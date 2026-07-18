# Qwen3.6-27B batch-one CUDA correctness runner

`q3x::runtime::ReferenceRunner` 是当前 Qwen3.6-27B dense text 模型的高层
correctness 路径。它把已经绑定的 `ModelWeights`、一个 `RequestState` 以及 reference
CUDA kernels 串成完整的 64 层单 token step，并提供最大 8 token 的 prompt-prefix tile。
它优先固定执行语义和检查点，不承诺大 prefill 或 serving 吞吐。

## 所有权和创建

`create_reference_runner` 在执行任何 CUDA 工作前严格检查：

- 全局和 64 层权重的精确 shape、非空 payload 与 3-linear/1-full variant schedule；
- batch 必须为 1，hidden/P0..P3/a/b/FP32 scratch、RoPE、48 份 conv/GDN state、
  16 对 KV cache 的容量和 schedule slot 必须满足精确 ABI；tile activation region
  必须覆盖 request plan 中声明的 `prefill_chunk_size`（1 到 8）；
- `RequestState` 必须有效，logical length 不能超过 capacity。

factory 创建一个自有 `cudaStreamNonBlocking` stream，并通过一次 `cudaHostAlloc` 预留
`float[248320]` logits。`ReferenceRunnerOptions::enable_trace=true` 时再通过一次
`cudaHostAlloc` 预留固定大小的 BF16 trace。factory 返回结构化
`ReferenceRunnerStatus`，不会用异常表达 CUDA/contract 错误。

`ReferenceRunnerOptions::projection_backend` 默认是 `kReference`；只有调用方显式选择
`kSm87WeightOnly` 时，层内 FP8/NVFP4 projection 才直接写 BF16，BF16 projection 仍回退
reference。未知 enum 会在 factory 阶段失败。最终 `lm_head` 始终走 FP32 reference 路径，
因此 backend 选项不会改变 logits 边界的类型与 host 分析流程。

runner 是 move-only，但不拥有 `ModelWeights` 或 `RequestState`。这两个**同一个对象**、
resident weight arena 和 request arena 必须比 runner 活得更久；runner 存活期间不得移动
或并发操作它们。析构会先收口自有 stream，再释放 pinned host storage。

## 单 token 顺序

workspace 固定复用 `RequestState` 中的 `H0/H1/H2`、`P0..P3`、`linear_a/b` 和
FP32 scratch。token 路径没有 `cudaMalloc`、`cudaHostAlloc`、C++ heap allocation 或
隐式 host/device 同步；所有 launch、D2D 和可选 D2H 都进入同一个 owned stream。成功
step 在末尾执行一次 `cudaStreamSynchronize`；中途失败也会先 drain 同一 stream 再返回。

1. `embed_tokens[input_id] -> H0`。
2. 对层 `0..63`：
   - `centered_rmsnorm(H0, input_layernorm) -> H1`；
   - linear 层依次投影 QKV/Z/A/B，原位 width-4 causal conv，原位 BF16 GDN state
     recurrence，逐 value-head plain norm × `SiLU(z)`，然后 `out_proj -> H1`；
   - full 层投影 `[Q|gate]`/K/V，将 per-head interleaved Q/gate 拆到 P3，Q/K 逐 head
     centered norm，使用 `RequestState` 中已 BF16-round 的 RoPE table 做 256/64 partial
     NeoX RoPE；当前 K/V 先 D2D 写入 position `p`，GQA 再读取 `0..p`，固定
     `24/4/256` 和 scale `1/16`；context 乘 `sigmoid(gate)` 后做 `o_proj -> H1`；
   - `H2 = BF16(H0 + attention_output)`；
   - post-attention centered norm，MLP gate/up，`SiLU(gate)*up`，`down_proj -> H1`；
   - `H0 = BF16(H2 + H1)`。
3. `centered_rmsnorm(H0, final_norm) -> H1`。
4. `compute_logits=true` 时执行 `lm_head -> FP32 scratch` 和异步 D2H。host 将每个
   logit 先按 BF16 RNE 量化、再扩展回 FP32，然后做稳定 logsumexp 和 argmax；相同最大
   值选择最小 token id。NaN 或 infinity 是硬错误。

`compute_logits=false` 仍执行 embedding、全部 64 层、所有 cache/state update 和 final
norm，只跳过 lm_head 与 logits D2H，适合调用方逐 token 建立 prompt prefix。

## Prompt-prefix tile

`prefill_prefix_tile(ids, M)` 接受 `M=1..8`，且不能超过 request plan 预留的 chunk
容量或剩余 sequence capacity。M1 直接委托给上述 `step(compute_logits=false)`；M2..M8
按 layer-major 顺序执行：每层先批量做 norm 与 projection，再按 token 顺序推进 causal
conv/GDN，或逐 position 写 K/V 并以 `first_position+t+1` 作为 GQA 的 causal length。
完整 tile 成功同步后才一次性提交新的 sequence length；任何校验、launch 或同步失败都
保持结构化错误并 poison runner。

SM87 FP8/NVFP4 projection 在 M2..M8 使用 small-M weight-reuse kernel。reference backend
和 BF16 weight 保留逐行 M1 fallback，因此接口语义不依赖 optimized backend。tile 不计算
logits，也不采集 trace；generation controller 只对 prompt 的最后一个 token 执行标量 logits
step。启用 trace 时 controller 会把 effective chunk 强制为 1，以保留逐 token trace ABI。

projection 的具体 BF16/FP8/NVFP4 计算策略完全由 `launch_projection_*` dispatch 决定；
runner 不复制也不改变底层量化语义。GDN canonical reference 保留 FP32 beta。vLLM
packed decode 的 BF16 beta 回写与 prefill/通用路径可能产生可解释的舍入差异，不能把
trace/state fixture 的逐位相等当作跨 backend 通用要求。

## commit、poison 和 reset

device 工作、可选 D2H 和 host logits 检查都成功后才调用 `RequestState::commit_token()`。
任意 step 错误（token 越界、capacity、variant、launch、同步、trace 配置、非有限 logits
或 commit）都会 poison runner；poison 后 step 会稳定返回 `kPoisoned`，不会继续修改
state。只有成功的 `reset()` 会清零完整 persistent span、同步 owned stream、把 logical
position 置零并恢复 runner。

每个成功结果包含输入 token 和提交前的绝对 `position`；有 logits 时包含 predicted id、
chosen/max logit、max log-probability 和 logsumexp。`measure_timing` 可附带包含末尾同步与
host logits 分析的端到端毫秒数。

## Trace 与 fixture

factory 启用 trace 后，step 的 `capture_trace=true` 才会排队 D2H；false 时没有任何 trace
copy。成功同步并 commit 后，`last_trace()` 暴露只读 pinned spans，布局为：

```text
embedding[5120]
for i in 0..63:
    hidden_i[5120]    # 本层 MLP down projection 输出 m_i
    residual_i[5120]  # attention residual r_i
final_norm[5120]
```

这些采样点与 `tests/fixtures/qwen36-27b-nvfp4-layers-bf16.json` 的 fixture 语义一致。
view 在 reset、runner 析构或下一次 captured step 后失效。trace 只借用 pinned storage，
不会增加 device arena，也不会改变 kernel 顺序。

## 当前验证边界

`reference_runner_host` 不加载 20GB 权重，验证纯 host BF16/logits oracle、tie/nonfinite
策略、trace offsets、固定 layer schedule、默认 `RequestMemoryPlan` 以及 null dependency
factory 错误。controller/unit gates 覆盖 19-token prompt 的 `8+8+2` prefix 拆分、C1/trace
fallback、tile malformed result 和失败传播；完整模型 fixture 在目标 Orin 上以 C1/C8
分别执行。runner 本身不会把缺少 checkpoint 误报为通过。
