---
q3x_document:
  id: q3x-qwen36-27b-runtime-contract
  class: contract
  status: active
  owner: runtime-maintainers
  authority: pinned Qwen3.5 and Qwen3.6 27B text runtime numerical and state contract
  effective: 2026-08-09
  last_reviewed: 2026-08-09
  supersedes: []
  superseded_by: []
  ssot_for: pinned 27B text Decode semantics, tensor boundaries, and recurrent state behavior
  review_trigger: any pinned model semantic, tensor boundary, Decode order, or state-contract change
---

# Qwen3.5 / Qwen3.6 27B text-only 单 token decode 运行时契约

> **权威边界。** 本组件契约细化 [系统 SDD](SDD.md)，并从属于该 SDD 与
> [工程宪法](ENGINEERING_CONSTITUTION.md)。当前实现、资格验证和默认生产路径的
> 动态真值只由 [`CURRENT_STATUS.md`](CURRENT_STATUS.md) 维护。本文中的局部机制、
> 性能门槛或实现选择，只有在被命名且处于活动状态的局部优化工作包内才具有约束力；
> 它们不能决定全局优先级、整体架构或生产晋级。

本文固定 Qwen3x-Orin 的 **27B Dense、batch=1、非 MTP、text-only**
decode 语义。目标不是规定某个 CUDA kernel 的物理布局，而是规定：相同 token、位置、
权重和进入本步前的状态，任何后端都必须产生可与可信 reference 对齐的逐层结果、更新
后的状态和 logits。

> 当前实现、资格验证和默认路径状态只以
> [`CURRENT_STATUS.md`](CURRENT_STATUS.md) 为准；未完成工作的先后顺序只以
> [`ROADMAP.md`](ROADMAP.md) 为准。本文是实现与 fixture 的入口契约，不是端到端
> 支持声明或进度清单。

## 1. 证据、版本和结论等级

审计日期为 **2026-07-18 (Asia/Shanghai)**。使用的本地源码 HEAD 为：

| 代号 | 仓库与固定提交 | 本文用途 | 许可证 |
|---|---|---|---|
| V | [vLLM `ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb`](https://github.com/vllm-project/vllm/tree/ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb) | Qwen3.5/3.6 loader、decoder、Gated DeltaNet、full-attention 和测试的主要运行时证据 | [Apache-2.0](https://github.com/vllm-project/vllm/blob/ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb/LICENSE)；其中 DeltaNet recurrent 文件声明了来自 flash-linear-attention 的 MIT 代码和版权信息，见 [文件头](https://github.com/vllm-project/vllm/blob/ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb/vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py#L1-L8) |
| T | [qwen35-thor `57e29777c2aff8a97f42df6e3d9487b1327f014f`](https://github.com/thomas-hiddenpeak/qwen35-thor/tree/57e29777c2aff8a97f42df6e3d9487b1327f014f) | Qwen3.5-27B C++/CUDA 路径、checkpoint 命名、状态分配和 decode 顺序的独立交叉证据 | [MIT](https://github.com/thomas-hiddenpeak/qwen35-thor/blob/57e29777c2aff8a97f42df6e3d9487b1327f014f/LICENSE#L1-L20) |

两者都不是 Qwen 模型发布方的权重 fixture。vLLM 是主要语义 oracle，qwen35-thor 是
独立实现证据；最终的支持结论仍需用固定官方 checkpoint 做逐层 fixture。

本文使用三个结论等级：

- **源码确认**：由上述固定提交中的运行时实现或测试直接确认。
- **派生**：只对源码确认的维度或公式做无歧义的算术/代数展开。
- **Fixture gate**：在真实 Qwen3.5 和 Qwen3.6 checkpoint 上仍必须验证，不能仅凭
  shape 相同宣称通过。

结论边界可以简化为：层调度、算子顺序、公式、logical tensor 名称/shape 和冷启动
状态语义属于**源码确认**；由这些常量计算的 cache element/byte count 属于**派生**；
具体官方 revision 的逐 tensor 物理 dtype/量化分配、backend 舍入、逐层数值和容差属于
**Fixture gate**。

Qwen3.5 与 Qwen3.6 的 27B engine-visible shape 相同，但权重不可互换；两者又共享
`qwen3_5` / `qwen3_5_text` 标识，因此 loader 必须保留显式 model series。固定配置
revision 见 [MODEL_SUPPORT.md](MODEL_SUPPORT.md#qwen35-versus-qwen36)。

## 2. 固定配置和层调度

下表是本契约的常量。vLLM 的 Qwen3.6 full-attention reference test 固定了 Q/K head、
head dim、partial RoPE、epsilon、最大位置和 theta
([V-test](https://github.com/vllm-project/vllm/blob/ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb/tests/kernels/test_fused_qk_norm_rope_gate.py#L12-L23))；
qwen35-thor 的 27B 默认配置及派生尺寸提供独立交叉检查
([T-config](https://github.com/thomas-hiddenpeak/qwen35-thor/blob/57e29777c2aff8a97f42df6e3d9487b1327f014f/src/engine/layer.h#L23-L47),
[T-derived](https://github.com/thomas-hiddenpeak/qwen35-thor/blob/57e29777c2aff8a97f42df6e3d9487b1327f014f/src/engine/layer.h#L193-L202))。

| 字段 | 固定值 |
|---|---:|
| vocabulary / hidden / dense intermediate | 248320 / 5120 / 17408 |
| decoder layers | 64 |
| layer schedule | 3 linear-attention + 1 full-attention，重复 16 次 |
| full-attention Q heads / KV heads / head dim | 24 / 4 / 256 |
| Q dim / gated Q projection dim / KV dim | 6144 / 12288 / 1024 |
| partial rotary factor / rotary dim | 1/4 / 64 |
| RoPE theta / maximum positions | 10000000 / 262144 |
| linear key heads × dim | 16 × 128 |
| linear value heads × dim | 48 × 128 |
| linear QKV projection dim / conv width | 10240 / 4 |
| RMS epsilon | `1e-6` |
| hidden activation | SiLU；Qwen3.6 的 `output_gate_type="swish"` 在运行时等价映射为 SiLU |
| tied embeddings | false；`lm_head.weight` 独立存在 |

以 0 为起点，full-attention 层是
`3, 7, 11, ..., 63`，其余 48 层是 linear-attention。vLLM 从
`layer_types[i]` 构造层，默认按 `(i + 1) % 4` 生成上述序列
([V-config](https://github.com/vllm-project/vllm/blob/ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb/vllm/transformers_utils/configs/qwen3_5.py#L42-L108))，
且 dense 27B 在每一层都实例化 dense MLP
([V-layer-init](https://github.com/vllm-project/vllm/blob/ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb/vllm/model_executor/models/qwen3_5.py#L123-L181))。

## 3. 一个 decode step 的总顺序

单 token decode 的输入必须是：

- 当前 token id `token_t`；
- 它的绝对位置 `p`；若请求的 logical context length 已包含当前 token，则
  `p = context_len - 1`；
- 16 层此前位置的 K/V；
- 48 层此前位置的 conv state 和 DeltaNet state。

qwen35-thor 的非 MTP 路径先 lookup 当前 token，再设置
`p = context_len - 1`，运行 64 层，最后做 final norm 和 lm-head
([T-decode](https://github.com/thomas-hiddenpeak/qwen35-thor/blob/57e29777c2aff8a97f42df6e3d9487b1327f014f/src/engine/engine.cpp#L1665-L1683),
[T-final](https://github.com/thomas-hiddenpeak/qwen35-thor/blob/57e29777c2aff8a97f42df6e3d9487b1327f014f/src/engine/engine.cpp#L1777-L1823))。

规范顺序为：

1. `x_0 = embed_tokens[token_t]`，shape `[5120]`。
2. 对 `i = 0..63`，按第 4 节的 pre-norm residual 规则执行本层；attention 分支
   由层类型选择第 5 或第 6 节。
3. `h = centered_rmsnorm(x_64, model.language_model.norm.weight)`。
4. `logits = lm_head.weight @ h`，shape `[248320]`。采样、penalty、tokenizer 和
   chat template 不属于模型 forward 数值契约。

vLLM 的模型循环及 final norm 顺序见
[V-model-forward](https://github.com/vllm-project/vllm/blob/ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb/vllm/model_executor/models/qwen3_next.py#L559-L601)，
lm-head 选择和 logits 计算见
[V-causal-lm](https://github.com/vllm-project/vllm/blob/ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb/vllm/model_executor/models/qwen3_5.py#L462-L529)。

## 4. RMSNorm、residual 和 dense MLP

### 4.1 两种不可混用的 RMSNorm

decoder input/post-attention norm、final norm、full-attention Q/K norm 都是
**centered-weight RMSNorm**：checkpoint 存储的是以 0 为中心的 `w`，有效 gamma 是
`1 + w`。

```text
centered_rmsnorm(x, w) = x * rsqrt(mean(x²) + 1e-6) * (1 + w)
```

vLLM 将 Qwen3.5 RMSNorm 绑定到 `GemmaRMSNorm`
([V-import](https://github.com/vllm-project/vllm/blob/ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb/vllm/model_executor/models/qwen3_5.py#L42-L48))，
其定义明确使用 `weight + 1`
([V-centered-norm](https://github.com/vllm-project/vllm/blob/ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb/vllm/model_executor/layers/layernorm.py#L127-L164))。

linear-attention 内部对 DeltaNet 输出使用的是 **plain-weight per-value-head
RMSNorm**，有效 gamma 就是 `w`：

```text
plain_rmsnorm(y, w) = y * rsqrt(mean(y²) + 1e-6) * w
```

该 norm 在 128 维 value head 内计算，然后再乘 output gate；vLLM 的
`RMSNormGated(norm_before_gate=True)` 定义见
[V-gated-norm](https://github.com/vllm-project/vllm/blob/ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb/vllm/model_executor/layers/layernorm.py#L169-L266)，
qwen35-thor 也明确标注该权重不是 centered
([T-gated-norm](https://github.com/thomas-hiddenpeak/qwen35-thor/blob/57e29777c2aff8a97f42df6e3d9487b1327f014f/src/engine/light_ops.cu#L1370-L1407))。

### 4.2 每层的规范 residual 顺序

无论 attention 类型，数学顺序都是：

```text
n_i       = centered_rmsnorm(x_i, input_layernorm.weight)
a_i       = attention_i(n_i)
r_i       = x_i + a_i
m_i_in    = centered_rmsnorm(r_i, post_attention_layernorm.weight)
m_i       = down_proj(silu(gate_proj(m_i_in)) * up_proj(m_i_in))
x_(i + 1) = r_i + m_i
```

所有 dense MLP projection 都无 bias。vLLM 的 residual-buffer 实现可能把最后一次
`+ m_i` 延迟到下一层的 fused add+norm 或 final norm，但数学检查点必须按上述
`r_i` 和 `x_(i+1)` 定义，不应把内部临时 buffer 当成层输出
([V-layer-forward](https://github.com/vllm-project/vllm/blob/ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb/vllm/model_executor/models/qwen3_next.py#L460-L517))。
MLP 的 `SiLU(gate) * up` 和 down projection 由
[V-MLP](https://github.com/vllm-project/vllm/blob/ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb/vllm/model_executor/models/qwen2_moe.py#L77-L122)
确认，qwen35-thor 的单 token 路径也执行相同顺序并将 down 输出加入 residual
([T-MLP](https://github.com/thomas-hiddenpeak/qwen35-thor/blob/57e29777c2aff8a97f42df6e3d9487b1327f014f/src/engine/layer.cu#L23-L112))。

## 5. Linear attention：Gated DeltaNet 精确语义

本节适用于 48 个 linear-attention 层。令输入为第 4 节的 `n_i`。

### 5.1 投影和 logical layout

```text
[q_raw, k_raw, v_raw] = in_proj_qkv(n_i)  # 2048 + 2048 + 6144
z                     = in_proj_z(n_i)    # 48 × 128
a                     = in_proj_a(n_i)    # 48 scalars
b                     = in_proj_b(n_i)    # 48 scalars
```

Qwen3.5/3.6 checkpoint 是非 interleaved layout：`in_proj_qkv` 内依次为完整 Q、K、V，
另有独立 Z、B、A tensor。vLLM 对该布局的 split 及 fused-loader mapping 见
[V-GDN-layout](https://github.com/vllm-project/vllm/blob/ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py#L566-L614)
和
[V-weight-map](https://github.com/vllm-project/vllm/blob/ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb/vllm/model_executor/models/qwen3_5.py#L279-L294)。

### 5.2 Width-4 depthwise causal conv

conv 只作用于拼接的 Q/K/V 共 10240 个 channel；Z、A、B 不经过 conv。每个 channel
有独立的 4 个权重、无 bias。进入本步前，`C[c] = [u_(t-3), u_(t-2), u_(t-1)]`
保存的是该 channel 的**原始 projection 输出**，不是此前的 conv 输出。单步语义是：

```text
conv_raw[c] = C[c,0] * Wconv[c,0]
            + C[c,1] * Wconv[c,1]
            + C[c,2] * Wconv[c,2]
            + u_t[c] * Wconv[c,3]
conv_qkv[c] = silu(conv_raw[c])
C'[c]       = [C[c,1], C[c,2], u_t[c]]
```

qwen35-thor 的 decode kernel 直接显示了权重顺序、SiLU 和 shift/update
([T-conv](https://github.com/thomas-hiddenpeak/qwen35-thor/blob/57e29777c2aff8a97f42df6e3d9487b1327f014f/src/engine/light_ops.cu#L634-L715))；
vLLM 在 projection 后调用 stateful `causal_conv1d_update`，再进入 recurrent rule
([V-conv-call](https://github.com/vllm-project/vllm/blob/ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py#L1644-L1695))。

### 5.3 Gated Delta rule

把 conv 输出拆成 16 个 Q heads、16 个 K heads 和 48 个 V heads，每个 head dim
都是 128。每 3 个连续 value heads 共享一个 key/query head：

```text
h(j) = floor(j / 3),  j = 0..47
q_hat_h = q_h * rsqrt(sum(q_h²) + 1e-6) / sqrt(128)
k_hat_h = k_h * rsqrt(sum(k_h²) + 1e-6)
alpha_j = exp(-exp(A_log[j]) * softplus(a[j] + dt_bias[j]))
beta_j  = sigmoid(b[j])
```

注意这里是 L2 normalization（`sum`），不是 RMSNorm（`mean`）；`1/sqrt(128)` 只乘
Q。令每个 value head 的 logical state `S_j` 为 `[V=128, K=128]`，则：

```text
P_j      = alpha_j * S_j
delta_j  = v_j - P_j @ k_hat_h(j)
S'_j     = P_j + beta_j * outer(delta_j, k_hat_h(j))
y_j      = S'_j @ q_hat_h(j)
```

vLLM fused recurrent kernel 逐项执行 decay、prediction error、beta update 和 output
([V-recurrence](https://github.com/vllm-project/vllm/blob/ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb/vllm/model_executor/layers/fla/ops/fused_recurrent.py#L292-L336))；
非 packed 路径给出同一公式及 state 写回
([V-recurrence-general](https://github.com/vllm-project/vllm/blob/ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb/vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py#L122-L170))；
qwen35-thor 的 C++/CUDA 路径独立确认
([T-recurrence](https://github.com/thomas-hiddenpeak/qwen35-thor/blob/57e29777c2aff8a97f42df6e3d9487b1327f014f/src/engine/light_ops.cu#L941-L1015))。

`S_j` 的物理存储可为 `[48,V,K]` 或等价转置/reshape，例如 qwen35-thor 使用
`[48,K,V]`。物理 layout 不是模型 ABI；repack 或 kernel registry 必须显式携带 layout，
fixture 应在 canonical `[48,V,K]` 视图比较。

### 5.4 Linear-attention 输出 gate

把 48 个 `y_j` 分别做共享 `[128]` plain-weight RMSNorm，再乘 `SiLU(z_j)`，flatten
到 6144 后做 output projection：

```text
gdn_j = plain_rmsnorm(y_j, linear_attn.norm.weight) * silu(z_j)
a_i   = linear_attn.out_proj(flatten(gdn_0 ... gdn_47))
```

Qwen3.6 的 `output_gate_type="swish"` 被 vLLM 映射成 SiLU；Qwen3.5 缺省也是 SiLU
([V-GDN-output](https://github.com/vllm-project/vllm/blob/ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py#L529-L552),
[V-GDN-project](https://github.com/vllm-project/vllm/blob/ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py#L851-L869))。
这与 full-attention 的 `sigmoid(gate)`、MLP 的 `SiLU(gate) * up` 是三个不同 gate，
实现和测试命名不得混用。

## 6. Full attention：GQA、partial RoPE 和 output gate

本节适用于 16 个 full-attention 层。输入仍为第 4 节的 `n_i`。

### 6.1 Q/Gate、K、V 投影

```text
q_gate_raw = self_attn.q_proj(n_i)  # [24, 2, 256]
k_raw      = self_attn.k_proj(n_i)  # [4, 256]
v          = self_attn.v_proj(n_i)  # [4, 256]
```

`q_proj` 的 12288 行不是 `[all Q | all gate]`，而是按 head 排列的
`[q_head_0 | gate_head_0 | q_head_1 | gate_head_1 | ...]`。vLLM 的 Qwen3.6 test
显式 reshape 为 `[num_tokens, 24, 2*256]` 后逐 head split，并要求 gate copy bit-exact
([V-QG-layout](https://github.com/vllm-project/vllm/blob/ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb/tests/kernels/test_fused_qk_norm_rope_gate.py#L27-L82),
[V-QG-check](https://github.com/vllm-project/vllm/blob/ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb/tests/kernels/test_fused_qk_norm_rope_gate.py#L128-L159))。

### 6.2 Q/K norm 和 partial NeoX RoPE

每个 Q/K head 单独做 centered-weight RMSNorm；然后只对 head 的前 64 维做
NeoX-style RoPE，后 192 维原样通过。RoPE 使用绝对位置 `p` 和 theta `1e7`：

```text
q = partial_neox_rope(centered_rmsnorm(q_raw, q_norm.weight), p)
k = partial_neox_rope(centered_rmsnorm(k_raw, k_norm.weight), p)
```

旋转的 64 维分成前后两个 32 维 half，执行标准 NeoX half rotation；精确 reference
见 [V-RoPE-reference](https://github.com/vllm-project/vllm/blob/ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb/tests/kernels/test_fused_qk_norm_rope_gate.py#L55-L81)。
V 不做 Q/K norm，也不做 RoPE。

### 6.3 Causal GQA、cache 写入和 gate

当前 post-norm/post-RoPE K 与当前 V 必须先在 logical position `p` 成为本层 cache 的
一部分；随后 query 能看到 `0..p`。每个 KV head 服务 6 个连续 Q heads：

```text
kv_head(hq) = floor(hq / 6)
score(hq,s) = dot(q[hq], K[s,kv_head(hq)]) / sqrt(256)
context[hq] = sum_s softmax(score(hq,0..p))[s] * V[s,kv_head(hq)]
gated[hq]   = context[hq] * sigmoid(gate_raw[hq])
a_i         = self_attn.o_proj(flatten(gated))
```

vLLM 的 attention 初始化、scale、GQA head 数及执行顺序见
[V-full-init](https://github.com/vllm-project/vllm/blob/ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb/vllm/model_executor/models/qwen3_next.py#L211-L309)
和 [V-full-forward](https://github.com/vllm-project/vllm/blob/ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb/vllm/model_executor/models/qwen3_next.py#L311-L378)。
qwen35-thor 独立显示当前 token 的 K/V 写入、`1/sqrt(head_dim)` 和 decode cache 路径
([T-KV-write](https://github.com/thomas-hiddenpeak/qwen35-thor/blob/57e29777c2aff8a97f42df6e3d9487b1327f014f/src/engine/layer.cu#L631-L672),
[T-attn-gate](https://github.com/thomas-hiddenpeak/qwen35-thor/blob/57e29777c2aff8a97f42df6e3d9487b1327f014f/src/engine/layer.cu#L797-L841))。

## 7. 必需 checkpoint tensor

以下是本 runtime 面向 Qwen3.5/3.6 ConditionalGeneration checkpoint 要求的
**logical tensor ABI**；固定源码 loader 直接确认这些名称和 shape。具体官方 revision
的完整 manifest 与物理 dtype 仍是 Fixture gate。令
`P_i = model.language_model.layers.{i}.`。shape 按 safetensors 中的 row-major logical
shape 表示；projection 权重均为 `[out, in]`。

### 7.1 全局 tensor

| Tensor | Shape | 基线 dtype / 规则 |
|---|---:|---|
| `model.language_model.embed_tokens.weight` | `[248320, 5120]` | BF16，必需 |
| `model.language_model.norm.weight` | `[5120]` | BF16 centered weight，必需 |
| `lm_head.weight` | `[248320, 5120]` | 27B `tie_word_embeddings=false`，独立且必需；物理精度服从 checkpoint metadata |

qwen35-thor 对这三个名字做强制绑定，缺失即报错
([T-global-weights](https://github.com/thomas-hiddenpeak/qwen35-thor/blob/57e29777c2aff8a97f42df6e3d9487b1327f014f/src/engine/model.cpp#L966-L980))。

### 7.2 每层公共 tensor（`i = 0..63`）

| Tensor | Shape | 基线 dtype / 规则 |
|---|---:|---|
| `P_i + input_layernorm.weight` | `[5120]` | BF16 centered weight |
| `P_i + post_attention_layernorm.weight` | `[5120]` | BF16 centered weight |
| `P_i + mlp.gate_proj.weight` | `[17408, 5120]` | metadata-directed linear weight |
| `P_i + mlp.up_proj.weight` | `[17408, 5120]` | metadata-directed linear weight |
| `P_i + mlp.down_proj.weight` | `[5120, 17408]` | metadata-directed linear weight |

### 7.3 Linear-attention 层 tensor

仅 `i ∉ {3,7,...,63}`：

| Tensor | Shape | 基线 dtype / 规则 |
|---|---:|---|
| `P_i + linear_attn.in_proj_qkv.weight` | `[10240, 5120]` | metadata-directed；rows 为 Q 2048、K 2048、V 6144 |
| `P_i + linear_attn.in_proj_z.weight` | `[6144, 5120]` | metadata-directed |
| `P_i + linear_attn.in_proj_a.weight` | `[48, 5120]` | 通常 BF16；无 bias |
| `P_i + linear_attn.in_proj_b.weight` | `[48, 5120]` | 通常 BF16；无 bias |
| `P_i + linear_attn.conv1d.weight` | `[10240, 1, 4]` | BF16；depthwise、无 bias；loader 可规范化为 `[10240,4]` |
| `P_i + linear_attn.A_log` | `[48]` | pinned Qwen3.6 artifact 序列化为 BF16；`exp` 前转 FP32 |
| `P_i + linear_attn.dt_bias` | `[48]` | pinned Qwen3.6 artifact 序列化为 BF16；softplus 前转 FP32 |
| `P_i + linear_attn.norm.weight` | `[128]` | BF16 plain weight，不加 1 |
| `P_i + linear_attn.out_proj.weight` | `[5120, 6144]` | metadata-directed |

vLLM 构造这些参数及维度见
[V-GDN-params](https://github.com/vllm-project/vllm/blob/ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py#L443-L552)，
qwen35-thor 的 checkpoint 绑定名字见
[T-linear-bind](https://github.com/thomas-hiddenpeak/qwen35-thor/blob/57e29777c2aff8a97f42df6e3d9487b1327f014f/src/engine/model.cpp#L404-L414)。

### 7.4 Full-attention 层 tensor

仅 `i ∈ {3,7,...,63}`：

| Tensor | Shape | 基线 dtype / 规则 |
|---|---:|---|
| `P_i + self_attn.q_proj.weight` | `[12288, 5120]` | metadata-directed；每 head 内 `[q(256), gate(256)]` |
| `P_i + self_attn.k_proj.weight` | `[1024, 5120]` | metadata-directed |
| `P_i + self_attn.v_proj.weight` | `[1024, 5120]` | metadata-directed |
| `P_i + self_attn.o_proj.weight` | `[5120, 6144]` | metadata-directed |
| `P_i + self_attn.q_norm.weight` | `[256]` | BF16 centered weight |
| `P_i + self_attn.k_norm.weight` | `[256]` | BF16 centered weight |

以上 attention projection 均无 bias。vLLM 的构造尺寸见
[V-full-params](https://github.com/vllm-project/vllm/blob/ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb/vllm/model_executor/models/qwen3_next.py#L221-L299)，
checkpoint 名称由 qwen35-thor 的绑定路径独立确认
([T-full-bind](https://github.com/thomas-hiddenpeak/qwen35-thor/blob/57e29777c2aff8a97f42df6e3d9487b1327f014f/src/engine/model.cpp#L394-L402))。

### 7.5 量化物理表示

“metadata-directed” 不允许根据 model 名或 projection 类型猜 dtype。对未量化的 base
checkpoint，它是 `[N,K]` BF16。对 ModelOpt mixed-precision checkpoint：

- W4A16 NVFP4 的 canonical tensor 是 `weight` U8 `[N,K/2]`、`weight_scale`
  E4M3 `[N,K/16]`、`weight_scale_2` FP32 scalar；vLLM 的固定源码明确记录了命名、
  dtype、group=16 和 `weight_scale_2 = amax/(6*448)`
  ([V-W4A16](https://github.com/vllm-project/vllm/blob/ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb/vllm/model_executor/layers/quantization/modelopt.py#L1244-L1254))。
- FP8 W8A16 的 `weight` 是 E4M3 `[N,K]`；scale 的具体 shape/policy 必须从
  `hf_quant_config.json` 与 tensor metadata 读取，不在本架构表硬编码。
- norm、`A_log`、`dt_bias` 等非 linear tensor 保持各自声明 dtype，不得因同层另一个
  projection 被量化而一起转换。完整 materialized shard 的 header 检查已确认 pinned
  Qwen3.6-27B 中 `A_log`、`dt_bias`、`conv1d.weight`、`in_proj_a/b.weight` 和
  linear-attention internal norm 均为 BF16；运行时提升到 FP32 是计算策略，不是对
  checkpoint dtype 的重写。

哪一些 projection 是 NVFP4、FP8 或 BF16 是 **Fixture gate**。Qwen3.5 与 Qwen3.6
官方打包不同，且同一 checkpoint 内并非全 FP4；loader 必须先输出完整 tensor manifest。

## 8. 可跳过的 tensor 和派生数据

text-only、非 MTP runner 可以且应当跳过：

- 所有 `model.visual.*`；
- 所有 `mtp.*`；
- checkpoint 若携带 `rotary_emb.inv_freq`，可忽略并由已验证 config 生成 RoPE cache。

vLLM 的 base text loader 跳过 `mtp.*` 和 `rotary_emb.inv_freq`
([V-loader-skip](https://github.com/vllm-project/vllm/blob/ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb/vllm/model_executor/models/qwen3_5.py#L317-L328),
[V-base-loader](https://github.com/vllm-project/vllm/blob/ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb/vllm/model_executor/models/qwen3_5.py#L531-L536))。
qwen35-thor 也把 MTP 和 vision 作为检测到完整权重时才启用的可选模块
([T-MTP-optional](https://github.com/thomas-hiddenpeak/qwen35-thor/blob/57e29777c2aff8a97f42df6e3d9487b1327f014f/src/engine/model.cpp#L982-L1017),
[T-vision-optional](https://github.com/thomas-hiddenpeak/qwen35-thor/blob/57e29777c2aff8a97f42df6e3d9487b1327f014f/src/engine/model.cpp#L1120-L1187))。

跳过表示不加载、不分配、不执行；不得把 vision/MTP tensor 的存在误报成 text runtime
不兼容。反过来，text ABI 中的必需 tensor 缺失必须 fail closed。

## 9. 初始化和 cache 尺寸

### 9.1 Linear-attention persistent state

非 speculative、TP=1 时每个 linear-attention 层有两个 logical state：

| State | Logical shape | BF16 elements / bytes |
|---|---:|---:|
| causal-conv history | `[10240, 3]` | 30720 / 61440 B = 60 KiB |
| DeltaNet matrix state | `[48, 128, 128]` | 786432 / 1572864 B = 1.5 MiB |

48 层合计 **78,446,592 bytes = 74.8125 MiB / request**。其中 DeltaNet 为 72 MiB，
conv 为 2.8125 MiB。这是由源码 shape 直接派生；vLLM 的 shape calculator 使用
`conv_kernel-1` 和 `[num_v_heads, head_v_dim, head_k_dim]`
([V-state-shape](https://github.com/vllm-project/vllm/blob/ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb/vllm/model_executor/layers/mamba/mamba_utils.py#L213-L234))，
qwen35-thor 按同一尺寸分配 BF16 state
([T-state-size](https://github.com/thomas-hiddenpeak/qwen35-thor/blob/57e29777c2aff8a97f42df6e3d9487b1327f014f/src/engine/cache_manager.cpp#L39-L50))。

每个新请求的 conv 和 DeltaNet state 必须清零；不能复用上一个请求的残留。实际分配
和清零证据见
[T-state-init](https://github.com/thomas-hiddenpeak/qwen35-thor/blob/57e29777c2aff8a97f42df6e3d9487b1327f014f/src/engine/cache_manager.cpp#L177-L201)。

运行时请求状态必须为 48 份 conv/GDN state、16 对 full-attention K/V、复用
workspace 与 BF16-rounded RoPE 表提供显式所有权，并在创建和 reset 时使完整
persistent span 满足零状态契约；当前实现与精确 byte budget/API 见
[REQUEST_STATE.md](REQUEST_STATE.md)。

vLLM 允许 state dtype 配置；`auto` 时 conv state 跟随 model/cache dtype，SSM state
默认跟随 conv state，也可单独指定
([V-state-dtype](https://github.com/vllm-project/vllm/blob/ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb/vllm/model_executor/layers/mamba/mamba_utils.py#L84-L116))。
兼容性数值契约采用 BF16 persistent state；kernel 内 decay、dot、outer update 应以 FP32
累加，再量化写回 BF16。若选择 FP32 SSM state，单请求 state 总量变为 146.8125 MiB，
且必须作为不同的数值策略单独验证。

### 9.2 Full-attention KV

每个 token 的 logical K/V 为：

```text
16 layers * 2 (K,V) * 4 KV heads * 256 = 32768 elements
```

BF16 下为 **65536 bytes = 64 KiB/token**；完整 262144 positions 为 **16 GiB**，尚未
计 allocator padding、page table 和 workspace。物理 paged layout、block size 和
K/V 交错方式不是模型 ABI，但每层必须可恢复 logical `[tokens,4,256]` K 和 V。

新请求以 logical KV length 0 开始；未写区域不要求物理清零，只要 length/page metadata
保证不可见。进入 decode 前，prompt prefill 必须同时建立 K/V、conv history 和 DeltaNet
state；只有 full-attention KV 而没有两类 linear state 不是有效 continuation cache。

## 10. 数值执行策略

canonical BF16 reference 策略是：

- activation、projection 输出和 persistent cache 基线为 BF16；
- RMS variance、Q/K L2 norm、conv 累加、`A_log`、softplus、alpha、DeltaNet
  dot/outer update 使用 FP32 中间值；
- centered norm 的 `weight + 1` 在 FP32 形成，再把输出转回 activation dtype；
- full-attention score scale 是 `1/sqrt(256)`，DeltaNet Q scale 是 `1/sqrt(128)`；
- softmax、GEMM/MMA 的具体 reduction tree 可以不同，但不得改变上述公式或 gate 顺序。

`beta` 的模型语义固定为 `sigmoid(b)`，但固定源码中存在一个需要 fixture 暴露的舍入差异：
qwen35-thor 和 vLLM 通用 recurrent 路径保留 FP32 `beta`，vLLM packed decode 路径则将
sigmoid 结果先 cast 到 `b.dtype`、再转回 FP32
([T-recurrence](https://github.com/thomas-hiddenpeak/qwen35-thor/blob/57e29777c2aff8a97f42df6e3d9487b1327f014f/src/engine/light_ops.cu#L980-L1015),
[V-recurrence-general](https://github.com/vllm-project/vllm/blob/ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb/vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py#L122-L170),
[V-recurrence-packed](https://github.com/vllm-project/vllm/blob/ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb/vllm/model_executor/layers/fla/ops/fused_recurrent.py#L316-L329))。
canonical reference 保留 FP32 `beta`；任何低精度舍入都必须记录成独立 backend policy，
并通过逐层 state/logits fixture，而不能被当成公式变更。

优化 kernel 必须记录其 accumulation policy。不得用一个全局宽松 tolerance 掩盖：
centered/plain norm 混淆、Q/Gate layout 错误、RoPE 旋转范围错误、conv state 顺序错误，
或 DeltaNet state 转置错误。

## 11. Fixture 计划和数值参考点

至少为 **Qwen3.5-27B 与 Qwen3.6-27B 各一份固定 revision** 建立两类 fixture：

1. 冷启动最短序列：state 全零，覆盖 position 0 与前几个 token。
2. 固定真实 prompt continuation：覆盖非零 conv/DeltaNet state 和多页 K/V 后的一个
   decode token。

tokenizer 与 chat template 先在 reference 侧固化为 token id 数组；本契约从 token id
开始。每个 fixture 记录 checkpoint revision/hash、原始 tensor manifest、量化 metadata、
reference runtime commit、dtype、cache dtype 和 position。

建议按以下顺序保存参考点；第一处不一致即停止并报告，而不是只看最终 token：

| 检查点 | 必须捕获的值 | 主要定位问题 |
|---|---|---|
| 0 | embedding `[5120]` | token id / embedding name / dtype |
| 每层入口 | canonical `x_i`、input norm 输出 | centered RMSNorm、residual 时序 |
| Linear projection | raw Q/K/V、Z、A、B | tensor 行顺序、量化 scale |
| Linear conv | conv Q/K/V、更新后的 `[10240,3]` | causal window、weight 顺序、SiLU |
| DeltaNet gates | normalized Q/K、alpha、beta | L2 vs RMS、A_log/dt、gate 非线性 |
| DeltaNet state | canonical `[48,V,K]` 的更新前后 state、`y` | decay/delta/outer、layout、cache dtype |
| Linear output | plain norm 后、SiLU(Z) 后、out-proj 后 | plain vs centered norm、output gate |
| Full projection | per-head Q、gate、K、V | Q/Gate interleave、projection shape |
| Full norm/RoPE | Q/K norm 后与 RoPE 后 | `+1` gamma、只旋转前 64 维、position/theta |
| Full cache/attention | 当前 K/V、scores/softmax 或 context、sigmoid gate 后、o-proj 后 | GQA mapping、current-token 可见性、scale |
| 每层 MLP | attention residual、post norm、gate/up、SwiGLU、down、`x_(i+1)` | residual 与 dense MLP |
| 输出 | final norm、完整 logits、top-k ids/values、greedy id | final norm、lm-head、整体漂移 |

已可先固定的严格检查：

- tensor 名称、shape、layer schedule、cache element count：exact；
- Q projection 的 gate slice：从源 projection 复制到 gate reference 必须 bit-exact，vLLM
  已以 `atol=rtol=0` 测试；
- state 初始化：exact zero；
- 重排/repack：在 dequantized logical view 上按对应 dtype policy 比较，不允许静默重标度。

仍为 **Fixture gate** 的项目：

- 两个官方 revision 的逐 tensor dtype 与 NVFP4/FP8/BF16 module 选择；
- Qwen3.5 artifact 的 `A_log`、`dt_bias` 和其他非 linear tensor 实际序列化 dtype；
- Qwen3.5 与 Qwen3.6 各自的逐层 activation、state 和 logits fixture；
- BF16 persistent state 相对可信 FP32-state reference 的逐 token 误差累积；
- packed decode 中 `beta` 舍入相对 canonical FP32 `beta` 的逐层影响；
- 每个 operation 的 `atol/rtol`、最终 logits 阈值和允许的 greedy-generation 长度。

在这些 fixture 进入仓库并在 Jetson AGX Orin 上复现以前，只能宣称本文的结构契约已
审计，不能把 Qwen3.5 或 Qwen3.6 27B 标为 `Verified`。

## 12. 一致性义务

- loader 必须先验证 model series、固定 revision/config 和完整 tensor manifest。
- 64 层调度必须为 48 linear + 16 full，且每层都有 dense MLP。
- centered RMSNorm 与 GDN plain RMSNorm 必须使用不同 kernel key 或显式 flag。
- GDN checkpoint layout 必须是 Q/K/V/Z 与 B/A non-interleaved；物理 state layout
  必须显式版本化。
- conv state 必须保存 projection input 的最近 3 项，并在输出计算后 shift。
- DeltaNet value head 到 key head 的映射必须为 3:1，Q/K 使用 L2 norm。
- full-attention Q/Gate 必须按每 head interleave，RoPE 只旋转前 64 维。
- full-attention cache 必须写入当前 K/V，decode query 可见位置 `0..p`。
- text-only 加载不得要求 `model.visual.*` 或 `mtp.*`。
- final centered RMSNorm 后才可执行独立 `lm_head.weight`。
- fixture 必须同时比较 activation、两类 linear state、K/V 与 logits，而非只比较 token。
- 所有第三方实现若被复制或改编，必须保留相应 Apache/MIT 头和 NOTICE；本文只总结
  运行时契约，没有复制第三方代码。
