# Prefill Gate+Up M64N8 paired-warp register-pipeline rejection

Date: 2026-08-03

Scope: Qwen3.6-27B-NVFP4, AGX Orin SM87, single-request Prefill

Performance authority: real checkpoint, OpenAI `/v1/completions`, external
EvalScope 1.9.1

## Decision

Reject the M64N8 paired-warp register-pipeline Gate+Up data flow.  The exact
candidate route was selected in the complete runner, but its first real P1853
direction gate regressed server Prefill by 478.04 ms (+25.01%), regressed
external TTFT by 478.09 ms (+24.95%), and reduced total throughput by
193.1826 token/s (-19.97%).  It remains build-time and runtime default off and
does not replace the production M64N128 K256 LDSM pair-feed route.

The result is large enough to reject the skeleton.  No cache-hint, tile-size,
lookahead-depth, or barrier scan follows it.

The rejected mode is:

```text
cumulative-prefill-current-best-mlp-k512-paired-warp-gateup-down-16warp-pairring-attention-k256-a-exchange-b4
```

## Structural hypothesis that failed

The candidate uses the authenticated paired-GateUp/canonical-Down
publication.  One persistent 512-thread CTA owns M64N512, and each of its 16
warps owns one N8 slice while computing both Gate and Up across four M16
panels.  For each K64 plane, one `ldmatrix.x4` A fragment feeds adjacent Gate
and Up IMMA instructions.  A uses a two-slot K512 `cp.async` ring; paired B
codes use one-plane register lookahead and one `ld.global.cg.v4.u32` fetch per
lane.

Gate and Up stay warp-local through exact BF16-RNE SiLU and multiplication.
The resulting M64N512 product is quantized directly to the production signed
A4/K512 Down-input ABI, so there is no global BF16 product seam or Gate shared
exchange.

The release resource contract is:

```text
threads/CTA       512
registers/thread  126
dynamic shared    99,584 bytes
active CTA/SM     1
stack/local/spill 0 / 0 / 0 bytes
```

The implementation passed bit-exact K512, K1536, and K5120 comparison,
P513 tail and guard coverage, alias rejection, nondefault-stream execution,
and two CUDA Graph replays.  SASS proved the intended `LDG.E.128.STRONG.GPU`,
`LDSM.16.M88.4`, and adjacent `IMMA.16864.S4.S4` path.  Synthetic inputs had
correctness and smoke authority only; they did not participate in the
performance decision.

## Real OpenAI API and external EvalScope result

Both routes use the same real checkpoint and authenticated K256 Attention and
K512 MLP weights.  The direction test uses one natural P1804 warmup, one
natural P1853 measured request, one generated token, concurrency one, and
external EvalScope.  The candidate request proves the intended mutually
exclusive route:

```text
GateUp M64N8 paired-warp launches          64
GateUp LDSM pair-feed launches              0
Down M128N128 16-warp pair-ring launches   64
Down incumbent pair-ring launches           0
Attention K256 A-exchange B4 launches     128
```

| Route | P1853 TTFT | Total throughput | Server Prefill |
|---|---:|---:|---:|
| Production LDSM pair-feed | 1,916.24 ms | 967.4983 tok/s | 1,911.30 ms |
| M64N8 paired-warp register pipeline | 2,394.33 ms | 774.3157 tok/s | 2,389.34 ms |
| Change | +478.09 ms (+24.95%) | -193.1826 tok/s (-19.97%) | +478.04 ms (+25.01%) |

The candidate warmup server Prefill was 2,381.85 ms.  Both warmup and measured
requests passed Gate, Down, Attention, publication-authentication, and dynamic
GDN accounting contracts.

## Request-scoped NSys attribution

NSys captured request index two through the same real API route.  Model and
sidecar loading occurred outside `cudaProfilerStart`/`cudaProfilerStop`.
Profiler-perturbed EvalScope latency is not a performance verdict; kernel time
is used only to attribute the already-negative real-path result.

| Kernel family | Production profile | Candidate profile | Change |
|---|---:|---:|---:|
| Gate+Up | 722.001952 ms | 1,184.924096 ms | **+462.922144 ms (+64.12%)** |
| Down16 | 288.796640 ms | 287.046656 ms | -1.749984 ms (-0.61%) |
| Attention projections | 404.839904 ms | 405.036224 ms | +0.196320 ms (+0.05%) |
| All GPU kernels | 1,911.120448 ms | 2,371.682560 ms | **+460.562112 ms (+24.10%)** |

Gate+Up explains 100.51% of the total kernel-time regression; every other
kernel family combined improved by 2.36 ms.  The failure is therefore the new
Gate+Up residency and schedule, not a selector miss, Down regression,
Attention change, GDN change, model loading, or EvalScope noise.  The checked
in `kernel-top20.csv` records the candidate's first 20 NSys kernel families.

The direct paired-B load removed a shared presentation, but it left only one
CTA/SM to cover global-B dependency latency and serialized four M64N128 cells
behind one M64N512 owner.  Same-warp Gate/Up ownership alone is not enough to
make that latency structure competitive with staged pair-feed.

## Evidence identity

```text
performance result root
  /home/rm01/q3x-results/paired-warp-p2k-evalscope1-run1

server ELF SHA256
  1365747c888d45c59eae0f3a675b9f37e43a518eb3ba998c897614bf6bbe94a2
benchmark summary SHA256
  d39355b457ad805b03ec5f560704515a182b3c4520446dcc856b53cb4c8597e1
server log SHA256
  249377273e38962f79aeb0fc1adfd811c1176a1ae0242cc16a95bd06e6b5b05c
provenance SHA256
  45393dae4f0dc0aed609581e99e962399548aef3035b70571a9d29c8ea3bbd19
EvalScope stdout SHA256
  3a5494c449e38f88c23d2898835b4d15bc6a5d2bba48819b0633aa86bc2218df

NSys report
  /home/rm01/q3x-results/paired-warp-p1853-candidate.nsys-rep
NSys report SHA256
  a54dbf5cae811dd0461f39349096b2e91efc3161db98646d4a15299f45e8564c
NSys SQLite SHA256
  eed8d7d7adb839ed1797695ebe7557d8dfc069a34fb2ee2823dc6559c3c2816d

production NSys report
  /tmp/q3x-gateup-ldmatrix-pairfeed-p1853-profile-64d198a.nsys-rep
production NSys report SHA256
  80aecad0ab6941e5a36860eac6d133b2ab18daa699cf19760a9240d877a02244
```

## Carry-forward

Keep the LDSM pair-feed production baseline unchanged.  Preserve this
default-off implementation and authenticated paired publication as a
reproducible correctness and negative-architecture record.

The next step is not another M64N512 variant.  Re-profile the current
pair-feed Gate+Up and Down16 kernels with the real P1853 activation and
weights, using valid single-pass or kernel-replay NCU metric groups.  Compare
Gate, Up, and Down separately against an isolated reference-only large-M GEMM
binary; cuBLASLt remains forbidden in the production ELF.  The resulting
occupancy, tensor-pipe, stall, shared-bank, L2, and DRAM gaps select a new
large-M skeleton with a different residency model.  Its first timing verdict
remains the real OpenAI API plus external EvalScope P1853 direction gate.
