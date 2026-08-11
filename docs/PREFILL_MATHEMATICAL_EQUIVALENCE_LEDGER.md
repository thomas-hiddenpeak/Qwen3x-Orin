---
q3x_document:
  id: q3x-prefill-mathematical-equivalence-ledger
  class: normative
  status: active
  owner: prefill-maintainers
  authority: Prefill mathematical equivalence, finite-precision identity, and production-observable liveness
  effective: 2026-08-11
  last_reviewed: 2026-08-11
  supersedes: []
  superseded_by: []
  ssot_for: Prefill architecture proof order, equivalence proof classes, P40 arithmetic ledger, and liveness-deletion eligibility
  review_trigger: model shape, numerical route, state ABI, API observable, or reference-arithmetic change
---

# Prefill mathematical-equivalence ledger

## 1. Authority and scope

This document is the normative ledger for deciding whether a Prefill
transformation preserves the pinned runner's mathematical and observable
contract. It refines the architecture-first and no-accuracy-regression rules
in the [engineering constitution](ENGINEERING_CONSTITUTION.md) and the
Prefill state boundary in
[the Prefill subsystem design](PREFILL_ARCHITECTURE_RESET.md).

It does not select the active work package, report implementation status, or
change a performance target. Delivery order belongs to
[`ROADMAP.md`](ROADMAP.md), implementation and qualification state belong to
[`CURRENT_STATUS.md`](CURRENT_STATUS.md), and measurement authority belongs
to [`REAL_MODEL_PERFORMANCE_POLICY.md`](REAL_MODEL_PERFORMANCE_POLICY.md).

The locked proof vehicle in this ledger is the pinned Qwen3.6-27B NVFP4 model
on SM87, with the P40 workload defined by:

| Symbol | Meaning | Value |
| --- | --- | ---: |
| `M` | prompt tokens | 40,000 |
| `H` | hidden width | 5,120 |
| `I` | MLP intermediate width | 17,408 |
| `L` | decoder layers | 64 |
| `L_gdn` | linear-Attention/GDN layers | 48 |
| `L_full` | full-Attention layers | 16 |
| `N_qg` | full-Attention Q plus gate output width | 12,288 |
| `N_q` | processed Q / Attention output width | 6,144 |
| `N_kv` | K or V width | 1,024 |

Any change to those shapes, the layer schedule, a dtype, a rounding point, the
Prefill-to-Decode ABI, or an API observable invalidates the affected row of
this ledger until it is rederived.

## 2. Mandatory architecture work order and proof layers

Every Prefill architecture candidate and local work package follows this
order before CUDA details are allowed to select the design:

1. derive the live computation over real numbers and prove each proposed
   algebraic rewrite or deletion in that domain;
2. bind decoded operand bits, scale placement, accumulation and reduction
   trees, dtypes, rounding, publication, and recurrent-state boundaries;
3. derive every production observable and the complete lifetime, owner,
   alias set, and reuse event of every live data, control, and workspace
   value; and
4. map only that proved live finite-precision graph to SM87 layouts,
   CTA/warp ownership, residency, pipelines, fusion, buffering, streams, and
   synchronization.

The fourth step is an engineering mapping, not an equivalence proof. If the
mapping needs a different reduction tree, publication point, state lifetime,
or observable, it returns to the applicable earlier proof layer and receives
a new numerical or route identity. A profiler may explain the selected
mapping after a real-route result; it may not reverse this order.

A transformation is not admitted by one undifferentiated claim of
"equivalence." It must state and satisfy the applicable layers below in
order. Proof obligations apply only to the live subgraph, but every value
classified as dead requires an independent production-observability proof.

### 2.1 Layer A: real-number equivalence

Real-number equivalence proves that the transformed equations compute the
same mathematical function over exact real values. It can justify such
directions as:

- concatenating Gate and Up as independent column ranges of one projection;
- composing online-softmax block summaries;
- composing unrounded affine recurrent transitions;
- moving a common real scale across a sum; or
- deleting a graph node whose value has no path to a production observable.

This layer is necessary for an algebraic rewrite, but it is never sufficient
for production. It says nothing about accumulator parenthesization, overflow,
underflow, signed zero, NaN behavior, quantized decode, dtype conversion,
rounding, publication, or state-transition timing.

### 2.2 Layer B: finite-precision and reduction-tree equivalence

Strict finite-precision equivalence requires the same live output bits for
every admitted input. The proof must bind all of the following:

- the exact decoded operand bits and special-value domain;
- tensor-global and block-scale application points;
- K-fragment order and ownership;
- every within-warp, cross-warp, split-K, and cross-CTA reduction parent;
- FP32 MMA accumulation and epilogue operation order;
- every BF16 publication and round-to-nearest-even boundary;
- residual, activation, normalization, and gate ordering; and
- overflow, underflow, NaN, infinity, and signed-zero behavior.

Preserving a real-number dot product while changing its reduction tree is a
new finite-precision route. It is not the same route merely because both use
FP32 accumulation or produce a correctly rounded final value. Such a route
remains `accuracy-unqualified` until the separately declared numerical and
no-regression contract passes; it may not borrow the incumbent's exact
identity, receipt, oracle result, or production qualification.

An implementation may use a different physical schedule without changing the
finite-precision tree only when the schedule still performs the same ordered
updates and publications. A proof by output sampling alone is insufficient
for calling that implementation strictly equivalent.

### 2.3 Layer C: production-observable liveness equivalence

Production-observable liveness asks which internal values can affect the
delivered API result or any state consumed by later Decode. A value may be
deleted only when the complete dependency graph proves that it reaches none
of the following:

- committed KV, recurrent, convolution, position, or RoPE state;
- the final representation, logits, selected token, or requested output;
- a lock, counter, barrier, completion word, or other control state whose
  value governs a later live operation;
- route, completion, cancellation, failure, or resource semantics; or
- any other field exposed by the admitted API mode.

Liveness equivalence is allowed to leave dead internal storage different from
a full-materialization oracle. It must not hide an observable mismatch by
calling the value dead. The candidate must preserve the incumbent
finite-precision function on every live path and prove that writes to dead
destinations are either absent or confined to an explicitly dead scratch
range.

The liveness set is API-contract-specific. Prompt logprobs, per-token logits,
hidden-state export, a training interface, or another feature that observes
earlier final-layer rows would make those rows live. A plan specialized for
ordinary next-token generation must fail closed rather than silently execute
when such an observable is requested.

A cross-phase control or workspace value must have one stable physical owner
for its complete live interval. A non-null address and a successful initial
clear do not prove this condition: every overlapping alias writer between the
clear and the last consumer must be excluded by construction or ordered and
re-established by the plan. In particular, a lock array whose zero state is
consumed by later split-K tails cannot reside in a family overlay that an
intervening Attention, GDN, projection, or residual phase may overwrite.
Receipts attest this lifetime and alias exclusion, not merely allocation.

### 2.4 Engineering mapping obligation

After Layers A--C are declared, the implementation maps the irreducible live
graph to hardware. The mapping record names operand residence and reuse,
physical buffer owners and lifetimes, communication volume, CTA/warp and
reduction ownership, pipeline stages, producer/consumer overlap, and each
required synchronization edge. Kernel names, tile sizes, cache operators,
stage counts, and double/triple buffering are consequences of that record;
they are not substitutes for it.

### 2.5 Classification rule

Every proposed transformation records one of these exact statuses:

| Status | Meaning |
| --- | --- |
| `real-equivalent-only` | Layer A is proved; finite-precision or observability obligations remain open |
| `finite-precision-equivalent` | Every live arithmetic output is bit-equivalent, but a new liveness or handoff claim remains open |
| `observable-state-equivalent` | Live arithmetic and the complete production handoff are equivalent for the declared API mode |
| `accuracy-unqualified` | The route intentionally changes a finite-precision tree or other numerical boundary and has not passed its separate qualification |
| `production-eligible` | The applicable three layers, route receipts, real-model oracles, and release gates all pass |

No status in this table changes production dispatch by itself.

## 3. Pinned finite-precision projection contract

Let `R_bf16` denote BF16 round-to-nearest-even. Let `Acc_fp32^K` denote the
complete declared K-fragment order, K-slice ownership, cross-CTA reduction
tree, and FP32 MMA accumulation. Let

```text
D4(q, s) = E2M1(q) * E4M3FN(s)
```

be the current NVFP4 per-value decode, with one block scale for each group of
16 weights. Exhaustive enumeration of all 16 E2M1 codes and all 254 finite
E4M3FN codes shows that each finite product is exactly representable in BF16,
including signed zero. The nominal BF16 conversion is therefore an identity
on that finite-code domain. This fact does not authorize the two E4M3FN NaN
encodings; a production artifact must validate that the checkpoint contains
no forbidden NaN scale code and preserve the declared special-value path.

For projection role `r`, authenticated tensor-global scale `a_r`, activation
`A`, packed code `Q_r`, and block scale `S_r`, the pinned publication is:

```text
P_r(A, Q_r, S_r)[m,n]
  = R_bf16(
      a_r * Acc_fp32^K(
        A[m,k] * D4(Q_r[n,k], S_r[n,floor(k/16)])
      )
    )
```

The exact MLP boundary is:

```text
G = P_gate(A, Q_gate, S_gate)
U = P_up(A,   Q_up,   S_up)
H_act[m,n] = R_bf16(SiLU(float(G[m,n])) * float(U[m,n]))
D = P_down(H_act, Q_down, S_down)
R_next[m,n] = R_bf16(float(D[m,n]) + float(R[m,n]))
```

This contract yields the following transformation rules:

1. Gate and Up may share A movement or one concatenated output traversal only
   while each keeps its independent weights, scales, K reduction, BF16
   publication, and SiLU/Up consumer semantics.
2. Down may fuse its residual consumer only after the Down value has passed
   the same tensor-global scaling and BF16 publication boundary. Moving the
   residual add across that point is not strictly equivalent.
3. Reusing a decoded BF16 weight is legal only inside a lifetime where the
   decoded word, block scale, special-value semantics, and consumer order are
   identical.
4. The real identity `sum(A_i * q_i * s) = s * sum(A_i * q_i)` does not move
   `s` outside a K16 dot in the finite-precision contract. That rewrite
   changes the BF16 products presented to MMA and usually changes the FP32
   accumulator tree.
5. A power-of-two scale is only a sufficient subcondition when all transformed
   operands and products remain exact, no exceptional path changes, and the
   original ordered FP32 updates are reproduced. It does not by itself delete
   the dot product.

The same principle governs FP8 W8A16 projection: preserving the real matrix
product is not enough; the route must bind exact E4M3FN decode, tensor scale,
accumulator tree, and BF16 publication.

## 4. Attention and GDN/SSM equivalence obligations

### 4.1 Exact causal Attention

In real arithmetic, a KV block can be summarized by online-softmax state
`(maximum, denominator, weighted_value)`, and block summaries can be composed
after rescaling to the combined maximum. That monoid is an algorithmic
opportunity, not a finite-precision proof.

An exact production Attention rewrite must additionally preserve:

- the causal membership of every query/key pair;
- Q, processed K, V, gate, and RoPE publication bits;
- maximum, exponential, denominator, weighted-value, and final-normalization
  order and dtype;
- the declared parallel reduction tree or an independently qualified new
  numerical route; and
- complete KV publication before the corresponding consumer is attested.

FlashAttention and FlashInfer are reference dataflows. Their familiar real
formula does not let a candidate inherit the incumbent's exact identity when
their parallel reduction order differs.

### 4.2 Per-token BF16 GDN/SSM state

For one GDN value row, normalized key `k_t`, value `v_t`, decay `alpha_t`, and
update gate `beta_t`, the unrounded real transition can be written:

```text
s_t = s_(t-1) (alpha_t I - alpha_t beta_t k_t k_t^T)
      + beta_t v_t k_t^T
```

Unrounded affine transitions compose hierarchically. The pinned production
semantics do not expose that unrounded associative monoid directly. They are:

```text
u_t        = F_fp32(s_(t-1)^bf16, x_t)
y_t        = G_fp32(u_t, x_t)          # consumes the unrounded update
s_t^bf16   = R_bf16(u_t)               # published after every token
```

Therefore a FlashLinearAttention WY/chunk construction, Mamba selective scan,
or another block recurrence is production-equivalent only if it proves, for
every token in order:

1. the incoming BF16 state bits equal the incumbent's state bits;
2. the FP32 update and token output consume operands in the declared order;
3. the emitted token output matches before the state is rounded;
4. the outgoing state is rounded to BF16 at the same token boundary and has
   identical bits;
5. convolution history, gates, normalized Q/K, and final boundary state are
   identical; and
6. chunking, cancellation, and handoff never expose an unrounded or partially
   advanced state.

Deferring BF16 state publication to the end of a chunk is not equivalent,
even when the unrounded affine block formula is exact over real numbers. FLA
and Mamba remain valuable references for chunk-local preparation, WY
factorization, scan structure, and fusion around the serial boundary; they do
not waive the per-token BF16 state proof. A path that deliberately changes
this semantic is a separately identified, accuracy-unqualified research
route and cannot enter the current production mainline.

### 4.3 GDN dependency width and state-residency lower bound

The recurrence is token-serial only within one value-head state chain. It is
not one request-wide scalar dependency. The fixed model has 48 independent
value heads of dimension 128. Every three consecutive value heads share one
Q/K head, yielding 16 independent QK groups. One value-head BF16 state is
`128 * 128 * 2 = 32,768` bytes; one three-value-head QK group owns 96 KiB of
live recurrent state.

Across all 48 GDN layers and P40, the strict recurrence body contains about
`4,529,848,320,000` MACs, or `9,059,696,640,000` conventional operations.
Reading and rewriting every complete BF16 state through global memory after
every token would move:

```text
48 layers * 40,000 tokens * 48 value heads
  * 128 * 128 BF16 values * 2 directions
  = 6,039,797,760,000 bytes
```

This is a mathematical lifetime result, not a cache hint: a competitive exact
route must keep each active state chain on chip across as many ordered token
updates as the per-token BF16 boundary permits. Parallel work is available
across the 48 value heads, across the 16 QK groups, and in Q/K normalization,
gate construction, chunk-local preparation, and output consumers. The only
irreducible serial edge is `s_(t-1)^bf16 -> s_t^bf16` within one value-head
chain. A CUDA design that serializes more than that has introduced an
engineering dependency absent from the model; one that serializes less has
changed the finite-precision function unless separately qualified.

### 4.4 Exact finite-domain nonlinearities

When a nonlinear consumer receives a value only after BF16 publication, its
input domain has exactly 65,536 bit patterns. A complete BF16-bit-indexed FP32
table for one device expression therefore occupies 256 KiB. For P40, the
fixed graph applies the relevant nonlinearities at these exact multiplicities:

| Boundary | P40 evaluations |
| --- | ---: |
| MLP Gate SiLU | 44,564,480,000 |
| GDN Z SiLU | 11,796,480,000 |
| Full-Attention output gate sigmoid | 3,932,160,000 |

Such a table is strictly eligible only when it is generated from the same
device expression and an exhaustive device oracle proves every FP32 output
bit for all 65,536 inputs, including every NaN encoding, infinities, subnormal
values, and signed zero. This finite-domain closure proves the arithmetic
substitution; it does not prove a speedup. The retained Decode experiment
already shows that one extra global lookup can merely exchange SFU work for
memory traffic and be neutral or negative. Consequently lookup reuse may be
placed inside a larger projection/Attention/GDN residency design, but it is
not an independent production candidate or a reason to start another local
parameter scan.

## 5. Quantitative P40 arithmetic ledger

The figures below count conventional operations as two operations per MAC.
They describe model work and candidate scope; they are not measured latency,
an SM87 roofline, or permission to lower the project target.

### 5.1 Projection work

| Family | Exact expression | Operations per prompt token | P40 operations |
| --- | ---: | ---: | ---: |
| NVFP4 Gate, Up, Down across 64 layers | `2 * 3 * 64 * H * I` | 34,225,520,640 | 1,369,020,825,600,000 |
| Linear-layer FP8 QKV, Z, O across 48 layers | role-shape sum | 11,072,962,560 | 442,918,502,400,000 |
| Full-layer FP8 Q, K, V, O across 16 layers | role-shape sum | 3,355,443,200 | 134,217,728,000,000 |
| Linear-layer BF16 A and B across 48 layers | `2 * 48 * 2 * H * 48` | 47,185,920 | 1,887,436,800,000 |
| **All listed projections** | sum | **48,701,112,320** | **1,948,044,492,800,000** |

Each Gate, Up, or Down role in one P40 layer contains exactly
`M * H * I = 3,565,158,400,000` MACs. The three roles across all 64 layers
contain `684,510,412,800,000` MACs, or
`1,369,020,825,600,000` conventional operations.

The 16 full-Attention layers' causal QK and PV work adds
`314,580,664,320,000` conventional operations at P40, excluding softmax,
gate application, normalization, and data movement. The comparison base used
for the final-layer liveness row is therefore:

```text
all listed projections + causal QK/PV
  = 2,262,625,157,120,000 conventional operations
```

GDN, softmax, normalization, gate application, embedding, logits, movement,
decode, launch, and synchronization cost remain outside this denominator.

### 5.2 Dense-BF16 planning proxy is not a target

At a published approximately 43-TFLOP/s dense BF16 Tensor Core rate, the MLP
operation count alone has an idealized floor of about 31.84 seconds, or about
1,256 prompt token/s at P40. All listed projections produce an idealized
45.30-second / 883-token/s proxy before Attention and GDN.

These values identify an execution-class problem; they are not a project or
hardware ceiling. The project owner's observed vLLM starting line of about
4.3K prompt token/s remains the planning constraint. At 4.3K token/s, P40 has
about 9.30 seconds of total Prefill budget, and the MLP operations alone would
represent about 147.17 effective TOP/s if lowered one-for-one to dense BF16
MMA. A conflict triggers an audit of arithmetic class, executed model work,
cache state, token accounting, backend, and route identity. It never lowers
the 4.3K target.

### 5.3 Exact arithmetic-class qualification gate

An order-of-magnitude successor cannot be selected from a nominal INT4, INT8,
BF16, or sparse peak. It must first prove how the real P40 operand domain maps
to the proposed execution algebra. For an exact block-floating, integer-limb,
or bit-plane route, the pre-kernel witness records per projection boundary:

- K16 and K64 activation exponent spans and aligned-significand trailing-zero
  distributions;
- the minimum exact signed-INT8 and signed-INT4 limb count per tile;
- residual-plane density and exact 2:4-encodable share;
- the fraction of tiles that preserve the incumbent ordered FP32 partials,
  scale rejoin, special-value path, and BF16 publication bits;
- the exact-fallback fraction; and
- the resulting average physical MMA pass count, including fallback and
  representation overhead.

The proposal advances to CUDA only when that measured pass count can fit the
5.0-second P40 projection allocation and, if extended to QK/PV, the
1.8-second Attention allocation. Otherwise the arithmetic class closes before
kernel implementation. This is distinct from approximation: calibrated W4A4,
truncated planes, changed reduction parents, or delayed state rounding remain
accuracy-unqualified regardless of their nominal peak rate.

## 6. Layer 63 production-liveness deletion

### 6.1 Dependency proof

The fixed layer schedule makes layers `3, 7, ..., 63` full-Attention layers.
All prompt-token hidden rows are live through layer 62 because a later layer
can consume them. Layer 63 has no later Prefill layer.

For ordinary next-token generation:

- Decode needs layer 63 K and V for every prompt position, so the complete
  layer-63 input normalization, K projection, K normalization/RoPE, V
  projection, and KV publication remain live for all `M` rows.
- The first output token depends on only the final prompt row after layer 63.
  Q plus gate, the causal query result, O projection, attention residual,
  post-Attention normalization, MLP, final normalization, and logits are
  therefore live only for row `M-1`.
- Layer-63 hidden outputs for rows `[0, M-1)` reach neither the committed
  Prefill-to-Decode ABI nor the ordinary generation API and are dead under
  this exact API mode.

The legal execution graph is consequently:

```text
layer62 hidden[0:M]
  -> input norm[0:M]
  -> K/V[0:M] -> processed K/RoPE[0:M] -> committed layer63 KV[0:M]

layer62 hidden[M-1]
  -> Q+gate[M-1] -> processed Q/gate(position=M-1)
  -> causal Attention(Q=1, KV=M)
  -> O/residual[M-1]
  -> post norm + MLP[M-1]
  -> final norm -> logits/token
```

### 6.2 Deletable work

The deletable prefix has `M-1 = 39,999` rows. Exact P40 counts are:

| Deleted layer-63 work | MAC expression | Deleted MACs |
| --- | ---: | ---: |
| Gate, Up, Down MLP | `(M-1) * 3 * H * I` | 10,695,207,813,120 |
| Q plus gate projection | `(M-1) * H * N_qg` | 2,516,519,485,440 |
| O projection | `(M-1) * N_q * H` | 1,258,259,742,720 |
| Causal QK and PV for earlier queries | `M*(M-1)/2 * 24 * (256+256)` | 9,830,154,240,000 |
| **Total** | sum | **24,300,141,281,280** |

This is `48,600,282,562,560` conventional operations, or
`2.1479599663%` of the `2,262,625,157,120,000`-operation comparison base.
Under a purely proportional-cost thought experiment that corresponds to at
most about `1/(1-0.0214796) = 1.02195x` throughput. Actual latency contribution
must be measured because kernels have different efficiencies and non-arithmetic
costs.

This is a useful unconditional deletion, but it is only an approximately
2.15% arithmetic-scope bound. It is not the complete P40 architecture, not a
4.3K solution, and not a reason to lower the 4.3K vLLM starting line. It also
does not delete layer-63 K/V, input normalization, final-row work, or any
earlier layer.

### 6.3 Mandatory observable set

A conforming implementation preserves and qualifies:

- all 48 linear-layer convolution and GDN/SSM final states;
- all 16 full-Attention K/V caches for every prompt position, including the
  complete processed layer-63 K and V;
- consumed token IDs/count, final position, sequence length, and RoPE state;
- the layer-63 final-row residual and final normalized hidden state;
- the requested logits or greedy token and generated output behavior;
- route identity, completion, cancellation, failure, and resource evidence;
  and
- the atomic rule that Decode sees nothing before `PrefillStateCommitted`.

The candidate must not compare earlier layer-63 output rows with an incumbent
that materializes them, because those are intentionally absent. It should
instead prove that the dead prompt-residual prefix remains at its layer-62
value and that no M1 kernel writes outside its declared destination.

## 7. Required plan, receipt, and oracle contracts

Layer-63 liveness is a topology change, not a hidden optimization inside an
existing whole-P40 tactic. An implementation must remain default-off until it
has all of the following.

### 7.1 Execution plan and progress

The bound plan has a distinct terminal-layer liveness identity and records at
least:

- terminal layer `63` and final row `M-1`;
- input/K/V rows `M`;
- Q, gate, Attention-output, O, post-norm, MLP, and final rows `1`;
- query position `M-1`, KV causal end `M`, and full-KV-before-query ordering;
- exact last-row write ranges and the dead residual prefix;
- a terminal progress transition that separately attests complete KV
  publication and final-row completion; and
- exact, native-only, non-MTP, no-cuBLASLt, no-fallback boundaries.

The implementation may retain the existing full-capacity P40 arena initially.
Capacity is not evidence that all rows were produced. Progress helpers and
phase counters may not claim five full-M drain panels or one full-P40 MLP when
the terminal route actually executed one M1 drain and one M1 MLP.

### 7.2 Role receipts and physical witness

The existing logical Q/K/V/O, Attention, Gate/Up, Down, residual, norm, and
handoff roles remain present. Their terminal physical shape must be
independently attested. Receipts bind:

- terminal K and V over all `M` rows, including processed-K publication;
- terminal Q/gate source row, output row count, and absolute RoPE position;
- terminal Attention with `query_m=1`, `kv_m=M`, `causal_end=M`, and
  `output_m=1`;
- terminal O and residual with a last-row-only write set;
- terminal post-norm, Gate+Up+SiLU, and Down+residual at M1; and
- the final M1 normalization/handoff chain.

Each receipt names its artifact, scale/sidecar owners, workspace, completion
domain, numerical contract, and write span. A Decode M1 kernel may be reused
only after its packing, scales, K order, reduction tree, epilogue, BF16
publication, and alias contract are proved equivalent. It may not silently
borrow the existing Prefill receipt.

The deployment-plan identifier and terminal physical counters are new and
versioned. The public logical route can still count one production execution
of each layer-63 role, but the physical witness must distinguish 15
whole-prompt full-Attention executions plus one terminal-query execution, 63
full-P40 MLP executions plus one terminal-M1 MLP, and 315 full drain panels
plus one terminal-M1 drain. Old witness schemas and counts remain byte-stable.

### 7.3 Liveness-aware oracle

A full-state comparator that hashes all layer-63 prompt-residual rows is not
the correct oracle for this candidate. The liveness oracle must:

1. compare every committed convolution/GDN state and every K/V row bitwise;
2. compare final-row Q/gate, Attention output, O/residual, post-norm,
   Gate/Up/activation, Down/residual, final hidden, logits, and token at their
   applicable phase boundaries;
3. compare final position, sequence length, route, progress, cancellation,
   and commit behavior;
4. exclude intentionally unmaterialized layer-63 hidden/scratch prefix rows
   from equality with the full-materialization incumbent;
5. hash the prompt-residual prefix before layer 63 and prove that it remains
   unchanged afterward; and
6. run guard and dual-poison tests showing that unused scratch rows cannot
   affect any observable and that M1 kernels do not write out of range.

Matching one generated token is a transport smoke result, not this oracle.
Likewise, a liveness-aware oracle does not qualify an already different
Attention reduction tree; that route retains its separate accuracy boundary.

## 8. Candidate ledger template

Every equivalence-derived work package records these fields before local
parameter work begins:

| Field | Required content |
| --- | --- |
| `transformation_id` | stable candidate and lineage identity |
| `api_mode` | exact observable contract for which liveness is claimed |
| `source_equation` / `target_equation` | mathematical functions before and after the rewrite |
| `real_equivalence_proof` | symbolic argument and admitted input domain |
| `live_set` / `dead_set` | dependency path to every handoff/API observable |
| `finite_precision_contract` | decoded bits, scales, K order, reduction tree, dtypes, rounding, special values |
| `state_transition_contract` | per-token and final state publication semantics |
| `plan_and_receipts` | tactic, layer/shape, artifacts, workspaces, write sets, completion and fallback boundaries |
| `oracles` | synthetic exhaustive, real-checkpoint, real-state, and API/state comparisons |
| `quantitative_scope` | exact removed MACs/bytes/launches and denominator |
| `status` | one classification from section 2.5 |
| `target_effect` | named Prefill/API budget; never a silently revised product target |

A profiler trace can explain how a proved transformation maps to SM87. It
cannot replace the proof, expand the dead set, or turn a different reduction
tree into an exact route.

## 9. Non-conclusions and change control

This ledger does not establish that the present P40 implementation is
production-qualified, that its whole-prompt FlashInfer reduction is exact, or
that the current performance target is unreachable. In particular:

- the layer-63 deletion is approximately 2.15% of the stated arithmetic base,
  not the missing order-of-magnitude solution;
- the 43-TFLOP/s dense-BF16 proxy identifies a need to inspect execution and
  arithmetic class; it is not an SM87 or project upper bound;
- FLA, Mamba, FlashAttention, FlashInfer, Triton, vLLM, and Humming are
  reference implementations and existence proofs, not automatic numerical
  qualification or production dependencies; and
- no algebraic observation changes the locked cold/no-cache 40K--60K API
  target, the approximately 130K target, the 4.3K vLLM starting line, the
  no-accuracy-loss rule, non-MTP scope, or cuBLASLt exclusion.

Amend this ledger whenever a model dimension, layer schedule, API observable,
state dtype, per-token rounding boundary, quantized decode, scale placement,
reduction tree, or `PrefillStateCommitted` field changes. Preserve superseded
rows as historical evidence rather than silently reusing their proof status.
