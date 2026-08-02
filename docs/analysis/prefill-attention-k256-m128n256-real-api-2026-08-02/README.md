# K256 M128N256 Attention real-API promotion (2026-08-02)

## Decision

The structural K256 Attention route is **retained as the new cumulative
current-best Prefill baseline**.  This is a production-shaped, explicit
admission route: all real Attention projections use one authenticated K256
publication, all four stage families fail closed if the new kernel does not
engage, and the old K128 Attention supermatrix route is excluded.  MLP stays
on the retained K512 GateUp-to-Down edge and Down v1 route.

The promotion decision comes from external EvalScope 1.9.1 through the real
OpenAI `/v1/completions` API and the pinned checkpoint.  Synthetic inputs are
used only for correctness and resource admission.

This is not the Prefill target.  At 843.5446 token/s the runner is only 42.2%
of the 2,000 token/s floor.  On this corpus, reaching that floor requires mean
TTFT near 962.4 ms, another 1,320.2 ms below the promoted result.  Work
therefore moves directly to a shape-specific MLP architecture replacement;
small parameter scans remain out of scope.

## External EvalScope/OpenAI result

The fixed natural P2K corpus has SHA-256
`41ab42aecfbf7157ece82df889df7a38a8f0ba2963b39958409e732a4681d4af`.
Each route runs one warm-up followed by four serial measured requests, with
`max_tokens=1`.  Mean input length is 1,924.75 tokens.

| Route | Success | Mean TTFT | Total throughput |
|---|---:|---:|---:|
| Retained K128 Attention baseline | 4/4 | 2,448.34 ms | 786.4375 token/s |
| K256 M128N256 Attention | 4/4 | 2,282.55 ms | 843.5446 token/s |
| **Change** | - | **-165.79 ms (-6.77%)** | **+7.26%** |

Every natural request improves, so the aggregate result is not caused by one
length or one padding boundary:

| Prompt tokens | K128 TTFT | K256 TTFT | Saving | Change |
|---:|---:|---:|---:|---:|
| 1,792 | 2,237.274 ms | 2,087.304 ms | 149.970 ms | -6.70% |
| 1,853 | 2,406.870 ms | 2,240.520 ms | 166.350 ms | -6.91% |
| 1,906 | 2,417.795 ms | 2,250.921 ms | 166.874 ms | -6.90% |
| 2,148 | 2,731.432 ms | 2,551.468 ms | 179.964 ms | -6.59% |

The tested source commit is `66587a919f5591bd3bcc39bdb5572e1509fb54b5`.
The exact server ELF SHA-256 is
`75512fec72e54a49b83cd21a6244ec601c0af5dd694efad048cda84e7d9b1224`.

## Executed architecture and publication

The Attention consumer is M128N256, 512 threads, 16 warps, one persistent CTA
per SM, and a three-stage combined A/B `cp.async` ring.  One stage is 49,920
bytes and the dynamic allocation is 149,760 bytes; a four-stage ring cannot
fit the pinned SM87 opt-in shared-memory limit.  Every warp owns M16N128 and
keeps 64 FP32 output values.  The fixed projection topologies are:

- Linear Attention: 48 two-Q/two-Z cells and 16 four-Q cells;
- Full Attention Q/K/V: eight two-Q/two-K cells, eight two-Q/two-V cells,
  and 40 four-Q cells;
- Attention output: 20 four-O cells.

The authenticated publication is:

```text
layout          sm87_s4_n64_packed_k64_scale_k256_consumer_v3
payload bytes   12353536000
manifest SHA256 0014dab96dbd6dc5060fe9739bc817fca21dbaf8fd5bdcaa749be0f6a9a535d6
policy SHA256   f345581fd2bec39a10c33a831befafd19ddbd7599f391836eef580b2a6c03717
payload SHA256  cdb3b1f54d0a1f406a0d055eb4fd5cba9f272b8b423b47a8746678e4cdb8f1d7
receipt SHA256  28ea67c00dd21ab95ea884fd0bdfe1d9db2aebcf52d09351d9731c54bf7a9bae
```

The independently authenticated K512 MLP overlay is rebound to that exact
base publication.  Its layout remains
`sm87_s4_n64_packed_k64_scale_k512_mlp_v1`, with payload SHA-256
`541480dcad50227288530b22ed24e5984cef99f513d1df17ede8d4b702a2d5ec`.
Cross-binding K128 and K256 base receipts is rejected.

## Request-scoped NSys attribution

One natural 1,853-token request was captured through the same API.  Model and
sidecar loading are outside the CUDA profiler range.  Profiled server Prefill
falls from 2,411.35 to 2,246.07 ms, a 165.28-ms saving consistent with the
unprofiled paired result.  Profiler timing is attribution only.

| Projection chain | K128 calls | K128 total | K256 calls | K256 total | Saving |
|---|---:|---:|---:|---:|---:|
| Linear Attention pair / topology 0 | 48 | 336.301 ms | 48 | 249.511 ms | 86.789 ms |
| Attention output | 64 | 165.631 ms | 64 | 124.466 ms | 41.165 ms |
| Full Attention Q/K/V / topology 1 | 16 | 98.303 ms | 16 | 73.347 ms | 24.956 ms |
| Attention activation quantizer | 128 | 38.086 ms | 128 | 28.539 ms | 9.547 ms |
| **Complete Attention projection chain** | **256** | **638.320 ms** | **256** | **475.863 ms** | **162.457 ms** |

The complete local chain is 1.3414x faster.  NSys contains exactly 48 topology
0, 16 topology 1, and 64 Attention-O K256 launches: 128 physical launches for
208 logical projections.  The three old K128 Attention supermatrix kernels
are absent.  Gate+Up changes by -0.531 ms and Down by -0.366 ms between the
captures, which isolates the real saving to the intended Attention plane.

The first 20 candidate kernel families are in
`kernel-top20.csv`.  Raw artifacts remain outside Git because they contain
machine-local profiler data:

```text
candidate NSys  /home/rm01/q3x-attention-k256-m128n256-p1853.nsys-rep
SHA256          2112a96f1ee6b1131b3489daca032a624dd20298b5a0317f986849b4dbf35d90
candidate SQL   /home/rm01/q3x-attention-k256-m128n256-p1853.sqlite
SHA256          2199ff0b80e269f410a7f18fba0d725075ba55d62abd7a9d86399e1e13edeae6
baseline NSys   /home/rm01/q3x-mlp-k512-edge-request2.nsys-rep
SHA256          8b72bb7412090fece5bfe6128a5d21b67249729f485be67c2c21554a4490a885
baseline SQL    /home/rm01/q3x-mlp-k512-edge-request2.sqlite
SHA256          0ede8ee3e5d9593cdf2adce63fd57a628efa03d7982b7723816f94491a3ef516
```

## Correctness and fail-closed admission

The CUDA kernel is bit-exact against the CPU reference for all three
topologies, including M129 padded to M256, K768, arbitrary panels, output
guards, immutable inputs, a captured CUDA graph, and two graph replays.  SASS
and resource admission report 128 registers/thread, zero local frame, zero
spill, 149,760 bytes dynamic shared memory, one active CTA/SM, `LDGSTS.128`,
`DEPBAR`, and `IMMA.16864` on the executed cell.

Seven focused CTests pass, including the CUDA correctness test, converter,
overlay binding, runtime host admission, and long-Prefill planner.  The
external harness contract suite also passes 38/38.  Runtime startup requires
400/400 K256 base projections, 192/192 K512 MLP projections, exact payload
hashes, and exact K256 launch accounting; missing or conflicting selectors,
receipts, stages, or hashes fail closed.

## Evidence roots

```text
baseline external result
  /home/rm01/q3x-mlp-k512-edge-evalscope-p2048
candidate external result
  /home/rm01/q3x-attention-k256-m128n256-evalscope-p2048
candidate profiled external run
  /home/rm01/q3x-attention-k256-m128n256-nsys-p2048
```

The candidate summary SHA-256 is
`eb840e93a7ea996841ff914dd2df1f3fe361e1be13cc80e974faa3241678ef52`;
its request database SHA-256 is
`cc38ba226dfa3bc6519aa9f63a59a0c59a40ad6e6ae884a9161d62050de793bf`.

