# SM87 A4W4 `ldmatrix` operand-feed P0

This is an isolated, default-off mapping proof. It does not add a production
selector and it is not a performance claim.

## Question

Can the established XOR-16B shared layout feed one native
`mma.m16n8k64.row.col.s32.s4.s4.s32` without scalar LDS reconstruction?

The probe stages one physical K64 plane selected from the exact production
K512 consumer layout `[K64=8][outer=64][packed-K64=32B]`, then compares four
independent operand combinations:

- scalar A + scalar B: 4 + 2 `LDS`;
- LDSM A + LDSM B: one `ldmatrix.x4` + one `ldmatrix.x2`;
- scalar A + LDSM B: 4 `LDS` + one `ldmatrix.x2`;
- LDSM A + scalar B: one `ldmatrix.x4` + 2 `LDS`.

Every path exports every lane's four raw S32 accumulator registers. Testing
only the first two paths is insufficient: matching A/B K permutations preserve
the dot product and can falsely pass. The two mixed paths independently pin A
and B against the scalar mapping. The CUDA correctness test compares all four
sets of 128 words bit-for-bit and against a CPU oracle for all eight K64
planes. Inputs include `-8`, `-7`, `0`, and `7`, plus a nonuniform K512
code-block pattern in the real consumer ordering.

## Production boundary

The combined A-x4/B-x2 probe establishes mappings needed by two different
future skeletons; it does not prescribe the same B path for both:

- Gate: migrate only A to `ldmatrix.x4`; keep the admitted paired-B direct
  `.ca.v4` feed from the fragment-native M128N64 one-CTA skeleton.
- Down: A `ldmatrix.x4` plus B `ldmatrix.x2` is the intended operand-feed
  basis.

## Binary gate

The host-only audit isolates all four kernels and requires exactly:

- scalar/scalar: 6 `LDS`, no `LDSM`;
- LDSM/LDSM: 0 `LDS`, one `LDSM.16.M88.4`, one `LDSM.16.M88.2`;
- scalar-A/LDSM-B: 4 `LDS`, no x4, one x2;
- LDSM-A/scalar-B: 2 `LDS`, one x4, no x2;
- every path: one `IMMA.16864.S4.S4` and zero `PRMT`, `SHFL`, `LDL`, or
  `STL`;
- 768 bytes static shared memory and zero stack/local storage.

Audit outputs are persisted in the build tree under
`sm87_a4w4_ldmatrix_operand_probe_audit/`.

## Build and validation

```bash
cmake -S . -B /tmp/q3x-ldmatrix-probe-build \
  -DBUILD_TESTING=ON \
  -DQ3X_BUILD_SM87_A4W4_LDMATRIX_OPERAND_PROBE=ON
cmake --build /tmp/q3x-ldmatrix-probe-build \
  --target q3x_sm87_a4w4_ldmatrix_operand_probe_test -j2
ctest --test-dir /tmp/q3x-ldmatrix-probe-build \
  -R sm87_a4w4_ldmatrix_operand_probe_binary_contract \
  --output-on-failure
ctest --test-dir /tmp/q3x-ldmatrix-probe-build \
  -R sm87_a4w4_ldmatrix_operand_probe_correctness \
  --output-on-failure
```

The final two commands are deliberately separate: the binary audit does not
use the GPU; the correctness test takes the shared GPU lock.

## Result

On CUDA 13.3 / SM87, both registered tests pass:

- scalar control SASS: 6 `LDS`, 1 `IMMA`, 26 registers, 768B shared, zero
  stack/spill/local;
- LDSM candidate SASS: 1 `LDSM.16.M88.4`, 1 `LDSM.16.M88.2`, 1 `IMMA`,
  26 registers, 768B shared, zero `PRMT`/`SHFL`/`LDL`/`STL` and zero
  stack/spill/local;
- scalar-A/LDSM-B SASS: 4 `LDS`, 1 `LDSM.16.M88.2`, 1 `IMMA`, 26
  registers, 768B shared, zero stack/spill/local;
- LDSM-A/scalar-B SASS: 2 `LDS`, 1 `LDSM.16.M88.4`, 1 `IMMA`, 22
  registers, 768B shared, zero stack/spill/local;
- correctness: scalar/scalar, LDSM/LDSM, both mixed paths, and CPU-oracle S32
  fragments are bit-identical for both input families, all eight K64 planes,
  all 32 lanes, and all four accumulator registers per lane; output guards
  remain intact.

The two mixed-oracle passes independently prove the A-x4 and B-x2 mappings;
the result no longer relies on permutation invariance of a fully replaced dot
product.

This admits the fragment mapping only. The next decision gate is a full
production-shaped skeleton followed by real-checkpoint/OpenAI-API/EvalScope
measurement; this probe has no timing result and must not be used as one.
