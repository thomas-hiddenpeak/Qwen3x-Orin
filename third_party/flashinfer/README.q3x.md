# FlashInfer source subset

This directory contains the transitive header closure used by the Qwen3.6
SM87 single-request prefill-attention specialization.  It is vendored from
`flashinfer-python` 0.6.12 and is compiled into `q3x_kernels`; no Python,
cubin, or runtime FlashInfer dependency is required.

- Upstream: <https://github.com/flashinfer-ai/flashinfer>
- Package: `flashinfer-python==0.6.12`
- Source wheel SHA-256:
  `09762e5a4a9cf40804a90c26d50a679de0fb362f4547457b12ea0c19a2f11aef`
- Consumer:
  `src/kernels/reference/full_attention_c512_register_pipeline.cu`
- Build admission: `Q3X_BUILD_FLASHINFER_PREFILL_ATTENTION_ADMISSION=ON`
- Runtime admission: `Q3X_FULL_ATTENTION_FLASHINFER_DIRECT=1`
- License: Apache-2.0; upstream and bundled third-party license texts are
  preserved in this directory.

The subset was obtained from the wheel's
`flashinfer/data/include/flashinfer` tree.  Keep the version and source hash
fixed when changing this closure so a clean build cannot silently select
headers from a host Python environment. `MANIFEST.sha256` records the exact
vendored header and license contents relative to this directory.
