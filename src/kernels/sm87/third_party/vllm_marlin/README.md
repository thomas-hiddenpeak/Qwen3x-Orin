# Vendored vLLM Marlin kernel core

The CUDA headers in this directory are copied from vLLM commit
`ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb`, under the Apache License 2.0.
They retain the upstream Marlin and Neural Magic copyright and license
headers. The local `core/scalar_type.hpp` is a dependency-free reduction of
vLLM's compile-time scalar-type ABI; it preserves the exact upstream type IDs.

Qwen3x-Orin's admitted production experiment instantiates the SM87 BF16 x
NVFP4 path used by Qwen3.6-27B: scheduler-wide M tiles, four `cp.async` stages,
group size 16, FP8 block scales, and the upstream persistent stripe scheduler.
The separate `Q3X_BUILD_FP8_MARLIN_PREFILL_ADMISSION` compile gate also
instantiates the frozen BF16 x E4M3FN W8A16 templates with channel-wise BF16
scales (`group_blocks=-1`). That second path is test-only and is not connected
to `ModelWeights` or production dispatch. No PyTorch or vLLM runtime dependency
is introduced.

Upstream source:
https://github.com/vllm-project/vllm/tree/ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb/csrc/libtorch_stable/quantization/marlin
