/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Modified for the fixed Q3X SM87 Decode admission, 2026-09-27.
// Fixed SM87 mapping of the vendored Apache-2.0 FlashInfer single-query path.
// See third_party/flashinfer/README.q3x.md for source and license provenance.
#include "decode_fused_gqa_internal.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <array>
#include <limits>

namespace q3x::runtime::fused_decode {
using Params = flashinfer::SinglePrefillParams<__nv_bfloat16, __nv_bfloat16,
                                             __nv_bfloat16>;
using BaseTraits = flashinfer::KernelTraits<
    flashinfer::MaskMode::kNone, 16, 1, 1, 16, 16, 1, 4,
    flashinfer::PosEncodingMode::kNone, __nv_bfloat16, __nv_bfloat16,
    __nv_bfloat16, float, typename Params::IdType,
    flashinfer::DefaultAttention<false, false, false, false>>;
struct Traits : BaseTraits {};
namespace {
static_assert(!Traits::IsInvalid());
static_assert(Traits::NUM_THREADS == 128);
static_assert(sizeof(typename Traits::SharedStorage) == 73760);
struct Span { std::uintptr_t begin; std::size_t bytes; };
bool valid_spans(const std::array<Span, 5>& spans) noexcept {
  for (std::size_t i = 0; i < spans.size(); ++i) {
    const auto a = spans[i];
    if (!a.begin || a.begin > std::numeric_limits<std::uintptr_t>::max() - a.bytes)
      return false;
    for (std::size_t j = 0; j < i; ++j) {
      const auto b = spans[j];
      if (a.begin < b.begin + b.bytes && b.begin < a.begin + a.bytes) return false;
    }
  }
  return true;
}
}  // namespace

}  // namespace q3x::runtime::fused_decode

namespace flashinfer {
namespace {
// Local, uniquely typed specialization: preserve the FP32 probability through
// two BF16 operands instead of discarding its residual. The vendored header
// and every other FlashInfer instantiation remain unchanged.
template <>
__device__ __forceinline__ void compute_sfm_v<q3x::runtime::fused_decode::Traits>(
    smem_t<q3x::runtime::fused_decode::Traits::SWIZZLE_MODE_KV>* v_smem,
    uint32_t* offset, uint8_t*, uint32_t,
    float (*probability)[1][8], float (*output)[16][8], float (*denominator)[2]) {
  alignas(16) __nv_bfloat16 hi[8];
  alignas(16) __nv_bfloat16 lo[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    hi[i] = __float2bfloat16_rn(probability[0][0][i]);
    lo[i] = __float2bfloat16_rn(probability[0][0][i] - __bfloat162float(hi[i]));
  }
  mma::m16k16_rowsum_f16f16f32(denominator[0], hi);
  mma::m16k16_rowsum_f16f16f32(denominator[0], lo);
#pragma unroll
  for (uint32_t d = 0; d < 16; ++d) {
    uint32_t values[4];
    v_smem->ldmatrix_m8n8x4_trans(*offset, values);
    mma::mma_sync_m16n16k16_row_col_f16f16f32<__nv_bfloat16>(
        output[0][d], reinterpret_cast<uint32_t*>(hi), values);
    mma::mma_sync_m16n16k16_row_col_f16f16f32<__nv_bfloat16>(
        output[0][d], reinterpret_cast<uint32_t*>(lo), values);
    *offset = v_smem->template advance_offset_by_column<2>(*offset, d);
  }
  *offset = v_smem->template advance_offset_by_row<16,
      q3x::runtime::fused_decode::Traits::UPCAST_STRIDE_V>(*offset) - 2 * 16;
  *offset -= 16 * q3x::runtime::fused_decode::Traits::UPCAST_STRIDE_V;
}
}  // namespace
}  // namespace flashinfer

namespace q3x::runtime::fused_decode {
namespace {
thread_local Observer observer = nullptr;
thread_local void* observer_context = nullptr;
}
void set_observer(Observer callback, void* context) noexcept {
  observer = callback;
  observer_context = context;
}
int observe(const Observation& observation) noexcept {
  return observer ? observer(observation, observer_context) : 0;
}

int prepare() noexcept {
  int device = 0;
  cudaDeviceProp props{};
  auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) return static_cast<int>(status);
  status = cudaGetDeviceProperties(&props, device);
  if (status != cudaSuccess) return static_cast<int>(status);
  if (props.major != 8 || props.minor != 7 || props.multiProcessorCount != 16)
    return static_cast<int>(cudaErrorNotSupported);
  return static_cast<int>(cudaFuncSetAttribute(
      flashinfer::SinglePrefillWithKVCacheKernel<Traits, Params>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      sizeof(typename Traits::SharedStorage)));
}

int launch(const std::uint16_t* query, const std::uint16_t* key,
           const std::uint16_t* value, const std::size_t sequence,
           void* workspace, const std::size_t workspace_bytes,
           std::uint16_t* output, void* opaque) noexcept {
  if (sequence < kMinimumSequence || sequence > kMaximumSequence ||
      workspace_bytes < kWorkspaceBytes ||
      reinterpret_cast<std::uintptr_t>(workspace) % 16 != 0 ||
      reinterpret_cast<std::uintptr_t>(query) % 16 != 0 ||
      reinterpret_cast<std::uintptr_t>(key) % 16 != 0 ||
      reinterpret_cast<std::uintptr_t>(value) % 16 != 0 ||
      reinterpret_cast<std::uintptr_t>(output) % 16 != 0 ||
      !valid_spans({{{reinterpret_cast<std::uintptr_t>(query), 24 * 256 * 2},
                     {reinterpret_cast<std::uintptr_t>(key), sequence * 4 * 256 * 2},
                     {reinterpret_cast<std::uintptr_t>(value), sequence * 4 * 256 * 2},
                     {reinterpret_cast<std::uintptr_t>(workspace), kWorkspaceBytes},
                     {reinterpret_cast<std::uintptr_t>(output), 24 * 256 * 2}}}))
    return static_cast<int>(cudaErrorInvalidValue);
  (void)cudaGetLastError();
  const auto stream = static_cast<cudaStream_t>(opaque);
  const auto chunk_size = std::max<std::size_t>((sequence + 7) / 8, 256);
  const auto chunks = (sequence + chunk_size - 1) / chunk_size;
  auto* partial = static_cast<__nv_bfloat16*>(workspace);
  auto* lse = reinterpret_cast<float*>(partial + chunks * 24 * 256);
  Params params(reinterpret_cast<__nv_bfloat16*>(const_cast<std::uint16_t*>(query)),
                reinterpret_cast<__nv_bfloat16*>(const_cast<std::uint16_t*>(key)),
                reinterpret_cast<__nv_bfloat16*>(const_cast<std::uint16_t*>(value)),
                nullptr, partial, lse, nullptr, 24, 4, 1,
                static_cast<std::uint32_t>(sequence), 24 * 256, 256,
                4 * 256, 256, 256, -1, 0.0f, 0.0625f, 1.0f, 10000.0f);
  params.partition_kv = true;
  void* args[] = {&params};
  auto status = cudaLaunchKernel(
      reinterpret_cast<const void*>(flashinfer::SinglePrefillWithKVCacheKernel<Traits, Params>),
      dim3(1, static_cast<unsigned>(chunks), 4), dim3(32, 1, 4), args,
      sizeof(typename Traits::SharedStorage), stream);
  if (status != cudaSuccess) return static_cast<int>(status);
  // Fixed merge, no per-token attribute setup or dispatch search.
  auto* merged = reinterpret_cast<__nv_bfloat16*>(output);
  float* merged_lse = nullptr;
  auto sets = static_cast<std::uint32_t>(chunks);
  std::uint32_t heads = 24;
  void* merge_args[] = {&partial, &lse, &merged, &merged_lse, &sets, &heads};
  return static_cast<int>(cudaLaunchKernel(
      reinterpret_cast<const void*>(flashinfer::MergeStatesLargeNumIndexSetsKernel<
          8, 32, 4, 4, __nv_bfloat16, __nv_bfloat16>),
      dim3(1, 24), dim3(32, 4), merge_args, 8704, stream));
}
}  // namespace q3x::runtime::fused_decode
