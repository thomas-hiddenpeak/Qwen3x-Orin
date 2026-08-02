#include "q3x/kernels/sm87_a4w4_ldmatrix_operand_probe.h"

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {
namespace {

inline constexpr std::size_t kAStageBytes = kSm87A4W4PackedABytes;
inline constexpr std::size_t kBStageBytes = kSm87A4W4PackedBBytes;

struct alignas(16) OperandStage final {
  alignas(16) std::uint8_t a[kAStageBytes];
  alignas(16) std::uint8_t b[kBStageBytes];
};

static_assert(kAStageBytes == 512U);
static_assert(kBStageBytes == 256U);
static_assert(sizeof(OperandStage) == 768U);

[[nodiscard]] constexpr bool pointer_is_aligned(
    const void* const pointer, const std::size_t alignment) noexcept {
  return pointer != nullptr &&
         reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0U;
}

[[nodiscard]] int reject_non_sm87(
    cudaDeviceProp* const properties = nullptr) noexcept {
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  cudaDeviceProp local{};
  status = cudaGetDeviceProperties(&local, device);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  if (local.major != kSm87A4W4RequiredComputeMajor ||
      local.minor != kSm87A4W4RequiredComputeMinor) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (properties != nullptr) {
    *properties = local;
  }
  return static_cast<int>(cudaSuccess);
}

__device__ __forceinline__ void stage_xor16b_k64(
    OperandStage& stage, const std::uint8_t* const packed_a_k512,
    const std::uint8_t* const packed_b_k512,
    const std::size_t k64_group, const unsigned int lane) noexcept {
  const std::size_t group_base =
      k64_group * kSm87A4W4LdmatrixProbeOuterBlock *
      kSm87A4W4LdmatrixProbePackedK64Bytes;

  // A: every lane moves one canonical 16-byte row-half. The destination
  // applies exactly the established XOR-16B permutation.
  const std::size_t a_row = lane >> 1U;
  const std::size_t a_half = lane & 1U;
  const auto a_value = *reinterpret_cast<const uint4*>(
      packed_a_k512 + group_base +
      a_row * kSm87A4W4LdmatrixProbePackedK64Bytes + a_half * 16U);
  *reinterpret_cast<uint4*>(
      stage.a + sm87_a4w4_swizzled_k64_byte_offset(
                    a_row, a_half * 16U)) = a_value;

  // B: the lower half-warp moves the eight logical rows.
  if (lane < 16U) {
    const std::size_t b_row = lane >> 1U;
    const std::size_t b_half = lane & 1U;
    const auto b_value = *reinterpret_cast<const uint4*>(
        packed_b_k512 + group_base +
        b_row * kSm87A4W4LdmatrixProbePackedK64Bytes + b_half * 16U);
    *reinterpret_cast<uint4*>(
        stage.b + sm87_a4w4_swizzled_k64_byte_offset(
                      b_row, b_half * 16U)) = b_value;
  }
  __syncwarp(0xffff'ffffU);
}

[[nodiscard]] __device__ __forceinline__ Sm87A4W4AFragment
load_a_ldmatrix_x4(const std::uint8_t* const shared_a,
                   const unsigned int lane) noexcept {
  // x4 providers are four groups of eight lanes. The four 8x8.b16 matrices
  // are ordered exactly as the native MMA A tuple:
  //   {M0..7/K0..31, M8..15/K0..31,
  //    M0..7/K32..63, M8..15/K32..63}.
  const unsigned int matrix = lane >> 3U;
  const unsigned int logical_row =
      (lane & 7U) + ((matrix & 1U) << 3U);
  const unsigned int logical_byte = (matrix >> 1U) * 16U;
  const auto* const source =
      shared_a + sm87_a4w4_swizzled_k64_byte_offset(
                     logical_row, logical_byte);
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(source));
  Sm87A4W4AFragment fragment{};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 870
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(fragment.x0), "=r"(fragment.x1),
        "=r"(fragment.x2), "=r"(fragment.x3)
      : "r"(shared_address)
      : "memory");
#else
  asm volatile("trap;");
#endif
  return fragment;
}

[[nodiscard]] __device__ __forceinline__ Sm87A4W4BFragment
load_b_ldmatrix_x2(const std::uint8_t* const shared_b,
                   const unsigned int lane) noexcept {
  // Lanes 0..15 provide the two 8x8.b16 matrices. Lanes 16..31 duplicate
  // valid addresses even though their address operands are ignored by x2.
  const unsigned int provider = lane & 15U;
  const unsigned int logical_row = provider & 7U;
  const unsigned int logical_byte = (provider >> 3U) * 16U;
  const auto* const source =
      shared_b + sm87_a4w4_swizzled_k64_byte_offset(
                     logical_row, logical_byte);
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(source));
  Sm87A4W4BFragment fragment{};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 870
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
      "{%0, %1}, [%2];"
      : "=r"(fragment.x0), "=r"(fragment.x1)
      : "r"(shared_address)
      : "memory");
#else
  asm volatile("trap;");
#endif
  return fragment;
}

__device__ __forceinline__ void export_accumulator(
    std::int32_t* const output, const unsigned int lane,
    const Sm87A4W4Accumulator& accumulator) noexcept {
  *reinterpret_cast<int4*>(output + lane * 4U) =
      make_int4(accumulator.x0, accumulator.x1,
                accumulator.x2, accumulator.x3);
}

}  // namespace

extern "C" __global__ __launch_bounds__(kSm87A4W4WarpThreads)
void q3x_sm87_a4w4_scalar_lds_operand_probe_kernel(
    const std::uint8_t* const packed_a_k512,
    const std::uint8_t* const packed_b_k512,
    const std::size_t k64_group,
    std::int32_t* const accumulators) {
  __shared__ OperandStage stage;
  const unsigned int lane = threadIdx.x;
  stage_xor16b_k64(
      stage, packed_a_k512, packed_b_k512, k64_group, lane);
  const Sm87A4W4AFragment a =
      sm87_a4w4_load_a_fragment_swizzled_shared(stage.a, lane);
  const Sm87A4W4BFragment b =
      sm87_a4w4_load_b_fragment_swizzled_shared(stage.b, lane);
  Sm87A4W4Accumulator accumulator{};
  sm87_a4w4_mma_m16n8k64(accumulator, a, b);
  export_accumulator(accumulators, lane, accumulator);
}

extern "C" __global__ __launch_bounds__(kSm87A4W4WarpThreads)
void q3x_sm87_a4w4_ldmatrix_operand_probe_kernel(
    const std::uint8_t* const packed_a_k512,
    const std::uint8_t* const packed_b_k512,
    const std::size_t k64_group,
    std::int32_t* const accumulators) {
  __shared__ OperandStage stage;
  const unsigned int lane = threadIdx.x;
  stage_xor16b_k64(
      stage, packed_a_k512, packed_b_k512, k64_group, lane);
  const Sm87A4W4AFragment a = load_a_ldmatrix_x4(stage.a, lane);
  const Sm87A4W4BFragment b = load_b_ldmatrix_x2(stage.b, lane);
  Sm87A4W4Accumulator accumulator{};
  sm87_a4w4_mma_m16n8k64(accumulator, a, b);
  export_accumulator(accumulators, lane, accumulator);
}

extern "C" __global__ __launch_bounds__(kSm87A4W4WarpThreads)
void q3x_sm87_a4w4_scalar_a_ldmatrix_b_operand_probe_kernel(
    const std::uint8_t* const packed_a_k512,
    const std::uint8_t* const packed_b_k512,
    const std::size_t k64_group,
    std::int32_t* const accumulators) {
  __shared__ OperandStage stage;
  const unsigned int lane = threadIdx.x;
  stage_xor16b_k64(
      stage, packed_a_k512, packed_b_k512, k64_group, lane);
  const Sm87A4W4AFragment a =
      sm87_a4w4_load_a_fragment_swizzled_shared(stage.a, lane);
  const Sm87A4W4BFragment b = load_b_ldmatrix_x2(stage.b, lane);
  Sm87A4W4Accumulator accumulator{};
  sm87_a4w4_mma_m16n8k64(accumulator, a, b);
  export_accumulator(accumulators, lane, accumulator);
}

extern "C" __global__ __launch_bounds__(kSm87A4W4WarpThreads)
void q3x_sm87_a4w4_ldmatrix_a_scalar_b_operand_probe_kernel(
    const std::uint8_t* const packed_a_k512,
    const std::uint8_t* const packed_b_k512,
    const std::size_t k64_group,
    std::int32_t* const accumulators) {
  __shared__ OperandStage stage;
  const unsigned int lane = threadIdx.x;
  stage_xor16b_k64(
      stage, packed_a_k512, packed_b_k512, k64_group, lane);
  const Sm87A4W4AFragment a = load_a_ldmatrix_x4(stage.a, lane);
  const Sm87A4W4BFragment b =
      sm87_a4w4_load_b_fragment_swizzled_shared(stage.b, lane);
  Sm87A4W4Accumulator accumulator{};
  sm87_a4w4_mma_m16n8k64(accumulator, a, b);
  export_accumulator(accumulators, lane, accumulator);
}

int launch_sm87_a4w4_ldmatrix_operand_probe_cuda(
    const std::uint8_t* const packed_a_k512,
    const std::size_t packed_a_capacity_bytes,
    const std::uint8_t* const packed_b_k512,
    const std::size_t packed_b_capacity_bytes,
    const std::size_t k64_group,
    std::int32_t* const scalar_accumulators,
    const std::size_t scalar_accumulator_capacity_words,
    std::int32_t* const ldmatrix_accumulators,
    const std::size_t ldmatrix_accumulator_capacity_words,
    std::int32_t* const scalar_a_ldmatrix_b_accumulators,
    const std::size_t scalar_a_ldmatrix_b_accumulator_capacity_words,
    std::int32_t* const ldmatrix_a_scalar_b_accumulators,
    const std::size_t ldmatrix_a_scalar_b_accumulator_capacity_words,
    void* const cuda_stream) noexcept {
  constexpr std::size_t kVectorAlignment = alignof(uint4);
  if (!pointer_is_aligned(packed_a_k512, kVectorAlignment) ||
      !pointer_is_aligned(packed_b_k512, kVectorAlignment) ||
      !pointer_is_aligned(scalar_accumulators, kVectorAlignment) ||
      !pointer_is_aligned(ldmatrix_accumulators, kVectorAlignment) ||
      !pointer_is_aligned(scalar_a_ldmatrix_b_accumulators,
                          kVectorAlignment) ||
      !pointer_is_aligned(ldmatrix_a_scalar_b_accumulators,
                          kVectorAlignment) ||
      packed_a_capacity_bytes < kSm87A4W4LdmatrixProbePayloadBytes ||
      packed_b_capacity_bytes < kSm87A4W4LdmatrixProbePayloadBytes ||
      scalar_accumulator_capacity_words <
          kSm87A4W4LdmatrixProbeAccumulatorWords ||
      ldmatrix_accumulator_capacity_words <
          kSm87A4W4LdmatrixProbeAccumulatorWords ||
      scalar_a_ldmatrix_b_accumulator_capacity_words <
          kSm87A4W4LdmatrixProbeAccumulatorWords ||
      ldmatrix_a_scalar_b_accumulator_capacity_words <
          kSm87A4W4LdmatrixProbeAccumulatorWords ||
      k64_group >= kSm87A4W4LdmatrixProbeK64Groups) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int architecture_status = reject_non_sm87();
  if (architecture_status != static_cast<int>(cudaSuccess)) {
    return architecture_status;
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  q3x_sm87_a4w4_scalar_lds_operand_probe_kernel<<<
      1U, static_cast<unsigned int>(kSm87A4W4WarpThreads), 0U, stream>>>(
      packed_a_k512, packed_b_k512, k64_group, scalar_accumulators);
  cudaError_t status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  q3x_sm87_a4w4_ldmatrix_operand_probe_kernel<<<
      1U, static_cast<unsigned int>(kSm87A4W4WarpThreads), 0U, stream>>>(
      packed_a_k512, packed_b_k512, k64_group, ldmatrix_accumulators);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  q3x_sm87_a4w4_scalar_a_ldmatrix_b_operand_probe_kernel<<<
      1U, static_cast<unsigned int>(kSm87A4W4WarpThreads), 0U, stream>>>(
      packed_a_k512, packed_b_k512, k64_group,
      scalar_a_ldmatrix_b_accumulators);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  q3x_sm87_a4w4_ldmatrix_a_scalar_b_operand_probe_kernel<<<
      1U, static_cast<unsigned int>(kSm87A4W4WarpThreads), 0U, stream>>>(
      packed_a_k512, packed_b_k512, k64_group,
      ldmatrix_a_scalar_b_accumulators);
  return static_cast<int>(cudaPeekAtLastError());
}

int query_sm87_a4w4_ldmatrix_operand_probe_resources_cuda(
    Sm87A4W4LdmatrixOperandProbeResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4LdmatrixOperandProbeResources{};

  cudaDeviceProp properties{};
  int status = reject_non_sm87(&properties);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  cudaFuncAttributes scalar_attributes{};
  cudaFuncAttributes ldmatrix_attributes{};
  cudaFuncAttributes scalar_a_ldmatrix_b_attributes{};
  cudaFuncAttributes ldmatrix_a_scalar_b_attributes{};
  cudaError_t cuda_status = cudaFuncGetAttributes(
      &scalar_attributes, q3x_sm87_a4w4_scalar_lds_operand_probe_kernel);
  if (cuda_status != cudaSuccess) {
    return static_cast<int>(cuda_status);
  }
  cuda_status = cudaFuncGetAttributes(
      &ldmatrix_attributes, q3x_sm87_a4w4_ldmatrix_operand_probe_kernel);
  if (cuda_status != cudaSuccess) {
    return static_cast<int>(cuda_status);
  }
  cuda_status = cudaFuncGetAttributes(
      &scalar_a_ldmatrix_b_attributes,
      q3x_sm87_a4w4_scalar_a_ldmatrix_b_operand_probe_kernel);
  if (cuda_status != cudaSuccess) {
    return static_cast<int>(cuda_status);
  }
  cuda_status = cudaFuncGetAttributes(
      &ldmatrix_a_scalar_b_attributes,
      q3x_sm87_a4w4_ldmatrix_a_scalar_b_operand_probe_kernel);
  if (cuda_status != cudaSuccess) {
    return static_cast<int>(cuda_status);
  }
  int scalar_active_blocks = 0;
  int ldmatrix_active_blocks = 0;
  int scalar_a_ldmatrix_b_active_blocks = 0;
  int ldmatrix_a_scalar_b_active_blocks = 0;
  cuda_status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &scalar_active_blocks,
      q3x_sm87_a4w4_scalar_lds_operand_probe_kernel,
      static_cast<int>(kSm87A4W4WarpThreads), 0U);
  if (cuda_status != cudaSuccess) {
    return static_cast<int>(cuda_status);
  }
  cuda_status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &ldmatrix_active_blocks,
      q3x_sm87_a4w4_ldmatrix_operand_probe_kernel,
      static_cast<int>(kSm87A4W4WarpThreads), 0U);
  if (cuda_status != cudaSuccess) {
    return static_cast<int>(cuda_status);
  }
  cuda_status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &scalar_a_ldmatrix_b_active_blocks,
      q3x_sm87_a4w4_scalar_a_ldmatrix_b_operand_probe_kernel,
      static_cast<int>(kSm87A4W4WarpThreads), 0U);
  if (cuda_status != cudaSuccess) {
    return static_cast<int>(cuda_status);
  }
  cuda_status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &ldmatrix_a_scalar_b_active_blocks,
      q3x_sm87_a4w4_ldmatrix_a_scalar_b_operand_probe_kernel,
      static_cast<int>(kSm87A4W4WarpThreads), 0U);
  if (cuda_status != cudaSuccess) {
    return static_cast<int>(cuda_status);
  }

  resources->scalar_registers_per_thread = scalar_attributes.numRegs;
  resources->ldmatrix_registers_per_thread = ldmatrix_attributes.numRegs;
  resources->scalar_a_ldmatrix_b_registers_per_thread =
      scalar_a_ldmatrix_b_attributes.numRegs;
  resources->ldmatrix_a_scalar_b_registers_per_thread =
      ldmatrix_a_scalar_b_attributes.numRegs;
  resources->scalar_static_shared_bytes = scalar_attributes.sharedSizeBytes;
  resources->ldmatrix_static_shared_bytes =
      ldmatrix_attributes.sharedSizeBytes;
  resources->scalar_a_ldmatrix_b_static_shared_bytes =
      scalar_a_ldmatrix_b_attributes.sharedSizeBytes;
  resources->ldmatrix_a_scalar_b_static_shared_bytes =
      ldmatrix_a_scalar_b_attributes.sharedSizeBytes;
  resources->scalar_local_bytes = scalar_attributes.localSizeBytes;
  resources->ldmatrix_local_bytes = ldmatrix_attributes.localSizeBytes;
  resources->scalar_a_ldmatrix_b_local_bytes =
      scalar_a_ldmatrix_b_attributes.localSizeBytes;
  resources->ldmatrix_a_scalar_b_local_bytes =
      ldmatrix_a_scalar_b_attributes.localSizeBytes;
  resources->scalar_active_blocks_per_sm = scalar_active_blocks;
  resources->ldmatrix_active_blocks_per_sm = ldmatrix_active_blocks;
  resources->scalar_a_ldmatrix_b_active_blocks_per_sm =
      scalar_a_ldmatrix_b_active_blocks;
  resources->ldmatrix_a_scalar_b_active_blocks_per_sm =
      ldmatrix_a_scalar_b_active_blocks;
  resources->compute_major = properties.major;
  resources->compute_minor = properties.minor;
  return static_cast<int>(cudaSuccess);
}

}  // namespace q3x::kernels
