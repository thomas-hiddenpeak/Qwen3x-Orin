#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels::p40_packed_detail {

inline constexpr unsigned int kWarpSize = 32U;
inline constexpr unsigned int kPackedProjectionSmCount = 16U;
inline constexpr unsigned int kPackedProjectionPersistentCtas = 32U;
inline constexpr unsigned int kPackedProjectionPipelineStages = 4U;
inline constexpr unsigned int kPackedProjectionTileK = 64U;

[[nodiscard]] __host__ __device__ constexpr unsigned int
transform_a_cell(const unsigned int logical_cell) noexcept {
  const unsigned int row = logical_cell / 8U;
  return row * 8U + ((logical_cell % 8U) ^ (row % 8U));
}

template <bool kCacheAll, bool kPredicate = false>
__device__ __forceinline__ void cp_async_16(
    void* const shared_destination, const void* const global_source,
    const bool valid = true) noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const std::uint32_t shared_address =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(shared_destination));
  if constexpr (kPredicate) {
    if constexpr (kCacheAll) {
      asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                   :
                   : "r"(shared_address), "l"(global_source),
                     "r"(valid ? 16U : 0U)
                   : "memory");
    } else {
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                   :
                   : "r"(shared_address), "l"(global_source),
                     "r"(valid ? 16U : 0U)
                   : "memory");
    }
  } else if constexpr (kCacheAll) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                 :
                 : "r"(shared_address), "l"(global_source)
                 : "memory");
  } else {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 :
                 : "r"(shared_address), "l"(global_source)
                 : "memory");
  }
#else
  *reinterpret_cast<uint4*>(shared_destination) =
      valid ? *reinterpret_cast<const uint4*>(global_source)
            : make_uint4(0U, 0U, 0U, 0U);
#endif
}

__device__ __forceinline__ void cp_async_commit_group() noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

template <unsigned int kGroups>
__device__ __forceinline__ void cp_async_wait_group() noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;" : : "n"(kGroups) : "memory");
#endif
}

[[nodiscard]] __host__ __device__ constexpr std::uint16_t
decode_e4m3fn_to_bf16_bits(const std::uint8_t code) noexcept {
  const std::uint16_t sign =
      static_cast<std::uint16_t>(code & 0x80U) << 8U;
  const std::uint16_t magnitude =
      static_cast<std::uint16_t>(code & 0x7fU);
  if (magnitude == 0x7fU) {
    return static_cast<std::uint16_t>(sign | 0x7fc0U);
  }
  const std::uint16_t exponent = magnitude >> 3U;
  const std::uint16_t mantissa = magnitude & 0x07U;
  if (exponent == 0U) {
    if (mantissa == 0U) {
      return sign;
    }
    const std::uint16_t leading =
        mantissa >= 4U ? 2U : (mantissa >= 2U ? 1U : 0U);
    const std::uint16_t bf16_exponent = 118U + leading;
    const std::uint16_t bf16_mantissa =
        (mantissa - (1U << leading)) << (7U - leading);
    return static_cast<std::uint16_t>(
        sign | (bf16_exponent << 7U) | bf16_mantissa);
  }
  return static_cast<std::uint16_t>(
      sign | ((120U + exponent) << 7U) | (mantissa << 4U));
}

[[nodiscard]] __device__ __forceinline__ std::uint32_t
multiply_bf16x2_bits(const std::uint32_t value_bits,
                     const std::uint16_t scale_bits) noexcept {
  const __nv_bfloat162_raw value_raw{
      static_cast<std::uint16_t>(value_bits),
      static_cast<std::uint16_t>(value_bits >> 16U)};
  const __nv_bfloat162_raw scale_raw{scale_bits, scale_bits};
  const __nv_bfloat162 values(value_raw);
  const __nv_bfloat162 scales(scale_raw);
  const __nv_bfloat162_raw result_raw =
      static_cast<__nv_bfloat162_raw>(__hmul2_rn(values, scales));
  return static_cast<std::uint32_t>(result_raw.x) |
         (static_cast<std::uint32_t>(result_raw.y) << 16U);
}

// Decode the four NVFP4 values consumed by one m16n8k16 B-fragment lane.
// PRMT constructs exact BF16 E2M1 values without a data-indexed table; the
// packed BF16 multiply applies the original E4M3FN block scale immediately
// before MMA. Signed zero and both E4M3FN NaN encodings are preserved.
[[nodiscard]] __device__ __forceinline__ uint2
decode_nvfp4x4_to_bf16x4(const std::uint16_t packed,
                         const std::uint8_t encoded_scale) noexcept {
  constexpr std::uint32_t kLowBytes0To3 = 0xc080'0000U;
  constexpr std::uint32_t kLowBytes4To7 = 0xc080'4000U;
  constexpr std::uint32_t kHighBytes0To3 = 0x3f3f'3f00U;
  constexpr std::uint32_t kHighBytes4To7 = 0x4040'4040U;
  constexpr std::uint32_t kMagnitudeSelectorMask = 0x7777U;
  constexpr std::uint32_t kSignByteMask = 0x8080'8080U;
  constexpr std::uint32_t kFirstFourSignSelector = 0xd9c8U;
  constexpr std::uint32_t kFirstPairInterleave = 0x5140U;
  constexpr std::uint32_t kSecondPairInterleave = 0x7362U;

  const std::uint32_t packed32 = static_cast<std::uint32_t>(packed);
  const std::uint32_t selector = packed32 & kMagnitudeSelectorMask;
  const std::uint32_t low =
      __byte_perm(kLowBytes0To3, kLowBytes4To7, selector);
  const std::uint32_t signs =
      __byte_perm(packed32 << 4U, packed32, kFirstFourSignSelector) &
      kSignByteMask;
  const std::uint32_t high =
      __byte_perm(kHighBytes0To3, kHighBytes4To7, selector) | signs;
  const std::uint16_t scale_bits =
      decode_e4m3fn_to_bf16_bits(encoded_scale);

  uint2 result{};
  result.x = multiply_bf16x2_bits(
      __byte_perm(low, high, kFirstPairInterleave), scale_bits);
  result.y = multiply_bf16x2_bits(
      __byte_perm(low, high, kSecondPairInterleave), scale_bits);
  return result;
}

[[nodiscard]] __host__ __device__ inline std::uint16_t encode_bf16_rne(
    const float value) noexcept {
#if defined(__CUDA_ARCH__)
  std::uint32_t bits = __float_as_uint(value);
#else
  std::uint32_t bits = 0U;
  __builtin_memcpy(&bits, &value, sizeof(bits));
#endif
  const std::uint32_t magnitude = bits & 0x7fff'ffffU;
  if (magnitude > 0x7f80'0000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] __host__ __device__ inline float decode_bf16(
    const std::uint16_t bits) noexcept {
  const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16U;
#if defined(__CUDA_ARCH__)
  return __uint_as_float(wide);
#else
  float value = 0.0F;
  __builtin_memcpy(&value, &wide, sizeof(value));
  return value;
#endif
}

[[nodiscard]] __device__ __forceinline__ std::uint32_t
pack_scaled_bf16_pair(const float low, const float high,
                      const float global_scale) noexcept {
  return static_cast<std::uint32_t>(encode_bf16_rne(low * global_scale)) |
         (static_cast<std::uint32_t>(
              encode_bf16_rne(high * global_scale))
          << 16U);
}

[[nodiscard]] __device__ __forceinline__ std::uint16_t
gate_up_silu_mul_bf16(const std::uint16_t gate_bits,
                      const std::uint16_t up_bits) noexcept {
  const float gate = decode_bf16(gate_bits);
  const float up = decode_bf16(up_bits);
  return encode_bf16_rne(gate / (1.0F + expf(-gate)) * up);
}

[[nodiscard]] __device__ __forceinline__ std::uint32_t
add_residual_bf16_pair(const std::uint32_t branch_bits,
                       const std::uint32_t residual_bits) noexcept {
  const float branch0 =
      decode_bf16(static_cast<std::uint16_t>(branch_bits));
  const float branch1 =
      decode_bf16(static_cast<std::uint16_t>(branch_bits >> 16U));
  const float residual0 =
      decode_bf16(static_cast<std::uint16_t>(residual_bits));
  const float residual1 =
      decode_bf16(static_cast<std::uint16_t>(residual_bits >> 16U));
  return static_cast<std::uint32_t>(
             encode_bf16_rne(branch0 + residual0)) |
         (static_cast<std::uint32_t>(
              encode_bf16_rne(branch1 + residual1))
          << 16U);
}

struct ByteRange {
  std::uintptr_t begin = 0U;
  std::uintptr_t end = 0U;
};

[[nodiscard]] inline constexpr bool multiply_would_overflow(
    const std::size_t left, const std::size_t right) noexcept {
  return right != 0U &&
         left > std::numeric_limits<std::size_t>::max() / right;
}

[[nodiscard]] inline bool make_range(const void* const pointer,
                                     const std::size_t bytes,
                                     ByteRange* const range) noexcept {
  if (pointer == nullptr || range == nullptr || bytes == 0U) {
    return false;
  }
  const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) {
    return false;
  }
  *range = {begin, begin + bytes};
  return true;
}

[[nodiscard]] inline constexpr bool overlaps(const ByteRange& left,
                                             const ByteRange& right) noexcept {
  return left.begin < right.end && right.begin < left.end;
}

[[nodiscard]] inline bool aligned(const void* const pointer,
                                  const std::size_t alignment) noexcept {
  return pointer != nullptr && alignment != 0U &&
         reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0U;
}

[[nodiscard]] inline cudaError_t validate_device_pointer(
    const void* const pointer, const int expected_device) noexcept {
  cudaPointerAttributes attributes{};
  const cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
  if (status != cudaSuccess) {
    return status;
  }
#if CUDART_VERSION >= 10000
  if (attributes.type != cudaMemoryTypeDevice ||
      attributes.device != expected_device) {
    return cudaErrorInvalidValue;
  }
#else
  if (attributes.memoryType != cudaMemoryTypeDevice ||
      attributes.device != expected_device) {
    return cudaErrorInvalidValue;
  }
#endif
  return cudaSuccess;
}

}  // namespace q3x::kernels::p40_packed_detail
