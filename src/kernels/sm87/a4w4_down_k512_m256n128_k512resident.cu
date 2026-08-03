#include "q3x/kernels/sm87_a4w4_down_k512_m256n128_k512resident.h"

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

inline constexpr unsigned int kRequiredSmCount = 16U;
inline constexpr unsigned int kTileM = static_cast<unsigned int>(
    kSm87A4W4DownK512M256N128ResidentTileM);
inline constexpr unsigned int kTileN = static_cast<unsigned int>(
    kSm87A4W4DownK512M256N128ResidentTileN);
inline constexpr unsigned int kThreads = static_cast<unsigned int>(
    kSm87A4W4DownK512M256N128ResidentThreads);
inline constexpr unsigned int kPackedK64Bytes = 32U;

struct alignas(16) ResidentCodes final {
  std::uint8_t
      a[kSm87A4W4DownK512M256N128ResidentPhysicalK64PerScale]
       [kTileM * kPackedK64Bytes];
  std::uint8_t
      b[kSm87A4W4DownK512M256N128ResidentPhysicalK64PerScale]
       [kTileN * kPackedK64Bytes];
};

struct alignas(16) ResidentScales final {
  std::uint16_t a[kTileM];
  std::uint16_t b[kTileN];
};

struct alignas(16) SharedStorage final {
  ResidentCodes codes;
  ResidentScales scales;
};

struct alignas(16) Float4 final {
  float x0{};
  float x1{};
  float x2{};
  float x3{};
};

// Named register state is intentional.  A runtime-indexed aggregate lets
// ptxas take its address and lower part of the persistent K-loop state to
// local memory.  These fields are the exact 4x4 FP32 output fragments and
// 4x2 S32 fragments owned by one M64N32 warp.
struct Outputs final {
  Float4 m0n0{};
  Float4 m0n1{};
  Float4 m0n2{};
  Float4 m0n3{};
  Float4 m1n0{};
  Float4 m1n1{};
  Float4 m1n2{};
  Float4 m1n3{};
  Float4 m2n0{};
  Float4 m2n1{};
  Float4 m2n2{};
  Float4 m2n3{};
  Float4 m3n0{};
  Float4 m3n1{};
  Float4 m3n2{};
  Float4 m3n3{};
};

struct Partials final {
  Sm87A4W4Accumulator m0n0{};
  Sm87A4W4Accumulator m0n1{};
  Sm87A4W4Accumulator m1n0{};
  Sm87A4W4Accumulator m1n1{};
  Sm87A4W4Accumulator m2n0{};
  Sm87A4W4Accumulator m2n1{};
  Sm87A4W4Accumulator m3n0{};
  Sm87A4W4Accumulator m3n1{};
};

static_assert(sizeof(ResidentCodes) ==
              kSm87A4W4DownK512M256N128ResidentABytes +
                  kSm87A4W4DownK512M256N128ResidentBBytes);
static_assert(sizeof(ResidentScales) ==
              kSm87A4W4DownK512M256N128ResidentScaleBytes);
static_assert(sizeof(SharedStorage) ==
              kSm87A4W4DownK512M256N128ResidentDynamicSharedBytes);
static_assert(sizeof(Float4) == 16U);
static_assert(sizeof(Outputs) == 256U);
static_assert(sizeof(Partials) == 128U);

[[nodiscard]] constexpr bool aligned(const void* const pointer,
                                     const std::size_t alignment) noexcept {
  return pointer != nullptr &&
         reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0U;
}

[[nodiscard]] bool byte_ranges_overlap(
    const void* const first, const std::size_t first_bytes,
    const void* const second, const std::size_t second_bytes) noexcept {
  const std::uintptr_t first_begin =
      reinterpret_cast<std::uintptr_t>(first);
  const std::uintptr_t second_begin =
      reinterpret_cast<std::uintptr_t>(second);
  constexpr std::uintptr_t maximum =
      std::numeric_limits<std::uintptr_t>::max();
  if (first_bytes > maximum - first_begin ||
      second_bytes > maximum - second_begin) {
    return true;
  }
  const std::uintptr_t first_end = first_begin + first_bytes;
  const std::uintptr_t second_end = second_begin + second_bytes;
  return first_begin < second_end && second_begin < first_end;
}

[[nodiscard]] __device__ __forceinline__ float decode_bf16(
    const std::uint16_t bits) noexcept {
  return __uint_as_float(static_cast<unsigned int>(bits) << 16U);
}

[[nodiscard]] __device__ __forceinline__ std::uint16_t encode_bf16(
    const float value) noexcept {
  unsigned int bits = __float_as_uint(value);
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

__device__ __forceinline__ void cp_async_cg_16(
    void* const destination, const void* const source,
    const unsigned int source_bytes = 16U) noexcept {
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(destination));
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 870
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
               :
               : "r"(shared_address), "l"(source), "r"(source_bytes)
               : "memory");
#else
  asm volatile("trap;");
#endif
}

__device__ __forceinline__ void cp_async_ca_16(
    void* const destination, const void* const source,
    const unsigned int source_bytes = 16U) noexcept {
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(destination));
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 870
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
               :
               : "r"(shared_address), "l"(source), "r"(source_bytes)
               : "memory");
#else
  asm volatile("trap;");
#endif
}

__device__ __forceinline__ void cp_async_commit() noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 870
  asm volatile("cp.async.commit_group;" : : : "memory");
#else
  asm volatile("trap;");
#endif
}

__device__ __forceinline__ void cp_async_wait_all() noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 870
  asm volatile("cp.async.wait_group 0;" : : : "memory");
#else
  asm volatile("trap;");
#endif
}

// Eight commits correspond exactly to the eight physical K64 planes in one
// numerical K512 epoch.  Plane seven also carries the scale vectors, avoiding
// a ninth outstanding cp.async group.
__device__ __forceinline__ void issue_k512_epoch(
    SharedStorage& shared, const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const packed_b,
    const std::uint16_t* const b_k512_scales_bf16,
    const unsigned int launch_token_count,
    const unsigned int m_tile_start, const unsigned int n_tile_start,
    const unsigned int group, const unsigned int k512_group_count,
    const unsigned int physical_k64_group_count) noexcept {
  constexpr unsigned int kAVectors = kTileM * kPackedK64Bytes / 16U;
  constexpr unsigned int kBVectors = kTileN * kPackedK64Bytes / 16U;
  constexpr unsigned int kVectors = kAVectors + kBVectors;
  static_assert(kAVectors == 512U);
  static_assert(kBVectors == 256U);
  static_assert(kVectors == kThreads + kThreads / 2U);

#pragma unroll 1
  for (unsigned int plane = 0U;
       plane <
       kSm87A4W4DownK512M256N128ResidentPhysicalK64PerScale;
       ++plane) {
    const unsigned int physical_k64_group = 8U * group + plane;
#pragma unroll
    for (unsigned int linear = threadIdx.x; linear < kVectors;
         linear += kThreads) {
      const bool is_a = linear < kAVectors;
      const unsigned int local_vector =
          is_a ? linear : linear - kAVectors;
      const unsigned int row = local_vector >> 1U;
      const unsigned int row_vector = local_vector & 1U;
      const unsigned int outer_start =
          is_a ? m_tile_start : n_tile_start;
      const unsigned int outer = outer_start + row;
      const bool valid = !is_a || outer < launch_token_count;
      const std::uint8_t* const operand = is_a ? packed_a : packed_b;
      const std::uint8_t* const source =
          valid
              ? operand + sm87_a4w4_down_k512_packed_offset(
                              outer, physical_k64_group,
                              16U * row_vector,
                              physical_k64_group_count)
              : operand;
      std::uint8_t* const destination =
          (is_a ? shared.codes.a[plane] : shared.codes.b[plane]) +
          sm87_a4w4_swizzled_k64_byte_offset(row,
                                             16U * row_vector);
      cp_async_cg_16(destination, source, valid ? 16U : 0U);
    }

    if (plane + 1U ==
            kSm87A4W4DownK512M256N128ResidentPhysicalK64PerScale &&
        threadIdx.x < 48U) {
      const bool is_a = threadIdx.x < 32U;
      const unsigned int vector = is_a ? threadIdx.x : threadIdx.x - 32U;
      const unsigned int row = 8U * vector;
      const unsigned int outer_start =
          is_a ? m_tile_start : n_tile_start;
      const unsigned int outer = outer_start + row;
      const bool valid = !is_a || outer < launch_token_count;
      const std::uint16_t* const operand =
          is_a ? a_k512_scales_bf16 : b_k512_scales_bf16;
      const std::uint16_t* const source =
          valid
              ? operand + sm87_a4w4_down_k512_scale_offset(
                              outer, group, k512_group_count)
              : operand;
      std::uint16_t* const destination =
          (is_a ? shared.scales.a : shared.scales.b) + row;
      cp_async_ca_16(destination, source, valid ? 16U : 0U);
    }
    cp_async_commit();
  }
}

[[nodiscard]] __device__ __forceinline__ Sm87A4W4AFragment
load_a_ldmatrix_x4(const std::uint8_t* const shared_a,
                   const unsigned int lane) noexcept {
  const unsigned int matrix = lane >> 3U;
  const unsigned int logical_row =
      (lane & 7U) + ((matrix & 1U) << 3U);
  const unsigned int logical_byte = (matrix >> 1U) * 16U;
  const auto* const source =
      shared_a +
      sm87_a4w4_swizzled_k64_byte_offset(logical_row, logical_byte);
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(source));
  Sm87A4W4AFragment fragment{};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 870
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(fragment.x0), "=r"(fragment.x1), "=r"(fragment.x2),
        "=r"(fragment.x3)
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
  const unsigned int provider = lane & 15U;
  const unsigned int logical_row = provider & 7U;
  const unsigned int logical_byte = (provider >> 3U) * 16U;
  const auto* const source =
      shared_b +
      sm87_a4w4_swizzled_k64_byte_offset(logical_row, logical_byte);
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(source));
  Sm87A4W4BFragment fragment{};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 870
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
      : "=r"(fragment.x0), "=r"(fragment.x1)
      : "r"(shared_address)
      : "memory");
#else
  asm volatile("trap;");
#endif
  return fragment;
}

template <unsigned int NPhase>
__device__ __forceinline__ void accumulate_n_phase(
    const ResidentCodes& codes, const unsigned int warp_m64,
    const unsigned int warp_n32, Partials& partials) noexcept {
  static_assert(NPhase < 2U);
  unsigned int lane = threadIdx.x & 31U;
  if constexpr (NPhase == 1U) {
    asm volatile("" : "+r"(lane));
  }
  const unsigned int local_m = warp_m64 * 64U;
  const unsigned int local_n = warp_n32 * 32U + NPhase * 16U;
#pragma unroll
  for (unsigned int plane = 0U;
       plane <
       kSm87A4W4DownK512M256N128ResidentPhysicalK64PerScale;
       ++plane) {
    const Sm87A4W4BFragment b0 = load_b_ldmatrix_x2(
        codes.b[plane] + (local_n + 0U) * kPackedK64Bytes, lane);
    const Sm87A4W4BFragment b1 = load_b_ldmatrix_x2(
        codes.b[plane] + (local_n + 8U) * kPackedK64Bytes, lane);

    const Sm87A4W4AFragment a0 = load_a_ldmatrix_x4(
        codes.a[plane] + (local_m + 0U) * kPackedK64Bytes, lane);
    sm87_a4w4_mma_m16n8k64(partials.m0n0, a0, b0);
    sm87_a4w4_mma_m16n8k64(partials.m0n1, a0, b1);

    const Sm87A4W4AFragment a1 = load_a_ldmatrix_x4(
        codes.a[plane] + (local_m + 16U) * kPackedK64Bytes, lane);
    sm87_a4w4_mma_m16n8k64(partials.m1n0, a1, b0);
    sm87_a4w4_mma_m16n8k64(partials.m1n1, a1, b1);

    const Sm87A4W4AFragment a2 = load_a_ldmatrix_x4(
        codes.a[plane] + (local_m + 32U) * kPackedK64Bytes, lane);
    sm87_a4w4_mma_m16n8k64(partials.m2n0, a2, b0);
    sm87_a4w4_mma_m16n8k64(partials.m2n1, a2, b1);

    const Sm87A4W4AFragment a3 = load_a_ldmatrix_x4(
        codes.a[plane] + (local_m + 48U) * kPackedK64Bytes, lane);
    sm87_a4w4_mma_m16n8k64(partials.m3n0, a3, b0);
    sm87_a4w4_mma_m16n8k64(partials.m3n1, a3, b1);
  }
}

__device__ __forceinline__ void update_one(
    Float4& output, const Sm87A4W4Accumulator& partial,
    const float a0, const float a1, const float b0,
    const float b1) noexcept {
  output.x0 = __fmaf_rn(static_cast<float>(partial.x0),
                        __fmul_rn(a0, b0), output.x0);
  output.x1 = __fmaf_rn(static_cast<float>(partial.x1),
                        __fmul_rn(a0, b1), output.x1);
  output.x2 = __fmaf_rn(static_cast<float>(partial.x2),
                        __fmul_rn(a1, b0), output.x2);
  output.x3 = __fmaf_rn(static_cast<float>(partial.x3),
                        __fmul_rn(a1, b1), output.x3);
}

template <unsigned int MFragment, unsigned int NPhase>
__device__ __forceinline__ void apply_m_fragment(
    Float4& output_n0, Float4& output_n1,
    const Sm87A4W4Accumulator& partial_n0,
    const Sm87A4W4Accumulator& partial_n1,
    const ResidentScales& scale, const unsigned int warp_m64,
    const unsigned int warp_n32) noexcept {
  static_assert(MFragment < 4U);
  static_assert(NPhase < 2U);
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int row0 = lane >> 2U;
  const unsigned int row1 = row0 + 8U;
  const unsigned int column0 = 2U * (lane & 3U);
  const unsigned int column1 = column0 + 1U;
  const unsigned int local_m = warp_m64 * 64U + MFragment * 16U;
  const unsigned int local_n = warp_n32 * 32U + NPhase * 16U;
  const float a0 = decode_bf16(scale.a[local_m + row0]);
  const float a1 = decode_bf16(scale.a[local_m + row1]);
  float b0 = decode_bf16(scale.b[local_n + column0]);
  float b1 = decode_bf16(scale.b[local_n + column1]);
  update_one(output_n0, partial_n0, a0, a1, b0, b1);
  b0 = decode_bf16(scale.b[local_n + 8U + column0]);
  b1 = decode_bf16(scale.b[local_n + 8U + column1]);
  update_one(output_n1, partial_n1, a0, a1, b0, b1);
}

template <unsigned int NPhase>
__device__ __forceinline__ void apply_n_phase(
    Outputs& outputs, const Partials& partials,
    const ResidentScales& scale, const unsigned int warp_m64,
    const unsigned int warp_n32) noexcept {
  static_assert(NPhase < 2U);
  if constexpr (NPhase == 0U) {
    apply_m_fragment<0U, 0U>(outputs.m0n0, outputs.m0n1,
                             partials.m0n0, partials.m0n1, scale,
                             warp_m64, warp_n32);
    apply_m_fragment<1U, 0U>(outputs.m1n0, outputs.m1n1,
                             partials.m1n0, partials.m1n1, scale,
                             warp_m64, warp_n32);
    apply_m_fragment<2U, 0U>(outputs.m2n0, outputs.m2n1,
                             partials.m2n0, partials.m2n1, scale,
                             warp_m64, warp_n32);
    apply_m_fragment<3U, 0U>(outputs.m3n0, outputs.m3n1,
                             partials.m3n0, partials.m3n1, scale,
                             warp_m64, warp_n32);
  } else {
    apply_m_fragment<0U, 1U>(outputs.m0n2, outputs.m0n3,
                             partials.m0n0, partials.m0n1, scale,
                             warp_m64, warp_n32);
    apply_m_fragment<1U, 1U>(outputs.m1n2, outputs.m1n3,
                             partials.m1n0, partials.m1n1, scale,
                             warp_m64, warp_n32);
    apply_m_fragment<2U, 1U>(outputs.m2n2, outputs.m2n3,
                             partials.m2n0, partials.m2n1, scale,
                             warp_m64, warp_n32);
    apply_m_fragment<3U, 1U>(outputs.m3n2, outputs.m3n3,
                             partials.m3n0, partials.m3n1, scale,
                             warp_m64, warp_n32);
  }
}

template <unsigned int MFragment, unsigned int NFragment>
__device__ __forceinline__ void store_fragment(
    const Float4& accumulator,
    const unsigned int logical_token_count,
    const unsigned int m_tile_start, const unsigned int n_tile_start,
    const unsigned int warp_m64, const unsigned int warp_n32,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements) noexcept {
  static_assert(MFragment < 4U);
  static_assert(NFragment < 4U);
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int row0 = lane >> 2U;
  const unsigned int row1 = row0 + 8U;
  const unsigned int column0 = 2U * (lane & 3U);
  const unsigned int column1 = column0 + 1U;
  const unsigned int base_m =
      m_tile_start + warp_m64 * 64U + MFragment * 16U;
  const unsigned int base_n =
      n_tile_start + warp_n32 * 32U + NFragment * 8U;
  if (base_m + row0 < logical_token_count) {
    output_bf16[static_cast<std::size_t>(base_m + row0) *
                        output_row_stride_elements +
                    base_n + column0] = encode_bf16(accumulator.x0);
    output_bf16[static_cast<std::size_t>(base_m + row0) *
                        output_row_stride_elements +
                    base_n + column1] = encode_bf16(accumulator.x1);
  }
  if (base_m + row1 < logical_token_count) {
    output_bf16[static_cast<std::size_t>(base_m + row1) *
                        output_row_stride_elements +
                    base_n + column0] = encode_bf16(accumulator.x2);
    output_bf16[static_cast<std::size_t>(base_m + row1) *
                        output_row_stride_elements +
                    base_n + column1] = encode_bf16(accumulator.x3);
  }
}

__device__ __forceinline__ void store_outputs(
    const Outputs& outputs, const unsigned int logical_token_count,
    const unsigned int m_tile_start, const unsigned int n_tile_start,
    const unsigned int warp_m64, const unsigned int warp_n32,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements) noexcept {
#define Q3X_STORE(M, N)                                                       \
  store_fragment<M##U, N##U>(outputs.m##M##n##N, logical_token_count,        \
                             m_tile_start, n_tile_start, warp_m64, warp_n32, \
                             output_bf16, output_row_stride_elements)
  Q3X_STORE(0, 0);
  Q3X_STORE(0, 1);
  Q3X_STORE(0, 2);
  Q3X_STORE(0, 3);
  Q3X_STORE(1, 0);
  Q3X_STORE(1, 1);
  Q3X_STORE(1, 2);
  Q3X_STORE(1, 3);
  Q3X_STORE(2, 0);
  Q3X_STORE(2, 1);
  Q3X_STORE(2, 2);
  Q3X_STORE(2, 3);
  Q3X_STORE(3, 0);
  Q3X_STORE(3, 1);
  Q3X_STORE(3, 2);
  Q3X_STORE(3, 3);
#undef Q3X_STORE
}

}  // namespace

extern "C" __global__
    __launch_bounds__(kSm87A4W4DownK512M256N128ResidentThreads,
                      kSm87A4W4DownK512M256N128ResidentCtasPerSm)
void q3x_sm87_a4w4_down_k512_m256n128_k512resident_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const packed_b,
    const std::uint16_t* const b_k512_scales_bf16,
    const unsigned int logical_token_count,
    const unsigned int launch_token_count,
    const unsigned int k512_group_count,
    const unsigned int physical_k64_group_count,
    const unsigned int m_tile_count, const unsigned int work_tile_count,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements) {
  extern __shared__ __align__(16) unsigned char dynamic_storage[];
  auto& shared = *reinterpret_cast<SharedStorage*>(dynamic_storage);
  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int warp_m64 = warp & 3U;
  const unsigned int warp_n32 = warp >> 2U;

  for (unsigned int work = blockIdx.x; work < work_tile_count;
       work += gridDim.x) {
    const unsigned int n_tile = work / m_tile_count;
    const unsigned int m_tile = work - n_tile * m_tile_count;
    const unsigned int m_tile_start = m_tile * kTileM;
    const unsigned int n_tile_start = n_tile * kTileN;
    Outputs outputs{};

    for (unsigned int group = 0U; group < k512_group_count; ++group) {
      issue_k512_epoch(
          shared, packed_a, a_k512_scales_bf16, packed_b,
          b_k512_scales_bf16, launch_token_count, m_tile_start,
          n_tile_start, group, k512_group_count,
          physical_k64_group_count);
      cp_async_wait_all();
      __syncthreads();

      {
        Partials partials{};
        accumulate_n_phase<0U>(shared.codes, warp_m64, warp_n32, partials);
        apply_n_phase<0U>(outputs, partials, shared.scales, warp_m64,
                          warp_n32);
      }
      {
        Partials partials{};
        accumulate_n_phase<1U>(shared.codes, warp_m64, warp_n32, partials);
        apply_n_phase<1U>(outputs, partials, shared.scales, warp_m64,
                          warp_n32);
      }
      __syncthreads();
    }

    store_outputs(outputs, logical_token_count, m_tile_start, n_tile_start,
                  warp_m64, warp_n32, output_bf16,
                  output_row_stride_elements);
    __syncthreads();
  }
}

namespace {

[[nodiscard]] int validate_target(
    cudaDeviceProp* const output_properties = nullptr) noexcept {
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  cudaDeviceProp properties{};
  status = cudaGetDeviceProperties(&properties, device);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  if (properties.major != kSm87A4W4RequiredComputeMajor ||
      properties.minor != kSm87A4W4RequiredComputeMinor ||
      properties.multiProcessorCount != static_cast<int>(kRequiredSmCount) ||
      properties.sharedMemPerBlockOptin <
          kSm87A4W4DownK512M256N128ResidentDynamicSharedBytes) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (output_properties != nullptr) {
    *output_properties = properties;
  }
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] cudaError_t configure_kernel() noexcept {
  cudaError_t status = cudaFuncSetAttribute(
      q3x_sm87_a4w4_down_k512_m256n128_k512resident_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(
          kSm87A4W4DownK512M256N128ResidentDynamicSharedBytes));
  if (status != cudaSuccess) {
    return status;
  }
  return cudaFuncSetAttribute(
      q3x_sm87_a4w4_down_k512_m256n128_k512resident_kernel,
      cudaFuncAttributePreferredSharedMemoryCarveout, 100);
}

[[nodiscard]] int launch_impl(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const packed_b,
    const std::size_t packed_b_capacity_bytes,
    const std::uint16_t* const b_k512_scales_bf16,
    const std::size_t b_scale_capacity_elements,
    const std::size_t logical_token_count,
    const std::size_t launch_token_count, const std::size_t output_size,
    const std::size_t input_size, std::uint16_t* const output_bf16,
    const std::size_t output_row_stride_elements,
    const std::size_t output_capacity_elements,
    const unsigned int maximum_launch_ctas, const bool production_shape,
    void* const cuda_stream) noexcept {
  const Sm87A4W4DownK512M256N128ResidentPlan plan =
      production_shape
          ? sm87_a4w4_down_k512_m256n128_k512resident_plan(
                logical_token_count, launch_token_count, output_size,
                input_size)
          : sm87_a4w4_down_k512_m256n128_k512resident_test_plan(
                logical_token_count, launch_token_count, output_size,
                input_size);
  if (plan.launch_ctas == 0U || maximum_launch_ctas == 0U ||
      !aligned(packed_a, 16U) ||
      !aligned(a_k512_scales_bf16, 16U) ||
      !aligned(packed_b, 16U) ||
      !aligned(b_k512_scales_bf16, 16U) ||
      !aligned(output_bf16, alignof(std::uint32_t)) ||
      output_row_stride_elements < output_size ||
      output_row_stride_elements % 2U != 0U ||
      output_row_stride_elements >
          std::numeric_limits<unsigned int>::max() ||
      plan.launch_token_count >
          std::numeric_limits<unsigned int>::max() ||
      plan.k512_groups > std::numeric_limits<unsigned int>::max() ||
      plan.physical_k64_groups >
          std::numeric_limits<unsigned int>::max() ||
      plan.m_tiles > std::numeric_limits<unsigned int>::max() ||
      plan.work_tiles > std::numeric_limits<unsigned int>::max() ||
      !sm87_a4w4_down_k512_product_fits(
          launch_token_count, output_row_stride_elements)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const std::size_t required_a_bytes =
      sm87_a4w4_down_k512_packed_capacity_bytes(launch_token_count,
                                                input_size);
  const std::size_t required_b_bytes =
      sm87_a4w4_down_k512_packed_capacity_bytes(output_size, input_size);
  const std::size_t required_a_scales =
      sm87_a4w4_down_k512_scale_capacity_elements(launch_token_count,
                                                  input_size);
  const std::size_t required_b_scales =
      sm87_a4w4_down_k512_scale_capacity_elements(output_size, input_size);
  const std::size_t required_output_elements =
      logical_token_count * output_row_stride_elements;
  if (required_a_bytes == 0U || required_b_bytes == 0U ||
      required_a_scales == 0U || required_b_scales == 0U ||
      packed_a_capacity_bytes < required_a_bytes ||
      packed_b_capacity_bytes < required_b_bytes ||
      a_scale_capacity_elements < required_a_scales ||
      b_scale_capacity_elements < required_b_scales ||
      output_capacity_elements < required_output_elements ||
      !sm87_a4w4_down_k512_product_fits(required_a_scales,
                                        sizeof(std::uint16_t)) ||
      !sm87_a4w4_down_k512_product_fits(required_b_scales,
                                        sizeof(std::uint16_t)) ||
      !sm87_a4w4_down_k512_product_fits(required_output_elements,
                                        sizeof(std::uint16_t))) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const std::size_t required_a_scale_bytes =
      required_a_scales * sizeof(std::uint16_t);
  const std::size_t required_b_scale_bytes =
      required_b_scales * sizeof(std::uint16_t);
  const std::size_t required_output_bytes =
      required_output_elements * sizeof(std::uint16_t);
  if (byte_ranges_overlap(output_bf16, required_output_bytes, packed_a,
                          required_a_bytes) ||
      byte_ranges_overlap(output_bf16, required_output_bytes,
                          a_k512_scales_bf16,
                          required_a_scale_bytes) ||
      byte_ranges_overlap(output_bf16, required_output_bytes, packed_b,
                          required_b_bytes) ||
      byte_ranges_overlap(output_bf16, required_output_bytes,
                          b_k512_scales_bf16,
                          required_b_scale_bytes)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  Sm87A4W4DownK512M256N128ResidentResources resources{};
  const int resource_status =
      query_sm87_a4w4_down_k512_m256n128_k512resident_resources_cuda(
          &resources);
  if (resource_status != static_cast<int>(cudaSuccess)) {
    return resource_status;
  }

  const unsigned int planned_ctas =
      static_cast<unsigned int>(plan.launch_ctas);
  const unsigned int launch_ctas =
      planned_ctas < maximum_launch_ctas ? planned_ctas
                                         : maximum_launch_ctas;
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  q3x_sm87_a4w4_down_k512_m256n128_k512resident_kernel
      <<<launch_ctas, kThreads,
         kSm87A4W4DownK512M256N128ResidentDynamicSharedBytes, stream>>>(
          packed_a, a_k512_scales_bf16, packed_b,
          b_k512_scales_bf16,
          static_cast<unsigned int>(plan.logical_token_count),
          static_cast<unsigned int>(plan.launch_token_count),
          static_cast<unsigned int>(plan.k512_groups),
          static_cast<unsigned int>(plan.physical_k64_groups),
          static_cast<unsigned int>(plan.m_tiles),
          static_cast<unsigned int>(plan.work_tiles), output_bf16,
          static_cast<unsigned int>(output_row_stride_elements));
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace

int query_sm87_a4w4_down_k512_m256n128_k512resident_resources_cuda(
    Sm87A4W4DownK512M256N128ResidentResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4DownK512M256N128ResidentResources{};
  cudaDeviceProp properties{};
  const int target_status = validate_target(&properties);
  if (target_status != static_cast<int>(cudaSuccess)) {
    return target_status;
  }
  const cudaError_t configure_status = configure_kernel();
  if (configure_status != cudaSuccess) {
    return static_cast<int>(configure_status);
  }

  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      q3x_sm87_a4w4_down_k512_m256n128_k512resident_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      q3x_sm87_a4w4_down_k512_m256n128_k512resident_kernel,
      static_cast<int>(kThreads),
      kSm87A4W4DownK512M256N128ResidentDynamicSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87A4W4DownK512M256N128ResidentDynamicSharedBytes;
  resources->configured_dynamic_shared_limit_bytes =
      attributes.maxDynamicSharedSizeBytes;
  resources->device_optin_shared_limit_bytes =
      properties.sharedMemPerBlockOptin;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  resources->compute_major = properties.major;
  resources->compute_minor = properties.minor;

  if (resources->registers_per_thread <= 0 ||
      resources->registers_per_thread >
          static_cast<int>(
              kSm87A4W4DownK512M256N128ResidentMaximumRegisters) ||
      resources->static_shared_bytes != 0U ||
      resources->dynamic_shared_bytes !=
          kSm87A4W4DownK512M256N128ResidentDynamicSharedBytes ||
      resources->configured_dynamic_shared_limit_bytes <
          kSm87A4W4DownK512M256N128ResidentDynamicSharedBytes ||
      resources->device_optin_shared_limit_bytes <
          kSm87A4W4DownK512M256N128ResidentDynamicSharedBytes ||
      resources->local_bytes != 0U ||
      resources->maximum_threads_per_block < static_cast<int>(kThreads) ||
      resources->active_blocks_per_sm <
          static_cast<int>(
              kSm87A4W4DownK512M256N128ResidentCtasPerSm)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_a4w4_down_k512_m256n128_k512resident_bf16_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const packed_b,
    const std::size_t packed_b_capacity_bytes,
    const std::uint16_t* const b_k512_scales_bf16,
    const std::size_t b_scale_capacity_elements,
    const std::size_t logical_token_count,
    const std::size_t launch_token_count, const std::size_t output_size,
    const std::size_t input_size, std::uint16_t* const output_bf16,
    const std::size_t output_row_stride_elements,
    const std::size_t output_capacity_elements,
    void* const cuda_stream) noexcept {
  return launch_impl(
      packed_a, packed_a_capacity_bytes, a_k512_scales_bf16,
      a_scale_capacity_elements, packed_b, packed_b_capacity_bytes,
      b_k512_scales_bf16, b_scale_capacity_elements,
      logical_token_count, launch_token_count, output_size, input_size,
      output_bf16, output_row_stride_elements, output_capacity_elements,
      static_cast<unsigned int>(
          kSm87A4W4DownK512M256N128ResidentPersistentCtas),
      true, cuda_stream);
}

int launch_sm87_a4w4_down_k512_m256n128_k512resident_test_bf16_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const packed_b,
    const std::size_t packed_b_capacity_bytes,
    const std::uint16_t* const b_k512_scales_bf16,
    const std::size_t b_scale_capacity_elements,
    const std::size_t logical_token_count,
    const std::size_t launch_token_count, const std::size_t output_size,
    const std::size_t input_size, std::uint16_t* const output_bf16,
    const std::size_t output_row_stride_elements,
    const std::size_t output_capacity_elements,
    const unsigned int maximum_launch_ctas,
    void* const cuda_stream) noexcept {
  return launch_impl(
      packed_a, packed_a_capacity_bytes, a_k512_scales_bf16,
      a_scale_capacity_elements, packed_b, packed_b_capacity_bytes,
      b_k512_scales_bf16, b_scale_capacity_elements,
      logical_token_count, launch_token_count, output_size, input_size,
      output_bf16, output_row_stride_elements, output_capacity_elements,
      maximum_launch_ctas, false, cuda_stream);
}

}  // namespace q3x::kernels
