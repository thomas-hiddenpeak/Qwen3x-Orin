#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>

namespace {

constexpr unsigned int kWarpSize = 32U;
constexpr unsigned int kExtent = 64U;
constexpr unsigned int kElements = kExtent * kExtent;

struct AFragment {
  std::uint32_t x0;
  std::uint32_t x1;
  std::uint32_t x2;
  std::uint32_t x3;
};

struct BFragment {
  std::uint32_t x0;
  std::uint32_t x1;
};

struct Accumulator {
  float x0;
  float x1;
  float x2;
  float x3;
};

[[nodiscard]] __device__ __forceinline__ unsigned int swizzled_chunk(
    const unsigned int row, const unsigned int logical_chunk) {
  return logical_chunk ^ (row & 7U);
}

__device__ __forceinline__ void load_a_direct(
    AFragment& fragment, const std::uint16_t* const canonical,
    const unsigned int k16, const unsigned int lane) {
  const unsigned int quadrant = lane / 8U;
  const unsigned int row =
      (lane % 8U) + (quadrant & 1U) * 8U;
  const unsigned int column =
      k16 * 16U + (quadrant >> 1U) * 8U;
  const unsigned int address = static_cast<unsigned int>(
      __cvta_generic_to_shared(canonical + row * kExtent + column));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(fragment.x0), "=r"(fragment.x1),
        "=r"(fragment.x2), "=r"(fragment.x3)
      : "r"(address)
      : "memory");
}

__device__ __forceinline__ void load_a_candidate(
    AFragment& fragment, const std::uint16_t* const swizzled,
    const unsigned int m_panel, const unsigned int k16,
    const unsigned int lane) {
  const unsigned int quadrant = lane / 8U;
  const unsigned int row =
      m_panel * 16U + (lane % 8U) + (quadrant & 1U) * 8U;
  const unsigned int logical_chunk =
      k16 * 2U + (quadrant >> 1U);
  const unsigned int physical_chunk =
      swizzled_chunk(row, logical_chunk);
  const unsigned int address = static_cast<unsigned int>(
      __cvta_generic_to_shared(
          swizzled + row * kExtent + physical_chunk * 8U));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(fragment.x0), "=r"(fragment.x1),
        "=r"(fragment.x2), "=r"(fragment.x3)
      : "r"(address)
      : "memory");
}

// Independent oracle loader: load the explicit row-major true [K,N]
// backing with x2.trans, one K16 x N8 matrix-B fragment at a time.
__device__ __forceinline__ void load_b_direct(
    BFragment& fragment, const std::uint16_t* const true_kn,
    const unsigned int n_panel, const unsigned int k16,
    const unsigned int lane) {
  const unsigned int row = k16 * 16U + (lane & 15U);
  const unsigned int column = n_panel * 8U;
  const unsigned int address = static_cast<unsigned int>(
      __cvta_generic_to_shared(
          true_kn + row * kExtent + column));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
      "{%0, %1}, [%2];"
      : "=r"(fragment.x0), "=r"(fragment.x1)
      : "r"(address)
      : "memory");
}

__device__ __forceinline__ void load_b_candidate_pair(
    BFragment& first, BFragment& second,
    const std::uint16_t* const swizzled_nk,
    const unsigned int n_panel, const unsigned int k32,
    const unsigned int lane) {
  const unsigned int row = n_panel * 8U + (lane & 7U);
  const unsigned int logical_chunk = k32 * 4U + (lane >> 3U);
  const unsigned int physical_chunk =
      swizzled_chunk(row, logical_chunk);
  const unsigned int address = static_cast<unsigned int>(
      __cvta_generic_to_shared(
          swizzled_nk + row * kExtent + physical_chunk * 8U));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(first.x0), "=r"(first.x1),
        "=r"(second.x0), "=r"(second.x1)
      : "r"(address)
      : "memory");
}

__device__ __forceinline__ void zero(Accumulator& accumulator) {
  accumulator.x0 = 0.0F;
  accumulator.x1 = 0.0F;
  accumulator.x2 = 0.0F;
  accumulator.x3 = 0.0F;
}

__device__ __forceinline__ void mma(
    Accumulator& accumulator, const AFragment& a,
    const BFragment& b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(accumulator.x0), "+f"(accumulator.x1),
        "+f"(accumulator.x2), "+f"(accumulator.x3)
      : "r"(a.x0), "r"(a.x1), "r"(a.x2), "r"(a.x3),
        "r"(b.x0), "r"(b.x1));
}

[[nodiscard]] __device__ __forceinline__ std::uint32_t pack_pair(
    const float low, const float high) {
  std::uint32_t packed = 0U;
  asm("cvt.rn.bf16x2.f32 %0, %2, %1;"
      : "=r"(packed)
      : "f"(low), "f"(high));
  return packed;
}

__device__ __forceinline__ void store_state_bank(
    const Accumulator (&state)[2][8],
    std::uint16_t* const shared_state, const unsigned int bank,
    const unsigned int warp, const unsigned int lane) {
  const unsigned int group = lane >> 2U;
  const unsigned int pair = lane & 3U;
  const unsigned int key0 = warp * 16U + group;
  const unsigned int key1 = key0 + 8U;
  for (unsigned int panel = 0U; panel < 8U; ++panel) {
    const unsigned int value0 = panel * 8U + pair * 2U;
    const unsigned int value1 = value0 + 1U;
    const unsigned int chunk00 = swizzled_chunk(value0, key0 / 8U);
    const unsigned int chunk01 = swizzled_chunk(value0, key1 / 8U);
    const unsigned int chunk10 = swizzled_chunk(value1, key0 / 8U);
    const unsigned int chunk11 = swizzled_chunk(value1, key1 / 8U);
    const std::uint32_t packed0 =
        pack_pair(state[bank][panel].x0, state[bank][panel].x2);
    const std::uint32_t packed1 =
        pack_pair(state[bank][panel].x1, state[bank][panel].x3);
    shared_state[value0 * kExtent + chunk00 * 8U + key0 % 8U] =
        static_cast<std::uint16_t>(packed0);
    shared_state[value0 * kExtent + chunk01 * 8U + key1 % 8U] =
        static_cast<std::uint16_t>(packed0 >> 16U);
    shared_state[value1 * kExtent + chunk10 * 8U + key0 % 8U] =
        static_cast<std::uint16_t>(packed1);
    shared_state[value1 * kExtent + chunk11 * 8U + key1 % 8U] =
        static_cast<std::uint16_t>(packed1 >> 16U);
  }
}

__device__ __forceinline__ void copy_state_bank(
    const std::uint16_t* const shared_state,
    std::uint16_t* const global_state, const unsigned int value_half,
    const unsigned int bank, const unsigned int thread) {
  for (unsigned int vector = thread; vector < 512U; vector += 128U) {
    const unsigned int value = vector / 8U;
    const unsigned int logical_chunk = vector % 8U;
    const unsigned int key_base = logical_chunk * 8U;
    const unsigned int physical_chunk =
        swizzled_chunk(value, logical_chunk);
    const uint4 packed = *reinterpret_cast<const uint4*>(
        shared_state + value * kExtent + physical_chunk * 8U);
    auto* const destination = reinterpret_cast<uint4*>(
        global_state +
        static_cast<std::size_t>(value_half * 64U + value) * 128U +
        bank * 64U + key_base);
    *destination = packed;
  }
}

__device__ __forceinline__ void count_a_words(
    const AFragment& expected, const AFragment& actual,
    unsigned int* const count) {
  const unsigned int local =
      (expected.x0 != actual.x0 ? 1U : 0U) +
      (expected.x1 != actual.x1 ? 1U : 0U) +
      (expected.x2 != actual.x2 ? 1U : 0U) +
      (expected.x3 != actual.x3 ? 1U : 0U);
  if (local != 0U) {
    atomicAdd(count, local);
  }
}

__device__ __forceinline__ void count_b_words(
    const BFragment& expected, const BFragment& actual,
    unsigned int* const count) {
  const unsigned int local =
      (expected.x0 != actual.x0 ? 1U : 0U) +
      (expected.x1 != actual.x1 ? 1U : 0U);
  if (local != 0U) {
    atomicAdd(count, local);
  }
}

__device__ __forceinline__ void count_accumulator_words(
    const Accumulator& expected, const Accumulator& actual,
    unsigned int* const count) {
  const unsigned int local =
      (__float_as_uint(expected.x0) != __float_as_uint(actual.x0) ? 1U : 0U) +
      (__float_as_uint(expected.x1) != __float_as_uint(actual.x1) ? 1U : 0U) +
      (__float_as_uint(expected.x2) != __float_as_uint(actual.x2) ? 1U : 0U) +
      (__float_as_uint(expected.x3) != __float_as_uint(actual.x3) ? 1U : 0U);
  if (local != 0U) {
    atomicAdd(count, local);
  }
}

__global__ void faithful_fragment_sentinel_kernel(
    const std::uint16_t* const global_a,
    const std::uint16_t* const global_b,
    unsigned int* const mismatches) {
  __shared__ __align__(16) std::uint16_t canonical_a[kElements];
  __shared__ __align__(16) std::uint16_t candidate_a[kElements];
  __shared__ __align__(16) std::uint16_t canonical_b[kElements];
  __shared__ __align__(16) std::uint16_t candidate_b[kElements];
  const unsigned int lane = threadIdx.x;
  for (unsigned int index = lane; index < kElements;
       index += kWarpSize) {
    const unsigned int row = index / kExtent;
    const unsigned int column = index % kExtent;
    const unsigned int physical_chunk =
        swizzled_chunk(row, column / 8U);
    const unsigned int physical =
        row * kExtent + physical_chunk * 8U + column % 8U;
    canonical_a[index] = global_a[index];
    candidate_a[physical] = global_a[index];
    canonical_b[column * kExtent + row] = global_b[index];
    candidate_b[physical] = global_b[index];
  }
  __syncwarp();

  AFragment direct_a[2]{};
  AFragment loaded_a[2]{};
  BFragment direct_b[2]{};
  BFragment loaded_b[2]{};
  for (unsigned int k16 = 0U; k16 < 2U; ++k16) {
    load_a_direct(direct_a[k16], canonical_a, k16, lane);
    load_a_candidate(loaded_a[k16], candidate_a, 0U, k16, lane);
    load_b_direct(direct_b[k16], canonical_b, 0U, k16, lane);
    count_a_words(direct_a[k16], loaded_a[k16], mismatches + 0U);
  }
  load_b_candidate_pair(loaded_b[0], loaded_b[1], candidate_b, 0U, 0U,
                        lane);
  for (unsigned int k16 = 0U; k16 < 2U; ++k16) {
    count_b_words(direct_b[k16], loaded_b[k16], mismatches + 1U);
  }

  Accumulator direct{};
  Accumulator a_probe{};
  Accumulator b_probe{};
  Accumulator combined{};
  zero(direct);
  zero(a_probe);
  zero(b_probe);
  zero(combined);
  for (unsigned int k16 = 0U; k16 < 2U; ++k16) {
    mma(direct, direct_a[k16], direct_b[k16]);
    mma(a_probe, loaded_a[k16], direct_b[k16]);
    mma(b_probe, direct_a[k16], loaded_b[k16]);
    mma(combined, loaded_a[k16], loaded_b[k16]);
  }
  count_accumulator_words(direct, a_probe, mismatches + 2U);
  count_accumulator_words(direct, b_probe, mismatches + 3U);
  count_accumulator_words(direct, combined, mismatches + 4U);
}

// Full candidate update epilogue with two K64 banks, two V64 halves, four
// K16-owning warps and all four reduction fragments. The coordinate-coded
// identity input makes any accumulator/store permutation observable.
__global__ __launch_bounds__(128)
void faithful_semantic_epilogue_sentinel_kernel(
    const std::uint16_t* const global_k_transpose,
    const std::uint16_t* const global_v_backing,
    std::uint16_t* const global_h) {
  __shared__ __align__(16) std::uint16_t shared_a0[kElements];
  __shared__ __align__(16) std::uint16_t shared_a1[kElements];
  __shared__ __align__(16) std::uint16_t shared_b[kElements];
  __shared__ __align__(16) std::uint16_t shared_state[kElements];
  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread / 32U;
  const unsigned int lane = thread % 32U;
  const unsigned int value_half = blockIdx.x;

  for (unsigned int index = thread; index < kElements; index += 128U) {
    const unsigned int row = index / kExtent;
    const unsigned int column = index % kExtent;
    const unsigned int physical_chunk =
        swizzled_chunk(row, column / 8U);
    const unsigned int physical =
        row * kExtent + physical_chunk * 8U + column % 8U;
    shared_a0[physical] = global_k_transpose[index];
    shared_a1[physical] = global_k_transpose[kElements + index];
    shared_b[physical] = global_v_backing[
        (value_half * 64U + row) * 64U + column];
  }
  __syncthreads();

  Accumulator state[2][8];
  for (unsigned int bank = 0U; bank < 2U; ++bank) {
    for (unsigned int panel = 0U; panel < 8U; ++panel) {
      zero(state[bank][panel]);
    }
  }
  for (unsigned int bank = 0U; bank < 2U; ++bank) {
    const std::uint16_t* const selected_a =
        bank == 0U ? shared_a0 : shared_a1;
    for (unsigned int token_pair = 0U; token_pair < 2U;
         ++token_pair) {
      AFragment a0{};
      AFragment a1{};
      load_a_candidate(a0, selected_a, warp, token_pair * 2U, lane);
      load_a_candidate(a1, selected_a, warp, token_pair * 2U + 1U,
                       lane);
      for (unsigned int panel = 0U; panel < 8U; ++panel) {
        BFragment b0{};
        BFragment b1{};
        load_b_candidate_pair(b0, b1, shared_b, panel, token_pair,
                              lane);
        mma(state[bank][panel], a0, b0);
        mma(state[bank][panel], a1, b1);
      }
    }
  }

  for (unsigned int bank = 0U; bank < 2U; ++bank) {
    store_state_bank(state, shared_state, bank, warp, lane);
    __syncthreads();
    copy_state_bank(shared_state, global_h, value_half, bank, thread);
    __syncthreads();
  }
}

[[nodiscard]] std::uint16_t encode_bf16(const float value) noexcept {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t rounded =
      bits + 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(rounded >> 16U);
}

[[nodiscard]] float decode_bf16(const std::uint16_t value) noexcept {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
  float decoded = 0.0F;
  std::memcpy(&decoded, &bits, sizeof(decoded));
  return decoded;
}

[[nodiscard]] bool run_semantic_epilogue_sentinel() {
  constexpr unsigned int kFullExtent = 128U;
  constexpr unsigned int kKElements = kFullExtent * 64U;
  constexpr unsigned int kVElements = kFullExtent * 64U;
  constexpr unsigned int kHElements = kFullExtent * kFullExtent;
  std::uint16_t* k_transpose = nullptr;
  std::uint16_t* v_backing = nullptr;
  std::uint16_t* h = nullptr;
  cudaError_t status = cudaMallocManaged(
      &k_transpose, kKElements * sizeof(*k_transpose));
  if (status == cudaSuccess) {
    status = cudaMallocManaged(&v_backing,
                               kVElements * sizeof(*v_backing));
  }
  if (status == cudaSuccess) {
    status = cudaMallocManaged(&h, kHElements * sizeof(*h));
  }
  if (status != cudaSuccess) {
    std::cerr << "semantic sentinel allocation failed: "
              << cudaGetErrorString(status) << '\n';
    (void)cudaFree(h);
    (void)cudaFree(v_backing);
    (void)cudaFree(k_transpose);
    return false;
  }

  std::memset(k_transpose, 0, kKElements * sizeof(*k_transpose));
  for (unsigned int bank = 0U; bank < 2U; ++bank) {
    const std::uint16_t diagonal = encode_bf16(
        bank == 0U ? 1.0F : 2.0F);
    for (unsigned int key = 0U; key < 64U; ++key) {
      const unsigned int token = (13U * key + 7U) % 64U;
      k_transpose[(bank * 64U + key) * 64U + token] = diagonal;
    }
  }
  for (unsigned int value = 0U; value < kFullExtent; ++value) {
    for (unsigned int token = 0U; token < 64U; ++token) {
      v_backing[value * 64U + token] = static_cast<std::uint16_t>(
          0x2000U + token * kFullExtent + value);
    }
  }
  std::memset(h, 0xff, kHElements * sizeof(*h));
  faithful_semantic_epilogue_sentinel_kernel<<<2U, 128U>>>(
      k_transpose, v_backing, h);
  status = cudaGetLastError();
  if (status == cudaSuccess) {
    status = cudaDeviceSynchronize();
  }
  if (status != cudaSuccess) {
    std::cerr << "semantic sentinel failed: "
              << cudaGetErrorString(status) << '\n';
    (void)cudaFree(h);
    (void)cudaFree(v_backing);
    (void)cudaFree(k_transpose);
    return false;
  }

  const auto expected = [&](const unsigned int value,
                            const unsigned int key) noexcept {
    const unsigned int local_key = key % 64U;
    const unsigned int token = (13U * local_key + 7U) % 64U;
    const std::uint16_t input =
        v_backing[value * 64U + token];
    return key < 64U
               ? input
               : encode_bf16(2.0F * decode_bf16(input));
  };
  std::size_t exact = 0U;
  std::size_t a_inverse_permutation = 0U;
  std::size_t transpose = 0U;
  std::size_t tile_transpose = 0U;
  std::size_t tile8_transpose = 0U;
  std::size_t tile16_transpose = 0U;
  std::size_t tile32_transpose = 0U;
  std::size_t key_bank_swap = 0U;
  std::size_t value_half_swap = 0U;
  std::size_t key_plus8_pair = 0U;
  std::size_t value_plus8_pair = 0U;
  std::size_t value_adjacent_pair = 0U;
  unsigned int first_value = kFullExtent;
  unsigned int first_key = kFullExtent;
  for (unsigned int value = 0U; value < kFullExtent; ++value) {
    for (unsigned int key = 0U; key < kFullExtent; ++key) {
      const std::uint16_t actual = h[value * kFullExtent + key];
      const bool differs = actual != expected(value, key);
      exact += differs ? 1U : 0U;
      if (differs && first_value == kFullExtent) {
        first_value = value;
        first_key = key;
      }
      transpose += actual != expected(key, value) ? 1U : 0U;
      const unsigned int inverse_token =
          (5U * ((key % 64U) + 64U - 7U)) % 64U;
      const std::uint16_t inverse_input =
          v_backing[value * 64U + inverse_token];
      const std::uint16_t inverse_expected =
          key < 64U
              ? inverse_input
              : encode_bf16(2.0F * decode_bf16(inverse_input));
      a_inverse_permutation +=
          actual != inverse_expected ? 1U : 0U;
      const unsigned int transposed_value =
          (value & 64U) + (key & 63U);
      const unsigned int transposed_key =
          (key & 64U) + (value & 63U);
      tile_transpose +=
          actual != expected(transposed_value, transposed_key) ? 1U : 0U;
      const auto block_transpose_mismatch =
          [&](const unsigned int extent) noexcept {
            const unsigned int block_mask = ~(extent - 1U);
            const unsigned int local_mask = extent - 1U;
            const unsigned int mapped_value =
                (value & block_mask) + (key & local_mask);
            const unsigned int mapped_key =
                (key & block_mask) + (value & local_mask);
            return actual != expected(mapped_value, mapped_key) ? 1U : 0U;
          };
      tile8_transpose += block_transpose_mismatch(8U);
      tile16_transpose += block_transpose_mismatch(16U);
      tile32_transpose += block_transpose_mismatch(32U);
      key_bank_swap +=
          actual != expected(value, key ^ 64U) ? 1U : 0U;
      value_half_swap +=
          actual != expected(value ^ 64U, key) ? 1U : 0U;
      key_plus8_pair +=
          actual != expected(value, key ^ 8U) ? 1U : 0U;
      value_plus8_pair +=
          actual != expected(value ^ 8U, key) ? 1U : 0U;
      value_adjacent_pair +=
          actual != expected(value ^ 1U, key) ? 1U : 0U;
    }
  }
  std::cout << "GDN_VLLM_FAITHFUL_SEMANTIC_EPILOGUE_SENTINEL"
            << " exact_mismatch=" << exact
            << " a_inverse_permutation_mismatch="
            << a_inverse_permutation
            << " transpose_mismatch=" << transpose
            << " tile_transpose_mismatch=" << tile_transpose
            << " tile8_transpose_mismatch=" << tile8_transpose
            << " tile16_transpose_mismatch=" << tile16_transpose
            << " tile32_transpose_mismatch=" << tile32_transpose
            << " key_bank_swap_mismatch=" << key_bank_swap
            << " value_half_swap_mismatch=" << value_half_swap
            << " key_plus8_pair_mismatch=" << key_plus8_pair
            << " value_plus8_pair_mismatch=" << value_plus8_pair
            << " value_adjacent_pair_mismatch=" << value_adjacent_pair;
  if (first_value != kFullExtent) {
    std::cout << " first_value=" << first_value
              << " first_key=" << first_key
              << " actual_bf16="
              << h[first_value * kFullExtent + first_key]
              << " expected_bf16=" << expected(first_value, first_key);
  }
  std::cout << '\n';
  (void)cudaFree(h);
  (void)cudaFree(v_backing);
  (void)cudaFree(k_transpose);
  return exact == 0U;
}

}  // namespace

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess ||
      device_count == 0) {
    std::cout << "SKIP: faithful fragment sentinel requires CUDA\n";
    return 77;
  }
  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess ||
      properties.major != 8 || properties.minor != 7) {
    std::cout << "SKIP: faithful fragment sentinel requires SM87\n";
    return 77;
  }

  std::uint16_t* a = nullptr;
  std::uint16_t* b = nullptr;
  unsigned int* mismatches = nullptr;
  cudaError_t status = cudaMallocManaged(&a, kElements * sizeof(*a));
  if (status == cudaSuccess) {
    status = cudaMallocManaged(&b, kElements * sizeof(*b));
  }
  if (status == cudaSuccess) {
    status = cudaMallocManaged(&mismatches, 5U * sizeof(*mismatches));
  }
  if (status != cudaSuccess) {
    std::cerr << "allocation failed: " << cudaGetErrorString(status) << '\n';
    (void)cudaFree(mismatches);
    (void)cudaFree(b);
    (void)cudaFree(a);
    return 2;
  }
  for (unsigned int row = 0U; row < kExtent; ++row) {
    for (unsigned int column = 0U; column < kExtent; ++column) {
      const unsigned int index = row * kExtent + column;
      const int a_code =
          static_cast<int>((row * 29U + column * 17U) % 61U) - 30;
      const int b_code =
          static_cast<int>((row * 13U + column * 31U) % 67U) - 33;
      a[index] = encode_bf16(static_cast<float>(a_code) / 16.0F);
      b[index] = encode_bf16(static_cast<float>(b_code) / 32.0F);
    }
  }
  std::memset(mismatches, 0, 5U * sizeof(*mismatches));
  faithful_fragment_sentinel_kernel<<<1U, kWarpSize>>>(a, b, mismatches);
  status = cudaGetLastError();
  if (status == cudaSuccess) {
    status = cudaDeviceSynchronize();
  }
  if (status != cudaSuccess) {
    std::cerr << "sentinel failed: " << cudaGetErrorString(status) << '\n';
    (void)cudaFree(mismatches);
    (void)cudaFree(b);
    (void)cudaFree(a);
    return 3;
  }
  std::cout << "GDN_VLLM_FAITHFUL_FRAGMENT_SENTINEL"
            << " a_fragment_word_mismatch=" << mismatches[0]
            << " b_fragment_word_mismatch=" << mismatches[1]
            << " a_only_mma_word_mismatch=" << mismatches[2]
            << " b_only_mma_word_mismatch=" << mismatches[3]
            << " combined_mma_word_mismatch=" << mismatches[4] << '\n';
  const bool fragments_passed =
      mismatches[0] == 0U && mismatches[1] == 0U &&
      mismatches[2] == 0U && mismatches[3] == 0U &&
      mismatches[4] == 0U;
  (void)cudaFree(mismatches);
  (void)cudaFree(b);
  (void)cudaFree(a);
  const bool semantic_passed = run_semantic_epilogue_sentinel();
  return fragments_passed && semantic_passed ? 0 : 1;
}
