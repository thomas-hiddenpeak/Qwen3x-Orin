#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {

namespace wmma = nvcuda::wmma;

constexpr unsigned int kWarpSize = 32U;
constexpr unsigned int kTileExtent = 16U;
constexpr unsigned int kTileElements = kTileExtent * kTileExtent;
constexpr unsigned int kFragmentElements = 8U;
constexpr unsigned int kProbeElements = kWarpSize * kFragmentElements;
constexpr unsigned int kGuardElements = 32U;
constexpr std::uint16_t kMarkerBase = 0x3e00U;
constexpr std::uint16_t kInputGuard = 0x55aaU;
constexpr std::uint16_t kProbeGuard = 0xa55aU;
constexpr std::uint32_t kOutputGuard = 0xdeadbeefU;

static_assert(kProbeElements == kTileElements);

[[nodiscard]] __device__ __forceinline__ std::uint32_t pack_bf16_pair(
    const __nv_bfloat16 low, const __nv_bfloat16 high) noexcept {
  return static_cast<std::uint32_t>(__bfloat16_as_ushort(low)) |
         (static_cast<std::uint32_t>(__bfloat16_as_ushort(high)) << 16U);
}

[[nodiscard]] __host__ __device__ constexpr std::uint16_t marker_for_linear(
    const unsigned int linear) noexcept {
  return static_cast<std::uint16_t>(kMarkerBase + linear);
}

// Frozen from the SM87 probe below. For a col-major B tile, every lane owns
// two adjacent K values in each of the lower/upper K8 and N8 quadrants.
[[nodiscard]] __host__ __device__ constexpr unsigned int
expected_k_for_lane_slot(const unsigned int lane,
                         const unsigned int slot) noexcept {
  return 2U * (lane % 4U) + (slot % 2U) + 8U * ((slot / 2U) % 2U);
}

[[nodiscard]] __host__ __device__ constexpr unsigned int
expected_n_for_lane_slot(const unsigned int lane,
                         const unsigned int slot) noexcept {
  return lane / 4U + 8U * (slot / 4U);
}

[[nodiscard]] __host__ __device__ constexpr unsigned int
expected_linear_for_lane_slot(const unsigned int lane,
                              const unsigned int slot) noexcept {
  return expected_k_for_lane_slot(lane, slot) +
         expected_n_for_lane_slot(lane, slot) * kTileExtent;
}

extern "C" __global__ void q3x_sm87_bf16_matrix_b_fragment_probe_kernel(
    std::uint16_t* const exported_raw) {
  __shared__ __align__(32) __nv_bfloat16 tile[kTileElements];
  const unsigned int lane = threadIdx.x;
  for (unsigned int linear = lane; linear < kTileElements;
       linear += kWarpSize) {
    tile[linear] = __ushort_as_bfloat16(marker_for_linear(linear));
  }
  __syncwarp();

  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                 wmma::col_major>
      fragment;
  static_assert(fragment.num_elements == kFragmentElements);
  wmma::load_matrix_sync(fragment, tile, kTileExtent);
#pragma unroll
  for (unsigned int slot = 0U; slot < kFragmentElements; ++slot) {
    exported_raw[lane * kFragmentElements + slot] =
        __bfloat16_as_ushort(fragment.x[slot]);
  }
}

extern "C" __global__ void q3x_sm87_bf16_matrix_b_register_feed_kernel(
    const std::uint16_t* const a_raw, const std::uint16_t* const b_raw,
    std::uint32_t* const shared_load_output,
    std::uint32_t* const register_feed_output) {
  __shared__ __align__(32) __nv_bfloat16 shared_a[kTileElements];
  __shared__ __align__(32) __nv_bfloat16 shared_b[kTileElements];
  const unsigned int lane = threadIdx.x;
  for (unsigned int linear = lane; linear < kTileElements;
       linear += kWarpSize) {
    shared_a[linear] = __ushort_as_bfloat16(a_raw[linear]);
    shared_b[linear] = __ushort_as_bfloat16(b_raw[linear]);
  }
  __syncwarp();

  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                 wmma::row_major>
      a_fragment;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                 wmma::col_major>
      shared_b_fragment;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                 wmma::col_major>
      register_b_fragment;
  static_assert(shared_b_fragment.num_elements == kFragmentElements);
  static_assert(register_b_fragment.num_elements == kFragmentElements);
  wmma::load_matrix_sync(a_fragment, shared_a, kTileExtent);
  wmma::load_matrix_sync(shared_b_fragment, shared_b, kTileExtent);
#pragma unroll
  for (unsigned int slot = 0U; slot < kFragmentElements; ++slot) {
    const unsigned int linear = expected_linear_for_lane_slot(lane, slot);
    register_b_fragment.x[slot] = __ushort_as_bfloat16(b_raw[linear]);
  }

  wmma::fragment<wmma::accumulator, 16, 16, 16, float>
      shared_accumulator;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float>
      register_accumulator;
  wmma::fill_fragment(shared_accumulator, 0.0F);
  wmma::fill_fragment(register_accumulator, 0.0F);
  wmma::mma_sync(shared_accumulator, a_fragment, shared_b_fragment,
                 shared_accumulator);
  wmma::mma_sync(register_accumulator, a_fragment, register_b_fragment,
                 register_accumulator);
  wmma::store_matrix_sync(reinterpret_cast<float*>(shared_load_output),
                          shared_accumulator, kTileExtent,
                          wmma::mem_row_major);
  wmma::store_matrix_sync(reinterpret_cast<float*>(register_feed_output),
                          register_accumulator, kTileExtent,
                          wmma::mem_row_major);
}

// Test-only sentinel for the inline instruction shape needed by the proposed
// M64xN256 Gate cell.  SM87 lowers one public WMMA m16n16 operation to two
// m16n8 instructions.  Feed those two instructions from the exact A/B
// fragment registers loaded by WMMA, then store their accumulator registers
// with the documented m16n8 lane ownership.  A bitwise match proves both the
// operand split and accumulator ownership before either is used in a full
// projection kernel.
extern "C" __global__ void q3x_sm87_bf16_mma_m16n8k16_sentinel_kernel(
    const std::uint16_t* const a_raw, const std::uint16_t* const b_raw,
    std::uint32_t* const wmma_output, std::uint32_t* const inline_output) {
  __shared__ __align__(32) __nv_bfloat16 shared_a[kTileElements];
  __shared__ __align__(32) __nv_bfloat16 shared_b[kTileElements];
  const unsigned int lane = threadIdx.x;
  for (unsigned int linear = lane; linear < kTileElements;
       linear += kWarpSize) {
    shared_a[linear] = __ushort_as_bfloat16(a_raw[linear]);
    shared_b[linear] = __ushort_as_bfloat16(b_raw[linear]);
  }
  __syncwarp();

  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                 wmma::row_major>
      a_fragment;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                 wmma::col_major>
      b_fragment;
  static_assert(a_fragment.num_elements == kFragmentElements);
  static_assert(b_fragment.num_elements == kFragmentElements);
  wmma::load_matrix_sync(a_fragment, shared_a, kTileExtent);
  wmma::load_matrix_sync(b_fragment, shared_b, kTileExtent);

  const std::uint32_t a0 = pack_bf16_pair(a_fragment.x[0], a_fragment.x[1]);
  const std::uint32_t a1 = pack_bf16_pair(a_fragment.x[2], a_fragment.x[3]);
  const std::uint32_t a2 = pack_bf16_pair(a_fragment.x[4], a_fragment.x[5]);
  const std::uint32_t a3 = pack_bf16_pair(a_fragment.x[6], a_fragment.x[7]);
  const std::uint32_t b0 = pack_bf16_pair(b_fragment.x[0], b_fragment.x[1]);
  const std::uint32_t b1 = pack_bf16_pair(b_fragment.x[2], b_fragment.x[3]);
  const std::uint32_t b2 = pack_bf16_pair(b_fragment.x[4], b_fragment.x[5]);
  const std::uint32_t b3 = pack_bf16_pair(b_fragment.x[6], b_fragment.x[7]);

  float d00 = 0.0F;
  float d01 = 0.0F;
  float d02 = 0.0F;
  float d03 = 0.0F;
  float d10 = 0.0F;
  float d11 = 0.0F;
  float d12 = 0.0F;
  float d13 = 0.0F;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(d00), "+f"(d01), "+f"(d02), "+f"(d03)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b2), "r"(b3));
#endif

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> reference;
  wmma::fill_fragment(reference, 0.0F);
  wmma::mma_sync(reference, a_fragment, b_fragment, reference);
  wmma::store_matrix_sync(reinterpret_cast<float*>(wmma_output), reference,
                          kTileExtent, wmma::mem_row_major);

  const unsigned int group = lane / 4U;
  const unsigned int thread_in_group = lane % 4U;
  const unsigned int column0 = 2U * thread_in_group;
  const unsigned int row0 = group;
  const unsigned int row1 = group + 8U;
  auto* const output = reinterpret_cast<float*>(inline_output);
  output[row0 * kTileExtent + column0] = d00;
  output[row0 * kTileExtent + column0 + 1U] = d01;
  output[row1 * kTileExtent + column0] = d02;
  output[row1 * kTileExtent + column0 + 1U] = d03;
  output[row0 * kTileExtent + column0 + 8U] = d10;
  output[row0 * kTileExtent + column0 + 9U] = d11;
  output[row1 * kTileExtent + column0 + 8U] = d12;
  output[row1 * kTileExtent + column0 + 9U] = d13;
}

class TestContext {
 public:
  void expect(const bool condition, const std::string& message) {
    if (!condition) {
      ++failures_;
      std::cerr << "FAIL: " << message << '\n';
    }
  }

  [[nodiscard]] bool cuda_ok(const cudaError_t status,
                             const std::string& operation) {
    expect(status == cudaSuccess,
           operation + ": " + cudaGetErrorString(status));
    return status == cudaSuccess;
  }

  [[nodiscard]] int failures() const noexcept { return failures_; }

 private:
  int failures_ = 0;
};

template <typename T>
class ManagedBuffer {
 public:
  ManagedBuffer() = default;
  ManagedBuffer(const ManagedBuffer&) = delete;
  ManagedBuffer& operator=(const ManagedBuffer&) = delete;

  ~ManagedBuffer() {
    if (data_ != nullptr) {
      (void)cudaFree(data_);
    }
  }

  [[nodiscard]] cudaError_t allocate(const std::size_t count) {
    count_ = count;
    return cudaMallocManaged(reinterpret_cast<void**>(&data_),
                             count * sizeof(T));
  }

  [[nodiscard]] T* data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return count_; }
  [[nodiscard]] T& operator[](const std::size_t index) noexcept {
    return data_[index];
  }
  [[nodiscard]] const T& operator[](const std::size_t index) const noexcept {
    return data_[index];
  }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0U;
};

template <typename T>
void fill_buffer(ManagedBuffer<T>& buffer, const T value) {
  std::fill_n(buffer.data(), buffer.size(), value);
}

template <typename T>
void expect_guards(TestContext& test, const ManagedBuffer<T>& buffer,
                   const std::size_t payload_elements, const T guard,
                   const std::string& label) {
  bool intact = true;
  for (std::size_t index = 0U; index < kGuardElements; ++index) {
    intact = intact && buffer[index] == guard;
    intact = intact &&
             buffer[kGuardElements + payload_elements + index] == guard;
  }
  test.expect(intact, label + " guard regions remain intact");
}

[[nodiscard]] std::uint16_t make_a_value(const unsigned int linear) {
  const std::uint16_t magnitude =
      static_cast<std::uint16_t>(0x3d00U + (linear % 0x0080U));
  return static_cast<std::uint16_t>(
      magnitude | ((linear & 1U) == 0U ? 0U : 0x8000U));
}

[[nodiscard]] std::uint16_t make_b_value(const unsigned int linear) {
  return marker_for_linear(linear);
}

[[nodiscard]] bool query_and_print_resources(TestContext& test) {
  const auto print_one = [&](const char* const name, const auto kernel) {
    cudaFuncAttributes attributes{};
    if (!test.cuda_ok(cudaFuncGetAttributes(&attributes, kernel),
                      std::string("query resources for ") + name)) {
      return false;
    }
    int active_blocks = 0;
    if (!test.cuda_ok(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                          &active_blocks, kernel,
                          static_cast<int>(kWarpSize), 0U),
                      std::string("query occupancy for ") + name)) {
      return false;
    }
    std::cout << "BF16_B_FRAGMENT_RESOURCE: kernel=" << name
              << " registers_per_thread=" << attributes.numRegs
              << " static_shared_bytes=" << attributes.sharedSizeBytes
              << " local_bytes=" << attributes.localSizeBytes
              << " max_threads_per_block=" << attributes.maxThreadsPerBlock
              << " active_blocks_per_sm=" << active_blocks
              << " binary_version=" << attributes.binaryVersion
              << " ptx_version=" << attributes.ptxVersion << '\n';
    return true;
  };

  const bool probe_ok = print_one(
      "q3x_sm87_bf16_matrix_b_fragment_probe_kernel",
      q3x_sm87_bf16_matrix_b_fragment_probe_kernel);
  const bool register_ok = print_one(
      "q3x_sm87_bf16_matrix_b_register_feed_kernel",
      q3x_sm87_bf16_matrix_b_register_feed_kernel);
  const bool inline_mma_ok = print_one(
      "q3x_sm87_bf16_mma_m16n8k16_sentinel_kernel",
      q3x_sm87_bf16_mma_m16n8k16_sentinel_kernel);
  std::cout
      << "BF16_B_FRAGMENT_SASS_INTERFACE: symbols="
      << "q3x_sm87_bf16_matrix_b_fragment_probe_kernel,"
      << "q3x_sm87_bf16_matrix_b_register_feed_kernel,"
      << "q3x_sm87_bf16_mma_m16n8k16_sentinel_kernel"
      << " tool=cuobjdump flags=--dump-resource-usage,--dump-sass\n";
  return probe_ok && register_ok && inline_mma_ok;
}

[[nodiscard]] bool collect_and_validate_mapping(
    TestContext& test, ManagedBuffer<std::uint16_t>& probe,
    std::array<unsigned int, kProbeElements>* const mapping) {
  fill_buffer(probe, kProbeGuard);
  q3x_sm87_bf16_matrix_b_fragment_probe_kernel<<<1U, kWarpSize>>>(
      probe.data() + kGuardElements);
  if (!test.cuda_ok(cudaGetLastError(), "launch matrix_b fragment probe") ||
      !test.cuda_ok(cudaDeviceSynchronize(),
                    "synchronize matrix_b fragment probe")) {
    return false;
  }
  expect_guards(test, probe, kProbeElements, kProbeGuard,
                "fragment probe output");

  std::array<unsigned int, kTileElements> seen{};
  bool marker_range_valid = true;
  bool frozen_mapping_valid = true;
  for (unsigned int index = 0U; index < kProbeElements; ++index) {
    const std::uint16_t raw = probe[kGuardElements + index];
    if (raw < kMarkerBase ||
        raw >= static_cast<std::uint16_t>(kMarkerBase + kTileElements)) {
      marker_range_valid = false;
      (*mapping)[index] = kTileElements;
      continue;
    }
    const unsigned int linear = static_cast<unsigned int>(raw - kMarkerBase);
    (*mapping)[index] = linear;
    ++seen[linear];
    const unsigned int lane = index / kFragmentElements;
    const unsigned int slot = index % kFragmentElements;
    frozen_mapping_valid =
        frozen_mapping_valid &&
        linear == expected_linear_for_lane_slot(lane, slot);
  }
  test.expect(marker_range_valid,
              "all exported fragment values are unique marker payloads");
  const bool bijective =
      std::all_of(seen.begin(), seen.end(),
                  [](const unsigned int count) { return count == 1U; });
  test.expect(bijective,
              "SM87 matrix_b lane/slot mapping is a 256-element bijection");
  test.expect(frozen_mapping_valid,
              "SM87 matrix_b lane/slot mapping matches the frozen formula");

  std::cout
      << "BF16_B_FRAGMENT_MAP_FORMULA: "
      << "k=2*(lane%4)+(slot%2)+8*((slot/2)%2) "
      << "n=lane/4+8*(slot/4) linear=k+16*n\n";

  for (unsigned int lane = 0U; lane < kWarpSize; ++lane) {
    std::ostringstream line;
    line << "BF16_B_FRAGMENT_MAP: lane=" << std::setw(2) << lane
         << " slots=";
    for (unsigned int slot = 0U; slot < kFragmentElements; ++slot) {
      const unsigned int linear =
          (*mapping)[lane * kFragmentElements + slot];
      const unsigned int k = linear % kTileExtent;
      const unsigned int n = linear / kTileExtent;
      line << (slot == 0U ? "" : ",") << slot << "->(" << k << "," << n
           << ")";
    }
    std::cout << line.str() << '\n';
  }
  return marker_range_valid && bijective && frozen_mapping_valid;
}

void test_register_feed(TestContext& test) {
  constexpr std::size_t kGuardedInputElements =
      kGuardElements + kTileElements + kGuardElements;
  constexpr std::size_t kGuardedOutputElements =
      kGuardElements + kTileElements + kGuardElements;
  ManagedBuffer<std::uint16_t> a;
  ManagedBuffer<std::uint16_t> b;
  ManagedBuffer<std::uint32_t> shared_first;
  ManagedBuffer<std::uint32_t> direct_first;
  ManagedBuffer<std::uint32_t> shared_replay;
  ManagedBuffer<std::uint32_t> direct_replay;
  bool ready = test.cuda_ok(a.allocate(kGuardedInputElements), "allocate A") &&
               test.cuda_ok(b.allocate(kGuardedInputElements), "allocate B") &&
               test.cuda_ok(shared_first.allocate(kGuardedOutputElements),
                            "allocate first shared output") &&
               test.cuda_ok(direct_first.allocate(kGuardedOutputElements),
                            "allocate first direct output") &&
               test.cuda_ok(shared_replay.allocate(kGuardedOutputElements),
                            "allocate replay shared output") &&
               test.cuda_ok(direct_replay.allocate(kGuardedOutputElements),
                            "allocate replay direct output");
  if (!ready) {
    return;
  }

  fill_buffer(a, kInputGuard);
  fill_buffer(b, kInputGuard);
  fill_buffer(shared_first, kOutputGuard);
  fill_buffer(direct_first, kOutputGuard);
  fill_buffer(shared_replay, kOutputGuard);
  fill_buffer(direct_replay, kOutputGuard);
  for (unsigned int linear = 0U; linear < kTileElements; ++linear) {
    a[kGuardElements + linear] = make_a_value(linear);
    b[kGuardElements + linear] = make_b_value(linear);
  }
  const auto launch = [&](ManagedBuffer<std::uint32_t>& shared_output,
                          ManagedBuffer<std::uint32_t>& direct_output,
                          const std::string& label) {
    q3x_sm87_bf16_matrix_b_register_feed_kernel<<<1U, kWarpSize>>>(
        a.data() + kGuardElements, b.data() + kGuardElements,
        shared_output.data() + kGuardElements,
        direct_output.data() + kGuardElements);
    ready = test.cuda_ok(cudaGetLastError(), label + " launch") && ready;
    ready = test.cuda_ok(cudaDeviceSynchronize(), label + " synchronize") &&
            ready;
  };
  launch(shared_first, direct_first, "first register-feed comparison");
  launch(shared_replay, direct_replay, "replay register-feed comparison");
  if (!ready) {
    return;
  }

  const auto payload_equal = [](const auto& lhs, const auto& rhs) {
    return std::equal(lhs.data() + kGuardElements,
                      lhs.data() + kGuardElements + kTileElements,
                      rhs.data() + kGuardElements);
  };
  test.expect(payload_equal(shared_first, direct_first),
              "direct fragment.x feed matches shared-load WMMA bitwise");
  test.expect(payload_equal(shared_first, shared_replay),
              "shared-load WMMA output is replay-bitwise");
  test.expect(payload_equal(direct_first, direct_replay),
              "direct fragment.x WMMA output is replay-bitwise");
  test.expect(payload_equal(shared_replay, direct_replay),
              "replay direct feed matches replay shared load bitwise");

  expect_guards(test, a, kTileElements, kInputGuard, "A input");
  expect_guards(test, b, kTileElements, kInputGuard, "B input");
  bool inputs_immutable = true;
  for (unsigned int linear = 0U; linear < kTileElements; ++linear) {
    inputs_immutable =
        inputs_immutable &&
        a[kGuardElements + linear] == make_a_value(linear) &&
        b[kGuardElements + linear] == make_b_value(linear);
  }
  test.expect(inputs_immutable, "A/B payloads remain bitwise immutable");
  expect_guards(test, shared_first, kTileElements, kOutputGuard,
                "first shared output");
  expect_guards(test, direct_first, kTileElements, kOutputGuard,
                "first direct output");
  expect_guards(test, shared_replay, kTileElements, kOutputGuard,
                "replay shared output");
  expect_guards(test, direct_replay, kTileElements, kOutputGuard,
                "replay direct output");

  bool finite_outputs = true;
  bool outputs_written = true;
  for (unsigned int linear = 0U; linear < kTileElements; ++linear) {
    const std::uint32_t bits = shared_first[kGuardElements + linear];
    finite_outputs = finite_outputs && (bits & 0x7f800000U) != 0x7f800000U;
    outputs_written = outputs_written && bits != kOutputGuard;
  }
  test.expect(finite_outputs, "WMMA comparison outputs are finite");
  test.expect(outputs_written, "WMMA comparison writes every output element");
  std::cout << "BF16_B_REGISTER_FEED_RESULT: mapping=bijective"
            << " shared_vs_direct=bitwise replay=bitwise guards=intact"
            << " inputs=immutable outputs=written finite=true\n";
}

void test_inline_m16n8(TestContext& test) {
  constexpr std::size_t kGuardedInputElements =
      kGuardElements + kTileElements + kGuardElements;
  constexpr std::size_t kGuardedOutputElements =
      kGuardElements + kTileElements + kGuardElements;
  ManagedBuffer<std::uint16_t> a;
  ManagedBuffer<std::uint16_t> b;
  ManagedBuffer<std::uint32_t> wmma_first;
  ManagedBuffer<std::uint32_t> inline_first;
  ManagedBuffer<std::uint32_t> wmma_replay;
  ManagedBuffer<std::uint32_t> inline_replay;
  bool ready = test.cuda_ok(a.allocate(kGuardedInputElements),
                            "allocate inline sentinel A") &&
               test.cuda_ok(b.allocate(kGuardedInputElements),
                            "allocate inline sentinel B") &&
               test.cuda_ok(wmma_first.allocate(kGuardedOutputElements),
                            "allocate first WMMA sentinel output") &&
               test.cuda_ok(inline_first.allocate(kGuardedOutputElements),
                            "allocate first inline sentinel output") &&
               test.cuda_ok(wmma_replay.allocate(kGuardedOutputElements),
                            "allocate replay WMMA sentinel output") &&
               test.cuda_ok(inline_replay.allocate(kGuardedOutputElements),
                            "allocate replay inline sentinel output");
  if (!ready) {
    return;
  }

  fill_buffer(a, kInputGuard);
  fill_buffer(b, kInputGuard);
  fill_buffer(wmma_first, kOutputGuard);
  fill_buffer(inline_first, kOutputGuard);
  fill_buffer(wmma_replay, kOutputGuard);
  fill_buffer(inline_replay, kOutputGuard);
  for (unsigned int linear = 0U; linear < kTileElements; ++linear) {
    a[kGuardElements + linear] = make_a_value(linear);
    b[kGuardElements + linear] = make_b_value(linear);
  }

  const auto launch = [&](ManagedBuffer<std::uint32_t>& wmma_result,
                          ManagedBuffer<std::uint32_t>& inline_result,
                          const std::string& label) {
    q3x_sm87_bf16_mma_m16n8k16_sentinel_kernel<<<1U, kWarpSize>>>(
        a.data() + kGuardElements, b.data() + kGuardElements,
        wmma_result.data() + kGuardElements,
        inline_result.data() + kGuardElements);
    ready = test.cuda_ok(cudaGetLastError(), label + " launch") && ready;
    ready = test.cuda_ok(cudaDeviceSynchronize(), label + " synchronize") &&
            ready;
  };
  launch(wmma_first, inline_first, "first inline m16n8 comparison");
  launch(wmma_replay, inline_replay, "replay inline m16n8 comparison");
  if (!ready) {
    return;
  }

  const auto payload_equal = [](const auto& lhs, const auto& rhs) {
    return std::equal(lhs.data() + kGuardElements,
                      lhs.data() + kGuardElements + kTileElements,
                      rhs.data() + kGuardElements);
  };
  test.expect(payload_equal(wmma_first, inline_first),
              "two inline m16n8 instructions match one WMMA m16n16 bitwise");
  test.expect(payload_equal(wmma_first, wmma_replay),
              "inline sentinel WMMA reference is replay-bitwise");
  test.expect(payload_equal(inline_first, inline_replay),
              "inline m16n8 output is replay-bitwise");
  test.expect(payload_equal(wmma_replay, inline_replay),
              "replay inline m16n8 matches replay WMMA bitwise");

  expect_guards(test, a, kTileElements, kInputGuard, "inline sentinel A");
  expect_guards(test, b, kTileElements, kInputGuard, "inline sentinel B");
  expect_guards(test, wmma_first, kTileElements, kOutputGuard,
                "first WMMA sentinel output");
  expect_guards(test, inline_first, kTileElements, kOutputGuard,
                "first inline sentinel output");
  expect_guards(test, wmma_replay, kTileElements, kOutputGuard,
                "replay WMMA sentinel output");
  expect_guards(test, inline_replay, kTileElements, kOutputGuard,
                "replay inline sentinel output");
  bool inputs_immutable = true;
  bool outputs_written = true;
  bool finite_outputs = true;
  for (unsigned int linear = 0U; linear < kTileElements; ++linear) {
    inputs_immutable =
        inputs_immutable &&
        a[kGuardElements + linear] == make_a_value(linear) &&
        b[kGuardElements + linear] == make_b_value(linear);
    const std::uint32_t bits = inline_first[kGuardElements + linear];
    outputs_written = outputs_written && bits != kOutputGuard;
    finite_outputs = finite_outputs && (bits & 0x7f800000U) != 0x7f800000U;
  }
  test.expect(inputs_immutable,
              "inline sentinel A/B payloads remain bitwise immutable");
  test.expect(outputs_written,
              "inline m16n8 sentinel writes every output element");
  test.expect(finite_outputs, "inline m16n8 sentinel outputs are finite");
  std::cout << "BF16_INLINE_M16N8_RESULT: wmma_m16n16_vs_two_m16n8=bitwise"
            << " replay=bitwise guards=intact inputs=immutable"
            << " outputs=written finite=true\n";
}

}  // namespace

int main() {
  TestContext test;
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status != cudaSuccess || device_count == 0) {
    std::cout << "SKIP: BF16 matrix_b register-feed sentinel requires a CUDA "
                 "device\n";
    return 77;
  }
  cudaDeviceProp properties{};
  if (!test.cuda_ok(cudaGetDeviceProperties(&properties, 0),
                    "query CUDA device")) {
    return 1;
  }
  if (properties.major != 8 || properties.minor != 7) {
    std::cout << "SKIP: BF16 matrix_b register-feed sentinel requires SM87; "
                 "got sm_"
              << properties.major << properties.minor << '\n';
    return 77;
  }
  std::cout << "BF16_B_FRAGMENT_DEVICE: name=" << properties.name
            << " sm=" << properties.major << properties.minor
            << " warp_size=" << properties.warpSize << '\n';
  test.expect(properties.warpSize == static_cast<int>(kWarpSize),
              "SM87 reports a 32-thread warp");

  (void)query_and_print_resources(test);
  ManagedBuffer<std::uint16_t> probe;
  if (!test.cuda_ok(
          probe.allocate(kGuardElements + kProbeElements + kGuardElements),
          "allocate fragment probe output")) {
    return 1;
  }
  std::array<unsigned int, kProbeElements> mapping{};
  if (collect_and_validate_mapping(test, probe, &mapping)) {
    test_register_feed(test);
    test_inline_m16n8(test);
  }

  if (test.failures() != 0) {
    std::cerr << test.failures()
              << " BF16 matrix_b register-feed sentinel assertion(s) failed\n";
    return 1;
  }
  std::cout << "BF16 matrix_b register-feed SM87 sentinel passed\n";
  return 0;
}
