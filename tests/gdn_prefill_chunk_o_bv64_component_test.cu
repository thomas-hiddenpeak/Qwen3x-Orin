#include "gdn_prefill_chunk_o_bv64_sm87.h"
#include "gdn_prefill_chunk64_native_sm87.h"

#include "q3x/runtime/decode_ops.h"
#include "q3x/runtime/gdn_decode.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string_view>

namespace {

namespace chunk_o = q3x::runtime::gdn_prefill_chunk_o_bv64_detail;
namespace native = q3x::runtime::gdn_prefill_chunk64_native_detail;
namespace runtime = q3x::runtime;

constexpr std::size_t kTokens = 64U;
constexpr std::size_t kQkHeads = runtime::kGdnQkHeadCount;
constexpr std::size_t kValueHeads = runtime::kGdnValueHeadCount;
constexpr std::size_t kDimension = runtime::kGdnHeadDimension;
constexpr std::size_t kCompactElements =
    kQkHeads * kTokens * kDimension;
constexpr std::size_t kBoundaryStateElements =
    kValueHeads * kDimension * kDimension;
constexpr std::size_t kValueElements =
    kValueHeads * kTokens * kDimension;
constexpr std::size_t kGateElements = kValueHeads * kTokens;
constexpr std::size_t kOutputElements =
    kTokens * kValueHeads * kDimension;
constexpr std::size_t kFragmentMatrixElements = 16U * 16U;
constexpr std::size_t kFragmentARegisters = 32U * 4U;
constexpr std::size_t kFragmentBRegisters = 32U * 2U;
constexpr std::size_t kFragmentAccumulatorRegisters = 32U * 4U;
constexpr std::uint16_t kZero = 0x0000U;
constexpr std::uint16_t kOne = 0x3f80U;
constexpr std::uint16_t kPoison = 0x7fc1U;
constexpr std::uint16_t kSmall = 0x3c80U;
constexpr float kEpsilon = 1.0e-6F;

static_assert(kQkHeads == 16U);
static_assert(kValueHeads == 48U);
static_assert(kDimension == 128U);

template <typename T>
class ManagedBuffer final {
 public:
  explicit ManagedBuffer(const std::size_t elements) noexcept
      : elements_(elements) {
    status_ = cudaMallocManaged(reinterpret_cast<void**>(&data_),
                                elements_ * sizeof(T));
  }

  ~ManagedBuffer() {
    if (data_ != nullptr) {
      (void)cudaFree(data_);
    }
  }

  ManagedBuffer(const ManagedBuffer&) = delete;
  ManagedBuffer& operator=(const ManagedBuffer&) = delete;

  [[nodiscard]] bool valid() const noexcept {
    return status_ == cudaSuccess && data_ != nullptr;
  }
  [[nodiscard]] T* data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return elements_; }

 private:
  T* data_ = nullptr;
  std::size_t elements_ = 0U;
  cudaError_t status_ = cudaErrorMemoryAllocation;
};

class TestContext final {
 public:
  void expect(const bool condition, const std::string_view message) {
    if (!condition) {
      ++failures_;
      std::cerr << "FAIL: " << message << '\n';
    }
  }

  [[nodiscard]] bool cuda_ok(const int status,
                             const std::string_view operation) {
    if (status == static_cast<int>(cudaSuccess)) {
      return true;
    }
    ++failures_;
    std::cerr << "CUDA FAIL: " << operation << " status=" << status
              << " message="
              << cudaGetErrorString(static_cast<cudaError_t>(status)) << '\n';
    return false;
  }

  [[nodiscard]] int result() const noexcept {
    return failures_ == 0U ? 0 : 1;
  }

 private:
  std::size_t failures_ = 0U;
};

struct Buffers final {
  ManagedBuffer<std::uint16_t> q{kCompactElements};
  ManagedBuffer<std::uint16_t> k{kCompactElements};
  ManagedBuffer<std::uint16_t> h{kBoundaryStateElements};
  ManagedBuffer<std::uint16_t> v{kValueElements};
  ManagedBuffer<float> gamma{kGateElements};
  ManagedBuffer<std::uint16_t> weight{kDimension};
  ManagedBuffer<std::uint16_t> gate{kOutputElements};
  ManagedBuffer<std::uint16_t> raw{kOutputElements};
  ManagedBuffer<std::uint16_t> output{kOutputElements};
  ManagedBuffer<std::uint16_t> reference{kOutputElements};
  ManagedBuffer<std::uint16_t> w{kValueElements};
  ManagedBuffer<std::uint16_t> u{kValueElements};
  ManagedBuffer<std::uint16_t> initial_state{kBoundaryStateElements};
  ManagedBuffer<std::uint16_t> baseline_state{kBoundaryStateElements};
  ManagedBuffer<std::uint16_t> candidate_state{kBoundaryStateElements};
  ManagedBuffer<std::uint16_t> candidate_raw{kOutputElements};
  ManagedBuffer<std::uint16_t> candidate_output{kOutputElements};
  ManagedBuffer<std::uint16_t> fragment_a{kFragmentMatrixElements};
  ManagedBuffer<std::uint16_t> fragment_b{kFragmentMatrixElements};
  ManagedBuffer<std::uint32_t> loaded_a{kFragmentARegisters};
  ManagedBuffer<std::uint32_t> direct_a{kFragmentARegisters};
  ManagedBuffer<std::uint32_t> loaded_b{kFragmentBRegisters};
  ManagedBuffer<std::uint32_t> direct_b{kFragmentBRegisters};
  ManagedBuffer<float> loaded_accumulator{kFragmentAccumulatorRegisters};
  ManagedBuffer<float> direct_accumulator{kFragmentAccumulatorRegisters};

  [[nodiscard]] bool valid() const noexcept {
    return q.valid() && k.valid() && h.valid() && v.valid() &&
           gamma.valid() && weight.valid() && gate.valid() && raw.valid() &&
           output.valid() && reference.valid() && w.valid() && u.valid() &&
           initial_state.valid() && baseline_state.valid() &&
           candidate_state.valid() && candidate_raw.valid() &&
           candidate_output.valid() && fragment_a.valid() &&
           fragment_b.valid() && loaded_a.valid() && direct_a.valid() &&
           loaded_b.valid() && direct_b.valid() &&
           loaded_accumulator.valid() && direct_accumulator.valid();
  }
};

void reset(Buffers& buffers) {
  std::fill_n(buffers.q.data(), buffers.q.size(), kZero);
  std::fill_n(buffers.k.data(), buffers.k.size(), kZero);
  std::fill_n(buffers.h.data(), buffers.h.size(), kZero);
  std::fill_n(buffers.v.data(), buffers.v.size(), kZero);
  std::fill_n(buffers.gamma.data(), buffers.gamma.size(), 0.0F);
  std::fill_n(buffers.weight.data(), buffers.weight.size(), kOne);
  std::fill_n(buffers.gate.data(), buffers.gate.size(), kOne);
  std::fill_n(buffers.raw.data(), buffers.raw.size(), kPoison);
  std::fill_n(buffers.output.data(), buffers.output.size(), kPoison);
  std::fill_n(buffers.reference.data(), buffers.reference.size(), kPoison);
  std::fill_n(buffers.w.data(), buffers.w.size(), kZero);
  std::fill_n(buffers.u.data(), buffers.u.size(), kZero);
  std::fill_n(buffers.initial_state.data(), buffers.initial_state.size(),
              kZero);
  std::fill_n(buffers.baseline_state.data(), buffers.baseline_state.size(),
              kPoison);
  std::fill_n(buffers.candidate_state.data(), buffers.candidate_state.size(),
              kPoison);
  std::fill_n(buffers.candidate_raw.data(), buffers.candidate_raw.size(),
              kPoison);
  std::fill_n(buffers.candidate_output.data(),
              buffers.candidate_output.size(), kPoison);
}

[[nodiscard]] std::size_t compact_index(const std::size_t qk_head,
                                        const std::size_t token,
                                        const std::size_t key) noexcept {
  return (qk_head * kTokens + token) * kDimension + key;
}

[[nodiscard]] std::size_t state_index(const std::size_t value_head,
                                      const std::size_t value,
                                      const std::size_t key) noexcept {
  return (value_head * kDimension + value) * kDimension + key;
}

[[nodiscard]] std::size_t value_index(const std::size_t value_head,
                                      const std::size_t token,
                                      const std::size_t value) noexcept {
  return (value_head * kTokens + token) * kDimension + value;
}

[[nodiscard]] std::size_t output_index(const std::size_t token,
                                       const std::size_t value_head,
                                       const std::size_t value) noexcept {
  return (token * kValueHeads + value_head) * kDimension + value;
}

[[nodiscard]] bool expect_single_raw(TestContext& test,
                                     const std::string_view label,
                                     const std::uint16_t* const raw,
                                     const std::size_t expected_index) {
  std::size_t unequal = 0U;
  std::size_t first = std::numeric_limits<std::size_t>::max();
  std::uint16_t first_expected = 0U;
  std::uint16_t first_actual = 0U;
  for (std::size_t index = 0U; index < kOutputElements; ++index) {
    const std::uint16_t expected = index == expected_index ? kOne : kZero;
    if (raw[index] != expected) {
      if (first == std::numeric_limits<std::size_t>::max()) {
        first = index;
        first_expected = expected;
        first_actual = raw[index];
      }
      ++unequal;
    }
  }
  std::cout << label << " unequal=" << unequal
            << " expected_single_index=" << expected_index
            << " first_unequal="
            << (first == std::numeric_limits<std::size_t>::max()
                    ? kOutputElements
                    : first)
            << " first_expected=0x" << std::hex << first_expected
            << " first_actual=0x" << first_actual << std::dec
            << " gate=" << (unequal == 0U ? "PASS" : "FAIL") << '\n';
  test.expect(unequal == 0U, label);
  return unequal == 0U;
}

[[nodiscard]] bool expect_equal(TestContext& test,
                                const std::string_view label,
                                const std::uint16_t* const expected,
                                const std::uint16_t* const actual,
                                const std::size_t elements) {
  std::size_t unequal = 0U;
  std::size_t first = std::numeric_limits<std::size_t>::max();
  for (std::size_t index = 0U; index < elements; ++index) {
    if (expected[index] != actual[index]) {
      if (first == std::numeric_limits<std::size_t>::max()) {
        first = index;
      }
      ++unequal;
    }
  }
  std::cout << label << " unequal=" << unequal
            << " first_unequal="
            << (first == std::numeric_limits<std::size_t>::max()
                    ? elements
                    : first);
  if (first != std::numeric_limits<std::size_t>::max()) {
    std::cout << " first_expected=0x" << std::hex << expected[first]
              << " first_actual=0x" << actual[first] << std::dec;
  }
  std::cout << " gate=" << (unequal == 0U ? "PASS" : "FAIL") << '\n';
  test.expect(unequal == 0U, label);
  return unequal == 0U;
}

[[nodiscard]] bool launch_component(Buffers& buffers) {
  return chunk_o::launch(
             buffers.q.data(), buffers.k.data(), buffers.h.data(),
             buffers.v.data(), buffers.gamma.data(), kTokens,
             buffers.weight.data(), buffers.gate.data(), kEpsilon,
             buffers.raw.data(), buffers.output.data()) ==
             static_cast<int>(cudaSuccess) &&
         cudaDeviceSynchronize() == cudaSuccess;
}

void check_component_epilogue(TestContext& test, Buffers& buffers,
                              const std::string_view label) {
  const int status =
      runtime::launch_headwise_plain_rms_norm_silu_gate_reference_cuda(
          buffers.raw.data(), buffers.weight.data(), buffers.gate.data(),
          kTokens * kValueHeads, kDimension, kEpsilon,
          buffers.reference.data());
  const bool ready = test.cuda_ok(status, "launch epilogue reference") &&
                     test.cuda_ok(static_cast<int>(cudaDeviceSynchronize()),
                                  "synchronize epilogue reference");
  if (ready) {
    (void)expect_equal(test, label, buffers.reference.data(),
                       buffers.output.data(), kOutputElements);
  }
}

void test_fragment_lane_mapping(TestContext& test, Buffers& buffers) {
  for (std::size_t index = 0U; index < kFragmentMatrixElements; ++index) {
    buffers.fragment_a.data()[index] = static_cast<std::uint16_t>(
        0x3f00U + (index & 0x007fU));
    buffers.fragment_b.data()[index] = static_cast<std::uint16_t>(
        0x3e80U + ((5U * index) & 0x007fU));
  }
  const int status = chunk_o::launch_fragment_sentinel(
      buffers.fragment_a.data(), buffers.fragment_b.data(),
      buffers.loaded_a.data(), buffers.direct_a.data(),
      buffers.loaded_b.data(), buffers.direct_b.data(),
      buffers.loaded_accumulator.data(), buffers.direct_accumulator.data());
  const bool ready =
      test.cuda_ok(status, "launch fragment lane sentinel") &&
      test.cuda_ok(static_cast<int>(cudaDeviceSynchronize()),
                   "synchronize fragment lane sentinel");
  if (!ready) {
    return;
  }

  const auto check_u32 = [&test](const std::string_view label,
                                 const std::uint32_t* const direct,
                                 const std::uint32_t* const loaded,
                                 const std::size_t elements,
                                 const std::size_t registers_per_lane) {
    std::size_t unequal = 0U;
    std::size_t first = elements;
    for (std::size_t index = 0U; index < elements; ++index) {
      if (direct[index] != loaded[index]) {
        if (first == elements) {
          first = index;
        }
        ++unequal;
      }
    }
    std::cout << label << " unequal=" << unequal
              << " first_lane="
              << (first == elements ? 32U : first / registers_per_lane)
              << " first_register="
              << (first == elements ? registers_per_lane
                                    : first % registers_per_lane);
    if (first != elements) {
      std::cout << " direct=0x" << std::hex << direct[first]
                << " loaded=0x" << loaded[first] << std::dec;
    }
    std::cout << " gate=" << (unequal == 0U ? "PASS" : "FAIL") << '\n';
    test.expect(unequal == 0U, label);
  };

  check_u32("GDN_CHUNK_O_BV64_FRAGMENT_A", buffers.direct_a.data(),
            buffers.loaded_a.data(), kFragmentARegisters, 4U);
  check_u32("GDN_CHUNK_O_BV64_FRAGMENT_B", buffers.direct_b.data(),
            buffers.loaded_b.data(), kFragmentBRegisters, 2U);

  std::size_t unequal = 0U;
  std::size_t first = kFragmentAccumulatorRegisters;
  for (std::size_t index = 0U; index < kFragmentAccumulatorRegisters;
       ++index) {
    if (buffers.direct_accumulator.data()[index] !=
        buffers.loaded_accumulator.data()[index]) {
      if (first == kFragmentAccumulatorRegisters) {
        first = index;
      }
      ++unequal;
    }
  }
  std::cout << "GDN_CHUNK_O_BV64_FRAGMENT_MMA unequal=" << unequal
            << " first_lane="
            << (first == kFragmentAccumulatorRegisters ? 32U : first / 4U)
            << " first_register="
            << (first == kFragmentAccumulatorRegisters ? 4U : first % 4U);
  if (first != kFragmentAccumulatorRegisters) {
    std::cout << " direct=" << buffers.direct_accumulator.data()[first]
              << " loaded=" << buffers.loaded_accumulator.data()[first];
  }
  std::cout << " gate=" << (unequal == 0U ? "PASS" : "FAIL") << '\n';
  test.expect(unequal == 0U, "GDN_CHUNK_O_BV64_FRAGMENT_MMA");
}

void test_state_only(TestContext& test, Buffers& buffers) {
  reset(buffers);
  constexpr std::size_t kQkHead = 0U;
  constexpr std::size_t kToken = 5U;
  constexpr std::size_t kKey = 7U;
  constexpr std::size_t kValueHead = 0U;
  constexpr std::size_t kValue = 9U;
  buffers.q.data()[compact_index(kQkHead, kToken, kKey)] = kOne;
  buffers.h.data()[state_index(kValueHead, kValue, kKey)] = kOne;
  const bool ready = launch_component(buffers);
  test.expect(ready, "state-only component launch");
  if (!ready) {
    return;
  }
  (void)expect_single_raw(
      test, "GDN_CHUNK_O_BV64_KN_TRANSPOSED_QH_RAW", buffers.raw.data(),
      output_index(kToken, kValueHead, kValue));
  check_component_epilogue(test, buffers,
                           "GDN_CHUNK_O_BV64_KN_TRANSPOSED_QH_EPILOGUE");
}

void test_qk_v_only(TestContext& test, Buffers& buffers) {
  reset(buffers);
  constexpr std::size_t kQkHead = 0U;
  constexpr std::size_t kQuery = 11U;
  constexpr std::size_t kSource = 2U;
  constexpr std::size_t kKey = 17U;
  constexpr std::size_t kValueHead = 2U;
  constexpr std::size_t kValue = 70U;
  static_assert(kSource <= kQuery);
  buffers.q.data()[compact_index(kQkHead, kQuery, kKey)] = kOne;
  buffers.k.data()[compact_index(kQkHead, kSource, kKey)] = kOne;
  buffers.v.data()[value_index(kValueHead, kSource, kValue)] = kOne;
  const bool ready = launch_component(buffers);
  test.expect(ready, "QK-V-only component launch");
  if (!ready) {
    return;
  }
  (void)expect_single_raw(
      test, "GDN_CHUNK_O_BV64_KN_TRANSPOSED_QK_V_RAW", buffers.raw.data(),
      output_index(kQuery, kValueHead, kValue));
  check_component_epilogue(test, buffers,
                           "GDN_CHUNK_O_BV64_KN_TRANSPOSED_QK_V_EPILOGUE");
}

void test_rows8_norm(TestContext& test, Buffers& buffers) {
  reset(buffers);
  constexpr std::size_t kRows = 8U;
  constexpr std::size_t kElements = kRows * kDimension;
  for (std::size_t row = 0U; row < kRows; ++row) {
    buffers.raw.data()[row * kDimension + (3U * row + 1U)] = kOne;
  }
  const int candidate_status = chunk_o::launch_norm_rows8(
      buffers.raw.data(), buffers.weight.data(), buffers.gate.data(), kRows,
      kEpsilon, buffers.output.data());
  const int reference_status =
      runtime::launch_headwise_plain_rms_norm_silu_gate_reference_cuda(
          buffers.raw.data(), buffers.weight.data(), buffers.gate.data(),
          kRows, kDimension, kEpsilon, buffers.reference.data());
  const bool ready = test.cuda_ok(candidate_status, "launch rows8 norm") &&
                     test.cuda_ok(reference_status,
                                  "launch rows8 norm reference") &&
                     test.cuda_ok(static_cast<int>(cudaDeviceSynchronize()),
                                  "synchronize rows8 norm");
  if (ready) {
    (void)expect_equal(test, "GDN_CHUNK_O_BV64_ROWS8_NORM",
                       buffers.reference.data(), buffers.output.data(),
                       kElements);
  }
}

void test_prompt_span_vertical_slice_c64(TestContext& test,
                                         Buffers& buffers) {
  reset(buffers);
  for (std::size_t index = 0U; index < buffers.initial_state.size();
       ++index) {
    if (index % 4'093U == 17U || index % 8'191U == 31U) {
      buffers.initial_state.data()[index] = kSmall;
    }
  }
  for (std::size_t index = 0U; index < buffers.w.size(); ++index) {
    if (index % 1'021U == 7U) {
      buffers.w.data()[index] = kSmall;
    }
    if (index % 769U == 11U) {
      buffers.u.data()[index] = kSmall;
    }
  }
  for (std::size_t index = 0U; index < buffers.q.size(); ++index) {
    if (index % 521U == 5U) {
      buffers.q.data()[index] = kSmall;
    }
    if (index % 509U == 3U) {
      buffers.k.data()[index] = kSmall;
    }
  }
  for (std::size_t value_head = 0U; value_head < kValueHeads;
       ++value_head) {
    for (std::size_t token = 0U; token < kTokens; ++token) {
      buffers.gamma.data()[value_head * kTokens + token] =
          -0.0078125F * static_cast<float>(token + 1U);
    }
  }

  int registers = 0;
  std::size_t static_shared = 0U;
  std::size_t local_bytes = 0U;
  int maximum_threads = 0;
  int active_blocks = 0;
  const int resource_status = native::query_prompt_span_state_o_resources(
      &registers, &static_shared, &local_bytes, &maximum_threads,
      &active_blocks);
  test.expect(test.cuda_ok(resource_status,
                           "query prompt-span state-o resources") &&
                  registers <= 255 && local_bytes == 0U &&
                  maximum_threads >= 128 && active_blocks >= 2,
              "prompt-span state-o retains two CTA/SM resource contract");
  std::cout << "GDN_PROMPT_SPAN_STATE_O_RESOURCES regs=" << registers
            << " static_shared=" << static_shared
            << " local_bytes=" << local_bytes
            << " max_threads=" << maximum_threads
            << " active_blocks_per_sm=" << active_blocks << '\n';

  const int baseline_status =
      native::launch_prompt_span_state_o_baseline_for_test(
          buffers.w.data(), buffers.u.data(), buffers.q.data(),
          buffers.k.data(), buffers.gamma.data(),
          buffers.initial_state.data(), buffers.baseline_state.data(),
          kTokens, buffers.v.data(), buffers.h.data(),
          buffers.weight.data(), buffers.gate.data(), kEpsilon,
          buffers.raw.data(), buffers.output.data());
  const int candidate_status = native::launch_prompt_span_state_o(
      buffers.w.data(), buffers.u.data(), buffers.q.data(),
      buffers.k.data(), buffers.gamma.data(), buffers.initial_state.data(),
      buffers.candidate_state.data(), kTokens, buffers.weight.data(),
      buffers.gate.data(), kEpsilon, buffers.candidate_raw.data(),
      buffers.candidate_output.data());
  const bool ready =
      test.cuda_ok(baseline_status, "launch prompt-span baseline C64") &&
      test.cuda_ok(candidate_status, "launch prompt-span candidate C64") &&
      test.cuda_ok(static_cast<int>(cudaDeviceSynchronize()),
                   "synchronize prompt-span C64");
  if (!ready) {
    return;
  }
  const std::size_t nonzero_raw = static_cast<std::size_t>(std::count_if(
      buffers.raw.data(), buffers.raw.data() + kOutputElements,
      [](const std::uint16_t value) { return value != kZero; }));
  std::size_t changed_state = 0U;
  for (std::size_t index = 0U; index < kBoundaryStateElements; ++index) {
    changed_state += buffers.baseline_state.data()[index] !=
                             buffers.initial_state.data()[index]
                         ? 1U
                         : 0U;
  }
  std::cout << "GDN_PROMPT_SPAN_NONEMPTY raw_nonzero=" << nonzero_raw
            << " changed_state=" << changed_state
            << " gate="
            << (nonzero_raw != 0U && changed_state != 0U ? "PASS"
                                                         : "FAIL")
            << '\n';
  test.expect(nonzero_raw != 0U && changed_state != 0U,
              "prompt-span fixture exercises nonzero output and state update");
  (void)expect_equal(test, "GDN_PROMPT_SPAN_STATE_C64",
                     buffers.baseline_state.data(),
                     buffers.candidate_state.data(), kBoundaryStateElements);
  (void)expect_equal(test, "GDN_PROMPT_SPAN_RAW_O_C64", buffers.raw.data(),
                     buffers.candidate_raw.data(), kOutputElements);
  (void)expect_equal(test, "GDN_PROMPT_SPAN_NORM_O_C64",
                     buffers.output.data(), buffers.candidate_output.data(),
                     kOutputElements);
}

}  // namespace

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::cout << "SKIP: GDN chunk-o BV64 component basis requires CUDA\n";
    return 77;
  }
  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) {
    return 1;
  }
  if (properties.major != 8 || properties.minor != 7) {
    std::cout << "SKIP: GDN chunk-o BV64 component basis requires SM87\n";
    return 77;
  }

  TestContext test;
  Buffers buffers;
  test.expect(buffers.valid(), "allocate component basis buffers");
  if (!buffers.valid()) {
    return test.result();
  }
  test_fragment_lane_mapping(test, buffers);
  test_state_only(test, buffers);
  test_qk_v_only(test, buffers);
  test_rows8_norm(test, buffers);
  test_prompt_span_vertical_slice_c64(test, buffers);
  std::cout << "GDN_CHUNK_O_BV64_COMPONENT_RESULT gate="
            << (test.result() == 0 ? "PASS" : "FAIL")
            << " authority=SYNTHETIC_CORRECTNESS_ONLY\n";
  return test.result();
}
