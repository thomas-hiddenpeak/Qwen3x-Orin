#include "gdn_prefill_chunk_o_bv64_sm87.h"
#include "gdn_prefill_chunk64_native_sm87.h"

#include "q3x/kernels/sm87_a4w4_attention_k256_m128n256.h"
#include "q3x/kernels/sm87_a4w4_factorized_lane_quantize.h"
#include "q3x/runtime/decode_ops.h"
#include "q3x/runtime/gdn_decode.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>

namespace {

namespace chunk_o = q3x::runtime::gdn_prefill_chunk_o_bv64_detail;
namespace kernels = q3x::kernels;
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
constexpr std::size_t kK256InputSize = kValueHeads * kDimension;
constexpr std::size_t kK256Groups = kK256InputSize / 256U;
constexpr std::size_t kK256PhysicalGroups = kK256InputSize / 64U;
constexpr std::size_t kPackedGuardBytes = 64U;
constexpr std::size_t kScaleGuardElements = 16U;
constexpr std::uint8_t kPackedSentinel = 0xa5U;
constexpr std::uint16_t kScaleSentinel = 0x5aa5U;
constexpr float kK256ClipRatio = 0.875F;
constexpr float kFactorizedLaneR1ClipRatio = 0.8125F;
constexpr std::size_t kFactorizedLaneR1ScaleGroups = 1U;
constexpr std::size_t kFactorizedLaneR1SharedBytes = 12'528U;

static_assert(kQkHeads == 16U);
static_assert(kValueHeads == 48U);
static_assert(kDimension == 128U);
static_assert(kK256InputSize == 6'144U);
static_assert(kK256Groups == 24U);
static_assert(kK256PhysicalGroups == 96U);
static_assert(kFactorizedLaneR1ScaleGroups == 1U);

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

class NonBlockingStream final {
 public:
  NonBlockingStream() noexcept {
    status_ = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
  }

  ~NonBlockingStream() {
    if (stream_ != nullptr) {
      (void)cudaStreamDestroy(stream_);
    }
  }

  NonBlockingStream(const NonBlockingStream&) = delete;
  NonBlockingStream& operator=(const NonBlockingStream&) = delete;

  [[nodiscard]] bool valid() const noexcept {
    return status_ == cudaSuccess && stream_ != nullptr;
  }
  [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
  cudaError_t status_ = cudaErrorInitializationError;
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

[[nodiscard]] bool expect_equal_bytes(TestContext& test,
                                      const std::string_view label,
                                      const std::uint8_t* const expected,
                                      const std::uint8_t* const actual,
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
    std::cout << " first_expected=0x" << std::hex
              << static_cast<unsigned int>(expected[first])
              << " first_actual=0x"
              << static_cast<unsigned int>(actual[first]) << std::dec;
  }
  std::cout << " gate=" << (unequal == 0U ? "PASS" : "FAIL") << '\n';
  test.expect(unequal == 0U, label);
  return unequal == 0U;
}

template <typename T>
[[nodiscard]] bool guards_are_sentinel(
    const T* const allocation, const std::size_t prefix_elements,
    const std::size_t payload_elements, const std::size_t suffix_elements,
    const T sentinel) {
  return std::all_of(allocation, allocation + prefix_elements,
                     [sentinel](const T value) { return value == sentinel; }) &&
         std::all_of(allocation + prefix_elements + payload_elements,
                     allocation + prefix_elements + payload_elements +
                         suffix_elements,
                     [sentinel](const T value) { return value == sentinel; });
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

struct K256PublishCase final {
  std::string_view name;
  std::size_t tile_logical_tokens;
  std::size_t destination_first_token;
};

void fill_k256_publish_inputs(std::uint16_t* const raw,
                              std::uint16_t* const gate,
                              std::uint16_t* const weight,
                              const std::size_t tokens) {
  constexpr std::uint16_t kRawPatterns[] = {
      0x0000U, 0x3d80U, 0xbd80U, 0x3e80U, 0xbe00U,
      0x3f00U, 0xbf40U, 0x4000U, 0xc020U};
  constexpr std::uint16_t kGatePatterns[] = {
      0x0000U, 0x3e00U, 0xbe80U, 0x3f00U,
      0xbf80U, 0x3fc0U, 0xc000U};
  constexpr std::uint16_t kWeightPatterns[] = {
      0x3f80U, 0x3f40U, 0x3fc0U, 0xbf00U, 0x3e80U};
  constexpr std::size_t kRawPatternCount =
      sizeof(kRawPatterns) / sizeof(kRawPatterns[0]);
  constexpr std::size_t kGatePatternCount =
      sizeof(kGatePatterns) / sizeof(kGatePatterns[0]);
  constexpr std::size_t kWeightPatternCount =
      sizeof(kWeightPatterns) / sizeof(kWeightPatterns[0]);

  for (std::size_t value = 0U; value < kDimension; ++value) {
    weight[value] = kWeightPatterns[(3U * value + 1U) % kWeightPatternCount];
  }
  for (std::size_t token = 0U; token < tokens; ++token) {
    for (std::size_t head = 0U; head < kValueHeads; ++head) {
      const std::size_t base =
          (token * kValueHeads + head) * kDimension;
      for (std::size_t value = 0U; value < kDimension; ++value) {
        raw[base + value] =
            kRawPatterns[(13U * token + 7U * head + 3U * value) %
                         kRawPatternCount];
        gate[base + value] =
            kGatePatterns[(5U * token + 11U * head + value) %
                          kGatePatternCount];
      }
    }
  }
}

[[nodiscard]] float decode_bf16_host(const std::uint16_t bits) {
  const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16U;
  float value = 0.0F;
  std::memcpy(&value, &word, sizeof(value));
  return value;
}

[[nodiscard]] std::uint16_t encode_quantizer_bf16_host(
    const float value) {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

void fill_factorized_lane_r1_inverse_alpha(float* const inverse_alpha) {
  constexpr float kPatterns[] = {
      0.5F, 0.75F, 1.0F, 1.25F, 1.5F, 1.75F, 2.0F};
  constexpr std::size_t kPatternCount =
      sizeof(kPatterns) / sizeof(kPatterns[0]);
  for (std::size_t k = 0U; k < kK256InputSize; ++k) {
    inverse_alpha[k] = kPatterns[(5U * k + k / 128U) % kPatternCount];
  }
}

void factorized_lane_r1_quantize_host(
    const std::uint16_t* const normalized,
    const float* const inverse_alpha,
    const std::size_t tile_logical_tokens,
    const std::size_t destination_first_token,
    const std::size_t whole_logical_tokens,
    const std::size_t launch_tokens,
    const float clip_ratio,
    std::uint8_t* const packed,
    std::uint16_t* const scales) {
  const bool final_slice =
      destination_first_token + tile_logical_tokens == whole_logical_tokens;
  const std::size_t padding_tokens =
      final_slice ? launch_tokens - whole_logical_tokens : 0U;
  const std::size_t publish_tokens =
      tile_logical_tokens + padding_tokens;
  for (std::size_t local_token = 0U; local_token < publish_tokens;
       ++local_token) {
    const std::size_t destination_token =
        destination_first_token + local_token;
    if (local_token >= tile_logical_tokens) {
      for (std::size_t pair = 0U; pair < kK256InputSize / 2U; ++pair) {
        const std::size_t global_even = pair * 2U;
        packed[kernels::sm87_a4w4_consumer_packed_offset(
            destination_token, global_even / 64U,
            (global_even % 64U) / 2U, kK256PhysicalGroups)] = 0U;
      }
      scales[kernels::sm87_a4w4_factorized_lane_scale_offset(
          destination_token, 0U, kFactorizedLaneR1ScaleGroups)] = kOne;
      continue;
    }

    const std::size_t input_base = local_token * kK256InputSize;
    float maximum = 0.0F;
    for (std::size_t k = 0U; k < kK256InputSize; ++k) {
      maximum = std::fmax(
          maximum,
          std::fabs(decode_bf16_host(normalized[input_base + k]) *
                    inverse_alpha[k]));
    }
    const float clipped_maximum = maximum * clip_ratio;
    std::uint16_t scale_bits = encode_quantizer_bf16_host(
        maximum == 0.0F ? 1.0F : clipped_maximum / 7.0F);
    float stored_scale = decode_bf16_host(scale_bits);
    if (maximum != 0.0F && stored_scale == 0.0F) {
      scale_bits = 1U;
      stored_scale = decode_bf16_host(scale_bits);
    }
    scales[kernels::sm87_a4w4_factorized_lane_scale_offset(
        destination_token, 0U, kFactorizedLaneR1ScaleGroups)] =
        scale_bits;

    for (std::size_t pair = 0U; pair < kK256InputSize / 2U; ++pair) {
      const std::size_t global_even = pair * 2U;
      float even = decode_bf16_host(
                       normalized[input_base + global_even]) *
                   inverse_alpha[global_even];
      float odd = decode_bf16_host(
                      normalized[input_base + global_even + 1U]) *
                  inverse_alpha[global_even + 1U];
      even = std::fmin(std::fmax(even, -clipped_maximum),
                       clipped_maximum);
      odd = std::fmin(std::fmax(odd, -clipped_maximum),
                      clipped_maximum);
      const long even_rounded = std::lrint(even / stored_scale);
      const long odd_rounded = std::lrint(odd / stored_scale);
      const int even_code = even_rounded < -7L
                                ? -7
                                : (even_rounded > 7L
                                       ? 7
                                       : static_cast<int>(even_rounded));
      const int odd_code = odd_rounded < -7L
                               ? -7
                               : (odd_rounded > 7L
                                      ? 7
                                      : static_cast<int>(odd_rounded));
      packed[kernels::sm87_a4w4_consumer_packed_offset(
          destination_token, global_even / 64U,
          (global_even % 64U) / 2U, kK256PhysicalGroups)] =
          kernels::sm87_a4w4_pack_signed_pair(even_code, odd_code);
    }
  }
}

[[nodiscard]] bool factorized_lane_r1_padding_is_zero_one(
    const std::uint8_t* const packed,
    const std::uint16_t* const scales,
    const std::size_t first_padding_token,
    const std::size_t launch_token_count) {
  for (std::size_t token = first_padding_token;
       token < launch_token_count; ++token) {
    if (scales[kernels::sm87_a4w4_factorized_lane_scale_offset(
            token, 0U, kFactorizedLaneR1ScaleGroups)] != kOne) {
      return false;
    }
    for (std::size_t group = 0U; group < kK256PhysicalGroups; ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        if (packed[kernels::sm87_a4w4_consumer_packed_offset(
                token, group, byte, kK256PhysicalGroups)] != 0U) {
          return false;
        }
      }
    }
  }
  return true;
}

[[nodiscard]] bool k256_padding_is_zero_one(
    const std::uint8_t* const packed,
    const std::uint16_t* const scales,
    const std::size_t first_padding_token,
    const std::size_t launch_token_count) {
  for (std::size_t token = first_padding_token;
       token < launch_token_count; ++token) {
    for (std::size_t group = 0U; group < kK256Groups; ++group) {
      if (scales[kernels::sm87_a4w4_attention_k256_scale_offset(
              token, group, kK256Groups)] != kOne) {
        return false;
      }
    }
    for (std::size_t group = 0U; group < kK256PhysicalGroups; ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        if (packed[kernels::sm87_a4w4_attention_k256_packed_offset(
                token, group, byte, kK256PhysicalGroups)] != 0U) {
          return false;
        }
      }
    }
  }
  return true;
}

void test_norm_k256_a4_resources(TestContext& test) {
  int registers = 0;
  std::size_t static_shared = 0U;
  std::size_t local_bytes = 0U;
  int maximum_threads = 0;
  int active_blocks = 0;
  const int status = chunk_o::query_norm_k256_a4_resources(
      &registers, &static_shared, &local_bytes, &maximum_threads,
      &active_blocks);
  const bool admitted =
      test.cuda_ok(status, "query direct norm K256 A4 resources") &&
      registers > 0 && registers <= 128 && static_shared == 32U &&
      local_bytes == 0U && maximum_threads >= 256 && active_blocks >= 2;
  test.expect(admitted,
              "direct norm K256 A4 retains two CTA/SM resource contract");
  std::cout << "GDN_CHUNK_O_BV64_NORM_K256_A4_RESOURCES regs=" << registers
            << " static_shared=" << static_shared
            << " local_bytes=" << local_bytes
            << " max_threads=" << maximum_threads
            << " active_blocks_per_sm=" << active_blocks
            << " gate=" << (admitted ? "PASS" : "FAIL") << '\n';
}

void test_norm_rows8_k256_a4_case(TestContext& test,
                                  const K256PublishCase& spec) {
  constexpr std::size_t kWholeLogicalTokens = 1'853U;
  constexpr std::size_t kWholeLaunchTokens = 1'920U;
  static_assert(
      kernels::sm87_a4w4_attention_k256_launch_token_count(
          kWholeLogicalTokens) == kWholeLaunchTokens);

  const std::size_t local_launch_tokens =
      kernels::sm87_a4w4_attention_k256_launch_token_count(
          spec.tile_logical_tokens);
  const std::size_t input_elements =
      spec.tile_logical_tokens * kK256InputSize;
  const std::size_t whole_packed_bytes =
      kernels::sm87_a4w4_attention_k256_packed_capacity_bytes(
          kWholeLaunchTokens, kK256InputSize);
  const std::size_t whole_scale_elements =
      kernels::sm87_a4w4_attention_k256_scale_capacity_elements(
          kWholeLaunchTokens, kK256InputSize);
  const std::size_t local_packed_bytes =
      kernels::sm87_a4w4_attention_k256_packed_capacity_bytes(
          local_launch_tokens, kK256InputSize);
  const std::size_t local_scale_elements =
      kernels::sm87_a4w4_attention_k256_scale_capacity_elements(
          local_launch_tokens, kK256InputSize);
  const bool capacities_valid =
      local_launch_tokens != 0U && input_elements != 0U &&
      whole_packed_bytes != 0U && whole_scale_elements != 0U &&
      local_packed_bytes != 0U && local_scale_elements != 0U;
  test.expect(capacities_valid,
              std::string(spec.name) + " direct K256 capacities");
  if (!capacities_valid) {
    return;
  }

  ManagedBuffer<std::uint16_t> raw(input_elements);
  ManagedBuffer<std::uint16_t> gate(input_elements);
  ManagedBuffer<std::uint16_t> weight(kDimension);
  ManagedBuffer<std::uint16_t> normalized(input_elements);
  ManagedBuffer<std::uint8_t> incumbent_packed(
      local_packed_bytes + 2U * kPackedGuardBytes);
  ManagedBuffer<std::uint16_t> incumbent_scales(
      local_scale_elements + 2U * kScaleGuardElements);
  ManagedBuffer<std::uint8_t> expected_packed(
      whole_packed_bytes + 2U * kPackedGuardBytes);
  ManagedBuffer<std::uint16_t> expected_scales(
      whole_scale_elements + 2U * kScaleGuardElements);
  ManagedBuffer<std::uint8_t> candidate_packed(
      whole_packed_bytes + 2U * kPackedGuardBytes);
  ManagedBuffer<std::uint16_t> candidate_scales(
      whole_scale_elements + 2U * kScaleGuardElements);
  NonBlockingStream stream;
  const bool allocated =
      raw.valid() && gate.valid() && weight.valid() && normalized.valid() &&
      incumbent_packed.valid() && incumbent_scales.valid() &&
      expected_packed.valid() && expected_scales.valid() &&
      candidate_packed.valid() && candidate_scales.valid() && stream.valid();
  test.expect(allocated,
              std::string(spec.name) +
                  " allocate direct K256 byte-for-byte fixture");
  test.expect(stream.valid() && stream.get() != nullptr,
              std::string(spec.name) + " uses a nondefault CUDA stream");
  if (!allocated) {
    return;
  }

  fill_k256_publish_inputs(raw.data(), gate.data(), weight.data(),
                           spec.tile_logical_tokens);
  std::fill_n(normalized.data(), normalized.size(), kPoison);
  std::fill_n(incumbent_packed.data(), incumbent_packed.size(),
              kPackedSentinel);
  std::fill_n(incumbent_scales.data(), incumbent_scales.size(),
              kScaleSentinel);
  std::fill_n(expected_packed.data(), expected_packed.size(),
              kPackedSentinel);
  std::fill_n(expected_scales.data(), expected_scales.size(),
              kScaleSentinel);
  std::fill_n(candidate_packed.data(), candidate_packed.size(),
              kPackedSentinel);
  std::fill_n(candidate_scales.data(), candidate_scales.size(),
              kScaleSentinel);

  std::uint8_t* const incumbent_packed_payload =
      incumbent_packed.data() + kPackedGuardBytes;
  std::uint16_t* const incumbent_scale_payload =
      incumbent_scales.data() + kScaleGuardElements;
  std::uint8_t* const expected_packed_payload =
      expected_packed.data() + kPackedGuardBytes;
  std::uint16_t* const expected_scale_payload =
      expected_scales.data() + kScaleGuardElements;
  std::uint8_t* const candidate_packed_payload =
      candidate_packed.data() + kPackedGuardBytes;
  std::uint16_t* const candidate_scale_payload =
      candidate_scales.data() + kScaleGuardElements;

  const int rejected_status = chunk_o::launch_norm_rows8_k256_a4(
      raw.data(), weight.data(), gate.data(), spec.tile_logical_tokens,
      kEpsilon, spec.destination_first_token, kWholeLogicalTokens,
      kWholeLaunchTokens, kK256ClipRatio, candidate_packed_payload,
      whole_packed_bytes - 1U, candidate_scale_payload,
      whole_scale_elements, stream.get());
  test.expect(rejected_status == static_cast<int>(cudaErrorInvalidValue),
              std::string(spec.name) +
                  " rejects one-byte-short packed capacity");
  const bool rejected_untouched =
      std::all_of(candidate_packed.data(),
                  candidate_packed.data() + candidate_packed.size(),
                  [](const std::uint8_t value) {
                    return value == kPackedSentinel;
                  }) &&
      std::all_of(candidate_scales.data(),
                  candidate_scales.data() + candidate_scales.size(),
                  [](const std::uint16_t value) {
                    return value == kScaleSentinel;
                  });
  test.expect(rejected_untouched,
              std::string(spec.name) +
                  " rejected launch leaves payload and guards untouched");

  const int norm_status = chunk_o::launch_norm_rows8(
      raw.data(), weight.data(), gate.data(),
      spec.tile_logical_tokens * kValueHeads, kEpsilon, normalized.data(),
      stream.get());
  const int incumbent_status =
      kernels::launch_sm87_a4_quantize_bf16_k256_cuda(
          normalized.data(), kK256InputSize, spec.tile_logical_tokens,
          local_launch_tokens, kK256InputSize, kK256ClipRatio,
          incumbent_packed_payload, local_packed_bytes,
          incumbent_scale_payload, local_scale_elements, stream.get());
  const int candidate_status = chunk_o::launch_norm_rows8_k256_a4(
      raw.data(), weight.data(), gate.data(), spec.tile_logical_tokens,
      kEpsilon, spec.destination_first_token, kWholeLogicalTokens,
      kWholeLaunchTokens, kK256ClipRatio, candidate_packed_payload,
      whole_packed_bytes, candidate_scale_payload, whole_scale_elements,
      stream.get());
  const bool launched =
      test.cuda_ok(norm_status, std::string(spec.name) +
                                    " launch incumbent BF16 norm") &&
      test.cuda_ok(incumbent_status, std::string(spec.name) +
                                         " launch incumbent K256 quantize") &&
      test.cuda_ok(candidate_status, std::string(spec.name) +
                                         " launch direct K256 publisher") &&
      test.cuda_ok(static_cast<int>(cudaStreamSynchronize(stream.get())),
                   std::string(spec.name) +
                       " synchronize nondefault CUDA stream");
  if (!launched) {
    return;
  }

  const bool final_slice =
      spec.destination_first_token + spec.tile_logical_tokens ==
      kWholeLogicalTokens;
  const std::size_t padding_tokens =
      final_slice ? kWholeLaunchTokens - kWholeLogicalTokens : 0U;
  const std::size_t publish_tokens =
      spec.tile_logical_tokens + padding_tokens;
  test.expect(publish_tokens <= local_launch_tokens,
              std::string(spec.name) +
                  " incumbent oracle covers direct publish rows");
  if (publish_tokens > local_launch_tokens) {
    return;
  }

  for (std::size_t local_token = 0U; local_token < publish_tokens;
       ++local_token) {
    const std::size_t destination_token =
        spec.destination_first_token + local_token;
    for (std::size_t group = 0U; group < kK256PhysicalGroups; ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        expected_packed_payload[
            kernels::sm87_a4w4_attention_k256_packed_offset(
                destination_token, group, byte, kK256PhysicalGroups)] =
            incumbent_packed_payload[
                kernels::sm87_a4w4_attention_k256_packed_offset(
                    local_token, group, byte, kK256PhysicalGroups)];
      }
    }
    for (std::size_t group = 0U; group < kK256Groups; ++group) {
      expected_scale_payload[
          kernels::sm87_a4w4_attention_k256_scale_offset(
              destination_token, group, kK256Groups)] =
          incumbent_scale_payload[
              kernels::sm87_a4w4_attention_k256_scale_offset(
                  local_token, group, kK256Groups)];
    }
  }

  const std::string label_prefix =
      "GDN_CHUNK_O_BV64_NORM_K256_A4_" + std::string(spec.name);
  (void)expect_equal_bytes(
      test, label_prefix + "_PACKED_DEST_SLICE_AND_GUARDS",
      expected_packed.data(), candidate_packed.data(),
      candidate_packed.size());
  (void)expect_equal(test, label_prefix + "_SCALES_DEST_SLICE_AND_GUARDS",
                     expected_scales.data(), candidate_scales.data(),
                     candidate_scales.size());

  const bool incumbent_guards =
      guards_are_sentinel(incumbent_packed.data(), kPackedGuardBytes,
                          local_packed_bytes, kPackedGuardBytes,
                          kPackedSentinel) &&
      guards_are_sentinel(incumbent_scales.data(), kScaleGuardElements,
                          local_scale_elements, kScaleGuardElements,
                          kScaleSentinel);
  const bool candidate_guards =
      guards_are_sentinel(candidate_packed.data(), kPackedGuardBytes,
                          whole_packed_bytes, kPackedGuardBytes,
                          kPackedSentinel) &&
      guards_are_sentinel(candidate_scales.data(), kScaleGuardElements,
                          whole_scale_elements, kScaleGuardElements,
                          kScaleSentinel);
  test.expect(incumbent_guards,
              label_prefix + " incumbent quantizer guards intact");
  test.expect(candidate_guards,
              label_prefix + " direct publisher guards intact");

  if (final_slice) {
    const bool padding_ok = k256_padding_is_zero_one(
        candidate_packed_payload, candidate_scale_payload,
        kWholeLogicalTokens, kWholeLaunchTokens);
    test.expect(padding_ok, label_prefix + " final P1920 padding zero/one");
    std::cout << label_prefix << "_FINAL_PADDING first="
              << kWholeLogicalTokens << " end=" << kWholeLaunchTokens
              << " gate=" << (padding_ok ? "PASS" : "FAIL") << '\n';
  }
}

void test_norm_rows8_k256_a4(TestContext& test) {
  // All cases publish into the same production-sized P1853/P1920 ABI.  The
  // short cases are final slices, proving that C1..C31 can own the tail and
  // its padding; C512 proves that an interior destination slice does not
  // disturb any other row.
  constexpr K256PublishCase kCases[] = {
      {"C1_FINAL_P1853", 1U, 1'852U},
      {"C31_FINAL_P1853", 31U, 1'822U},
      {"C317_FINAL_P1853", 317U, 1'536U},
      {"C512_INTERIOR_P1853", 512U, 1'024U},
  };
  test_norm_k256_a4_resources(test);
  for (const K256PublishCase& spec : kCases) {
    test_norm_rows8_k256_a4_case(test, spec);
  }
}

void test_norm_factorized_lane_r1_a4_resources(TestContext& test) {
  int registers = 0;
  std::size_t static_shared = 0U;
  std::size_t local_bytes = 0U;
  int maximum_threads = 0;
  int active_blocks = 0;
  const int status =
      chunk_o::query_norm_factorized_lane_r1_a4_resources(
          &registers, &static_shared, &local_bytes, &maximum_threads,
          &active_blocks);
  const bool admitted =
      test.cuda_ok(status, "query direct norm factorized R1 A4 resources") &&
      registers > 0 && registers <= 128 &&
      static_shared == kFactorizedLaneR1SharedBytes &&
      local_bytes == 0U && maximum_threads >= 256 && active_blocks >= 2;
  test.expect(
      admitted,
      "direct norm factorized R1 A4 retains two CTA/SM resource contract");
  std::cout << "GDN_CHUNK_O_BV64_NORM_FACTORIZED_R1_A4_RESOURCES regs="
            << registers << " static_shared=" << static_shared
            << " local_bytes=" << local_bytes
            << " max_threads=" << maximum_threads
            << " active_blocks_per_sm=" << active_blocks
            << " gate=" << (admitted ? "PASS" : "FAIL") << '\n';
}

void test_norm_rows8_factorized_lane_r1_a4_case(
    TestContext& test, const K256PublishCase& spec) {
  constexpr std::size_t kWholeLogicalTokens = 1'853U;
  constexpr std::size_t kWholeLaunchTokens = 1'920U;
  const auto whole_plan =
      kernels::sm87_a4w4_factorized_lane_quantize_plan(
          kWholeLogicalTokens, kWholeLaunchTokens, kK256InputSize,
          kFactorizedLaneR1ScaleGroups);
  const std::size_t input_elements =
      spec.tile_logical_tokens * kK256InputSize;
  const bool capacities_valid = whole_plan.valid() && input_elements != 0U;
  test.expect(capacities_valid,
              std::string(spec.name) + " direct factorized R1 capacities");
  if (!capacities_valid) {
    return;
  }

  ManagedBuffer<std::uint16_t> raw(input_elements);
  ManagedBuffer<std::uint16_t> raw_snapshot(input_elements);
  ManagedBuffer<std::uint16_t> gate(input_elements);
  ManagedBuffer<std::uint16_t> weight(kDimension);
  ManagedBuffer<float> inverse_alpha(kK256InputSize);
  ManagedBuffer<std::uint16_t> normalized(input_elements);
  ManagedBuffer<std::uint8_t> expected_packed(
      whole_plan.packed_capacity_bytes + 2U * kPackedGuardBytes);
  ManagedBuffer<std::uint16_t> expected_scales(
      whole_plan.scale_capacity_elements + 2U * kScaleGuardElements);
  ManagedBuffer<std::uint8_t> candidate_packed(
      whole_plan.packed_capacity_bytes + 2U * kPackedGuardBytes);
  ManagedBuffer<std::uint16_t> candidate_scales(
      whole_plan.scale_capacity_elements + 2U * kScaleGuardElements);
  NonBlockingStream stream;
  const bool allocated =
      raw.valid() && raw_snapshot.valid() && gate.valid() && weight.valid() &&
      inverse_alpha.valid() && normalized.valid() &&
      expected_packed.valid() && expected_scales.valid() &&
      candidate_packed.valid() && candidate_scales.valid() && stream.valid();
  test.expect(allocated,
              std::string(spec.name) +
                  " allocate direct factorized R1 byte fixture");
  test.expect(stream.valid() && stream.get() != nullptr,
              std::string(spec.name) +
                  " direct factorized R1 uses nondefault stream");
  if (!allocated) {
    return;
  }

  fill_k256_publish_inputs(raw.data(), gate.data(), weight.data(),
                           spec.tile_logical_tokens);
  std::copy_n(raw.data(), input_elements, raw_snapshot.data());
  fill_factorized_lane_r1_inverse_alpha(inverse_alpha.data());
  std::fill_n(normalized.data(), normalized.size(), kPoison);
  std::fill_n(expected_packed.data(), expected_packed.size(),
              kPackedSentinel);
  std::fill_n(expected_scales.data(), expected_scales.size(),
              kScaleSentinel);
  std::fill_n(candidate_packed.data(), candidate_packed.size(),
              kPackedSentinel);
  std::fill_n(candidate_scales.data(), candidate_scales.size(),
              kScaleSentinel);

  std::uint8_t* const expected_packed_payload =
      expected_packed.data() + kPackedGuardBytes;
  std::uint16_t* const expected_scale_payload =
      expected_scales.data() + kScaleGuardElements;
  std::uint8_t* const candidate_packed_payload =
      candidate_packed.data() + kPackedGuardBytes;
  std::uint16_t* const candidate_scale_payload =
      candidate_scales.data() + kScaleGuardElements;

  const int rejected_status =
      chunk_o::launch_norm_rows8_factorized_lane_r1_a4(
          raw.data(), weight.data(), gate.data(), spec.tile_logical_tokens,
          kEpsilon, inverse_alpha.data(), inverse_alpha.size(),
          spec.destination_first_token, kWholeLogicalTokens,
          kWholeLaunchTokens, kFactorizedLaneR1ClipRatio,
          candidate_packed_payload, whole_plan.packed_capacity_bytes - 1U,
          candidate_scale_payload, whole_plan.scale_capacity_elements,
          stream.get());
  test.expect(rejected_status == static_cast<int>(cudaErrorInvalidValue),
              std::string(spec.name) +
                  " rejects one-byte-short factorized R1 packed capacity");
  const bool rejected_untouched =
      std::all_of(candidate_packed.data(),
                  candidate_packed.data() + candidate_packed.size(),
                  [](const std::uint8_t value) {
                    return value == kPackedSentinel;
                  }) &&
      std::all_of(candidate_scales.data(),
                  candidate_scales.data() + candidate_scales.size(),
                  [](const std::uint16_t value) {
                    return value == kScaleSentinel;
                  });
  test.expect(rejected_untouched,
              std::string(spec.name) +
                  " rejected factorized R1 launch leaves buffers untouched");

  const int norm_status = chunk_o::launch_norm_rows8(
      raw.data(), weight.data(), gate.data(),
      spec.tile_logical_tokens * kValueHeads, kEpsilon, normalized.data(),
      stream.get());
  const int candidate_status =
      chunk_o::launch_norm_rows8_factorized_lane_r1_a4(
          raw.data(), weight.data(), gate.data(), spec.tile_logical_tokens,
          kEpsilon, inverse_alpha.data(), inverse_alpha.size(),
          spec.destination_first_token, kWholeLogicalTokens,
          kWholeLaunchTokens, kFactorizedLaneR1ClipRatio,
          candidate_packed_payload, whole_plan.packed_capacity_bytes,
          candidate_scale_payload, whole_plan.scale_capacity_elements,
          stream.get());
  const bool launched =
      test.cuda_ok(norm_status,
                   std::string(spec.name) +
                       " launch incumbent BF16 norm for factorized R1") &&
      test.cuda_ok(candidate_status,
                   std::string(spec.name) +
                       " launch direct factorized R1 publisher") &&
      test.cuda_ok(static_cast<int>(cudaStreamSynchronize(stream.get())),
                   std::string(spec.name) +
                       " synchronize direct factorized R1 stream");
  if (!launched) {
    return;
  }

  factorized_lane_r1_quantize_host(
      normalized.data(), inverse_alpha.data(), spec.tile_logical_tokens,
      spec.destination_first_token, kWholeLogicalTokens, kWholeLaunchTokens,
      kFactorizedLaneR1ClipRatio, expected_packed_payload,
      expected_scale_payload);
  const std::string label_prefix =
      "GDN_CHUNK_O_BV64_NORM_FACTORIZED_R1_A4_" +
      std::string(spec.name);
  (void)expect_equal_bytes(
      test, label_prefix + "_PACKED_DEST_SLICE_AND_GUARDS",
      expected_packed.data(), candidate_packed.data(),
      candidate_packed.size());
  (void)expect_equal(
      test, label_prefix + "_SCALES_DEST_SLICE_AND_GUARDS",
      expected_scales.data(), candidate_scales.data(),
      candidate_scales.size());
  (void)expect_equal(test, label_prefix + "_RAW_INPUT_UNCHANGED",
                     raw_snapshot.data(), raw.data(), input_elements);

  const bool candidate_guards =
      guards_are_sentinel(candidate_packed.data(), kPackedGuardBytes,
                          whole_plan.packed_capacity_bytes,
                          kPackedGuardBytes, kPackedSentinel) &&
      guards_are_sentinel(candidate_scales.data(), kScaleGuardElements,
                          whole_plan.scale_capacity_elements,
                          kScaleGuardElements, kScaleSentinel);
  test.expect(candidate_guards,
              label_prefix + " direct publisher guards intact");
  if (spec.destination_first_token + spec.tile_logical_tokens ==
      kWholeLogicalTokens) {
    const bool padding_ok = factorized_lane_r1_padding_is_zero_one(
        candidate_packed_payload, candidate_scale_payload,
        kWholeLogicalTokens, kWholeLaunchTokens);
    test.expect(padding_ok,
                label_prefix + " final P1920 padding zero/one");
  }
}

void test_norm_rows8_factorized_lane_r1_a4(TestContext& test) {
  constexpr K256PublishCase kCases[] = {
      {"C1_FINAL_P1853", 1U, 1'852U},
      {"C31_FINAL_P1853", 31U, 1'822U},
      {"C317_FINAL_P1853", 317U, 1'536U},
      {"C512_INTERIOR_P1853", 512U, 1'024U},
  };
  test_norm_factorized_lane_r1_a4_resources(test);
  for (const K256PublishCase& spec : kCases) {
    test_norm_rows8_factorized_lane_r1_a4_case(test, spec);
  }
}

#if defined(Q3X_ENABLE_GDN_STATE_O_BV64_FUSION_ADMISSION)
void test_prompt_span_state_o_factorized_lane_r1_a4(
    TestContext& test, Buffers& buffers) {
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
  std::copy_n(buffers.initial_state.data(), kBoundaryStateElements,
              buffers.candidate_state.data());
  std::fill_n(buffers.candidate_raw.data(), kOutputElements, kPoison);

  constexpr std::size_t kWholeLogicalTokens = kTokens;
  const std::size_t whole_launch_tokens =
      kernels::sm87_a4w4_attention_k256_launch_token_count(
          kWholeLogicalTokens);
  const auto plan = kernels::sm87_a4w4_factorized_lane_quantize_plan(
      kWholeLogicalTokens, whole_launch_tokens, kK256InputSize,
      kFactorizedLaneR1ScaleGroups);
  ManagedBuffer<float> inverse_alpha(kK256InputSize);
  ManagedBuffer<std::uint8_t> expected_packed(plan.packed_capacity_bytes);
  ManagedBuffer<std::uint8_t> candidate_packed(plan.packed_capacity_bytes);
  ManagedBuffer<std::uint16_t> expected_scales(
      plan.scale_capacity_elements);
  ManagedBuffer<std::uint16_t> candidate_scales(
      plan.scale_capacity_elements);
  const bool allocated = plan.valid() && inverse_alpha.valid() &&
                         expected_packed.valid() && candidate_packed.valid() &&
                         expected_scales.valid() && candidate_scales.valid();
  test.expect(allocated,
              "allocate prompt-span state+O direct-R1 composition fixture");
  if (!allocated) {
    return;
  }
  fill_factorized_lane_r1_inverse_alpha(inverse_alpha.data());
  std::fill_n(expected_packed.data(), expected_packed.size(),
              kPackedSentinel);
  std::fill_n(candidate_packed.data(), candidate_packed.size(),
              kPackedSentinel);
  std::fill_n(expected_scales.data(), expected_scales.size(),
              kScaleSentinel);
  std::fill_n(candidate_scales.data(), candidate_scales.size(),
              kScaleSentinel);

  // raw_output and packed_a deliberately alias. The complete composition
  // must reject this before modifying either the in-place state or raw-O.
  const int alias_status =
      native::launch_prompt_span_state_o_factorized_lane_r1_a4(
          buffers.w.data(), buffers.u.data(), buffers.q.data(),
          buffers.k.data(), buffers.gamma.data(),
          buffers.candidate_state.data(), kTokens, buffers.weight.data(),
          buffers.gate.data(), kEpsilon, inverse_alpha.data(),
          inverse_alpha.size(), 0U, kWholeLogicalTokens,
          whole_launch_tokens, kFactorizedLaneR1ClipRatio,
          buffers.candidate_raw.data(),
          reinterpret_cast<std::uint8_t*>(buffers.candidate_raw.data()),
          plan.packed_capacity_bytes, candidate_scales.data(),
          plan.scale_capacity_elements);
  const bool alias_untouched =
      std::equal(buffers.candidate_state.data(),
                 buffers.candidate_state.data() + kBoundaryStateElements,
                 buffers.initial_state.data()) &&
      std::all_of(buffers.candidate_raw.data(),
                  buffers.candidate_raw.data() + kOutputElements,
                  [](const std::uint16_t value) {
                    return value == kPoison;
                  });
  test.expect(alias_status == static_cast<int>(cudaErrorInvalidValue) &&
                  alias_untouched,
              "prompt-span state+O direct-R1 alias preflight is launch-free");

  const int baseline_status =
      native::launch_prompt_span_state_o_baseline_for_test(
          buffers.w.data(), buffers.u.data(), buffers.q.data(),
          buffers.k.data(), buffers.gamma.data(),
          buffers.initial_state.data(), buffers.baseline_state.data(),
          kTokens, buffers.v.data(), buffers.h.data(),
          buffers.weight.data(), buffers.gate.data(), kEpsilon,
          buffers.raw.data(), buffers.output.data());
  const int expected_publish_status =
      chunk_o::launch_norm_rows8_factorized_lane_r1_a4(
          buffers.raw.data(), buffers.weight.data(), buffers.gate.data(),
          kTokens, kEpsilon, inverse_alpha.data(), inverse_alpha.size(), 0U,
          kWholeLogicalTokens, whole_launch_tokens,
          kFactorizedLaneR1ClipRatio, expected_packed.data(),
          plan.packed_capacity_bytes, expected_scales.data(),
          plan.scale_capacity_elements);
  const int candidate_status =
      native::launch_prompt_span_state_o_factorized_lane_r1_a4(
          buffers.w.data(), buffers.u.data(), buffers.q.data(),
          buffers.k.data(), buffers.gamma.data(),
          buffers.candidate_state.data(), kTokens, buffers.weight.data(),
          buffers.gate.data(), kEpsilon, inverse_alpha.data(),
          inverse_alpha.size(), 0U, kWholeLogicalTokens,
          whole_launch_tokens, kFactorizedLaneR1ClipRatio,
          buffers.candidate_raw.data(), candidate_packed.data(),
          plan.packed_capacity_bytes, candidate_scales.data(),
          plan.scale_capacity_elements);
  const bool ready =
      test.cuda_ok(baseline_status,
                   "launch prompt-span state+O direct-R1 baseline") &&
      test.cuda_ok(expected_publish_status,
                   "launch prompt-span expected direct-R1 publisher") &&
      test.cuda_ok(candidate_status,
                   "launch prompt-span fused state+O direct-R1 candidate") &&
      test.cuda_ok(static_cast<int>(cudaDeviceSynchronize()),
                   "synchronize prompt-span fused state+O direct-R1");
  if (!ready) {
    return;
  }
  (void)expect_equal(test, "GDN_STATE_O_R1_FUSED_STATE_C64",
                     buffers.baseline_state.data(),
                     buffers.candidate_state.data(),
                     kBoundaryStateElements);
  (void)expect_equal(test, "GDN_STATE_O_R1_FUSED_RAW_C64",
                     buffers.raw.data(), buffers.candidate_raw.data(),
                     kOutputElements);
  (void)expect_equal_bytes(test, "GDN_STATE_O_R1_FUSED_PACKED_C64",
                           expected_packed.data(), candidate_packed.data(),
                           plan.packed_capacity_bytes);
  (void)expect_equal(test, "GDN_STATE_O_R1_FUSED_SCALES_C64",
                     expected_scales.data(), candidate_scales.data(),
                     plan.scale_capacity_elements);
}
#endif

void test_native_factorized_lane_r1_alias_preflight(TestContext& test) {
  constexpr std::size_t kWholeLogicalTokens = kTokens;
  constexpr std::size_t kWholeLaunchTokens = 128U;
  const auto plan = kernels::sm87_a4w4_factorized_lane_quantize_plan(
      kWholeLogicalTokens, kWholeLaunchTokens, kK256InputSize,
      kFactorizedLaneR1ScaleGroups);
  ManagedBuffer<std::uint8_t> workspace(native::workspace_bytes());
  ManagedBuffer<std::uint16_t> conv_qkv(
      kTokens * runtime::kGdnQkvChannels);
  ManagedBuffer<std::uint16_t> a(kTokens * kValueHeads);
  ManagedBuffer<std::uint16_t> b(kTokens * kValueHeads);
  ManagedBuffer<std::uint16_t> a_log(kValueHeads);
  ManagedBuffer<std::uint16_t> dt_bias(kValueHeads);
  ManagedBuffer<std::uint16_t> state_input(kBoundaryStateElements);
  ManagedBuffer<std::uint16_t> state_output(kBoundaryStateElements);
  ManagedBuffer<std::uint16_t> norm_weight(kDimension);
  ManagedBuffer<std::uint16_t> silu_gate(kOutputElements);
  ManagedBuffer<float> inverse_alpha(kK256InputSize);
  ManagedBuffer<std::uint16_t> scales(plan.scale_capacity_elements);
  const bool allocated =
      plan.valid() && workspace.valid() && conv_qkv.valid() && a.valid() &&
      b.valid() && a_log.valid() && dt_bias.valid() && state_input.valid() &&
      state_output.valid() && norm_weight.valid() && silu_gate.valid() &&
      inverse_alpha.valid() && scales.valid();
  test.expect(allocated,
              "allocate native factorized R1 alias-preflight fixture");
  if (!allocated) {
    return;
  }

  std::fill_n(state_input.data(), state_input.size(), kZero);
  std::fill_n(state_output.data(), state_output.size(), kPoison);
  std::fill_n(workspace.data(), std::min<std::size_t>(workspace.size(), 64U),
              kPackedSentinel);
  std::fill_n(inverse_alpha.data(), inverse_alpha.size(), 1.0F);
  std::fill_n(scales.data(), scales.size(), kScaleSentinel);

  const auto launch_with_packed = [&](std::uint8_t* const packed) {
    return native::launch_factorized_lane_r1_a4(
        workspace.data(), workspace.size(), conv_qkv.data(), kTokens,
        a.data(), b.data(), a_log.data(), dt_bias.data(), state_input.data(),
        state_output.data(), kEpsilon, norm_weight.data(), silu_gate.data(),
        kEpsilon, inverse_alpha.data(), inverse_alpha.size(), 0U,
        kWholeLogicalTokens, kWholeLaunchTokens,
        kFactorizedLaneR1ClipRatio, packed, plan.packed_capacity_bytes,
        scales.data(), scales.size());
  };

  const int state_alias_status = launch_with_packed(
      reinterpret_cast<std::uint8_t*>(state_output.data()));
  test.expect(state_alias_status == static_cast<int>(cudaErrorInvalidValue),
              "native factorized R1 rejects packed/state-output alias before enqueue");
  test.expect(
      std::all_of(state_output.data(),
                  state_output.data() + state_output.size(),
                  [](const std::uint16_t value) { return value == kPoison; }),
      "state-output alias rejection leaves recurrent state untouched");

  const int workspace_alias_status = launch_with_packed(workspace.data());
  test.expect(workspace_alias_status ==
                  static_cast<int>(cudaErrorInvalidValue),
              "native factorized R1 rejects packed/workspace alias before enqueue");
  test.expect(
      std::all_of(workspace.data(),
                  workspace.data() +
                      std::min<std::size_t>(workspace.size(), 64U),
                  [](const std::uint8_t value) {
                    return value == kPackedSentinel;
                  }) &&
          std::all_of(state_output.data(),
                      state_output.data() + state_output.size(),
                      [](const std::uint16_t value) {
                        return value == kPoison;
                      }),
      "workspace alias rejection leaves workspace prefix and state untouched");
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
  test_norm_rows8_k256_a4(test);
  test_norm_rows8_factorized_lane_r1_a4(test);
#if defined(Q3X_ENABLE_GDN_STATE_O_BV64_FUSION_ADMISSION)
  test_prompt_span_state_o_factorized_lane_r1_a4(test, buffers);
#endif
  test_native_factorized_lane_r1_alias_preflight(test);
  std::cout << "GDN_CHUNK_O_BV64_COMPONENT_RESULT gate="
            << (test.result() == 0 ? "PASS" : "FAIL")
            << " authority=SYNTHETIC_CORRECTNESS_ONLY\n";
  return test.result();
}
