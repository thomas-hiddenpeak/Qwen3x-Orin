#include "gdn_prefill_chunk64_native_sm87.h"
#include "gdn_prefill_prompt_span_macro_sm87.h"
#include "gdn_prefill_wy_vllm_layout_sm87.h"

#include "q3x/runtime/gdn_decode.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>

namespace {

namespace macro =
    q3x::runtime::gdn_prefill_prompt_span_macro_detail;
namespace native = q3x::runtime::gdn_prefill_chunk64_native_detail;
namespace wy = q3x::runtime::gdn_prefill_wy_vllm_layout_detail;
namespace runtime = q3x::runtime;

constexpr std::size_t kTokens = 545U;
constexpr std::size_t kChunk = 64U;
constexpr std::size_t kChunks = (kTokens + kChunk - 1U) / kChunk;
constexpr std::size_t kPaddedTokens = kChunks * kChunk;
constexpr std::size_t kQkHeads = runtime::kGdnQkHeadCount;
constexpr std::size_t kValueHeads = runtime::kGdnValueHeadCount;
constexpr std::size_t kDimension = runtime::kGdnHeadDimension;
constexpr std::size_t kCompactChunkElements =
    kQkHeads * kChunk * kDimension;
constexpr std::size_t kCompactElements =
    kChunks * kCompactChunkElements;
constexpr std::size_t kValueChunkElements =
    kValueHeads * kChunk * kDimension;
constexpr std::size_t kValueElements = kChunks * kValueChunkElements;
constexpr std::size_t kStateElements =
    kValueHeads * kDimension * kDimension;
constexpr std::size_t kTransformChunkElements =
    kValueHeads * kChunk * kChunk;
constexpr std::size_t kTransformElements =
    kChunks * kTransformChunkElements;
constexpr std::size_t kRawGramChunkElements =
    kQkHeads * kChunk * kChunk;
constexpr std::size_t kRawGramElements =
    kChunks * kRawGramChunkElements;
constexpr std::size_t kScalarChunkElements = kValueHeads * kChunk;
constexpr std::size_t kScalarElements =
    kChunks * kScalarChunkElements;
constexpr std::size_t kBoundaryElements = kChunks * kStateElements;
constexpr std::size_t kConvElements =
    kTokens * runtime::kGdnQkvChannels;
constexpr std::size_t kRawOutputElements =
    kPaddedTokens * kValueHeads * kDimension;
constexpr std::size_t kOutputElements =
    kTokens * kValueHeads * kDimension;
constexpr std::uint16_t kZero = 0x0000U;
constexpr std::uint16_t kHalf = 0x3f00U;
constexpr std::uint16_t kQuarter = 0x3e80U;
constexpr std::uint16_t kEighth = 0x3e00U;
constexpr std::uint16_t kSixteenth = 0x3d80U;
constexpr std::uint16_t kSmall = 0x3c80U;
constexpr std::uint16_t kTiny = 0x3c00U;
constexpr std::uint16_t kPoison = 0x7fc1U;
constexpr float kNormEpsilon = 1.0e-6F;

static_assert(kTokens == 8U * kChunk + 33U);
static_assert(kChunks == 9U);
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
              << cudaGetErrorString(static_cast<cudaError_t>(status))
              << '\n';
    return false;
  }
  [[nodiscard]] int result() const noexcept {
    return failures_ == 0U ? 0 : 1;
  }

 private:
  std::size_t failures_ = 0U;
};

struct Buffers final {
  ManagedBuffer<std::uint16_t> compact_q{kCompactElements};
  ManagedBuffer<std::uint16_t> compact_k{kCompactElements};
  ManagedBuffer<std::uint16_t> conv_qkv{kConvElements};
  ManagedBuffer<float> gamma{kScalarElements};
  ManagedBuffer<float> beta{kScalarElements};
  ManagedBuffer<float> raw_gram{kRawGramElements};
  ManagedBuffer<std::uint16_t> transform{kTransformElements};
  ManagedBuffer<std::uint16_t> w{kValueElements};
  ManagedBuffer<std::uint16_t> u{kValueElements};
  ManagedBuffer<std::uint16_t> initial_state{kStateElements};
  ManagedBuffer<std::uint16_t> bridge_state{kStateElements};
  ManagedBuffer<std::uint16_t> baseline_state{kStateElements};
  ManagedBuffer<std::uint16_t> candidate_state{kStateElements};
  ManagedBuffer<std::uint16_t> alias_state{kStateElements};
  ManagedBuffer<std::uint16_t> boundary_state{kBoundaryElements};
  ManagedBuffer<std::uint16_t> candidate_boundary_state{kBoundaryElements};
  ManagedBuffer<std::uint16_t> v_new{kValueElements};
  ManagedBuffer<std::uint16_t> candidate_transform{kTransformElements};
  ManagedBuffer<std::uint16_t> candidate_w{kValueElements};
  ManagedBuffer<std::uint16_t> candidate_u{kValueElements};
  ManagedBuffer<std::uint16_t> candidate_v_new{kValueElements};
  ManagedBuffer<std::uint16_t> norm_weight{kDimension};
  ManagedBuffer<std::uint16_t> silu_gate{kOutputElements};
  ManagedBuffer<std::uint16_t> alias_gate{kOutputElements};
  ManagedBuffer<std::uint16_t> baseline_raw{kRawOutputElements};
  ManagedBuffer<std::uint16_t> candidate_raw{kRawOutputElements};
  ManagedBuffer<std::uint16_t> baseline_output{kOutputElements};
  ManagedBuffer<std::uint16_t> candidate_output{kOutputElements};

  [[nodiscard]] bool valid() const noexcept {
    return compact_q.valid() && compact_k.valid() && conv_qkv.valid() &&
           gamma.valid() && beta.valid() && raw_gram.valid() &&
           transform.valid() && w.valid() && u.valid() &&
           initial_state.valid() && bridge_state.valid() &&
           baseline_state.valid() && candidate_state.valid() &&
           alias_state.valid() && boundary_state.valid() &&
           candidate_boundary_state.valid() && v_new.valid() &&
           candidate_transform.valid() && candidate_w.valid() &&
           candidate_u.valid() && candidate_v_new.valid() &&
           norm_weight.valid() && silu_gate.valid() && alias_gate.valid() &&
           baseline_raw.valid() && candidate_raw.valid() &&
           baseline_output.valid() && candidate_output.valid();
  }
};

void initialize(Buffers& buffers) {
  std::fill_n(buffers.compact_q.data(), buffers.compact_q.size(), kZero);
  std::fill_n(buffers.compact_k.data(), buffers.compact_k.size(), kZero);
  std::fill_n(buffers.conv_qkv.data(), buffers.conv_qkv.size(), kZero);
  std::fill_n(buffers.raw_gram.data(), buffers.raw_gram.size(), 0.0F);
  std::fill_n(buffers.transform.data(), buffers.transform.size(), kPoison);
  std::fill_n(buffers.w.data(), buffers.w.size(), kPoison);
  std::fill_n(buffers.u.data(), buffers.u.size(), kPoison);
  std::fill_n(buffers.initial_state.data(), buffers.initial_state.size(),
              kZero);
  std::fill_n(buffers.bridge_state.data(), buffers.bridge_state.size(),
              kPoison);
  std::fill_n(buffers.baseline_state.data(), buffers.baseline_state.size(),
              kPoison);
  std::fill_n(buffers.candidate_state.data(), buffers.candidate_state.size(),
              kPoison);
  std::fill_n(buffers.boundary_state.data(), buffers.boundary_state.size(),
              kPoison);
  std::fill_n(buffers.candidate_boundary_state.data(),
              buffers.candidate_boundary_state.size(), kPoison);
  std::fill_n(buffers.v_new.data(), buffers.v_new.size(), kPoison);
  std::fill_n(buffers.candidate_transform.data(),
              buffers.candidate_transform.size(), kPoison);
  std::fill_n(buffers.candidate_w.data(), buffers.candidate_w.size(),
              kPoison);
  std::fill_n(buffers.candidate_u.data(), buffers.candidate_u.size(),
              kPoison);
  std::fill_n(buffers.candidate_v_new.data(),
              buffers.candidate_v_new.size(), kPoison);
  std::fill_n(buffers.baseline_raw.data(), buffers.baseline_raw.size(),
              kPoison);
  std::fill_n(buffers.candidate_raw.data(), buffers.candidate_raw.size(),
              kPoison);
  std::fill_n(buffers.baseline_output.data(), buffers.baseline_output.size(),
              kPoison);
  std::fill_n(buffers.candidate_output.data(),
              buffers.candidate_output.size(), kPoison);

  // A shared K dimension makes every strictly-lower Gram block nonzero;
  // the second dimension varies by token/head so the fixture is not rank-1.
  for (std::size_t chunk = 0U; chunk < kChunks; ++chunk) {
    for (std::size_t head = 0U; head < kQkHeads; ++head) {
      for (std::size_t token = 0U; token < kChunk; ++token) {
        const std::size_t global_token = chunk * kChunk + token;
        if (global_token >= kTokens) {
          continue;
        }
        const std::size_t base =
            ((chunk * kQkHeads + head) * kChunk + token) * kDimension;
        const std::size_t common = (head * 7U + 3U) % kDimension;
        std::size_t unique =
            (head * 17U + global_token * 13U + 1U) % kDimension;
        if (unique == common) {
          unique = (unique + 1U) % kDimension;
        }
        buffers.compact_k.data()[base + common] = kHalf;
        buffers.compact_k.data()[base + unique] = kSmall;
        buffers.compact_q.data()[base + common] = kQuarter;
        buffers.compact_q.data()[base + unique] = kSixteenth;
      }
    }
  }

  for (std::size_t chunk = 0U; chunk < kChunks; ++chunk) {
    const std::size_t valid_in_chunk =
        std::min(kChunk, kTokens - chunk * kChunk);
    for (std::size_t head = 0U; head < kValueHeads; ++head) {
      const float head_bias =
          0.0001220703125F * static_cast<float>(head & 7U);
      const float tail_gamma =
          -0.001953125F * static_cast<float>(valid_in_chunk) - head_bias;
      for (std::size_t token = 0U; token < kChunk; ++token) {
        const std::size_t scalar =
            (chunk * kValueHeads + head) * kChunk + token;
        buffers.gamma.data()[scalar] =
            token < valid_in_chunk
                ? -0.001953125F * static_cast<float>(token + 1U) -
                      head_bias
                : tail_gamma;
        buffers.beta.data()[scalar] =
            token < valid_in_chunk
                ? 0.40625F +
                      0.0078125F *
                          static_cast<float>((head + token) & 7U)
                : 0.0F;
      }
    }
  }

  for (std::size_t token = 0U; token < kTokens; ++token) {
    for (std::size_t head = 0U; head < kValueHeads; ++head) {
      const std::size_t value_base =
          token * runtime::kGdnQkvChannels + runtime::kGdnQElements +
          runtime::kGdnKElements + head * kDimension;
      const std::size_t common = (head * 11U + 5U) % kDimension;
      std::size_t unique =
          (head * 19U + token * 23U + 7U) % kDimension;
      if (unique == common) {
        unique = (unique + 1U) % kDimension;
      }
      buffers.conv_qkv.data()[value_base + common] = kEighth;
      buffers.conv_qkv.data()[value_base + unique] = kSmall;
    }
  }

  for (std::size_t head = 0U; head < kValueHeads; ++head) {
    for (std::size_t value = 0U; value < kDimension; ++value) {
      for (std::size_t key = 0U; key < kDimension; ++key) {
        const std::size_t index =
            (head * kDimension + value) * kDimension + key;
        if ((head * 5U + value * 3U + key) % 29U == 0U) {
          buffers.initial_state.data()[index] =
              ((value + key) & 1U) == 0U ? kTiny : kSmall;
        }
      }
    }
  }

  constexpr std::array<std::uint16_t, 4U> kNormValues{
      kHalf, 0x3f20U, 0x3f40U, 0x3f60U};
  constexpr std::array<std::uint16_t, 4U> kGateValues{
      kEighth, kQuarter, kHalf, 0x3f20U};
  for (std::size_t value = 0U; value < kDimension; ++value) {
    buffers.norm_weight.data()[value] = kNormValues[value & 3U];
  }
  for (std::size_t token = 0U; token < kTokens; ++token) {
    for (std::size_t head = 0U; head < kValueHeads; ++head) {
      for (std::size_t value = 0U; value < kDimension; ++value) {
        const std::size_t index =
            (token * kValueHeads + head) * kDimension + value;
        buffers.silu_gate.data()[index] =
            kGateValues[(token + head + value) & 3U];
      }
    }
  }
  std::copy_n(buffers.initial_state.data(), kStateElements,
              buffers.alias_state.data());
  std::copy_n(buffers.silu_gate.data(), kOutputElements,
              buffers.alias_gate.data());
}

[[nodiscard]] bool expect_equal(TestContext& test,
                                const std::string_view label,
                                const std::uint16_t* const expected,
                                const std::uint16_t* const actual,
                                const std::size_t elements) {
  std::size_t unequal = 0U;
  std::size_t first = elements;
  for (std::size_t index = 0U; index < elements; ++index) {
    if (expected[index] != actual[index]) {
      if (first == elements) {
        first = index;
      }
      ++unequal;
    }
  }
  std::cout << label << " unequal=" << unequal
            << " first_unequal=" << first;
  if (first != elements) {
    std::cout << " expected=0x" << std::hex << expected[first]
              << " actual=0x" << actual[first] << std::dec;
  }
  std::cout << " gate=" << (unequal == 0U ? "PASS" : "FAIL") << '\n';
  test.expect(unequal == 0U, label);
  return unequal == 0U;
}

[[nodiscard]] std::size_t count_nonzero(
    const std::uint16_t* const values, const std::size_t elements) {
  std::size_t count = 0U;
  for (std::size_t index = 0U; index < elements; ++index) {
    count += (values[index] & 0x7fffU) != 0U ? 1U : 0U;
  }
  return count;
}

[[nodiscard]] std::size_t count_changed(
    const std::uint16_t* const first, const std::uint16_t* const second,
    const std::size_t elements) {
  std::size_t count = 0U;
  for (std::size_t index = 0U; index < elements; ++index) {
    count += first[index] != second[index] ? 1U : 0U;
  }
  return count;
}

[[nodiscard]] std::size_t count_strictly_lower_gram_nonzero(
    const float* const raw_gram) {
  std::size_t count = 0U;
  for (std::size_t matrix = 0U; matrix < kChunks * kQkHeads; ++matrix) {
    for (std::size_t row = 1U; row < kChunk; ++row) {
      for (std::size_t column = 0U; column < row; ++column) {
        const std::size_t index =
            matrix * kChunk * kChunk + row * kChunk + column;
        count += raw_gram[index] != 0.0F ? 1U : 0U;
      }
    }
  }
  return count;
}

void test_resources(TestContext& test) {
  int registers = 0;
  std::size_t static_shared = 0U;
  std::size_t dynamic_shared = 0U;
  std::size_t local_bytes = 0U;
  int maximum_threads = 0;
  int active_blocks = 0;
  const int status = macro::query_c64_resources(
      &registers, &static_shared, &dynamic_shared, &local_bytes,
      &maximum_threads, &active_blocks);
  const bool queried = test.cuda_ok(status, "query prompt macro resources");
  test.expect(queried && registers <= 224 && static_shared == 0U &&
                  local_bytes == 0U && dynamic_shared == 115'200U &&
                  maximum_threads >= 256 && active_blocks == 1,
              "prompt macro hard resource contract");
  std::cout << "GDN_PROMPT_MACRO_RESOURCES regs=" << registers
            << " static_shared=" << static_shared
            << " dynamic_shared=" << dynamic_shared
            << " local_bytes=" << local_bytes
            << " max_threads=" << maximum_threads
            << " active_blocks_per_sm=" << active_blocks << '\n';
}

void test_bitwise_p545(TestContext& test, Buffers& buffers) {
  initialize(buffers);

  constexpr std::size_t kFirstTokens = 512U;
  constexpr std::size_t kFirstChunks = 8U;
  constexpr std::size_t kTailTokens = kTokens - kFirstTokens;
  const int wy_first_status = wy::launch_packless(
      buffers.compact_k.data(), buffers.gamma.data(), buffers.beta.data(),
      buffers.conv_qkv.data(), kFirstTokens, kFirstChunks,
      buffers.raw_gram.data(), buffers.transform.data(), buffers.w.data(),
      buffers.u.data(), nullptr);
  const int wy_tail_status = wy::launch_packless(
      buffers.compact_k.data() + kFirstChunks * kCompactChunkElements,
      buffers.gamma.data() + kFirstChunks * kScalarChunkElements,
      buffers.beta.data() + kFirstChunks * kScalarChunkElements,
      buffers.conv_qkv.data() +
          kFirstTokens * runtime::kGdnQkvChannels,
      kTailTokens, 1U,
      buffers.raw_gram.data() + kFirstChunks * kRawGramChunkElements,
      buffers.transform.data() + kFirstChunks * kTransformChunkElements,
      buffers.w.data() + kFirstChunks * kValueChunkElements,
      buffers.u.data() + kFirstChunks * kValueChunkElements, nullptr);

  const int baseline_first_status =
      native::launch_prompt_span_state_o_baseline_for_test(
          buffers.w.data(), buffers.u.data(), buffers.compact_q.data(),
          buffers.compact_k.data(), buffers.gamma.data(),
          buffers.initial_state.data(), buffers.bridge_state.data(),
          kFirstTokens, buffers.v_new.data(), buffers.boundary_state.data(),
          buffers.norm_weight.data(), buffers.silu_gate.data(),
          kNormEpsilon, buffers.baseline_raw.data(),
          buffers.baseline_output.data());
  const int baseline_tail_status =
      native::launch_prompt_span_state_o_baseline_for_test(
          buffers.w.data() + kFirstChunks * kValueChunkElements,
          buffers.u.data() + kFirstChunks * kValueChunkElements,
          buffers.compact_q.data() +
              kFirstChunks * kCompactChunkElements,
          buffers.compact_k.data() +
              kFirstChunks * kCompactChunkElements,
          buffers.gamma.data() + kFirstChunks * kScalarChunkElements,
          buffers.bridge_state.data(), buffers.baseline_state.data(),
          kTailTokens,
          buffers.v_new.data() + kFirstChunks * kValueChunkElements,
          buffers.boundary_state.data() +
              kFirstChunks * kStateElements,
          buffers.norm_weight.data(),
          buffers.silu_gate.data() +
              kFirstTokens * kValueHeads * kDimension,
          kNormEpsilon,
          buffers.baseline_raw.data() +
              kFirstTokens * kValueHeads * kDimension,
          buffers.baseline_output.data() +
              kFirstTokens * kValueHeads * kDimension);

  const macro::DiagnosticBoundaries diagnostics{
      buffers.candidate_transform.data(),
      buffers.candidate_w.data(),
      buffers.candidate_u.data(),
      buffers.candidate_v_new.data(),
      buffers.candidate_boundary_state.data(),
      buffers.candidate_raw.data()};
  const int candidate_status = macro::launch_c64_diagnostic(
      buffers.raw_gram.data(), buffers.gamma.data(), buffers.beta.data(),
      buffers.compact_q.data(), buffers.compact_k.data(),
      buffers.conv_qkv.data(), kTokens, buffers.initial_state.data(),
      buffers.candidate_state.data(), buffers.norm_weight.data(),
      buffers.silu_gate.data(), kNormEpsilon,
      buffers.candidate_output.data(), diagnostics);
  const bool ready =
      test.cuda_ok(wy_first_status, "launch incumbent WY P512") &&
      test.cuda_ok(wy_tail_status, "launch incumbent WY P33 tail") &&
      test.cuda_ok(baseline_first_status,
                   "launch incumbent state/O P512") &&
      test.cuda_ok(baseline_tail_status,
                   "launch incumbent state/O P33 tail") &&
      test.cuda_ok(candidate_status, "launch macro diagnostic P545") &&
      test.cuda_ok(static_cast<int>(cudaDeviceSynchronize()),
                   "synchronize P545 bitwise oracle");
  if (!ready) {
    return;
  }

  const std::size_t lower_gram_nonzero =
      count_strictly_lower_gram_nonzero(buffers.raw_gram.data());
  const std::size_t transform_nonzero = count_nonzero(
      buffers.transform.data(), kTransformElements);
  const std::size_t boundary_nonzero = count_nonzero(
      buffers.boundary_state.data(), kBoundaryElements);
  const std::size_t final_state_nonzero = count_nonzero(
      buffers.baseline_state.data(), kStateElements);
  const std::size_t final_state_changed = count_changed(
      buffers.initial_state.data(), buffers.baseline_state.data(),
      kStateElements);
  const std::size_t vnew_nonzero =
      count_nonzero(buffers.v_new.data(), kValueElements);
  const std::size_t raw_nonzero =
      count_nonzero(buffers.baseline_raw.data(), kOutputElements);
  const std::size_t output_nonzero =
      count_nonzero(buffers.baseline_output.data(), kOutputElements);
  const std::size_t tail_vnew_nonzero = count_nonzero(
      buffers.v_new.data() + kFirstChunks * kValueChunkElements,
      kValueChunkElements);
  const std::size_t tail_raw_nonzero = count_nonzero(
      buffers.baseline_raw.data() +
          kFirstTokens * kValueHeads * kDimension,
      kTailTokens * kValueHeads * kDimension);
  const std::size_t tail_output_nonzero = count_nonzero(
      buffers.baseline_output.data() +
          kFirstTokens * kValueHeads * kDimension,
      kTailTokens * kValueHeads * kDimension);
  const auto* const chunk8_boundary =
      buffers.boundary_state.data() + kFirstChunks * kStateElements;
  const std::size_t chunk8_boundary_nonzero =
      count_nonzero(chunk8_boundary, kStateElements);
  const std::size_t chunk8_boundary_changed = count_changed(
      buffers.initial_state.data(), chunk8_boundary, kStateElements);
  std::cout << "GDN_PROMPT_MACRO_NONDEGENERACY lower_gram="
            << lower_gram_nonzero << " transform=" << transform_nonzero
            << " boundary_state=" << boundary_nonzero
            << " final_state=" << final_state_nonzero
            << " final_state_changed=" << final_state_changed
            << " vnew=" << vnew_nonzero << " raw_o=" << raw_nonzero
            << " output=" << output_nonzero
            << " tail_vnew=" << tail_vnew_nonzero
            << " tail_raw_o=" << tail_raw_nonzero
            << " tail_output=" << tail_output_nonzero
            << " chunk8_boundary=" << chunk8_boundary_nonzero
            << " chunk8_boundary_changed=" << chunk8_boundary_changed
            << '\n';
  test.expect(lower_gram_nonzero > 0U, "strictly-lower Gram is nonzero");
  test.expect(transform_nonzero > 0U, "transform is nonzero");
  test.expect(boundary_nonzero > 0U, "boundary state is nonzero");
  test.expect(final_state_nonzero > 0U, "final state is nonzero");
  test.expect(final_state_changed > 0U, "state update changes state");
  test.expect(vnew_nonzero > 0U, "Vnew is nonzero");
  test.expect(raw_nonzero > 0U, "raw O is nonzero");
  test.expect(output_nonzero > 0U, "normalized/gated output is nonzero");
  test.expect(tail_vnew_nonzero > 0U, "P33 tail Vnew is nonzero");
  test.expect(tail_raw_nonzero > 0U, "P33 tail raw O is nonzero");
  test.expect(tail_output_nonzero > 0U,
              "P33 tail normalized/gated output is nonzero");
  test.expect(chunk8_boundary_nonzero > 0U,
              "chunk8 boundary state is nonzero");
  test.expect(chunk8_boundary_changed > 0U,
              "chunk8 boundary state differs from initial state");

  (void)expect_equal(test, "GDN_PROMPT_MACRO_TRANSFORM_P545",
                     buffers.transform.data(),
                     buffers.candidate_transform.data(),
                     kTransformElements);
  (void)expect_equal(test, "GDN_PROMPT_MACRO_W_P545", buffers.w.data(),
                     buffers.candidate_w.data(), kValueElements);
  (void)expect_equal(test, "GDN_PROMPT_MACRO_U_P545", buffers.u.data(),
                     buffers.candidate_u.data(), kValueElements);
  (void)expect_equal(test, "GDN_PROMPT_MACRO_BOUNDARY_STATE_P545",
                     buffers.boundary_state.data(),
                     buffers.candidate_boundary_state.data(),
                     kBoundaryElements);
  (void)expect_equal(test, "GDN_PROMPT_MACRO_VNEW_P545",
                     buffers.v_new.data(), buffers.candidate_v_new.data(),
                     kValueElements);
  (void)expect_equal(test, "GDN_PROMPT_MACRO_RAW_O_P545",
                     buffers.baseline_raw.data(),
                     buffers.candidate_raw.data(), kRawOutputElements);
  (void)expect_equal(test, "GDN_PROMPT_MACRO_STATE_P545",
                     buffers.baseline_state.data(),
                     buffers.candidate_state.data(), kStateElements);
  (void)expect_equal(test, "GDN_PROMPT_MACRO_OUTPUT_P545",
                     buffers.baseline_output.data(),
                     buffers.candidate_output.data(), kOutputElements);

  // Exercise both sanctioned exact aliases at once: state input/output and
  // the final SiLU-gate/consumer buffer.
  const int alias_status = macro::launch_c64(
      buffers.raw_gram.data(), buffers.gamma.data(), buffers.beta.data(),
      buffers.compact_q.data(), buffers.compact_k.data(),
      buffers.conv_qkv.data(), kTokens, buffers.alias_state.data(),
      buffers.alias_state.data(), buffers.norm_weight.data(),
      buffers.alias_gate.data(), kNormEpsilon, buffers.alias_gate.data());
  const bool alias_ready =
      test.cuda_ok(alias_status, "launch true in-place P545") &&
      test.cuda_ok(static_cast<int>(cudaDeviceSynchronize()),
                   "synchronize true in-place P545");
  if (alias_ready) {
    (void)expect_equal(test, "GDN_PROMPT_MACRO_INPLACE_SECONDARY_P545",
                       buffers.candidate_output.data(),
                       buffers.alias_gate.data(), kOutputElements);
    (void)expect_equal(test, "GDN_PROMPT_MACRO_INPLACE_STATE_P545",
                       buffers.candidate_state.data(),
                       buffers.alias_state.data(), kStateElements);
  }
}

}  // namespace

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::cout << "SKIP: GDN prompt macro P545 requires CUDA\n";
    return 77;
  }
  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) {
    return 1;
  }
  if (properties.major != 8 || properties.minor != 7) {
    std::cout << "SKIP: GDN prompt macro P545 requires SM87\n";
    return 77;
  }
  TestContext test;
  Buffers buffers;
  test.expect(buffers.valid(), "allocate P545 macro buffers");
  if (!buffers.valid()) {
    return test.result();
  }
  test_resources(test);
  test_bitwise_p545(test, buffers);
  std::cout << "GDN_PROMPT_MACRO_P545_RESULT gate="
            << (test.result() == 0 ? "PASS" : "FAIL")
            << " authority=SYNTHETIC_CORRECTNESS_ONLY\n";
  return test.result();
}
