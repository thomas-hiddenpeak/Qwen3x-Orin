#include "q3x/runtime/prefill_mlp_factorized_lane_r4_candidate_converter.h"

#include <iostream>
#include <string_view>

namespace {

namespace runtime = q3x::runtime;

class Test final {
 public:
  void expect(const bool condition, const std::string_view message) {
    if (!condition) {
      std::cerr << "FAILED: " << message << '\n';
      ++failures_;
    }
  }

  [[nodiscard]] int result() const noexcept {
    return failures_ == 0 ? 0 : 1;
  }

 private:
  int failures_ = 0;
};

static_assert(
    runtime::kPrefillMLPFactorizedLaneR4PublicationPayloadBytes ==
    8'583'954'432ULL);
static_assert(runtime::kPrefillMLPFactorizedLaneR4PublicationLaneCount == 4U);

void expect_invalid_options(
    Test& test,
    const runtime::PrefillMLPFactorizedLaneR4IdentityCandidateConversionOptions&
        options,
    const std::string_view message) {
  const auto result = runtime::
      convert_pinned_qwen36_27b_prefill_mlp_factorized_lane_r4_identity_candidate(
          options);
  test.expect(!result, message);
  test.expect(
      result.diagnostic.code == runtime::
                                    PrefillMLPFactorizedLaneR4CandidateConverterErrorCode::
                                        kInvalidOption,
      "invalid direction-gate options fail before checkpoint or output I/O");
  test.expect(!result.manifest.has_value() && !result.policy.has_value() &&
                  !result.receipt.has_value(),
              "invalid options cannot create a publication identity");
  test.expect(result.stats.source_shards_authenticated == 0U &&
                  result.stats.source_shard_bytes_hashed == 0U,
              "invalid options cannot claim source-shard authentication");
}

}  // namespace

int main() {
  Test test;

  runtime::PrefillMLPFactorizedLaneR4IdentityCandidateConversionOptions
      options;
  expect_invalid_options(test, options,
                         "empty model/output and default clips are rejected");

  options.model_directory = "/does/not/matter";
  options.output_path = "/also/not/reached";
  options.weight_clip_ratio = 1.0;
  expect_invalid_options(test, options,
                         "activation clip has no implicit identity default");

  options.activation_clip_ratio = 1.0;
  options.row_chunk_size = 65U;
  expect_invalid_options(test, options,
                         "non-N64 streaming chunk is rejected before I/O");

  options.row_chunk_size = 320U;
  expect_invalid_options(test, options,
                         "direction publisher remains bounded to N256");

  options.row_chunk_size = 256U;
  options.weight_clip_ratio = 0.0;
  expect_invalid_options(test, options,
                         "weight clip has no implicit identity default");

  test.expect(
      runtime::kPrefillMLPFactorizedLaneR4IdentityCandidateAlpha5120 ==
          "builtin/q3x/identity-alpha-f32-v1/k5120" &&
          runtime::kPrefillMLPFactorizedLaneR4IdentityCandidateAlpha17408 ==
              "builtin/q3x/identity-alpha-f32-v1/k17408",
      "identity alpha binding is explicit and shape-specific");
  test.expect(
      runtime::kPrefillMLPFactorizedLaneR4IdentityCandidateAlpha5120Sha256 ==
              "42010c1c68b632e2ab15c82bca6edef2cac2026c889dd0202d609602b756f568" &&
          runtime::
                  kPrefillMLPFactorizedLaneR4IdentityCandidateAlpha17408Sha256 ==
              "15cd4df15b3bcb53816bb119e9d52efa3c0bbee237fa17c5a7c351dc9bfdcbcd",
      "identity FP32LE one[K] digests are pinned reproducibly");
  test.expect(
      runtime::to_string(runtime::
                             PrefillMLPFactorizedLaneR4CandidateConverterErrorCode::
                                 kPublicationConflict) ==
          "publication_conflict",
      "candidate diagnostics have a stable CLI spelling");

  return test.result();
}
