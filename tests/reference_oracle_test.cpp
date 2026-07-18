#include "q3x/runtime/reference_oracle.h"

#include <unistd.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace runtime = q3x::runtime;

class TestContext {
 public:
  void expect(const bool condition, const std::string_view message) {
    if (!condition) {
      ++failures_;
      std::cerr << "FAIL: " << message << '\n';
    }
  }

  [[nodiscard]] int failures() const noexcept { return failures_; }

 private:
  int failures_ = 0;
};

[[nodiscard]] std::string read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    return {};
  }
  const std::streampos end = input.tellg();
  if (end < 0) {
    return {};
  }
  std::string bytes(static_cast<std::size_t>(end), '\0');
  input.seekg(0, std::ios::beg);
  if (!bytes.empty()) {
    input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  return input ? bytes : std::string{};
}

[[nodiscard]] bool write_file(const std::filesystem::path& path,
                              const std::string& bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  return static_cast<bool>(output);
}

[[nodiscard]] bool replace_once(std::string& text,
                                const std::string_view old_value,
                                const std::string_view new_value) {
  const std::size_t position = text.find(old_value);
  if (position == std::string::npos) {
    return false;
  }
  text.replace(position, old_value.size(), new_value);
  return true;
}

class FixtureCopies {
 public:
  FixtureCopies(const std::filesystem::path& source_greedy,
                const std::filesystem::path& source_layers)
      : directory_(std::filesystem::temp_directory_path() /
                   ("q3x-reference-oracle-" +
                    std::to_string(static_cast<long long>(::getpid())))),
        greedy_path_(directory_ / source_greedy.filename()),
        layers_path_(directory_ / source_layers.filename()),
        greedy_(read_file(source_greedy)),
        layers_(read_file(source_layers)) {
    std::error_code error;
    std::filesystem::remove_all(directory_, error);
    std::filesystem::create_directories(directory_, error);
    valid_ = !error && !greedy_.empty() && !layers_.empty() && flush();
  }

  FixtureCopies(const FixtureCopies&) = delete;
  FixtureCopies& operator=(const FixtureCopies&) = delete;
  ~FixtureCopies() {
    std::error_code error;
    std::filesystem::remove_all(directory_, error);
  }

  [[nodiscard]] bool valid() const noexcept { return valid_; }
  [[nodiscard]] const std::filesystem::path& greedy_path() const noexcept {
    return greedy_path_;
  }
  [[nodiscard]] const std::filesystem::path& layers_path() const noexcept {
    return layers_path_;
  }
  [[nodiscard]] std::string& greedy() noexcept { return greedy_; }
  [[nodiscard]] std::string& layers() noexcept { return layers_; }
  [[nodiscard]] bool flush() const {
    return write_file(greedy_path_, greedy_) &&
           write_file(layers_path_, layers_);
  }

 private:
  std::filesystem::path directory_;
  std::filesystem::path greedy_path_;
  std::filesystem::path layers_path_;
  std::string greedy_;
  std::string layers_;
  bool valid_ = false;
};

void test_real_fixture(TestContext& test,
                       const std::filesystem::path& greedy_path,
                       const std::filesystem::path& layers_path) {
  const runtime::ReferenceOracleResult result =
      runtime::load_reference_oracle(greedy_path, layers_path);
  test.expect(result.ok(), "real BF16 oracle fixture pair loads");
  if (!result) {
    std::cerr << "oracle diagnostic: " << result.diagnostic.message << " at "
              << result.diagnostic.path << '\n';
    return;
  }
  const runtime::ReferenceOracle& oracle = *result.value;
  test.expect(oracle.greedy.source_revision ==
                  "0893e1606ff3d5f97a441f405d5fc541a6bdf404",
              "pinned source revision is retained");
  test.expect(oracle.greedy.prompt_token_ids.size() == 19U &&
                  oracle.greedy.prompt_token_ids.back() == 271U,
              "exact 19-token prompt is retained");
  test.expect(oracle.greedy.expected_token_ids.size() == 26U &&
                  oracle.greedy.expected_token_ids[0] == 77517U &&
                  oracle.greedy.expected_token_ids[1] == 220U &&
                  oracle.greedy.stop_token_id == 248046U,
              "expected greedy tokens and stop token are retained");
  test.expect(oracle.layers.phases[0].position == 18U &&
                  oracle.layers.phases[1].position == 19U,
              "two canonical phase positions are retained");
  test.expect(oracle.layers.phases[0].layers[63].index == 63U,
              "all 64 ordered layer records are retained");
  test.expect(
      oracle.layers.phases[0].embedding.sha256_raw ==
          "fb567db6f074fca439da6c7a200dd212c9936cb38a95d084f646da30eba4f0b3",
      "prefill embedding raw hash is retained");
  test.expect(
      oracle.layers.phases[1].final_norm.sha256_raw ==
          "95c5243bd88236100fb8cbe314341db7dac71b64dfacb8face11f6da39514a4b",
      "decode final-norm raw hash is retained");
  test.expect(oracle.layers.phases[0].logits.top20[0].token_id == 77517U &&
                  oracle.layers.phases[1].logits.top20[0].token_id == 220U,
              "phase predictions agree with top logits");
  test.expect(oracle.layers.sample_indices.front() == 0U &&
                  oracle.layers.sample_indices.back() == 5119U,
              "canonical sample positions are retained");
}

void test_limits(TestContext& test,
                 const std::filesystem::path& greedy_path,
                 const std::filesystem::path& layers_path) {
  test.expect(runtime::load_reference_oracle({}, layers_path)
                      .diagnostic.code ==
                  runtime::ReferenceOracleErrorCode::kInvalidArgument,
              "empty fixture path is rejected");
  test.expect(runtime::load_reference_oracle(greedy_path.parent_path(),
                                             layers_path)
                      .diagnostic.code ==
                  runtime::ReferenceOracleErrorCode::kNotRegularFile,
              "directory fixture path is rejected");

  runtime::ReferenceOracleLimits limits;
  limits.max_file_bytes = 100U;
  test.expect(runtime::load_reference_oracle(greedy_path, layers_path, limits)
                      .diagnostic.code ==
                  runtime::ReferenceOracleErrorCode::kFileTooLarge,
              "file byte limit is enforced before allocation");

  limits = {};
  limits.max_json_nodes = 10U;
  test.expect(runtime::load_reference_oracle(greedy_path, layers_path, limits)
                      .diagnostic.code ==
                  runtime::ReferenceOracleErrorCode::kResourceLimit,
              "global JSON node limit is enforced");

  limits = {};
  limits.max_nesting_depth = 2U;
  test.expect(runtime::load_reference_oracle(greedy_path, layers_path, limits)
                      .diagnostic.code ==
                  runtime::ReferenceOracleErrorCode::kResourceLimit,
              "JSON nesting-depth limit is enforced");

  limits = {};
  limits.max_container_items = 10U;
  test.expect(runtime::load_reference_oracle(greedy_path, layers_path, limits)
                      .diagnostic.code ==
                  runtime::ReferenceOracleErrorCode::kResourceLimit,
              "global JSON container-item limit is enforced");

  limits = {};
  limits.max_string_bytes = 4U;
  test.expect(runtime::load_reference_oracle(greedy_path, layers_path, limits)
                      .diagnostic.code ==
                  runtime::ReferenceOracleErrorCode::kResourceLimit,
              "per-string byte limit is enforced");

  limits = {};
  limits.max_array_length = 10U;
  test.expect(runtime::load_reference_oracle(greedy_path, layers_path, limits)
                      .diagnostic.code ==
                  runtime::ReferenceOracleErrorCode::kResourceLimit,
              "per-array length limit is enforced");

  limits = {};
  limits.max_file_bytes = 0U;
  test.expect(runtime::load_reference_oracle(greedy_path, layers_path, limits)
                      .diagnostic.code ==
                  runtime::ReferenceOracleErrorCode::kInvalidLimit,
              "zero oracle limit is rejected");
}

void test_mutations(TestContext& test,
                    const std::filesystem::path& source_greedy,
                    const std::filesystem::path& source_layers) {
  {
    FixtureCopies copies(source_greedy, source_layers);
    test.expect(copies.valid(), "schema mutation fixture copies are created");
    test.expect(replace_once(copies.greedy(), "\"schema_version\": 1",
                             "\"schema_version\": 2") &&
                    copies.flush(),
                "schema version mutation is written");
    test.expect(runtime::load_reference_oracle(copies.greedy_path(),
                                               copies.layers_path())
                        .diagnostic.code ==
                    runtime::ReferenceOracleErrorCode::kSchemaMismatch,
                "unknown schema version is rejected");
  }
  {
    FixtureCopies copies(source_greedy, source_layers);
    test.expect(replace_once(copies.greedy(), "{", "{\"extra\":0,") &&
                    copies.flush(),
                "extra-field mutation is written");
    test.expect(runtime::load_reference_oracle(copies.greedy_path(),
                                               copies.layers_path())
                        .diagnostic.code ==
                    runtime::ReferenceOracleErrorCode::kSchemaMismatch,
                "unexpected top-level field is rejected");
  }
  {
    FixtureCopies copies(source_greedy, source_layers);
    test.expect(replace_once(copies.greedy(),
                             "\"enable_thinking\": false",
                             "\"enable_thinking\": 0") &&
                    copies.flush(),
                "wrong-type mutation is written");
    test.expect(runtime::load_reference_oracle(copies.greedy_path(),
                                               copies.layers_path())
                        .diagnostic.code ==
                    runtime::ReferenceOracleErrorCode::kSchemaMismatch,
                "wrong scalar type is rejected");
  }
  {
    FixtureCopies copies(source_greedy, source_layers);
    test.expect(replace_once(copies.layers(), "\"length\": 5120",
                             "\"length\": 5119") &&
                    copies.flush(),
                "boundary length mutation is written");
    test.expect(runtime::load_reference_oracle(copies.greedy_path(),
                                               copies.layers_path())
                        .diagnostic.code ==
                    runtime::ReferenceOracleErrorCode::kSchemaMismatch,
                "wrong boundary length is rejected");
  }
  {
    FixtureCopies copies(source_greedy, source_layers);
    test.expect(replace_once(copies.layers(), "4095,\n      5119",
                             "4095,\n      5120") &&
                    copies.flush(),
                "sample-index mutation is written");
    test.expect(runtime::load_reference_oracle(copies.greedy_path(),
                                               copies.layers_path())
                        .diagnostic.code ==
                    runtime::ReferenceOracleErrorCode::kSchemaMismatch,
                "noncanonical sample index is rejected");
  }
  {
    FixtureCopies copies(source_greedy, source_layers);
    test.expect(replace_once(
                    copies.layers(),
                    "qwen36-27b-nvfp4-greedy-bf16.json", "wrong.json") &&
                    copies.flush(),
                "cross-file filename mutation is written");
    test.expect(runtime::load_reference_oracle(copies.greedy_path(),
                                               copies.layers_path())
                        .diagnostic.code ==
                    runtime::ReferenceOracleErrorCode::kCrossFileMismatch,
                "layer source filename mismatch is rejected");
  }
  {
    FixtureCopies copies(source_greedy, source_layers);
    test.expect(replace_once(
                    copies.layers(),
                    "0893e1606ff3d5f97a441f405d5fc541a6bdf404",
                    "1893e1606ff3d5f97a441f405d5fc541a6bdf404") &&
                    copies.flush(),
                "cross-file revision mutation is written");
    test.expect(runtime::load_reference_oracle(copies.greedy_path(),
                                               copies.layers_path())
                        .diagnostic.code ==
                    runtime::ReferenceOracleErrorCode::kCrossFileMismatch,
                "source revision mismatch is rejected");
  }
}

void test_bf16_summary_and_comparison(TestContext& test) {
  const std::vector<std::uint16_t> values = {0x3f80U, 0xc000U, 0x3f00U,
                                              0x0000U};
  const std::vector<std::size_t> indices = {0U, 2U, 3U};
  const runtime::BoundarySummaryResult summarized =
      runtime::summarize_bf16_span(values.data(), values.size(), indices);
  test.expect(summarized.ok(), "synthetic BF16 span is summarized");
  if (!summarized) {
    return;
  }
  const runtime::BoundarySummary& summary = *summarized.value;
  test.expect(
      summary.sha256_raw ==
          "8c2c6a65c7c224cc1521e4272dd00b0b47a70a11dc5819e2e193ed41207fef58",
      "BF16 hash uses canonical little-endian raw words");
  test.expect(summary.mean == -0.125 && summary.minimum == -2.0 &&
                  summary.maximum == 1.0 &&
                  std::fabs(summary.rms - std::sqrt(1.3125)) < 1.0e-15,
              "BF16 summary statistics are exact for synthetic values");
  test.expect(summary.samples == std::vector<double>({1.0, 0.5, 0.0}),
              "BF16 samples use requested indices");

  runtime::BoundaryComparisonResult comparison =
      runtime::compare_boundary_samples(values.data(), values.size(), summary,
                                         indices, 0.0, 0.0);
  test.expect(comparison.ok() && comparison.matches,
              "identical BF16 samples compare equal");

  std::vector<std::uint16_t> changed = values;
  changed[2] = 0x3f80U;
  comparison = runtime::compare_boundary_samples(
      changed.data(), changed.size(), summary, indices, 0.0, 0.0);
  test.expect(comparison.ok() && !comparison.matches &&
                  comparison.first_mismatch.has_value() &&
                  comparison.first_mismatch->sample_ordinal == 1U &&
                  comparison.first_mismatch->element_index == 2U,
              "comparison reports the first mismatching sample");

  changed[2] = 0x3f01U;
  comparison = runtime::compare_boundary_samples(
      changed.data(), changed.size(), summary, indices, 0.01, 0.0);
  test.expect(comparison.ok() && comparison.matches,
              "absolute tolerance permits a small cross-backend difference");

  const std::vector<std::size_t> bad_indices = {0U, 4U};
  test.expect(runtime::summarize_bf16_span(values.data(), values.size(),
                                           bad_indices)
                      .diagnostic.code ==
                  runtime::ReferenceOracleErrorCode::kInvalidSampleIndex,
              "out-of-range sample index is rejected");
  const std::vector<std::uint16_t> nonfinite = {0x7fc0U};
  test.expect(runtime::summarize_bf16_span(nonfinite.data(), nonfinite.size(),
                                           {0U})
                      .diagnostic.code ==
                  runtime::ReferenceOracleErrorCode::kNonFiniteValue,
              "NaN BF16 span is rejected");
}

}  // namespace

int main(const int argc, char** const argv) {
  if (argc != 3) {
    std::cerr << "usage: reference_oracle_test GREEDY_JSON LAYERS_JSON\n";
    return 2;
  }
  const std::filesystem::path greedy_path(argv[1]);
  const std::filesystem::path layers_path(argv[2]);
  TestContext test;
  test_real_fixture(test, greedy_path, layers_path);
  test_limits(test, greedy_path, layers_path);
  test_mutations(test, greedy_path, layers_path);
  test_bf16_summary_and_comparison(test);
  if (test.failures() != 0) {
    std::cerr << test.failures() << " reference oracle test(s) failed\n";
    return 1;
  }
  std::cout << "reference oracle tests passed\n";
  return 0;
}
