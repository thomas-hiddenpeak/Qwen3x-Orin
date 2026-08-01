#include "q3x/server/evaluation_server.h"
#include "q3x/server/evaluation_server_cli.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>

namespace {

namespace server = q3x::server;

class TestContext {
 public:
  void expect(const bool condition, const std::string_view message) {
    if (!condition) {
      std::cerr << "FAILED: " << message << '\n';
      ++failures_;
    }
  }

  [[nodiscard]] int failures() const noexcept { return failures_; }

 private:
  int failures_ = 0;
};

template <std::size_t Size>
[[nodiscard]] bool parse(const char* const (&arguments)[Size],
                         server::EvaluationServerOptions& options,
                         std::string& error) {
  return server::parse_evaluation_server_arguments(
      static_cast<int>(Size), arguments, options, error);
}

void test_disabled_default(TestContext& test) {
  const char* const arguments[] = {"qwen3x-eval-server", "/model"};
  server::EvaluationServerOptions options;
  std::string error;
  test.expect(parse(arguments, options, error) && error.empty() &&
                  options.model_directory == "/model" &&
                  options.profile_request_index == 0U,
              "omitting the diagnostic switch leaves profiling disabled");

  const char* const explicit_zero[] = {
      "qwen3x-eval-server", "/model", "--profile-request-index", "0"};
  options = {};
  error = "stale";
  test.expect(parse(explicit_zero, options, error) && error.empty() &&
                  options.profile_request_index == 0U,
              "an explicit zero preserves the disabled path");
}

void test_valid_index(TestContext& test) {
  const char* const arguments[] = {
      "qwen3x-eval-server", "/model", "--profile-request-index", "513"};
  server::EvaluationServerOptions options;
  std::string error;
  test.expect(parse(arguments, options, error) && error.empty() &&
                  options.profile_request_index == 513U,
              "a positive 1-based completion index parses exactly");
  test.expect(server::evaluation_server_request_profiling_compiled(),
              "BUILD_TESTING gateway exports the CUDA profiler link contract");
}

void test_invalid_indices(TestContext& test) {
  {
    const char* const arguments[] = {
        "qwen3x-eval-server", "/model", "--profile-request-index", "-1"};
    server::EvaluationServerOptions options;
    std::string error;
    test.expect(!parse(arguments, options, error) &&
                    error.find("unsigned integer") != std::string::npos,
                "negative profiler index is rejected");
  }
  {
    const char* const arguments[] = {
        "qwen3x-eval-server", "/model", "--profile-request-index", "1x"};
    server::EvaluationServerOptions options;
    std::string error;
    test.expect(!parse(arguments, options, error) &&
                    error.find("unsigned integer") != std::string::npos,
                "partially parsed profiler index is rejected");
  }
  {
    const char* const arguments[] = {
        "qwen3x-eval-server", "/model", "--profile-request-index",
        "18446744073709551616"};
    server::EvaluationServerOptions options;
    std::string error;
    test.expect(!parse(arguments, options, error) &&
                    error.find("unsigned integer") != std::string::npos,
                "overflowing profiler index is rejected");
  }
  {
    const char* const arguments[] = {
        "qwen3x-eval-server", "/model", "--profile-request-index", "1",
        "--profile-request-index", "2"};
    server::EvaluationServerOptions options;
    std::string error;
    test.expect(!parse(arguments, options, error) &&
                    error.find("one unsigned integer") != std::string::npos,
                "duplicate profiler selection is rejected");
  }
  {
    const char* const arguments[] = {
        "qwen3x-eval-server", "/model", "--profile-request-index"};
    server::EvaluationServerOptions options;
    std::string error;
    test.expect(!parse(arguments, options, error) &&
                    error.find("requires a value") != std::string::npos,
                "missing profiler index is rejected");
  }
}

void test_mlp_k512_publication(TestContext& test) {
  const char* const valid[] = {
      "qwen3x-eval-server",
      "/model",
      "--prefill-a4-payload",
      "/publication/a4.bin",
      "--prefill-a4-policy",
      "/publication/a4-policy.json",
      "--prefill-mlp-k512-payload",
      "/publication/mlp-k512.bin",
      "--prefill-mlp-k512-policy",
      "/publication/mlp-k512-policy.json",
      "--prefill-mlp-k512-receipt",
      "/publication/mlp-k512-receipt.json"};
  server::EvaluationServerOptions options;
  std::string error;
  test.expect(parse(valid, options, error) && error.empty() &&
                  options.prefill_mlp_k512_payload_path ==
                      "/publication/mlp-k512.bin" &&
                  options.prefill_mlp_k512_policy_path ==
                      "/publication/mlp-k512-policy.json" &&
                  options.prefill_mlp_k512_receipt_path ==
                      "/publication/mlp-k512-receipt.json",
              "a complete authenticated MLP K512 publication parses");

  const char* const incomplete[] = {
      "qwen3x-eval-server",
      "/model",
      "--prefill-a4-payload",
      "/publication/a4.bin",
      "--prefill-a4-policy",
      "/publication/a4-policy.json",
      "--prefill-mlp-k512-payload",
      "/publication/mlp-k512.bin",
      "--prefill-mlp-k512-policy",
      "/publication/mlp-k512-policy.json"};
  options = {};
  error.clear();
  test.expect(!parse(incomplete, options, error) &&
                  error.find("required together") != std::string::npos,
              "an incomplete MLP K512 publication is rejected");

  const char* const missing_base[] = {
      "qwen3x-eval-server",
      "/model",
      "--prefill-mlp-k512-payload",
      "/publication/mlp-k512.bin",
      "--prefill-mlp-k512-policy",
      "/publication/mlp-k512-policy.json",
      "--prefill-mlp-k512-receipt",
      "/publication/mlp-k512-receipt.json"};
  options = {};
  error.clear();
  test.expect(!parse(missing_base, options, error) &&
                  error.find("explicit K128 A4") != std::string::npos,
              "MLP K512 cannot be admitted without the K128 A4 base");
}

void test_mlp_k512_fragment_native_publication(TestContext& test) {
  const char* const valid[] = {
      "qwen3x-eval-server",
      "/model",
      "--prefill-a4-payload",
      "/publication/a4.bin",
      "--prefill-a4-policy",
      "/publication/a4-policy.json",
      "--prefill-mlp-k512-fragment-native-payload",
      "/publication/mlp-k512-v2.bin",
      "--prefill-mlp-k512-fragment-native-policy",
      "/publication/mlp-k512-v1-policy.json",
      "--prefill-mlp-k512-fragment-native-receipt",
      "/publication/mlp-k512-v2-receipt.json"};
  server::EvaluationServerOptions options;
  std::string error;
  test.expect(
      parse(valid, options, error) && error.empty() &&
          options.prefill_mlp_k512_fragment_native_payload_path ==
              "/publication/mlp-k512-v2.bin" &&
          options.prefill_mlp_k512_fragment_native_policy_path ==
              "/publication/mlp-k512-v1-policy.json" &&
          options.prefill_mlp_k512_fragment_native_receipt_path ==
              "/publication/mlp-k512-v2-receipt.json",
      "a complete authenticated fragment-native MLP K512 publication parses");

  const char* const incomplete[] = {
      "qwen3x-eval-server",
      "/model",
      "--prefill-a4-payload",
      "/publication/a4.bin",
      "--prefill-a4-policy",
      "/publication/a4-policy.json",
      "--prefill-mlp-k512-fragment-native-payload",
      "/publication/mlp-k512-v2.bin",
      "--prefill-mlp-k512-fragment-native-policy",
      "/publication/mlp-k512-v1-policy.json"};
  options = {};
  error.clear();
  test.expect(!parse(incomplete, options, error) &&
                  error.find("required together") != std::string::npos,
              "an incomplete fragment-native MLP K512 publication is "
              "rejected");

  const char* const missing_base[] = {
      "qwen3x-eval-server",
      "/model",
      "--prefill-mlp-k512-fragment-native-payload",
      "/publication/mlp-k512-v2.bin",
      "--prefill-mlp-k512-fragment-native-policy",
      "/publication/mlp-k512-v1-policy.json",
      "--prefill-mlp-k512-fragment-native-receipt",
      "/publication/mlp-k512-v2-receipt.json"};
  options = {};
  error.clear();
  test.expect(!parse(missing_base, options, error) &&
                  error.find("explicit K128 A4") != std::string::npos,
              "fragment-native MLP K512 requires the K128 A4 base");

  const char* const both_versions[] = {
      "qwen3x-eval-server",
      "/model",
      "--prefill-a4-payload",
      "/publication/a4.bin",
      "--prefill-a4-policy",
      "/publication/a4-policy.json",
      "--prefill-mlp-k512-payload",
      "/publication/mlp-k512-v1.bin",
      "--prefill-mlp-k512-policy",
      "/publication/mlp-k512-v1-policy.json",
      "--prefill-mlp-k512-receipt",
      "/publication/mlp-k512-v1-receipt.json",
      "--prefill-mlp-k512-fragment-native-payload",
      "/publication/mlp-k512-v2.bin",
      "--prefill-mlp-k512-fragment-native-policy",
      "/publication/mlp-k512-v1-policy.json",
      "--prefill-mlp-k512-fragment-native-receipt",
      "/publication/mlp-k512-v2-receipt.json"};
  options = {};
  error.clear();
  test.expect(!parse(both_versions, options, error) &&
                  error.find("mutually exclusive") != std::string::npos,
              "MLP K512 v1 and fragment-native v2 cannot coexist");

  const char* const duplicate_payload[] = {
      "qwen3x-eval-server",
      "/model",
      "--prefill-a4-payload",
      "/publication/a4.bin",
      "--prefill-a4-policy",
      "/publication/a4-policy.json",
      "--prefill-mlp-k512-fragment-native-payload",
      "/publication/mlp-k512-v2.bin",
      "--prefill-mlp-k512-fragment-native-policy",
      "/publication/mlp-k512-v1-policy.json",
      "--prefill-mlp-k512-fragment-native-receipt",
      "/publication/mlp-k512-v2-receipt.json",
      "--prefill-mlp-k512-fragment-native-payload",
      "/publication/duplicate.bin"};
  options = {};
  error.clear();
  test.expect(!parse(duplicate_payload, options, error) &&
                  error.find("one non-empty FILE") != std::string::npos,
              "a duplicate fragment-native payload flag is rejected");

  const char* const duplicate_policy[] = {
      "qwen3x-eval-server",
      "/model",
      "--prefill-a4-payload",
      "/publication/a4.bin",
      "--prefill-a4-policy",
      "/publication/a4-policy.json",
      "--prefill-mlp-k512-fragment-native-payload",
      "/publication/mlp-k512-v2.bin",
      "--prefill-mlp-k512-fragment-native-policy",
      "/publication/mlp-k512-v1-policy.json",
      "--prefill-mlp-k512-fragment-native-receipt",
      "/publication/mlp-k512-v2-receipt.json",
      "--prefill-mlp-k512-fragment-native-policy",
      "/publication/duplicate-policy.json"};
  options = {};
  error.clear();
  test.expect(!parse(duplicate_policy, options, error) &&
                  error.find("one non-empty FILE") != std::string::npos,
              "a duplicate fragment-native policy flag is rejected");

  const char* const duplicate_receipt[] = {
      "qwen3x-eval-server",
      "/model",
      "--prefill-a4-payload",
      "/publication/a4.bin",
      "--prefill-a4-policy",
      "/publication/a4-policy.json",
      "--prefill-mlp-k512-fragment-native-payload",
      "/publication/mlp-k512-v2.bin",
      "--prefill-mlp-k512-fragment-native-policy",
      "/publication/mlp-k512-v1-policy.json",
      "--prefill-mlp-k512-fragment-native-receipt",
      "/publication/mlp-k512-v2-receipt.json",
      "--prefill-mlp-k512-fragment-native-receipt",
      "/publication/duplicate-receipt.json"};
  options = {};
  error.clear();
  test.expect(!parse(duplicate_receipt, options, error) &&
                  error.find("one non-empty FILE") != std::string::npos,
              "a duplicate fragment-native receipt flag is rejected");
}

}  // namespace

int main() {
  TestContext test;
  test_disabled_default(test);
  test_valid_index(test);
  test_invalid_indices(test);
  test_mlp_k512_publication(test);
  test_mlp_k512_fragment_native_publication(test);
  if (test.failures() != 0) {
    std::cerr << test.failures()
              << " evaluation server CLI test(s) failed\n";
    return 1;
  }
  std::cout << "All evaluation server CLI tests passed\n";
  return 0;
}
