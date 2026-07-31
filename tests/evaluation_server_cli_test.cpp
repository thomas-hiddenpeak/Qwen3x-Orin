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

}  // namespace

int main() {
  TestContext test;
  test_disabled_default(test);
  test_valid_index(test);
  test_invalid_indices(test);
  if (test.failures() != 0) {
    std::cerr << test.failures()
              << " evaluation server CLI test(s) failed\n";
    return 1;
  }
  std::cout << "All evaluation server CLI tests passed\n";
  return 0;
}
