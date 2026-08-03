#include "q3x/runtime/prefill_r1_projection_plane_v2.h"

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace runtime = q3x::runtime;

class Test final {
 public:
  void expect(const bool condition, const std::string_view message) {
    if (!condition) {
      std::cerr << "FAILED: " << message << '\n';
      ++failures_;
    }
  }
  [[nodiscard]] int result() const noexcept { return failures_ == 0 ? 0 : 1; }

 private:
  int failures_ = 0;
};

[[nodiscard]] bool write_sealed(const fs::path& path,
                                const std::string_view bytes) {
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
                        S_IRUSR | S_IWUSR);
  if (fd < 0) return false;
  std::size_t completed = 0U;
  while (completed < bytes.size()) {
    const ssize_t count =
        ::write(fd, bytes.data() + completed, bytes.size() - completed);
    if (count < 0) {
      if (errno == EINTR) continue;
      (void)::close(fd);
      return false;
    }
    if (count == 0) {
      (void)::close(fd);
      return false;
    }
    completed += static_cast<std::size_t>(count);
  }
  bool ok = ::fsync(fd) == 0;
  ok = (::fchmod(fd, S_IRUSR) == 0) && ok;
  ok = (::fsync(fd) == 0) && ok;
  ok = (::close(fd) == 0) && ok;
  return ok;
}

[[nodiscard]] fs::path make_fixture_directory() {
  fs::path base = fs::temp_directory_path() /
                  "q3x-r1-v2-file-test.XXXXXX";
  std::string pattern = base.string();
  std::vector<char> mutable_pattern(pattern.begin(), pattern.end());
  mutable_pattern.push_back('\0');
  const char* const created = ::mkdtemp(mutable_pattern.data());
  return created == nullptr ? fs::path{} : fs::path(created);
}

void test_file_option_path_alias(Test& test, const fs::path& directory) {
  const fs::path shared = directory / "shared.input";
  test.expect(write_sealed(shared, "x"), "tiny alias fixture is created");
  runtime::PrefillR1ProjectionPlaneV2FileConversionOptions options;
  options.model_directory = directory;
  options.base_k256_payload_path = shared;
  options.base_k256_policy_path = shared;
  options.base_k256_receipt_path = shared;
  options.mlp_r1_payload_path = shared;
  options.mlp_r1_policy_path = shared;
  options.mlp_r1_receipt_path = shared;
  options.attention_r1_payload_path = shared;
  options.attention_r1_policy_path = shared;
  options.attention_r1_receipt_path = shared;
  options.output_path = directory / "aliased.output";
  const auto result =
      runtime::convert_authenticated_prefill_r1_projection_plane_v2_files(
          options);
  test.expect(!result &&
                  result.diagnostic.code ==
                      runtime::PrefillR1ProjectionPlaneV2ErrorCode::kUnsafePath,
              "file composer rejects path aliases before authentication or allocation");
  test.expect(!fs::exists(options.output_path) &&
                  !fs::exists(options.output_path.string() +
                              ".manifest.json") &&
                  !fs::exists(options.output_path.string() + ".policy.json") &&
                  !fs::exists(options.output_path.string() + ".receipt.json"),
              "path rejection leaves no publication residuals");
}

void test_file_converter_no_replace_preflight(Test& test,
                                              const fs::path& directory) {
  std::array<fs::path, 9U> inputs{};
  for (std::size_t index = 0U; index < inputs.size(); ++index) {
    inputs[index] = directory / ("distinct.input." + std::to_string(index));
    test.expect(write_sealed(inputs[index], "x"),
                "distinct tiny composer input is created");
  }
  runtime::PrefillR1ProjectionPlaneV2FileConversionOptions options;
  options.model_directory = directory;
  options.base_k256_payload_path = inputs[0U];
  options.base_k256_policy_path = inputs[1U];
  options.base_k256_receipt_path = inputs[2U];
  options.mlp_r1_payload_path = inputs[3U];
  options.mlp_r1_policy_path = inputs[4U];
  options.mlp_r1_receipt_path = inputs[5U];
  options.attention_r1_payload_path = inputs[6U];
  options.attention_r1_policy_path = inputs[7U];
  options.attention_r1_receipt_path = inputs[8U];
  options.output_path = directory / "existing.output";
  test.expect(write_sealed(options.output_path, "keep"),
              "pre-existing composer output is created");
  const auto result =
      runtime::convert_authenticated_prefill_r1_projection_plane_v2_files(
          options);
  test.expect(
      !result && result.diagnostic.code ==
                     runtime::PrefillR1ProjectionPlaneV2ErrorCode::
                         kPublicationConflict,
      "file composer rejects an existing target before source authentication");
  test.expect(fs::file_size(options.output_path) == 4U &&
                  !fs::exists(options.output_path.string() +
                              ".manifest.json") &&
                  !fs::exists(options.output_path.string() + ".policy.json") &&
                  !fs::exists(options.output_path.string() + ".receipt.json"),
              "composer no-replace preflight preserves old target and emits nothing");
}

[[nodiscard]] std::array<fs::path, 4U> make_sealed_set(
    Test& test, const fs::path& directory, const std::string_view prefix) {
  std::array<fs::path, 4U> files{};
  for (std::size_t index = 0U; index < files.size(); ++index) {
    files[index] = directory /
                   (std::string(prefix) + "." + std::to_string(index));
    test.expect(write_sealed(files[index], std::string(index + 1U, 'a')),
                "tiny sealed transaction member is created");
  }
  return files;
}

void test_no_replace_and_rollback(Test& test, const fs::path& directory) {
  // This publication contract is host tooling and is intentionally testable
  // without enabling the GPU/runtime projection-plane admission.
  auto temporary = make_sealed_set(test, directory, "no-replace.temp");
  std::array<fs::path, 4U> targets{};
  for (std::size_t index = 0U; index < targets.size(); ++index) {
    targets[index] = directory /
                     ("no-replace.target." + std::to_string(index));
  }
  test.expect(write_sealed(targets[0U], "original"),
              "pre-existing no-replace target is created");
  const auto conflict =
      runtime::publish_prefill_r1_projection_plane_v2_file_set_no_replace(
          temporary, targets);
  test.expect(conflict.code ==
                  runtime::PrefillR1ProjectionPlaneV2ErrorCode::
                      kPublicationConflict,
              "existing target is never replaced");
  test.expect(fs::exists(targets[0U]) && !fs::exists(targets[1U]) &&
                  !fs::exists(targets[2U]) && !fs::exists(targets[3U]),
              "no-replace conflict publishes no other member");

  temporary = make_sealed_set(test, directory, "rollback.temp");
  for (std::size_t index = 0U; index < targets.size(); ++index) {
    targets[index] = directory /
                     ("rollback.target." + std::to_string(index));
  }
  test.expect(write_sealed(targets[2U], "late-conflict"),
              "late rollback conflict target is created");
  const auto rolled_back =
      runtime::publish_prefill_r1_projection_plane_v2_file_set_no_replace(
          temporary, targets);
  test.expect(rolled_back.code ==
                  runtime::PrefillR1ProjectionPlaneV2ErrorCode::
                      kPublicationConflict,
              "late link conflict is reported");
  test.expect(!fs::exists(targets[0U]) && !fs::exists(targets[1U]) &&
                  fs::exists(targets[2U]) && !fs::exists(targets[3U]),
              "late publication failure rolls back every newly linked target");
}

void test_successful_four_file_set(Test& test, const fs::path& directory) {
  auto temporary = make_sealed_set(test, directory, "success.temp");
  std::array<fs::path, 4U> targets{};
  for (std::size_t index = 0U; index < targets.size(); ++index) {
    targets[index] = directory /
                     ("success.target." + std::to_string(index));
  }
  const auto published =
      runtime::publish_prefill_r1_projection_plane_v2_file_set_no_replace(
          temporary, targets);
  test.expect(published.ok(), "four sealed files publish as one host contract");
  for (std::size_t index = 0U; index < targets.size(); ++index) {
    test.expect(fs::exists(targets[index]) && !fs::exists(temporary[index]),
                "successful publication retains target and removes temp link");
  }
}

}  // namespace

int main() {
  Test test;
  const fs::path directory = make_fixture_directory();
  test.expect(!directory.empty(), "small fixture directory is created");
  if (directory.empty()) return test.result();
  test_file_option_path_alias(test, directory);
  test_file_converter_no_replace_preflight(test, directory);
  test_no_replace_and_rollback(test, directory);
  test_successful_four_file_set(test, directory);
  std::error_code cleanup_error;
  fs::permissions(directory, fs::perms::owner_all,
                  fs::perm_options::add, cleanup_error);
  fs::remove_all(directory, cleanup_error);
  test.expect(!cleanup_error, "small fixture directory is removed");
  return test.result();
}
