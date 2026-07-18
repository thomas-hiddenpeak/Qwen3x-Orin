#include "q3x/runtime/reference_oracle.h"

#include "q3x/core/sha256.h"
#include "q3x/io/json.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <new>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace q3x::runtime {
namespace {

namespace json = q3x::io::json;

constexpr std::string_view kGreedyFixture =
    "qwen36-27b-nvfp4-text-only-greedy-bf16-cache";
constexpr std::string_view kLayerFixture =
    "qwen36-27b-nvfp4-layer-boundaries-bf16-cache";
constexpr std::string_view kRepository = "nvidia/Qwen3.6-27B-NVFP4";
constexpr std::string_view kRevision =
    "0893e1606ff3d5f97a441f405d5fc541a6bdf404";
constexpr std::string_view kBf16 = "bfloat16";

constexpr std::array<std::size_t, kReferenceOracleSampleCount>
    kPinnedSampleIndices = {0U,   1U,    2U,    3U,    7U,    15U,
                            31U,  63U,   127U,  255U,  511U,  1023U,
                            2047U, 3071U, 4095U, 5119U};

constexpr std::size_t kAbsoluteFileBytes = 64U * 1024U * 1024U;
constexpr std::size_t kAbsoluteJsonNodes = 1'000'000U;
constexpr std::size_t kAbsoluteDepth = 128U;
constexpr std::size_t kAbsoluteContainerItems = 1'000'000U;
constexpr std::size_t kAbsoluteStringBytes = 8U * 1024U * 1024U;
constexpr std::size_t kAbsoluteArrayLength = 1'000'000U;

class FileDescriptor {
 public:
  explicit FileDescriptor(const int value) noexcept : value_(value) {}
  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;
  ~FileDescriptor() {
    if (value_ >= 0) {
      (void)::close(value_);
    }
  }
  [[nodiscard]] int get() const noexcept { return value_; }

 private:
  int value_ = -1;
};

[[nodiscard]] float decode_bf16(const std::uint16_t value) noexcept {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
  float decoded = 0.0F;
  static_assert(sizeof(decoded) == sizeof(bits), "float must be binary32");
  std::memcpy(&decoded, &bits, sizeof(decoded));
  return decoded;
}

[[nodiscard]] bool lowercase_sha256(const std::string& value) noexcept {
  if (value.size() != 64U) {
    return false;
  }
  for (const char character : value) {
    const bool digit = character >= '0' && character <= '9';
    const bool lower_hex = character >= 'a' && character <= 'f';
    if (!digit && !lower_hex) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool limits_valid(const ReferenceOracleLimits& limits) noexcept {
  return limits.max_file_bytes != 0U && limits.max_json_nodes != 0U &&
         limits.max_nesting_depth != 0U &&
         limits.max_container_items != 0U &&
         limits.max_string_bytes != 0U && limits.max_array_length != 0U &&
         limits.max_file_bytes <= kAbsoluteFileBytes &&
         limits.max_json_nodes <= kAbsoluteJsonNodes &&
         limits.max_nesting_depth <= kAbsoluteDepth &&
         limits.max_container_items <= kAbsoluteContainerItems &&
         limits.max_string_bytes <= kAbsoluteStringBytes &&
         limits.max_array_length <= kAbsoluteArrayLength;
}

class OracleLoader {
 public:
  OracleLoader(const std::filesystem::path& greedy_path,
               const std::filesystem::path& layers_path,
               const ReferenceOracleLimits& limits)
      : greedy_path_(greedy_path), layers_path_(layers_path), limits_(limits) {}

  [[nodiscard]] ReferenceOracleResult run() {
    ReferenceOracleResult result;
    if (!limits_valid(limits_)) {
      fail(ReferenceOracleErrorCode::kInvalidLimit, "$limits",
           "oracle limits are zero or exceed absolute safety ceilings");
      result.diagnostic = std::move(diagnostic_);
      return result;
    }
    json::Value greedy_root;
    json::Value layers_root;
    if (!load_json(greedy_path_, greedy_root) ||
        !load_json(layers_path_, layers_root)) {
      result.diagnostic = std::move(diagnostic_);
      return result;
    }

    ReferenceOracle oracle;
    if (!parse_greedy(greedy_root, oracle.greedy) ||
        !parse_layers(layers_root, oracle.layers) ||
        !cross_validate(oracle)) {
      result.diagnostic = std::move(diagnostic_);
      return result;
    }
    result.value.emplace(std::move(oracle));
    return result;
  }

 private:
  void fail(const ReferenceOracleErrorCode code, std::string path,
            std::string message, std::string expected = {},
            std::string actual = {}, const int system_error = 0,
            const std::size_t json_offset = 0U) {
    if (diagnostic_.code != ReferenceOracleErrorCode::kNone) {
      return;
    }
    diagnostic_.code = code;
    diagnostic_.path = std::move(path);
    diagnostic_.message = std::move(message);
    diagnostic_.expected = std::move(expected);
    diagnostic_.actual = std::move(actual);
    diagnostic_.system_error = system_error;
    diagnostic_.json_offset = json_offset;
  }

  [[nodiscard]] bool read_regular_file(const std::filesystem::path& path,
                                       std::string& output) {
    if (path.empty()) {
      fail(ReferenceOracleErrorCode::kInvalidArgument, "$file",
           "oracle fixture path is empty");
      return false;
    }
    const std::string native = path.string();
    const int raw_fd = ::open(native.c_str(),
                              O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    if (raw_fd < 0) {
      const int error = errno;
      fail(error == ELOOP ? ReferenceOracleErrorCode::kNotRegularFile
                          : ReferenceOracleErrorCode::kOpenFailed,
           native, "could not open oracle fixture as a non-symlink file", {},
           {}, error);
      return false;
    }
    FileDescriptor fd(raw_fd);
    struct stat before {};
    if (::fstat(fd.get(), &before) != 0) {
      const int error = errno;
      fail(ReferenceOracleErrorCode::kIoFailure, native,
           "could not stat open oracle fixture", {}, {}, error);
      return false;
    }
    if (!S_ISREG(before.st_mode)) {
      fail(ReferenceOracleErrorCode::kNotRegularFile, native,
           "oracle fixture is not a regular file");
      return false;
    }
    if (before.st_size < 0 ||
        static_cast<std::uintmax_t>(before.st_size) > limits_.max_file_bytes ||
        static_cast<std::uintmax_t>(before.st_size) >
            std::numeric_limits<std::size_t>::max()) {
      fail(ReferenceOracleErrorCode::kFileTooLarge, native,
           "oracle fixture exceeds max_file_bytes",
           std::to_string(limits_.max_file_bytes),
           before.st_size < 0 ? "negative size"
                              : std::to_string(before.st_size));
      return false;
    }
    output.resize(static_cast<std::size_t>(before.st_size));
    std::size_t offset = 0U;
    while (offset < output.size()) {
      const ssize_t read_size =
          ::read(fd.get(), output.data() + offset, output.size() - offset);
      if (read_size < 0 && errno == EINTR) {
        continue;
      }
      if (read_size <= 0) {
        const int error = read_size < 0 ? errno : 0;
        fail(ReferenceOracleErrorCode::kIoFailure, native,
             "short read while loading oracle fixture", {}, {}, error);
        return false;
      }
      offset += static_cast<std::size_t>(read_size);
    }
    struct stat after {};
    if (::fstat(fd.get(), &after) != 0) {
      const int error = errno;
      fail(ReferenceOracleErrorCode::kIoFailure, native,
           "could not restat oracle fixture after reading", {}, {}, error);
      return false;
    }
    if (after.st_size != before.st_size || after.st_dev != before.st_dev ||
        after.st_ino != before.st_ino) {
      fail(ReferenceOracleErrorCode::kIoFailure, native,
           "oracle fixture changed while it was being read");
      return false;
    }
    return true;
  }

  [[nodiscard]] bool resource_walk(const json::Value& value,
                                   const std::string_view path) {
    if (const std::string* text = value.as_string(); text != nullptr) {
      if (text->size() > limits_.max_string_bytes) {
        fail(ReferenceOracleErrorCode::kResourceLimit, std::string(path),
             "JSON string exceeds max_string_bytes");
        return false;
      }
      return true;
    }
    if (const json::Value::Array* array = value.as_array(); array != nullptr) {
      if (array->size() > limits_.max_array_length) {
        fail(ReferenceOracleErrorCode::kResourceLimit, std::string(path),
             "JSON array exceeds max_array_length");
        return false;
      }
      for (const json::Value& item : *array) {
        if (!resource_walk(item, path)) {
          return false;
        }
      }
      return true;
    }
    if (const json::Value::Object* object = value.as_object();
        object != nullptr) {
      for (const auto& member : *object) {
        if (member.first.size() > limits_.max_string_bytes) {
          fail(ReferenceOracleErrorCode::kResourceLimit, std::string(path),
               "JSON object key exceeds max_string_bytes");
          return false;
        }
        if (!resource_walk(member.second, path)) {
          return false;
        }
      }
    }
    return true;
  }

  [[nodiscard]] bool load_json(const std::filesystem::path& path,
                               json::Value& output) {
    std::string bytes;
    if (!read_regular_file(path, bytes)) {
      return false;
    }
    json::ParseOptions options;
    options.max_input_bytes = limits_.max_file_bytes;
    options.max_nesting_depth = limits_.max_nesting_depth;
    options.max_values = limits_.max_json_nodes;
    options.max_container_items = limits_.max_container_items;
    json::ParseResult parsed = json::parse(bytes, options);
    if (!parsed) {
      const bool resource =
          parsed.error.code == json::ErrorCode::kNestingTooDeep ||
          parsed.error.code == json::ErrorCode::kTooManyValues ||
          parsed.error.code == json::ErrorCode::kTooManyContainerItems;
      fail(resource ? ReferenceOracleErrorCode::kResourceLimit
                    : ReferenceOracleErrorCode::kJsonRejected,
           path.string(), std::string(parsed.error.message()), {}, {}, 0,
           parsed.error.offset);
      return false;
    }
    if (!resource_walk(*parsed.value, "$")) {
      return false;
    }
    output = std::move(*parsed.value);
    return true;
  }

  [[nodiscard]] bool exact_keys(
      const json::Value& value,
      const std::initializer_list<std::string_view> keys,
      const std::string_view path) {
    const json::Value::Object* const object = value.as_object();
    if (object == nullptr || object->size() != keys.size()) {
      fail(ReferenceOracleErrorCode::kSchemaMismatch, std::string(path),
           "object has unexpected, missing, or wrongly typed fields");
      return false;
    }
    for (const std::string_view key : keys) {
      if (object->find(key) == object->end()) {
        fail(ReferenceOracleErrorCode::kSchemaMismatch, std::string(path),
             "object is missing required field", std::string(key));
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] const json::Value* member(const json::Value& object,
                                           const std::string_view key,
                                           const std::string_view path) {
    const json::Value* const value = object.find(key);
    if (value == nullptr) {
      fail(ReferenceOracleErrorCode::kSchemaMismatch,
           std::string(path) + "." + std::string(key),
           "required field is missing");
    }
    return value;
  }

  [[nodiscard]] bool string_value(const json::Value& object,
                                  const std::string_view key,
                                  const std::string_view path,
                                  std::string& output) {
    const json::Value* const value = member(object, key, path);
    const std::string* const text = value == nullptr ? nullptr : value->as_string();
    if (text == nullptr) {
      fail(ReferenceOracleErrorCode::kSchemaMismatch,
           std::string(path) + "." + std::string(key),
           "field must be a string");
      return false;
    }
    output = *text;
    return true;
  }

  [[nodiscard]] bool expect_string(const json::Value& object,
                                   const std::string_view key,
                                   const std::string_view expected,
                                   const std::string_view path) {
    std::string actual;
    if (!string_value(object, key, path, actual)) {
      return false;
    }
    if (actual != expected) {
      fail(ReferenceOracleErrorCode::kSchemaMismatch,
           std::string(path) + "." + std::string(key),
           "string does not match the pinned schema", std::string(expected),
           actual);
      return false;
    }
    return true;
  }

  [[nodiscard]] bool bool_value(const json::Value& object,
                                const std::string_view key,
                                const std::string_view path, bool& output) {
    const json::Value* const value = member(object, key, path);
    const bool* const boolean = value == nullptr ? nullptr : value->as_bool();
    if (boolean == nullptr) {
      fail(ReferenceOracleErrorCode::kSchemaMismatch,
           std::string(path) + "." + std::string(key),
           "field must be a boolean");
      return false;
    }
    output = *boolean;
    return true;
  }

  [[nodiscard]] bool expect_bool(const json::Value& object,
                                 const std::string_view key,
                                 const bool expected,
                                 const std::string_view path) {
    bool actual = false;
    if (!bool_value(object, key, path, actual)) {
      return false;
    }
    if (actual != expected) {
      fail(ReferenceOracleErrorCode::kSchemaMismatch,
           std::string(path) + "." + std::string(key),
           "boolean does not match the pinned schema",
           expected ? "true" : "false", actual ? "true" : "false");
      return false;
    }
    return true;
  }

  [[nodiscard]] bool uint64_value(const json::Value& object,
                                  const std::string_view key,
                                  const std::string_view path,
                                  std::uint64_t& output) {
    const json::Value* const value = member(object, key, path);
    const json::Number* const number =
        value == nullptr ? nullptr : value->as_number();
    if (number == nullptr || !number->to_uint64(output)) {
      fail(ReferenceOracleErrorCode::kSchemaMismatch,
           std::string(path) + "." + std::string(key),
           "field must be an unsigned integer");
      return false;
    }
    return true;
  }

  [[nodiscard]] bool double_value(const json::Value& object,
                                  const std::string_view key,
                                  const std::string_view path, double& output) {
    const json::Value* const value = member(object, key, path);
    const json::Number* const number =
        value == nullptr ? nullptr : value->as_number();
    if (number == nullptr || !number->to_double(output)) {
      fail(ReferenceOracleErrorCode::kSchemaMismatch,
           std::string(path) + "." + std::string(key),
           "field must be a number");
      return false;
    }
    if (!std::isfinite(output)) {
      fail(ReferenceOracleErrorCode::kNonFiniteValue,
           std::string(path) + "." + std::string(key),
           "numeric field is not finite");
      return false;
    }
    return true;
  }

  [[nodiscard]] const json::Value::Array* array_value(
      const json::Value& object, const std::string_view key,
      const std::string_view path) {
    const json::Value* const value = member(object, key, path);
    const json::Value::Array* const array =
        value == nullptr ? nullptr : value->as_array();
    if (array == nullptr) {
      fail(ReferenceOracleErrorCode::kSchemaMismatch,
           std::string(path) + "." + std::string(key),
           "field must be an array");
    }
    return array;
  }

  [[nodiscard]] bool validate_cache_policy(const json::Value& value,
                                           const std::string_view path,
                                           const bool includes_forced,
                                           CachePolicyReference& output) {
    if (!exact_keys(value,
                    includes_forced
                        ? std::initializer_list<std::string_view>{
                              "kv_cache_dtype", "mamba_cache_dtype",
                              "mamba_ssm_cache_dtype", "forced"}
                        : std::initializer_list<std::string_view>{
                              "kv_cache_dtype", "mamba_cache_dtype",
                              "mamba_ssm_cache_dtype"},
                    path) ||
        !expect_string(value, "kv_cache_dtype", kBf16, path) ||
        !expect_string(value, "mamba_cache_dtype", kBf16, path) ||
        !expect_string(value, "mamba_ssm_cache_dtype", kBf16, path) ||
        (includes_forced && !expect_bool(value, "forced", true, path))) {
      return false;
    }
    output.kv_cache_dtype = std::string(kBf16);
    output.mamba_cache_dtype = std::string(kBf16);
    output.mamba_ssm_cache_dtype = std::string(kBf16);
    return true;
  }

  [[nodiscard]] bool validate_string_fields(
      const json::Value& value,
      const std::initializer_list<std::string_view> fields,
      const std::string_view path) {
    for (const std::string_view field : fields) {
      std::string ignored;
      if (!string_value(value, field, path, ignored)) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool validate_source(const json::Value& value,
                                     GreedyReference& output) {
    if (!exact_keys(value, {"repository", "revision", "files"},
                    "$.source") ||
        !expect_string(value, "repository", kRepository, "$.source") ||
        !expect_string(value, "revision", kRevision, "$.source")) {
      return false;
    }
    output.source_repository = std::string(kRepository);
    output.source_revision = std::string(kRevision);
    const json::Value* const files = member(value, "files", "$.source");
    if (files == nullptr ||
        !exact_keys(*files,
                    {"model-00001-of-00003.safetensors",
                     "model-00002-of-00003.safetensors",
                     "model-00003-of-00003.safetensors", "tokenizer.json"},
                    "$.source.files")) {
      return false;
    }
    struct Identity {
      std::string_view filename;
      std::uint64_t size_bytes;
      std::string_view sha256;
    };
    constexpr std::array<Identity, 4U> kIdentities = {{
        {"model-00001-of-00003.safetensors", 9965652512ULL,
         "b4a0d9a57ff1859dac1144b53ca285011db072737d8813fc16d8d1e07ecae17d"},
        {"model-00002-of-00003.safetensors", 9985757032ULL,
         "06da4242b0f491118d19d4d4c7564307a7bd6059c6bed284e08c93f6fc5a556d"},
        {"model-00003-of-00003.safetensors", 1970287640ULL,
         "e90f5b2bb16814a0565de284ea179edec201edfb120d13f1debaab66f9e60845"},
        {"tokenizer.json", 12807982ULL,
         "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"},
    }};
    for (const Identity& identity : kIdentities) {
      const json::Value* const file =
          member(*files, identity.filename, "$.source.files");
      const std::string path =
          "$.source.files." + std::string(identity.filename);
      std::uint64_t size = 0U;
      std::string digest;
      if (file == nullptr || !exact_keys(*file, {"size_bytes", "sha256"}, path) ||
          !uint64_value(*file, "size_bytes", path, size) || size == 0U ||
          !string_value(*file, "sha256", path, digest) ||
          !lowercase_sha256(digest) || size != identity.size_bytes ||
          digest != identity.sha256) {
        if (diagnostic_.code == ReferenceOracleErrorCode::kNone) {
          fail(ReferenceOracleErrorCode::kSchemaMismatch, path,
               "source file identity is invalid");
        }
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool parse_token_ids(const json::Value& object,
                                     const std::string_view key,
                                     const std::size_t expected_size,
                                     const std::string_view path,
                                     std::vector<std::uint32_t>& output) {
    const json::Value::Array* const array = array_value(object, key, path);
    if (array == nullptr || array->size() != expected_size) {
      fail(ReferenceOracleErrorCode::kSchemaMismatch,
           std::string(path) + "." + std::string(key),
           "token-id array has the wrong length",
           std::to_string(expected_size),
           array == nullptr ? "not an array" : std::to_string(array->size()));
      return false;
    }
    output.clear();
    output.reserve(array->size());
    for (std::size_t index = 0U; index < array->size(); ++index) {
      const json::Number* const number = (*array)[index].as_number();
      std::uint64_t token = 0U;
      if (number == nullptr || !number->to_uint64(token) ||
          token >= kReferenceOracleVocabularySize) {
        fail(ReferenceOracleErrorCode::kSchemaMismatch,
             std::string(path) + "." + std::string(key) + "[" +
                 std::to_string(index) + "]",
             "token id is outside the pinned vocabulary");
        return false;
      }
      output.push_back(static_cast<std::uint32_t>(token));
    }
    return true;
  }

  [[nodiscard]] bool parse_greedy(const json::Value& root,
                                  GreedyReference& output) {
    if (!exact_keys(root,
                    {"schema_version", "fixture", "oracle", "engine",
                     "sampling", "repeatability", "comparison_policy",
                     "source", "prompt", "expected"},
                    "$")) {
      return false;
    }
    std::uint64_t schema = 0U;
    if (!uint64_value(root, "schema_version", "$", schema) || schema != 1U ||
        !expect_string(root, "fixture", kGreedyFixture, "$")) {
      if (diagnostic_.code == ReferenceOracleErrorCode::kNone) {
        fail(ReferenceOracleErrorCode::kSchemaMismatch, "$.schema_version",
             "only reference oracle schema_version 1 is supported", "1",
             std::to_string(schema));
      }
      return false;
    }
    output.fixture = std::string(kGreedyFixture);

    const json::Value* const oracle = member(root, "oracle", "$" );
    if (oracle == nullptr ||
        !exact_keys(*oracle,
                    {"device", "compute_capability", "python", "torch",
                     "cuda", "vllm", "vllm_revision", "transformers",
                     "safetensors", "tokenizers", "quantization_resolved",
                     "fp8_linear_backend", "nvfp4_linear_backend",
                     "sampling_backend", "environment", "cache_policy"},
                    "$.oracle") ||
        !validate_string_fields(
            *oracle,
            {"device", "compute_capability", "python", "torch", "cuda",
             "vllm", "vllm_revision", "transformers", "safetensors",
             "tokenizers", "quantization_resolved", "fp8_linear_backend",
             "nvfp4_linear_backend", "sampling_backend"},
            "$.oracle")) {
      return false;
    }
    const json::Value* const environment =
        member(*oracle, "environment", "$.oracle");
    if (environment == nullptr ||
        !exact_keys(*environment,
                    {"VLLM_ENABLE_V1_MULTIPROCESSING",
                     "VLLM_USE_FLASHINFER_SAMPLER"},
                    "$.oracle.environment") ||
        !validate_string_fields(
            *environment,
            {"VLLM_ENABLE_V1_MULTIPROCESSING",
             "VLLM_USE_FLASHINFER_SAMPLER"},
            "$.oracle.environment")) {
      return false;
    }
    const json::Value* const cache =
        member(*oracle, "cache_policy", "$.oracle");
    if (cache == nullptr ||
        !validate_cache_policy(*cache, "$.oracle.cache_policy", true,
                               output.cache_policy)) {
      return false;
    }

    if (!validate_greedy_engine(root) || !validate_sampling(root) ||
        !validate_repeatability(root) || !validate_comparison_policy(root)) {
      return false;
    }
    const json::Value* const source = member(root, "source", "$" );
    if (source == nullptr || !validate_source(*source, output)) {
      return false;
    }
    return parse_prompt(root, output) && parse_expected(root, output);
  }

  [[nodiscard]] bool validate_greedy_engine(const json::Value& root) {
    const json::Value* const engine = member(root, "engine", "$" );
    if (engine == nullptr ||
        !exact_keys(*engine,
                    {"language_model_only", "dtype", "kv_cache_dtype",
                     "max_model_len", "max_num_seqs",
                     "max_num_batched_tokens", "gpu_memory_utilization",
                     "enforce_eager", "enable_prefix_caching",
                     "mm_processor_cache_gb", "mamba_cache_dtype",
                     "mamba_ssm_cache_dtype"},
                    "$.engine") ||
        !expect_bool(*engine, "language_model_only", true, "$.engine") ||
        !expect_string(*engine, "dtype", kBf16, "$.engine") ||
        !expect_string(*engine, "kv_cache_dtype", kBf16, "$.engine") ||
        !expect_string(*engine, "mamba_cache_dtype", kBf16, "$.engine") ||
        !expect_string(*engine, "mamba_ssm_cache_dtype", kBf16, "$.engine")) {
      return false;
    }
    std::uint64_t max_length = 0U;
    std::uint64_t max_sequences = 0U;
    std::uint64_t max_tokens = 0U;
    std::uint64_t multimedia_cache = 0U;
    double utilization = 0.0;
    if (!uint64_value(*engine, "max_model_len", "$.engine", max_length) ||
        !uint64_value(*engine, "max_num_seqs", "$.engine", max_sequences) ||
        !uint64_value(*engine, "max_num_batched_tokens", "$.engine",
                      max_tokens) ||
        !double_value(*engine, "gpu_memory_utilization", "$.engine",
                      utilization) ||
        !expect_bool(*engine, "enforce_eager", true, "$.engine") ||
        !expect_bool(*engine, "enable_prefix_caching", false, "$.engine") ||
        !uint64_value(*engine, "mm_processor_cache_gb", "$.engine",
                      multimedia_cache)) {
      return false;
    }
    if (max_length != 128U || max_sequences != 1U || max_tokens != 128U ||
        utilization != 0.8 || multimedia_cache != 0U) {
      fail(ReferenceOracleErrorCode::kSchemaMismatch, "$.engine",
           "engine settings do not match the pinned BF16 oracle policy");
      return false;
    }
    return true;
  }

  [[nodiscard]] bool validate_sampling(const json::Value& root) {
    const json::Value* const value = member(root, "sampling", "$" );
    if (value == nullptr ||
        !exact_keys(*value,
                    {"temperature", "max_tokens", "logprobs", "seed"},
                    "$.sampling")) {
      return false;
    }
    double temperature = 0.0;
    std::uint64_t integer = 0U;
    if (!double_value(*value, "temperature", "$.sampling", temperature) ||
        temperature != 0.0 ||
        !uint64_value(*value, "max_tokens", "$.sampling", integer) ||
        integer != 32U ||
        !uint64_value(*value, "logprobs", "$.sampling", integer) ||
        integer != 5U ||
        !uint64_value(*value, "seed", "$.sampling", integer) ||
        integer != 0U) {
      if (diagnostic_.code == ReferenceOracleErrorCode::kNone) {
        fail(ReferenceOracleErrorCode::kSchemaMismatch, "$.sampling",
             "fixture is not the pinned deterministic greedy policy");
      }
      return false;
    }
    return true;
  }

  [[nodiscard]] bool validate_repeatability(const json::Value& root) {
    const json::Value* const value = member(root, "repeatability", "$" );
    if (value == nullptr ||
        !exact_keys(*value,
                    {"runs", "single_run_policy_capture",
                     "token_ids_match_primary_oracle",
                     "text_matches_primary_oracle", "run_seconds"},
                    "$.repeatability")) {
      return false;
    }
    std::uint64_t runs = 0U;
    if (!uint64_value(*value, "runs", "$.repeatability", runs) ||
        !expect_bool(*value, "single_run_policy_capture", true,
                     "$.repeatability") ||
        !expect_bool(*value, "token_ids_match_primary_oracle", true,
                     "$.repeatability") ||
        !expect_bool(*value, "text_matches_primary_oracle", true,
                     "$.repeatability")) {
      return false;
    }
    if (runs != 1U) {
      fail(ReferenceOracleErrorCode::kSchemaMismatch,
           "$.repeatability.runs",
           "BF16 oracle fixture must contain exactly one captured run", "1",
           std::to_string(runs));
      return false;
    }
    const json::Value::Array* const seconds =
        array_value(*value, "run_seconds", "$.repeatability");
    if (seconds == nullptr || seconds->size() != runs) {
      fail(ReferenceOracleErrorCode::kSchemaMismatch,
           "$.repeatability.run_seconds",
           "run_seconds length does not match runs");
      return false;
    }
    for (const json::Value& item : *seconds) {
      double elapsed = 0.0;
      const json::Number* const number = item.as_number();
      if (number == nullptr || !number->to_double(elapsed) ||
          !std::isfinite(elapsed) || elapsed < 0.0) {
        fail(ReferenceOracleErrorCode::kNonFiniteValue,
             "$.repeatability.run_seconds",
             "run duration must be finite and non-negative");
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool validate_comparison_policy(const json::Value& root) {
    const json::Value* const value =
        member(root, "comparison_policy", "$" );
    if (value == nullptr ||
        !exact_keys(*value,
                    {"token_ids", "text",
                     "chosen_logprobs_same_oracle_absolute_tolerance",
                     "chosen_logprobs_cross_runtime_diagnostic_tolerance",
                     "timings_are_normative"},
                    "$.comparison_policy") ||
        !expect_string(*value, "token_ids", "exact",
                       "$.comparison_policy") ||
        !expect_string(*value, "text", "exact UTF-8",
                       "$.comparison_policy") ||
        !expect_bool(*value, "timings_are_normative", false,
                     "$.comparison_policy")) {
      return false;
    }
    double same_oracle_tolerance = 0.0;
    double cross_runtime_tolerance = 0.0;
    if (!double_value(
            *value, "chosen_logprobs_same_oracle_absolute_tolerance",
            "$.comparison_policy", same_oracle_tolerance) ||
        !double_value(
            *value,
            "chosen_logprobs_cross_runtime_diagnostic_tolerance",
            "$.comparison_policy", cross_runtime_tolerance)) {
      return false;
    }
    if (same_oracle_tolerance < 0.0 || cross_runtime_tolerance < 0.0) {
      fail(ReferenceOracleErrorCode::kSchemaMismatch,
           "$.comparison_policy",
           "comparison tolerances must be non-negative");
      return false;
    }
    return true;
  }

  [[nodiscard]] bool parse_prompt(const json::Value& root,
                                  GreedyReference& output) {
    const json::Value* const prompt = member(root, "prompt", "$" );
    if (prompt == nullptr ||
        !exact_keys(*prompt,
                    {"user_text", "enable_thinking", "rendered",
                     "token_ids"},
                    "$.prompt") ||
        !expect_bool(*prompt, "enable_thinking", false, "$.prompt")) {
      return false;
    }
    std::string ignored;
    return string_value(*prompt, "user_text", "$.prompt", ignored) &&
           string_value(*prompt, "rendered", "$.prompt", ignored) &&
           parse_token_ids(*prompt, "token_ids", 19U, "$.prompt",
                           output.prompt_token_ids);
  }

  [[nodiscard]] bool parse_expected(const json::Value& root,
                                    GreedyReference& output) {
    const json::Value* const expected = member(root, "expected", "$" );
    if (expected == nullptr ||
        !exact_keys(*expected,
                    {"token_ids", "text", "finish_reason", "stop_token_id",
                     "chosen_logprobs"},
                    "$.expected") ||
        !parse_token_ids(*expected, "token_ids", 26U, "$.expected",
                         output.expected_token_ids) ||
        !string_value(*expected, "text", "$.expected", output.expected_text) ||
        !expect_string(*expected, "finish_reason", "stop", "$.expected")) {
      return false;
    }
    output.finish_reason = "stop";
    std::uint64_t stop = 0U;
    if (!uint64_value(*expected, "stop_token_id", "$.expected", stop) ||
        stop >= kReferenceOracleVocabularySize ||
        output.expected_token_ids.empty() ||
        stop != output.expected_token_ids.back()) {
      fail(ReferenceOracleErrorCode::kSchemaMismatch,
           "$.expected.stop_token_id",
           "stop token must equal the final expected token id");
      return false;
    }
    output.stop_token_id = static_cast<std::uint32_t>(stop);
    const json::Value::Array* const logprobs =
        array_value(*expected, "chosen_logprobs", "$.expected");
    if (logprobs == nullptr ||
        logprobs->size() != output.expected_token_ids.size()) {
      fail(ReferenceOracleErrorCode::kSchemaMismatch,
           "$.expected.chosen_logprobs",
           "chosen_logprobs length must equal expected token count");
      return false;
    }
    output.chosen_logprobs.clear();
    output.chosen_logprobs.reserve(logprobs->size());
    for (std::size_t index = 0U; index < logprobs->size(); ++index) {
      double value = 0.0;
      const json::Number* const number = (*logprobs)[index].as_number();
      if (number == nullptr || !number->to_double(value) ||
          !std::isfinite(value) || value > 0.0) {
        fail(ReferenceOracleErrorCode::kNonFiniteValue,
             "$.expected.chosen_logprobs[" + std::to_string(index) + "]",
             "chosen logprob must be finite and non-positive");
        return false;
      }
      output.chosen_logprobs.push_back(value);
    }
    return true;
  }

  [[nodiscard]] bool parse_boundary(const json::Value& value,
                                    const std::string& path,
                                    BoundarySummary& output) {
    if (!exact_keys(value,
                    {"dtype", "length", "sha256_raw", "mean", "rms",
                     "minimum", "maximum", "samples"},
                    path) ||
        !expect_string(value, "dtype", kBf16, path)) {
      return false;
    }
    output.dtype = std::string(kBf16);
    std::uint64_t length = 0U;
    if (!uint64_value(value, "length", path, length) ||
        length != kReferenceOracleHiddenSize) {
      fail(ReferenceOracleErrorCode::kSchemaMismatch, path + ".length",
           "boundary length must be 5120", "5120", std::to_string(length));
      return false;
    }
    output.length = static_cast<std::size_t>(length);
    if (!string_value(value, "sha256_raw", path, output.sha256_raw) ||
        !lowercase_sha256(output.sha256_raw) ||
        !double_value(value, "mean", path, output.mean) ||
        !double_value(value, "rms", path, output.rms) ||
        !double_value(value, "minimum", path, output.minimum) ||
        !double_value(value, "maximum", path, output.maximum) ||
        output.rms < 0.0 || output.minimum > output.maximum) {
      if (diagnostic_.code == ReferenceOracleErrorCode::kNone) {
        fail(ReferenceOracleErrorCode::kSchemaMismatch, path,
             "boundary hash or finite statistics are invalid");
      }
      return false;
    }
    const json::Value::Array* const samples =
        array_value(value, "samples", path);
    if (samples == nullptr || samples->size() != kReferenceOracleSampleCount) {
      fail(ReferenceOracleErrorCode::kSchemaMismatch, path + ".samples",
           "boundary must contain exactly 16 diagnostic samples");
      return false;
    }
    output.samples.clear();
    output.samples.reserve(samples->size());
    for (std::size_t index = 0U; index < samples->size(); ++index) {
      double sample = 0.0;
      const json::Number* const number = (*samples)[index].as_number();
      if (number == nullptr || !number->to_double(sample) ||
          !std::isfinite(sample)) {
        fail(ReferenceOracleErrorCode::kNonFiniteValue,
             path + ".samples[" + std::to_string(index) + "]",
             "boundary sample must be finite");
        return false;
      }
      output.samples.push_back(sample);
    }
    return true;
  }

  [[nodiscard]] bool parse_logits(const json::Value& value,
                                  const std::string& path,
                                  const std::uint32_t predicted_token,
                                  LogitsReference& output) {
    if (!exact_keys(value,
                    {"dtype", "length", "logsumexp", "top20",
                     "chosen_logprob"},
                    path) ||
        !expect_string(value, "dtype", kBf16, path)) {
      return false;
    }
    output.dtype = std::string(kBf16);
    std::uint64_t length = 0U;
    if (!uint64_value(value, "length", path, length) ||
        length != kReferenceOracleVocabularySize ||
        !double_value(value, "logsumexp", path, output.logsumexp) ||
        !double_value(value, "chosen_logprob", path,
                      output.chosen_logprob) ||
        output.chosen_logprob > 0.0) {
      if (diagnostic_.code == ReferenceOracleErrorCode::kNone) {
        fail(ReferenceOracleErrorCode::kSchemaMismatch, path,
             "logits dimensions or statistics are invalid");
      }
      return false;
    }
    output.length = static_cast<std::size_t>(length);
    const json::Value::Array* const top20 =
        array_value(value, "top20", path);
    if (top20 == nullptr || top20->size() != output.top20.size()) {
      fail(ReferenceOracleErrorCode::kSchemaMismatch, path + ".top20",
           "logits must contain exactly 20 top entries");
      return false;
    }
    std::set<std::uint32_t> seen;
    double previous = std::numeric_limits<double>::infinity();
    for (std::size_t index = 0U; index < top20->size(); ++index) {
      const json::Value& item = (*top20)[index];
      const std::string item_path =
          path + ".top20[" + std::to_string(index) + "]";
      std::uint64_t token = 0U;
      double logit = 0.0;
      if (!exact_keys(item, {"token_id", "logit"}, item_path) ||
          !uint64_value(item, "token_id", item_path, token) ||
          token >= kReferenceOracleVocabularySize ||
          !double_value(item, "logit", item_path, logit) ||
          logit > previous ||
          !seen.emplace(static_cast<std::uint32_t>(token)).second) {
        if (diagnostic_.code == ReferenceOracleErrorCode::kNone) {
          fail(ReferenceOracleErrorCode::kSchemaMismatch, item_path,
               "top-logit entry is invalid, duplicated, or out of order");
        }
        return false;
      }
      previous = logit;
      output.top20[index] =
          TopLogitReference{static_cast<std::uint32_t>(token), logit};
    }
    if (output.top20.front().token_id != predicted_token) {
      fail(ReferenceOracleErrorCode::kSchemaMismatch, path + ".top20[0]",
           "highest-logit token does not equal predicted_token_id");
      return false;
    }
    return true;
  }

  [[nodiscard]] bool parse_layers(const json::Value& root,
                                  LayerReference& output) {
    if (!exact_keys(root,
                    {"schema_version", "fixture", "oracle",
                     "source_greedy_fixture", "source_revision", "semantics",
                     "phases"},
                    "$layers")) {
      return false;
    }
    std::uint64_t schema = 0U;
    if (!uint64_value(root, "schema_version", "$layers", schema) ||
        schema != 1U ||
        !expect_string(root, "fixture", kLayerFixture, "$layers") ||
        !string_value(root, "source_greedy_fixture", "$layers",
                      output.source_greedy_fixture) ||
        !string_value(root, "source_revision", "$layers",
                      output.source_revision)) {
      if (diagnostic_.code == ReferenceOracleErrorCode::kNone) {
        fail(ReferenceOracleErrorCode::kSchemaMismatch,
             "$layers.schema_version",
             "only layer oracle schema_version 1 is supported");
      }
      return false;
    }
    output.fixture = std::string(kLayerFixture);
    if (!parse_layer_oracle_metadata(root, output) ||
        !parse_semantics(root, output)) {
      return false;
    }
    const json::Value::Array* const phases =
        array_value(root, "phases", "$layers");
    if (phases == nullptr || phases->size() != output.phases.size()) {
      fail(ReferenceOracleErrorCode::kSchemaMismatch, "$layers.phases",
           "layer oracle must contain exactly two phases");
      return false;
    }
    for (std::size_t index = 0U; index < phases->size(); ++index) {
      if (!parse_phase((*phases)[index], index, output.phases[index])) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool parse_layer_oracle_metadata(const json::Value& root,
                                                 LayerReference& output) {
    const json::Value* const oracle = member(root, "oracle", "$layers");
    if (oracle == nullptr ||
        !exact_keys(*oracle,
                    {"vllm", "device", "compute_capability",
                     "engine_init_seconds", "generation_seconds",
                     "cache_options"},
                    "$layers.oracle") ||
        !validate_string_fields(*oracle, {"vllm", "device"},
                                "$layers.oracle")) {
      return false;
    }
    double initialization_seconds = 0.0;
    double generation_seconds = 0.0;
    if (!double_value(*oracle, "engine_init_seconds", "$layers.oracle",
                      initialization_seconds) ||
        !double_value(*oracle, "generation_seconds", "$layers.oracle",
                      generation_seconds)) {
      return false;
    }
    if (initialization_seconds < 0.0 || generation_seconds < 0.0) {
      fail(ReferenceOracleErrorCode::kSchemaMismatch, "$layers.oracle",
           "oracle timings must be finite and non-negative");
      return false;
    }
    const json::Value::Array* const capability =
        array_value(*oracle, "compute_capability", "$layers.oracle");
    if (capability == nullptr || capability->size() != 2U) {
      fail(ReferenceOracleErrorCode::kSchemaMismatch,
           "$layers.oracle.compute_capability",
           "compute capability must be [8,7]");
      return false;
    }
    for (std::size_t index = 0U; index < 2U; ++index) {
      const json::Number* const number = (*capability)[index].as_number();
      std::uint64_t value = 0U;
      if (number == nullptr || !number->to_uint64(value) ||
          value != (index == 0U ? 8U : 7U)) {
        fail(ReferenceOracleErrorCode::kSchemaMismatch,
             "$layers.oracle.compute_capability",
             "compute capability must be [8,7]");
        return false;
      }
    }
    const json::Value* const cache =
        member(*oracle, "cache_options", "$layers.oracle");
    return cache != nullptr &&
           validate_cache_policy(*cache, "$layers.oracle.cache_options",
                                 false, output.cache_policy);
  }

  [[nodiscard]] bool parse_semantics(const json::Value& root,
                                     LayerReference& output) {
    const json::Value* const semantics =
        member(root, "semantics", "$layers");
    if (semantics == nullptr ||
        !exact_keys(*semantics,
                    {"layer_hidden", "layer_residual", "hash", "samples",
                     "timings_are_normative"},
                    "$layers.semantics") ||
        !validate_string_fields(*semantics,
                                {"layer_hidden", "layer_residual", "hash"},
                                "$layers.semantics") ||
        !expect_bool(*semantics, "timings_are_normative", false,
                     "$layers.semantics")) {
      return false;
    }
    const json::Value::Array* const samples =
        array_value(*semantics, "samples", "$layers.semantics");
    if (samples == nullptr || samples->size() != kPinnedSampleIndices.size()) {
      fail(ReferenceOracleErrorCode::kSchemaMismatch,
           "$layers.semantics.samples",
           "sample index list must contain exactly 16 entries");
      return false;
    }
    for (std::size_t index = 0U; index < samples->size(); ++index) {
      const json::Number* const number = (*samples)[index].as_number();
      std::uint64_t value = 0U;
      if (number == nullptr || !number->to_uint64(value) ||
          value != kPinnedSampleIndices[index]) {
        fail(ReferenceOracleErrorCode::kSchemaMismatch,
             "$layers.semantics.samples[" + std::to_string(index) + "]",
             "sample index does not match the canonical position",
             std::to_string(kPinnedSampleIndices[index]),
             std::to_string(value));
        return false;
      }
      output.sample_indices[index] = static_cast<std::size_t>(value);
    }
    return true;
  }

  [[nodiscard]] bool parse_phase(const json::Value& value,
                                 const std::size_t phase_index,
                                 PhaseReference& output) {
    const std::string path =
        "$layers.phases[" + std::to_string(phase_index) + "]";
    if (!exact_keys(value,
                    {"name", "position", "input_token_id",
                     "predicted_token_id", "embedding", "layers",
                     "final_norm", "logits"},
                    path)) {
      return false;
    }
    const std::string_view expected_name =
        phase_index == 0U ? "prefill_last_prompt_token"
                          : "decode_after_token_77517";
    const std::uint64_t expected_position = phase_index == 0U ? 18U : 19U;
    const std::uint64_t expected_input = phase_index == 0U ? 271U : 77517U;
    const std::uint64_t expected_prediction = phase_index == 0U ? 77517U : 220U;
    std::uint64_t position = 0U;
    std::uint64_t input = 0U;
    std::uint64_t prediction = 0U;
    if (!expect_string(value, "name", expected_name, path) ||
        !uint64_value(value, "position", path, position) ||
        !uint64_value(value, "input_token_id", path, input) ||
        !uint64_value(value, "predicted_token_id", path, prediction) ||
        position != expected_position || input != expected_input ||
        prediction != expected_prediction) {
      if (diagnostic_.code == ReferenceOracleErrorCode::kNone) {
        fail(ReferenceOracleErrorCode::kSchemaMismatch, path,
             "phase identity does not match the pinned two-step oracle");
      }
      return false;
    }
    output.name = std::string(expected_name);
    output.position = static_cast<std::size_t>(position);
    output.input_token_id = static_cast<std::uint32_t>(input);
    output.predicted_token_id = static_cast<std::uint32_t>(prediction);
    const json::Value* const embedding = member(value, "embedding", path);
    const json::Value* const final_norm = member(value, "final_norm", path);
    const json::Value* const logits = member(value, "logits", path);
    if (embedding == nullptr || final_norm == nullptr || logits == nullptr ||
        !parse_boundary(*embedding, path + ".embedding", output.embedding) ||
        !parse_boundary(*final_norm, path + ".final_norm",
                        output.final_norm) ||
        !parse_logits(*logits, path + ".logits", output.predicted_token_id,
                      output.logits)) {
      return false;
    }
    const json::Value::Array* const layers =
        array_value(value, "layers", path);
    if (layers == nullptr || layers->size() != output.layers.size()) {
      fail(ReferenceOracleErrorCode::kSchemaMismatch, path + ".layers",
           "phase must contain exactly 64 layer boundaries");
      return false;
    }
    for (std::size_t index = 0U; index < layers->size(); ++index) {
      const json::Value& layer = (*layers)[index];
      const std::string layer_path =
          path + ".layers[" + std::to_string(index) + "]";
      std::uint64_t parsed_index = 0U;
      if (!exact_keys(layer, {"index", "hidden", "residual"}, layer_path) ||
          !uint64_value(layer, "index", layer_path, parsed_index) ||
          parsed_index != index) {
        fail(ReferenceOracleErrorCode::kSchemaMismatch,
             layer_path + ".index",
             "layer indices must be contiguous from 0 through 63");
        return false;
      }
      const json::Value* const hidden = member(layer, "hidden", layer_path);
      const json::Value* const residual =
          member(layer, "residual", layer_path);
      output.layers[index].index = index;
      if (hidden == nullptr || residual == nullptr ||
          !parse_boundary(*hidden, layer_path + ".hidden",
                          output.layers[index].hidden) ||
          !parse_boundary(*residual, layer_path + ".residual",
                          output.layers[index].residual)) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool cross_validate(const ReferenceOracle& oracle) {
    const std::string greedy_filename = greedy_path_.filename().string();
    if (oracle.layers.source_greedy_fixture != greedy_filename) {
      fail(ReferenceOracleErrorCode::kCrossFileMismatch,
           "$layers.source_greedy_fixture",
           "layer fixture names a different greedy source file",
           greedy_filename, oracle.layers.source_greedy_fixture);
      return false;
    }
    if (oracle.greedy.source_revision != oracle.layers.source_revision ||
        oracle.layers.source_revision != kRevision) {
      fail(ReferenceOracleErrorCode::kCrossFileMismatch,
           "$layers.source_revision",
           "greedy and layer fixtures use different source revisions",
           oracle.greedy.source_revision, oracle.layers.source_revision);
      return false;
    }
    if (oracle.greedy.expected_token_ids.size() < 2U ||
        oracle.layers.phases[0].predicted_token_id !=
            oracle.greedy.expected_token_ids[0] ||
        oracle.layers.phases[1].predicted_token_id !=
            oracle.greedy.expected_token_ids[1] ||
        oracle.greedy.prompt_token_ids.empty() ||
        oracle.layers.phases[0].input_token_id !=
            oracle.greedy.prompt_token_ids.back() ||
        oracle.layers.phases[1].input_token_id !=
            oracle.greedy.expected_token_ids[0]) {
      fail(ReferenceOracleErrorCode::kCrossFileMismatch, "$layers.phases",
           "phase inputs or predictions disagree with greedy token sequence");
      return false;
    }
    return true;
  }

  const std::filesystem::path& greedy_path_;
  const std::filesystem::path& layers_path_;
  const ReferenceOracleLimits& limits_;
  ReferenceOracleDiagnostic diagnostic_;
};

[[nodiscard]] ReferenceOracleDiagnostic diagnostic(
    const ReferenceOracleErrorCode code, std::string path,
    std::string message) {
  ReferenceOracleDiagnostic value;
  value.code = code;
  value.path = std::move(path);
  value.message = std::move(message);
  return value;
}

[[nodiscard]] bool valid_sample_indices(
    const std::vector<std::size_t>& indices, const std::size_t length) noexcept {
  if (indices.empty()) {
    return false;
  }
  for (std::size_t index = 0U; index < indices.size(); ++index) {
    if (indices[index] >= length ||
        (index != 0U && indices[index] <= indices[index - 1U])) {
      return false;
    }
  }
  return true;
}

}  // namespace

ReferenceOracleResult load_reference_oracle(
    const std::filesystem::path& greedy_path,
    const std::filesystem::path& layers_path,
    const ReferenceOracleLimits& limits) {
  try {
    return OracleLoader(greedy_path, layers_path, limits).run();
  } catch (const std::bad_alloc&) {
    ReferenceOracleResult result;
    result.diagnostic = diagnostic(
        ReferenceOracleErrorCode::kAllocationFailure, "$",
        "allocation failed while loading reference oracle fixtures");
    return result;
  }
}

BoundarySummaryResult summarize_bf16_span(
    const std::uint16_t* const values, const std::size_t length,
    const std::vector<std::size_t>& sample_indices) {
  BoundarySummaryResult result;
  if (values == nullptr || length == 0U) {
    result.diagnostic = diagnostic(ReferenceOracleErrorCode::kInvalidArgument,
                                   "$actual", "BF16 span is empty or null");
    return result;
  }
  if (!valid_sample_indices(sample_indices, length)) {
    result.diagnostic = diagnostic(
        ReferenceOracleErrorCode::kInvalidSampleIndex, "$sample_indices",
        "sample indices must be non-empty, strictly increasing, and in range");
    return result;
  }
  if (length > std::numeric_limits<std::size_t>::max() / sizeof(*values)) {
    result.diagnostic = diagnostic(
        ReferenceOracleErrorCode::kArithmeticOverflow, "$actual",
        "BF16 span byte count overflows size_t");
    return result;
  }

  BoundarySummary summary;
  summary.dtype = std::string(kBf16);
  summary.length = length;
  summary.minimum = std::numeric_limits<double>::infinity();
  summary.maximum = -std::numeric_limits<double>::infinity();
  double sum = 0.0;
  double sum_squares = 0.0;
  q3x::core::Sha256 hash;
  std::array<std::uint8_t, 4096U> bytes{};
  std::size_t buffered = 0U;
  for (std::size_t index = 0U; index < length; ++index) {
    const float decoded = decode_bf16(values[index]);
    if (!std::isfinite(decoded)) {
      result.diagnostic = diagnostic(
          ReferenceOracleErrorCode::kNonFiniteValue,
          "$actual[" + std::to_string(index) + "]",
          "BF16 span contains NaN or infinity");
      return result;
    }
    const double value = decoded;
    sum += value;
    sum_squares += value * value;
    summary.minimum = std::min(summary.minimum, value);
    summary.maximum = std::max(summary.maximum, value);
    bytes[buffered++] = static_cast<std::uint8_t>(values[index] & 0xffU);
    bytes[buffered++] = static_cast<std::uint8_t>(values[index] >> 8U);
    if (buffered == bytes.size()) {
      if (!hash.update(bytes.data(), buffered)) {
        result.diagnostic = diagnostic(
            ReferenceOracleErrorCode::kArithmeticOverflow, "$actual",
            "SHA-256 input length overflowed");
        return result;
      }
      buffered = 0U;
    }
  }
  if (buffered != 0U && !hash.update(bytes.data(), buffered)) {
    result.diagnostic = diagnostic(
        ReferenceOracleErrorCode::kArithmeticOverflow, "$actual",
        "SHA-256 input length overflowed");
    return result;
  }
  summary.sha256_raw = hash.finalize().hex();
  summary.mean = sum / static_cast<double>(length);
  summary.rms = std::sqrt(sum_squares / static_cast<double>(length));
  summary.samples.reserve(sample_indices.size());
  for (const std::size_t index : sample_indices) {
    summary.samples.push_back(decode_bf16(values[index]));
  }
  result.value.emplace(std::move(summary));
  return result;
}

BoundaryComparisonResult compare_boundary_samples(
    const std::uint16_t* const actual, const std::size_t actual_length,
    const BoundarySummary& expected,
    const std::vector<std::size_t>& sample_indices,
    const double absolute_tolerance, const double relative_tolerance) {
  BoundaryComparisonResult result;
  if (actual == nullptr || actual_length == 0U ||
      expected.dtype != kBf16 || expected.length != actual_length ||
      expected.samples.size() != sample_indices.size() ||
      !valid_sample_indices(sample_indices, actual_length) ||
      !std::isfinite(absolute_tolerance) || absolute_tolerance < 0.0 ||
      !std::isfinite(relative_tolerance) || relative_tolerance < 0.0) {
    result.diagnostic = diagnostic(
        ReferenceOracleErrorCode::kInvalidArgument, "$comparison",
        "comparison span, summary, indices, or tolerances are invalid");
    return result;
  }
  for (std::size_t index = 0U; index < actual_length; ++index) {
    if (!std::isfinite(decode_bf16(actual[index]))) {
      result.diagnostic = diagnostic(
          ReferenceOracleErrorCode::kNonFiniteValue,
          "$actual[" + std::to_string(index) + "]",
          "comparison span contains NaN or infinity");
      return result;
    }
  }
  result.matches = true;
  for (std::size_t ordinal = 0U; ordinal < sample_indices.size(); ++ordinal) {
    const double expected_value = expected.samples[ordinal];
    const double actual_value = decode_bf16(actual[sample_indices[ordinal]]);
    if (!std::isfinite(expected_value)) {
      result.diagnostic = diagnostic(
          ReferenceOracleErrorCode::kNonFiniteValue,
          "$expected.samples[" + std::to_string(ordinal) + "]",
          "expected sample is NaN or infinity");
      result.matches = false;
      return result;
    }
    const double tolerance =
        absolute_tolerance + relative_tolerance * std::fabs(expected_value);
    if (!std::isfinite(tolerance)) {
      result.diagnostic = diagnostic(
          ReferenceOracleErrorCode::kInvalidArgument, "$comparison",
          "combined comparison tolerance is not finite");
      result.matches = false;
      return result;
    }
    if (std::fabs(actual_value - expected_value) > tolerance) {
      result.matches = false;
      result.first_mismatch = BoundarySampleMismatch{
          ordinal, sample_indices[ordinal], expected_value, actual_value,
          tolerance};
      return result;
    }
  }
  return result;
}

std::string_view to_string(const ReferenceOracleErrorCode code) noexcept {
  switch (code) {
    case ReferenceOracleErrorCode::kNone:
      return "none";
    case ReferenceOracleErrorCode::kInvalidArgument:
      return "invalid argument";
    case ReferenceOracleErrorCode::kInvalidLimit:
      return "invalid limit";
    case ReferenceOracleErrorCode::kOpenFailed:
      return "open failed";
    case ReferenceOracleErrorCode::kNotRegularFile:
      return "not a regular file";
    case ReferenceOracleErrorCode::kFileTooLarge:
      return "file too large";
    case ReferenceOracleErrorCode::kIoFailure:
      return "I/O failure";
    case ReferenceOracleErrorCode::kJsonRejected:
      return "JSON rejected";
    case ReferenceOracleErrorCode::kResourceLimit:
      return "resource limit";
    case ReferenceOracleErrorCode::kSchemaMismatch:
      return "schema mismatch";
    case ReferenceOracleErrorCode::kCrossFileMismatch:
      return "cross-file mismatch";
    case ReferenceOracleErrorCode::kNonFiniteValue:
      return "non-finite value";
    case ReferenceOracleErrorCode::kInvalidSampleIndex:
      return "invalid sample index";
    case ReferenceOracleErrorCode::kArithmeticOverflow:
      return "arithmetic overflow";
    case ReferenceOracleErrorCode::kAllocationFailure:
      return "allocation failure";
  }
  return "unknown";
}

}  // namespace q3x::runtime
