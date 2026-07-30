#include "q3x/runtime/prefill_a4_sidecar_converter.h"

#include "q3x/core/sha256.h"
#include "q3x/io/json.h"
#include "q3x/model/weight_manifest.h"
#include "q3x/quantization/nvfp4.h"
#include "q3x/runtime/resident_weights.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <new>
#include <sstream>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <sys/file.h>
#include <sys/types.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace q3x::runtime {
namespace {

namespace fs = std::filesystem;
namespace json = q3x::io::json;
namespace mw = q3x::model::weights;

constexpr std::string_view kPolicySchema =
    "q3x.prefill.a4.calibration-policy";
constexpr std::string_view kReceiptSchema =
    "q3x.prefill.a4.publication-receipt";
constexpr std::string_view kEqualizationScheme =
    "input_channel_multiply_f32_v1";

[[nodiscard]] PrefillA4ConverterDiagnostic make_diagnostic(
    const PrefillA4ConverterErrorCode code, std::string context,
    std::string message, std::string expected = {}, std::string actual = {},
    const int system_error = 0) {
  PrefillA4ConverterDiagnostic diagnostic;
  diagnostic.code = code;
  diagnostic.context = std::move(context);
  diagnostic.message = std::move(message);
  diagnostic.expected = std::move(expected);
  diagnostic.actual = std::move(actual);
  diagnostic.system_error = system_error;
  return diagnostic;
}

[[nodiscard]] bool lowercase_sha256(const std::string_view digest) noexcept {
  if (digest.size() != 64U) {
    return false;
  }
  return std::all_of(digest.begin(), digest.end(), [](const char value) {
    return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
  });
}

[[nodiscard]] bool checked_multiply(const std::uint64_t left,
                                    const std::uint64_t right,
                                    std::uint64_t& output) noexcept {
  if (left != 0U &&
      right > std::numeric_limits<std::uint64_t>::max() / left) {
    return false;
  }
  output = left * right;
  return true;
}

[[nodiscard]] bool checked_add(const std::uint64_t left,
                               const std::uint64_t right,
                               std::uint64_t& output) noexcept {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    return false;
  }
  output = left + right;
  return true;
}

class UniqueFd {
 public:
  UniqueFd() noexcept = default;
  explicit UniqueFd(const int fd) noexcept : fd_(fd) {}
  ~UniqueFd() {
    if (fd_ >= 0) {
      (void)::close(fd_);
    }
  }
  UniqueFd(const UniqueFd&) = delete;
  UniqueFd& operator=(const UniqueFd&) = delete;
  UniqueFd(UniqueFd&& other) noexcept : fd_(other.fd_) { other.fd_ = -1; }
  UniqueFd& operator=(UniqueFd&& other) noexcept {
    if (this != &other) {
      if (fd_ >= 0) {
        (void)::close(fd_);
      }
      fd_ = other.fd_;
      other.fd_ = -1;
    }
    return *this;
  }
  [[nodiscard]] int get() const noexcept { return fd_; }
  [[nodiscard]] int release() noexcept {
    const int fd = fd_;
    fd_ = -1;
    return fd;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return fd_ >= 0; }

 private:
  int fd_ = -1;
};

[[nodiscard]] bool offset_fits(const std::uint64_t offset) noexcept {
  return offset <= static_cast<std::uint64_t>(
                       std::numeric_limits<off_t>::max());
}

[[nodiscard]] bool pread_exact(const int fd, void* const destination,
                               const std::size_t bytes,
                               const std::uint64_t offset,
                               int& error) noexcept {
  if (!offset_fits(offset)) {
    error = EOVERFLOW;
    return false;
  }
  auto* output = static_cast<std::uint8_t*>(destination);
  std::size_t completed = 0U;
  while (completed < bytes) {
    const std::uint64_t position = offset + completed;
    if (position < offset || !offset_fits(position)) {
      error = EOVERFLOW;
      return false;
    }
    const ssize_t count = ::pread(fd, output + completed, bytes - completed,
                                  static_cast<off_t>(position));
    if (count < 0) {
      if (errno == EINTR) {
        continue;
      }
      error = errno;
      return false;
    }
    if (count == 0) {
      error = EIO;
      return false;
    }
    completed += static_cast<std::size_t>(count);
  }
  return true;
}

[[nodiscard]] bool pwrite_exact(const int fd, const void* const source,
                                const std::size_t bytes,
                                const std::uint64_t offset,
                                int& error) noexcept {
  if (!offset_fits(offset)) {
    error = EOVERFLOW;
    return false;
  }
  const auto* input = static_cast<const std::uint8_t*>(source);
  std::size_t completed = 0U;
  while (completed < bytes) {
    const std::uint64_t position = offset + completed;
    if (position < offset || !offset_fits(position)) {
      error = EOVERFLOW;
      return false;
    }
    const ssize_t count = ::pwrite(fd, input + completed, bytes - completed,
                                   static_cast<off_t>(position));
    if (count < 0) {
      if (errno == EINTR) {
        continue;
      }
      error = errno;
      return false;
    }
    if (count == 0) {
      error = EIO;
      return false;
    }
    completed += static_cast<std::size_t>(count);
  }
  return true;
}

[[nodiscard]] PrefillA4ConverterDiagnostic hash_open_file(
    const int fd, const std::uint64_t bytes, std::string& digest) {
  try {
    constexpr std::size_t kChunk = 8U * 1024U * 1024U;
    std::vector<std::uint8_t> buffer(
        static_cast<std::size_t>(std::min<std::uint64_t>(bytes, kChunk)));
    core::Sha256 hash;
    std::uint64_t offset = 0U;
    while (offset < bytes) {
      const std::size_t count = static_cast<std::size_t>(
          std::min<std::uint64_t>(buffer.size(), bytes - offset));
      int error = 0;
      if (!pread_exact(fd, buffer.data(), count, offset, error)) {
        return make_diagnostic(PrefillA4ConverterErrorCode::kIoFailure,
                               "sha256", "failed to read file while hashing",
                               {}, {}, error);
      }
      if (!hash.update(buffer.data(), count)) {
        return make_diagnostic(
            PrefillA4ConverterErrorCode::kArithmeticOverflow, "sha256",
            "SHA-256 byte count overflowed");
      }
      offset += count;
    }
    digest = hash.finalize().hex();
    return {};
  } catch (const std::bad_alloc&) {
    return make_diagnostic(PrefillA4ConverterErrorCode::kAllocationFailure,
                           "sha256", "hash buffer allocation failed");
  }
}

[[nodiscard]] bool read_bounded_file(const fs::path& path,
                                     const std::uint64_t maximum_bytes,
                                     std::string& output,
                                     PrefillA4ConverterDiagnostic& diagnostic) {
  UniqueFd input(::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  if (!input) {
    diagnostic = make_diagnostic(
        PrefillA4ConverterErrorCode::kOpenFailed, path.string(),
        "failed to open regular non-symlink input", {}, {}, errno);
    return false;
  }
  struct stat before {};
  if (::fstat(input.get(), &before) != 0 || !S_ISREG(before.st_mode) ||
      before.st_size < 0) {
    diagnostic = make_diagnostic(PrefillA4ConverterErrorCode::kIoFailure,
                                 path.string(),
                                 "input fstat/regular-file check failed", {},
                                 {}, errno);
    return false;
  }
  const std::uint64_t bytes = static_cast<std::uint64_t>(before.st_size);
  if (bytes > maximum_bytes || bytes > std::numeric_limits<std::size_t>::max()) {
    diagnostic = make_diagnostic(
        PrefillA4ConverterErrorCode::kInvalidOption, path.string(),
        "input exceeds its bounded size", std::to_string(maximum_bytes),
        std::to_string(bytes));
    return false;
  }
  try {
    output.resize(static_cast<std::size_t>(bytes));
  } catch (const std::bad_alloc&) {
    diagnostic = make_diagnostic(
        PrefillA4ConverterErrorCode::kAllocationFailure, path.string(),
        "input buffer allocation failed");
    return false;
  }
  int error = 0;
  if (!output.empty() &&
      !pread_exact(input.get(), output.data(), output.size(), 0U, error)) {
    diagnostic = make_diagnostic(PrefillA4ConverterErrorCode::kIoFailure,
                                 path.string(), "short input file read", {},
                                 {}, error);
    return false;
  }
  std::uint8_t extra = 0U;
  ssize_t extra_count = 0;
  do {
    extra_count = ::pread(input.get(), &extra, 1U,
                          static_cast<off_t>(bytes));
  } while (extra_count < 0 && errno == EINTR);
  struct stat after {};
  if (extra_count != 0 || ::fstat(input.get(), &after) != 0 ||
      before.st_dev != after.st_dev || before.st_ino != after.st_ino ||
      before.st_size != after.st_size ||
      before.st_mtim.tv_sec != after.st_mtim.tv_sec ||
      before.st_mtim.tv_nsec != after.st_mtim.tv_nsec ||
      before.st_ctim.tv_sec != after.st_ctim.tv_sec ||
      before.st_ctim.tv_nsec != after.st_ctim.tv_nsec) {
    diagnostic = make_diagnostic(PrefillA4ConverterErrorCode::kIoFailure,
                                 path.string(),
                                 "input changed during bounded read", {}, {},
                                 extra_count < 0 ? errno : 0);
    return false;
  }
  return true;
}

[[nodiscard]] const json::Value::Object* require_object(
    const json::Value& value, const std::string_view context,
    PrefillA4ConverterDiagnostic& diagnostic) {
  const auto* const object = value.as_object();
  if (object == nullptr) {
    diagnostic = make_diagnostic(PrefillA4ConverterErrorCode::kInvalidPolicy,
                                 std::string(context), "expected JSON object");
  }
  return object;
}

[[nodiscard]] bool exact_keys(
    const json::Value::Object& object,
    const std::initializer_list<std::string_view> keys,
    const std::string_view context,
    PrefillA4ConverterDiagnostic& diagnostic) {
  if (object.size() != keys.size()) {
    diagnostic = make_diagnostic(PrefillA4ConverterErrorCode::kInvalidPolicy,
                                 std::string(context),
                                 "JSON object field count is not exact");
    return false;
  }
  for (const std::string_view key : keys) {
    if (object.find(key) == object.end()) {
      diagnostic = make_diagnostic(PrefillA4ConverterErrorCode::kInvalidPolicy,
                                   std::string(context),
                                   "missing required JSON field",
                                   std::string(key));
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool string_member(const json::Value::Object& object,
                                 const std::string_view key,
                                 std::string& output,
                                 const std::string_view context,
                                 PrefillA4ConverterDiagnostic& diagnostic) {
  const auto found = object.find(key);
  const std::string* const value =
      found == object.end() ? nullptr : found->second.as_string();
  if (value == nullptr) {
    diagnostic = make_diagnostic(PrefillA4ConverterErrorCode::kInvalidPolicy,
                                 std::string(context),
                                 "expected JSON string", std::string(key));
    return false;
  }
  output = *value;
  return true;
}

[[nodiscard]] bool uint64_member(const json::Value::Object& object,
                                 const std::string_view key,
                                 std::uint64_t& output,
                                 const std::string_view context,
                                 PrefillA4ConverterDiagnostic& diagnostic) {
  const auto found = object.find(key);
  const json::Number* const value =
      found == object.end() ? nullptr : found->second.as_number();
  if (value == nullptr || !value->to_uint64(output)) {
    diagnostic = make_diagnostic(PrefillA4ConverterErrorCode::kInvalidPolicy,
                                 std::string(context),
                                 "expected unsigned JSON integer",
                                 std::string(key));
    return false;
  }
  return true;
}

[[nodiscard]] bool double_member(const json::Value::Object& object,
                                 const std::string_view key, double& output,
                                 const std::string_view context,
                                 PrefillA4ConverterDiagnostic& diagnostic) {
  const auto found = object.find(key);
  const json::Number* const value =
      found == object.end() ? nullptr : found->second.as_number();
  if (value == nullptr || !value->to_double(output) || !std::isfinite(output)) {
    diagnostic = make_diagnostic(PrefillA4ConverterErrorCode::kInvalidPolicy,
                                 std::string(context),
                                 "expected finite JSON number",
                                 std::string(key));
    return false;
  }
  return true;
}

[[nodiscard]] bool safe_relative_metadata_path(const fs::path& path) {
  if (path.empty() || path.is_absolute() || path.has_root_path() ||
      path.string().size() > 4096U) {
    return false;
  }
  for (const fs::path& component : path) {
    if (component.empty() || component == "." || component == "..") {
      return false;
    }
  }
  const std::string text = path.generic_string();
  return text.find('\\') == std::string::npos &&
         text.find(':') == std::string::npos;
}

[[nodiscard]] bool valid_clip_ratio(const double value) noexcept {
  const float runtime_value = static_cast<float>(value);
  return std::isfinite(value) && value >= kPrefillA4MinimumClipRatio &&
         value <= 1.0 && std::isfinite(runtime_value) &&
         runtime_value >= static_cast<float>(kPrefillA4MinimumClipRatio) &&
         runtime_value <= 1.0F;
}

[[nodiscard]] std::string activation_boundary_key(
    const PrefillProjectionSidecarEntry& entry) {
  const std::string layer = std::to_string(entry.layer_index);
  switch (entry.family) {
    case PrefillProjectionFamily::kMlpGate:
    case PrefillProjectionFamily::kMlpUp:
      return layer + ":mlp_gate_up_input";
    case PrefillProjectionFamily::kMlpDown:
      return layer + ":mlp_down_input";
    case PrefillProjectionFamily::kLinearQkv:
    case PrefillProjectionFamily::kLinearZ:
      return layer + ":linear_qkv_z_input";
    case PrefillProjectionFamily::kLinearO:
      return layer + ":linear_o_input";
    case PrefillProjectionFamily::kFullQ:
    case PrefillProjectionFamily::kFullK:
    case PrefillProjectionFamily::kFullV:
      return layer + ":full_qkv_input";
    case PrefillProjectionFamily::kFullO:
      return layer + ":full_o_input";
    case PrefillProjectionFamily::kCount:
      break;
  }
  return layer + ":invalid";
}

[[nodiscard]] bool same_equalization(
    const std::optional<PrefillA4ChannelEqualization>& left,
    const std::optional<PrefillA4ChannelEqualization>& right) {
  if (left.has_value() != right.has_value()) {
    return false;
  }
  if (!left.has_value()) {
    return true;
  }
  return left->scheme == right->scheme &&
         left->factors_path == right->factors_path &&
         left->factors_sha256 == right->factors_sha256 &&
         left->factor_count == right->factor_count;
}

[[nodiscard]] bool parse_version(const json::Value& value,
                                 std::uint32_t& major,
                                 std::uint32_t& minor,
                                 const std::string_view context,
                                 PrefillA4ConverterDiagnostic& diagnostic) {
  const auto* const object = require_object(value, context, diagnostic);
  if (object == nullptr ||
      !exact_keys(*object, {"major", "minor"}, context, diagnostic)) {
    return false;
  }
  std::uint64_t parsed_major = 0U;
  std::uint64_t parsed_minor = 0U;
  if (!uint64_member(*object, "major", parsed_major, context, diagnostic) ||
      !uint64_member(*object, "minor", parsed_minor, context, diagnostic) ||
      parsed_major > std::numeric_limits<std::uint32_t>::max() ||
      parsed_minor > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  major = static_cast<std::uint32_t>(parsed_major);
  minor = static_cast<std::uint32_t>(parsed_minor);
  return true;
}

[[nodiscard]] bool parse_projection_calibration(
    const json::Value& value, const std::size_t index,
    PrefillA4ProjectionCalibration& output,
    PrefillA4ConverterDiagnostic& diagnostic) {
  const std::string context =
      "policy.projections[" + std::to_string(index) + "]";
  const auto* const object = require_object(value, context, diagnostic);
  if (object == nullptr ||
      !exact_keys(*object,
                  {"ordinal", "source_module", "source_sha256",
                   "weight_clip_ratio", "activation_clip_ratio",
                   "activation_scale_group_size", "rounding",
                   "channel_equalization"},
                  context, diagnostic)) {
    return false;
  }
  std::uint64_t ordinal = 0U;
  std::uint64_t activation_scale_group_size = 0U;
  std::string rounding;
  if (!uint64_member(*object, "ordinal", ordinal, context, diagnostic) ||
      ordinal > std::numeric_limits<std::uint32_t>::max() ||
      !string_member(*object, "source_module", output.source_module, context,
                     diagnostic) ||
      !string_member(*object, "source_sha256", output.source_sha256, context,
                     diagnostic) ||
      !double_member(*object, "weight_clip_ratio",
                     output.weight_clip_ratio, context, diagnostic) ||
      !double_member(*object, "activation_clip_ratio",
                     output.activation_clip_ratio, context, diagnostic) ||
      !uint64_member(*object, "activation_scale_group_size",
                     activation_scale_group_size, context, diagnostic) ||
      !string_member(*object, "rounding", rounding, context, diagnostic)) {
    return false;
  }
  output.ordinal = static_cast<std::uint32_t>(ordinal);
  if (activation_scale_group_size >
      std::numeric_limits<std::uint32_t>::max()) {
    diagnostic = make_diagnostic(
        PrefillA4ConverterErrorCode::kUnsupportedCalibration,
        context + ".activation_scale_group_size",
        "activation group size overflows the policy ABI");
    return false;
  }
  output.activation_scale_group_size =
      static_cast<std::uint32_t>(activation_scale_group_size);
  if (rounding != "nearest_even_v1") {
    diagnostic = make_diagnostic(
        PrefillA4ConverterErrorCode::kUnsupportedCalibration,
        context + ".rounding", "unsupported rounding policy",
        "nearest_even_v1", rounding);
    return false;
  }
  output.rounding = PrefillA4Rounding::kNearestEvenV1;

  const json::Value& equalization = object->at("channel_equalization");
  if (equalization.is_null()) {
    output.channel_equalization.reset();
    return true;
  }
  const auto* const equalization_object =
      require_object(equalization, context + ".channel_equalization",
                     diagnostic);
  if (equalization_object == nullptr ||
      !exact_keys(*equalization_object,
                  {"scheme", "factors_path", "factors_sha256",
                   "factor_count"},
                  context + ".channel_equalization", diagnostic)) {
    return false;
  }
  PrefillA4ChannelEqualization parsed;
  std::string factors_path;
  if (!string_member(*equalization_object, "scheme", parsed.scheme, context,
                     diagnostic) ||
      !string_member(*equalization_object, "factors_path", factors_path,
                     context, diagnostic) ||
      !string_member(*equalization_object, "factors_sha256",
                     parsed.factors_sha256, context, diagnostic) ||
      !uint64_member(*equalization_object, "factor_count", parsed.factor_count,
                     context, diagnostic)) {
    return false;
  }
  parsed.factors_path = fs::path(factors_path);
  if (parsed.scheme != kEqualizationScheme ||
      !safe_relative_metadata_path(parsed.factors_path) ||
      !lowercase_sha256(parsed.factors_sha256)) {
    diagnostic = make_diagnostic(
        PrefillA4ConverterErrorCode::kUnsupportedCalibration,
        context + ".channel_equalization",
        "invalid channel equalization scheme, path, or digest",
        std::string(kEqualizationScheme));
    return false;
  }
  output.channel_equalization.emplace(std::move(parsed));
  return true;
}

[[nodiscard]] std::uint16_t float_to_bf16_nearest_even(
    const float value) noexcept {
  std::uint32_t bits = 0U;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t exponent = bits & 0x7f800000U;
  if (exponent == 0x7f800000U) {
    // Preserve infinities and quiet NaNs. Conversion inputs reject non-finite
    // values, but keeping the helper total avoids undefined edge behavior.
    return static_cast<std::uint16_t>((bits + 0x0000ffffU) >> 16U);
  }
  const std::uint32_t bias = 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>((bits + bias) >> 16U);
}

[[nodiscard]] float bf16_to_float(const std::uint16_t value) noexcept {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
  float output = 0.0F;
  std::memcpy(&output, &bits, sizeof(output));
  return output;
}

[[nodiscard]] int round_nearest_even(const float value) noexcept {
  const float base_value = std::floor(value);
  const float fraction = value - base_value;
  int base = static_cast<int>(base_value);
  if (fraction > 0.5F || (fraction == 0.5F && (base & 1) != 0)) {
    ++base;
  }
  return base;
}

void write_little_u16(const std::uint16_t value,
                      std::uint8_t* const destination) noexcept {
  destination[0] = static_cast<std::uint8_t>(value);
  destination[1] = static_cast<std::uint8_t>(value >> 8U);
}

[[nodiscard]] float read_little_f32(const std::uint8_t* const bytes) noexcept {
  const std::uint32_t bits = static_cast<std::uint32_t>(bytes[0]) |
                             (static_cast<std::uint32_t>(bytes[1]) << 8U) |
                             (static_cast<std::uint32_t>(bytes[2]) << 16U) |
                             (static_cast<std::uint32_t>(bytes[3]) << 24U);
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

void write_quoted(std::ostream& output, const std::string_view value) {
  static constexpr char kHex[] = "0123456789abcdef";
  output.put('"');
  for (const char character : value) {
    const auto byte = static_cast<unsigned char>(character);
    switch (byte) {
      case '"':
        output << "\\\"";
        break;
      case '\\':
        output << "\\\\";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        if (byte < 0x20U) {
          output << "\\u00" << kHex[(byte >> 4U) & 0x0fU]
                 << kHex[byte & 0x0fU];
        } else {
          output.put(static_cast<char>(byte));
        }
        break;
    }
  }
  output.put('"');
}

[[nodiscard]] std::string serialize_receipt(
    const PrefillA4PublicationReceipt& receipt) {
  std::ostringstream output;
  output << "{\n  \"schema\": ";
  write_quoted(output, kReceiptSchema);
  output << ",\n  \"version\": {\"major\": " << receipt.version_major
         << ", \"minor\": " << receipt.version_minor << "},"
         << "\n  \"mode\": ";
  write_quoted(output, to_string(receipt.mode));
  output << ",\n  \"production_residency_eligible\": "
         << (receipt.production_residency_eligible ? "true" : "false")
         << ",\n  \"physical_layout\": ";
  write_quoted(output, receipt.physical_layout);
  output << ",\n  \"source_checkpoint_id\": ";
  write_quoted(output, receipt.source_checkpoint_id);
  output << ",\n  \"source_config_sha256\": ";
  write_quoted(output, receipt.source_config_sha256);
  output << ",\n  \"source_index_sha256\": ";
  write_quoted(output, receipt.source_index_sha256);
  output << ",\n  \"manifest_sha256\": ";
  write_quoted(output, receipt.manifest_sha256);
  output << ",\n  \"policy_sha256\": ";
  write_quoted(output, receipt.policy_sha256);
  output << ",\n  \"policy_bytes\": " << receipt.policy_bytes
         << ",\n  \"payload_sha256\": ";
  write_quoted(output, receipt.payload_sha256);
  output << ",\n  \"payload_bytes\": " << receipt.payload_bytes
         << ",\n  \"projection_count\": " << receipt.projection_count
         << "\n}\n";
  return output.str();
}

[[nodiscard]] PrefillA4ConverterDiagnostic verify_regular_fd(
    const int fd, const std::uint64_t expected_size,
    const std::string_view context) {
  struct stat status {};
  if (::fstat(fd, &status) != 0) {
    return make_diagnostic(PrefillA4ConverterErrorCode::kIoFailure,
                           std::string(context), "fstat failed", {}, {},
                           errno);
  }
  if (!S_ISREG(status.st_mode) || status.st_size < 0 ||
      static_cast<std::uint64_t>(status.st_size) != expected_size) {
    return make_diagnostic(PrefillA4ConverterErrorCode::kSourceTensorMismatch,
                           std::string(context),
                           "regular file size differs from contract",
                           std::to_string(expected_size),
                           status.st_size < 0
                               ? "negative"
                               : std::to_string(status.st_size));
  }
  return {};
}

struct FileSnapshot {
  std::uint64_t device_id = 0U;
  std::uint64_t inode = 0U;
  std::uint64_t size = 0U;
  std::int64_t modification_seconds = 0;
  std::int64_t modification_nanoseconds = 0;
  std::int64_t change_seconds = 0;
  std::int64_t change_nanoseconds = 0;
};

[[nodiscard]] bool capture_snapshot(const int fd, FileSnapshot& output,
                                    int& error) noexcept {
  struct stat status {};
  if (::fstat(fd, &status) != 0 || !S_ISREG(status.st_mode) ||
      status.st_size < 0) {
    error = errno != 0 ? errno : EINVAL;
    return false;
  }
  output.device_id = static_cast<std::uint64_t>(status.st_dev);
  output.inode = static_cast<std::uint64_t>(status.st_ino);
  output.size = static_cast<std::uint64_t>(status.st_size);
  output.modification_seconds = status.st_mtim.tv_sec;
  output.modification_nanoseconds = status.st_mtim.tv_nsec;
  output.change_seconds = status.st_ctim.tv_sec;
  output.change_nanoseconds = status.st_ctim.tv_nsec;
  return true;
}

[[nodiscard]] bool same_snapshot(const FileSnapshot& left,
                                 const FileSnapshot& right) noexcept {
  return left.device_id == right.device_id && left.inode == right.inode &&
         left.size == right.size &&
         left.modification_seconds == right.modification_seconds &&
         left.modification_nanoseconds == right.modification_nanoseconds &&
         left.change_seconds == right.change_seconds &&
         left.change_nanoseconds == right.change_nanoseconds;
}

struct AuthenticatedSources {
  UniqueFd root;
  std::map<std::string, UniqueFd, std::less<>> shards;
  std::map<std::string, FileSnapshot, std::less<>> snapshots;
};

[[nodiscard]] PrefillA4ConverterDiagnostic open_and_authenticate_sources(
    const fs::path& directory, AuthenticatedSources& sources,
    PrefillA4SidecarConversionStats& stats) {
  sources.root = UniqueFd(::open(directory.c_str(),
                                 O_RDONLY | O_DIRECTORY | O_CLOEXEC |
                                     O_NOFOLLOW));
  if (!sources.root) {
    return make_diagnostic(PrefillA4ConverterErrorCode::kOpenFailed,
                           directory.string(),
                           "failed to open checkpoint directory", {}, {},
                           errno);
  }
  for (const ShardIdentity& identity : pinned_qwen36_27b_shards()) {
    UniqueFd shard(::openat(sources.root.get(), identity.filename.c_str(),
                            O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
    if (!shard) {
      return make_diagnostic(PrefillA4ConverterErrorCode::kOpenFailed,
                             identity.filename,
                             "failed to open pinned checkpoint shard", {},
                             {}, errno);
    }
    if (::flock(shard.get(), LOCK_SH | LOCK_NB) != 0) {
      return make_diagnostic(
          PrefillA4ConverterErrorCode::kSourceAuthenticationFailed,
          identity.filename,
          "checkpoint shard is concurrently locked for mutation", {}, {},
          errno);
    }
    PrefillA4ConverterDiagnostic diagnostic =
        verify_regular_fd(shard.get(), identity.file_size, identity.filename);
    if (!diagnostic) {
      return diagnostic;
    }
    FileSnapshot before;
    int snapshot_error = 0;
    if (!capture_snapshot(shard.get(), before, snapshot_error)) {
      return make_diagnostic(PrefillA4ConverterErrorCode::kIoFailure,
                             identity.filename,
                             "failed to snapshot checkpoint shard", {}, {},
                             snapshot_error);
    }
    std::string digest;
    diagnostic = hash_open_file(shard.get(), identity.file_size, digest);
    if (!diagnostic) {
      diagnostic.context = identity.filename;
      return diagnostic;
    }
    stats.source_bytes_read += identity.file_size;
    if (digest != identity.sha256) {
      return make_diagnostic(
          PrefillA4ConverterErrorCode::kSourceAuthenticationFailed,
          identity.filename, "checkpoint shard SHA-256 mismatch",
          identity.sha256, digest);
    }
    FileSnapshot after;
    if (!capture_snapshot(shard.get(), after, snapshot_error) ||
        !same_snapshot(before, after)) {
      return make_diagnostic(
          PrefillA4ConverterErrorCode::kSourceAuthenticationFailed,
          identity.filename,
          "checkpoint shard changed during initial authentication", {}, {},
          snapshot_error);
    }
    sources.snapshots.emplace(identity.filename, after);
    sources.shards.emplace(identity.filename, std::move(shard));
  }
  return {};
}

[[nodiscard]] PrefillA4ConverterDiagnostic reauthenticate_sources(
    const AuthenticatedSources& sources,
    PrefillA4SidecarConversionStats& stats) {
  for (const ShardIdentity& identity : pinned_qwen36_27b_shards()) {
    const auto found = sources.shards.find(identity.filename);
    const auto snapshot_found = sources.snapshots.find(identity.filename);
    if (found == sources.shards.end() ||
        snapshot_found == sources.snapshots.end()) {
      return make_diagnostic(
          PrefillA4ConverterErrorCode::kSourceAuthenticationFailed,
          identity.filename,
          "authenticated shard handle disappeared during conversion");
    }
    PrefillA4ConverterDiagnostic diagnostic = verify_regular_fd(
        found->second.get(), identity.file_size, identity.filename);
    if (!diagnostic) {
      return diagnostic;
    }
    FileSnapshot before;
    int snapshot_error = 0;
    if (!capture_snapshot(found->second.get(), before, snapshot_error) ||
        !same_snapshot(snapshot_found->second, before)) {
      return make_diagnostic(
          PrefillA4ConverterErrorCode::kSourceAuthenticationFailed,
          identity.filename,
          "checkpoint shard metadata changed during conversion", {}, {},
          snapshot_error);
    }
    std::string digest;
    diagnostic = hash_open_file(found->second.get(), identity.file_size,
                                digest);
    stats.source_bytes_read += identity.file_size;
    if (!diagnostic) {
      diagnostic.context = identity.filename;
      return diagnostic;
    }
    if (digest != identity.sha256) {
      return make_diagnostic(
          PrefillA4ConverterErrorCode::kSourceAuthenticationFailed,
          identity.filename,
          "checkpoint shard changed during sidecar conversion",
          identity.sha256, digest);
    }
    FileSnapshot after;
    if (!capture_snapshot(found->second.get(), after, snapshot_error) ||
        !same_snapshot(snapshot_found->second, after)) {
      return make_diagnostic(
          PrefillA4ConverterErrorCode::kSourceAuthenticationFailed,
          identity.filename,
          "checkpoint shard changed during final authentication", {}, {},
          snapshot_error);
    }
  }
  return {};
}

[[nodiscard]] const UniqueFd* source_fd(
    const AuthenticatedSources& sources,
    const mw::TensorLocator& locator) noexcept {
  const auto found = sources.shards.find(locator.shard);
  return found == sources.shards.end() ? nullptr : &found->second;
}

[[nodiscard]] PrefillA4ConverterDiagnostic read_tensor_range(
    const AuthenticatedSources& sources, const mw::TensorLocator& locator,
    const std::uint64_t relative_offset, void* const output,
    const std::size_t bytes, const std::string_view context,
    PrefillA4SidecarConversionStats& stats) {
  std::uint64_t end = 0U;
  if (!checked_add(relative_offset, bytes, end) || end > locator.byte_size) {
    return make_diagnostic(
        PrefillA4ConverterErrorCode::kSourceTensorMismatch,
        std::string(context), "tensor read exceeds authenticated locator");
  }
  const UniqueFd* const fd = source_fd(sources, locator);
  if (fd == nullptr) {
    return make_diagnostic(PrefillA4ConverterErrorCode::kSourceTensorMismatch,
                           std::string(context),
                           "tensor references an unauthenticated shard");
  }
  std::uint64_t absolute = 0U;
  if (!checked_add(locator.file_begin, relative_offset, absolute)) {
    return make_diagnostic(
        PrefillA4ConverterErrorCode::kArithmeticOverflow,
        std::string(context), "tensor source offset overflowed");
  }
  int error = 0;
  if (!pread_exact(fd->get(), output, bytes, absolute, error)) {
    return make_diagnostic(PrefillA4ConverterErrorCode::kIoFailure,
                           std::string(context), "tensor payload read failed",
                           {}, {}, error);
  }
  stats.source_bytes_read += bytes;
  return {};
}

[[nodiscard]] PrefillA4ConverterDiagnostic load_equalization_factors(
    const PrefillA4ProjectionCalibration& calibration,
    const fs::path& policy_directory, const std::uint64_t input_size,
    std::vector<float>& factors, PrefillA4SidecarConversionStats& stats) {
  factors.clear();
  if (!calibration.channel_equalization.has_value()) {
    return {};
  }
  const PrefillA4ChannelEqualization& contract =
      *calibration.channel_equalization;
  if (contract.factor_count != input_size) {
    return make_diagnostic(
        PrefillA4ConverterErrorCode::kPolicyCoverageMismatch,
        calibration.source_module + ".channel_equalization.factor_count",
        "equalization factor count differs from projection K",
        std::to_string(input_size), std::to_string(contract.factor_count));
  }
  const fs::path path = policy_directory / contract.factors_path;
  UniqueFd fd(::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  if (!fd) {
    return make_diagnostic(PrefillA4ConverterErrorCode::kOpenFailed,
                           path.string(),
                           "failed to open channel equalization factors", {},
                           {}, errno);
  }
  std::uint64_t bytes = 0U;
  if (!checked_multiply(input_size, sizeof(float), bytes)) {
    return make_diagnostic(PrefillA4ConverterErrorCode::kArithmeticOverflow,
                           path.string(), "factor byte count overflowed");
  }
  PrefillA4ConverterDiagnostic diagnostic =
      verify_regular_fd(fd.get(), bytes, path.string());
  if (!diagnostic) {
    return diagnostic;
  }
  try {
    factors.resize(static_cast<std::size_t>(input_size));
    std::vector<std::uint8_t> encoded(static_cast<std::size_t>(bytes));
    int error = 0;
    if (!pread_exact(fd.get(), encoded.data(), encoded.size(), 0U, error)) {
      return make_diagnostic(PrefillA4ConverterErrorCode::kIoFailure,
                             path.string(), "factor payload read failed", {},
                             {}, error);
    }
    stats.source_bytes_read += bytes;
    const std::string digest = core::sha256(std::string_view(
        reinterpret_cast<const char*>(encoded.data()), encoded.size())).hex();
    if (digest != contract.factors_sha256) {
      return make_diagnostic(PrefillA4ConverterErrorCode::kDigestMismatch,
                             path.string(),
                             "equalization factor digest mismatch",
                             contract.factors_sha256, digest);
    }
    for (std::size_t index = 0U; index < factors.size(); ++index) {
      factors[index] = read_little_f32(encoded.data() + index * 4U);
      const float inverse = 1.0F / factors[index];
      if (!std::isfinite(factors[index]) || factors[index] <= 0.0F ||
          !std::isfinite(inverse) || inverse <= 0.0F) {
        return make_diagnostic(
            PrefillA4ConverterErrorCode::kUnsupportedCalibration,
            path.string(),
            "equalization factors and their runtime inverse must be finite and positive",
            {}, std::to_string(index));
      }
    }
  } catch (const std::bad_alloc&) {
    return make_diagnostic(PrefillA4ConverterErrorCode::kAllocationFailure,
                           path.string(), "factor allocation failed");
  }
  return {};
}

void remove_if_present(const fs::path& path) noexcept {
  std::error_code ignored;
  (void)fs::remove(path, ignored);
}

[[nodiscard]] UniqueFd create_temporary_file_near(
    const fs::path& target, const std::string_view tag, fs::path& path,
    PrefillA4ConverterDiagnostic& diagnostic) {
  try {
    std::string pattern = target.string() + ".tmp." + std::string(tag) +
                          ".XXXXXX";
    std::vector<char> mutable_pattern(pattern.begin(), pattern.end());
    mutable_pattern.push_back('\0');
    const int fd = ::mkstemp(mutable_pattern.data());
    if (fd < 0) {
      diagnostic = make_diagnostic(
          PrefillA4ConverterErrorCode::kOpenFailed, target.string(),
          "failed to create unique temporary publication file", {}, {},
          errno);
      return {};
    }
    UniqueFd output(fd);
    path = fs::path(mutable_pattern.data());
    const int descriptor_flags = ::fcntl(fd, F_GETFD);
    if (descriptor_flags < 0 ||
        ::fcntl(fd, F_SETFD, descriptor_flags | FD_CLOEXEC) != 0 ||
        ::fchmod(fd, S_IRUSR | S_IWUSR) != 0) {
      diagnostic = make_diagnostic(
          PrefillA4ConverterErrorCode::kIoFailure, path.string(),
          "failed to secure temporary publication descriptor", {}, {}, errno);
      return {};
    }
    diagnostic = {};
    return output;
  } catch (const std::bad_alloc&) {
    diagnostic = make_diagnostic(
        PrefillA4ConverterErrorCode::kAllocationFailure, target.string(),
        "temporary pathname allocation failed");
    return {};
  }
}

[[nodiscard]] bool target_absent(const fs::path& path,
                                 PrefillA4ConverterDiagnostic& diagnostic) {
  struct stat status {};
  if (::lstat(path.c_str(), &status) == 0) {
    diagnostic = make_diagnostic(
        PrefillA4ConverterErrorCode::kPublicationConflict, path.string(),
        "publication target already exists and will not be replaced");
    return false;
  }
  if (errno != ENOENT) {
    diagnostic = make_diagnostic(PrefillA4ConverterErrorCode::kIoFailure,
                                 path.string(), "failed to inspect target", {},
                                 {}, errno);
    return false;
  }
  return true;
}

[[nodiscard]] PrefillA4ConverterDiagnostic atomic_link_publication(
    const fs::path& payload_temp, const int payload_fd,
    const fs::path& payload, const fs::path& receipt_temp,
    const int receipt_fd, const fs::path& receipt) {
  struct stat payload_path_status {};
  struct stat payload_fd_status {};
  struct stat receipt_path_status {};
  struct stat receipt_fd_status {};
  if (::lstat(payload_temp.c_str(), &payload_path_status) != 0 ||
      ::fstat(payload_fd, &payload_fd_status) != 0 ||
      ::lstat(receipt_temp.c_str(), &receipt_path_status) != 0 ||
      ::fstat(receipt_fd, &receipt_fd_status) != 0 ||
      !S_ISREG(payload_path_status.st_mode) ||
      !S_ISREG(receipt_path_status.st_mode) ||
      payload_path_status.st_dev != payload_fd_status.st_dev ||
      payload_path_status.st_ino != payload_fd_status.st_ino ||
      receipt_path_status.st_dev != receipt_fd_status.st_dev ||
      receipt_path_status.st_ino != receipt_fd_status.st_ino ||
      payload_fd_status.st_nlink != 1 || receipt_fd_status.st_nlink != 1 ||
      (payload_fd_status.st_mode & (S_IWUSR | S_IWGRP | S_IWOTH)) != 0 ||
      (receipt_fd_status.st_mode & (S_IWUSR | S_IWGRP | S_IWOTH)) != 0) {
    return make_diagnostic(
        PrefillA4ConverterErrorCode::kPublicationConflict,
        payload_temp.string(),
        "temporary publication path no longer names its secured descriptor",
        {}, {}, errno);
  }
  if (::link(payload_temp.c_str(), payload.c_str()) != 0) {
    return make_diagnostic(PrefillA4ConverterErrorCode::kPublicationConflict,
                           payload.string(),
                           "failed to atomically publish payload without replace",
                           {}, {}, errno);
  }
  if (::link(receipt_temp.c_str(), receipt.c_str()) != 0) {
    const int saved = errno;
    (void)::unlink(payload.c_str());
    return make_diagnostic(PrefillA4ConverterErrorCode::kPublicationConflict,
                           receipt.string(),
                           "failed to atomically publish receipt without replace",
                           {}, {}, saved);
  }
  if (::unlink(payload_temp.c_str()) != 0 ||
      ::unlink(receipt_temp.c_str()) != 0) {
    const int saved = errno;
    (void)::unlink(receipt.c_str());
    (void)::unlink(payload.c_str());
    return make_diagnostic(PrefillA4ConverterErrorCode::kIoFailure,
                           payload.string(),
                           "temporary unlink failed; publication rolled back",
                           {}, {}, saved);
  }
  const fs::path parent = payload.parent_path().empty()
                              ? fs::path(".")
                              : payload.parent_path();
  UniqueFd directory(
      ::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  if (!directory || ::fsync(directory.get()) != 0) {
    const int saved = errno;
    (void)::unlink(receipt.c_str());
    (void)::unlink(payload.c_str());
    if (directory) {
      (void)::fsync(directory.get());
    }
    return make_diagnostic(PrefillA4ConverterErrorCode::kIoFailure,
                           parent.string(),
                           "directory sync failed; publication rolled back", {},
                           {}, saved);
  }
  return {};
}

}  // namespace

PrefillA4CalibrationPolicyResult parse_prefill_a4_calibration_policy(
    const std::string_view document,
    const PrefillSidecarManifest& manifest) {
  PrefillA4CalibrationPolicyResult result;
  try {
    json::ParseOptions parse_options;
    parse_options.max_input_bytes = 16U * 1024U * 1024U;
    parse_options.max_nesting_depth = 16U;
    parse_options.max_values = 10'000U;
    parse_options.max_container_items = 10'000U;
    const json::ParseResult parsed = json::parse(document, parse_options);
    if (!parsed) {
      result.diagnostic = make_diagnostic(
          PrefillA4ConverterErrorCode::kInvalidPolicy, "policy.json",
          "failed to parse calibration policy JSON",
          std::string(parsed.error.message()), std::to_string(parsed.error.offset));
      return result;
    }
    const auto* const root =
        require_object(*parsed.value, "policy", result.diagnostic);
    if (root == nullptr ||
        !exact_keys(*root,
                    {"schema", "version", "mode", "sidecar_kind",
                     "physical_layout",
                     "source_checkpoint_id", "source_config_sha256",
                     "source_index_sha256", "manifest_sha256", "projections"},
                    "policy", result.diagnostic)) {
      return result;
    }
    PrefillA4CalibrationPolicy policy;
    std::string schema;
    std::string mode;
    std::string sidecar_kind;
    if (!string_member(*root, "schema", schema, "policy", result.diagnostic) ||
        schema != kPolicySchema ||
        !parse_version(root->at("version"), policy.version_major,
                       policy.version_minor, "policy.version",
                       result.diagnostic) ||
        !string_member(*root, "mode", mode, "policy", result.diagnostic) ||
        !string_member(*root, "sidecar_kind", sidecar_kind, "policy",
                       result.diagnostic) ||
        !string_member(*root, "physical_layout", policy.physical_layout,
                       "policy", result.diagnostic) ||
        !string_member(*root, "source_checkpoint_id",
                       policy.source_checkpoint_id, "policy",
                       result.diagnostic) ||
        !string_member(*root, "source_config_sha256",
                       policy.source_config_sha256, "policy",
                       result.diagnostic) ||
        !string_member(*root, "source_index_sha256",
                       policy.source_index_sha256, "policy",
                       result.diagnostic) ||
        !string_member(*root, "manifest_sha256", policy.manifest_sha256,
                       "policy", result.diagnostic)) {
      if (result.diagnostic.ok()) {
        result.diagnostic = make_diagnostic(
            PrefillA4ConverterErrorCode::kInvalidPolicy, "policy.schema",
            "unsupported calibration policy schema", std::string(kPolicySchema),
            schema);
      }
      return result;
    }
    if (mode == "production_calibrated") {
      policy.mode = PrefillA4ConversionMode::kProductionCalibrated;
    } else if (mode == "experimental_nearest_even_smoke") {
      policy.mode = PrefillA4ConversionMode::kExperimentalNearestEvenSmoke;
    } else {
      result.diagnostic = make_diagnostic(
          PrefillA4ConverterErrorCode::kInvalidPolicy, "policy.mode",
          "unsupported conversion mode", "production_calibrated", mode);
      return result;
    }
    if (sidecar_kind != "a4_k64") {
      result.diagnostic = make_diagnostic(
          PrefillA4ConverterErrorCode::kInvalidPolicy, "policy.sidecar_kind",
          "only A4-K64 policy is supported", "a4_k64", sidecar_kind);
      return result;
    }
    policy.sidecar_kind = PrefillSidecarKind::kA4K64;
    if (policy.physical_layout != kPrefillA4PhysicalLayout) {
      result.diagnostic = make_diagnostic(
          PrefillA4ConverterErrorCode::kInvalidPolicy,
          "policy.physical_layout",
          "unsupported A4 physical payload layout",
          std::string(kPrefillA4PhysicalLayout), policy.physical_layout);
      return result;
    }
    const json::Value::Array* const projections =
        root->at("projections").as_array();
    if (projections == nullptr) {
      result.diagnostic = make_diagnostic(
          PrefillA4ConverterErrorCode::kInvalidPolicy, "policy.projections",
          "expected projection policy array");
      return result;
    }
    policy.projections.reserve(projections->size());
    for (std::size_t index = 0U; index < projections->size(); ++index) {
      PrefillA4ProjectionCalibration calibration;
      if (!parse_projection_calibration((*projections)[index], index,
                                        calibration, result.diagnostic)) {
        return result;
      }
      policy.projections.emplace_back(std::move(calibration));
    }
    policy.policy_sha256 = core::sha256(document).hex();
    policy.policy_bytes = document.size();
    result.diagnostic =
        validate_prefill_a4_calibration_policy(policy, manifest);
    if (!result.diagnostic) {
      return result;
    }
    result.value.emplace(std::move(policy));
    return result;
  } catch (const std::bad_alloc&) {
    result.diagnostic = make_diagnostic(
        PrefillA4ConverterErrorCode::kAllocationFailure, "policy",
        "allocation failed while parsing calibration policy");
    return result;
  } catch (...) {
    result.diagnostic = make_diagnostic(
        PrefillA4ConverterErrorCode::kInvalidPolicy, "policy",
        "unexpected calibration policy parsing failure");
    return result;
  }
}

PrefillA4ConverterDiagnostic validate_prefill_a4_calibration_policy(
    const PrefillA4CalibrationPolicy& policy,
    const PrefillSidecarManifest& manifest) {
  const PrefillContractDiagnostic manifest_diagnostic =
      validate_prefill_sidecar_manifest(manifest);
  if (!manifest_diagnostic || manifest.kind != PrefillSidecarKind::kA4K64 ||
      manifest.projections.size() != kQwen36PrefillProjectionCount) {
    return make_diagnostic(PrefillA4ConverterErrorCode::kInvalidManifest,
                           "manifest",
                           "policy requires a valid 400-entry A4-K64 manifest");
  }
  if (policy.version_major != kPrefillA4CalibrationPolicyVersionMajor ||
      policy.version_minor != kPrefillA4CalibrationPolicyVersionMinor ||
      policy.sidecar_kind != PrefillSidecarKind::kA4K64 ||
      policy.physical_layout != kPrefillA4PhysicalLayout ||
      !lowercase_sha256(policy.policy_sha256) || policy.policy_bytes == 0U ||
      policy.policy_bytes > 16ULL * 1024ULL * 1024ULL) {
    return make_diagnostic(PrefillA4ConverterErrorCode::kInvalidPolicy,
                           "policy.header",
                           "policy version, kind, or digest is invalid");
  }
  if (policy.source_checkpoint_id != manifest.source_checkpoint_id ||
      policy.source_config_sha256 != manifest.source_config_sha256 ||
      policy.source_index_sha256 != manifest.source_index_sha256 ||
      policy.manifest_sha256 != manifest.manifest_sha256) {
    return make_diagnostic(PrefillA4ConverterErrorCode::kSourceBindingMismatch,
                           "policy.source_binding",
                           "policy is not bound to the supplied manifest");
  }
  if (policy.mode != PrefillA4ConversionMode::kProductionCalibrated) {
    return make_diagnostic(
        PrefillA4ConverterErrorCode::kPublicationRejected, "policy.mode",
        "experimental nearest-round policy is not production-admissible");
  }
  if (policy.projections.size() != manifest.projections.size()) {
    return make_diagnostic(
        PrefillA4ConverterErrorCode::kPolicyCoverageMismatch,
        "policy.projections",
        "production policy must cover all 400 projections exactly",
        std::to_string(manifest.projections.size()),
        std::to_string(policy.projections.size()));
  }
  std::map<std::string, std::size_t, std::less<>> activation_boundaries;
  for (std::size_t index = 0U; index < manifest.projections.size(); ++index) {
    const PrefillProjectionSidecarEntry& entry = manifest.projections[index];
    const PrefillA4ProjectionCalibration& calibration =
        policy.projections[index];
    if (calibration.ordinal != entry.ordinal ||
        calibration.source_module != entry.source_module ||
        calibration.source_sha256 != entry.source_sha256) {
      return make_diagnostic(
          PrefillA4ConverterErrorCode::kPolicyCoverageMismatch,
          "policy.projections[" + std::to_string(index) + "]",
          "projection policy identity differs from manifest");
    }
    if (!valid_clip_ratio(calibration.weight_clip_ratio) ||
        !valid_clip_ratio(calibration.activation_clip_ratio) ||
        calibration.activation_scale_group_size !=
            kPrefillA4WeightGroupSize ||
        calibration.rounding != PrefillA4Rounding::kNearestEvenV1) {
      return make_diagnostic(
          PrefillA4ConverterErrorCode::kUnsupportedCalibration,
          calibration.source_module,
          "weight/activation clip, K64 group, or rounding policy is unsupported");
    }
    if (calibration.channel_equalization.has_value()) {
      const PrefillA4ChannelEqualization& equalization =
          *calibration.channel_equalization;
      if (equalization.scheme != kEqualizationScheme ||
          !safe_relative_metadata_path(equalization.factors_path) ||
          !lowercase_sha256(equalization.factors_sha256) ||
          equalization.factor_count != entry.input_size) {
        return make_diagnostic(
            PrefillA4ConverterErrorCode::kUnsupportedCalibration,
            calibration.source_module + ".channel_equalization",
            "channel equalization metadata differs from projection K64 ABI");
      }
    }
    const std::string boundary = activation_boundary_key(entry);
    const auto inserted = activation_boundaries.emplace(boundary, index);
    if (!inserted.second) {
      const PrefillA4ProjectionCalibration& first =
          policy.projections[inserted.first->second];
      if (calibration.activation_clip_ratio !=
              first.activation_clip_ratio ||
          calibration.activation_scale_group_size !=
              first.activation_scale_group_size ||
          calibration.rounding != first.rounding ||
          !same_equalization(calibration.channel_equalization,
                             first.channel_equalization)) {
        return make_diagnostic(
            PrefillA4ConverterErrorCode::kPolicyCoverageMismatch,
            "policy.activation_boundary." + boundary,
            "projections sharing one activation must share clip/group/rounding/equalization metadata",
            manifest.projections[inserted.first->second].source_module,
            calibration.source_module);
      }
    }
  }
  return {};
}

PrefillA4ConverterDiagnostic quantize_prefill_a4_k64_consumer_blocks(
    const float* const source_rows, const std::size_t row_count,
    const std::size_t input_size,
    const PrefillA4ProjectionCalibration& calibration,
    const PrefillA4ConversionMode mode,
    std::uint8_t* const packed_signed_w4,
    const std::size_t packed_signed_w4_bytes,
    std::uint8_t* const bf16_scales_little_endian,
    const std::size_t bf16_scale_bytes) {
  if (row_count == 0U || input_size == 0U ||
      (row_count % 64U) != 0U ||
      (input_size % kPrefillA4WeightGroupSize) != 0U ||
      source_rows == nullptr || packed_signed_w4 == nullptr ||
      bf16_scales_little_endian == nullptr) {
    return make_diagnostic(PrefillA4ConverterErrorCode::kInvalidOption,
                           "a4_k64_quantize",
                           "N64 blocks, K64, or buffer arguments are invalid");
  }
  if (input_size > std::numeric_limits<std::size_t>::max() / row_count) {
    return make_diagnostic(PrefillA4ConverterErrorCode::kArithmeticOverflow,
                           "a4_k64_quantize",
                           "input matrix element count overflowed");
  }
  const std::size_t elements = row_count * input_size;
  const std::size_t expected_packed_bytes = elements / 2U;
  const std::size_t expected_scale_bytes =
      elements / kPrefillA4WeightGroupSize * 2U;
  if (packed_signed_w4_bytes != expected_packed_bytes ||
      bf16_scale_bytes != expected_scale_bytes) {
    return make_diagnostic(
        PrefillA4ConverterErrorCode::kInvalidOption, "a4_k64_quantize",
        "consumer output buffer lengths differ from exact N64/K64 ABI",
        std::to_string(expected_packed_bytes) + ":" +
            std::to_string(expected_scale_bytes),
        std::to_string(packed_signed_w4_bytes) + ":" +
            std::to_string(bf16_scale_bytes));
  }
  double clip_ratio = calibration.weight_clip_ratio;
  switch (mode) {
    case PrefillA4ConversionMode::kProductionCalibrated:
      if (!valid_clip_ratio(calibration.activation_clip_ratio) ||
          calibration.activation_scale_group_size !=
              kPrefillA4WeightGroupSize) {
        return make_diagnostic(
            PrefillA4ConverterErrorCode::kUnsupportedCalibration,
            "a4_k64_quantize",
            "production quantization requires explicit activation K64 calibration");
      }
      break;
    case PrefillA4ConversionMode::kExperimentalNearestEvenSmoke:
      if (clip_ratio == 0.0) {
        clip_ratio = 1.0;
      }
      break;
    default:
      return make_diagnostic(PrefillA4ConverterErrorCode::kInvalidOption,
                             "a4_k64_quantize",
                             "unknown conversion mode");
  }
  if (!valid_clip_ratio(clip_ratio) ||
      calibration.rounding != PrefillA4Rounding::kNearestEvenV1) {
    return make_diagnostic(PrefillA4ConverterErrorCode::kUnsupportedCalibration,
                           "a4_k64_quantize",
                           "explicit clip and nearest-even policy required");
  }
  const std::size_t n_blocks = row_count / 64U;
  const std::size_t k_blocks = input_size / 64U;
  for (std::size_t n_block = 0U; n_block < n_blocks; ++n_block) {
    for (std::size_t k_block = 0U; k_block < k_blocks; ++k_block) {
      for (std::size_t local_n = 0U; local_n < 64U; ++local_n) {
        const std::size_t row = n_block * 64U + local_n;
        const std::size_t begin = k_block * 64U;
        const float* const input = source_rows + row * input_size;
        std::uint8_t* const packed =
            packed_signed_w4 +
            (((n_block * k_blocks + k_block) * 64U + local_n) * 32U);
        std::uint8_t* const scale =
            bf16_scales_little_endian +
            (((n_block * k_blocks + k_block) * 64U + local_n) * 2U);
        float maximum = 0.0F;
        for (std::size_t index = 0U; index < kPrefillA4WeightGroupSize;
             ++index) {
          const float value = input[begin + index];
          if (!std::isfinite(value)) {
            return make_diagnostic(
                PrefillA4ConverterErrorCode::kQuantizationFailure,
                "a4_k64_quantize", "source contains non-finite value", {},
                std::to_string(row * input_size + begin + index));
          }
          maximum = std::max(maximum, std::fabs(value));
        }
        const float threshold = maximum * static_cast<float>(clip_ratio);
        std::uint16_t scale_bits =
            float_to_bf16_nearest_even(threshold / 7.0F);
        float stored_scale = bf16_to_float(scale_bits);
        if (maximum != 0.0F && stored_scale == 0.0F) {
          scale_bits = 1U;
          stored_scale = bf16_to_float(scale_bits);
        }
        write_little_u16(scale_bits, scale);
        for (std::size_t pair = 0U; pair < kPrefillA4WeightGroupSize / 2U;
             ++pair) {
          std::uint8_t encoded = 0U;
          for (std::size_t lane = 0U; lane < 2U; ++lane) {
            const float value = input[begin + pair * 2U + lane];
            int quantized = 0;
            if (stored_scale != 0.0F) {
              const float clipped =
                  std::max(-threshold, std::min(threshold, value));
              quantized = round_nearest_even(clipped / stored_scale);
              quantized = std::max(-7, std::min(7, quantized));
            }
            const std::uint8_t nibble =
                static_cast<std::uint8_t>(quantized) & 0x0fU;
            encoded = static_cast<std::uint8_t>(
                encoded | static_cast<std::uint8_t>(nibble << (lane * 4U)));
          }
          packed[pair] = encoded;
        }
      }
    }
  }
  return {};
}

PrefillA4SidecarConversionResult
convert_pinned_qwen36_27b_prefill_a4_k64_sidecar(
    const PrefillA4SidecarConversionOptions& options) {
  PrefillA4SidecarConversionResult result;
  fs::path payload_temp;
  fs::path receipt_temp;
  struct TemporaryCleanup {
    fs::path* payload = nullptr;
    fs::path* receipt = nullptr;
    ~TemporaryCleanup() {
      if (payload != nullptr && !payload->empty()) {
        remove_if_present(*payload);
      }
      if (receipt != nullptr && !receipt->empty()) {
        remove_if_present(*receipt);
      }
    }
  } cleanup{&payload_temp, &receipt_temp};

  try {
    if (options.model_directory.empty() ||
        options.calibration_policy_path.empty() || options.output_path.empty() ||
        options.row_chunk_size == 0U || options.row_chunk_size > 256U ||
        (options.row_chunk_size % 64U) != 0U ||
        options.max_policy_bytes == 0U ||
        options.max_policy_bytes > 16ULL * 1024ULL * 1024ULL) {
      result.diagnostic = make_diagnostic(
          PrefillA4ConverterErrorCode::kInvalidOption, "conversion_options",
          "directory, policy, output, row chunk, or policy bound is invalid");
      return result;
    }

    const mw::ManifestResult source_manifest =
        mw::build_qwen36_27b_text_manifest(options.model_directory);
    if (!source_manifest) {
      std::string message = "pinned checkpoint manifest validation failed";
      if (!source_manifest.diagnostics.empty()) {
        message += ": " + source_manifest.diagnostics.front().message;
      }
      result.diagnostic = make_diagnostic(
          PrefillA4ConverterErrorCode::kInvalidManifest,
          options.model_directory.string(), std::move(message));
      return result;
    }
    PrefillSidecarManifestOptions manifest_options;
    manifest_options.kind = PrefillSidecarKind::kA4K64;
    const PrefillSidecarManifestResult built_manifest =
        build_qwen36_27b_prefill_sidecar_manifest(
            *source_manifest.value, pinned_qwen36_27b_shards(),
            manifest_options);
    if (!built_manifest) {
      result.diagnostic = make_diagnostic(
          PrefillA4ConverterErrorCode::kInvalidManifest,
          built_manifest.diagnostic.context,
          built_manifest.diagnostic.message);
      return result;
    }
    const PrefillSidecarManifest& manifest = *built_manifest.value;

    std::string policy_document;
    if (!read_bounded_file(options.calibration_policy_path,
                           options.max_policy_bytes, policy_document,
                           result.diagnostic)) {
      return result;
    }
    const PrefillA4CalibrationPolicyResult parsed_policy =
        parse_prefill_a4_calibration_policy(policy_document, manifest);
    if (!parsed_policy) {
      result.diagnostic = parsed_policy.diagnostic;
      return result;
    }
    const PrefillA4CalibrationPolicy& policy = *parsed_policy.value;
    const auto equalized = std::find_if(
        policy.projections.begin(), policy.projections.end(),
        [](const PrefillA4ProjectionCalibration& calibration) {
          return calibration.channel_equalization.has_value();
        });
    if (equalized != policy.projections.end()) {
      result.diagnostic = make_diagnostic(
          PrefillA4ConverterErrorCode::kUnsupportedCalibration,
          equalized->source_module + ".channel_equalization",
          "production conversion rejects equalization until the runner retains authenticated inverse factors");
      return result;
    }

    const fs::path receipt_path =
        fs::path(options.output_path.string() + ".receipt.json");
    if (!target_absent(options.output_path, result.diagnostic) ||
        !target_absent(receipt_path, result.diagnostic)) {
      return result;
    }
    const fs::path output_parent = options.output_path.parent_path().empty()
                                       ? fs::path(".")
                                       : options.output_path.parent_path();
    std::error_code filesystem_error;
    const fs::file_status parent_status =
        fs::symlink_status(output_parent, filesystem_error);
    if (filesystem_error || !fs::is_directory(parent_status)) {
      result.diagnostic = make_diagnostic(
          PrefillA4ConverterErrorCode::kOpenFailed, output_parent.string(),
          "output parent must be an existing non-symlink directory", {},
          filesystem_error.message());
      return result;
    }

    AuthenticatedSources sources;
    result.diagnostic = open_and_authenticate_sources(
        options.model_directory, sources, result.stats);
    if (!result.diagnostic) {
      return result;
    }

    UniqueFd output = create_temporary_file_near(
        options.output_path, "payload", payload_temp, result.diagnostic);
    if (!output) {
      return result;
    }
    if (!offset_fits(manifest.summary.arena_bytes) ||
        ::ftruncate(output.get(),
                    static_cast<off_t>(manifest.summary.arena_bytes)) != 0) {
      result.diagnostic = make_diagnostic(
          PrefillA4ConverterErrorCode::kIoFailure, payload_temp.string(),
          "failed to size temporary sidecar", {}, {}, errno);
      return result;
    }
    if (options.preallocate_output) {
      const int allocation_error =
          ::posix_fallocate(output.get(), 0,
                            static_cast<off_t>(manifest.summary.arena_bytes));
      if (allocation_error != 0) {
        result.diagnostic = make_diagnostic(
            PrefillA4ConverterErrorCode::kIoFailure, payload_temp.string(),
            "failed to preallocate complete sidecar", {}, {},
            allocation_error);
        return result;
      }
    }

    const fs::path policy_directory =
        options.calibration_policy_path.parent_path().empty()
            ? fs::path(".")
            : options.calibration_policy_path.parent_path();
    for (std::size_t projection_index = 0U;
         projection_index < manifest.projections.size(); ++projection_index) {
      const PrefillProjectionSidecarEntry& entry =
          manifest.projections[projection_index];
      const PrefillA4ProjectionCalibration& calibration =
          policy.projections[projection_index];
      if ((entry.output_size % 64U) != 0U ||
          (entry.input_size % 64U) != 0U) {
        result.diagnostic = make_diagnostic(
            PrefillA4ConverterErrorCode::kInvalidManifest,
            entry.source_module,
            "A4 consumer-prepack requires N64 and K64 projection shapes");
        return result;
      }
      const bool nvfp4_source =
          entry.family == PrefillProjectionFamily::kMlpGate ||
          entry.family == PrefillProjectionFamily::kMlpUp ||
          entry.family == PrefillProjectionFamily::kMlpDown;
      const mw::TensorLocator* const weight =
          source_manifest.value->find(entry.source_module + ".weight");
      const mw::TensorLocator* const weight_scale =
          source_manifest.value->find(entry.source_module + ".weight_scale");
      const mw::TensorLocator* const weight_scale_2 =
          nvfp4_source
              ? source_manifest.value->find(entry.source_module +
                                            ".weight_scale_2")
              : nullptr;
      if (weight == nullptr || weight_scale == nullptr ||
          (nvfp4_source && weight_scale_2 == nullptr)) {
        result.diagnostic = make_diagnostic(
            PrefillA4ConverterErrorCode::kSourceTensorMismatch,
            entry.source_module, "projection source components are missing");
        return result;
      }

      std::array<std::uint8_t, 4U> scalar_bytes{};
      const mw::TensorLocator& scalar_locator =
          nvfp4_source ? *weight_scale_2 : *weight_scale;
      result.diagnostic = read_tensor_range(
          sources, scalar_locator, 0U, scalar_bytes.data(), scalar_bytes.size(),
          entry.source_module +
              (nvfp4_source ? ".weight_scale_2" : ".weight_scale"),
          result.stats);
      if (!result.diagnostic) {
        return result;
      }
      const float tensor_scale = read_little_f32(scalar_bytes.data());
      if (!std::isfinite(tensor_scale) || tensor_scale < 0.0F) {
        result.diagnostic = make_diagnostic(
            PrefillA4ConverterErrorCode::kSourceTensorMismatch,
            entry.source_module, "source tensor scale is not finite/nonnegative");
        return result;
      }

      std::vector<float> equalization_factors;
      result.diagnostic = load_equalization_factors(
          calibration, policy_directory, entry.input_size,
          equalization_factors, result.stats);
      if (!result.diagnostic) {
        return result;
      }

      const std::size_t maximum_rows = static_cast<std::size_t>(
          std::min<std::uint64_t>(options.row_chunk_size, entry.output_size));
      const std::size_t input_size =
          static_cast<std::size_t>(entry.input_size);
      const std::size_t source_weight_stride =
          nvfp4_source ? input_size / 2U : input_size;
      const std::size_t source_scale_stride =
          nvfp4_source ? input_size / 16U : 0U;
      const std::size_t output_weight_stride = input_size / 2U;
      const std::size_t output_scale_stride = input_size / 64U * 2U;
      std::vector<std::uint8_t> source_weights(
          maximum_rows * source_weight_stride);
      std::vector<std::uint8_t> source_scales(
          maximum_rows * source_scale_stride);
      std::vector<float> decoded(maximum_rows * input_size);
      std::vector<std::uint8_t> packed(maximum_rows * output_weight_stride);
      std::vector<std::uint8_t> scales(maximum_rows * output_scale_stride);
      const std::uint64_t working_bytes =
          source_weights.capacity() + source_scales.capacity() +
          decoded.capacity() * sizeof(float) + packed.capacity() +
          scales.capacity() + equalization_factors.capacity() * sizeof(float);
      result.stats.peak_working_bytes =
          std::max(result.stats.peak_working_bytes, working_bytes);

      for (std::uint64_t row = 0U; row < entry.output_size;
           row += maximum_rows) {
        const std::size_t rows = static_cast<std::size_t>(
            std::min<std::uint64_t>(maximum_rows, entry.output_size - row));
        const std::size_t source_weight_bytes = rows * source_weight_stride;
        result.diagnostic = read_tensor_range(
            sources, *weight, row * source_weight_stride,
            source_weights.data(), source_weight_bytes,
            entry.source_module + ".weight", result.stats);
        if (!result.diagnostic) {
          return result;
        }
        if (nvfp4_source) {
          const std::size_t source_scale_bytes = rows * source_scale_stride;
          result.diagnostic = read_tensor_range(
              sources, *weight_scale, row * source_scale_stride,
              source_scales.data(), source_scale_bytes,
              entry.source_module + ".weight_scale", result.stats);
          if (!result.diagnostic) {
            return result;
          }
        }

        for (std::size_t local_row = 0U; local_row < rows; ++local_row) {
          for (std::size_t column = 0U; column < input_size; ++column) {
            float value = 0.0F;
            if (nvfp4_source) {
              const std::uint8_t packed_source =
                  source_weights[local_row * source_weight_stride +
                                 column / 2U];
              const std::uint8_t block_scale =
                  source_scales[local_row * source_scale_stride +
                                column / 16U];
              value = quantization::dequantize_nvfp4_value(
                  packed_source, (column & 1U) != 0U, block_scale,
                  tensor_scale);
            } else {
              value = quantization::decode_e4m3fn(
                          source_weights[local_row * source_weight_stride +
                                         column]) *
                      tensor_scale;
            }
            if (!equalization_factors.empty()) {
              value *= equalization_factors[column];
            }
            decoded[local_row * input_size + column] = value;
          }
        }
        const std::size_t packed_bytes = rows * output_weight_stride;
        const std::size_t scale_bytes = rows * output_scale_stride;
        result.diagnostic = quantize_prefill_a4_k64_consumer_blocks(
            decoded.data(), rows, input_size, calibration,
            PrefillA4ConversionMode::kProductionCalibrated, packed.data(),
            packed_bytes, scales.data(), scale_bytes);
        if (!result.diagnostic) {
          result.diagnostic.context = entry.source_module + ":" +
                                      result.diagnostic.context;
          return result;
        }

        std::uint64_t weight_offset = 0U;
        std::uint64_t scale_offset = 0U;
        if (!checked_add(entry.sidecar_offset,
                         row * output_weight_stride, weight_offset) ||
            !checked_add(entry.sidecar_offset, entry.weight_bytes,
                         scale_offset) ||
            !checked_add(scale_offset, row * output_scale_stride,
                         scale_offset)) {
          result.diagnostic = make_diagnostic(
              PrefillA4ConverterErrorCode::kArithmeticOverflow,
              entry.source_module, "sidecar output offset overflowed");
          return result;
        }
        int error = 0;
        if (!pwrite_exact(output.get(), packed.data(), packed_bytes,
                          weight_offset, error) ||
            !pwrite_exact(output.get(), scales.data(), scale_bytes,
                          scale_offset, error)) {
          result.diagnostic = make_diagnostic(
              PrefillA4ConverterErrorCode::kIoFailure, entry.source_module,
              "sidecar projection write failed", {}, {}, error);
          return result;
        }
        result.stats.output_bytes_written += packed_bytes + scale_bytes;
        result.stats.rows_converted += rows;
      }
      ++result.stats.projections_converted;
    }

    if (result.stats.projections_converted !=
            manifest.summary.projection_count ||
        result.stats.output_bytes_written != manifest.summary.payload_bytes) {
      result.diagnostic = make_diagnostic(
          PrefillA4ConverterErrorCode::kIoFailure, payload_temp.string(),
          "conversion did not write the complete 400-projection payload",
          std::to_string(manifest.summary.payload_bytes),
          std::to_string(result.stats.output_bytes_written));
      return result;
    }
    result.diagnostic = reauthenticate_sources(sources, result.stats);
    if (!result.diagnostic) {
      return result;
    }
    if (::fsync(output.get()) != 0) {
      result.diagnostic = make_diagnostic(
          PrefillA4ConverterErrorCode::kIoFailure, payload_temp.string(),
          "failed to synchronize temporary payload", {}, {}, errno);
      return result;
    }
    result.diagnostic = verify_regular_fd(
        output.get(), manifest.summary.arena_bytes, payload_temp.string());
    if (!result.diagnostic) {
      return result;
    }
    std::string payload_sha256;
    result.diagnostic = hash_open_file(
        output.get(), manifest.summary.arena_bytes, payload_sha256);
    if (!result.diagnostic) {
      result.diagnostic.context = payload_temp.string();
      return result;
    }
    if (::fchmod(output.get(), S_IRUSR) != 0 || ::fsync(output.get()) != 0) {
      result.diagnostic = make_diagnostic(
          PrefillA4ConverterErrorCode::kIoFailure, payload_temp.string(),
          "failed to seal temporary payload read-only", {}, {}, errno);
      return result;
    }

    PrefillA4PublicationReceipt receipt;
    receipt.mode = PrefillA4ConversionMode::kProductionCalibrated;
    receipt.production_residency_eligible = true;
    receipt.physical_layout = std::string(kPrefillA4PhysicalLayout);
    receipt.source_checkpoint_id = manifest.source_checkpoint_id;
    receipt.source_config_sha256 = manifest.source_config_sha256;
    receipt.source_index_sha256 = manifest.source_index_sha256;
    receipt.manifest_sha256 = manifest.manifest_sha256;
    receipt.policy_sha256 = policy.policy_sha256;
    receipt.policy_bytes = policy.policy_bytes;
    receipt.payload_sha256 = std::move(payload_sha256);
    receipt.payload_bytes = manifest.summary.arena_bytes;
    receipt.projection_count = manifest.summary.projection_count;
    const std::string serialized_receipt = serialize_receipt(receipt);

    UniqueFd receipt_output = create_temporary_file_near(
        receipt_path, "receipt", receipt_temp, result.diagnostic);
    if (!receipt_output) {
      return result;
    }
    int error = 0;
    if (!pwrite_exact(receipt_output.get(), serialized_receipt.data(),
                      serialized_receipt.size(), 0U, error) ||
        ::fsync(receipt_output.get()) != 0) {
      result.diagnostic = make_diagnostic(
          PrefillA4ConverterErrorCode::kIoFailure, receipt_temp.string(),
          "failed to write/synchronize publication receipt", {}, {},
          error != 0 ? error : errno);
      return result;
    }
    if (::fchmod(receipt_output.get(), S_IRUSR) != 0 ||
        ::fsync(receipt_output.get()) != 0) {
      result.diagnostic = make_diagnostic(
          PrefillA4ConverterErrorCode::kIoFailure, receipt_temp.string(),
          "failed to seal temporary receipt read-only", {}, {}, errno);
      return result;
    }
    result.diagnostic = atomic_link_publication(
        payload_temp, output.get(), options.output_path, receipt_temp,
        receipt_output.get(), receipt_path);
    if (!result.diagnostic) {
      return result;
    }
    payload_temp.clear();
    receipt_temp.clear();
    result.receipt.emplace(std::move(receipt));
    return result;
  } catch (const std::bad_alloc&) {
    result.diagnostic = make_diagnostic(
        PrefillA4ConverterErrorCode::kAllocationFailure, "conversion",
        "bounded conversion buffer allocation failed");
    return result;
  } catch (const std::exception& error) {
    result.diagnostic = make_diagnostic(
        PrefillA4ConverterErrorCode::kIoFailure, "conversion",
        "unexpected conversion failure", {}, error.what());
    return result;
  } catch (...) {
    result.diagnostic = make_diagnostic(
        PrefillA4ConverterErrorCode::kIoFailure, "conversion",
        "unexpected conversion failure");
    return result;
  }
}

std::optional<PrefillA4PublicationReceipt>
parse_prefill_a4_publication_receipt(
    const std::string_view document,
    PrefillA4ConverterDiagnostic& diagnostic) {
  diagnostic = {};
  try {
    json::ParseOptions options;
    options.max_input_bytes = 64U * 1024U;
    options.max_nesting_depth = 8U;
    options.max_values = 128U;
    options.max_container_items = 128U;
    const json::ParseResult parsed = json::parse(document, options);
    if (!parsed) {
      diagnostic = make_diagnostic(
          PrefillA4ConverterErrorCode::kPublicationRejected, "receipt.json",
          "failed to parse publication receipt",
          std::string(parsed.error.message()), std::to_string(parsed.error.offset));
      return std::nullopt;
    }
    const auto* const root = require_object(*parsed.value, "receipt", diagnostic);
    if (root == nullptr ||
        !exact_keys(*root,
                    {"schema", "version", "mode",
                     "production_residency_eligible", "source_checkpoint_id",
                     "physical_layout",
                     "source_config_sha256", "source_index_sha256",
                     "manifest_sha256", "policy_sha256", "policy_bytes",
                     "payload_sha256", "payload_bytes", "projection_count"},
                    "receipt", diagnostic)) {
      return std::nullopt;
    }
    PrefillA4PublicationReceipt receipt;
    std::string schema;
    std::string mode;
    if (!string_member(*root, "schema", schema, "receipt", diagnostic) ||
        schema != kReceiptSchema ||
        !parse_version(root->at("version"), receipt.version_major,
                       receipt.version_minor, "receipt.version", diagnostic) ||
        !string_member(*root, "mode", mode, "receipt", diagnostic) ||
        !string_member(*root, "physical_layout", receipt.physical_layout,
                       "receipt", diagnostic) ||
        !string_member(*root, "source_checkpoint_id",
                       receipt.source_checkpoint_id, "receipt", diagnostic) ||
        !string_member(*root, "source_config_sha256",
                       receipt.source_config_sha256, "receipt", diagnostic) ||
        !string_member(*root, "source_index_sha256",
                       receipt.source_index_sha256, "receipt", diagnostic) ||
        !string_member(*root, "manifest_sha256", receipt.manifest_sha256,
                       "receipt", diagnostic) ||
        !string_member(*root, "policy_sha256", receipt.policy_sha256,
                       "receipt", diagnostic) ||
        !uint64_member(*root, "policy_bytes", receipt.policy_bytes,
                       "receipt", diagnostic) ||
        !string_member(*root, "payload_sha256", receipt.payload_sha256,
                       "receipt", diagnostic) ||
        !uint64_member(*root, "payload_bytes", receipt.payload_bytes,
                       "receipt", diagnostic) ||
        !uint64_member(*root, "projection_count", receipt.projection_count,
                       "receipt", diagnostic)) {
      if (diagnostic.ok()) {
        diagnostic = make_diagnostic(
            PrefillA4ConverterErrorCode::kPublicationRejected,
            "receipt.schema", "unsupported receipt schema",
            std::string(kReceiptSchema), schema);
      }
      return std::nullopt;
    }
    const bool* const eligible =
        root->at("production_residency_eligible").as_bool();
    if (eligible == nullptr) {
      diagnostic = make_diagnostic(
          PrefillA4ConverterErrorCode::kPublicationRejected,
          "receipt.production_residency_eligible", "expected JSON boolean");
      return std::nullopt;
    }
    receipt.production_residency_eligible = *eligible;
    if (mode == "production_calibrated") {
      receipt.mode = PrefillA4ConversionMode::kProductionCalibrated;
    } else if (mode == "experimental_nearest_even_smoke") {
      receipt.mode = PrefillA4ConversionMode::kExperimentalNearestEvenSmoke;
    } else {
      diagnostic = make_diagnostic(
          PrefillA4ConverterErrorCode::kPublicationRejected, "receipt.mode",
          "unsupported receipt mode", {}, mode);
      return std::nullopt;
    }
    if (receipt.physical_layout != kPrefillA4PhysicalLayout) {
      diagnostic = make_diagnostic(
          PrefillA4ConverterErrorCode::kPublicationRejected,
          "receipt.physical_layout",
          "unsupported A4 physical payload layout",
          std::string(kPrefillA4PhysicalLayout), receipt.physical_layout);
      return std::nullopt;
    }
    return receipt;
  } catch (const std::bad_alloc&) {
    diagnostic = make_diagnostic(
        PrefillA4ConverterErrorCode::kAllocationFailure, "receipt",
        "receipt allocation failed");
    return std::nullopt;
  } catch (...) {
    diagnostic = make_diagnostic(
        PrefillA4ConverterErrorCode::kPublicationRejected, "receipt",
        "unexpected receipt parsing failure");
    return std::nullopt;
  }
}

PrefillA4AuthenticatedPublication::~PrefillA4AuthenticatedPublication() {
  if (payload_fd_ >= 0) {
    (void)::close(payload_fd_);
  }
}

PrefillA4AuthenticatedPublication::PrefillA4AuthenticatedPublication(
    PrefillA4AuthenticatedPublication&& other) noexcept
    : payload_fd_(other.payload_fd_),
      receipt_(std::move(other.receipt_)),
      policy_(std::move(other.policy_)),
      device_id_(other.device_id_),
      inode_(other.inode_),
      file_size_(other.file_size_),
      modification_seconds_(other.modification_seconds_),
      modification_nanoseconds_(other.modification_nanoseconds_),
      change_seconds_(other.change_seconds_),
      change_nanoseconds_(other.change_nanoseconds_) {
  other.payload_fd_ = -1;
}

PrefillA4AuthenticatedPublication&
PrefillA4AuthenticatedPublication::operator=(
    PrefillA4AuthenticatedPublication&& other) noexcept {
  if (this != &other) {
    if (payload_fd_ >= 0) {
      (void)::close(payload_fd_);
    }
    payload_fd_ = other.payload_fd_;
    other.payload_fd_ = -1;
    receipt_ = std::move(other.receipt_);
    policy_ = std::move(other.policy_);
    device_id_ = other.device_id_;
    inode_ = other.inode_;
    file_size_ = other.file_size_;
    modification_seconds_ = other.modification_seconds_;
    modification_nanoseconds_ = other.modification_nanoseconds_;
    change_seconds_ = other.change_seconds_;
    change_nanoseconds_ = other.change_nanoseconds_;
  }
  return *this;
}

PrefillA4ConverterDiagnostic
PrefillA4AuthenticatedPublication::revalidate_unchanged_after_consumption()
    const {
  if (payload_fd_ < 0) {
    return make_diagnostic(PrefillA4ConverterErrorCode::kPublicationRejected,
                           "residency.payload",
                           "authenticated payload handle is empty");
  }
  FileSnapshot current;
  int error = 0;
  if (!capture_snapshot(payload_fd_, current, error)) {
    return make_diagnostic(PrefillA4ConverterErrorCode::kIoFailure,
                           "residency.payload",
                           "failed to revalidate held payload descriptor", {},
                           {}, error);
  }
  FileSnapshot expected;
  expected.device_id = device_id_;
  expected.inode = inode_;
  expected.size = file_size_;
  expected.modification_seconds = modification_seconds_;
  expected.modification_nanoseconds = modification_nanoseconds_;
  expected.change_seconds = change_seconds_;
  expected.change_nanoseconds = change_nanoseconds_;
  if (!same_snapshot(expected, current)) {
    return make_diagnostic(PrefillA4ConverterErrorCode::kDigestMismatch,
                           "residency.payload",
                           "held payload inode changed during consumption");
  }
  return {};
}

PrefillA4PublicationAuthenticationResult
authenticate_prefill_a4_publication_for_residency(
    const PrefillSidecarManifest& manifest,
    const PrefillA4PublicationReceipt& receipt,
    const fs::path& payload_path,
    const fs::path& calibration_policy_path) {
  PrefillA4PublicationAuthenticationResult result;
  try {
    const PrefillContractDiagnostic manifest_diagnostic =
        validate_prefill_sidecar_manifest(manifest);
    if (!manifest_diagnostic || manifest.kind != PrefillSidecarKind::kA4K64) {
      result.diagnostic = make_diagnostic(
          PrefillA4ConverterErrorCode::kInvalidManifest,
          "residency.manifest",
          "publication requires a valid A4-K64 manifest");
      return result;
    }
    if (receipt.version_major != kPrefillA4PublicationVersionMajor ||
        receipt.version_minor != kPrefillA4PublicationVersionMinor ||
        receipt.mode != PrefillA4ConversionMode::kProductionCalibrated ||
        !receipt.production_residency_eligible ||
        receipt.physical_layout != kPrefillA4PhysicalLayout) {
      result.diagnostic = make_diagnostic(
          PrefillA4ConverterErrorCode::kPublicationRejected,
          "residency.receipt",
          "experimental or unsupported publication cannot enter production");
      return result;
    }
    if (receipt.source_checkpoint_id != manifest.source_checkpoint_id ||
        receipt.source_config_sha256 != manifest.source_config_sha256 ||
        receipt.source_index_sha256 != manifest.source_index_sha256 ||
        receipt.manifest_sha256 != manifest.manifest_sha256 ||
        receipt.payload_bytes != manifest.summary.arena_bytes ||
        receipt.projection_count != manifest.summary.projection_count ||
        !lowercase_sha256(receipt.policy_sha256) ||
        receipt.policy_bytes == 0U ||
        receipt.policy_bytes > 16ULL * 1024ULL * 1024ULL ||
        !lowercase_sha256(receipt.payload_sha256)) {
      result.diagnostic = make_diagnostic(
          PrefillA4ConverterErrorCode::kSourceBindingMismatch,
          "residency.receipt",
          "receipt source/manifest/size binding differs from production contract");
      return result;
    }
    std::string policy_document;
    if (!read_bounded_file(calibration_policy_path, receipt.policy_bytes,
                           policy_document, result.diagnostic)) {
      return result;
    }
    if (policy_document.size() != receipt.policy_bytes) {
      result.diagnostic = make_diagnostic(
          PrefillA4ConverterErrorCode::kDigestMismatch,
          calibration_policy_path.string(),
          "calibration policy byte length mismatch",
          std::to_string(receipt.policy_bytes),
          std::to_string(policy_document.size()));
      return result;
    }
    PrefillA4CalibrationPolicyResult policy =
        parse_prefill_a4_calibration_policy(policy_document, manifest);
    if (!policy) {
      result.diagnostic = policy.diagnostic;
      return result;
    }
    if (policy.value->policy_sha256 != receipt.policy_sha256) {
      result.diagnostic = make_diagnostic(
          PrefillA4ConverterErrorCode::kDigestMismatch,
          calibration_policy_path.string(),
          "calibration policy SHA-256 differs from receipt",
          receipt.policy_sha256, policy.value->policy_sha256);
      return result;
    }
    const auto equalized = std::find_if(
        policy.value->projections.begin(), policy.value->projections.end(),
        [](const PrefillA4ProjectionCalibration& calibration) {
          return calibration.channel_equalization.has_value();
        });
    if (equalized != policy.value->projections.end()) {
      result.diagnostic = make_diagnostic(
          PrefillA4ConverterErrorCode::kUnsupportedCalibration,
          equalized->source_module + ".channel_equalization",
          "production residency rejects equalization until the runner retains authenticated inverse factors");
      return result;
    }

    UniqueFd payload(::open(payload_path.c_str(),
                            O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
    if (!payload) {
      result.diagnostic = make_diagnostic(
          PrefillA4ConverterErrorCode::kOpenFailed, payload_path.string(),
          "failed to open publication payload", {}, {}, errno);
      return result;
    }
    if (::flock(payload.get(), LOCK_SH | LOCK_NB) != 0) {
      result.diagnostic = make_diagnostic(
          PrefillA4ConverterErrorCode::kPublicationRejected,
          payload_path.string(),
          "payload is concurrently locked for mutation", {}, {}, errno);
      return result;
    }
    result.diagnostic = verify_regular_fd(
        payload.get(), receipt.payload_bytes, payload_path.string());
    if (!result.diagnostic) {
      return result;
    }
    struct stat status {};
    if (::fstat(payload.get(), &status) != 0 ||
        (status.st_mode & (S_IWUSR | S_IWGRP | S_IWOTH)) != 0 ||
        status.st_uid != ::geteuid() || status.st_nlink != 1) {
      result.diagnostic = make_diagnostic(
          PrefillA4ConverterErrorCode::kPublicationRejected,
          payload_path.string(),
          "payload must be owner-held, read-only, and singly linked", {}, {},
          errno);
      return result;
    }
    FileSnapshot before;
    int snapshot_error = 0;
    if (!capture_snapshot(payload.get(), before, snapshot_error)) {
      result.diagnostic = make_diagnostic(
          PrefillA4ConverterErrorCode::kIoFailure, payload_path.string(),
          "failed to snapshot publication payload", {}, {}, snapshot_error);
      return result;
    }
    std::string digest;
    result.diagnostic = hash_open_file(
        payload.get(), receipt.payload_bytes, digest);
    if (!result.diagnostic) {
      result.diagnostic.context = payload_path.string();
      return result;
    }
    FileSnapshot after;
    if (!capture_snapshot(payload.get(), after, snapshot_error) ||
        !same_snapshot(before, after)) {
      result.diagnostic = make_diagnostic(
          PrefillA4ConverterErrorCode::kDigestMismatch, payload_path.string(),
          "payload changed during authentication", {}, {}, snapshot_error);
      return result;
    }
    if (digest != receipt.payload_sha256) {
      result.diagnostic = make_diagnostic(
          PrefillA4ConverterErrorCode::kDigestMismatch, payload_path.string(),
          "published payload SHA-256 mismatch", receipt.payload_sha256,
          digest);
      return result;
    }

    PrefillA4AuthenticatedPublication authenticated;
    authenticated.payload_fd_ = payload.release();
    authenticated.receipt_ = receipt;
    authenticated.policy_ = std::move(*policy.value);
    authenticated.device_id_ = after.device_id;
    authenticated.inode_ = after.inode;
    authenticated.file_size_ = after.size;
    authenticated.modification_seconds_ = after.modification_seconds;
    authenticated.modification_nanoseconds_ = after.modification_nanoseconds;
    authenticated.change_seconds_ = after.change_seconds;
    authenticated.change_nanoseconds_ = after.change_nanoseconds;
    result.value.emplace(std::move(authenticated));
    return result;
  } catch (const std::bad_alloc&) {
    result.diagnostic = make_diagnostic(
        PrefillA4ConverterErrorCode::kAllocationFailure,
        "residency.publication", "authentication allocation failed");
    return result;
  } catch (...) {
    result.diagnostic = make_diagnostic(
        PrefillA4ConverterErrorCode::kPublicationRejected,
        "residency.publication", "unexpected publication authentication failure");
    return result;
  }
}

std::string_view to_string(const PrefillA4ConversionMode mode) noexcept {
  switch (mode) {
    case PrefillA4ConversionMode::kProductionCalibrated:
      return "production_calibrated";
    case PrefillA4ConversionMode::kExperimentalNearestEvenSmoke:
      return "experimental_nearest_even_smoke";
  }
  return "invalid";
}

std::string_view to_string(const PrefillA4Rounding rounding) noexcept {
  switch (rounding) {
    case PrefillA4Rounding::kNearestEvenV1:
      return "nearest_even_v1";
  }
  return "invalid";
}

std::string_view to_string(
    const PrefillA4ConverterErrorCode code) noexcept {
  switch (code) {
    case PrefillA4ConverterErrorCode::kNone:
      return "none";
    case PrefillA4ConverterErrorCode::kInvalidOption:
      return "invalid_option";
    case PrefillA4ConverterErrorCode::kInvalidManifest:
      return "invalid_manifest";
    case PrefillA4ConverterErrorCode::kInvalidPolicy:
      return "invalid_policy";
    case PrefillA4ConverterErrorCode::kPolicyCoverageMismatch:
      return "policy_coverage_mismatch";
    case PrefillA4ConverterErrorCode::kSourceBindingMismatch:
      return "source_binding_mismatch";
    case PrefillA4ConverterErrorCode::kUnsupportedCalibration:
      return "unsupported_calibration";
    case PrefillA4ConverterErrorCode::kUnsafePath:
      return "unsafe_path";
    case PrefillA4ConverterErrorCode::kOpenFailed:
      return "open_failed";
    case PrefillA4ConverterErrorCode::kIoFailure:
      return "io_failure";
    case PrefillA4ConverterErrorCode::kSourceAuthenticationFailed:
      return "source_authentication_failed";
    case PrefillA4ConverterErrorCode::kSourceTensorMismatch:
      return "source_tensor_mismatch";
    case PrefillA4ConverterErrorCode::kArithmeticOverflow:
      return "arithmetic_overflow";
    case PrefillA4ConverterErrorCode::kQuantizationFailure:
      return "quantization_failure";
    case PrefillA4ConverterErrorCode::kPublicationConflict:
      return "publication_conflict";
    case PrefillA4ConverterErrorCode::kPublicationRejected:
      return "publication_rejected";
    case PrefillA4ConverterErrorCode::kDigestMismatch:
      return "digest_mismatch";
    case PrefillA4ConverterErrorCode::kAllocationFailure:
      return "allocation_failure";
  }
  return "invalid";
}

}  // namespace q3x::runtime
