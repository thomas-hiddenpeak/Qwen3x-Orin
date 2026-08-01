#include "q3x/runtime/prefill_mlp_k512_overlay.h"

#include "q3x/core/sha256.h"
#include "q3x/io/json.h"
#include "q3x/quantization/nvfp4.h"
#include "q3x/runtime/prefill_quantized_contract.h"

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
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace q3x::runtime {
namespace {

namespace fs = std::filesystem;
namespace json = q3x::io::json;
namespace mw = q3x::model::weights;

constexpr std::string_view kManifestSchema =
    "q3x.prefill.mlp-k512.overlay-manifest";
constexpr std::string_view kPolicySchema =
    "q3x.prefill.mlp-k512.calibration-policy";
constexpr std::string_view kReceiptSchema =
    "q3x.prefill.mlp-k512.publication-receipt";

[[nodiscard]] PrefillMLPK512OverlayDiagnostic make_diagnostic(
    const PrefillMLPK512OverlayErrorCode code, std::string context,
    std::string message, std::string expected = {}, std::string actual = {},
    const int system_error = 0) {
  PrefillMLPK512OverlayDiagnostic result;
  result.code = code;
  result.context = std::move(context);
  result.message = std::move(message);
  result.expected = std::move(expected);
  result.actual = std::move(actual);
  result.system_error = system_error;
  return result;
}

[[nodiscard]] bool lower_sha256(const std::string_view value) noexcept {
  return value.size() == 64U &&
         std::all_of(value.begin(), value.end(), [](const char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

[[nodiscard]] bool valid_clip_ratio(const double value) noexcept {
  const float narrowed = static_cast<float>(value);
  return std::isfinite(value) && value >= kPrefillA4MinimumClipRatio &&
         value <= 1.0 && std::isfinite(narrowed) &&
         narrowed >= static_cast<float>(kPrefillA4MinimumClipRatio) &&
         narrowed <= 1.0F;
}

[[nodiscard]] bool checked_add(const std::uint64_t first,
                               const std::uint64_t second,
                               std::uint64_t& result) noexcept {
  if (second > std::numeric_limits<std::uint64_t>::max() - first) {
    return false;
  }
  result = first + second;
  return true;
}

class UniqueFd final {
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
  [[nodiscard]] explicit operator bool() const noexcept { return fd_ >= 0; }

 private:
  int fd_ = -1;
};

[[nodiscard]] bool pread_exact(const int fd, void* const destination,
                               const std::size_t bytes,
                               const std::uint64_t offset,
                               int& error) noexcept {
  if (offset > static_cast<std::uint64_t>(
                   std::numeric_limits<off_t>::max())) {
    error = EOVERFLOW;
    return false;
  }
  auto* output = static_cast<std::uint8_t*>(destination);
  std::size_t completed = 0U;
  while (completed < bytes) {
    const std::uint64_t position = offset + completed;
    if (position < offset ||
        position > static_cast<std::uint64_t>(
                       std::numeric_limits<off_t>::max())) {
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
  if (offset > static_cast<std::uint64_t>(
                   std::numeric_limits<off_t>::max())) {
    error = EOVERFLOW;
    return false;
  }
  const auto* input = static_cast<const std::uint8_t*>(source);
  std::size_t completed = 0U;
  while (completed < bytes) {
    const std::uint64_t position = offset + completed;
    if (position < offset ||
        position > static_cast<std::uint64_t>(
                       std::numeric_limits<off_t>::max())) {
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

[[nodiscard]] PrefillMLPK512OverlayDiagnostic hash_fd(
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
        return make_diagnostic(
            PrefillMLPK512OverlayErrorCode::kIoFailure, "sha256",
            "failed to read file while hashing", {}, {}, error);
      }
      if (!hash.update(buffer.data(), count)) {
        return make_diagnostic(
            PrefillMLPK512OverlayErrorCode::kArithmeticOverflow,
            "sha256", "SHA-256 byte count overflowed");
      }
      offset += count;
    }
    digest = hash.finalize().hex();
    return {};
  } catch (const std::bad_alloc&) {
    return make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kAllocationFailure, "sha256",
        "hash buffer allocation failed");
  }
}

[[nodiscard]] std::string sha256_text(const std::string_view text) {
  core::Sha256 hash;
  if (!hash.update(text.data(), text.size())) {
    return {};
  }
  return hash.finalize().hex();
}

void write_quoted(std::ostream& output, const std::string_view value) {
  output << '"';
  for (const unsigned char character : value) {
    switch (character) {
      case '"':
        output << "\\\"";
        break;
      case '\\':
        output << "\\\\";
        break;
      case '\b':
        output << "\\b";
        break;
      case '\f':
        output << "\\f";
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
        if (character < 0x20U) {
          output << "\\u00" << std::hex << std::setw(2)
                 << std::setfill('0') << static_cast<unsigned int>(character)
                 << std::dec << std::setfill(' ');
        } else {
          output << static_cast<char>(character);
        }
        break;
    }
  }
  output << '"';
}

void write_base(std::ostream& output,
                const PrefillMLPK512BaseBinding& base) {
  output << "{\"sidecar_kind\":\"a4_k128\",\"physical_layout\":";
  write_quoted(output, base.physical_layout);
  output << ",\"manifest_sha256\":";
  write_quoted(output, base.manifest_sha256);
  output << ",\"policy_sha256\":";
  write_quoted(output, base.policy_sha256);
  output << ",\"payload_sha256\":";
  write_quoted(output, base.payload_sha256);
  output << '}';
}

[[nodiscard]] bool valid_base(
    const PrefillMLPK512BaseBinding& base) noexcept {
  return base.physical_layout == kPrefillA4K128PhysicalLayout &&
         lower_sha256(base.manifest_sha256) &&
         lower_sha256(base.policy_sha256) &&
         lower_sha256(base.payload_sha256);
}

[[nodiscard]] std::string manifest_body(
    const PrefillMLPK512OverlayManifest& manifest) {
  std::ostringstream output;
  output << "schema=" << kManifestSchema << "\nversion="
         << manifest.version_major << '.' << manifest.version_minor
         << "\nlayout=" << manifest.physical_layout
         << "\ncheckpoint=" << manifest.source_checkpoint_id
         << "\nconfig=" << manifest.source_config_sha256
         << "\nindex=" << manifest.source_index_sha256 << "\nbase="
         << manifest.required_base.physical_layout << ':'
         << manifest.required_base.manifest_sha256 << ':'
         << manifest.required_base.policy_sha256 << ':'
         << manifest.required_base.payload_sha256 << "\npayload="
         << manifest.payload_bytes << '\n';
  for (const PrefillMLPK512OverlayEntry& entry :
       manifest.projections) {
    output << entry.ordinal << ':' << entry.layer_index << ':' << entry.family
           << ':' << entry.source_module << ':' << entry.source_sha256 << ':'
           << entry.output_size << ':' << entry.input_size << ':'
           << entry.sidecar_offset << ':' << entry.weight_bytes << ':'
           << entry.scale_bytes << '\n';
  }
  return output.str();
}

[[nodiscard]] bool expected_mlp_family(
    const PrefillProjectionFamily family) noexcept {
  return family == PrefillProjectionFamily::kMlpGate ||
         family == PrefillProjectionFamily::kMlpUp ||
         family == PrefillProjectionFamily::kMlpDown;
}

[[nodiscard]] std::string_view mlp_family_name(
    const PrefillProjectionFamily family) noexcept {
  switch (family) {
    case PrefillProjectionFamily::kMlpGate:
      return "gate";
    case PrefillProjectionFamily::kMlpUp:
      return "up";
    case PrefillProjectionFamily::kMlpDown:
      return "down";
    default:
      return {};
  }
}

[[nodiscard]] std::uint32_t float_bits(const float value) noexcept {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

[[nodiscard]] float bits_float(const std::uint32_t bits) noexcept {
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

[[nodiscard]] std::uint16_t float_to_bf16(const float value) noexcept {
  std::uint32_t bits = float_bits(value);
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] float bf16_to_float(const std::uint16_t value) noexcept {
  return bits_float(static_cast<std::uint32_t>(value) << 16U);
}

void write_little_u16(const std::uint16_t value,
                      std::uint8_t* const output) noexcept {
  output[0U] = static_cast<std::uint8_t>(value & 0xffU);
  output[1U] = static_cast<std::uint8_t>(value >> 8U);
}

[[nodiscard]] float read_little_f32(const std::uint8_t* const input) noexcept {
  const std::uint32_t bits = static_cast<std::uint32_t>(input[0U]) |
                             (static_cast<std::uint32_t>(input[1U]) << 8U) |
                             (static_cast<std::uint32_t>(input[2U]) << 16U) |
                             (static_cast<std::uint32_t>(input[3U]) << 24U);
  return bits_float(bits);
}

[[nodiscard]] int round_nearest_even(const float value) noexcept {
  const float floor_value = std::floor(value);
  const float fraction = value - floor_value;
  if (fraction < 0.5F) {
    return static_cast<int>(floor_value);
  }
  if (fraction > 0.5F) {
    return static_cast<int>(floor_value) + 1;
  }
  const int floor_integer = static_cast<int>(floor_value);
  return (floor_integer & 1) == 0 ? floor_integer : floor_integer + 1;
}

[[nodiscard]] bool exact_keys(
    const json::Value::Object& object,
    const std::initializer_list<std::string_view> keys) {
  if (object.size() != keys.size()) {
    return false;
  }
  return std::all_of(keys.begin(), keys.end(), [&object](const auto key) {
    return object.find(key) != object.end();
  });
}

[[nodiscard]] bool json_string(const json::Value::Object& object,
                               const std::string_view key,
                               std::string& output) {
  const auto found = object.find(key);
  const std::string* value =
      found == object.end() ? nullptr : found->second.as_string();
  if (value == nullptr) {
    return false;
  }
  output = *value;
  return true;
}

[[nodiscard]] bool json_uint(const json::Value::Object& object,
                             const std::string_view key,
                             std::uint64_t& output) {
  const auto found = object.find(key);
  const json::Number* value =
      found == object.end() ? nullptr : found->second.as_number();
  return value != nullptr && value->to_uint64(output);
}

[[nodiscard]] bool json_double(const json::Value::Object& object,
                               const std::string_view key, double& output) {
  const auto found = object.find(key);
  const json::Number* value =
      found == object.end() ? nullptr : found->second.as_number();
  return value != nullptr && value->to_double(output) && std::isfinite(output);
}

[[nodiscard]] bool parse_version(const json::Value& value,
                                 std::uint32_t& major,
                                 std::uint32_t& minor) {
  const auto* object = value.as_object();
  std::uint64_t major_value = 0U;
  std::uint64_t minor_value = 0U;
  if (object == nullptr || !exact_keys(*object, {"major", "minor"}) ||
      !json_uint(*object, "major", major_value) ||
      !json_uint(*object, "minor", minor_value) ||
      major_value > std::numeric_limits<std::uint32_t>::max() ||
      minor_value > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  major = static_cast<std::uint32_t>(major_value);
  minor = static_cast<std::uint32_t>(minor_value);
  return true;
}

[[nodiscard]] bool parse_base(const json::Value& value,
                              PrefillMLPK512BaseBinding& base) {
  const auto* object = value.as_object();
  std::string kind;
  return object != nullptr &&
         exact_keys(*object,
                    {"sidecar_kind", "physical_layout", "manifest_sha256",
                     "policy_sha256", "payload_sha256"}) &&
         json_string(*object, "sidecar_kind", kind) && kind == "a4_k128" &&
         json_string(*object, "physical_layout", base.physical_layout) &&
         json_string(*object, "manifest_sha256", base.manifest_sha256) &&
         json_string(*object, "policy_sha256", base.policy_sha256) &&
         json_string(*object, "payload_sha256", base.payload_sha256) &&
         valid_base(base);
}

[[nodiscard]] bool same_base(
    const PrefillMLPK512BaseBinding& first,
    const PrefillMLPK512BaseBinding& second) noexcept {
  return first.physical_layout == second.physical_layout &&
         first.manifest_sha256 == second.manifest_sha256 &&
         first.policy_sha256 == second.policy_sha256 &&
         first.payload_sha256 == second.payload_sha256;
}

[[nodiscard]] bool read_bounded_file(
    const fs::path& path, const std::uint64_t maximum_bytes,
    std::string& output, PrefillMLPK512OverlayDiagnostic& diagnostic) {
  UniqueFd input(::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  if (!input) {
    diagnostic = make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kIoFailure, path.string(),
        "failed to open regular non-symlink input", {}, {}, errno);
    return false;
  }
  struct stat status {};
  if (::fstat(input.get(), &status) != 0 || !S_ISREG(status.st_mode) ||
      status.st_size < 0 ||
      static_cast<std::uint64_t>(status.st_size) > maximum_bytes ||
      static_cast<std::uint64_t>(status.st_size) >
          std::numeric_limits<std::size_t>::max()) {
    diagnostic = make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kIoFailure, path.string(),
        "bounded regular-file check failed", {}, {}, errno);
    return false;
  }
  try {
    output.resize(static_cast<std::size_t>(status.st_size));
  } catch (const std::bad_alloc&) {
    diagnostic = make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kAllocationFailure,
        path.string(), "input allocation failed");
    return false;
  }
  int error = 0;
  if (!output.empty() &&
      !pread_exact(input.get(), output.data(), output.size(), 0U, error)) {
    diagnostic = make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kIoFailure, path.string(),
        "bounded input read failed", {}, {}, error);
    return false;
  }
  return true;
}

}  // namespace

PrefillMLPK512OverlayManifestResult
build_qwen36_27b_prefill_mlp_k512_overlay_manifest(
    const mw::WeightManifest& source_manifest,
    const std::vector<ShardIdentity>& authenticated_shards,
    const PrefillMLPK512BaseBinding& required_base) {
  PrefillMLPK512OverlayManifestResult result;
  if (!valid_base(required_base)) {
    result.diagnostic = make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kInvalidManifest,
        "required_base", "valid authenticated A4-K128 binding is required");
    return result;
  }
  PrefillSidecarManifestOptions options;
  options.kind = PrefillSidecarKind::kA4K128;
  const PrefillSidecarManifestResult full =
      build_qwen36_27b_prefill_sidecar_manifest(source_manifest,
                                                 authenticated_shards,
                                                 options);
  if (!full) {
    result.diagnostic = make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kInvalidManifest,
        full.diagnostic.context, full.diagnostic.message);
    return result;
  }
  if (required_base.manifest_sha256 != full.value->manifest_sha256) {
    result.diagnostic = make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kInvalidManifest,
        "required_base.manifest_sha256",
        "base K128 manifest is not the one rebuilt from the original checkpoint",
        full.value->manifest_sha256, required_base.manifest_sha256);
    return result;
  }

  PrefillMLPK512OverlayManifest manifest;
  manifest.physical_layout = std::string(kPrefillMLPK512OverlayLayout);
  manifest.source_checkpoint_id = full.value->source_checkpoint_id;
  manifest.source_config_sha256 = full.value->source_config_sha256;
  manifest.source_index_sha256 = full.value->source_index_sha256;
  manifest.required_base = required_base;
  manifest.payload_bytes = kPrefillMLPK512OverlayPayloadBytes;
  manifest.projections.reserve(kPrefillMLPK512OverlayProjectionCount);
  for (const PrefillProjectionSidecarEntry& source :
       full.value->projections) {
    if (!expected_mlp_family(source.family)) {
      continue;
    }
    const std::uint32_t ordinal =
        static_cast<std::uint32_t>(manifest.projections.size());
    PrefillMLPK512OverlayEntry entry;
    entry.ordinal = ordinal;
    entry.layer_index = source.layer_index;
    entry.family = std::string(mlp_family_name(source.family));
    entry.source_module = source.source_module;
    entry.source_sha256 = source.source_sha256;
    const bool down = source.family == PrefillProjectionFamily::kMlpDown;
    entry.output_size = down ? kPrefillMLPK512OverlayDownOutputSize
                             : kPrefillMLPK512OverlayGateUpOutputSize;
    entry.input_size = down ? kPrefillMLPK512OverlayDownInputSize
                            : kPrefillMLPK512OverlayGateUpInputSize;
    entry.sidecar_offset =
        static_cast<std::uint64_t>(ordinal) *
        kPrefillMLPK512OverlayProjectionBytes;
    entry.weight_bytes = kPrefillMLPK512OverlayProjectionWeightBytes;
    entry.scale_bytes = kPrefillMLPK512OverlayProjectionScaleBytes;
    manifest.projections.emplace_back(std::move(entry));
  }
  manifest.manifest_sha256 = sha256_text(manifest_body(manifest));
  result.diagnostic =
      validate_prefill_mlp_k512_overlay_manifest(manifest);
  if (!result.diagnostic) {
    return result;
  }
  result.value.emplace(std::move(manifest));
  return result;
}

PrefillMLPK512OverlayDiagnostic
validate_prefill_mlp_k512_overlay_manifest(
    const PrefillMLPK512OverlayManifest& manifest) {
  if (manifest.version_major != kPrefillMLPK512OverlayVersionMajor ||
      manifest.version_minor != kPrefillMLPK512OverlayVersionMinor ||
      manifest.physical_layout != kPrefillMLPK512OverlayLayout ||
      manifest.source_checkpoint_id.empty() ||
      !lower_sha256(manifest.source_config_sha256) ||
      !lower_sha256(manifest.source_index_sha256) ||
      !valid_base(manifest.required_base) ||
      manifest.projections.size() !=
          kPrefillMLPK512OverlayProjectionCount ||
      manifest.payload_bytes != kPrefillMLPK512OverlayPayloadBytes ||
      !lower_sha256(manifest.manifest_sha256)) {
    return make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kInvalidManifest, "manifest",
        "K512 MLP manifest header is invalid");
  }
  for (std::size_t index = 0U; index < manifest.projections.size(); ++index) {
    const auto& entry = manifest.projections[index];
    const std::size_t layer = index / 3U;
    const std::size_t position = index % 3U;
    const bool down = position == 2U;
    const std::string_view family =
        position == 0U ? std::string_view("gate")
                       : (position == 1U ? std::string_view("up")
                                         : std::string_view("down"));
    const std::string_view projection =
        position == 0U ? std::string_view("gate_proj")
                       : (position == 1U ? std::string_view("up_proj")
                                         : std::string_view("down_proj"));
    const std::string expected_module =
        "model.language_model.layers." + std::to_string(layer) +
        ".mlp." + std::string(projection);
    if (entry.ordinal != index || entry.layer_index != layer ||
        entry.family != family ||
        entry.source_module != expected_module ||
        !lower_sha256(entry.source_sha256) ||
        entry.output_size !=
            (down ? kPrefillMLPK512OverlayDownOutputSize
                  : kPrefillMLPK512OverlayGateUpOutputSize) ||
        entry.input_size !=
            (down ? kPrefillMLPK512OverlayDownInputSize
                  : kPrefillMLPK512OverlayGateUpInputSize) ||
        entry.sidecar_offset !=
            index * kPrefillMLPK512OverlayProjectionBytes ||
        entry.weight_bytes !=
            kPrefillMLPK512OverlayProjectionWeightBytes ||
        entry.scale_bytes !=
            kPrefillMLPK512OverlayProjectionScaleBytes) {
      return make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kInvalidManifest,
          "manifest.projections[" + std::to_string(index) + "]",
          "fixed gate/up/down projection inventory or byte layout is invalid");
    }
  }
  const std::string digest = sha256_text(manifest_body(manifest));
  if (digest != manifest.manifest_sha256) {
    return make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kDigestMismatch,
        "manifest.manifest_sha256", "manifest digest mismatch",
        manifest.manifest_sha256, digest);
  }
  return {};
}

PrefillMLPK512OverlayDiagnostic
quantize_prefill_mlp_k512_consumer_blocks(
    const float* const source_rows, const std::size_t row_count,
    const std::size_t input_size, const double weight_clip_ratio,
    std::uint8_t* const packed_signed_w4,
    const std::size_t packed_signed_w4_bytes,
    std::uint8_t* const bf16_scales_little_endian,
    const std::size_t bf16_scale_bytes) {
  if (source_rows == nullptr || packed_signed_w4 == nullptr ||
      bf16_scales_little_endian == nullptr || row_count == 0U ||
      row_count % 64U != 0U || input_size == 0U || input_size % 512U != 0U ||
      !valid_clip_ratio(weight_clip_ratio) ||
      input_size > std::numeric_limits<std::size_t>::max() / row_count) {
    return make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kInvalidOption,
        "a4_k512_quantize", "N64/K512 input or explicit clip is invalid");
  }
  const std::size_t elements = row_count * input_size;
  const std::size_t expected_weights = elements / 2U;
  const std::size_t expected_scales = elements / 512U * 2U;
  if (packed_signed_w4_bytes != expected_weights ||
      bf16_scale_bytes != expected_scales) {
    return make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kInvalidOption,
        "a4_k512_quantize", "output sizes differ from K512 ABI",
        std::to_string(expected_weights) + ":" +
            std::to_string(expected_scales),
        std::to_string(packed_signed_w4_bytes) + ":" +
            std::to_string(bf16_scale_bytes));
  }
  const std::size_t n_blocks = row_count / 64U;
  const std::size_t k64_blocks = input_size / 64U;
  const std::size_t k512_blocks = input_size / 512U;
  for (std::size_t n_block = 0U; n_block < n_blocks; ++n_block) {
    for (std::size_t group = 0U; group < k512_blocks; ++group) {
      for (std::size_t local_n = 0U; local_n < 64U; ++local_n) {
        const std::size_t row = n_block * 64U + local_n;
        const float* input = source_rows + row * input_size;
        const std::size_t begin = group * 512U;
        float maximum = 0.0F;
        for (std::size_t column = 0U; column < 512U; ++column) {
          const float value = input[begin + column];
          if (!std::isfinite(value)) {
            return make_diagnostic(
                PrefillMLPK512OverlayErrorCode::kQuantizationFailure,
                "a4_k512_quantize", "source contains non-finite value", {},
                std::to_string(row * input_size + begin + column));
          }
          maximum = std::max(maximum, std::fabs(value));
        }
        const float threshold =
            maximum * static_cast<float>(weight_clip_ratio);
        std::uint16_t scale_bits =
            float_to_bf16(maximum == 0.0F ? 1.0F : threshold / 7.0F);
        float stored_scale = bf16_to_float(scale_bits);
        if (maximum != 0.0F && stored_scale == 0.0F) {
          scale_bits = 1U;
          stored_scale = bf16_to_float(scale_bits);
        }
        const std::size_t scale_offset =
            (((n_block * k512_blocks + group) * 64U + local_n) * 2U);
        write_little_u16(scale_bits,
                         bf16_scales_little_endian + scale_offset);
        for (std::size_t plane = 0U; plane < 8U; ++plane) {
          const std::size_t physical_group = group * 8U + plane;
          std::uint8_t* packed =
              packed_signed_w4 +
              (((n_block * k64_blocks + physical_group) * 64U + local_n) *
               32U);
          for (std::size_t pair = 0U; pair < 32U; ++pair) {
            std::uint8_t encoded = 0U;
            for (std::size_t lane = 0U; lane < 2U; ++lane) {
              const float value =
                  input[begin + plane * 64U + pair * 2U + lane];
              const float clipped =
                  std::max(-threshold, std::min(threshold, value));
              int code = stored_scale == 0.0F
                             ? 0
                             : round_nearest_even(clipped / stored_scale);
              code = std::max(-7, std::min(7, code));
              encoded |= static_cast<std::uint8_t>(
                  (static_cast<unsigned int>(code) & 0x0fU) << (4U * lane));
            }
            packed[pair] = encoded;
          }
        }
      }
    }
  }
  return {};
}

namespace {

[[nodiscard]] std::string serialize_policy(
    const PrefillMLPK512OverlayManifest& manifest,
    const double weight_clip_ratio, const double activation_clip_ratio) {
  std::ostringstream output;
  output << std::setprecision(17)
         << "{\n  \"schema\": \"" << kPolicySchema
         << "\",\n  \"version\": {\"major\": 1, \"minor\": 0},"
            "\n  \"mode\": \"production_calibrated\","
            "\n  \"physical_layout\": ";
  write_quoted(output, manifest.physical_layout);
  output << ",\n  \"packed_k_group_size\": 64,"
            "\n  \"scale_group_size\": 512,"
            "\n  \"source_checkpoint_id\": ";
  write_quoted(output, manifest.source_checkpoint_id);
  output << ",\n  \"source_config_sha256\": ";
  write_quoted(output, manifest.source_config_sha256);
  output << ",\n  \"source_index_sha256\": ";
  write_quoted(output, manifest.source_index_sha256);
  output << ",\n  \"manifest_sha256\": ";
  write_quoted(output, manifest.manifest_sha256);
  output << ",\n  \"required_base\": ";
  write_base(output, manifest.required_base);
  output << ",\n  \"projections\": [\n";
  for (std::size_t index = 0U; index < manifest.projections.size(); ++index) {
    const auto& entry = manifest.projections[index];
    output << "    {\"ordinal\":" << entry.ordinal
           << ",\"source_module\":";
    write_quoted(output, entry.source_module);
    output << ",\"source_sha256\":";
    write_quoted(output, entry.source_sha256);
    output << ",\"weight_clip_ratio\":" << weight_clip_ratio
           << ",\"activation_clip_ratio\":" << activation_clip_ratio
           << ",\"activation_scale_group_size\":512,"
              "\"rounding\":\"nearest_even_v1\"}";
    output << (index + 1U == manifest.projections.size() ? "\n" : ",\n");
  }
  output << "  ]\n}\n";
  return output.str();
}

[[nodiscard]] PrefillMLPK512OverlayDiagnostic publish_read_only_file(
    const fs::path& target, const std::string_view bytes) {
  if (target.empty()) {
    return make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kInvalidOption, "output",
        "non-empty output path is required");
  }
  struct stat existing {};
  if (::lstat(target.c_str(), &existing) == 0 || errno != ENOENT) {
    return make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kPublicationConflict,
        target.string(), "publication target already exists", {}, {},
        errno == ENOENT ? 0 : errno);
  }
  const fs::path parent =
      target.parent_path().empty() ? fs::path(".") : target.parent_path();
  const fs::path temporary =
      fs::path(target.string() + ".tmp." + std::to_string(::getpid()));
  UniqueFd output(::open(temporary.c_str(),
                         O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
                         S_IRUSR | S_IWUSR));
  if (!output) {
    return make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kIoFailure,
        temporary.string(), "failed to create temporary publication", {},
        {}, errno);
  }
  int error = 0;
  if ((!bytes.empty() &&
       !pwrite_exact(output.get(), bytes.data(), bytes.size(), 0U, error)) ||
      ::fchmod(output.get(), S_IRUSR) != 0 || ::fsync(output.get()) != 0) {
    const int saved = error != 0 ? error : errno;
    (void)::unlink(temporary.c_str());
    return make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kIoFailure,
        temporary.string(), "failed to seal temporary publication", {}, {},
        saved);
  }
  if (::link(temporary.c_str(), target.c_str()) != 0) {
    const int saved = errno;
    (void)::unlink(temporary.c_str());
    return make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kPublicationConflict,
        target.string(), "atomic no-replace publication failed", {}, {},
        saved);
  }
  if (::unlink(temporary.c_str()) != 0) {
    const int saved = errno;
    (void)::unlink(target.c_str());
    return make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kIoFailure, target.string(),
        "temporary unlink failed; publication rolled back", {}, {}, saved);
  }
  UniqueFd directory(::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC |
                                                O_NOFOLLOW));
  if (!directory || ::fsync(directory.get()) != 0) {
    const int saved = errno;
    (void)::unlink(target.c_str());
    return make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kIoFailure, parent.string(),
        "directory sync failed; publication rolled back", {}, {}, saved);
  }
  return {};
}

}  // namespace

PrefillMLPK512OverlayPolicyResult
parse_prefill_mlp_k512_overlay_policy(
    const std::string_view document,
    const PrefillMLPK512OverlayManifest& manifest) {
  PrefillMLPK512OverlayPolicyResult result;
  const PrefillMLPK512OverlayDiagnostic manifest_diagnostic =
      validate_prefill_mlp_k512_overlay_manifest(manifest);
  if (!manifest_diagnostic) {
    result.diagnostic = manifest_diagnostic;
    return result;
  }
  try {
    json::ParseOptions options;
    options.max_input_bytes = 4U * 1024U * 1024U;
    options.max_nesting_depth = 12U;
    options.max_values = 2'000U;
    options.max_container_items = 2'000U;
    const json::ParseResult parsed = json::parse(document, options);
    const auto* root = parsed ? parsed.value->as_object() : nullptr;
    if (root == nullptr ||
        !exact_keys(*root,
                    {"schema", "version", "mode", "physical_layout",
                     "packed_k_group_size", "scale_group_size",
                     "source_checkpoint_id", "source_config_sha256",
                     "source_index_sha256", "manifest_sha256",
                     "required_base", "projections"})) {
      result.diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kInvalidPolicy, "policy",
          "strict policy JSON schema mismatch");
      return result;
    }
    PrefillMLPK512OverlayPolicy policy;
    std::string schema;
    std::string mode;
    std::uint64_t packed_k = 0U;
    std::uint64_t scale_k = 0U;
    if (!json_string(*root, "schema", schema) || schema != kPolicySchema ||
        !parse_version(root->at("version"), policy.version_major,
                       policy.version_minor) ||
        policy.version_major != kPrefillMLPK512OverlayVersionMajor ||
        policy.version_minor != kPrefillMLPK512OverlayVersionMinor ||
        !json_string(*root, "mode", mode) ||
        mode != "production_calibrated" ||
        !json_string(*root, "physical_layout", policy.physical_layout) ||
        policy.physical_layout != manifest.physical_layout ||
        !json_uint(*root, "packed_k_group_size", packed_k) ||
        packed_k != kPrefillMLPK512OverlayPackedK ||
        !json_uint(*root, "scale_group_size", scale_k) ||
        scale_k != kPrefillMLPK512OverlayScaleK ||
        !json_string(*root, "source_checkpoint_id",
                     policy.source_checkpoint_id) ||
        policy.source_checkpoint_id != manifest.source_checkpoint_id ||
        !json_string(*root, "source_config_sha256",
                     policy.source_config_sha256) ||
        policy.source_config_sha256 != manifest.source_config_sha256 ||
        !json_string(*root, "source_index_sha256",
                     policy.source_index_sha256) ||
        policy.source_index_sha256 != manifest.source_index_sha256 ||
        !json_string(*root, "manifest_sha256", policy.manifest_sha256) ||
        policy.manifest_sha256 != manifest.manifest_sha256 ||
        !parse_base(root->at("required_base"), policy.required_base) ||
        !same_base(policy.required_base, manifest.required_base)) {
      result.diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kInvalidPolicy, "policy",
          "policy identity or fixed K512 ABI differs from manifest");
      return result;
    }
    const auto* projections = root->at("projections").as_array();
    if (projections == nullptr || projections->size() !=
                                      manifest.projections.size()) {
      result.diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kInvalidPolicy,
          "policy.projections",
          "policy must cover all 192 gate/up/down projections");
      return result;
    }
    policy.projections.reserve(projections->size());
    for (std::size_t index = 0U; index < projections->size(); ++index) {
      const auto* object = (*projections)[index].as_object();
      PrefillMLPK512OverlayCalibration calibration;
      std::uint64_t ordinal = 0U;
      std::uint64_t activation_group = 0U;
      std::string rounding;
      if (object == nullptr ||
          !exact_keys(*object,
                      {"ordinal", "source_module", "source_sha256",
                       "weight_clip_ratio", "activation_clip_ratio",
                       "activation_scale_group_size", "rounding"}) ||
          !json_uint(*object, "ordinal", ordinal) || ordinal != index ||
          !json_string(*object, "source_module", calibration.source_module) ||
          calibration.source_module !=
              manifest.projections[index].source_module ||
          !json_string(*object, "source_sha256", calibration.source_sha256) ||
          calibration.source_sha256 !=
              manifest.projections[index].source_sha256 ||
          !json_double(*object, "weight_clip_ratio",
                       calibration.weight_clip_ratio) ||
          !valid_clip_ratio(calibration.weight_clip_ratio) ||
          !json_double(*object, "activation_clip_ratio",
                       calibration.activation_clip_ratio) ||
          !valid_clip_ratio(calibration.activation_clip_ratio) ||
          !json_uint(*object, "activation_scale_group_size",
                     activation_group) ||
          activation_group != kPrefillMLPK512OverlayScaleK ||
          !json_string(*object, "rounding", rounding) ||
          rounding != "nearest_even_v1") {
        result.diagnostic = make_diagnostic(
            PrefillMLPK512OverlayErrorCode::kInvalidPolicy,
            "policy.projections[" + std::to_string(index) + "]",
            "calibration entry differs from fixed source-bound K512 ABI");
        return result;
      }
      calibration.ordinal = static_cast<std::uint32_t>(ordinal);
      calibration.activation_scale_group_size =
          static_cast<std::uint32_t>(activation_group);
      policy.projections.emplace_back(std::move(calibration));
    }
    for (std::size_t layer = 0U;
         layer < policy.projections.size() / 3U; ++layer) {
      const std::size_t gate_index = 3U * layer;
      const std::size_t up_index = gate_index + 1U;
      const float gate_clip = static_cast<float>(
          policy.projections[gate_index].activation_clip_ratio);
      const float up_clip = static_cast<float>(
          policy.projections[up_index].activation_clip_ratio);
      if (gate_clip != up_clip) {
        result.diagnostic = make_diagnostic(
            PrefillMLPK512OverlayErrorCode::kInvalidPolicy,
            "policy.projections[" + std::to_string(gate_index) + "," +
                std::to_string(up_index) + "]",
            "fused Gate+Up requires one shared activation clip ratio after "
            "float narrowing",
            std::to_string(gate_clip), std::to_string(up_clip));
        return result;
      }
    }
    policy.policy_sha256 = sha256_text(document);
    policy.policy_bytes = document.size();
    result.value.emplace(std::move(policy));
    return result;
  } catch (const std::bad_alloc&) {
    result.diagnostic = make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kAllocationFailure, "policy",
        "policy allocation failed");
    return result;
  } catch (...) {
    result.diagnostic = make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kInvalidPolicy, "policy",
        "unexpected policy parse failure");
    return result;
  }
}

PrefillMLPK512OverlayPolicyResult
write_prefill_mlp_k512_overlay_policy_template(
    const PrefillMLPK512OverlayManifest& manifest,
    const fs::path& output_path, const double weight_clip_ratio,
    const double activation_clip_ratio) {
  PrefillMLPK512OverlayPolicyResult result;
  const auto manifest_diagnostic =
      validate_prefill_mlp_k512_overlay_manifest(manifest);
  if (!manifest_diagnostic || output_path.empty() ||
      !valid_clip_ratio(weight_clip_ratio) ||
      !valid_clip_ratio(activation_clip_ratio)) {
    result.diagnostic = manifest_diagnostic
                            ? make_diagnostic(
                                  PrefillMLPK512OverlayErrorCode::
                                      kInvalidOption,
                                  "policy_template",
                                  "output and two explicit clips are required")
                            : manifest_diagnostic;
    return result;
  }
  const std::string document = serialize_policy(
      manifest, weight_clip_ratio, activation_clip_ratio);
  result = parse_prefill_mlp_k512_overlay_policy(document, manifest);
  if (!result) {
    return result;
  }
  result.diagnostic = publish_read_only_file(output_path, document);
  if (!result.diagnostic) {
    result.value.reset();
  }
  return result;
}

PrefillMLPK512OverlayPolicyResult
write_qwen36_27b_prefill_mlp_k512_overlay_policy_template(
    const PrefillMLPK512OverlayPolicyTemplateOptions& options) {
  PrefillMLPK512OverlayPolicyResult result;
  if (options.model_directory.empty() ||
      options.base_k128_receipt_path.empty() || options.output_path.empty() ||
      !valid_clip_ratio(options.weight_clip_ratio) ||
      !valid_clip_ratio(options.activation_clip_ratio)) {
    result.diagnostic = make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kInvalidOption,
        "policy_template",
        "model/base receipt/output and two explicit clip ratios are required");
    return result;
  }
  std::string base_document;
  if (!read_bounded_file(options.base_k128_receipt_path,
                         1ULL * 1024ULL * 1024ULL, base_document,
                         result.diagnostic)) {
    return result;
  }
  PrefillA4ConverterDiagnostic base_diagnostic;
  const std::optional<PrefillA4PublicationReceipt> base_receipt =
      parse_prefill_a4_publication_receipt(base_document, base_diagnostic);
  if (!base_receipt.has_value() || !base_diagnostic ||
      !base_receipt->production_residency_eligible ||
      base_receipt->sidecar_kind != PrefillSidecarKind::kA4K128 ||
      base_receipt->packed_k_group_size != 64U ||
      base_receipt->scale_group_size != 128U ||
      base_receipt->physical_layout != kPrefillA4K128PhysicalLayout) {
    result.diagnostic = make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kInvalidReceipt,
        options.base_k128_receipt_path.string(),
        "base receipt is not an authenticated production A4-K128 ABI");
    return result;
  }
  PrefillMLPK512BaseBinding base;
  base.physical_layout = base_receipt->physical_layout;
  base.manifest_sha256 = base_receipt->manifest_sha256;
  base.policy_sha256 = base_receipt->policy_sha256;
  base.payload_sha256 = base_receipt->payload_sha256;
  const mw::ManifestResult source =
      mw::build_qwen36_27b_text_manifest(options.model_directory);
  if (!source) {
    result.diagnostic = make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kInvalidManifest,
        options.model_directory.string(),
        "pinned checkpoint manifest validation failed");
    return result;
  }
  const auto built =
      build_qwen36_27b_prefill_mlp_k512_overlay_manifest(
          *source.value, pinned_qwen36_27b_shards(), base);
  if (!built) {
    result.diagnostic = built.diagnostic;
    return result;
  }
  return write_prefill_mlp_k512_overlay_policy_template(
      *built.value, options.output_path, options.weight_clip_ratio,
      options.activation_clip_ratio);
}

namespace {

[[nodiscard]] std::string serialize_receipt(
    const PrefillMLPK512OverlayReceipt& receipt) {
  std::ostringstream output;
  output << "{\"schema\":\"" << kReceiptSchema
         << "\",\"version\":{\"major\":1,\"minor\":0},"
            "\"mode\":\"production_calibrated\","
            "\"production_residency_eligible\":true,"
            "\"physical_layout\":";
  write_quoted(output, receipt.physical_layout);
  output << ",\"packed_k_group_size\":64,\"scale_group_size\":512,"
            "\"source_checkpoint_id\":";
  write_quoted(output, receipt.source_checkpoint_id);
  output << ",\"source_config_sha256\":";
  write_quoted(output, receipt.source_config_sha256);
  output << ",\"source_index_sha256\":";
  write_quoted(output, receipt.source_index_sha256);
  output << ",\"manifest_sha256\":";
  write_quoted(output, receipt.manifest_sha256);
  output << ",\"policy_sha256\":";
  write_quoted(output, receipt.policy_sha256);
  output << ",\"policy_bytes\":" << receipt.policy_bytes
         << ",\"required_base\":";
  write_base(output, receipt.required_base);
  output << ",\"payload_sha256\":";
  write_quoted(output, receipt.payload_sha256);
  output << ",\"payload_bytes\":" << receipt.payload_bytes
         << ",\"projection_count\":" << receipt.projection_count << "}\n";
  return output.str();
}

[[nodiscard]] bool extract_policy_base(
    const std::string_view document,
    PrefillMLPK512BaseBinding& base,
    PrefillMLPK512OverlayDiagnostic& diagnostic) {
  json::ParseOptions options;
  options.max_input_bytes = 4U * 1024U * 1024U;
  options.max_nesting_depth = 12U;
  options.max_values = 2'000U;
  options.max_container_items = 2'000U;
  const json::ParseResult parsed = json::parse(document, options);
  const auto* root = parsed ? parsed.value->as_object() : nullptr;
  if (root == nullptr) {
    diagnostic = make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kInvalidPolicy, "policy",
        "policy JSON parse failed before manifest reconstruction");
    return false;
  }
  const auto found = root->find("required_base");
  if (found == root->end() || !parse_base(found->second, base)) {
    diagnostic = make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kInvalidPolicy,
        "policy.required_base", "valid A4-K128 base binding is required");
    return false;
  }
  return true;
}

struct FileSnapshot final {
  std::uint64_t device = 0U;
  std::uint64_t inode = 0U;
  std::uint64_t size = 0U;
  std::int64_t mtime_s = 0;
  std::int64_t mtime_ns = 0;
  std::int64_t ctime_s = 0;
  std::int64_t ctime_ns = 0;
};

[[nodiscard]] bool snapshot_fd(const int fd, FileSnapshot& output,
                               int& error) noexcept {
  struct stat status {};
  if (::fstat(fd, &status) != 0 || !S_ISREG(status.st_mode) ||
      status.st_size < 0) {
    error = errno != 0 ? errno : EINVAL;
    return false;
  }
  output.device = status.st_dev;
  output.inode = status.st_ino;
  output.size = static_cast<std::uint64_t>(status.st_size);
  output.mtime_s = status.st_mtim.tv_sec;
  output.mtime_ns = status.st_mtim.tv_nsec;
  output.ctime_s = status.st_ctim.tv_sec;
  output.ctime_ns = status.st_ctim.tv_nsec;
  return true;
}

[[nodiscard]] bool same_snapshot(const FileSnapshot& first,
                                 const FileSnapshot& second) noexcept {
  return first.device == second.device && first.inode == second.inode &&
         first.size == second.size && first.mtime_s == second.mtime_s &&
         first.mtime_ns == second.mtime_ns && first.ctime_s == second.ctime_s &&
         first.ctime_ns == second.ctime_ns;
}

struct AuthenticatedSources final {
  UniqueFd root;
  std::map<std::string, UniqueFd, std::less<>> shards;
  std::map<std::string, FileSnapshot, std::less<>> snapshots;
};

[[nodiscard]] PrefillMLPK512OverlayDiagnostic authenticate_sources(
    const fs::path& directory, AuthenticatedSources& sources,
    PrefillMLPK512OverlayConversionStats& stats) {
  sources.root = UniqueFd(::open(directory.c_str(),
                                 O_RDONLY | O_DIRECTORY | O_CLOEXEC |
                                     O_NOFOLLOW));
  if (!sources.root) {
    return make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kSourceAuthenticationFailed,
        directory.string(), "failed to open checkpoint directory", {}, {},
        errno);
  }
  for (const ShardIdentity& identity : pinned_qwen36_27b_shards()) {
    UniqueFd shard(::openat(sources.root.get(), identity.filename.c_str(),
                            O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
    if (!shard || ::flock(shard.get(), LOCK_SH | LOCK_NB) != 0) {
      return make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kSourceAuthenticationFailed,
          identity.filename, "failed to lock pinned source shard", {}, {},
          errno);
    }
    FileSnapshot before;
    FileSnapshot after;
    int error = 0;
    if (!snapshot_fd(shard.get(), before, error) ||
        before.size != identity.file_size) {
      return make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kSourceAuthenticationFailed,
          identity.filename, "source shard size differs from pinned identity",
          std::to_string(identity.file_size), std::to_string(before.size),
          error);
    }
    std::string digest;
    auto diagnostic = hash_fd(shard.get(), identity.file_size, digest);
    stats.source_bytes_read += identity.file_size;
    if (!diagnostic) {
      return diagnostic;
    }
    if (digest != identity.sha256 ||
        !snapshot_fd(shard.get(), after, error) ||
        !same_snapshot(before, after)) {
      return make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kSourceAuthenticationFailed,
          identity.filename,
          "source shard digest or immutable snapshot differs from pin",
          identity.sha256, digest, error);
    }
    sources.snapshots.emplace(identity.filename, after);
    sources.shards.emplace(identity.filename, std::move(shard));
  }
  return {};
}

[[nodiscard]] PrefillMLPK512OverlayDiagnostic reauthenticate_sources(
    const AuthenticatedSources& sources,
    PrefillMLPK512OverlayConversionStats& stats) {
  for (const ShardIdentity& identity : pinned_qwen36_27b_shards()) {
    const auto shard = sources.shards.find(identity.filename);
    const auto original = sources.snapshots.find(identity.filename);
    if (shard == sources.shards.end() || original == sources.snapshots.end()) {
      return make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kSourceAuthenticationFailed,
          identity.filename, "authenticated shard handle disappeared");
    }
    FileSnapshot before;
    FileSnapshot after;
    int error = 0;
    if (!snapshot_fd(shard->second.get(), before, error) ||
        !same_snapshot(before, original->second)) {
      return make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kSourceAuthenticationFailed,
          identity.filename, "source shard changed during conversion", {}, {},
          error);
    }
    std::string digest;
    auto diagnostic = hash_fd(shard->second.get(), identity.file_size, digest);
    stats.source_bytes_read += identity.file_size;
    if (!diagnostic) {
      return diagnostic;
    }
    if (digest != identity.sha256 ||
        !snapshot_fd(shard->second.get(), after, error) ||
        !same_snapshot(after, original->second)) {
      return make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kSourceAuthenticationFailed,
          identity.filename, "source shard failed final authentication",
          identity.sha256, digest, error);
    }
  }
  return {};
}

[[nodiscard]] PrefillMLPK512OverlayDiagnostic read_tensor(
    const AuthenticatedSources& sources, const mw::TensorLocator& locator,
    const std::uint64_t relative_offset, void* const output,
    const std::size_t bytes, const std::string_view context,
    PrefillMLPK512OverlayConversionStats& stats) {
  std::uint64_t end = 0U;
  std::uint64_t absolute = 0U;
  const auto shard = sources.shards.find(locator.shard);
  if (!checked_add(relative_offset, bytes, end) || end > locator.byte_size ||
      !checked_add(locator.file_begin, relative_offset, absolute) ||
      shard == sources.shards.end()) {
    return make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kSourceTensorMismatch,
        std::string(context), "tensor range is outside authenticated source");
  }
  int error = 0;
  if (!pread_exact(shard->second.get(), output, bytes, absolute, error)) {
    return make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kIoFailure,
        std::string(context), "source tensor read failed", {}, {}, error);
  }
  stats.source_bytes_read += bytes;
  return {};
}

}  // namespace

std::optional<PrefillMLPK512OverlayReceipt>
parse_prefill_mlp_k512_overlay_receipt(
    const std::string_view document,
    PrefillMLPK512OverlayDiagnostic& diagnostic) {
  diagnostic = {};
  try {
    json::ParseOptions options;
    options.max_input_bytes = 1U * 1024U * 1024U;
    options.max_nesting_depth = 8U;
    options.max_values = 128U;
    options.max_container_items = 128U;
    const json::ParseResult parsed = json::parse(document, options);
    const auto* root = parsed ? parsed.value->as_object() : nullptr;
    if (root == nullptr ||
        !exact_keys(*root,
                    {"schema", "version", "mode",
                     "production_residency_eligible", "physical_layout",
                     "packed_k_group_size", "scale_group_size",
                     "source_checkpoint_id", "source_config_sha256",
                     "source_index_sha256", "manifest_sha256",
                     "policy_sha256", "policy_bytes", "required_base",
                     "payload_sha256", "payload_bytes", "projection_count"})) {
      diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kInvalidReceipt, "receipt",
          "strict receipt JSON schema mismatch");
      return std::nullopt;
    }
    PrefillMLPK512OverlayReceipt receipt;
    std::string schema;
    std::string mode;
    std::uint64_t packed_k = 0U;
    std::uint64_t scale_k = 0U;
    const bool* eligible =
        root->at("production_residency_eligible").as_bool();
    if (!json_string(*root, "schema", schema) || schema != kReceiptSchema ||
        !parse_version(root->at("version"), receipt.version_major,
                       receipt.version_minor) ||
        receipt.version_major != kPrefillMLPK512OverlayVersionMajor ||
        receipt.version_minor != kPrefillMLPK512OverlayVersionMinor ||
        !json_string(*root, "mode", mode) ||
        mode != "production_calibrated" || eligible == nullptr || !*eligible ||
        !json_string(*root, "physical_layout", receipt.physical_layout) ||
        receipt.physical_layout != kPrefillMLPK512OverlayLayout ||
        !json_uint(*root, "packed_k_group_size", packed_k) || packed_k != 64U ||
        !json_uint(*root, "scale_group_size", scale_k) || scale_k != 512U ||
        !json_string(*root, "source_checkpoint_id",
                     receipt.source_checkpoint_id) ||
        receipt.source_checkpoint_id.empty() ||
        !json_string(*root, "source_config_sha256",
                     receipt.source_config_sha256) ||
        !lower_sha256(receipt.source_config_sha256) ||
        !json_string(*root, "source_index_sha256",
                     receipt.source_index_sha256) ||
        !lower_sha256(receipt.source_index_sha256) ||
        !json_string(*root, "manifest_sha256", receipt.manifest_sha256) ||
        !lower_sha256(receipt.manifest_sha256) ||
        !json_string(*root, "policy_sha256", receipt.policy_sha256) ||
        !lower_sha256(receipt.policy_sha256) ||
        !json_uint(*root, "policy_bytes", receipt.policy_bytes) ||
        receipt.policy_bytes == 0U ||
        !parse_base(root->at("required_base"), receipt.required_base) ||
        !json_string(*root, "payload_sha256", receipt.payload_sha256) ||
        !lower_sha256(receipt.payload_sha256) ||
        !json_uint(*root, "payload_bytes", receipt.payload_bytes) ||
        receipt.payload_bytes != kPrefillMLPK512OverlayPayloadBytes ||
        !json_uint(*root, "projection_count", receipt.projection_count) ||
        receipt.projection_count !=
            kPrefillMLPK512OverlayProjectionCount) {
      diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kInvalidReceipt, "receipt",
          "receipt identity or fixed K512 ABI is invalid");
      return std::nullopt;
    }
    receipt.production_residency_eligible = true;
    return receipt;
  } catch (...) {
    diagnostic = make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kInvalidReceipt, "receipt",
        "unexpected receipt parse failure");
    return std::nullopt;
  }
}

PrefillMLPK512OverlayConversionResult
convert_pinned_qwen36_27b_prefill_mlp_k512_overlay(
    const PrefillMLPK512OverlayConversionOptions& options) {
  PrefillMLPK512OverlayConversionResult result;
  fs::path payload_temporary;
  fs::path receipt_temporary;
  struct Cleanup final {
    fs::path* first;
    fs::path* second;
    ~Cleanup() {
      if (first != nullptr && !first->empty()) {
        (void)::unlink(first->c_str());
      }
      if (second != nullptr && !second->empty()) {
        (void)::unlink(second->c_str());
      }
    }
  } cleanup{&payload_temporary, &receipt_temporary};

  try {
    if (options.model_directory.empty() ||
        options.calibration_policy_path.empty() ||
        options.output_path.empty() || options.row_chunk_size == 0U ||
        options.row_chunk_size % 64U != 0U ||
        options.row_chunk_size > 512U || options.max_policy_bytes == 0U) {
      result.diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kInvalidOption, "convert",
          "model/policy/output and a bounded N64 row chunk are required");
      return result;
    }
    const fs::path receipt_path =
        fs::path(options.output_path.string() + ".receipt.json");
    struct stat status {};
    errno = 0;
    if (::lstat(options.output_path.c_str(), &status) == 0 || errno != ENOENT) {
      result.diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kPublicationConflict,
          options.output_path.string(), "payload target already exists");
      return result;
    }
    errno = 0;
    if (::lstat(receipt_path.c_str(), &status) == 0 || errno != ENOENT) {
      result.diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kPublicationConflict,
          receipt_path.string(), "receipt target already exists");
      return result;
    }

    std::string policy_document;
    if (!read_bounded_file(options.calibration_policy_path,
                           options.max_policy_bytes, policy_document,
                           result.diagnostic)) {
      return result;
    }
    PrefillMLPK512BaseBinding required_base;
    if (!extract_policy_base(policy_document, required_base,
                             result.diagnostic)) {
      return result;
    }
    const mw::ManifestResult source_manifest =
        mw::build_qwen36_27b_text_manifest(options.model_directory);
    if (!source_manifest) {
      result.diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kInvalidManifest,
          options.model_directory.string(),
          "pinned checkpoint manifest validation failed");
      return result;
    }
    const auto built =
        build_qwen36_27b_prefill_mlp_k512_overlay_manifest(
            *source_manifest.value, pinned_qwen36_27b_shards(),
            required_base);
    if (!built) {
      result.diagnostic = built.diagnostic;
      return result;
    }
    const PrefillMLPK512OverlayPolicyResult parsed_policy =
        parse_prefill_mlp_k512_overlay_policy(policy_document,
                                                       *built.value);
    if (!parsed_policy) {
      result.diagnostic = parsed_policy.diagnostic;
      return result;
    }

    AuthenticatedSources sources;
    result.diagnostic =
        authenticate_sources(options.model_directory, sources, result.stats);
    if (!result.diagnostic) {
      return result;
    }

    const fs::path parent = options.output_path.parent_path().empty()
                                ? fs::path(".")
                                : options.output_path.parent_path();
    payload_temporary = fs::path(options.output_path.string() + ".tmp." +
                                 std::to_string(::getpid()));
    receipt_temporary = fs::path(receipt_path.string() + ".tmp." +
                                 std::to_string(::getpid()));
    UniqueFd output(::open(payload_temporary.c_str(),
                           O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
                           S_IRUSR | S_IWUSR));
    if (!output) {
      result.diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kIoFailure,
          payload_temporary.string(), "failed to create temporary payload",
          {}, {}, errno);
      return result;
    }
    if (kPrefillMLPK512OverlayPayloadBytes >
            static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) ||
        ::ftruncate(output.get(), static_cast<off_t>(
                                      kPrefillMLPK512OverlayPayloadBytes)) !=
            0) {
      result.diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kIoFailure,
          payload_temporary.string(), "failed to size temporary payload", {},
          {}, errno);
      return result;
    }
    if (options.preallocate_output) {
      const int error = ::posix_fallocate(
          output.get(), 0,
          static_cast<off_t>(kPrefillMLPK512OverlayPayloadBytes));
      if (error != 0) {
        result.diagnostic = make_diagnostic(
            PrefillMLPK512OverlayErrorCode::kIoFailure,
            payload_temporary.string(), "payload preallocation failed", {},
            {}, error);
        return result;
      }
    }

    for (std::size_t projection_index = 0U;
         projection_index < built.value->projections.size();
         ++projection_index) {
      const auto& entry = built.value->projections[projection_index];
      const auto& calibration =
          parsed_policy.value->projections[projection_index];
      const mw::TensorLocator* weight =
          source_manifest.value->find(entry.source_module + ".weight");
      const mw::TensorLocator* weight_scale =
          source_manifest.value->find(entry.source_module + ".weight_scale");
      const mw::TensorLocator* weight_scale_2 =
          source_manifest.value->find(entry.source_module +
                                      ".weight_scale_2");
      if (weight == nullptr || weight_scale == nullptr ||
          weight_scale_2 == nullptr ||
          weight->dtype != io::safetensors::DType::kU8 ||
          weight->shape != std::vector<std::uint64_t>{
                               entry.output_size, entry.input_size / 2U} ||
          weight->byte_size != entry.output_size * entry.input_size / 2U ||
          weight_scale->dtype != io::safetensors::DType::kF8E4M3 ||
          weight_scale->shape != std::vector<std::uint64_t>{
                                     entry.output_size,
                                     entry.input_size / 16U} ||
          weight_scale->byte_size !=
              entry.output_size * entry.input_size / 16U ||
          weight_scale_2->dtype != io::safetensors::DType::kF32 ||
          !weight_scale_2->shape.empty() ||
          weight_scale_2->byte_size != 4U) {
        result.diagnostic = make_diagnostic(
            PrefillMLPK512OverlayErrorCode::kSourceTensorMismatch,
            entry.source_module,
            "MLP projection must be original packed NVFP4 plus per-K16 E4M3 and F32 secondary scales");
        return result;
      }
      std::array<std::uint8_t, 4U> scalar_bytes{};
      result.diagnostic = read_tensor(
          sources, *weight_scale_2, 0U, scalar_bytes.data(),
          scalar_bytes.size(), entry.source_module + ".weight_scale_2",
          result.stats);
      if (!result.diagnostic) {
        return result;
      }
      const float tensor_scale = read_little_f32(scalar_bytes.data());
      if (!std::isfinite(tensor_scale) || tensor_scale < 0.0F) {
        result.diagnostic = make_diagnostic(
            PrefillMLPK512OverlayErrorCode::kSourceTensorMismatch,
            entry.source_module + ".weight_scale_2",
            "source NVFP4 secondary scale is not finite/nonnegative");
        return result;
      }

      const std::size_t rows_per_chunk = std::min<std::size_t>(
          options.row_chunk_size,
          static_cast<std::size_t>(entry.output_size));
      const std::size_t input_size =
          static_cast<std::size_t>(entry.input_size);
      const std::size_t source_weight_stride = input_size / 2U;
      const std::size_t source_scale_stride = input_size / 16U;
      const std::size_t packed_stride = input_size / 2U;
      const std::size_t scale_stride = input_size / 512U * 2U;
      std::vector<std::uint8_t> source_weights(rows_per_chunk *
                                               source_weight_stride);
      std::vector<std::uint8_t> source_scales(rows_per_chunk *
                                              source_scale_stride);
      std::vector<float> decoded(rows_per_chunk * input_size);
      std::vector<std::uint8_t> packed(rows_per_chunk * packed_stride);
      std::vector<std::uint8_t> scales(rows_per_chunk * scale_stride);
      const std::uint64_t working =
          source_weights.capacity() + source_scales.capacity() +
          decoded.capacity() * sizeof(float) + packed.capacity() +
          scales.capacity();
      result.stats.peak_working_bytes =
          std::max(result.stats.peak_working_bytes, working);

      for (std::uint64_t row = 0U; row < entry.output_size;
           row += rows_per_chunk) {
        const std::size_t rows = static_cast<std::size_t>(
            std::min<std::uint64_t>(rows_per_chunk, entry.output_size - row));
        const std::size_t source_weight_bytes = rows * source_weight_stride;
        result.diagnostic = read_tensor(
            sources, *weight, row * source_weight_stride,
            source_weights.data(), source_weight_bytes,
            entry.source_module + ".weight", result.stats);
        if (!result.diagnostic) {
          return result;
        }
        const std::size_t source_scale_bytes = rows * source_scale_stride;
        result.diagnostic = read_tensor(
            sources, *weight_scale, row * source_scale_stride,
            source_scales.data(), source_scale_bytes,
            entry.source_module + ".weight_scale", result.stats);
        if (!result.diagnostic) {
          return result;
        }
        for (std::size_t local_row = 0U; local_row < rows; ++local_row) {
          for (std::size_t column = 0U; column < input_size; ++column) {
            const std::uint8_t packed_source =
                source_weights[local_row * source_weight_stride +
                               column / 2U];
            const std::uint8_t block_scale =
                source_scales[local_row * source_scale_stride +
                              column / 16U];
            decoded[local_row * input_size + column] =
                quantization::dequantize_nvfp4_value(
                    packed_source, (column & 1U) != 0U, block_scale,
                    tensor_scale);
          }
        }
        const std::size_t packed_bytes = rows * packed_stride;
        const std::size_t scale_bytes = rows * scale_stride;
        result.diagnostic = quantize_prefill_mlp_k512_consumer_blocks(
            decoded.data(), rows, input_size, calibration.weight_clip_ratio,
            packed.data(), packed_bytes, scales.data(), scale_bytes);
        if (!result.diagnostic) {
          result.diagnostic.context = entry.source_module + ":" +
                                      result.diagnostic.context;
          return result;
        }
        std::uint64_t weight_offset = 0U;
        std::uint64_t scale_offset = 0U;
        if (!checked_add(entry.sidecar_offset, row * packed_stride,
                         weight_offset) ||
            !checked_add(entry.sidecar_offset, entry.weight_bytes,
                         scale_offset) ||
            !checked_add(scale_offset, row * scale_stride, scale_offset)) {
          result.diagnostic = make_diagnostic(
              PrefillMLPK512OverlayErrorCode::kArithmeticOverflow,
              entry.source_module, "payload output offset overflowed");
          return result;
        }
        int error = 0;
        if (!pwrite_exact(output.get(), packed.data(), packed_bytes,
                          weight_offset, error) ||
            !pwrite_exact(output.get(), scales.data(), scale_bytes,
                          scale_offset, error)) {
          result.diagnostic = make_diagnostic(
              PrefillMLPK512OverlayErrorCode::kIoFailure,
              entry.source_module, "payload projection write failed", {}, {},
              error);
          return result;
        }
        result.stats.output_bytes_written += packed_bytes + scale_bytes;
        result.stats.rows_converted += rows;
      }
      ++result.stats.projections_converted;
    }
    if (result.stats.projections_converted !=
            kPrefillMLPK512OverlayProjectionCount ||
        result.stats.output_bytes_written !=
            kPrefillMLPK512OverlayPayloadBytes) {
      result.diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kIoFailure,
          payload_temporary.string(),
          "conversion did not write the complete 192-projection payload");
      return result;
    }
    if (::fsync(output.get()) != 0) {
      result.diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kIoFailure,
          payload_temporary.string(), "payload fsync failed", {}, {}, errno);
      return result;
    }
    result.diagnostic = reauthenticate_sources(sources, result.stats);
    if (!result.diagnostic) {
      return result;
    }
    std::string payload_sha256;
    result.diagnostic = hash_fd(output.get(),
                                kPrefillMLPK512OverlayPayloadBytes,
                                payload_sha256);
    if (!result.diagnostic) {
      return result;
    }
    if (::fchmod(output.get(), S_IRUSR) != 0 || ::fsync(output.get()) != 0) {
      result.diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kIoFailure,
          payload_temporary.string(), "failed to seal payload read-only", {},
          {}, errno);
      return result;
    }

    PrefillMLPK512OverlayReceipt receipt;
    receipt.production_residency_eligible = true;
    receipt.physical_layout = built.value->physical_layout;
    receipt.source_checkpoint_id = built.value->source_checkpoint_id;
    receipt.source_config_sha256 = built.value->source_config_sha256;
    receipt.source_index_sha256 = built.value->source_index_sha256;
    receipt.manifest_sha256 = built.value->manifest_sha256;
    receipt.policy_sha256 = parsed_policy.value->policy_sha256;
    receipt.policy_bytes = parsed_policy.value->policy_bytes;
    receipt.required_base = built.value->required_base;
    receipt.payload_sha256 = payload_sha256;
    receipt.payload_bytes = kPrefillMLPK512OverlayPayloadBytes;
    receipt.projection_count = kPrefillMLPK512OverlayProjectionCount;
    const std::string receipt_document = serialize_receipt(receipt);
    PrefillMLPK512OverlayDiagnostic receipt_parse_diagnostic;
    if (!parse_prefill_mlp_k512_overlay_receipt(
            receipt_document, receipt_parse_diagnostic)) {
      result.diagnostic = receipt_parse_diagnostic;
      return result;
    }
    UniqueFd receipt_output(::open(
        receipt_temporary.c_str(),
        O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
        S_IRUSR | S_IWUSR));
    if (!receipt_output) {
      result.diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kIoFailure,
          receipt_temporary.string(), "failed to create temporary receipt",
          {}, {}, errno);
      return result;
    }
    int receipt_error = 0;
    if (!pwrite_exact(receipt_output.get(), receipt_document.data(),
                      receipt_document.size(), 0U, receipt_error) ||
        ::fchmod(receipt_output.get(), S_IRUSR) != 0 ||
        ::fsync(receipt_output.get()) != 0) {
      result.diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kIoFailure,
          receipt_temporary.string(), "failed to seal temporary receipt", {},
          {}, receipt_error != 0 ? receipt_error : errno);
      return result;
    }

    if (::link(payload_temporary.c_str(), options.output_path.c_str()) != 0) {
      result.diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kPublicationConflict,
          options.output_path.string(), "payload no-replace link failed", {},
          {}, errno);
      return result;
    }
    if (::link(receipt_temporary.c_str(), receipt_path.c_str()) != 0) {
      const int saved = errno;
      (void)::unlink(options.output_path.c_str());
      result.diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kPublicationConflict,
          receipt_path.string(), "receipt no-replace link failed; rolled back",
          {}, {}, saved);
      return result;
    }
    if (::unlink(payload_temporary.c_str()) != 0 ||
        ::unlink(receipt_temporary.c_str()) != 0) {
      const int saved = errno;
      (void)::unlink(receipt_path.c_str());
      (void)::unlink(options.output_path.c_str());
      result.diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kIoFailure, parent.string(),
          "temporary cleanup failed; publication rolled back", {}, {}, saved);
      return result;
    }
    payload_temporary.clear();
    receipt_temporary.clear();
    UniqueFd directory(::open(parent.c_str(), O_RDONLY | O_DIRECTORY |
                                                  O_CLOEXEC | O_NOFOLLOW));
    if (!directory || ::fsync(directory.get()) != 0) {
      const int saved = errno;
      (void)::unlink(receipt_path.c_str());
      (void)::unlink(options.output_path.c_str());
      result.diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kIoFailure, parent.string(),
          "publication directory sync failed; rolled back", {}, {}, saved);
      return result;
    }
    result.receipt.emplace(std::move(receipt));
    result.diagnostic = {};
    return result;
  } catch (const std::bad_alloc&) {
    result.diagnostic = make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kAllocationFailure, "convert",
        "conversion allocation failed");
    return result;
  } catch (...) {
    result.diagnostic = make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kIoFailure, "convert",
        "unexpected conversion failure");
    return result;
  }
}

std::string_view to_string(
    const PrefillMLPK512OverlayErrorCode code) noexcept {
  switch (code) {
    case PrefillMLPK512OverlayErrorCode::kNone:
      return "none";
    case PrefillMLPK512OverlayErrorCode::kInvalidOption:
      return "invalid_option";
    case PrefillMLPK512OverlayErrorCode::kInvalidManifest:
      return "invalid_manifest";
    case PrefillMLPK512OverlayErrorCode::kInvalidPolicy:
      return "invalid_policy";
    case PrefillMLPK512OverlayErrorCode::kInvalidReceipt:
      return "invalid_receipt";
    case PrefillMLPK512OverlayErrorCode::kSourceAuthenticationFailed:
      return "source_authentication_failed";
    case PrefillMLPK512OverlayErrorCode::kSourceTensorMismatch:
      return "source_tensor_mismatch";
    case PrefillMLPK512OverlayErrorCode::kIoFailure:
      return "io_failure";
    case PrefillMLPK512OverlayErrorCode::kDigestMismatch:
      return "digest_mismatch";
    case PrefillMLPK512OverlayErrorCode::kArithmeticOverflow:
      return "arithmetic_overflow";
    case PrefillMLPK512OverlayErrorCode::kQuantizationFailure:
      return "quantization_failure";
    case PrefillMLPK512OverlayErrorCode::kPublicationConflict:
      return "publication_conflict";
    case PrefillMLPK512OverlayErrorCode::kAllocationFailure:
      return "allocation_failure";
  }
  return "unknown";
}

}  // namespace q3x::runtime
