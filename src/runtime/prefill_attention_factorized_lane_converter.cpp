#include "q3x/runtime/prefill_attention_factorized_lane_converter.h"

#include "q3x/core/sha256.h"
#include "q3x/io/json.h"
#include "q3x/runtime/prefill_mlp_factorized_lane_converter.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fcntl.h>
#include <filesystem>
#include <iomanip>
#include <limits>
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
    "q3x.prefill.attention-factorized-r1.overlay-manifest";
constexpr std::string_view kPolicySchema =
    "q3x.prefill.attention-factorized-r1.policy";
constexpr std::string_view kReceiptSchema =
    "q3x.prefill.attention-factorized-r1.receipt";

[[nodiscard]] PrefillAttentionFactorizedLaneConverterDiagnostic make_diagnostic(
    const PrefillAttentionFactorizedLaneConverterErrorCode code,
    std::string context, std::string message, std::string expected = {},
    std::string actual = {}) {
  PrefillAttentionFactorizedLaneConverterDiagnostic result;
  result.code = code;
  result.context = std::move(context);
  result.message = std::move(message);
  result.expected = std::move(expected);
  result.actual = std::move(actual);
  return result;
}

[[nodiscard]] bool checked_add(const std::uint64_t left,
                               const std::uint64_t right,
                               std::uint64_t& output) noexcept {
  return prefill_a4_factorized_lane_contract_detail::checked_add(
      left, right, output);
}

[[nodiscard]] bool checked_multiply(const std::uint64_t left,
                                    const std::uint64_t right,
                                    std::uint64_t& output) noexcept {
  return prefill_a4_factorized_lane_contract_detail::checked_multiply(
      left, right, output);
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

[[nodiscard]] std::string sha256_text(const std::string_view bytes) {
  core::Sha256 hasher;
  if (!hasher.update(bytes.data(), bytes.size())) {
    return {};
  }
  return hasher.finalize().hex();
}

[[nodiscard]] std::string digest_hex(
    const std::array<std::uint8_t,
                     kPrefillA4FactorizedLaneMetadataDigestBytes>& bytes) {
  core::Sha256Digest digest;
  digest.bytes = bytes;
  return digest.hex();
}

void write_quoted(std::ostream& output, const std::string_view value) {
  output << '"';
  for (const unsigned char character : value) {
    switch (character) {
      case '"': output << "\\\""; break;
      case '\\': output << "\\\\"; break;
      case '\b': output << "\\b"; break;
      case '\f': output << "\\f"; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (character < 0x20U) {
          output << "\\u00" << std::hex << std::setw(2)
                 << std::setfill('0')
                 << static_cast<unsigned int>(character) << std::dec
                 << std::setfill(' ');
        } else {
          output << static_cast<char>(character);
        }
        break;
    }
  }
  output << '"';
}

[[nodiscard]] bool expected_attention_family(
    const PrefillProjectionFamily family) noexcept {
  switch (family) {
    case PrefillProjectionFamily::kLinearQkv:
    case PrefillProjectionFamily::kLinearZ:
    case PrefillProjectionFamily::kLinearO:
    case PrefillProjectionFamily::kFullQ:
    case PrefillProjectionFamily::kFullK:
    case PrefillProjectionFamily::kFullV:
    case PrefillProjectionFamily::kFullO:
      return true;
    default:
      return false;
  }
}

[[nodiscard]] PrefillAttentionFactorizedLaneProjectionFamily overlay_family(
    const PrefillProjectionFamily family) noexcept {
  switch (family) {
    case PrefillProjectionFamily::kLinearQkv:
      return PrefillAttentionFactorizedLaneProjectionFamily::kLinearQkv;
    case PrefillProjectionFamily::kLinearZ:
      return PrefillAttentionFactorizedLaneProjectionFamily::kLinearZ;
    case PrefillProjectionFamily::kLinearO:
      return PrefillAttentionFactorizedLaneProjectionFamily::kLinearO;
    case PrefillProjectionFamily::kFullQ:
      return PrefillAttentionFactorizedLaneProjectionFamily::kFullQ;
    case PrefillProjectionFamily::kFullK:
      return PrefillAttentionFactorizedLaneProjectionFamily::kFullK;
    case PrefillProjectionFamily::kFullV:
      return PrefillAttentionFactorizedLaneProjectionFamily::kFullV;
    case PrefillProjectionFamily::kFullO:
      return PrefillAttentionFactorizedLaneProjectionFamily::kFullO;
    default:
      return static_cast<PrefillAttentionFactorizedLaneProjectionFamily>(
          0xffU);
  }
}

[[nodiscard]] std::string_view family_name(
    const PrefillAttentionFactorizedLaneProjectionFamily family) noexcept {
  switch (family) {
    case PrefillAttentionFactorizedLaneProjectionFamily::kLinearQkv:
      return "linear_qkv";
    case PrefillAttentionFactorizedLaneProjectionFamily::kLinearZ:
      return "linear_z";
    case PrefillAttentionFactorizedLaneProjectionFamily::kLinearO:
      return "linear_o";
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullQ:
      return "full_q";
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullK:
      return "full_k";
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullV:
      return "full_v";
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullO:
      return "full_o";
  }
  return "invalid";
}

[[nodiscard]] std::string expected_source_module(
    const std::uint32_t layer,
    const PrefillAttentionFactorizedLaneProjectionFamily family) {
  std::string suffix;
  switch (family) {
    case PrefillAttentionFactorizedLaneProjectionFamily::kLinearQkv:
      suffix = "linear_attn.in_proj_qkv";
      break;
    case PrefillAttentionFactorizedLaneProjectionFamily::kLinearZ:
      suffix = "linear_attn.in_proj_z";
      break;
    case PrefillAttentionFactorizedLaneProjectionFamily::kLinearO:
      suffix = "linear_attn.out_proj";
      break;
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullQ:
      suffix = "self_attn.q_proj";
      break;
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullK:
      suffix = "self_attn.k_proj";
      break;
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullV:
      suffix = "self_attn.v_proj";
      break;
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullO:
      suffix = "self_attn.o_proj";
      break;
  }
  return "model.language_model.layers." + std::to_string(layer) + "." +
         suffix;
}

[[nodiscard]] const PrefillA4FactorizedLaneProjectionLayoutPlan*
projection_layout(
    const PrefillAttentionFactorizedLaneOverlayLayoutPlan& plan,
    const PrefillAttentionFactorizedLaneProjectionFamily family) noexcept {
  switch (family) {
    case PrefillAttentionFactorizedLaneProjectionFamily::kLinearQkv:
      return &plan.linear_qkv;
    case PrefillAttentionFactorizedLaneProjectionFamily::kLinearZ:
      return &plan.linear_z;
    case PrefillAttentionFactorizedLaneProjectionFamily::kLinearO:
      return &plan.linear_o;
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullQ:
      return &plan.full_q;
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullK:
      return &plan.full_k;
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullV:
      return &plan.full_v;
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullO:
      return &plan.full_o;
  }
  return nullptr;
}

[[nodiscard]] bool expected_shape(
    const PrefillAttentionFactorizedLaneProjectionFamily family,
    std::uint64_t& output_size, std::uint64_t& input_size) noexcept {
  input_size = kPrefillAttentionFactorizedLaneHiddenSize;
  switch (family) {
    case PrefillAttentionFactorizedLaneProjectionFamily::kLinearQkv:
      output_size = kPrefillAttentionFactorizedLaneLinearQkvOutputSize;
      return true;
    case PrefillAttentionFactorizedLaneProjectionFamily::kLinearZ:
      output_size = kPrefillAttentionFactorizedLaneLinearZOutputSize;
      return true;
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullQ:
      output_size = kPrefillAttentionFactorizedLaneFullQOutputSize;
      return true;
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullK:
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullV:
      output_size = kPrefillAttentionFactorizedLaneFullKvOutputSize;
      return true;
    case PrefillAttentionFactorizedLaneProjectionFamily::kLinearO:
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullO:
      output_size = kPrefillAttentionFactorizedLaneHiddenSize;
      input_size = kPrefillAttentionFactorizedLaneAttentionOutputSize;
      return true;
  }
  return false;
}

[[nodiscard]] PrefillAttentionFactorizedLaneProjectionFamily
expected_family_at_position(const std::uint32_t layer,
                            const std::uint32_t position) noexcept {
  if (prefill_attention_factorized_lane_is_full_layer(layer)) {
    constexpr std::array<PrefillAttentionFactorizedLaneProjectionFamily, 4U>
        families = {
            PrefillAttentionFactorizedLaneProjectionFamily::kFullQ,
            PrefillAttentionFactorizedLaneProjectionFamily::kFullK,
            PrefillAttentionFactorizedLaneProjectionFamily::kFullV,
            PrefillAttentionFactorizedLaneProjectionFamily::kFullO,
        };
    return position < families.size()
               ? families[position]
               : static_cast<
                     PrefillAttentionFactorizedLaneProjectionFamily>(0xffU);
  }
  constexpr std::array<PrefillAttentionFactorizedLaneProjectionFamily, 3U>
      families = {
          PrefillAttentionFactorizedLaneProjectionFamily::kLinearQkv,
          PrefillAttentionFactorizedLaneProjectionFamily::kLinearZ,
          PrefillAttentionFactorizedLaneProjectionFamily::kLinearO,
      };
  return position < families.size()
             ? families[position]
             : static_cast<PrefillAttentionFactorizedLaneProjectionFamily>(
                   0xffU);
}

[[nodiscard]] bool same_base(
    const PrefillAttentionFactorizedLaneBaseK256Binding& left,
    const PrefillAttentionFactorizedLaneBaseK256Binding& right) noexcept {
  return left.physical_layout == right.physical_layout &&
         left.packed_k_group_size == right.packed_k_group_size &&
         left.scale_group_size == right.scale_group_size &&
         left.manifest_sha256 == right.manifest_sha256 &&
         left.policy_sha256 == right.policy_sha256 &&
         left.payload_sha256 == right.payload_sha256 &&
         left.receipt_sha256 == right.receipt_sha256;
}

[[nodiscard]] bool valid_base(
    const PrefillAttentionFactorizedLaneBaseK256Binding& base) noexcept {
  return base.physical_layout ==
             kPrefillAttentionFactorizedLaneRequiredBaseK256Layout &&
         base.packed_k_group_size ==
             kPrefillAttentionFactorizedLaneRequiredBasePackedK &&
         base.scale_group_size ==
             kPrefillAttentionFactorizedLaneRequiredBaseScaleK &&
         lower_sha256(base.manifest_sha256) &&
         lower_sha256(base.policy_sha256) &&
         lower_sha256(base.payload_sha256) &&
         lower_sha256(base.receipt_sha256);
}

[[nodiscard]] std::string identity_factor_digest(
    const std::uint64_t input_size) {
  if (input_size > std::numeric_limits<std::size_t>::max()) {
    return {};
  }
  std::vector<float> ones(static_cast<std::size_t>(input_size), 1.0F);
  const auto metadata = serialize_prefill_attention_factorized_lane_metadata(
      kPrefillAttentionFactorizedLaneR1LaneCount, ones.data(), ones.size());
  return metadata ? digest_hex(metadata.inverse_alpha_sha256) : std::string{};
}

[[nodiscard]] std::string manifest_body(
    const PrefillAttentionFactorizedLaneOverlayManifestBinding& manifest) {
  std::ostringstream output;
  output << "schema=" << kManifestSchema << "\nversion="
         << manifest.version_major << '.' << manifest.version_minor
         << "\nlayout=" << manifest.physical_layout << "\ncheckpoint="
         << manifest.source_checkpoint_id << "\nconfig="
         << manifest.source_config_sha256 << "\nindex="
         << manifest.source_index_sha256 << "\nlane="
         << manifest.lane_count << "\nbase="
         << manifest.required_base_k256.physical_layout << ':'
         << manifest.required_base_k256.packed_k_group_size << ':'
         << manifest.required_base_k256.scale_group_size << ':'
         << manifest.required_base_k256.manifest_sha256 << ':'
         << manifest.required_base_k256.policy_sha256 << ':'
         << manifest.required_base_k256.payload_sha256 << ':'
         << manifest.required_base_k256.receipt_sha256 << "\npayload="
         << manifest.payload_bytes << '\n';
  for (const auto& entry : manifest.projections) {
    output << entry.ordinal << ':' << entry.layer_index << ':'
           << family_name(entry.family) << ':' << entry.source_module << ':'
           << entry.source_sha256 << ':' << entry.output_size << ':'
           << entry.input_size << ':' << entry.payload_offset << ':'
           << entry.payload_bytes << '\n';
  }
  return output.str();
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

struct FileSnapshot final {
  std::uint64_t device = 0U;
  std::uint64_t inode = 0U;
  std::uint64_t size = 0U;
  std::int64_t mtime_s = 0;
  std::int64_t mtime_ns = 0;
  std::int64_t ctime_s = 0;
  std::int64_t ctime_ns = 0;
};

struct LockedDocument final {
  UniqueFd fd;
  FileSnapshot snapshot;
  std::string document;
  std::string sha256;
};

[[nodiscard]] PrefillAttentionFactorizedLaneConverterDiagnostic
file_diagnostic(PrefillAttentionFactorizedLaneConverterErrorCode code,
                std::string context, std::string message, int error);
[[nodiscard]] bool capture_snapshot(int fd, FileSnapshot& output,
                                    int& error) noexcept;
[[nodiscard]] bool same_snapshot(const FileSnapshot& left,
                                 const FileSnapshot& right) noexcept;
[[nodiscard]] bool offset_fits(std::uint64_t offset) noexcept;
[[nodiscard]] bool pread_exact(int fd, void* destination,
                               std::size_t byte_count, std::uint64_t offset,
                               int& error) noexcept;
[[nodiscard]] bool pwrite_exact(int fd, const void* source,
                                std::size_t byte_count, std::uint64_t offset,
                                int& error) noexcept;
[[nodiscard]] bool pwrite_zeros(int fd, std::uint64_t byte_count,
                                std::uint64_t offset, int& error) noexcept;
[[nodiscard]] PrefillAttentionFactorizedLaneConverterDiagnostic hash_open_file(
    int fd, std::uint64_t byte_count, std::string& digest);
[[nodiscard]] PrefillAttentionFactorizedLaneConverterDiagnostic
read_locked_document(const fs::path& path, std::uint64_t maximum_bytes,
                     LockedDocument& output);
[[nodiscard]] PrefillAttentionFactorizedLaneConverterDiagnostic
revalidate_locked_document(const LockedDocument& document,
                           const fs::path& path);
void remove_if_present(const fs::path& path) noexcept;
[[nodiscard]] UniqueFd create_temporary_file_near(
    const fs::path& target, std::string_view tag, fs::path& path,
    PrefillAttentionFactorizedLaneConverterDiagnostic& diagnostic);
[[nodiscard]] bool target_absent(
    const fs::path& path,
    PrefillAttentionFactorizedLaneConverterDiagnostic& diagnostic);
[[nodiscard]] bool same_normalized_path(const fs::path& left,
                                        const fs::path& right);
[[nodiscard]] PrefillAttentionFactorizedLaneConverterDiagnostic seal_document(
    int fd, const fs::path& path, std::string_view document);
[[nodiscard]] PrefillAttentionFactorizedLaneConverterDiagnostic
publish_three_no_replace(
    const fs::path& payload_temp, int payload_fd, const fs::path& payload,
    const fs::path& policy_temp, int policy_fd, const fs::path& policy,
    const fs::path& receipt_temp, int receipt_fd, const fs::path& receipt);
[[nodiscard]] const PrefillProjectionSidecarEntry* find_base_projection(
    const PrefillSidecarManifest& base, std::string_view module);

}  // namespace

PrefillAttentionFactorizedLaneR1ConversionResult
convert_authenticated_k256_to_prefill_attention_factorized_lane_r1(
    const PrefillAttentionFactorizedLaneR1ConversionOptions& options) {
  PrefillAttentionFactorizedLaneR1ConversionResult result;
  fs::path payload_temp;
  fs::path policy_temp;
  fs::path receipt_temp;
  struct TemporaryCleanup final {
    fs::path* payload;
    fs::path* policy;
    fs::path* receipt;
    ~TemporaryCleanup() {
      remove_if_present(*payload);
      remove_if_present(*policy);
      remove_if_present(*receipt);
    }
  } cleanup{&payload_temp, &policy_temp, &receipt_temp};

  try {
    const fs::path policy_path =
        fs::path(options.output_path.string() + ".policy.json");
    const fs::path receipt_path =
        fs::path(options.output_path.string() + ".receipt.json");
    if (options.model_directory.empty() ||
        options.base_k256_payload_path.empty() ||
        options.base_k256_policy_path.empty() ||
        options.base_k256_receipt_path.empty() || options.output_path.empty() ||
        !valid_clip_ratio(options.weight_clip_ratio) ||
        !valid_clip_ratio(options.activation_clip_ratio) ||
        options.max_base_receipt_bytes == 0U ||
        options.max_base_receipt_bytes > 1ULL * 1024ULL * 1024ULL ||
        options.output_path.filename().empty() ||
        options.output_path.filename() == "." ||
        options.output_path.filename() == "..") {
      result.diagnostic = make_diagnostic(
          PrefillAttentionFactorizedLaneConverterErrorCode::kInvalidOption,
          "conversion_options",
          "model/base triple/output, bounded receipt, and two clips required");
      return result;
    }
    if (same_normalized_path(options.output_path,
                             options.base_k256_payload_path) ||
        same_normalized_path(options.output_path,
                             options.base_k256_policy_path) ||
        same_normalized_path(options.output_path,
                             options.base_k256_receipt_path) ||
        same_normalized_path(policy_path, options.base_k256_payload_path) ||
        same_normalized_path(policy_path, options.base_k256_policy_path) ||
        same_normalized_path(policy_path, options.base_k256_receipt_path) ||
        same_normalized_path(receipt_path, options.base_k256_payload_path) ||
        same_normalized_path(receipt_path, options.base_k256_policy_path) ||
        same_normalized_path(receipt_path,
                             options.base_k256_receipt_path)) {
      result.diagnostic = make_diagnostic(
          PrefillAttentionFactorizedLaneConverterErrorCode::kUnsafePath,
          options.output_path.string(),
          "output publication set must not alias any base input");
      return result;
    }
    if (!target_absent(options.output_path, result.diagnostic) ||
        !target_absent(policy_path, result.diagnostic) ||
        !target_absent(receipt_path, result.diagnostic)) {
      return result;
    }
    const fs::path output_parent =
        options.output_path.parent_path().empty()
            ? fs::path(".")
            : options.output_path.parent_path();
    std::error_code filesystem_error;
    const fs::file_status parent_status =
        fs::symlink_status(output_parent, filesystem_error);
    if (filesystem_error || !fs::is_directory(parent_status) ||
        fs::is_symlink(parent_status)) {
      result.diagnostic = make_diagnostic(
          PrefillAttentionFactorizedLaneConverterErrorCode::kUnsafePath,
          output_parent.string(),
          "output parent must be an existing non-symlink directory", {},
          filesystem_error.message());
      return result;
    }

    const mw::ManifestResult source_manifest =
        mw::build_qwen36_27b_text_manifest(options.model_directory);
    if (!source_manifest) {
      result.diagnostic = make_diagnostic(
          PrefillAttentionFactorizedLaneConverterErrorCode::kInvalidManifest,
          options.model_directory.string(),
          "pinned Qwen3.6-27B checkpoint manifest validation failed");
      return result;
    }
    PrefillSidecarManifestOptions base_manifest_options;
    base_manifest_options.kind = PrefillSidecarKind::kA4K256;
    const PrefillSidecarManifestResult base_manifest_result =
        build_qwen36_27b_prefill_sidecar_manifest(
            *source_manifest.value, pinned_qwen36_27b_shards(),
            base_manifest_options);
    if (!base_manifest_result) {
      result.diagnostic = make_diagnostic(
          PrefillAttentionFactorizedLaneConverterErrorCode::kInvalidManifest,
          base_manifest_result.diagnostic.context,
          base_manifest_result.diagnostic.message);
      return result;
    }
    const PrefillSidecarManifest& base_manifest =
        *base_manifest_result.value;

    LockedDocument base_receipt_document;
    result.diagnostic = read_locked_document(
        options.base_k256_receipt_path, options.max_base_receipt_bytes,
        base_receipt_document);
    if (!result.diagnostic) {
      return result;
    }
    PrefillA4ConverterDiagnostic base_receipt_diagnostic;
    const std::optional<PrefillA4PublicationReceipt> parsed_base_receipt =
        parse_prefill_a4_publication_receipt(
            base_receipt_document.document, base_receipt_diagnostic);
    if (!parsed_base_receipt.has_value() || !base_receipt_diagnostic) {
      result.diagnostic = make_diagnostic(
          PrefillAttentionFactorizedLaneConverterErrorCode::
              kInvalidBaseReceipt,
          options.base_k256_receipt_path.string(),
          "strict base K256 receipt parsing failed",
          base_receipt_diagnostic.message, base_receipt_diagnostic.context);
      return result;
    }
    const PrefillAttentionFactorizedLaneManifestResult overlay_manifest_result =
        build_prefill_attention_factorized_lane_r1_manifest(
            base_manifest, *parsed_base_receipt,
            base_receipt_document.sha256);
    if (!overlay_manifest_result) {
      result.diagnostic = overlay_manifest_result.diagnostic;
      return result;
    }
    PrefillAttentionFactorizedLaneOverlayManifestBinding overlay_manifest =
        *overlay_manifest_result.value;

    PrefillA4PublicationAuthenticationResult base_authentication =
        authenticate_prefill_a4_publication_for_residency(
            base_manifest, *parsed_base_receipt,
            options.base_k256_payload_path, options.base_k256_policy_path);
    if (!base_authentication) {
      result.diagnostic = make_diagnostic(
          PrefillAttentionFactorizedLaneConverterErrorCode::
              kBaseAuthenticationFailed,
          base_authentication.diagnostic.context,
          base_authentication.diagnostic.message,
          base_authentication.diagnostic.expected,
          base_authentication.diagnostic.actual);
      return result;
    }
    PrefillA4AuthenticatedPublication base_publication =
        std::move(*base_authentication.value);
    if (base_publication.receipt().sidecar_kind !=
            PrefillSidecarKind::kA4K256 ||
        base_publication.receipt().manifest_sha256 !=
            overlay_manifest.required_base_k256.manifest_sha256 ||
        base_publication.receipt().policy_sha256 !=
            overlay_manifest.required_base_k256.policy_sha256 ||
        base_publication.receipt().payload_sha256 !=
            overlay_manifest.required_base_k256.payload_sha256) {
      result.diagnostic = make_diagnostic(
          PrefillAttentionFactorizedLaneConverterErrorCode::
              kSourceBindingMismatch,
          "base.publication",
          "authenticated K256 publication differs from derivative binding");
      return result;
    }

    PrefillAttentionFactorizedLaneR1PolicyResult policy_result =
        build_prefill_attention_factorized_lane_r1_policy(
            overlay_manifest, options.weight_clip_ratio,
            options.activation_clip_ratio);
    if (!policy_result) {
      result.diagnostic = policy_result.diagnostic;
      return result;
    }
    const std::string policy_document = policy_result.canonical_document;
    PrefillAttentionFactorizedLaneR1Policy policy = *policy_result.value;
    const auto policy_reparsed =
        parse_prefill_attention_factorized_lane_r1_policy(
            policy_document, overlay_manifest);
    if (!policy_reparsed ||
        policy_reparsed.value->binding.policy_sha256 !=
            policy.binding.policy_sha256 ||
        policy_reparsed.value->binding.policy_bytes !=
            policy.binding.policy_bytes) {
      result.diagnostic = policy_reparsed
                              ? make_diagnostic(
                                    PrefillAttentionFactorizedLaneConverterErrorCode::
                                        kDigestMismatch,
                                    "r1.policy",
                                    "canonical policy failed exact reparse")
                              : policy_reparsed.diagnostic;
      return result;
    }

    UniqueFd output = create_temporary_file_near(
        options.output_path, "payload", payload_temp, result.diagnostic);
    if (!output) {
      return result;
    }
    if (!offset_fits(overlay_manifest.payload_bytes) ||
        ::ftruncate(output.get(),
                    static_cast<off_t>(overlay_manifest.payload_bytes)) != 0) {
      result.diagnostic = file_diagnostic(
          PrefillAttentionFactorizedLaneConverterErrorCode::kIoFailure,
          payload_temp.string(), "failed to size temporary R1 payload", errno);
      return result;
    }
    if (options.preallocate_output) {
      const int allocation_error = ::posix_fallocate(
          output.get(), 0, static_cast<off_t>(overlay_manifest.payload_bytes));
      if (allocation_error != 0) {
        result.diagnostic = file_diagnostic(
            PrefillAttentionFactorizedLaneConverterErrorCode::kIoFailure,
            payload_temp.string(),
            "failed to preallocate complete R1 payload", allocation_error);
        return result;
      }
    }

    const auto plan = prefill_attention_factorized_lane_overlay_layout_plan(
        kPrefillAttentionFactorizedLaneR1LaneCount);
    std::vector<float> hidden_ones(
        static_cast<std::size_t>(
            kPrefillAttentionFactorizedLaneHiddenSize),
        1.0F);
    std::vector<float> attention_output_ones(
        static_cast<std::size_t>(
            kPrefillAttentionFactorizedLaneAttentionOutputSize),
        1.0F);
    const auto hidden_metadata =
        serialize_prefill_attention_factorized_lane_metadata(
            kPrefillAttentionFactorizedLaneR1LaneCount, hidden_ones.data(),
            hidden_ones.size());
    const auto attention_output_metadata =
        serialize_prefill_attention_factorized_lane_metadata(
            kPrefillAttentionFactorizedLaneR1LaneCount,
            attention_output_ones.data(), attention_output_ones.size());
    if (!plan || !hidden_metadata || !attention_output_metadata) {
      result.diagnostic = make_diagnostic(
          PrefillAttentionFactorizedLaneConverterErrorCode::kAllocationFailure,
          "r1.metadata", "failed to materialize identity-alpha metadata");
      return result;
    }

    for (std::size_t projection_index = 0U;
         projection_index < overlay_manifest.projections.size();
         ++projection_index) {
      const auto& overlay_entry =
          overlay_manifest.projections[projection_index];
      const PrefillProjectionSidecarEntry* const base_entry =
          find_base_projection(base_manifest, overlay_entry.source_module);
      const auto* const layout = projection_layout(plan, overlay_entry.family);
      const bool family_matches =
          base_entry != nullptr &&
          overlay_family(base_entry->family) == overlay_entry.family;
      if (!family_matches || layout == nullptr ||
          base_entry->output_size != overlay_entry.output_size ||
          base_entry->input_size != overlay_entry.input_size ||
          base_entry->source_sha256 != overlay_entry.source_sha256 ||
          base_entry->scale_group_size !=
              kPrefillAttentionFactorizedLaneRequiredBaseScaleK) {
        result.diagnostic = make_diagnostic(
            PrefillAttentionFactorizedLaneConverterErrorCode::
                kSourceBindingMismatch,
            overlay_entry.source_module,
            "base K256 projection differs from R1 manifest");
        return result;
      }
      const std::uint64_t n_blocks = overlay_entry.output_size / 64U;
      std::uint64_t packed_block_bytes = 0U;
      if (!checked_multiply(overlay_entry.input_size, 32U,
                            packed_block_bytes)) {
        result.diagnostic = make_diagnostic(
            PrefillAttentionFactorizedLaneConverterErrorCode::
                kArithmeticOverflow,
            overlay_entry.source_module, "N64 packed byte count overflowed");
        return result;
      }
      const std::uint64_t base_scale_block_bytes =
          overlay_entry.input_size / 2U;
      constexpr std::uint64_t r1_scale_block_bytes = 64U * 2U;
      std::uint64_t expected_weight_bytes = 0U;
      std::uint64_t expected_base_scale_bytes = 0U;
      std::uint64_t expected_r1_scale_bytes = 0U;
      if (!checked_multiply(n_blocks, packed_block_bytes,
                            expected_weight_bytes) ||
          !checked_multiply(n_blocks, base_scale_block_bytes,
                            expected_base_scale_bytes) ||
          !checked_multiply(n_blocks, r1_scale_block_bytes,
                            expected_r1_scale_bytes) ||
          base_entry->weight_bytes != expected_weight_bytes ||
          base_entry->scale_bytes != expected_base_scale_bytes ||
          layout->packed_weight_bytes != expected_weight_bytes ||
          layout->weight_scale_bytes != expected_r1_scale_bytes ||
          packed_block_bytes > std::numeric_limits<std::size_t>::max() ||
          base_scale_block_bytes >
              std::numeric_limits<std::size_t>::max()) {
        result.diagnostic = make_diagnostic(
            PrefillAttentionFactorizedLaneConverterErrorCode::kInvalidManifest,
            overlay_entry.source_module,
            "base or R1 projection byte arithmetic is invalid");
        return result;
      }
      std::vector<std::uint8_t> base_packed(
          static_cast<std::size_t>(packed_block_bytes));
      std::vector<std::uint8_t> base_scales(
          static_cast<std::size_t>(base_scale_block_bytes));
      std::vector<std::uint8_t> r1_packed(
          static_cast<std::size_t>(packed_block_bytes));
      std::array<std::uint8_t, r1_scale_block_bytes> r1_scales{};
      const std::uint64_t working_bytes =
          base_packed.capacity() + base_scales.capacity() +
          r1_packed.capacity() + r1_scales.size() +
          64U * overlay_entry.input_size * sizeof(float);
      result.stats.peak_working_bytes =
          std::max(result.stats.peak_working_bytes, working_bytes);

      for (std::uint64_t n_block = 0U; n_block < n_blocks; ++n_block) {
        std::uint64_t base_weight_offset = 0U;
        std::uint64_t base_scale_offset = 0U;
        std::uint64_t output_weight_offset = 0U;
        std::uint64_t output_scale_offset = 0U;
        std::uint64_t block_relative = 0U;
        if (!checked_multiply(n_block, packed_block_bytes, block_relative) ||
            !checked_add(base_entry->sidecar_offset, block_relative,
                         base_weight_offset) ||
            !checked_add(base_entry->sidecar_offset,
                         base_entry->weight_bytes, base_scale_offset) ||
            !checked_multiply(n_block, base_scale_block_bytes,
                              block_relative) ||
            !checked_add(base_scale_offset, block_relative,
                         base_scale_offset) ||
            !checked_add(overlay_entry.payload_offset,
                         layout->packed_weight_offset,
                         output_weight_offset) ||
            !checked_multiply(n_block, packed_block_bytes, block_relative) ||
            !checked_add(output_weight_offset, block_relative,
                         output_weight_offset) ||
            !checked_add(overlay_entry.payload_offset,
                         layout->weight_scale_offset,
                         output_scale_offset) ||
            !checked_multiply(n_block, r1_scale_block_bytes,
                              block_relative) ||
            !checked_add(output_scale_offset, block_relative,
                         output_scale_offset)) {
          result.diagnostic = make_diagnostic(
              PrefillAttentionFactorizedLaneConverterErrorCode::
                  kArithmeticOverflow,
              overlay_entry.source_module,
              "projection block offset overflowed");
          return result;
        }
        int error = 0;
        if (!pread_exact(base_publication.payload_fd(), base_packed.data(),
                         base_packed.size(), base_weight_offset, error) ||
            !pread_exact(base_publication.payload_fd(), base_scales.data(),
                         base_scales.size(), base_scale_offset, error)) {
          result.diagnostic = file_diagnostic(
              PrefillAttentionFactorizedLaneConverterErrorCode::kIoFailure,
              overlay_entry.source_module,
              "failed to read authenticated K256 N64 block", error);
          return result;
        }
        result.stats.base_bytes_read +=
            base_packed.size() + base_scales.size();
        result.diagnostic =
            transform_prefill_attention_k256_to_factorized_r1_consumer_blocks(
                base_packed.data(), base_packed.size(), base_scales.data(),
                base_scales.size(), 64U,
                static_cast<std::size_t>(overlay_entry.input_size),
                policy.binding.projections[projection_index]
                    .weight_clip_ratio,
                r1_packed.data(), r1_packed.size(), r1_scales.data(),
                r1_scales.size());
        if (!result.diagnostic) {
          result.diagnostic.context =
              overlay_entry.source_module + ":" + result.diagnostic.context;
          return result;
        }
        if (!pwrite_exact(output.get(), r1_packed.data(), r1_packed.size(),
                          output_weight_offset, error) ||
            !pwrite_exact(output.get(), r1_scales.data(), r1_scales.size(),
                          output_scale_offset, error)) {
          result.diagnostic = file_diagnostic(
              PrefillAttentionFactorizedLaneConverterErrorCode::kIoFailure,
              overlay_entry.source_module, "failed to write R1 N64 block",
              error);
          return result;
        }
        result.stats.output_bytes_written +=
            r1_packed.size() + r1_scales.size();
        ++result.stats.n64_blocks_converted;
      }

      const auto& metadata =
          overlay_entry.input_size ==
                  kPrefillAttentionFactorizedLaneHiddenSize
              ? hidden_metadata
              : attention_output_metadata;
      const std::string metadata_digest =
          digest_hex(metadata.inverse_alpha_sha256);
      if (metadata.bytes.size() != layout->metadata_bytes ||
          metadata_digest !=
              policy.binding.projections[projection_index]
                  .factor_source.sha256) {
        result.diagnostic = make_diagnostic(
            PrefillAttentionFactorizedLaneConverterErrorCode::kDigestMismatch,
            overlay_entry.source_module,
            "identity metadata differs from policy factor binding");
        return result;
      }
      std::uint64_t packed_end = 0U;
      std::uint64_t scale_end = 0U;
      std::uint64_t metadata_end = 0U;
      if (!checked_add(layout->packed_weight_offset,
                       layout->packed_weight_bytes, packed_end) ||
          !checked_add(layout->weight_scale_offset,
                       layout->weight_scale_bytes, scale_end) ||
          !checked_add(layout->metadata_offset, layout->metadata_bytes,
                       metadata_end) ||
          packed_end > layout->weight_scale_offset ||
          scale_end > layout->metadata_offset ||
          metadata_end > layout->projection_bytes) {
        result.diagnostic = make_diagnostic(
            PrefillAttentionFactorizedLaneConverterErrorCode::kInvalidManifest,
            overlay_entry.source_module, "projection padding ranges overlap");
        return result;
      }
      int error = 0;
      std::uint64_t absolute = 0U;
      if (!checked_add(overlay_entry.payload_offset, packed_end, absolute) ||
          !pwrite_zeros(output.get(),
                        layout->weight_scale_offset - packed_end, absolute,
                        error) ||
          !checked_add(overlay_entry.payload_offset, scale_end, absolute) ||
          !pwrite_zeros(output.get(), layout->metadata_offset - scale_end,
                        absolute, error) ||
          !checked_add(overlay_entry.payload_offset, layout->metadata_offset,
                       absolute) ||
          !pwrite_exact(output.get(), metadata.bytes.data(),
                        metadata.bytes.size(), absolute, error) ||
          !checked_add(overlay_entry.payload_offset, metadata_end, absolute) ||
          !pwrite_zeros(output.get(),
                        layout->projection_bytes - metadata_end, absolute,
                        error)) {
        result.diagnostic = file_diagnostic(
            PrefillAttentionFactorizedLaneConverterErrorCode::kIoFailure,
            overlay_entry.source_module,
            "failed to explicitly write projection metadata/padding", error);
        return result;
      }
      result.stats.output_bytes_written +=
          (layout->weight_scale_offset - packed_end) +
          (layout->metadata_offset - scale_end) + metadata.bytes.size() +
          (layout->projection_bytes - metadata_end);
      ++result.stats.projections_converted;
    }

    if (result.stats.projections_converted !=
            kPrefillAttentionFactorizedLaneProjectionCount ||
        result.stats.output_bytes_written != overlay_manifest.payload_bytes) {
      result.diagnostic = make_diagnostic(
          PrefillAttentionFactorizedLaneConverterErrorCode::kIoFailure,
          payload_temp.string(),
          "conversion did not explicitly cover the complete R1 payload",
          std::to_string(overlay_manifest.payload_bytes),
          std::to_string(result.stats.output_bytes_written));
      return result;
    }
    const PrefillA4ConverterDiagnostic base_revalidation =
        base_publication.revalidate_unchanged_after_consumption();
    if (!base_revalidation) {
      result.diagnostic = make_diagnostic(
          PrefillAttentionFactorizedLaneConverterErrorCode::
              kBaseAuthenticationFailed,
          base_revalidation.context, base_revalidation.message,
          base_revalidation.expected, base_revalidation.actual);
      return result;
    }
    result.diagnostic = revalidate_locked_document(
        base_receipt_document, options.base_k256_receipt_path);
    if (!result.diagnostic) {
      return result;
    }

    if (::fsync(output.get()) != 0) {
      result.diagnostic = file_diagnostic(
          PrefillAttentionFactorizedLaneConverterErrorCode::kIoFailure,
          payload_temp.string(), "failed to synchronize temporary payload",
          errno);
      return result;
    }
    FileSnapshot payload_before;
    int snapshot_error = 0;
    if (!capture_snapshot(output.get(), payload_before, snapshot_error) ||
        payload_before.size != overlay_manifest.payload_bytes) {
      result.diagnostic = file_diagnostic(
          PrefillAttentionFactorizedLaneConverterErrorCode::kIoFailure,
          payload_temp.string(), "temporary payload size/snapshot invalid",
          snapshot_error);
      return result;
    }
    std::string payload_sha256;
    result.diagnostic = hash_open_file(
        output.get(), overlay_manifest.payload_bytes, payload_sha256);
    if (!result.diagnostic) {
      result.diagnostic.context = payload_temp.string();
      return result;
    }
    FileSnapshot payload_after;
    if (!capture_snapshot(output.get(), payload_after, snapshot_error) ||
        !same_snapshot(payload_before, payload_after)) {
      result.diagnostic = file_diagnostic(
          PrefillAttentionFactorizedLaneConverterErrorCode::kDigestMismatch,
          payload_temp.string(), "temporary payload changed while hashing",
          snapshot_error);
      return result;
    }
    if (::fchmod(output.get(), S_IRUSR) != 0 || ::fsync(output.get()) != 0) {
      result.diagnostic = file_diagnostic(
          PrefillAttentionFactorizedLaneConverterErrorCode::kIoFailure,
          payload_temp.string(), "failed to seal R1 payload read-only", errno);
      return result;
    }

    PrefillAttentionFactorizedLaneR1ReceiptResult receipt_result =
        build_prefill_attention_factorized_lane_r1_receipt(
            overlay_manifest, policy, payload_sha256);
    if (!receipt_result) {
      result.diagnostic = receipt_result.diagnostic;
      return result;
    }
    const std::string receipt_document = receipt_result.canonical_document;
    PrefillAttentionFactorizedLaneR1Receipt receipt = *receipt_result.value;
    const auto receipt_reparsed =
        parse_prefill_attention_factorized_lane_r1_receipt(
            receipt_document, overlay_manifest, policy);
    if (!receipt_reparsed) {
      result.diagnostic = receipt_reparsed.diagnostic;
      return result;
    }

    UniqueFd policy_output = create_temporary_file_near(
        policy_path, "policy", policy_temp, result.diagnostic);
    if (!policy_output) {
      return result;
    }
    result.diagnostic =
        seal_document(policy_output.get(), policy_temp, policy_document);
    if (!result.diagnostic) {
      return result;
    }
    UniqueFd receipt_output = create_temporary_file_near(
        receipt_path, "receipt", receipt_temp, result.diagnostic);
    if (!receipt_output) {
      return result;
    }
    result.diagnostic =
        seal_document(receipt_output.get(), receipt_temp, receipt_document);
    if (!result.diagnostic) {
      return result;
    }
    result.diagnostic = publish_three_no_replace(
        payload_temp, output.get(), options.output_path, policy_temp,
        policy_output.get(), policy_path, receipt_temp, receipt_output.get(),
        receipt_path);
    if (!result.diagnostic) {
      return result;
    }
    payload_temp.clear();
    policy_temp.clear();
    receipt_temp.clear();
    result.manifest.emplace(std::move(overlay_manifest));
    result.policy.emplace(std::move(policy));
    result.receipt.emplace(std::move(receipt));
    return result;
  } catch (const std::bad_alloc&) {
    result.diagnostic = make_diagnostic(
        PrefillAttentionFactorizedLaneConverterErrorCode::kAllocationFailure,
        "conversion", "bounded R1 conversion allocation failed");
    return result;
  } catch (const std::exception& error) {
    result.diagnostic = make_diagnostic(
        PrefillAttentionFactorizedLaneConverterErrorCode::kIoFailure,
        "conversion", "unexpected R1 conversion failure", {}, error.what());
    return result;
  } catch (...) {
    result.diagnostic = make_diagnostic(
        PrefillAttentionFactorizedLaneConverterErrorCode::kIoFailure,
        "conversion", "unexpected R1 conversion failure");
    return result;
  }
}


std::string prefill_attention_factorized_lane_r1_manifest_sha256(
    const PrefillAttentionFactorizedLaneOverlayManifestBinding& manifest) {
  return sha256_text(manifest_body(manifest));
}

PrefillAttentionFactorizedLaneConverterDiagnostic
validate_prefill_attention_factorized_lane_r1_manifest(
    const PrefillAttentionFactorizedLaneOverlayManifestBinding& manifest) {
  const auto plan = prefill_attention_factorized_lane_overlay_layout_plan(
      kPrefillAttentionFactorizedLaneR1LaneCount);
  if (!plan ||
      manifest.version_major !=
          kPrefillAttentionFactorizedLaneOverlayVersionMajor ||
      manifest.version_minor !=
          kPrefillAttentionFactorizedLaneOverlayVersionMinor ||
      manifest.physical_layout !=
          kPrefillAttentionFactorizedLaneOverlayLayout ||
      manifest.source_checkpoint_id.empty() ||
      !lower_sha256(manifest.source_config_sha256) ||
      !lower_sha256(manifest.source_index_sha256) ||
      !valid_base(manifest.required_base_k256) ||
      manifest.lane_count != kPrefillAttentionFactorizedLaneR1LaneCount ||
      manifest.projections.size() !=
          kPrefillAttentionFactorizedLaneProjectionCount ||
      manifest.payload_bytes != kPrefillAttentionFactorizedLaneR1PayloadBytes ||
      manifest.payload_bytes != plan.payload_bytes ||
      !lower_sha256(manifest.manifest_sha256)) {
    return make_diagnostic(
        PrefillAttentionFactorizedLaneConverterErrorCode::kInvalidManifest,
        "overlay.manifest", "R1 Attention overlay header is invalid");
  }

  std::size_t index = 0U;
  for (std::uint32_t layer = 0U;
       layer < kPrefillAttentionFactorizedLaneLayerCount; ++layer) {
    const std::uint32_t projection_count =
        prefill_attention_factorized_lane_is_full_layer(layer)
            ? kPrefillAttentionFactorizedLaneFullProjectionsPerLayer
            : kPrefillAttentionFactorizedLaneLinearProjectionsPerLayer;
    for (std::uint32_t position = 0U; position < projection_count;
         ++position, ++index) {
      const auto family = expected_family_at_position(layer, position);
      const auto* const layout = projection_layout(plan, family);
      std::uint64_t output_size = 0U;
      std::uint64_t input_size = 0U;
      const auto& entry = manifest.projections[index];
      const std::uint64_t expected_offset =
          prefill_attention_factorized_lane_projection_absolute_offset(
              plan, layer, family);
      const std::uint32_t expected_ordinal =
          prefill_attention_factorized_lane_projection_ordinal(layer,
                                                                family);
      if (layout == nullptr || !expected_shape(family, output_size, input_size) ||
          expected_ordinal != index || entry.ordinal != expected_ordinal ||
          entry.layer_index != layer || entry.family != family ||
          entry.source_module != expected_source_module(layer, family) ||
          !lower_sha256(entry.source_sha256) ||
          entry.output_size != output_size || entry.input_size != input_size ||
          entry.payload_offset != expected_offset ||
          entry.payload_bytes != layout->projection_bytes) {
        return make_diagnostic(
            PrefillAttentionFactorizedLaneConverterErrorCode::kInvalidManifest,
            "overlay.manifest.projections[" + std::to_string(index) + "]",
            "fixed layer-major Attention inventory is invalid");
      }
    }
  }
  if (index != manifest.projections.size()) {
    return make_diagnostic(
        PrefillAttentionFactorizedLaneConverterErrorCode::kInvalidManifest,
        "overlay.manifest.projections",
        "Attention projection inventory cardinality is invalid");
  }
  const std::string digest =
      prefill_attention_factorized_lane_r1_manifest_sha256(manifest);
  if (digest != manifest.manifest_sha256) {
    return make_diagnostic(
        PrefillAttentionFactorizedLaneConverterErrorCode::kDigestMismatch,
        "overlay.manifest_sha256", "manifest body digest mismatch",
        manifest.manifest_sha256, digest);
  }
  return {};
}

PrefillAttentionFactorizedLaneManifestResult
build_prefill_attention_factorized_lane_r1_manifest(
    const PrefillSidecarManifest& base_k256_manifest,
    const PrefillA4PublicationReceipt& base_k256_receipt,
    const std::string_view base_receipt_sha256) {
  PrefillAttentionFactorizedLaneManifestResult result;
  const PrefillContractDiagnostic base_manifest_diagnostic =
      validate_prefill_sidecar_manifest(base_k256_manifest);
  if (!base_manifest_diagnostic ||
      base_k256_manifest.kind != PrefillSidecarKind::kA4K256 ||
      base_k256_manifest.projections.size() !=
          kQwen36PrefillProjectionCount) {
    result.diagnostic = make_diagnostic(
        PrefillAttentionFactorizedLaneConverterErrorCode::kInvalidManifest,
        "base.manifest", "valid pinned 400-projection K256 manifest required");
    return result;
  }
  if (base_k256_receipt.version_major !=
          kPrefillA4K256PublicationVersionMajor ||
      base_k256_receipt.version_minor !=
          kPrefillA4K256PublicationVersionMinor ||
      base_k256_receipt.mode !=
          PrefillA4ConversionMode::kProductionCalibrated ||
      !base_k256_receipt.production_residency_eligible ||
      base_k256_receipt.sidecar_kind != PrefillSidecarKind::kA4K256 ||
      base_k256_receipt.packed_k_group_size !=
          kPrefillAttentionFactorizedLaneRequiredBasePackedK ||
      base_k256_receipt.scale_group_size !=
          kPrefillAttentionFactorizedLaneRequiredBaseScaleK ||
      base_k256_receipt.physical_layout !=
          kPrefillAttentionFactorizedLaneRequiredBaseK256Layout ||
      base_k256_receipt.source_checkpoint_id !=
          base_k256_manifest.source_checkpoint_id ||
      base_k256_receipt.source_config_sha256 !=
          base_k256_manifest.source_config_sha256 ||
      base_k256_receipt.source_index_sha256 !=
          base_k256_manifest.source_index_sha256 ||
      base_k256_receipt.manifest_sha256 !=
          base_k256_manifest.manifest_sha256 ||
      base_k256_receipt.payload_bytes !=
          base_k256_manifest.summary.arena_bytes ||
      base_k256_receipt.projection_count !=
          base_k256_manifest.summary.projection_count ||
      !lower_sha256(base_k256_receipt.policy_sha256) ||
      base_k256_receipt.policy_bytes == 0U ||
      !lower_sha256(base_k256_receipt.payload_sha256) ||
      !lower_sha256(base_receipt_sha256)) {
    result.diagnostic = make_diagnostic(
        PrefillAttentionFactorizedLaneConverterErrorCode::kInvalidBaseReceipt,
        "base.receipt", "production K256 receipt does not bind the manifest");
    return result;
  }

  try {
    const auto plan = prefill_attention_factorized_lane_overlay_layout_plan(
        kPrefillAttentionFactorizedLaneR1LaneCount);
    PrefillAttentionFactorizedLaneOverlayManifestBinding manifest;
    manifest.physical_layout =
        std::string(kPrefillAttentionFactorizedLaneOverlayLayout);
    manifest.source_checkpoint_id = base_k256_manifest.source_checkpoint_id;
    manifest.source_config_sha256 = base_k256_manifest.source_config_sha256;
    manifest.source_index_sha256 = base_k256_manifest.source_index_sha256;
    manifest.required_base_k256.physical_layout =
        base_k256_receipt.physical_layout;
    manifest.required_base_k256.packed_k_group_size =
        base_k256_receipt.packed_k_group_size;
    manifest.required_base_k256.scale_group_size =
        base_k256_receipt.scale_group_size;
    manifest.required_base_k256.manifest_sha256 =
        base_k256_receipt.manifest_sha256;
    manifest.required_base_k256.policy_sha256 =
        base_k256_receipt.policy_sha256;
    manifest.required_base_k256.payload_sha256 =
        base_k256_receipt.payload_sha256;
    manifest.required_base_k256.receipt_sha256 =
        std::string(base_receipt_sha256);
    manifest.lane_count = kPrefillAttentionFactorizedLaneR1LaneCount;
    manifest.payload_bytes = plan.payload_bytes;
    manifest.projections.reserve(kPrefillAttentionFactorizedLaneProjectionCount);

    for (const auto& source : base_k256_manifest.projections) {
      if (!expected_attention_family(source.family)) {
        continue;
      }
      const std::size_t index = manifest.projections.size();
      const auto family = overlay_family(source.family);
      const std::uint32_t expected_ordinal =
          prefill_attention_factorized_lane_projection_ordinal(
              source.layer_index, family);
      const auto* const layout = projection_layout(plan, family);
      std::uint64_t output_size = 0U;
      std::uint64_t input_size = 0U;
      if (index >= kPrefillAttentionFactorizedLaneProjectionCount ||
          expected_ordinal != index || layout == nullptr ||
          !expected_shape(family, output_size, input_size) ||
          source.source_module !=
              expected_source_module(source.layer_index, family) ||
          source.output_size != output_size ||
          source.input_size != input_size ||
          !lower_sha256(source.source_sha256)) {
        result.diagnostic = make_diagnostic(
            PrefillAttentionFactorizedLaneConverterErrorCode::kInvalidManifest,
            source.source_module,
            "base Attention inventory differs from fixed layer-major ABI");
        return result;
      }
      PrefillAttentionFactorizedLaneManifestProjection entry;
      entry.ordinal = expected_ordinal;
      entry.layer_index = source.layer_index;
      entry.family = family;
      entry.source_module = source.source_module;
      entry.source_sha256 = source.source_sha256;
      entry.output_size = source.output_size;
      entry.input_size = source.input_size;
      entry.payload_offset =
          prefill_attention_factorized_lane_projection_absolute_offset(
              plan, source.layer_index, family);
      entry.payload_bytes = layout->projection_bytes;
      manifest.projections.emplace_back(std::move(entry));
    }

    manifest.manifest_sha256 =
        prefill_attention_factorized_lane_r1_manifest_sha256(manifest);
    result.diagnostic =
        validate_prefill_attention_factorized_lane_r1_manifest(manifest);
    if (!result.diagnostic) {
      return result;
    }
    result.value.emplace(std::move(manifest));
    return result;
  } catch (const std::bad_alloc&) {
    result.diagnostic = make_diagnostic(
        PrefillAttentionFactorizedLaneConverterErrorCode::kAllocationFailure,
        "overlay.manifest", "manifest allocation failed");
    return result;
  }
}

namespace {

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
  const std::string* const value =
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
  const json::Number* const value =
      found == object.end() ? nullptr : found->second.as_number();
  return value != nullptr && value->to_uint64(output);
}

[[nodiscard]] bool json_double(const json::Value::Object& object,
                               const std::string_view key, double& output) {
  const auto found = object.find(key);
  const json::Number* const value =
      found == object.end() ? nullptr : found->second.as_number();
  return value != nullptr && value->to_double(output) && std::isfinite(output);
}

[[nodiscard]] bool json_bool(const json::Value::Object& object,
                             const std::string_view key, bool& output) {
  const auto found = object.find(key);
  const bool* const value =
      found == object.end() ? nullptr : found->second.as_bool();
  if (value == nullptr) {
    return false;
  }
  output = *value;
  return true;
}

[[nodiscard]] bool parse_version(const json::Value& value,
                                 std::uint32_t& major,
                                 std::uint32_t& minor) {
  const auto* const object = value.as_object();
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

void write_base(
    std::ostream& output,
    const PrefillAttentionFactorizedLaneBaseK256Binding& base) {
  output << "{\"sidecar_kind\":\"a4_k256\",\"physical_layout\":";
  write_quoted(output, base.physical_layout);
  output << ",\"packed_k_group_size\":" << base.packed_k_group_size
         << ",\"scale_group_size\":" << base.scale_group_size
         << ",\"manifest_sha256\":";
  write_quoted(output, base.manifest_sha256);
  output << ",\"policy_sha256\":";
  write_quoted(output, base.policy_sha256);
  output << ",\"payload_sha256\":";
  write_quoted(output, base.payload_sha256);
  output << ",\"receipt_sha256\":";
  write_quoted(output, base.receipt_sha256);
  output << '}';
}

[[nodiscard]] bool parse_base(
    const json::Value& value,
    PrefillAttentionFactorizedLaneBaseK256Binding& base) {
  const auto* const object = value.as_object();
  std::string kind;
  std::uint64_t packed_k = 0U;
  std::uint64_t scale_k = 0U;
  if (object == nullptr ||
      !exact_keys(*object,
                  {"sidecar_kind", "physical_layout",
                   "packed_k_group_size", "scale_group_size",
                   "manifest_sha256", "policy_sha256", "payload_sha256",
                   "receipt_sha256"}) ||
      !json_string(*object, "sidecar_kind", kind) || kind != "a4_k256" ||
      !json_string(*object, "physical_layout", base.physical_layout) ||
      !json_uint(*object, "packed_k_group_size", packed_k) ||
      packed_k > std::numeric_limits<std::uint32_t>::max() ||
      !json_uint(*object, "scale_group_size", scale_k) ||
      scale_k > std::numeric_limits<std::uint32_t>::max() ||
      !json_string(*object, "manifest_sha256", base.manifest_sha256) ||
      !json_string(*object, "policy_sha256", base.policy_sha256) ||
      !json_string(*object, "payload_sha256", base.payload_sha256) ||
      !json_string(*object, "receipt_sha256", base.receipt_sha256)) {
    return false;
  }
  base.packed_k_group_size = static_cast<std::uint32_t>(packed_k);
  base.scale_group_size = static_cast<std::uint32_t>(scale_k);
  return valid_base(base);
}

[[nodiscard]] bool same_factor(
    const PrefillAttentionFactorizedLaneFactorSourceBinding& left,
    const PrefillAttentionFactorizedLaneFactorSourceBinding& right) noexcept {
  return left.scheme == right.scheme && left.path == right.path &&
         left.sha256 == right.sha256 &&
         left.element_count == right.element_count;
}

[[nodiscard]] bool shared_activation_and_factor(
    const PrefillAttentionFactorizedLaneProjectionCalibrationBinding& left,
    const PrefillAttentionFactorizedLaneProjectionCalibrationBinding& right)
    noexcept {
  return static_cast<float>(left.activation_clip_ratio) ==
             static_cast<float>(right.activation_clip_ratio) &&
         same_factor(left.factor_source, right.factor_source);
}

[[nodiscard]] std::string serialize_policy(
    const PrefillAttentionFactorizedLaneR1Policy& policy) {
  const auto& binding = policy.binding;
  std::ostringstream output;
  output << std::setprecision(17) << "{\"schema\":";
  write_quoted(output, kPolicySchema);
  output << ",\"version\":{\"major\":" << binding.version_major
         << ",\"minor\":" << binding.version_minor << "},\"mode\":";
  write_quoted(output, policy.mode);
  output << ",\"performance_upper_bound_only\":"
         << (policy.performance_upper_bound_only ? "true" : "false")
         << ",\"quality_production_eligible\":"
         << (policy.quality_production_eligible ? "true" : "false")
         << ",\"converter_abi\":";
  write_quoted(output, policy.converter_abi);
  output << ",\"physical_layout\":";
  write_quoted(output, binding.physical_layout);
  output << ",\"source_checkpoint_id\":";
  write_quoted(output, binding.source_checkpoint_id);
  output << ",\"source_config_sha256\":";
  write_quoted(output, binding.source_config_sha256);
  output << ",\"source_index_sha256\":";
  write_quoted(output, binding.source_index_sha256);
  output << ",\"manifest_sha256\":";
  write_quoted(output, binding.manifest_sha256);
  output << ",\"required_base_k256\":";
  write_base(output, binding.required_base_k256);
  output << ",\"lane_count\":" << binding.lane_count
         << ",\"projections\":[";
  for (std::size_t index = 0U; index < binding.projections.size(); ++index) {
    const auto& projection = binding.projections[index];
    output << (index == 0U ? "" : ",") << "{\"ordinal\":"
           << projection.ordinal << ",\"source_module\":";
    write_quoted(output, projection.source_module);
    output << ",\"source_sha256\":";
    write_quoted(output, projection.source_sha256);
    output << ",\"weight_clip_ratio\":" << projection.weight_clip_ratio
           << ",\"activation_clip_ratio\":"
           << projection.activation_clip_ratio
           << ",\"rounding\":\"nearest_even_v1\",\"factor\":{\"scheme\":";
    write_quoted(output, projection.factor_source.scheme);
    output << ",\"path\":";
    write_quoted(output, projection.factor_source.path);
    output << ",\"sha256\":";
    write_quoted(output, projection.factor_source.sha256);
    output << ",\"count\":" << projection.factor_source.element_count
           << "}}";
  }
  output << "]}\n";
  return output.str();
}

[[nodiscard]] bool policy_identity_matches_manifest(
    const PrefillAttentionFactorizedLaneR1Policy& policy,
    const PrefillAttentionFactorizedLaneOverlayManifestBinding& manifest)
    noexcept {
  const auto& binding = policy.binding;
  return binding.version_major ==
             kPrefillAttentionFactorizedLaneOverlayVersionMajor &&
         binding.version_minor ==
             kPrefillAttentionFactorizedLaneOverlayVersionMinor &&
         policy.converter_abi ==
             kPrefillAttentionFactorizedLaneR1ConverterAbi &&
         policy.mode == kPrefillAttentionFactorizedLaneR1Mode &&
         policy.performance_upper_bound_only &&
         !policy.quality_production_eligible &&
         binding.physical_layout == manifest.physical_layout &&
         binding.source_checkpoint_id == manifest.source_checkpoint_id &&
         binding.source_config_sha256 == manifest.source_config_sha256 &&
         binding.source_index_sha256 == manifest.source_index_sha256 &&
         binding.manifest_sha256 == manifest.manifest_sha256 &&
         same_base(binding.required_base_k256,
                   manifest.required_base_k256) &&
         binding.lane_count == kPrefillAttentionFactorizedLaneR1LaneCount;
}

[[nodiscard]] PrefillAttentionFactorizedLaneConverterDiagnostic
validate_policy_entries(
    const PrefillAttentionFactorizedLaneR1Policy& policy,
    const PrefillAttentionFactorizedLaneOverlayManifestBinding& manifest) {
  if (!policy_identity_matches_manifest(policy, manifest) ||
      policy.binding.projections.size() != manifest.projections.size()) {
    return make_diagnostic(
        PrefillAttentionFactorizedLaneConverterErrorCode::kInvalidPolicy,
        "r1.policy", "policy identity differs from R1 Attention manifest");
  }

  const std::string hidden_digest = identity_factor_digest(
      kPrefillAttentionFactorizedLaneHiddenSize);
  const std::string attention_output_digest = identity_factor_digest(
      kPrefillAttentionFactorizedLaneAttentionOutputSize);
  if (!lower_sha256(hidden_digest) ||
      !lower_sha256(attention_output_digest)) {
    return make_diagnostic(
        PrefillAttentionFactorizedLaneConverterErrorCode::kAllocationFailure,
        "r1.policy.factor", "failed to materialize identity factor digest");
  }

  for (std::size_t index = 0U; index < manifest.projections.size(); ++index) {
    const auto& calibration = policy.binding.projections[index];
    const auto& entry = manifest.projections[index];
    const std::string& expected_digest =
        entry.input_size == kPrefillAttentionFactorizedLaneHiddenSize
            ? hidden_digest
            : attention_output_digest;
    if (calibration.ordinal != index ||
        calibration.source_module != entry.source_module ||
        calibration.source_sha256 != entry.source_sha256 ||
        !valid_clip_ratio(calibration.weight_clip_ratio) ||
        !valid_clip_ratio(calibration.activation_clip_ratio) ||
        calibration.factor_source.scheme !=
            kPrefillAttentionFactorizedLaneR1FactorScheme ||
        !calibration.factor_source.path.empty() ||
        calibration.factor_source.sha256 != expected_digest ||
        calibration.factor_source.element_count != entry.input_size) {
      return make_diagnostic(
          PrefillAttentionFactorizedLaneConverterErrorCode::kInvalidPolicy,
          "r1.policy.projections[" + std::to_string(index) + "]",
          "projection clip/source/identity-factor binding is invalid");
    }
  }

  for (std::uint32_t layer = 0U;
       layer < kPrefillAttentionFactorizedLaneLayerCount; ++layer) {
    if (prefill_attention_factorized_lane_is_full_layer(layer)) {
      const std::uint32_t q =
          prefill_attention_factorized_lane_projection_ordinal(
              layer, PrefillAttentionFactorizedLaneProjectionFamily::kFullQ);
      const std::uint32_t k =
          prefill_attention_factorized_lane_projection_ordinal(
              layer, PrefillAttentionFactorizedLaneProjectionFamily::kFullK);
      const std::uint32_t v =
          prefill_attention_factorized_lane_projection_ordinal(
              layer, PrefillAttentionFactorizedLaneProjectionFamily::kFullV);
      if (!shared_activation_and_factor(policy.binding.projections[q],
                                        policy.binding.projections[k]) ||
          !shared_activation_and_factor(policy.binding.projections[q],
                                        policy.binding.projections[v])) {
        return make_diagnostic(
            PrefillAttentionFactorizedLaneConverterErrorCode::kInvalidPolicy,
            "r1.policy.full_qkv_group",
            "Full Q/K/V must share activation clip and identity factor");
      }
    } else {
      const std::uint32_t qkv =
          prefill_attention_factorized_lane_projection_ordinal(
              layer,
              PrefillAttentionFactorizedLaneProjectionFamily::kLinearQkv);
      const std::uint32_t z =
          prefill_attention_factorized_lane_projection_ordinal(
              layer, PrefillAttentionFactorizedLaneProjectionFamily::kLinearZ);
      if (!shared_activation_and_factor(policy.binding.projections[qkv],
                                        policy.binding.projections[z])) {
        return make_diagnostic(
            PrefillAttentionFactorizedLaneConverterErrorCode::kInvalidPolicy,
            "r1.policy.linear_qkv_z_group",
            "Linear QKV/Z must share activation clip and identity factor");
      }
    }
  }
  const bool has_document_identity =
      !policy.binding.policy_sha256.empty() ||
      policy.binding.policy_bytes != 0U;
  if (has_document_identity) {
    const std::string canonical = serialize_policy(policy);
    if (!lower_sha256(policy.binding.policy_sha256) ||
        policy.binding.policy_bytes != canonical.size() ||
        policy.binding.policy_sha256 != sha256_text(canonical)) {
      return make_diagnostic(
          PrefillAttentionFactorizedLaneConverterErrorCode::kDigestMismatch,
          "r1.policy", "policy object differs from its canonical byte identity");
    }
  }
  return {};
}

}  // namespace

PrefillAttentionFactorizedLaneR1PolicyResult
build_prefill_attention_factorized_lane_r1_policy(
    const PrefillAttentionFactorizedLaneOverlayManifestBinding& manifest,
    const double weight_clip_ratio, const double activation_clip_ratio) {
  PrefillAttentionFactorizedLaneR1PolicyResult result;
  result.diagnostic =
      validate_prefill_attention_factorized_lane_r1_manifest(manifest);
  if (!result.diagnostic || !valid_clip_ratio(weight_clip_ratio) ||
      !valid_clip_ratio(activation_clip_ratio)) {
    if (result.diagnostic) {
      result.diagnostic = make_diagnostic(
          PrefillAttentionFactorizedLaneConverterErrorCode::kInvalidOption,
          "r1.policy", "two explicit clip ratios in [1/256,1] are required");
    }
    return result;
  }

  try {
    const std::string hidden_digest = identity_factor_digest(
        kPrefillAttentionFactorizedLaneHiddenSize);
    const std::string attention_output_digest = identity_factor_digest(
        kPrefillAttentionFactorizedLaneAttentionOutputSize);
    if (!lower_sha256(hidden_digest) ||
        !lower_sha256(attention_output_digest)) {
      result.diagnostic = make_diagnostic(
          PrefillAttentionFactorizedLaneConverterErrorCode::kAllocationFailure,
          "r1.policy.factor", "failed to materialize identity factor digest");
      return result;
    }

    PrefillAttentionFactorizedLaneR1Policy policy;
    policy.binding.physical_layout = manifest.physical_layout;
    policy.binding.source_checkpoint_id = manifest.source_checkpoint_id;
    policy.binding.source_config_sha256 = manifest.source_config_sha256;
    policy.binding.source_index_sha256 = manifest.source_index_sha256;
    policy.binding.manifest_sha256 = manifest.manifest_sha256;
    policy.binding.required_base_k256 = manifest.required_base_k256;
    policy.binding.lane_count = kPrefillAttentionFactorizedLaneR1LaneCount;
    policy.converter_abi =
        std::string(kPrefillAttentionFactorizedLaneR1ConverterAbi);
    policy.mode = std::string(kPrefillAttentionFactorizedLaneR1Mode);
    policy.performance_upper_bound_only = true;
    policy.quality_production_eligible = false;
    policy.binding.projections.reserve(manifest.projections.size());
    for (const auto& entry : manifest.projections) {
      PrefillAttentionFactorizedLaneProjectionCalibrationBinding calibration;
      calibration.ordinal = entry.ordinal;
      calibration.source_module = entry.source_module;
      calibration.source_sha256 = entry.source_sha256;
      calibration.weight_clip_ratio = weight_clip_ratio;
      calibration.activation_clip_ratio = activation_clip_ratio;
      calibration.factor_source.scheme =
          std::string(kPrefillAttentionFactorizedLaneR1FactorScheme);
      calibration.factor_source.path.clear();
      calibration.factor_source.sha256 =
          entry.input_size == kPrefillAttentionFactorizedLaneHiddenSize
              ? hidden_digest
              : attention_output_digest;
      calibration.factor_source.element_count = entry.input_size;
      policy.binding.projections.emplace_back(std::move(calibration));
    }
    result.diagnostic = validate_policy_entries(policy, manifest);
    if (!result.diagnostic) {
      return result;
    }
    result.canonical_document = serialize_policy(policy);
    policy.binding.policy_sha256 = sha256_text(result.canonical_document);
    policy.binding.policy_bytes = result.canonical_document.size();
    result.value.emplace(std::move(policy));
    return result;
  } catch (const std::bad_alloc&) {
    result.diagnostic = make_diagnostic(
        PrefillAttentionFactorizedLaneConverterErrorCode::kAllocationFailure,
        "r1.policy", "policy allocation failed");
    return result;
  }
}

PrefillAttentionFactorizedLaneR1PolicyResult
parse_prefill_attention_factorized_lane_r1_policy(
    const std::string_view document,
    const PrefillAttentionFactorizedLaneOverlayManifestBinding& manifest) {
  PrefillAttentionFactorizedLaneR1PolicyResult result;
  result.diagnostic =
      validate_prefill_attention_factorized_lane_r1_manifest(manifest);
  if (!result.diagnostic) {
    return result;
  }

  try {
    json::ParseOptions options;
    options.max_input_bytes = 2U * 1024U * 1024U;
    options.max_nesting_depth = 12U;
    options.max_values = 3'000U;
    options.max_container_items = 3'000U;
    const json::ParseResult parsed = json::parse(document, options);
    const auto* const root = parsed ? parsed.value->as_object() : nullptr;
    if (root == nullptr ||
        !exact_keys(*root,
                    {"schema", "version", "mode",
                     "performance_upper_bound_only",
                     "quality_production_eligible", "converter_abi",
                     "physical_layout", "source_checkpoint_id",
                     "source_config_sha256", "source_index_sha256",
                     "manifest_sha256", "required_base_k256", "lane_count",
                     "projections"})) {
      result.diagnostic = make_diagnostic(
          PrefillAttentionFactorizedLaneConverterErrorCode::kInvalidPolicy,
          "r1.policy", "strict policy JSON schema mismatch");
      return result;
    }

    PrefillAttentionFactorizedLaneR1Policy policy;
    std::string schema;
    std::uint64_t lane_count = 0U;
    if (!json_string(*root, "schema", schema) || schema != kPolicySchema ||
        !parse_version(root->at("version"), policy.binding.version_major,
                       policy.binding.version_minor) ||
        !json_string(*root, "mode", policy.mode) ||
        !json_bool(*root, "performance_upper_bound_only",
                   policy.performance_upper_bound_only) ||
        !json_bool(*root, "quality_production_eligible",
                   policy.quality_production_eligible) ||
        !json_string(*root, "converter_abi", policy.converter_abi) ||
        !json_string(*root, "physical_layout",
                     policy.binding.physical_layout) ||
        !json_string(*root, "source_checkpoint_id",
                     policy.binding.source_checkpoint_id) ||
        !json_string(*root, "source_config_sha256",
                     policy.binding.source_config_sha256) ||
        !json_string(*root, "source_index_sha256",
                     policy.binding.source_index_sha256) ||
        !json_string(*root, "manifest_sha256",
                     policy.binding.manifest_sha256) ||
        !parse_base(root->at("required_base_k256"),
                    policy.binding.required_base_k256) ||
        !json_uint(*root, "lane_count", lane_count) ||
        lane_count > std::numeric_limits<std::uint32_t>::max()) {
      result.diagnostic = make_diagnostic(
          PrefillAttentionFactorizedLaneConverterErrorCode::kInvalidPolicy,
          "r1.policy", "policy identity fields are invalid");
      return result;
    }
    policy.binding.lane_count = static_cast<std::uint32_t>(lane_count);
    if (!policy_identity_matches_manifest(policy, manifest)) {
      result.diagnostic = make_diagnostic(
          PrefillAttentionFactorizedLaneConverterErrorCode::
              kSourceBindingMismatch,
          "r1.policy", "policy differs from manifest/base/upper-bound ABI");
      return result;
    }

    const auto* const projections = root->at("projections").as_array();
    if (projections == nullptr ||
        projections->size() != manifest.projections.size()) {
      result.diagnostic = make_diagnostic(
          PrefillAttentionFactorizedLaneConverterErrorCode::kInvalidPolicy,
          "r1.policy.projections", "policy must cover all 208 projections");
      return result;
    }
    policy.binding.projections.reserve(projections->size());
    for (std::size_t index = 0U; index < projections->size(); ++index) {
      const auto* const object = (*projections)[index].as_object();
      PrefillAttentionFactorizedLaneProjectionCalibrationBinding calibration;
      std::uint64_t ordinal = 0U;
      std::string rounding;
      if (object == nullptr ||
          !exact_keys(*object,
                      {"ordinal", "source_module", "source_sha256",
                       "weight_clip_ratio", "activation_clip_ratio",
                       "rounding", "factor"}) ||
          !json_uint(*object, "ordinal", ordinal) || ordinal != index ||
          !json_string(*object, "source_module",
                       calibration.source_module) ||
          !json_string(*object, "source_sha256",
                       calibration.source_sha256) ||
          !json_double(*object, "weight_clip_ratio",
                       calibration.weight_clip_ratio) ||
          !json_double(*object, "activation_clip_ratio",
                       calibration.activation_clip_ratio) ||
          !json_string(*object, "rounding", rounding) ||
          rounding != "nearest_even_v1") {
        result.diagnostic = make_diagnostic(
            PrefillAttentionFactorizedLaneConverterErrorCode::kInvalidPolicy,
            "r1.policy.projections[" + std::to_string(index) + "]",
            "strict projection calibration schema mismatch");
        return result;
      }
      const auto* const factor = object->at("factor").as_object();
      std::uint64_t factor_count = 0U;
      if (factor == nullptr ||
          !exact_keys(*factor, {"scheme", "path", "sha256", "count"}) ||
          !json_string(*factor, "scheme",
                       calibration.factor_source.scheme) ||
          !json_string(*factor, "path", calibration.factor_source.path) ||
          !json_string(*factor, "sha256",
                       calibration.factor_source.sha256) ||
          !json_uint(*factor, "count", factor_count)) {
        result.diagnostic = make_diagnostic(
            PrefillAttentionFactorizedLaneConverterErrorCode::kInvalidPolicy,
            "r1.policy.projections[" + std::to_string(index) + "].factor",
            "strict identity factor schema mismatch");
        return result;
      }
      calibration.ordinal = static_cast<std::uint32_t>(ordinal);
      calibration.factor_source.element_count = factor_count;
      policy.binding.projections.emplace_back(std::move(calibration));
    }
    result.diagnostic = validate_policy_entries(policy, manifest);
    if (!result.diagnostic) {
      return result;
    }
    policy.binding.policy_sha256 = sha256_text(document);
    policy.binding.policy_bytes = document.size();
    result.value.emplace(std::move(policy));
    return result;
  } catch (const std::bad_alloc&) {
    result.diagnostic = make_diagnostic(
        PrefillAttentionFactorizedLaneConverterErrorCode::kAllocationFailure,
        "r1.policy", "policy parser allocation failed");
    return result;
  } catch (...) {
    result.diagnostic = make_diagnostic(
        PrefillAttentionFactorizedLaneConverterErrorCode::kInvalidPolicy,
        "r1.policy", "unexpected strict policy parse failure");
    return result;
  }
}

namespace {

[[nodiscard]] std::string serialize_receipt(
    const PrefillAttentionFactorizedLaneR1Receipt& receipt) {
  const auto& binding = receipt.binding;
  std::ostringstream output;
  output << "{\"schema\":";
  write_quoted(output, kReceiptSchema);
  output << ",\"version\":{\"major\":" << binding.version_major
         << ",\"minor\":" << binding.version_minor << "},\"mode\":";
  write_quoted(output, receipt.mode);
  output << ",\"production_residency_eligible\":"
         << (binding.production_residency_eligible ? "true" : "false")
         << ",\"residency_eligibility_scope\":";
  write_quoted(output, receipt.residency_eligibility_scope);
  output << ",\"performance_upper_bound_only\":"
         << (receipt.performance_upper_bound_only ? "true" : "false")
         << ",\"quality_production_eligible\":"
         << (binding.quality_production_eligible ? "true" : "false")
         << ",\"converter_abi\":";
  write_quoted(output, receipt.converter_abi);
  output << ",\"physical_layout\":";
  write_quoted(output, binding.physical_layout);
  output << ",\"source_checkpoint_id\":";
  write_quoted(output, binding.source_checkpoint_id);
  output << ",\"source_config_sha256\":";
  write_quoted(output, binding.source_config_sha256);
  output << ",\"source_index_sha256\":";
  write_quoted(output, binding.source_index_sha256);
  output << ",\"manifest_sha256\":";
  write_quoted(output, binding.manifest_sha256);
  output << ",\"policy_sha256\":";
  write_quoted(output, binding.policy_sha256);
  output << ",\"policy_bytes\":" << binding.policy_bytes
         << ",\"required_base_k256\":";
  write_base(output, binding.required_base_k256);
  output << ",\"lane_count\":" << binding.lane_count
         << ",\"factor_scheme\":";
  write_quoted(output, kPrefillAttentionFactorizedLaneR1FactorScheme);
  output << ",\"payload_sha256\":";
  write_quoted(output, binding.payload.sha256);
  output << ",\"payload_bytes\":" << binding.payload.bytes
         << ",\"projection_count\":" << binding.projection_count << "}\n";
  return output.str();
}

[[nodiscard]] bool receipt_identity_matches(
    const PrefillAttentionFactorizedLaneR1Receipt& receipt,
    const PrefillAttentionFactorizedLaneOverlayManifestBinding& manifest,
    const PrefillAttentionFactorizedLaneR1Policy& policy) noexcept {
  const auto& binding = receipt.binding;
  return binding.version_major ==
             kPrefillAttentionFactorizedLaneOverlayVersionMajor &&
         binding.version_minor ==
             kPrefillAttentionFactorizedLaneOverlayVersionMinor &&
         !binding.production_residency_eligible &&
         !binding.quality_production_eligible &&
         receipt.residency_eligibility_scope ==
             kPrefillAttentionFactorizedLaneR1EligibilityScope &&
         receipt.converter_abi ==
             kPrefillAttentionFactorizedLaneR1ConverterAbi &&
         receipt.mode == kPrefillAttentionFactorizedLaneR1Mode &&
         receipt.performance_upper_bound_only &&
         !receipt.quality_production_eligible &&
         binding.physical_layout == manifest.physical_layout &&
         binding.source_checkpoint_id == manifest.source_checkpoint_id &&
         binding.source_config_sha256 == manifest.source_config_sha256 &&
         binding.source_index_sha256 == manifest.source_index_sha256 &&
         binding.manifest_sha256 == manifest.manifest_sha256 &&
         binding.policy_sha256 == policy.binding.policy_sha256 &&
         binding.policy_bytes == policy.binding.policy_bytes &&
         same_base(binding.required_base_k256,
                   manifest.required_base_k256) &&
         binding.lane_count == kPrefillAttentionFactorizedLaneR1LaneCount &&
         lower_sha256(binding.payload.sha256) &&
         binding.payload.bytes == kPrefillAttentionFactorizedLaneR1PayloadBytes &&
         binding.payload.bytes == manifest.payload_bytes &&
         binding.projection_count ==
             kPrefillAttentionFactorizedLaneProjectionCount;
}

}  // namespace

PrefillAttentionFactorizedLaneR1ReceiptResult
build_prefill_attention_factorized_lane_r1_receipt(
    const PrefillAttentionFactorizedLaneOverlayManifestBinding& manifest,
    const PrefillAttentionFactorizedLaneR1Policy& policy,
    const std::string_view payload_sha256) {
  PrefillAttentionFactorizedLaneR1ReceiptResult result;
  result.diagnostic =
      validate_prefill_attention_factorized_lane_r1_manifest(manifest);
  if (!result.diagnostic) {
    return result;
  }
  result.diagnostic = validate_policy_entries(policy, manifest);
  if (!result.diagnostic || !lower_sha256(payload_sha256) ||
      !lower_sha256(policy.binding.policy_sha256) ||
      policy.binding.policy_bytes == 0U) {
    if (result.diagnostic) {
      result.diagnostic = make_diagnostic(
          PrefillAttentionFactorizedLaneConverterErrorCode::kInvalidReceipt,
          "r1.receipt", "validated policy and payload SHA-256 are required");
    }
    return result;
  }

  try {
    PrefillAttentionFactorizedLaneR1Receipt receipt;
    receipt.binding.production_residency_eligible = false;
    receipt.binding.quality_production_eligible = false;
    receipt.binding.physical_layout = manifest.physical_layout;
    receipt.binding.source_checkpoint_id = manifest.source_checkpoint_id;
    receipt.binding.source_config_sha256 = manifest.source_config_sha256;
    receipt.binding.source_index_sha256 = manifest.source_index_sha256;
    receipt.binding.required_base_k256 = manifest.required_base_k256;
    receipt.binding.lane_count = kPrefillAttentionFactorizedLaneR1LaneCount;
    receipt.binding.manifest_sha256 = manifest.manifest_sha256;
    receipt.binding.policy_sha256 = policy.binding.policy_sha256;
    receipt.binding.policy_bytes = policy.binding.policy_bytes;
    receipt.binding.payload.sha256 = std::string(payload_sha256);
    receipt.binding.payload.bytes = manifest.payload_bytes;
    receipt.binding.projection_count = manifest.projections.size();
    receipt.converter_abi =
        std::string(kPrefillAttentionFactorizedLaneR1ConverterAbi);
    receipt.mode = std::string(kPrefillAttentionFactorizedLaneR1Mode);
    receipt.residency_eligibility_scope =
        std::string(kPrefillAttentionFactorizedLaneR1EligibilityScope);
    receipt.performance_upper_bound_only = true;
    receipt.quality_production_eligible = false;
    if (!receipt_identity_matches(receipt, manifest, policy)) {
      result.diagnostic = make_diagnostic(
          PrefillAttentionFactorizedLaneConverterErrorCode::kInvalidReceipt,
          "r1.receipt", "constructed receipt identity is inconsistent");
      return result;
    }
    result.canonical_document = serialize_receipt(receipt);
    result.value.emplace(std::move(receipt));
    return result;
  } catch (const std::bad_alloc&) {
    result.diagnostic = make_diagnostic(
        PrefillAttentionFactorizedLaneConverterErrorCode::kAllocationFailure,
        "r1.receipt", "receipt allocation failed");
    return result;
  }
}

PrefillAttentionFactorizedLaneR1ReceiptResult
parse_prefill_attention_factorized_lane_r1_receipt(
    const std::string_view document,
    const PrefillAttentionFactorizedLaneOverlayManifestBinding& manifest,
    const PrefillAttentionFactorizedLaneR1Policy& policy) {
  PrefillAttentionFactorizedLaneR1ReceiptResult result;
  result.diagnostic =
      validate_prefill_attention_factorized_lane_r1_manifest(manifest);
  if (!result.diagnostic) {
    return result;
  }
  result.diagnostic = validate_policy_entries(policy, manifest);
  if (!result.diagnostic) {
    return result;
  }

  try {
    json::ParseOptions options;
    options.max_input_bytes = 64U * 1024U;
    options.max_nesting_depth = 8U;
    options.max_values = 128U;
    options.max_container_items = 128U;
    const json::ParseResult parsed = json::parse(document, options);
    const auto* const root = parsed ? parsed.value->as_object() : nullptr;
    if (root == nullptr ||
        !exact_keys(*root,
                    {"schema", "version", "mode",
                     "production_residency_eligible",
                     "residency_eligibility_scope",
                     "performance_upper_bound_only",
                     "quality_production_eligible", "converter_abi",
                     "physical_layout", "source_checkpoint_id",
                     "source_config_sha256", "source_index_sha256",
                     "manifest_sha256", "policy_sha256", "policy_bytes",
                     "required_base_k256", "lane_count", "factor_scheme",
                     "payload_sha256", "payload_bytes",
                     "projection_count"})) {
      result.diagnostic = make_diagnostic(
          PrefillAttentionFactorizedLaneConverterErrorCode::kInvalidReceipt,
          "r1.receipt", "strict receipt JSON schema mismatch");
      return result;
    }

    PrefillAttentionFactorizedLaneR1Receipt receipt;
    std::string schema;
    std::string factor_scheme;
    std::uint64_t lane_count = 0U;
    bool quality_production_eligible = false;
    if (!json_string(*root, "schema", schema) || schema != kReceiptSchema ||
        !parse_version(root->at("version"), receipt.binding.version_major,
                       receipt.binding.version_minor) ||
        !json_string(*root, "mode", receipt.mode) ||
        !json_bool(*root, "production_residency_eligible",
                   receipt.binding.production_residency_eligible) ||
        !json_string(*root, "residency_eligibility_scope",
                     receipt.residency_eligibility_scope) ||
        !json_bool(*root, "performance_upper_bound_only",
                   receipt.performance_upper_bound_only) ||
        !json_bool(*root, "quality_production_eligible",
                   quality_production_eligible) ||
        !json_string(*root, "converter_abi", receipt.converter_abi) ||
        !json_string(*root, "physical_layout",
                     receipt.binding.physical_layout) ||
        !json_string(*root, "source_checkpoint_id",
                     receipt.binding.source_checkpoint_id) ||
        !json_string(*root, "source_config_sha256",
                     receipt.binding.source_config_sha256) ||
        !json_string(*root, "source_index_sha256",
                     receipt.binding.source_index_sha256) ||
        !json_string(*root, "manifest_sha256",
                     receipt.binding.manifest_sha256) ||
        !json_string(*root, "policy_sha256",
                     receipt.binding.policy_sha256) ||
        !json_uint(*root, "policy_bytes", receipt.binding.policy_bytes) ||
        !parse_base(root->at("required_base_k256"),
                    receipt.binding.required_base_k256) ||
        !json_uint(*root, "lane_count", lane_count) ||
        lane_count > std::numeric_limits<std::uint32_t>::max() ||
        !json_string(*root, "factor_scheme", factor_scheme) ||
        !json_string(*root, "payload_sha256",
                     receipt.binding.payload.sha256) ||
        !json_uint(*root, "payload_bytes", receipt.binding.payload.bytes) ||
        !json_uint(*root, "projection_count",
                   receipt.binding.projection_count)) {
      result.diagnostic = make_diagnostic(
          PrefillAttentionFactorizedLaneConverterErrorCode::kInvalidReceipt,
          "r1.receipt", "receipt identity fields are invalid");
      return result;
    }
    receipt.binding.lane_count = static_cast<std::uint32_t>(lane_count);
    receipt.binding.quality_production_eligible =
        quality_production_eligible;
    receipt.quality_production_eligible = quality_production_eligible;
    if (factor_scheme != kPrefillAttentionFactorizedLaneR1FactorScheme ||
        !receipt_identity_matches(receipt, manifest, policy)) {
      result.diagnostic = make_diagnostic(
          PrefillAttentionFactorizedLaneConverterErrorCode::
              kSourceBindingMismatch,
          "r1.receipt",
          "receipt differs from manifest/policy/upper-bound-only ABI");
      return result;
    }
    result.value.emplace(std::move(receipt));
    return result;
  } catch (const std::bad_alloc&) {
    result.diagnostic = make_diagnostic(
        PrefillAttentionFactorizedLaneConverterErrorCode::kAllocationFailure,
        "r1.receipt", "receipt parser allocation failed");
    return result;
  } catch (...) {
    result.diagnostic = make_diagnostic(
        PrefillAttentionFactorizedLaneConverterErrorCode::kInvalidReceipt,
        "r1.receipt", "unexpected strict receipt parse failure");
    return result;
  }
}

PrefillAttentionFactorizedLaneConverterDiagnostic
transform_prefill_attention_k256_to_factorized_r1_consumer_blocks(
    const std::uint8_t* const base_packed_signed_w4,
    const std::size_t base_packed_signed_w4_bytes,
    const std::uint8_t* const base_bf16_scales_little_endian,
    const std::size_t base_bf16_scale_bytes, const std::size_t row_count,
    const std::size_t input_size, const double weight_clip_ratio,
    std::uint8_t* const r1_packed_signed_w4,
    const std::size_t r1_packed_signed_w4_bytes,
    std::uint8_t* const r1_bf16_scales_little_endian,
    const std::size_t r1_bf16_scale_bytes) {
  const auto source =
      transform_prefill_mlp_k256_to_factorized_r1_consumer_blocks(
          base_packed_signed_w4, base_packed_signed_w4_bytes,
          base_bf16_scales_little_endian, base_bf16_scale_bytes, row_count,
          input_size, weight_clip_ratio, r1_packed_signed_w4,
          r1_packed_signed_w4_bytes, r1_bf16_scales_little_endian,
          r1_bf16_scale_bytes);
  if (source.ok()) {
    return {};
  }
  PrefillAttentionFactorizedLaneConverterErrorCode code =
      PrefillAttentionFactorizedLaneConverterErrorCode::kQuantizationFailure;
  switch (source.code) {
    case PrefillMLPFactorizedLaneConverterErrorCode::kInvalidOption:
      code = PrefillAttentionFactorizedLaneConverterErrorCode::kInvalidOption;
      break;
    case PrefillMLPFactorizedLaneConverterErrorCode::kArithmeticOverflow:
      code = PrefillAttentionFactorizedLaneConverterErrorCode::
          kArithmeticOverflow;
      break;
    case PrefillMLPFactorizedLaneConverterErrorCode::kAllocationFailure:
      code = PrefillAttentionFactorizedLaneConverterErrorCode::
          kAllocationFailure;
      break;
    case PrefillMLPFactorizedLaneConverterErrorCode::kQuantizationFailure:
      code = PrefillAttentionFactorizedLaneConverterErrorCode::
          kQuantizationFailure;
      break;
    default:
      break;
  }
  return make_diagnostic(code, source.context, source.message,
                         source.expected, source.actual);
}

namespace {

[[nodiscard]] PrefillAttentionFactorizedLaneConverterDiagnostic
file_diagnostic(const PrefillAttentionFactorizedLaneConverterErrorCode code,
                std::string context, std::string message,
                const int error = 0) {
  return make_diagnostic(code, std::move(context), std::move(message), {},
                         error == 0 ? std::string{}
                                    : std::to_string(error));
}

[[nodiscard]] bool capture_snapshot(const int fd, FileSnapshot& output,
                                    int& error) noexcept {
  struct stat status {};
  if (::fstat(fd, &status) != 0 || !S_ISREG(status.st_mode) ||
      status.st_size < 0) {
    error = errno != 0 ? errno : EINVAL;
    return false;
  }
  output.device = static_cast<std::uint64_t>(status.st_dev);
  output.inode = static_cast<std::uint64_t>(status.st_ino);
  output.size = static_cast<std::uint64_t>(status.st_size);
  output.mtime_s = status.st_mtim.tv_sec;
  output.mtime_ns = status.st_mtim.tv_nsec;
  output.ctime_s = status.st_ctim.tv_sec;
  output.ctime_ns = status.st_ctim.tv_nsec;
  return true;
}

[[nodiscard]] bool same_snapshot(const FileSnapshot& left,
                                 const FileSnapshot& right) noexcept {
  return left.device == right.device && left.inode == right.inode &&
         left.size == right.size && left.mtime_s == right.mtime_s &&
         left.mtime_ns == right.mtime_ns && left.ctime_s == right.ctime_s &&
         left.ctime_ns == right.ctime_ns;
}

[[nodiscard]] bool offset_fits(const std::uint64_t offset) noexcept {
  return offset <=
         static_cast<std::uint64_t>(std::numeric_limits<off_t>::max());
}

[[nodiscard]] bool pread_exact(const int fd, void* const destination,
                               const std::size_t byte_count,
                               const std::uint64_t offset,
                               int& error) noexcept {
  if (!offset_fits(offset)) {
    error = EOVERFLOW;
    return false;
  }
  auto* output = static_cast<std::uint8_t*>(destination);
  std::size_t completed = 0U;
  while (completed < byte_count) {
    std::uint64_t position = 0U;
    if (!checked_add(offset, completed, position) || !offset_fits(position)) {
      error = EOVERFLOW;
      return false;
    }
    const ssize_t count =
        ::pread(fd, output + completed, byte_count - completed,
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
                                const std::size_t byte_count,
                                const std::uint64_t offset,
                                int& error) noexcept {
  if (!offset_fits(offset)) {
    error = EOVERFLOW;
    return false;
  }
  const auto* input = static_cast<const std::uint8_t*>(source);
  std::size_t completed = 0U;
  while (completed < byte_count) {
    std::uint64_t position = 0U;
    if (!checked_add(offset, completed, position) || !offset_fits(position)) {
      error = EOVERFLOW;
      return false;
    }
    const ssize_t count =
        ::pwrite(fd, input + completed, byte_count - completed,
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

[[nodiscard]] bool pwrite_zeros(const int fd,
                                const std::uint64_t byte_count,
                                const std::uint64_t offset,
                                int& error) noexcept {
  std::array<std::uint8_t, 4096U> zeros{};
  std::uint64_t completed = 0U;
  while (completed < byte_count) {
    const std::size_t count = static_cast<std::size_t>(
        std::min<std::uint64_t>(zeros.size(), byte_count - completed));
    std::uint64_t position = 0U;
    if (!checked_add(offset, completed, position) ||
        !pwrite_exact(fd, zeros.data(), count, position, error)) {
      return false;
    }
    completed += count;
  }
  return true;
}

[[nodiscard]] PrefillAttentionFactorizedLaneConverterDiagnostic hash_open_file(
    const int fd, const std::uint64_t byte_count, std::string& digest) {
  try {
    constexpr std::size_t kChunk = 8U * 1024U * 1024U;
    std::vector<std::uint8_t> buffer(static_cast<std::size_t>(
        std::min<std::uint64_t>(byte_count, kChunk)));
    core::Sha256 hasher;
    std::uint64_t offset = 0U;
    while (offset < byte_count) {
      const std::size_t count = static_cast<std::size_t>(
          std::min<std::uint64_t>(buffer.size(), byte_count - offset));
      int error = 0;
      if (!pread_exact(fd, buffer.data(), count, offset, error)) {
        return file_diagnostic(
            PrefillAttentionFactorizedLaneConverterErrorCode::kIoFailure,
            "sha256", "failed to read open file while hashing", error);
      }
      if (!hasher.update(buffer.data(), count)) {
        return make_diagnostic(
            PrefillAttentionFactorizedLaneConverterErrorCode::
                kArithmeticOverflow,
            "sha256", "SHA-256 input length overflowed");
      }
      offset += count;
    }
    digest = hasher.finalize().hex();
    return {};
  } catch (const std::bad_alloc&) {
    return make_diagnostic(
        PrefillAttentionFactorizedLaneConverterErrorCode::kAllocationFailure,
        "sha256", "hash buffer allocation failed");
  }
}

[[nodiscard]] PrefillAttentionFactorizedLaneConverterDiagnostic
read_locked_document(const fs::path& path,
                     const std::uint64_t maximum_bytes,
                     LockedDocument& output) {
  output.fd = UniqueFd(::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  if (!output.fd) {
    return file_diagnostic(
        PrefillAttentionFactorizedLaneConverterErrorCode::kOpenFailed,
        path.string(), "failed to open non-symlink document", errno);
  }
  if (::flock(output.fd.get(), LOCK_SH | LOCK_NB) != 0) {
    return file_diagnostic(
        PrefillAttentionFactorizedLaneConverterErrorCode::
            kBaseAuthenticationFailed,
        path.string(), "document is concurrently locked for mutation", errno);
  }
  struct stat status {};
  int error = 0;
  if (::fstat(output.fd.get(), &status) != 0 ||
      !S_ISREG(status.st_mode) || status.st_size <= 0 ||
      static_cast<std::uint64_t>(status.st_size) > maximum_bytes ||
      static_cast<std::uint64_t>(status.st_size) >
          std::numeric_limits<std::size_t>::max() ||
      (status.st_mode & (S_IWUSR | S_IWGRP | S_IWOTH)) != 0 ||
      status.st_uid != ::geteuid() || status.st_nlink != 1 ||
      !capture_snapshot(output.fd.get(), output.snapshot, error)) {
    return file_diagnostic(
        PrefillAttentionFactorizedLaneConverterErrorCode::
            kBaseAuthenticationFailed,
        path.string(),
        "document must be bounded, owner-held, read-only, and singly linked",
        error != 0 ? error : errno);
  }
  try {
    output.document.resize(static_cast<std::size_t>(status.st_size));
  } catch (const std::bad_alloc&) {
    return make_diagnostic(
        PrefillAttentionFactorizedLaneConverterErrorCode::kAllocationFailure,
        path.string(), "document allocation failed");
  }
  if (!pread_exact(output.fd.get(), output.document.data(),
                   output.document.size(), 0U, error)) {
    return file_diagnostic(
        PrefillAttentionFactorizedLaneConverterErrorCode::kIoFailure,
        path.string(), "failed to read complete locked document", error);
  }
  FileSnapshot after;
  if (!capture_snapshot(output.fd.get(), after, error) ||
      !same_snapshot(output.snapshot, after)) {
    return file_diagnostic(
        PrefillAttentionFactorizedLaneConverterErrorCode::kDigestMismatch,
        path.string(), "document changed while being read", error);
  }
  output.sha256 = sha256_text(output.document);
  if (!lower_sha256(output.sha256)) {
    return make_diagnostic(
        PrefillAttentionFactorizedLaneConverterErrorCode::kDigestMismatch,
        path.string(), "failed to hash exact document bytes");
  }
  return {};
}

[[nodiscard]] PrefillAttentionFactorizedLaneConverterDiagnostic
revalidate_locked_document(const LockedDocument& document,
                           const fs::path& path) {
  FileSnapshot after;
  int error = 0;
  if (!capture_snapshot(document.fd.get(), after, error)) {
    return file_diagnostic(
        PrefillAttentionFactorizedLaneConverterErrorCode::kIoFailure,
        path.string(), "failed to revalidate locked document", error);
  }
  if (!same_snapshot(document.snapshot, after)) {
    return make_diagnostic(
        PrefillAttentionFactorizedLaneConverterErrorCode::kDigestMismatch,
        path.string(), "locked document changed during conversion");
  }
  return {};
}

void remove_if_present(const fs::path& path) noexcept {
  if (!path.empty()) {
    (void)::unlink(path.c_str());
  }
}

[[nodiscard]] UniqueFd create_temporary_file_near(
    const fs::path& target, const std::string_view tag, fs::path& path,
    PrefillAttentionFactorizedLaneConverterDiagnostic& diagnostic) {
  try {
    std::string pattern = target.string() + ".tmp." + std::string(tag) +
                          ".XXXXXX";
    std::vector<char> mutable_pattern(pattern.begin(), pattern.end());
    mutable_pattern.push_back('\0');
    const int fd = ::mkstemp(mutable_pattern.data());
    if (fd < 0) {
      diagnostic = file_diagnostic(
          PrefillAttentionFactorizedLaneConverterErrorCode::kOpenFailed,
          target.string(), "failed to create unique temporary file", errno);
      return {};
    }
    UniqueFd output(fd);
    path = fs::path(mutable_pattern.data());
    const int flags = ::fcntl(fd, F_GETFD);
    if (flags < 0 || ::fcntl(fd, F_SETFD, flags | FD_CLOEXEC) != 0 ||
        ::fchmod(fd, S_IRUSR | S_IWUSR) != 0 ||
        ::flock(fd, LOCK_EX | LOCK_NB) != 0) {
      diagnostic = file_diagnostic(
          PrefillAttentionFactorizedLaneConverterErrorCode::kIoFailure,
          path.string(), "failed to secure temporary descriptor", errno);
      return {};
    }
    return output;
  } catch (const std::bad_alloc&) {
    diagnostic = make_diagnostic(
        PrefillAttentionFactorizedLaneConverterErrorCode::kAllocationFailure,
        target.string(), "temporary pathname allocation failed");
    return {};
  }
}

[[nodiscard]] bool target_absent(
    const fs::path& path,
    PrefillAttentionFactorizedLaneConverterDiagnostic& diagnostic) {
  struct stat status {};
  if (::lstat(path.c_str(), &status) == 0) {
    diagnostic = make_diagnostic(
        PrefillAttentionFactorizedLaneConverterErrorCode::
            kPublicationConflict,
        path.string(), "publication target already exists");
    return false;
  }
  if (errno != ENOENT) {
    diagnostic = file_diagnostic(
        PrefillAttentionFactorizedLaneConverterErrorCode::kIoFailure,
        path.string(), "failed to inspect publication target", errno);
    return false;
  }
  return true;
}

[[nodiscard]] bool same_normalized_path(const fs::path& left,
                                        const fs::path& right) {
  std::error_code left_error;
  std::error_code right_error;
  const fs::path left_absolute =
      fs::absolute(left, left_error).lexically_normal();
  const fs::path right_absolute =
      fs::absolute(right, right_error).lexically_normal();
  return !left_error && !right_error && left_absolute == right_absolute;
}

[[nodiscard]] PrefillAttentionFactorizedLaneConverterDiagnostic seal_document(
    const int fd, const fs::path& path, const std::string_view document) {
  int error = 0;
  if (!pwrite_exact(fd, document.data(), document.size(), 0U, error) ||
      !offset_fits(document.size()) ||
      ::ftruncate(fd, static_cast<off_t>(document.size())) != 0 ||
      ::fsync(fd) != 0) {
    return file_diagnostic(
        PrefillAttentionFactorizedLaneConverterErrorCode::kIoFailure,
        path.string(), "failed to write publication document",
        error != 0 ? error : errno);
  }
  FileSnapshot before;
  FileSnapshot after;
  int snapshot_error = 0;
  std::string digest;
  if (!capture_snapshot(fd, before, snapshot_error) ||
      before.size != document.size()) {
    return file_diagnostic(
        PrefillAttentionFactorizedLaneConverterErrorCode::kIoFailure,
        path.string(), "publication document snapshot/size is invalid",
        snapshot_error);
  }
  auto diagnostic = hash_open_file(fd, before.size, digest);
  if (!diagnostic) {
    diagnostic.context = path.string();
    return diagnostic;
  }
  const std::string expected_digest = sha256_text(document);
  if (!capture_snapshot(fd, after, snapshot_error) ||
      !same_snapshot(before, after) || digest != expected_digest) {
    return make_diagnostic(
        PrefillAttentionFactorizedLaneConverterErrorCode::kDigestMismatch,
        path.string(), "publication document changed or failed readback hash",
        expected_digest, digest);
  }
  if (::fchmod(fd, S_IRUSR) != 0 || ::fsync(fd) != 0) {
    return file_diagnostic(
        PrefillAttentionFactorizedLaneConverterErrorCode::kIoFailure,
        path.string(), "failed to seal publication document read-only", errno);
  }
  return {};
}

[[nodiscard]] bool secured_temp_matches_fd(const fs::path& path,
                                           const int fd) noexcept {
  struct stat path_status {};
  struct stat fd_status {};
  return ::lstat(path.c_str(), &path_status) == 0 &&
         ::fstat(fd, &fd_status) == 0 && S_ISREG(path_status.st_mode) &&
         path_status.st_dev == fd_status.st_dev &&
         path_status.st_ino == fd_status.st_ino && fd_status.st_nlink == 1 &&
         (fd_status.st_mode & (S_IWUSR | S_IWGRP | S_IWOTH)) == 0;
}

[[nodiscard]] PrefillAttentionFactorizedLaneConverterDiagnostic
publish_three_no_replace(
    const fs::path& payload_temp, const int payload_fd,
    const fs::path& payload, const fs::path& policy_temp, const int policy_fd,
    const fs::path& policy, const fs::path& receipt_temp,
    const int receipt_fd, const fs::path& receipt) {
  if (!secured_temp_matches_fd(payload_temp, payload_fd) ||
      !secured_temp_matches_fd(policy_temp, policy_fd) ||
      !secured_temp_matches_fd(receipt_temp, receipt_fd)) {
    return make_diagnostic(
        PrefillAttentionFactorizedLaneConverterErrorCode::
            kPublicationConflict,
        payload_temp.string(),
        "temporary path no longer names its secured read-only descriptor");
  }
  bool payload_linked = false;
  bool policy_linked = false;
  bool receipt_linked = false;
  const auto rollback = [&]() noexcept {
    if (receipt_linked) (void)::unlink(receipt.c_str());
    if (policy_linked) (void)::unlink(policy.c_str());
    if (payload_linked) (void)::unlink(payload.c_str());
  };
  if (::link(payload_temp.c_str(), payload.c_str()) != 0) {
    return file_diagnostic(
        PrefillAttentionFactorizedLaneConverterErrorCode::
            kPublicationConflict,
        payload.string(), "payload no-replace link failed", errno);
  }
  payload_linked = true;
  if (::link(policy_temp.c_str(), policy.c_str()) != 0) {
    const int saved = errno;
    rollback();
    return file_diagnostic(
        PrefillAttentionFactorizedLaneConverterErrorCode::
            kPublicationConflict,
        policy.string(), "policy no-replace link failed", saved);
  }
  policy_linked = true;
  if (::link(receipt_temp.c_str(), receipt.c_str()) != 0) {
    const int saved = errno;
    rollback();
    return file_diagnostic(
        PrefillAttentionFactorizedLaneConverterErrorCode::
            kPublicationConflict,
        receipt.string(), "receipt no-replace link failed", saved);
  }
  receipt_linked = true;
  if (::unlink(payload_temp.c_str()) != 0 ||
      ::unlink(policy_temp.c_str()) != 0 ||
      ::unlink(receipt_temp.c_str()) != 0) {
    const int saved = errno;
    rollback();
    return file_diagnostic(
        PrefillAttentionFactorizedLaneConverterErrorCode::kIoFailure,
        payload.string(), "temporary unlink failed; publication rolled back",
        saved);
  }
  const fs::path parent =
      payload.parent_path().empty() ? fs::path(".") : payload.parent_path();
  UniqueFd directory(::open(parent.c_str(),
                            O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
  if (!directory || ::fsync(directory.get()) != 0) {
    const int saved = errno;
    rollback();
    if (directory) {
      (void)::fsync(directory.get());
    }
    return file_diagnostic(
        PrefillAttentionFactorizedLaneConverterErrorCode::kIoFailure,
        parent.string(), "directory sync failed; publication rolled back",
        saved);
  }
  return {};
}

[[nodiscard]] const PrefillProjectionSidecarEntry* find_base_projection(
    const PrefillSidecarManifest& base, const std::string_view module) {
  const auto found = std::find_if(
      base.projections.begin(), base.projections.end(),
      [module](const PrefillProjectionSidecarEntry& entry) {
        return entry.source_module == module;
      });
  return found == base.projections.end() ? nullptr : &*found;
}

}  // namespace

std::string_view to_string(
    const PrefillAttentionFactorizedLaneConverterErrorCode code) noexcept {
  switch (code) {
    case PrefillAttentionFactorizedLaneConverterErrorCode::kNone:
      return "none";
    case PrefillAttentionFactorizedLaneConverterErrorCode::kInvalidOption:
      return "invalid_option";
    case PrefillAttentionFactorizedLaneConverterErrorCode::kInvalidManifest:
      return "invalid_manifest";
    case PrefillAttentionFactorizedLaneConverterErrorCode::kInvalidBaseReceipt:
      return "invalid_base_receipt";
    case PrefillAttentionFactorizedLaneConverterErrorCode::
        kBaseAuthenticationFailed:
      return "base_authentication_failed";
    case PrefillAttentionFactorizedLaneConverterErrorCode::
        kSourceBindingMismatch:
      return "source_binding_mismatch";
    case PrefillAttentionFactorizedLaneConverterErrorCode::kInvalidPolicy:
      return "invalid_policy";
    case PrefillAttentionFactorizedLaneConverterErrorCode::kInvalidReceipt:
      return "invalid_receipt";
    case PrefillAttentionFactorizedLaneConverterErrorCode::kArithmeticOverflow:
      return "arithmetic_overflow";
    case PrefillAttentionFactorizedLaneConverterErrorCode::kQuantizationFailure:
      return "quantization_failure";
    case PrefillAttentionFactorizedLaneConverterErrorCode::kDigestMismatch:
      return "digest_mismatch";
    case PrefillAttentionFactorizedLaneConverterErrorCode::kUnsafePath:
      return "unsafe_path";
    case PrefillAttentionFactorizedLaneConverterErrorCode::kOpenFailed:
      return "open_failed";
    case PrefillAttentionFactorizedLaneConverterErrorCode::kIoFailure:
      return "io_failure";
    case PrefillAttentionFactorizedLaneConverterErrorCode::
        kPublicationConflict:
      return "publication_conflict";
    case PrefillAttentionFactorizedLaneConverterErrorCode::kAllocationFailure:
      return "allocation_failure";
  }
  return "invalid";
}

}  // namespace q3x::runtime
