#include "q3x/runtime/prefill_mlp_factorized_lane_r4_publication.h"

#include "q3x/core/sha256.h"
#include "q3x/io/json.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <new>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace q3x::runtime {
namespace {

namespace json = q3x::io::json;

constexpr std::string_view kManifestSchema =
    "q3x.prefill.mlp-factorized-r4.direct-manifest";
constexpr std::string_view kPolicySchema =
    "q3x.prefill.mlp-factorized-r4.direct-policy";
constexpr std::string_view kReceiptSchema =
    "q3x.prefill.mlp-factorized-r4.direct-receipt";
constexpr std::string_view kRoundingScheme = "nearest_even_v1";

[[nodiscard]] PrefillMLPFactorizedLaneR4PublicationDiagnostic
make_diagnostic(
    const PrefillMLPFactorizedLaneR4PublicationErrorCode code,
    std::string context, std::string message, std::string expected = {},
    std::string actual = {}) {
  PrefillMLPFactorizedLaneR4PublicationDiagnostic result;
  result.code = code;
  result.context = std::move(context);
  result.message = std::move(message);
  result.expected = std::move(expected);
  result.actual = std::move(actual);
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
  return std::isfinite(value) &&
         value >=
             kPrefillMLPFactorizedLaneR4PublicationMinimumClipRatio &&
         value <= 1.0 && std::isfinite(narrowed) &&
         narrowed >= static_cast<float>(
                         kPrefillMLPFactorizedLaneR4PublicationMinimumClipRatio) &&
         narrowed <= 1.0F;
}

[[nodiscard]] bool valid_factor_path(const std::string_view value) noexcept {
  if (value.empty() || value.front() == '/' || value.back() == '/') {
    return false;
  }
  std::size_t segment_begin = 0U;
  for (std::size_t index = 0U; index <= value.size(); ++index) {
    if (index != value.size() && value[index] != '/') {
      const unsigned char character =
          static_cast<unsigned char>(value[index]);
      if (character < 0x20U || value[index] == '\\') {
        return false;
      }
      continue;
    }
    const std::string_view segment =
        value.substr(segment_begin, index - segment_begin);
    if (segment.empty() || segment == "." || segment == "..") {
      return false;
    }
    if (index != value.size()) {
      segment_begin = index + 1U;
    }
  }
  return true;
}

[[nodiscard]] std::string sha256_text(const std::string_view bytes) {
  core::Sha256 hasher;
  if (!hasher.update(bytes.data(), bytes.size())) {
    return {};
  }
  return hasher.finalize().hex();
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
  return value != nullptr && value->to_double(output) &&
         std::isfinite(output);
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

[[nodiscard]] std::string_view family_name(
    const PrefillMLPFactorizedLaneProjectionFamily family) noexcept {
  switch (family) {
    case PrefillMLPFactorizedLaneProjectionFamily::kGate:
      return "gate";
    case PrefillMLPFactorizedLaneProjectionFamily::kUp:
      return "up";
    case PrefillMLPFactorizedLaneProjectionFamily::kDown:
      return "down";
  }
  return "invalid";
}

[[nodiscard]] bool parse_family(
    const std::string_view name,
    PrefillMLPFactorizedLaneProjectionFamily& family) noexcept {
  if (name == "gate") {
    family = PrefillMLPFactorizedLaneProjectionFamily::kGate;
    return true;
  }
  if (name == "up") {
    family = PrefillMLPFactorizedLaneProjectionFamily::kUp;
    return true;
  }
  if (name == "down") {
    family = PrefillMLPFactorizedLaneProjectionFamily::kDown;
    return true;
  }
  return false;
}

[[nodiscard]] bool expected_mlp_family(
    const PrefillProjectionFamily family) noexcept {
  return family == PrefillProjectionFamily::kMlpGate ||
         family == PrefillProjectionFamily::kMlpUp ||
         family == PrefillProjectionFamily::kMlpDown;
}

[[nodiscard]] PrefillMLPFactorizedLaneProjectionFamily overlay_family(
    const PrefillProjectionFamily family) noexcept {
  switch (family) {
    case PrefillProjectionFamily::kMlpGate:
      return PrefillMLPFactorizedLaneProjectionFamily::kGate;
    case PrefillProjectionFamily::kMlpUp:
      return PrefillMLPFactorizedLaneProjectionFamily::kUp;
    case PrefillProjectionFamily::kMlpDown:
      return PrefillMLPFactorizedLaneProjectionFamily::kDown;
    default:
      return static_cast<PrefillMLPFactorizedLaneProjectionFamily>(0xffU);
  }
}

[[nodiscard]] std::string expected_source_module(
    const std::uint32_t layer,
    const PrefillMLPFactorizedLaneProjectionFamily family) {
  const std::string_view projection =
      family == PrefillMLPFactorizedLaneProjectionFamily::kGate
          ? "gate_proj"
      : family == PrefillMLPFactorizedLaneProjectionFamily::kUp
          ? "up_proj"
          : "down_proj";
  return "model.language_model.layers." + std::to_string(layer) +
         ".mlp." + std::string(projection);
}

[[nodiscard]] const PrefillA4FactorizedLaneProjectionLayoutPlan&
projection_layout(
    const PrefillMLPFactorizedLaneOverlayLayoutPlan& plan,
    const PrefillMLPFactorizedLaneProjectionFamily family) noexcept {
  return family == PrefillMLPFactorizedLaneProjectionFamily::kDown
             ? plan.down
             : (family == PrefillMLPFactorizedLaneProjectionFamily::kUp
                    ? plan.up
                    : plan.gate);
}

[[nodiscard]] bool same_direct_source(
    const PrefillMLPFactorizedLaneR4DirectSourceManifestBinding& left,
    const PrefillMLPFactorizedLaneR4DirectSourceManifestBinding&
        right) noexcept {
  return left.source_manifest_kind == right.source_manifest_kind &&
         left.source_checkpoint_id == right.source_checkpoint_id &&
         left.source_config_sha256 == right.source_config_sha256 &&
         left.source_index_sha256 == right.source_index_sha256 &&
         left.source_manifest_sha256 == right.source_manifest_sha256;
}

[[nodiscard]] bool valid_direct_source(
    const PrefillMLPFactorizedLaneR4DirectSourceManifestBinding&
        source) noexcept {
  return source.source_manifest_kind ==
             kPrefillMLPFactorizedLaneR4DirectSourceKind &&
         !source.source_checkpoint_id.empty() &&
         lower_sha256(source.source_config_sha256) &&
         lower_sha256(source.source_index_sha256) &&
         lower_sha256(source.source_manifest_sha256);
}

[[nodiscard]] PrefillMLPFactorizedLaneR4DirectSourceManifestBinding
direct_source_from(const PrefillSidecarManifest& source) {
  PrefillMLPFactorizedLaneR4DirectSourceManifestBinding binding;
  binding.source_manifest_kind =
      std::string(kPrefillMLPFactorizedLaneR4DirectSourceKind);
  binding.source_checkpoint_id = source.source_checkpoint_id;
  binding.source_config_sha256 = source.source_config_sha256;
  binding.source_index_sha256 = source.source_index_sha256;
  binding.source_manifest_sha256 = source.manifest_sha256;
  return binding;
}

void write_direct_source(
    std::ostream& output,
    const PrefillMLPFactorizedLaneR4DirectSourceManifestBinding& source) {
  output << "{\"source_manifest_kind\":";
  write_quoted(output, source.source_manifest_kind);
  output << ",\"source_checkpoint_id\":";
  write_quoted(output, source.source_checkpoint_id);
  output << ",\"source_config_sha256\":";
  write_quoted(output, source.source_config_sha256);
  output << ",\"source_index_sha256\":";
  write_quoted(output, source.source_index_sha256);
  output << ",\"source_manifest_sha256\":";
  write_quoted(output, source.source_manifest_sha256);
  output << '}';
}

[[nodiscard]] bool parse_direct_source(
    const json::Value& value,
    PrefillMLPFactorizedLaneR4DirectSourceManifestBinding& source) {
  const auto* const object = value.as_object();
  return object != nullptr &&
         exact_keys(*object,
                    {"source_manifest_kind", "source_checkpoint_id",
                     "source_config_sha256", "source_index_sha256",
                     "source_manifest_sha256"}) &&
         json_string(*object, "source_manifest_kind",
                     source.source_manifest_kind) &&
         json_string(*object, "source_checkpoint_id",
                     source.source_checkpoint_id) &&
         json_string(*object, "source_config_sha256",
                     source.source_config_sha256) &&
         json_string(*object, "source_index_sha256",
                     source.source_index_sha256) &&
         json_string(*object, "source_manifest_sha256",
                     source.source_manifest_sha256) &&
         valid_direct_source(source);
}

void write_manifest_projection(
    std::ostream& output,
    const PrefillMLPFactorizedLaneManifestProjection& projection) {
  output << "{\"ordinal\":" << projection.ordinal
         << ",\"layer_index\":" << projection.layer_index
         << ",\"family\":";
  write_quoted(output, family_name(projection.family));
  output << ",\"source_module\":";
  write_quoted(output, projection.source_module);
  output << ",\"source_sha256\":";
  write_quoted(output, projection.source_sha256);
  output << ",\"output_size\":" << projection.output_size
         << ",\"input_size\":" << projection.input_size
         << ",\"payload_offset\":" << projection.payload_offset
         << ",\"payload_bytes\":" << projection.payload_bytes << '}';
}

[[nodiscard]] bool parse_manifest_projection(
    const json::Value& value,
    PrefillMLPFactorizedLaneManifestProjection& projection) {
  const auto* const object = value.as_object();
  std::uint64_t ordinal = 0U;
  std::uint64_t layer_index = 0U;
  std::string family;
  if (object == nullptr ||
      !exact_keys(*object,
                  {"ordinal", "layer_index", "family", "source_module",
                   "source_sha256", "output_size", "input_size",
                   "payload_offset", "payload_bytes"}) ||
      !json_uint(*object, "ordinal", ordinal) ||
      ordinal > std::numeric_limits<std::uint32_t>::max() ||
      !json_uint(*object, "layer_index", layer_index) ||
      layer_index > std::numeric_limits<std::uint32_t>::max() ||
      !json_string(*object, "family", family) ||
      !parse_family(family, projection.family) ||
      !json_string(*object, "source_module", projection.source_module) ||
      !json_string(*object, "source_sha256", projection.source_sha256) ||
      !json_uint(*object, "output_size", projection.output_size) ||
      !json_uint(*object, "input_size", projection.input_size) ||
      !json_uint(*object, "payload_offset", projection.payload_offset) ||
      !json_uint(*object, "payload_bytes", projection.payload_bytes)) {
    return false;
  }
  projection.ordinal = static_cast<std::uint32_t>(ordinal);
  projection.layer_index = static_cast<std::uint32_t>(layer_index);
  return true;
}

[[nodiscard]] std::string serialize_manifest(
    const PrefillMLPFactorizedLaneR4Manifest& manifest,
    const bool include_publication_identity) {
  std::ostringstream output;
  output << "{\"schema\":";
  write_quoted(output, kManifestSchema);
  output << ",\"version\":{\"major\":" << manifest.version_major
         << ",\"minor\":" << manifest.version_minor
         << "},\"physical_layout\":";
  write_quoted(output, manifest.physical_layout);
  output << ",\"direct_source\":";
  write_direct_source(output, manifest.direct_source);
  output << ",\"lane_count\":" << manifest.lane_count
         << ",\"payload_bytes\":" << manifest.payload_bytes
         << ",\"projection_count\":" << manifest.projections.size();
  if (include_publication_identity) {
    output << ",\"manifest_sha256\":";
    write_quoted(output, manifest.manifest_sha256);
    output << ",\"manifest_bytes\":" << manifest.manifest_bytes;
  }
  output << ",\"projections\":[";
  for (std::size_t index = 0U; index < manifest.projections.size(); ++index) {
    output << (index == 0U ? "" : ",");
    write_manifest_projection(output, manifest.projections[index]);
  }
  output << "]}\n";
  return output.str();
}

[[nodiscard]] PrefillMLPFactorizedLaneR4PublicationDiagnostic
validate_exact_source(const PrefillSidecarManifest& source) {
  const PrefillContractDiagnostic diagnostic =
      validate_prefill_sidecar_manifest(source);
  if (!diagnostic || source.kind != PrefillSidecarKind::kExact ||
      source.residency_class != PrefillSidecarResidencyClass::kExact ||
      source.projections.size() != kQwen36PrefillProjectionCount ||
      source.summary.mlp_projection_count !=
          kPrefillMLPFactorizedLaneProjectionCount) {
    return make_diagnostic(
        PrefillMLPFactorizedLaneR4PublicationErrorCode::
            kInvalidSourceManifest,
        "r4.direct_source",
        "validated pinned Exact PrefillSidecarManifest is required");
  }
  return {};
}

[[nodiscard]] PrefillMLPFactorizedLaneR4PublicationDiagnostic
validate_manifest_structure(
    const PrefillMLPFactorizedLaneR4Manifest& manifest) {
  const auto plan = prefill_mlp_factorized_lane_overlay_layout_plan(
      kPrefillMLPFactorizedLaneR4PublicationLaneCount);
  if (!plan ||
      manifest.version_major !=
          kPrefillMLPFactorizedLaneR4PublicationVersionMajor ||
      manifest.version_minor !=
          kPrefillMLPFactorizedLaneR4PublicationVersionMinor ||
      manifest.physical_layout != kPrefillMLPFactorizedLaneOverlayLayout ||
      !valid_direct_source(manifest.direct_source) ||
      manifest.lane_count !=
          kPrefillMLPFactorizedLaneR4PublicationLaneCount ||
      manifest.projections.size() !=
          kPrefillMLPFactorizedLaneProjectionCount ||
      manifest.payload_bytes !=
          kPrefillMLPFactorizedLaneR4PublicationPayloadBytes ||
      manifest.payload_bytes != plan.payload_bytes ||
      !lower_sha256(manifest.manifest_sha256) ||
      manifest.manifest_bytes == 0U) {
    return make_diagnostic(
        PrefillMLPFactorizedLaneR4PublicationErrorCode::kInvalidManifest,
        "r4.manifest", "R4 direct manifest header is invalid");
  }
  for (std::size_t index = 0U; index < manifest.projections.size(); ++index) {
    const auto& projection = manifest.projections[index];
    const auto family =
        static_cast<PrefillMLPFactorizedLaneProjectionFamily>(index % 3U);
    const std::uint32_t layer = static_cast<std::uint32_t>(index / 3U);
    const bool down =
        family == PrefillMLPFactorizedLaneProjectionFamily::kDown;
    if (projection.ordinal != index || projection.layer_index != layer ||
        projection.family != family ||
        projection.source_module != expected_source_module(layer, family) ||
        !lower_sha256(projection.source_sha256) ||
        projection.output_size !=
            (down ? kPrefillMLPFactorizedLaneDownOutputSize
                  : kPrefillMLPFactorizedLaneGateUpOutputSize) ||
        projection.input_size !=
            (down ? kPrefillMLPFactorizedLaneDownInputSize
                  : kPrefillMLPFactorizedLaneGateUpInputSize) ||
        projection.payload_offset !=
            prefill_mlp_factorized_lane_projection_absolute_offset(
                plan, layer, family) ||
        projection.payload_bytes !=
            projection_layout(plan, family).projection_bytes) {
      return make_diagnostic(
          PrefillMLPFactorizedLaneR4PublicationErrorCode::kInvalidManifest,
          "r4.manifest.projections[" + std::to_string(index) + "]",
          "fixed layer-major Gate/Up/Down projection is invalid");
    }
  }
  const std::string digest =
      sha256_text(serialize_manifest(manifest, false));
  if (digest != manifest.manifest_sha256) {
    return make_diagnostic(
        PrefillMLPFactorizedLaneR4PublicationErrorCode::kDigestMismatch,
        "r4.manifest.manifest_sha256", "manifest body digest mismatch",
        manifest.manifest_sha256, digest);
  }
  const std::uint64_t bytes = serialize_manifest(manifest, true).size();
  if (bytes != manifest.manifest_bytes) {
    return make_diagnostic(
        PrefillMLPFactorizedLaneR4PublicationErrorCode::kDigestMismatch,
        "r4.manifest.manifest_bytes", "canonical manifest size mismatch",
        std::to_string(manifest.manifest_bytes), std::to_string(bytes));
  }
  return {};
}

[[nodiscard]] json::ParseOptions manifest_parse_options() {
  json::ParseOptions options;
  options.max_input_bytes = 512U * 1024U;
  options.max_nesting_depth = 8U;
  options.max_values = 4'000U;
  options.max_container_items = 4'000U;
  return options;
}

}  // namespace

std::string prefill_mlp_factorized_lane_r4_manifest_sha256(
    const PrefillMLPFactorizedLaneR4Manifest& manifest) {
  return sha256_text(serialize_manifest(manifest, false));
}

PrefillMLPFactorizedLaneR4PublicationDiagnostic
validate_prefill_mlp_factorized_lane_r4_direct_manifest(
    const PrefillMLPFactorizedLaneR4Manifest& manifest,
    const PrefillSidecarManifest& exact_source_manifest) {
  auto diagnostic = validate_exact_source(exact_source_manifest);
  if (!diagnostic) {
    return diagnostic;
  }
  diagnostic = validate_manifest_structure(manifest);
  if (!diagnostic) {
    return diagnostic;
  }
  const auto expected_source = direct_source_from(exact_source_manifest);
  if (!same_direct_source(manifest.direct_source, expected_source)) {
    return make_diagnostic(
        PrefillMLPFactorizedLaneR4PublicationErrorCode::
            kSourceBindingMismatch,
        "r4.manifest.direct_source",
        "manifest direct-source identity differs from pinned Exact source");
  }
  std::size_t mlp_index = 0U;
  for (const auto& source : exact_source_manifest.projections) {
    if (!expected_mlp_family(source.family)) {
      continue;
    }
    if (mlp_index >= manifest.projections.size()) {
      return make_diagnostic(
          PrefillMLPFactorizedLaneR4PublicationErrorCode::kInvalidManifest,
          "r4.manifest.projections", "source contains too many MLP entries");
    }
    const auto& projection = manifest.projections[mlp_index];
    if (projection.family != overlay_family(source.family) ||
        projection.layer_index != source.layer_index ||
        projection.source_module != source.source_module ||
        projection.source_sha256 != source.source_sha256 ||
        projection.output_size != source.output_size ||
        projection.input_size != source.input_size) {
      return make_diagnostic(
          PrefillMLPFactorizedLaneR4PublicationErrorCode::
              kSourceBindingMismatch,
          "r4.manifest.projections[" + std::to_string(mlp_index) + "]",
          "projection differs from authenticated Exact source entry");
    }
    ++mlp_index;
  }
  if (mlp_index != manifest.projections.size()) {
    return make_diagnostic(
        PrefillMLPFactorizedLaneR4PublicationErrorCode::kInvalidManifest,
        "r4.manifest.projections", "source MLP inventory is incomplete");
  }
  return {};
}

PrefillMLPFactorizedLaneR4ManifestResult
build_prefill_mlp_factorized_lane_r4_direct_manifest(
    const PrefillSidecarManifest& exact_source_manifest) {
  PrefillMLPFactorizedLaneR4ManifestResult result;
  result.diagnostic = validate_exact_source(exact_source_manifest);
  if (!result.diagnostic) {
    return result;
  }
  try {
    const auto plan = prefill_mlp_factorized_lane_overlay_layout_plan(
        kPrefillMLPFactorizedLaneR4PublicationLaneCount);
    PrefillMLPFactorizedLaneR4Manifest manifest;
    manifest.physical_layout =
        std::string(kPrefillMLPFactorizedLaneOverlayLayout);
    manifest.direct_source = direct_source_from(exact_source_manifest);
    manifest.lane_count =
        kPrefillMLPFactorizedLaneR4PublicationLaneCount;
    manifest.payload_bytes = plan.payload_bytes;
    manifest.projections.reserve(
        kPrefillMLPFactorizedLaneProjectionCount);
    for (const auto& source : exact_source_manifest.projections) {
      if (!expected_mlp_family(source.family)) {
        continue;
      }
      const std::size_t index = manifest.projections.size();
      if (index >= kPrefillMLPFactorizedLaneProjectionCount) {
        result.diagnostic = make_diagnostic(
            PrefillMLPFactorizedLaneR4PublicationErrorCode::kInvalidManifest,
            "r4.manifest.projections",
            "source contains too many MLP projections");
        return result;
      }
      const auto family = overlay_family(source.family);
      const std::uint32_t layer = static_cast<std::uint32_t>(index / 3U);
      if (family !=
              static_cast<PrefillMLPFactorizedLaneProjectionFamily>(
                  index % 3U) ||
          source.layer_index != layer) {
        result.diagnostic = make_diagnostic(
            PrefillMLPFactorizedLaneR4PublicationErrorCode::kInvalidManifest,
            "r4.manifest.projections[" + std::to_string(index) + "]",
            "Exact source is not layer-major Gate/Up/Down");
        return result;
      }
      PrefillMLPFactorizedLaneManifestProjection projection;
      projection.ordinal = static_cast<std::uint32_t>(index);
      projection.layer_index = layer;
      projection.family = family;
      projection.source_module = source.source_module;
      projection.source_sha256 = source.source_sha256;
      projection.output_size = source.output_size;
      projection.input_size = source.input_size;
      projection.payload_offset =
          prefill_mlp_factorized_lane_projection_absolute_offset(
              plan, layer, family);
      projection.payload_bytes =
          projection_layout(plan, family).projection_bytes;
      manifest.projections.emplace_back(std::move(projection));
    }
    manifest.manifest_sha256 =
        prefill_mlp_factorized_lane_r4_manifest_sha256(manifest);
    // manifest_sha256 is fixed width.  Iterate only because manifest_bytes
    // itself changes the decimal spelling of the complete document.
    for (int iteration = 0; iteration < 4; ++iteration) {
      const std::uint64_t bytes = serialize_manifest(manifest, true).size();
      if (bytes == manifest.manifest_bytes) {
        break;
      }
      manifest.manifest_bytes = bytes;
    }
    result.diagnostic =
        validate_prefill_mlp_factorized_lane_r4_direct_manifest(
            manifest, exact_source_manifest);
    if (!result.diagnostic) {
      return result;
    }
    result.canonical_document = serialize_manifest(manifest, true);
    result.value.emplace(std::move(manifest));
    return result;
  } catch (const std::bad_alloc&) {
    result.diagnostic = make_diagnostic(
        PrefillMLPFactorizedLaneR4PublicationErrorCode::kAllocationFailure,
        "r4.manifest", "manifest allocation failed");
    return result;
  }
}

PrefillMLPFactorizedLaneR4ManifestResult
parse_prefill_mlp_factorized_lane_r4_direct_manifest(
    const std::string_view document,
    const PrefillSidecarManifest& exact_source_manifest) {
  PrefillMLPFactorizedLaneR4ManifestResult result;
  result.diagnostic = validate_exact_source(exact_source_manifest);
  if (!result.diagnostic) {
    return result;
  }
  try {
    const json::ParseResult parsed =
        json::parse(document, manifest_parse_options());
    const auto* const root = parsed ? parsed.value->as_object() : nullptr;
    if (root == nullptr ||
        !exact_keys(*root,
                    {"schema", "version", "physical_layout",
                     "direct_source", "lane_count", "payload_bytes",
                     "projection_count", "manifest_sha256",
                     "manifest_bytes", "projections"})) {
      result.diagnostic = make_diagnostic(
          PrefillMLPFactorizedLaneR4PublicationErrorCode::kInvalidManifest,
          "r4.manifest", "strict manifest JSON schema mismatch");
      return result;
    }
    PrefillMLPFactorizedLaneR4Manifest manifest;
    std::string schema;
    std::uint64_t lane_count = 0U;
    std::uint64_t projection_count = 0U;
    if (!json_string(*root, "schema", schema) || schema != kManifestSchema ||
        !parse_version(root->at("version"), manifest.version_major,
                       manifest.version_minor) ||
        !json_string(*root, "physical_layout", manifest.physical_layout) ||
        !parse_direct_source(root->at("direct_source"),
                             manifest.direct_source) ||
        !json_uint(*root, "lane_count", lane_count) ||
        lane_count > std::numeric_limits<std::uint32_t>::max() ||
        !json_uint(*root, "payload_bytes", manifest.payload_bytes) ||
        !json_uint(*root, "projection_count", projection_count) ||
        !json_string(*root, "manifest_sha256",
                     manifest.manifest_sha256) ||
        !json_uint(*root, "manifest_bytes", manifest.manifest_bytes)) {
      result.diagnostic = make_diagnostic(
          PrefillMLPFactorizedLaneR4PublicationErrorCode::kInvalidManifest,
          "r4.manifest", "manifest identity fields are invalid");
      return result;
    }
    manifest.lane_count = static_cast<std::uint32_t>(lane_count);
    const auto* const projections = root->at("projections").as_array();
    if (projections == nullptr || projections->size() != projection_count ||
        projection_count != kPrefillMLPFactorizedLaneProjectionCount) {
      result.diagnostic = make_diagnostic(
          PrefillMLPFactorizedLaneR4PublicationErrorCode::kInvalidManifest,
          "r4.manifest.projections",
          "manifest must contain exactly 192 projections");
      return result;
    }
    manifest.projections.reserve(projections->size());
    for (std::size_t index = 0U; index < projections->size(); ++index) {
      PrefillMLPFactorizedLaneManifestProjection projection;
      if (!parse_manifest_projection((*projections)[index], projection)) {
        result.diagnostic = make_diagnostic(
            PrefillMLPFactorizedLaneR4PublicationErrorCode::kInvalidManifest,
            "r4.manifest.projections[" + std::to_string(index) + "]",
            "strict projection schema mismatch");
        return result;
      }
      manifest.projections.emplace_back(std::move(projection));
    }
    result.diagnostic =
        validate_prefill_mlp_factorized_lane_r4_direct_manifest(
            manifest, exact_source_manifest);
    if (!result.diagnostic) {
      return result;
    }
    const std::string canonical = serialize_manifest(manifest, true);
    if (canonical != document) {
      result.diagnostic = make_diagnostic(
          PrefillMLPFactorizedLaneR4PublicationErrorCode::kInvalidManifest,
          "r4.manifest", "manifest JSON is not canonical");
      return result;
    }
    result.canonical_document = canonical;
    result.value.emplace(std::move(manifest));
    return result;
  } catch (const std::bad_alloc&) {
    result.diagnostic = make_diagnostic(
        PrefillMLPFactorizedLaneR4PublicationErrorCode::kAllocationFailure,
        "r4.manifest", "manifest parser allocation failed");
    return result;
  } catch (...) {
    result.diagnostic = make_diagnostic(
        PrefillMLPFactorizedLaneR4PublicationErrorCode::kInvalidManifest,
        "r4.manifest", "unexpected strict manifest parse failure");
    return result;
  }
}

namespace {

[[nodiscard]] std::string serialize_policy(
    const PrefillMLPFactorizedLaneR4Policy& policy) {
  std::ostringstream output;
  output << std::setprecision(17) << "{\"schema\":";
  write_quoted(output, kPolicySchema);
  output << ",\"version\":{\"major\":" << policy.version_major
         << ",\"minor\":" << policy.version_minor << "},\"mode\":";
  write_quoted(output, policy.mode);
  output << ",\"converter_abi\":";
  write_quoted(output, policy.converter_abi);
  output << ",\"performance_candidate_only\":"
         << (policy.performance_candidate_only ? "true" : "false")
         << ",\"production_residency_eligible\":"
         << (policy.production_residency_eligible ? "true" : "false")
         << ",\"quality_production_eligible\":"
         << (policy.quality_production_eligible ? "true" : "false")
         << ",\"physical_layout\":";
  write_quoted(output, policy.physical_layout);
  output << ",\"direct_source\":";
  write_direct_source(output, policy.direct_source);
  output << ",\"manifest_sha256\":";
  write_quoted(output, policy.manifest_sha256);
  output << ",\"manifest_bytes\":" << policy.manifest_bytes
         << ",\"lane_count\":" << policy.lane_count
         << ",\"projection_count\":" << policy.projections.size()
         << ",\"projections\":[";
  for (std::size_t index = 0U; index < policy.projections.size(); ++index) {
    const auto& projection = policy.projections[index];
    output << (index == 0U ? "" : ",") << "{\"ordinal\":"
           << projection.ordinal << ",\"source_module\":";
    write_quoted(output, projection.source_module);
    output << ",\"source_sha256\":";
    write_quoted(output, projection.source_sha256);
    output << ",\"weight_clip_ratio\":" << projection.weight_clip_ratio
           << ",\"activation_clip_ratio\":"
           << projection.activation_clip_ratio << ",\"rounding\":";
    write_quoted(output, kRoundingScheme);
    output << ",\"factor\":{\"scheme\":";
    write_quoted(output, projection.factor_scheme);
    output << ",\"path\":";
    write_quoted(output, projection.factor_path);
    output << ",\"sha256\":";
    write_quoted(output, projection.factor_sha256);
    output << ",\"count\":" << projection.factor_element_count << "}}";
  }
  output << "]}\n";
  return output.str();
}

[[nodiscard]] bool same_factor_identity(
    const PrefillMLPFactorizedLaneR4ProjectionPolicyBinding& left,
    const PrefillMLPFactorizedLaneR4ProjectionPolicyBinding& right) noexcept {
  return left.factor_scheme == right.factor_scheme &&
         left.factor_path == right.factor_path &&
         left.factor_sha256 == right.factor_sha256 &&
         left.factor_element_count == right.factor_element_count;
}

[[nodiscard]] PrefillMLPFactorizedLaneR4PublicationDiagnostic
validate_policy_core(
    const PrefillMLPFactorizedLaneR4Policy& policy,
    const PrefillMLPFactorizedLaneR4Manifest& manifest,
    const bool require_policy_identity) {
  auto diagnostic = validate_manifest_structure(manifest);
  if (!diagnostic) {
    return diagnostic;
  }
  if (policy.version_major !=
          kPrefillMLPFactorizedLaneR4PublicationVersionMajor ||
      policy.version_minor !=
          kPrefillMLPFactorizedLaneR4PublicationVersionMinor ||
      policy.mode != kPrefillMLPFactorizedLaneR4PublicationMode ||
      policy.converter_abi != kPrefillMLPFactorizedLaneR4PublicationAbi ||
      !policy.performance_candidate_only ||
      policy.production_residency_eligible ||
      policy.quality_production_eligible ||
      policy.physical_layout != manifest.physical_layout ||
      !same_direct_source(policy.direct_source, manifest.direct_source) ||
      policy.manifest_sha256 != manifest.manifest_sha256 ||
      policy.manifest_bytes != manifest.manifest_bytes ||
      policy.lane_count !=
          kPrefillMLPFactorizedLaneR4PublicationLaneCount ||
      policy.projections.size() != manifest.projections.size()) {
    return make_diagnostic(
        PrefillMLPFactorizedLaneR4PublicationErrorCode::kInvalidPolicy,
        "r4.policy", "policy identity differs from direct R4 manifest");
  }
  for (std::size_t index = 0U; index < policy.projections.size(); ++index) {
    const auto& binding = policy.projections[index];
    const auto& projection = manifest.projections[index];
    if (binding.ordinal != index ||
        binding.source_module != projection.source_module ||
        binding.source_sha256 != projection.source_sha256 ||
        !valid_clip_ratio(binding.weight_clip_ratio) ||
        !valid_clip_ratio(binding.activation_clip_ratio) ||
        binding.factor_scheme !=
            kPrefillMLPFactorizedLaneR4PublicationFactorScheme ||
        !valid_factor_path(binding.factor_path) ||
        !lower_sha256(binding.factor_sha256) ||
        binding.factor_element_count != projection.input_size) {
      return make_diagnostic(
          PrefillMLPFactorizedLaneR4PublicationErrorCode::kInvalidPolicy,
          "r4.policy.projections[" + std::to_string(index) + "]",
          "projection clip/source/calibrated-alpha binding is invalid");
    }
  }
  for (std::size_t layer = 0U;
       layer < kPrefillMLPFactorizedLaneLayerCount; ++layer) {
    const auto& gate = policy.projections[layer * 3U];
    const auto& up = policy.projections[layer * 3U + 1U];
    if (gate.activation_clip_ratio != up.activation_clip_ratio ||
        !same_factor_identity(gate, up)) {
      return make_diagnostic(
          PrefillMLPFactorizedLaneR4PublicationErrorCode::kInvalidPolicy,
          "r4.policy.gate_up[" + std::to_string(layer) + "]",
          "Gate/Up must share activation clip and calibrated alpha identity");
    }
  }
  if (require_policy_identity) {
    const std::string canonical = serialize_policy(policy);
    const std::string digest = sha256_text(canonical);
    if (!lower_sha256(policy.policy_sha256) ||
        policy.policy_sha256 != digest ||
        policy.policy_bytes != canonical.size()) {
      return make_diagnostic(
          PrefillMLPFactorizedLaneR4PublicationErrorCode::kDigestMismatch,
          "r4.policy.identity", "canonical policy digest/size mismatch",
          policy.policy_sha256, digest);
    }
  }
  return {};
}

[[nodiscard]] json::ParseOptions policy_parse_options() {
  json::ParseOptions options;
  options.max_input_bytes = 512U * 1024U;
  options.max_nesting_depth = 10U;
  options.max_values = 5'000U;
  options.max_container_items = 5'000U;
  return options;
}

}  // namespace

PrefillMLPFactorizedLaneR4PolicyResult
build_prefill_mlp_factorized_lane_r4_policy(
    const PrefillMLPFactorizedLaneR4Manifest& manifest,
    const std::vector<PrefillMLPFactorizedLaneR4CalibrationSpec>&
        calibration) {
  PrefillMLPFactorizedLaneR4PolicyResult result;
  result.diagnostic = validate_manifest_structure(manifest);
  if (!result.diagnostic) {
    return result;
  }
  if (calibration.size() != manifest.projections.size()) {
    result.diagnostic = make_diagnostic(
        PrefillMLPFactorizedLaneR4PublicationErrorCode::kInvalidPolicy,
        "r4.policy.calibration",
        "calibration must cover all 192 projections");
    return result;
  }
  try {
    PrefillMLPFactorizedLaneR4Policy policy;
    policy.mode =
        std::string(kPrefillMLPFactorizedLaneR4PublicationMode);
    policy.converter_abi =
        std::string(kPrefillMLPFactorizedLaneR4PublicationAbi);
    policy.physical_layout = manifest.physical_layout;
    policy.direct_source = manifest.direct_source;
    policy.manifest_sha256 = manifest.manifest_sha256;
    policy.manifest_bytes = manifest.manifest_bytes;
    policy.lane_count = kPrefillMLPFactorizedLaneR4PublicationLaneCount;
    policy.projections.reserve(manifest.projections.size());
    for (std::size_t index = 0U; index < manifest.projections.size(); ++index) {
      const auto& source = manifest.projections[index];
      const auto& spec = calibration[index];
      PrefillMLPFactorizedLaneR4ProjectionPolicyBinding binding;
      binding.ordinal = source.ordinal;
      binding.source_module = source.source_module;
      binding.source_sha256 = source.source_sha256;
      binding.weight_clip_ratio = spec.weight_clip_ratio;
      binding.activation_clip_ratio = spec.activation_clip_ratio;
      binding.factor_scheme =
          std::string(kPrefillMLPFactorizedLaneR4PublicationFactorScheme);
      binding.factor_path = spec.alpha_path;
      binding.factor_sha256 = spec.alpha_sha256;
      binding.factor_element_count = spec.alpha_element_count;
      policy.projections.emplace_back(std::move(binding));
    }
    result.diagnostic = validate_policy_core(policy, manifest, false);
    if (!result.diagnostic) {
      return result;
    }
    result.canonical_document = serialize_policy(policy);
    policy.policy_sha256 = sha256_text(result.canonical_document);
    policy.policy_bytes = result.canonical_document.size();
    result.diagnostic = validate_policy_core(policy, manifest, true);
    if (!result.diagnostic) {
      result.canonical_document.clear();
      return result;
    }
    result.value.emplace(std::move(policy));
    return result;
  } catch (const std::bad_alloc&) {
    result.diagnostic = make_diagnostic(
        PrefillMLPFactorizedLaneR4PublicationErrorCode::kAllocationFailure,
        "r4.policy", "policy allocation failed");
    return result;
  }
}

PrefillMLPFactorizedLaneR4PolicyResult
parse_prefill_mlp_factorized_lane_r4_policy(
    const std::string_view document,
    const PrefillMLPFactorizedLaneR4Manifest& manifest) {
  PrefillMLPFactorizedLaneR4PolicyResult result;
  result.diagnostic = validate_manifest_structure(manifest);
  if (!result.diagnostic) {
    return result;
  }
  try {
    const json::ParseResult parsed =
        json::parse(document, policy_parse_options());
    const auto* const root = parsed ? parsed.value->as_object() : nullptr;
    if (root == nullptr ||
        !exact_keys(*root,
                    {"schema", "version", "mode", "converter_abi",
                     "performance_candidate_only",
                     "production_residency_eligible",
                     "quality_production_eligible", "physical_layout",
                     "direct_source", "manifest_sha256", "manifest_bytes",
                     "lane_count", "projection_count", "projections"})) {
      result.diagnostic = make_diagnostic(
          PrefillMLPFactorizedLaneR4PublicationErrorCode::kInvalidPolicy,
          "r4.policy", "strict policy JSON schema mismatch");
      return result;
    }
    PrefillMLPFactorizedLaneR4Policy policy;
    std::string schema;
    std::uint64_t lane_count = 0U;
    std::uint64_t projection_count = 0U;
    if (!json_string(*root, "schema", schema) || schema != kPolicySchema ||
        !parse_version(root->at("version"), policy.version_major,
                       policy.version_minor) ||
        !json_string(*root, "mode", policy.mode) ||
        !json_string(*root, "converter_abi", policy.converter_abi) ||
        !json_bool(*root, "performance_candidate_only",
                   policy.performance_candidate_only) ||
        !json_bool(*root, "production_residency_eligible",
                   policy.production_residency_eligible) ||
        !json_bool(*root, "quality_production_eligible",
                   policy.quality_production_eligible) ||
        !json_string(*root, "physical_layout", policy.physical_layout) ||
        !parse_direct_source(root->at("direct_source"),
                             policy.direct_source) ||
        !json_string(*root, "manifest_sha256",
                     policy.manifest_sha256) ||
        !json_uint(*root, "manifest_bytes", policy.manifest_bytes) ||
        !json_uint(*root, "lane_count", lane_count) ||
        lane_count > std::numeric_limits<std::uint32_t>::max() ||
        !json_uint(*root, "projection_count", projection_count)) {
      result.diagnostic = make_diagnostic(
          PrefillMLPFactorizedLaneR4PublicationErrorCode::kInvalidPolicy,
          "r4.policy", "policy identity fields are invalid");
      return result;
    }
    policy.lane_count = static_cast<std::uint32_t>(lane_count);
    const auto* const projections = root->at("projections").as_array();
    if (projections == nullptr || projections->size() != projection_count ||
        projection_count != manifest.projections.size()) {
      result.diagnostic = make_diagnostic(
          PrefillMLPFactorizedLaneR4PublicationErrorCode::kInvalidPolicy,
          "r4.policy.projections",
          "policy must cover all manifest projections");
      return result;
    }
    policy.projections.reserve(projections->size());
    for (std::size_t index = 0U; index < projections->size(); ++index) {
      const auto* const object = (*projections)[index].as_object();
      PrefillMLPFactorizedLaneR4ProjectionPolicyBinding binding;
      std::uint64_t ordinal = 0U;
      std::string rounding;
      if (object == nullptr ||
          !exact_keys(*object,
                      {"ordinal", "source_module", "source_sha256",
                       "weight_clip_ratio", "activation_clip_ratio",
                       "rounding", "factor"}) ||
          !json_uint(*object, "ordinal", ordinal) ||
          ordinal > std::numeric_limits<std::uint32_t>::max() ||
          !json_string(*object, "source_module", binding.source_module) ||
          !json_string(*object, "source_sha256", binding.source_sha256) ||
          !json_double(*object, "weight_clip_ratio",
                       binding.weight_clip_ratio) ||
          !json_double(*object, "activation_clip_ratio",
                       binding.activation_clip_ratio) ||
          !json_string(*object, "rounding", rounding) ||
          rounding != kRoundingScheme) {
        result.diagnostic = make_diagnostic(
            PrefillMLPFactorizedLaneR4PublicationErrorCode::kInvalidPolicy,
            "r4.policy.projections[" + std::to_string(index) + "]",
            "strict projection policy schema mismatch");
        return result;
      }
      binding.ordinal = static_cast<std::uint32_t>(ordinal);
      const auto* const factor = object->at("factor").as_object();
      if (factor == nullptr ||
          !exact_keys(*factor, {"scheme", "path", "sha256", "count"}) ||
          !json_string(*factor, "scheme", binding.factor_scheme) ||
          !json_string(*factor, "path", binding.factor_path) ||
          !json_string(*factor, "sha256", binding.factor_sha256) ||
          !json_uint(*factor, "count", binding.factor_element_count)) {
        result.diagnostic = make_diagnostic(
            PrefillMLPFactorizedLaneR4PublicationErrorCode::kInvalidPolicy,
            "r4.policy.projections[" + std::to_string(index) + "].factor",
            "strict calibrated-alpha schema mismatch");
        return result;
      }
      policy.projections.emplace_back(std::move(binding));
    }
    result.diagnostic = validate_policy_core(policy, manifest, false);
    if (!result.diagnostic) {
      return result;
    }
    const std::string canonical = serialize_policy(policy);
    if (canonical != document) {
      result.diagnostic = make_diagnostic(
          PrefillMLPFactorizedLaneR4PublicationErrorCode::kInvalidPolicy,
          "r4.policy", "policy JSON is not canonical");
      return result;
    }
    policy.policy_sha256 = sha256_text(document);
    policy.policy_bytes = document.size();
    result.diagnostic = validate_policy_core(policy, manifest, true);
    if (!result.diagnostic) {
      return result;
    }
    result.canonical_document = canonical;
    result.value.emplace(std::move(policy));
    return result;
  } catch (const std::bad_alloc&) {
    result.diagnostic = make_diagnostic(
        PrefillMLPFactorizedLaneR4PublicationErrorCode::kAllocationFailure,
        "r4.policy", "policy parser allocation failed");
    return result;
  } catch (...) {
    result.diagnostic = make_diagnostic(
        PrefillMLPFactorizedLaneR4PublicationErrorCode::kInvalidPolicy,
        "r4.policy", "unexpected strict policy parse failure");
    return result;
  }
}

PrefillMLPFactorizedLaneR4PublicationDiagnostic
validate_prefill_mlp_factorized_lane_r4_policy_binding(
    const PrefillMLPFactorizedLaneR4Policy& policy,
    const PrefillMLPFactorizedLaneR4Manifest& manifest) {
  try {
    return validate_policy_core(policy, manifest, true);
  } catch (const std::bad_alloc&) {
    return make_diagnostic(
        PrefillMLPFactorizedLaneR4PublicationErrorCode::kAllocationFailure,
        "r4.policy", "policy validation allocation failed");
  } catch (...) {
    return make_diagnostic(
        PrefillMLPFactorizedLaneR4PublicationErrorCode::kInvalidPolicy,
        "r4.policy", "unexpected strict policy validation failure");
  }
}

namespace {

[[nodiscard]] std::string serialize_receipt(
    const PrefillMLPFactorizedLaneR4Receipt& receipt) {
  std::ostringstream output;
  output << "{\"schema\":";
  write_quoted(output, kReceiptSchema);
  output << ",\"version\":{\"major\":" << receipt.version_major
         << ",\"minor\":" << receipt.version_minor << "},\"mode\":";
  write_quoted(output, receipt.mode);
  output << ",\"converter_abi\":";
  write_quoted(output, receipt.converter_abi);
  output << ",\"performance_candidate_only\":"
         << (receipt.performance_candidate_only ? "true" : "false")
         << ",\"production_residency_eligible\":"
         << (receipt.production_residency_eligible ? "true" : "false")
         << ",\"quality_production_eligible\":"
         << (receipt.quality_production_eligible ? "true" : "false")
         << ",\"physical_layout\":";
  write_quoted(output, receipt.physical_layout);
  output << ",\"direct_source\":";
  write_direct_source(output, receipt.direct_source);
  output << ",\"lane_count\":" << receipt.lane_count
         << ",\"factor_scheme\":";
  write_quoted(output,
               kPrefillMLPFactorizedLaneR4PublicationFactorScheme);
  output << ",\"manifest_sha256\":";
  write_quoted(output, receipt.manifest_sha256);
  output << ",\"manifest_bytes\":" << receipt.manifest_bytes
         << ",\"policy_sha256\":";
  write_quoted(output, receipt.policy_sha256);
  output << ",\"policy_bytes\":" << receipt.policy_bytes
         << ",\"payload_sha256\":";
  write_quoted(output, receipt.payload_sha256);
  output << ",\"payload_bytes\":" << receipt.payload_bytes
         << ",\"projection_count\":" << receipt.projection_count
         << "}\n";
  return output.str();
}

[[nodiscard]] PrefillMLPFactorizedLaneR4PublicationDiagnostic
validate_receipt_core(
    const PrefillMLPFactorizedLaneR4Receipt& receipt,
    const PrefillMLPFactorizedLaneR4Manifest& manifest,
    const PrefillMLPFactorizedLaneR4Policy& policy) {
  auto diagnostic = validate_policy_core(policy, manifest, true);
  if (!diagnostic) {
    return diagnostic;
  }
  if (receipt.version_major !=
          kPrefillMLPFactorizedLaneR4PublicationVersionMajor ||
      receipt.version_minor !=
          kPrefillMLPFactorizedLaneR4PublicationVersionMinor ||
      receipt.mode != kPrefillMLPFactorizedLaneR4PublicationMode ||
      receipt.converter_abi !=
          kPrefillMLPFactorizedLaneR4PublicationAbi ||
      !receipt.performance_candidate_only ||
      receipt.production_residency_eligible ||
      receipt.quality_production_eligible ||
      receipt.physical_layout != manifest.physical_layout ||
      !same_direct_source(receipt.direct_source,
                          manifest.direct_source) ||
      receipt.lane_count !=
          kPrefillMLPFactorizedLaneR4PublicationLaneCount ||
      receipt.manifest_sha256 != manifest.manifest_sha256 ||
      receipt.manifest_bytes != manifest.manifest_bytes ||
      receipt.policy_sha256 != policy.policy_sha256 ||
      receipt.policy_bytes != policy.policy_bytes ||
      !lower_sha256(receipt.payload_sha256) ||
      receipt.payload_bytes != manifest.payload_bytes ||
      receipt.payload_bytes !=
          kPrefillMLPFactorizedLaneR4PublicationPayloadBytes ||
      receipt.projection_count != manifest.projections.size()) {
    return make_diagnostic(
        PrefillMLPFactorizedLaneR4PublicationErrorCode::kInvalidReceipt,
        "r4.receipt",
        "receipt must bind the exact manifest, policy, and payload identity");
  }
  return {};
}

[[nodiscard]] json::ParseOptions receipt_parse_options() {
  json::ParseOptions options;
  options.max_input_bytes = 64U * 1024U;
  options.max_nesting_depth = 6U;
  options.max_values = 128U;
  options.max_container_items = 128U;
  return options;
}

}  // namespace

PrefillMLPFactorizedLaneR4ReceiptResult
build_prefill_mlp_factorized_lane_r4_receipt(
    const PrefillMLPFactorizedLaneR4Manifest& manifest,
    const PrefillMLPFactorizedLaneR4Policy& policy,
    const std::string_view payload_sha256) {
  PrefillMLPFactorizedLaneR4ReceiptResult result;
  result.diagnostic = validate_policy_core(policy, manifest, true);
  if (!result.diagnostic) {
    return result;
  }
  if (!lower_sha256(payload_sha256)) {
    result.diagnostic = make_diagnostic(
        PrefillMLPFactorizedLaneR4PublicationErrorCode::kInvalidReceipt,
        "r4.receipt.payload_sha256",
        "lowercase payload SHA-256 is required");
    return result;
  }
  try {
    PrefillMLPFactorizedLaneR4Receipt receipt;
    receipt.mode =
        std::string(kPrefillMLPFactorizedLaneR4PublicationMode);
    receipt.converter_abi =
        std::string(kPrefillMLPFactorizedLaneR4PublicationAbi);
    receipt.physical_layout = manifest.physical_layout;
    receipt.direct_source = manifest.direct_source;
    receipt.lane_count = kPrefillMLPFactorizedLaneR4PublicationLaneCount;
    receipt.manifest_sha256 = manifest.manifest_sha256;
    receipt.manifest_bytes = manifest.manifest_bytes;
    receipt.policy_sha256 = policy.policy_sha256;
    receipt.policy_bytes = policy.policy_bytes;
    receipt.payload_sha256 = std::string(payload_sha256);
    receipt.payload_bytes = manifest.payload_bytes;
    receipt.projection_count = manifest.projections.size();
    result.diagnostic = validate_receipt_core(receipt, manifest, policy);
    if (!result.diagnostic) {
      return result;
    }
    result.canonical_document = serialize_receipt(receipt);
    result.value.emplace(std::move(receipt));
    return result;
  } catch (const std::bad_alloc&) {
    result.diagnostic = make_diagnostic(
        PrefillMLPFactorizedLaneR4PublicationErrorCode::kAllocationFailure,
        "r4.receipt", "receipt allocation failed");
    return result;
  }
}

PrefillMLPFactorizedLaneR4ReceiptResult
parse_prefill_mlp_factorized_lane_r4_receipt(
    const std::string_view document,
    const PrefillMLPFactorizedLaneR4Manifest& manifest,
    const PrefillMLPFactorizedLaneR4Policy& policy) {
  PrefillMLPFactorizedLaneR4ReceiptResult result;
  result.diagnostic = validate_policy_core(policy, manifest, true);
  if (!result.diagnostic) {
    return result;
  }
  try {
    const json::ParseResult parsed =
        json::parse(document, receipt_parse_options());
    const auto* const root = parsed ? parsed.value->as_object() : nullptr;
    if (root == nullptr ||
        !exact_keys(*root,
                    {"schema", "version", "mode", "converter_abi",
                     "performance_candidate_only",
                     "production_residency_eligible",
                     "quality_production_eligible", "physical_layout",
                     "direct_source", "lane_count", "factor_scheme",
                     "manifest_sha256", "manifest_bytes", "policy_sha256",
                     "policy_bytes", "payload_sha256", "payload_bytes",
                     "projection_count"})) {
      result.diagnostic = make_diagnostic(
          PrefillMLPFactorizedLaneR4PublicationErrorCode::kInvalidReceipt,
          "r4.receipt", "strict receipt JSON schema mismatch");
      return result;
    }
    PrefillMLPFactorizedLaneR4Receipt receipt;
    std::string schema;
    std::string factor_scheme;
    std::uint64_t lane_count = 0U;
    if (!json_string(*root, "schema", schema) || schema != kReceiptSchema ||
        !parse_version(root->at("version"), receipt.version_major,
                       receipt.version_minor) ||
        !json_string(*root, "mode", receipt.mode) ||
        !json_string(*root, "converter_abi", receipt.converter_abi) ||
        !json_bool(*root, "performance_candidate_only",
                   receipt.performance_candidate_only) ||
        !json_bool(*root, "production_residency_eligible",
                   receipt.production_residency_eligible) ||
        !json_bool(*root, "quality_production_eligible",
                   receipt.quality_production_eligible) ||
        !json_string(*root, "physical_layout", receipt.physical_layout) ||
        !parse_direct_source(root->at("direct_source"),
                             receipt.direct_source) ||
        !json_uint(*root, "lane_count", lane_count) ||
        lane_count > std::numeric_limits<std::uint32_t>::max() ||
        !json_string(*root, "factor_scheme", factor_scheme) ||
        !json_string(*root, "manifest_sha256",
                     receipt.manifest_sha256) ||
        !json_uint(*root, "manifest_bytes", receipt.manifest_bytes) ||
        !json_string(*root, "policy_sha256", receipt.policy_sha256) ||
        !json_uint(*root, "policy_bytes", receipt.policy_bytes) ||
        !json_string(*root, "payload_sha256", receipt.payload_sha256) ||
        !json_uint(*root, "payload_bytes", receipt.payload_bytes) ||
        !json_uint(*root, "projection_count",
                   receipt.projection_count)) {
      result.diagnostic = make_diagnostic(
          PrefillMLPFactorizedLaneR4PublicationErrorCode::kInvalidReceipt,
          "r4.receipt", "receipt identity fields are invalid");
      return result;
    }
    receipt.lane_count = static_cast<std::uint32_t>(lane_count);
    if (factor_scheme !=
        kPrefillMLPFactorizedLaneR4PublicationFactorScheme) {
      result.diagnostic = make_diagnostic(
          PrefillMLPFactorizedLaneR4PublicationErrorCode::kInvalidReceipt,
          "r4.receipt.factor_scheme",
          "receipt factor scheme differs from calibrated R4 ABI");
      return result;
    }
    result.diagnostic = validate_receipt_core(receipt, manifest, policy);
    if (!result.diagnostic) {
      return result;
    }
    const std::string canonical = serialize_receipt(receipt);
    if (canonical != document) {
      result.diagnostic = make_diagnostic(
          PrefillMLPFactorizedLaneR4PublicationErrorCode::kInvalidReceipt,
          "r4.receipt", "receipt JSON is not canonical");
      return result;
    }
    result.canonical_document = canonical;
    result.value.emplace(std::move(receipt));
    return result;
  } catch (const std::bad_alloc&) {
    result.diagnostic = make_diagnostic(
        PrefillMLPFactorizedLaneR4PublicationErrorCode::kAllocationFailure,
        "r4.receipt", "receipt parser allocation failed");
    return result;
  } catch (...) {
    result.diagnostic = make_diagnostic(
        PrefillMLPFactorizedLaneR4PublicationErrorCode::kInvalidReceipt,
        "r4.receipt", "unexpected strict receipt parse failure");
    return result;
  }
}

std::string_view to_string(
    const PrefillMLPFactorizedLaneR4PublicationErrorCode code) noexcept {
  switch (code) {
    case PrefillMLPFactorizedLaneR4PublicationErrorCode::kNone:
      return "none";
    case PrefillMLPFactorizedLaneR4PublicationErrorCode::
        kInvalidSourceManifest:
      return "invalid_source_manifest";
    case PrefillMLPFactorizedLaneR4PublicationErrorCode::kInvalidManifest:
      return "invalid_manifest";
    case PrefillMLPFactorizedLaneR4PublicationErrorCode::kInvalidPolicy:
      return "invalid_policy";
    case PrefillMLPFactorizedLaneR4PublicationErrorCode::kInvalidReceipt:
      return "invalid_receipt";
    case PrefillMLPFactorizedLaneR4PublicationErrorCode::
        kSourceBindingMismatch:
      return "source_binding_mismatch";
    case PrefillMLPFactorizedLaneR4PublicationErrorCode::kDigestMismatch:
      return "digest_mismatch";
    case PrefillMLPFactorizedLaneR4PublicationErrorCode::kAllocationFailure:
      return "allocation_failure";
  }
  return "unknown";
}

}  // namespace q3x::runtime
