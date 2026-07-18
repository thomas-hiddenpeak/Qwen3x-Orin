#include "q3x/model/checkpoint_metadata.h"

#include "q3x/core/sha256.h"
#include "q3x/io/safetensors.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <new>
#include <set>
#include <system_error>
#include <utility>

namespace q3x::model::checkpoint {
namespace {

using JsonObject = io::json::Value::Object;

const std::array<KnownCheckpointDescriptor, kKnownCheckpointCount> kCheckpoints = {
    KnownCheckpointDescriptor{
        KnownModel::kQwen36_27B,
        "nvidia-qwen3.6-27b-nvfp4@0893e160",
        "nvidia/Qwen3.6-27B-NVFP4",
        "0893e1606ff3d5f97a441f405d5fc541a6bdf404",
        "c04a19ba293737ad7be4f6e96d6666cb7e479cbe19ecc0c289fad267135b0338",
        "fd7200cd8bca2a8a5d777061521abf83e2deb97ab6bc2f04e7a0a3d3f8ecd5c1",
        "7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2",
        "0.45.0",
        208,
        193,
        2194,
        3,
        21921428072ULL,
        true,
        ""},
    KnownCheckpointDescriptor{
        KnownModel::kQwen36_35BA3B,
        "nvidia-qwen3.6-35b-a3b-nvfp4@491c2f1e",
        "nvidia/Qwen3.6-35B-A3B-NVFP4",
        "491c2f1ea524c639598bf8fa787a93fed5a6fbce",
        "58aefa1c9eff7989f431d748f2ddec39446cb1fd2a69acc46e285c6a37b0cecc",
        "75fe7cc8d5836b58734e05ee67423a4ce91d602aaad45c8173a1b7597cd57663",
        "d67403a4e9793c0ba8a136baf14b3b76ec7b32c822267978084895e07ebd8a3e",
        "0.44.0",
        130,
        161,
        124468,
        3,
        23407580856ULL,
        true,
        "The pinned config.json contains stale embedded ModelOpt 0.37.0 "
        "metadata with no KV-cache scheme and an activation-quantized NVFP4 "
        "group. The pinned external hf_quant_config.json (ModelOpt 0.44.0) "
        "is authoritative; its per-layer table agrees with the embedded table."},
};

Diagnostic make_diagnostic(DiagnosticCode code,
                           std::string source,
                           std::string pointer,
                           std::string message,
                           std::string expected = {},
                           std::string actual = {},
                           Severity severity = Severity::kError) {
  Diagnostic diagnostic;
  diagnostic.code = code;
  diagnostic.severity = severity;
  diagnostic.source = std::move(source);
  diagnostic.json_pointer = std::move(pointer);
  diagnostic.expected = std::move(expected);
  diagnostic.actual = std::move(actual);
  diagnostic.message = std::move(message);
  return diagnostic;
}

template <typename T>
Result<T> failure(Diagnostic diagnostic) {
  Result<T> result;
  result.diagnostics.emplace_back(std::move(diagnostic));
  return result;
}

const io::json::Value* find_value(const JsonObject& object,
                                  std::string_view key) noexcept {
  const auto iterator = object.find(key);
  return iterator == object.end() ? nullptr : &iterator->second;
}

const JsonObject* require_object(const io::json::Value& value,
                                 const std::string& source,
                                 std::string pointer,
                                 std::vector<Diagnostic>& diagnostics) {
  const auto* object = value.as_object();
  if (object == nullptr) {
    diagnostics.emplace_back(make_diagnostic(
        DiagnosticCode::kWrongType,
        source,
        std::move(pointer),
        "expected a JSON object",
        "object",
        std::string(io::json::to_string(value.type()))));
  }
  return object;
}

const io::json::Value* require_field(const JsonObject& object,
                                     std::string_view key,
                                     const std::string& source,
                                     std::string pointer,
                                     std::vector<Diagnostic>& diagnostics) {
  const io::json::Value* value = find_value(object, key);
  if (value == nullptr) {
    diagnostics.emplace_back(make_diagnostic(
        DiagnosticCode::kMissingField,
        source,
        std::move(pointer),
        "required field is missing"));
  }
  return value;
}

bool read_string(const JsonObject& object,
                 std::string_view key,
                 const std::string& source,
                 std::string pointer,
                 std::string& output,
                 std::vector<Diagnostic>& diagnostics) {
  const io::json::Value* value =
      require_field(object, key, source, pointer, diagnostics);
  if (value == nullptr) {
    return false;
  }
  const std::string* string = value->as_string();
  if (string == nullptr) {
    diagnostics.emplace_back(make_diagnostic(
        DiagnosticCode::kWrongType,
        source,
        std::move(pointer),
        "expected a JSON string",
        "string",
        std::string(io::json::to_string(value->type()))));
    return false;
  }
  output = *string;
  return true;
}

bool read_bool(const JsonObject& object,
               std::string_view key,
               const std::string& source,
               std::string pointer,
               bool& output,
               std::vector<Diagnostic>& diagnostics) {
  const io::json::Value* value =
      require_field(object, key, source, pointer, diagnostics);
  if (value == nullptr) {
    return false;
  }
  const bool* boolean = value->as_bool();
  if (boolean == nullptr) {
    diagnostics.emplace_back(make_diagnostic(
        DiagnosticCode::kWrongType,
        source,
        std::move(pointer),
        "expected a JSON boolean",
        "boolean",
        std::string(io::json::to_string(value->type()))));
    return false;
  }
  output = *boolean;
  return true;
}

bool read_uint32(const JsonObject& object,
                 std::string_view key,
                 const std::string& source,
                 std::string pointer,
                 std::uint32_t& output,
                 std::vector<Diagnostic>& diagnostics) {
  const io::json::Value* value =
      require_field(object, key, source, pointer, diagnostics);
  if (value == nullptr) {
    return false;
  }
  const io::json::Number* number = value->as_number();
  std::uint64_t parsed = 0;
  if (number == nullptr || !number->to_uint64(parsed)) {
    diagnostics.emplace_back(make_diagnostic(
        DiagnosticCode::kWrongType,
        source,
        std::move(pointer),
        "expected a non-negative JSON integer",
        "uint32",
        number == nullptr ? std::string(io::json::to_string(value->type()))
                          : number->text()));
    return false;
  }
  if (parsed > std::numeric_limits<std::uint32_t>::max()) {
    diagnostics.emplace_back(make_diagnostic(
        DiagnosticCode::kIntegerOutOfRange,
        source,
        std::move(pointer),
        "integer is outside uint32 range",
        "0..4294967295",
        number->text()));
    return false;
  }
  output = static_cast<std::uint32_t>(parsed);
  return true;
}

bool read_double(const JsonObject& object,
                 std::string_view key,
                 const std::string& source,
                 std::string pointer,
                 double& output,
                 std::vector<Diagnostic>& diagnostics) {
  const io::json::Value* value =
      require_field(object, key, source, pointer, diagnostics);
  if (value == nullptr) {
    return false;
  }
  const io::json::Number* number = value->as_number();
  if (number == nullptr || !number->to_double(output)) {
    diagnostics.emplace_back(make_diagnostic(
        DiagnosticCode::kWrongType,
        source,
        std::move(pointer),
        "expected a finite JSON number",
        "number",
        number == nullptr ? std::string(io::json::to_string(value->type()))
                          : number->text()));
    return false;
  }
  return true;
}

bool check_allowed_keys(const JsonObject& object,
                        const std::set<std::string, std::less<>>& allowed,
                        const std::string& source,
                        const std::string& pointer,
                        std::vector<Diagnostic>& diagnostics) {
  bool valid = true;
  for (const auto& entry : object) {
    if (allowed.find(entry.first) == allowed.end()) {
      diagnostics.emplace_back(make_diagnostic(
          DiagnosticCode::kUnknownField,
          source,
          pointer + "/" + entry.first,
          "unknown field in strict metadata object"));
      valid = false;
    }
  }
  return valid;
}

bool almost_equal(double left, double right) noexcept {
  const double scale = std::max({1.0, std::fabs(left), std::fabs(right)});
  return std::fabs(left - right) <= scale * 1.0e-12;
}

bool matches_catalog_shape(const HfConfigSummary& summary,
                           const ModelConfig& config) noexcept {
  if (summary.architecture != config.hf_architecture ||
      summary.model_type != config.hf_model_type ||
      summary.text_model_type != config.hf_text_model_type ||
      summary.dtype != "bfloat16" || summary.hidden_activation != "silu" ||
      summary.vocab_size != config.vocab_size ||
      summary.hidden_size != config.hidden_size ||
      summary.intermediate_size != config.intermediate_size ||
      summary.num_hidden_layers != config.num_hidden_layers ||
      summary.max_position_embeddings != config.max_position_embeddings ||
      !almost_equal(summary.rms_norm_eps, config.rms_norm_eps) ||
      summary.num_attention_heads != config.num_attention_heads ||
      summary.num_key_value_heads != config.num_key_value_heads ||
      summary.head_dim != config.head_dim ||
      summary.full_attention_interval != config.full_attention_interval ||
      !almost_equal(summary.partial_rotary_factor, 0.25) ||
      !almost_equal(summary.rope_theta, config.rope_theta) ||
      summary.attention_bias ||
      summary.attention_output_gate != config.attention_output_gate ||
      summary.linear_num_key_heads != config.linear_num_key_heads ||
      summary.linear_key_head_dim != config.linear_key_head_dim ||
      summary.linear_num_value_heads != config.linear_num_value_heads ||
      summary.linear_value_head_dim != config.linear_value_head_dim ||
      summary.linear_conv_kernel_dim != config.linear_conv_kernel_dim ||
      summary.num_experts != config.num_experts ||
      summary.num_experts_per_token != config.num_experts_per_token ||
      summary.moe_intermediate_size != config.moe_intermediate_size ||
      summary.shared_expert_intermediate_size !=
          config.shared_expert_intermediate_size ||
      summary.mtp_num_hidden_layers != config.mtp_num_hidden_layers ||
      summary.mtp_uses_dedicated_embeddings !=
          config.mtp_uses_dedicated_embeddings ||
      summary.layer_types.size() != config.num_hidden_layers) {
    return false;
  }

  const VisionConfig& left = summary.vision;
  const VisionConfig& right = config.vision;
  if (left.present != right.present || left.depth != right.depth ||
      left.hidden_size != right.hidden_size ||
      left.intermediate_size != right.intermediate_size ||
      left.num_heads != right.num_heads ||
      left.out_hidden_size != right.out_hidden_size ||
      left.num_position_embeddings != right.num_position_embeddings ||
      left.patch_size != right.patch_size ||
      left.spatial_merge_size != right.spatial_merge_size ||
      left.temporal_patch_size != right.temporal_patch_size) {
    return false;
  }

  for (std::size_t layer = 0; layer < summary.layer_types.size(); ++layer) {
    if (summary.layer_types[layer] != config.layer_type(layer)) {
      return false;
    }
  }
  return true;
}

}  // namespace

const std::array<KnownCheckpointDescriptor, kKnownCheckpointCount>&
known_checkpoint_catalog() {
  return kCheckpoints;
}

const KnownCheckpointDescriptor* find_known_checkpoint_by_config_sha256(
    const std::string_view digest) noexcept {
  for (const KnownCheckpointDescriptor& checkpoint : kCheckpoints) {
    if (checkpoint.config_sha256 == digest) {
      return &checkpoint;
    }
  }
  return nullptr;
}

std::string_view to_string(const InspectionStatus status) noexcept {
  switch (status) {
    case InspectionStatus::kMetadataCompatible:
      return "metadata-compatible";
    case InspectionStatus::kInvalidMetadata:
      return "invalid-metadata";
    case InspectionStatus::kUnknownRevision:
      return "unknown-revision";
    case InspectionStatus::kUnsupportedArchitecture:
      return "unsupported-architecture";
    case InspectionStatus::kUnsupportedQuantization:
      return "unsupported-quantization";
    case InspectionStatus::kMissingRequiredFile:
      return "missing-required-file";
    case InspectionStatus::kIoError:
      return "io-error";
    case InspectionStatus::kMissingShard:
      return "missing-shard";
  }
  return "unknown";
}

std::string_view to_string(const QuantAlgorithm algorithm) noexcept {
  switch (algorithm) {
    case QuantAlgorithm::kFp8:
      return "FP8";
    case QuantAlgorithm::kW4A16Nvfp4:
      return "W4A16_NVFP4";
  }
  return "unknown";
}

std::string_view to_string(const DiagnosticCode code) noexcept {
  switch (code) {
    case DiagnosticCode::kNone:
      return "none";
    case DiagnosticCode::kFileMissing:
      return "file-missing";
    case DiagnosticCode::kFileReadFailed:
      return "file-read-failed";
    case DiagnosticCode::kInputTooLarge:
      return "input-too-large";
    case DiagnosticCode::kInvalidJson:
      return "invalid-json";
    case DiagnosticCode::kRootNotObject:
      return "root-not-object";
    case DiagnosticCode::kMissingField:
      return "missing-field";
    case DiagnosticCode::kWrongType:
      return "wrong-type";
    case DiagnosticCode::kUnknownField:
      return "unknown-field";
    case DiagnosticCode::kInvalidValue:
      return "invalid-value";
    case DiagnosticCode::kIntegerOutOfRange:
      return "integer-out-of-range";
    case DiagnosticCode::kUnsupportedArchitecture:
      return "unsupported-architecture";
    case DiagnosticCode::kUnsupportedQuantization:
      return "unsupported-quantization";
    case DiagnosticCode::kUnknownRevision:
      return "unknown-revision";
    case DiagnosticCode::kHashMismatch:
      return "hash-mismatch";
    case DiagnosticCode::kProducerMismatch:
      return "producer-mismatch";
    case DiagnosticCode::kQuantizationCountMismatch:
      return "quantization-count-mismatch";
    case DiagnosticCode::kMissingQuantizedModule:
      return "missing-quantized-module";
    case DiagnosticCode::kUnexpectedQuantizedModule:
      return "unexpected-quantized-module";
    case DiagnosticCode::kQuantizedModuleMismatch:
      return "quantized-module-mismatch";
    case DiagnosticCode::kInvalidIndex:
      return "invalid-index";
    case DiagnosticCode::kUnsafeShardPath:
      return "unsafe-shard-path";
    case DiagnosticCode::kShardMissing:
      return "shard-missing";
    case DiagnosticCode::kShardNotRegular:
      return "shard-not-regular";
    case DiagnosticCode::kShardHeaderInvalid:
      return "shard-header-invalid";
    case DiagnosticCode::kUnexpectedShardTensor:
      return "unexpected-shard-tensor";
    case DiagnosticCode::kWrongShardTensor:
      return "wrong-shard-tensor";
    case DiagnosticCode::kMissingShardTensor:
      return "missing-shard-tensor";
    case DiagnosticCode::kShardPayloadSizeMismatch:
      return "shard-payload-size-mismatch";
    case DiagnosticCode::kAllocationFailure:
      return "allocation-failure";
  }
  return "unknown";
}

Result<HfConfigSummary> parse_hf_config(const io::json::Value& root,
                                        std::string source) {
  Result<HfConfigSummary> result;
  std::vector<Diagnostic> diagnostics;
  const JsonObject* object =
      require_object(root, source, "", diagnostics);
  if (object == nullptr) {
    result.diagnostics = std::move(diagnostics);
    return result;
  }

  HfConfigSummary summary;
  const io::json::Value* architectures_value = require_field(
      *object, "architectures", source, "/architectures", diagnostics);
  if (architectures_value != nullptr) {
    const auto* architectures = architectures_value->as_array();
    if (architectures == nullptr || architectures->size() != 1U ||
        (*architectures)[0].as_string() == nullptr) {
      diagnostics.emplace_back(make_diagnostic(
          DiagnosticCode::kWrongType,
          source,
          "/architectures",
          "architectures must be an array containing exactly one string",
          "[\"architecture\"]"));
    } else {
      summary.architecture = *(*architectures)[0].as_string();
    }
  }
  read_string(*object,
              "model_type",
              source,
              "/model_type",
              summary.model_type,
              diagnostics);

  const bool dense =
      summary.architecture == "Qwen3_5ForConditionalGeneration";
  const bool moe =
      summary.architecture == "Qwen3_5MoeForConditionalGeneration";
  if (!summary.architecture.empty() && !dense && !moe) {
    diagnostics.emplace_back(make_diagnostic(
        DiagnosticCode::kUnsupportedArchitecture,
        source,
        "/architectures/0",
        "checkpoint architecture is outside the supported catalog",
        "Qwen3_5ForConditionalGeneration or "
        "Qwen3_5MoeForConditionalGeneration",
        summary.architecture));
    result.diagnostics = std::move(diagnostics);
    return result;
  }

  const io::json::Value* text_value =
      require_field(*object, "text_config", source, "/text_config", diagnostics);
  const JsonObject* text = nullptr;
  if (text_value != nullptr) {
    text = require_object(*text_value, source, "/text_config", diagnostics);
  }
  if (text == nullptr) {
    result.diagnostics = std::move(diagnostics);
    return result;
  }

  read_string(*text,
              "model_type",
              source,
              "/text_config/model_type",
              summary.text_model_type,
              diagnostics);
  read_string(*text,
              "dtype",
              source,
              "/text_config/dtype",
              summary.dtype,
              diagnostics);
  read_string(*text,
              "hidden_act",
              source,
              "/text_config/hidden_act",
              summary.hidden_activation,
              diagnostics);
  read_uint32(*text,
              "vocab_size",
              source,
              "/text_config/vocab_size",
              summary.vocab_size,
              diagnostics);
  read_uint32(*text,
              "hidden_size",
              source,
              "/text_config/hidden_size",
              summary.hidden_size,
              diagnostics);
  read_uint32(*text,
              "num_hidden_layers",
              source,
              "/text_config/num_hidden_layers",
              summary.num_hidden_layers,
              diagnostics);
  read_uint32(*text,
              "max_position_embeddings",
              source,
              "/text_config/max_position_embeddings",
              summary.max_position_embeddings,
              diagnostics);
  read_double(*text,
              "rms_norm_eps",
              source,
              "/text_config/rms_norm_eps",
              summary.rms_norm_eps,
              diagnostics);
  read_uint32(*text,
              "num_attention_heads",
              source,
              "/text_config/num_attention_heads",
              summary.num_attention_heads,
              diagnostics);
  read_uint32(*text,
              "num_key_value_heads",
              source,
              "/text_config/num_key_value_heads",
              summary.num_key_value_heads,
              diagnostics);
  read_uint32(*text,
              "head_dim",
              source,
              "/text_config/head_dim",
              summary.head_dim,
              diagnostics);
  read_uint32(*text,
              "full_attention_interval",
              source,
              "/text_config/full_attention_interval",
              summary.full_attention_interval,
              diagnostics);
  read_bool(*text,
            "attention_bias",
            source,
            "/text_config/attention_bias",
            summary.attention_bias,
            diagnostics);
  read_bool(*text,
            "attn_output_gate",
            source,
            "/text_config/attn_output_gate",
            summary.attention_output_gate,
            diagnostics);
  read_uint32(*text,
              "linear_num_key_heads",
              source,
              "/text_config/linear_num_key_heads",
              summary.linear_num_key_heads,
              diagnostics);
  read_uint32(*text,
              "linear_key_head_dim",
              source,
              "/text_config/linear_key_head_dim",
              summary.linear_key_head_dim,
              diagnostics);
  read_uint32(*text,
              "linear_num_value_heads",
              source,
              "/text_config/linear_num_value_heads",
              summary.linear_num_value_heads,
              diagnostics);
  read_uint32(*text,
              "linear_value_head_dim",
              source,
              "/text_config/linear_value_head_dim",
              summary.linear_value_head_dim,
              diagnostics);
  read_uint32(*text,
              "linear_conv_kernel_dim",
              source,
              "/text_config/linear_conv_kernel_dim",
              summary.linear_conv_kernel_dim,
              diagnostics);
  read_uint32(*text,
              "mtp_num_hidden_layers",
              source,
              "/text_config/mtp_num_hidden_layers",
              summary.mtp_num_hidden_layers,
              diagnostics);
  read_bool(*text,
            "mtp_use_dedicated_embeddings",
            source,
            "/text_config/mtp_use_dedicated_embeddings",
            summary.mtp_uses_dedicated_embeddings,
            diagnostics);

  if (dense) {
    read_uint32(*text,
                "intermediate_size",
                source,
                "/text_config/intermediate_size",
                summary.intermediate_size,
                diagnostics);
  } else if (moe) {
    read_uint32(*text,
                "num_experts",
                source,
                "/text_config/num_experts",
                summary.num_experts,
                diagnostics);
    read_uint32(*text,
                "num_experts_per_tok",
                source,
                "/text_config/num_experts_per_tok",
                summary.num_experts_per_token,
                diagnostics);
    read_uint32(*text,
                "moe_intermediate_size",
                source,
                "/text_config/moe_intermediate_size",
                summary.moe_intermediate_size,
                diagnostics);
    read_uint32(*text,
                "shared_expert_intermediate_size",
                source,
                "/text_config/shared_expert_intermediate_size",
                summary.shared_expert_intermediate_size,
                diagnostics);
  }

  const io::json::Value* rope_value = require_field(
      *text, "rope_parameters", source, "/text_config/rope_parameters", diagnostics);
  const JsonObject* rope = nullptr;
  if (rope_value != nullptr) {
    rope = require_object(
        *rope_value, source, "/text_config/rope_parameters", diagnostics);
  }
  if (rope != nullptr) {
    read_double(*rope,
                "partial_rotary_factor",
                source,
                "/text_config/rope_parameters/partial_rotary_factor",
                summary.partial_rotary_factor,
                diagnostics);
    read_double(*rope,
                "rope_theta",
                source,
                "/text_config/rope_parameters/rope_theta",
                summary.rope_theta,
                diagnostics);
    const io::json::Value* text_partial =
        find_value(*text, "partial_rotary_factor");
    if (text_partial != nullptr) {
      double parsed = 0.0;
      const io::json::Number* number = text_partial->as_number();
      if (number == nullptr || !number->to_double(parsed)) {
        diagnostics.emplace_back(make_diagnostic(
            DiagnosticCode::kWrongType,
            source,
            "/text_config/partial_rotary_factor",
            "expected a finite JSON number"));
      } else if (!almost_equal(parsed, summary.partial_rotary_factor)) {
        diagnostics.emplace_back(make_diagnostic(
            DiagnosticCode::kInvalidValue,
            source,
            "/text_config/partial_rotary_factor",
            "text and RoPE partial rotary factors disagree",
            std::to_string(summary.partial_rotary_factor),
            number->text()));
      }
    }
  }

  const io::json::Value* layer_types_value = require_field(
      *text, "layer_types", source, "/text_config/layer_types", diagnostics);
  if (layer_types_value != nullptr) {
    const auto* layers = layer_types_value->as_array();
    if (layers == nullptr) {
      diagnostics.emplace_back(make_diagnostic(
          DiagnosticCode::kWrongType,
          source,
          "/text_config/layer_types",
          "expected an array of layer type strings"));
    } else {
      summary.layer_types.reserve(layers->size());
      for (std::size_t index = 0; index < layers->size(); ++index) {
        const std::string* name = (*layers)[index].as_string();
        if (name == nullptr ||
            (*name != "linear_attention" && *name != "full_attention")) {
          diagnostics.emplace_back(make_diagnostic(
              DiagnosticCode::kInvalidValue,
              source,
              "/text_config/layer_types/" + std::to_string(index),
              "unsupported layer type",
              "linear_attention or full_attention",
              name == nullptr
                  ? std::string(io::json::to_string((*layers)[index].type()))
                  : *name));
          continue;
        }
        summary.layer_types.emplace_back(
            *name == "linear_attention" ? LayerType::kLinearAttention
                                         : LayerType::kFullAttention);
      }
    }
  }

  const io::json::Value* vision_value = require_field(
      *object, "vision_config", source, "/vision_config", diagnostics);
  const JsonObject* vision = nullptr;
  if (vision_value != nullptr) {
    vision = require_object(*vision_value, source, "/vision_config", diagnostics);
  }
  if (vision != nullptr) {
    summary.vision.present = true;
    read_uint32(*vision,
                "depth",
                source,
                "/vision_config/depth",
                summary.vision.depth,
                diagnostics);
    read_uint32(*vision,
                "hidden_size",
                source,
                "/vision_config/hidden_size",
                summary.vision.hidden_size,
                diagnostics);
    read_uint32(*vision,
                "intermediate_size",
                source,
                "/vision_config/intermediate_size",
                summary.vision.intermediate_size,
                diagnostics);
    read_uint32(*vision,
                "num_heads",
                source,
                "/vision_config/num_heads",
                summary.vision.num_heads,
                diagnostics);
    read_uint32(*vision,
                "out_hidden_size",
                source,
                "/vision_config/out_hidden_size",
                summary.vision.out_hidden_size,
                diagnostics);
    read_uint32(*vision,
                "num_position_embeddings",
                source,
                "/vision_config/num_position_embeddings",
                summary.vision.num_position_embeddings,
                diagnostics);
    read_uint32(*vision,
                "patch_size",
                source,
                "/vision_config/patch_size",
                summary.vision.patch_size,
                diagnostics);
    read_uint32(*vision,
                "spatial_merge_size",
                source,
                "/vision_config/spatial_merge_size",
                summary.vision.spatial_merge_size,
                diagnostics);
    read_uint32(*vision,
                "temporal_patch_size",
                source,
                "/vision_config/temporal_patch_size",
                summary.vision.temporal_patch_size,
                diagnostics);
  }

  const io::json::Value* embedded_quantization =
      find_value(*object, "quantization_config");
  if (embedded_quantization != nullptr) {
    const JsonObject* quantization = require_object(
        *embedded_quantization, source, "/quantization_config", diagnostics);
    if (quantization != nullptr) {
      const io::json::Value* method = find_value(*quantization, "quant_method");
      if (method != nullptr) {
        const std::string* name = method->as_string();
        if (name == nullptr) {
          diagnostics.emplace_back(make_diagnostic(
              DiagnosticCode::kWrongType,
              source,
              "/quantization_config/quant_method",
              "quant_method must be a string"));
        } else {
          summary.embedded_quantization_method = *name;
        }
      }
    }
  }

  if (!diagnostics.empty()) {
    result.diagnostics = std::move(diagnostics);
    return result;
  }

  for (const ModelConfig& candidate : known_model_catalog()) {
    if (matches_catalog_shape(summary, candidate)) {
      summary.shape_candidates.emplace_back(candidate.known_model);
    }
  }
  result.value.emplace(std::move(summary));
  return result;
}

Result<ModelOptSummary> parse_modelopt_quant_config(
    const io::json::Value& root, std::string source) {
  Result<ModelOptSummary> result;
  std::vector<Diagnostic> diagnostics;
  const JsonObject* object =
      require_object(root, source, "", diagnostics);
  if (object == nullptr) {
    result.diagnostics = std::move(diagnostics);
    return result;
  }
  check_allowed_keys(*object,
                     {"producer", "quantization"},
                     source,
                     "",
                     diagnostics);

  ModelOptSummary summary;
  const io::json::Value* producer_value =
      require_field(*object, "producer", source, "/producer", diagnostics);
  const JsonObject* producer = nullptr;
  if (producer_value != nullptr) {
    producer =
        require_object(*producer_value, source, "/producer", diagnostics);
  }
  if (producer != nullptr) {
    check_allowed_keys(*producer,
                       {"name", "version"},
                       source,
                       "/producer",
                       diagnostics);
    read_string(*producer,
                "name",
                source,
                "/producer/name",
                summary.producer_name,
                diagnostics);
    read_string(*producer,
                "version",
                source,
                "/producer/version",
                summary.producer_version,
                diagnostics);
    if (!summary.producer_name.empty() &&
        summary.producer_name != "modelopt") {
      diagnostics.emplace_back(make_diagnostic(
          DiagnosticCode::kUnsupportedQuantization,
          source,
          "/producer/name",
          "unsupported quantization producer",
          "modelopt",
          summary.producer_name));
    }
  }

  const io::json::Value* quantization_value = require_field(
      *object, "quantization", source, "/quantization", diagnostics);
  const JsonObject* quantization = nullptr;
  if (quantization_value != nullptr) {
    quantization = require_object(
        *quantization_value, source, "/quantization", diagnostics);
  }
  if (quantization == nullptr) {
    result.diagnostics = std::move(diagnostics);
    return result;
  }
  check_allowed_keys(*quantization,
                     {"quant_algo",
                      "kv_cache_quant_algo",
                      "exclude_modules",
                      "quantized_layers"},
                     source,
                     "/quantization",
                     diagnostics);
  read_string(*quantization,
              "quant_algo",
              source,
              "/quantization/quant_algo",
              summary.quantization_algorithm,
              diagnostics);
  read_string(*quantization,
              "kv_cache_quant_algo",
              source,
              "/quantization/kv_cache_quant_algo",
              summary.kv_cache_quantization_algorithm,
              diagnostics);
  if (!summary.quantization_algorithm.empty() &&
      summary.quantization_algorithm != "MIXED_PRECISION") {
    diagnostics.emplace_back(make_diagnostic(
        DiagnosticCode::kUnsupportedQuantization,
        source,
        "/quantization/quant_algo",
        "only checkpoint-directed mixed precision is supported",
        "MIXED_PRECISION",
        summary.quantization_algorithm));
  }
  if (!summary.kv_cache_quantization_algorithm.empty() &&
      summary.kv_cache_quantization_algorithm != "FP8") {
    diagnostics.emplace_back(make_diagnostic(
        DiagnosticCode::kUnsupportedQuantization,
        source,
        "/quantization/kv_cache_quant_algo",
        "unsupported KV-cache quantization",
        "FP8",
        summary.kv_cache_quantization_algorithm));
  }

  const io::json::Value* exclude_value = require_field(
      *quantization,
      "exclude_modules",
      source,
      "/quantization/exclude_modules",
      diagnostics);
  if (exclude_value != nullptr) {
    const auto* exclude = exclude_value->as_array();
    if (exclude == nullptr) {
      diagnostics.emplace_back(make_diagnostic(
          DiagnosticCode::kWrongType,
          source,
          "/quantization/exclude_modules",
          "exclude_modules must be an array of strings"));
    } else {
      std::set<std::string, std::less<>> seen;
      for (std::size_t index = 0; index < exclude->size(); ++index) {
        const std::string* pattern = (*exclude)[index].as_string();
        const std::string pointer =
            "/quantization/exclude_modules/" + std::to_string(index);
        if (pattern == nullptr || pattern->empty()) {
          diagnostics.emplace_back(make_diagnostic(
              DiagnosticCode::kWrongType,
              source,
              pointer,
              "exclude pattern must be a non-empty string"));
          continue;
        }
        if (!seen.emplace(*pattern).second) {
          diagnostics.emplace_back(make_diagnostic(
              DiagnosticCode::kInvalidValue,
              source,
              pointer,
              "duplicate exclude pattern",
              {},
              *pattern));
          continue;
        }
        summary.exclude_modules.emplace_back(*pattern);
      }
    }
  }

  const io::json::Value* layers_value = require_field(
      *quantization,
      "quantized_layers",
      source,
      "/quantization/quantized_layers",
      diagnostics);
  const JsonObject* layers = nullptr;
  if (layers_value != nullptr) {
    layers = require_object(*layers_value,
                            source,
                            "/quantization/quantized_layers",
                            diagnostics);
  }
  if (layers != nullptr) {
    constexpr std::size_t kMaximumQuantizedModules = 100'000;
    if (layers->size() > kMaximumQuantizedModules) {
      diagnostics.emplace_back(make_diagnostic(
          DiagnosticCode::kInputTooLarge,
          source,
          "/quantization/quantized_layers",
          "too many quantized module records",
          std::to_string(kMaximumQuantizedModules),
          std::to_string(layers->size())));
    } else {
      for (const auto& layer : *layers) {
        const std::string pointer =
            "/quantization/quantized_layers/" + layer.first;
        if (layer.first.empty()) {
          diagnostics.emplace_back(make_diagnostic(
              DiagnosticCode::kInvalidValue,
              source,
              pointer,
              "quantized module path cannot be empty"));
          continue;
        }
        const JsonObject* entry =
            require_object(layer.second, source, pointer, diagnostics);
        if (entry == nullptr) {
          continue;
        }
        const io::json::Value* algorithm_value = require_field(
            *entry, "quant_algo", source, pointer + "/quant_algo", diagnostics);
        if (algorithm_value == nullptr) {
          continue;
        }
        const std::string* algorithm_name = algorithm_value->as_string();
        if (algorithm_name == nullptr) {
          diagnostics.emplace_back(make_diagnostic(
              DiagnosticCode::kWrongType,
              source,
              pointer + "/quant_algo",
              "quant_algo must be a string"));
          continue;
        }

        QuantizedModule module;
        if (*algorithm_name == "FP8") {
          check_allowed_keys(
              *entry, {"quant_algo"}, source, pointer, diagnostics);
          module.algorithm = QuantAlgorithm::kFp8;
          ++summary.fp8_count;
        } else if (*algorithm_name == "W4A16_NVFP4") {
          check_allowed_keys(*entry,
                             {"quant_algo", "group_size"},
                             source,
                             pointer,
                             diagnostics);
          std::uint32_t group_size = 0;
          if (read_uint32(*entry,
                          "group_size",
                          source,
                          pointer + "/group_size",
                          group_size,
                          diagnostics)) {
            if (group_size != 16U) {
              diagnostics.emplace_back(make_diagnostic(
                  DiagnosticCode::kUnsupportedQuantization,
                  source,
                  pointer + "/group_size",
                  "unsupported NVFP4 group size",
                  "16",
                  std::to_string(group_size)));
            }
            module.group_size = group_size;
          }
          module.algorithm = QuantAlgorithm::kW4A16Nvfp4;
          ++summary.nvfp4_count;
        } else {
          diagnostics.emplace_back(make_diagnostic(
              DiagnosticCode::kUnsupportedQuantization,
              source,
              pointer + "/quant_algo",
              "unsupported per-module quantization algorithm",
              "FP8 or W4A16_NVFP4",
              *algorithm_name));
          continue;
        }
        summary.quantized_modules.emplace(layer.first, std::move(module));
      }
    }
  }

  if (!diagnostics.empty()) {
    result.diagnostics = std::move(diagnostics);
    return result;
  }
  result.value.emplace(std::move(summary));
  return result;
}

std::map<std::string, QuantizedModule, std::less<>>
expected_quantized_modules(const KnownModel model) {
  std::map<std::string, QuantizedModule, std::less<>> expected;
  const ModelConfig* config = find_known_model(model);
  if (config == nullptr ||
      (model != KnownModel::kQwen35_27B &&
       model != KnownModel::kQwen35_35BA3B &&
       model != KnownModel::kQwen36_27B &&
       model != KnownModel::kQwen36_35BA3B)) {
    return expected;
  }

  const QuantizedModule fp8{QuantAlgorithm::kFp8, std::nullopt};
  const QuantizedModule nvfp4{QuantAlgorithm::kW4A16Nvfp4, 16U};
  for (std::uint32_t layer = 0; layer < config->num_hidden_layers; ++layer) {
    const std::string prefix =
        "model.language_model.layers." + std::to_string(layer);
    if (config->layer_type(layer) == LayerType::kLinearAttention) {
      expected.emplace(prefix + ".linear_attn.out_proj", fp8);
      expected.emplace(prefix + ".linear_attn.in_proj_qkv", fp8);
      expected.emplace(prefix + ".linear_attn.in_proj_z", fp8);
    } else {
      expected.emplace(prefix + ".self_attn.q_proj", fp8);
      expected.emplace(prefix + ".self_attn.k_proj", fp8);
      expected.emplace(prefix + ".self_attn.v_proj", fp8);
      expected.emplace(prefix + ".self_attn.o_proj", fp8);
    }

    if (config->is_moe()) {
      expected.emplace(prefix + ".mlp.experts", nvfp4);
      expected.emplace(prefix + ".mlp.shared_expert.gate_proj", nvfp4);
      expected.emplace(prefix + ".mlp.shared_expert.up_proj", nvfp4);
      expected.emplace(prefix + ".mlp.shared_expert.down_proj", nvfp4);
    } else {
      expected.emplace(prefix + ".mlp.gate_proj", nvfp4);
      expected.emplace(prefix + ".mlp.up_proj", nvfp4);
      expected.emplace(prefix + ".mlp.down_proj", nvfp4);
    }
  }
  expected.emplace("lm_head", nvfp4);
  return expected;
}

std::vector<Diagnostic> validate_modelopt_quantization(
    const ModelOptSummary& summary,
    const KnownCheckpointDescriptor& checkpoint) {
  std::vector<Diagnostic> diagnostics;
  const std::string source = "hf_quant_config.json";
  if (summary.producer_name != "modelopt" ||
      summary.producer_version != checkpoint.modelopt_version) {
    diagnostics.emplace_back(make_diagnostic(
        DiagnosticCode::kProducerMismatch,
        source,
        "/producer",
        "ModelOpt producer does not match the pinned descriptor",
        "modelopt " + checkpoint.modelopt_version,
        summary.producer_name + " " + summary.producer_version));
  }
  if (summary.quantization_algorithm != "MIXED_PRECISION" ||
      summary.kv_cache_quantization_algorithm != "FP8") {
    diagnostics.emplace_back(make_diagnostic(
        DiagnosticCode::kUnsupportedQuantization,
        source,
        "/quantization",
        "pinned checkpoint requires MIXED_PRECISION with FP8 KV cache"));
  }

  std::set<std::string, std::less<>> excludes(summary.exclude_modules.begin(),
                                               summary.exclude_modules.end());
  const std::set<std::string, std::less<>> expected_excludes = {
      "mtp*", "mtp.layers.0*"};
  if (excludes != expected_excludes ||
      excludes.size() != summary.exclude_modules.size()) {
    diagnostics.emplace_back(make_diagnostic(
        DiagnosticCode::kInvalidValue,
        source,
        "/quantization/exclude_modules",
        "exclude_modules must contain exactly the two pinned MTP patterns",
        "mtp*, mtp.layers.0*"));
  }

  if (summary.fp8_count != checkpoint.expected_fp8_modules ||
      summary.nvfp4_count != checkpoint.expected_nvfp4_modules) {
    diagnostics.emplace_back(make_diagnostic(
        DiagnosticCode::kQuantizationCountMismatch,
        source,
        "/quantization/quantized_layers",
        "per-algorithm module counts do not match the pinned descriptor",
        "FP8=" + std::to_string(checkpoint.expected_fp8_modules) +
            ", W4A16_NVFP4=" +
            std::to_string(checkpoint.expected_nvfp4_modules),
        "FP8=" + std::to_string(summary.fp8_count) +
            ", W4A16_NVFP4=" + std::to_string(summary.nvfp4_count)));
  }

  const auto expected = expected_quantized_modules(checkpoint.model);
  for (const auto& item : expected) {
    const auto actual = summary.quantized_modules.find(item.first);
    if (actual == summary.quantized_modules.end()) {
      diagnostics.emplace_back(make_diagnostic(
          DiagnosticCode::kMissingQuantizedModule,
          source,
          "/quantization/quantized_layers/" + item.first,
          "required quantized module is missing",
          std::string(to_string(item.second.algorithm))));
      continue;
    }
    if (actual->second.algorithm != item.second.algorithm ||
        actual->second.group_size != item.second.group_size) {
      diagnostics.emplace_back(make_diagnostic(
          DiagnosticCode::kQuantizedModuleMismatch,
          source,
          "/quantization/quantized_layers/" + item.first,
          "quantized module algorithm or group size is incorrect",
          std::string(to_string(item.second.algorithm)),
          std::string(to_string(actual->second.algorithm))));
    }
  }
  for (const auto& item : summary.quantized_modules) {
    if (expected.find(item.first) == expected.end()) {
      diagnostics.emplace_back(make_diagnostic(
          DiagnosticCode::kUnexpectedQuantizedModule,
          source,
          "/quantization/quantized_layers/" + item.first,
          "module is not in the pinned architecture quantization set",
          {},
          std::string(to_string(item.second.algorithm))));
    }
  }
  return diagnostics;
}

bool is_safe_relative_shard_path(const std::string_view path) noexcept {
  return io::safetensors::is_safe_relative_shard_path(path);
}

namespace {

struct LoadedJsonFile {
  io::json::Value root;
  FileEvidence evidence;
};

struct LoadedJsonResult {
  std::optional<LoadedJsonFile> value;
  Diagnostic diagnostic;
};

LoadedJsonResult load_json_file(const std::filesystem::path& path,
                                const std::uint64_t maximum_bytes,
                                std::string logical_name) {
  LoadedJsonResult result;
  std::error_code error;
  const bool exists = std::filesystem::exists(path, error);
  if (error || !exists) {
    result.diagnostic = make_diagnostic(
        DiagnosticCode::kFileMissing,
        std::move(logical_name),
        "",
        "required metadata file is missing",
        path.string());
    return result;
  }
  if (!std::filesystem::is_regular_file(path, error) || error) {
    result.diagnostic = make_diagnostic(
        DiagnosticCode::kFileReadFailed,
        std::move(logical_name),
        "",
        "metadata path is not a readable regular file",
        {},
        path.string());
    return result;
  }
  const std::uintmax_t file_size = std::filesystem::file_size(path, error);
  if (error) {
    result.diagnostic = make_diagnostic(
        DiagnosticCode::kFileReadFailed,
        std::move(logical_name),
        "",
        "could not determine metadata file size",
        {},
        error.message());
    return result;
  }
  if (file_size > maximum_bytes ||
      file_size > std::numeric_limits<std::size_t>::max() ||
      file_size > static_cast<std::uintmax_t>(
                      std::numeric_limits<std::streamsize>::max())) {
    result.diagnostic = make_diagnostic(
        DiagnosticCode::kInputTooLarge,
        std::move(logical_name),
        "",
        "metadata file exceeds the configured byte limit",
        std::to_string(maximum_bytes),
        std::to_string(file_size));
    return result;
  }

  try {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
      result.diagnostic = make_diagnostic(
          DiagnosticCode::kFileReadFailed,
          std::move(logical_name),
          "",
          "could not open metadata file",
          {},
          path.string());
      return result;
    }
    std::string bytes(static_cast<std::size_t>(file_size), '\0');
    if (!bytes.empty()) {
      input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }
    if (!input || input.peek() != std::char_traits<char>::eof()) {
      result.diagnostic = make_diagnostic(
          DiagnosticCode::kFileReadFailed,
          std::move(logical_name),
          "",
          "metadata file changed or could not be read completely",
          {},
          path.string());
      return result;
    }

    io::json::ParseOptions parse_options;
    parse_options.max_input_bytes = static_cast<std::size_t>(maximum_bytes);
    parse_options.max_nesting_depth = 128;
    io::json::ParseResult parsed = io::json::parse(bytes, parse_options);
    if (!parsed) {
      result.diagnostic = make_diagnostic(
          DiagnosticCode::kInvalidJson,
          logical_name,
          "",
          "invalid JSON at byte " + std::to_string(parsed.error.offset) +
              ": " + std::string(parsed.error.message()),
          {},
          std::string(io::json::to_string(parsed.error.code)));
      return result;
    }

    LoadedJsonFile loaded;
    loaded.root = std::move(*parsed.value);
    loaded.evidence.name = std::move(logical_name);
    loaded.evidence.size = static_cast<std::uint64_t>(file_size);
    loaded.evidence.sha256 = core::sha256(bytes).hex();
    result.value.emplace(std::move(loaded));
    return result;
  } catch (const std::bad_alloc&) {
    result.diagnostic = make_diagnostic(
        DiagnosticCode::kAllocationFailure,
        std::move(logical_name),
        "",
        "allocation failed while reading metadata file");
    return result;
  }
}

struct ParsedIndex {
  IndexSummary summary;
  io::safetensors::Index contract;
};

Result<ParsedIndex> parse_index(const io::json::Value& root,
                                const InspectionOptions& options,
                                const std::string& source) {
  Result<ParsedIndex> result;
  std::vector<Diagnostic> diagnostics;
  const JsonObject* object =
      require_object(root, source, "", diagnostics);
  if (object == nullptr) {
    result.diagnostics = std::move(diagnostics);
    return result;
  }
  check_allowed_keys(
      *object, {"metadata", "weight_map"}, source, "", diagnostics);

  ParsedIndex parsed;
  const io::json::Value* weight_map_value = require_field(
      *object, "weight_map", source, "/weight_map", diagnostics);
  const JsonObject* weight_map = nullptr;
  if (weight_map_value != nullptr) {
    weight_map =
        require_object(*weight_map_value, source, "/weight_map", diagnostics);
  }
  if (weight_map != nullptr) {
    if (weight_map->size() > options.max_index_tensors) {
      diagnostics.emplace_back(make_diagnostic(
          DiagnosticCode::kInputTooLarge,
          source,
          "/weight_map",
          "index contains too many tensor records",
          std::to_string(options.max_index_tensors),
          std::to_string(weight_map->size())));
    } else {
      parsed.summary.tensor_count = weight_map->size();
      std::set<std::string, std::less<>> shards;
      for (const auto& item : *weight_map) {
        const std::string* shard = item.second.as_string();
        if (item.first.empty() || shard == nullptr || shard->empty()) {
          diagnostics.emplace_back(make_diagnostic(
              DiagnosticCode::kInvalidIndex,
              source,
              "/weight_map/" + item.first,
              "tensor and shard names must be non-empty strings"));
          continue;
        }
        if (!is_safe_relative_shard_path(*shard)) {
          diagnostics.emplace_back(make_diagnostic(
              DiagnosticCode::kUnsafeShardPath,
              source,
              "/weight_map/" + item.first,
              "unsafe shard path in index",
              "relative *.safetensors path without dot components",
              *shard));
          continue;
        }
        shards.emplace(*shard);
        if (options.require_shards) {
          parsed.contract.weight_map.emplace(item.first, *shard);
        }
      }
      if (shards.size() > options.max_index_shards) {
        diagnostics.emplace_back(make_diagnostic(
            DiagnosticCode::kInputTooLarge,
            source,
            "/weight_map",
            "index references too many shards",
            std::to_string(options.max_index_shards),
            std::to_string(shards.size())));
      } else {
        parsed.summary.shards.assign(shards.begin(), shards.end());
        if (options.require_shards) {
          parsed.contract.shards = parsed.summary.shards;
        }
      }
    }
  }

  const io::json::Value* metadata_value = find_value(*object, "metadata");
  if (metadata_value != nullptr) {
    const JsonObject* metadata =
        require_object(*metadata_value, source, "/metadata", diagnostics);
    if (metadata != nullptr) {
      const io::json::Value* total_size = find_value(*metadata, "total_size");
      if (total_size != nullptr) {
        const io::json::Number* number = total_size->as_number();
        std::uint64_t parsed_total_size = 0;
        if (number == nullptr || !number->to_uint64(parsed_total_size)) {
          diagnostics.emplace_back(make_diagnostic(
              DiagnosticCode::kInvalidIndex,
              source,
              "/metadata/total_size",
              "total_size must be a non-negative uint64 integer"));
        } else {
          parsed.summary.total_size = parsed_total_size;
          if (options.require_shards) {
            parsed.contract.total_size = parsed_total_size;
          }
        }
      }
    }
  }

  if (!diagnostics.empty()) {
    result.diagnostics = std::move(diagnostics);
    return result;
  }
  result.value.emplace(std::move(parsed));
  return result;
}

InspectionResult inspection_failure(InspectionStatus status,
                                    std::vector<Diagnostic> diagnostics) {
  InspectionResult result;
  result.status = status;
  result.diagnostics = std::move(diagnostics);
  return result;
}

InspectionResult inspection_failure(InspectionStatus status,
                                    Diagnostic diagnostic) {
  std::vector<Diagnostic> diagnostics;
  diagnostics.emplace_back(std::move(diagnostic));
  return inspection_failure(status, std::move(diagnostics));
}

bool has_code(const std::vector<Diagnostic>& diagnostics,
              const DiagnosticCode code) noexcept {
  return std::any_of(diagnostics.begin(), diagnostics.end(),
                     [code](const Diagnostic& diagnostic) {
                       return diagnostic.code == code;
                     });
}

InspectionStatus status_for_parse_failure(
    const std::vector<Diagnostic>& diagnostics) noexcept {
  if (has_code(diagnostics, DiagnosticCode::kUnsupportedArchitecture)) {
    return InspectionStatus::kUnsupportedArchitecture;
  }
  if (has_code(diagnostics, DiagnosticCode::kUnsupportedQuantization)) {
    return InspectionStatus::kUnsupportedQuantization;
  }
  return InspectionStatus::kInvalidMetadata;
}

InspectionStatus status_for_shard_error(
    const io::safetensors::ErrorCode code) noexcept {
  switch (code) {
    case io::safetensors::ErrorCode::kShardMissing:
    case io::safetensors::ErrorCode::kShardNotRegular:
      return InspectionStatus::kMissingShard;
    case io::safetensors::ErrorCode::kOpenFailed:
    case io::safetensors::ErrorCode::kIoFailure:
    case io::safetensors::ErrorCode::kAllocationFailure:
      return InspectionStatus::kIoError;
    default:
      return InspectionStatus::kInvalidMetadata;
  }
}

Diagnostic shard_validation_diagnostic(
    const io::safetensors::Error& error) {
  DiagnosticCode code = DiagnosticCode::kShardHeaderInvalid;
  std::string source = error.context.empty()
                           ? "model.safetensors.index.json"
                           : error.context;
  std::string pointer = "/weight_map";

  switch (error.code) {
    case io::safetensors::ErrorCode::kUnsafeShardPath:
      code = DiagnosticCode::kUnsafeShardPath;
      source = "model.safetensors.index.json";
      break;
    case io::safetensors::ErrorCode::kShardMissing:
      code = DiagnosticCode::kShardMissing;
      break;
    case io::safetensors::ErrorCode::kShardNotRegular:
      code = DiagnosticCode::kShardNotRegular;
      break;
    case io::safetensors::ErrorCode::kUnexpectedTensor:
      code = DiagnosticCode::kUnexpectedShardTensor;
      source = "model.safetensors.index.json";
      pointer += "/" + error.context;
      break;
    case io::safetensors::ErrorCode::kTensorInWrongShard:
      code = DiagnosticCode::kWrongShardTensor;
      source = "model.safetensors.index.json";
      pointer += "/" + error.context;
      break;
    case io::safetensors::ErrorCode::kMissingIndexedTensor:
      code = DiagnosticCode::kMissingShardTensor;
      source = "model.safetensors.index.json";
      pointer += "/" + error.context;
      break;
    case io::safetensors::ErrorCode::kMissingTotalSize:
    case io::safetensors::ErrorCode::kPayloadSizeMismatch:
      code = DiagnosticCode::kShardPayloadSizeMismatch;
      source = "model.safetensors.index.json";
      pointer = "/metadata/total_size";
      break;
    case io::safetensors::ErrorCode::kInvalidOption:
    case io::safetensors::ErrorCode::kTooManyTensors:
    case io::safetensors::ErrorCode::kInvalidWeightMap:
    case io::safetensors::ErrorCode::kTooManyShards:
    case io::safetensors::ErrorCode::kShardSetMismatch:
    case io::safetensors::ErrorCode::kArithmeticOverflow:
      code = DiagnosticCode::kInvalidIndex;
      source = "model.safetensors.index.json";
      break;
    case io::safetensors::ErrorCode::kOpenFailed:
    case io::safetensors::ErrorCode::kIoFailure:
      code = DiagnosticCode::kFileReadFailed;
      break;
    case io::safetensors::ErrorCode::kAllocationFailure:
      code = DiagnosticCode::kAllocationFailure;
      break;
    default:
      break;
  }

  return make_diagnostic(code,
                         std::move(source),
                         std::move(pointer),
                         std::string(error.message()),
                         error.expected,
                         error.actual);
}

}  // namespace

InspectionResult inspect_directory(const std::filesystem::path& directory,
                                   const InspectionOptions& options) {
  if (options.max_config_bytes == 0U ||
      options.max_quant_config_bytes == 0U ||
      options.max_index_bytes == 0U || options.max_index_tensors == 0U ||
      options.max_index_shards == 0U) {
    return inspection_failure(
        InspectionStatus::kInvalidMetadata,
        make_diagnostic(DiagnosticCode::kInvalidValue,
                        "options",
                        "",
                        "inspection limits must be non-zero"));
  }

  const auto config_file = load_json_file(directory / "config.json",
                                          options.max_config_bytes,
                                          "config.json");
  if (!config_file.value) {
    const InspectionStatus status =
        config_file.diagnostic.code == DiagnosticCode::kFileMissing
            ? InspectionStatus::kMissingRequiredFile
            : InspectionStatus::kIoError;
    return inspection_failure(status, config_file.diagnostic);
  }

  Result<HfConfigSummary> config =
      parse_hf_config(config_file.value->root, "config.json");
  if (!config) {
    return inspection_failure(status_for_parse_failure(config.diagnostics),
                              std::move(config.diagnostics));
  }
  if (config.value->embedded_quantization_method &&
      *config.value->embedded_quantization_method != "modelopt") {
    return inspection_failure(
        InspectionStatus::kUnsupportedQuantization,
        make_diagnostic(
            DiagnosticCode::kUnsupportedQuantization,
            "config.json",
            "/quantization_config/quant_method",
            "checkpoint uses an unsupported quantization method",
            "modelopt",
            *config.value->embedded_quantization_method));
  }
  if (config.value->shape_candidates.empty()) {
    return inspection_failure(
        InspectionStatus::kUnsupportedArchitecture,
        make_diagnostic(
            DiagnosticCode::kUnsupportedArchitecture,
            "config.json",
            "",
            "configuration identifiers or dimensions do not match a supported "
            "Qwen3.5/Qwen3.6 catalog shape"));
  }

  const KnownCheckpointDescriptor* checkpoint =
      find_known_checkpoint_by_config_sha256(
          config_file.value->evidence.sha256);
  if (checkpoint == nullptr) {
    return inspection_failure(
        InspectionStatus::kUnknownRevision,
        make_diagnostic(
            DiagnosticCode::kUnknownRevision,
            "config.json",
            "",
            "configuration shape is known but its exact revision is not pinned",
            "one of the built-in NVIDIA config SHA-256 digests",
            config_file.value->evidence.sha256));
  }
  if (std::find(config.value->shape_candidates.begin(),
                config.value->shape_candidates.end(),
                checkpoint->model) == config.value->shape_candidates.end()) {
    return inspection_failure(
        InspectionStatus::kInvalidMetadata,
        make_diagnostic(
            DiagnosticCode::kInvalidValue,
            "config.json",
            "",
            "pinned config hash does not match the parsed architecture shape"));
  }

  const auto quant_file = load_json_file(directory / "hf_quant_config.json",
                                         options.max_quant_config_bytes,
                                         "hf_quant_config.json");
  if (!quant_file.value) {
    const InspectionStatus status =
        quant_file.diagnostic.code == DiagnosticCode::kFileMissing
            ? InspectionStatus::kMissingRequiredFile
            : InspectionStatus::kIoError;
    return inspection_failure(status, quant_file.diagnostic);
  }
  Result<ModelOptSummary> quantization = parse_modelopt_quant_config(
      quant_file.value->root, "hf_quant_config.json");
  if (!quantization) {
    return inspection_failure(
        status_for_parse_failure(quantization.diagnostics),
        std::move(quantization.diagnostics));
  }
  std::vector<Diagnostic> quantization_diagnostics =
      validate_modelopt_quantization(*quantization.value, *checkpoint);
  if (!quantization_diagnostics.empty()) {
    return inspection_failure(InspectionStatus::kInvalidMetadata,
                              std::move(quantization_diagnostics));
  }
  if (quant_file.value->evidence.sha256 != checkpoint->quant_config_sha256) {
    return inspection_failure(
        InspectionStatus::kUnknownRevision,
        make_diagnostic(
            DiagnosticCode::kHashMismatch,
            "hf_quant_config.json",
            "",
            "quantization metadata is valid but not the pinned artifact",
            checkpoint->quant_config_sha256,
            quant_file.value->evidence.sha256));
  }

  const auto index_file = load_json_file(
      directory / "model.safetensors.index.json",
      options.max_index_bytes,
      "model.safetensors.index.json");
  if (!index_file.value) {
    const InspectionStatus status =
        index_file.diagnostic.code == DiagnosticCode::kFileMissing
            ? InspectionStatus::kMissingRequiredFile
            : InspectionStatus::kIoError;
    return inspection_failure(status, index_file.diagnostic);
  }
  Result<ParsedIndex> index = parse_index(index_file.value->root,
                                          options,
                                          "model.safetensors.index.json");
  if (!index) {
    return inspection_failure(InspectionStatus::kInvalidMetadata,
                              std::move(index.diagnostics));
  }
  if (index_file.value->evidence.sha256 != checkpoint->index_sha256) {
    return inspection_failure(
        InspectionStatus::kUnknownRevision,
        make_diagnostic(DiagnosticCode::kHashMismatch,
                        "model.safetensors.index.json",
                        "",
                        "safetensors index is not the pinned artifact",
                        checkpoint->index_sha256,
                        index_file.value->evidence.sha256));
  }
  if (index.value->summary.tensor_count !=
          checkpoint->expected_index_tensors ||
      index.value->summary.shards.size() !=
          checkpoint->expected_index_shards ||
      index.value->summary.total_size !=
          checkpoint->expected_index_total_size) {
    return inspection_failure(
        InspectionStatus::kInvalidMetadata,
        make_diagnostic(
            DiagnosticCode::kInvalidIndex,
            "model.safetensors.index.json",
            "",
            "index statistics do not match the pinned descriptor"));
  }

  InspectionReport report;
  report.status = InspectionStatus::kMetadataCompatible;
  report.checkpoint = *checkpoint;
  report.config = std::move(*config.value);
  report.quantization = std::move(*quantization.value);
  report.index = std::move(index.value->summary);
  report.files.emplace_back(config_file.value->evidence);
  report.files.emplace_back(quant_file.value->evidence);
  report.files.emplace_back(index_file.value->evidence);

  if (options.require_shards) {
    io::safetensors::ShardValidationOptions validation_options;
    validation_options.max_shards = options.max_index_shards;
    validation_options.max_tensors = options.max_index_tensors;
    validation_options.header_options.max_tensors =
        options.max_index_tensors;
    const io::safetensors::Result<io::safetensors::ShardValidationSummary>
        validated = io::safetensors::validate_index_shards(
            directory, index.value->contract, validation_options);
    if (!validated) {
      return inspection_failure(status_for_shard_error(validated.error.code),
                                shard_validation_diagnostic(validated.error));
    }

    report.shard_contract_validated = true;
    report.validated_shards = validated.value->validated_shards;
    report.validated_tensors = validated.value->validated_tensors;
    report.validated_payload_bytes = validated.value->payload_bytes;
    report.present_shards = validated.value->validated_shards;
  } else {
    for (const std::string& shard : report.index.shards) {
      const std::filesystem::path shard_path = directory / shard;
      std::error_code error;
      const bool exists = std::filesystem::exists(shard_path, error);
      if (error || !exists) {
        ++report.missing_shards;
        report.diagnostics.emplace_back(make_diagnostic(
            DiagnosticCode::kShardMissing,
            "model.safetensors.index.json",
            "/weight_map",
            "referenced shard is not present; metadata-only inspection "
            "remains valid",
            {},
            shard,
            Severity::kWarning));
        continue;
      }
      if (!std::filesystem::is_regular_file(shard_path, error) || error) {
        ++report.missing_shards;
        report.diagnostics.emplace_back(make_diagnostic(
            DiagnosticCode::kShardNotRegular,
            "model.safetensors.index.json",
            "/weight_map",
            "referenced shard is not a regular file",
            {},
            shard,
            Severity::kWarning));
        continue;
      }
      ++report.present_shards;
    }
  }

  InspectionResult result;
  result.status = InspectionStatus::kMetadataCompatible;
  result.diagnostics = report.diagnostics;
  result.report.emplace(std::move(report));
  return result;
}

}  // namespace q3x::model::checkpoint
