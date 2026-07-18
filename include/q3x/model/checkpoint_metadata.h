#pragma once

#include "q3x/io/json.h"
#include "q3x/model/model_config.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace q3x::model::checkpoint {

enum class InspectionStatus : std::uint8_t {
  kMetadataCompatible,
  kInvalidMetadata,
  kUnknownRevision,
  kUnsupportedArchitecture,
  kUnsupportedQuantization,
  kMissingRequiredFile,
  kIoError,
  kMissingShard,
};

enum class Severity : std::uint8_t {
  kWarning,
  kError,
};

enum class DiagnosticCode : std::uint8_t {
  kNone,
  kFileMissing,
  kFileReadFailed,
  kInputTooLarge,
  kInvalidJson,
  kRootNotObject,
  kMissingField,
  kWrongType,
  kUnknownField,
  kInvalidValue,
  kIntegerOutOfRange,
  kUnsupportedArchitecture,
  kUnsupportedQuantization,
  kUnknownRevision,
  kHashMismatch,
  kProducerMismatch,
  kQuantizationCountMismatch,
  kMissingQuantizedModule,
  kUnexpectedQuantizedModule,
  kQuantizedModuleMismatch,
  kInvalidIndex,
  kUnsafeShardPath,
  kShardMissing,
  kShardNotRegular,
  kShardHeaderInvalid,
  kUnexpectedShardTensor,
  kWrongShardTensor,
  kMissingShardTensor,
  kShardPayloadSizeMismatch,
  kAllocationFailure,
};

struct Diagnostic {
  DiagnosticCode code = DiagnosticCode::kNone;
  Severity severity = Severity::kError;
  std::string source;
  std::string json_pointer;
  std::string expected;
  std::string actual;
  std::string message;
};

template <typename T>
struct Result {
  std::optional<T> value;
  std::vector<Diagnostic> diagnostics;

  [[nodiscard]] bool ok() const noexcept { return value.has_value(); }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

struct HfConfigSummary {
  std::string architecture;
  std::string model_type;
  std::string text_model_type;
  std::string dtype;
  std::string hidden_activation;

  std::uint32_t vocab_size = 0;
  std::uint32_t hidden_size = 0;
  std::uint32_t intermediate_size = 0;
  std::uint32_t num_hidden_layers = 0;
  std::uint32_t max_position_embeddings = 0;
  double rms_norm_eps = 0.0;

  std::uint32_t num_attention_heads = 0;
  std::uint32_t num_key_value_heads = 0;
  std::uint32_t head_dim = 0;
  std::uint32_t full_attention_interval = 0;
  double partial_rotary_factor = 0.0;
  double rope_theta = 0.0;
  bool attention_bias = false;
  bool attention_output_gate = false;

  std::uint32_t linear_num_key_heads = 0;
  std::uint32_t linear_key_head_dim = 0;
  std::uint32_t linear_num_value_heads = 0;
  std::uint32_t linear_value_head_dim = 0;
  std::uint32_t linear_conv_kernel_dim = 0;

  std::uint32_t num_experts = 0;
  std::uint32_t num_experts_per_token = 0;
  std::uint32_t moe_intermediate_size = 0;
  std::uint32_t shared_expert_intermediate_size = 0;

  std::uint32_t mtp_num_hidden_layers = 0;
  bool mtp_uses_dedicated_embeddings = false;
  VisionConfig vision;
  std::vector<LayerType> layer_types;

  std::optional<std::string> embedded_quantization_method;
  std::vector<KnownModel> shape_candidates;
};

enum class QuantAlgorithm : std::uint8_t {
  kFp8,
  kW4A16Nvfp4,
};

struct QuantizedModule {
  QuantAlgorithm algorithm = QuantAlgorithm::kFp8;
  std::optional<std::uint32_t> group_size;
};

struct ModelOptSummary {
  std::string producer_name;
  std::string producer_version;
  std::string quantization_algorithm;
  std::string kv_cache_quantization_algorithm;
  std::vector<std::string> exclude_modules;
  std::map<std::string, QuantizedModule, std::less<>> quantized_modules;
  std::size_t fp8_count = 0;
  std::size_t nvfp4_count = 0;
};

struct KnownCheckpointDescriptor {
  KnownModel model = KnownModel::kQwen36_27B;
  std::string id;
  std::string repository;
  std::string revision;
  std::string config_sha256;
  std::string quant_config_sha256;
  std::string index_sha256;
  std::string modelopt_version;
  std::size_t expected_fp8_modules = 0;
  std::size_t expected_nvfp4_modules = 0;
  std::size_t expected_index_tensors = 0;
  std::size_t expected_index_shards = 0;
  std::optional<std::uint64_t> expected_index_total_size;
  bool external_quantization_authoritative = true;
  std::string known_metadata_quirk;
};

inline constexpr std::size_t kKnownCheckpointCount = 2;

struct FileEvidence {
  std::string name;
  std::uint64_t size = 0;
  std::string sha256;
};

struct IndexSummary {
  std::size_t tensor_count = 0;
  std::vector<std::string> shards;
  std::optional<std::uint64_t> total_size;
};

struct InspectionOptions {
  bool require_shards = false;
  std::uint64_t max_config_bytes = 1024U * 1024U;
  std::uint64_t max_quant_config_bytes = 4U * 1024U * 1024U;
  std::uint64_t max_index_bytes = 64U * 1024U * 1024U;
  std::size_t max_index_tensors = 2'000'000;
  std::size_t max_index_shards = 100'000;
};

struct InspectionReport {
  InspectionStatus status = InspectionStatus::kInvalidMetadata;
  KnownCheckpointDescriptor checkpoint;
  HfConfigSummary config;
  ModelOptSummary quantization;
  IndexSummary index;
  std::vector<FileEvidence> files;
  std::size_t present_shards = 0;
  std::size_t missing_shards = 0;
  bool shard_contract_validated = false;
  std::size_t validated_shards = 0;
  std::size_t validated_tensors = 0;
  std::uint64_t validated_payload_bytes = 0;
  std::vector<Diagnostic> diagnostics;
};

struct InspectionResult {
  InspectionStatus status = InspectionStatus::kInvalidMetadata;
  std::optional<InspectionReport> report;
  std::vector<Diagnostic> diagnostics;

  [[nodiscard]] bool ok() const noexcept {
    return status == InspectionStatus::kMetadataCompatible &&
           report.has_value();
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

[[nodiscard]] Result<HfConfigSummary> parse_hf_config(
    const io::json::Value& root,
    std::string source = "config.json");

[[nodiscard]] Result<ModelOptSummary> parse_modelopt_quant_config(
    const io::json::Value& root,
    std::string source = "hf_quant_config.json");

[[nodiscard]] std::map<std::string, QuantizedModule, std::less<>>
expected_quantized_modules(KnownModel model);

[[nodiscard]] std::vector<Diagnostic> validate_modelopt_quantization(
    const ModelOptSummary& summary,
    const KnownCheckpointDescriptor& checkpoint);

[[nodiscard]] const std::array<KnownCheckpointDescriptor,
                               kKnownCheckpointCount>&
known_checkpoint_catalog();

[[nodiscard]] const KnownCheckpointDescriptor*
find_known_checkpoint_by_config_sha256(std::string_view digest) noexcept;

[[nodiscard]] bool is_safe_relative_shard_path(
    std::string_view path) noexcept;

[[nodiscard]] InspectionResult inspect_directory(
    const std::filesystem::path& directory,
    const InspectionOptions& options = {});

[[nodiscard]] std::string_view to_string(InspectionStatus status) noexcept;
[[nodiscard]] std::string_view to_string(DiagnosticCode code) noexcept;
[[nodiscard]] std::string_view to_string(QuantAlgorithm algorithm) noexcept;

}  // namespace q3x::model::checkpoint
