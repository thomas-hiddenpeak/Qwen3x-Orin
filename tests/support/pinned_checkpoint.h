#pragma once

#include "q3x/io/safetensors.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace q3x::test::support {

// A data-driven, test-only contract for loading a small set of payloads from
// a pinned checkpoint revision.  The generic loader deliberately lives under
// tests/support: production checkpoint ownership remains with ModelWeights.
struct PinnedFile {
  std::string relative_path;
  std::string sha256;
  // Zero means the SHA pin is authoritative and the byte size is recorded but
  // not independently pinned.
  std::uint64_t file_bytes = 0U;
};

struct PinnedShard {
  std::string relative_path;
  std::uint64_t file_bytes = 0U;
  std::uint64_t header_bytes = 0U;
  std::uint64_t data_offset = 0U;
  std::string sha256;
};

struct PinnedTensor {
  std::string name;
  std::string shard;
  io::safetensors::DType dtype = io::safetensors::DType::kBool;
  std::vector<std::uint64_t> shape;
  std::uint64_t file_begin = 0U;
  std::uint64_t file_end = 0U;
  std::string sha256;
  // NVFP4 weight_scale_2 pins use both the payload SHA and exact little-endian
  // F32 bits.  Other tensor kinds leave this empty.
  std::optional<std::uint32_t> exact_f32_bits;
  bool require_finite_positive_f32 = false;
};

struct PinnedModelRevision {
  std::string id;
  std::string repository;
  std::string revision;
  std::string index_file;
  std::vector<PinnedFile> metadata_files;
  std::vector<PinnedShard> shards;
};

struct PinnedBundleDescriptor {
  std::string id;
  const PinnedModelRevision* model = nullptr;
  std::vector<PinnedTensor> tensors;
  bool require_single_shard = false;
};

enum class PinnedCheckpointErrorCode : std::uint8_t {
  kNone,
  kInvalidDescriptor,
  kInvalidRequest,
  kPathMissing,
  kPathTypeMismatch,
  kSymlinkRejected,
  kCanonicalizationFailed,
  kPathEscape,
  kFileSizeMismatch,
  kFileHashFailure,
  kFileHashMismatch,
  kIndexReadFailed,
  kIndexMappingMismatch,
  kShardHeaderReadFailed,
  kShardHeaderMismatch,
  kTensorMissing,
  kTensorMetadataMismatch,
  kRangeNotRepresentable,
  kPayloadReadFailed,
  kPayloadHashMismatch,
  kScalarMismatch,
  kAllocationFailure,
};

struct PinnedCheckpointError {
  PinnedCheckpointErrorCode code = PinnedCheckpointErrorCode::kNone;
  std::string stage;
  std::string path;
  std::string tensor;
  std::string expected;
  std::string actual;
  std::string message;
};

struct PinnedFileEvidence {
  std::string relative_path;
  std::string canonical_path;
  std::uint64_t file_bytes = 0U;
  std::string sha256;
  bool opened_read_only = false;
};

struct PinnedShardEvidence {
  std::string relative_path;
  std::string canonical_path;
  std::uint64_t file_bytes = 0U;
  std::uint64_t header_bytes = 0U;
  std::uint64_t data_offset = 0U;
  std::string expected_sha256;
  std::string observed_sha256;
  bool full_sha256_verified = false;
  bool opened_read_only = false;
};

struct PinnedTensorPayload {
  std::string name;
  std::string shard;
  io::safetensors::DType dtype = io::safetensors::DType::kBool;
  std::vector<std::uint64_t> shape;
  std::uint64_t file_begin = 0U;
  std::uint64_t file_end = 0U;
  std::vector<std::uint8_t> bytes;
  std::string sha256;
  std::optional<std::uint32_t> f32_bits;
  std::optional<float> f32_value;
};

struct LoadedPinnedBundle {
  std::string descriptor_id;
  std::string model_id;
  std::string repository;
  std::string revision;
  std::string canonical_directory;
  std::vector<PinnedFileEvidence> metadata_files;
  std::vector<PinnedShardEvidence> shards;
  std::vector<PinnedTensorPayload> tensors;

  [[nodiscard]] const PinnedTensorPayload* find_tensor(
      std::string_view name) const noexcept;
};

struct PinnedBundleLoadOptions {
  // Empty selects every tensor in the bundle descriptor.  A non-empty list is
  // useful for Gate-only, Down-only, or Gate+Up performance cells without
  // reading the other layer-0 payloads.
  std::vector<std::string> tensor_names;
  // Full-shard SHA is supported for revision audits but intentionally opt-in:
  // the pinned Qwen shard is almost 10 GB.  Metadata and selected tensor
  // payload SHA checks are always mandatory.
  bool verify_full_shard_sha256 = false;
  std::uint64_t maximum_total_payload_bytes = 512U * 1024U * 1024U;
};

struct PinnedBundleLoadResult {
  std::optional<LoadedPinnedBundle> value;
  std::optional<PinnedCheckpointError> error;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && !error.has_value();
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

[[nodiscard]] PinnedBundleLoadResult load_pinned_checkpoint_bundle(
    const std::filesystem::path& checkpoint_directory,
    const PinnedBundleDescriptor& descriptor,
    const PinnedBundleLoadOptions& options = {});

[[nodiscard]] const PinnedTensor* find_pinned_tensor(
    const PinnedBundleDescriptor& descriptor,
    std::string_view name) noexcept;

[[nodiscard]] std::string_view to_string(
    PinnedCheckpointErrorCode code) noexcept;
[[nodiscard]] std::string describe_pinned_checkpoint_error(
    const PinnedCheckpointError& error);

// Pinned production evidence for nvidia/Qwen3.6-27B-NVFP4 revision
// 0893e1606ff3d5f97a441f405d5fc541a6bdf404.  The descriptor covers the
// layer-0 Gate, Up, and Down NVFP4 weight/weight_scale/weight_scale_2 bundle.
[[nodiscard]] const PinnedModelRevision&
qwen36_27b_nvfp4_model_revision();
[[nodiscard]] const PinnedBundleDescriptor&
qwen36_27b_nvfp4_layer0_mlp_bundle();

inline constexpr std::string_view kQwen36Layer0GateWeight =
    "model.language_model.layers.0.mlp.gate_proj.weight";
inline constexpr std::string_view kQwen36Layer0GateBlockScale =
    "model.language_model.layers.0.mlp.gate_proj.weight_scale";
inline constexpr std::string_view kQwen36Layer0GateWeightScale2 =
    "model.language_model.layers.0.mlp.gate_proj.weight_scale_2";
inline constexpr std::string_view kQwen36Layer0UpWeight =
    "model.language_model.layers.0.mlp.up_proj.weight";
inline constexpr std::string_view kQwen36Layer0UpBlockScale =
    "model.language_model.layers.0.mlp.up_proj.weight_scale";
inline constexpr std::string_view kQwen36Layer0UpWeightScale2 =
    "model.language_model.layers.0.mlp.up_proj.weight_scale_2";
inline constexpr std::string_view kQwen36Layer0DownWeight =
    "model.language_model.layers.0.mlp.down_proj.weight";
inline constexpr std::string_view kQwen36Layer0DownBlockScale =
    "model.language_model.layers.0.mlp.down_proj.weight_scale";
inline constexpr std::string_view kQwen36Layer0DownWeightScale2 =
    "model.language_model.layers.0.mlp.down_proj.weight_scale_2";

}  // namespace q3x::test::support
