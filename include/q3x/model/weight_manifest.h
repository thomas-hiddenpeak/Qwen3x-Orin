#pragma once

#include "q3x/io/safetensors.h"
#include "q3x/model/checkpoint_metadata.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace q3x::model::weights {

enum class TensorCategory : std::uint8_t {
    kText,
    kVision,
    kMtp,
    kUnknown,
};

enum class ManifestErrorCode : std::uint8_t {
    kNone,
    kInvalidOption,
    kCheckpointRejected,
    kUnsupportedCheckpoint,
    kIndexRejected,
    kShardRejected,
    kUnsafeShardPath,
    kDuplicateTensor,
    kUnknownTensor,
    kUnexpectedTextTensor,
    kMissingTensor,
    kDTypeMismatch,
    kShapeMismatch,
    kByteSizeMismatch,
    kOffsetMismatch,
    kQuantizationMismatch,
    kCountMismatch,
    kSizeMismatch,
    kArithmeticOverflow,
    kIoFailure,
    kAllocationFailure,
};

struct ManifestDiagnostic {
    ManifestErrorCode code = ManifestErrorCode::kNone;
    std::string context;
    std::string expected;
    std::string actual;
    std::string message;
};

// A locator never owns or reads tensor payload. file_begin/file_end are
// absolute byte offsets within `file`, not offsets relative to the safetensors
// data buffer.
struct TensorLocator {
    TensorCategory category = TensorCategory::kUnknown;
    std::string shard;
    std::filesystem::path file;
    std::uint64_t file_begin = 0;
    std::uint64_t file_end = 0;
    std::uint64_t byte_size = 0;
    io::safetensors::DType dtype = io::safetensors::DType::kBool;
    std::vector<std::uint64_t> shape;
};

struct WeightManifestSummary {
    std::size_t shard_count = 0;
    std::size_t tensor_count = 0;
    std::size_t text_tensor_count = 0;
    std::size_t vision_tensor_count = 0;
    std::size_t mtp_tensor_count = 0;
    std::size_t fp8_module_count = 0;
    std::size_t nvfp4_module_count = 0;

    std::uint64_t raw_text_bytes = 0;
    std::uint64_t vision_bytes = 0;
    std::uint64_t mtp_bytes = 0;
    std::uint64_t skipped_bytes = 0;
    std::uint64_t arena_alignment = 0;
    std::uint64_t estimated_text_arena_bytes = 0;
};

struct WeightManifest {
    checkpoint::KnownCheckpointDescriptor checkpoint;
    WeightManifestSummary summary;
    std::map<std::string, TensorLocator, std::less<>> tensors;

    [[nodiscard]] const TensorLocator* find(
        std::string_view name) const noexcept;
};

struct ManifestOptions {
    // Each text tensor begins at this alignment in the estimated runtime
    // arena. Must be a non-zero power of two.
    std::uint64_t arena_alignment = 256;
    io::safetensors::IndexReadOptions index_options;
    io::safetensors::ShardValidationOptions shard_options;
};

struct ManifestResult {
    std::optional<WeightManifest> value;
    std::vector<ManifestDiagnostic> diagnostics;

    [[nodiscard]] bool ok() const noexcept { return value.has_value(); }
    [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Low-level ABI validator for an already collected locator table. This entry
// point exists for synthetic tests and loader composition. It validates the
// complete Qwen3.6-27B text tensor ABI against the supplied, already validated
// ModelOpt table, classifies vision/MTP tensors as skipped, and rejects every
// unknown or additional text tensor.
[[nodiscard]] ManifestResult validate_qwen36_27b_text_manifest(
    std::map<std::string, TensorLocator, std::less<>> tensors,
    const checkpoint::ModelOptSummary& quantization,
    std::uint64_t arena_alignment = 256);

// End-to-end pinned artifact gate. It first requires successful strict
// checkpoint inspection (including all shard headers and the index contract),
// then rebuilds locators from headers and applies the complete text ABI. Tensor
// payload bytes are never read.
[[nodiscard]] ManifestResult build_qwen36_27b_text_manifest(
    const std::filesystem::path& directory,
    const ManifestOptions& options = {});

[[nodiscard]] TensorCategory classify_tensor(std::string_view name) noexcept;
[[nodiscard]] std::string_view to_string(TensorCategory category) noexcept;
[[nodiscard]] std::string_view to_string(ManifestErrorCode code) noexcept;

inline constexpr std::size_t kPinnedQwen36_27BTextTensorCount = 1846;
inline constexpr std::size_t kPinnedQwen36_27BVisionTensorCount = 333;
inline constexpr std::size_t kPinnedQwen36_27BMtpTensorCount = 15;
inline constexpr std::uint64_t kPinnedQwen36_27BTextBytes = 20150569096ULL;
inline constexpr std::uint64_t kPinnedQwen36_27BVisionBytes = 921460192ULL;
inline constexpr std::uint64_t kPinnedQwen36_27BMtpBytes = 849398784ULL;

}  // namespace q3x::model::weights
