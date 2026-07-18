#pragma once

#include "q3x/io/json.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace q3x::io::safetensors {

// Mirrors the dtype spellings accepted by safetensors 0.8. Sub-byte types are
// represented by their bit width and are accepted only when a whole tensor is
// byte aligned.
enum class DType : std::uint8_t {
    kBool,
    kF4,
    kF6E2M3,
    kF6E3M2,
    kU8,
    kI8,
    kF8E5M2,
    kF8E4M3,
    kF8E8M0,
    kF8E4M3Fnuz,
    kF8E5M2Fnuz,
    kI16,
    kU16,
    kF16,
    kBf16,
    kI32,
    kU32,
    kF32,
    kC64,
    kF64,
    kI64,
    kU64,
};

[[nodiscard]] std::optional<DType> parse_dtype(std::string_view name) noexcept;
[[nodiscard]] std::string_view to_string(DType dtype) noexcept;
[[nodiscard]] std::uint8_t bit_width(DType dtype) noexcept;

enum class ErrorCode : std::uint8_t {
    kNone,
    kInvalidOption,
    kOpenFailed,
    kIoFailure,
    kFileTooSmall,
    kFileTooLarge,
    kHeaderTooLarge,
    kInvalidHeaderLength,
    kInvalidHeaderStart,
    kInvalidJson,
    kHeaderNotObject,
    kTooManyTensors,
    kTensorDescriptorNotObject,
    kMissingTensorField,
    kUnknownTensorField,
    kUnsupportedDType,
    kInvalidShape,
    kRankLimitExceeded,
    kArithmeticOverflow,
    kInvalidDataOffsets,
    kMisalignedTensor,
    kTensorSizeMismatch,
    kDataOutOfRange,
    kOverlappingData,
    kDataGap,
    kDataNotFullyCovered,
    kInvalidMetadata,
    kIndexTooLarge,
    kIndexNotObject,
    kMissingWeightMap,
    kInvalidWeightMap,
    kTooManyShards,
    kUnsafeShardPath,
    kShardMissing,
    kShardNotRegular,
    kShardSetMismatch,
    kUnexpectedTensor,
    kTensorInWrongShard,
    kMissingIndexedTensor,
    kMissingTotalSize,
    kPayloadSizeMismatch,
    kAllocationFailure,
};

inline constexpr std::uint64_t kUnknownOffset =
    std::numeric_limits<std::uint64_t>::max();
inline constexpr std::uint64_t kDefaultMaxHeaderBytes = 100'000'000;

struct Error {
    ErrorCode code = ErrorCode::kNone;
    std::uint64_t offset = kUnknownOffset;
    // Tensor name, field name, shard name, or path associated with the error.
    std::string context;
    // Optional machine-readable comparison details used by contract checks.
    std::string expected;
    std::string actual;

    [[nodiscard]] constexpr bool ok() const noexcept {
        return code == ErrorCode::kNone;
    }

    [[nodiscard]] std::string_view message() const noexcept;
};

template <typename T>
struct Result {
    std::optional<T> value;
    Error error;

    [[nodiscard]] bool ok() const noexcept {
        return value.has_value() && error.ok();
    }

    [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

struct TensorInfo {
    DType dtype = DType::kBool;
    std::vector<std::uint64_t> shape;
    std::uint64_t element_count = 0;
    std::uint64_t byte_size = 0;
    // Relative to Header::data_offset, as defined by safetensors.
    std::uint64_t data_begin = 0;
    std::uint64_t data_end = 0;
    // Absolute positions in the source file, useful for bounded pread/mmap.
    std::uint64_t file_begin = 0;
    std::uint64_t file_end = 0;
};

struct Header {
    std::uint64_t file_size = 0;
    std::uint64_t header_size = 0;
    std::uint64_t data_offset = 0;
    std::uint64_t data_size = 0;
    std::map<std::string, TensorInfo, std::less<>> tensors;
    std::map<std::string, std::string, std::less<>> metadata;

    [[nodiscard]] const TensorInfo* find_tensor(std::string_view name) const noexcept;
};

struct ReadOptions {
    std::uint64_t max_file_bytes = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t max_header_bytes = kDefaultMaxHeaderBytes;
    std::size_t max_tensors = 2'000'000;
    std::size_t max_rank = 64;
    json::ParseOptions json_options;
};

// Reads and validates only the prefix/header; tensor payload bytes are never
// copied. Validation includes exact shape*dtype sizing, checked arithmetic,
// in-range non-overlapping offsets, and complete coverage of the data buffer.
[[nodiscard]] Result<Header> read_header(const std::string& path,
                                         const ReadOptions& options = {});

struct Index {
    std::map<std::string, std::string, std::less<>> weight_map;
    std::vector<std::string> shards;
    json::Value::Object metadata;
    std::optional<std::uint64_t> total_size;

    [[nodiscard]] const std::string* shard_for(
        std::string_view tensor_name) const noexcept;
};

struct IndexReadOptions {
    std::uint64_t max_file_bytes = 64U * 1024U * 1024U;
    std::size_t max_tensors = 2'000'000;
    std::size_t max_shards = 100'000;
    json::ParseOptions json_options;
};

// Parses a Hugging Face *.safetensors.index.json file. `shards` is the sorted,
// deduplicated set of filenames referenced by weight_map.
[[nodiscard]] Result<Index> read_index(const std::string& path,
                                       const IndexReadOptions& options = {});

struct ShardValidationOptions {
    std::size_t max_shards = 100'000;
    std::size_t max_tensors = 2'000'000;
    ReadOptions header_options;
};

struct ShardValidationSummary {
    std::size_t validated_shards = 0;
    std::size_t validated_tensors = 0;
    std::uint64_t payload_bytes = 0;
};

// Lexical policy for shard paths supplied by an untrusted index. Backslashes,
// absolute/rooted paths, dot components, control characters, and non-
// safetensors extensions are rejected.
[[nodiscard]] bool is_safe_relative_shard_path(
    std::string_view path) noexcept;

// Validates the complete contract between an index and every referenced shard.
// Each shard is required to be a regular non-symlink file and is inspected via
// read_header(); payload bytes are never read. The validator checks exact
// tensor membership and shard ownership, defensive limits, checked aggregate
// counts, and metadata.total_size against the sum of shard payload sizes.
[[nodiscard]] Result<ShardValidationSummary> validate_index_shards(
    const std::filesystem::path& directory,
    const Index& index,
    const ShardValidationOptions& options = {});

[[nodiscard]] std::string_view to_string(ErrorCode code) noexcept;

}  // namespace q3x::io::safetensors
