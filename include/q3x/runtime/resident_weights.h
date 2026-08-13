#pragma once

#include "q3x/io/safetensors.h"
#include "q3x/model/weight_manifest.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace q3x::runtime {

inline constexpr std::uint64_t kResidentTensorAlignment = 256U;
inline constexpr std::uint64_t kPinnedQwen36_27BArenaBytes =
    20'150'786'560ULL;

struct ShardIdentity {
    std::string filename;
    std::uint64_t file_size = 0;
    std::string sha256;
};

// The immutable identity of the exact NVIDIA Qwen3.6-27B-NVFP4 payload used by
// the runtime. The filenames, full-file byte sizes, and SHA-256 digests are
// part of the loader trust boundary.
[[nodiscard]] const std::vector<ShardIdentity>&
pinned_qwen36_27b_shards() noexcept;

enum class ResidentLoadErrorCode : std::uint8_t {
    kNone,
    kInvalidOption,
    kInvalidManifest,
    kUnsafeShardPath,
    kDuplicateShard,
    kMissingShardIdentity,
    kUnexpectedShardIdentity,
    kInvalidShardIdentity,
    kArithmeticOverflow,
    kOpenDirectoryFailed,
    kOpenShardFailed,
    kShardNotRegular,
    kShardSizeMismatch,
    kIoFailure,
    kSha256BackendUnavailable,
    kSha256Failure,
    kSha256Mismatch,
    kCudaFailure,
    kInsufficientDeviceMemory,
    kAllocationFailure,
};

enum class ResidentSha256Backend : std::uint8_t {
    // Prefer the Linux AF_ALG SHA-256 implementation, but fall back to Q3X's
    // in-process implementation if AF_ALG cannot be initialized before any
    // shard bytes are consumed. The in-process implementation preserves the
    // portable digest/API contract and may transparently dispatch to a
    // runtime-detected CPU ISA implementation such as ARMv8 SHA2.
    kAuto,
    // Select Q3X's in-process implementation. This stable serialized name is
    // historical; it does not force the scalar compression function.
    kPortable,
    kLinuxAfAlg,
};

struct ResidentLoadDiagnostic {
    ResidentLoadErrorCode code = ResidentLoadErrorCode::kNone;
    std::string message;
    std::string context;
    std::string shard;
    std::uint64_t offset = 0;
    int system_error = 0;
    int cuda_error = 0;
    std::string expected;
    std::string actual;
};

struct PlannedTensor {
    std::string name;
    std::string shard;
    std::uint64_t source_begin = 0;
    std::uint64_t source_end = 0;
    std::uint64_t arena_offset = 0;
    std::uint64_t byte_size = 0;
    io::safetensors::DType dtype = io::safetensors::DType::kBool;
    std::vector<std::uint64_t> shape;
};

struct PlannedShard {
    ShardIdentity identity;
    // Indices into ResidentLoadPlan::tensors, in increasing source-offset
    // order. Only text tensors appear here; all other bytes are hash-only.
    std::vector<std::size_t> tensor_indices;
    std::uint64_t copied_bytes = 0;
};

struct ResidentLoadPlan {
    std::uint64_t arena_bytes = 0;
    std::uint64_t copied_bytes = 0;
    std::uint64_t source_bytes = 0;
    std::vector<PlannedTensor> tensors;
    std::vector<PlannedShard> shards;
};

struct PlanResult {
    std::optional<ResidentLoadPlan> value;
    ResidentLoadDiagnostic diagnostic;

    [[nodiscard]] bool ok() const noexcept {
        return value.has_value() &&
               diagnostic.code == ResidentLoadErrorCode::kNone;
    }
    [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Pure CPU planning and validation. It performs no file I/O and makes no CUDA
// calls. Every locator is range-checked against an identity; all source ranges
// must be disjoint; text tensors receive deterministic 256-byte-aligned arena
// offsets in (shard, source offset, tensor name) order.
[[nodiscard]] PlanResult build_resident_load_plan(
    const model::weights::WeightManifest& manifest,
    const std::vector<ShardIdentity>& identities);

struct DeviceTensorView {
    std::uint64_t arena_offset = 0;
    std::uint64_t byte_size = 0;
    const void* device_data = nullptr;
    io::safetensors::DType dtype = io::safetensors::DType::kBool;
    std::vector<std::uint64_t> shape;
};

struct ShardLoadStats {
    std::string filename;
    std::uint64_t bytes_read = 0;
    std::uint64_t bytes_copied = 0;
    std::uint64_t chunks = 0;
    std::uint64_t memcpy_operations = 0;
    std::string sha256;
};

struct ResidentLoadStats {
    std::uint64_t bytes_read = 0;
    std::uint64_t bytes_copied = 0;
    std::uint64_t bytes_skipped = 0;
    std::uint64_t chunks = 0;
    std::uint64_t memcpy_operations = 0;
    std::size_t shard_workers = 0U;
    std::uint64_t pinned_staging_bytes = 0U;
    std::uint64_t device_free_before = 0;
    std::uint64_t device_total = 0;
    // The concrete backend used for every shard. A successful load never
    // reports kAuto and never mixes backends between shards.
    ResidentSha256Backend sha256_backend = ResidentSha256Backend::kPortable;
    std::vector<ShardLoadStats> shards;
};

struct ResidentLoadOptions {
    // Each active shard worker uses two page-locked staging buffers of this
    // size in ping-pong fashion.
    std::uint64_t chunk_bytes = 64ULL * 1024ULL * 1024ULL;
    // cudaMalloc is attempted only if this many bytes will remain according
    // to cudaMemGetInfo. The default deliberately fails closed on tight Orin
    // unified-memory configurations.
    std::uint64_t min_free_bytes_after_load = 8ULL * 1024ULL * 1024ULL * 1024ULL;
    std::uint64_t max_arena_bytes = 32ULL * 1024ULL * 1024ULL * 1024ULL;
    std::size_t max_tensors = 10'000U;
    std::size_t max_shards = 16U;
    std::uint64_t max_memcpy_operations = 1'000'000ULL;
    ResidentSha256Backend sha256_backend = ResidentSha256Backend::kAuto;
    // Bound concurrent shard read/hash/scatter pipelines. The effective count
    // is also capped by the number of authenticated shards in the plan.
    std::size_t max_parallel_shards = 3U;
};

struct ResidentLoadResult;

class ResidentWeights {
  public:
    ResidentWeights() noexcept = default;
    ~ResidentWeights();

    ResidentWeights(const ResidentWeights&) = delete;
    ResidentWeights& operator=(const ResidentWeights&) = delete;
    ResidentWeights(ResidentWeights&& other) noexcept;
    ResidentWeights& operator=(ResidentWeights&& other) noexcept;

    [[nodiscard]] const void* arena_data() const noexcept { return arena_; }
    [[nodiscard]] std::uint64_t size_bytes() const noexcept {
        return arena_bytes_;
    }
    [[nodiscard]] std::size_t tensor_count() const noexcept {
        return tensors_.size();
    }
    [[nodiscard]] const DeviceTensorView* find(
        std::string_view name) const noexcept;
    [[nodiscard]] const ResidentLoadStats& stats() const noexcept {
        return stats_;
    }
    [[nodiscard]] explicit operator bool() const noexcept {
        return arena_ != nullptr;
    }

  private:
    friend ResidentLoadResult load_resident_weights(
        const std::filesystem::path&,
        const model::weights::WeightManifest&,
        const std::vector<ShardIdentity>&,
        const ResidentLoadOptions&);

    void release() noexcept;

    void* arena_ = nullptr;
    std::uint64_t arena_bytes_ = 0;
    std::map<std::string, DeviceTensorView, std::less<>> tensors_;
    ResidentLoadStats stats_;
};

struct ResidentLoadResult {
    std::optional<ResidentWeights> value;
    ResidentLoadDiagnostic diagnostic;

    [[nodiscard]] bool ok() const noexcept {
        return value.has_value() &&
               diagnostic.code == ResidentLoadErrorCode::kNone;
    }
    [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Executes a validated plan through one sequential read of every full shard.
// Shards may load concurrently, but each shard has an independent sequential
// pipeline. The same bytes feed SHA-256 and any intersecting H2D copies;
// payloads are never reread for hashing. Files are opened relative to an
// already-open root directory with O_NOFOLLOW on every path component.
[[nodiscard]] ResidentLoadResult load_resident_weights(
    const std::filesystem::path& directory,
    const model::weights::WeightManifest& manifest,
    const std::vector<ShardIdentity>& identities,
    const ResidentLoadOptions& options = {});

// Builds the strict pinned manifest, requires the exact 20,150,786,560-byte
// text arena contract, then loads only text tensors while authenticating every
// byte of all three official shard files.
[[nodiscard]] ResidentLoadResult load_pinned_qwen36_27b(
    const std::filesystem::path& directory,
    const ResidentLoadOptions& options = {});

[[nodiscard]] std::string_view to_string(
    ResidentLoadErrorCode code) noexcept;
[[nodiscard]] std::string_view to_string(
    ResidentSha256Backend backend) noexcept;

}  // namespace q3x::runtime
