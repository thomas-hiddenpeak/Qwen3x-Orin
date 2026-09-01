#pragma once

#if !defined(Q3X_ENABLE_SM87_AOT_SYSTEM_V1_ARITHMETIC_WITNESS)
#error "The P40 projection catalog is a private, default-off test admission"
#endif

#include "q3x/core/sha256.h"
#include "q3x/model/weight_manifest.h"
#include "sm87_aot_system_v1_checkpoint_reader_internal.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace q3x::test::sm87_aot_p40_projection_catalog {

enum class ProjectionRole : std::uint8_t {
  kInvalid = 0U,
  kNvFp4GateUp,
  kNvFp4Down,
  kFp8GdnQkvZ,
  kFp8FullQkv,
  kFp8AttentionOutput,
};

enum class TensorKind : std::uint8_t {
  kWeight = 0U,
  kWeightScale,
  kWeightScale2,
  kInputScale,
};

struct TensorRecord {
  std::string name;
  std::uint32_t layer = 0U;
  ProjectionRole role = ProjectionRole::kInvalid;
  TensorKind kind = TensorKind::kWeight;
  io::safetensors::DType dtype = io::safetensors::DType::kBool;
  std::vector<std::uint64_t> shape;
  std::string shard;
  std::filesystem::path file;
  std::uint64_t file_begin = 0U;
  std::uint64_t file_end = 0U;
  core::Sha256Digest payload_sha256{};
  bool scalar_scale_bits_present = false;
  std::uint32_t scalar_scale_bits = 0U;
};

struct ModuleRecord {
  std::string name;
  std::uint32_t layer = 0U;
  ProjectionRole role = ProjectionRole::kInvalid;
  std::vector<std::size_t> tensor_indices;
};

struct Catalog {
  std::vector<ModuleRecord> modules;
  std::vector<TensorRecord> tensors;
  std::vector<sm87_aot_checkpoint_reader::ReaderReceipt> shard_receipts;
  core::Sha256Digest catalog_sha256{};
};

enum class ErrorCode : std::uint8_t {
  kNone = 0U,
  kInvalidManifest,
  kMissingModuleTensor,
  kTensorContractMismatch,
  kDuplicateTensor,
  kShardContractMismatch,
  kReaderFailure,
  kPayloadFailure,
  kCountMismatch,
  kArithmeticOverflow,
  kAllocationFailure,
  kInternalFailure,
};

struct Error {
  ErrorCode code = ErrorCode::kNone;
  std::string context;
  sm87_aot_checkpoint_reader::ReaderError reader_error{};
};

struct Result {
  std::optional<Catalog> value;
  Error error{};

  [[nodiscard]] bool ok() const noexcept { return value.has_value(); }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Builds the frozen 400-module P40 projection catalog. The supplied public
// manifest remains the metadata authority. Each of the three pinned shards is
// opened once and read sequentially by the authenticated checkpoint reader;
// no payload is retained. The result is published only after every shard and
// every selected tensor has authenticated successfully.
[[nodiscard]] Result build_p40_projection_tensor_catalog(
    const std::filesystem::path& checkpoint_root,
    const model::weights::WeightManifest& manifest,
    const sm87_aot_checkpoint_reader::ReaderOptions& reader_options = {});

[[nodiscard]] std::string_view to_string(ProjectionRole role) noexcept;
[[nodiscard]] std::string_view to_string(TensorKind kind) noexcept;
[[nodiscard]] std::string_view to_string(ErrorCode code) noexcept;
[[nodiscard]] bool run_p40_projection_catalog_protocol_self_test() noexcept;

inline constexpr std::size_t kP40ProjectionModuleCount = 400U;
inline constexpr std::size_t kP40ProjectionTensorCount = 1'392U;
inline constexpr std::string_view kP40ProjectionCatalogDomain =
    "q3x.sm87.aot-system-v1.p40-projection-tensor-catalog.v1";

}  // namespace q3x::test::sm87_aot_p40_projection_catalog
