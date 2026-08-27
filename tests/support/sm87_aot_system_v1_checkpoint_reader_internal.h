#pragma once

#if !defined(Q3X_ENABLE_SM87_AOT_SYSTEM_V1_ARITHMETIC_WITNESS)
#error "The SM87 AOT checkpoint reader is a private, default-off test admission"
#endif

#include "q3x/core/sha256.h"
#include "q3x/io/safetensors.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace q3x::test::sm87_aot_checkpoint_reader {

struct PinnedShardIdentity {
  std::string_view filename;
  std::uint64_t file_size = 0U;
  core::Sha256Digest full_sha256{};
};

struct TensorRangeRequest {
  std::string tensor_name;
  io::safetensors::DType dtype = io::safetensors::DType::kBool;
  std::vector<std::uint64_t> shape;
  std::uint64_t file_begin = 0U;
  std::uint64_t file_end = 0U;
};

struct SinkTransaction {
  std::uint64_t expected_file_size = 0U;
  std::uint64_t expected_selected_bytes = 0U;
  std::size_t expected_range_count = 0U;
};

struct SinkConsumeResult {
  bool accepted = false;
  std::size_t consumed_bytes = 0U;
};

// The sink owns provisional state once begin() is invoked. Every path other
// than commit()==true receives exactly one abort(), including begin()==false,
// a thrown callback, and commit()==false. Only a successful commit suppresses
// abort. consume() receives only bytes that came directly from the reader's
// current sequential read buffer.
class ProvisionalRangeSink {
 public:
  virtual ~ProvisionalRangeSink() = default;

  [[nodiscard]] virtual bool begin(const SinkTransaction& transaction) = 0;
  [[nodiscard]] virtual SinkConsumeResult consume(
      std::size_t range_index,
      std::uint64_t absolute_file_offset,
      std::uint64_t tensor_byte_offset,
      const std::uint8_t* bytes,
      std::size_t byte_count) = 0;
  [[nodiscard]] virtual bool commit() = 0;
  virtual void abort() noexcept = 0;
};

struct ReaderOptions {
  std::size_t chunk_bytes = 1024U * 1024U;
  std::size_t max_selected_ranges = 4096U;
  io::safetensors::ReadOptions header_options{};
};

enum class ReaderErrorCode : std::uint8_t {
  kNone = 0U,
  kInvalidOption,
  kUnsafeShardFilename,
  kUnknownPinnedShard,
  kRootOpenFailed,
  kShardOpenFailed,
  kStatFailed,
  kShardNotRegular,
  kShardSizeMismatch,
  kIoFailure,
  kUnexpectedEof,
  kUnexpectedTrailingData,
  kInvalidPrefix,
  kHeaderInvalid,
  kTooManyRanges,
  kInvalidRange,
  kDuplicateRange,
  kRangesNotOrdered,
  kRangesOverlap,
  kMissingTensor,
  kTensorMetadataMismatch,
  kArithmeticOverflow,
  kSinkBeginFailure,
  kSinkFailure,
  kSinkShortDelivery,
  kRangeDeliveryIncomplete,
  kShardHashMismatch,
  kShardMutated,
  kSinkCommitFailure,
  kAllocationFailure,
  kInternalFailure,
};

struct ReaderError {
  ReaderErrorCode code = ReaderErrorCode::kNone;
  std::size_t range_index = static_cast<std::size_t>(-1);
  int system_error = 0;
  io::safetensors::Error header_error{};

  [[nodiscard]] constexpr bool ok() const noexcept {
    return code == ReaderErrorCode::kNone;
  }
};

struct ReaderReceipt {
  std::array<char, 256U> shard_filename{};
  std::size_t shard_filename_size = 0U;
  std::uint64_t expected_file_size = 0U;
  core::Sha256Digest expected_full_sha256{};
  std::uint64_t file_size = 0U;
  std::uint64_t bytes_read = 0U;
  std::uint64_t selected_bytes = 0U;
  std::size_t selected_range_count = 0U;
  std::size_t sequential_read_calls = 0U;
  std::size_t sink_delivery_calls = 0U;
  core::Sha256Digest full_sha256{};
  core::Sha256Digest range_inventory_sha256{};
  core::Sha256Digest receipt_sha256{};
  bool committed = false;
};

struct ReaderResult {
  std::optional<ReaderReceipt> receipt;
  ReaderError error{};

  [[nodiscard]] bool ok() const noexcept {
    return receipt.has_value() && receipt->committed && error.ok();
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Returns an identity only for the frozen three-shard checkpoint. The table
// itself remains private to the reader implementation.
[[nodiscard]] std::optional<PinnedShardIdentity> pinned_shard_identity(
    std::string_view filename) noexcept;

// Opens root once as O_DIRECTORY|O_NOFOLLOW and the frozen single-component
// shard once with openat(O_RDONLY|O_CLOEXEC|O_NOFOLLOW|O_NONBLOCK). O_NONBLOCK
// prevents a special-file substitution from blocking before fstat rejects it.
// The same shard fd is then consumed sequentially through prefix, header,
// payload, SHA-256, EOF, and post-read identity validation.
[[nodiscard]] ReaderResult read_pinned_shard_ranges(
    const std::filesystem::path& checkpoint_root,
    std::string_view pinned_shard_filename,
    const std::vector<TensorRangeRequest>& ranges,
    ProvisionalRangeSink& sink,
    const ReaderOptions& options = {});

[[nodiscard]] std::string_view to_string(ReaderErrorCode code) noexcept;

// Synthetic host-only adversarial coverage. It never opens the pinned model.
[[nodiscard]] bool run_checkpoint_reader_self_test() noexcept;

}  // namespace q3x::test::sm87_aot_checkpoint_reader
