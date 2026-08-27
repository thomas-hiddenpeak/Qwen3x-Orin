#include "sm87_aot_system_v1_checkpoint_reader_internal.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <new>
#include <set>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <utility>

namespace q3x::test::sm87_aot_checkpoint_reader {
namespace {

inline constexpr std::size_t kMaximumChunkBytes = 64U * 1024U * 1024U;
inline constexpr std::size_t kMaximumShardFilenameBytes = 255U;
inline constexpr std::string_view kReceiptDomain =
    "q3x.sm87.aot-system-v1.checkpoint-range-receipt.v1";

constexpr std::uint8_t hex_nibble(const char value) noexcept {
  return value >= '0' && value <= '9'
             ? static_cast<std::uint8_t>(value - '0')
         : value >= 'a' && value <= 'f'
             ? static_cast<std::uint8_t>(10 + value - 'a')
         : value >= 'A' && value <= 'F'
             ? static_cast<std::uint8_t>(10 + value - 'A')
             : 0U;
}

template <std::size_t Size>
constexpr core::Sha256Digest digest_from_hex(
    const char (&text)[Size]) noexcept {
  static_assert(Size == 65U, "a SHA-256 literal has 64 hexadecimal digits");
  core::Sha256Digest digest{};
  for (std::size_t index = 0U; index < digest.bytes.size(); ++index) {
    digest.bytes[index] = static_cast<std::uint8_t>(
        (hex_nibble(text[index * 2U]) << 4U) |
        hex_nibble(text[index * 2U + 1U]));
  }
  return digest;
}

inline constexpr std::array<PinnedShardIdentity, 3U> kPinnedShards = {{
    {"model-00001-of-00003.safetensors",
     9'965'652'512ULL,
     digest_from_hex(
         "b4a0d9a57ff1859dac1144b53ca285011db072737d8813fc16d8d1e07ecae17d")},
    {"model-00002-of-00003.safetensors",
     9'985'757'032ULL,
     digest_from_hex(
         "06da4242b0f491118d19d4d4c7564307a7bd6059c6bed284e08c93f6fc5a556d")},
    {"model-00003-of-00003.safetensors",
     1'970'287'640ULL,
     digest_from_hex(
         "e90f5b2bb16814a0565de284ea179edec201edfb120d13f1debaab66f9e60845")},
}};

class UniqueFd {
 public:
  explicit UniqueFd(const int value = -1) noexcept : value_(value) {}
  ~UniqueFd() {
    if (value_ >= 0) {
      (void)::close(value_);
    }
  }

  UniqueFd(const UniqueFd&) = delete;
  UniqueFd& operator=(const UniqueFd&) = delete;

  [[nodiscard]] int get() const noexcept { return value_; }

 private:
  int value_ = -1;
};

class SinkAbortGuard {
 public:
  explicit SinkAbortGuard(ProvisionalRangeSink& sink) noexcept : sink_(sink) {}
  ~SinkAbortGuard() {
    if (begin_invoked_ && !committed_) {
      sink_.abort();
    }
  }

  void mark_begin_invoked() noexcept { begin_invoked_ = true; }
  void mark_committed() noexcept { committed_ = true; }

 private:
  ProvisionalRangeSink& sink_;
  bool begin_invoked_ = false;
  bool committed_ = false;
};

struct FileIdentity {
  dev_t device = 0;
  ino_t inode = 0;
  mode_t type = 0;
  off_t size = 0;
  timespec modification{};
  timespec change{};
};

[[nodiscard]] FileIdentity file_identity(const struct stat& status) noexcept {
  return {status.st_dev,
          status.st_ino,
          static_cast<mode_t>(status.st_mode & S_IFMT),
          status.st_size,
          status.st_mtim,
          status.st_ctim};
}

[[nodiscard]] bool same_timespec(const timespec& left,
                                 const timespec& right) noexcept {
  return left.tv_sec == right.tv_sec && left.tv_nsec == right.tv_nsec;
}

[[nodiscard]] bool same_file_identity(const FileIdentity& left,
                                      const FileIdentity& right) noexcept {
  return left.device == right.device && left.inode == right.inode &&
         left.type == right.type && left.size == right.size &&
         same_timespec(left.modification, right.modification) &&
         same_timespec(left.change, right.change);
}

[[nodiscard]] bool is_safe_single_shard_filename(
    const std::string_view filename) noexcept {
  if (filename.empty() || filename.size() > kMaximumShardFilenameBytes ||
      filename == "." || filename == ".." ||
      filename.size() < std::string_view(".safetensors").size() ||
      filename.substr(filename.size() -
                      std::string_view(".safetensors").size()) !=
          ".safetensors") {
    return false;
  }
  for (const char character : filename) {
    const auto byte = static_cast<unsigned char>(character);
    if (byte < 0x20U || byte == 0x7fU || character == '/' ||
        character == '\\' || character == ':') {
      return false;
    }
  }
  return true;
}

[[nodiscard]] ReaderResult failure(
    const ReaderErrorCode code,
    const std::size_t range_index = static_cast<std::size_t>(-1),
    const int system_error = 0) noexcept {
  ReaderResult result;
  result.error.code = code;
  result.error.range_index = range_index;
  result.error.system_error = system_error;
  return result;
}

[[nodiscard]] ReaderResult header_failure(
    io::safetensors::Error error) noexcept {
  ReaderResult result;
  result.error.code = ReaderErrorCode::kHeaderInvalid;
  result.error.header_error = std::move(error);
  return result;
}

[[nodiscard]] bool checked_add(const std::uint64_t left,
                               const std::uint64_t right,
                               std::uint64_t& output) noexcept {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    return false;
  }
  output = left + right;
  return true;
}

void hash_u64(core::Sha256& hasher, const std::uint64_t value) noexcept {
  std::array<std::uint8_t, 8U> bytes{};
  for (std::size_t index = 0U; index < bytes.size(); ++index) {
    bytes[index] = static_cast<std::uint8_t>(value >> (index * 8U));
  }
  (void)hasher.update(bytes.data(), bytes.size());
}

void hash_bytes(core::Sha256& hasher,
                const void* const bytes,
                const std::size_t byte_count) noexcept {
  (void)hasher.update(bytes, byte_count);
}

[[nodiscard]] core::Sha256Digest range_inventory_digest(
    const std::vector<TensorRangeRequest>& ranges) noexcept {
  core::Sha256 hasher;
  constexpr std::string_view domain =
      "q3x.sm87.aot-system-v1.selected-safetensors-ranges.v1";
  hash_bytes(hasher, domain.data(), domain.size());
  hash_u64(hasher, static_cast<std::uint64_t>(ranges.size()));
  for (const TensorRangeRequest& range : ranges) {
    hash_u64(hasher, static_cast<std::uint64_t>(range.tensor_name.size()));
    hash_bytes(hasher, range.tensor_name.data(), range.tensor_name.size());
    hash_u64(hasher, static_cast<std::uint64_t>(range.dtype));
    hash_u64(hasher, static_cast<std::uint64_t>(range.shape.size()));
    for (const std::uint64_t dimension : range.shape) {
      hash_u64(hasher, dimension);
    }
    hash_u64(hasher, range.file_begin);
    hash_u64(hasher, range.file_end);
  }
  return hasher.finalize();
}

[[nodiscard]] core::Sha256Digest sign_receipt(
    const ReaderReceipt& receipt) noexcept {
  core::Sha256 hasher;
  hash_bytes(hasher, kReceiptDomain.data(), kReceiptDomain.size());
  hash_u64(hasher, static_cast<std::uint64_t>(receipt.shard_filename_size));
  hash_bytes(hasher,
             receipt.shard_filename.data(),
             receipt.shard_filename_size);
  hash_u64(hasher, receipt.expected_file_size);
  hash_bytes(hasher,
             receipt.expected_full_sha256.bytes.data(),
             receipt.expected_full_sha256.bytes.size());
  hash_u64(hasher, receipt.file_size);
  hash_u64(hasher, receipt.bytes_read);
  hash_u64(hasher, receipt.selected_bytes);
  hash_u64(hasher,
           static_cast<std::uint64_t>(receipt.selected_range_count));
  hash_u64(hasher, static_cast<std::uint64_t>(receipt.sequential_read_calls));
  hash_u64(hasher, static_cast<std::uint64_t>(receipt.sink_delivery_calls));
  hash_bytes(hasher,
             receipt.full_sha256.bytes.data(),
             receipt.full_sha256.bytes.size());
  hash_bytes(hasher,
             receipt.range_inventory_sha256.bytes.data(),
             receipt.range_inventory_sha256.bytes.size());
  constexpr std::uint8_t committed = 1U;
  hash_bytes(hasher, &committed, sizeof(committed));
  return hasher.finalize();
}

class SequentialHasher {
 public:
  SequentialHasher(const int descriptor,
                   const std::uint64_t expected_size,
                   const std::size_t maximum_read_bytes) noexcept
      : descriptor_(descriptor),
        expected_size_(expected_size),
        maximum_read_bytes_(maximum_read_bytes) {}

  enum class ReadState : std::uint8_t { kData, kEof, kError };

  [[nodiscard]] ReadState read_some(std::uint8_t* const destination,
                                    const std::size_t capacity,
                                    std::size_t& read_bytes) noexcept {
    read_bytes = 0U;
    if (capacity == 0U || total_bytes_ >= expected_size_) {
      return ReadState::kEof;
    }
    const std::uint64_t remaining = expected_size_ - total_bytes_;
    const std::size_t request = static_cast<std::size_t>(
        std::min<std::uint64_t>(remaining,
                                static_cast<std::uint64_t>(
                                    std::min(capacity,
                                             maximum_read_bytes_))));
    ssize_t count = -1;
    do {
      count = ::read(descriptor_, destination, request);
    } while (count < 0 && errno == EINTR);
    ++read_calls_;
    if (count < 0) {
      error_ = errno;
      return ReadState::kError;
    }
    if (count == 0) {
      return ReadState::kEof;
    }
    read_bytes = static_cast<std::size_t>(count);
    if (!hasher_.update(destination, read_bytes) ||
        !checked_add(total_bytes_,
                     static_cast<std::uint64_t>(read_bytes),
                     total_bytes_)) {
      error_ = EOVERFLOW;
      return ReadState::kError;
    }
    return ReadState::kData;
  }

  [[nodiscard]] bool read_exact(std::uint8_t* destination,
                                std::size_t byte_count) noexcept {
    while (byte_count != 0U) {
      std::size_t read_bytes = 0U;
      const ReadState state = read_some(destination, byte_count, read_bytes);
      if (state != ReadState::kData) {
        unexpected_eof_ = state == ReadState::kEof;
        return false;
      }
      destination += read_bytes;
      byte_count -= read_bytes;
    }
    return true;
  }

  [[nodiscard]] bool require_eof() noexcept {
    std::uint8_t extra = 0U;
    ssize_t count = -1;
    do {
      count = ::read(descriptor_, &extra, sizeof(extra));
    } while (count < 0 && errno == EINTR);
    ++read_calls_;
    if (count < 0) {
      error_ = errno;
      return false;
    }
    trailing_data_ = count != 0;
    return count == 0;
  }

  [[nodiscard]] core::Sha256Digest finalize() noexcept {
    return hasher_.finalize();
  }
  [[nodiscard]] std::uint64_t total_bytes() const noexcept {
    return total_bytes_;
  }
  [[nodiscard]] std::size_t read_calls() const noexcept { return read_calls_; }
  [[nodiscard]] int error() const noexcept { return error_; }
  [[nodiscard]] bool unexpected_eof() const noexcept { return unexpected_eof_; }
  [[nodiscard]] bool trailing_data() const noexcept { return trailing_data_; }

 private:
  int descriptor_ = -1;
  std::uint64_t expected_size_ = 0U;
  std::size_t maximum_read_bytes_ = 0U;
  core::Sha256 hasher_{};
  std::uint64_t total_bytes_ = 0U;
  std::size_t read_calls_ = 0U;
  int error_ = 0;
  bool unexpected_eof_ = false;
  bool trailing_data_ = false;
};

[[nodiscard]] std::uint64_t decode_little_endian_u64(
    const std::array<std::uint8_t, 8U>& bytes) noexcept {
  std::uint64_t value = 0U;
  for (std::size_t index = 0U; index < bytes.size(); ++index) {
    value |= static_cast<std::uint64_t>(bytes[index]) << (index * 8U);
  }
  return value;
}

[[nodiscard]] ReaderResult validate_ranges(
    const io::safetensors::Header& header,
    const std::vector<TensorRangeRequest>& ranges,
    const ReaderOptions& options,
    std::uint64_t& selected_bytes) {
  if (ranges.empty()) {
    return failure(ReaderErrorCode::kInvalidRange);
  }
  if (ranges.size() > options.max_selected_ranges) {
    return failure(ReaderErrorCode::kTooManyRanges);
  }

  std::set<std::string_view, std::less<>> names;
  selected_bytes = 0U;
  for (std::size_t index = 0U; index < ranges.size(); ++index) {
    const TensorRangeRequest& range = ranges[index];
    if (range.tensor_name.empty() ||
        range.tensor_name.find('\0') != std::string::npos ||
        range.file_begin >= range.file_end ||
        range.file_begin < header.data_offset ||
        range.file_end > header.file_size) {
      return failure(ReaderErrorCode::kInvalidRange, index);
    }
    if (!names.emplace(range.tensor_name).second) {
      return failure(ReaderErrorCode::kDuplicateRange, index);
    }
    if (index != 0U) {
      const TensorRangeRequest& previous = ranges[index - 1U];
      if (range.file_begin < previous.file_begin) {
        return failure(ReaderErrorCode::kRangesNotOrdered, index);
      }
      if (range.file_begin < previous.file_end) {
        return failure(ReaderErrorCode::kRangesOverlap, index);
      }
    }

    const io::safetensors::TensorInfo* const tensor =
        header.find_tensor(range.tensor_name);
    if (tensor == nullptr) {
      return failure(ReaderErrorCode::kMissingTensor, index);
    }
    if (tensor->dtype != range.dtype || tensor->shape != range.shape ||
        tensor->file_begin != range.file_begin ||
        tensor->file_end != range.file_end ||
        tensor->byte_size != range.file_end - range.file_begin) {
      return failure(ReaderErrorCode::kTensorMetadataMismatch, index);
    }
    if (!checked_add(selected_bytes,
                     range.file_end - range.file_begin,
                     selected_bytes)) {
      return failure(ReaderErrorCode::kArithmeticOverflow, index);
    }
  }
  return {};
}

[[nodiscard]] ReaderResult read_authenticated_ranges(
    const std::filesystem::path& checkpoint_root,
    const PinnedShardIdentity& expected,
    const std::vector<TensorRangeRequest>& ranges,
    ProvisionalRangeSink& sink,
    const ReaderOptions& options) {
  try {
    if (options.chunk_bytes == 0U ||
        options.chunk_bytes > kMaximumChunkBytes ||
        options.max_selected_ranges == 0U ||
        options.header_options.max_header_bytes == 0U ||
        options.header_options.max_tensors == 0U ||
        options.header_options.max_rank == 0U) {
      return failure(ReaderErrorCode::kInvalidOption);
    }
    if (!is_safe_single_shard_filename(expected.filename)) {
      return failure(ReaderErrorCode::kUnsafeShardFilename);
    }
    if (ranges.empty()) {
      return failure(ReaderErrorCode::kInvalidRange);
    }
    if (ranges.size() > options.max_selected_ranges) {
      return failure(ReaderErrorCode::kTooManyRanges);
    }

    // Freeze caller-owned descriptors before opening the shard. A sink
    // callback cannot alter the metadata that governs later delivery.
    const std::vector<TensorRangeRequest> selected_ranges = ranges;

    const std::string filename(expected.filename);
    const UniqueFd root_descriptor(::open(checkpoint_root.c_str(),
                                          O_RDONLY | O_DIRECTORY | O_CLOEXEC |
                                              O_NOFOLLOW));
    if (root_descriptor.get() < 0) {
      return failure(ReaderErrorCode::kRootOpenFailed,
                     static_cast<std::size_t>(-1),
                     errno);
    }
    const UniqueFd shard_descriptor(
        ::openat(root_descriptor.get(),
                 filename.c_str(),
                 O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK));
    if (shard_descriptor.get() < 0) {
      return failure(ReaderErrorCode::kShardOpenFailed,
                     static_cast<std::size_t>(-1),
                     errno);
    }

    struct stat pre_status {};
    if (::fstat(shard_descriptor.get(), &pre_status) != 0) {
      return failure(ReaderErrorCode::kStatFailed,
                     static_cast<std::size_t>(-1),
                     errno);
    }
    if (!S_ISREG(pre_status.st_mode)) {
      return failure(ReaderErrorCode::kShardNotRegular);
    }
    if (pre_status.st_size < 0 ||
        static_cast<std::uint64_t>(pre_status.st_size) != expected.file_size) {
      return failure(ReaderErrorCode::kShardSizeMismatch);
    }
    const FileIdentity pre_identity = file_identity(pre_status);

    SequentialHasher reader(
        shard_descriptor.get(), expected.file_size, options.chunk_bytes);
    std::array<std::uint8_t, 8U> prefix{};
    if (!reader.read_exact(prefix.data(), prefix.size())) {
      return failure(reader.unexpected_eof() ? ReaderErrorCode::kUnexpectedEof
                                             : ReaderErrorCode::kIoFailure,
                     static_cast<std::size_t>(-1),
                     reader.error());
    }
    const std::uint64_t header_size = decode_little_endian_u64(prefix);
    if (header_size == 0U ||
        header_size > options.header_options.max_header_bytes ||
        header_size >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max()) ||
        header_size > expected.file_size - prefix.size()) {
      return failure(ReaderErrorCode::kInvalidPrefix);
    }

    std::string header_text(static_cast<std::size_t>(header_size), '\0');
    if (!reader.read_exact(
            reinterpret_cast<std::uint8_t*>(header_text.data()),
            header_text.size())) {
      return failure(reader.unexpected_eof() ? ReaderErrorCode::kUnexpectedEof
                                             : ReaderErrorCode::kIoFailure,
                     static_cast<std::size_t>(-1),
                     reader.error());
    }
    io::safetensors::ReadOptions header_options = options.header_options;
    header_options.max_file_bytes = expected.file_size;
    const io::safetensors::Result<io::safetensors::Header> parsed_header =
        io::safetensors::parse_header_text(
            header_text, expected.file_size, header_options);
    if (!parsed_header) {
      return header_failure(parsed_header.error);
    }

    std::uint64_t selected_bytes = 0U;
    ReaderResult range_validation =
        validate_ranges(
            *parsed_header.value, selected_ranges, options, selected_bytes);
    if (!range_validation.error.ok()) {
      return range_validation;
    }

    std::vector<std::uint64_t> delivered(selected_ranges.size(), 0U);
    std::vector<std::uint8_t> buffer(options.chunk_bytes);
    const core::Sha256Digest inventory_digest =
        range_inventory_digest(selected_ranges);

    SinkAbortGuard abort_guard(sink);
    abort_guard.mark_begin_invoked();
    if (!sink.begin(
            {expected.file_size, selected_bytes, selected_ranges.size()})) {
      return failure(ReaderErrorCode::kSinkBeginFailure);
    }

    std::size_t next_range = 0U;
    std::size_t delivery_calls = 0U;
    while (reader.total_bytes() < expected.file_size) {
      const std::uint64_t chunk_begin = reader.total_bytes();
      std::size_t chunk_bytes = 0U;
      const SequentialHasher::ReadState state =
          reader.read_some(buffer.data(), buffer.size(), chunk_bytes);
      if (state == SequentialHasher::ReadState::kEof) {
        return failure(ReaderErrorCode::kUnexpectedEof);
      }
      if (state == SequentialHasher::ReadState::kError) {
        return failure(ReaderErrorCode::kIoFailure,
                       static_cast<std::size_t>(-1),
                       reader.error());
      }
      const std::uint64_t chunk_end = reader.total_bytes();
      while (next_range < selected_ranges.size() &&
             selected_ranges[next_range].file_end <= chunk_begin) {
        ++next_range;
      }
      for (std::size_t index = next_range;
           index < selected_ranges.size() &&
           selected_ranges[index].file_begin < chunk_end;
           ++index) {
        const TensorRangeRequest& range = selected_ranges[index];
        const std::uint64_t intersection_begin =
            std::max(chunk_begin, range.file_begin);
        const std::uint64_t intersection_end =
            std::min(chunk_end, range.file_end);
        if (intersection_begin >= intersection_end) {
          continue;
        }
        const std::size_t buffer_offset = static_cast<std::size_t>(
            intersection_begin - chunk_begin);
        const std::size_t intersection_bytes = static_cast<std::size_t>(
            intersection_end - intersection_begin);
        const SinkConsumeResult consumed = sink.consume(
            index,
            intersection_begin,
            intersection_begin - range.file_begin,
            buffer.data() + buffer_offset,
            intersection_bytes);
        ++delivery_calls;
        if (!consumed.accepted) {
          return failure(ReaderErrorCode::kSinkFailure, index);
        }
        if (consumed.consumed_bytes != intersection_bytes) {
          return failure(ReaderErrorCode::kSinkShortDelivery, index);
        }
        if (!checked_add(delivered[index],
                         static_cast<std::uint64_t>(intersection_bytes),
                         delivered[index])) {
          return failure(ReaderErrorCode::kArithmeticOverflow, index);
        }
      }
      (void)chunk_bytes;
    }

    if (!reader.require_eof()) {
      return failure(reader.trailing_data()
                         ? ReaderErrorCode::kUnexpectedTrailingData
                         : ReaderErrorCode::kIoFailure,
                     static_cast<std::size_t>(-1),
                     reader.error());
    }
    for (std::size_t index = 0U; index < selected_ranges.size(); ++index) {
      if (delivered[index] !=
          selected_ranges[index].file_end -
              selected_ranges[index].file_begin) {
        return failure(ReaderErrorCode::kRangeDeliveryIncomplete, index);
      }
    }

    const core::Sha256Digest full_digest = reader.finalize();
    struct stat post_status {};
    if (::fstat(shard_descriptor.get(), &post_status) != 0) {
      return failure(ReaderErrorCode::kStatFailed,
                     static_cast<std::size_t>(-1),
                     errno);
    }
    if (!S_ISREG(post_status.st_mode) ||
        !same_file_identity(pre_identity, file_identity(post_status))) {
      return failure(ReaderErrorCode::kShardMutated);
    }
    if (!(full_digest == expected.full_sha256)) {
      return failure(ReaderErrorCode::kShardHashMismatch);
    }

    ReaderReceipt receipt;
    std::copy(expected.filename.begin(),
              expected.filename.end(),
              receipt.shard_filename.begin());
    receipt.shard_filename_size = expected.filename.size();
    receipt.expected_file_size = expected.file_size;
    receipt.expected_full_sha256 = expected.full_sha256;
    receipt.file_size = expected.file_size;
    receipt.bytes_read = reader.total_bytes();
    receipt.selected_bytes = selected_bytes;
    receipt.selected_range_count = selected_ranges.size();
    receipt.sequential_read_calls = reader.read_calls();
    receipt.sink_delivery_calls = delivery_calls;
    receipt.full_sha256 = full_digest;
    receipt.range_inventory_sha256 = inventory_digest;
    if (!sink.commit()) {
      return failure(ReaderErrorCode::kSinkCommitFailure);
    }
    abort_guard.mark_committed();
    receipt.committed = true;
    receipt.receipt_sha256 = sign_receipt(receipt);

    ReaderResult result;
    result.receipt.emplace(receipt);
    return result;
  } catch (const std::bad_alloc&) {
    return failure(ReaderErrorCode::kAllocationFailure);
  } catch (const std::length_error&) {
    return failure(ReaderErrorCode::kAllocationFailure);
  } catch (...) {
    return failure(ReaderErrorCode::kInternalFailure);
  }
}

[[nodiscard]] bool digest_nonzero(const core::Sha256Digest& digest) noexcept {
  return std::any_of(digest.bytes.begin(),
                     digest.bytes.end(),
                     [](const std::uint8_t byte) { return byte != 0U; });
}

class ScopedTestDirectory {
 public:
  ScopedTestDirectory() noexcept {
    try {
      const std::filesystem::path current = std::filesystem::current_path();
      for (std::size_t attempt = 0U; attempt < 64U; ++attempt) {
        path_ = current /
                (".q3x-sm87-aot-reader-self-test-" +
                 std::to_string(static_cast<unsigned long long>(::getpid())) +
                 "-" + std::to_string(attempt));
        std::error_code error;
        if (std::filesystem::create_directory(path_, error)) {
          valid_ = true;
          return;
        }
        if (error && error != std::errc::file_exists) {
          path_.clear();
          return;
        }
      }
      path_.clear();
    } catch (...) {
      path_.clear();
    }
  }

  ~ScopedTestDirectory() {
    if (!path_.empty()) {
      std::error_code ignored;
      (void)std::filesystem::remove_all(path_, ignored);
    }
  }

  [[nodiscard]] bool valid() const noexcept { return valid_; }
  [[nodiscard]] const std::filesystem::path& path() const noexcept {
    return path_;
  }

 private:
  std::filesystem::path path_;
  bool valid_ = false;
};

[[nodiscard]] bool write_all(const std::filesystem::path& path,
                             const std::string_view bytes) noexcept {
  const UniqueFd descriptor(
      ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600));
  if (descriptor.get() < 0) {
    return false;
  }
  std::size_t offset = 0U;
  while (offset < bytes.size()) {
    ssize_t count = -1;
    do {
      count = ::write(descriptor.get(),
                      bytes.data() + offset,
                      bytes.size() - offset);
    } while (count < 0 && errno == EINTR);
    if (count <= 0) {
      return false;
    }
    offset += static_cast<std::size_t>(count);
  }
  return true;
}

[[nodiscard]] std::string make_safetensors_bytes(
    const std::string_view header,
    const std::string_view payload) {
  std::string bytes(8U, '\0');
  const std::uint64_t header_size =
      static_cast<std::uint64_t>(header.size());
  for (std::size_t index = 0U; index < 8U; ++index) {
    bytes[index] = static_cast<char>(header_size >> (index * 8U));
  }
  bytes.append(header.data(), header.size());
  bytes.append(payload.data(), payload.size());
  return bytes;
}

struct SyntheticFixture {
  std::string filename;
  std::string bytes;
  std::uint64_t data_offset = 0U;
  PinnedShardIdentity identity{};
};

[[nodiscard]] SyntheticFixture make_valid_fixture() {
  constexpr std::string_view header =
      R"({"alpha":{"dtype":"U8","shape":[5],"data_offsets":[0,5]},"beta":{"dtype":"BF16","shape":[3],"data_offsets":[5,11]},"gamma":{"dtype":"U8","shape":[4],"data_offsets":[11,15]}})";
  const std::array<std::uint8_t, 15U> payload = {
      0x10U, 0x11U, 0x12U, 0x13U, 0x14U,
      0x20U, 0x21U, 0x22U, 0x23U, 0x24U, 0x25U,
      0x30U, 0x31U, 0x32U, 0x33U};
  SyntheticFixture fixture;
  fixture.filename = "fixture.safetensors";
  fixture.bytes = make_safetensors_bytes(
      header,
      std::string_view(reinterpret_cast<const char*>(payload.data()),
                       payload.size()));
  fixture.data_offset = 8U + static_cast<std::uint64_t>(header.size());
  fixture.identity = {fixture.filename,
                      static_cast<std::uint64_t>(fixture.bytes.size()),
                      core::sha256(fixture.bytes)};
  return fixture;
}

[[nodiscard]] TensorRangeRequest alpha_range(
    const SyntheticFixture& fixture) {
  return {"alpha",
          io::safetensors::DType::kU8,
          {5U},
          fixture.data_offset,
          fixture.data_offset + 5U};
}

[[nodiscard]] TensorRangeRequest beta_range(
    const SyntheticFixture& fixture) {
  return {"beta",
          io::safetensors::DType::kBf16,
          {3U},
          fixture.data_offset + 5U,
          fixture.data_offset + 11U};
}

[[nodiscard]] TensorRangeRequest gamma_range(
    const SyntheticFixture& fixture) {
  return {"gamma",
          io::safetensors::DType::kU8,
          {4U},
          fixture.data_offset + 11U,
          fixture.data_offset + 15U};
}

class RecordingSink final : public ProvisionalRangeSink {
 public:
  enum class Mode : std::uint8_t {
    kNormal,
    kBeginFail,
    kShort,
    kFail,
    kCommitFail,
  };
  using Mutation = bool (*)(void*) noexcept;

  RecordingSink(std::vector<std::vector<std::uint8_t>> expected,
                const Mode mode = Mode::kNormal,
                Mutation mutation = nullptr,
                void* mutation_context = nullptr)
      : expected_(std::move(expected)),
        mode_(mode),
        mutation_(mutation),
        mutation_context_(mutation_context) {}

  [[nodiscard]] bool begin(const SinkTransaction& transaction) override {
    ++begin_calls;
    transaction_ = transaction;
    provisional_.assign(expected_.size(), {});
    begun_ = true;
    return mode_ != Mode::kBeginFail;
  }

  [[nodiscard]] SinkConsumeResult consume(
      const std::size_t range_index,
      const std::uint64_t,
      const std::uint64_t,
      const std::uint8_t* const bytes,
      const std::size_t byte_count) override {
    ++consume_calls;
    if (!begun_ || range_index >= provisional_.size() || bytes == nullptr) {
      return {};
    }
    if (!mutation_attempted_ && mutation_ != nullptr) {
      mutation_attempted_ = true;
      mutation_succeeded_ = mutation_(mutation_context_);
      if (!mutation_succeeded_) {
        return {};
      }
    }
    if (mode_ == Mode::kFail) {
      return {};
    }
    if (mode_ == Mode::kShort) {
      mode_ = Mode::kNormal;
      return {true, byte_count == 0U ? 0U : byte_count - 1U};
    }
    provisional_[range_index].insert(
        provisional_[range_index].end(), bytes, bytes + byte_count);
    delivered_bytes += byte_count;
    return {true, byte_count};
  }

  [[nodiscard]] bool commit() override {
    ++commit_calls;
    if (mode_ == Mode::kCommitFail || !begun_ ||
        provisional_ != expected_) {
      return false;
    }
    committed_ = true;
    return true;
  }

  void abort() noexcept override {
    ++abort_calls;
    bytes_before_abort = delivered_bytes;
    provisional_matched_before_abort = provisional_ == expected_;
    for (auto& bytes : provisional_) {
      bytes.clear();
    }
    committed_ = false;
  }

  [[nodiscard]] bool provisional_empty() const noexcept {
    return std::all_of(provisional_.begin(),
                       provisional_.end(),
                       [](const auto& bytes) { return bytes.empty(); });
  }

  SinkTransaction transaction_{};
  std::size_t begin_calls = 0U;
  std::size_t consume_calls = 0U;
  std::size_t commit_calls = 0U;
  std::size_t abort_calls = 0U;
  std::size_t delivered_bytes = 0U;
  std::size_t bytes_before_abort = 0U;
  bool mutation_succeeded_ = false;
  bool provisional_matched_before_abort = false;

 private:
  std::vector<std::vector<std::uint8_t>> expected_;
  std::vector<std::vector<std::uint8_t>> provisional_;
  Mode mode_ = Mode::kNormal;
  Mutation mutation_ = nullptr;
  void* mutation_context_ = nullptr;
  bool begun_ = false;
  bool committed_ = false;
  bool mutation_attempted_ = false;
};

struct TestState {
  void expect(const bool condition, const char* const message) noexcept {
    if (!condition) {
      passed = false;
      std::fprintf(stderr, "checkpoint reader self-test failed: %s\n", message);
    }
  }
  bool passed = true;
};

[[nodiscard]] std::vector<std::uint8_t> bytes(
    std::initializer_list<std::uint8_t> values) {
  return {values.begin(), values.end()};
}

struct MutationContext {
  std::filesystem::path path;
};

struct RenameReplaceContext {
  std::filesystem::path path;
  std::filesystem::path moved_path;
  std::string replacement_bytes;
};

struct ResizeContext {
  std::filesystem::path path;
  off_t size = 0;
};

struct ByteMutationContext {
  std::filesystem::path path;
  off_t offset = 0;
  std::uint8_t value = 0U;
};

[[nodiscard]] bool mutate_mtime(void* const opaque) noexcept {
  auto* const context = static_cast<MutationContext*>(opaque);
  struct stat status {};
  if (context == nullptr || ::stat(context->path.c_str(), &status) != 0) {
    return false;
  }
  std::array<timespec, 2U> times = {status.st_atim, status.st_mtim};
  if (times[1U].tv_sec == std::numeric_limits<time_t>::max()) {
    --times[1U].tv_sec;
  } else {
    ++times[1U].tv_sec;
  }
  return ::utimensat(AT_FDCWD, context->path.c_str(), times.data(), 0) == 0;
}

[[nodiscard]] bool rename_and_replace(void* const opaque) noexcept {
  auto* const context = static_cast<RenameReplaceContext*>(opaque);
  if (context == nullptr ||
      ::rename(context->path.c_str(), context->moved_path.c_str()) != 0) {
    return false;
  }
  return write_all(context->path, context->replacement_bytes);
}

[[nodiscard]] bool truncate_file(void* const opaque) noexcept {
  auto* const context = static_cast<ResizeContext*>(opaque);
  return context != nullptr &&
         ::truncate(context->path.c_str(), context->size) == 0;
}

[[nodiscard]] bool append_byte(void* const opaque) noexcept {
  auto* const context = static_cast<MutationContext*>(opaque);
  if (context == nullptr) {
    return false;
  }
  const UniqueFd descriptor(
      ::open(context->path.c_str(), O_WRONLY | O_APPEND | O_CLOEXEC));
  if (descriptor.get() < 0) {
    return false;
  }
  constexpr std::uint8_t byte = 0xa5U;
  ssize_t count = -1;
  do {
    count = ::write(descriptor.get(), &byte, sizeof(byte));
  } while (count < 0 && errno == EINTR);
  return count == static_cast<ssize_t>(sizeof(byte));
}

[[nodiscard]] bool overwrite_byte(void* const opaque) noexcept {
  auto* const context = static_cast<ByteMutationContext*>(opaque);
  if (context == nullptr) {
    return false;
  }
  const UniqueFd descriptor(
      ::open(context->path.c_str(), O_WRONLY | O_CLOEXEC));
  if (descriptor.get() < 0) {
    return false;
  }
  ssize_t count = -1;
  do {
    count = ::pwrite(descriptor.get(),
                     &context->value,
                     sizeof(context->value),
                     context->offset);
  } while (count < 0 && errno == EINTR);
  return count == static_cast<ssize_t>(sizeof(context->value));
}

[[nodiscard]] bool restore_fixture(const std::filesystem::path& path,
                                   const std::string_view bytes) noexcept {
  return ::unlink(path.c_str()) == 0 && write_all(path, bytes);
}

}  // namespace

std::optional<PinnedShardIdentity> pinned_shard_identity(
    const std::string_view filename) noexcept {
  for (const PinnedShardIdentity& identity : kPinnedShards) {
    if (identity.filename == filename) {
      return identity;
    }
  }
  return std::nullopt;
}

ReaderResult read_pinned_shard_ranges(
    const std::filesystem::path& checkpoint_root,
    const std::string_view pinned_shard_filename,
    const std::vector<TensorRangeRequest>& ranges,
    ProvisionalRangeSink& sink,
    const ReaderOptions& options) {
  if (!is_safe_single_shard_filename(pinned_shard_filename)) {
    return failure(ReaderErrorCode::kUnsafeShardFilename);
  }
  const std::optional<PinnedShardIdentity> identity =
      pinned_shard_identity(pinned_shard_filename);
  if (!identity.has_value()) {
    return failure(ReaderErrorCode::kUnknownPinnedShard);
  }
  return read_authenticated_ranges(
      checkpoint_root, *identity, ranges, sink, options);
}

std::string_view to_string(const ReaderErrorCode code) noexcept {
  switch (code) {
    case ReaderErrorCode::kNone:
      return "authenticated ranges committed";
    case ReaderErrorCode::kInvalidOption:
      return "invalid reader option";
    case ReaderErrorCode::kUnsafeShardFilename:
      return "unsafe shard filename";
    case ReaderErrorCode::kUnknownPinnedShard:
      return "shard is not in the frozen checkpoint identity";
    case ReaderErrorCode::kRootOpenFailed:
      return "failed to open checkpoint root";
    case ReaderErrorCode::kShardOpenFailed:
      return "failed to open non-symlink shard";
    case ReaderErrorCode::kStatFailed:
      return "failed to inspect shard identity";
    case ReaderErrorCode::kShardNotRegular:
      return "shard is not a regular file";
    case ReaderErrorCode::kShardSizeMismatch:
      return "shard size does not match the frozen identity";
    case ReaderErrorCode::kIoFailure:
      return "sequential shard read failed";
    case ReaderErrorCode::kUnexpectedEof:
      return "shard ended before its frozen size";
    case ReaderErrorCode::kUnexpectedTrailingData:
      return "shard grew beyond its frozen size";
    case ReaderErrorCode::kInvalidPrefix:
      return "invalid safetensors prefix";
    case ReaderErrorCode::kHeaderInvalid:
      return "invalid safetensors header";
    case ReaderErrorCode::kTooManyRanges:
      return "selected range count exceeds the reader limit";
    case ReaderErrorCode::kInvalidRange:
      return "selected range is invalid";
    case ReaderErrorCode::kDuplicateRange:
      return "selected tensor is duplicated";
    case ReaderErrorCode::kRangesNotOrdered:
      return "selected ranges are not ordered";
    case ReaderErrorCode::kRangesOverlap:
      return "selected ranges overlap";
    case ReaderErrorCode::kMissingTensor:
      return "selected tensor is missing from the same-fd header";
    case ReaderErrorCode::kTensorMetadataMismatch:
      return "selected tensor metadata disagrees with the same-fd header";
    case ReaderErrorCode::kArithmeticOverflow:
      return "range accounting overflow";
    case ReaderErrorCode::kSinkBeginFailure:
      return "provisional sink begin failed";
    case ReaderErrorCode::kSinkFailure:
      return "provisional sink rejected bytes";
    case ReaderErrorCode::kSinkShortDelivery:
      return "provisional sink accepted a short delivery";
    case ReaderErrorCode::kRangeDeliveryIncomplete:
      return "selected range delivery is incomplete";
    case ReaderErrorCode::kShardHashMismatch:
      return "full shard SHA-256 mismatch";
    case ReaderErrorCode::kShardMutated:
      return "shard identity changed during the read";
    case ReaderErrorCode::kSinkCommitFailure:
      return "provisional sink commit failed";
    case ReaderErrorCode::kAllocationFailure:
      return "reader allocation failed";
    case ReaderErrorCode::kInternalFailure:
      return "reader or sink raised an unexpected failure";
  }
  return "unknown checkpoint reader error";
}

bool run_checkpoint_reader_self_test() noexcept {
  try {
    TestState test;
    ScopedTestDirectory directory;
    test.expect(directory.valid(), "workspace-local RAII directory is created");
    if (!directory.valid()) {
      return false;
    }

    SyntheticFixture fixture = make_valid_fixture();
    const std::filesystem::path fixture_path =
        directory.path() / fixture.filename;
    test.expect(write_all(fixture_path, fixture.bytes),
                "valid safetensors fixture is written");

    ReaderOptions options;
    options.chunk_bytes = 3U;
    options.max_selected_ranges = 16U;
    options.header_options.max_header_bytes = 4096U;
    options.header_options.max_tensors = 16U;
    options.header_options.max_rank = 8U;
    options.header_options.json_options.max_input_bytes = 4096U;
    options.header_options.json_options.max_values = 128U;
    options.header_options.json_options.max_container_items = 128U;

    const std::vector<TensorRangeRequest> multi_ranges = {
        alpha_range(fixture), gamma_range(fixture)};
    {
      RecordingSink sink({bytes({0x10U, 0x11U, 0x12U, 0x13U, 0x14U}),
                          bytes({0x30U, 0x31U, 0x32U, 0x33U})});
      const ReaderResult result = read_authenticated_ranges(
          directory.path(), fixture.identity, multi_ranges, sink, options);
      test.expect(result.ok(), "valid multi-range small-chunk read commits");
      test.expect(result.receipt.has_value() &&
                      result.receipt->shard_filename_size ==
                          fixture.filename.size() &&
                      std::string_view(result.receipt->shard_filename.data(),
                                       result.receipt->shard_filename_size) ==
                          fixture.filename &&
                      result.receipt->expected_file_size ==
                          fixture.identity.file_size &&
                      result.receipt->expected_full_sha256 ==
                          fixture.identity.full_sha256 &&
                      result.receipt->bytes_read == fixture.bytes.size() &&
                      result.receipt->selected_bytes == 9U &&
                      result.receipt->selected_range_count == 2U &&
                      result.receipt->sink_delivery_calls >= 4U &&
                      result.receipt->full_sha256 ==
                          fixture.identity.full_sha256 &&
                      digest_nonzero(result.receipt->range_inventory_sha256) &&
                      digest_nonzero(result.receipt->receipt_sha256),
                  "valid receipt binds full file and selected ranges");
      test.expect(sink.begin_calls == 1U && sink.commit_calls == 1U &&
                      sink.abort_calls == 0U,
                  "valid transaction commits exactly once");
    }

    {
      const std::filesystem::path link_path =
          directory.path() / "link.safetensors";
      test.expect(::symlink(fixture.filename.c_str(), link_path.c_str()) == 0,
                  "symlink fixture is created");
      PinnedShardIdentity link_identity = fixture.identity;
      link_identity.filename = "link.safetensors";
      RecordingSink sink({bytes({0x10U, 0x11U, 0x12U, 0x13U, 0x14U})});
      const ReaderResult result = read_authenticated_ranges(
          directory.path(), link_identity, {alpha_range(fixture)}, sink, options);
      test.expect(result.error.code == ReaderErrorCode::kShardOpenFailed &&
                      sink.begin_calls == 0U,
                  "O_NOFOLLOW rejects a shard symlink before begin");
    }

    {
      const std::string fifo_name = "fifo.safetensors";
      const std::filesystem::path fifo_path = directory.path() / fifo_name;
      test.expect(::mkfifo(fifo_path.c_str(), 0600) == 0,
                  "FIFO shard fixture is created");
      PinnedShardIdentity fifo_identity{};
      fifo_identity.filename = fifo_name;
      RecordingSink sink({});
      const ReaderResult result = read_authenticated_ranges(
          directory.path(), fifo_identity, {alpha_range(fixture)}, sink, options);
      test.expect(result.error.code == ReaderErrorCode::kShardNotRegular &&
                      sink.begin_calls == 0U,
                  "O_NONBLOCK reaches fstat and rejects FIFO without blocking");
    }

    {
      PinnedShardIdentity unsafe_identity = fixture.identity;
      unsafe_identity.filename = "nested/fixture.safetensors";
      RecordingSink sink({bytes({0x10U, 0x11U, 0x12U, 0x13U, 0x14U})});
      const ReaderResult result = read_authenticated_ranges(
          directory.path(), unsafe_identity, {alpha_range(fixture)}, sink, options);
      test.expect(result.error.code ==
                          ReaderErrorCode::kUnsafeShardFilename &&
                      sink.begin_calls == 0U,
                  "unsafe multi-component path is rejected");
    }

    {
      PinnedShardIdentity wrong_size = fixture.identity;
      ++wrong_size.file_size;
      RecordingSink sink({bytes({0x10U, 0x11U, 0x12U, 0x13U, 0x14U})});
      const ReaderResult result = read_authenticated_ranges(
          directory.path(), wrong_size, {alpha_range(fixture)}, sink, options);
      test.expect(result.error.code == ReaderErrorCode::kShardSizeMismatch &&
                      sink.begin_calls == 0U,
                  "wrong frozen shard size is rejected before delivery");
    }

    {
      PinnedShardIdentity wrong_hash = fixture.identity;
      wrong_hash.full_sha256.bytes[0U] ^= 0x80U;
      RecordingSink sink({bytes({0x10U, 0x11U, 0x12U, 0x13U, 0x14U})});
      const ReaderResult result = read_authenticated_ranges(
          directory.path(), wrong_hash, {alpha_range(fixture)}, sink, options);
      test.expect(result.error.code == ReaderErrorCode::kShardHashMismatch &&
                      sink.commit_calls == 0U && sink.abort_calls == 1U,
                  "wrong full shard hash aborts the provisional sink");
    }

    {
      constexpr std::string_view malformed_header = R"({"broken":)";
      const std::string malformed_bytes =
          make_safetensors_bytes(malformed_header, {});
      const std::string malformed_name = "malformed.safetensors";
      const std::filesystem::path malformed_path =
          directory.path() / malformed_name;
      test.expect(write_all(malformed_path, malformed_bytes),
                  "malformed header fixture is written");
      const PinnedShardIdentity malformed_identity = {
          malformed_name,
          static_cast<std::uint64_t>(malformed_bytes.size()),
          core::sha256(malformed_bytes)};
      RecordingSink sink({});
      const ReaderResult result = read_authenticated_ranges(
          directory.path(), malformed_identity, {alpha_range(fixture)}, sink, options);
      test.expect(result.error.code == ReaderErrorCode::kHeaderInvalid &&
                      sink.begin_calls == 0U,
                  "malformed same-fd header is rejected before begin");
    }

    {
      TensorRangeRequest mismatch = alpha_range(fixture);
      mismatch.dtype = io::safetensors::DType::kBf16;
      RecordingSink sink({bytes({0x10U, 0x11U, 0x12U, 0x13U, 0x14U})});
      const ReaderResult result = read_authenticated_ranges(
          directory.path(), fixture.identity, {mismatch}, sink, options);
      test.expect(result.error.code ==
                          ReaderErrorCode::kTensorMetadataMismatch &&
                      sink.begin_calls == 0U,
                  "dtype metadata mismatch is rejected");
    }

    {
      TensorRangeRequest overlap = beta_range(fixture);
      overlap.file_begin = fixture.data_offset + 3U;
      RecordingSink sink({});
      const ReaderResult result = read_authenticated_ranges(
          directory.path(), fixture.identity,
          {alpha_range(fixture), overlap}, sink, options);
      test.expect(result.error.code == ReaderErrorCode::kRangesOverlap &&
                      sink.begin_calls == 0U,
                  "overlapping selected ranges are rejected");
    }

    {
      const TensorRangeRequest alpha = alpha_range(fixture);
      RecordingSink sink({});
      const ReaderResult result = read_authenticated_ranges(
          directory.path(), fixture.identity, {alpha, alpha}, sink, options);
      test.expect(result.error.code == ReaderErrorCode::kDuplicateRange &&
                      sink.begin_calls == 0U,
                  "duplicate selected tensor is rejected");
    }

    {
      TensorRangeRequest missing = alpha_range(fixture);
      missing.tensor_name = "missing";
      RecordingSink sink({});
      const ReaderResult result = read_authenticated_ranges(
          directory.path(), fixture.identity, {missing}, sink, options);
      test.expect(result.error.code == ReaderErrorCode::kMissingTensor &&
                      sink.begin_calls == 0U,
                  "missing selected tensor is rejected");
    }

    {
      RecordingSink sink(
          {bytes({0x10U, 0x11U, 0x12U, 0x13U, 0x14U})},
          RecordingSink::Mode::kBeginFail);
      const ReaderResult result = read_authenticated_ranges(
          directory.path(), fixture.identity, {alpha_range(fixture)}, sink, options);
      test.expect(result.error.code == ReaderErrorCode::kSinkBeginFailure &&
                      sink.begin_calls == 1U && sink.commit_calls == 0U &&
                      sink.abort_calls == 1U && !result.receipt.has_value(),
                  "begin failure receives exactly one abort and no receipt");
    }

    {
      RecordingSink sink(
          {bytes({0x10U, 0x11U, 0x12U, 0x13U, 0x14U})},
          RecordingSink::Mode::kShort);
      const ReaderResult result = read_authenticated_ranges(
          directory.path(), fixture.identity, {alpha_range(fixture)}, sink, options);
      test.expect(result.error.code == ReaderErrorCode::kSinkShortDelivery &&
                      sink.commit_calls == 0U && sink.abort_calls == 1U &&
                      sink.provisional_empty(),
                  "short sink delivery aborts without a receipt");
    }

    {
      RecordingSink sink(
          {bytes({0x10U, 0x11U, 0x12U, 0x13U, 0x14U})},
          RecordingSink::Mode::kFail);
      const ReaderResult result = read_authenticated_ranges(
          directory.path(), fixture.identity, {alpha_range(fixture)}, sink, options);
      test.expect(result.error.code == ReaderErrorCode::kSinkFailure &&
                      sink.commit_calls == 0U && sink.abort_calls == 1U &&
                      sink.provisional_empty(),
                  "sink failure aborts without a receipt");
    }

    {
      RecordingSink sink(
          {bytes({0x10U, 0x11U, 0x12U, 0x13U, 0x14U})},
          RecordingSink::Mode::kCommitFail);
      const ReaderResult result = read_authenticated_ranges(
          directory.path(), fixture.identity, {alpha_range(fixture)}, sink, options);
      test.expect(result.error.code == ReaderErrorCode::kSinkCommitFailure &&
                      sink.begin_calls == 1U && sink.commit_calls == 1U &&
                      sink.abort_calls == 1U && sink.provisional_empty() &&
                      !result.receipt.has_value(),
                  "commit failure receives exactly one abort and no receipt");
    }

    {
      const std::filesystem::path moved_path =
          directory.path() / "fixture-open-fd.safetensors";
      RenameReplaceContext replacement{
          fixture_path,
          moved_path,
          std::string(fixture.bytes.size(), static_cast<char>(0x5a))};
      RecordingSink sink(
          {bytes({0x10U, 0x11U, 0x12U, 0x13U, 0x14U})},
          RecordingSink::Mode::kNormal,
          rename_and_replace,
          &replacement);
      const ReaderResult result = read_authenticated_ranges(
          directory.path(), fixture.identity, {alpha_range(fixture)}, sink, options);
      const bool old_fd_preserved =
          sink.mutation_succeeded_ && sink.delivered_bytes == 5U &&
          ((result.ok() && sink.commit_calls == 1U &&
            sink.abort_calls == 0U) ||
           (result.error.code == ReaderErrorCode::kShardMutated &&
            sink.bytes_before_abort == 5U &&
            sink.provisional_matched_before_abort &&
            sink.commit_calls == 0U && sink.abort_calls == 1U));
      test.expect(old_fd_preserved,
                  "path replacement cannot redirect the open fd and post-stat is applied");
      test.expect(::unlink(fixture_path.c_str()) == 0 &&
                      ::rename(moved_path.c_str(), fixture_path.c_str()) == 0,
                  "rename replacement fixture is restored");
    }

    {
      ResizeContext truncation{
          fixture_path,
          static_cast<off_t>(fixture.data_offset + options.chunk_bytes)};
      RecordingSink sink(
          {bytes({0x10U, 0x11U, 0x12U, 0x13U, 0x14U})},
          RecordingSink::Mode::kNormal,
          truncate_file,
          &truncation);
      const ReaderResult result = read_authenticated_ranges(
          directory.path(), fixture.identity, {alpha_range(fixture)}, sink, options);
      test.expect(sink.mutation_succeeded_ &&
                      result.error.code == ReaderErrorCode::kUnexpectedEof &&
                      sink.commit_calls == 0U && sink.abort_calls == 1U,
                  "read-time truncation aborts on unexpected EOF");
      test.expect(restore_fixture(fixture_path, fixture.bytes),
                  "truncated fixture is restored");
    }

    {
      MutationContext growth{fixture_path};
      RecordingSink sink(
          {bytes({0x10U, 0x11U, 0x12U, 0x13U, 0x14U})},
          RecordingSink::Mode::kNormal,
          append_byte,
          &growth);
      const ReaderResult result = read_authenticated_ranges(
          directory.path(), fixture.identity, {alpha_range(fixture)}, sink, options);
      test.expect(sink.mutation_succeeded_ &&
                      result.error.code ==
                          ReaderErrorCode::kUnexpectedTrailingData &&
                      sink.commit_calls == 0U && sink.abort_calls == 1U,
                  "read-time append aborts at the mandatory EOF read");
      test.expect(restore_fixture(fixture_path, fixture.bytes),
                  "grown fixture is restored");
    }

    {
      ByteMutationContext mutation{
          fixture_path,
          static_cast<off_t>(fixture.bytes.size() - 1U),
          0x7eU};
      RecordingSink sink(
          {bytes({0x10U, 0x11U, 0x12U, 0x13U, 0x14U})},
          RecordingSink::Mode::kNormal,
          overwrite_byte,
          &mutation);
      const ReaderResult result = read_authenticated_ranges(
          directory.path(), fixture.identity, {alpha_range(fixture)}, sink, options);
      test.expect(sink.mutation_succeeded_ &&
                      (result.error.code == ReaderErrorCode::kShardMutated ||
                       result.error.code ==
                           ReaderErrorCode::kShardHashMismatch) &&
                      sink.commit_calls == 0U && sink.abort_calls == 1U,
                  "same-size byte mutation fails closed after delivery");
      test.expect(restore_fixture(fixture_path, fixture.bytes),
                  "same-size mutated fixture is restored");
    }

    {
      MutationContext mutation{fixture_path};
      RecordingSink sink(
          {bytes({0x10U, 0x11U, 0x12U, 0x13U, 0x14U})},
          RecordingSink::Mode::kNormal,
          mutate_mtime,
          &mutation);
      const ReaderResult result = read_authenticated_ranges(
          directory.path(), fixture.identity, {alpha_range(fixture)}, sink, options);
      test.expect(sink.mutation_succeeded_ &&
                      result.error.code == ReaderErrorCode::kShardMutated &&
                      sink.commit_calls == 0U && sink.abort_calls == 1U &&
                      sink.provisional_empty(),
                  "read-time metadata mutation fails post-fstat and aborts");

      struct stat status {};
      if (::stat(fixture_path.c_str(), &status) == 0) {
        std::array<timespec, 2U> restore = {status.st_atim, status.st_mtim};
        if (restore[1U].tv_sec == std::numeric_limits<time_t>::min()) {
          ++restore[1U].tv_sec;
        } else {
          --restore[1U].tv_sec;
        }
        (void)::utimensat(AT_FDCWD, fixture_path.c_str(), restore.data(), 0);
      }
    }

    {
      PinnedShardIdentity late_hash = fixture.identity;
      late_hash.full_sha256.bytes[31U] ^= 0x01U;
      RecordingSink sink({bytes({0x10U, 0x11U, 0x12U, 0x13U, 0x14U}),
                          bytes({0x30U, 0x31U, 0x32U, 0x33U})});
      const ReaderResult result = read_authenticated_ranges(
          directory.path(), late_hash, multi_ranges, sink, options);
      test.expect(result.error.code == ReaderErrorCode::kShardHashMismatch &&
                      sink.bytes_before_abort == 9U &&
                      sink.commit_calls == 0U && sink.abort_calls == 1U &&
                      sink.provisional_empty() && !result.receipt.has_value(),
                  "late full-hash failure aborts fully delivered provisional bytes");
    }

    return test.passed;
  } catch (...) {
    return false;
  }
}

}  // namespace q3x::test::sm87_aot_checkpoint_reader
