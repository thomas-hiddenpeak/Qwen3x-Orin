#include "q3x/runtime/prefill_mlp_factorized_lane_r4_candidate_converter.h"

#include "q3x/core/sha256.h"
#include "q3x/io/safetensors.h"
#include "q3x/model/weight_manifest.h"
#include "q3x/runtime/prefill_mlp_factorized_lane_r4_converter.h"
#include "q3x/runtime/resident_weights.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace q3x::runtime {
namespace {

namespace fs = std::filesystem;
namespace mw = q3x::model::weights;
namespace st = q3x::io::safetensors;

using ErrorCode = PrefillMLPFactorizedLaneR4CandidateConverterErrorCode;
using Diagnostic = PrefillMLPFactorizedLaneR4CandidateConverterDiagnostic;
using Stats = PrefillMLPFactorizedLaneR4IdentityCandidateConversionStats;

constexpr std::uint64_t kExpectedSourceBytes = 9'625'928'448ULL;
constexpr std::size_t kMaximumRowChunk = 256U;

[[nodiscard]] Diagnostic make_diagnostic(
    const ErrorCode code, std::string context, std::string message,
    std::string expected = {}, std::string actual = {},
    const int system_error = 0) {
  Diagnostic result;
  result.code = code;
  result.context = std::move(context);
  result.message = std::move(message);
  result.expected = std::move(expected);
  result.actual = std::move(actual);
  result.system_error = system_error;
  return result;
}

[[nodiscard]] bool checked_add(const std::uint64_t left,
                               const std::uint64_t right,
                               std::uint64_t& output) noexcept {
  return prefill_a4_factorized_lane_contract_detail::checked_add(
      left, right, output);
}

[[nodiscard]] bool checked_multiply(const std::uint64_t left,
                                    const std::uint64_t right,
                                    std::uint64_t& output) noexcept {
  return prefill_a4_factorized_lane_contract_detail::checked_multiply(
      left, right, output);
}

[[nodiscard]] bool valid_clip_ratio(const double value) noexcept {
  const float narrowed = static_cast<float>(value);
  return std::isfinite(value) &&
         value >= kPrefillMLPFactorizedLaneR4PublicationMinimumClipRatio &&
         value <= 1.0 && std::isfinite(narrowed) &&
         narrowed >= static_cast<float>(
                         kPrefillMLPFactorizedLaneR4PublicationMinimumClipRatio) &&
         narrowed <= 1.0F;
}

[[nodiscard]] bool valid_row_chunk(const std::size_t value) noexcept {
  return value != 0U && value <= kMaximumRowChunk &&
         value % kPrefillA4FactorizedLaneOuterBlock == 0U;
}

[[nodiscard]] std::string sha256_text(const std::string_view bytes) {
  core::Sha256 hasher;
  if (!hasher.update(bytes.data(), bytes.size())) {
    return {};
  }
  return hasher.finalize().hex();
}

class UniqueFd final {
 public:
  UniqueFd() noexcept = default;
  explicit UniqueFd(const int fd) noexcept : fd_(fd) {}
  ~UniqueFd() {
    if (fd_ >= 0) {
      (void)::close(fd_);
    }
  }
  UniqueFd(const UniqueFd&) = delete;
  UniqueFd& operator=(const UniqueFd&) = delete;
  UniqueFd(UniqueFd&& other) noexcept : fd_(other.fd_) { other.fd_ = -1; }
  UniqueFd& operator=(UniqueFd&& other) noexcept {
    if (this != &other) {
      if (fd_ >= 0) {
        (void)::close(fd_);
      }
      fd_ = other.fd_;
      other.fd_ = -1;
    }
    return *this;
  }
  [[nodiscard]] int get() const noexcept { return fd_; }
  [[nodiscard]] explicit operator bool() const noexcept { return fd_ >= 0; }

 private:
  int fd_ = -1;
};

struct FileSnapshot final {
  std::uint64_t device = 0U;
  std::uint64_t inode = 0U;
  std::uint64_t size = 0U;
  std::int64_t mtime_s = 0;
  std::int64_t mtime_ns = 0;
  std::int64_t ctime_s = 0;
  std::int64_t ctime_ns = 0;
};

[[nodiscard]] bool capture_snapshot(const int fd, FileSnapshot& output,
                                    int& error) noexcept {
  struct stat status {};
  if (::fstat(fd, &status) != 0 || !S_ISREG(status.st_mode) ||
      status.st_size < 0) {
    error = errno != 0 ? errno : EINVAL;
    return false;
  }
  output.device = static_cast<std::uint64_t>(status.st_dev);
  output.inode = static_cast<std::uint64_t>(status.st_ino);
  output.size = static_cast<std::uint64_t>(status.st_size);
  output.mtime_s = status.st_mtim.tv_sec;
  output.mtime_ns = status.st_mtim.tv_nsec;
  output.ctime_s = status.st_ctim.tv_sec;
  output.ctime_ns = status.st_ctim.tv_nsec;
  return true;
}

[[nodiscard]] bool same_snapshot(const FileSnapshot& left,
                                 const FileSnapshot& right) noexcept {
  return left.device == right.device && left.inode == right.inode &&
         left.size == right.size && left.mtime_s == right.mtime_s &&
         left.mtime_ns == right.mtime_ns && left.ctime_s == right.ctime_s &&
         left.ctime_ns == right.ctime_ns;
}

[[nodiscard]] bool offset_fits(const std::uint64_t offset) noexcept {
  return offset <=
         static_cast<std::uint64_t>(std::numeric_limits<off_t>::max());
}

[[nodiscard]] bool pread_exact(const int fd, void* const destination,
                               const std::size_t byte_count,
                               const std::uint64_t offset,
                               int& error) noexcept {
  if (!offset_fits(offset)) {
    error = EOVERFLOW;
    return false;
  }
  auto* output = static_cast<std::uint8_t*>(destination);
  std::size_t completed = 0U;
  while (completed < byte_count) {
    std::uint64_t position = 0U;
    if (!checked_add(offset, completed, position) || !offset_fits(position)) {
      error = EOVERFLOW;
      return false;
    }
    const ssize_t count =
        ::pread(fd, output + completed, byte_count - completed,
                static_cast<off_t>(position));
    if (count < 0) {
      if (errno == EINTR) {
        continue;
      }
      error = errno;
      return false;
    }
    if (count == 0) {
      error = EIO;
      return false;
    }
    completed += static_cast<std::size_t>(count);
  }
  return true;
}

[[nodiscard]] bool pwrite_exact(const int fd, const void* const source,
                                const std::size_t byte_count,
                                const std::uint64_t offset,
                                int& error) noexcept {
  if (!offset_fits(offset)) {
    error = EOVERFLOW;
    return false;
  }
  const auto* input = static_cast<const std::uint8_t*>(source);
  std::size_t completed = 0U;
  while (completed < byte_count) {
    std::uint64_t position = 0U;
    if (!checked_add(offset, completed, position) || !offset_fits(position)) {
      error = EOVERFLOW;
      return false;
    }
    const ssize_t count =
        ::pwrite(fd, input + completed, byte_count - completed,
                 static_cast<off_t>(position));
    if (count < 0) {
      if (errno == EINTR) {
        continue;
      }
      error = errno;
      return false;
    }
    if (count == 0) {
      error = EIO;
      return false;
    }
    completed += static_cast<std::size_t>(count);
  }
  return true;
}

[[nodiscard]] Diagnostic hash_open_file(const int fd,
                                        const std::uint64_t byte_count,
                                        std::string& digest) {
  try {
    constexpr std::size_t kChunk = 8U * 1024U * 1024U;
    std::vector<std::uint8_t> buffer(static_cast<std::size_t>(
        std::min<std::uint64_t>(byte_count, kChunk)));
    core::Sha256 hasher;
    std::uint64_t offset = 0U;
    while (offset < byte_count) {
      const std::size_t count = static_cast<std::size_t>(
          std::min<std::uint64_t>(buffer.size(), byte_count - offset));
      int error = 0;
      if (!pread_exact(fd, buffer.data(), count, offset, error)) {
        return make_diagnostic(ErrorCode::kIoFailure, "sha256",
                               "failed to read complete open file while hashing",
                               {}, {}, error);
      }
      if (!hasher.update(buffer.data(), count)) {
        return make_diagnostic(ErrorCode::kArithmeticOverflow, "sha256",
                               "SHA-256 input length overflowed");
      }
      offset += count;
    }
    digest = hasher.finalize().hex();
    return {};
  } catch (const std::bad_alloc&) {
    return make_diagnostic(ErrorCode::kAllocationFailure, "sha256",
                           "hash buffer allocation failed");
  }
}

struct OpenedShard final {
  std::string filename;
  std::uint64_t expected_size = 0U;
  UniqueFd fd;
  FileSnapshot snapshot;
};

struct SourceSet final {
  UniqueFd directory;
  std::vector<OpenedShard> shards;
};

[[nodiscard]] const OpenedShard* find_shard(
    const SourceSet& sources, const std::string_view filename) noexcept {
  const auto found = std::find_if(
      sources.shards.begin(), sources.shards.end(),
      [filename](const OpenedShard& shard) {
        return shard.filename == filename;
      });
  return found == sources.shards.end() ? nullptr : &*found;
}

[[nodiscard]] Diagnostic open_sources(const fs::path& model_directory,
                                      SourceSet& sources, Stats& stats) {
  sources.directory = UniqueFd(::open(model_directory.c_str(),
                                      O_RDONLY | O_DIRECTORY | O_CLOEXEC |
                                          O_NOFOLLOW));
  if (!sources.directory) {
    return make_diagnostic(ErrorCode::kOpenFailed,
                           model_directory.string(),
                           "failed to open non-symlink model directory", {},
                           {}, errno);
  }
  try {
    const auto& pinned = pinned_qwen36_27b_shards();
    sources.shards.reserve(pinned.size());
    for (const auto& identity : pinned) {
      if (!st::is_safe_relative_shard_path(identity.filename)) {
        return make_diagnostic(ErrorCode::kUnsafePath, identity.filename,
                               "pinned shard filename failed lexical policy");
      }
      UniqueFd fd(::openat(sources.directory.get(), identity.filename.c_str(),
                           O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
      if (!fd) {
        return make_diagnostic(ErrorCode::kOpenFailed, identity.filename,
                               "failed to open pinned source shard", {}, {},
                               errno);
      }
      if (::flock(fd.get(), LOCK_SH | LOCK_NB) != 0) {
        return make_diagnostic(
            ErrorCode::kIoFailure, identity.filename,
            "source shard is concurrently locked for mutation", {}, {},
            errno);
      }
      FileSnapshot snapshot;
      int error = 0;
      if (!capture_snapshot(fd.get(), snapshot, error)) {
        return make_diagnostic(ErrorCode::kIoFailure, identity.filename,
                               "source shard is not a regular file", {}, {},
                               error);
      }
      if (snapshot.size != identity.file_size) {
        return make_diagnostic(
            ErrorCode::kSourceTensorMismatch, identity.filename,
            "pinned source shard file size differs",
            std::to_string(identity.file_size), std::to_string(snapshot.size));
      }
      std::string digest;
      Diagnostic hash_diagnostic =
          hash_open_file(fd.get(), identity.file_size, digest);
      if (!hash_diagnostic) {
        hash_diagnostic.context = identity.filename;
        return hash_diagnostic;
      }
      FileSnapshot after_hash;
      if (!capture_snapshot(fd.get(), after_hash, error) ||
          !same_snapshot(snapshot, after_hash)) {
        return make_diagnostic(
            ErrorCode::kDigestMismatch, identity.filename,
            "source shard changed during complete pinned SHA-256", {}, {},
            error);
      }
      if (digest != identity.sha256) {
        return make_diagnostic(
            ErrorCode::kDigestMismatch, identity.filename,
            "source shard failed its pinned complete-file SHA-256",
            identity.sha256, digest);
      }
      if (!checked_add(stats.source_shard_bytes_hashed, identity.file_size,
                       stats.source_shard_bytes_hashed) ||
          !checked_add(stats.source_shards_authenticated, 1U,
                       stats.source_shards_authenticated)) {
        return make_diagnostic(
            ErrorCode::kArithmeticOverflow, identity.filename,
            "source authentication accounting overflowed");
      }
      OpenedShard shard;
      shard.filename = identity.filename;
      shard.expected_size = identity.file_size;
      shard.fd = std::move(fd);
      shard.snapshot = snapshot;
      sources.shards.emplace_back(std::move(shard));
    }
    return {};
  } catch (const std::bad_alloc&) {
    return make_diagnostic(ErrorCode::kAllocationFailure,
                           model_directory.string(),
                           "source descriptor inventory allocation failed");
  }
}

[[nodiscard]] Diagnostic revalidate_sources(const SourceSet& sources) {
  for (const auto& shard : sources.shards) {
    FileSnapshot after;
    struct stat named_status {};
    int error = 0;
    if (!capture_snapshot(shard.fd.get(), after, error)) {
      return make_diagnostic(ErrorCode::kIoFailure, shard.filename,
                             "failed to revalidate source shard", {}, {},
                             error);
    }
    if (!same_snapshot(shard.snapshot, after) ||
        after.size != shard.expected_size) {
      return make_diagnostic(ErrorCode::kDigestMismatch, shard.filename,
                             "source shard changed during conversion");
    }
    const int named_result =
        ::fstatat(sources.directory.get(), shard.filename.c_str(),
                  &named_status, AT_SYMLINK_NOFOLLOW);
    const int named_error = named_result == 0 ? 0 : errno;
    if (named_result != 0 ||
        !S_ISREG(named_status.st_mode) ||
        static_cast<std::uint64_t>(named_status.st_dev) !=
            shard.snapshot.device ||
        static_cast<std::uint64_t>(named_status.st_ino) !=
            shard.snapshot.inode) {
      return make_diagnostic(
          ErrorCode::kDigestMismatch, shard.filename,
          "model-directory entry no longer names the locked source shard", {},
          {}, named_error);
    }
  }
  return {};
}

[[nodiscard]] Diagnostic validate_locator(
    const SourceSet& sources, const mw::TensorLocator& locator,
    const st::DType dtype, const std::vector<std::uint64_t>& shape,
    const std::uint64_t byte_size, const std::string_view context) {
  const OpenedShard* const shard = find_shard(sources, locator.shard);
  std::uint64_t computed_end = 0U;
  if (shard == nullptr || locator.dtype != dtype || locator.shape != shape ||
      locator.byte_size != byte_size || locator.file_end < locator.file_begin ||
      locator.file_end - locator.file_begin != locator.byte_size ||
      !checked_add(locator.file_begin, locator.byte_size, computed_end) ||
      computed_end != locator.file_end || locator.file_end >
                                             (shard == nullptr ? 0U
                                                               : shard->snapshot.size) ||
      locator.file.filename() != fs::path(locator.shard)) {
    return make_diagnostic(
        ErrorCode::kSourceTensorMismatch, std::string(context),
        "source tensor locator/dtype/shape/range differs from direct NVFP4 ABI");
  }
  return {};
}

[[nodiscard]] Diagnostic read_locator_range(
    const SourceSet& sources, const mw::TensorLocator& locator,
    const std::uint64_t relative_offset, void* const destination,
    const std::size_t byte_count, const std::string_view context,
    Stats& stats) {
  std::uint64_t relative_end = 0U;
  std::uint64_t absolute = 0U;
  if (!checked_add(relative_offset, byte_count, relative_end) ||
      relative_end > locator.byte_size ||
      !checked_add(locator.file_begin, relative_offset, absolute)) {
    return make_diagnostic(ErrorCode::kArithmeticOverflow,
                           std::string(context),
                           "source tensor subrange is outside its locator");
  }
  const OpenedShard* const shard = find_shard(sources, locator.shard);
  if (shard == nullptr) {
    return make_diagnostic(ErrorCode::kSourceTensorMismatch,
                           std::string(context),
                           "source tensor names an unopened shard");
  }
  int error = 0;
  if (!pread_exact(shard->fd.get(), destination, byte_count, absolute,
                   error)) {
    return make_diagnostic(ErrorCode::kIoFailure, std::string(context),
                           "short or failed source tensor read", {}, {},
                           error);
  }
  if (!checked_add(stats.source_bytes_read, byte_count,
                   stats.source_bytes_read)) {
    return make_diagnostic(ErrorCode::kArithmeticOverflow,
                           std::string(context),
                           "source byte counter overflowed");
  }
  return {};
}

[[nodiscard]] float read_f32_little_endian(
    const std::uint8_t* const bytes) noexcept {
  const std::uint32_t bits = static_cast<std::uint32_t>(bytes[0]) |
                             (static_cast<std::uint32_t>(bytes[1]) << 8U) |
                             (static_cast<std::uint32_t>(bytes[2]) << 16U) |
                             (static_cast<std::uint32_t>(bytes[3]) << 24U);
  float output = 0.0F;
  std::memcpy(&output, &bits, sizeof(output));
  return output;
}

struct IdentityAlpha final {
  std::vector<float> values;
  std::string path;
  std::string sha256;
  PrefillMLPFactorizedLaneR4MetadataResult metadata;
};

[[nodiscard]] Diagnostic build_identity_alpha(const std::size_t input_size,
                                              IdentityAlpha& output) {
  try {
    if (input_size != 5'120U && input_size != 17'408U) {
      return make_diagnostic(ErrorCode::kInvalidManifest, "identity_alpha",
                             "identity direction gate supports only pinned MLP K");
    }
    output.values.assign(input_size, 1.0F);
    output.path = input_size == 5'120U
                      ? std::string(
                            kPrefillMLPFactorizedLaneR4IdentityCandidateAlpha5120)
                      : std::string(
                            kPrefillMLPFactorizedLaneR4IdentityCandidateAlpha17408);
    std::vector<std::uint8_t> bytes(input_size * sizeof(float));
    for (std::size_t index = 0U; index < input_size; ++index) {
      bytes[index * 4U] = 0x00U;
      bytes[index * 4U + 1U] = 0x00U;
      bytes[index * 4U + 2U] = 0x80U;
      bytes[index * 4U + 3U] = 0x3fU;
    }
    core::Sha256 hasher;
    if (!hasher.update(bytes.data(), bytes.size())) {
      return make_diagnostic(ErrorCode::kArithmeticOverflow,
                             "identity_alpha",
                             "identity-alpha SHA-256 length overflowed");
    }
    output.sha256 = hasher.finalize().hex();
    const std::string_view expected_sha256 =
        input_size == 5'120U
            ? kPrefillMLPFactorizedLaneR4IdentityCandidateAlpha5120Sha256
            : kPrefillMLPFactorizedLaneR4IdentityCandidateAlpha17408Sha256;
    if (output.sha256 != expected_sha256) {
      return make_diagnostic(
          ErrorCode::kDigestMismatch, "identity_alpha",
          "materialized FP32LE identity alpha differs from the pinned binding",
          std::string(expected_sha256), output.sha256);
    }
    output.metadata = build_prefill_mlp_factorized_lane_r4_metadata(
        output.values.data(), output.values.size());
    if (!output.metadata) {
      return make_diagnostic(
          ErrorCode::kMetadataFailure, "identity_alpha.metadata",
          "failed to build direct R4 identity inverse-alpha metadata",
          {}, std::string(to_string(output.metadata.diagnostic.code)));
    }
    const auto parsed = parse_prefill_mlp_factorized_lane_metadata(
        output.metadata.metadata.bytes.data(),
        output.metadata.metadata.bytes.size(),
        kPrefillMLPFactorizedLaneR4LaneCount, input_size);
    if (!parsed || parsed.inverse_alpha.size() != input_size ||
        !std::all_of(parsed.inverse_alpha.begin(), parsed.inverse_alpha.end(),
                     [](const float value) { return value == 1.0F; })) {
      return make_diagnostic(ErrorCode::kMetadataFailure,
                             "identity_alpha.metadata",
                             "identity metadata failed canonical reparse");
    }
    return {};
  } catch (const std::bad_alloc&) {
    return make_diagnostic(ErrorCode::kAllocationFailure, "identity_alpha",
                           "identity-alpha allocation failed");
  } catch (const std::length_error&) {
    return make_diagnostic(ErrorCode::kAllocationFailure, "identity_alpha",
                           "identity-alpha size exceeded host limits");
  }
}

void remove_if_present(const fs::path& path) noexcept {
  if (!path.empty()) {
    (void)::unlink(path.c_str());
  }
}

[[nodiscard]] bool target_absent(const fs::path& path,
                                 Diagnostic& diagnostic) {
  struct stat status {};
  if (::lstat(path.c_str(), &status) == 0) {
    diagnostic = make_diagnostic(ErrorCode::kPublicationConflict,
                                 path.string(),
                                 "publication target already exists");
    return false;
  }
  if (errno != ENOENT) {
    diagnostic = make_diagnostic(ErrorCode::kIoFailure, path.string(),
                                 "failed to inspect publication target", {},
                                 {}, errno);
    return false;
  }
  return true;
}

[[nodiscard]] UniqueFd create_temporary_file_near(
    const fs::path& target, const std::string_view tag, fs::path& path,
    Diagnostic& diagnostic) {
  try {
    std::string pattern = target.string() + ".tmp." + std::string(tag) +
                          ".XXXXXX";
    std::vector<char> mutable_pattern(pattern.begin(), pattern.end());
    mutable_pattern.push_back('\0');
    const int fd = ::mkstemp(mutable_pattern.data());
    if (fd < 0) {
      diagnostic = make_diagnostic(ErrorCode::kOpenFailed, target.string(),
                                   "failed to create temporary file", {}, {},
                                   errno);
      return {};
    }
    UniqueFd output(fd);
    path = fs::path(mutable_pattern.data());
    const int flags = ::fcntl(fd, F_GETFD);
    if (flags < 0 || ::fcntl(fd, F_SETFD, flags | FD_CLOEXEC) != 0 ||
        ::fchmod(fd, S_IRUSR | S_IWUSR) != 0 ||
        ::flock(fd, LOCK_EX | LOCK_NB) != 0) {
      diagnostic = make_diagnostic(ErrorCode::kIoFailure, path.string(),
                                   "failed to secure temporary file", {}, {},
                                   errno);
      return {};
    }
    return output;
  } catch (const std::bad_alloc&) {
    diagnostic = make_diagnostic(ErrorCode::kAllocationFailure,
                                 target.string(),
                                 "temporary filename allocation failed");
    return {};
  }
}

[[nodiscard]] Diagnostic seal_document(const int fd, const fs::path& path,
                                       const std::string_view document) {
  int error = 0;
  if (!pwrite_exact(fd, document.data(), document.size(), 0U, error) ||
      ::ftruncate(fd, static_cast<off_t>(document.size())) != 0 ||
      ::fsync(fd) != 0) {
    return make_diagnostic(ErrorCode::kIoFailure, path.string(),
                           "failed to write publication document", {}, {},
                           error != 0 ? error : errno);
  }
  FileSnapshot before;
  FileSnapshot after;
  int snapshot_error = 0;
  std::string digest;
  if (!capture_snapshot(fd, before, snapshot_error) ||
      before.size != document.size()) {
    return make_diagnostic(ErrorCode::kIoFailure, path.string(),
                           "publication document size/snapshot is invalid", {},
                           {}, snapshot_error);
  }
  Diagnostic diagnostic = hash_open_file(fd, before.size, digest);
  if (!diagnostic) {
    diagnostic.context = path.string();
    return diagnostic;
  }
  const std::string expected = sha256_text(document);
  if (!capture_snapshot(fd, after, snapshot_error) ||
      !same_snapshot(before, after) || digest != expected) {
    return make_diagnostic(ErrorCode::kDigestMismatch, path.string(),
                           "publication document failed readback hash", expected,
                           digest, snapshot_error);
  }
  if (::fchmod(fd, S_IRUSR) != 0 || ::fsync(fd) != 0) {
    return make_diagnostic(ErrorCode::kIoFailure, path.string(),
                           "failed to seal publication document read-only", {},
                           {}, errno);
  }
  return {};
}

[[nodiscard]] bool secured_temp_matches_fd(const fs::path& path,
                                           const int fd) noexcept {
  struct stat path_status {};
  struct stat fd_status {};
  return ::lstat(path.c_str(), &path_status) == 0 &&
         ::fstat(fd, &fd_status) == 0 && S_ISREG(path_status.st_mode) &&
         path_status.st_dev == fd_status.st_dev &&
         path_status.st_ino == fd_status.st_ino && fd_status.st_nlink == 1 &&
         (fd_status.st_mode & (S_IWUSR | S_IWGRP | S_IWOTH)) == 0;
}

struct PublicationFile final {
  fs::path temp;
  int fd = -1;
  fs::path target;
};

[[nodiscard]] Diagnostic publish_four_no_replace(
    const std::array<PublicationFile, 4U>& files) {
  for (const auto& file : files) {
    if (!secured_temp_matches_fd(file.temp, file.fd)) {
      return make_diagnostic(
          ErrorCode::kPublicationConflict, file.temp.string(),
          "temporary path no longer names its secured read-only descriptor");
    }
  }
  std::size_t linked = 0U;
  const auto rollback = [&]() noexcept {
    while (linked != 0U) {
      --linked;
      (void)::unlink(files[linked].target.c_str());
    }
  };
  for (; linked < files.size(); ++linked) {
    if (::link(files[linked].temp.c_str(), files[linked].target.c_str()) != 0) {
      const int saved = errno;
      const fs::path failed = files[linked].target;
      rollback();
      return make_diagnostic(ErrorCode::kPublicationConflict, failed.string(),
                             "no-replace publication link failed", {}, {},
                             saved);
    }
  }
  for (const auto& file : files) {
    if (::unlink(file.temp.c_str()) != 0) {
      const int saved = errno;
      rollback();
      return make_diagnostic(ErrorCode::kIoFailure, file.temp.string(),
                             "temporary unlink failed; publication rolled back",
                             {}, {}, saved);
    }
  }
  const fs::path parent = files[0].target.parent_path().empty()
                              ? fs::path(".")
                              : files[0].target.parent_path();
  UniqueFd directory(::open(parent.c_str(), O_RDONLY | O_DIRECTORY |
                                                O_CLOEXEC | O_NOFOLLOW));
  if (!directory || ::fsync(directory.get()) != 0) {
    const int saved = errno;
    rollback();
    if (directory) {
      (void)::fsync(directory.get());
    }
    return make_diagnostic(ErrorCode::kIoFailure, parent.string(),
                           "directory sync failed; publication rolled back", {},
                           {}, saved);
  }
  return {};
}

[[nodiscard]] Diagnostic publication_diagnostic(
    const PrefillMLPFactorizedLaneR4PublicationDiagnostic& source,
    const std::string_view context) {
  return make_diagnostic(ErrorCode::kPublicationFailure,
                         std::string(context) + ":" + source.context,
                         source.message, source.expected, source.actual);
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

}  // namespace

PrefillMLPFactorizedLaneR4IdentityCandidateConversionResult
convert_pinned_qwen36_27b_prefill_mlp_factorized_lane_r4_identity_candidate(
    const PrefillMLPFactorizedLaneR4IdentityCandidateConversionOptions&
        options) {
  PrefillMLPFactorizedLaneR4IdentityCandidateConversionResult result;
  fs::path payload_temp;
  fs::path manifest_temp;
  fs::path policy_temp;
  fs::path receipt_temp;
  struct TemporaryCleanup final {
    fs::path* payload;
    fs::path* manifest;
    fs::path* policy;
    fs::path* receipt;
    ~TemporaryCleanup() {
      remove_if_present(*payload);
      remove_if_present(*manifest);
      remove_if_present(*policy);
      remove_if_present(*receipt);
    }
  } cleanup{&payload_temp, &manifest_temp, &policy_temp, &receipt_temp};

  try {
    const fs::path manifest_path(
        options.output_path.string() + ".manifest.json");
    const fs::path policy_path(options.output_path.string() + ".policy.json");
    const fs::path receipt_path(
        options.output_path.string() + ".receipt.json");
    if (options.model_directory.empty() || options.output_path.empty() ||
        options.output_path.filename().empty() ||
        options.output_path.filename() == "." ||
        options.output_path.filename() == ".." ||
        !valid_clip_ratio(options.weight_clip_ratio) ||
        !valid_clip_ratio(options.activation_clip_ratio) ||
        !valid_row_chunk(options.row_chunk_size)) {
      result.diagnostic = make_diagnostic(
          ErrorCode::kInvalidOption, "conversion_options",
          "model/output, explicit independent clips, and N64 row chunk <=256 are required");
      return result;
    }
    if (!target_absent(options.output_path, result.diagnostic) ||
        !target_absent(manifest_path, result.diagnostic) ||
        !target_absent(policy_path, result.diagnostic) ||
        !target_absent(receipt_path, result.diagnostic)) {
      return result;
    }
    const fs::path output_parent = options.output_path.parent_path().empty()
                                       ? fs::path(".")
                                       : options.output_path.parent_path();
    std::error_code filesystem_error;
    const fs::file_status parent_status =
        fs::symlink_status(output_parent, filesystem_error);
    if (filesystem_error || !fs::is_directory(parent_status) ||
        fs::is_symlink(parent_status)) {
      result.diagnostic = make_diagnostic(
          ErrorCode::kUnsafePath, output_parent.string(),
          "output parent must be an existing non-symlink directory", {},
          filesystem_error.message());
      return result;
    }

    const mw::ManifestResult source_manifest =
        mw::build_qwen36_27b_text_manifest(options.model_directory);
    if (!source_manifest) {
      result.diagnostic = make_diagnostic(
          ErrorCode::kInvalidManifest, options.model_directory.string(),
          "pinned Qwen3.6-27B checkpoint manifest validation failed");
      return result;
    }
    PrefillSidecarManifestOptions exact_options;
    exact_options.kind = PrefillSidecarKind::kExact;
    const PrefillSidecarManifestResult exact_result =
        build_qwen36_27b_prefill_sidecar_manifest(
            *source_manifest.value, pinned_qwen36_27b_shards(), exact_options);
    if (!exact_result) {
      result.diagnostic = make_diagnostic(
          ErrorCode::kInvalidManifest, exact_result.diagnostic.context,
          exact_result.diagnostic.message, exact_result.diagnostic.expected,
          exact_result.diagnostic.actual);
      return result;
    }
    const PrefillSidecarManifest& exact_manifest = *exact_result.value;
    const auto manifest_result =
        build_prefill_mlp_factorized_lane_r4_direct_manifest(exact_manifest);
    if (!manifest_result) {
      result.diagnostic =
          publication_diagnostic(manifest_result.diagnostic, "manifest");
      return result;
    }
    const auto manifest_reparse =
        parse_prefill_mlp_factorized_lane_r4_direct_manifest(
            manifest_result.canonical_document, exact_manifest);
    if (!manifest_reparse ||
        manifest_reparse.canonical_document !=
            manifest_result.canonical_document) {
      result.diagnostic = make_diagnostic(
          ErrorCode::kPublicationFailure, "manifest",
          "constructed direct manifest failed canonical strict reparse");
      return result;
    }
    PrefillMLPFactorizedLaneR4Manifest manifest = *manifest_result.value;

    IdentityAlpha alpha5120;
    IdentityAlpha alpha17408;
    result.diagnostic = build_identity_alpha(5'120U, alpha5120);
    if (!result.diagnostic) {
      return result;
    }
    result.diagnostic = build_identity_alpha(17'408U, alpha17408);
    if (!result.diagnostic) {
      return result;
    }
    std::vector<PrefillMLPFactorizedLaneR4CalibrationSpec> calibration;
    calibration.reserve(manifest.projections.size());
    for (const auto& projection : manifest.projections) {
      const IdentityAlpha& alpha =
          projection.input_size == 5'120U ? alpha5120 : alpha17408;
      PrefillMLPFactorizedLaneR4CalibrationSpec spec;
      spec.alpha_scheme = std::string(
          kPrefillMLPFactorizedLaneR4IdentityCandidateFactorScheme);
      spec.weight_clip_ratio = options.weight_clip_ratio;
      spec.activation_clip_ratio = options.activation_clip_ratio;
      spec.alpha_path = alpha.path;
      spec.alpha_sha256 = alpha.sha256;
      spec.alpha_element_count = projection.input_size;
      calibration.emplace_back(std::move(spec));
    }
    const auto policy_result =
        build_prefill_mlp_factorized_lane_r4_policy(manifest, calibration);
    if (!policy_result) {
      result.diagnostic =
          publication_diagnostic(policy_result.diagnostic, "policy");
      return result;
    }
    const auto policy_reparse = parse_prefill_mlp_factorized_lane_r4_policy(
        policy_result.canonical_document, manifest);
    if (!policy_reparse ||
        policy_reparse.canonical_document != policy_result.canonical_document) {
      result.diagnostic = make_diagnostic(
          ErrorCode::kPublicationFailure, "policy",
          "identity-alpha direction policy failed canonical strict reparse");
      return result;
    }
    PrefillMLPFactorizedLaneR4Policy policy = *policy_result.value;

    SourceSet sources;
    result.diagnostic =
        open_sources(options.model_directory, sources, result.stats);
    if (!result.diagnostic) {
      return result;
    }
    std::uint64_t expected_authenticated_bytes = 0U;
    for (const auto& identity : pinned_qwen36_27b_shards()) {
      if (!checked_add(expected_authenticated_bytes, identity.file_size,
                       expected_authenticated_bytes)) {
        result.diagnostic = make_diagnostic(
            ErrorCode::kArithmeticOverflow, "source_authentication",
            "pinned shard byte inventory overflowed");
        return result;
      }
    }
    if (result.stats.source_shards_authenticated !=
            pinned_qwen36_27b_shards().size() ||
        result.stats.source_shard_bytes_hashed !=
            expected_authenticated_bytes) {
      result.diagnostic = make_diagnostic(
          ErrorCode::kDigestMismatch, "source_authentication",
          "complete pinned shard SHA-256 coverage is incomplete",
          std::to_string(expected_authenticated_bytes),
          std::to_string(result.stats.source_shard_bytes_hashed));
      return result;
    }

    UniqueFd output = create_temporary_file_near(
        options.output_path, "r4-identity-payload", payload_temp,
        result.diagnostic);
    if (!output) {
      return result;
    }
    if (!offset_fits(manifest.payload_bytes) ||
        ::ftruncate(output.get(), static_cast<off_t>(manifest.payload_bytes)) !=
            0) {
      result.diagnostic = make_diagnostic(
          ErrorCode::kIoFailure, payload_temp.string(),
          "failed to size fixed direct R4 payload", {}, {}, errno);
      return result;
    }
    if (options.preallocate_output) {
      const int error = ::posix_fallocate(
          output.get(), 0, static_cast<off_t>(manifest.payload_bytes));
      if (error != 0) {
        result.diagnostic = make_diagnostic(
            ErrorCode::kIoFailure, payload_temp.string(),
            "failed to preallocate complete direct R4 payload", {}, {}, error);
        return result;
      }
    }

    const auto overlay_plan = prefill_mlp_factorized_lane_overlay_layout_plan(
        kPrefillMLPFactorizedLaneR4LaneCount);
    if (!overlay_plan || overlay_plan.payload_bytes != manifest.payload_bytes) {
      result.diagnostic = make_diagnostic(
          ErrorCode::kInvalidManifest, "r4.layout",
          "fixed direct R4 overlay plan is invalid");
      return result;
    }

    for (const auto& projection : manifest.projections) {
      const std::string weight_name = projection.source_module + ".weight";
      const std::string scale_name =
          projection.source_module + ".weight_scale";
      const std::string scalar_name =
          projection.source_module + ".weight_scale_2";
      const mw::TensorLocator* const weight =
          source_manifest.value->find(weight_name);
      const mw::TensorLocator* const scale =
          source_manifest.value->find(scale_name);
      const mw::TensorLocator* const scalar =
          source_manifest.value->find(scalar_name);
      if (weight == nullptr || scale == nullptr || scalar == nullptr) {
        result.diagnostic = make_diagnostic(
            ErrorCode::kSourceTensorMismatch, projection.source_module,
            "direct NVFP4 weight/weight_scale/weight_scale_2 is missing");
        return result;
      }
      std::uint64_t packed_bytes_u64 = 0U;
      std::uint64_t source_scale_bytes_u64 = 0U;
      if (!checked_multiply(projection.output_size,
                            projection.input_size / 2U,
                            packed_bytes_u64) ||
          !checked_multiply(projection.output_size,
                            projection.input_size / 16U,
                            source_scale_bytes_u64)) {
        result.diagnostic = make_diagnostic(
            ErrorCode::kArithmeticOverflow, projection.source_module,
            "direct source tensor byte count overflowed");
        return result;
      }
      result.diagnostic = validate_locator(
          sources, *weight, st::DType::kU8,
          {projection.output_size, projection.input_size / 2U},
          packed_bytes_u64, weight_name);
      if (!result.diagnostic) {
        return result;
      }
      result.diagnostic = validate_locator(
          sources, *scale, st::DType::kF8E4M3,
          {projection.output_size, projection.input_size / 16U},
          source_scale_bytes_u64, scale_name);
      if (!result.diagnostic) {
        return result;
      }
      result.diagnostic = validate_locator(sources, *scalar, st::DType::kF32,
                                           {}, sizeof(float), scalar_name);
      if (!result.diagnostic) {
        return result;
      }

      std::array<std::uint8_t, sizeof(float)> scalar_bytes{};
      result.diagnostic = read_locator_range(
          sources, *scalar, 0U, scalar_bytes.data(), scalar_bytes.size(),
          scalar_name, result.stats);
      if (!result.diagnostic) {
        return result;
      }
      const float weight_scale_2 =
          read_f32_little_endian(scalar_bytes.data());
      if (!std::isfinite(weight_scale_2) || weight_scale_2 < 0.0F) {
        result.diagnostic = make_diagnostic(
            ErrorCode::kSourceTensorMismatch, scalar_name,
            "direct NVFP4 weight_scale_2 must be finite and nonnegative");
        return result;
      }

      const auto& layout = projection_layout(overlay_plan, projection.family);
      if (!layout || layout.output_size != projection.output_size ||
          layout.input_size != projection.input_size ||
          layout.projection_bytes != projection.payload_bytes) {
        result.diagnostic = make_diagnostic(
            ErrorCode::kInvalidManifest, projection.source_module,
            "projection differs from fixed R4 payload layout");
        return result;
      }
      const std::size_t input_size =
          static_cast<std::size_t>(projection.input_size);
      const std::size_t source_weight_stride = input_size / 2U;
      const std::size_t source_scale_stride = input_size / 16U;
      const std::size_t output_scale_stride =
          kPrefillMLPFactorizedLaneR4LaneCount * sizeof(std::uint16_t);
      const std::size_t maximum_rows = std::min<std::size_t>(
          options.row_chunk_size,
          static_cast<std::size_t>(projection.output_size));
      std::vector<std::uint8_t> source_weights(maximum_rows *
                                                source_weight_stride);
      std::vector<std::uint8_t> source_scales(maximum_rows *
                                               source_scale_stride);
      std::vector<std::uint8_t> packed(maximum_rows * source_weight_stride);
      std::vector<std::uint8_t> scales(maximum_rows * output_scale_stride);
      const IdentityAlpha& alpha =
          projection.input_size == 5'120U ? alpha5120 : alpha17408;
      const std::uint64_t identity_bytes =
          (alpha5120.values.capacity() + alpha17408.values.capacity()) *
              sizeof(float) +
          alpha5120.metadata.metadata.bytes.capacity() +
          alpha17408.metadata.metadata.bytes.capacity();
      const std::uint64_t primitive_decode_bytes =
          kPrefillA4FactorizedLaneOuterBlock * input_size * sizeof(float);
      const std::uint64_t working_bytes =
          source_weights.capacity() + source_scales.capacity() +
          packed.capacity() + scales.capacity() + identity_bytes +
          primitive_decode_bytes;
      result.stats.peak_working_bytes =
          std::max(result.stats.peak_working_bytes, working_bytes);

      for (std::uint64_t row = 0U; row < projection.output_size;
           row += maximum_rows) {
        const std::size_t rows = static_cast<std::size_t>(
            std::min<std::uint64_t>(maximum_rows,
                                    projection.output_size - row));
        const std::size_t weight_bytes = rows * source_weight_stride;
        const std::size_t source_scale_bytes = rows * source_scale_stride;
        const std::size_t output_scale_bytes = rows * output_scale_stride;
        result.diagnostic = read_locator_range(
            sources, *weight, row * source_weight_stride,
            source_weights.data(), weight_bytes, weight_name, result.stats);
        if (!result.diagnostic) {
          return result;
        }
        result.diagnostic = read_locator_range(
            sources, *scale, row * source_scale_stride, source_scales.data(),
            source_scale_bytes, scale_name, result.stats);
        if (!result.diagnostic) {
          return result;
        }
        const auto transform =
            transform_prefill_mlp_nvfp4_to_factorized_r4_consumer_blocks(
                source_weights.data(), weight_bytes, source_scales.data(),
                source_scale_bytes, weight_scale_2, rows, input_size,
                alpha.values.data(), alpha.values.size(),
                options.weight_clip_ratio, packed.data(), weight_bytes,
                scales.data(), output_scale_bytes);
        if (!transform) {
          result.diagnostic = make_diagnostic(
              ErrorCode::kQuantizationFailure, projection.source_module,
              transform.message, {}, std::string(to_string(transform.code)));
          return result;
        }
        std::uint64_t packed_row_offset = 0U;
        std::uint64_t scale_row_offset = 0U;
        std::uint64_t packed_relative = 0U;
        std::uint64_t scale_relative = 0U;
        std::uint64_t packed_offset = 0U;
        std::uint64_t scale_offset = 0U;
        if (!checked_multiply(row, source_weight_stride,
                              packed_row_offset) ||
            !checked_multiply(row, output_scale_stride, scale_row_offset) ||
            !checked_add(layout.packed_weight_offset, packed_row_offset,
                         packed_relative) ||
            !checked_add(layout.weight_scale_offset, scale_row_offset,
                         scale_relative) ||
            !checked_add(projection.payload_offset, packed_relative,
                         packed_offset) ||
            !checked_add(projection.payload_offset, scale_relative,
                         scale_offset)) {
          result.diagnostic = make_diagnostic(
              ErrorCode::kArithmeticOverflow, projection.source_module,
              "R4 output chunk offset overflowed");
          return result;
        }
        int error = 0;
        if (!pwrite_exact(output.get(), packed.data(), weight_bytes,
                          packed_offset, error) ||
            !pwrite_exact(output.get(), scales.data(), output_scale_bytes,
                          scale_offset, error)) {
          result.diagnostic = make_diagnostic(
              ErrorCode::kIoFailure, payload_temp.string(),
              "failed to write complete R4 projection chunk", {}, {}, error);
          return result;
        }
        std::uint64_t written = 0U;
        if (!checked_add(weight_bytes, output_scale_bytes, written) ||
            !checked_add(result.stats.output_bytes_written, written,
                         result.stats.output_bytes_written)) {
          result.diagnostic = make_diagnostic(
              ErrorCode::kArithmeticOverflow, projection.source_module,
              "R4 output byte counter overflowed");
          return result;
        }
        if (!checked_add(result.stats.n64_blocks_converted,
                         rows / kPrefillA4FactorizedLaneOuterBlock,
                         result.stats.n64_blocks_converted)) {
          result.diagnostic = make_diagnostic(
              ErrorCode::kArithmeticOverflow, projection.source_module,
              "R4 N64 block counter overflowed");
          return result;
        }
      }

      std::uint64_t metadata_offset = 0U;
      if (!checked_add(projection.payload_offset, layout.metadata_offset,
                       metadata_offset)) {
        result.diagnostic = make_diagnostic(
            ErrorCode::kArithmeticOverflow, projection.source_module,
            "R4 metadata output offset overflowed");
        return result;
      }
      int error = 0;
      if (!pwrite_exact(output.get(), alpha.metadata.metadata.bytes.data(),
                        alpha.metadata.metadata.bytes.size(), metadata_offset,
                        error)) {
        result.diagnostic = make_diagnostic(
            ErrorCode::kIoFailure, payload_temp.string(),
            "failed to write R4 identity metadata", {}, {}, error);
        return result;
      }
      std::vector<std::uint8_t> metadata_readback(
          alpha.metadata.metadata.bytes.size());
      if (!pread_exact(output.get(), metadata_readback.data(),
                       metadata_readback.size(), metadata_offset, error)) {
        result.diagnostic = make_diagnostic(
            ErrorCode::kIoFailure, payload_temp.string(),
            "failed to read back complete R4 metadata", {}, {}, error);
        return result;
      }
      const auto metadata_reparse = parse_prefill_mlp_factorized_lane_metadata(
          metadata_readback.data(), metadata_readback.size(),
          kPrefillMLPFactorizedLaneR4LaneCount, projection.input_size);
      if (metadata_readback != alpha.metadata.metadata.bytes ||
          !metadata_reparse || metadata_reparse.inverse_alpha.size() !=
                                   projection.input_size ||
          !std::all_of(metadata_reparse.inverse_alpha.begin(),
                       metadata_reparse.inverse_alpha.end(),
                       [](const float value) { return value == 1.0F; })) {
        result.diagnostic = make_diagnostic(
            ErrorCode::kMetadataFailure, projection.source_module,
            "published R4 metadata failed strict readback reparse");
        return result;
      }
      if (!checked_add(result.stats.output_bytes_written,
                       metadata_readback.size(),
                       result.stats.output_bytes_written)) {
        result.diagnostic = make_diagnostic(
            ErrorCode::kArithmeticOverflow, projection.source_module,
            "R4 metadata byte counter overflowed");
        return result;
      }
      ++result.stats.projections_converted;
    }

    if (result.stats.source_bytes_read != kExpectedSourceBytes ||
        result.stats.projections_converted !=
            kPrefillMLPFactorizedLaneProjectionCount) {
      result.diagnostic = make_diagnostic(
          ErrorCode::kDigestMismatch, "conversion_accounting",
          "direct source coverage is incomplete",
          std::to_string(kExpectedSourceBytes),
          std::to_string(result.stats.source_bytes_read));
      return result;
    }
    result.diagnostic = revalidate_sources(sources);
    if (!result.diagnostic) {
      return result;
    }
    if (::fsync(output.get()) != 0) {
      result.diagnostic = make_diagnostic(
          ErrorCode::kIoFailure, payload_temp.string(),
          "failed to sync complete R4 payload", {}, {}, errno);
      return result;
    }
    FileSnapshot payload_before;
    FileSnapshot payload_after;
    int snapshot_error = 0;
    if (!capture_snapshot(output.get(), payload_before, snapshot_error) ||
        payload_before.size != manifest.payload_bytes) {
      result.diagnostic = make_diagnostic(
          ErrorCode::kIoFailure, payload_temp.string(),
          "fixed R4 payload size/snapshot is invalid",
          std::to_string(manifest.payload_bytes),
          std::to_string(payload_before.size), snapshot_error);
      return result;
    }
    std::string payload_sha256;
    result.diagnostic =
        hash_open_file(output.get(), manifest.payload_bytes, payload_sha256);
    if (!result.diagnostic) {
      result.diagnostic.context = payload_temp.string();
      return result;
    }
    if (!capture_snapshot(output.get(), payload_after, snapshot_error) ||
        !same_snapshot(payload_before, payload_after)) {
      result.diagnostic = make_diagnostic(
          ErrorCode::kDigestMismatch, payload_temp.string(),
          "R4 payload changed during complete SHA-256 readback", {}, {},
          snapshot_error);
      return result;
    }
    if (::fchmod(output.get(), S_IRUSR) != 0 || ::fsync(output.get()) != 0) {
      result.diagnostic = make_diagnostic(
          ErrorCode::kIoFailure, payload_temp.string(),
          "failed to seal complete R4 payload read-only", {}, {}, errno);
      return result;
    }

    const auto receipt_result =
        build_prefill_mlp_factorized_lane_r4_receipt(manifest, policy,
                                                     payload_sha256);
    if (!receipt_result) {
      result.diagnostic =
          publication_diagnostic(receipt_result.diagnostic, "receipt");
      return result;
    }
    const auto receipt_reparse =
        parse_prefill_mlp_factorized_lane_r4_receipt(
            receipt_result.canonical_document, manifest, policy);
    if (!receipt_reparse ||
        receipt_reparse.canonical_document != receipt_result.canonical_document) {
      result.diagnostic = make_diagnostic(
          ErrorCode::kPublicationFailure, "receipt",
          "direct R4 receipt failed canonical strict reparse");
      return result;
    }

    UniqueFd manifest_output = create_temporary_file_near(
        manifest_path, "r4-manifest", manifest_temp, result.diagnostic);
    if (!manifest_output) {
      return result;
    }
    UniqueFd policy_output = create_temporary_file_near(
        policy_path, "r4-policy", policy_temp, result.diagnostic);
    if (!policy_output) {
      return result;
    }
    UniqueFd receipt_output = create_temporary_file_near(
        receipt_path, "r4-receipt", receipt_temp, result.diagnostic);
    if (!receipt_output) {
      return result;
    }
    result.diagnostic = seal_document(manifest_output.get(), manifest_temp,
                                      manifest_result.canonical_document);
    if (!result.diagnostic) {
      return result;
    }
    result.diagnostic = seal_document(policy_output.get(), policy_temp,
                                      policy_result.canonical_document);
    if (!result.diagnostic) {
      return result;
    }
    result.diagnostic = seal_document(receipt_output.get(), receipt_temp,
                                      receipt_result.canonical_document);
    if (!result.diagnostic) {
      return result;
    }
    result.diagnostic = revalidate_sources(sources);
    if (!result.diagnostic) {
      return result;
    }
    const std::array<PublicationFile, 4U> files{{
        {payload_temp, output.get(), options.output_path},
        {manifest_temp, manifest_output.get(), manifest_path},
        {policy_temp, policy_output.get(), policy_path},
        {receipt_temp, receipt_output.get(), receipt_path},
    }};
    result.diagnostic = publish_four_no_replace(files);
    if (!result.diagnostic) {
      return result;
    }
    payload_temp.clear();
    manifest_temp.clear();
    policy_temp.clear();
    receipt_temp.clear();
    result.manifest.emplace(std::move(manifest));
    result.policy.emplace(std::move(policy));
    result.receipt.emplace(*receipt_result.value);
    return result;
  } catch (const std::bad_alloc&) {
    result.diagnostic = make_diagnostic(
        ErrorCode::kAllocationFailure, "r4_identity_conversion",
        "direct R4 identity candidate allocation failed");
    return result;
  } catch (const std::length_error&) {
    result.diagnostic = make_diagnostic(
        ErrorCode::kAllocationFailure, "r4_identity_conversion",
        "direct R4 identity candidate exceeded host container limits");
    return result;
  } catch (...) {
    result.diagnostic = make_diagnostic(
        ErrorCode::kIoFailure, "r4_identity_conversion",
        "unexpected direct R4 identity candidate conversion failure");
    return result;
  }
}

std::string_view to_string(
    const PrefillMLPFactorizedLaneR4CandidateConverterErrorCode code) noexcept {
  switch (code) {
    case ErrorCode::kNone:
      return "none";
    case ErrorCode::kInvalidOption:
      return "invalid_option";
    case ErrorCode::kInvalidManifest:
      return "invalid_manifest";
    case ErrorCode::kSourceTensorMismatch:
      return "source_tensor_mismatch";
    case ErrorCode::kUnsafePath:
      return "unsafe_path";
    case ErrorCode::kOpenFailed:
      return "open_failed";
    case ErrorCode::kIoFailure:
      return "io_failure";
    case ErrorCode::kArithmeticOverflow:
      return "arithmetic_overflow";
    case ErrorCode::kQuantizationFailure:
      return "quantization_failure";
    case ErrorCode::kMetadataFailure:
      return "metadata_failure";
    case ErrorCode::kPublicationFailure:
      return "publication_failure";
    case ErrorCode::kDigestMismatch:
      return "digest_mismatch";
    case ErrorCode::kPublicationConflict:
      return "publication_conflict";
    case ErrorCode::kAllocationFailure:
      return "allocation_failure";
  }
  return "unknown";
}

}  // namespace q3x::runtime
