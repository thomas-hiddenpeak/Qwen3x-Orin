#include "sm87_target_aot_projection_persistence_internal.h"

#include "q3x/core/sha256.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <linux/fs.h>
#include <limits>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace q3x::runtime::target_aot_persistence_detail {
namespace {

constexpr std::array<std::uint8_t, 8U> kSuperblockMagic{{
    'Q', '3', 'X', 'A', 'O', 'T', 'B', '1'}};
constexpr std::array<std::uint8_t, 8U> kRecordMagic{{
    'Q', '3', 'X', 'A', 'O', 'T', 'R', '1'}};
constexpr std::uint16_t kLayerMajorGateUpThenDown = 1U;

constexpr std::size_t kSuperblockRepositoryOffset = 72U;
constexpr std::size_t kSuperblockRevisionOffset =
    kSuperblockRepositoryOffset + kPinnedStringCapacity;
constexpr std::size_t kSuperblockCheckpointDigestOffset =
    kSuperblockRevisionOffset + kPinnedStringCapacity;
constexpr std::size_t kSuperblockCatalogDigestOffset =
    kSuperblockCheckpointDigestOffset + kHeaderDigestBytes;
constexpr std::size_t kSuperblockRecordHeaderCatalogDigestOffset =
    kSuperblockCatalogDigestOffset + kHeaderDigestBytes;
constexpr std::size_t kSuperblockReservedOffset =
    kSuperblockHeaderDigestOffset + kHeaderDigestBytes;

constexpr std::size_t kRecordFixedReservedOffset = 104U;
constexpr std::size_t kManifestEncodedBytes = 430U;
constexpr std::size_t kTransformReceiptEncodedBytes = 574U;
constexpr std::size_t kSourceBindingEncodedBytes = 105U;
constexpr std::size_t kManifestFixedEncodedBytes = 42U;
constexpr std::size_t kManifestTailEncodedBytes = 73U;
constexpr std::size_t kPartitionTransformReceiptEncodedBytes = 162U;
constexpr std::size_t kTransformReceiptFixedEncodedBytes = 28U;
constexpr std::size_t kTransformReceiptTailEncodedBytes = 60U;

static_assert(kSuperblockCheckpointDigestOffset == 200U);
static_assert(kSuperblockCatalogDigestOffset == 232U);
static_assert(kSuperblockRecordHeaderCatalogDigestOffset == 264U);
static_assert(kSuperblockHeaderDigestOffset == 296U);
static_assert(kSuperblockReservedOffset == 328U);
static_assert(kManifestEncodedBytes ==
              kManifestFixedEncodedBytes +
                  kernels::kSm87TargetAotProjectionPackedMaxPartitions *
                      kSourceBindingEncodedBytes +
                  kManifestTailEncodedBytes);
static_assert(kTransformReceiptEncodedBytes ==
              kTransformReceiptFixedEncodedBytes +
                  kernels::kSm87TargetAotProjectionPackedMaxPartitions *
                      kPartitionTransformReceiptEncodedBytes +
                  kTransformReceiptTailEncodedBytes);
static_assert(kManifestEncodedBytes <= kRecordManifestSectionBytes);
static_assert(kTransformReceiptEncodedBytes <=
              kRecordTransformReceiptSectionBytes);
static_assert(sizeof(off_t) >= sizeof(std::int64_t));

std::atomic<std::uint64_t> g_persistence_file_serial{1U};

[[nodiscard]] std::string errno_message(const std::string_view operation,
                                        const int error) {
  return std::string(operation) + ": " + std::strerror(error);
}

[[nodiscard]] bool fsync_nointr(const int fd) noexcept {
  int status = 0;
  do {
    status = ::fsync(fd);
  } while (status != 0 && errno == EINTR);
  return status == 0;
}

[[nodiscard]] bool stale_temporary_absent(
    const int directory_fd, const std::string& final_name,
    std::string& message) {
  const int scan_fd = ::dup(directory_fd);
  if (scan_fd < 0) {
    message = errno_message(
        "duplicating target-AOT bundle directory for stale-temp audit",
        errno);
    return false;
  }
  DIR* const directory = ::fdopendir(scan_fd);
  if (directory == nullptr) {
    const int error = errno;
    (void)::close(scan_fd);
    message = errno_message(
        "opening target-AOT bundle directory for stale-temp audit", error);
    return false;
  }
  const std::string prefix = "." + final_name + ".q3x-tmp-";
  bool ok = true;
  errno = 0;
  while (const dirent* const entry = ::readdir(directory)) {
    const std::string_view name(entry->d_name);
    if (name.size() < prefix.size() ||
        name.compare(0U, prefix.size(), prefix) != 0) {
      continue;
    }
    struct stat status {};
    const bool stat_ok =
        ::fstatat(directory_fd, entry->d_name, &status,
                  AT_SYMLINK_NOFOLLOW) == 0;
    message = "stale target-AOT temporary file requires manual audit: " +
              std::string(name);
    if (stat_ok) {
      message += " size=" +
                 std::to_string(static_cast<long long>(status.st_size)) +
                 " uid=" +
                 std::to_string(static_cast<unsigned long long>(status.st_uid)) +
                 " mtime=" +
                 std::to_string(static_cast<long long>(status.st_mtim.tv_sec));
    }
    ok = false;
    break;
  }
  const int scan_error = errno;
  if (::closedir(directory) != 0 && ok) {
    message = errno_message(
        "closing target-AOT stale-temp directory audit failed", errno);
    return false;
  }
  if (ok && scan_error != 0) {
    message = errno_message(
        "reading target-AOT bundle directory for stale-temp audit",
        scan_error);
    return false;
  }
  return ok;
}

class ByteWriter final {
 public:
  ByteWriter(std::uint8_t* const data, const std::size_t begin,
             const std::size_t end) noexcept
      : data_(data), cursor_(begin), end_(end) {}

  [[nodiscard]] bool put_u8(const std::uint8_t value) noexcept {
    if (cursor_ >= end_) {
      return false;
    }
    data_[cursor_++] = value;
    return true;
  }

  [[nodiscard]] bool put_u16(const std::uint16_t value) noexcept {
    return put_unsigned(value);
  }

  [[nodiscard]] bool put_u32(const std::uint32_t value) noexcept {
    return put_unsigned(value);
  }

  [[nodiscard]] bool put_u64(const std::uint64_t value) noexcept {
    return put_unsigned(value);
  }

  [[nodiscard]] bool put_bool(const bool value) noexcept {
    return put_u8(value ? 1U : 0U);
  }

  template <std::size_t Size>
  [[nodiscard]] bool put_array(
      const std::array<std::uint8_t, Size>& value) noexcept {
    if (Size > end_ - cursor_) {
      return false;
    }
    std::copy(value.begin(), value.end(), data_ + cursor_);
    cursor_ += Size;
    return true;
  }

  [[nodiscard]] bool put_digest(const PersistenceDigest& digest) noexcept {
    return put_array(digest.bytes);
  }

  [[nodiscard]] bool put_string_slot(const std::string_view value,
                                     const std::size_t capacity) noexcept {
    if (value.size() > capacity || capacity > end_ - cursor_) {
      return false;
    }
    std::copy(value.begin(), value.end(), data_ + cursor_);
    cursor_ += capacity;
    return true;
  }

  [[nodiscard]] bool seek(const std::size_t position) noexcept {
    if (position < cursor_ || position > end_) {
      return false;
    }
    cursor_ = position;
    return true;
  }

  [[nodiscard]] std::size_t position() const noexcept { return cursor_; }

 private:
  template <typename Unsigned>
  [[nodiscard]] bool put_unsigned(Unsigned value) noexcept {
    if (sizeof(Unsigned) > end_ - cursor_) {
      return false;
    }
    for (std::size_t index = 0U; index < sizeof(Unsigned); ++index) {
      data_[cursor_++] = static_cast<std::uint8_t>(value >> (8U * index));
    }
    return true;
  }

  std::uint8_t* data_ = nullptr;
  std::size_t cursor_ = 0U;
  std::size_t end_ = 0U;
};

class ByteReader final {
 public:
  ByteReader(const std::uint8_t* const data, const std::size_t begin,
             const std::size_t end) noexcept
      : data_(data), cursor_(begin), end_(end) {}

  [[nodiscard]] bool get_u8(std::uint8_t* const value) noexcept {
    if (value == nullptr || cursor_ >= end_) {
      return false;
    }
    *value = data_[cursor_++];
    return true;
  }

  [[nodiscard]] bool get_u16(std::uint16_t* const value) noexcept {
    return get_unsigned(value);
  }

  [[nodiscard]] bool get_u32(std::uint32_t* const value) noexcept {
    return get_unsigned(value);
  }

  [[nodiscard]] bool get_u64(std::uint64_t* const value) noexcept {
    return get_unsigned(value);
  }

  [[nodiscard]] bool get_bool(bool* const value) noexcept {
    std::uint8_t encoded = 0U;
    if (value == nullptr || !get_u8(&encoded) || encoded > 1U) {
      return false;
    }
    *value = encoded != 0U;
    return true;
  }

  template <std::size_t Size>
  [[nodiscard]] bool get_array(
      std::array<std::uint8_t, Size>* const value) noexcept {
    if (value == nullptr || Size > end_ - cursor_) {
      return false;
    }
    std::copy_n(data_ + cursor_, Size, value->begin());
    cursor_ += Size;
    return true;
  }

  [[nodiscard]] bool get_digest(PersistenceDigest* const digest) noexcept {
    return digest != nullptr && get_array(&digest->bytes);
  }

  [[nodiscard]] bool seek(const std::size_t position) noexcept {
    if (position < cursor_ || position > end_) {
      return false;
    }
    cursor_ = position;
    return true;
  }

  [[nodiscard]] std::size_t position() const noexcept { return cursor_; }

 private:
  template <typename Unsigned>
  [[nodiscard]] bool get_unsigned(Unsigned* const value) noexcept {
    if (value == nullptr || sizeof(Unsigned) > end_ - cursor_) {
      return false;
    }
    std::uint64_t decoded = 0U;
    for (std::size_t index = 0U; index < sizeof(Unsigned); ++index) {
      decoded |= static_cast<std::uint64_t>(data_[cursor_++])
                 << (8U * index);
    }
    *value = static_cast<Unsigned>(decoded);
    return true;
  }

  const std::uint8_t* data_ = nullptr;
  std::size_t cursor_ = 0U;
  std::size_t end_ = 0U;
};

[[nodiscard]] bool bytes_equal(const std::uint8_t* const input,
                               const std::array<std::uint8_t, 8U>& expected)
    noexcept {
  return input != nullptr &&
         std::equal(expected.begin(), expected.end(), input);
}

[[nodiscard]] bool all_zero(const std::uint8_t* const input,
                            const std::size_t begin,
                            const std::size_t end) noexcept {
  return input != nullptr && begin <= end &&
         std::all_of(input + begin, input + end,
                     [](const std::uint8_t byte) { return byte == 0U; });
}

[[nodiscard]] bool string_slot_matches(const std::uint8_t* const input,
                                       const std::size_t offset,
                                       const std::size_t capacity,
                                       const std::string_view expected)
    noexcept {
  if (input == nullptr || expected.size() > capacity ||
      !std::equal(expected.begin(), expected.end(), input + offset)) {
    return false;
  }
  return all_zero(input, offset + expected.size(), offset + capacity);
}

[[nodiscard]] PersistenceDigest compute_header_digest(
    const std::uint8_t* const input, const std::size_t header_bytes,
    const std::size_t digest_offset) noexcept {
  PersistenceDigest result;
  if (input == nullptr || digest_offset > header_bytes ||
      kHeaderDigestBytes > header_bytes - digest_offset) {
    return result;
  }
  core::Sha256 hasher;
  const std::array<std::uint8_t, kHeaderDigestBytes> zero_digest{};
  const bool ok = hasher.update(input, digest_offset) &&
                  hasher.update(zero_digest.data(), zero_digest.size()) &&
                  hasher.update(input + digest_offset + kHeaderDigestBytes,
                                header_bytes - digest_offset -
                                    kHeaderDigestBytes);
  if (!ok) {
    return result;
  }
  const core::Sha256Digest digest = hasher.finalize();
  result.bytes = digest.bytes;
  return result;
}

void write_digest_at(std::uint8_t* const output, const std::size_t offset,
                     const PersistenceDigest& digest) noexcept {
  std::copy(digest.bytes.begin(), digest.bytes.end(), output + offset);
}

[[nodiscard]] PersistenceDigest read_digest_at(
    const std::uint8_t* const input, const std::size_t offset) noexcept {
  PersistenceDigest digest;
  std::copy_n(input + offset, digest.bytes.size(), digest.bytes.begin());
  return digest;
}

[[nodiscard]] bool valid_projection_role_value(
    const std::uint8_t value) noexcept {
  return value < static_cast<std::uint8_t>(
                     kernels::Sm87TargetAotProjectionRole::kCount);
}

[[nodiscard]] bool valid_logical_role_value(const std::uint8_t value) noexcept {
  return value <= static_cast<std::uint8_t>(
                      kernels::Sm87TargetAotLogicalRole::
                          kFp8AttentionOutput);
}

[[nodiscard]] bool valid_plan_identity_value(
    const std::uint16_t value) noexcept {
  return value <= static_cast<std::uint16_t>(
                      kernels::Sm87TargetAotProjectionPackedPlanIdentity::
                          kFp8AttentionOutputM128N256K64V1);
}

[[nodiscard]] bool valid_layout_identity_value(
    const std::uint16_t value) noexcept {
  return value <= static_cast<std::uint16_t>(
                      kernels::Sm87TargetAotProjectionPackedLayoutIdentity::
                          kConsumerN64K16LaneComponentV1);
}

[[nodiscard]] bool valid_encoding_value(const std::uint8_t value) noexcept {
  return value <= static_cast<std::uint8_t>(
                      kernels::Sm87TargetAotProjectionEncoding::
                          kFp8E4M3FnTensorScale);
}

[[nodiscard]] bool valid_transform_identity_value(
    const std::uint16_t value) noexcept {
  return value <= static_cast<std::uint16_t>(
                      kernels::Sm87TargetAotProjectionPackedTransformIdentity::
                          kCanonicalNkToConsumerN64K16LaneComponentV1);
}

[[nodiscard]] bool write_source_binding(
    ByteWriter& writer,
    const kernels::Sm87TargetAotProjectionPackedSourceBinding& source)
    noexcept {
  return writer.put_u8(static_cast<std::uint8_t>(source.logical_role)) &&
         writer.put_u32(source.partition_index) &&
         writer.put_u64(source.tensor_identity) &&
         writer.put_digest(source.weight_digest) &&
         writer.put_digest(source.scale_digest) &&
         writer.put_u32(source.output_features) &&
         writer.put_u32(source.input_features) &&
         writer.put_u32(source.tensor_scale_bits) &&
         writer.put_u64(source.payload_offset) &&
         writer.put_u64(source.payload_bytes);
}

[[nodiscard]] PersistenceError read_source_binding(
    ByteReader& reader,
    kernels::Sm87TargetAotProjectionPackedSourceBinding* const source)
    noexcept {
  if (source == nullptr) {
    return PersistenceError::kInvalidArgument;
  }
  std::uint8_t logical_role = 0U;
  if (!reader.get_u8(&logical_role) ||
      !valid_logical_role_value(logical_role)) {
    return PersistenceError::kInvalidEnum;
  }
  source->logical_role =
      static_cast<kernels::Sm87TargetAotLogicalRole>(logical_role);
  if (!reader.get_u32(&source->partition_index) ||
      !reader.get_u64(&source->tensor_identity) ||
      !reader.get_digest(&source->weight_digest) ||
      !reader.get_digest(&source->scale_digest) ||
      !reader.get_u32(&source->output_features) ||
      !reader.get_u32(&source->input_features) ||
      !reader.get_u32(&source->tensor_scale_bits) ||
      !reader.get_u64(&source->payload_offset) ||
      !reader.get_u64(&source->payload_bytes)) {
    return PersistenceError::kTruncated;
  }
  return PersistenceError::kNone;
}

[[nodiscard]] bool write_manifest(
    ByteWriter& writer,
    const kernels::Sm87TargetAotProjectionPackedManifest& manifest)
    noexcept {
  if (!writer.put_array(manifest.magic) ||
      !writer.put_u16(manifest.abi_major) ||
      !writer.put_u16(manifest.abi_minor) ||
      !writer.put_u32(manifest.header_bytes) ||
      !writer.put_u64(manifest.artifact_identity) ||
      !writer.put_u8(static_cast<std::uint8_t>(manifest.role)) ||
      !writer.put_u16(static_cast<std::uint16_t>(manifest.plan_identity)) ||
      !writer.put_u16(static_cast<std::uint16_t>(manifest.layout_identity)) ||
      !writer.put_u8(static_cast<std::uint8_t>(manifest.encoding)) ||
      !writer.put_u64(manifest.source_inventory_identity) ||
      !writer.put_u32(manifest.source_count)) {
    return false;
  }
  for (const auto& source : manifest.sources) {
    if (!write_source_binding(writer, source)) {
      return false;
    }
  }
  return writer.put_u64(manifest.payload_offset) &&
         writer.put_u64(manifest.payload_bytes) &&
         writer.put_u64(manifest.artifact_bytes) &&
         writer.put_u32(manifest.payload_alignment) &&
         writer.put_digest(manifest.payload_digest) &&
         writer.put_bool(manifest.token_count_independent) &&
         writer.put_bool(manifest.cuda_implementation_present) &&
         writer.put_bool(manifest.static_resources_qualified) &&
         writer.put_bool(manifest.numerical_contract_qualified) &&
         writer.put_bool(manifest.production_dispatch_eligible) &&
         writer.put_u64(manifest.seal.value);
}

[[nodiscard]] PersistenceError read_manifest(
    ByteReader& reader,
    kernels::Sm87TargetAotProjectionPackedManifest* const manifest)
    noexcept {
  if (manifest == nullptr) {
    return PersistenceError::kInvalidArgument;
  }
  if (!reader.get_array(&manifest->magic) ||
      !reader.get_u16(&manifest->abi_major) ||
      !reader.get_u16(&manifest->abi_minor) ||
      !reader.get_u32(&manifest->header_bytes) ||
      !reader.get_u64(&manifest->artifact_identity)) {
    return PersistenceError::kTruncated;
  }
  std::uint8_t role = 0U;
  std::uint16_t plan = 0U;
  std::uint16_t layout = 0U;
  std::uint8_t encoding = 0U;
  if (!reader.get_u8(&role) || !valid_projection_role_value(role) ||
      !reader.get_u16(&plan) || !valid_plan_identity_value(plan) ||
      !reader.get_u16(&layout) || !valid_layout_identity_value(layout) ||
      !reader.get_u8(&encoding) || !valid_encoding_value(encoding)) {
    return PersistenceError::kInvalidEnum;
  }
  manifest->role =
      static_cast<kernels::Sm87TargetAotProjectionRole>(role);
  manifest->plan_identity =
      static_cast<kernels::Sm87TargetAotProjectionPackedPlanIdentity>(plan);
  manifest->layout_identity =
      static_cast<kernels::Sm87TargetAotProjectionPackedLayoutIdentity>(
          layout);
  manifest->encoding =
      static_cast<kernels::Sm87TargetAotProjectionEncoding>(encoding);
  if (!reader.get_u64(&manifest->source_inventory_identity) ||
      !reader.get_u32(&manifest->source_count)) {
    return PersistenceError::kTruncated;
  }
  for (auto& source : manifest->sources) {
    const PersistenceError error = read_source_binding(reader, &source);
    if (error != PersistenceError::kNone) {
      return error;
    }
  }
  if (!reader.get_u64(&manifest->payload_offset) ||
      !reader.get_u64(&manifest->payload_bytes) ||
      !reader.get_u64(&manifest->artifact_bytes) ||
      !reader.get_u32(&manifest->payload_alignment) ||
      !reader.get_digest(&manifest->payload_digest)) {
    return PersistenceError::kTruncated;
  }
  if (!reader.get_bool(&manifest->token_count_independent) ||
      !reader.get_bool(&manifest->cuda_implementation_present) ||
      !reader.get_bool(&manifest->static_resources_qualified) ||
      !reader.get_bool(&manifest->numerical_contract_qualified) ||
      !reader.get_bool(&manifest->production_dispatch_eligible)) {
    return PersistenceError::kInvalidBoolean;
  }
  if (!reader.get_u64(&manifest->seal.value)) {
    return PersistenceError::kTruncated;
  }
  return PersistenceError::kNone;
}

[[nodiscard]] bool write_partition_transform_receipt(
    ByteWriter& writer,
    const kernels::Sm87TargetAotProjectionPackedPartitionTransformReceipt&
        receipt) noexcept {
  return writer.put_u8(static_cast<std::uint8_t>(receipt.logical_role)) &&
         writer.put_u32(receipt.partition_index) &&
         writer.put_u64(receipt.tensor_identity) &&
         writer.put_digest(receipt.observed_source_weight_digest) &&
         writer.put_digest(receipt.observed_source_scale_digest) &&
         writer.put_u64(receipt.source_weight_bytes_hashed) &&
         writer.put_u64(receipt.source_scale_bytes_hashed) &&
         writer.put_u64(receipt.repacked_weight_values) &&
         writer.put_u64(receipt.repacked_block_scale_values) &&
         writer.put_u64(
             receipt.source_block_scale_e4m3fn_bytes_scanned) &&
         writer.put_u64(
             receipt.payload_block_scale_e4m3fn_bytes_scanned) &&
         writer.put_u64(receipt.source_forbidden_block_scale_codes) &&
         writer.put_u64(receipt.payload_forbidden_block_scale_codes) &&
         writer.put_u64(receipt.payload_offset) &&
         writer.put_u64(receipt.payload_bytes) &&
         writer.put_bool(
             receipt.source_digests_computed_from_tensor_bytes) &&
         writer.put_bool(receipt.canonical_address_bijection_applied) &&
         writer.put_bool(receipt.bit_exact_weight_permutation) &&
         writer.put_bool(receipt.bit_exact_block_scale_permutation) &&
         writer.put_bool(receipt.tensor_scale_kept_external);
}

[[nodiscard]] PersistenceError read_partition_transform_receipt(
    ByteReader& reader,
    kernels::Sm87TargetAotProjectionPackedPartitionTransformReceipt* const
        receipt) noexcept {
  if (receipt == nullptr) {
    return PersistenceError::kInvalidArgument;
  }
  std::uint8_t logical_role = 0U;
  if (!reader.get_u8(&logical_role) ||
      !valid_logical_role_value(logical_role)) {
    return PersistenceError::kInvalidEnum;
  }
  receipt->logical_role =
      static_cast<kernels::Sm87TargetAotLogicalRole>(logical_role);
  if (!reader.get_u32(&receipt->partition_index) ||
      !reader.get_u64(&receipt->tensor_identity) ||
      !reader.get_digest(&receipt->observed_source_weight_digest) ||
      !reader.get_digest(&receipt->observed_source_scale_digest) ||
      !reader.get_u64(&receipt->source_weight_bytes_hashed) ||
      !reader.get_u64(&receipt->source_scale_bytes_hashed) ||
      !reader.get_u64(&receipt->repacked_weight_values) ||
      !reader.get_u64(&receipt->repacked_block_scale_values) ||
      !reader.get_u64(
          &receipt->source_block_scale_e4m3fn_bytes_scanned) ||
      !reader.get_u64(
          &receipt->payload_block_scale_e4m3fn_bytes_scanned) ||
      !reader.get_u64(&receipt->source_forbidden_block_scale_codes) ||
      !reader.get_u64(&receipt->payload_forbidden_block_scale_codes) ||
      !reader.get_u64(&receipt->payload_offset) ||
      !reader.get_u64(&receipt->payload_bytes)) {
    return PersistenceError::kTruncated;
  }
  if (!reader.get_bool(
          &receipt->source_digests_computed_from_tensor_bytes) ||
      !reader.get_bool(&receipt->canonical_address_bijection_applied) ||
      !reader.get_bool(&receipt->bit_exact_weight_permutation) ||
      !reader.get_bool(&receipt->bit_exact_block_scale_permutation) ||
      !reader.get_bool(&receipt->tensor_scale_kept_external)) {
    return PersistenceError::kInvalidBoolean;
  }
  return PersistenceError::kNone;
}

[[nodiscard]] bool write_transform_receipt(
    ByteWriter& writer,
    const kernels::Sm87TargetAotProjectionPackedTransformReceipt& receipt)
    noexcept {
  if (!writer.put_u64(receipt.artifact_identity) ||
      !writer.put_u64(receipt.source_inventory_identity) ||
      !writer.put_u8(static_cast<std::uint8_t>(receipt.role)) ||
      !writer.put_u16(static_cast<std::uint16_t>(receipt.plan_identity)) ||
      !writer.put_u16(static_cast<std::uint16_t>(receipt.layout_identity)) ||
      !writer.put_u8(static_cast<std::uint8_t>(receipt.encoding)) ||
      !writer.put_u16(
          static_cast<std::uint16_t>(receipt.transform_identity)) ||
      !writer.put_u32(receipt.partition_count)) {
    return false;
  }
  for (const auto& partition : receipt.partitions) {
    if (!write_partition_transform_receipt(writer, partition)) {
      return false;
    }
  }
  return writer.put_u64(receipt.payload.artifact_identity) &&
         writer.put_u64(receipt.payload.observed_payload_offset) &&
         writer.put_u64(receipt.payload.observed_payload_bytes) &&
         writer.put_digest(receipt.payload.observed_payload_digest) &&
         writer.put_bool(
             receipt.payload.digest_computed_from_payload_bytes) &&
         writer.put_bool(receipt.deterministic_transform) &&
         writer.put_bool(receipt.no_arithmetic_conversion) &&
         writer.put_bool(receipt.no_request_time_repacking);
}

[[nodiscard]] PersistenceError read_transform_receipt(
    ByteReader& reader,
    kernels::Sm87TargetAotProjectionPackedTransformReceipt* const receipt)
    noexcept {
  if (receipt == nullptr) {
    return PersistenceError::kInvalidArgument;
  }
  if (!reader.get_u64(&receipt->artifact_identity) ||
      !reader.get_u64(&receipt->source_inventory_identity)) {
    return PersistenceError::kTruncated;
  }
  std::uint8_t role = 0U;
  std::uint16_t plan = 0U;
  std::uint16_t layout = 0U;
  std::uint8_t encoding = 0U;
  std::uint16_t transform = 0U;
  if (!reader.get_u8(&role) || !valid_projection_role_value(role) ||
      !reader.get_u16(&plan) || !valid_plan_identity_value(plan) ||
      !reader.get_u16(&layout) || !valid_layout_identity_value(layout) ||
      !reader.get_u8(&encoding) || !valid_encoding_value(encoding) ||
      !reader.get_u16(&transform) ||
      !valid_transform_identity_value(transform)) {
    return PersistenceError::kInvalidEnum;
  }
  receipt->role =
      static_cast<kernels::Sm87TargetAotProjectionRole>(role);
  receipt->plan_identity =
      static_cast<kernels::Sm87TargetAotProjectionPackedPlanIdentity>(plan);
  receipt->layout_identity =
      static_cast<kernels::Sm87TargetAotProjectionPackedLayoutIdentity>(
          layout);
  receipt->encoding =
      static_cast<kernels::Sm87TargetAotProjectionEncoding>(encoding);
  receipt->transform_identity = static_cast<
      kernels::Sm87TargetAotProjectionPackedTransformIdentity>(transform);
  if (!reader.get_u32(&receipt->partition_count)) {
    return PersistenceError::kTruncated;
  }
  for (auto& partition : receipt->partitions) {
    const PersistenceError error =
        read_partition_transform_receipt(reader, &partition);
    if (error != PersistenceError::kNone) {
      return error;
    }
  }
  if (!reader.get_u64(&receipt->payload.artifact_identity) ||
      !reader.get_u64(&receipt->payload.observed_payload_offset) ||
      !reader.get_u64(&receipt->payload.observed_payload_bytes) ||
      !reader.get_digest(&receipt->payload.observed_payload_digest)) {
    return PersistenceError::kTruncated;
  }
  if (!reader.get_bool(
          &receipt->payload.digest_computed_from_payload_bytes) ||
      !reader.get_bool(&receipt->deterministic_transform) ||
      !reader.get_bool(&receipt->no_arithmetic_conversion) ||
      !reader.get_bool(&receipt->no_request_time_repacking)) {
    return PersistenceError::kInvalidBoolean;
  }
  return PersistenceError::kNone;
}

[[nodiscard]] kernels::Sm87TargetAotProjectionPackedSourceInventory
inventory_from_manifest(
    const kernels::Sm87TargetAotProjectionPackedManifest& manifest) noexcept {
  kernels::Sm87TargetAotProjectionPackedSourceInventory inventory;
  inventory.identity = manifest.source_inventory_identity;
  inventory.role = manifest.role;
  inventory.source_count = manifest.source_count;
  inventory.sources = manifest.sources;
  return inventory;
}

[[nodiscard]] bool record_structurally_valid(
    const std::size_t layer_index,
    const kernels::Sm87TargetAotProjectionPackedManifest& manifest,
    const kernels::Sm87TargetAotProjectionPackedTransformReceipt& receipt,
    kernels::Sm87TargetAotProjectionPackedSourceInventory* const inventory)
    noexcept {
  if (inventory == nullptr || layer_index >= kPersistedLayerCount ||
      !persistence_role_is_nvfp4(manifest.role)) {
    return false;
  }
  *inventory = inventory_from_manifest(manifest);
  return kernels::sm87_target_aot_projection_validate_packed_manifest(
             manifest, *inventory) &&
         kernels::sm87_target_aot_projection_validate_transform_receipt(
             manifest, *inventory, receipt);
}

}  // namespace

UniqueFd::UniqueFd(const int fd) noexcept : fd_(fd) {}

UniqueFd::~UniqueFd() { reset(); }

UniqueFd::UniqueFd(UniqueFd&& other) noexcept : fd_(other.release()) {}

UniqueFd& UniqueFd::operator=(UniqueFd&& other) noexcept {
  if (this != &other) {
    reset(other.release());
  }
  return *this;
}

int UniqueFd::get() const noexcept { return fd_; }

UniqueFd::operator bool() const noexcept { return fd_ >= 0; }

int UniqueFd::release() noexcept {
  const int result = fd_;
  fd_ = -1;
  return result;
}

void UniqueFd::reset(const int next) noexcept {
  if (fd_ >= 0) {
    // Linux has already released the descriptor when close reports EINTR;
    // retrying could close an unrelated descriptor reused by another thread.
    (void)::close(fd_);
  }
  fd_ = next;
}

bool secure_absolute_parent(const std::filesystem::path& path,
                            SecureParent& result,
                            std::string& message) {
  result = {};
  if (!path.is_absolute() || path.filename().empty()) {
    message = "target-AOT bundle path must be an absolute file path";
    return false;
  }
  std::vector<std::string> components;
  for (const auto& component_path : path.relative_path()) {
    const std::string component = component_path.native();
    if (component.empty() || component == "." || component == ".." ||
        component.find('\0') != std::string::npos) {
      message = "target-AOT bundle path contains a forbidden component";
      return false;
    }
    components.push_back(component);
  }
  if (components.empty()) {
    message = "target-AOT bundle path has no filename";
    return false;
  }

  UniqueFd directory(
      ::open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
  if (!directory) {
    const int error = errno;
    message = errno_message("opening the filesystem root failed", error);
    return false;
  }
  for (std::size_t index = 0U; index + 1U < components.size(); ++index) {
    const int next = ::openat(directory.get(), components[index].c_str(),
                              O_RDONLY | O_DIRECTORY | O_CLOEXEC |
                                  O_NOFOLLOW);
    if (next < 0) {
      const int error = errno;
      message = errno_message(
          "opening a target-AOT bundle parent directory failed", error);
      return false;
    }
    directory.reset(next);
  }
  result.directory = std::move(directory);
  result.filename = components.back();
  return true;
}

bool same_snapshot(const FileSnapshot& left,
                   const FileSnapshot& right) noexcept {
  return left.device == right.device && left.inode == right.inode &&
         left.mode == right.mode && left.links == right.links &&
         left.bytes == right.bytes &&
         left.modified_seconds == right.modified_seconds &&
         left.modified_nanoseconds == right.modified_nanoseconds &&
         left.changed_seconds == right.changed_seconds &&
         left.changed_nanoseconds == right.changed_nanoseconds;
}

bool snapshot_regular_file_exact(const int fd,
                                 const std::uint64_t expected_bytes,
                                 FileSnapshot& snapshot,
                                 std::string& message) {
  struct stat status {};
  if (fd < 0 || ::fstat(fd, &status) != 0) {
    const int error = fd < 0 ? EBADF : errno;
    message = errno_message("fstat on target-AOT bundle failed", error);
    return false;
  }
  if (!S_ISREG(status.st_mode) || status.st_nlink != 1U ||
      status.st_size < 0 ||
      static_cast<std::uint64_t>(status.st_size) != expected_bytes) {
    message = "target-AOT bundle is not one regular, single-link, exact-size "
              "file";
    return false;
  }
  snapshot.device = static_cast<std::uint64_t>(status.st_dev);
  snapshot.inode = static_cast<std::uint64_t>(status.st_ino);
  snapshot.mode = static_cast<std::uint64_t>(status.st_mode);
  snapshot.links = static_cast<std::uint64_t>(status.st_nlink);
  snapshot.bytes = static_cast<std::uint64_t>(status.st_size);
  snapshot.modified_seconds = status.st_mtim.tv_sec;
  snapshot.modified_nanoseconds = status.st_mtim.tv_nsec;
  snapshot.changed_seconds = status.st_ctim.tv_sec;
  snapshot.changed_nanoseconds = status.st_ctim.tv_nsec;
  return true;
}

bool open_regular_file_exact(const std::filesystem::path& path,
                             const std::uint64_t expected_bytes,
                             UniqueFd& file, FileSnapshot& snapshot,
                             std::string& message) {
  file.reset();
  SecureParent parent;
  if (!secure_absolute_parent(path, parent, message)) {
    return false;
  }
  const int fd = ::openat(parent.directory.get(), parent.filename.c_str(),
                          O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
  if (fd < 0) {
    const int error = errno;
    message = errno_message("opening target-AOT bundle failed", error);
    return false;
  }
  file.reset(fd);
  if (!snapshot_regular_file_exact(file.get(), expected_bytes, snapshot,
                                   message)) {
    file.reset();
    return false;
  }
  return true;
}

bool pread_exact_counted(const int fd, void* const output,
                         const std::size_t bytes,
                         const std::uint64_t offset,
                         std::uint64_t& bytes_read,
                         std::string& message) {
  if (fd < 0 || (output == nullptr && bytes != 0U) ||
      offset > static_cast<std::uint64_t>(
                   std::numeric_limits<off_t>::max()) ||
      bytes > static_cast<std::uint64_t>(
                  std::numeric_limits<off_t>::max()) - offset) {
    message = "target-AOT bundle read range is invalid";
    return false;
  }
  auto* cursor = static_cast<std::uint8_t*>(output);
  std::size_t completed = 0U;
  while (completed < bytes) {
    const std::size_t request = std::min(
        bytes - completed,
        static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
    const ssize_t observed =
        ::pread(fd, cursor + completed, request,
                static_cast<off_t>(offset + completed));
    if (observed < 0) {
      if (errno == EINTR) {
        continue;
      }
      const int error = errno;
      message = errno_message("reading target-AOT bundle failed", error);
      return false;
    }
    if (observed == 0) {
      message = "target-AOT bundle ended before the requested range";
      return false;
    }
    const auto consumed = static_cast<std::size_t>(observed);
    completed += consumed;
    if (bytes_read > std::numeric_limits<std::uint64_t>::max() - consumed) {
      message = "target-AOT bundle byte accounting overflowed";
      return false;
    }
    bytes_read += consumed;
  }
  return true;
}

bool pwrite_exact_counted(const int fd, const void* const input,
                          const std::size_t bytes,
                          const std::uint64_t offset,
                          std::uint64_t& bytes_written,
                          std::string& message) {
  if (fd < 0 || (input == nullptr && bytes != 0U) ||
      offset > static_cast<std::uint64_t>(
                   std::numeric_limits<off_t>::max()) ||
      bytes > static_cast<std::uint64_t>(
                  std::numeric_limits<off_t>::max()) - offset) {
    message = "target-AOT bundle write range is invalid";
    return false;
  }
  const auto* cursor = static_cast<const std::uint8_t*>(input);
  std::size_t completed = 0U;
  while (completed < bytes) {
    const std::size_t request = std::min(
        bytes - completed,
        static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
    const ssize_t observed =
        ::pwrite(fd, cursor + completed, request,
                 static_cast<off_t>(offset + completed));
    if (observed < 0) {
      if (errno == EINTR) {
        continue;
      }
      const int error = errno;
      message = errno_message("writing target-AOT bundle failed", error);
      return false;
    }
    if (observed == 0) {
      message = "target-AOT bundle write made no progress";
      return false;
    }
    const auto consumed = static_cast<std::size_t>(observed);
    completed += consumed;
    if (bytes_written >
        std::numeric_limits<std::uint64_t>::max() - consumed) {
      message = "target-AOT bundle write accounting overflowed";
      return false;
    }
    bytes_written += consumed;
  }
  return true;
}

AtomicCreateOnlyFile::~AtomicCreateOnlyFile() { abandon(); }

bool AtomicCreateOnlyFile::open_create_only(
    const std::filesystem::path& output,
    const std::uint64_t expected_bytes, const bool reserve_storage,
    std::string& message) {
  if (is_open() || expected_bytes == 0U ||
      expected_bytes > static_cast<std::uint64_t>(
                           std::numeric_limits<off_t>::max())) {
    message = is_open() ? "target-AOT atomic file was already initialized"
                        : "target-AOT atomic file size is invalid";
    return false;
  }
  SecureParent parent;
  if (!secure_absolute_parent(output, parent, message)) {
    return false;
  }
  struct stat existing {};
  if (::fstatat(parent.directory.get(), parent.filename.c_str(), &existing,
                AT_SYMLINK_NOFOLLOW) == 0) {
    message = "target-AOT bundle destination already exists";
    return false;
  }
  if (errno != ENOENT) {
    const int error = errno;
    message = errno_message(
        "checking target-AOT bundle destination failed", error);
    return false;
  }
  if (!stale_temporary_absent(parent.directory.get(), parent.filename,
                              message)) {
    return false;
  }

  const std::uint64_t serial =
      g_persistence_file_serial.fetch_add(1U, std::memory_order_relaxed);
  temporary_name_ = "." + parent.filename + ".q3x-tmp-" +
                    std::to_string(static_cast<unsigned long long>(
                        static_cast<std::uint64_t>(::getpid()))) +
                    "-" + std::to_string(
                                static_cast<unsigned long long>(serial));
  final_name_ = parent.filename;
  expected_bytes_ = expected_bytes;
  const int fd = ::openat(parent.directory.get(), temporary_name_.c_str(),
                          O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC,
                          S_IRUSR | S_IWUSR);
  if (fd < 0) {
    const int error = errno;
    temporary_name_.clear();
    final_name_.clear();
    expected_bytes_ = 0U;
    message = errno_message(
        "creating target-AOT bundle temporary file failed", error);
    return false;
  }
  fd_.reset(fd);
  directory_ = std::move(parent.directory);
  if (reserve_storage) {
    const int allocate_status = ::posix_fallocate(
        fd_.get(), 0, static_cast<off_t>(expected_bytes_));
    if (allocate_status != 0) {
      message = errno_message(
          "reserving target-AOT bundle storage failed", allocate_status);
      abandon();
      return false;
    }
  }
  if (::ftruncate(fd_.get(), static_cast<off_t>(expected_bytes_)) != 0) {
    const int error = errno;
    message = errno_message(
        "sizing target-AOT bundle temporary file failed", error);
    abandon();
    return false;
  }
  return true;
}

bool AtomicCreateOnlyFile::sync_data_and_drop_cache(
    const std::uint64_t offset, const std::uint64_t bytes,
    std::string& message) {
  if (!is_open() || bytes == 0U || offset > expected_bytes_ ||
      bytes > expected_bytes_ - offset ||
      offset > static_cast<std::uint64_t>(
                   std::numeric_limits<off_t>::max()) ||
      bytes > static_cast<std::uint64_t>(
                  std::numeric_limits<off_t>::max())) {
    message = "target-AOT bundle sync range is invalid";
    return false;
  }
  if (::fdatasync(fd_.get()) != 0) {
    const int error = errno;
    message = errno_message(
        "syncing one target-AOT bundle range failed", error);
    return false;
  }
  const int advise = ::posix_fadvise(
      fd_.get(), static_cast<off_t>(offset), static_cast<off_t>(bytes),
      POSIX_FADV_DONTNEED);
  if (advise != 0) {
    message = errno_message(
        "dropping one synced target-AOT bundle range from page cache failed",
        advise);
    return false;
  }
  return true;
}

AtomicPublishOutcome AtomicCreateOnlyFile::publish(std::string& message) {
  if (!is_open() || temporary_name_.empty() || final_name_.empty()) {
    message = "target-AOT atomic file is not ready for publication";
    return AtomicPublishOutcome::kNotPublished;
  }
  if (!fsync_nointr(fd_.get())) {
    const int error = errno;
    message = errno_message("syncing target-AOT bundle failed", error);
    return AtomicPublishOutcome::kNotPublished;
  }
  FileSnapshot snapshot;
  if (!snapshot_regular_file_exact(fd_.get(), expected_bytes_, snapshot,
                                   message)) {
    return AtomicPublishOutcome::kNotPublished;
  }
  if (::syscall(SYS_renameat2, directory_.get(), temporary_name_.c_str(),
                directory_.get(), final_name_.c_str(),
                RENAME_NOREPLACE) != 0) {
    const int error = errno;
    message = errno_message(
        "publishing target-AOT bundle without replacement failed", error);
    return AtomicPublishOutcome::kNotPublished;
  }
  // The complete file is visible only at the final name. If the directory
  // fsync fails, retain it and report uncertain durability; deleting a
  // possibly durable authenticated artifact would be less safe.
  temporary_name_.clear();
  if (!fsync_nointr(directory_.get())) {
    const int error = errno;
    message = errno_message(
        "target-AOT bundle final name is visible but directory durability is "
        "uncertain; do not retry or delete it before secure direct-load "
        "reauthentication",
        error);
    return AtomicPublishOutcome::kPublishedDurabilityUncertain;
  }
  return AtomicPublishOutcome::kPublishedDurable;
}

bool AtomicCreateOnlyFile::is_open() const noexcept {
  return static_cast<bool>(fd_) && static_cast<bool>(directory_) &&
         !final_name_.empty() && expected_bytes_ != 0U;
}

int AtomicCreateOnlyFile::fd() const noexcept { return fd_.get(); }

std::uint64_t AtomicCreateOnlyFile::expected_bytes() const noexcept {
  return expected_bytes_;
}

void AtomicCreateOnlyFile::abandon() noexcept {
  fd_.reset();
  if (directory_ && !temporary_name_.empty()) {
    (void)::unlinkat(directory_.get(), temporary_name_.c_str(), 0);
  }
  temporary_name_.clear();
  final_name_.clear();
  expected_bytes_ = 0U;
  directory_.reset();
}

PersistenceStatus encode_superblock(
    const PersistenceSuperblock& superblock, std::uint8_t* const output,
    const std::size_t output_bytes) noexcept {
  if (output == nullptr) {
    return {PersistenceError::kInvalidArgument};
  }
  if (output_bytes < kBundleSuperblockBytes) {
    return {PersistenceError::kOutputTooSmall};
  }
  std::fill_n(output, kBundleSuperblockBytes, std::uint8_t{0U});
  if (kernels::sm87_target_aot_projection_digest_is_zero(
          superblock.checkpoint_identity_sha256) ||
      kernels::sm87_target_aot_projection_digest_is_zero(
          superblock.authenticated_payload_catalog_sha256) ||
      kernels::sm87_target_aot_projection_digest_is_zero(
          superblock.record_header_catalog_sha256)) {
    return {PersistenceError::kInvalidArgument};
  }

  ByteWriter writer(output, 0U, kBundleSuperblockBytes);
  const bool encoded =
      writer.put_array(kSuperblockMagic) &&
      writer.put_u16(kPersistenceAbiMajor) &&
      writer.put_u16(kPersistenceAbiMinor) &&
      writer.put_u32(kLittleEndianTag) &&
      writer.put_u32(static_cast<std::uint32_t>(kBundleSuperblockBytes)) &&
      writer.put_u32(static_cast<std::uint32_t>(kRecordHeaderBytes)) &&
      writer.put_u16(kTargetSmMajor) && writer.put_u16(kTargetSmMinor) &&
      writer.put_u32(static_cast<std::uint32_t>(kPersistedLayerCount)) &&
      writer.put_u32(static_cast<std::uint32_t>(kPersistedArtifactCount)) &&
      writer.put_u32(static_cast<std::uint32_t>(kPersistedSourceCount)) &&
      writer.put_u16(kLayerMajorGateUpThenDown) &&
      writer.put_u16(
          static_cast<std::uint16_t>(kPinnedModelRepository.size())) &&
      writer.put_u16(
          static_cast<std::uint16_t>(kPinnedModelRevision.size())) &&
      writer.seek(48U) && writer.put_u64(kPersistedArenaBytes) &&
      writer.put_u64(kPersistedBundleBytes) &&
      writer.put_u64(kBundleSuperblockBytes) &&
      writer.put_string_slot(kPinnedModelRepository,
                             kPinnedStringCapacity) &&
      writer.put_string_slot(kPinnedModelRevision, kPinnedStringCapacity) &&
      writer.put_digest(superblock.checkpoint_identity_sha256) &&
      writer.put_digest(
          superblock.authenticated_payload_catalog_sha256) &&
      writer.put_digest(superblock.record_header_catalog_sha256) &&
      writer.seek(kSuperblockReservedOffset);
  if (!encoded || writer.position() != kSuperblockReservedOffset) {
    std::fill_n(output, kBundleSuperblockBytes, std::uint8_t{0U});
    return {PersistenceError::kInvalidArgument};
  }
  const PersistenceDigest header_digest = compute_header_digest(
      output, kBundleSuperblockBytes, kSuperblockHeaderDigestOffset);
  if (kernels::sm87_target_aot_projection_digest_is_zero(header_digest)) {
    std::fill_n(output, kBundleSuperblockBytes, std::uint8_t{0U});
    return {PersistenceError::kInvalidArgument};
  }
  write_digest_at(output, kSuperblockHeaderDigestOffset, header_digest);
  return {};
}

PersistenceStatus decode_superblock(
    const std::uint8_t* const input, const std::size_t input_bytes,
    PersistenceSuperblock* const superblock) noexcept {
  if (input == nullptr || superblock == nullptr) {
    return {PersistenceError::kInvalidArgument};
  }
  if (input_bytes < kBundleSuperblockBytes) {
    return {PersistenceError::kTruncated};
  }
  if (!bytes_equal(input, kSuperblockMagic)) {
    return {PersistenceError::kInvalidMagic};
  }

  ByteReader reader(input, 8U, kBundleSuperblockBytes);
  std::uint16_t major = 0U;
  std::uint16_t minor = 0U;
  std::uint32_t endian = 0U;
  if (!reader.get_u16(&major) || !reader.get_u16(&minor) ||
      !reader.get_u32(&endian)) {
    return {PersistenceError::kTruncated};
  }
  if (major != kPersistenceAbiMajor || minor != kPersistenceAbiMinor) {
    return {PersistenceError::kUnsupportedVersion};
  }
  if (endian != kLittleEndianTag) {
    return {PersistenceError::kInvalidEndian};
  }

  std::uint32_t superblock_bytes = 0U;
  std::uint32_t record_header_bytes = 0U;
  std::uint16_t sm_major = 0U;
  std::uint16_t sm_minor = 0U;
  std::uint32_t layers = 0U;
  std::uint32_t artifacts = 0U;
  std::uint32_t sources = 0U;
  std::uint16_t order = 0U;
  std::uint16_t repository_bytes = 0U;
  std::uint16_t revision_bytes = 0U;
  if (!reader.get_u32(&superblock_bytes) ||
      !reader.get_u32(&record_header_bytes) ||
      !reader.get_u16(&sm_major) || !reader.get_u16(&sm_minor) ||
      !reader.get_u32(&layers) || !reader.get_u32(&artifacts) ||
      !reader.get_u32(&sources) || !reader.get_u16(&order) ||
      !reader.get_u16(&repository_bytes) ||
      !reader.get_u16(&revision_bytes)) {
    return {PersistenceError::kTruncated};
  }
  if (superblock_bytes != kBundleSuperblockBytes ||
      record_header_bytes != kRecordHeaderBytes ||
      sm_major != kTargetSmMajor || sm_minor != kTargetSmMinor ||
      layers != kPersistedLayerCount ||
      artifacts != kPersistedArtifactCount ||
      sources != kPersistedSourceCount ||
      order != kLayerMajorGateUpThenDown ||
      repository_bytes != kPinnedModelRepository.size() ||
      revision_bytes != kPinnedModelRevision.size()) {
    return {PersistenceError::kFixedFieldMismatch};
  }
  if (!all_zero(input, 46U, 48U) ||
      !string_slot_matches(input, kSuperblockRepositoryOffset,
                           kPinnedStringCapacity, kPinnedModelRepository) ||
      !string_slot_matches(input, kSuperblockRevisionOffset,
                           kPinnedStringCapacity, kPinnedModelRevision) ||
      !all_zero(input, kSuperblockReservedOffset,
                kBundleSuperblockBytes)) {
    return {PersistenceError::kReservedBytesNonZero};
  }
  if (!reader.seek(48U)) {
    return {PersistenceError::kTruncated};
  }
  std::uint64_t arena_bytes = 0U;
  std::uint64_t bundle_bytes = 0U;
  std::uint64_t records_offset = 0U;
  if (!reader.get_u64(&arena_bytes) || !reader.get_u64(&bundle_bytes) ||
      !reader.get_u64(&records_offset)) {
    return {PersistenceError::kTruncated};
  }
  if (arena_bytes != kPersistedArenaBytes ||
      bundle_bytes != kPersistedBundleBytes ||
      records_offset != kBundleSuperblockBytes) {
    return {PersistenceError::kFixedFieldMismatch};
  }

  const PersistenceDigest stored_digest =
      read_digest_at(input, kSuperblockHeaderDigestOffset);
  const PersistenceDigest computed_digest = compute_header_digest(
      input, kBundleSuperblockBytes, kSuperblockHeaderDigestOffset);
  if (stored_digest != computed_digest ||
      kernels::sm87_target_aot_projection_digest_is_zero(stored_digest)) {
    return {PersistenceError::kHeaderDigestMismatch};
  }

  PersistenceSuperblock decoded;
  decoded.checkpoint_identity_sha256 =
      read_digest_at(input, kSuperblockCheckpointDigestOffset);
  decoded.authenticated_payload_catalog_sha256 =
      read_digest_at(input, kSuperblockCatalogDigestOffset);
  decoded.record_header_catalog_sha256 =
      read_digest_at(input, kSuperblockRecordHeaderCatalogDigestOffset);
  if (kernels::sm87_target_aot_projection_digest_is_zero(
          decoded.checkpoint_identity_sha256) ||
      kernels::sm87_target_aot_projection_digest_is_zero(
          decoded.authenticated_payload_catalog_sha256) ||
      kernels::sm87_target_aot_projection_digest_is_zero(
          decoded.record_header_catalog_sha256)) {
    return {PersistenceError::kFixedFieldMismatch};
  }
  *superblock = decoded;
  return {};
}

PersistenceStatus encode_record_header(
    const std::size_t layer_index,
    const kernels::Sm87TargetAotProjectionPackedManifest& manifest,
    const kernels::Sm87TargetAotProjectionPackedTransformReceipt&
        transform_receipt,
    std::uint8_t* const output, const std::size_t output_bytes) noexcept {
  if (output == nullptr) {
    return {PersistenceError::kInvalidArgument};
  }
  if (output_bytes < kRecordHeaderBytes) {
    return {PersistenceError::kOutputTooSmall};
  }
  std::fill_n(output, kRecordHeaderBytes, std::uint8_t{0U});
  kernels::Sm87TargetAotProjectionPackedSourceInventory inventory;
  if (!record_structurally_valid(layer_index, manifest, transform_receipt,
                                 &inventory)) {
    return {PersistenceError::kManifestMismatch};
  }

  const std::size_t ordinal =
      persistence_record_ordinal(layer_index, manifest.role);
  const std::uint64_t arena_offset =
      persistence_expected_arena_offset(layer_index, manifest.role);
  const std::uint64_t bundle_header_offset =
      persistence_expected_bundle_header_offset(layer_index, manifest.role);
  const std::uint64_t bundle_payload_offset =
      bundle_header_offset + kRecordHeaderBytes;
  const std::uint64_t bundle_record_bytes =
      kRecordHeaderBytes + manifest.payload_bytes;

  ByteWriter fixed(output, 0U, kRecordManifestOffset);
  const bool fixed_encoded =
      fixed.put_array(kRecordMagic) &&
      fixed.put_u16(kPersistenceAbiMajor) &&
      fixed.put_u16(kPersistenceAbiMinor) &&
      fixed.put_u32(kLittleEndianTag) &&
      fixed.put_u32(static_cast<std::uint32_t>(kRecordHeaderBytes)) &&
      fixed.put_u32(static_cast<std::uint32_t>(ordinal)) &&
      fixed.put_u32(static_cast<std::uint32_t>(layer_index)) &&
      fixed.put_u8(static_cast<std::uint8_t>(manifest.role)) &&
      fixed.seek(32U) && fixed.put_u64(arena_offset) &&
      fixed.put_u64(bundle_header_offset) &&
      fixed.put_u64(bundle_payload_offset) &&
      fixed.put_u64(bundle_record_bytes) &&
      fixed.put_u32(
          static_cast<std::uint32_t>(kRecordManifestSectionBytes)) &&
      fixed.put_u32(static_cast<std::uint32_t>(
          kRecordTransformReceiptSectionBytes)) &&
      fixed.seek(kRecordFixedReservedOffset);
  ByteWriter manifest_writer(
      output, kRecordManifestOffset,
      kRecordManifestOffset + kRecordManifestSectionBytes);
  ByteWriter receipt_writer(
      output, kRecordTransformReceiptOffset,
      kRecordTransformReceiptOffset + kRecordTransformReceiptSectionBytes);
  const bool sections_encoded =
      write_manifest(manifest_writer, manifest) &&
      manifest_writer.position() ==
          kRecordManifestOffset + kManifestEncodedBytes &&
      write_transform_receipt(receipt_writer, transform_receipt) &&
      receipt_writer.position() ==
          kRecordTransformReceiptOffset + kTransformReceiptEncodedBytes;
  if (!fixed_encoded || fixed.position() != kRecordFixedReservedOffset ||
      !sections_encoded) {
    std::fill_n(output, kRecordHeaderBytes, std::uint8_t{0U});
    return {PersistenceError::kInvalidArgument};
  }

  const PersistenceDigest header_digest = compute_header_digest(
      output, kRecordHeaderBytes, kRecordHeaderDigestOffset);
  if (kernels::sm87_target_aot_projection_digest_is_zero(header_digest)) {
    std::fill_n(output, kRecordHeaderBytes, std::uint8_t{0U});
    return {PersistenceError::kInvalidArgument};
  }
  write_digest_at(output, kRecordHeaderDigestOffset, header_digest);
  return {};
}

PersistenceStatus decode_record_header(
    const std::uint8_t* const input, const std::size_t input_bytes,
    PersistenceRecord* const record) noexcept {
  if (input == nullptr || record == nullptr) {
    return {PersistenceError::kInvalidArgument};
  }
  if (input_bytes < kRecordHeaderBytes) {
    return {PersistenceError::kTruncated};
  }
  if (!bytes_equal(input, kRecordMagic)) {
    return {PersistenceError::kInvalidMagic};
  }

  ByteReader fixed(input, 8U, kRecordManifestOffset);
  std::uint16_t major = 0U;
  std::uint16_t minor = 0U;
  std::uint32_t endian = 0U;
  if (!fixed.get_u16(&major) || !fixed.get_u16(&minor) ||
      !fixed.get_u32(&endian)) {
    return {PersistenceError::kTruncated};
  }
  if (major != kPersistenceAbiMajor || minor != kPersistenceAbiMinor) {
    return {PersistenceError::kUnsupportedVersion};
  }
  if (endian != kLittleEndianTag) {
    return {PersistenceError::kInvalidEndian};
  }

  std::uint32_t header_bytes = 0U;
  std::uint32_t ordinal = 0U;
  std::uint32_t layer_index = 0U;
  std::uint8_t role_value = 0U;
  if (!fixed.get_u32(&header_bytes) || !fixed.get_u32(&ordinal) ||
      !fixed.get_u32(&layer_index) || !fixed.get_u8(&role_value)) {
    return {PersistenceError::kTruncated};
  }
  if (!valid_projection_role_value(role_value)) {
    return {PersistenceError::kInvalidEnum};
  }
  const auto role =
      static_cast<kernels::Sm87TargetAotProjectionRole>(role_value);
  if (header_bytes != kRecordHeaderBytes ||
      layer_index >= kPersistedLayerCount ||
      !persistence_role_is_nvfp4(role) ||
      ordinal != persistence_record_ordinal(layer_index, role)) {
    return {PersistenceError::kFixedFieldMismatch};
  }
  if (!all_zero(input, 29U, 32U) ||
      !all_zero(input, kRecordFixedReservedOffset,
                kRecordManifestOffset) ||
      !all_zero(input,
                kRecordManifestOffset + kManifestEncodedBytes,
                kRecordTransformReceiptOffset) ||
      !all_zero(input,
                kRecordTransformReceiptOffset +
                    kTransformReceiptEncodedBytes,
                kRecordTransformReceiptOffset +
                    kRecordTransformReceiptSectionBytes) ||
      !all_zero(input,
                kRecordTransformReceiptOffset +
                    kRecordTransformReceiptSectionBytes,
                kRecordHeaderBytes)) {
    return {PersistenceError::kReservedBytesNonZero};
  }

  if (!fixed.seek(32U)) {
    return {PersistenceError::kTruncated};
  }
  std::uint64_t arena_offset = 0U;
  std::uint64_t bundle_header_offset = 0U;
  std::uint64_t bundle_payload_offset = 0U;
  std::uint64_t bundle_record_bytes = 0U;
  std::uint32_t manifest_section_bytes = 0U;
  std::uint32_t receipt_section_bytes = 0U;
  if (!fixed.get_u64(&arena_offset) ||
      !fixed.get_u64(&bundle_header_offset) ||
      !fixed.get_u64(&bundle_payload_offset) ||
      !fixed.get_u64(&bundle_record_bytes) ||
      !fixed.get_u32(&manifest_section_bytes) ||
      !fixed.get_u32(&receipt_section_bytes)) {
    return {PersistenceError::kTruncated};
  }
  if (manifest_section_bytes != kRecordManifestSectionBytes ||
      receipt_section_bytes != kRecordTransformReceiptSectionBytes ||
      arena_offset != persistence_expected_arena_offset(layer_index, role) ||
      bundle_header_offset !=
          persistence_expected_bundle_header_offset(layer_index, role) ||
      bundle_payload_offset != bundle_header_offset + kRecordHeaderBytes) {
    return {PersistenceError::kFixedFieldMismatch};
  }

  const PersistenceDigest stored_digest =
      read_digest_at(input, kRecordHeaderDigestOffset);
  const PersistenceDigest computed_digest = compute_header_digest(
      input, kRecordHeaderBytes, kRecordHeaderDigestOffset);
  if (stored_digest != computed_digest ||
      kernels::sm87_target_aot_projection_digest_is_zero(stored_digest)) {
    return {PersistenceError::kHeaderDigestMismatch};
  }

  PersistenceRecord decoded;
  decoded.ordinal = ordinal;
  decoded.layer_index = layer_index;
  decoded.role = role;
  decoded.arena_offset = arena_offset;
  decoded.bundle_header_offset = bundle_header_offset;
  decoded.bundle_payload_offset = bundle_payload_offset;
  decoded.bundle_record_bytes = bundle_record_bytes;

  ByteReader manifest_reader(
      input, kRecordManifestOffset,
      kRecordManifestOffset + kRecordManifestSectionBytes);
  PersistenceError error =
      read_manifest(manifest_reader, &decoded.manifest);
  if (error != PersistenceError::kNone) {
    return {error};
  }
  if (manifest_reader.position() !=
      kRecordManifestOffset + kManifestEncodedBytes) {
    return {PersistenceError::kManifestMismatch};
  }
  ByteReader receipt_reader(
      input, kRecordTransformReceiptOffset,
      kRecordTransformReceiptOffset + kRecordTransformReceiptSectionBytes);
  error = read_transform_receipt(receipt_reader,
                                 &decoded.transform_receipt);
  if (error != PersistenceError::kNone) {
    return {error};
  }
  if (receipt_reader.position() !=
      kRecordTransformReceiptOffset + kTransformReceiptEncodedBytes) {
    return {PersistenceError::kTransformReceiptMismatch};
  }

  if (decoded.manifest.role != role ||
      bundle_record_bytes !=
          kRecordHeaderBytes + decoded.manifest.payload_bytes) {
    return {PersistenceError::kFixedFieldMismatch};
  }
  decoded.source_inventory = inventory_from_manifest(decoded.manifest);
  if (!kernels::sm87_target_aot_projection_validate_packed_manifest(
          decoded.manifest, decoded.source_inventory)) {
    return {PersistenceError::kManifestMismatch};
  }
  if (!kernels::sm87_target_aot_projection_validate_transform_receipt(
          decoded.manifest, decoded.source_inventory,
          decoded.transform_receipt)) {
    return {PersistenceError::kTransformReceiptMismatch};
  }
  *record = decoded;
  return {};
}

}  // namespace q3x::runtime::target_aot_persistence_detail
