#pragma once

#include "q3x/kernels/sm87_target_aot_projection_layout.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

// Source-private canonical disk codec for the NVFP4 projection subset of
// AC-PREFILL-SM87-AOT-SYSTEM-v1.  This is deliberately not an installed API:
// persistence grants no device pointer, upload receipt, view, or execution
// capability.
namespace q3x::runtime::target_aot_persistence_detail {

inline constexpr std::string_view kPinnedModelRepository =
    "nvidia/Qwen3.6-27B-NVFP4";
inline constexpr std::string_view kPinnedModelRevision =
    "0893e1606ff3d5f97a441f405d5fc541a6bdf404";

inline constexpr std::size_t kBundleSuperblockBytes = 4'096U;
inline constexpr std::size_t kRecordHeaderBytes = 4'096U;
inline constexpr std::size_t kPersistedLayerCount = 64U;
inline constexpr std::size_t kPersistedArtifactsPerLayer = 2U;
inline constexpr std::size_t kPersistedArtifactCount =
    kPersistedLayerCount * kPersistedArtifactsPerLayer;
inline constexpr std::size_t kPersistedSourceCount =
    kPersistedLayerCount * 3U;
inline constexpr std::uint64_t kPersistedArenaBytes = 9'625'927'680ULL;
inline constexpr std::uint64_t kPersistedBundleBytes = 9'626'456'064ULL;
inline constexpr std::uint16_t kPersistenceAbiMajor = 1U;
inline constexpr std::uint16_t kPersistenceAbiMinor = 0U;
inline constexpr std::uint16_t kTargetSmMajor = 8U;
inline constexpr std::uint16_t kTargetSmMinor = 7U;
inline constexpr std::uint32_t kLittleEndianTag = 0x0102'0304U;
inline constexpr std::size_t kPinnedStringCapacity = 64U;

inline constexpr std::size_t kSuperblockHeaderDigestOffset = 296U;
inline constexpr std::size_t kRecordHeaderDigestOffset = 72U;
inline constexpr std::size_t kHeaderDigestBytes = 32U;
inline constexpr std::size_t kRecordManifestOffset = 128U;
inline constexpr std::size_t kRecordManifestSectionBytes = 512U;
inline constexpr std::size_t kRecordTransformReceiptOffset =
    kRecordManifestOffset + kRecordManifestSectionBytes;
inline constexpr std::size_t kRecordTransformReceiptSectionBytes = 640U;

using PersistenceDigest =
    kernels::Sm87TargetAotProjectionSha256Digest;
using SuperblockBytes = std::array<std::uint8_t, kBundleSuperblockBytes>;
using RecordHeaderBytes = std::array<std::uint8_t, kRecordHeaderBytes>;

enum class PersistenceError : std::uint8_t {
  kNone = 0U,
  kInvalidArgument,
  kOutputTooSmall,
  kTruncated,
  kInvalidMagic,
  kUnsupportedVersion,
  kInvalidEndian,
  kInvalidEnum,
  kInvalidBoolean,
  kFixedFieldMismatch,
  kReservedBytesNonZero,
  kHeaderDigestMismatch,
  kManifestMismatch,
  kTransformReceiptMismatch,
};

struct PersistenceStatus final {
  PersistenceError error = PersistenceError::kNone;

  [[nodiscard]] constexpr bool ok() const noexcept {
    return error == PersistenceError::kNone;
  }
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok();
  }
};

// Stable bundle fields. Repository, revision, target, counts, offsets, and
// byte sizes are format constants and are nevertheless encoded in every
// superblock. The checkpoint and verified-payload catalog digests are external
// authentication inputs. record_header_catalog_sha256 is SHA-256 over the
// direct concatenation of the 128 complete canonical record-header byte images
// in ordinal order, and closes their exact sequence before payload streaming.
struct PersistenceSuperblock final {
  PersistenceDigest checkpoint_identity_sha256{};
  PersistenceDigest authenticated_payload_catalog_sha256{};
  PersistenceDigest record_header_catalog_sha256{};
};

// Stable record fields reconstructed from one canonical record header.
// Payload bytes are not retained here; the loader hashes the corresponding
// bundle byte interval before it can issue any runtime upload authority.
struct PersistenceRecord final {
  std::size_t ordinal = 0U;
  std::size_t layer_index = 0U;
  kernels::Sm87TargetAotProjectionRole role =
      kernels::Sm87TargetAotProjectionRole::kInvalid;
  std::uint64_t arena_offset = 0U;
  std::uint64_t bundle_header_offset = 0U;
  std::uint64_t bundle_payload_offset = 0U;
  std::uint64_t bundle_record_bytes = 0U;
  kernels::Sm87TargetAotProjectionPackedSourceInventory source_inventory{};
  kernels::Sm87TargetAotProjectionPackedManifest manifest{};
  kernels::Sm87TargetAotProjectionPackedTransformReceipt transform_receipt{};
};

// Linux file-descriptor and publication primitives shared by the production
// writer/loader and the host-only security tests. They remain source-private:
// the installed runtime API exposes only an authenticated bundle path.
class UniqueFd final {
 public:
  UniqueFd() noexcept = default;
  explicit UniqueFd(int fd) noexcept;
  ~UniqueFd();

  UniqueFd(const UniqueFd&) = delete;
  UniqueFd& operator=(const UniqueFd&) = delete;
  UniqueFd(UniqueFd&& other) noexcept;
  UniqueFd& operator=(UniqueFd&& other) noexcept;

  [[nodiscard]] int get() const noexcept;
  [[nodiscard]] explicit operator bool() const noexcept;
  [[nodiscard]] int release() noexcept;
  void reset(int next = -1) noexcept;

 private:
  int fd_ = -1;
};

struct SecureParent final {
  UniqueFd directory;
  std::string filename;
};

struct FileSnapshot final {
  std::uint64_t device = 0U;
  std::uint64_t inode = 0U;
  std::uint64_t mode = 0U;
  std::uint64_t links = 0U;
  std::uint64_t bytes = 0U;
  std::int64_t modified_seconds = 0;
  std::int64_t modified_nanoseconds = 0;
  std::int64_t changed_seconds = 0;
  std::int64_t changed_nanoseconds = 0;
};

enum class AtomicPublishOutcome : std::uint8_t {
  kNotPublished = 0U,
  kPublishedDurable,
  kPublishedDurabilityUncertain,
};

[[nodiscard]] bool secure_absolute_parent(
    const std::filesystem::path& path, SecureParent& result,
    std::string& message);

[[nodiscard]] bool same_snapshot(const FileSnapshot& left,
                                 const FileSnapshot& right) noexcept;

[[nodiscard]] bool snapshot_regular_file_exact(
    int fd, std::uint64_t expected_bytes, FileSnapshot& snapshot,
    std::string& message);

[[nodiscard]] bool open_regular_file_exact(
    const std::filesystem::path& path, std::uint64_t expected_bytes,
    UniqueFd& file, FileSnapshot& snapshot, std::string& message);

[[nodiscard]] bool pread_exact_counted(
    int fd, void* output, std::size_t bytes, std::uint64_t offset,
    std::uint64_t& bytes_read, std::string& message);

[[nodiscard]] bool pwrite_exact_counted(
    int fd, const void* input, std::size_t bytes, std::uint64_t offset,
    std::uint64_t& bytes_written, std::string& message);

class AtomicCreateOnlyFile final {
 public:
  AtomicCreateOnlyFile() noexcept = default;
  ~AtomicCreateOnlyFile();

  AtomicCreateOnlyFile(const AtomicCreateOnlyFile&) = delete;
  AtomicCreateOnlyFile& operator=(const AtomicCreateOnlyFile&) = delete;
  AtomicCreateOnlyFile(AtomicCreateOnlyFile&&) = delete;
  AtomicCreateOnlyFile& operator=(AtomicCreateOnlyFile&&) = delete;

  // Creates an unpublished same-directory temporary file. reserve_storage is
  // required by the production bundle writer; host tests may select a sparse
  // file while exercising identical naming and publication mechanics.
  [[nodiscard]] bool open_create_only(
      const std::filesystem::path& output, std::uint64_t expected_bytes,
      bool reserve_storage, std::string& message);
  [[nodiscard]] bool sync_data_and_drop_cache(
      std::uint64_t offset, std::uint64_t bytes, std::string& message);
  // fsync(file) -> renameat2(RENAME_NOREPLACE) -> fsync(directory). Once the
  // rename succeeds, the final name is never removed even if directory fsync
  // fails; the distinct outcome requires later secure reauthentication.
  [[nodiscard]] AtomicPublishOutcome publish(std::string& message);

  [[nodiscard]] bool is_open() const noexcept;
  [[nodiscard]] int fd() const noexcept;
  [[nodiscard]] std::uint64_t expected_bytes() const noexcept;

 private:
  void abandon() noexcept;

  UniqueFd directory_;
  UniqueFd fd_;
  std::string temporary_name_;
  std::string final_name_;
  std::uint64_t expected_bytes_ = 0U;
};

[[nodiscard]] constexpr bool persistence_role_is_nvfp4(
    kernels::Sm87TargetAotProjectionRole role) noexcept {
  return role == kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp ||
         role == kernels::Sm87TargetAotProjectionRole::kNvFp4Down;
}

[[nodiscard]] constexpr std::size_t persistence_record_ordinal(
    const std::size_t layer_index,
    const kernels::Sm87TargetAotProjectionRole role) noexcept {
  if (layer_index >= kPersistedLayerCount ||
      !persistence_role_is_nvfp4(role)) {
    return kPersistedArtifactCount;
  }
  return layer_index * kPersistedArtifactsPerLayer +
         (role == kernels::Sm87TargetAotProjectionRole::kNvFp4Down ? 1U
                                                                   : 0U);
}

[[nodiscard]] constexpr std::uint64_t persistence_expected_arena_offset(
    const std::size_t layer_index,
    const kernels::Sm87TargetAotProjectionRole role) noexcept {
  if (persistence_record_ordinal(layer_index, role) >=
      kPersistedArtifactCount) {
    return kPersistedArenaBytes;
  }
  constexpr std::uint64_t kGateUpBytes =
      kernels::kSm87TargetAotNvFp4GateUpPackedLayout.payload_bytes;
  constexpr std::uint64_t kDownBytes =
      kernels::kSm87TargetAotNvFp4DownPackedLayout.payload_bytes;
  return static_cast<std::uint64_t>(layer_index) *
             (kGateUpBytes + kDownBytes) +
         (role == kernels::Sm87TargetAotProjectionRole::kNvFp4Down
              ? kGateUpBytes
              : 0U);
}

[[nodiscard]] constexpr std::uint64_t persistence_expected_bundle_header_offset(
    const std::size_t layer_index,
    const kernels::Sm87TargetAotProjectionRole role) noexcept {
  const std::size_t ordinal = persistence_record_ordinal(layer_index, role);
  return ordinal >= kPersistedArtifactCount
             ? kPersistedBundleBytes
             : static_cast<std::uint64_t>(kBundleSuperblockBytes) +
                   static_cast<std::uint64_t>(ordinal) * kRecordHeaderBytes +
                   persistence_expected_arena_offset(layer_index, role);
}

[[nodiscard]] PersistenceStatus encode_superblock(
    const PersistenceSuperblock& superblock, std::uint8_t* output,
    std::size_t output_bytes) noexcept;

[[nodiscard]] PersistenceStatus decode_superblock(
    const std::uint8_t* input, std::size_t input_bytes,
    PersistenceSuperblock* superblock) noexcept;

[[nodiscard]] PersistenceStatus encode_record_header(
    std::size_t layer_index,
    const kernels::Sm87TargetAotProjectionPackedManifest& manifest,
    const kernels::Sm87TargetAotProjectionPackedTransformReceipt&
        transform_receipt,
    std::uint8_t* output, std::size_t output_bytes) noexcept;

[[nodiscard]] PersistenceStatus decode_record_header(
    const std::uint8_t* input, std::size_t input_bytes,
    PersistenceRecord* record) noexcept;

static_assert(kPinnedModelRepository.size() <= kPinnedStringCapacity);
static_assert(kPinnedModelRevision.size() <= kPinnedStringCapacity);
static_assert(kBundleSuperblockBytes == 4'096U);
static_assert(kRecordHeaderBytes == 4'096U);
static_assert(kRecordHeaderBytes ==
              kernels::kSm87TargetAotProjectionPackedHeaderBytes);
static_assert(kernels::kSm87TargetAotProjectionPackedMaxPartitions == 3U);
static_assert(kPersistedLayerCount == 64U);
static_assert(kPersistedArtifactsPerLayer == 2U);
static_assert(kPersistedArtifactCount == 128U);
static_assert(kPersistedSourceCount == 192U);
static_assert(kPersistedArenaBytes ==
              kPersistedLayerCount *
                  (kernels::kSm87TargetAotNvFp4GateUpPackedLayout
                       .payload_bytes +
                   kernels::kSm87TargetAotNvFp4DownPackedLayout
                       .payload_bytes));
static_assert(kPersistedBundleBytes ==
              kPersistedArenaBytes + kBundleSuperblockBytes +
                  kPersistedArtifactCount * kRecordHeaderBytes);
static_assert(kRecordTransformReceiptOffset +
                      kRecordTransformReceiptSectionBytes <=
                  kRecordHeaderBytes);

// Persistence ABI v1 writes enum values directly. Freeze every value accepted
// by the codec so a source enum insertion cannot silently reinterpret an
// already-persisted bundle without an explicit ABI revision.
static_assert(static_cast<std::uint8_t>(
                  kernels::Sm87TargetAotProjectionRole::kInvalid) == 0U);
static_assert(static_cast<std::uint8_t>(
                  kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp) == 1U);
static_assert(static_cast<std::uint8_t>(
                  kernels::Sm87TargetAotProjectionRole::kNvFp4Down) == 2U);
static_assert(static_cast<std::uint8_t>(
                  kernels::Sm87TargetAotProjectionRole::kFp8GdnQkvZ) == 3U);
static_assert(static_cast<std::uint8_t>(
                  kernels::Sm87TargetAotProjectionRole::kFp8FullQkv) == 4U);
static_assert(static_cast<std::uint8_t>(
                  kernels::Sm87TargetAotProjectionRole::kFp8AttentionOutput) ==
              5U);
static_assert(static_cast<std::uint8_t>(
                  kernels::Sm87TargetAotProjectionRole::kCount) == 6U);

static_assert(static_cast<std::uint8_t>(
                  kernels::Sm87TargetAotLogicalRole::kInvalid) == 0U);
static_assert(static_cast<std::uint8_t>(
                  kernels::Sm87TargetAotLogicalRole::kNvFp4Gate) == 1U);
static_assert(static_cast<std::uint8_t>(
                  kernels::Sm87TargetAotLogicalRole::kNvFp4Up) == 2U);
static_assert(static_cast<std::uint8_t>(
                  kernels::Sm87TargetAotLogicalRole::kNvFp4Down) == 3U);
static_assert(static_cast<std::uint8_t>(
                  kernels::Sm87TargetAotLogicalRole::kFp8GdnQkv) == 4U);
static_assert(static_cast<std::uint8_t>(
                  kernels::Sm87TargetAotLogicalRole::kFp8GdnZ) == 5U);
static_assert(static_cast<std::uint8_t>(
                  kernels::Sm87TargetAotLogicalRole::kFp8FullQGate) == 6U);
static_assert(static_cast<std::uint8_t>(
                  kernels::Sm87TargetAotLogicalRole::kFp8FullK) == 7U);
static_assert(static_cast<std::uint8_t>(
                  kernels::Sm87TargetAotLogicalRole::kFp8FullV) == 8U);
static_assert(static_cast<std::uint8_t>(
                  kernels::Sm87TargetAotLogicalRole::kFp8AttentionOutput) ==
              9U);

static_assert(static_cast<std::uint16_t>(
                  kernels::Sm87TargetAotProjectionPackedPlanIdentity::
                      kInvalid) == 0U);
static_assert(static_cast<std::uint16_t>(
                  kernels::Sm87TargetAotProjectionPackedPlanIdentity::
                      kNvFp4GateUpM128N256K64V1) == 1U);
static_assert(static_cast<std::uint16_t>(
                  kernels::Sm87TargetAotProjectionPackedPlanIdentity::
                      kNvFp4DownM128N256K64V1) == 2U);
static_assert(static_cast<std::uint16_t>(
                  kernels::Sm87TargetAotProjectionPackedPlanIdentity::
                      kFp8GdnQkvZM128N256K64V1) == 3U);
static_assert(static_cast<std::uint16_t>(
                  kernels::Sm87TargetAotProjectionPackedPlanIdentity::
                      kFp8FullQkvM128N256K64V1) == 4U);
static_assert(static_cast<std::uint16_t>(
                  kernels::Sm87TargetAotProjectionPackedPlanIdentity::
                      kFp8AttentionOutputM128N256K64V1) == 5U);

static_assert(static_cast<std::uint16_t>(
                  kernels::Sm87TargetAotProjectionPackedLayoutIdentity::
                      kInvalid) == 0U);
static_assert(static_cast<std::uint16_t>(
                  kernels::Sm87TargetAotProjectionPackedLayoutIdentity::
                      kConsumerN64K16LaneComponentV1) == 1U);
static_assert(static_cast<std::uint8_t>(
                  kernels::Sm87TargetAotProjectionEncoding::kInvalid) == 0U);
static_assert(static_cast<std::uint8_t>(
                  kernels::Sm87TargetAotProjectionEncoding::
                      kNvFp4E2M1Block16E4M3FnScale) == 1U);
static_assert(static_cast<std::uint8_t>(
                  kernels::Sm87TargetAotProjectionEncoding::
                      kFp8E4M3FnTensorScale) == 2U);
static_assert(static_cast<std::uint16_t>(
                  kernels::Sm87TargetAotProjectionPackedTransformIdentity::
                      kInvalid) == 0U);
static_assert(static_cast<std::uint16_t>(
                  kernels::Sm87TargetAotProjectionPackedTransformIdentity::
                      kCanonicalNkToConsumerN64K16LaneComponentV1) == 1U);

}  // namespace q3x::runtime::target_aot_persistence_detail
