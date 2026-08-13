#include "sm87_target_aot_projection_persistence_internal.h"

#include "q3x/core/sha256.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace persistence =
    q3x::runtime::target_aot_persistence_detail;
namespace kernels = q3x::kernels;

namespace {

class TestContext final {
 public:
  void expect(const bool condition, const std::string& message) {
    if (!condition) {
      ++failures_;
      std::cerr << "FAIL: " << message << '\n';
    }
  }

  [[nodiscard]] int failures() const noexcept { return failures_; }

 private:
  int failures_ = 0;
};

class ScratchRoot final {
 public:
  explicit ScratchRoot(std::filesystem::path path) : path_(std::move(path)) {
    if (!path_.is_absolute() ||
        path_.filename() != "q3x-sm87-target-aot-persistence-host") {
      return;
    }
    std::error_code error;
    (void)std::filesystem::remove_all(path_, error);
    error.clear();
    ready_ = std::filesystem::create_directories(path_, error) && !error;
  }

  ~ScratchRoot() {
    if (ready_) {
      std::error_code error;
      (void)std::filesystem::remove_all(path_, error);
    }
  }

  ScratchRoot(const ScratchRoot&) = delete;
  ScratchRoot& operator=(const ScratchRoot&) = delete;

  [[nodiscard]] bool ready() const noexcept { return ready_; }
  [[nodiscard]] const std::filesystem::path& path() const noexcept {
    return path_;
  }

 private:
  std::filesystem::path path_;
  bool ready_ = false;
};

[[nodiscard]] bool create_sparse_file(const std::filesystem::path& path,
                                      const std::uint64_t bytes) {
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
                        S_IRUSR | S_IWUSR);
  if (fd < 0) {
    return false;
  }
  const bool sized = bytes <=
                         static_cast<std::uint64_t>(
                             std::numeric_limits<off_t>::max()) &&
                     ::ftruncate(fd, static_cast<off_t>(bytes)) == 0;
  (void)::close(fd);
  return sized;
}

[[nodiscard]] persistence::PersistenceDigest make_digest(
    const std::uint64_t seed) noexcept {
  persistence::PersistenceDigest digest;
  std::uint64_t state = seed | 1U;
  for (std::size_t index = 0U; index < digest.bytes.size(); ++index) {
    state ^= state << 13U;
    state ^= state >> 7U;
    state ^= state << 17U;
    digest.bytes[index] = static_cast<std::uint8_t>(state >> 24U);
  }
  if (kernels::sm87_target_aot_projection_digest_is_zero(digest)) {
    digest.bytes[0U] = 1U;
  }
  return digest;
}

[[nodiscard]] kernels::Sm87TargetAotProjectionPackedSourceInventory
make_inventory(
    const kernels::Sm87TargetAotProjectionPackedLayout& layout,
    const std::uint64_t seed) noexcept {
  kernels::Sm87TargetAotProjectionPackedSourceInventory inventory;
  inventory.identity = 0x5100'0000'0000'0000ULL + seed;
  inventory.role = layout.role;
  inventory.source_count = layout.partition_count;
  constexpr std::array<std::uint32_t, 3U> kScaleBits{{
      0x3f80'0000U,
      0x3f00'0000U,
      0x3e80'0000U,
  }};
  for (std::size_t index = 0U; index < layout.partition_count; ++index) {
    inventory.sources[index] =
        kernels::sm87_target_aot_projection_packed_source_binding(
            layout, index,
            0x6100'0000'0000'0000ULL + seed * 4U + index,
            make_digest(seed * 17U + index * 2U + 1U),
            make_digest(seed * 17U + index * 2U + 2U),
            kScaleBits[index]);
  }
  return inventory;
}

[[nodiscard]] kernels::Sm87TargetAotProjectionPackedTransformReceipt
make_transform_receipt(
    const kernels::Sm87TargetAotProjectionPackedLayout& layout,
    const kernels::Sm87TargetAotProjectionPackedSourceInventory& inventory,
    const kernels::Sm87TargetAotProjectionPackedManifest& manifest) noexcept {
  kernels::Sm87TargetAotProjectionPackedTransformReceipt receipt;
  receipt.artifact_identity = manifest.artifact_identity;
  receipt.source_inventory_identity = inventory.identity;
  receipt.role = layout.role;
  receipt.plan_identity = layout.plan_identity;
  receipt.layout_identity = layout.layout_identity;
  receipt.encoding = layout.encoding;
  receipt.transform_identity = kernels::
      Sm87TargetAotProjectionPackedTransformIdentity::
          kCanonicalNkToConsumerN64K16LaneComponentV1;
  receipt.partition_count = layout.partition_count;
  receipt.payload = {manifest.artifact_identity,
                     manifest.payload_offset,
                     manifest.payload_bytes,
                     manifest.payload_digest,
                     true};
  receipt.deterministic_transform = true;
  receipt.no_arithmetic_conversion = true;
  receipt.no_request_time_repacking = true;
  for (std::size_t index = 0U; index < layout.partition_count; ++index) {
    const auto& partition = layout.partitions[index];
    const auto& source = inventory.sources[index];
    auto& observed = receipt.partitions[index];
    const std::uint64_t values =
        static_cast<std::uint64_t>(partition.output_features) *
        partition.input_features;
    const std::uint64_t block_scale_values =
        partition.block_scale_group_k == 0U
            ? 0U
            : values / partition.block_scale_group_k;
    observed.logical_role = partition.logical_role;
    observed.partition_index = static_cast<std::uint32_t>(index);
    observed.tensor_identity = source.tensor_identity;
    observed.observed_source_weight_digest = source.weight_digest;
    observed.observed_source_scale_digest = source.scale_digest;
    observed.source_weight_bytes_hashed =
        values * partition.weight_bits / 8U;
    observed.source_scale_bytes_hashed =
        block_scale_values + sizeof(std::uint32_t);
    observed.repacked_weight_values = values;
    observed.repacked_block_scale_values = block_scale_values;
    observed.source_block_scale_e4m3fn_bytes_scanned = block_scale_values;
    observed.payload_block_scale_e4m3fn_bytes_scanned = block_scale_values;
    observed.payload_offset = partition.payload_offset;
    observed.payload_bytes = partition.payload_bytes;
    observed.source_digests_computed_from_tensor_bytes = true;
    observed.canonical_address_bijection_applied = true;
    observed.bit_exact_weight_permutation = true;
    observed.bit_exact_block_scale_permutation =
        block_scale_values != 0U;
    observed.tensor_scale_kept_external = true;
  }
  return receipt;
}

struct RecordFixture final {
  kernels::Sm87TargetAotProjectionPackedManifest manifest{};
  kernels::Sm87TargetAotProjectionPackedTransformReceipt receipt{};
};

[[nodiscard]] RecordFixture make_record_fixture(
    const kernels::Sm87TargetAotProjectionRole role,
    const std::uint64_t seed) noexcept {
  const auto layout =
      kernels::sm87_target_aot_projection_packed_layout(role);
  const auto inventory = make_inventory(layout, seed);
  RecordFixture fixture;
  fixture.manifest =
      kernels::sm87_target_aot_projection_make_packed_manifest(
          role, 0x7100'0000'0000'0000ULL + seed, inventory,
          make_digest(0x8000U + seed));
  fixture.receipt =
      make_transform_receipt(layout, inventory, fixture.manifest);
  return fixture;
}

[[nodiscard]] std::uint64_t read_u64_le(const std::uint8_t* const input)
    noexcept {
  std::uint64_t value = 0U;
  for (std::size_t index = 0U; index < sizeof(value); ++index) {
    value |= static_cast<std::uint64_t>(input[index]) << (8U * index);
  }
  return value;
}

void write_u16_le(std::uint8_t* const output,
                  const std::uint16_t value) noexcept {
  for (std::size_t index = 0U; index < sizeof(value); ++index) {
    output[index] = static_cast<std::uint8_t>(value >> (8U * index));
  }
}

void write_u32_le(std::uint8_t* const output,
                  const std::uint32_t value) noexcept {
  for (std::size_t index = 0U; index < sizeof(value); ++index) {
    output[index] = static_cast<std::uint8_t>(value >> (8U * index));
  }
}

void write_u64_le(std::uint8_t* const output,
                  const std::uint64_t value) noexcept {
  for (std::size_t index = 0U; index < sizeof(value); ++index) {
    output[index] = static_cast<std::uint8_t>(value >> (8U * index));
  }
}

[[nodiscard]] std::string sha256_hex(const std::uint8_t* const input,
                                     const std::size_t bytes) noexcept {
  q3x::core::Sha256 hasher;
  if (!hasher.update(input, bytes)) {
    return {};
  }
  return hasher.finalize().hex();
}

void reseal_header(std::uint8_t* const header, const std::size_t bytes,
                   const std::size_t digest_offset) noexcept {
  std::fill_n(header + digest_offset, persistence::kHeaderDigestBytes,
              std::uint8_t{0U});
  q3x::core::Sha256 hasher;
  (void)hasher.update(header, bytes);
  const q3x::core::Sha256Digest digest = hasher.finalize();
  std::copy(digest.bytes.begin(), digest.bytes.end(),
            header + digest_offset);
}

void test_superblock(TestContext& test) {
  persistence::PersistenceSuperblock source;
  source.checkpoint_identity_sha256 = make_digest(1U);
  source.authenticated_payload_catalog_sha256 = make_digest(2U);
  source.record_header_catalog_sha256 = make_digest(3U);
  persistence::SuperblockBytes encoded{};
  test.expect(persistence::encode_superblock(source, encoded.data(),
                                             encoded.size())
                  .ok(),
              "valid superblock encodes");
  test.expect(
      std::equal(encoded.begin(), encoded.begin() + 8U,
                 std::array<std::uint8_t, 8U>{
                     'Q', '3', 'X', 'A', 'O', 'T', 'B', '1'}
                     .begin()) &&
          encoded[8U] == 1U && encoded[9U] == 0U &&
          encoded[12U] == 0x04U && encoded[13U] == 0x03U &&
          encoded[14U] == 0x02U && encoded[15U] == 0x01U &&
          read_u64_le(encoded.data() + 48U) ==
              persistence::kPersistedArenaBytes,
      "superblock golden prefix is explicit little-endian");

  constexpr std::string_view kGoldenSha256 =
      "fab99245e5efefdf02cf75cdb0b78e1ce6d95782a8cc3357d4d9e1373909fdfa";
  const std::string observed_sha256 =
      sha256_hex(encoded.data(), encoded.size());
  test.expect(observed_sha256 == kGoldenSha256,
              "superblock golden SHA-256 matches: observed=" +
                  observed_sha256);

  persistence::PersistenceSuperblock decoded;
  test.expect(persistence::decode_superblock(encoded.data(), encoded.size(),
                                             &decoded)
                  .ok(),
              "valid superblock decodes");
  test.expect(decoded.checkpoint_identity_sha256 ==
                  source.checkpoint_identity_sha256 &&
                  decoded.authenticated_payload_catalog_sha256 ==
                      source.authenticated_payload_catalog_sha256 &&
                  decoded.record_header_catalog_sha256 ==
                      source.record_header_catalog_sha256,
              "superblock round-trip preserves all catalog digests");
  persistence::SuperblockBytes replay{};
  test.expect(persistence::encode_superblock(decoded, replay.data(),
                                             replay.size()) &&
                  replay == encoded,
              "superblock decode/re-encode is canonical");

  auto corrupted = encoded;
  corrupted[persistence::kSuperblockHeaderDigestOffset] ^= 1U;
  test.expect(
      persistence::decode_superblock(corrupted.data(), corrupted.size(),
                                     &decoded)
              .error == persistence::PersistenceError::kHeaderDigestMismatch,
      "superblock digest corruption fails closed");
  corrupted = encoded;
  corrupted.back() = 1U;
  test.expect(
      persistence::decode_superblock(corrupted.data(), corrupted.size(),
                                     &decoded)
              .error ==
          persistence::PersistenceError::kReservedBytesNonZero,
      "superblock reserved bytes must remain zero");
  test.expect(
      persistence::decode_superblock(encoded.data(), encoded.size() - 1U,
                                     &decoded)
              .error == persistence::PersistenceError::kTruncated,
      "truncated superblock is rejected before parsing");

  corrupted = encoded;
  write_u16_le(corrupted.data() + 8U, 2U);
  reseal_header(corrupted.data(), corrupted.size(),
                persistence::kSuperblockHeaderDigestOffset);
  test.expect(
      persistence::decode_superblock(corrupted.data(), corrupted.size(),
                                     &decoded)
              .error == persistence::PersistenceError::kUnsupportedVersion,
      "well-digested superblock ABI mismatch fails closed");
  corrupted = encoded;
  write_u32_le(corrupted.data() + 12U, 0x0102'0305U);
  reseal_header(corrupted.data(), corrupted.size(),
                persistence::kSuperblockHeaderDigestOffset);
  test.expect(
      persistence::decode_superblock(corrupted.data(), corrupted.size(),
                                     &decoded)
              .error == persistence::PersistenceError::kInvalidEndian,
      "well-digested superblock endian mismatch fails closed");
  corrupted = encoded;
  write_u32_le(corrupted.data() + 20U, 2'048U);
  reseal_header(corrupted.data(), corrupted.size(),
                persistence::kSuperblockHeaderDigestOffset);
  test.expect(
      persistence::decode_superblock(corrupted.data(), corrupted.size(),
                                     &decoded)
              .error == persistence::PersistenceError::kFixedFieldMismatch,
      "well-digested superblock fixed-size mismatch fails closed");
  corrupted = encoded;
  std::fill_n(corrupted.data() + 264U, persistence::kHeaderDigestBytes,
              std::uint8_t{0U});
  reseal_header(corrupted.data(), corrupted.size(),
                persistence::kSuperblockHeaderDigestOffset);
  test.expect(
      persistence::decode_superblock(corrupted.data(), corrupted.size(),
                                     &decoded)
              .error == persistence::PersistenceError::kFixedFieldMismatch,
      "zero record-header catalog digest fails closed");

  persistence::PersistenceSuperblock zero_digest;
  test.expect(
      persistence::encode_superblock(zero_digest, replay.data(), replay.size())
              .error == persistence::PersistenceError::kInvalidArgument,
      "zero external authentication digests cannot encode");
  auto missing_record_catalog = source;
  missing_record_catalog.record_header_catalog_sha256 = {};
  test.expect(
      persistence::encode_superblock(missing_record_catalog, replay.data(),
                                     replay.size())
              .error == persistence::PersistenceError::kInvalidArgument,
      "zero record-header catalog digest cannot encode");
}

void test_record_header(TestContext& test) {
  constexpr std::size_t kLayer = 7U;
  const RecordFixture fixture = make_record_fixture(
      kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp, 7U);
  persistence::RecordHeaderBytes encoded{};
  test.expect(persistence::encode_record_header(
                  kLayer, fixture.manifest, fixture.receipt, encoded.data(),
                  encoded.size())
                  .ok(),
              "valid GateUp record header encodes");
  test.expect(encoded[0U] == 'Q' && encoded[7U] == '1' &&
                  encoded[8U] == 1U && encoded[12U] == 0x04U &&
                  encoded[20U] == 14U && encoded[24U] == kLayer &&
                  encoded[28U] == static_cast<std::uint8_t>(
                                      kernels::Sm87TargetAotProjectionRole::
                                          kNvFp4GateUp) &&
                  read_u64_le(encoded.data() + 32U) ==
                      persistence::persistence_expected_arena_offset(
                          kLayer,
                          kernels::Sm87TargetAotProjectionRole::
                              kNvFp4GateUp),
              "record golden prefix fixes ordering and little-endian offsets");

  constexpr std::string_view kGoldenSha256 =
      "f8a08ffcf55ed2dedf06eaa2cadc8b7901b1431d1726365f83b5c691e6eac24a";
  const std::string observed_sha256 =
      sha256_hex(encoded.data(), encoded.size());
  test.expect(observed_sha256 == kGoldenSha256,
              "record-header golden SHA-256 matches: observed=" +
                  observed_sha256);

  persistence::PersistenceRecord decoded;
  test.expect(persistence::decode_record_header(encoded.data(), encoded.size(),
                                                &decoded)
                  .ok(),
              "valid record header decodes");
  test.expect(decoded.ordinal == 14U && decoded.layer_index == kLayer &&
                  decoded.role ==
                      kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp &&
                  decoded.manifest.payload_digest ==
                      fixture.manifest.payload_digest &&
                  kernels::sm87_target_aot_projection_validate_packed_manifest(
                      decoded.manifest, decoded.source_inventory) &&
                  kernels::
                      sm87_target_aot_projection_validate_transform_receipt(
                          decoded.manifest, decoded.source_inventory,
                          decoded.transform_receipt),
              "record reconstructs inventory and closes manifest/transform "
              "validation");
  persistence::RecordHeaderBytes replay{};
  test.expect(persistence::encode_record_header(
                  decoded.layer_index, decoded.manifest,
                  decoded.transform_receipt, replay.data(), replay.size()) &&
                  replay == encoded,
              "record decode/re-encode is canonical");

  auto corrupted = encoded;
  corrupted[persistence::kRecordHeaderDigestOffset] ^= 1U;
  test.expect(
      persistence::decode_record_header(corrupted.data(), corrupted.size(),
                                        &decoded)
              .error == persistence::PersistenceError::kHeaderDigestMismatch,
      "record header digest corruption fails closed");
  corrupted = encoded;
  corrupted.back() = 1U;
  test.expect(
      persistence::decode_record_header(corrupted.data(), corrupted.size(),
                                        &decoded)
              .error ==
          persistence::PersistenceError::kReservedBytesNonZero,
      "record reserved bytes must remain zero");
  test.expect(
      persistence::decode_record_header(encoded.data(), encoded.size() - 1U,
                                        &decoded)
              .error == persistence::PersistenceError::kTruncated,
      "truncated record header is rejected before parsing");

  corrupted = encoded;
  corrupted[28U] = 0xffU;
  reseal_header(corrupted.data(), corrupted.size(),
                persistence::kRecordHeaderDigestOffset);
  test.expect(persistence::decode_record_header(
                  corrupted.data(), corrupted.size(), &decoded)
                  .error == persistence::PersistenceError::kInvalidEnum,
              "well-digested unknown enum still fails closed");

  // The first manifest boolean follows 417 canonical manifest bytes.
  corrupted = encoded;
  corrupted[persistence::kRecordManifestOffset + 417U] = 2U;
  reseal_header(corrupted.data(), corrupted.size(),
                persistence::kRecordHeaderDigestOffset);
  test.expect(persistence::decode_record_header(
                  corrupted.data(), corrupted.size(), &decoded)
                  .error == persistence::PersistenceError::kInvalidBoolean,
              "well-digested noncanonical boolean still fails closed");

  corrupted = encoded;
  corrupted[persistence::kRecordManifestOffset + 429U] ^= 1U;
  reseal_header(corrupted.data(), corrupted.size(),
                persistence::kRecordHeaderDigestOffset);
  test.expect(persistence::decode_record_header(
                  corrupted.data(), corrupted.size(), &decoded)
                  .error == persistence::PersistenceError::kManifestMismatch,
              "well-digested manifest corruption cannot retain authority");

  corrupted = encoded;
  write_u16_le(corrupted.data() + 8U, 2U);
  reseal_header(corrupted.data(), corrupted.size(),
                persistence::kRecordHeaderDigestOffset);
  test.expect(persistence::decode_record_header(
                  corrupted.data(), corrupted.size(), &decoded)
                  .error ==
              persistence::PersistenceError::kUnsupportedVersion,
              "well-digested record ABI mismatch fails closed");
  corrupted = encoded;
  write_u32_le(corrupted.data() + 12U, 0x0102'0305U);
  reseal_header(corrupted.data(), corrupted.size(),
                persistence::kRecordHeaderDigestOffset);
  test.expect(persistence::decode_record_header(
                  corrupted.data(), corrupted.size(), &decoded)
                  .error == persistence::PersistenceError::kInvalidEndian,
              "well-digested record endian mismatch fails closed");
  corrupted = encoded;
  write_u32_le(corrupted.data() + 16U, 2'048U);
  reseal_header(corrupted.data(), corrupted.size(),
                persistence::kRecordHeaderDigestOffset);
  test.expect(persistence::decode_record_header(
                  corrupted.data(), corrupted.size(), &decoded)
                  .error == persistence::PersistenceError::kFixedFieldMismatch,
              "well-digested record header-size mismatch fails closed");
  corrupted = encoded;
  write_u32_le(corrupted.data() + 20U, 15U);
  reseal_header(corrupted.data(), corrupted.size(),
                persistence::kRecordHeaderDigestOffset);
  test.expect(persistence::decode_record_header(
                  corrupted.data(), corrupted.size(), &decoded)
                  .error == persistence::PersistenceError::kFixedFieldMismatch,
              "well-digested record ordinal mismatch fails closed");
  corrupted = encoded;
  write_u64_le(corrupted.data() + 40U,
               read_u64_le(encoded.data() + 40U) + 4'096U);
  reseal_header(corrupted.data(), corrupted.size(),
                persistence::kRecordHeaderDigestOffset);
  test.expect(persistence::decode_record_header(
                  corrupted.data(), corrupted.size(), &decoded)
                  .error == persistence::PersistenceError::kFixedFieldMismatch,
              "well-digested record file-offset mismatch fails closed");
  corrupted = encoded;
  write_u32_le(corrupted.data() + 64U, 256U);
  reseal_header(corrupted.data(), corrupted.size(),
                persistence::kRecordHeaderDigestOffset);
  test.expect(persistence::decode_record_header(
                  corrupted.data(), corrupted.size(), &decoded)
                  .error == persistence::PersistenceError::kFixedFieldMismatch,
              "well-digested record section-size mismatch fails closed");
}

void test_exact_bundle_cover(TestContext& test) {
  const RecordFixture fixture = make_record_fixture(
      kernels::Sm87TargetAotProjectionRole::kNvFp4Down, 64U);
  persistence::RecordHeaderBytes encoded{};
  persistence::PersistenceRecord decoded;
  test.expect(
      persistence::encode_record_header(
          63U, fixture.manifest, fixture.receipt, encoded.data(),
          encoded.size()) &&
          persistence::decode_record_header(encoded.data(), encoded.size(),
                                            &decoded),
      "last layer Down record round-trips");
  constexpr std::string_view kDownGoldenSha256 =
      "b0fafc9552a6ccc0f08123408a38a70adccc34a7a87fe396bf4864b64cb8e144";
  const std::string observed_down_sha256 =
      sha256_hex(encoded.data(), encoded.size());
  test.expect(observed_down_sha256 == kDownGoldenSha256,
              "Down record-header golden SHA-256 matches: observed=" +
                  observed_down_sha256);
  test.expect(decoded.ordinal == 127U &&
                  decoded.arena_offset + decoded.manifest.payload_bytes ==
                      persistence::kPersistedArenaBytes &&
                  decoded.bundle_header_offset +
                          decoded.bundle_record_bytes ==
                      persistence::kPersistedBundleBytes,
              "fixed layer-major records exactly cover arena and bundle");
}

void test_secure_file_open(TestContext& test,
                           const std::filesystem::path& root) {
  constexpr std::uint64_t kSmallBytes = 4'096U;
  const auto expect_rejected = [&test](const std::filesystem::path& path,
                                       const std::uint64_t expected_bytes,
                                       const std::string& label) {
    persistence::UniqueFd file;
    persistence::FileSnapshot snapshot;
    std::string message;
    test.expect(!persistence::open_regular_file_exact(
                    path, expected_bytes, file, snapshot, message) &&
                    !file && !message.empty(),
                label);
  };

  expect_rejected("relative/bundle.aot", kSmallBytes,
                  "relative bundle path is rejected");
  expect_rejected(root / "." / "dot-component.aot", kSmallBytes,
                  "dot path component is rejected");
  expect_rejected(root / "child" / ".." / "dotdot-component.aot",
                  kSmallBytes, "dot-dot path component is rejected");
  std::string embedded_nul = (root / "embedded-nul").string();
  embedded_nul.push_back('\0');
  embedded_nul.append("suffix.aot");
  expect_rejected(std::filesystem::path(embedded_nul), kSmallBytes,
                  "embedded NUL path component is rejected");

  const auto exact = root / "exact-sparse.aot";
  test.expect(create_sparse_file(exact, persistence::kPersistedBundleBytes),
              "exact production-size sparse bundle fixture is created");
  persistence::UniqueFd exact_file;
  persistence::FileSnapshot exact_snapshot;
  std::string message;
  test.expect(persistence::open_regular_file_exact(
                  exact, persistence::kPersistedBundleBytes, exact_file,
                  exact_snapshot, message) &&
                  exact_file &&
                  exact_snapshot.bytes == persistence::kPersistedBundleBytes,
              "exact production-size sparse regular file opens securely");

  const auto short_file = root / "short.aot";
  const auto long_file = root / "long.aot";
  test.expect(create_sparse_file(short_file, kSmallBytes - 1U) &&
                  create_sparse_file(long_file, kSmallBytes + 1U),
              "size mismatch fixtures are created");
  expect_rejected(short_file, kSmallBytes,
                  "one-byte-short bundle is rejected");
  expect_rejected(long_file, kSmallBytes,
                  "one-byte-long bundle is rejected");

  const auto directory = root / "directory.aot";
  std::error_code error;
  test.expect(std::filesystem::create_directory(directory, error) && !error,
              "directory fixture is created");
  expect_rejected(directory, kSmallBytes,
                  "directory cannot masquerade as a bundle");

  const auto fifo = root / "fifo.aot";
  test.expect(::mkfifo(fifo.c_str(), S_IRUSR | S_IWUSR) == 0,
              "FIFO fixture is created");
  expect_rejected(fifo, kSmallBytes,
                  "FIFO cannot block or masquerade as a bundle");

  const auto symlink_target = root / "symlink-target.aot";
  const auto final_symlink = root / "final-symlink.aot";
  test.expect(create_sparse_file(symlink_target, kSmallBytes),
              "final symlink target is created");
  error.clear();
  std::filesystem::create_symlink(symlink_target.filename(), final_symlink,
                                  error);
  test.expect(!error, "final symlink fixture is created");
  expect_rejected(final_symlink, kSmallBytes,
                  "final symlink is rejected without traversal");

  const auto real_parent = root / "real-parent";
  const auto parent_symlink = root / "parent-symlink";
  error.clear();
  test.expect(std::filesystem::create_directory(real_parent, error) && !error,
              "real parent fixture is created");
  test.expect(create_sparse_file(real_parent / "bundle.aot", kSmallBytes),
              "parent symlink target bundle is created");
  error.clear();
  std::filesystem::create_directory_symlink(real_parent.filename(),
                                            parent_symlink, error);
  test.expect(!error, "parent symlink fixture is created");
  expect_rejected(parent_symlink / "bundle.aot", kSmallBytes,
                  "symlinked parent directory is rejected");

  const auto hardlink_source = root / "hardlink-source.aot";
  const auto hardlink_alias = root / "hardlink-alias.aot";
  test.expect(create_sparse_file(hardlink_source, kSmallBytes),
              "hardlink source fixture is created");
  error.clear();
  std::filesystem::create_hard_link(hardlink_source, hardlink_alias, error);
  test.expect(!error, "hardlink fixture is created");
  expect_rejected(hardlink_source, kSmallBytes,
                  "multi-link regular file is rejected");

  const auto changing = root / "changing.aot";
  test.expect(create_sparse_file(changing, kSmallBytes),
              "snapshot mutation fixture is created");
  persistence::UniqueFd changing_file;
  persistence::FileSnapshot before;
  message.clear();
  test.expect(persistence::open_regular_file_exact(
                  changing, kSmallBytes, changing_file, before, message),
              "snapshot mutation fixture opens");
  test.expect(persistence::same_snapshot(before, before),
              "identical file snapshots compare equal");
  test.expect(::fchmod(changing_file.get(), S_IRUSR) == 0,
              "opened bundle metadata is mutated deterministically");
  persistence::FileSnapshot after;
  message.clear();
  test.expect(persistence::snapshot_regular_file_exact(
                  changing_file.get(), kSmallBytes, after, message) &&
                  !persistence::same_snapshot(before, after),
              "snapshot comparison detects a same-size metadata change");
}

void test_atomic_create_only_publication(
    TestContext& test, const std::filesystem::path& root) {
  constexpr std::size_t kBytes = 4'096U;
  std::vector<std::uint8_t> expected(kBytes);
  for (std::size_t index = 0U; index < expected.size(); ++index) {
    expected[index] = static_cast<std::uint8_t>((index * 37U + 11U) & 0xffU);
  }
  const auto destination = root / "atomic-published.aot";
  persistence::AtomicCreateOnlyFile writer;
  std::string message;
  test.expect(writer.open_create_only(destination, expected.size(), false,
                                      message) &&
                  writer.is_open() &&
                  writer.expected_bytes() == expected.size(),
              "small create-only atomic writer opens an unpublished file");
  std::uint64_t bytes_written = 0U;
  message.clear();
  test.expect(persistence::pwrite_exact_counted(
                  writer.fd(), expected.data(), expected.size(), 0U,
                  bytes_written, message) &&
                  bytes_written == expected.size(),
              "atomic writer uses the shared exact counted write path");
  message.clear();
  test.expect(writer.sync_data_and_drop_cache(0U, expected.size(), message),
              "small atomic file exercises the production sync policy");
  message.clear();
  test.expect(writer.publish(message) ==
                      persistence::AtomicPublishOutcome::kPublishedDurable &&
                  std::filesystem::is_regular_file(destination),
              "create-only publication atomically exposes the complete file");

  persistence::UniqueFd published;
  persistence::FileSnapshot snapshot;
  message.clear();
  test.expect(persistence::open_regular_file_exact(
                  destination, expected.size(), published, snapshot,
                  message),
              "published file reopens through the production secure path");
  std::vector<std::uint8_t> observed(kBytes);
  std::uint64_t bytes_read = 0U;
  message.clear();
  test.expect(persistence::pread_exact_counted(
                  published.get(), observed.data(), observed.size(), 0U,
                  bytes_read, message) &&
                  bytes_read == observed.size() && observed == expected,
              "published bytes exactly match the pre-publication payload");

  persistence::AtomicCreateOnlyFile replacement;
  message.clear();
  test.expect(!replacement.open_create_only(
                  destination, expected.size(), false, message) &&
                  !replacement.is_open() && !message.empty(),
              "create-only publication refuses an existing destination");
  std::fill(observed.begin(), observed.end(), 0U);
  bytes_read = 0U;
  message.clear();
  test.expect(persistence::pread_exact_counted(
                  published.get(), observed.data(), observed.size(), 0U,
                  bytes_read, message) &&
                  observed == expected,
              "refused replacement leaves the published file unchanged");

  const auto abandoned = root / "abandoned.aot";
  {
    persistence::AtomicCreateOnlyFile unpublished;
    message.clear();
    test.expect(unpublished.open_create_only(
                    abandoned, expected.size(), false, message),
                "unpublished create-only file opens");
  }
  test.expect(!std::filesystem::exists(abandoned),
              "destroying an unpublished writer exposes no final file");
  bool temporary_left = false;
  for (const auto& entry : std::filesystem::directory_iterator(root)) {
    const std::string name = entry.path().filename().string();
    temporary_left = temporary_left ||
                     name.find(".abandoned.aot.q3x-tmp-") == 0U;
  }
  test.expect(!temporary_left,
              "destroying an unpublished writer removes its exact temp file");

  const auto stale_destination = root / "stale-destination.aot";
  const auto stale_temporary =
      root / ".stale-destination.aot.q3x-tmp-4242-7";
  test.expect(create_sparse_file(stale_temporary, 17U),
              "stale temporary fixture is created");
  persistence::AtomicCreateOnlyFile stale_guard;
  message.clear();
  test.expect(!stale_guard.open_create_only(
                  stale_destination, expected.size(), false, message) &&
                  !stale_guard.is_open() &&
                  message.find("requires manual audit") != std::string::npos &&
                  std::filesystem::exists(stale_temporary) &&
                  !std::filesystem::exists(stale_destination),
              "stale same-target temp fails closed without automatic deletion");
}

}  // namespace

int main(const int argc, char** const argv) {
  static_assert(persistence::kPersistedLayerCount == 64U);
  static_assert(persistence::kPersistedArtifactCount == 128U);
  static_assert(persistence::kPersistedSourceCount == 192U);
  static_assert(persistence::kPersistedArenaBytes == 9'625'927'680ULL);
  static_assert(persistence::kPersistedBundleBytes == 9'626'456'064ULL);

  TestContext test;
  test_superblock(test);
  test_record_header(test);
  test_exact_bundle_cover(test);
  test.expect(argc == 2, "host filesystem test requires one workspace path");
  if (argc == 2) {
    ScratchRoot scratch(argv[1]);
    test.expect(scratch.ready(),
                "host filesystem scratch directory is repository-local");
    if (scratch.ready()) {
      test_secure_file_open(test, scratch.path());
      test_atomic_create_only_publication(test, scratch.path());
    }
  }
  if (test.failures() != 0) {
    std::cerr << test.failures()
              << " target-AOT persistence host check(s) failed\n";
    return 1;
  }
  std::cout << "SM87 target-AOT persistence host checks passed\n";
  return 0;
}
