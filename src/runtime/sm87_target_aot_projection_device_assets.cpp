#include "q3x/runtime/sm87_target_aot_projection_device_assets.h"

#include "q3x/core/sha256.h"

#if defined(Q3X_ENABLE_SM87_TARGET_AOT_DEVICE_ASSETS_V1_ADMISSION)
#include "sm87_target_aot_projection_persistence_internal.h"

#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace q3x::runtime {
namespace {

#if defined(Q3X_ENABLE_SM87_TARGET_AOT_DEVICE_ASSETS_V1_ADMISSION)

namespace st = io::safetensors;
namespace persistence = target_aot_persistence_detail;

constexpr std::string_view kModelRepository =
    "nvidia/Qwen3.6-27B-NVFP4";
constexpr std::string_view kModelRevision =
    "0893e1606ff3d5f97a441f405d5fc541a6bdf404";
constexpr std::string_view kCheckpointIdentityDomain =
    "q3x.sm87.target-aot.device-assets.checkpoint.v1";
constexpr std::string_view kSourceIdentityDomain =
    "q3x.sm87.target-aot.device-assets.source.v1";
constexpr std::string_view kInventoryIdentityDomain =
    "q3x.sm87.target-aot.device-assets.inventory.v1";
constexpr std::string_view kArtifactIdentityDomain =
    "q3x.sm87.target-aot.device-assets.artifact.v1";
constexpr std::string_view kOwnerIdentityDomain =
    "q3x.sm87.target-aot.device-assets.owner.v1";
constexpr std::string_view kAllocationIdentityDomain =
    "q3x.sm87.target-aot.device-assets.allocation.v1";
constexpr std::string_view kStreamIdentityDomain =
    "q3x.sm87.target-aot.device-assets.stream.v1";
constexpr std::string_view kEventIdentityDomain =
    "q3x.sm87.target-aot.device-assets.event.v1";
constexpr std::string_view kVerifiedPayloadCatalogDomain =
    "q3x.sm87.target-aot.device-assets.verified-payload-catalog.v1";

static_assert(persistence::kPinnedModelRepository == kModelRepository);
static_assert(persistence::kPinnedModelRevision == kModelRevision);
static_assert(persistence::kPersistedLayerCount ==
              kSm87TargetAotProjectionDeviceLayerCount);
static_assert(persistence::kPersistedArtifactCount ==
              kSm87TargetAotProjectionDeviceArtifactCount);
static_assert(persistence::kPersistedSourceCount ==
              kSm87TargetAotProjectionDeviceSourceCount);
static_assert(persistence::kPersistedArenaBytes ==
              kSm87TargetAotProjectionDeviceArenaBytes);
static_assert(persistence::kPersistedBundleBytes ==
              kSm87TargetAotProjectionPersistentBundleBytes);
std::atomic<std::uint64_t> g_device_asset_transaction_serial{1U};

struct PlannedSource final {
  kernels::Sm87TargetAotLogicalRole logical_role =
      kernels::Sm87TargetAotLogicalRole::kInvalid;
  std::string module_name;
  const NvFp4LinearWeight* projection = nullptr;
  const DeviceTensorView* weight = nullptr;
  const DeviceTensorView* block_scale = nullptr;
  const DeviceTensorView* tensor_scale = nullptr;
  std::uint64_t tensor_identity = 0U;
};

struct PlannedArtifact final {
  std::size_t layer_index = 0U;
  kernels::Sm87TargetAotProjectionRole role =
      kernels::Sm87TargetAotProjectionRole::kInvalid;
  std::uint64_t inventory_identity = 0U;
  std::uint64_t artifact_identity = 0U;
  std::uint64_t device_arena_offset = 0U;
  std::uint32_t source_count = 0U;
  std::array<PlannedSource,
             kernels::kSm87TargetAotProjectionPackedMaxPartitions>
      sources{};
};

using ArtifactPlans =
    std::array<PlannedArtifact,
               kSm87TargetAotProjectionDeviceArtifactCount>;

template <typename Unsigned>
[[nodiscard]] bool hash_unsigned(core::Sha256& hasher,
                                 Unsigned value) noexcept {
  static_assert(std::is_unsigned_v<Unsigned>);
  std::array<std::uint8_t, sizeof(Unsigned)> bytes{};
  for (std::size_t index = 0U; index < bytes.size(); ++index) {
    bytes[index] = static_cast<std::uint8_t>(value >> (8U * index));
  }
  return hasher.update(bytes.data(), bytes.size());
}

[[nodiscard]] bool hash_string(core::Sha256& hasher,
                               const std::string_view value) noexcept {
  return hash_unsigned(hasher, static_cast<std::uint64_t>(value.size())) &&
         hasher.update(value.data(), value.size());
}

[[nodiscard]] kernels::Sm87TargetAotProjectionSha256Digest copy_digest(
    const core::Sha256Digest& source) noexcept {
  kernels::Sm87TargetAotProjectionSha256Digest result;
  result.bytes = source.bytes;
  return result;
}

[[nodiscard]] bool parse_lower_hex_digest(
    const std::string_view text,
    kernels::Sm87TargetAotProjectionSha256Digest& digest) noexcept {
  digest = {};
  if (text.size() != 64U) {
    return false;
  }
  const auto nibble = [](const char value, std::uint8_t& output) noexcept {
    if (value >= '0' && value <= '9') {
      output = static_cast<std::uint8_t>(value - '0');
      return true;
    }
    if (value >= 'a' && value <= 'f') {
      output = static_cast<std::uint8_t>(value - 'a' + 10);
      return true;
    }
    return false;
  };
  for (std::size_t index = 0U; index < digest.bytes.size(); ++index) {
    std::uint8_t high = 0U;
    std::uint8_t low = 0U;
    if (!nibble(text[2U * index], high) ||
        !nibble(text[2U * index + 1U], low)) {
      digest = {};
      return false;
    }
    digest.bytes[index] =
        static_cast<std::uint8_t>((high << 4U) | low);
  }
  return !kernels::sm87_target_aot_projection_digest_is_zero(digest);
}

[[nodiscard]] bool initialize_record_header_catalog(
    core::Sha256& hasher) noexcept {
  // The fixed ordinal and record count are already authenticated by every
  // canonical header plus the superblock. The v1 wire definition is SHA-256
  // over the 128 complete 4096-byte headers in ordinal order.
  (void)hasher;
  return true;
}

[[nodiscard]] bool append_record_header_catalog_entry(
    core::Sha256& hasher, const std::size_t ordinal,
    const std::uint8_t* const header,
    const std::size_t header_bytes) noexcept {
  return ordinal < persistence::kPersistedArtifactCount &&
         header != nullptr &&
         header_bytes == persistence::kRecordHeaderBytes &&
         hasher.update(header, header_bytes);
}

class PersistentBundleWriter final {
 public:
  PersistentBundleWriter() noexcept = default;
  ~PersistentBundleWriter() = default;

  PersistentBundleWriter(const PersistentBundleWriter&) = delete;
  PersistentBundleWriter& operator=(const PersistentBundleWriter&) = delete;

  [[nodiscard]] bool open_create_only(const std::filesystem::path& output,
                                      std::string& message) {
    return file_.open_create_only(output, persistence::kPersistedBundleBytes,
                                  true, message) &&
           initialize_record_header_catalog(header_catalog_);
  }

  [[nodiscard]] bool write_record(
      const std::size_t ordinal,
      const persistence::PersistenceRecord& record,
      const std::uint8_t* const payload, const std::size_t payload_bytes,
      std::string& message) {
    if (!file_.is_open() || ordinal != records_written_ ||
        ordinal >= persistence::kPersistedArtifactCount ||
        record.ordinal != ordinal || payload == nullptr ||
        payload_bytes != record.manifest.payload_bytes) {
      message = "target-AOT persistent writer record order is invalid";
      return false;
    }
    persistence::RecordHeaderBytes header{};
    if (!persistence::encode_record_header(
            record.layer_index, record.manifest, record.transform_receipt,
            header.data(), header.size())) {
      message = "encoding target-AOT persistent record header failed";
      return false;
    }
    if (!persistence::pwrite_exact_counted(
            file_.fd(), header.data(), header.size(),
            record.bundle_header_offset, bytes_written_, message) ||
        !persistence::pwrite_exact_counted(
            file_.fd(), payload, payload_bytes,
            record.bundle_payload_offset, bytes_written_, message) ||
        !append_record_header_catalog_entry(
            header_catalog_, ordinal, header.data(), header.size())) {
      if (message.empty()) {
        message = "updating target-AOT record-header catalog failed";
      }
      return false;
    }
    ++records_written_;
    if ((ordinal & 1U) != 0U) {
      const std::uint64_t layer_begin =
          persistence::persistence_expected_bundle_header_offset(
              record.layer_index,
              kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp);
      const std::uint64_t layer_end =
          record.bundle_payload_offset + record.manifest.payload_bytes;
      if (!file_.sync_data_and_drop_cache(
              layer_begin, layer_end - layer_begin, message)) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool publish(
      const kernels::Sm87TargetAotProjectionSha256Digest& checkpoint_digest,
      const kernels::Sm87TargetAotProjectionSha256Digest& payload_catalog,
      std::string& header_catalog_sha256, std::string& message) {
    if (!file_.is_open() ||
        records_written_ != persistence::kPersistedArtifactCount) {
      message = "target-AOT persistent writer did not close all records";
      return false;
    }
    const core::Sha256Digest header_catalog = header_catalog_.finalize();
    const auto persistent_header_catalog = copy_digest(header_catalog);
    if (kernels::sm87_target_aot_projection_digest_is_zero(
            persistent_header_catalog)) {
      message = "target-AOT record-header catalog digest was zero";
      return false;
    }
    const std::string header_catalog_hex = header_catalog.hex();
    if (header_catalog_hex.size() != 64U) {
      message = "target-AOT record-header catalog encoding failed";
      return false;
    }
    // Complete the only potentially allocating result publication before
    // the filesystem rename. Once the final name is visible, no ordinary
    // allocation failure may obscure the file transaction outcome.
    header_catalog_sha256 = header_catalog_hex;
    persistence::PersistenceSuperblock superblock;
    superblock.checkpoint_identity_sha256 = checkpoint_digest;
    superblock.authenticated_payload_catalog_sha256 = payload_catalog;
    superblock.record_header_catalog_sha256 = persistent_header_catalog;
    persistence::SuperblockBytes encoded{};
    if (!persistence::encode_superblock(superblock, encoded.data(),
                                        encoded.size()) ||
        !persistence::pwrite_exact_counted(
            file_.fd(), encoded.data(), encoded.size(), 0U,
            bytes_written_, message)) {
      if (message.empty()) {
        message = "encoding target-AOT persistent superblock failed";
      }
      return false;
    }
    if (bytes_written_ != persistence::kPersistedBundleBytes) {
      message = "target-AOT persistent writer byte accounting did not close";
      return false;
    }
    const persistence::AtomicPublishOutcome publish_outcome =
        file_.publish(message);
    if (publish_outcome !=
        persistence::AtomicPublishOutcome::kPublishedDurable) {
      return false;
    }
    return true;
  }

  [[nodiscard]] std::uint64_t bytes_written() const noexcept {
    return bytes_written_;
  }

 private:
  persistence::AtomicCreateOnlyFile file_;
  core::Sha256 header_catalog_;
  std::size_t records_written_ = 0U;
  std::uint64_t bytes_written_ = 0U;
};

[[nodiscard]] std::uint64_t digest_identity(
    const core::Sha256Digest& digest) noexcept {
  std::uint64_t value = 0U;
  for (std::size_t index = 0U; index < sizeof(value); ++index) {
    value |= static_cast<std::uint64_t>(digest.bytes[index]) <<
             (8U * index);
  }
  return value;
}

[[nodiscard]] std::uint64_t finish_identity(core::Sha256& hasher) noexcept {
  return digest_identity(hasher.finalize());
}

[[nodiscard]] std::uint32_t float_bits(const float value) noexcept {
  std::uint32_t bits = 0U;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

[[nodiscard]] bool empty_p40_packed_artifact_view(
    const kernels::Sm87P40PackedProjectionDeviceView& view) noexcept {
  return view.payload == nullptr && view.payload_bytes == 0U &&
         view.artifact_identity == 0U &&
         view.role == kernels::Sm87P40PackedProjectionRole::kInvalid &&
         view.tactic == kernels::Sm87P40PackedTactic::kInvalid &&
         view.source_count == 0U &&
         std::all_of(view.scalar_scales.begin(), view.scalar_scales.end(),
                     [](const float scale) { return scale == 0.0F; });
}

[[nodiscard]] bool empty_nvfp4_marlin_p40_parity_view(
    const NvFp4MarlinP40ParityDeviceView& view) noexcept {
  const auto digest_empty = [](const NvFp4MarlinP40ParityDigest& digest) {
    return std::all_of(digest.begin(), digest.end(),
                       [](const std::uint8_t byte) { return byte == 0U; });
  };
  const auto source_empty = [&digest_empty](
                                const NvFp4MarlinP40ParitySourceManifest&
                                    source) {
    return source.role == NvFp4MarlinP40ParitySourceRole::kInvalid &&
           source.tensor_identity == 0U &&
           digest_empty(source.weight_digest) &&
           digest_empty(source.scale_digest) &&
           source.global_scale_bits == 0U;
  };
  const auto& manifest = view.manifest;
  return view.weight == nullptr && view.scales == nullptr &&
         view.global_scale == nullptr &&
         manifest.version == kNvFp4MarlinP40ParityManifestVersion &&
         manifest.layer_index == 0U &&
         manifest.role == NvFp4MarlinP40ParityRole::kInvalid &&
         manifest.layout == NvFp4MarlinP40ParityLayout::kInvalid &&
         manifest.output_features == 0U && manifest.input_features == 0U &&
         manifest.weight_bytes == 0U && manifest.scale_bytes == 0U &&
         manifest.artifact_identity == 0U &&
         digest_empty(manifest.transformation_digest) &&
         manifest.source_count == 0U &&
         std::all_of(manifest.sources.begin(), manifest.sources.end(),
                     source_empty);
}

[[nodiscard]] bool resident_range_contains(
    const ResidentWeights& resident, const DeviceTensorView& view) noexcept {
  if (!resident || resident.arena_data() == nullptr ||
      resident.size_bytes() == 0U || view.device_data == nullptr ||
      view.byte_size == 0U || view.arena_offset > resident.size_bytes() ||
      view.byte_size > resident.size_bytes() - view.arena_offset) {
    return false;
  }
  const auto base = reinterpret_cast<std::uintptr_t>(resident.arena_data());
  if (resident.size_bytes() >
          std::numeric_limits<std::uintptr_t>::max() - base ||
      view.arena_offset >
          std::numeric_limits<std::uintptr_t>::max() - base) {
    return false;
  }
  return reinterpret_cast<std::uintptr_t>(view.device_data) ==
         base + static_cast<std::uintptr_t>(view.arena_offset);
}

[[nodiscard]] const DeviceTensorView* exact_resident_tensor(
    const ResidentWeights& resident, const std::string& name,
    const void* const expected_pointer, const st::DType expected_dtype,
    const std::initializer_list<std::uint64_t> expected_shape,
    const std::uint64_t expected_bytes) noexcept {
  const DeviceTensorView* const view = resident.find(name);
  if (view == nullptr || view->device_data != expected_pointer ||
      view->dtype != expected_dtype || view->byte_size != expected_bytes ||
      view->shape.size() != expected_shape.size() ||
      !resident_range_contains(resident, *view) ||
      !std::equal(view->shape.begin(), view->shape.end(),
                  expected_shape.begin(), expected_shape.end())) {
    return nullptr;
  }
  return view;
}

[[nodiscard]] bool verify_observed_checkpoint(
    const ResidentWeights& resident, core::Sha256Digest& checkpoint_digest,
    std::string& message) {
  if (!resident || resident.arena_data() == nullptr ||
      resident.size_bytes() != kPinnedQwen36_27BArenaBytes) {
    message = "resident arena is not the exact pinned Qwen3.6-27B arena";
    return false;
  }
  const auto& pinned = pinned_qwen36_27b_shards();
  const auto& observed = resident.stats().shards;
  if (pinned.empty() || observed.size() != pinned.size()) {
    message = "resident loader did not report the complete pinned shard set";
    return false;
  }
  std::array<bool, 16U> used{};
  if (observed.size() > used.size()) {
    message = "pinned shard set exceeds the fixed provenance bound";
    return false;
  }
  core::Sha256 hasher;
  bool ok = hash_string(hasher, kCheckpointIdentityDomain) &&
            hash_string(hasher, kModelRepository) &&
            hash_string(hasher, kModelRevision) &&
            hash_unsigned(hasher,
                          static_cast<std::uint64_t>(pinned.size()));
  for (const ShardIdentity& expected : pinned) {
    std::size_t match = observed.size();
    for (std::size_t index = 0U; index < observed.size(); ++index) {
      if (!used[index] && observed[index].filename == expected.filename) {
        match = index;
        break;
      }
    }
    if (match == observed.size() ||
        observed[match].sha256 != expected.sha256 ||
        observed[match].bytes_read != expected.file_size) {
      message = "resident loader shard identity differs from the pinned "
                "checkpoint at " +
                expected.filename;
      return false;
    }
    used[match] = true;
    ok = ok && hash_string(hasher, expected.filename) &&
         hash_unsigned(hasher, expected.file_size) &&
         hash_string(hasher, observed[match].sha256);
  }
  checkpoint_digest = hasher.finalize();
  if (!ok || digest_identity(checkpoint_digest) == 0U) {
    checkpoint_digest = {};
    message = "failed to derive the pinned checkpoint identity";
    return false;
  }
  return true;
}

[[nodiscard]] std::uint64_t make_source_identity(
    const core::Sha256Digest& checkpoint_digest,
    const std::string_view module_name,
    const kernels::Sm87TargetAotLogicalRole role) noexcept {
  core::Sha256 hasher;
  const bool ok = hash_string(hasher, kSourceIdentityDomain) &&
                  hasher.update(checkpoint_digest.bytes.data(),
                                checkpoint_digest.bytes.size()) &&
                  hash_string(hasher, module_name) &&
                  hash_unsigned(hasher, static_cast<std::uint8_t>(role));
  const std::uint64_t identity = finish_identity(hasher);
  return ok ? identity : 0U;
}

[[nodiscard]] std::uint64_t make_inventory_identity(
    const core::Sha256Digest& checkpoint_digest,
    const PlannedArtifact& planned) noexcept {
  core::Sha256 hasher;
  bool ok = hash_string(hasher, kInventoryIdentityDomain) &&
            hasher.update(checkpoint_digest.bytes.data(),
                          checkpoint_digest.bytes.size()) &&
            hash_unsigned(hasher,
                          static_cast<std::uint64_t>(planned.layer_index)) &&
            hash_unsigned(hasher, static_cast<std::uint8_t>(planned.role)) &&
            hash_unsigned(hasher, planned.source_count);
  for (std::size_t index = 0U; index < planned.source_count; ++index) {
    ok = ok && hash_unsigned(hasher, planned.sources[index].tensor_identity);
  }
  const std::uint64_t identity = finish_identity(hasher);
  return ok ? identity : 0U;
}

[[nodiscard]] std::uint64_t make_artifact_identity(
    const core::Sha256Digest& checkpoint_digest,
    const PlannedArtifact& planned) noexcept {
  core::Sha256 hasher;
  const bool ok = hash_string(hasher, kArtifactIdentityDomain) &&
                  hasher.update(checkpoint_digest.bytes.data(),
                                checkpoint_digest.bytes.size()) &&
                  hash_unsigned(
                      hasher,
                      static_cast<std::uint64_t>(planned.layer_index)) &&
                  hash_unsigned(hasher,
                                static_cast<std::uint8_t>(planned.role)) &&
                  hash_unsigned(hasher, planned.inventory_identity) &&
                  hash_unsigned(
                      hasher,
                      static_cast<std::uint16_t>(
                          kernels::Sm87TargetAotProjectionPackedTransformIdentity::
                              kCanonicalNkToConsumerN64K16LaneComponentV1));
  const std::uint64_t identity = finish_identity(hasher);
  return ok ? identity : 0U;
}

[[nodiscard]] std::uint64_t make_runtime_identity(
    const std::string_view domain, const core::Sha256Digest& checkpoint_digest,
    const std::uint64_t transaction_serial, const std::uint64_t first,
    const std::uint64_t second = 0U,
    const std::uint64_t third = 0U) noexcept {
  core::Sha256 hasher;
  const bool ok = hash_string(hasher, domain) &&
                  hasher.update(checkpoint_digest.bytes.data(),
                                checkpoint_digest.bytes.size()) &&
                  hash_unsigned(hasher, transaction_serial) &&
                  hash_unsigned(hasher, first) &&
                  hash_unsigned(hasher, second) &&
                  hash_unsigned(hasher, third);
  const std::uint64_t identity = finish_identity(hasher);
  return ok ? identity : 0U;
}

[[nodiscard]] bool append_verified_payload_catalog_entry(
    core::Sha256& hasher, const PlannedArtifact& planned,
    const kernels::Sm87TargetAotProjectionPackedManifest& manifest,
    const kernels::Sm87TargetAotProjectionPackedSourceInventory& inventory,
    const kernels::Sm87TargetAotProjectionPackedTransformReceipt&
        transform_receipt,
    const kernels::Sm87TargetAotProjectionSha256Digest&
        verified_payload_digest) noexcept {
  bool ok =
      hash_unsigned(hasher,
                    static_cast<std::uint64_t>(planned.layer_index)) &&
      hash_unsigned(hasher, static_cast<std::uint8_t>(planned.role)) &&
      hasher.update(manifest.magic.data(), manifest.magic.size()) &&
      hash_unsigned(hasher, manifest.abi_major) &&
      hash_unsigned(hasher, manifest.abi_minor) &&
      hash_unsigned(hasher, manifest.header_bytes) &&
      hash_unsigned(hasher, planned.device_arena_offset) &&
      hash_unsigned(hasher, manifest.artifact_identity) &&
      hash_unsigned(hasher, inventory.identity) &&
      hash_unsigned(hasher,
                    static_cast<std::uint16_t>(manifest.plan_identity)) &&
      hash_unsigned(hasher,
                    static_cast<std::uint16_t>(manifest.layout_identity)) &&
      hash_unsigned(hasher, static_cast<std::uint8_t>(manifest.encoding)) &&
      hash_unsigned(
          hasher,
          static_cast<std::uint16_t>(transform_receipt.transform_identity)) &&
      hash_unsigned(hasher, manifest.payload_offset) &&
      hash_unsigned(hasher, manifest.payload_bytes) &&
      hash_unsigned(hasher, manifest.artifact_bytes) &&
      hash_unsigned(hasher, manifest.payload_alignment) &&
      hasher.update(manifest.payload_digest.bytes.data(),
                    manifest.payload_digest.bytes.size()) &&
      hasher.update(verified_payload_digest.bytes.data(),
                    verified_payload_digest.bytes.size()) &&
      hash_unsigned(hasher, static_cast<std::uint8_t>(
                                manifest.token_count_independent)) &&
      hash_unsigned(hasher, static_cast<std::uint8_t>(
                                manifest.cuda_implementation_present)) &&
      hash_unsigned(hasher, static_cast<std::uint8_t>(
                                manifest.static_resources_qualified)) &&
      hash_unsigned(hasher, static_cast<std::uint8_t>(
                                manifest.numerical_contract_qualified)) &&
      hash_unsigned(hasher, static_cast<std::uint8_t>(
                                manifest.production_dispatch_eligible)) &&
      hash_unsigned(hasher, manifest.seal.value) &&
      hash_unsigned(hasher, inventory.source_count);
  for (std::size_t source_index = 0U;
       source_index < inventory.source_count; ++source_index) {
    const auto& source = inventory.sources[source_index];
    ok = ok &&
         hash_unsigned(hasher,
                       static_cast<std::uint8_t>(source.logical_role)) &&
         hash_unsigned(hasher, source.partition_index) &&
         hash_unsigned(hasher, source.tensor_identity) &&
         hasher.update(source.weight_digest.bytes.data(),
                       source.weight_digest.bytes.size()) &&
         hasher.update(source.scale_digest.bytes.data(),
                       source.scale_digest.bytes.size()) &&
         hash_unsigned(hasher, source.output_features) &&
         hash_unsigned(hasher, source.input_features) &&
         hash_unsigned(hasher, source.tensor_scale_bits) &&
         hash_unsigned(hasher, source.payload_offset) &&
         hash_unsigned(hasher, source.payload_bytes);
  }
  return ok;
}

struct VerifiedDeviceUploadContext final {
  std::uint8_t* arena = nullptr;
  std::uint64_t arena_bytes = 0U;
  std::uint64_t allocation_identity = 0U;
  std::uint64_t owner_identity = 0U;
  std::int32_t device_ordinal = -1;
  std::uint64_t stream_identity = 0U;
  std::uint64_t transaction_serial = 0U;
  const core::Sha256Digest* checkpoint_digest = nullptr;
};

[[nodiscard]] bool make_verified_device_descriptor(
    const VerifiedDeviceUploadContext& context,
    const PlannedArtifact& planned, const std::size_t artifact_index,
    const kernels::Sm87TargetAotProjectionPackedManifest& manifest,
    const kernels::Sm87TargetAotProjectionPackedSourceInventory& inventory,
    const kernels::Sm87TargetAotProjectionPackedTransformReceipt&
        transform_receipt,
    const kernels::Sm87TargetAotProjectionSha256Digest& readback_digest,
    Sm87TargetAotProjectionDeviceAssetDescriptor& descriptor) noexcept {
  const auto layout =
      kernels::sm87_target_aot_projection_packed_layout(planned.role);
  if (!layout.valid() || context.arena == nullptr ||
      context.arena_bytes != kSm87TargetAotProjectionDeviceArenaBytes ||
      context.allocation_identity == 0U || context.owner_identity == 0U ||
      context.device_ordinal < 0 || context.stream_identity == 0U ||
      context.transaction_serial == 0U ||
      context.checkpoint_digest == nullptr ||
      planned.device_arena_offset > context.arena_bytes ||
      layout.payload_bytes >
          context.arena_bytes - planned.device_arena_offset ||
      manifest.role != planned.role ||
      manifest.payload_digest != readback_digest ||
      !kernels::sm87_target_aot_projection_validate_transform_receipt(
          manifest, inventory, transform_receipt)) {
    return false;
  }
  std::uint8_t* const destination =
      context.arena + static_cast<std::size_t>(planned.device_arena_offset);
  const std::uint64_t upload_event_identity = make_runtime_identity(
      kEventIdentityDomain, *context.checkpoint_digest,
      context.transaction_serial, planned.artifact_identity, artifact_index,
      1U);
  const std::uint64_t verification_event_identity = make_runtime_identity(
      kEventIdentityDomain, *context.checkpoint_digest,
      context.transaction_serial, planned.artifact_identity, artifact_index,
      2U);

  kernels::Sm87TargetAotNvFp4CudaDeviceUploadReceipt upload;
  upload.artifact_identity = manifest.artifact_identity;
  upload.source_inventory_identity = inventory.identity;
  upload.role = manifest.role;
  upload.plan_identity = manifest.plan_identity;
  upload.layout_identity = manifest.layout_identity;
  upload.transform_identity = transform_receipt.transform_identity;
  upload.host_payload_offset = manifest.payload_offset;
  upload.host_payload_bytes = manifest.payload_bytes;
  upload.host_payload_digest = manifest.payload_digest;
  upload.host_manifest_seal = manifest.seal;
  upload.tensor_scale_count = manifest.source_count;
  for (std::size_t source_index = 0U;
       source_index < manifest.source_count; ++source_index) {
    upload.tensor_scale_bits[source_index] =
        manifest.sources[source_index].tensor_scale_bits;
  }
  upload.device_allocation_identity = context.allocation_identity;
  upload.device_allocation_owner_identity = context.owner_identity;
  upload.device_ordinal = context.device_ordinal;
  upload.device_allocation_begin =
      reinterpret_cast<std::uintptr_t>(context.arena);
  upload.device_allocation_bytes = context.arena_bytes;
  upload.device_allocation_end =
      upload.device_allocation_begin +
      static_cast<std::uintptr_t>(context.arena_bytes);
  upload.device_payload_begin = reinterpret_cast<std::uintptr_t>(destination);
  upload.device_payload_bytes = layout.payload_bytes;
  upload.device_payload_end =
      upload.device_payload_begin +
      static_cast<std::uintptr_t>(layout.payload_bytes);
  upload.upload_stream_owner_identity = context.owner_identity;
  upload.upload_stream_identity = context.stream_identity;
  upload.upload_completion_event_identity = upload_event_identity;
  upload.verification_stream_owner_identity = context.owner_identity;
  upload.verification_stream_identity = context.stream_identity;
  upload.verification_completion_event_identity =
      verification_event_identity;
  upload.verification_readback_bytes = layout.payload_bytes;
  upload.verification_readback_digest = readback_digest;
  upload.host_payload_digest_verified_before_copy = true;
  upload.host_payload_immutable_until_completion = true;
  upload.copy_enqueued_to_exact_payload_range = true;
  upload.completion_event_recorded_after_copy = true;
  upload.completion_event_observed = true;
  upload.upload_completed = true;
  upload.verification_copy_enqueued_from_exact_payload_range = true;
  upload.verification_event_recorded_after_copy = true;
  upload.verification_event_observed = true;
  upload.verification_completed = true;
  upload.device_payload_matches_host_payload = true;
  upload.allocation_retained_for_asset_lifetime = true;
  upload.receipt_identity =
      kernels::sm87_target_aot_nvfp4_cuda_compute_upload_receipt_identity(
          upload);
  const auto view = kernels::sm87_target_aot_bind_nvfp4_cuda_asset(
      manifest, inventory, transform_receipt, upload);
  if (upload.receipt_identity == 0U || !view.valid ||
      !kernels::sm87_target_aot_nvfp4_cuda_asset_valid(view)) {
    return false;
  }

  descriptor = {};
  descriptor.layer_index = planned.layer_index;
  descriptor.role = planned.role;
  descriptor.device_arena_offset = planned.device_arena_offset;
  descriptor.source_inventory = inventory;
  descriptor.manifest = manifest;
  descriptor.transform_receipt = transform_receipt;
  descriptor.upload_receipt = upload;
  descriptor.view = view;
  return true;
}

[[nodiscard]] bool plan_source(
    const ResidentWeights& resident, const NvFp4LinearWeight& projection,
    const std::string& module_name,
    const kernels::Sm87TargetAotLogicalRole logical_role,
    const std::uint64_t expected_output,
    const std::uint64_t expected_input,
    const core::Sha256Digest& checkpoint_digest, PlannedSource& source) {
  if (projection.packed_weight == nullptr ||
      projection.block_scale == nullptr ||
      projection.weight_scale_2_device == nullptr ||
      projection.output_size != expected_output ||
      projection.input_size != expected_input ||
      !std::isfinite(projection.weight_scale_2) ||
      projection.weight_scale_2 <= 0.0F ||
      reinterpret_cast<std::uintptr_t>(projection.packed_weight) % 16U !=
          0U ||
      reinterpret_cast<std::uintptr_t>(projection.block_scale) % 16U != 0U ||
      reinterpret_cast<std::uintptr_t>(projection.weight_scale_2_device) %
              alignof(float) !=
          0U ||
      projection.prefill_marlin_gate_up_layout !=
          NvFp4MarlinGateUpLayout::kUnbound ||
      projection.prefill_marlin_weight != nullptr ||
      projection.prefill_marlin_scales != nullptr ||
      projection.prefill_marlin_global_scale != nullptr ||
      !empty_p40_packed_artifact_view(
          projection.prefill_p40_packed_artifact) ||
      !empty_nvfp4_marlin_p40_parity_view(
          projection.prefill_p40_vllm_marlin_parity)) {
    return false;
  }
  const std::uint64_t values = expected_output * expected_input;
  const DeviceTensorView* const weight = exact_resident_tensor(
      resident, module_name + ".weight", projection.packed_weight,
      st::DType::kU8, {expected_output, expected_input / 2U}, values / 2U);
  const DeviceTensorView* const block_scale = exact_resident_tensor(
      resident, module_name + ".weight_scale", projection.block_scale,
      st::DType::kF8E4M3, {expected_output, expected_input / 16U},
      values / 16U);
  const DeviceTensorView* const tensor_scale = exact_resident_tensor(
      resident, module_name + ".weight_scale_2",
      projection.weight_scale_2_device, st::DType::kF32, {}, sizeof(float));
  const std::uint64_t tensor_identity =
      make_source_identity(checkpoint_digest, module_name, logical_role);
  if (weight == nullptr || block_scale == nullptr || tensor_scale == nullptr ||
      tensor_identity == 0U) {
    return false;
  }
  source.logical_role = logical_role;
  source.module_name = module_name;
  source.projection = &projection;
  source.weight = weight;
  source.block_scale = block_scale;
  source.tensor_scale = tensor_scale;
  source.tensor_identity = tensor_identity;
  return true;
}

[[nodiscard]] bool plan_inventory(
    const ResidentWeights& resident, const ModelWeights& model_weights,
    const core::Sha256Digest& checkpoint_digest, ArtifactPlans& plans,
    std::string& message) {
  std::array<std::uint64_t,
             kSm87TargetAotProjectionDeviceSourceCount>
      source_identities{};
  std::array<std::uint64_t,
             kSm87TargetAotProjectionDeviceArtifactCount>
      inventory_identities{};
  std::array<std::uint64_t,
             kSm87TargetAotProjectionDeviceArtifactCount>
      artifact_identities{};
  std::size_t source_identity_count = 0U;
  std::size_t plan_count = 0U;
  std::uint64_t arena_offset = 0U;

  for (std::size_t layer_index = 0U;
       layer_index < kSm87TargetAotProjectionDeviceLayerCount;
       ++layer_index) {
    const DecoderLayerWeights& layer = model_weights.layer(layer_index);
    const auto* const gate =
        std::get_if<NvFp4LinearWeight>(&layer.mlp.gate_proj);
    const auto* const up =
        std::get_if<NvFp4LinearWeight>(&layer.mlp.up_proj);
    const auto* const down =
        std::get_if<NvFp4LinearWeight>(&layer.mlp.down_proj);
    if (gate == nullptr || up == nullptr || down == nullptr) {
      message = "non-NVFP4 MLP projection at layer " +
                std::to_string(layer_index);
      return false;
    }
    const std::string prefix = "model.language_model.layers." +
                               std::to_string(layer_index) + ".mlp.";
    PlannedArtifact gate_up;
    gate_up.layer_index = layer_index;
    gate_up.role = kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp;
    gate_up.source_count = 2U;
    PlannedArtifact down_artifact;
    down_artifact.layer_index = layer_index;
    down_artifact.role = kernels::Sm87TargetAotProjectionRole::kNvFp4Down;
    down_artifact.source_count = 1U;
    if (!plan_source(resident, *gate, prefix + "gate_proj",
                     kernels::Sm87TargetAotLogicalRole::kNvFp4Gate, 17'408U,
                     5'120U, checkpoint_digest, gate_up.sources[0U]) ||
        !plan_source(resident, *up, prefix + "up_proj",
                     kernels::Sm87TargetAotLogicalRole::kNvFp4Up, 17'408U,
                     5'120U, checkpoint_digest, gate_up.sources[1U]) ||
        !plan_source(resident, *down, prefix + "down_proj",
                     kernels::Sm87TargetAotLogicalRole::kNvFp4Down, 5'120U,
                     17'408U, checkpoint_digest,
                     down_artifact.sources[0U])) {
      message = "ineligible or non-resident target-AOT NVFP4 inventory at "
                "layer " +
                std::to_string(layer_index);
      return false;
    }

    for (PlannedArtifact* const artifact :
         std::array<PlannedArtifact*, 2U>{{&gate_up, &down_artifact}}) {
      const auto layout =
          kernels::sm87_target_aot_projection_packed_layout(artifact->role);
      if (!layout.valid() || layout.partition_count != artifact->source_count ||
          arena_offset % layout.payload_alignment != 0U ||
          layout.payload_bytes >
              kSm87TargetAotProjectionDeviceArenaBytes - arena_offset ||
          plan_count >= plans.size()) {
        message = "target-AOT layout does not fit the fixed device arena";
        return false;
      }
      artifact->device_arena_offset = arena_offset;
      for (std::size_t source_index = 0U;
           source_index < artifact->source_count; ++source_index) {
        const std::uint64_t identity =
            artifact->sources[source_index].tensor_identity;
        if (source_identity_count >= source_identities.size() ||
            identity == 0U ||
            std::find(source_identities.begin(),
                      source_identities.begin() + source_identity_count,
                      identity) !=
                source_identities.begin() + source_identity_count) {
          message = "target-AOT source identities are not globally unique";
          return false;
        }
        source_identities[source_identity_count++] = identity;
      }
      artifact->inventory_identity =
          make_inventory_identity(checkpoint_digest, *artifact);
      artifact->artifact_identity =
          make_artifact_identity(checkpoint_digest, *artifact);
      if (artifact->inventory_identity == 0U ||
          artifact->artifact_identity == 0U ||
          std::find(inventory_identities.begin(),
                    inventory_identities.begin() + plan_count,
                    artifact->inventory_identity) !=
              inventory_identities.begin() + plan_count ||
          std::find(artifact_identities.begin(),
                    artifact_identities.begin() + plan_count,
                    artifact->artifact_identity) !=
              artifact_identities.begin() + plan_count) {
        message = "target-AOT artifact identities are zero or duplicated";
        return false;
      }
      inventory_identities[plan_count] = artifact->inventory_identity;
      artifact_identities[plan_count] = artifact->artifact_identity;
      plans[plan_count++] = std::move(*artifact);
      arena_offset += layout.payload_bytes;
    }
  }
  if (plan_count != plans.size() ||
      source_identity_count != source_identities.size() ||
      arena_offset != kSm87TargetAotProjectionDeviceArenaBytes) {
    message = "target-AOT inventory did not close 128 artifacts, 192 "
              "sources, and the exact arena";
    return false;
  }
  return true;
}

[[nodiscard]] bool persisted_record_matches_plan(
    const persistence::PersistenceRecord& record,
    const PlannedArtifact& planned, const std::size_t ordinal,
    std::string& message) {
  const auto layout =
      kernels::sm87_target_aot_projection_packed_layout(planned.role);
  const std::uint64_t expected_header_offset =
      persistence::persistence_expected_bundle_header_offset(
          planned.layer_index, planned.role);
  if (!layout.valid() || ordinal >= persistence::kPersistedArtifactCount ||
      record.ordinal != ordinal || record.layer_index != planned.layer_index ||
      record.role != planned.role ||
      record.arena_offset != planned.device_arena_offset ||
      record.bundle_header_offset != expected_header_offset ||
      record.bundle_payload_offset !=
          expected_header_offset + persistence::kRecordHeaderBytes ||
      record.bundle_record_bytes !=
          persistence::kRecordHeaderBytes + layout.payload_bytes ||
      record.manifest.role != planned.role ||
      record.manifest.artifact_identity != planned.artifact_identity ||
      record.manifest.source_inventory_identity !=
          planned.inventory_identity ||
      record.source_inventory.identity != planned.inventory_identity ||
      record.source_inventory.role != planned.role ||
      record.manifest.source_count != planned.source_count ||
      record.source_inventory.source_count != planned.source_count ||
      record.manifest.plan_identity != layout.plan_identity ||
      record.manifest.layout_identity != layout.layout_identity ||
      record.manifest.encoding != layout.encoding ||
      record.manifest.payload_offset !=
          kernels::kSm87TargetAotProjectionPackedHeaderBytes ||
      record.manifest.payload_bytes != layout.payload_bytes ||
      record.manifest.artifact_bytes !=
          kernels::kSm87TargetAotProjectionPackedHeaderBytes +
              layout.payload_bytes ||
      record.manifest.payload_alignment != layout.payload_alignment ||
      !kernels::sm87_target_aot_projection_validate_packed_manifest(
          record.manifest, record.source_inventory) ||
      !kernels::sm87_target_aot_projection_validate_transform_receipt(
          record.manifest, record.source_inventory,
          record.transform_receipt)) {
    message = "target-AOT persisted record does not match the live plan at "
              "ordinal " +
              std::to_string(ordinal);
    return false;
  }
  if (record.bundle_payload_offset > persistence::kPersistedBundleBytes ||
      layout.payload_bytes >
          persistence::kPersistedBundleBytes -
              record.bundle_payload_offset ||
      record.arena_offset > persistence::kPersistedArenaBytes ||
      layout.payload_bytes >
          persistence::kPersistedArenaBytes - record.arena_offset) {
    message = "target-AOT persisted record escapes the fixed bundle or arena";
    return false;
  }

  for (std::size_t source_index = 0U;
       source_index < planned.source_count; ++source_index) {
    const PlannedSource& expected = planned.sources[source_index];
    const auto& source = record.source_inventory.sources[source_index];
    const auto& partition = layout.partitions[source_index];
    if (expected.projection == nullptr ||
        source.logical_role != expected.logical_role ||
        source.partition_index != source_index ||
        source.tensor_identity != expected.tensor_identity ||
        source.output_features != partition.output_features ||
        source.input_features != partition.input_features ||
        source.tensor_scale_bits !=
            float_bits(expected.projection->weight_scale_2) ||
        !kernels::sm87_target_aot_projection_scale_bits_valid(
            source.tensor_scale_bits) ||
        source.payload_offset != partition.payload_offset ||
        source.payload_bytes != partition.payload_bytes) {
      message = "target-AOT persisted source does not match the live tensor "
                "plan at ordinal " +
                std::to_string(ordinal) + ", source " +
                std::to_string(source_index);
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool scan_persisted_nvfp4_payload_scales(
    const kernels::Sm87TargetAotProjectionPackedLayout& layout,
    const std::uint8_t* const payload, const std::size_t payload_bytes,
    std::string& message) noexcept {
  if (!layout.valid() || payload == nullptr ||
      payload_bytes != layout.payload_bytes ||
      (layout.role !=
           kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp &&
       layout.role != kernels::Sm87TargetAotProjectionRole::kNvFp4Down)) {
    message = "target-AOT persisted NVFP4 payload range is invalid";
    return false;
  }
  for (std::size_t partition_index = 0U;
       partition_index < layout.partition_count; ++partition_index) {
    const auto& partition = layout.partitions[partition_index];
    if (partition.block_scale_group_k == 0U) {
      message = "target-AOT persisted NVFP4 partition has no block scales";
      return false;
    }
    for (std::size_t n_tile = 0U; n_tile < partition.n_tiles; ++n_tile) {
      for (std::size_t k_tile = 0U; k_tile < partition.k_tiles; ++k_tile) {
        const auto cell = kernels::sm87_target_aot_projection_packed_cell(
            layout, partition_index, n_tile, k_tile);
        if (!cell.valid || cell.block_scale_offset > payload_bytes ||
            cell.block_scale_bytes >
                payload_bytes - cell.block_scale_offset) {
          message = "target-AOT persisted block-scale range escaped payload";
          return false;
        }
        for (std::uint64_t byte = 0U; byte < cell.block_scale_bytes; ++byte) {
          if (kernels::
                  sm87_target_aot_projection_block_scale_e4m3fn_code_is_forbidden(
                      payload[static_cast<std::size_t>(
                          cell.block_scale_offset + byte)])) {
            message = "target-AOT persisted payload contains a forbidden "
                      "NVFP4 block-scale encoding";
            return false;
          }
        }
      }
    }
  }
  return true;
}

void cleanup_stream(cudaStream_t& stream, cudaEvent_t& upload_event,
                    cudaEvent_t& verification_event) noexcept;

class PendingDeviceAssetLoad final {
 public:
  PendingDeviceAssetLoad() noexcept = default;
  ~PendingDeviceAssetLoad() { reset(); }

  PendingDeviceAssetLoad(const PendingDeviceAssetLoad&) = delete;
  PendingDeviceAssetLoad& operator=(const PendingDeviceAssetLoad&) = delete;

  void reset() noexcept {
    cleanup_stream(stream, upload_event, verification_event);
    if (arena != nullptr) {
      (void)cudaFree(arena);
    }
    arena = nullptr;
    bytes = 0U;
    descriptors = {};
    descriptor_count = 0U;
  }

  std::uint8_t* arena = nullptr;
  std::uint64_t bytes = 0U;
  cudaStream_t stream = nullptr;
  cudaEvent_t upload_event = nullptr;
  cudaEvent_t verification_event = nullptr;
  std::array<Sm87TargetAotProjectionDeviceAssetDescriptor,
             kSm87TargetAotProjectionDeviceArtifactCount>
      descriptors{};
  std::size_t descriptor_count = 0U;
};

[[nodiscard]] bool copy_and_bind_sources(
    const PlannedArtifact& planned, cudaStream_t stream,
    std::vector<std::uint8_t>& canonical,
    Sm87TargetAotProjectionSourceSet& sources,
    std::uint64_t& copied_bytes, int& cuda_error,
    std::string& message) {
  const auto layout =
      kernels::sm87_target_aot_projection_packed_layout(planned.role);
  if (!layout.valid() || planned.source_count != layout.partition_count ||
      canonical.size() < layout.payload_bytes) {
    message = "invalid bounded source staging contract";
    return false;
  }
  sources = {};
  sources.role = planned.role;
  sources.inventory_identity = planned.inventory_identity;
  sources.source_count = planned.source_count;
  std::array<std::uint32_t,
             kernels::kSm87TargetAotProjectionPackedMaxPartitions>
      observed_tensor_scale_bits{};
  std::uint64_t cursor = 0U;
  for (std::size_t index = 0U; index < planned.source_count; ++index) {
    const PlannedSource& planned_source = planned.sources[index];
    const auto& partition = layout.partitions[index];
    const std::uint64_t values =
        static_cast<std::uint64_t>(partition.output_features) *
        partition.input_features;
    const std::uint64_t weight_bytes = values / 2U;
    const std::uint64_t scale_bytes = values / 16U;
    if (planned_source.weight == nullptr ||
        planned_source.block_scale == nullptr ||
        planned_source.tensor_scale == nullptr ||
        planned_source.projection == nullptr ||
        weight_bytes > layout.payload_bytes - cursor) {
      message = "planned target-AOT source is incomplete";
      return false;
    }
    std::uint8_t* const weight_host =
        canonical.data() + static_cast<std::size_t>(cursor);
    cudaError_t status = cudaMemcpyAsync(
        weight_host, planned_source.weight->device_data,
        static_cast<std::size_t>(weight_bytes), cudaMemcpyDeviceToHost,
        stream);
    if (status != cudaSuccess) {
      cuda_error = static_cast<int>(status);
      message = "D2H failed for " + planned_source.module_name + ".weight";
      return false;
    }
    cursor += weight_bytes;
    if (scale_bytes > layout.payload_bytes - cursor) {
      message = "planned target-AOT scale staging overflow";
      return false;
    }
    std::uint8_t* const scale_host =
        canonical.data() + static_cast<std::size_t>(cursor);
    status = cudaMemcpyAsync(
        scale_host, planned_source.block_scale->device_data,
        static_cast<std::size_t>(scale_bytes), cudaMemcpyDeviceToHost,
        stream);
    if (status != cudaSuccess) {
      cuda_error = static_cast<int>(status);
      message = "D2H failed for " + planned_source.module_name +
                ".weight_scale";
      return false;
    }
    cursor += scale_bytes;
    status = cudaMemcpyAsync(&observed_tensor_scale_bits[index],
                             planned_source.tensor_scale->device_data,
                             sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                             stream);
    if (status != cudaSuccess) {
      cuda_error = static_cast<int>(status);
      message = "D2H failed for scalar scales at " +
                planned_source.module_name;
      return false;
    }
    auto& source = sources.sources[index];
    source.logical_role = planned_source.logical_role;
    source.tensor_identity = planned_source.tensor_identity;
    source.output_features = partition.output_features;
    source.input_features = partition.input_features;
    source.packed_weight = {weight_host,
                            static_cast<std::size_t>(weight_bytes)};
    source.block_scale = {scale_host,
                          static_cast<std::size_t>(scale_bytes)};
  }
  cudaError_t status = cudaStreamSynchronize(stream);
  if (status != cudaSuccess) {
    cuda_error = static_cast<int>(status);
    message = "source D2H stream synchronization failed";
    return false;
  }
  for (std::size_t index = 0U; index < planned.source_count; ++index) {
    const NvFp4LinearWeight& projection =
        *planned.sources[index].projection;
    const std::uint32_t expected_tensor = float_bits(projection.weight_scale_2);
    if (observed_tensor_scale_bits[index] != expected_tensor ||
        !kernels::sm87_target_aot_projection_scale_bits_valid(
            observed_tensor_scale_bits[index])) {
      message = "device scalar scale bits differ from ModelWeights at " +
                planned.sources[index].module_name;
      return false;
    }
    sources.sources[index].tensor_scale_bits =
        observed_tensor_scale_bits[index];
  }
  if (cursor != layout.payload_bytes) {
    message = "canonical source bytes did not equal the artifact payload";
    return false;
  }
  copied_bytes +=
      cursor + sizeof(std::uint32_t) * planned.source_count;
  return true;
}

void cleanup_stream(cudaStream_t& stream, cudaEvent_t& upload_event,
                    cudaEvent_t& verification_event) noexcept {
  if (stream != nullptr) {
    (void)cudaStreamSynchronize(stream);
  }
  if (upload_event != nullptr) {
    (void)cudaEventDestroy(upload_event);
    upload_event = nullptr;
  }
  if (verification_event != nullptr) {
    (void)cudaEventDestroy(verification_event);
    verification_event = nullptr;
  }
  if (stream != nullptr) {
    (void)cudaStreamDestroy(stream);
    stream = nullptr;
  }
}

[[nodiscard]] Sm87TargetAotProjectionDevicePreparationStats
load_persisted_impl(
    const ResidentWeights& resident, const ModelWeights& model_weights,
    const std::filesystem::path& bundle_path,
    const std::string_view expected_verified_payload_catalog_sha256,
    const std::uint64_t minimum_free_bytes_after_load,
    Sm87TargetAotProjectionDeviceAssets& owner,
    std::uint8_t*& owner_arena, std::uint64_t& owner_bytes,
    std::uint64_t& allocation_identity, std::uint64_t& owner_identity,
    std::int32_t& owner_device_ordinal,
    std::array<Sm87TargetAotProjectionDeviceAssetDescriptor,
               kSm87TargetAotProjectionDeviceArtifactCount>& descriptors,
    std::size_t& descriptor_count) {
  Sm87TargetAotProjectionDevicePreparationStats result;
  result.source =
      Sm87TargetAotProjectionDeviceAssetSource::kPersistedBundle;
  if (!owner.empty()) {
    result.hard_failure = true;
    result.message = "target-AOT persisted device asset owner was not empty";
    return result;
  }

  kernels::Sm87TargetAotProjectionSha256Digest expected_catalog{};
  if (!parse_lower_hex_digest(
          expected_verified_payload_catalog_sha256, expected_catalog)) {
    result.hard_failure = true;
    result.message = "target-AOT persisted load requires one nonzero "
                     "lowercase external catalog SHA-256";
    return result;
  }
  core::Sha256Digest checkpoint_digest{};
  if (!verify_observed_checkpoint(resident, checkpoint_digest,
                                  result.message)) {
    result.hard_failure = true;
    return result;
  }
  const auto persistent_checkpoint_digest = copy_digest(checkpoint_digest);
  ArtifactPlans plans{};
  if (!plan_inventory(resident, model_weights, checkpoint_digest, plans,
                      result.message)) {
    result.hard_failure = true;
    return result;
  }

  persistence::UniqueFd bundle;
  persistence::FileSnapshot initial_snapshot;
  if (!persistence::open_regular_file_exact(
          bundle_path, persistence::kPersistedBundleBytes, bundle,
          initial_snapshot, result.message)) {
    result.hard_failure = true;
    return result;
  }
  persistence::SuperblockBytes superblock_bytes{};
  if (!persistence::pread_exact_counted(
          bundle.get(), superblock_bytes.data(), superblock_bytes.size(), 0U,
          result.persistent_bundle_file_bytes_read, result.message)) {
    result.hard_failure = true;
    return result;
  }
  persistence::PersistenceSuperblock superblock;
  if (!persistence::decode_superblock(superblock_bytes.data(),
                                      superblock_bytes.size(), &superblock)) {
    result.hard_failure = true;
    result.message = "target-AOT persisted bundle superblock is invalid";
    return result;
  }
  if (superblock.checkpoint_identity_sha256 !=
          persistent_checkpoint_digest ||
      superblock.authenticated_payload_catalog_sha256 != expected_catalog) {
    result.hard_failure = true;
    result.message = "target-AOT persisted bundle does not match the live "
                     "checkpoint and external catalog trust root";
    return result;
  }

  std::array<persistence::PersistenceRecord,
             persistence::kPersistedArtifactCount>
      records{};
  std::array<kernels::Sm87TargetAotProjectionSha256Digest,
             persistence::kPersistedArtifactCount>
      authenticated_header_digests{};
  persistence::RecordHeaderBytes header{};
  core::Sha256 record_header_catalog;
  bool record_header_catalog_ok =
      initialize_record_header_catalog(record_header_catalog);
  for (std::size_t index = 0U; index < records.size(); ++index) {
    const PlannedArtifact& planned = plans[index];
    const std::uint64_t header_offset =
        persistence::persistence_expected_bundle_header_offset(
            planned.layer_index, planned.role);
    if (!persistence::pread_exact_counted(
            bundle.get(), header.data(), header.size(), header_offset,
            result.persistent_bundle_file_bytes_read, result.message) ||
        !sm87_target_aot_projection_sha256(
            {header.data(), header.size()},
            &authenticated_header_digests[index]) ||
        !persistence::decode_record_header(
            header.data(), header.size(), &records[index]) ||
        !persisted_record_matches_plan(records[index], planned, index,
                                       result.message)) {
      result.hard_failure = true;
      if (result.message.empty()) {
        result.message = "target-AOT persisted record header is invalid at "
                         "ordinal " +
                         std::to_string(index);
      }
      return result;
    }
    record_header_catalog_ok =
        record_header_catalog_ok &&
        append_record_header_catalog_entry(record_header_catalog, index,
                                           header.data(), header.size());
  }
  const core::Sha256Digest observed_record_header_catalog_digest =
      record_header_catalog.finalize();
  const auto observed_record_header_catalog =
      copy_digest(observed_record_header_catalog_digest);
  if (!record_header_catalog_ok ||
      observed_record_header_catalog !=
          superblock.record_header_catalog_sha256) {
    result.hard_failure = true;
    result.message =
        "target-AOT persisted record-header catalog SHA-256 mismatch";
    return result;
  }
  result.persistent_record_header_catalog_sha256 =
      observed_record_header_catalog_digest.hex();
  if (result.persistent_record_header_catalog_sha256.size() != 64U) {
    result.hard_failure = true;
    result.message =
        "target-AOT persisted record-header catalog encoding failed";
    return result;
  }

  // Authenticate every persisted payload and the complete externally pinned
  // catalog before issuing even a device query. This first bounded-staging
  // pass makes a damaged late record a purely host-side failure. The upload
  // pass deliberately reads and authenticates the same bytes again to close
  // the descriptor-level TOCTOU window without retaining the 9.6 GB arena in
  // host memory.
  std::vector<std::uint8_t> staging(
      static_cast<std::size_t>(
          kSm87TargetAotProjectionMaximumArtifactPayloadBytes));
  result.host_staging_peak_bytes =
      kSm87TargetAotProjectionMaximumArtifactPayloadBytes;
  core::Sha256 host_payload_catalog_hasher;
  bool host_payload_catalog_hash_ok = hash_string(
      host_payload_catalog_hasher, kVerifiedPayloadCatalogDomain);
  std::uint64_t host_payload_catalog_bytes = 0U;
  for (std::size_t index = 0U; index < plans.size(); ++index) {
    const PlannedArtifact& planned = plans[index];
    const persistence::PersistenceRecord& record = records[index];
    const auto layout =
        kernels::sm87_target_aot_projection_packed_layout(planned.role);
    if (!layout.valid() || layout.payload_bytes > staging.size() ||
        planned.device_arena_offset != host_payload_catalog_bytes ||
        planned.device_arena_offset >
            kSm87TargetAotProjectionDeviceArenaBytes ||
        layout.payload_bytes >
            kSm87TargetAotProjectionDeviceArenaBytes -
                planned.device_arena_offset) {
      result.hard_failure = true;
      result.message =
          "persisted target-AOT payload escaped bounded host authentication";
      return result;
    }
    const std::size_t payload_bytes =
        static_cast<std::size_t>(layout.payload_bytes);
    if (!persistence::pread_exact_counted(
            bundle.get(), staging.data(), payload_bytes,
            record.bundle_payload_offset,
            result.persistent_bundle_file_bytes_read, result.message)) {
      result.hard_failure = true;
      return result;
    }
    kernels::Sm87TargetAotProjectionSha256Digest host_digest{};
    if (!sm87_target_aot_projection_sha256(
            {staging.data(), payload_bytes}, &host_digest) ||
        host_digest != record.manifest.payload_digest ||
        host_digest !=
            record.transform_receipt.payload.observed_payload_digest ||
        !scan_persisted_nvfp4_payload_scales(
            layout, staging.data(), payload_bytes, result.message)) {
      result.hard_failure = true;
      if (result.message.empty()) {
        result.message =
            "persisted target-AOT pre-CUDA payload authentication failed";
      }
      return result;
    }
    host_payload_catalog_hash_ok =
        host_payload_catalog_hash_ok &&
        append_verified_payload_catalog_entry(
            host_payload_catalog_hasher, planned, record.manifest,
            record.source_inventory, record.transform_receipt, host_digest);
    host_payload_catalog_bytes += layout.payload_bytes;
  }
  persistence::FileSnapshot authenticated_snapshot;
  const std::string host_payload_catalog_sha256 =
      host_payload_catalog_hash_ok
          ? host_payload_catalog_hasher.finalize().hex()
          : std::string{};
  if (!host_payload_catalog_hash_ok ||
      host_payload_catalog_bytes !=
          kSm87TargetAotProjectionDeviceArenaBytes ||
      host_payload_catalog_sha256 !=
          expected_verified_payload_catalog_sha256 ||
      !persistence::snapshot_regular_file_exact(
          bundle.get(), persistence::kPersistedBundleBytes,
          authenticated_snapshot, result.message) ||
      !persistence::same_snapshot(initial_snapshot, authenticated_snapshot) ||
      result.persistent_bundle_file_bytes_read !=
          persistence::kPersistedBundleBytes) {
    result.hard_failure = true;
    if (result.message.empty()) {
      result.message =
          "target-AOT persisted bundle failed complete pre-CUDA authentication";
    }
    return result;
  }
  result.persistent_bundle_host_authentication_passes = 1U;

  int device_ordinal = -1;
  cudaError_t status = cudaGetDevice(&device_ordinal);
  if (status != cudaSuccess || device_ordinal < 0) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message =
        "cudaGetDevice failed before target-AOT persisted loading";
    return result;
  }
  cudaPointerAttributes resident_attributes{};
  status =
      cudaPointerGetAttributes(&resident_attributes, resident.arena_data());
  if (status != cudaSuccess ||
      resident_attributes.type != cudaMemoryTypeDevice ||
      resident_attributes.device != device_ordinal) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message = "resident checkpoint arena is not device memory on the "
                     "current CUDA ordinal";
    return result;
  }

  std::size_t free_bytes = 0U;
  std::size_t total_bytes = 0U;
  status = cudaMemGetInfo(&free_bytes, &total_bytes);
  (void)total_bytes;
  const std::uint64_t required_before =
      kSm87TargetAotProjectionDeviceArenaBytes +
      kSm87TargetAotProjectionMaximumArtifactPayloadBytes;
  if (status != cudaSuccess || required_before > free_bytes ||
      minimum_free_bytes_after_load > free_bytes - required_before) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message =
        status == cudaSuccess
            ? "insufficient memory for the target-AOT persisted arena, "
              "bounded staging, and retained margin"
            : "cudaMemGetInfo failed before target-AOT persisted staging";
    return result;
  }

  status = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (status != cudaSuccess ||
      kSm87TargetAotProjectionDeviceArenaBytes > free_bytes ||
      minimum_free_bytes_after_load >
          free_bytes - kSm87TargetAotProjectionDeviceArenaBytes) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message =
        status == cudaSuccess
            ? "bounded host staging consumed the target-AOT persisted "
              "post-allocation margin"
            : "cudaMemGetInfo failed after target-AOT persisted staging";
    return result;
  }

  PendingDeviceAssetLoad pending;
  void* allocation = nullptr;
  status = cudaMalloc(
      &allocation,
      static_cast<std::size_t>(kSm87TargetAotProjectionDeviceArenaBytes));
  if (status != cudaSuccess || allocation == nullptr) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message =
        "cudaMalloc failed for the exact persisted target-AOT arena";
    return result;
  }
  pending.arena = static_cast<std::uint8_t*>(allocation);
  pending.bytes = kSm87TargetAotProjectionDeviceArenaBytes;
  cudaPointerAttributes allocation_attributes{};
  status = cudaPointerGetAttributes(&allocation_attributes, pending.arena);
  if (status != cudaSuccess ||
      allocation_attributes.type != cudaMemoryTypeDevice ||
      allocation_attributes.device != device_ordinal) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message = "persisted target-AOT allocation is not device memory "
                     "on the current CUDA ordinal";
    return result;
  }

  const std::uint64_t transaction_serial =
      g_device_asset_transaction_serial.fetch_add(1U,
                                                   std::memory_order_relaxed);
  const std::uint64_t pending_owner_identity = make_runtime_identity(
      kOwnerIdentityDomain, checkpoint_digest, transaction_serial,
      static_cast<std::uint64_t>(device_ordinal), pending.bytes);
  const std::uint64_t pending_allocation_identity = make_runtime_identity(
      kAllocationIdentityDomain, checkpoint_digest, transaction_serial,
      pending_owner_identity, pending.bytes,
      static_cast<std::uint64_t>(device_ordinal));
  if (reinterpret_cast<std::uintptr_t>(pending.arena) %
              kernels::kSm87TargetAotProjectionPackedAlignment !=
          0U ||
      pending_owner_identity == 0U || pending_allocation_identity == 0U) {
    result.hard_failure = true;
    result.message =
        "persisted target-AOT allocation alignment or identity failed";
    return result;
  }
  status = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (status != cudaSuccess ||
      static_cast<std::uint64_t>(free_bytes) <
          minimum_free_bytes_after_load) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message =
        status == cudaSuccess
            ? "persisted target-AOT allocation violated retained margin"
            : "cudaMemGetInfo failed after persisted target-AOT allocation";
    return result;
  }

  status = cudaStreamCreateWithFlags(&pending.stream, cudaStreamNonBlocking);
  if (status != cudaSuccess) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message =
        "failed to create persisted target-AOT loader stream";
    return result;
  }
  const std::uint64_t stream_identity = make_runtime_identity(
      kStreamIdentityDomain, checkpoint_digest, transaction_serial,
      pending_owner_identity, 1U);
  if (stream_identity == 0U) {
    result.hard_failure = true;
    result.message =
        "failed to derive persisted target-AOT stream identity";
    return result;
  }
  const VerifiedDeviceUploadContext upload_context{
      pending.arena,
      pending.bytes,
      pending_allocation_identity,
      pending_owner_identity,
      device_ordinal,
      stream_identity,
      transaction_serial,
      &checkpoint_digest};
  persistence::SuperblockBytes second_superblock_bytes{};
  if (!persistence::pread_exact_counted(
          bundle.get(), second_superblock_bytes.data(),
          second_superblock_bytes.size(), 0U,
          result.persistent_bundle_file_bytes_read, result.message) ||
      second_superblock_bytes != superblock_bytes) {
    result.hard_failure = true;
    if (result.message.empty()) {
      result.message =
          "persisted target-AOT superblock changed before device upload";
    }
    return result;
  }
  core::Sha256 second_record_header_catalog;
  bool second_record_header_catalog_ok =
      initialize_record_header_catalog(second_record_header_catalog);
  core::Sha256 second_host_payload_catalog_hasher;
  bool second_host_payload_catalog_hash_ok = hash_string(
      second_host_payload_catalog_hasher, kVerifiedPayloadCatalogDomain);
  std::uint64_t second_host_payload_catalog_bytes = 0U;
  core::Sha256 verified_payload_catalog_hasher;
  bool verified_payload_catalog_hash_ok = hash_string(
      verified_payload_catalog_hasher, kVerifiedPayloadCatalogDomain);
  std::uint64_t verified_payload_catalog_bytes = 0U;

  for (std::size_t index = 0U; index < plans.size(); ++index) {
    const PlannedArtifact& planned = plans[index];
    const persistence::PersistenceRecord& record = records[index];
    const std::uint64_t header_offset =
        persistence::persistence_expected_bundle_header_offset(
            planned.layer_index, planned.role);
    kernels::Sm87TargetAotProjectionSha256Digest second_header_digest{};
    persistence::PersistenceRecord second_record;
    if (!persistence::pread_exact_counted(
            bundle.get(), header.data(), header.size(), header_offset,
            result.persistent_bundle_file_bytes_read, result.message) ||
        !sm87_target_aot_projection_sha256(
            {header.data(), header.size()}, &second_header_digest) ||
        second_header_digest != authenticated_header_digests[index] ||
        !persistence::decode_record_header(
            header.data(), header.size(), &second_record) ||
        !persisted_record_matches_plan(second_record, planned, index,
                                       result.message) ||
        !append_record_header_catalog_entry(
            second_record_header_catalog, index, header.data(),
            header.size())) {
      result.hard_failure = true;
      if (result.message.empty()) {
        result.message =
            "persisted target-AOT record header changed before device upload";
      }
      return result;
    }
    const auto layout =
        kernels::sm87_target_aot_projection_packed_layout(planned.role);
    if (!layout.valid() || layout.payload_bytes > staging.size() ||
        planned.device_arena_offset != verified_payload_catalog_bytes ||
        planned.device_arena_offset > pending.bytes ||
        layout.payload_bytes >
            pending.bytes - planned.device_arena_offset) {
      result.hard_failure = true;
      result.message = "persisted target-AOT payload escaped bounded storage";
      return result;
    }
    const std::size_t payload_bytes =
        static_cast<std::size_t>(layout.payload_bytes);
    if (!persistence::pread_exact_counted(
            bundle.get(), staging.data(), payload_bytes,
            record.bundle_payload_offset,
            result.persistent_bundle_file_bytes_read, result.message)) {
      result.hard_failure = true;
      return result;
    }
    kernels::Sm87TargetAotProjectionSha256Digest host_digest{};
    if (!sm87_target_aot_projection_sha256(
            {staging.data(), payload_bytes}, &host_digest) ||
        host_digest != record.manifest.payload_digest ||
        host_digest !=
            record.transform_receipt.payload.observed_payload_digest ||
        !scan_persisted_nvfp4_payload_scales(
            layout, staging.data(), payload_bytes, result.message)) {
      result.hard_failure = true;
      if (result.message.empty()) {
        result.message =
            "persisted target-AOT host payload authentication failed";
      }
      return result;
    }
    second_host_payload_catalog_hash_ok =
        second_host_payload_catalog_hash_ok &&
        append_verified_payload_catalog_entry(
            second_host_payload_catalog_hasher, planned, record.manifest,
            record.source_inventory, record.transform_receipt, host_digest);
    second_host_payload_catalog_bytes += layout.payload_bytes;

    status = cudaEventCreateWithFlags(&pending.upload_event,
                                      cudaEventDisableTiming);
    if (status == cudaSuccess) {
      status = cudaEventCreateWithFlags(&pending.verification_event,
                                        cudaEventDisableTiming);
    }
    if (status != cudaSuccess) {
      result.hard_failure = true;
      result.cuda_error = static_cast<int>(status);
      result.message =
          "failed to create persisted target-AOT completion events";
      return result;
    }
    std::uint8_t* const destination =
        pending.arena +
        static_cast<std::size_t>(planned.device_arena_offset);
    status = cudaMemcpyAsync(destination, staging.data(), payload_bytes,
                             cudaMemcpyHostToDevice, pending.stream);
    if (status == cudaSuccess) {
      status = cudaEventRecord(pending.upload_event, pending.stream);
    }
    if (status == cudaSuccess) {
      status = cudaEventSynchronize(pending.upload_event);
    }
    if (status != cudaSuccess) {
      result.hard_failure = true;
      result.cuda_error = static_cast<int>(status);
      result.message = "persisted target-AOT H2D or upload event failed";
      return result;
    }
    result.payload_h2d_bytes += layout.payload_bytes;

    status = cudaMemcpyAsync(staging.data(), destination, payload_bytes,
                             cudaMemcpyDeviceToHost, pending.stream);
    if (status == cudaSuccess) {
      status = cudaEventRecord(pending.verification_event, pending.stream);
    }
    if (status == cudaSuccess) {
      status = cudaEventSynchronize(pending.verification_event);
    }
    if (status != cudaSuccess) {
      result.hard_failure = true;
      result.cuda_error = static_cast<int>(status);
      result.message =
          "persisted target-AOT verification D2H or event failed";
      return result;
    }
    kernels::Sm87TargetAotProjectionSha256Digest readback_digest{};
    if (!sm87_target_aot_projection_sha256(
            {staging.data(), payload_bytes}, &readback_digest) ||
        readback_digest != record.manifest.payload_digest) {
      result.hard_failure = true;
      result.message =
          "persisted target-AOT device readback SHA-256 mismatch";
      return result;
    }
    verified_payload_catalog_hash_ok =
        verified_payload_catalog_hash_ok &&
        append_verified_payload_catalog_entry(
            verified_payload_catalog_hasher, planned, record.manifest,
            record.source_inventory, record.transform_receipt,
            readback_digest);
    if (!verified_payload_catalog_hash_ok) {
      result.hard_failure = true;
      result.message =
          "persisted target-AOT verified-payload catalog hashing failed";
      return result;
    }
    verified_payload_catalog_bytes += layout.payload_bytes;
    result.verification_d2h_bytes += layout.payload_bytes;

    if (!make_verified_device_descriptor(
            upload_context, planned, index, record.manifest,
            record.source_inventory, record.transform_receipt,
            readback_digest, pending.descriptors[index])) {
      result.hard_failure = true;
      result.message =
          "persisted target-AOT receipt or bound device view failed";
      return result;
    }
    ++pending.descriptor_count;

    const cudaError_t upload_destroy =
        cudaEventDestroy(pending.upload_event);
    pending.upload_event = nullptr;
    const cudaError_t verification_destroy =
        cudaEventDestroy(pending.verification_event);
    pending.verification_event = nullptr;
    if (upload_destroy != cudaSuccess ||
        verification_destroy != cudaSuccess) {
      result.hard_failure = true;
      result.cuda_error = static_cast<int>(
          upload_destroy != cudaSuccess ? upload_destroy
                                        : verification_destroy);
      result.message =
          "persisted target-AOT completion event destruction failed";
      return result;
    }
  }

  status = cudaStreamDestroy(pending.stream);
  pending.stream = nullptr;
  if (status != cudaSuccess) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message =
        "persisted target-AOT loader stream destruction failed";
    return result;
  }
  if (pending.descriptor_count != plans.size() ||
      verified_payload_catalog_bytes != pending.bytes ||
      !verified_payload_catalog_hash_ok ||
      second_host_payload_catalog_bytes != pending.bytes ||
      !second_host_payload_catalog_hash_ok ||
      !second_record_header_catalog_ok) {
    result.hard_failure = true;
    result.message =
        "persisted target-AOT inventory or payload catalog did not close";
    return result;
  }
  result.verified_payload_catalog_sha256 =
      verified_payload_catalog_hasher.finalize().hex();
  const std::string second_host_payload_catalog_sha256 =
      second_host_payload_catalog_hasher.finalize().hex();
  const auto second_record_header_catalog_digest =
      copy_digest(second_record_header_catalog.finalize());
  if (result.verified_payload_catalog_sha256 !=
          expected_verified_payload_catalog_sha256 ||
      second_host_payload_catalog_sha256 !=
          expected_verified_payload_catalog_sha256 ||
      second_record_header_catalog_digest !=
          superblock.record_header_catalog_sha256 ||
      result.verified_payload_catalog_sha256.size() != 64U) {
    result.hard_failure = true;
    result.message =
        "persisted target-AOT payload catalog differs from the external "
        "trust root";
    return result;
  }
  persistence::FileSnapshot final_snapshot;
  if (!persistence::snapshot_regular_file_exact(
          bundle.get(), persistence::kPersistedBundleBytes, final_snapshot,
          result.message) ||
      !persistence::same_snapshot(authenticated_snapshot, final_snapshot) ||
      result.persistent_bundle_file_bytes_read !=
          kSm87TargetAotProjectionPersistentDirectLoadFileBytesRead) {
    result.hard_failure = true;
    if (result.message.empty()) {
      result.message =
          "target-AOT persisted bundle changed during authenticated loading";
    }
    return result;
  }
  result.persistent_bundle_host_authentication_passes = 2U;
  status = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (status != cudaSuccess ||
      static_cast<std::uint64_t>(free_bytes) <
          minimum_free_bytes_after_load) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message =
        status == cudaSuccess
            ? "completed persisted target-AOT load violated the retained "
              "memory margin"
            : "cudaMemGetInfo failed after persisted target-AOT loading";
    return result;
  }

  owner_arena = pending.arena;
  owner_bytes = pending.bytes;
  allocation_identity = pending_allocation_identity;
  owner_identity = pending_owner_identity;
  owner_device_ordinal = device_ordinal;
  descriptors = pending.descriptors;
  descriptor_count = pending.descriptor_count;
  pending.arena = nullptr;
  pending.bytes = 0U;
  pending.descriptor_count = 0U;

  result.enabled = true;
  result.artifacts = descriptor_count;
  result.sources = kSm87TargetAotProjectionDeviceSourceCount;
  result.arena_bytes = owner_bytes;
  result.owner_identity = owner_identity;
  result.allocation_identity = allocation_identity;
  result.device_ordinal = owner_device_ordinal;
  return result;
}

[[nodiscard]] Sm87TargetAotProjectionDevicePreparationStats prepare_impl(
    const ResidentWeights& resident, const ModelWeights& model_weights,
    const std::filesystem::path& create_bundle_path,
    const std::string_view expected_verified_payload_catalog_sha256,
    const std::uint64_t minimum_free_bytes_after_prepare,
    Sm87TargetAotProjectionDeviceAssets& owner,
    std::uint8_t*& owner_arena, std::uint64_t& owner_bytes,
    std::uint64_t& allocation_identity, std::uint64_t& owner_identity,
    std::int32_t& owner_device_ordinal,
    std::array<Sm87TargetAotProjectionDeviceAssetDescriptor,
               kSm87TargetAotProjectionDeviceArtifactCount>& descriptors,
    std::size_t& descriptor_count) {
  Sm87TargetAotProjectionDevicePreparationStats result;
  result.source =
      Sm87TargetAotProjectionDeviceAssetSource::kOnlineCheckpointTransform;
  if (!owner.empty()) {
    result.hard_failure = true;
    result.message = "target-AOT device asset owner was not empty";
    return result;
  }

  const bool create_bundle = !create_bundle_path.empty();
  kernels::Sm87TargetAotProjectionSha256Digest expected_catalog{};
  if (create_bundle !=
          !expected_verified_payload_catalog_sha256.empty() ||
      (create_bundle &&
       !parse_lower_hex_digest(
           expected_verified_payload_catalog_sha256, expected_catalog))) {
    result.hard_failure = true;
    result.message =
        "target-AOT offline creation requires one absolute output path and "
        "one nonzero lowercase external catalog SHA-256";
    return result;
  }

  core::Sha256Digest checkpoint_digest{};
  if (!verify_observed_checkpoint(resident, checkpoint_digest,
                                  result.message)) {
    result.hard_failure = true;
    return result;
  }
  ArtifactPlans plans{};
  if (!plan_inventory(resident, model_weights, checkpoint_digest, plans,
                      result.message)) {
    result.hard_failure = true;
    return result;
  }

  PersistentBundleWriter persistent_writer;

  int device_ordinal = -1;
  cudaError_t status = cudaGetDevice(&device_ordinal);
  if (status != cudaSuccess || device_ordinal < 0) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message = "cudaGetDevice failed before target-AOT preparation";
    return result;
  }
  cudaPointerAttributes resident_attributes{};
  status = cudaPointerGetAttributes(&resident_attributes,
                                    resident.arena_data());
  if (status != cudaSuccess ||
      resident_attributes.type != cudaMemoryTypeDevice ||
      resident_attributes.device != device_ordinal) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message = "resident checkpoint arena is not device memory on the "
                     "current CUDA ordinal";
    return result;
  }
  std::size_t free_bytes = 0U;
  std::size_t total_bytes = 0U;
  status = cudaMemGetInfo(&free_bytes, &total_bytes);
  (void)total_bytes;
  const std::uint64_t required_before =
      kSm87TargetAotProjectionDeviceArenaBytes +
      kSm87TargetAotProjectionMaximumHostStagingBytes;
  if (status != cudaSuccess || required_before > free_bytes ||
      minimum_free_bytes_after_prepare > free_bytes - required_before) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message = status == cudaSuccess
                         ? "insufficient memory for the target-AOT arena, "
                           "bounded dual staging, and retained margin"
                         : "cudaMemGetInfo failed before target-AOT staging";
    return result;
  }

  std::vector<std::uint8_t> canonical(
      static_cast<std::size_t>(
          kSm87TargetAotProjectionMaximumArtifactPayloadBytes));
  std::vector<std::uint8_t> payload(
      static_cast<std::size_t>(
          kSm87TargetAotProjectionMaximumArtifactPayloadBytes));
  result.host_staging_peak_bytes =
      kSm87TargetAotProjectionMaximumHostStagingBytes;
  status = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (status != cudaSuccess ||
      kSm87TargetAotProjectionDeviceArenaBytes > free_bytes ||
      minimum_free_bytes_after_prepare >
          free_bytes - kSm87TargetAotProjectionDeviceArenaBytes) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message = status == cudaSuccess
                         ? "bounded host staging consumed the target-AOT "
                           "post-allocation margin"
                         : "cudaMemGetInfo failed after host staging";
    return result;
  }

  void* allocation = nullptr;
  status = cudaMalloc(
      &allocation,
      static_cast<std::size_t>(kSm87TargetAotProjectionDeviceArenaBytes));
  if (status != cudaSuccess || allocation == nullptr) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message = "cudaMalloc failed for the exact target-AOT arena";
    return result;
  }
  owner_arena = static_cast<std::uint8_t*>(allocation);
  owner_bytes = kSm87TargetAotProjectionDeviceArenaBytes;
  owner_device_ordinal = device_ordinal;
  cudaPointerAttributes allocation_attributes{};
  status = cudaPointerGetAttributes(&allocation_attributes, owner_arena);
  if (status != cudaSuccess ||
      allocation_attributes.type != cudaMemoryTypeDevice ||
      allocation_attributes.device != device_ordinal) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message = "target-AOT allocation is not device memory on the "
                     "current CUDA ordinal";
    owner.release();
    return result;
  }
  const std::uint64_t transaction_serial =
      g_device_asset_transaction_serial.fetch_add(1U,
                                                   std::memory_order_relaxed);
  owner_identity = make_runtime_identity(
      kOwnerIdentityDomain, checkpoint_digest, transaction_serial,
      static_cast<std::uint64_t>(device_ordinal), owner_bytes);
  allocation_identity = make_runtime_identity(
      kAllocationIdentityDomain, checkpoint_digest, transaction_serial,
      owner_identity, owner_bytes,
      static_cast<std::uint64_t>(device_ordinal));
  if (reinterpret_cast<std::uintptr_t>(owner_arena) %
              kernels::kSm87TargetAotProjectionPackedAlignment !=
          0U ||
      owner_identity == 0U || allocation_identity == 0U) {
    result.hard_failure = true;
    result.message = "target-AOT allocation alignment or identity failed";
    owner.release();
    return result;
  }
  status = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (status != cudaSuccess ||
      static_cast<std::uint64_t>(free_bytes) <
          minimum_free_bytes_after_prepare) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message = status == cudaSuccess
                         ? "target-AOT allocation violated retained margin"
                         : "cudaMemGetInfo failed after target-AOT allocation";
    owner.release();
    return result;
  }

  cudaStream_t stream = nullptr;
  status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (status != cudaSuccess) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message = "failed to create target-AOT loader stream";
    owner.release();
    return result;
  }
  cudaEvent_t upload_event = nullptr;
  cudaEvent_t verification_event = nullptr;
  core::Sha256 verified_payload_catalog_hasher;
  bool verified_payload_catalog_hash_ok =
      hash_string(verified_payload_catalog_hasher,
                  kVerifiedPayloadCatalogDomain);
  std::uint64_t verified_payload_catalog_bytes = 0U;
  const std::uint64_t stream_identity = make_runtime_identity(
      kStreamIdentityDomain, checkpoint_digest, transaction_serial,
      owner_identity, 1U);
  if (stream_identity == 0U) {
    result.hard_failure = true;
    result.message = "failed to derive target-AOT stream identity";
    cleanup_stream(stream, upload_event, verification_event);
    owner.release();
    return result;
  }
  const VerifiedDeviceUploadContext upload_context{
      owner_arena,        owner_bytes,          allocation_identity,
      owner_identity,     owner_device_ordinal, stream_identity,
      transaction_serial, &checkpoint_digest};
  if (create_bundle &&
      !persistent_writer.open_create_only(create_bundle_path,
                                          result.message)) {
    result.hard_failure = true;
    cleanup_stream(stream, upload_event, verification_event);
    owner.release();
    return result;
  }

  for (std::size_t index = 0U; index < plans.size(); ++index) {
    const PlannedArtifact& planned = plans[index];
    const auto layout =
        kernels::sm87_target_aot_projection_packed_layout(planned.role);
    if (!layout.valid() || layout.payload_bytes > canonical.size() ||
        layout.payload_bytes > payload.size() ||
        planned.device_arena_offset > owner_bytes ||
        layout.payload_bytes > owner_bytes - planned.device_arena_offset) {
      result.hard_failure = true;
      result.message = "target-AOT artifact escaped bounded storage";
      cleanup_stream(stream, upload_event, verification_event);
      owner.release();
      return result;
    }

    Sm87TargetAotProjectionSourceSet sources;
    if (!copy_and_bind_sources(planned, stream, canonical, sources,
                               result.source_d2h_bytes, result.cuda_error,
                               result.message)) {
      result.hard_failure = true;
      cleanup_stream(stream, upload_event, verification_event);
      owner.release();
      return result;
    }
    const auto inspection =
        sm87_target_aot_projection_inspect_sources(sources);
    if (!inspection) {
      result.hard_failure = true;
      result.message = "target-AOT source inspection failed at layer " +
                       std::to_string(planned.layer_index) + ": " +
                       sm87_target_aot_projection_asset_error_string(
                           inspection.error);
      cleanup_stream(stream, upload_event, verification_event);
      owner.release();
      return result;
    }
    const auto build = sm87_target_aot_projection_build_asset(
        planned.artifact_identity, sources, inspection.inventory,
        {payload.data(), static_cast<std::size_t>(layout.payload_bytes)});
    if (!build ||
        sm87_target_aot_projection_validate_asset(
            build.manifest, build.transform_receipt, sources,
            inspection.inventory,
            {payload.data(), static_cast<std::size_t>(layout.payload_bytes)}) !=
            Sm87TargetAotProjectionAssetError::kSuccess) {
      result.hard_failure = true;
      result.message = "target-AOT host build/validation failed at layer " +
                       std::to_string(planned.layer_index) + ": " +
                       sm87_target_aot_projection_asset_error_string(
                           build.error);
      cleanup_stream(stream, upload_event, verification_event);
      owner.release();
      return result;
    }
    kernels::Sm87TargetAotProjectionSha256Digest before_copy_digest{};
    if (!sm87_target_aot_projection_sha256(
            {payload.data(), static_cast<std::size_t>(layout.payload_bytes)},
            &before_copy_digest) ||
        before_copy_digest != build.manifest.payload_digest) {
      result.hard_failure = true;
      result.message = "target-AOT payload changed before H2D";
      cleanup_stream(stream, upload_event, verification_event);
      owner.release();
      return result;
    }

    status = cudaEventCreateWithFlags(&upload_event, cudaEventDisableTiming);
    if (status == cudaSuccess) {
      status = cudaEventCreateWithFlags(&verification_event,
                                        cudaEventDisableTiming);
    }
    if (status != cudaSuccess) {
      result.hard_failure = true;
      result.cuda_error = static_cast<int>(status);
      result.message = "failed to create target-AOT completion events";
      cleanup_stream(stream, upload_event, verification_event);
      owner.release();
      return result;
    }
    std::uint8_t* const destination =
        owner_arena + static_cast<std::size_t>(planned.device_arena_offset);
    status = cudaMemcpyAsync(destination, payload.data(),
                             static_cast<std::size_t>(layout.payload_bytes),
                             cudaMemcpyHostToDevice, stream);
    if (status == cudaSuccess) {
      status = cudaEventRecord(upload_event, stream);
    }
    if (status == cudaSuccess) {
      status = cudaEventSynchronize(upload_event);
    }
    if (status != cudaSuccess) {
      result.hard_failure = true;
      result.cuda_error = static_cast<int>(status);
      result.message = "target-AOT H2D or upload event failed";
      cleanup_stream(stream, upload_event, verification_event);
      owner.release();
      return result;
    }
    result.payload_h2d_bytes += layout.payload_bytes;

    status = cudaMemcpyAsync(canonical.data(), destination,
                             static_cast<std::size_t>(layout.payload_bytes),
                             cudaMemcpyDeviceToHost, stream);
    if (status == cudaSuccess) {
      status = cudaEventRecord(verification_event, stream);
    }
    if (status == cudaSuccess) {
      status = cudaEventSynchronize(verification_event);
    }
    if (status != cudaSuccess) {
      result.hard_failure = true;
      result.cuda_error = static_cast<int>(status);
      result.message = "target-AOT verification D2H or event failed";
      cleanup_stream(stream, upload_event, verification_event);
      owner.release();
      return result;
    }
    kernels::Sm87TargetAotProjectionSha256Digest readback_digest{};
    if (!sm87_target_aot_projection_sha256(
            {canonical.data(),
             static_cast<std::size_t>(layout.payload_bytes)},
            &readback_digest) ||
        readback_digest != build.manifest.payload_digest) {
      result.hard_failure = true;
      result.message = "target-AOT device readback SHA-256 mismatch";
      cleanup_stream(stream, upload_event, verification_event);
      owner.release();
      return result;
    }
    if (planned.device_arena_offset != verified_payload_catalog_bytes) {
      result.hard_failure = true;
      result.message =
          "target-AOT verified-payload catalog order failed";
      cleanup_stream(stream, upload_event, verification_event);
      owner.release();
      return result;
    }
    verified_payload_catalog_hash_ok =
        verified_payload_catalog_hash_ok &&
        append_verified_payload_catalog_entry(
            verified_payload_catalog_hasher, planned, build.manifest,
            inspection.inventory, build.transform_receipt, readback_digest);
    if (!verified_payload_catalog_hash_ok) {
      result.hard_failure = true;
      result.message = "target-AOT verified-payload catalog hashing failed";
      cleanup_stream(stream, upload_event, verification_event);
      owner.release();
      return result;
    }
    verified_payload_catalog_bytes += layout.payload_bytes;
    result.verification_d2h_bytes += layout.payload_bytes;

    if (create_bundle) {
      persistence::PersistenceRecord persistent_record;
      persistent_record.ordinal = index;
      persistent_record.layer_index = planned.layer_index;
      persistent_record.role = planned.role;
      persistent_record.arena_offset = planned.device_arena_offset;
      persistent_record.bundle_header_offset =
          persistence::persistence_expected_bundle_header_offset(
              planned.layer_index, planned.role);
      persistent_record.bundle_payload_offset =
          persistent_record.bundle_header_offset +
          persistence::kRecordHeaderBytes;
      persistent_record.bundle_record_bytes =
          persistence::kRecordHeaderBytes + layout.payload_bytes;
      persistent_record.source_inventory = inspection.inventory;
      persistent_record.manifest = build.manifest;
      persistent_record.transform_receipt = build.transform_receipt;
      if (!persisted_record_matches_plan(persistent_record, planned, index,
                                         result.message) ||
          !persistent_writer.write_record(
              index, persistent_record, canonical.data(),
              static_cast<std::size_t>(layout.payload_bytes),
              result.message)) {
        result.hard_failure = true;
        cleanup_stream(stream, upload_event, verification_event);
        owner.release();
        return result;
      }
    }

    Sm87TargetAotProjectionDeviceAssetDescriptor descriptor;
    if (!make_verified_device_descriptor(
            upload_context, planned, index, build.manifest,
            inspection.inventory, build.transform_receipt, readback_digest,
            descriptor)) {
      result.hard_failure = true;
      result.message = "target-AOT receipt or bound device view failed";
      cleanup_stream(stream, upload_event, verification_event);
      owner.release();
      return result;
    }
    descriptors[index] = descriptor;

    const cudaError_t upload_destroy = cudaEventDestroy(upload_event);
    upload_event = nullptr;
    const cudaError_t verification_destroy =
        cudaEventDestroy(verification_event);
    verification_event = nullptr;
    if (upload_destroy != cudaSuccess ||
        verification_destroy != cudaSuccess) {
      result.hard_failure = true;
      result.cuda_error = static_cast<int>(
          upload_destroy != cudaSuccess ? upload_destroy
                                        : verification_destroy);
      result.message = "target-AOT completion event destruction failed";
      cleanup_stream(stream, upload_event, verification_event);
      owner.release();
      return result;
    }
  }

  status = cudaStreamDestroy(stream);
  stream = nullptr;
  if (status != cudaSuccess) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message = "target-AOT loader stream destruction failed";
    owner.release();
    return result;
  }
  if (verified_payload_catalog_bytes != owner_bytes ||
      !verified_payload_catalog_hash_ok) {
    result.hard_failure = true;
    result.message =
        "target-AOT verified-payload catalog did not close the exact arena";
    owner.release();
    return result;
  }
  result.verified_payload_catalog_sha256 =
      verified_payload_catalog_hasher.finalize().hex();
  if (result.verified_payload_catalog_sha256.size() != 64U ||
      result.verified_payload_catalog_sha256 == std::string(64U, '0')) {
    result.hard_failure = true;
    result.message = "target-AOT verified-payload catalog digest was invalid";
    owner.release();
    return result;
  }
  if (create_bundle &&
      result.verified_payload_catalog_sha256 !=
          expected_verified_payload_catalog_sha256) {
    result.hard_failure = true;
    result.message =
        "target-AOT offline payload catalog differs from the external "
        "trust root";
    owner.release();
    return result;
  }
  descriptor_count = descriptors.size();
  for (std::size_t layer = 0U;
       layer < kSm87TargetAotProjectionDeviceLayerCount; ++layer) {
    if (!owner.has_asset(
            layer,
            kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp) ||
        !owner.has_asset(
            layer, kernels::Sm87TargetAotProjectionRole::kNvFp4Down)) {
      result.hard_failure = true;
      result.message = "retained target-AOT inventory is incomplete";
      owner.release();
      return result;
    }
  }
  status = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (status != cudaSuccess ||
      static_cast<std::uint64_t>(free_bytes) <
          minimum_free_bytes_after_prepare) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message = status == cudaSuccess
                         ? "completed target-AOT preparation violated the "
                           "retained memory margin"
                         : "cudaMemGetInfo failed after target-AOT "
                           "preparation";
    owner.release();
    return result;
  }
  if (create_bundle) {
    if (!persistent_writer.publish(
            copy_digest(checkpoint_digest), expected_catalog,
            result.persistent_record_header_catalog_sha256,
            result.message)) {
      result.hard_failure = true;
      owner.release();
      return result;
    }
    result.persistent_bundle_created = true;
    result.persistent_bundle_file_bytes_written =
        persistent_writer.bytes_written();
  }
  result.enabled = true;
  result.artifacts = descriptor_count;
  result.sources = kSm87TargetAotProjectionDeviceSourceCount;
  result.arena_bytes = owner_bytes;
  result.owner_identity = owner_identity;
  result.allocation_identity = allocation_identity;
  result.device_ordinal = owner_device_ordinal;
  return result;
}

#endif  // Q3X_ENABLE_SM87_TARGET_AOT_DEVICE_ASSETS_V1_ADMISSION

}  // namespace

Sm87TargetAotProjectionDeviceAssets::~Sm87TargetAotProjectionDeviceAssets() {
  release_unconditionally();
}

bool Sm87TargetAotProjectionDeviceAssets::release() noexcept {
  if (attached_model_weights_ != nullptr) {
    return false;
  }
  release_unconditionally();
  return true;
}

void Sm87TargetAotProjectionDeviceAssets::release_unconditionally() noexcept {
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_DEVICE_ASSETS_V1_ADMISSION)
  if (arena_ != nullptr) {
    (void)cudaFree(arena_);
  }
#endif
  arena_ = nullptr;
  bytes_ = 0U;
  allocation_identity_ = 0U;
  owner_identity_ = 0U;
  device_ordinal_ = -1;
  descriptors_ = {};
  descriptor_count_ = 0U;
  prepared_model_weights_ = nullptr;
  attached_model_weights_ = nullptr;
}

const Sm87TargetAotProjectionDeviceAssetDescriptor*
Sm87TargetAotProjectionDeviceAssets::find(
    const std::size_t layer_index,
    const kernels::Sm87TargetAotProjectionRole role) const noexcept {
  if (layer_index >= kSm87TargetAotProjectionDeviceLayerCount ||
      (role != kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp &&
       role != kernels::Sm87TargetAotProjectionRole::kNvFp4Down)) {
    return nullptr;
  }
  for (std::size_t index = 0U; index < descriptor_count_; ++index) {
    if (descriptors_[index].layer_index == layer_index &&
        descriptors_[index].role == role) {
      return &descriptors_[index];
    }
  }
  return nullptr;
}

Sm87TargetAotProjectionDevicePreparationStats
Sm87TargetAotProjectionDeviceAssets::prepare(
    const ResidentWeights& resident, const ModelWeights& model_weights,
    const std::filesystem::path& create_bundle_path,
    const std::string_view expected_verified_payload_catalog_sha256,
    const std::uint64_t minimum_free_bytes_after_prepare) {
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_DEVICE_ASSETS_V1_ADMISSION)
  try {
    Sm87TargetAotProjectionDevicePreparationStats result = prepare_impl(
        resident, model_weights, create_bundle_path,
        expected_verified_payload_catalog_sha256,
        minimum_free_bytes_after_prepare, *this, arena_, bytes_,
        allocation_identity_, owner_identity_, device_ordinal_, descriptors_,
        descriptor_count_);
    if (result.enabled && !result.hard_failure) {
      prepared_model_weights_ = &model_weights;
    } else {
      (void)release();
    }
    return result;
  } catch (const std::exception& error) {
    (void)release();
    Sm87TargetAotProjectionDevicePreparationStats result;
    result.source =
        Sm87TargetAotProjectionDeviceAssetSource::kOnlineCheckpointTransform;
    result.hard_failure = true;
    result.message =
        std::string("exception during target-AOT device preparation: ") +
        error.what();
    return result;
  } catch (...) {
    (void)release();
    Sm87TargetAotProjectionDevicePreparationStats result;
    result.source =
        Sm87TargetAotProjectionDeviceAssetSource::kOnlineCheckpointTransform;
    result.hard_failure = true;
    result.message =
        "unknown exception during target-AOT device preparation";
    return result;
  }
#else
  (void)resident;
  (void)model_weights;
  (void)create_bundle_path;
  (void)expected_verified_payload_catalog_sha256;
  (void)minimum_free_bytes_after_prepare;
  Sm87TargetAotProjectionDevicePreparationStats result;
  result.source =
      Sm87TargetAotProjectionDeviceAssetSource::kOnlineCheckpointTransform;
  result.hard_failure = true;
  result.message = "target-AOT device asset preparation is not compiled";
  return result;
#endif
}

Sm87TargetAotProjectionDevicePreparationStats
Sm87TargetAotProjectionDeviceAssets::load_persisted(
    const ResidentWeights& resident, const ModelWeights& model_weights,
    const std::filesystem::path& bundle_path,
    const std::string_view expected_verified_payload_catalog_sha256,
    const std::uint64_t minimum_free_bytes_after_load) {
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_DEVICE_ASSETS_V1_ADMISSION)
  try {
    Sm87TargetAotProjectionDevicePreparationStats result =
        load_persisted_impl(
            resident, model_weights, bundle_path,
            expected_verified_payload_catalog_sha256,
            minimum_free_bytes_after_load, *this, arena_, bytes_,
            allocation_identity_, owner_identity_, device_ordinal_,
            descriptors_, descriptor_count_);
    if (result.enabled && !result.hard_failure) {
      prepared_model_weights_ = &model_weights;
    } else {
      (void)release();
    }
    return result;
  } catch (const std::exception& error) {
    (void)release();
    Sm87TargetAotProjectionDevicePreparationStats result;
    result.source =
        Sm87TargetAotProjectionDeviceAssetSource::kPersistedBundle;
    result.hard_failure = true;
    result.message =
        std::string("exception during target-AOT persisted loading: ") +
        error.what();
    return result;
  } catch (...) {
    (void)release();
    Sm87TargetAotProjectionDevicePreparationStats result;
    result.source =
        Sm87TargetAotProjectionDeviceAssetSource::kPersistedBundle;
    result.hard_failure = true;
    result.message =
        "unknown exception during target-AOT persisted loading";
    return result;
  }
#else
  (void)resident;
  (void)model_weights;
  (void)bundle_path;
  (void)expected_verified_payload_catalog_sha256;
  (void)minimum_free_bytes_after_load;
  Sm87TargetAotProjectionDevicePreparationStats result;
  result.source =
      Sm87TargetAotProjectionDeviceAssetSource::kPersistedBundle;
  result.hard_failure = true;
  result.message = "target-AOT persisted device asset loading is not compiled";
  return result;
#endif
}

}  // namespace q3x::runtime
