#include "q3x/kernels/sm87_p40_packed_projection.h"

#include "q3x/core/sha256.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>

namespace q3x::kernels {
namespace {

using ManifestIssueBits = std::underlying_type_t<Sm87P40PackedManifestIssue>;
using InventoryIssueBits = std::underlying_type_t<Sm87P40PackedInventoryIssue>;

constexpr char kManifestHashDomain[] =
    "q3x.sm87.p40.packed-projection.manifest.v1";

void flag(Sm87P40PackedManifestValidation* const validation,
          const Sm87P40PackedManifestIssue issue) noexcept {
  validation->issue_mask |= static_cast<ManifestIssueBits>(issue);
}

void flag(Sm87P40PackedInventoryValidation* const validation,
          const Sm87P40PackedInventoryIssue issue) noexcept {
  validation->issue_mask |= static_cast<InventoryIssueBits>(issue);
}

template <typename Unsigned>
bool update_little_endian(core::Sha256* const hasher,
                          Unsigned value) noexcept {
  static_assert(std::is_unsigned_v<Unsigned>);
  std::array<std::uint8_t, sizeof(Unsigned)> bytes{};
  for (std::size_t index = 0U; index < bytes.size(); ++index) {
    bytes[index] = static_cast<std::uint8_t>(value >> (8U * index));
  }
  return hasher->update(bytes.data(), bytes.size());
}

bool update_digest(core::Sha256* const hasher,
                   const Sm87P40PackedDigest& digest) noexcept {
  return hasher->update(digest.bytes.data(), digest.bytes.size());
}

bool same_magic(const std::array<std::uint8_t, 8U>& magic) noexcept {
  for (std::size_t index = 0U; index < magic.size(); ++index) {
    if (magic[index] != kSm87P40PackedProjectionMagic[index]) {
      return false;
    }
  }
  return true;
}

bool finite_positive_scale(const std::uint32_t bits) noexcept {
  float value = 0.0F;
  static_assert(sizeof(value) == sizeof(bits));
  std::memcpy(&value, &bits, sizeof(value));
  return std::isfinite(value) && value > 0.0F;
}

bool source_is_canonical_zero(
    const Sm87P40PackedSourceIdentity& source) noexcept {
  return source.role == Sm87P40PackedLogicalRole::kInvalid &&
         source.tensor_identity == 0U &&
         sm87_p40_packed_digest_is_zero(source.weight_digest) &&
         sm87_p40_packed_digest_is_zero(source.scale_digest) &&
         source.global_scale_bits == 0U;
}

bool role_matches_layer(const Sm87P40PackedProjectionRole role,
                        const std::size_t layer_index) noexcept {
  if (layer_index >= kSm87P40PackedProjectionLayerCount) {
    return false;
  }
  if (role == Sm87P40PackedProjectionRole::kFp8LinearQkvZ) {
    return !sm87_p40_packed_is_full_layer(layer_index);
  }
  if (role == Sm87P40PackedProjectionRole::kFp8FullQkv) {
    return sm87_p40_packed_is_full_layer(layer_index);
  }
  return role == Sm87P40PackedProjectionRole::kNvFp4GateUp ||
         role == Sm87P40PackedProjectionRole::kNvFp4Down ||
         role == Sm87P40PackedProjectionRole::kFp8AttentionOutput;
}

Sm87P40PackedManifestValidation validate_manifest_impl(
    const Sm87P40PackedArtifactManifest& manifest,
    const bool check_manifest_digest) noexcept {
  Sm87P40PackedManifestValidation result;
  if (!same_magic(manifest.magic)) {
    flag(&result, Sm87P40PackedManifestIssue::kMagic);
  }
  if (manifest.abi_major != kSm87P40PackedProjectionAbiMajor ||
      manifest.abi_minor != kSm87P40PackedProjectionAbiMinor) {
    flag(&result, Sm87P40PackedManifestIssue::kVersion);
  }
  if (manifest.header_bytes != kSm87P40PackedProjectionHeaderBytes) {
    flag(&result, Sm87P40PackedManifestIssue::kHeader);
  }
  if (manifest.layer_index >= kSm87P40PackedProjectionLayerCount) {
    flag(&result, Sm87P40PackedManifestIssue::kLayer);
  }
  if (manifest.artifact_identity == 0U) {
    flag(&result, Sm87P40PackedManifestIssue::kArtifactIdentity);
  }

  const auto plan = sm87_p40_packed_projection_plan(manifest.role);
  if (!plan.valid() || !role_matches_layer(manifest.role,
                                           manifest.layer_index)) {
    flag(&result, Sm87P40PackedManifestIssue::kRole);
  }
  if (!plan.valid() || manifest.tactic != plan.tactic) {
    flag(&result, Sm87P40PackedManifestIssue::kTactic);
  }
  if (manifest.provider != Sm87P40PackedProvider::kNativeSm87) {
    flag(&result, Sm87P40PackedManifestIssue::kProvider);
  }
  if (manifest.policy != kSm87P40PackedRequiredPolicy) {
    flag(&result, Sm87P40PackedManifestIssue::kPolicy);
  }
  if (!plan.valid() || manifest.source_count != plan.source_count) {
    flag(&result, Sm87P40PackedManifestIssue::kSourceCount);
  }

  const std::size_t active_sources =
      plan.valid() ? plan.source_count : 0U;
  for (std::size_t index = 0U; index < manifest.sources.size(); ++index) {
    const auto& source = manifest.sources[index];
    if (index >= active_sources) {
      if (!source_is_canonical_zero(source)) {
        flag(&result, Sm87P40PackedManifestIssue::kSourceRole);
        if (source.tensor_identity != 0U) {
          flag(&result, Sm87P40PackedManifestIssue::kSourceIdentity);
        }
        if (!sm87_p40_packed_digest_is_zero(source.weight_digest) ||
            !sm87_p40_packed_digest_is_zero(source.scale_digest)) {
          flag(&result, Sm87P40PackedManifestIssue::kSourceDigest);
        }
        if (source.global_scale_bits != 0U) {
          flag(&result, Sm87P40PackedManifestIssue::kScale);
        }
      }
      continue;
    }
    if (source.role != plan.partitions[index].role) {
      flag(&result, Sm87P40PackedManifestIssue::kSourceRole);
    }
    if (source.tensor_identity == 0U) {
      flag(&result, Sm87P40PackedManifestIssue::kSourceIdentity);
    }
    for (std::size_t other = 0U; other < index; ++other) {
      if (source.tensor_identity != 0U &&
          source.tensor_identity ==
              manifest.sources[other].tensor_identity) {
        flag(&result, Sm87P40PackedManifestIssue::kSourceIdentity);
      }
    }
    if (sm87_p40_packed_digest_is_zero(source.weight_digest) ||
        sm87_p40_packed_digest_is_zero(source.scale_digest)) {
      flag(&result, Sm87P40PackedManifestIssue::kSourceDigest);
    }
    if (!finite_positive_scale(source.global_scale_bits)) {
      flag(&result, Sm87P40PackedManifestIssue::kScale);
    }
  }

  bool payload_range_valid = plan.valid();
  payload_range_valid =
      payload_range_valid &&
      manifest.payload_offset == kSm87P40PackedProjectionHeaderBytes &&
      manifest.payload_offset % kSm87P40PackedProjectionPayloadAlignment ==
          0U &&
      manifest.payload_bytes == plan.payload_bytes;
  if (manifest.payload_offset >
      std::numeric_limits<std::uint64_t>::max() - manifest.payload_bytes) {
    payload_range_valid = false;
  } else {
    payload_range_valid =
        payload_range_valid &&
        manifest.artifact_bytes ==
            manifest.payload_offset + manifest.payload_bytes;
  }
  if (!payload_range_valid) {
    flag(&result, Sm87P40PackedManifestIssue::kPayloadRange);
  }
  if (sm87_p40_packed_digest_is_zero(manifest.payload_digest)) {
    flag(&result, Sm87P40PackedManifestIssue::kPayloadDigest);
  }
  if (sm87_p40_packed_digest_is_zero(manifest.model_digest) ||
      sm87_p40_packed_digest_is_zero(manifest.checkpoint_digest)) {
    flag(&result, Sm87P40PackedManifestIssue::kModelDigest);
  }
  if (check_manifest_digest) {
    Sm87P40PackedDigest expected;
    if (!compute_sm87_p40_packed_manifest_digest(manifest, &expected) ||
        sm87_p40_packed_digest_is_zero(manifest.manifest_digest) ||
        manifest.manifest_digest != expected) {
      flag(&result, Sm87P40PackedManifestIssue::kManifestDigest);
    }
  }
  return result;
}

}  // namespace

Sm87P40PackedArtifactManifest make_sm87_p40_packed_artifact_manifest(
    const Sm87P40PackedProjectionRole role,
    const std::size_t layer_index) noexcept {
  const auto plan = sm87_p40_packed_projection_plan(role);
  if (!plan.valid() || !role_matches_layer(role, layer_index)) {
    return {};
  }
  Sm87P40PackedArtifactManifest manifest;
  manifest.magic = kSm87P40PackedProjectionMagic;
  manifest.abi_major = kSm87P40PackedProjectionAbiMajor;
  manifest.abi_minor = kSm87P40PackedProjectionAbiMinor;
  manifest.header_bytes = kSm87P40PackedProjectionHeaderBytes;
  manifest.layer_index = static_cast<std::uint32_t>(layer_index);
  manifest.role = role;
  manifest.tactic = plan.tactic;
  manifest.provider = Sm87P40PackedProvider::kNativeSm87;
  manifest.policy = kSm87P40PackedRequiredPolicy;
  manifest.source_count = plan.source_count;
  manifest.payload_offset = kSm87P40PackedProjectionHeaderBytes;
  manifest.payload_bytes = plan.payload_bytes;
  manifest.artifact_bytes = manifest.payload_offset + manifest.payload_bytes;
  for (std::size_t index = 0U; index < plan.source_count; ++index) {
    manifest.sources[index].role = plan.partitions[index].role;
  }
  return manifest;
}

bool compute_sm87_p40_packed_manifest_digest(
    const Sm87P40PackedArtifactManifest& manifest,
    Sm87P40PackedDigest* const digest) noexcept {
  if (digest == nullptr) {
    return false;
  }
  core::Sha256 hasher;
  bool ok = hasher.update(kManifestHashDomain,
                          sizeof(kManifestHashDomain) - 1U);
  ok = ok && hasher.update(manifest.magic.data(), manifest.magic.size());
  ok = ok && update_little_endian(&hasher, manifest.abi_major);
  ok = ok && update_little_endian(&hasher, manifest.abi_minor);
  ok = ok && update_little_endian(&hasher, manifest.header_bytes);
  ok = ok && update_little_endian(&hasher, manifest.layer_index);
  ok = ok && update_little_endian(&hasher, manifest.artifact_identity);
  ok = ok && update_little_endian(
                 &hasher, static_cast<std::uint8_t>(manifest.role));
  ok = ok && update_little_endian(
                 &hasher, static_cast<std::uint8_t>(manifest.tactic));
  ok = ok && update_little_endian(
                 &hasher, static_cast<std::uint8_t>(manifest.provider));
  ok = ok && update_little_endian(&hasher, manifest.policy);
  ok = ok && update_little_endian(&hasher, manifest.source_count);
  ok = ok && update_little_endian(&hasher, manifest.payload_offset);
  ok = ok && update_little_endian(&hasher, manifest.payload_bytes);
  ok = ok && update_little_endian(&hasher, manifest.artifact_bytes);
  ok = ok && update_digest(&hasher, manifest.model_digest);
  ok = ok && update_digest(&hasher, manifest.checkpoint_digest);
  ok = ok && update_digest(&hasher, manifest.payload_digest);
  for (const auto& source : manifest.sources) {
    ok = ok && update_little_endian(
                   &hasher, static_cast<std::uint8_t>(source.role));
    ok = ok && update_little_endian(&hasher, source.tensor_identity);
    ok = ok && update_digest(&hasher, source.weight_digest);
    ok = ok && update_digest(&hasher, source.scale_digest);
    ok = ok && update_little_endian(&hasher, source.global_scale_bits);
  }
  if (!ok) {
    *digest = {};
    return false;
  }
  digest->bytes = hasher.finalize().bytes;
  return true;
}

bool seal_sm87_p40_packed_artifact_manifest(
    Sm87P40PackedArtifactManifest* const manifest) noexcept {
  if (manifest == nullptr) {
    return false;
  }
  manifest->manifest_digest = {};
  if (!validate_manifest_impl(*manifest, false).valid() ||
      !compute_sm87_p40_packed_manifest_digest(
          *manifest, &manifest->manifest_digest)) {
    manifest->manifest_digest = {};
    return false;
  }
  return validate_manifest_impl(*manifest, true).valid();
}

Sm87P40PackedManifestValidation
validate_sm87_p40_packed_artifact_manifest(
    const Sm87P40PackedArtifactManifest& manifest) noexcept {
  return validate_manifest_impl(manifest, true);
}

Sm87P40PackedInventoryValidation
validate_sm87_p40_packed_artifact_inventory(
    const Sm87P40PackedArtifactManifest* const manifests,
    const std::size_t manifest_count) noexcept {
  Sm87P40PackedInventoryValidation result;
  result.artifact_count = manifest_count;
  if (manifests == nullptr) {
    flag(&result, Sm87P40PackedInventoryIssue::kNull);
    if (manifest_count != 0U) {
      flag(&result, Sm87P40PackedInventoryIssue::kCount);
    }
    return result;
  }
  if (manifest_count != kSm87P40PackedProjectionArtifactCount) {
    flag(&result, Sm87P40PackedInventoryIssue::kCount);
  }

  constexpr std::size_t kRoleCount =
      static_cast<std::size_t>(Sm87P40PackedProjectionRole::kCount);
  std::array<std::array<bool, kSm87P40PackedProjectionLayerCount>,
             kRoleCount>
      coverage{};
  std::array<std::uint64_t,
             kSm87P40PackedProjectionSourceIdentityCount>
      source_identities{};
  std::size_t source_identity_count = 0U;
  const Sm87P40PackedDigest common_model =
      manifest_count == 0U ? Sm87P40PackedDigest{}
                           : manifests[0U].model_digest;
  const Sm87P40PackedDigest common_checkpoint =
      manifest_count == 0U ? Sm87P40PackedDigest{}
                           : manifests[0U].checkpoint_digest;

  for (std::size_t index = 0U; index < manifest_count; ++index) {
    const auto& manifest = manifests[index];
    if (!validate_sm87_p40_packed_artifact_manifest(manifest).valid()) {
      flag(&result, Sm87P40PackedInventoryIssue::kManifest);
    }
    if (manifest.model_digest != common_model) {
      flag(&result, Sm87P40PackedInventoryIssue::kModelMismatch);
    }
    if (manifest.checkpoint_digest != common_checkpoint) {
      flag(&result, Sm87P40PackedInventoryIssue::kCheckpointMismatch);
    }
    for (std::size_t prior = 0U; prior < index; ++prior) {
      if (manifest.artifact_identity != 0U &&
          manifest.artifact_identity ==
              manifests[prior].artifact_identity) {
        flag(&result, Sm87P40PackedInventoryIssue::kDuplicateArtifact);
      }
    }

    const std::size_t role_index = static_cast<std::size_t>(manifest.role);
    if (role_index < kRoleCount &&
        manifest.layer_index < kSm87P40PackedProjectionLayerCount) {
      if (coverage[role_index][manifest.layer_index]) {
        flag(&result, Sm87P40PackedInventoryIssue::kDuplicateRoleLayer);
      }
      coverage[role_index][manifest.layer_index] = true;
      if (!role_matches_layer(manifest.role, manifest.layer_index)) {
        flag(&result, Sm87P40PackedInventoryIssue::kWrongLayerType);
      }
    }

    if (manifest.role == Sm87P40PackedProjectionRole::kNvFp4GateUp) {
      ++result.gate_up_artifacts;
    } else if (manifest.role ==
               Sm87P40PackedProjectionRole::kNvFp4Down) {
      ++result.down_artifacts;
    } else if (manifest.role ==
                   Sm87P40PackedProjectionRole::kFp8LinearQkvZ ||
               manifest.role ==
                   Sm87P40PackedProjectionRole::kFp8FullQkv ||
               manifest.role ==
                   Sm87P40PackedProjectionRole::kFp8AttentionOutput) {
      ++result.fp8_artifacts;
      result.fp8_logical_roles += manifest.source_count;
    }

    const std::size_t bounded_sources = std::min<std::size_t>(
        manifest.source_count, manifest.sources.size());
    result.source_identities += bounded_sources;
    for (std::size_t source = 0U; source < bounded_sources; ++source) {
      const std::uint64_t identity =
          manifest.sources[source].tensor_identity;
      if (identity != 0U) {
        for (std::size_t prior = 0U; prior < source_identity_count;
             ++prior) {
          if (source_identities[prior] == identity) {
            flag(&result, Sm87P40PackedInventoryIssue::kDuplicateSource);
          }
        }
      }
      if (source_identity_count < source_identities.size()) {
        source_identities[source_identity_count++] = identity;
      } else {
        flag(&result,
             Sm87P40PackedInventoryIssue::kSourceIdentityCount);
      }
    }
  }

  for (std::size_t layer = 0U;
       layer < kSm87P40PackedProjectionLayerCount; ++layer) {
    const auto require = [&](const Sm87P40PackedProjectionRole role) {
      if (!coverage[static_cast<std::size_t>(role)][layer]) {
        flag(&result, Sm87P40PackedInventoryIssue::kMissingRoleLayer);
      }
    };
    require(Sm87P40PackedProjectionRole::kNvFp4GateUp);
    require(Sm87P40PackedProjectionRole::kNvFp4Down);
    require(Sm87P40PackedProjectionRole::kFp8AttentionOutput);
    require(sm87_p40_packed_is_full_layer(layer)
                ? Sm87P40PackedProjectionRole::kFp8FullQkv
                : Sm87P40PackedProjectionRole::kFp8LinearQkvZ);
  }
  if (result.fp8_logical_roles !=
      kSm87P40PackedProjectionFp8LogicalRoleCount) {
    flag(&result, Sm87P40PackedInventoryIssue::kFp8LogicalCount);
  }
  if (result.source_identities !=
      kSm87P40PackedProjectionSourceIdentityCount) {
    flag(&result, Sm87P40PackedInventoryIssue::kSourceIdentityCount);
  }
  return result;
}

}  // namespace q3x::kernels
