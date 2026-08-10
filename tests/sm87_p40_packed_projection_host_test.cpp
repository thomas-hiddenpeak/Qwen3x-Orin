#include "q3x/kernels/sm87_p40_packed_projection.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <type_traits>
#include <vector>

namespace {

using q3x::kernels::Sm87P40PackedArtifactManifest;
using q3x::kernels::Sm87P40PackedDigest;
using q3x::kernels::Sm87P40PackedInventoryIssue;
using q3x::kernels::Sm87P40PackedLogicalRole;
using q3x::kernels::Sm87P40PackedManifestIssue;
using q3x::kernels::Sm87P40PackedProjectionPlan;
using q3x::kernels::Sm87P40PackedProjectionRole;
using q3x::kernels::Sm87P40PackedTactic;
using q3x::kernels::kSm87P40PackedProjectionArtifactCount;
using q3x::kernels::kSm87P40PackedProjectionFp8LogicalRoleCount;
using q3x::kernels::kSm87P40PackedProjectionLayerCount;
using q3x::kernels::kSm87P40PackedProjectionSourceIdentityCount;
using q3x::kernels::make_sm87_p40_packed_artifact_manifest;
using q3x::kernels::seal_sm87_p40_packed_artifact_manifest;
using q3x::kernels::sm87_p40_packed_fragment;
using q3x::kernels::sm87_p40_packed_is_full_layer;
using q3x::kernels::sm87_p40_packed_layout_cell;
using q3x::kernels::sm87_p40_packed_projection_plan;
using q3x::kernels::sm87_p40_packed_task;
using q3x::kernels::validate_sm87_p40_packed_artifact_inventory;
using q3x::kernels::validate_sm87_p40_packed_artifact_manifest;

constexpr auto kGate = sm87_p40_packed_projection_plan(
    Sm87P40PackedProjectionRole::kNvFp4GateUp);
constexpr auto kDown = sm87_p40_packed_projection_plan(
    Sm87P40PackedProjectionRole::kNvFp4Down);
constexpr auto kLinear = sm87_p40_packed_projection_plan(
    Sm87P40PackedProjectionRole::kFp8LinearQkvZ);
constexpr auto kFull = sm87_p40_packed_projection_plan(
    Sm87P40PackedProjectionRole::kFp8FullQkv);
constexpr auto kOutput = sm87_p40_packed_projection_plan(
    Sm87P40PackedProjectionRole::kFp8AttentionOutput);

static_assert(kGate.valid() && kDown.valid() && kLinear.valid() &&
              kFull.valid() && kOutput.valid());
static_assert(kGate.token_count == 40'000U && kGate.tile_m == 64U &&
              kGate.grid_m == 625U);
static_assert(kGate.grid_n == 136U && kGate.logical_tasks == 85'000U &&
              kGate.source_count == 2U &&
              kGate.payload_bytes == 100'270'080U);
static_assert(kGate.partitions[0U].warps == 4U &&
              kGate.partitions[0U].fragment_weight_bytes == 256U &&
              kGate.partitions[0U].fragment_scale_bytes == 32U &&
              kGate.partitions[0U].cell_bytes == 4'608U);
static_assert(kDown.grid_n == 40U && kDown.logical_tasks == 25'000U &&
              kDown.payload_bytes == 50'135'040U);
static_assert(kLinear.grid_n == 128U &&
              kLinear.logical_tasks == 80'000U &&
              kLinear.payload_bytes == 83'886'080U);
static_assert(kFull.grid_n == 128U && kFull.logical_tasks == 80'000U &&
              kFull.payload_bytes == 73'400'320U);
static_assert(kFull.tactic ==
                  Sm87P40PackedTactic::kFp8FullQkvMixedPersistent &&
              kFull.partitions[0U].tactic ==
                  Sm87P40PackedTactic::kFp8WideM64N128K64Gm4 &&
              kFull.partitions[1U].tactic ==
                  Sm87P40PackedTactic::kFp8SmallKvM64N64K128Gm4 &&
              kFull.partitions[2U].tactic ==
                  Sm87P40PackedTactic::kFp8SmallKvM64N64K128Gm4);
static_assert(kOutput.grid_n == 40U &&
              kOutput.logical_tasks == 25'000U &&
              kOutput.payload_bytes == 31'457'280U);
static_assert(kSm87P40PackedProjectionArtifactCount == 256U &&
              kSm87P40PackedProjectionSourceIdentityCount == 400U &&
              kSm87P40PackedProjectionFp8LogicalRoleCount == 208U);
static_assert(!sm87_p40_packed_is_full_layer(0U) &&
              sm87_p40_packed_is_full_layer(3U) &&
              sm87_p40_packed_is_full_layer(63U));
static_assert(!sm87_p40_packed_projection_plan(
                   Sm87P40PackedProjectionRole::kInvalid)
                   .valid());
static_assert(sm87_p40_packed_projection_plan(
                  Sm87P40PackedProjectionRole::kNvFp4GateUp, 40'000U)
                  .valid() &&
              !sm87_p40_packed_projection_plan(
                   Sm87P40PackedProjectionRole::kNvFp4GateUp, 39'999U)
                   .valid() &&
              !sm87_p40_packed_projection_plan(
                   Sm87P40PackedProjectionRole::kNvFp4GateUp, 40'001U)
                   .valid() &&
              !sm87_p40_packed_projection_plan(
                   Sm87P40PackedProjectionRole::kNvFp4GateUp, 60'000U)
                   .valid());

using Prepare = int (*)(
    Sm87P40PackedProjectionRole,
    const q3x::kernels::Sm87P40PackedCanonicalSource*, std::size_t,
    std::uint8_t*, std::size_t, void*) noexcept;
using ResourceQuery = int (*)(
    Sm87P40PackedProjectionRole,
    q3x::kernels::Sm87P40PackedProjectionResources*) noexcept;
using GateLauncher = int (*)(
    const std::uint16_t*,
    const q3x::kernels::Sm87P40PackedProjectionDeviceView&, std::size_t,
    std::uint16_t*, void*) noexcept;
using Fp8Launcher = int (*)(
    const std::uint16_t*,
    const q3x::kernels::Sm87P40PackedProjectionDeviceView&, std::size_t,
    const std::array<std::uint16_t*,
                     q3x::kernels::kSm87P40PackedProjectionMaximumSources>&,
    void*) noexcept;

static_assert(std::is_same_v<
              decltype(&q3x::kernels::prepare_sm87_p40_packed_projection_cuda),
              Prepare>);
static_assert(std::is_same_v<
              decltype(&q3x::kernels::
                           query_sm87_p40_packed_projection_resources_cuda),
              ResourceQuery>);
static_assert(std::is_same_v<
              decltype(&q3x::kernels::
                           launch_sm87_p40_packed_nvfp4_gate_up_cuda),
              GateLauncher>);
static_assert(std::is_same_v<
              decltype(&q3x::kernels::launch_sm87_p40_packed_fp8_cuda),
              Fp8Launcher>);

[[nodiscard]] bool check(const bool condition, const char* const message) {
  if (!condition) {
    std::cerr << message << '\n';
  }
  return condition;
}

template <typename Issue>
[[nodiscard]] bool has_issue(const std::uint64_t issue_mask,
                             const Issue issue) noexcept {
  return (issue_mask & static_cast<std::uint64_t>(issue)) != 0U;
}

[[nodiscard]] Sm87P40PackedDigest make_digest(std::uint64_t seed) noexcept {
  Sm87P40PackedDigest result;
  for (std::size_t index = 0U; index < result.bytes.size(); ++index) {
    seed ^= seed << 13U;
    seed ^= seed >> 7U;
    seed ^= seed << 17U;
    result.bytes[index] = static_cast<std::uint8_t>(seed >> 11U);
  }
  if (q3x::kernels::sm87_p40_packed_digest_is_zero(result)) {
    result.bytes[0U] = 1U;
  }
  return result;
}

struct InventoryFixture {
  std::vector<Sm87P40PackedArtifactManifest> manifests;
  std::size_t source_count = 0U;
  std::size_t nvfp4_sources = 0U;
  std::size_t fp8_sources = 0U;
};

[[nodiscard]] bool append_manifest(
    InventoryFixture* const fixture, const std::size_t layer,
    const Sm87P40PackedProjectionRole role,
    std::uint64_t* const artifact_identity,
    std::uint64_t* const source_identity) {
  auto manifest = make_sm87_p40_packed_artifact_manifest(role, layer);
  if (manifest.role != role) {
    return false;
  }
  manifest.artifact_identity = (*artifact_identity)++;
  manifest.model_digest = make_digest(0x1001U);
  manifest.checkpoint_digest = make_digest(0x2001U);
  manifest.payload_digest = make_digest(0x3000U + manifest.artifact_identity);
  const bool fp8 = role == Sm87P40PackedProjectionRole::kFp8LinearQkvZ ||
                   role == Sm87P40PackedProjectionRole::kFp8FullQkv ||
                   role == Sm87P40PackedProjectionRole::kFp8AttentionOutput;
  for (std::size_t index = 0U; index < manifest.source_count; ++index) {
    auto& source = manifest.sources[index];
    source.tensor_identity = (*source_identity)++;
    source.weight_digest = make_digest(0x4000U + source.tensor_identity);
    source.scale_digest = make_digest(0x8000U + source.tensor_identity);
    source.global_scale_bits =
        0x3f000000U + static_cast<std::uint32_t>(source.tensor_identity);
    ++fixture->source_count;
    if (fp8) {
      ++fixture->fp8_sources;
    } else {
      ++fixture->nvfp4_sources;
    }
  }
  if (!seal_sm87_p40_packed_artifact_manifest(&manifest)) {
    return false;
  }
  fixture->manifests.push_back(manifest);
  return true;
}

[[nodiscard]] InventoryFixture make_inventory() {
  InventoryFixture fixture;
  fixture.manifests.reserve(kSm87P40PackedProjectionArtifactCount);
  std::uint64_t artifact_identity = 1U;
  std::uint64_t source_identity = 1U;
  for (std::size_t layer = 0U;
       layer < kSm87P40PackedProjectionLayerCount; ++layer) {
    if (!append_manifest(&fixture, layer,
                         Sm87P40PackedProjectionRole::kNvFp4GateUp,
                         &artifact_identity, &source_identity) ||
        !append_manifest(&fixture, layer,
                         Sm87P40PackedProjectionRole::kNvFp4Down,
                         &artifact_identity, &source_identity) ||
        !append_manifest(
            &fixture, layer,
            sm87_p40_packed_is_full_layer(layer)
                ? Sm87P40PackedProjectionRole::kFp8FullQkv
                : Sm87P40PackedProjectionRole::kFp8LinearQkvZ,
            &artifact_identity, &source_identity) ||
        !append_manifest(&fixture, layer,
                         Sm87P40PackedProjectionRole::kFp8AttentionOutput,
                         &artifact_identity, &source_identity)) {
      fixture.manifests.clear();
      return fixture;
    }
  }
  return fixture;
}

[[nodiscard]] bool exhaustive_task_bijection(
    const Sm87P40PackedProjectionPlan& plan) {
  std::vector<std::uint8_t> visited(plan.logical_tasks, 0U);
  for (std::uint64_t linear = 0U; linear < plan.logical_tasks; ++linear) {
    const auto task = sm87_p40_packed_task(plan, linear);
    if (!task.valid || task.m_tile >= plan.grid_m ||
        task.n_tile >= plan.grid_n) {
      return false;
    }
    const std::uint64_t canonical =
        static_cast<std::uint64_t>(task.m_tile) * plan.grid_n + task.n_tile;
    if (canonical >= visited.size() || visited[canonical] != 0U) {
      return false;
    }
    visited[canonical] = 1U;
    if (plan.role == Sm87P40PackedProjectionRole::kNvFp4GateUp) {
      if (task.partition_index != 0xffU ||
          task.source_partition_mask != 0x03U ||
          task.local_n_tile != task.n_tile) {
        return false;
      }
    } else {
      if (task.partition_index >= plan.source_count ||
          task.source_partition_mask !=
              static_cast<std::uint8_t>(1U << task.partition_index)) {
        return false;
      }
      const auto& partition = plan.partitions[task.partition_index];
      if (task.n_tile != partition.first_task_n_tile +
                             task.local_n_tile) {
        return false;
      }
    }
  }
  if (std::find(visited.begin(), visited.end(), 0U) != visited.end() ||
      sm87_p40_packed_task(plan, plan.logical_tasks).valid) {
    return false;
  }

  std::fill(visited.begin(), visited.end(), 0U);
  for (std::uint64_t cta = 0U;
       cta < q3x::kernels::kSm87P40PackedProjectionPersistentCtas; ++cta) {
    for (std::uint64_t linear = cta; linear < plan.logical_tasks;
         linear += q3x::kernels::kSm87P40PackedProjectionPersistentCtas) {
      ++visited[linear];
    }
  }
  return std::find_if(visited.begin(), visited.end(),
                      [](const std::uint8_t visits) {
                        return visits != 1U;
                      }) == visited.end();
}

[[nodiscard]] bool exhaustive_layout_bijection(
    const Sm87P40PackedProjectionPlan& plan) {
  std::uint64_t observed_bytes = 0U;
  if (plan.role == Sm87P40PackedProjectionRole::kNvFp4GateUp) {
    const auto& branch = plan.partitions[0U];
    for (std::uint32_t n_tile = 0U; n_tile < plan.grid_n; ++n_tile) {
      for (std::uint32_t k_tile = 0U; k_tile < branch.k_tiles; ++k_tile) {
        const auto cell = sm87_p40_packed_layout_cell(plan, n_tile, k_tile);
        const std::uint64_t expected =
            (static_cast<std::uint64_t>(n_tile) * branch.k_tiles + k_tile) *
            2U * branch.cell_bytes;
        if (!cell.valid || cell.payload_offset != expected ||
            cell.cell_bytes != 2U * branch.cell_bytes ||
            cell.source_partition_mask != 0x03U) {
          return false;
        }
        observed_bytes += cell.cell_bytes;
      }
    }
    return observed_bytes == plan.payload_bytes &&
           !sm87_p40_packed_layout_cell(plan, plan.grid_n, 0U).valid &&
           !sm87_p40_packed_layout_cell(plan, 0U, branch.k_tiles).valid;
  }

  for (std::uint32_t partition_index = 0U;
       partition_index < plan.source_count; ++partition_index) {
    const auto& partition = plan.partitions[partition_index];
    std::uint64_t partition_bytes = 0U;
    for (std::uint32_t local_n = 0U;
         local_n < partition.task_n_tiles; ++local_n) {
      const std::uint32_t n_tile = partition.first_task_n_tile + local_n;
      for (std::uint32_t k_tile = 0U; k_tile < partition.k_tiles; ++k_tile) {
        const auto cell = sm87_p40_packed_layout_cell(plan, n_tile, k_tile);
        const std::uint64_t expected =
            partition.payload_offset +
            (static_cast<std::uint64_t>(local_n) * partition.k_tiles +
             k_tile) *
                partition.cell_bytes;
        if (!cell.valid || cell.partition_index != partition_index ||
            cell.payload_offset != expected ||
            cell.cell_bytes != partition.cell_bytes) {
          return false;
        }
        partition_bytes += cell.cell_bytes;
      }
    }
    if (partition_bytes != partition.payload_bytes) {
      return false;
    }
    observed_bytes += partition_bytes;
  }
  return observed_bytes == plan.payload_bytes &&
         !sm87_p40_packed_layout_cell(plan, plan.grid_n, 0U).valid;
}

[[nodiscard]] bool exhaustive_fragment_bijection(
    const Sm87P40PackedProjectionPlan& plan) {
  for (std::uint32_t n_tile = 0U; n_tile < plan.grid_n; ++n_tile) {
    std::uint8_t partition_index = 0U;
    if (plan.role != Sm87P40PackedProjectionRole::kNvFp4GateUp) {
      while (partition_index < plan.source_count &&
             n_tile >= plan.partitions[partition_index].first_task_n_tile +
                           plan.partitions[partition_index].task_n_tiles) {
        ++partition_index;
      }
      if (partition_index >= plan.source_count) {
        return false;
      }
    }
    const auto& partition = plan.partitions[partition_index];
    const std::uint32_t physical_warps =
        plan.role == Sm87P40PackedProjectionRole::kNvFp4GateUp
            ? 8U
            : partition.warps;
    const std::uint32_t total_k16 = partition.input_features / 16U;
    const std::uint32_t k16_per_tile = partition.tile_k / 16U;
    for (std::uint32_t global_k16 = 0U; global_k16 < total_k16;
         ++global_k16) {
      const std::uint32_t k_tile = global_k16 / k16_per_tile;
      const std::uint32_t local_k16 = global_k16 % k16_per_tile;
      const auto cell = sm87_p40_packed_layout_cell(plan, n_tile, k_tile);
      for (std::uint32_t warp = 0U; warp < physical_warps; ++warp) {
        const auto fragment =
            sm87_p40_packed_fragment(plan, n_tile, global_k16, warp);
        const auto& fragment_partition =
            plan.partitions[fragment.partition_index];
        const std::uint64_t expected =
            cell.payload_offset +
            (static_cast<std::uint64_t>(local_k16) * physical_warps + warp) *
                (fragment_partition.fragment_weight_bytes +
                 fragment_partition.fragment_scale_bytes);
        if (!fragment.valid || fragment.weight_offset != expected ||
            fragment.weight_bytes !=
                fragment_partition.fragment_weight_bytes ||
            fragment.scale_offset != expected + fragment.weight_bytes ||
            fragment.scale_bytes !=
                fragment_partition.fragment_scale_bytes ||
            fragment.scale_offset + fragment.scale_bytes >
                cell.payload_offset + cell.cell_bytes) {
          return false;
        }
      }
    }
    if (sm87_p40_packed_fragment(plan, n_tile, total_k16, 0U).valid ||
        sm87_p40_packed_fragment(plan, n_tile, 0U, physical_warps).valid) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool exhaustive_inventory_paths(
    const InventoryFixture& fixture) {
  std::array<std::size_t, 5U> artifact_roles{};
  std::size_t fp8_receipts = 0U;
  for (const auto& manifest : fixture.manifests) {
    if (!validate_sm87_p40_packed_artifact_manifest(manifest).valid()) {
      return false;
    }
    const auto plan = sm87_p40_packed_projection_plan(manifest.role);
    if (!exhaustive_task_bijection(plan) ||
        !exhaustive_layout_bijection(plan)) {
      return false;
    }
    ++artifact_roles[static_cast<std::size_t>(manifest.role)];
    if (manifest.role == Sm87P40PackedProjectionRole::kFp8LinearQkvZ ||
        manifest.role == Sm87P40PackedProjectionRole::kFp8FullQkv ||
        manifest.role ==
            Sm87P40PackedProjectionRole::kFp8AttentionOutput) {
      fp8_receipts += manifest.source_count;
    }
  }
  return artifact_roles[static_cast<std::size_t>(
             Sm87P40PackedProjectionRole::kNvFp4GateUp)] == 64U &&
         artifact_roles[static_cast<std::size_t>(
             Sm87P40PackedProjectionRole::kNvFp4Down)] == 64U &&
         artifact_roles[static_cast<std::size_t>(
             Sm87P40PackedProjectionRole::kFp8LinearQkvZ)] == 48U &&
         artifact_roles[static_cast<std::size_t>(
             Sm87P40PackedProjectionRole::kFp8FullQkv)] == 16U &&
         artifact_roles[static_cast<std::size_t>(
             Sm87P40PackedProjectionRole::kFp8AttentionOutput)] == 64U &&
         fp8_receipts == kSm87P40PackedProjectionFp8LogicalRoleCount;
}

[[nodiscard]] bool manifest_fail_closed_tests(
    const InventoryFixture& fixture) {
  bool ok = true;
  const auto expect_issue = [&](Sm87P40PackedArtifactManifest manifest,
                                const Sm87P40PackedManifestIssue issue,
                                const char* const message) {
    const auto validation =
        validate_sm87_p40_packed_artifact_manifest(manifest);
    ok &= check(!validation.valid() &&
                    has_issue(validation.issue_mask, issue),
                message);
  };

  auto mutation = fixture.manifests[0U];
  mutation.magic[0U] ^= 1U;
  expect_issue(mutation, Sm87P40PackedManifestIssue::kMagic,
               "bad magic must fail closed");
  mutation = fixture.manifests[0U];
  ++mutation.abi_minor;
  expect_issue(mutation, Sm87P40PackedManifestIssue::kVersion,
               "unknown ABI minor must fail closed");
  mutation = fixture.manifests[0U];
  mutation.header_bytes = 0U;
  expect_issue(mutation, Sm87P40PackedManifestIssue::kHeader,
               "bad header extent must fail closed");
  mutation = fixture.manifests[0U];
  mutation.artifact_identity = 0U;
  expect_issue(mutation, Sm87P40PackedManifestIssue::kArtifactIdentity,
               "zero artifact identity must fail closed");
  mutation = fixture.manifests[2U];
  mutation.layer_index = 3U;
  expect_issue(mutation, Sm87P40PackedManifestIssue::kRole,
               "linear artifact on a full layer must fail closed");
  mutation = fixture.manifests[0U];
  mutation.tactic = Sm87P40PackedTactic::kInvalid;
  expect_issue(mutation, Sm87P40PackedManifestIssue::kTactic,
               "unknown tactic must fail closed");
  mutation = fixture.manifests[0U];
  mutation.policy &= ~q3x::kernels::kSm87P40PackedFullKAccumulation;
  expect_issue(mutation, Sm87P40PackedManifestIssue::kPolicy,
               "weakened exact policy must fail closed");
  mutation = fixture.manifests[0U];
  mutation.sources[1U].role = Sm87P40PackedLogicalRole::kNvFp4Gate;
  expect_issue(mutation, Sm87P40PackedManifestIssue::kSourceRole,
               "Gate and Up role identities must remain independent");
  mutation = fixture.manifests[0U];
  mutation.sources[1U].tensor_identity =
      mutation.sources[0U].tensor_identity;
  expect_issue(mutation, Sm87P40PackedManifestIssue::kSourceIdentity,
               "Gate and Up tensor identities must remain independent");
  mutation = fixture.manifests[0U];
  mutation.sources[0U].weight_digest = {};
  expect_issue(mutation, Sm87P40PackedManifestIssue::kSourceDigest,
               "zero source digest must fail closed");
  mutation = fixture.manifests[0U];
  mutation.sources[0U].global_scale_bits = 0x7fc00000U;
  expect_issue(mutation, Sm87P40PackedManifestIssue::kScale,
               "NaN global scale must fail closed");
  mutation = fixture.manifests[1U];
  mutation.sources[1U].role = Sm87P40PackedLogicalRole::kNvFp4Down;
  expect_issue(mutation, Sm87P40PackedManifestIssue::kSourceRole,
               "noncanonical unused source must fail closed");
  mutation = fixture.manifests[0U];
  ++mutation.payload_offset;
  expect_issue(mutation, Sm87P40PackedManifestIssue::kPayloadRange,
               "misaligned payload must fail closed");
  mutation = fixture.manifests[0U];
  mutation.payload_offset = std::numeric_limits<std::uint64_t>::max();
  expect_issue(mutation, Sm87P40PackedManifestIssue::kPayloadRange,
               "overflowing artifact extent must fail closed");
  mutation = fixture.manifests[0U];
  mutation.payload_digest = {};
  expect_issue(mutation, Sm87P40PackedManifestIssue::kPayloadDigest,
               "zero payload digest must fail closed");
  mutation = fixture.manifests[0U];
  mutation.model_digest = {};
  expect_issue(mutation, Sm87P40PackedManifestIssue::kModelDigest,
               "zero model identity must fail closed");
  mutation = fixture.manifests[0U];
  mutation.payload_digest.bytes[0U] ^= 1U;
  expect_issue(mutation, Sm87P40PackedManifestIssue::kManifestDigest,
               "unsealed mutation must fail authenticated hash");

  Sm87P40PackedDigest first;
  Sm87P40PackedDigest second;
  ok &= check(q3x::kernels::compute_sm87_p40_packed_manifest_digest(
                  fixture.manifests[0U], &first) &&
                  q3x::kernels::compute_sm87_p40_packed_manifest_digest(
                      fixture.manifests[0U], &second) &&
                  first == second,
              "canonical manifest hash must be deterministic");
  mutation = fixture.manifests[0U];
  ++mutation.artifact_identity;
  ok &= check(q3x::kernels::compute_sm87_p40_packed_manifest_digest(
                  mutation, &second) &&
                  first != second,
              "authenticated field mutation must change manifest hash");
  ok &= check(!q3x::kernels::compute_sm87_p40_packed_manifest_digest(
                  fixture.manifests[0U], nullptr) &&
                  !seal_sm87_p40_packed_artifact_manifest(nullptr),
              "null hash/seal outputs must fail closed");
  return ok;
}

[[nodiscard]] bool inventory_fail_closed_tests(
    const InventoryFixture& fixture) {
  bool ok = true;
  const auto valid = validate_sm87_p40_packed_artifact_inventory(
      fixture.manifests.data(), fixture.manifests.size());
  ok &= check(valid.valid() && valid.artifact_count == 256U &&
                  valid.gate_up_artifacts == 64U &&
                  valid.down_artifacts == 64U &&
                  valid.fp8_artifacts == 128U &&
                  valid.fp8_logical_roles == 208U &&
                  valid.source_identities == 400U,
              "complete 256-artifact/400-source inventory must validate");

  auto mutation = fixture.manifests;
  mutation[1U].artifact_identity = mutation[0U].artifact_identity;
  ok &= check(seal_sm87_p40_packed_artifact_manifest(&mutation[1U]),
              "duplicate artifact fixture must remain locally valid");
  auto result = validate_sm87_p40_packed_artifact_inventory(
      mutation.data(), mutation.size());
  ok &= check(has_issue(result.issue_mask,
                        Sm87P40PackedInventoryIssue::kDuplicateArtifact),
              "inventory must reject duplicate artifact identity");

  mutation = fixture.manifests;
  mutation[1U].sources[0U].tensor_identity =
      mutation[0U].sources[0U].tensor_identity;
  ok &= check(seal_sm87_p40_packed_artifact_manifest(&mutation[1U]),
              "cross-artifact source duplicate must remain locally valid");
  result = validate_sm87_p40_packed_artifact_inventory(mutation.data(),
                                                        mutation.size());
  ok &= check(has_issue(result.issue_mask,
                        Sm87P40PackedInventoryIssue::kDuplicateSource),
              "inventory must reject duplicate source identity");

  mutation = fixture.manifests;
  mutation[1U] = mutation[0U];
  result = validate_sm87_p40_packed_artifact_inventory(mutation.data(),
                                                        mutation.size());
  ok &= check(has_issue(result.issue_mask,
                        Sm87P40PackedInventoryIssue::kDuplicateRoleLayer),
              "inventory must reject duplicate role/layer ownership");

  result = validate_sm87_p40_packed_artifact_inventory(
      fixture.manifests.data(), fixture.manifests.size() - 1U);
  ok &= check(has_issue(result.issue_mask,
                        Sm87P40PackedInventoryIssue::kCount) &&
                  has_issue(result.issue_mask,
                            Sm87P40PackedInventoryIssue::kMissingRoleLayer),
              "truncated inventory must fail count and coverage");

  mutation = fixture.manifests;
  mutation[1U].model_digest = make_digest(0xdeadU);
  ok &= check(seal_sm87_p40_packed_artifact_manifest(&mutation[1U]),
              "model mismatch fixture must remain locally valid");
  result = validate_sm87_p40_packed_artifact_inventory(mutation.data(),
                                                        mutation.size());
  ok &= check(has_issue(result.issue_mask,
                        Sm87P40PackedInventoryIssue::kModelMismatch),
              "inventory must reject mixed model identities");

  mutation = fixture.manifests;
  mutation[1U].checkpoint_digest = make_digest(0xbeefU);
  ok &= check(seal_sm87_p40_packed_artifact_manifest(&mutation[1U]),
              "checkpoint mismatch fixture must remain locally valid");
  result = validate_sm87_p40_packed_artifact_inventory(mutation.data(),
                                                        mutation.size());
  ok &= check(has_issue(result.issue_mask,
                        Sm87P40PackedInventoryIssue::kCheckpointMismatch),
              "inventory must reject mixed checkpoint identities");

  mutation = fixture.manifests;
  mutation[2U].layer_index = 3U;
  result = validate_sm87_p40_packed_artifact_inventory(mutation.data(),
                                                        mutation.size());
  ok &= check(has_issue(result.issue_mask,
                        Sm87P40PackedInventoryIssue::kWrongLayerType),
              "inventory must reject linear/full layer type mismatch");

  result = validate_sm87_p40_packed_artifact_inventory(nullptr, 1U);
  ok &= check(has_issue(result.issue_mask,
                        Sm87P40PackedInventoryIssue::kNull) &&
                  has_issue(result.issue_mask,
                            Sm87P40PackedInventoryIssue::kCount),
              "null nonempty inventory must fail closed");
  return ok;
}

}  // namespace

int main() {
  bool ok = true;
  const InventoryFixture fixture = make_inventory();
  ok &= check(fixture.manifests.size() ==
                  kSm87P40PackedProjectionArtifactCount &&
                  fixture.source_count ==
                      kSm87P40PackedProjectionSourceIdentityCount &&
                  fixture.nvfp4_sources == 192U &&
                  fixture.fp8_sources == 208U,
              "fixture must contain 256 physical artifacts and 400 sources");
  if (!ok) {
    return 1;
  }
  ok &= check(exhaustive_inventory_paths(fixture),
              "all 64 Gate+Up, 64 Down, and 208 FP8 receipts must have "
              "bijective P40 task/layout ownership");
  ok &= check(exhaustive_fragment_bijection(kGate) &&
                  exhaustive_fragment_bijection(kDown) &&
                  exhaustive_fragment_bijection(kLinear) &&
                  exhaustive_fragment_bijection(kFull) &&
                  exhaustive_fragment_bijection(kOutput),
              "all consumer fragments must bijectively cover packed cells");
  ok &= manifest_fail_closed_tests(fixture);
  ok &= inventory_fail_closed_tests(fixture);
  return ok ? 0 : 1;
}
