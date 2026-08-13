#pragma once

#if !defined(Q3X_ENABLE_P40_PROJECTION_WORK_MANIFEST_DIAGNOSTIC)
#error "The P40 projection work manifest is a default-off test diagnostic"
#endif

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

namespace q3x::tests::p40_work_manifest {

inline constexpr std::uint32_t kPromptTokens = 40'000U;
inline constexpr std::uint32_t kPanelTokens = 8'000U;
inline constexpr std::size_t kPanelCount = 5U;
inline constexpr std::size_t kLayerCount = 64U;
inline constexpr std::size_t kLinearLayerCount = 48U;
inline constexpr std::size_t kFullAttentionLayerCount = 16U;

inline constexpr std::size_t kFp8LogicalRoleCount = 208U;
inline constexpr std::size_t kNvFp4LogicalRoleCount = 192U;
inline constexpr std::size_t kBf16LogicalRoleCount = 96U;
inline constexpr std::size_t kLogicalRoleCount =
    kFp8LogicalRoleCount + kNvFp4LogicalRoleCount + kBf16LogicalRoleCount;

inline constexpr std::size_t kFp8OuterOperationCount = 128U;
inline constexpr std::size_t kNvFp4OuterOperationCount = 128U;
inline constexpr std::size_t kBf16OuterOperationCount = 48U;
inline constexpr std::size_t kOuterOperationCount =
    kFp8OuterOperationCount + kNvFp4OuterOperationCount +
    kBf16OuterOperationCount;

inline constexpr std::size_t kFp8PhysicalLaunchCount = 1'040U;
inline constexpr std::size_t kNvFp4PhysicalLaunchCount = 128U;
inline constexpr std::size_t kBf16PhysicalLaunchCount = 48U;
inline constexpr std::size_t kPhysicalLaunchCount =
    kFp8PhysicalLaunchCount + kNvFp4PhysicalLaunchCount +
    kBf16PhysicalLaunchCount;

inline constexpr std::size_t kQuantizedSourceScalePartitionCount = 400U;
inline constexpr std::size_t kQuantizedPhysicalArtifactCount = 336U;
inline constexpr std::uint16_t kNoScaleIdentity =
    std::numeric_limits<std::uint16_t>::max();
inline constexpr std::uint8_t kWholePromptPanelOrdinal =
    std::numeric_limits<std::uint8_t>::max();

inline constexpr std::uint64_t kFp8ConventionalOperations =
    577'136'230'400'000ULL;
inline constexpr std::uint64_t kNvFp4ConventionalOperations =
    1'369'020'825'600'000ULL;
inline constexpr std::uint64_t kBf16ConventionalOperations =
    1'887'436'800'000ULL;
inline constexpr std::uint64_t kProjectionConventionalOperations =
    1'948'044'492'800'000ULL;

inline constexpr std::string_view kSchemaIdentity =
    "q3x.p40.projection-work-manifest.v1";
inline constexpr std::string_view kRouteIdentity =
    "q3x.sm87.ac-prefill-prompt-wide-v2.native-p40-whole-core.v1";

enum class ProjectionFamily : std::uint8_t {
  kFp8W8A16 = 0,
  kNvFp4W4A16,
  kBf16,
};

enum class LayerKind : std::uint8_t {
  kLinearAttention = 0,
  kFullAttention,
};

// These are checkpoint/output roles, not CUDA-kernel names. In particular,
// linear QKV is one checkpoint tensor, full Q includes the gate columns, and
// Gate and Up remain independent checkpoint sources even when one physical
// persistent kernel consumes their merged sidecar.
enum class CheckpointSubrole : std::uint8_t {
  kLinearQkv = 0,
  kLinearZ,
  kLinearO,
  kFullQGate,
  kFullK,
  kFullV,
  kFullO,
  kMlpGate,
  kMlpUp,
  kMlpDown,
  kLinearA,
  kLinearB,
};

enum class FusedOuterOperationKind : std::uint8_t {
  kLinearQkvZInput = 0,
  kLinearOutput,
  kFullQkvInput,
  kFullOutput,
  kMlpGateUp,
  kMlpDownResidual,
  kLinearAb,
};

enum class Algorithm : std::uint8_t {
  kFp8MarlinW8A16P8000 = 0,
  kNvFp4PersistentMarlinGateUpP40000,
  kNvFp4PersistentMarlinDownResidualP40000,
  kBf16AbSingleGridP40000,
};

enum class ActivationDtype : std::uint8_t {
  kBf16 = 0,
};

enum class ScalePartitionKind : std::uint8_t {
  kNone = 0,
  // One scalar checkpoint tensor scale is expanded into the physical
  // per-output-channel BF16 Marlin scale vector for this exact source role.
  kFp8TensorGlobalExpandedPerOutputChannel,
  // One independent K16 E4M3FN block-scale partition plus the source role's
  // tensor-global weight_scale_2. Gate and Up keep separate source partitions.
  kNvFp4K16BlockAndTensorGlobal,
};

enum class PromptPartitionKind : std::uint8_t {
  kP8000Panel = 0,
  kP40000WholePrompt,
};

[[nodiscard]] constexpr std::string_view name(
    const ProjectionFamily value) noexcept {
  switch (value) {
    case ProjectionFamily::kFp8W8A16:
      return "fp8-w8a16";
    case ProjectionFamily::kNvFp4W4A16:
      return "nvfp4-w4a16";
    case ProjectionFamily::kBf16:
      return "bf16";
  }
  return "invalid";
}

[[nodiscard]] constexpr std::string_view name(
    const LayerKind value) noexcept {
  switch (value) {
    case LayerKind::kLinearAttention:
      return "linear_attention";
    case LayerKind::kFullAttention:
      return "full_attention";
  }
  return "invalid";
}

[[nodiscard]] constexpr std::string_view name(
    const CheckpointSubrole value) noexcept {
  switch (value) {
    case CheckpointSubrole::kLinearQkv:
      return "linear_attention.in_proj_qkv";
    case CheckpointSubrole::kLinearZ:
      return "linear_attention.in_proj_z";
    case CheckpointSubrole::kLinearO:
      return "linear_attention.out_proj";
    case CheckpointSubrole::kFullQGate:
      return "full_attention.q_proj[q+gate]";
    case CheckpointSubrole::kFullK:
      return "full_attention.k_proj";
    case CheckpointSubrole::kFullV:
      return "full_attention.v_proj";
    case CheckpointSubrole::kFullO:
      return "full_attention.o_proj";
    case CheckpointSubrole::kMlpGate:
      return "mlp.gate_proj";
    case CheckpointSubrole::kMlpUp:
      return "mlp.up_proj";
    case CheckpointSubrole::kMlpDown:
      return "mlp.down_proj";
    case CheckpointSubrole::kLinearA:
      return "linear_attention.in_proj_a";
    case CheckpointSubrole::kLinearB:
      return "linear_attention.in_proj_b";
  }
  return "invalid";
}

[[nodiscard]] constexpr std::string_view name(
    const FusedOuterOperationKind value) noexcept {
  switch (value) {
    case FusedOuterOperationKind::kLinearQkvZInput:
      return "linear-qkv+z-input";
    case FusedOuterOperationKind::kLinearOutput:
      return "linear-output";
    case FusedOuterOperationKind::kFullQkvInput:
      return "full-q+k+v-input";
    case FusedOuterOperationKind::kFullOutput:
      return "full-output";
    case FusedOuterOperationKind::kMlpGateUp:
      return "mlp-gate+up";
    case FusedOuterOperationKind::kMlpDownResidual:
      return "mlp-down+residual";
    case FusedOuterOperationKind::kLinearAb:
      return "linear-a+b";
  }
  return "invalid";
}

[[nodiscard]] constexpr std::string_view name(
    const Algorithm value) noexcept {
  switch (value) {
    case Algorithm::kFp8MarlinW8A16P8000:
      return "q3x-v10-fp8-marlin-w8a16-p8000";
    case Algorithm::kNvFp4PersistentMarlinGateUpP40000:
      return "q3x-v10-nvfp4-persistent-marlin-gate-up-p40000";
    case Algorithm::kNvFp4PersistentMarlinDownResidualP40000:
      return "q3x-v10-nvfp4-persistent-marlin-down-residual-p40000";
    case Algorithm::kBf16AbSingleGridP40000:
      return "q3x-v10-bf16-a-b-single-grid-p40000";
  }
  return "invalid";
}

[[nodiscard]] constexpr std::string_view name(
    const ActivationDtype value) noexcept {
  switch (value) {
    case ActivationDtype::kBf16:
      return "bf16";
  }
  return "invalid";
}

[[nodiscard]] constexpr std::string_view name(
    const ScalePartitionKind value) noexcept {
  switch (value) {
    case ScalePartitionKind::kNone:
      return "none";
    case ScalePartitionKind::kFp8TensorGlobalExpandedPerOutputChannel:
      return "fp8-tensor-global->marlin-output-channel";
    case ScalePartitionKind::kNvFp4K16BlockAndTensorGlobal:
      return "nvfp4-k16-block+tensor-global";
  }
  return "invalid";
}

[[nodiscard]] constexpr std::string_view name(
    const PromptPartitionKind value) noexcept {
  switch (value) {
    case PromptPartitionKind::kP8000Panel:
      return "p8000-panel";
    case PromptPartitionKind::kP40000WholePrompt:
      return "p40000-whole-prompt";
  }
  return "invalid";
}

struct ScalePartitionIdentity {
  ScalePartitionKind kind = ScalePartitionKind::kNone;
  std::uint16_t source_partition = kNoScaleIdentity;
  std::uint16_t physical_artifact = kNoScaleIdentity;
  std::uint32_t physical_output_offset = 0U;
  std::uint32_t output_channels = 0U;
};

struct LogicalProjectionRole {
  std::uint16_t ordinal = 0U;
  std::uint16_t layer = 0U;
  std::uint16_t outer_operation = 0U;
  LayerKind layer_kind = LayerKind::kLinearAttention;
  ProjectionFamily family = ProjectionFamily::kFp8W8A16;
  CheckpointSubrole subrole = CheckpointSubrole::kLinearQkv;
  Algorithm algorithm = Algorithm::kFp8MarlinW8A16P8000;
  ActivationDtype activation_dtype = ActivationDtype::kBf16;
  std::uint32_t m = 0U;
  std::uint32_t n = 0U;
  std::uint32_t k = 0U;
  ScalePartitionIdentity scale{};
  std::uint64_t conventional_operations = 0U;
};

struct FusedOuterOperation {
  std::uint16_t ordinal = 0U;
  std::uint16_t layer = 0U;
  LayerKind layer_kind = LayerKind::kLinearAttention;
  ProjectionFamily family = ProjectionFamily::kFp8W8A16;
  FusedOuterOperationKind kind =
      FusedOuterOperationKind::kLinearQkvZInput;
  Algorithm algorithm = Algorithm::kFp8MarlinW8A16P8000;
  ActivationDtype activation_dtype = ActivationDtype::kBf16;
  std::uint16_t logical_begin = 0U;
  std::uint8_t logical_count = 0U;
  std::uint16_t physical_begin = 0U;
  std::uint8_t physical_count = 0U;
  std::uint64_t conventional_operations = 0U;
};

struct PhysicalCudaLaunch {
  std::uint16_t ordinal = 0U;
  std::uint16_t layer = 0U;
  std::uint16_t outer_operation = 0U;
  LayerKind layer_kind = LayerKind::kLinearAttention;
  ProjectionFamily family = ProjectionFamily::kFp8W8A16;
  Algorithm algorithm = Algorithm::kFp8MarlinW8A16P8000;
  ActivationDtype activation_dtype = ActivationDtype::kBf16;
  PromptPartitionKind prompt_partition = PromptPartitionKind::kP8000Panel;
  std::uint32_t token_offset = 0U;
  std::uint32_t token_count = 0U;
  std::uint8_t panel_ordinal = 0U;
  std::uint8_t logical_member_count = 0U;
  std::array<std::uint16_t, 3U> logical_members{};
};

struct ProjectionWorkManifest {
  std::uint32_t version = 1U;
  bool diagnostic_only = true;
  bool production_reachable = false;
  std::uint32_t prompt_tokens = kPromptTokens;
  std::uint32_t panel_tokens = kPanelTokens;
  std::uint8_t panel_count = static_cast<std::uint8_t>(kPanelCount);
  std::array<LogicalProjectionRole, kLogicalRoleCount> logical_roles{};
  std::array<FusedOuterOperation, kOuterOperationCount> outer_operations{};
  std::array<PhysicalCudaLaunch, kPhysicalLaunchCount> physical_launches{};
};

struct LogicalMemberSpec {
  CheckpointSubrole subrole = CheckpointSubrole::kLinearQkv;
  std::uint32_t n = 0U;
  std::uint32_t k = 0U;
};

[[nodiscard]] constexpr std::uint64_t conventional_operations(
    const std::uint32_t m, const std::uint32_t n,
    const std::uint32_t k) noexcept {
  return 2ULL * static_cast<std::uint64_t>(m) *
         static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(k);
}

[[nodiscard]] constexpr bool is_full_attention_layer(
    const std::size_t layer) noexcept {
  return layer < kLayerCount && ((layer + 1U) % 4U) == 0U;
}

class ManifestBuilder final {
 public:
  constexpr void append_outer(
      const std::uint16_t layer, const LayerKind layer_kind,
      const ProjectionFamily family, const FusedOuterOperationKind kind,
      const Algorithm algorithm,
      const std::array<LogicalMemberSpec, 3U>& members,
      const std::uint8_t member_count) noexcept {
    const std::uint16_t outer_ordinal =
        static_cast<std::uint16_t>(outer_count_);
    const std::uint16_t logical_begin =
        static_cast<std::uint16_t>(logical_count_);
    const std::uint16_t physical_begin =
        static_cast<std::uint16_t>(physical_count_);
    const std::uint16_t shared_artifact =
        static_cast<std::uint16_t>(artifact_count_);
    std::uint32_t shared_output_offset = 0U;
    std::uint64_t outer_operations = 0U;

    for (std::size_t index = 0U; index < member_count; ++index) {
      const LogicalMemberSpec& member = members[index];
      LogicalProjectionRole logical;
      logical.ordinal = static_cast<std::uint16_t>(logical_count_);
      logical.layer = layer;
      logical.outer_operation = outer_ordinal;
      logical.layer_kind = layer_kind;
      logical.family = family;
      logical.subrole = member.subrole;
      logical.algorithm = algorithm;
      logical.activation_dtype = ActivationDtype::kBf16;
      logical.m = kPromptTokens;
      logical.n = member.n;
      logical.k = member.k;
      logical.conventional_operations =
          conventional_operations(logical.m, logical.n, logical.k);
      outer_operations += logical.conventional_operations;

      if (family == ProjectionFamily::kFp8W8A16) {
        logical.scale.kind =
            ScalePartitionKind::kFp8TensorGlobalExpandedPerOutputChannel;
        logical.scale.source_partition =
            static_cast<std::uint16_t>(scale_source_count_++);
        logical.scale.physical_artifact =
            static_cast<std::uint16_t>(artifact_count_++);
        logical.scale.output_channels = member.n;
      } else if (family == ProjectionFamily::kNvFp4W4A16) {
        logical.scale.kind =
            ScalePartitionKind::kNvFp4K16BlockAndTensorGlobal;
        logical.scale.source_partition =
            static_cast<std::uint16_t>(scale_source_count_++);
        logical.scale.physical_artifact = shared_artifact;
        logical.scale.physical_output_offset = shared_output_offset;
        logical.scale.output_channels = member.n;
        shared_output_offset += member.n;
      }
      manifest_.logical_roles[logical_count_++] = logical;
    }
    if (family == ProjectionFamily::kNvFp4W4A16) {
      ++artifact_count_;
    }

    if (family == ProjectionFamily::kFp8W8A16) {
      for (std::size_t panel = 0U; panel < kPanelCount; ++panel) {
        for (std::size_t member = 0U; member < member_count; ++member) {
          PhysicalCudaLaunch launch;
          launch.ordinal = static_cast<std::uint16_t>(physical_count_);
          launch.layer = layer;
          launch.outer_operation = outer_ordinal;
          launch.layer_kind = layer_kind;
          launch.family = family;
          launch.algorithm = algorithm;
          launch.activation_dtype = ActivationDtype::kBf16;
          launch.prompt_partition = PromptPartitionKind::kP8000Panel;
          launch.token_offset = static_cast<std::uint32_t>(panel) *
                                kPanelTokens;
          launch.token_count = kPanelTokens;
          launch.panel_ordinal = static_cast<std::uint8_t>(panel);
          launch.logical_member_count = 1U;
          launch.logical_members[0U] = static_cast<std::uint16_t>(
              logical_begin + member);
          manifest_.physical_launches[physical_count_++] = launch;
        }
      }
    } else {
      PhysicalCudaLaunch launch;
      launch.ordinal = static_cast<std::uint16_t>(physical_count_);
      launch.layer = layer;
      launch.outer_operation = outer_ordinal;
      launch.layer_kind = layer_kind;
      launch.family = family;
      launch.algorithm = algorithm;
      launch.activation_dtype = ActivationDtype::kBf16;
      launch.prompt_partition = PromptPartitionKind::kP40000WholePrompt;
      launch.token_count = kPromptTokens;
      launch.panel_ordinal = kWholePromptPanelOrdinal;
      launch.logical_member_count = member_count;
      for (std::size_t member = 0U; member < member_count; ++member) {
        launch.logical_members[member] = static_cast<std::uint16_t>(
            logical_begin + member);
      }
      manifest_.physical_launches[physical_count_++] = launch;
    }

    FusedOuterOperation outer;
    outer.ordinal = outer_ordinal;
    outer.layer = layer;
    outer.layer_kind = layer_kind;
    outer.family = family;
    outer.kind = kind;
    outer.algorithm = algorithm;
    outer.activation_dtype = ActivationDtype::kBf16;
    outer.logical_begin = logical_begin;
    outer.logical_count = member_count;
    outer.physical_begin = physical_begin;
    outer.physical_count = static_cast<std::uint8_t>(
        physical_count_ - physical_begin);
    outer.conventional_operations = outer_operations;
    manifest_.outer_operations[outer_count_++] = outer;
  }

  [[nodiscard]] constexpr ProjectionWorkManifest finish() const noexcept {
    return manifest_;
  }

 private:
  ProjectionWorkManifest manifest_{};
  std::size_t logical_count_ = 0U;
  std::size_t outer_count_ = 0U;
  std::size_t physical_count_ = 0U;
  std::size_t scale_source_count_ = 0U;
  std::size_t artifact_count_ = 0U;
};

[[nodiscard]] constexpr ProjectionWorkManifest make_manifest() noexcept {
  ManifestBuilder builder;
  for (std::size_t layer = 0U; layer < kLayerCount; ++layer) {
    const bool full = is_full_attention_layer(layer);
    const LayerKind layer_kind = full ? LayerKind::kFullAttention
                                      : LayerKind::kLinearAttention;
    if (full) {
      builder.append_outer(
          static_cast<std::uint16_t>(layer), layer_kind,
          ProjectionFamily::kFp8W8A16,
          FusedOuterOperationKind::kFullQkvInput,
          Algorithm::kFp8MarlinW8A16P8000,
          {{{CheckpointSubrole::kFullQGate, 12'288U, 5'120U},
            {CheckpointSubrole::kFullK, 1'024U, 5'120U},
            {CheckpointSubrole::kFullV, 1'024U, 5'120U}}},
          3U);
      builder.append_outer(
          static_cast<std::uint16_t>(layer), layer_kind,
          ProjectionFamily::kFp8W8A16,
          FusedOuterOperationKind::kFullOutput,
          Algorithm::kFp8MarlinW8A16P8000,
          {{{CheckpointSubrole::kFullO, 5'120U, 6'144U}, {}, {}}}, 1U);
    } else {
      builder.append_outer(
          static_cast<std::uint16_t>(layer), layer_kind,
          ProjectionFamily::kFp8W8A16,
          FusedOuterOperationKind::kLinearQkvZInput,
          Algorithm::kFp8MarlinW8A16P8000,
          {{{CheckpointSubrole::kLinearQkv, 10'240U, 5'120U},
            {CheckpointSubrole::kLinearZ, 6'144U, 5'120U},
            {}}},
          2U);
      builder.append_outer(
          static_cast<std::uint16_t>(layer), layer_kind,
          ProjectionFamily::kBf16, FusedOuterOperationKind::kLinearAb,
          Algorithm::kBf16AbSingleGridP40000,
          {{{CheckpointSubrole::kLinearA, 48U, 5'120U},
            {CheckpointSubrole::kLinearB, 48U, 5'120U},
            {}}},
          2U);
      builder.append_outer(
          static_cast<std::uint16_t>(layer), layer_kind,
          ProjectionFamily::kFp8W8A16,
          FusedOuterOperationKind::kLinearOutput,
          Algorithm::kFp8MarlinW8A16P8000,
          {{{CheckpointSubrole::kLinearO, 5'120U, 6'144U}, {}, {}}}, 1U);
    }
    builder.append_outer(
        static_cast<std::uint16_t>(layer), layer_kind,
        ProjectionFamily::kNvFp4W4A16,
        FusedOuterOperationKind::kMlpGateUp,
        Algorithm::kNvFp4PersistentMarlinGateUpP40000,
        {{{CheckpointSubrole::kMlpGate, 17'408U, 5'120U},
          {CheckpointSubrole::kMlpUp, 17'408U, 5'120U},
          {}}},
        2U);
    builder.append_outer(
        static_cast<std::uint16_t>(layer), layer_kind,
        ProjectionFamily::kNvFp4W4A16,
        FusedOuterOperationKind::kMlpDownResidual,
        Algorithm::kNvFp4PersistentMarlinDownResidualP40000,
        {{{CheckpointSubrole::kMlpDown, 5'120U, 17'408U}, {}, {}}}, 1U);
  }
  return builder.finish();
}

inline const ProjectionWorkManifest kManifest = make_manifest();

struct FamilyTotals {
  std::size_t logical_roles = 0U;
  std::size_t outer_operations = 0U;
  std::size_t physical_launches = 0U;
  std::uint64_t logical_conventional_operations = 0U;
  std::uint64_t physical_conventional_operations = 0U;
};

struct ManifestTotals {
  FamilyTotals fp8{};
  FamilyTotals nvfp4{};
  FamilyTotals bf16{};
  std::uint64_t logical_conventional_operations = 0U;
  std::uint64_t physical_conventional_operations = 0U;
};

enum class ManifestIssue : std::uint8_t {
  kNone = 0,
  kHeader,
  kLogicalRole,
  kScalePartition,
  kOuterOperation,
  kPhysicalLaunch,
  kPromptCoverage,
  kExactV10Identity,
  kTotals,
};

struct ManifestCheckResult {
  bool valid = false;
  ManifestIssue issue = ManifestIssue::kHeader;
  std::size_t index = 0U;
  ManifestTotals totals{};
};

[[nodiscard]] inline FamilyTotals& family_totals(
    ManifestTotals& totals, const ProjectionFamily family) noexcept {
  switch (family) {
    case ProjectionFamily::kFp8W8A16:
      return totals.fp8;
    case ProjectionFamily::kNvFp4W4A16:
      return totals.nvfp4;
    case ProjectionFamily::kBf16:
      return totals.bf16;
  }
  return totals.bf16;
}

[[nodiscard]] inline bool same_scale(
    const ScalePartitionIdentity& left,
    const ScalePartitionIdentity& right) noexcept {
  return left.kind == right.kind &&
         left.source_partition == right.source_partition &&
         left.physical_artifact == right.physical_artifact &&
         left.physical_output_offset == right.physical_output_offset &&
         left.output_channels == right.output_channels;
}

[[nodiscard]] inline bool same_logical(
    const LogicalProjectionRole& left,
    const LogicalProjectionRole& right) noexcept {
  return left.ordinal == right.ordinal && left.layer == right.layer &&
         left.outer_operation == right.outer_operation &&
         left.layer_kind == right.layer_kind && left.family == right.family &&
         left.subrole == right.subrole && left.algorithm == right.algorithm &&
         left.activation_dtype == right.activation_dtype && left.m == right.m &&
         left.n == right.n && left.k == right.k &&
         same_scale(left.scale, right.scale) &&
         left.conventional_operations == right.conventional_operations;
}

[[nodiscard]] inline bool same_outer(
    const FusedOuterOperation& left,
    const FusedOuterOperation& right) noexcept {
  return left.ordinal == right.ordinal && left.layer == right.layer &&
         left.layer_kind == right.layer_kind && left.family == right.family &&
         left.kind == right.kind && left.algorithm == right.algorithm &&
         left.activation_dtype == right.activation_dtype &&
         left.logical_begin == right.logical_begin &&
         left.logical_count == right.logical_count &&
         left.physical_begin == right.physical_begin &&
         left.physical_count == right.physical_count &&
         left.conventional_operations == right.conventional_operations;
}

[[nodiscard]] inline bool same_physical(
    const PhysicalCudaLaunch& left,
    const PhysicalCudaLaunch& right) noexcept {
  if (left.ordinal != right.ordinal || left.layer != right.layer ||
      left.outer_operation != right.outer_operation ||
      left.layer_kind != right.layer_kind || left.family != right.family ||
      left.algorithm != right.algorithm ||
      left.activation_dtype != right.activation_dtype ||
      left.prompt_partition != right.prompt_partition ||
      left.token_offset != right.token_offset ||
      left.token_count != right.token_count ||
      left.panel_ordinal != right.panel_ordinal ||
      left.logical_member_count != right.logical_member_count) {
    return false;
  }
  for (std::size_t index = 0U; index < left.logical_members.size(); ++index) {
    if (left.logical_members[index] != right.logical_members[index]) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] inline bool exact_v10_identity(
    const ProjectionWorkManifest& manifest) noexcept {
  for (std::size_t index = 0U; index < manifest.logical_roles.size(); ++index) {
    if (!same_logical(manifest.logical_roles[index],
                      kManifest.logical_roles[index])) {
      return false;
    }
  }
  for (std::size_t index = 0U; index < manifest.outer_operations.size();
       ++index) {
    if (!same_outer(manifest.outer_operations[index],
                    kManifest.outer_operations[index])) {
      return false;
    }
  }
  for (std::size_t index = 0U; index < manifest.physical_launches.size();
       ++index) {
    if (!same_physical(manifest.physical_launches[index],
                       kManifest.physical_launches[index])) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] inline ManifestCheckResult check_manifest(
    const ProjectionWorkManifest& manifest) noexcept {
  ManifestCheckResult result;
  if (manifest.version != 1U || !manifest.diagnostic_only ||
      manifest.production_reachable || manifest.prompt_tokens != kPromptTokens ||
      manifest.panel_tokens != kPanelTokens ||
      manifest.panel_count != kPanelCount) {
    result.issue = ManifestIssue::kHeader;
    return result;
  }

  std::array<bool, kQuantizedSourceScalePartitionCount> scale_sources{};
  std::array<bool, kQuantizedPhysicalArtifactCount> scale_artifacts{};
  std::array<std::uint32_t, kLogicalRoleCount> covered_tokens{};
  std::array<std::uint8_t, kLogicalRoleCount> panel_masks{};
  std::array<std::uint8_t, kLogicalRoleCount> physical_memberships{};

  for (std::size_t index = 0U; index < manifest.logical_roles.size(); ++index) {
    const LogicalProjectionRole& logical = manifest.logical_roles[index];
    if (logical.ordinal != index || logical.layer >= kLayerCount ||
        logical.outer_operation >= kOuterOperationCount ||
        logical.activation_dtype != ActivationDtype::kBf16 ||
        logical.m != kPromptTokens || logical.n == 0U || logical.k == 0U ||
        logical.conventional_operations !=
            conventional_operations(logical.m, logical.n, logical.k)) {
      result.issue = ManifestIssue::kLogicalRole;
      result.index = index;
      return result;
    }
    FamilyTotals& totals = family_totals(result.totals, logical.family);
    ++totals.logical_roles;
    totals.logical_conventional_operations +=
        logical.conventional_operations;
    result.totals.logical_conventional_operations +=
        logical.conventional_operations;

    if (logical.family == ProjectionFamily::kBf16) {
      if (logical.scale.kind != ScalePartitionKind::kNone ||
          logical.scale.source_partition != kNoScaleIdentity ||
          logical.scale.physical_artifact != kNoScaleIdentity ||
          logical.scale.physical_output_offset != 0U ||
          logical.scale.output_channels != 0U) {
        result.issue = ManifestIssue::kScalePartition;
        result.index = index;
        return result;
      }
    } else {
      const ScalePartitionKind expected_kind =
          logical.family == ProjectionFamily::kFp8W8A16
              ? ScalePartitionKind::
                    kFp8TensorGlobalExpandedPerOutputChannel
              : ScalePartitionKind::kNvFp4K16BlockAndTensorGlobal;
      if (logical.scale.kind != expected_kind ||
          logical.scale.source_partition >= scale_sources.size() ||
          logical.scale.physical_artifact >= scale_artifacts.size() ||
          logical.scale.output_channels != logical.n ||
          scale_sources[logical.scale.source_partition]) {
        result.issue = ManifestIssue::kScalePartition;
        result.index = index;
        return result;
      }
      scale_sources[logical.scale.source_partition] = true;
      scale_artifacts[logical.scale.physical_artifact] = true;
    }
  }
  for (std::size_t index = 0U; index < scale_sources.size(); ++index) {
    if (!scale_sources[index]) {
      result.issue = ManifestIssue::kScalePartition;
      result.index = index;
      return result;
    }
  }
  for (std::size_t index = 0U; index < scale_artifacts.size(); ++index) {
    if (!scale_artifacts[index]) {
      result.issue = ManifestIssue::kScalePartition;
      result.index = index;
      return result;
    }
  }

  for (std::size_t index = 0U; index < manifest.outer_operations.size();
       ++index) {
    const FusedOuterOperation& outer = manifest.outer_operations[index];
    if (outer.ordinal != index || outer.layer >= kLayerCount ||
        outer.logical_count == 0U || outer.logical_count > 3U ||
        outer.logical_begin + outer.logical_count > kLogicalRoleCount ||
        outer.physical_count == 0U ||
        outer.physical_begin + outer.physical_count > kPhysicalLaunchCount ||
        outer.activation_dtype != ActivationDtype::kBf16) {
      result.issue = ManifestIssue::kOuterOperation;
      result.index = index;
      return result;
    }
    std::uint64_t logical_operations = 0U;
    for (std::size_t member = 0U; member < outer.logical_count; ++member) {
      const LogicalProjectionRole& logical =
          manifest.logical_roles[outer.logical_begin + member];
      if (logical.outer_operation != index || logical.layer != outer.layer ||
          logical.layer_kind != outer.layer_kind ||
          logical.family != outer.family || logical.algorithm != outer.algorithm ||
          logical.activation_dtype != outer.activation_dtype) {
        result.issue = ManifestIssue::kOuterOperation;
        result.index = index;
        return result;
      }
      logical_operations += logical.conventional_operations;
    }
    if (logical_operations != outer.conventional_operations) {
      result.issue = ManifestIssue::kOuterOperation;
      result.index = index;
      return result;
    }
    ++family_totals(result.totals, outer.family).outer_operations;
  }

  for (std::size_t index = 0U; index < manifest.physical_launches.size();
       ++index) {
    const PhysicalCudaLaunch& launch = manifest.physical_launches[index];
    if (launch.ordinal != index || launch.layer >= kLayerCount ||
        launch.outer_operation >= kOuterOperationCount ||
        launch.logical_member_count == 0U ||
        launch.logical_member_count > launch.logical_members.size() ||
        launch.activation_dtype != ActivationDtype::kBf16) {
      result.issue = ManifestIssue::kPhysicalLaunch;
      result.index = index;
      return result;
    }
    const FusedOuterOperation& outer =
        manifest.outer_operations[launch.outer_operation];
    if (index < outer.physical_begin ||
        index >= outer.physical_begin + outer.physical_count ||
        launch.layer != outer.layer || launch.layer_kind != outer.layer_kind ||
        launch.family != outer.family || launch.algorithm != outer.algorithm ||
        launch.activation_dtype != outer.activation_dtype) {
      result.issue = ManifestIssue::kPhysicalLaunch;
      result.index = index;
      return result;
    }
    if (launch.family == ProjectionFamily::kFp8W8A16) {
      if (launch.prompt_partition != PromptPartitionKind::kP8000Panel ||
          launch.token_count != kPanelTokens ||
          launch.panel_ordinal >= kPanelCount ||
          launch.token_offset !=
              static_cast<std::uint32_t>(launch.panel_ordinal) *
                  kPanelTokens ||
          launch.logical_member_count != 1U) {
        result.issue = ManifestIssue::kPhysicalLaunch;
        result.index = index;
        return result;
      }
    } else if (launch.prompt_partition !=
                   PromptPartitionKind::kP40000WholePrompt ||
               launch.token_offset != 0U ||
               launch.token_count != kPromptTokens ||
               launch.panel_ordinal != kWholePromptPanelOrdinal ||
               launch.logical_member_count != outer.logical_count) {
      result.issue = ManifestIssue::kPhysicalLaunch;
      result.index = index;
      return result;
    }

    FamilyTotals& totals = family_totals(result.totals, launch.family);
    ++totals.physical_launches;
    for (std::size_t member = 0U; member < launch.logical_member_count;
         ++member) {
      const std::size_t logical_index = launch.logical_members[member];
      if (logical_index < outer.logical_begin ||
          logical_index >= outer.logical_begin + outer.logical_count) {
        result.issue = ManifestIssue::kPhysicalLaunch;
        result.index = index;
        return result;
      }
      for (std::size_t earlier = 0U; earlier < member; ++earlier) {
        if (launch.logical_members[earlier] == logical_index) {
          result.issue = ManifestIssue::kPhysicalLaunch;
          result.index = index;
          return result;
        }
      }
      const LogicalProjectionRole& logical =
          manifest.logical_roles[logical_index];
      covered_tokens[logical_index] += launch.token_count;
      ++physical_memberships[logical_index];
      if (launch.family == ProjectionFamily::kFp8W8A16) {
        const std::uint8_t bit = static_cast<std::uint8_t>(
            1U << launch.panel_ordinal);
        if ((panel_masks[logical_index] & bit) != 0U) {
          result.issue = ManifestIssue::kPromptCoverage;
          result.index = logical_index;
          return result;
        }
        panel_masks[logical_index] = static_cast<std::uint8_t>(
            panel_masks[logical_index] | bit);
      }
      const std::uint64_t physical_operations = conventional_operations(
          launch.token_count, logical.n, logical.k);
      totals.physical_conventional_operations += physical_operations;
      result.totals.physical_conventional_operations += physical_operations;
    }
  }

  for (std::size_t index = 0U; index < manifest.logical_roles.size(); ++index) {
    const LogicalProjectionRole& logical = manifest.logical_roles[index];
    const bool fp8 = logical.family == ProjectionFamily::kFp8W8A16;
    if (covered_tokens[index] != kPromptTokens ||
        (fp8 && (panel_masks[index] != 0x1fU ||
                 physical_memberships[index] != kPanelCount)) ||
        (!fp8 && physical_memberships[index] != 1U)) {
      result.issue = ManifestIssue::kPromptCoverage;
      result.index = index;
      return result;
    }
  }

  if (!exact_v10_identity(manifest)) {
    result.issue = ManifestIssue::kExactV10Identity;
    return result;
  }

  const bool exact_totals =
      result.totals.fp8.logical_roles == kFp8LogicalRoleCount &&
      result.totals.nvfp4.logical_roles == kNvFp4LogicalRoleCount &&
      result.totals.bf16.logical_roles == kBf16LogicalRoleCount &&
      result.totals.fp8.outer_operations == kFp8OuterOperationCount &&
      result.totals.nvfp4.outer_operations == kNvFp4OuterOperationCount &&
      result.totals.bf16.outer_operations == kBf16OuterOperationCount &&
      result.totals.fp8.physical_launches == kFp8PhysicalLaunchCount &&
      result.totals.nvfp4.physical_launches == kNvFp4PhysicalLaunchCount &&
      result.totals.bf16.physical_launches == kBf16PhysicalLaunchCount &&
      result.totals.fp8.logical_conventional_operations ==
          kFp8ConventionalOperations &&
      result.totals.nvfp4.logical_conventional_operations ==
          kNvFp4ConventionalOperations &&
      result.totals.bf16.logical_conventional_operations ==
          kBf16ConventionalOperations &&
      result.totals.fp8.physical_conventional_operations ==
          kFp8ConventionalOperations &&
      result.totals.nvfp4.physical_conventional_operations ==
          kNvFp4ConventionalOperations &&
      result.totals.bf16.physical_conventional_operations ==
          kBf16ConventionalOperations &&
      result.totals.logical_conventional_operations ==
          kProjectionConventionalOperations &&
      result.totals.physical_conventional_operations ==
          kProjectionConventionalOperations;
  if (!exact_totals) {
    result.issue = ManifestIssue::kTotals;
    return result;
  }
  result.valid = true;
  result.issue = ManifestIssue::kNone;
  return result;
}

}  // namespace q3x::tests::p40_work_manifest
