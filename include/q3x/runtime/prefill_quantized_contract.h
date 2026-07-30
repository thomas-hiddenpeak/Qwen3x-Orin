#pragma once

#include "q3x/model/weight_manifest.h"
#include "q3x/runtime/resident_weights.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace q3x::runtime {

inline constexpr std::uint32_t kPrefillSidecarManifestVersionMajor = 1U;
inline constexpr std::uint32_t kPrefillSidecarManifestVersionMinor = 0U;
inline constexpr std::size_t kQwen36PrefillProjectionCount = 400U;
inline constexpr std::size_t kQwen36PrefillMlpProjectionCount = 192U;
inline constexpr std::size_t kQwen36PrefillAttentionProjectionCount = 208U;
inline constexpr std::uint64_t kPrefillSidecarAlignment = 256U;
inline constexpr std::uint64_t kPrefillPromptArenaMaximumTokens = 40'000U;
inline constexpr std::uint64_t kPrefillPromptHiddenWidth = 5'120U;
inline constexpr std::uint64_t kPrefillPromptAttentionOutputWidth = 6'144U;
inline constexpr std::uint64_t kPrefillPromptIntermediateWidth = 17'408U;

// Raw payload totals intentionally exclude projection-to-projection alignment.
// They are fixed arithmetic identities for the pinned 400-projection model.
inline constexpr std::uint64_t kPrefillExactSidecarPayloadBytes =
    16'840'132'160ULL;
inline constexpr std::uint64_t kPrefillA8SafeSidecarPayloadBytes =
    24'707'072'000ULL;
inline constexpr std::uint64_t kPrefillA8CompactSidecarPayloadBytes =
    16'952'853'248ULL;
inline constexpr std::uint64_t kPrefillA4K64SidecarPayloadBytes =
    12'923'699'200ULL;
inline constexpr std::uint64_t kPrefillA4K128SidecarPayloadBytes =
    12'543'590'400ULL;

enum class PrefillProjectionFamily : std::uint8_t {
  kMlpGate = 0,
  kMlpUp,
  kMlpDown,
  kLinearQkv,
  kLinearZ,
  kLinearO,
  kFullQ,
  kFullK,
  kFullV,
  kFullO,
  kCount,
};

enum class PrefillSidecarKind : std::uint8_t {
  kExact = 0,
  kA8Safe,
  kA8Compact,
  kA4K64,
  kA4K128,
};

enum class PrefillSidecarResidencyClass : std::uint8_t {
  kExact = 0,
  kA8,
  kA4,
};

enum class PrefillWeightQuantization : std::uint8_t {
  kExactNvfp4E2m1 = 0,
  kExactFp8E4m3,
  kSymmetricW8,
  kSymmetricW4,
};

enum class PrefillSidecarLayout : std::uint8_t {
  kExactNvfp4MarlinConsumer = 0,
  kExactFp8SupermatrixConsumer,
  kSm87S8K128Consumer,
  kSm87U4B8K32Consumer,
  kSm87S4K64Consumer,
  kSm87S4K128Consumer,
};

enum class PrefillContractErrorCode : std::uint8_t {
  kNone = 0,
  kInvalidOption,
  kUnsupportedCheckpoint,
  kInvalidSourceIdentity,
  kMissingSourceTensor,
  kSourceTensorMismatch,
  kInvalidManifest,
  kCountMismatch,
  kSizeMismatch,
  kOffsetMismatch,
  kDigestMismatch,
  kArithmeticOverflow,
  kArenaLimitExceeded,
  kResidencyConflict,
  kActivationMismatch,
  kAllocationFailure,
};

struct PrefillContractDiagnostic {
  PrefillContractErrorCode code = PrefillContractErrorCode::kNone;
  std::string context;
  std::string message;
  std::string expected;
  std::string actual;

  [[nodiscard]] bool ok() const noexcept {
    return code == PrefillContractErrorCode::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

struct PrefillProjectionSidecarEntry {
  std::uint32_t ordinal = 0U;
  std::uint32_t layer_index = 0U;
  PrefillProjectionFamily family = PrefillProjectionFamily::kMlpGate;
  std::string source_module;

  // SHA-256 over the ordered authenticated source components. The digest
  // binds component tensor names, shard SHA-256 identities, source ranges,
  // dtypes, and shapes; it is not a digest of an unauthenticated filename.
  std::string source_sha256;

  // Logical GEMM shape is [output_size, input_size] == [N, K].
  std::uint64_t output_size = 0U;
  std::uint64_t input_size = 0U;
  PrefillWeightQuantization quantization =
      PrefillWeightQuantization::kExactNvfp4E2m1;
  std::uint32_t scale_group_size = 0U;
  PrefillSidecarLayout layout =
      PrefillSidecarLayout::kExactNvfp4MarlinConsumer;

  std::uint64_t weight_bytes = 0U;
  std::uint64_t scale_bytes = 0U;
  std::uint64_t metadata_bytes = 0U;
  std::uint64_t sidecar_offset = 0U;
  std::uint64_t sidecar_byte_size = 0U;
};

struct PrefillSidecarManifestSummary {
  std::size_t projection_count = 0U;
  std::size_t mlp_projection_count = 0U;
  std::size_t attention_projection_count = 0U;
  std::array<std::size_t,
             static_cast<std::size_t>(PrefillProjectionFamily::kCount)>
      family_counts{};
  std::uint64_t logical_weight_elements = 0U;
  std::uint64_t payload_bytes = 0U;
  std::uint64_t arena_bytes = 0U;
};

struct PrefillSidecarManifest {
  std::uint32_t version_major = kPrefillSidecarManifestVersionMajor;
  std::uint32_t version_minor = kPrefillSidecarManifestVersionMinor;
  PrefillSidecarKind kind = PrefillSidecarKind::kExact;
  PrefillSidecarResidencyClass residency_class =
      PrefillSidecarResidencyClass::kExact;
  std::uint64_t arena_alignment = kPrefillSidecarAlignment;

  std::string source_checkpoint_id;
  std::string source_config_sha256;
  std::string source_index_sha256;
  std::vector<PrefillProjectionSidecarEntry> projections;
  PrefillSidecarManifestSummary summary;

  // Digest over the complete versioned manifest body above. Offline sidecar
  // writers must persist and verify it before device residency is considered.
  std::string manifest_sha256;
};

struct PrefillSidecarManifestOptions {
  PrefillSidecarKind kind = PrefillSidecarKind::kExact;
  std::uint64_t arena_alignment = kPrefillSidecarAlignment;
};

struct PrefillSidecarManifestResult {
  std::optional<PrefillSidecarManifest> value;
  PrefillContractDiagnostic diagnostic;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && diagnostic.ok();
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Builds the sidecar manifest only from an already validated checkpoint
// manifest and its authenticated full-file shard identities. It reads no
// payload bytes and performs no CUDA operation. All 400 projection source
// components must be present and match the pinned Qwen3.6-27B physical ABI.
[[nodiscard]] PrefillSidecarManifestResult
build_qwen36_27b_prefill_sidecar_manifest(
    const model::weights::WeightManifest& source_manifest,
    const std::vector<ShardIdentity>& authenticated_shards,
    const PrefillSidecarManifestOptions& options = {});

// Revalidates version, inventory, shape/family mapping, format/group/layout,
// byte arithmetic, disjoint aligned offsets, summaries, and manifest digest.
[[nodiscard]] PrefillContractDiagnostic validate_prefill_sidecar_manifest(
    const PrefillSidecarManifest& manifest);

enum class PrefillPromptActivation : std::uint8_t {
  kA8 = 0,
  kA4,
};

struct PrefillPromptArenaRegion {
  std::uint64_t arena_offset = 0U;
  std::uint64_t byte_size = 0U;
  std::uint64_t logical_element_capacity = 0U;
  std::uint32_t element_bits = 0U;
};

struct PrefillPromptArenaOptions {
  std::uint64_t prompt_token_count = 0U;
  PrefillPromptActivation activation = PrefillPromptActivation::kA8;
  // A8 requires K128. A4 accepts K64 or K128.
  std::uint32_t activation_scale_group_size = 128U;
  // Zero means prompt_token_count. A smaller nonzero value is the layer-local
  // microspan fallback; the two BF16 hidden slabs always span the full prompt.
  std::uint64_t staging_token_capacity = 0U;
  std::uint64_t arena_alignment = kPrefillSidecarAlignment;
  std::uint64_t max_arena_bytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;
};

struct PrefillPromptArenaPlan {
  std::uint64_t prompt_token_count = 0U;
  PrefillPromptActivation activation = PrefillPromptActivation::kA8;
  std::uint32_t activation_scale_group_size = 0U;
  std::uint64_t staging_token_capacity = 0U;
  std::uint64_t arena_alignment = 0U;
  bool whole_prompt_staging = false;

  // Ping-pong canonical residual storage [P, 5120] BF16.
  std::array<PrefillPromptArenaRegion, 2U> hidden_bf16;
  // Reusable normalized projection input [S, 5120].
  PrefillPromptArenaRegion hidden_quantized;
  PrefillPromptArenaRegion hidden_scales_bf16;
  // O-projection input [S, 6144]. These two regions intentionally alias the
  // prefix of the larger intermediate regions below: O completes before the
  // Gate+Up epilogue publishes Down input, so their lifetimes are disjoint.
  PrefillPromptArenaRegion attention_output_input_quantized;
  PrefillPromptArenaRegion attention_output_input_scales_bf16;
  // Paired Gate+Up epilogue output [S, 17408], never two BF16 matrices.
  PrefillPromptArenaRegion intermediate_quantized;
  PrefillPromptArenaRegion intermediate_scales_bf16;
  // One FP32 cross-N sum-of-squares value per full-prompt row.
  PrefillPromptArenaRegion row_sum_squares_fp32;
  std::uint64_t arena_bytes = 0U;
};

struct PrefillPromptArenaPlanResult {
  std::optional<PrefillPromptArenaPlan> value;
  PrefillContractDiagnostic diagnostic;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && diagnostic.ok();
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Pure-host, overflow-safe planner. The ordinary RequestState arena and all
// production dispatch remain unchanged until a future gated route consumes
// this contract.
[[nodiscard]] PrefillPromptArenaPlanResult build_prefill_prompt_arena_plan(
    const PrefillPromptArenaOptions& options);

struct PrefillSidecarResidencyRequest {
  // Exactly one slot must be non-null. A process may never retain Exact+A8,
  // Exact+A4, or A8+A4 full-model Prefill sidecars simultaneously.
  const PrefillSidecarManifest* exact = nullptr;
  const PrefillSidecarManifest* a8 = nullptr;
  const PrefillSidecarManifest* a4 = nullptr;
  // Required for A8/A4 and must use the matching activation family. Exact
  // omits it because its established W4A16/W8A16 workspace is outside Stage 0.
  const PrefillPromptArenaPlan* prompt_arena = nullptr;
  std::uint64_t max_total_resident_bytes =
      32ULL * 1024ULL * 1024ULL * 1024ULL;
};

struct PrefillSidecarResidencyPlan {
  PrefillSidecarKind sidecar_kind = PrefillSidecarKind::kExact;
  PrefillSidecarResidencyClass residency_class =
      PrefillSidecarResidencyClass::kExact;
  std::uint64_t sidecar_bytes = 0U;
  std::uint64_t prompt_arena_bytes = 0U;
  std::uint64_t peak_resident_bytes = 0U;
};

struct PrefillSidecarResidencyPlanResult {
  std::optional<PrefillSidecarResidencyPlan> value;
  PrefillContractDiagnostic diagnostic;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && diagnostic.ok();
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

[[nodiscard]] PrefillSidecarResidencyPlanResult
build_prefill_sidecar_residency_plan(
    const PrefillSidecarResidencyRequest& request);

[[nodiscard]] PrefillSidecarResidencyClass residency_class_for(
    PrefillSidecarKind kind) noexcept;
[[nodiscard]] std::string_view to_string(
    PrefillProjectionFamily family) noexcept;
[[nodiscard]] std::string_view to_string(PrefillSidecarKind kind) noexcept;
[[nodiscard]] std::string_view to_string(
    PrefillSidecarResidencyClass residency) noexcept;
[[nodiscard]] std::string_view to_string(
    PrefillWeightQuantization quantization) noexcept;
[[nodiscard]] std::string_view to_string(
    PrefillSidecarLayout layout) noexcept;
[[nodiscard]] std::string_view to_string(
    PrefillContractErrorCode code) noexcept;

}  // namespace q3x::runtime
