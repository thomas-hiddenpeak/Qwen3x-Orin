#pragma once

#include "q3x/kernels/sm87_target_aot_projection_fp8_cuda.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {

// Default-off C8000 FP8 constituent of AC-PREFILL-SM87-MACROFEED-v4.
// It consumes the canonical target-AOT Marlin payload without repacking and
// preserves the V2 raw-code decode, partition-private tensor scales,
// ascending full-K FP32 accumulation, and BF16-RNE publication boundaries.
// This header exposes no selector and grants no production authority.
enum class Sm87MacroFeedV4Fp8Identity : std::uint64_t {
  kInvalid = 0U,
  kGdnQkvZM64N128K64OrdinaryGridV1 = 0x5133'4d46'5634'4601ULL,
  kFullQkvM64N128K64OrdinaryGridV1 = 0x5133'4d46'5634'4602ULL,
  kAttentionOutputM64N128K64OrdinaryGridV1 =
      0x5133'4d46'5634'4603ULL,
};

inline constexpr std::size_t kSm87MacroFeedV4Fp8Tokens = 8'000U;
inline constexpr std::size_t kSm87MacroFeedV4Fp8BlockM = 64U;
inline constexpr std::size_t kSm87MacroFeedV4Fp8BlockN = 128U;
inline constexpr std::size_t kSm87MacroFeedV4Fp8BlockK = 64U;
inline constexpr std::size_t kSm87MacroFeedV4Fp8Threads = 256U;
inline constexpr std::size_t kSm87MacroFeedV4Fp8Warps = 8U;
inline constexpr std::size_t kSm87MacroFeedV4Fp8PipelineStages = 3U;
inline constexpr std::size_t kSm87MacroFeedV4Fp8GridM = 125U;
inline constexpr std::size_t kSm87MacroFeedV4Fp8SmCount = 16U;
inline constexpr std::size_t kSm87MacroFeedV4Fp8RequiredCtasPerSm = 2U;
inline constexpr std::size_t kSm87MacroFeedV4Fp8MaximumRegisters = 128U;
inline constexpr std::size_t kSm87MacroFeedV4Fp8ActivationBytesPerStage =
    kSm87MacroFeedV4Fp8BlockM * kSm87MacroFeedV4Fp8BlockK *
    sizeof(std::uint16_t);
inline constexpr std::size_t kSm87MacroFeedV4Fp8WeightBytesPerStage =
    kSm87MacroFeedV4Fp8BlockN * kSm87MacroFeedV4Fp8BlockK;
inline constexpr std::size_t kSm87MacroFeedV4Fp8DynamicSharedBytes =
    kSm87MacroFeedV4Fp8PipelineStages *
    (kSm87MacroFeedV4Fp8ActivationBytesPerStage +
     kSm87MacroFeedV4Fp8WeightBytesPerStage);

inline constexpr std::size_t kSm87MacroFeedV4Fp8HiddenRowStride = 5'120U;
inline constexpr std::size_t kSm87MacroFeedV4Fp8ScratchRowStride = 17'408U;
inline constexpr std::size_t kSm87MacroFeedV4Fp8KvNhdRowStride = 1'024U;
inline constexpr std::size_t kSm87MacroFeedV4Fp8GdnQkvFeatures = 10'240U;
inline constexpr std::size_t kSm87MacroFeedV4Fp8GdnZOffset = 10'240U;
inline constexpr std::size_t kSm87MacroFeedV4Fp8GdnZFeatures = 6'144U;
inline constexpr std::size_t kSm87MacroFeedV4Fp8FullQFeatures = 6'144U;
inline constexpr std::size_t kSm87MacroFeedV4Fp8FullGateLogicalOffset =
    6'144U;
inline constexpr std::size_t kSm87MacroFeedV4Fp8FullGateFeatures = 6'144U;
inline constexpr std::size_t kSm87MacroFeedV4Fp8FullQGateFeatures = 12'288U;
inline constexpr std::size_t kSm87MacroFeedV4Fp8AttentionOutputInputFeatures =
    6'144U;
inline constexpr std::size_t kSm87MacroFeedV4Fp8AttentionHeads = 24U;
inline constexpr std::size_t kSm87MacroFeedV4Fp8AttentionHeadFeatures = 256U;
inline constexpr std::size_t kSm87MacroFeedV4Fp8QGateHeadStride = 512U;

inline constexpr std::size_t kSm87MacroFeedV4Fp8TestInputFeatures = 256U;
inline constexpr std::size_t kSm87MacroFeedV4Fp8TestKTiles = 4U;
inline constexpr std::size_t kSm87MacroFeedV4Fp8TestPayloadBytes = 65'536U;
inline constexpr std::size_t
    kSm87MacroFeedV4Fp8TestAttentionOutputLogicalFirstK = 192U;

// Scratch preserves the model-native [24 heads, Q/G, 256 features] row.
// Attention overwrites Q in place and leaves Gate intact.  O consumes the Q
// slots through this logical-to-physical mapping, without a compact copy.
[[nodiscard]] constexpr std::size_t
sm87_macrofeed_v4_fp8_interleaved_q_physical_offset(
    const std::size_t logical_q) noexcept {
  return (logical_q / kSm87MacroFeedV4Fp8AttentionHeadFeatures) *
             kSm87MacroFeedV4Fp8QGateHeadStride +
         logical_q % kSm87MacroFeedV4Fp8AttentionHeadFeatures;
}

[[nodiscard]] constexpr std::size_t
sm87_macrofeed_v4_fp8_interleaved_q_gate_physical_offset(
    const std::size_t logical_q_gate) noexcept {
  const bool gate = logical_q_gate >= kSm87MacroFeedV4Fp8FullQFeatures;
  const std::size_t local =
      gate ? logical_q_gate - kSm87MacroFeedV4Fp8FullQFeatures
           : logical_q_gate;
  return sm87_macrofeed_v4_fp8_interleaved_q_physical_offset(local) +
         (gate ? kSm87MacroFeedV4Fp8AttentionHeadFeatures : 0U);
}

[[nodiscard]] constexpr bool sm87_macrofeed_v4_fp8_role(
    const Sm87TargetAotProjectionRole role) noexcept {
  return role == Sm87TargetAotProjectionRole::kFp8GdnQkvZ ||
         role == Sm87TargetAotProjectionRole::kFp8FullQkv ||
         role == Sm87TargetAotProjectionRole::kFp8AttentionOutput;
}

[[nodiscard]] constexpr Sm87MacroFeedV4Fp8Identity
sm87_macrofeed_v4_fp8_identity(
    const Sm87TargetAotProjectionRole role) noexcept {
  switch (role) {
    case Sm87TargetAotProjectionRole::kFp8GdnQkvZ:
      return Sm87MacroFeedV4Fp8Identity::
          kGdnQkvZM64N128K64OrdinaryGridV1;
    case Sm87TargetAotProjectionRole::kFp8FullQkv:
      return Sm87MacroFeedV4Fp8Identity::
          kFullQkvM64N128K64OrdinaryGridV1;
    case Sm87TargetAotProjectionRole::kFp8AttentionOutput:
      return Sm87MacroFeedV4Fp8Identity::
          kAttentionOutputM64N128K64OrdinaryGridV1;
    default:
      return Sm87MacroFeedV4Fp8Identity::kInvalid;
  }
}

[[nodiscard]] constexpr std::uint16_t
sm87_macrofeed_v4_fp8_bias_shift_bf16_bits(
    const std::uint8_t code) noexcept {
  return static_cast<std::uint16_t>(
      (static_cast<std::uint16_t>(code & 0x80U) << 8U) |
      (static_cast<std::uint16_t>(code & 0x7fU) << 4U));
}

struct Sm87MacroFeedV4Fp8Plan final {
  Sm87MacroFeedV4Fp8Identity identity = Sm87MacroFeedV4Fp8Identity::kInvalid;
  Sm87TargetAotProjectionRole role =
      Sm87TargetAotProjectionRole::kInvalid;
  std::size_t token_count = 0U;
  std::size_t input_features = 0U;
  std::size_t input_physical_span = 0U;
  std::size_t input_row_stride = 0U;
  std::size_t projected_output_features = 0U;
  std::size_t primary_output_features = 0U;
  std::size_t primary_output_row_stride = 0U;
  std::size_t key_output_features = 0U;
  std::size_t key_output_row_stride = 0U;
  std::size_t value_output_features = 0U;
  std::size_t value_output_row_stride = 0U;
  std::size_t partition_count = 0U;
  std::array<std::size_t, 3U> partition_first_n{};
  std::array<std::size_t, 3U> partition_features{};
  std::array<std::size_t, 3U> partition_payload_offsets{};
  std::size_t grid_m = 0U;
  std::size_t grid_n = 0U;
  std::size_t k_tiles = 0U;
  std::size_t logical_tasks = 0U;
  std::size_t payload_bytes = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  bool ordinary_full_grid = false;
  bool m_major_n_adjacent = false;
  bool role_specific_direct_scatter = false;
  bool private_nhd_kv = false;
  bool input_base_offset_permitted = false;
  bool full_q_gate_head_interleaved = false;
  bool attention_output_gathers_interleaved_q = false;
  bool attention_gate_and_gap_preserved = false;
  bool exact_fp8_marlin_semantics = false;
  bool authenticated_asset_zero_copy = false;
  bool fallback_permitted = true;
  bool selector_present = true;
  bool request_jit_repack_or_autotune = true;
  bool numerical_contract_qualified = true;
  bool production_dispatch_eligible = true;

  [[nodiscard]] constexpr bool valid() const noexcept;
};

[[nodiscard]] constexpr Sm87MacroFeedV4Fp8Plan
sm87_macrofeed_v4_fp8_plan(const Sm87TargetAotProjectionRole role,
                           const std::size_t token_count) noexcept {
  if (!sm87_macrofeed_v4_fp8_role(role) ||
      token_count != kSm87MacroFeedV4Fp8Tokens) {
    return {};
  }
  Sm87MacroFeedV4Fp8Plan plan;
  plan.identity = sm87_macrofeed_v4_fp8_identity(role);
  plan.role = role;
  plan.token_count = token_count;
  plan.grid_m = kSm87MacroFeedV4Fp8GridM;
  plan.dynamic_shared_bytes = kSm87MacroFeedV4Fp8DynamicSharedBytes;
  plan.ordinary_full_grid = true;
  plan.m_major_n_adjacent = true;
  plan.role_specific_direct_scatter = true;
  plan.exact_fp8_marlin_semantics = true;
  plan.authenticated_asset_zero_copy = true;
  plan.fallback_permitted = false;
  plan.selector_present = false;
  plan.request_jit_repack_or_autotune = false;
  plan.numerical_contract_qualified = false;
  plan.production_dispatch_eligible = false;
  if (role == Sm87TargetAotProjectionRole::kFp8GdnQkvZ) {
    plan.input_features = 5'120U;
    plan.input_physical_span = 5'120U;
    plan.input_row_stride = kSm87MacroFeedV4Fp8HiddenRowStride;
    plan.projected_output_features = 16'384U;
    plan.primary_output_features = 16'384U;
    plan.primary_output_row_stride = kSm87MacroFeedV4Fp8ScratchRowStride;
    plan.partition_count = 2U;
    plan.partition_first_n = {0U, 10'240U, 0U};
    plan.partition_features = {10'240U, 6'144U, 0U};
    plan.partition_payload_offsets = {0U, 52'428'800U, 0U};
    plan.grid_n = 128U;
    plan.k_tiles = 80U;
  } else if (role == Sm87TargetAotProjectionRole::kFp8FullQkv) {
    plan.input_features = 5'120U;
    plan.input_physical_span = 5'120U;
    plan.input_row_stride = kSm87MacroFeedV4Fp8HiddenRowStride;
    plan.projected_output_features = 14'336U;
    plan.primary_output_features = 12'288U;
    plan.primary_output_row_stride = kSm87MacroFeedV4Fp8ScratchRowStride;
    plan.key_output_features = 1'024U;
    plan.key_output_row_stride = kSm87MacroFeedV4Fp8KvNhdRowStride;
    plan.value_output_features = 1'024U;
    plan.value_output_row_stride = kSm87MacroFeedV4Fp8KvNhdRowStride;
    plan.partition_count = 3U;
    plan.partition_first_n = {0U, 12'288U, 13'312U};
    plan.partition_features = {12'288U, 1'024U, 1'024U};
    plan.partition_payload_offsets = {0U, 62'914'560U, 68'157'440U};
    plan.grid_n = 112U;
    plan.k_tiles = 80U;
    plan.private_nhd_kv = true;
    plan.full_q_gate_head_interleaved = true;
  } else {
    plan.input_features = 6'144U;
    plan.input_physical_span = kSm87MacroFeedV4Fp8FullQGateFeatures;
    plan.input_row_stride = kSm87MacroFeedV4Fp8ScratchRowStride;
    plan.projected_output_features = 5'120U;
    plan.primary_output_features = 5'120U;
    plan.primary_output_row_stride = kSm87MacroFeedV4Fp8HiddenRowStride;
    plan.partition_count = 1U;
    plan.partition_first_n = {0U, 0U, 0U};
    plan.partition_features = {5'120U, 0U, 0U};
    plan.partition_payload_offsets = {0U, 0U, 0U};
    plan.grid_n = 40U;
    plan.k_tiles = 96U;
    plan.attention_output_gathers_interleaved_q = true;
    plan.attention_gate_and_gap_preserved = true;
  }
  plan.logical_tasks = plan.grid_m * plan.grid_n;
  const auto layout = sm87_target_aot_projection_packed_layout(role);
  plan.payload_bytes = layout.payload_bytes;
  return plan;
}

constexpr bool Sm87MacroFeedV4Fp8Plan::valid() const noexcept {
  if (identity != sm87_macrofeed_v4_fp8_identity(role) ||
      token_count != kSm87MacroFeedV4Fp8Tokens ||
      grid_m != kSm87MacroFeedV4Fp8GridM ||
      input_physical_span == 0U || input_row_stride < input_physical_span ||
      input_features != k_tiles * kSm87MacroFeedV4Fp8BlockK ||
      projected_output_features != grid_n * kSm87MacroFeedV4Fp8BlockN ||
      logical_tasks != grid_m * grid_n ||
      dynamic_shared_bytes != kSm87MacroFeedV4Fp8DynamicSharedBytes ||
      !ordinary_full_grid || !m_major_n_adjacent ||
      !role_specific_direct_scatter || !exact_fp8_marlin_semantics ||
      !authenticated_asset_zero_copy || fallback_permitted ||
      selector_present || request_jit_repack_or_autotune ||
      numerical_contract_qualified || production_dispatch_eligible) {
    return false;
  }
  const auto layout = sm87_target_aot_projection_packed_layout(role);
  if (!layout.valid() || layout.payload_bytes != payload_bytes ||
      layout.partition_count != partition_count) {
    return false;
  }
  for (std::size_t index = 0U; index < partition_count; ++index) {
    if (layout.partitions[index].global_n_offset !=
            partition_first_n[index] ||
        layout.partitions[index].output_features !=
            partition_features[index] ||
        layout.partitions[index].payload_offset !=
            partition_payload_offsets[index]) {
      return false;
    }
  }
  if (role == Sm87TargetAotProjectionRole::kFp8GdnQkvZ) {
    return input_physical_span == kSm87MacroFeedV4Fp8HiddenRowStride &&
           input_row_stride == kSm87MacroFeedV4Fp8HiddenRowStride &&
           primary_output_features == 16'384U &&
           primary_output_row_stride ==
               kSm87MacroFeedV4Fp8ScratchRowStride &&
           key_output_features == 0U && value_output_features == 0U &&
           !private_nhd_kv && !input_base_offset_permitted &&
           !full_q_gate_head_interleaved &&
           !attention_output_gathers_interleaved_q &&
           !attention_gate_and_gap_preserved;
  }
  if (role == Sm87TargetAotProjectionRole::kFp8FullQkv) {
    return input_physical_span == kSm87MacroFeedV4Fp8HiddenRowStride &&
           input_row_stride == kSm87MacroFeedV4Fp8HiddenRowStride &&
           primary_output_features == kSm87MacroFeedV4Fp8FullQGateFeatures &&
           primary_output_row_stride ==
               kSm87MacroFeedV4Fp8ScratchRowStride &&
           key_output_features == kSm87MacroFeedV4Fp8KvNhdRowStride &&
           key_output_row_stride == kSm87MacroFeedV4Fp8KvNhdRowStride &&
           value_output_features == kSm87MacroFeedV4Fp8KvNhdRowStride &&
           value_output_row_stride == kSm87MacroFeedV4Fp8KvNhdRowStride &&
           private_nhd_kv && !input_base_offset_permitted &&
           full_q_gate_head_interleaved &&
           !attention_output_gathers_interleaved_q &&
           !attention_gate_and_gap_preserved;
  }
  return input_physical_span == kSm87MacroFeedV4Fp8FullQGateFeatures &&
         input_row_stride == kSm87MacroFeedV4Fp8ScratchRowStride &&
         primary_output_features == kSm87MacroFeedV4Fp8HiddenRowStride &&
         primary_output_row_stride ==
             kSm87MacroFeedV4Fp8HiddenRowStride &&
         key_output_features == 0U && value_output_features == 0U &&
         !private_nhd_kv && !input_base_offset_permitted &&
         !full_q_gate_head_interleaved &&
         attention_output_gathers_interleaved_q &&
         attention_gate_and_gap_preserved;
}

struct Sm87MacroFeedV4Fp8LayoutBinding final {
  Sm87TargetAotProjectionRole role =
      Sm87TargetAotProjectionRole::kInvalid;
  std::size_t token_count = 0U;
  const std::uint16_t* input = nullptr;
  std::size_t input_row_stride = 0U;
  std::uint16_t* primary_output = nullptr;
  std::size_t primary_output_row_stride = 0U;
  std::uint16_t* key_output = nullptr;
  std::size_t key_output_row_stride = 0U;
  std::uint16_t* value_output = nullptr;
  std::size_t value_output_row_stride = 0U;
};

struct Sm87MacroFeedV4Fp8ByteRange final {
  std::uintptr_t begin = 0U;
  std::uintptr_t end = 0U;
  bool valid = false;
};

[[nodiscard]] constexpr Sm87MacroFeedV4Fp8ByteRange
sm87_macrofeed_v4_fp8_strided_range(
    const void* const pointer, const std::size_t rows,
    const std::size_t row_stride, const std::size_t row_width) noexcept {
  if (pointer == nullptr || rows == 0U || row_width == 0U ||
      row_stride < row_width ||
      row_stride > std::numeric_limits<std::uintptr_t>::max() /
                       sizeof(std::uint16_t)) {
    return {};
  }
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  const std::size_t last_row = rows - 1U;
  if (last_row != 0U &&
      row_stride > (std::numeric_limits<std::uintptr_t>::max() /
                    sizeof(std::uint16_t) - row_width) /
                       last_row) {
    return {};
  }
  const std::uintptr_t elements = last_row * row_stride + row_width;
  if (begin > std::numeric_limits<std::uintptr_t>::max() -
                  elements * sizeof(std::uint16_t)) {
    return {};
  }
  return {begin, begin + elements * sizeof(std::uint16_t), true};
}

[[nodiscard]] constexpr bool sm87_macrofeed_v4_fp8_ranges_overlap(
    const Sm87MacroFeedV4Fp8ByteRange& left,
    const Sm87MacroFeedV4Fp8ByteRange& right) noexcept {
  return left.valid && right.valid && left.begin < right.end &&
         right.begin < left.end;
}

[[nodiscard]] constexpr bool sm87_macrofeed_v4_fp8_layout_valid(
    const Sm87MacroFeedV4Fp8LayoutBinding& binding) noexcept {
  const auto plan = sm87_macrofeed_v4_fp8_plan(binding.role,
                                               binding.token_count);
  if (!plan.valid() || binding.input == nullptr ||
      binding.primary_output == nullptr ||
      binding.input_row_stride != plan.input_row_stride ||
      binding.primary_output_row_stride !=
          plan.primary_output_row_stride ||
      reinterpret_cast<std::uintptr_t>(binding.input) % 16U != 0U ||
      reinterpret_cast<std::uintptr_t>(binding.primary_output) % 16U != 0U ||
      binding.input_row_stride % 8U != 0U ||
      binding.primary_output_row_stride % 8U != 0U) {
    return false;
  }
  if (binding.role == Sm87TargetAotProjectionRole::kFp8FullQkv) {
    if (binding.key_output == nullptr || binding.value_output == nullptr ||
        binding.key_output_row_stride != plan.key_output_row_stride ||
        binding.value_output_row_stride != plan.value_output_row_stride ||
        reinterpret_cast<std::uintptr_t>(binding.key_output) % 16U != 0U ||
        reinterpret_cast<std::uintptr_t>(binding.value_output) % 16U != 0U) {
      return false;
    }
  } else if (binding.key_output != nullptr ||
             binding.value_output != nullptr ||
             binding.key_output_row_stride != 0U ||
             binding.value_output_row_stride != 0U) {
    return false;
  }

  std::array<Sm87MacroFeedV4Fp8ByteRange, 4U> ranges{};
  std::size_t count = 0U;
  ranges[count++] = sm87_macrofeed_v4_fp8_strided_range(
      binding.input, binding.token_count, binding.input_row_stride,
      plan.input_physical_span);
  ranges[count++] = sm87_macrofeed_v4_fp8_strided_range(
      binding.primary_output, binding.token_count,
      binding.primary_output_row_stride, plan.primary_output_features);
  if (binding.role == Sm87TargetAotProjectionRole::kFp8FullQkv) {
    ranges[count++] = sm87_macrofeed_v4_fp8_strided_range(
        binding.key_output, binding.token_count,
        binding.key_output_row_stride, plan.key_output_features);
    ranges[count++] = sm87_macrofeed_v4_fp8_strided_range(
        binding.value_output, binding.token_count,
        binding.value_output_row_stride, plan.value_output_features);
  }
  for (std::size_t first = 0U; first < count; ++first) {
    if (!ranges[first].valid) {
      return false;
    }
    for (std::size_t second = first + 1U; second < count; ++second) {
      if (sm87_macrofeed_v4_fp8_ranges_overlap(ranges[first],
                                                ranges[second])) {
        return false;
      }
    }
  }
  return true;
}

struct Sm87MacroFeedV4Fp8CudaResources final {
  Sm87MacroFeedV4Fp8Identity identity = Sm87MacroFeedV4Fp8Identity::kInvalid;
  Sm87TargetAotProjectionRole role =
      Sm87TargetAotProjectionRole::kInvalid;
  std::int32_t device_ordinal = -1;
  std::int32_t compute_major = 0;
  std::int32_t compute_minor = 0;
  std::int32_t sm_count = 0;
  std::int32_t binary_version = 0;
  std::int32_t registers_per_thread = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  std::int32_t maximum_threads_per_block = 0;
  std::int32_t active_blocks_per_sm = 0;
  std::size_t shared_bytes_per_sm = 0U;
  std::size_t optin_shared_bytes_per_block = 0U;
  bool kernel_compiled = false;
  bool static_resource_gate_passed = false;
  bool numerical_contract_qualified = true;
  bool production_dispatch_eligible = true;
};

[[nodiscard]] constexpr bool sm87_macrofeed_v4_fp8_resource_gate(
    const Sm87MacroFeedV4Fp8CudaResources& resources) noexcept {
  return resources.identity == sm87_macrofeed_v4_fp8_identity(resources.role) &&
         resources.device_ordinal >= 0 && resources.compute_major == 8 &&
         resources.compute_minor == 7 &&
         resources.sm_count ==
             static_cast<std::int32_t>(kSm87MacroFeedV4Fp8SmCount) &&
         resources.binary_version == 87 && resources.kernel_compiled &&
         resources.registers_per_thread > 0 &&
         resources.registers_per_thread <=
             static_cast<std::int32_t>(
                 kSm87MacroFeedV4Fp8MaximumRegisters) &&
         resources.static_shared_bytes == 0U &&
         resources.dynamic_shared_bytes ==
             kSm87MacroFeedV4Fp8DynamicSharedBytes &&
         resources.local_bytes == 0U &&
         resources.maximum_threads_per_block >=
             static_cast<std::int32_t>(kSm87MacroFeedV4Fp8Threads) &&
         resources.active_blocks_per_sm >=
             static_cast<std::int32_t>(
                 kSm87MacroFeedV4Fp8RequiredCtasPerSm) &&
         resources.shared_bytes_per_sm >=
             kSm87MacroFeedV4Fp8RequiredCtasPerSm *
                 kSm87MacroFeedV4Fp8DynamicSharedBytes &&
         resources.optin_shared_bytes_per_block >=
             kSm87MacroFeedV4Fp8DynamicSharedBytes &&
         !resources.numerical_contract_qualified &&
         !resources.production_dispatch_eligible;
}

// Public, caller-constructible T1 observation only.  The identity below is a
// deterministic coherence checksum, not a signature, an opaque package
// token, or an execution capability.  Until the private V4 startup package
// binds this constituent, this snapshot can only admit explicit T1 probes.
struct Sm87MacroFeedV4Fp8T1AdmissionSnapshot final {
  std::uint64_t snapshot_identity = 0U;
  Sm87MacroFeedV4Fp8CudaResources resources{};
  bool dynamic_shared_attribute_observed = false;
  bool resource_query_completed = false;
  bool caller_constructible = true;
  bool startup_package_bound = false;
  bool execution_capability = false;
  bool admission_only = true;
  bool default_off = true;
  bool selector_present = false;
  bool production_dispatch_eligible = false;
};

[[nodiscard]] constexpr std::uint64_t sm87_macrofeed_v4_fp8_hash_u64(
    std::uint64_t hash, const std::uint64_t value) noexcept {
  for (std::size_t byte = 0U; byte < sizeof(value); ++byte) {
    hash ^= static_cast<std::uint8_t>(value >> (8U * byte));
    hash *= 1'099'511'628'211ULL;
  }
  return hash;
}

[[nodiscard]] constexpr std::uint64_t
sm87_macrofeed_v4_fp8_compute_t1_admission_snapshot_identity(
    const Sm87MacroFeedV4Fp8T1AdmissionSnapshot& snapshot) noexcept {
  std::uint64_t hash = 14'695'981'039'346'656'037ULL;
  hash = sm87_macrofeed_v4_fp8_hash_u64(
      hash, static_cast<std::uint64_t>(snapshot.resources.identity));
  hash = sm87_macrofeed_v4_fp8_hash_u64(
      hash, static_cast<std::uint64_t>(snapshot.resources.role));
  hash = sm87_macrofeed_v4_fp8_hash_u64(
      hash, static_cast<std::uint32_t>(snapshot.resources.device_ordinal));
  hash = sm87_macrofeed_v4_fp8_hash_u64(
      hash,
      static_cast<std::uint32_t>(snapshot.resources.registers_per_thread));
  hash = sm87_macrofeed_v4_fp8_hash_u64(
      hash, snapshot.resources.dynamic_shared_bytes);
  hash = sm87_macrofeed_v4_fp8_hash_u64(
      hash,
      static_cast<std::uint32_t>(snapshot.resources.active_blocks_per_sm));
  hash = sm87_macrofeed_v4_fp8_hash_u64(
      hash, snapshot.resources.static_resource_gate_passed);
  hash = sm87_macrofeed_v4_fp8_hash_u64(
      hash, snapshot.dynamic_shared_attribute_observed);
  hash = sm87_macrofeed_v4_fp8_hash_u64(
      hash, snapshot.resource_query_completed);
  hash = sm87_macrofeed_v4_fp8_hash_u64(hash,
                                         snapshot.caller_constructible);
  hash = sm87_macrofeed_v4_fp8_hash_u64(hash,
                                         snapshot.startup_package_bound);
  hash = sm87_macrofeed_v4_fp8_hash_u64(hash,
                                         snapshot.execution_capability);
  hash = sm87_macrofeed_v4_fp8_hash_u64(hash, snapshot.admission_only);
  hash = sm87_macrofeed_v4_fp8_hash_u64(hash, snapshot.default_off);
  hash = sm87_macrofeed_v4_fp8_hash_u64(hash, snapshot.selector_present);
  return sm87_macrofeed_v4_fp8_hash_u64(hash,
                                         snapshot.production_dispatch_eligible);
}

[[nodiscard]] constexpr bool
sm87_macrofeed_v4_fp8_t1_admission_snapshot_valid(
    const Sm87MacroFeedV4Fp8T1AdmissionSnapshot& snapshot) noexcept {
  return snapshot.snapshot_identity != 0U &&
         snapshot.snapshot_identity ==
             sm87_macrofeed_v4_fp8_compute_t1_admission_snapshot_identity(
                 snapshot) &&
         sm87_macrofeed_v4_fp8_resource_gate(snapshot.resources) &&
         snapshot.resources.static_resource_gate_passed &&
         snapshot.dynamic_shared_attribute_observed &&
         snapshot.resource_query_completed && snapshot.caller_constructible &&
         !snapshot.startup_package_bound && !snapshot.execution_capability &&
         snapshot.admission_only && snapshot.default_off &&
         !snapshot.selector_present &&
         !snapshot.production_dispatch_eligible;
}

struct Sm87MacroFeedV4Fp8Arguments final {
  Sm87TargetAotProjectionRole role =
      Sm87TargetAotProjectionRole::kInvalid;
  const std::uint16_t* input = nullptr;
  std::size_t input_row_stride = 0U;
  Sm87TargetAotFp8CudaAssetView asset{};
  std::size_t token_count = 0U;
  std::uint16_t* primary_output = nullptr;
  std::size_t primary_output_row_stride = 0U;
  std::uint16_t* key_output = nullptr;
  std::size_t key_output_row_stride = 0U;
  std::uint16_t* value_output = nullptr;
  std::size_t value_output_row_stride = 0U;
  void* cuda_stream = nullptr;
};

struct Sm87MacroFeedV4Fp8T1AdmissionLaunchReceipt final {
  Sm87MacroFeedV4Fp8Identity identity = Sm87MacroFeedV4Fp8Identity::kInvalid;
  Sm87TargetAotProjectionRole role =
      Sm87TargetAotProjectionRole::kInvalid;
  std::uint64_t artifact_identity = 0U;
  std::int32_t device_ordinal = -1;
  std::size_t token_count = 0U;
  std::size_t logical_tasks = 0U;
  std::uint32_t physical_kernel_launches = 0U;
  std::uint32_t fallback_launches = 0U;
  bool ordinary_full_grid = false;
  bool role_specific_direct_scatter = false;
  bool private_nhd_kv = false;
  bool authenticated_asset_zero_copy = false;
  bool launch_enqueued = false;
  bool completion_observed = false;
  bool admission_only = false;
  bool caller_constructible_snapshot = false;
  bool startup_package_bound = true;
  bool execution_capability = true;
  bool current_device_matches_snapshot = false;
  bool asset_upload_device_matches_current = false;
  bool live_resource_snapshot_verified = false;
  bool caller_stream_non_null = false;
  bool stream_owner_verified = false;
  bool live_cuda_ranges_verified = false;
  bool production_dispatch_eligible = true;

  [[nodiscard]] constexpr bool valid_t1_admission_enqueue_receipt()
      const noexcept {
    const auto plan = sm87_macrofeed_v4_fp8_plan(role, token_count);
    return plan.valid() && identity == plan.identity &&
           artifact_identity != 0U && device_ordinal >= 0 &&
           logical_tasks == plan.logical_tasks &&
           physical_kernel_launches == 1U && fallback_launches == 0U &&
           ordinary_full_grid && role_specific_direct_scatter &&
           private_nhd_kv == plan.private_nhd_kv &&
           authenticated_asset_zero_copy && launch_enqueued &&
           !completion_observed && admission_only &&
           caller_constructible_snapshot && !startup_package_bound &&
           !execution_capability && current_device_matches_snapshot &&
           asset_upload_device_matches_current &&
           live_resource_snapshot_verified &&
           caller_stream_non_null && !stream_owner_verified &&
           live_cuda_ranges_verified &&
           !production_dispatch_eligible;
  }
};

struct Sm87MacroFeedV4Fp8TileTestArguments final {
  Sm87TargetAotProjectionRole role =
      Sm87TargetAotProjectionRole::kInvalid;
  std::size_t partition_index = 0U;
  std::size_t partition_n256_tile = 0U;
  const std::uint16_t* input = nullptr;
  std::size_t input_row_stride = 0U;
  std::size_t logical_input_first_k = 0U;
  const std::uint8_t* canonical_payload_four_k64_cells = nullptr;
  std::uint16_t compensated_scale_bf16_bits = 0U;
  std::size_t valid_rows = 0U;
  std::size_t canonical_n128_half = 0U;
  std::uint16_t* primary_output = nullptr;
  std::size_t primary_output_row_stride = 0U;
  std::uint16_t* key_output = nullptr;
  std::size_t key_output_row_stride = 0U;
  std::uint16_t* value_output = nullptr;
  std::size_t value_output_row_stride = 0U;
  void* cuda_stream = nullptr;
};

[[nodiscard]] bool sm87_macrofeed_v4_fp8_arguments_valid(
    const Sm87MacroFeedV4Fp8Arguments& arguments) noexcept;

[[nodiscard]] int query_sm87_macrofeed_v4_fp8_cuda_resources(
    Sm87TargetAotProjectionRole role,
    Sm87MacroFeedV4Fp8CudaResources* resources) noexcept;

[[nodiscard]] int capture_sm87_macrofeed_v4_fp8_t1_admission_snapshot_cuda(
    Sm87TargetAotProjectionRole role,
    Sm87MacroFeedV4Fp8T1AdmissionSnapshot* snapshot) noexcept;

// Explicit default-off T1 admission launch.  Its public snapshot is caller
// constructible and grants no execution authority.  This seam is not a
// startup-package-issued seal, not a production hot path, and must not be
// bound to request dispatch.  It performs no fallback, JIT, repack, autotune,
// or selector action.
[[nodiscard]] int launch_sm87_macrofeed_v4_fp8_t1_admission_cuda(
    const Sm87MacroFeedV4Fp8Arguments& arguments,
    const Sm87MacroFeedV4Fp8T1AdmissionSnapshot& snapshot,
    Sm87MacroFeedV4Fp8T1AdmissionLaunchReceipt* receipt) noexcept;

// Synthetic T1 helper only. Four canonical K64 cells exercise all three
// pipeline slots and the ring turnover. valid_rows admits M64 or a predicated
// tail in [1,63]; canonical_n128_half selects either half of each unchanged
// N256 payload cell. partition_n256_tile controls only the logical publication
// site.  O fixes logical_input_first_k=192 so four K64 cells cross a Q/G head
// boundary and prove the interleaved Q gather.  This helper has no real-model
// or performance authority.
[[nodiscard]] int launch_sm87_macrofeed_v4_fp8_tile_test_cuda(
    const Sm87MacroFeedV4Fp8TileTestArguments& arguments) noexcept;

static_assert(kSm87MacroFeedV4Fp8GridM * kSm87MacroFeedV4Fp8BlockM ==
              kSm87MacroFeedV4Fp8Tokens);
static_assert(kSm87MacroFeedV4Fp8DynamicSharedBytes == 49'152U);
static_assert(kSm87MacroFeedV4Fp8AttentionHeads *
                      kSm87MacroFeedV4Fp8AttentionHeadFeatures ==
                  kSm87MacroFeedV4Fp8FullQFeatures &&
              kSm87MacroFeedV4Fp8AttentionHeads *
                      kSm87MacroFeedV4Fp8QGateHeadStride ==
                  kSm87MacroFeedV4Fp8FullQGateFeatures);
static_assert(sm87_macrofeed_v4_fp8_interleaved_q_physical_offset(255U) ==
                  255U &&
              sm87_macrofeed_v4_fp8_interleaved_q_physical_offset(256U) ==
                  512U &&
              sm87_macrofeed_v4_fp8_interleaved_q_gate_physical_offset(
                  6'144U) == 256U &&
              sm87_macrofeed_v4_fp8_interleaved_q_gate_physical_offset(
                  6'400U) == 768U);
static_assert(sm87_macrofeed_v4_fp8_plan(
                  Sm87TargetAotProjectionRole::kFp8GdnQkvZ, 8'000U)
                  .valid());
static_assert(sm87_macrofeed_v4_fp8_plan(
                  Sm87TargetAotProjectionRole::kFp8FullQkv, 8'000U)
                  .valid());
static_assert(sm87_macrofeed_v4_fp8_plan(
                  Sm87TargetAotProjectionRole::kFp8AttentionOutput, 8'000U)
                  .valid());
static_assert(!sm87_macrofeed_v4_fp8_plan(
                   Sm87TargetAotProjectionRole::kFp8FullQkv, 7'999U)
                   .valid());
static_assert(sm87_macrofeed_v4_fp8_bias_shift_bf16_bits(0x7fU) ==
              0x07f0U);
static_assert(sm87_macrofeed_v4_fp8_bias_shift_bf16_bits(0xffU) ==
              0x87f0U);

}  // namespace q3x::kernels
