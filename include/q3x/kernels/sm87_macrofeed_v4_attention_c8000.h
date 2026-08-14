#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {

// Default-off exact Full-Attention core constituent for
// AC-PREFILL-SM87-MACROFEED-v4.  It consumes one already-preprocessed C8000
// Q/G scratch panel and the complete private NHD K/V allocation for one model
// layer.  This public T1 seam is deliberately startup-package-unbound and
// grants no selector, whole-model, numerical-qualification, or production
// authority.
enum class Sm87MacroFeedV4AttentionC8000Identity : std::uint64_t {
  kInvalid = 0U,
  kExactQ128Kv32TwoStageInPlaceV1 = 0x5133'4d46'5634'4102ULL,
};

inline constexpr Sm87MacroFeedV4AttentionC8000Identity
    kSm87MacroFeedV4AttentionC8000Identity =
        Sm87MacroFeedV4AttentionC8000Identity::
            kExactQ128Kv32TwoStageInPlaceV1;
inline constexpr std::size_t kSm87MacroFeedV4AttentionC8000Tokens = 8'000U;
inline constexpr std::size_t
    kSm87MacroFeedV4AttentionC8000MaximumPositions = 40'000U;
inline constexpr std::size_t kSm87MacroFeedV4AttentionC8000PanelCount = 5U;
inline constexpr std::size_t
    kSm87MacroFeedV4AttentionC8000ScratchRowStride = 17'408U;
inline constexpr std::size_t kSm87MacroFeedV4AttentionC8000QueryHeads = 24U;
inline constexpr std::size_t kSm87MacroFeedV4AttentionC8000KvHeads = 4U;
inline constexpr std::size_t
    kSm87MacroFeedV4AttentionC8000QueriesPerKv = 6U;
inline constexpr std::size_t
    kSm87MacroFeedV4AttentionC8000HeadDimension = 256U;
inline constexpr std::size_t
    kSm87MacroFeedV4AttentionC8000QGateHeadStride = 512U;
inline constexpr std::size_t kSm87MacroFeedV4AttentionC8000QSlotOffset = 0U;
inline constexpr std::size_t kSm87MacroFeedV4AttentionC8000GateSlotOffset =
    256U;
inline constexpr std::size_t kSm87MacroFeedV4AttentionC8000QGateSpan =
    12'288U;
inline constexpr std::size_t kSm87MacroFeedV4AttentionC8000KvRowStride =
    1'024U;
inline constexpr std::size_t kSm87MacroFeedV4AttentionC8000QueryTile = 128U;
inline constexpr std::size_t kSm87MacroFeedV4AttentionC8000KvTile = 32U;
inline constexpr std::size_t kSm87MacroFeedV4AttentionC8000Threads = 256U;
inline constexpr std::size_t kSm87MacroFeedV4AttentionC8000Warps = 8U;
inline constexpr std::size_t
    kSm87MacroFeedV4AttentionC8000QueryRowsPerWarp = 16U;
inline constexpr std::size_t
    kSm87MacroFeedV4AttentionC8000PipelineStages = 2U;
inline constexpr std::size_t kSm87MacroFeedV4AttentionC8000GridX = 375U;
inline constexpr std::size_t kSm87MacroFeedV4AttentionC8000GridY = 1U;
inline constexpr std::size_t kSm87MacroFeedV4AttentionC8000GridZ = 4U;
inline constexpr std::size_t
    kSm87MacroFeedV4AttentionC8000PhysicalCtas = 1'500U;
inline constexpr std::size_t
    kSm87MacroFeedV4AttentionC8000DynamicSharedBytes = 128U * 1'024U;
inline constexpr std::size_t
    kSm87MacroFeedV4AttentionC8000SplitKvWorkspaceBytes = 0U;
inline constexpr std::size_t
    kSm87MacroFeedV4AttentionC8000PointerAlignment = 16U;
inline constexpr std::size_t kSm87MacroFeedV4AttentionC8000SmCount = 16U;
inline constexpr std::size_t
    kSm87MacroFeedV4AttentionC8000MaximumRegisters = 255U;
inline constexpr std::size_t
    kSm87MacroFeedV4AttentionC8000RequiredCtasPerSm = 1U;

inline constexpr std::uint64_t kSm87MacroFeedV4AttentionC8000ScratchBytes =
    kSm87MacroFeedV4AttentionC8000Tokens *
    kSm87MacroFeedV4AttentionC8000ScratchRowStride *
    sizeof(std::uint16_t);
inline constexpr std::uint64_t kSm87MacroFeedV4AttentionC8000KvCacheBytes =
    kSm87MacroFeedV4AttentionC8000MaximumPositions *
    kSm87MacroFeedV4AttentionC8000KvRowStride * sizeof(std::uint16_t);

[[nodiscard]] constexpr bool
sm87_macrofeed_v4_attention_c8000_first_position_supported(
    const std::size_t first_position) noexcept {
  return first_position % kSm87MacroFeedV4AttentionC8000Tokens == 0U &&
         first_position / kSm87MacroFeedV4AttentionC8000Tokens <
             kSm87MacroFeedV4AttentionC8000PanelCount;
}

[[nodiscard]] constexpr std::size_t
sm87_macrofeed_v4_attention_c8000_q_physical_offset(
    const std::size_t token, const std::size_t query_head,
    const std::size_t dimension) noexcept {
  return token * kSm87MacroFeedV4AttentionC8000ScratchRowStride +
         query_head * kSm87MacroFeedV4AttentionC8000QGateHeadStride +
         kSm87MacroFeedV4AttentionC8000QSlotOffset + dimension;
}

[[nodiscard]] constexpr std::size_t
sm87_macrofeed_v4_attention_c8000_gate_physical_offset(
    const std::size_t token, const std::size_t query_head,
    const std::size_t dimension) noexcept {
  return token * kSm87MacroFeedV4AttentionC8000ScratchRowStride +
         query_head * kSm87MacroFeedV4AttentionC8000QGateHeadStride +
         kSm87MacroFeedV4AttentionC8000GateSlotOffset + dimension;
}

struct Sm87MacroFeedV4AttentionC8000Plan final {
  Sm87MacroFeedV4AttentionC8000Identity identity =
      Sm87MacroFeedV4AttentionC8000Identity::kInvalid;
  std::size_t first_position = 0U;
  std::size_t token_count = 0U;
  std::size_t ready_end = 0U;
  std::size_t scratch_row_stride = 0U;
  std::size_t kv_position_capacity = 0U;
  std::size_t kv_row_stride = 0U;
  std::size_t query_tile = 0U;
  std::size_t kv_tile = 0U;
  std::size_t grid_x = 0U;
  std::size_t grid_y = 0U;
  std::size_t grid_z = 0U;
  std::size_t threads_per_cta = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  std::size_t split_kv_workspace_bytes = 1U;
  std::uint32_t physical_kernel_launches = 0U;
  bool exact_causal = false;
  bool online_softmax = false;
  bool q64_subgroup_reduction_order = false;
  bool q128_fully_staged_before_store = false;
  bool q_output_aliases_q_input = false;
  bool gate_fused_after_bf16_attention = false;
  bool gate_slots_preserved = false;
  bool scratch_gap_preserved = false;
  bool private_nhd_kv_allocation_origin = false;
  bool kv32_ascending_two_stage = false;
  bool partition_kv = true;
  bool merge_kernel_present = true;
  bool temporary_output_present = true;
  bool default_off = false;
  bool selector_present = true;
  bool fallback_permitted = true;
  bool numerical_contract_qualified = true;
  bool production_dispatch_eligible = true;
  bool startup_package_unbound = true;
  bool execution_capability = false;
  bool caller_snapshot_grants_production_authority = false;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return identity == kSm87MacroFeedV4AttentionC8000Identity &&
           sm87_macrofeed_v4_attention_c8000_first_position_supported(
               first_position) &&
           token_count == kSm87MacroFeedV4AttentionC8000Tokens &&
           ready_end == first_position + token_count &&
           ready_end <= kSm87MacroFeedV4AttentionC8000MaximumPositions &&
           scratch_row_stride ==
               kSm87MacroFeedV4AttentionC8000ScratchRowStride &&
           kv_position_capacity ==
               kSm87MacroFeedV4AttentionC8000MaximumPositions &&
           kv_row_stride == kSm87MacroFeedV4AttentionC8000KvRowStride &&
           query_tile == kSm87MacroFeedV4AttentionC8000QueryTile &&
           kv_tile == kSm87MacroFeedV4AttentionC8000KvTile &&
           grid_x == kSm87MacroFeedV4AttentionC8000GridX &&
           grid_y == kSm87MacroFeedV4AttentionC8000GridY &&
           grid_z == kSm87MacroFeedV4AttentionC8000GridZ &&
           threads_per_cta == kSm87MacroFeedV4AttentionC8000Threads &&
           dynamic_shared_bytes ==
               kSm87MacroFeedV4AttentionC8000DynamicSharedBytes &&
           split_kv_workspace_bytes == 0U && physical_kernel_launches == 1U &&
           exact_causal && online_softmax && q64_subgroup_reduction_order &&
           q128_fully_staged_before_store && q_output_aliases_q_input &&
           gate_fused_after_bf16_attention && gate_slots_preserved &&
           scratch_gap_preserved && private_nhd_kv_allocation_origin &&
           kv32_ascending_two_stage && !partition_kv &&
           !merge_kernel_present && !temporary_output_present && default_off &&
           !selector_present && !fallback_permitted &&
           !numerical_contract_qualified && !production_dispatch_eligible &&
           startup_package_unbound && !execution_capability &&
           !caller_snapshot_grants_production_authority;
  }
};

[[nodiscard]] constexpr Sm87MacroFeedV4AttentionC8000Plan
sm87_macrofeed_v4_attention_c8000_plan(
    const std::size_t first_position, const std::size_t token_count) noexcept {
  if (token_count != kSm87MacroFeedV4AttentionC8000Tokens ||
      !sm87_macrofeed_v4_attention_c8000_first_position_supported(
          first_position) ||
      first_position > kSm87MacroFeedV4AttentionC8000MaximumPositions -
                           token_count) {
    return {};
  }
  return {kSm87MacroFeedV4AttentionC8000Identity,
          first_position,
          token_count,
          first_position + token_count,
          kSm87MacroFeedV4AttentionC8000ScratchRowStride,
          kSm87MacroFeedV4AttentionC8000MaximumPositions,
          kSm87MacroFeedV4AttentionC8000KvRowStride,
          kSm87MacroFeedV4AttentionC8000QueryTile,
          kSm87MacroFeedV4AttentionC8000KvTile,
          kSm87MacroFeedV4AttentionC8000GridX,
          kSm87MacroFeedV4AttentionC8000GridY,
          kSm87MacroFeedV4AttentionC8000GridZ,
          kSm87MacroFeedV4AttentionC8000Threads,
          kSm87MacroFeedV4AttentionC8000DynamicSharedBytes,
          0U,
          1U,
          true,
          true,
          true,
          true,
          true,
          true,
          true,
          true,
          true,
          true,
          false,
          false,
          false,
          true,
          false,
          false,
          false,
          false,
          true,
          false,
          false};
}

struct Sm87MacroFeedV4AttentionC8000ByteRange final {
  std::uintptr_t begin = 0U;
  std::uintptr_t end = 0U;
  bool valid = false;
};

[[nodiscard]] constexpr Sm87MacroFeedV4AttentionC8000ByteRange
sm87_macrofeed_v4_attention_c8000_byte_range(
    const void* const pointer, const std::uint64_t bytes) noexcept {
  if (pointer == nullptr || bytes == 0U ||
      bytes > std::numeric_limits<std::uintptr_t>::max()) {
    return {};
  }
  const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
  const auto width = static_cast<std::uintptr_t>(bytes);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - width) {
    return {};
  }
  return {begin, begin + width, true};
}

[[nodiscard]] constexpr bool sm87_macrofeed_v4_attention_c8000_ranges_disjoint(
    const Sm87MacroFeedV4AttentionC8000ByteRange& left,
    const Sm87MacroFeedV4AttentionC8000ByteRange& right) noexcept {
  return left.valid && right.valid &&
         (left.end <= right.begin || right.end <= left.begin);
}

struct Sm87MacroFeedV4AttentionC8000Arguments final {
  // One mutable [8000,17408] BF16 plane.  Each head is [Q256,Gate256].
  // The core overwrites only Q slots after the complete CTA-owned Q128 tile
  // has reached shared memory; Gate slots and [12288,17408) remain unchanged.
  std::uint16_t* q_gate_scratch = nullptr;
  std::size_t token_count = 0U;
  std::size_t scratch_row_stride = 0U;
  // Logical allocation origins, not append-slice pointers.  Both caches are
  // physically [40000,4,256] NHD BF16 and contain the complete ready prefix.
  const std::uint16_t* key_cache = nullptr;
  const std::uint16_t* value_cache = nullptr;
  std::size_t kv_position_capacity = 0U;
  std::size_t kv_row_stride = 0U;
  std::size_t first_position = 0U;
  // Caller precondition: Full-QKV and in-place Q/K preprocessing are ordered
  // before this launch on the supplied live stream.  This T1 seam observes
  // the stream device but does not authenticate owner-issued events.
  void* cuda_stream = nullptr;
};

[[nodiscard]] bool sm87_macrofeed_v4_attention_c8000_arguments_valid(
    const Sm87MacroFeedV4AttentionC8000Arguments& arguments) noexcept;

struct Sm87MacroFeedV4AttentionC8000KernelResources final {
  std::int32_t registers_per_thread = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  std::int32_t maximum_threads_per_block = 0;
  std::int32_t active_blocks_per_sm = 0;
  std::int32_t threads_per_block = 0;
  std::int32_t grid_x = 0;
  std::int32_t grid_y = 0;
  std::int32_t grid_z = 0;
  std::int32_t physical_grid_ctas = 0;
};

// Caller-fillable T0/T1 evidence only.  Launch re-observes the current device
// and kernel and requires an exact field-for-field match before enqueue.
struct Sm87MacroFeedV4AttentionC8000AdmissionResourceSnapshot final {
  Sm87MacroFeedV4AttentionC8000Identity identity =
      Sm87MacroFeedV4AttentionC8000Identity::kInvalid;
  std::int32_t device_ordinal = -1;
  std::int32_t compute_major = 0;
  std::int32_t compute_minor = 0;
  std::int32_t sm_count = 0;
  std::int32_t binary_version = 0;
  Sm87MacroFeedV4AttentionC8000KernelResources kernel{};
  bool kernel_compiled = false;
  bool exact_geometry = false;
  bool static_resource_gate_passed = false;
  bool numerical_contract_qualified = false;
  bool production_dispatch_eligible = false;
  bool startup_package_unbound = true;
  bool execution_capability = false;
  bool caller_snapshot_grants_production_authority = false;
};

[[nodiscard]] constexpr bool
sm87_macrofeed_v4_attention_c8000_admission_resource_gate(
    const Sm87MacroFeedV4AttentionC8000AdmissionResourceSnapshot&
        resources) noexcept {
  return resources.identity == kSm87MacroFeedV4AttentionC8000Identity &&
         resources.device_ordinal >= 0 && resources.compute_major == 8 &&
         resources.compute_minor == 7 &&
         resources.sm_count ==
             static_cast<std::int32_t>(
                 kSm87MacroFeedV4AttentionC8000SmCount) &&
         resources.binary_version == 87 && resources.kernel_compiled &&
         resources.exact_geometry && resources.static_resource_gate_passed &&
         resources.kernel.registers_per_thread > 0 &&
         resources.kernel.registers_per_thread <=
             static_cast<std::int32_t>(
                 kSm87MacroFeedV4AttentionC8000MaximumRegisters) &&
         resources.kernel.static_shared_bytes == 0U &&
         resources.kernel.dynamic_shared_bytes ==
             kSm87MacroFeedV4AttentionC8000DynamicSharedBytes &&
         resources.kernel.local_bytes == 0U &&
         resources.kernel.maximum_threads_per_block >=
             static_cast<std::int32_t>(
                 kSm87MacroFeedV4AttentionC8000Threads) &&
         resources.kernel.active_blocks_per_sm ==
             static_cast<std::int32_t>(
                 kSm87MacroFeedV4AttentionC8000RequiredCtasPerSm) &&
         resources.kernel.threads_per_block ==
             static_cast<std::int32_t>(
                 kSm87MacroFeedV4AttentionC8000Threads) &&
         resources.kernel.grid_x ==
             static_cast<std::int32_t>(
                 kSm87MacroFeedV4AttentionC8000GridX) &&
         resources.kernel.grid_y ==
             static_cast<std::int32_t>(
                 kSm87MacroFeedV4AttentionC8000GridY) &&
         resources.kernel.grid_z ==
             static_cast<std::int32_t>(
                 kSm87MacroFeedV4AttentionC8000GridZ) &&
         resources.kernel.physical_grid_ctas ==
             static_cast<std::int32_t>(
                 kSm87MacroFeedV4AttentionC8000PhysicalCtas) &&
         !resources.numerical_contract_qualified &&
         !resources.production_dispatch_eligible &&
         resources.startup_package_unbound && !resources.execution_capability &&
         !resources.caller_snapshot_grants_production_authority;
}

[[nodiscard]] int
query_sm87_macrofeed_v4_attention_c8000_admission_resources_cuda(
    Sm87MacroFeedV4AttentionC8000AdmissionResourceSnapshot* resources) noexcept;

struct Sm87MacroFeedV4AttentionC8000AdmissionLaunchReceipt final {
  Sm87MacroFeedV4AttentionC8000Identity identity =
      Sm87MacroFeedV4AttentionC8000Identity::kInvalid;
  std::int32_t device_ordinal = -1;
  std::size_t first_position = 0U;
  std::size_t token_count = 0U;
  std::size_t ready_end = 0U;
  std::size_t scratch_row_stride = 0U;
  std::size_t kv_position_capacity = 0U;
  std::size_t kv_row_stride = 0U;
  std::size_t split_kv_workspace_bytes = 1U;
  std::uint32_t physical_kernel_launches = 0U;
  bool exact_causal = false;
  bool online_softmax = false;
  bool q64_subgroup_reduction_order = false;
  bool q128_fully_staged_before_store = false;
  bool q_output_aliases_q_input = false;
  bool gate_fused_after_bf16_attention = false;
  bool gate_slots_preserved = false;
  bool scratch_gap_preserved = false;
  bool private_nhd_kv_allocation_origin = false;
  bool kv32_ascending_two_stage = false;
  bool partition_kv = true;
  bool merge_kernel_present = true;
  bool temporary_output_present = true;
  bool current_device_revalidated = false;
  bool caller_snapshot_exact_observed_match = false;
  bool device_allocation_ranges_owned = false;
  bool caller_stream_non_null = false;
  bool live_stream_device_observed = false;
  bool stream_owner_verified = false;
  bool preprocess_completion_event_bound = false;
  bool launch_enqueued = false;
  bool completion_observed = false;
  bool numerical_contract_qualified = true;
  bool production_dispatch_eligible = true;
  bool startup_package_unbound = true;
  bool execution_capability = false;
  bool caller_snapshot_grants_production_authority = false;

  [[nodiscard]] constexpr bool valid_enqueue_receipt() const noexcept {
    const auto plan = sm87_macrofeed_v4_attention_c8000_plan(first_position,
                                                             token_count);
    return plan.valid() && identity == plan.identity && device_ordinal >= 0 &&
           ready_end == plan.ready_end &&
           scratch_row_stride == plan.scratch_row_stride &&
           kv_position_capacity == plan.kv_position_capacity &&
           kv_row_stride == plan.kv_row_stride &&
           split_kv_workspace_bytes == 0U && physical_kernel_launches == 1U &&
           exact_causal && online_softmax && q64_subgroup_reduction_order &&
           q128_fully_staged_before_store && q_output_aliases_q_input &&
           gate_fused_after_bf16_attention && gate_slots_preserved &&
           scratch_gap_preserved && private_nhd_kv_allocation_origin &&
           kv32_ascending_two_stage && !partition_kv &&
           !merge_kernel_present && !temporary_output_present &&
           current_device_revalidated &&
           caller_snapshot_exact_observed_match &&
           device_allocation_ranges_owned && caller_stream_non_null &&
           live_stream_device_observed && !stream_owner_verified &&
           !preprocess_completion_event_bound && launch_enqueued &&
           !completion_observed && !numerical_contract_qualified &&
           !production_dispatch_eligible && startup_package_unbound &&
           !execution_capability &&
           !caller_snapshot_grants_production_authority;
  }
};

[[nodiscard]] int launch_sm87_macrofeed_v4_attention_c8000_admission_cuda(
    const Sm87MacroFeedV4AttentionC8000Arguments& arguments,
    const Sm87MacroFeedV4AttentionC8000AdmissionResourceSnapshot& resources,
    Sm87MacroFeedV4AttentionC8000AdmissionLaunchReceipt* receipt) noexcept;

// Bounded correctness seam.  The exact same Q128/KV32 two-stage body is used
// with tail predicates.  output_q_gate_scratch may equal q_gate_scratch
// exactly (the production alias) or be a disjoint V4-layout plane; partial
// overlap is rejected.  Only token counts 1 and 65 are admitted so this seam
// cannot acquire performance or whole-model authority.
struct Sm87MacroFeedV4AttentionC8000OracleArguments final {
  const std::uint16_t* q_gate_scratch = nullptr;
  std::uint16_t* output_q_gate_scratch = nullptr;
  std::size_t token_count = 0U;
  std::size_t scratch_row_stride = 0U;
  const std::uint16_t* key_cache = nullptr;
  const std::uint16_t* value_cache = nullptr;
  std::size_t kv_position_capacity = 0U;
  std::size_t kv_row_stride = 0U;
  std::size_t first_position = 0U;
  void* cuda_stream = nullptr;
};

[[nodiscard]] int launch_sm87_macrofeed_v4_attention_c8000_oracle_cuda(
    const Sm87MacroFeedV4AttentionC8000OracleArguments& arguments) noexcept;

static_assert(kSm87MacroFeedV4AttentionC8000QueryHeads *
                      kSm87MacroFeedV4AttentionC8000QGateHeadStride ==
                  kSm87MacroFeedV4AttentionC8000QGateSpan &&
              kSm87MacroFeedV4AttentionC8000QueryHeads ==
                  kSm87MacroFeedV4AttentionC8000KvHeads *
                      kSm87MacroFeedV4AttentionC8000QueriesPerKv &&
              kSm87MacroFeedV4AttentionC8000KvRowStride ==
                  kSm87MacroFeedV4AttentionC8000KvHeads *
                      kSm87MacroFeedV4AttentionC8000HeadDimension &&
              kSm87MacroFeedV4AttentionC8000GridX *
                      kSm87MacroFeedV4AttentionC8000QueryTile ==
                  kSm87MacroFeedV4AttentionC8000Tokens *
                      kSm87MacroFeedV4AttentionC8000QueriesPerKv &&
              kSm87MacroFeedV4AttentionC8000PhysicalCtas ==
                  kSm87MacroFeedV4AttentionC8000GridX *
                      kSm87MacroFeedV4AttentionC8000GridZ &&
              kSm87MacroFeedV4AttentionC8000ScratchBytes == 278'528'000U &&
              kSm87MacroFeedV4AttentionC8000KvCacheBytes == 81'920'000U);
static_assert(sm87_macrofeed_v4_attention_c8000_plan(0U, 8'000U).valid());
static_assert(
    sm87_macrofeed_v4_attention_c8000_plan(32'000U, 8'000U).valid());
static_assert(
    !sm87_macrofeed_v4_attention_c8000_plan(1U, 8'000U).valid());

}  // namespace q3x::kernels
