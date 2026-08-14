#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {

// Default-off C8000 Full-Attention preprocessing constituent of
// AC-PREFILL-SM87-MACROFEED-v4.  It owns no selector and grants no startup,
// whole-model, numerical-qualification, or production execution authority.
enum class Sm87MacroFeedV4FullAttentionPreprocessIdentity : std::uint64_t {
  kInvalid = 0U,
  kInPlaceQKPromptWide128ExactTreeV1 = 0x5133'4d46'5634'4101ULL,
};

inline constexpr Sm87MacroFeedV4FullAttentionPreprocessIdentity
    kSm87MacroFeedV4FullAttentionPreprocessIdentity =
        Sm87MacroFeedV4FullAttentionPreprocessIdentity::
            kInPlaceQKPromptWide128ExactTreeV1;
inline constexpr std::size_t
    kSm87MacroFeedV4FullAttentionPreprocessTokens = 8'000U;
inline constexpr std::size_t
    kSm87MacroFeedV4FullAttentionPreprocessMaximumPositions = 40'000U;
inline constexpr std::size_t
    kSm87MacroFeedV4FullAttentionPreprocessPanelCount = 5U;
inline constexpr std::size_t
    kSm87MacroFeedV4FullAttentionPreprocessScratchRowStride = 17'408U;
inline constexpr std::size_t
    kSm87MacroFeedV4FullAttentionPreprocessQueryHeads = 24U;
inline constexpr std::size_t
    kSm87MacroFeedV4FullAttentionPreprocessLogicalKvHeads = 4U;
inline constexpr std::size_t
    kSm87MacroFeedV4FullAttentionPreprocessPhysicalKvHeads = 8U;
inline constexpr std::size_t
    kSm87MacroFeedV4FullAttentionPreprocessHeadDimension = 256U;
inline constexpr std::size_t
    kSm87MacroFeedV4FullAttentionPreprocessPhysicalKvHeadDimension = 128U;
inline constexpr std::size_t
    kSm87MacroFeedV4FullAttentionPreprocessQGateHeadStride = 512U;
inline constexpr std::size_t
    kSm87MacroFeedV4FullAttentionPreprocessQGateSpan = 12'288U;
inline constexpr std::size_t
    kSm87MacroFeedV4FullAttentionPreprocessKeyRowStride = 1'024U;
inline constexpr std::size_t
    kSm87MacroFeedV4FullAttentionPreprocessRotaryDimension = 64U;
inline constexpr std::size_t
    kSm87MacroFeedV4FullAttentionPreprocessRotaryHalf = 32U;
inline constexpr std::size_t
    kSm87MacroFeedV4FullAttentionPreprocessThreads = 128U;
inline constexpr std::size_t
    kSm87MacroFeedV4FullAttentionPreprocessCombinedHeads = 28U;
inline constexpr std::size_t
    kSm87MacroFeedV4FullAttentionPreprocessPhysicalCtas = 224'000U;
inline constexpr std::size_t
    kSm87MacroFeedV4FullAttentionPreprocessStaticSharedBytes = 516U;
inline constexpr std::size_t
    kSm87MacroFeedV4FullAttentionPreprocessRequiredCtasPerSm = 4U;
inline constexpr std::size_t
    kSm87MacroFeedV4FullAttentionPreprocessMaximumRegisters = 128U;
inline constexpr std::size_t
    kSm87MacroFeedV4FullAttentionPreprocessPointerAlignment = 16U;
inline constexpr std::size_t
    kSm87MacroFeedV4FullAttentionPreprocessSmCount = 16U;
inline constexpr std::uint32_t
    kSm87MacroFeedV4FullAttentionPreprocessEpsilonFp32Bits = 0x3586'37bdU;

inline constexpr std::uint64_t
    kSm87MacroFeedV4FullAttentionPreprocessScratchBytes =
        kSm87MacroFeedV4FullAttentionPreprocessTokens *
        kSm87MacroFeedV4FullAttentionPreprocessScratchRowStride *
        sizeof(std::uint16_t);
inline constexpr std::uint64_t
    kSm87MacroFeedV4FullAttentionPreprocessKeyCacheBytes =
        kSm87MacroFeedV4FullAttentionPreprocessMaximumPositions *
        kSm87MacroFeedV4FullAttentionPreprocessKeyRowStride *
        sizeof(std::uint16_t);
inline constexpr std::uint64_t
    kSm87MacroFeedV4FullAttentionPreprocessNormWeightBytes =
        kSm87MacroFeedV4FullAttentionPreprocessHeadDimension *
        sizeof(std::uint16_t);
inline constexpr std::uint64_t
    kSm87MacroFeedV4FullAttentionPreprocessRopeTableBytes =
        kSm87MacroFeedV4FullAttentionPreprocessMaximumPositions *
        kSm87MacroFeedV4FullAttentionPreprocessRotaryHalf * sizeof(float);

[[nodiscard]] constexpr bool
sm87_macrofeed_v4_full_attention_preprocess_first_position_supported(
    const std::size_t first_position) noexcept {
  return first_position %
                     kSm87MacroFeedV4FullAttentionPreprocessTokens ==
                 0U &&
         first_position /
                 kSm87MacroFeedV4FullAttentionPreprocessTokens <
             kSm87MacroFeedV4FullAttentionPreprocessPanelCount;
}

struct Sm87MacroFeedV4FullAttentionPreprocessPlan final {
  Sm87MacroFeedV4FullAttentionPreprocessIdentity identity =
      Sm87MacroFeedV4FullAttentionPreprocessIdentity::kInvalid;
  std::size_t first_position = 0U;
  std::size_t token_count = 0U;
  std::size_t scratch_row_stride = 0U;
  std::size_t key_cache_position_capacity = 0U;
  std::size_t key_cache_row_stride = 0U;
  std::size_t rope_position_capacity = 0U;
  std::size_t rope_row_stride = 0U;
  std::size_t grid_x = 0U;
  std::size_t grid_y = 0U;
  std::size_t threads_per_cta = 0U;
  std::uint32_t epsilon_fp32_bits = 0U;
  std::uint32_t physical_kernel_launches = 0U;
  bool centered_qk_rmsnorm = false;
  bool exact_prompt_wide_reduction_tree = false;
  bool bf16_boundary_before_rope = false;
  bool partial_d64_neox_rope = false;
  bool q_in_place = false;
  bool k_in_place = false;
  bool gate_bitwise_preserved = false;
  bool scratch_gap_bitwise_preserved = false;
  bool private_nhd_key_cache = false;
  bool value_cache_addressable = true;
  bool default_off = false;
  bool selector_present = true;
  bool fallback_permitted = true;
  bool numerical_contract_qualified = true;
  bool production_dispatch_eligible = true;
  bool startup_package_unbound = true;
  bool execution_capability = false;
  bool caller_snapshot_grants_production_authority = false;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return identity ==
               kSm87MacroFeedV4FullAttentionPreprocessIdentity &&
           sm87_macrofeed_v4_full_attention_preprocess_first_position_supported(
               first_position) &&
           token_count ==
               kSm87MacroFeedV4FullAttentionPreprocessTokens &&
           first_position <=
               kSm87MacroFeedV4FullAttentionPreprocessMaximumPositions -
                   token_count &&
           scratch_row_stride ==
               kSm87MacroFeedV4FullAttentionPreprocessScratchRowStride &&
           key_cache_position_capacity ==
               kSm87MacroFeedV4FullAttentionPreprocessMaximumPositions &&
           key_cache_row_stride ==
               kSm87MacroFeedV4FullAttentionPreprocessKeyRowStride &&
           rope_position_capacity ==
               kSm87MacroFeedV4FullAttentionPreprocessMaximumPositions &&
           rope_row_stride ==
               kSm87MacroFeedV4FullAttentionPreprocessRotaryHalf &&
           grid_x == token_count &&
           grid_y ==
               kSm87MacroFeedV4FullAttentionPreprocessCombinedHeads &&
           threads_per_cta ==
               kSm87MacroFeedV4FullAttentionPreprocessThreads &&
           epsilon_fp32_bits ==
               kSm87MacroFeedV4FullAttentionPreprocessEpsilonFp32Bits &&
           physical_kernel_launches == 1U && centered_qk_rmsnorm &&
           exact_prompt_wide_reduction_tree && bf16_boundary_before_rope &&
           partial_d64_neox_rope && q_in_place && k_in_place &&
           gate_bitwise_preserved && scratch_gap_bitwise_preserved &&
           private_nhd_key_cache && !value_cache_addressable && default_off &&
           !selector_present && !fallback_permitted &&
           !numerical_contract_qualified &&
           !production_dispatch_eligible && startup_package_unbound &&
           !execution_capability &&
           !caller_snapshot_grants_production_authority;
  }
};

[[nodiscard]] constexpr Sm87MacroFeedV4FullAttentionPreprocessPlan
sm87_macrofeed_v4_full_attention_preprocess_plan(
    const std::size_t first_position,
    const std::size_t token_count) noexcept {
  if (token_count != kSm87MacroFeedV4FullAttentionPreprocessTokens ||
      !sm87_macrofeed_v4_full_attention_preprocess_first_position_supported(
          first_position) ||
      first_position >
          kSm87MacroFeedV4FullAttentionPreprocessMaximumPositions -
              token_count) {
    return {};
  }
  Sm87MacroFeedV4FullAttentionPreprocessPlan plan;
  plan.identity = kSm87MacroFeedV4FullAttentionPreprocessIdentity;
  plan.first_position = first_position;
  plan.token_count = token_count;
  plan.scratch_row_stride =
      kSm87MacroFeedV4FullAttentionPreprocessScratchRowStride;
  plan.key_cache_position_capacity =
      kSm87MacroFeedV4FullAttentionPreprocessMaximumPositions;
  plan.key_cache_row_stride =
      kSm87MacroFeedV4FullAttentionPreprocessKeyRowStride;
  plan.rope_position_capacity =
      kSm87MacroFeedV4FullAttentionPreprocessMaximumPositions;
  plan.rope_row_stride =
      kSm87MacroFeedV4FullAttentionPreprocessRotaryHalf;
  plan.grid_x = token_count;
  plan.grid_y = kSm87MacroFeedV4FullAttentionPreprocessCombinedHeads;
  plan.threads_per_cta =
      kSm87MacroFeedV4FullAttentionPreprocessThreads;
  plan.epsilon_fp32_bits =
      kSm87MacroFeedV4FullAttentionPreprocessEpsilonFp32Bits;
  plan.physical_kernel_launches = 1U;
  plan.centered_qk_rmsnorm = true;
  plan.exact_prompt_wide_reduction_tree = true;
  plan.bf16_boundary_before_rope = true;
  plan.partial_d64_neox_rope = true;
  plan.q_in_place = true;
  plan.k_in_place = true;
  plan.gate_bitwise_preserved = true;
  plan.scratch_gap_bitwise_preserved = true;
  plan.private_nhd_key_cache = true;
  plan.value_cache_addressable = false;
  plan.default_off = true;
  plan.selector_present = false;
  plan.fallback_permitted = false;
  plan.numerical_contract_qualified = false;
  plan.production_dispatch_eligible = false;
  plan.startup_package_unbound = true;
  plan.execution_capability = false;
  plan.caller_snapshot_grants_production_authority = false;
  return plan;
}

struct Sm87MacroFeedV4FullAttentionPreprocessByteRange final {
  std::uintptr_t begin = 0U;
  std::uintptr_t end = 0U;
  bool valid = false;
};

[[nodiscard]] constexpr Sm87MacroFeedV4FullAttentionPreprocessByteRange
sm87_macrofeed_v4_full_attention_preprocess_byte_range(
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

[[nodiscard]] constexpr bool
sm87_macrofeed_v4_full_attention_preprocess_ranges_disjoint(
    const Sm87MacroFeedV4FullAttentionPreprocessByteRange& left,
    const Sm87MacroFeedV4FullAttentionPreprocessByteRange& right) noexcept {
  return left.valid && right.valid &&
         (left.end <= right.begin || right.end <= left.begin);
}

struct Sm87MacroFeedV4FullAttentionPreprocessArguments final {
  // Local C8000 scratch: [token,17408], with 24 contiguous [Q256,Gate256]
  // head slots in [0,12288).  Q is overwritten in place; Gate and the
  // [12288,17408) row gap remain bitwise unchanged.
  std::uint16_t* q_gate_scratch = nullptr;
  std::size_t token_count = 0U;
  std::size_t scratch_row_stride = 0U;
  // Full private cache base, physically [40000,8,128] and logically
  // [40000,4,256].  Only [first_position,first_position+token_count) changes.
  std::uint16_t* key_cache = nullptr;
  std::size_t key_cache_position_capacity = 0U;
  std::size_t key_cache_row_stride = 0U;
  const std::uint16_t* q_norm_weight = nullptr;  // [256], centered gamma.
  const std::uint16_t* k_norm_weight = nullptr;  // [256], centered gamma.
  const float* cosines = nullptr;                // [40000,32].
  const float* sines = nullptr;                  // [40000,32].
  std::size_t rope_position_capacity = 0U;
  std::size_t rope_row_stride = 0U;
  std::size_t first_position = 0U;
  std::uint32_t epsilon_fp32_bits = 0U;
  // T1 caller precondition only.  This public seam does not authenticate
  // arbitrary or stale opaque stream handles; private startup binding owns
  // that future capability boundary.
  void* cuda_stream = nullptr;
};

[[nodiscard]] bool sm87_macrofeed_v4_full_attention_preprocess_arguments_valid(
    const Sm87MacroFeedV4FullAttentionPreprocessArguments& arguments) noexcept;

struct Sm87MacroFeedV4FullAttentionPreprocessKernelResources final {
  std::int32_t registers_per_thread = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  std::int32_t maximum_threads_per_block = 0;
  std::int32_t active_blocks_per_sm = 0;
  std::int32_t threads_per_block = 0;
  std::int32_t grid_x = 0;
  std::int32_t grid_y = 0;
  std::int32_t physical_grid_ctas = 0;
};

// Caller-fillable T0/T1 evidence only.  Launch re-queries the current device
// and kernel and requires an exact field-for-field match before enqueue.
struct Sm87MacroFeedV4FullAttentionPreprocessAdmissionResourceSnapshot final {
  Sm87MacroFeedV4FullAttentionPreprocessIdentity identity =
      Sm87MacroFeedV4FullAttentionPreprocessIdentity::kInvalid;
  std::int32_t device_ordinal = -1;
  std::int32_t compute_major = 0;
  std::int32_t compute_minor = 0;
  std::int32_t sm_count = 0;
  std::int32_t binary_version = 0;
  Sm87MacroFeedV4FullAttentionPreprocessKernelResources kernel{};
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
sm87_macrofeed_v4_full_attention_preprocess_admission_resource_gate(
    const Sm87MacroFeedV4FullAttentionPreprocessAdmissionResourceSnapshot&
        resources) noexcept {
  return resources.identity ==
             kSm87MacroFeedV4FullAttentionPreprocessIdentity &&
         resources.device_ordinal >= 0 && resources.compute_major == 8 &&
         resources.compute_minor == 7 &&
         resources.sm_count == static_cast<std::int32_t>(
                                   kSm87MacroFeedV4FullAttentionPreprocessSmCount) &&
         resources.binary_version == 87 && resources.kernel_compiled &&
         resources.exact_geometry && resources.static_resource_gate_passed &&
         resources.kernel.registers_per_thread > 0 &&
         resources.kernel.registers_per_thread <=
             static_cast<std::int32_t>(
                 kSm87MacroFeedV4FullAttentionPreprocessMaximumRegisters) &&
         resources.kernel.static_shared_bytes ==
             kSm87MacroFeedV4FullAttentionPreprocessStaticSharedBytes &&
         resources.kernel.local_bytes == 0U &&
         resources.kernel.maximum_threads_per_block >=
             static_cast<std::int32_t>(
                 kSm87MacroFeedV4FullAttentionPreprocessThreads) &&
         resources.kernel.active_blocks_per_sm >=
             static_cast<std::int32_t>(
                 kSm87MacroFeedV4FullAttentionPreprocessRequiredCtasPerSm) &&
         resources.kernel.threads_per_block ==
             static_cast<std::int32_t>(
                 kSm87MacroFeedV4FullAttentionPreprocessThreads) &&
         resources.kernel.grid_x ==
             static_cast<std::int32_t>(
                 kSm87MacroFeedV4FullAttentionPreprocessTokens) &&
         resources.kernel.grid_y ==
             static_cast<std::int32_t>(
                 kSm87MacroFeedV4FullAttentionPreprocessCombinedHeads) &&
         resources.kernel.physical_grid_ctas ==
             static_cast<std::int32_t>(
                 kSm87MacroFeedV4FullAttentionPreprocessPhysicalCtas) &&
         !resources.numerical_contract_qualified &&
         !resources.production_dispatch_eligible &&
         resources.startup_package_unbound && !resources.execution_capability &&
         !resources.caller_snapshot_grants_production_authority;
}

[[nodiscard]] int
query_sm87_macrofeed_v4_full_attention_preprocess_admission_resources_cuda(
    Sm87MacroFeedV4FullAttentionPreprocessAdmissionResourceSnapshot*
        resources) noexcept;

// Public structural enqueue observation only.  It is caller-constructible,
// is not a completion receipt, and cannot impersonate a private startup
// package or a production execution capability.
struct Sm87MacroFeedV4FullAttentionPreprocessAdmissionLaunchReceipt final {
  Sm87MacroFeedV4FullAttentionPreprocessIdentity identity =
      Sm87MacroFeedV4FullAttentionPreprocessIdentity::kInvalid;
  std::int32_t device_ordinal = -1;
  std::size_t first_position = 0U;
  std::size_t token_count = 0U;
  std::size_t scratch_row_stride = 0U;
  std::uint32_t physical_kernel_launches = 0U;
  bool centered_qk_rmsnorm = false;
  bool exact_prompt_wide_reduction_tree = false;
  bool bf16_boundary_before_rope = false;
  bool partial_d64_neox_rope = false;
  bool q_in_place = false;
  bool k_in_place = false;
  bool gate_bitwise_preserved = false;
  bool scratch_gap_bitwise_preserved = false;
  bool private_nhd_key_cache = false;
  bool value_cache_unaddressable = false;
  bool current_device_revalidated = false;
  bool caller_snapshot_exact_observed_match = false;
  bool device_allocation_ranges_owned = false;
  bool caller_stream_non_null = false;
  bool stream_owner_verified = false;
  bool launch_enqueued = false;
  bool completion_observed = false;
  bool numerical_contract_qualified = true;
  bool production_dispatch_eligible = true;
  bool startup_package_unbound = true;
  bool execution_capability = false;
  bool caller_snapshot_grants_production_authority = false;

  [[nodiscard]] constexpr bool valid_enqueue_receipt() const noexcept {
    const auto plan = sm87_macrofeed_v4_full_attention_preprocess_plan(
        first_position, token_count);
    return plan.valid() && identity == plan.identity && device_ordinal >= 0 &&
           scratch_row_stride == plan.scratch_row_stride &&
           physical_kernel_launches == 1U && centered_qk_rmsnorm &&
           exact_prompt_wide_reduction_tree && bf16_boundary_before_rope &&
           partial_d64_neox_rope && q_in_place && k_in_place &&
           gate_bitwise_preserved && scratch_gap_bitwise_preserved &&
           private_nhd_key_cache && value_cache_unaddressable &&
           current_device_revalidated &&
           caller_snapshot_exact_observed_match &&
           device_allocation_ranges_owned && caller_stream_non_null &&
           !stream_owner_verified && launch_enqueued &&
           !completion_observed && !numerical_contract_qualified &&
           !production_dispatch_eligible && startup_package_unbound &&
           !execution_capability &&
           !caller_snapshot_grants_production_authority;
  }
};

// Default-off T1 executor.  The caller-fillable resource snapshot is
// re-observed exactly on the current device, and every complete CUDA
// allocation range is checked before enqueue.  The non-null stream remains a
// caller-live precondition: this public seam deliberately does not claim
// opaque-handle ownership.  A future private package must bind the stream,
// completion event, panel commit, numerical qualification, and production
// route authority.
[[nodiscard]] int
launch_sm87_macrofeed_v4_full_attention_preprocess_admission_cuda(
    const Sm87MacroFeedV4FullAttentionPreprocessArguments& arguments,
    const Sm87MacroFeedV4FullAttentionPreprocessAdmissionResourceSnapshot&
        resources,
    Sm87MacroFeedV4FullAttentionPreprocessAdmissionLaunchReceipt*
        receipt) noexcept;

// Bounded T1 correctness seam.  Both entries use the V4 in-place layout;
// candidate-128 reproduces the admitted kernel while reference-256 preserves
// the established production reduction tree directly.  Neither grants timing,
// selector, startup, whole-model, or production authority.
inline constexpr std::size_t
    kSm87MacroFeedV4FullAttentionPreprocessOracleMaximumTokens = 128U;

struct Sm87MacroFeedV4FullAttentionPreprocessOracleArguments final {
  std::uint16_t* q_gate_scratch = nullptr;
  std::size_t token_count = 0U;
  std::size_t scratch_row_stride = 0U;
  std::uint16_t* key_cache = nullptr;
  std::size_t key_cache_position_capacity = 0U;
  std::size_t key_cache_row_stride = 0U;
  const std::uint16_t* q_norm_weight = nullptr;
  const std::uint16_t* k_norm_weight = nullptr;
  const float* cosines = nullptr;
  const float* sines = nullptr;
  std::size_t rope_position_capacity = 0U;
  std::size_t rope_row_stride = 0U;
  std::size_t first_position = 0U;
  std::uint32_t epsilon_fp32_bits = 0U;
  void* cuda_stream = nullptr;
};

[[nodiscard]] int
launch_sm87_macrofeed_v4_full_attention_preprocess_candidate_128_oracle_cuda(
    const Sm87MacroFeedV4FullAttentionPreprocessOracleArguments&
        arguments) noexcept;

[[nodiscard]] int
launch_sm87_macrofeed_v4_full_attention_preprocess_reference_256_oracle_cuda(
    const Sm87MacroFeedV4FullAttentionPreprocessOracleArguments&
        arguments) noexcept;

static_assert(kSm87MacroFeedV4FullAttentionPreprocessQueryHeads *
                      kSm87MacroFeedV4FullAttentionPreprocessQGateHeadStride ==
                  kSm87MacroFeedV4FullAttentionPreprocessQGateSpan &&
              kSm87MacroFeedV4FullAttentionPreprocessPhysicalKvHeads *
                      kSm87MacroFeedV4FullAttentionPreprocessPhysicalKvHeadDimension ==
                  kSm87MacroFeedV4FullAttentionPreprocessLogicalKvHeads *
                      kSm87MacroFeedV4FullAttentionPreprocessHeadDimension &&
              kSm87MacroFeedV4FullAttentionPreprocessKeyRowStride == 1'024U &&
              kSm87MacroFeedV4FullAttentionPreprocessPhysicalCtas ==
                  kSm87MacroFeedV4FullAttentionPreprocessTokens *
                      kSm87MacroFeedV4FullAttentionPreprocessCombinedHeads &&
              kSm87MacroFeedV4FullAttentionPreprocessScratchBytes ==
                  278'528'000U &&
              kSm87MacroFeedV4FullAttentionPreprocessKeyCacheBytes ==
                  81'920'000U &&
              kSm87MacroFeedV4FullAttentionPreprocessRopeTableBytes ==
                  5'120'000U);
static_assert(sm87_macrofeed_v4_full_attention_preprocess_plan(0U, 8'000U)
                  .valid());
static_assert(sm87_macrofeed_v4_full_attention_preprocess_plan(32'000U,
                                                               8'000U)
                  .valid());
static_assert(!sm87_macrofeed_v4_full_attention_preprocess_plan(1U, 8'000U)
                   .valid());

}  // namespace q3x::kernels
