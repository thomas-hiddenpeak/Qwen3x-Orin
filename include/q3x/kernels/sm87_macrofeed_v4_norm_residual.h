#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {

// Default-off, admission-only exact normalization/residual constituent for
// AC-PREFILL-SM87-MACROFEED-v4.  Both operations reuse the established
// decode_ops arithmetic kernels.  This surface owns no selector, startup
// capability, completion event, numerical qualification, or production
// dispatch authority.
enum class Sm87MacroFeedV4NormResidualIdentity : std::uint64_t {
  kInvalid = 0U,
  kExactCenteredRmsNormAndAliasResidualV1 =
      0x5133'4d46'5634'4e01ULL,
};

enum class Sm87MacroFeedV4NormResidualOperation : std::uint8_t {
  kInvalid = 0U,
  kInputCenteredRmsNorm,
  kBranchResidualPostCenteredRmsNorm,
};

inline constexpr Sm87MacroFeedV4NormResidualIdentity
    kSm87MacroFeedV4NormResidualIdentity =
        Sm87MacroFeedV4NormResidualIdentity::
            kExactCenteredRmsNormAndAliasResidualV1;
inline constexpr std::size_t kSm87MacroFeedV4NormResidualTokens = 8'000U;
inline constexpr std::size_t kSm87MacroFeedV4NormResidualHidden = 5'120U;
inline constexpr std::size_t
    kSm87MacroFeedV4NormResidualInputNormThreads = 256U;
inline constexpr std::size_t
    kSm87MacroFeedV4NormResidualFusedThreads = 512U;
inline constexpr std::size_t kSm87MacroFeedV4NormResidualGridCtas = 8'000U;
inline constexpr std::size_t kSm87MacroFeedV4NormResidualLaunches = 1U;
inline constexpr std::size_t kSm87MacroFeedV4NormResidualSmCount = 16U;
inline constexpr std::size_t
    kSm87MacroFeedV4NormResidualRequiredCtasPerSm = 2U;
inline constexpr std::size_t kSm87MacroFeedV4NormResidualPointerAlignment =
    16U;
inline constexpr std::size_t
    kSm87MacroFeedV4NormResidualInputNormStaticSharedBytes = 1'024U;
inline constexpr std::size_t
    kSm87MacroFeedV4NormResidualFusedStaticSharedBytes = 11'264U;
inline constexpr std::uint32_t
    kSm87MacroFeedV4NormResidualEpsilonFp32Bits = 0x3586'37bdU;
inline constexpr std::size_t kSm87MacroFeedV4NormResidualOracleMaximumTokens =
    65U;
inline constexpr std::uint64_t kSm87MacroFeedV4NormResidualHiddenBytes =
    kSm87MacroFeedV4NormResidualTokens *
    kSm87MacroFeedV4NormResidualHidden * sizeof(std::uint16_t);
inline constexpr std::uint64_t kSm87MacroFeedV4NormResidualWeightBytes =
    kSm87MacroFeedV4NormResidualHidden * sizeof(std::uint16_t);

static_assert(kSm87MacroFeedV4NormResidualHiddenBytes == 81'920'000U);
static_assert(kSm87MacroFeedV4NormResidualWeightBytes == 10'240U);

struct Sm87MacroFeedV4NormResidualPlan final {
  Sm87MacroFeedV4NormResidualIdentity identity =
      Sm87MacroFeedV4NormResidualIdentity::kInvalid;
  std::size_t token_count = 0U;
  std::size_t hidden_size = 0U;
  std::size_t grid_ctas = 0U;
  std::size_t input_norm_threads_per_cta = 0U;
  std::size_t fused_residual_norm_threads_per_cta = 0U;
  std::uint32_t epsilon_fp32_bits = 0U;
  std::uint32_t input_norm_kernel_launches = 0U;
  std::uint32_t fused_residual_norm_kernel_launches = 0U;
  bool established_exact_norm_body_reused = false;
  bool established_exact_residual_norm_body_reused = false;
  bool residual_published_in_place_to_right = false;
  bool normalized_published_in_place_to_left = false;
  bool third_hidden_plane_present = true;
  bool copy_kernel_present = true;
  bool selector_present = true;
  bool fallback_permitted = true;
  bool jit_permitted = true;
  bool runtime_repack_permitted = true;
  bool autotune_permitted = true;
  bool default_off = false;
  bool numerical_contract_qualified = true;
  bool production_dispatch_eligible = true;
  bool startup_package_unbound = true;
  bool execution_capability = false;
  bool caller_snapshot_grants_production_authority = false;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return identity == kSm87MacroFeedV4NormResidualIdentity &&
           token_count == kSm87MacroFeedV4NormResidualTokens &&
           hidden_size == kSm87MacroFeedV4NormResidualHidden &&
           grid_ctas == kSm87MacroFeedV4NormResidualGridCtas &&
           input_norm_threads_per_cta ==
               kSm87MacroFeedV4NormResidualInputNormThreads &&
           fused_residual_norm_threads_per_cta ==
               kSm87MacroFeedV4NormResidualFusedThreads &&
           epsilon_fp32_bits ==
               kSm87MacroFeedV4NormResidualEpsilonFp32Bits &&
           input_norm_kernel_launches ==
               kSm87MacroFeedV4NormResidualLaunches &&
           fused_residual_norm_kernel_launches ==
               kSm87MacroFeedV4NormResidualLaunches &&
           established_exact_norm_body_reused &&
           established_exact_residual_norm_body_reused &&
           residual_published_in_place_to_right &&
           normalized_published_in_place_to_left &&
           !third_hidden_plane_present && !copy_kernel_present &&
           !selector_present && !fallback_permitted && !jit_permitted &&
           !runtime_repack_permitted && !autotune_permitted && default_off &&
           !numerical_contract_qualified &&
           !production_dispatch_eligible && startup_package_unbound &&
           !execution_capability &&
           !caller_snapshot_grants_production_authority;
  }
};

[[nodiscard]] constexpr Sm87MacroFeedV4NormResidualPlan
sm87_macrofeed_v4_norm_residual_plan(const std::size_t token_count,
                                     const std::size_t hidden_size) noexcept {
  if (token_count != kSm87MacroFeedV4NormResidualTokens ||
      hidden_size != kSm87MacroFeedV4NormResidualHidden) {
    return {};
  }
  Sm87MacroFeedV4NormResidualPlan plan;
  plan.identity = kSm87MacroFeedV4NormResidualIdentity;
  plan.token_count = token_count;
  plan.hidden_size = hidden_size;
  plan.grid_ctas = kSm87MacroFeedV4NormResidualGridCtas;
  plan.input_norm_threads_per_cta =
      kSm87MacroFeedV4NormResidualInputNormThreads;
  plan.fused_residual_norm_threads_per_cta =
      kSm87MacroFeedV4NormResidualFusedThreads;
  plan.epsilon_fp32_bits = kSm87MacroFeedV4NormResidualEpsilonFp32Bits;
  plan.input_norm_kernel_launches = 1U;
  plan.fused_residual_norm_kernel_launches = 1U;
  plan.established_exact_norm_body_reused = true;
  plan.established_exact_residual_norm_body_reused = true;
  plan.residual_published_in_place_to_right = true;
  plan.normalized_published_in_place_to_left = true;
  plan.third_hidden_plane_present = false;
  plan.copy_kernel_present = false;
  plan.selector_present = false;
  plan.fallback_permitted = false;
  plan.jit_permitted = false;
  plan.runtime_repack_permitted = false;
  plan.autotune_permitted = false;
  plan.default_off = true;
  plan.numerical_contract_qualified = false;
  plan.production_dispatch_eligible = false;
  plan.startup_package_unbound = true;
  plan.execution_capability = false;
  plan.caller_snapshot_grants_production_authority = false;
  return plan;
}

struct Sm87MacroFeedV4NormResidualByteRange final {
  std::uintptr_t begin = 0U;
  std::uintptr_t end = 0U;
  bool valid = false;
};

[[nodiscard]] constexpr Sm87MacroFeedV4NormResidualByteRange
sm87_macrofeed_v4_norm_residual_byte_range(
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

[[nodiscard]] constexpr bool sm87_macrofeed_v4_norm_residual_ranges_disjoint(
    const Sm87MacroFeedV4NormResidualByteRange& left,
    const Sm87MacroFeedV4NormResidualByteRange& right) noexcept {
  return left.valid && right.valid &&
         (left.end <= right.begin || right.end <= left.begin);
}

struct Sm87MacroFeedV4InputNormArguments final {
  const std::uint16_t* input_hidden = nullptr;  // BF16 [8000,5120].
  const std::uint16_t* centered_weight = nullptr;  // BF16 [5120].
  std::uint16_t* output_hidden = nullptr;  // BF16 [8000,5120].
  std::size_t token_count = 0U;
  std::size_t hidden_size = 0U;
  std::uint32_t epsilon_fp32_bits = 0U;
  void* cuda_stream = nullptr;
};

struct Sm87MacroFeedV4ResidualPostNormArguments final {
  // Exactly two hidden planes.  The established kernel first publishes
  // R_bf16(left + right) in place to right, then consumes that published
  // residual with its unchanged reduction tree and writes normalized output
  // in place to left.
  std::uint16_t* left_residual_then_normalized = nullptr;
  std::uint16_t* right_branch_then_residual = nullptr;
  const std::uint16_t* centered_weight = nullptr;  // BF16 [5120].
  std::size_t token_count = 0U;
  std::size_t hidden_size = 0U;
  std::uint32_t epsilon_fp32_bits = 0U;
  void* cuda_stream = nullptr;
};

[[nodiscard]] bool sm87_macrofeed_v4_input_norm_arguments_valid(
    const Sm87MacroFeedV4InputNormArguments& arguments) noexcept;
[[nodiscard]] bool sm87_macrofeed_v4_residual_post_norm_arguments_valid(
    const Sm87MacroFeedV4ResidualPostNormArguments& arguments) noexcept;

struct Sm87MacroFeedV4NormResidualKernelResources final {
  std::int32_t registers_per_thread = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  std::int32_t maximum_threads_per_block = 0;
  std::int32_t active_blocks_per_sm = 0;
  std::int32_t threads_per_block = 0;
  std::int32_t physical_grid_ctas = 0;
};

struct Sm87MacroFeedV4NormResidualAdmissionResourceSnapshot final {
  Sm87MacroFeedV4NormResidualIdentity identity =
      Sm87MacroFeedV4NormResidualIdentity::kInvalid;
  std::int32_t device_ordinal = -1;
  std::int32_t compute_major = 0;
  std::int32_t compute_minor = 0;
  std::int32_t sm_count = 0;
  std::int32_t binary_version = 0;
  Sm87MacroFeedV4NormResidualKernelResources input_norm{};
  Sm87MacroFeedV4NormResidualKernelResources fused_residual_norm{};
  bool kernels_compiled = false;
  bool exact_geometry = false;
  bool static_resource_gate_passed = false;
  bool numerical_contract_qualified = false;
  bool production_dispatch_eligible = false;
  bool startup_package_unbound = true;
  bool execution_capability = false;
  bool caller_snapshot_grants_production_authority = false;
};

[[nodiscard]] constexpr bool sm87_macrofeed_v4_norm_residual_resource_gate(
    const Sm87MacroFeedV4NormResidualAdmissionResourceSnapshot& r) noexcept {
  return r.identity == kSm87MacroFeedV4NormResidualIdentity &&
         r.device_ordinal >= 0 && r.compute_major == 8 &&
         r.compute_minor == 7 &&
         r.sm_count ==
             static_cast<std::int32_t>(kSm87MacroFeedV4NormResidualSmCount) &&
         r.binary_version == 87 && r.kernels_compiled && r.exact_geometry &&
         r.static_resource_gate_passed &&
         r.input_norm.registers_per_thread > 0 &&
         r.input_norm.static_shared_bytes ==
             kSm87MacroFeedV4NormResidualInputNormStaticSharedBytes &&
         r.input_norm.local_bytes == 0U &&
         r.input_norm.maximum_threads_per_block >=
             static_cast<std::int32_t>(
                 kSm87MacroFeedV4NormResidualInputNormThreads) &&
         r.input_norm.active_blocks_per_sm >=
             static_cast<std::int32_t>(
                 kSm87MacroFeedV4NormResidualRequiredCtasPerSm) &&
         r.input_norm.threads_per_block ==
             static_cast<std::int32_t>(
                 kSm87MacroFeedV4NormResidualInputNormThreads) &&
         r.input_norm.physical_grid_ctas ==
             static_cast<std::int32_t>(
                 kSm87MacroFeedV4NormResidualGridCtas) &&
         r.fused_residual_norm.registers_per_thread > 0 &&
         r.fused_residual_norm.static_shared_bytes ==
             kSm87MacroFeedV4NormResidualFusedStaticSharedBytes &&
         r.fused_residual_norm.local_bytes == 0U &&
         r.fused_residual_norm.maximum_threads_per_block >=
             static_cast<std::int32_t>(
                 kSm87MacroFeedV4NormResidualFusedThreads) &&
         r.fused_residual_norm.active_blocks_per_sm >=
             static_cast<std::int32_t>(
                 kSm87MacroFeedV4NormResidualRequiredCtasPerSm) &&
         r.fused_residual_norm.threads_per_block ==
             static_cast<std::int32_t>(
                 kSm87MacroFeedV4NormResidualFusedThreads) &&
         r.fused_residual_norm.physical_grid_ctas ==
             static_cast<std::int32_t>(
                 kSm87MacroFeedV4NormResidualGridCtas) &&
         !r.numerical_contract_qualified &&
         !r.production_dispatch_eligible && r.startup_package_unbound &&
         !r.execution_capability &&
         !r.caller_snapshot_grants_production_authority;
}

[[nodiscard]] int
query_sm87_macrofeed_v4_norm_residual_admission_resources_cuda(
    Sm87MacroFeedV4NormResidualAdmissionResourceSnapshot* resources) noexcept;

struct Sm87MacroFeedV4NormResidualAdmissionLaunchReceipt final {
  Sm87MacroFeedV4NormResidualIdentity identity =
      Sm87MacroFeedV4NormResidualIdentity::kInvalid;
  Sm87MacroFeedV4NormResidualOperation operation =
      Sm87MacroFeedV4NormResidualOperation::kInvalid;
  std::int32_t device_ordinal = -1;
  std::size_t token_count = 0U;
  std::size_t hidden_size = 0U;
  std::uint32_t physical_kernel_launches = 0U;
  bool established_exact_body_reused = false;
  bool exact_alias_contract = false;
  bool third_hidden_plane_absent = false;
  bool current_device_revalidated = false;
  bool caller_snapshot_exact_observed_match = false;
  bool live_stream_device_observed = false;
  bool device_allocation_ranges_owned = false;
  bool launch_enqueued = false;
  bool completion_observed = true;
  bool numerical_contract_qualified = true;
  bool production_dispatch_eligible = true;
  bool startup_package_unbound = true;
  bool execution_capability = false;
  bool caller_snapshot_grants_production_authority = false;

  [[nodiscard]] constexpr bool valid_enqueue_receipt() const noexcept {
    return identity == kSm87MacroFeedV4NormResidualIdentity &&
           (operation ==
                Sm87MacroFeedV4NormResidualOperation::kInputCenteredRmsNorm ||
            operation == Sm87MacroFeedV4NormResidualOperation::
                             kBranchResidualPostCenteredRmsNorm) &&
           device_ordinal >= 0 &&
           token_count == kSm87MacroFeedV4NormResidualTokens &&
           hidden_size == kSm87MacroFeedV4NormResidualHidden &&
           physical_kernel_launches == 1U &&
           established_exact_body_reused && exact_alias_contract &&
           third_hidden_plane_absent && current_device_revalidated &&
           caller_snapshot_exact_observed_match &&
           live_stream_device_observed && device_allocation_ranges_owned &&
           launch_enqueued && !completion_observed &&
           !numerical_contract_qualified &&
           !production_dispatch_eligible && startup_package_unbound &&
           !execution_capability &&
           !caller_snapshot_grants_production_authority;
  }
};

[[nodiscard]] int launch_sm87_macrofeed_v4_input_norm_admission_cuda(
    const Sm87MacroFeedV4InputNormArguments& arguments,
    const Sm87MacroFeedV4NormResidualAdmissionResourceSnapshot& resources,
    Sm87MacroFeedV4NormResidualAdmissionLaunchReceipt* receipt) noexcept;

[[nodiscard]] int launch_sm87_macrofeed_v4_residual_post_norm_admission_cuda(
    const Sm87MacroFeedV4ResidualPostNormArguments& arguments,
    const Sm87MacroFeedV4NormResidualAdmissionResourceSnapshot& resources,
    Sm87MacroFeedV4NormResidualAdmissionLaunchReceipt* receipt) noexcept;

// Bounded T1 correctness seams.  They launch the identical production bodies
// for C1..C65 and have no timing or completion authority.
[[nodiscard]] int launch_sm87_macrofeed_v4_input_norm_oracle_cuda(
    const Sm87MacroFeedV4InputNormArguments& arguments) noexcept;
[[nodiscard]] int launch_sm87_macrofeed_v4_residual_post_norm_oracle_cuda(
    const Sm87MacroFeedV4ResidualPostNormArguments& arguments) noexcept;

static_assert(sm87_macrofeed_v4_norm_residual_plan(8'000U, 5'120U).valid());
static_assert(!sm87_macrofeed_v4_norm_residual_plan(7'999U, 5'120U).valid());

}  // namespace q3x::kernels
