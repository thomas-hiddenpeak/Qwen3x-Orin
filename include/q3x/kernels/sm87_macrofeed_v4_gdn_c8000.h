#pragma once

#include "q3x/kernels/sm87_target_aot_gdn_plan.h"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {

// Default-off exact GDN continuation constituent for
// AC-PREFILL-SM87-MACROFEED-v4.  One invocation advances one linear layer
// across exactly one contiguous C8000 prompt panel.  It owns no selector and
// grants no whole-model, numerical-qualification, or production authority.
enum class Sm87MacroFeedV4GdnC8000Identity : std::uint64_t {
  kInvalid = 0U,
  kExactPerTokenBf16DualEpochInPlaceV1 = 0x5133'4d46'5634'4401ULL,
};

inline constexpr Sm87MacroFeedV4GdnC8000Identity
    kSm87MacroFeedV4GdnC8000Identity =
        Sm87MacroFeedV4GdnC8000Identity::
            kExactPerTokenBf16DualEpochInPlaceV1;
inline constexpr std::size_t kSm87MacroFeedV4GdnC8000Tokens = 8'000U;
inline constexpr std::size_t kSm87MacroFeedV4GdnScratchRowStride = 17'408U;
inline constexpr std::size_t kSm87MacroFeedV4GdnQkvOffset = 0U;
inline constexpr std::size_t kSm87MacroFeedV4GdnQkvChannels = 10'240U;
inline constexpr std::size_t kSm87MacroFeedV4GdnZOffset = 10'240U;
inline constexpr std::size_t kSm87MacroFeedV4GdnZChannels = 6'144U;
inline constexpr std::size_t kSm87MacroFeedV4GdnAOffset = 16'384U;
inline constexpr std::size_t kSm87MacroFeedV4GdnBOffset = 16'432U;
inline constexpr std::size_t kSm87MacroFeedV4GdnScalarChannels = 48U;
inline constexpr std::size_t kSm87MacroFeedV4GdnOutputOffset = 4'096U;
inline constexpr std::size_t kSm87MacroFeedV4GdnOutputChannels = 6'144U;
inline constexpr std::size_t kSm87MacroFeedV4GdnConvThreads = 256U;
inline constexpr std::size_t kSm87MacroFeedV4GdnConvCtas =
    kSm87TargetAotGdnTotalConvChannels / kSm87MacroFeedV4GdnConvThreads;
inline constexpr std::size_t kSm87MacroFeedV4GdnRecurrenceThreads = 256U;
inline constexpr std::size_t kSm87MacroFeedV4GdnRecurrenceCtas =
    kSm87TargetAotGdnValueHeads;
inline constexpr std::size_t kSm87MacroFeedV4GdnPhysicalKernelLaunches = 2U;
inline constexpr std::size_t kSm87MacroFeedV4GdnHistoryCopies = 1U;
inline constexpr std::size_t kSm87MacroFeedV4GdnCancellationQuantum = 256U;
inline constexpr std::size_t kSm87MacroFeedV4GdnPointerAlignment = 16U;
inline constexpr std::size_t kSm87MacroFeedV4GdnSmCount = 16U;
inline constexpr std::size_t kSm87MacroFeedV4GdnConvStaticSharedBytes = 4U;
inline constexpr std::size_t
    kSm87MacroFeedV4GdnRecurrenceStaticSharedBytes = 34'316U;

inline constexpr std::uint64_t kSm87MacroFeedV4GdnScratchBytes =
    kSm87MacroFeedV4GdnC8000Tokens * kSm87MacroFeedV4GdnScratchRowStride *
    kSm87TargetAotGdnBf16Bytes;
inline constexpr std::uint64_t kSm87MacroFeedV4GdnConvWeightBytes =
    kSm87TargetAotGdnConvWeightElements * kSm87TargetAotGdnBf16Bytes;
inline constexpr std::uint64_t kSm87MacroFeedV4GdnConvHistoryBytes =
    kSm87TargetAotGdnTotalConvHistoryBytes;
inline constexpr std::uint64_t kSm87MacroFeedV4GdnHeadVectorBytes =
    kSm87TargetAotGdnScalarHeadElements * kSm87TargetAotGdnBf16Bytes;
inline constexpr std::uint64_t kSm87MacroFeedV4GdnNormWeightBytes =
    kSm87TargetAotGdnNormWeightElements * kSm87TargetAotGdnBf16Bytes;
inline constexpr std::uint64_t kSm87MacroFeedV4GdnStateBytes =
    kSm87TargetAotGdnTotalStateBytes;

static_assert(kSm87MacroFeedV4GdnQkvOffset == 0U);
static_assert(kSm87MacroFeedV4GdnQkvChannels ==
              kSm87TargetAotGdnTotalConvChannels);
static_assert(kSm87MacroFeedV4GdnZOffset ==
              kSm87MacroFeedV4GdnQkvOffset +
                  kSm87MacroFeedV4GdnQkvChannels);
static_assert(kSm87MacroFeedV4GdnAOffset ==
              kSm87MacroFeedV4GdnZOffset +
                  kSm87MacroFeedV4GdnZChannels);
static_assert(kSm87MacroFeedV4GdnBOffset ==
              kSm87MacroFeedV4GdnAOffset +
                  kSm87MacroFeedV4GdnScalarChannels);
static_assert(kSm87MacroFeedV4GdnBOffset +
                      kSm87MacroFeedV4GdnScalarChannels <=
                  kSm87MacroFeedV4GdnScratchRowStride);
static_assert(kSm87MacroFeedV4GdnOutputOffset ==
              kSm87TargetAotGdnRawVOffset);
static_assert(kSm87MacroFeedV4GdnOutputOffset +
                      kSm87MacroFeedV4GdnOutputChannels ==
                  kSm87MacroFeedV4GdnQkvChannels);
static_assert(kSm87MacroFeedV4GdnScratchBytes == 278'528'000U);
static_assert(kSm87MacroFeedV4GdnConvHistoryBytes == 61'440U);
static_assert(kSm87MacroFeedV4GdnStateBytes == 1'572'864U);

enum Sm87MacroFeedV4GdnPolicy : std::uint64_t {
  kSm87MacroFeedV4GdnExactPerTokenBf16State = 1ULL << 0U,
  kSm87MacroFeedV4GdnPreRoundFp32TokenOutput = 1ULL << 1U,
  kSm87MacroFeedV4GdnActiveStateConst = 1ULL << 2U,
  kSm87MacroFeedV4GdnCandidateStateFullyAssigned = 1ULL << 3U,
  kSm87MacroFeedV4GdnHistoryCopyThenInPlaceUpdate = 1ULL << 4U,
  kSm87MacroFeedV4GdnInPlaceQkvConvolution = 1ULL << 5U,
  kSm87MacroFeedV4GdnInPlaceVOutput = 1ULL << 6U,
  kSm87MacroFeedV4GdnNoWholeEpochCopy = 1ULL << 7U,
  kSm87MacroFeedV4GdnNoAssociativeScan = 1ULL << 8U,
  kSm87MacroFeedV4GdnNoFp32AuthoritativeState = 1ULL << 9U,
  kSm87MacroFeedV4GdnNoProductionSelector = 1ULL << 10U,
  kSm87MacroFeedV4GdnRawBf16BeforeNormGate = 1ULL << 11U,
  kSm87MacroFeedV4GdnExactReductionTrees = 1ULL << 12U,
  kSm87MacroFeedV4GdnBoundedCancellationPolling = 1ULL << 13U,
  kSm87MacroFeedV4GdnCurrentDeviceRevalidated = 1ULL << 14U,
  kSm87MacroFeedV4GdnCallerSnapshotExactObservedMatch = 1ULL << 15U,
  kSm87MacroFeedV4GdnCallerSuppliedLiveStreamRequired = 1ULL << 16U,
  kSm87MacroFeedV4GdnDeviceAllocationRangesOwned = 1ULL << 17U,
  kSm87MacroFeedV4GdnLiveStreamDeviceObserved = 1ULL << 18U,
};

inline constexpr std::uint64_t kSm87MacroFeedV4GdnRequiredPolicy =
    kSm87MacroFeedV4GdnExactPerTokenBf16State |
    kSm87MacroFeedV4GdnPreRoundFp32TokenOutput |
    kSm87MacroFeedV4GdnActiveStateConst |
    kSm87MacroFeedV4GdnCandidateStateFullyAssigned |
    kSm87MacroFeedV4GdnHistoryCopyThenInPlaceUpdate |
    kSm87MacroFeedV4GdnInPlaceQkvConvolution |
    kSm87MacroFeedV4GdnInPlaceVOutput |
    kSm87MacroFeedV4GdnNoWholeEpochCopy |
    kSm87MacroFeedV4GdnNoAssociativeScan |
    kSm87MacroFeedV4GdnNoFp32AuthoritativeState |
    kSm87MacroFeedV4GdnNoProductionSelector |
    kSm87MacroFeedV4GdnRawBf16BeforeNormGate |
    kSm87MacroFeedV4GdnExactReductionTrees |
    kSm87MacroFeedV4GdnBoundedCancellationPolling |
    kSm87MacroFeedV4GdnCurrentDeviceRevalidated |
    kSm87MacroFeedV4GdnCallerSnapshotExactObservedMatch |
    kSm87MacroFeedV4GdnCallerSuppliedLiveStreamRequired |
    kSm87MacroFeedV4GdnDeviceAllocationRangesOwned |
    kSm87MacroFeedV4GdnLiveStreamDeviceObserved;

struct Sm87MacroFeedV4GdnC8000Plan final {
  Sm87MacroFeedV4GdnC8000Identity identity =
      Sm87MacroFeedV4GdnC8000Identity::kInvalid;
  std::size_t token_count = 0U;
  std::size_t scratch_row_stride = 0U;
  std::uint64_t policy = 0U;
  std::size_t conv_history_copy_bytes = 0U;
  std::size_t whole_recurrent_epoch_copy_bytes = 0U;
  std::uint32_t physical_kernel_launches = 0U;
  std::uint32_t asynchronous_d2d_copies = 0U;
  bool default_off = false;
  bool selector_present = true;
  bool numerical_contract_qualified = true;
  bool production_dispatch_eligible = true;
  bool startup_package_unbound = true;
  bool execution_capability = false;  // Production-route capability only.
  bool caller_snapshot_grants_production_authority = false;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return identity == kSm87MacroFeedV4GdnC8000Identity &&
           token_count == kSm87MacroFeedV4GdnC8000Tokens &&
           scratch_row_stride == kSm87MacroFeedV4GdnScratchRowStride &&
           policy == kSm87MacroFeedV4GdnRequiredPolicy &&
           conv_history_copy_bytes == kSm87MacroFeedV4GdnConvHistoryBytes &&
           whole_recurrent_epoch_copy_bytes == 0U &&
           physical_kernel_launches ==
               kSm87MacroFeedV4GdnPhysicalKernelLaunches &&
           asynchronous_d2d_copies == kSm87MacroFeedV4GdnHistoryCopies &&
           default_off && !selector_present &&
           !numerical_contract_qualified &&
           !production_dispatch_eligible && startup_package_unbound &&
           !execution_capability &&
           !caller_snapshot_grants_production_authority;
  }
};

[[nodiscard]] constexpr Sm87MacroFeedV4GdnC8000Plan
sm87_macrofeed_v4_gdn_c8000_plan(const std::size_t token_count,
                                 const std::size_t row_stride) noexcept {
  if (token_count != kSm87MacroFeedV4GdnC8000Tokens ||
      row_stride != kSm87MacroFeedV4GdnScratchRowStride) {
    return {};
  }
  return {kSm87MacroFeedV4GdnC8000Identity,
          token_count,
          row_stride,
          kSm87MacroFeedV4GdnRequiredPolicy,
          static_cast<std::size_t>(kSm87MacroFeedV4GdnConvHistoryBytes),
          0U,
          static_cast<std::uint32_t>(
              kSm87MacroFeedV4GdnPhysicalKernelLaunches),
          static_cast<std::uint32_t>(kSm87MacroFeedV4GdnHistoryCopies),
          true,
          false,
          false,
          false,
          true,
          false,
          false};
}

struct Sm87MacroFeedV4GdnByteRange final {
  std::uintptr_t begin = 0U;
  std::uintptr_t end = 0U;
  bool valid = false;
};

[[nodiscard]] constexpr Sm87MacroFeedV4GdnByteRange
sm87_macrofeed_v4_gdn_byte_range(const void* const pointer,
                                 const std::uint64_t bytes) noexcept {
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

[[nodiscard]] constexpr bool sm87_macrofeed_v4_gdn_ranges_disjoint(
    const Sm87MacroFeedV4GdnByteRange& left,
    const Sm87MacroFeedV4GdnByteRange& right) noexcept {
  return left.valid && right.valid &&
         (left.end <= right.begin || right.end <= left.begin);
}

[[nodiscard]] constexpr bool sm87_macrofeed_v4_gdn_pointer_aligned(
    const void* const pointer) noexcept {
  return pointer != nullptr &&
         reinterpret_cast<std::uintptr_t>(pointer) %
                 kSm87MacroFeedV4GdnPointerAlignment ==
             0U;
}

struct Sm87MacroFeedV4GdnC8000Arguments final {
  // One phase-aliased [8000,17408] BF16 plane.  QKV is convolved in place;
  // the final GDN result overwrites only V [4096,10240).
  std::uint16_t* scratch = nullptr;
  std::size_t token_count = 0U;
  std::size_t scratch_row_stride = 0U;
  const std::uint16_t* conv_weight = nullptr;          // [10240,4]
  const std::uint16_t* a_log = nullptr;                // [48]
  const std::uint16_t* dt_bias = nullptr;              // [48]
  const std::uint16_t* norm_weight = nullptr;          // [128]
  const std::uint16_t* active_conv_history = nullptr;  // [10240,3]
  std::uint16_t* candidate_conv_history = nullptr;     // [10240,3]
  const std::uint16_t* active_recurrent_state = nullptr;  // [48,128,128]
  std::uint16_t* candidate_recurrent_state = nullptr;     // [48,128,128]
  const std::uint32_t* cancellation_signal = nullptr;
  std::uint32_t l2_epsilon_fp32_bits = 0U;
  std::uint32_t norm_epsilon_fp32_bits = 0U;
  // T1 caller precondition: a live CUDA Runtime stream.  This admission seam
  // observes its device but cannot safely authenticate arbitrary opaque or
  // stale handles; a future private owner-issued package must do that.
  void* cuda_stream = nullptr;
};

// Structural/range validation only.  Host-only fake ranges may satisfy this
// seam for T0 tests; only the admission launcher binds the current device,
// observed resource snapshot, caller-preconditioned live-stream device, and
// every complete device allocation range before enqueue.
[[nodiscard]] bool sm87_macrofeed_v4_gdn_c8000_arguments_valid(
    const Sm87MacroFeedV4GdnC8000Arguments& arguments) noexcept;

struct Sm87MacroFeedV4GdnKernelResourceSnapshot final {
  std::int32_t registers_per_thread = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  std::int32_t maximum_threads_per_block = 0;
  std::int32_t active_blocks_per_sm = 0;
  std::int32_t threads_per_block = 0;
  std::int32_t physical_grid_ctas = 0;
};

// Caller-fillable T0/T1 resource evidence only.  This snapshot is not a
// startup-issued package and cannot grant production execution authority.
struct Sm87MacroFeedV4GdnC8000AdmissionResourceSnapshot final {
  Sm87MacroFeedV4GdnC8000Identity identity =
      Sm87MacroFeedV4GdnC8000Identity::kInvalid;
  std::int32_t device_ordinal = -1;
  std::int32_t compute_major = 0;
  std::int32_t compute_minor = 0;
  std::int32_t sm_count = 0;
  std::int32_t binary_version = 0;
  Sm87MacroFeedV4GdnKernelResourceSnapshot convolution{};
  Sm87MacroFeedV4GdnKernelResourceSnapshot recurrence_epilogue{};
  bool kernels_compiled = false;
  bool exact_geometry = false;
  bool static_resource_gate_passed = false;
  bool numerical_contract_qualified = false;
  bool production_dispatch_eligible = false;
  bool startup_package_unbound = true;
  bool execution_capability = false;  // Production-route capability only.
  bool caller_snapshot_grants_production_authority = false;
};

[[nodiscard]] constexpr bool
sm87_macrofeed_v4_gdn_c8000_admission_resource_gate(
    const Sm87MacroFeedV4GdnC8000AdmissionResourceSnapshot&
        resources) noexcept {
  const auto kernel_valid =
      [](const Sm87MacroFeedV4GdnKernelResourceSnapshot& item,
         const std::int32_t threads, const std::int32_t ctas,
         const std::int32_t minimum_active_ctas,
         const std::size_t expected_static_shared) {
    return item.registers_per_thread > 0 && item.local_bytes == 0U &&
           item.static_shared_bytes == expected_static_shared &&
           item.maximum_threads_per_block >= threads &&
           item.active_blocks_per_sm >= minimum_active_ctas &&
           item.threads_per_block == threads &&
           item.physical_grid_ctas == ctas;
  };
  return resources.identity == kSm87MacroFeedV4GdnC8000Identity &&
         resources.device_ordinal >= 0 && resources.compute_major == 8 &&
         resources.compute_minor == 7 &&
         resources.sm_count ==
             static_cast<std::int32_t>(kSm87MacroFeedV4GdnSmCount) &&
         resources.binary_version == 87 && resources.kernels_compiled &&
         resources.exact_geometry && resources.static_resource_gate_passed &&
         kernel_valid(resources.convolution,
                      static_cast<std::int32_t>(
                          kSm87MacroFeedV4GdnConvThreads),
                      static_cast<std::int32_t>(kSm87MacroFeedV4GdnConvCtas),
                      2, kSm87MacroFeedV4GdnConvStaticSharedBytes) &&
         kernel_valid(
             resources.recurrence_epilogue,
             static_cast<std::int32_t>(
                 kSm87MacroFeedV4GdnRecurrenceThreads),
             static_cast<std::int32_t>(
                 kSm87MacroFeedV4GdnRecurrenceCtas),
             3, kSm87MacroFeedV4GdnRecurrenceStaticSharedBytes) &&
         !resources.numerical_contract_qualified &&
         !resources.production_dispatch_eligible &&
         resources.startup_package_unbound &&
         !resources.execution_capability &&
         !resources.caller_snapshot_grants_production_authority;
}

[[nodiscard]] int
query_sm87_macrofeed_v4_gdn_c8000_admission_resource_snapshot_cuda(
    Sm87MacroFeedV4GdnC8000AdmissionResourceSnapshot* resources) noexcept;

struct Sm87MacroFeedV4GdnC8000AdmissionLaunchReceipt final {
  Sm87MacroFeedV4GdnC8000Identity identity =
      Sm87MacroFeedV4GdnC8000Identity::kInvalid;
  std::size_t token_count = 0U;
  std::size_t scratch_row_stride = 0U;
  std::uint64_t conv_history_copy_bytes = 0U;
  std::uint64_t whole_recurrent_epoch_copy_bytes = 0U;
  std::uint32_t physical_kernel_launches = 0U;
  std::uint32_t asynchronous_d2d_copies = 0U;
  bool active_state_const = false;
  bool candidate_state_full_assignment_required = false;
  bool in_place_qkv_convolution = false;
  bool in_place_v_output = false;
  bool current_device_revalidated = false;
  bool caller_snapshot_exact_observed_match = false;
  bool caller_supplied_live_stream_required = false;
  bool live_stream_device_observed = false;
  bool device_allocation_ranges_owned = false;
  bool launch_enqueued = false;
  bool numerical_contract_qualified = true;
  bool production_dispatch_eligible = true;
  bool startup_package_unbound = true;
  bool execution_capability = false;  // Production-route capability only.
  bool caller_snapshot_grants_production_authority = false;

  [[nodiscard]] constexpr bool valid_enqueue_receipt() const noexcept {
    return identity == kSm87MacroFeedV4GdnC8000Identity &&
           token_count == kSm87MacroFeedV4GdnC8000Tokens &&
           scratch_row_stride == kSm87MacroFeedV4GdnScratchRowStride &&
           conv_history_copy_bytes ==
               kSm87MacroFeedV4GdnConvHistoryBytes &&
           whole_recurrent_epoch_copy_bytes == 0U &&
           physical_kernel_launches ==
               kSm87MacroFeedV4GdnPhysicalKernelLaunches &&
           asynchronous_d2d_copies == kSm87MacroFeedV4GdnHistoryCopies &&
           active_state_const && candidate_state_full_assignment_required &&
           in_place_qkv_convolution && in_place_v_output &&
           current_device_revalidated &&
           caller_snapshot_exact_observed_match &&
           caller_supplied_live_stream_required &&
           live_stream_device_observed &&
           device_allocation_ranges_owned && launch_enqueued &&
           !numerical_contract_qualified &&
           !production_dispatch_eligible && startup_package_unbound &&
           !execution_capability &&
           !caller_snapshot_grants_production_authority;
  }
};

// Default-off admission-only executor.  The caller-fillable snapshot is only
// T0/T1 resource evidence: it is startup-package-unbound, has no production
// execution capability, and grants no production authority.  The returned
// receipt attests an admission enqueue only.  T1 requires a caller-supplied
// live stream and observes its device; it does not authenticate arbitrary or
// stale opaque handles.  A future private startup-issued package must own that
// stream and bind device-owned completion/panel-commit events.
[[nodiscard]] int launch_sm87_macrofeed_v4_gdn_c8000_admission_cuda(
    const Sm87MacroFeedV4GdnC8000Arguments& arguments,
    const Sm87MacroFeedV4GdnC8000AdmissionResourceSnapshot& resources,
    Sm87MacroFeedV4GdnC8000AdmissionLaunchReceipt* receipt) noexcept;

// Synthetic correctness seam only.  It uses the identical two production
// kernels for a bounded token count while retaining the fixed V4 row layout.
// It has no timing, selector, whole-model, or production authority.
struct Sm87MacroFeedV4GdnOracleArguments final {
  std::uint16_t* scratch = nullptr;
  std::size_t token_count = 0U;
  std::size_t scratch_row_stride = 0U;
  const std::uint16_t* conv_weight = nullptr;
  const std::uint16_t* a_log = nullptr;
  const std::uint16_t* dt_bias = nullptr;
  const std::uint16_t* norm_weight = nullptr;
  const std::uint16_t* active_conv_history = nullptr;
  std::uint16_t* candidate_conv_history = nullptr;
  const std::uint16_t* active_recurrent_state = nullptr;
  std::uint16_t* candidate_recurrent_state = nullptr;
  const std::uint32_t* cancellation_signal = nullptr;
  std::uint32_t l2_epsilon_fp32_bits = 0U;
  std::uint32_t norm_epsilon_fp32_bits = 0U;
  void* cuda_stream = nullptr;
};

inline constexpr std::size_t kSm87MacroFeedV4GdnOracleMaximumTokens = 128U;

[[nodiscard]] int launch_sm87_macrofeed_v4_gdn_oracle_cuda(
    const Sm87MacroFeedV4GdnOracleArguments& arguments) noexcept;

static_assert(sm87_macrofeed_v4_gdn_c8000_plan(
                  kSm87MacroFeedV4GdnC8000Tokens,
                  kSm87MacroFeedV4GdnScratchRowStride)
                  .valid());
static_assert(!sm87_macrofeed_v4_gdn_c8000_plan(
                   kSm87MacroFeedV4GdnC8000Tokens - 1U,
                   kSm87MacroFeedV4GdnScratchRowStride)
                   .valid());
static_assert(kSm87MacroFeedV4GdnRequiredPolicy == 0x7ffffULL);

}  // namespace q3x::kernels
