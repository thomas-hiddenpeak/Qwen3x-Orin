#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {

// Default-off, admission-only exact BF16 A/B projection constituent for
// AC-PREFILL-SM87-MACROFEED-v4.  It translates the established
// M64xN96xK64 arithmetic body to the fixed C8000 V4 scratch layout.  This
// public T1 surface owns no selector or startup capability and grants no
// whole-model, numerical-qualification, completion, or production authority.
enum class Sm87MacroFeedV4Bf16AbIdentity : std::uint64_t {
  kInvalid = 0U,
  kExactM64N96K64DirectScratchV1 = 0x5133'4d46'5634'4201ULL,
};

inline constexpr Sm87MacroFeedV4Bf16AbIdentity
    kSm87MacroFeedV4Bf16AbIdentity =
        Sm87MacroFeedV4Bf16AbIdentity::kExactM64N96K64DirectScratchV1;
inline constexpr std::size_t kSm87MacroFeedV4Bf16AbTokens = 8'000U;
inline constexpr std::size_t kSm87MacroFeedV4Bf16AbInputFeatures = 5'120U;
inline constexpr std::size_t kSm87MacroFeedV4Bf16AbRowsPerProjection = 48U;
inline constexpr std::size_t kSm87MacroFeedV4Bf16AbLogicalRows = 96U;
inline constexpr std::size_t kSm87MacroFeedV4Bf16AbScratchRowStride =
    17'408U;
inline constexpr std::size_t kSm87MacroFeedV4Bf16AbAOffset = 16'384U;
inline constexpr std::size_t kSm87MacroFeedV4Bf16AbBOffset = 16'432U;
inline constexpr std::size_t kSm87MacroFeedV4Bf16AbTileTokens = 64U;
inline constexpr std::size_t kSm87MacroFeedV4Bf16AbThreads = 256U;
inline constexpr std::size_t kSm87MacroFeedV4Bf16AbGridCtas = 125U;
inline constexpr std::size_t kSm87MacroFeedV4Bf16AbPhysicalKernelLaunches =
    1U;
inline constexpr std::size_t kSm87MacroFeedV4Bf16AbDynamicSharedBytes =
    46'080U;
inline constexpr std::size_t kSm87MacroFeedV4Bf16AbPointerAlignment = 16U;
inline constexpr std::size_t kSm87MacroFeedV4Bf16AbSmCount = 16U;
inline constexpr std::size_t kSm87MacroFeedV4Bf16AbRequiredCtasPerSm = 2U;
inline constexpr std::size_t kSm87MacroFeedV4Bf16AbOracleTokens = 64U;

inline constexpr std::uint64_t kSm87MacroFeedV4Bf16AbWeightElements =
    kSm87MacroFeedV4Bf16AbRowsPerProjection *
    kSm87MacroFeedV4Bf16AbInputFeatures;
inline constexpr std::uint64_t kSm87MacroFeedV4Bf16AbWeightBytes =
    kSm87MacroFeedV4Bf16AbWeightElements * sizeof(std::uint16_t);
inline constexpr std::uint64_t kSm87MacroFeedV4Bf16AbInputBytes =
    kSm87MacroFeedV4Bf16AbTokens *
    kSm87MacroFeedV4Bf16AbInputFeatures * sizeof(std::uint16_t);
inline constexpr std::uint64_t kSm87MacroFeedV4Bf16AbScratchBytes =
    kSm87MacroFeedV4Bf16AbTokens *
    kSm87MacroFeedV4Bf16AbScratchRowStride * sizeof(std::uint16_t);

static_assert(kSm87MacroFeedV4Bf16AbLogicalRows ==
              2U * kSm87MacroFeedV4Bf16AbRowsPerProjection);
static_assert(kSm87MacroFeedV4Bf16AbBOffset ==
              kSm87MacroFeedV4Bf16AbAOffset +
                  kSm87MacroFeedV4Bf16AbRowsPerProjection);
static_assert(kSm87MacroFeedV4Bf16AbBOffset +
                      kSm87MacroFeedV4Bf16AbRowsPerProjection <=
                  kSm87MacroFeedV4Bf16AbScratchRowStride);
static_assert(kSm87MacroFeedV4Bf16AbTokens %
                      kSm87MacroFeedV4Bf16AbTileTokens ==
                  0U);
static_assert(kSm87MacroFeedV4Bf16AbGridCtas *
                      kSm87MacroFeedV4Bf16AbTileTokens ==
                  kSm87MacroFeedV4Bf16AbTokens);
static_assert(kSm87MacroFeedV4Bf16AbWeightBytes == 491'520U);
static_assert(kSm87MacroFeedV4Bf16AbInputBytes == 81'920'000U);
static_assert(kSm87MacroFeedV4Bf16AbScratchBytes == 278'528'000U);

struct Sm87MacroFeedV4Bf16AbPlan final {
  Sm87MacroFeedV4Bf16AbIdentity identity =
      Sm87MacroFeedV4Bf16AbIdentity::kInvalid;
  std::size_t token_count = 0U;
  std::size_t input_features = 0U;
  std::size_t rows_per_projection = 0U;
  std::size_t scratch_row_stride = 0U;
  std::size_t a_offset = 0U;
  std::size_t b_offset = 0U;
  std::size_t tile_tokens = 0U;
  std::size_t grid_ctas = 0U;
  std::size_t threads_per_cta = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  std::uint32_t physical_kernel_launches = 0U;
  bool established_exact_body_reused = false;
  bool shared_input_dual_projection = false;
  bool direct_scratch_scatter = false;
  bool compact_bridge_present = true;
  bool tail_present = true;
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
    return identity == kSm87MacroFeedV4Bf16AbIdentity &&
           token_count == kSm87MacroFeedV4Bf16AbTokens &&
           input_features == kSm87MacroFeedV4Bf16AbInputFeatures &&
           rows_per_projection ==
               kSm87MacroFeedV4Bf16AbRowsPerProjection &&
           scratch_row_stride ==
               kSm87MacroFeedV4Bf16AbScratchRowStride &&
           a_offset == kSm87MacroFeedV4Bf16AbAOffset &&
           b_offset == kSm87MacroFeedV4Bf16AbBOffset &&
           tile_tokens == kSm87MacroFeedV4Bf16AbTileTokens &&
           grid_ctas == kSm87MacroFeedV4Bf16AbGridCtas &&
           grid_ctas * tile_tokens == token_count &&
           threads_per_cta == kSm87MacroFeedV4Bf16AbThreads &&
           dynamic_shared_bytes ==
               kSm87MacroFeedV4Bf16AbDynamicSharedBytes &&
           physical_kernel_launches ==
               kSm87MacroFeedV4Bf16AbPhysicalKernelLaunches &&
           established_exact_body_reused && shared_input_dual_projection &&
           direct_scratch_scatter && !compact_bridge_present &&
           !tail_present && !selector_present && !fallback_permitted &&
           !jit_permitted && !runtime_repack_permitted &&
           !autotune_permitted && default_off &&
           !numerical_contract_qualified &&
           !production_dispatch_eligible && startup_package_unbound &&
           !execution_capability &&
           !caller_snapshot_grants_production_authority;
  }
};

[[nodiscard]] constexpr Sm87MacroFeedV4Bf16AbPlan
sm87_macrofeed_v4_bf16_ab_plan(const std::size_t token_count,
                               const std::size_t scratch_row_stride) noexcept {
  if (token_count != kSm87MacroFeedV4Bf16AbTokens ||
      scratch_row_stride != kSm87MacroFeedV4Bf16AbScratchRowStride) {
    return {};
  }
  return {kSm87MacroFeedV4Bf16AbIdentity,
          token_count,
          kSm87MacroFeedV4Bf16AbInputFeatures,
          kSm87MacroFeedV4Bf16AbRowsPerProjection,
          scratch_row_stride,
          kSm87MacroFeedV4Bf16AbAOffset,
          kSm87MacroFeedV4Bf16AbBOffset,
          kSm87MacroFeedV4Bf16AbTileTokens,
          kSm87MacroFeedV4Bf16AbGridCtas,
          kSm87MacroFeedV4Bf16AbThreads,
          kSm87MacroFeedV4Bf16AbDynamicSharedBytes,
          static_cast<std::uint32_t>(
              kSm87MacroFeedV4Bf16AbPhysicalKernelLaunches),
          true,
          true,
          true,
          false,
          false,
          false,
          false,
          false,
          false,
          false,
          true,
          false,
          false,
          true,
          false,
          false};
}

struct Sm87MacroFeedV4Bf16AbByteRange final {
  std::uintptr_t begin = 0U;
  std::uintptr_t end = 0U;
  bool valid = false;
};

[[nodiscard]] constexpr Sm87MacroFeedV4Bf16AbByteRange
sm87_macrofeed_v4_bf16_ab_byte_range(
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

[[nodiscard]] constexpr bool sm87_macrofeed_v4_bf16_ab_ranges_disjoint(
    const Sm87MacroFeedV4Bf16AbByteRange& left,
    const Sm87MacroFeedV4Bf16AbByteRange& right) noexcept {
  return left.valid && right.valid &&
         (left.end <= right.begin || right.end <= left.begin);
}

[[nodiscard]] constexpr bool sm87_macrofeed_v4_bf16_ab_pointer_aligned(
    const void* const pointer) noexcept {
  return pointer != nullptr &&
         reinterpret_cast<std::uintptr_t>(pointer) %
                 kSm87MacroFeedV4Bf16AbPointerAlignment ==
             0U;
}

struct Sm87MacroFeedV4Bf16AbArguments final {
  const std::uint16_t* a_weights = nullptr;  // canonical [48,5120]
  const std::uint16_t* b_weights = nullptr;  // canonical [48,5120]
  const std::uint16_t* input = nullptr;      // token-major [8000,5120]
  std::uint16_t* scratch = nullptr;          // BF16 [8000,17408] origin
  std::size_t token_count = 0U;
  std::size_t scratch_row_stride = 0U;
  // T1 caller precondition only.  A future private startup-issued package
  // must own/authenticate the stream and completion event.
  void* cuda_stream = nullptr;
};

// Structural/range validation only.  Host-only fake ranges may satisfy this
// seam for T0 tests.  The admission launcher additionally revalidates the
// current device, live-stream device, exact resource snapshot, and complete
// CUDA device allocation ranges before enqueue.
[[nodiscard]] bool sm87_macrofeed_v4_bf16_ab_arguments_valid(
    const Sm87MacroFeedV4Bf16AbArguments& arguments) noexcept;

struct Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot final {
  Sm87MacroFeedV4Bf16AbIdentity identity =
      Sm87MacroFeedV4Bf16AbIdentity::kInvalid;
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
  std::int32_t threads_per_block = 0;
  std::int32_t physical_grid_ctas = 0;
  bool kernel_compiled = false;
  bool exact_geometry = false;
  bool static_resource_gate_passed = false;
  bool numerical_contract_qualified = false;
  bool production_dispatch_eligible = false;
  bool startup_package_unbound = true;
  bool execution_capability = false;
  bool caller_snapshot_grants_production_authority = false;
};

// local_bytes is deliberately observed rather than assumed.  The gate binds
// the exact kernel geometry, dynamic shared-memory footprint and required
// 2-CTA/SM occupancy; it does not turn a caller-fillable snapshot into an
// execution capability.
[[nodiscard]] constexpr bool
sm87_macrofeed_v4_bf16_ab_admission_resource_gate(
    const Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot& resources) noexcept {
  return resources.identity == kSm87MacroFeedV4Bf16AbIdentity &&
         resources.device_ordinal >= 0 && resources.compute_major == 8 &&
         resources.compute_minor == 7 &&
         resources.sm_count ==
             static_cast<std::int32_t>(kSm87MacroFeedV4Bf16AbSmCount) &&
         resources.binary_version == 87 &&
         resources.registers_per_thread > 0 &&
         resources.dynamic_shared_bytes ==
             kSm87MacroFeedV4Bf16AbDynamicSharedBytes &&
         resources.maximum_threads_per_block >=
             static_cast<std::int32_t>(kSm87MacroFeedV4Bf16AbThreads) &&
         resources.active_blocks_per_sm >=
             static_cast<std::int32_t>(
                 kSm87MacroFeedV4Bf16AbRequiredCtasPerSm) &&
         resources.threads_per_block ==
             static_cast<std::int32_t>(kSm87MacroFeedV4Bf16AbThreads) &&
         resources.physical_grid_ctas ==
             static_cast<std::int32_t>(kSm87MacroFeedV4Bf16AbGridCtas) &&
         resources.kernel_compiled && resources.exact_geometry &&
         resources.static_resource_gate_passed &&
         !resources.numerical_contract_qualified &&
         !resources.production_dispatch_eligible &&
         resources.startup_package_unbound &&
         !resources.execution_capability &&
         !resources.caller_snapshot_grants_production_authority;
}

[[nodiscard]] int
query_sm87_macrofeed_v4_bf16_ab_admission_resource_snapshot_cuda(
    Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot* resources) noexcept;

struct Sm87MacroFeedV4Bf16AbAdmissionLaunchReceipt final {
  Sm87MacroFeedV4Bf16AbIdentity identity =
      Sm87MacroFeedV4Bf16AbIdentity::kInvalid;
  std::size_t token_count = 0U;
  std::size_t scratch_row_stride = 0U;
  std::size_t a_offset = 0U;
  std::size_t b_offset = 0U;
  std::uint32_t physical_kernel_launches = 0U;
  bool established_exact_body_reused = false;
  bool shared_input_dual_projection = false;
  bool direct_scratch_scatter = false;
  bool compact_bridge_absent = false;
  bool current_device_revalidated = false;
  bool caller_snapshot_exact_observed_match = false;
  bool caller_supplied_live_stream_required = false;
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
    return identity == kSm87MacroFeedV4Bf16AbIdentity &&
           token_count == kSm87MacroFeedV4Bf16AbTokens &&
           scratch_row_stride ==
               kSm87MacroFeedV4Bf16AbScratchRowStride &&
           a_offset == kSm87MacroFeedV4Bf16AbAOffset &&
           b_offset == kSm87MacroFeedV4Bf16AbBOffset &&
           physical_kernel_launches ==
               kSm87MacroFeedV4Bf16AbPhysicalKernelLaunches &&
           established_exact_body_reused && shared_input_dual_projection &&
           direct_scratch_scatter && compact_bridge_absent &&
           current_device_revalidated &&
           caller_snapshot_exact_observed_match &&
           caller_supplied_live_stream_required &&
           live_stream_device_observed && device_allocation_ranges_owned &&
           launch_enqueued && !completion_observed &&
           !numerical_contract_qualified &&
           !production_dispatch_eligible && startup_package_unbound &&
           !execution_capability &&
           !caller_snapshot_grants_production_authority;
  }
};

// Default-off admission-only executor.  It enqueues one fixed
// dim3(125,1,1) x dim3(256,1,1) grid and writes A/B directly to scratch
// columns [16384,16432) and [16432,16480).  The receipt proves enqueue and
// observed admission facts only; it is not a completion or production token.
[[nodiscard]] int launch_sm87_macrofeed_v4_bf16_ab_admission_cuda(
    const Sm87MacroFeedV4Bf16AbArguments& arguments,
    const Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot& resources,
    Sm87MacroFeedV4Bf16AbAdmissionLaunchReceipt* receipt) noexcept;

// Synthetic correctness seam only.  It launches one CTA of the identical
// production arithmetic body for exactly M64 and retains the V4 row layout.
// It has no timing, selector, startup, whole-model, or production authority.
struct Sm87MacroFeedV4Bf16AbOracleArguments final {
  const std::uint16_t* a_weights = nullptr;
  const std::uint16_t* b_weights = nullptr;
  const std::uint16_t* input = nullptr;  // [64,5120]
  std::uint16_t* scratch = nullptr;      // [64,17408]
  std::size_t token_count = 0U;
  std::size_t scratch_row_stride = 0U;
  void* cuda_stream = nullptr;
};

[[nodiscard]] int launch_sm87_macrofeed_v4_bf16_ab_oracle_cuda(
    const Sm87MacroFeedV4Bf16AbOracleArguments& arguments) noexcept;

// Synthetic compact-output counterpart for the bitwise test above.  This is
// a V4-owned oracle, not the legacy large-M admission ABI: it launches the
// same body once with independent compact [64,48] A/B outputs and has no tail
// or production authority.
struct Sm87MacroFeedV4Bf16AbCompactOracleArguments final {
  const std::uint16_t* a_weights = nullptr;
  const std::uint16_t* b_weights = nullptr;
  const std::uint16_t* input = nullptr;  // [64,5120]
  std::uint16_t* a_output = nullptr;     // [64,48]
  std::uint16_t* b_output = nullptr;     // [64,48]
  std::size_t token_count = 0U;
  void* cuda_stream = nullptr;
};

[[nodiscard]] int launch_sm87_macrofeed_v4_bf16_ab_compact_oracle_cuda(
    const Sm87MacroFeedV4Bf16AbCompactOracleArguments& arguments) noexcept;

static_assert(sm87_macrofeed_v4_bf16_ab_plan(
                  kSm87MacroFeedV4Bf16AbTokens,
                  kSm87MacroFeedV4Bf16AbScratchRowStride)
                  .valid());
static_assert(!sm87_macrofeed_v4_bf16_ab_plan(
                   kSm87MacroFeedV4Bf16AbTokens - 1U,
                   kSm87MacroFeedV4Bf16AbScratchRowStride)
                   .valid());

}  // namespace q3x::kernels
