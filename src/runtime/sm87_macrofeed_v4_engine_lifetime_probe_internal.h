#pragma once

#if !defined(Q3X_ENABLE_SM87_MACROFEED_V3_P40_EXECUTOR_ADMISSION) || \
    !defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION) || \
    !defined(Q3X_ENABLE_SM87_MACROFEED_V4_FULL_ATTENTION_PREPROCESS_ADMISSION) || \
    !defined(Q3X_ENABLE_SM87_MACROFEED_V4_ATTENTION_C8000_ADMISSION)
#error "the V4 Engine lifetime probe requires combined V3/V4 plus both Full-Attention admissions"
#endif

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::runtime::sm87_macrofeed_v4_engine_lifetime_probe_detail {

inline constexpr std::size_t
    kSm87MacroFeedV4EngineLifetimeExpectedCompleteGdnBindings = 48U;
inline constexpr std::size_t
    kSm87MacroFeedV4EngineLifetimeExpectedMlpPairBindings = 64U;
inline constexpr std::size_t
    kSm87MacroFeedV4EngineLifetimeExpectedFullAttentionBindings = 16U;
inline constexpr std::size_t
    kSm87MacroFeedV4EngineLifetimeExpectedRequestBoundaryBindings = 1U;
inline constexpr std::size_t
    kSm87MacroFeedV4EngineLifetimeExpectedRequestBoundaryResourceQueries = 4U;
// Device-only Execution ownership.  Pinned host ownership is accounted
// separately below and never enters cudaMemGetInfo reserve arithmetic.
inline constexpr std::uint64_t
    kSm87MacroFeedV4EngineLifetimeExpectedOwnedBytes = 3'220'701'184U;
inline constexpr std::uint64_t
    kSm87MacroFeedV4EngineLifetimeExpectedHostOwnedBytes = 160'008U;
inline constexpr std::uint64_t
    kSm87MacroFeedV4EngineLifetimeExpectedTotalOwnedBytes = 3'220'861'192U;
inline constexpr std::uint32_t
    kSm87MacroFeedV4EngineLifetimeExpectedPinnedHostFlags = 1U;
inline constexpr std::uint64_t
    kSm87MacroFeedV4EngineLifetimeExpectedScratchAliasSpanBytes = 507'144U;
inline constexpr std::uint64_t
    kSm87MacroFeedV4EngineLifetimeExpectedRequestBoundaryResidentBytes =
        20'150'786'560U;
inline constexpr std::uint64_t
    kSm87MacroFeedV4EngineLifetimeExpectedTransientBytes = 442'368'000U;
inline constexpr std::uint64_t
    kSm87MacroFeedV4EngineLifetimeMinimumOwnedArenaBytes = 156'893'184U;
inline constexpr std::uint64_t
    kSm87MacroFeedV4EngineLifetimeExpectedKvArenaBytes = 2'621'440'000U;
inline constexpr std::uint64_t
    kSm87MacroFeedV4EngineLifetimeExpectedRopeBytes = 67'108'864U;
inline constexpr std::size_t
    kSm87MacroFeedV4EngineLifetimeExpectedRopePositions = 262'144U;
inline constexpr std::size_t
    kSm87MacroFeedV4EngineLifetimeExpectedRopePairs = 32U;
inline constexpr std::uint64_t
    kSm87MacroFeedV4EngineLifetimeExpectedAnchoredBytes = 3'287'810'048U;
inline constexpr std::uint64_t
    kSm87MacroFeedV4EngineLifetimeExpectedTotalAnchoredBytes = 3'287'970'056U;
inline constexpr std::uint64_t
    kSm87MacroFeedV4EngineLifetimeLegacyRequestArenaBytes = 8'640'542'976U;
inline constexpr std::uint64_t
    kSm87MacroFeedV4EngineLifetimeFutureAfterExecutionBytes =
        kSm87MacroFeedV4EngineLifetimeLegacyRequestArenaBytes;
inline constexpr std::uint64_t
    kSm87MacroFeedV4EngineLifetimeFutureAfterRopeBytes =
        kSm87MacroFeedV4EngineLifetimeExpectedOwnedBytes +
        kSm87MacroFeedV4EngineLifetimeLegacyRequestArenaBytes;
inline constexpr std::uint64_t
    kSm87MacroFeedV4EngineLifetimeFutureAfterCompleteAotBytes =
        kSm87MacroFeedV4EngineLifetimeExpectedRopeBytes +
        kSm87MacroFeedV4EngineLifetimeFutureAfterRopeBytes;

static_assert(kSm87MacroFeedV4EngineLifetimeExpectedOwnedBytes ==
              kSm87MacroFeedV4EngineLifetimeExpectedTransientBytes +
                  kSm87MacroFeedV4EngineLifetimeMinimumOwnedArenaBytes +
                  kSm87MacroFeedV4EngineLifetimeExpectedKvArenaBytes);
static_assert(kSm87MacroFeedV4EngineLifetimeExpectedAnchoredBytes ==
              kSm87MacroFeedV4EngineLifetimeExpectedOwnedBytes +
                  kSm87MacroFeedV4EngineLifetimeExpectedRopeBytes);
static_assert(kSm87MacroFeedV4EngineLifetimeExpectedTotalOwnedBytes ==
              kSm87MacroFeedV4EngineLifetimeExpectedOwnedBytes +
                  kSm87MacroFeedV4EngineLifetimeExpectedHostOwnedBytes);
static_assert(kSm87MacroFeedV4EngineLifetimeExpectedTotalAnchoredBytes ==
              kSm87MacroFeedV4EngineLifetimeExpectedTotalOwnedBytes +
                  kSm87MacroFeedV4EngineLifetimeExpectedRopeBytes);
static_assert(kSm87MacroFeedV4EngineLifetimeFutureAfterRopeBytes ==
              11'861'244'160U);
static_assert(kSm87MacroFeedV4EngineLifetimeFutureAfterCompleteAotBytes ==
              11'928'353'024U);

// Capability-free construction snapshot.  It contains no ModelWeights,
// launch-capable typed device pointer, CUDA asset, stream/event handle,
// launch receipt, or route selector.  Integer allocation bounds are evidence
// only.  The Engine publishes it only after the normal startup and execution
// packages have been assigned to their final lifetime root.
struct Sm87MacroFeedV4EngineLifetimeConstructionSnapshot final {
  std::uint64_t lifetime_root_identity = 0U;
  std::uint64_t startup_package_identity = 0U;
  std::uint64_t execution_package_identity = 0U;
  std::uint64_t execution_startup_package_identity = 0U;
  std::uint64_t startup_owner_identity = 0U;
  std::uint64_t startup_allocation_identity = 0U;
  std::uint64_t startup_device_identity = 0U;
  std::int32_t startup_device_ordinal = -1;
  std::int32_t execution_device_ordinal = -1;
  std::uint64_t startup_complete_gdn_source_catalog_identity = 0U;
  std::uint64_t startup_mlp_source_catalog_identity = 0U;
  std::uint64_t startup_full_attention_source_catalog_identity = 0U;
  std::uint64_t startup_request_boundary_source_catalog_identity = 0U;
  std::uint64_t startup_request_boundary_resident_root_identity = 0U;
  std::uint64_t startup_request_boundary_resident_arena_bytes = 0U;
  std::size_t startup_request_boundary_source_bindings = 0U;
  std::uint64_t execution_complete_gdn_catalog_identity = 0U;
  std::size_t execution_complete_gdn_binding_count = 0U;
  std::uint64_t execution_mlp_pair_catalog_identity = 0U;
  std::size_t execution_mlp_pair_binding_count = 0U;
  std::uint64_t execution_full_attention_catalog_identity = 0U;
  std::size_t execution_full_attention_binding_count = 0U;
  std::uint64_t retained_complete_gdn_catalog_fold_identity = 0U;
  std::uint64_t retained_mlp_pair_catalog_fold_identity = 0U;
  std::uint64_t retained_full_attention_catalog_fold_identity = 0U;
  std::uint64_t full_attention_resource_bundle_identity = 0U;
  std::uint64_t execution_request_boundary_catalog_identity = 0U;
  std::uint64_t retained_request_boundary_catalog_fold_identity = 0U;
  std::uint64_t request_boundary_source_catalog_identity = 0U;
  std::uint64_t request_boundary_resource_bundle_identity = 0U;
  std::uint64_t request_boundary_binding_identity = 0U;
  std::uint64_t request_boundary_resident_root_identity = 0U;
  std::uint64_t request_boundary_resident_arena_bytes = 0U;
  std::size_t execution_request_boundary_binding_count = 0U;
  std::size_t execution_request_boundary_resource_queries = 0U;
  std::uint64_t transient_allocation_identity = 0U;
  std::uint64_t recurrent_allocation_identity = 0U;
  std::uint64_t kv_allocation_identity = 0U;
  std::uint64_t request_state_kv_allocation_identity = 0U;
  std::uintptr_t kv_allocation_begin = 0U;
  std::uint64_t engine_rope_owner_identity = 0U;
  std::uint64_t engine_rope_binding_identity = 0U;
  std::uintptr_t engine_rope_allocation_begin = 0U;
  std::int32_t engine_rope_device_ordinal = -1;
  std::size_t engine_rope_positions = 0U;
  std::size_t engine_rope_pairs = 0U;
  std::uint64_t execution_events_owner_identity = 0U;
  std::uint64_t request_boundary_host_staging_allocation_identity = 0U;
  std::uintptr_t request_boundary_host_staging_begin = 0U;
  std::uint64_t request_boundary_scratch_alias_identity = 0U;
  std::uint64_t transient_bytes = 0U;
  std::uint64_t recurrent_bytes = 0U;
  std::uint64_t kv_allocation_bytes = 0U;
  std::uint64_t request_state_kv_allocation_bytes = 0U;
  std::uint64_t engine_rope_allocation_bytes = 0U;
  std::uint64_t request_boundary_host_staging_bytes = 0U;
  std::uint64_t request_boundary_host_owned_bytes = 0U;
  std::uint64_t request_boundary_scratch_alias_span_bytes = 0U;
  std::uint32_t request_boundary_host_staging_flags = 0U;
  std::uint64_t execution_required_device_allocation_bytes = 0U;
  std::uint64_t execution_minimum_free_bytes_after_create = 0U;
  std::uint64_t execution_free_bytes_before_allocations = 0U;
  std::uint64_t execution_free_bytes_after_allocations = 0U;
  // Retained compatibility field: device-only Execution ownership.
  std::uint64_t owned_bytes = 0U;
  std::uint64_t total_owned_bytes = 0U;
  std::uint64_t anchored_bytes = 0U;
  std::uint64_t total_anchored_bytes = 0U;
  std::uint64_t legacy_request_arena_bytes = 0U;
  std::uint64_t minimum_free_bytes_after_legacy_create = 0U;
  std::uint64_t minimum_free_bytes_after_rope_create = 0U;
  std::uint64_t minimum_free_bytes_after_complete_aot_create = 0U;
  bool request_state_kv_physical_owner_bound = false;
  bool execution_aggregate_memory_gate_passed = false;
  bool startup_request_boundary_normal_resident_authority = false;
  bool startup_request_boundary_host_test_resident_authority = true;
  bool request_boundary_execution_catalog_bound = false;
  bool request_boundary_source_private_resource_queries = false;
  bool request_boundary_normal_resident_authority = false;
  bool request_boundary_host_test_resident_authority = true;
  bool request_boundary_synthetic_unbound = true;
  bool request_boundary_host_staging_pinned = false;
  bool request_boundary_host_staging_construction_zero_initialized = false;
  bool request_boundary_scratch_aliases_exact = false;
  bool request_boundary_request_selectable = true;
  bool request_boundary_launcher_authority = true;
  bool request_boundary_production_dispatch_eligible = true;
  bool normal_factory_branch = false;
  bool synthetic_t1_gdn_layer0_source = true;
  bool complete_gdn_layer0_bound = false;
  bool whole_model_executor_bound = true;
  bool selector_bound = true;
  bool api_route_bound = true;
  bool default_off = false;
  bool production_dispatch_eligible = true;
  bool lifetime_chain_sealed = false;

  [[nodiscard]] constexpr std::uint64_t
  compute_lifetime_root_identity() const noexcept {
    const auto mix = [](std::uint64_t state,
                        const std::uint64_t value) constexpr noexcept {
      state ^= value + 0x9e37'79b9'7f4a'7c15ULL + (state << 6U) +
               (state >> 2U);
      state ^= state >> 30U;
      state *= 0xbf58'476d'1ce4'e5b9ULL;
      state ^= state >> 27U;
      state *= 0x94d0'49bb'1331'11ebULL;
      state ^= state >> 31U;
      return state;
    };
    std::uint64_t identity = 0x5133'4d46'5634'4c52ULL;
    identity = mix(identity, startup_package_identity);
    identity = mix(identity, execution_package_identity);
    identity = mix(identity, execution_startup_package_identity);
    identity = mix(identity, startup_owner_identity);
    identity = mix(identity, startup_allocation_identity);
    identity = mix(identity, startup_device_identity);
    identity = mix(identity,
                   static_cast<std::uint64_t>(startup_device_ordinal + 1));
    identity = mix(identity,
                   static_cast<std::uint64_t>(execution_device_ordinal + 1));
    identity = mix(identity,
                   startup_complete_gdn_source_catalog_identity);
    identity = mix(identity, startup_mlp_source_catalog_identity);
    identity = mix(identity, startup_full_attention_source_catalog_identity);
    identity = mix(identity,
                   startup_request_boundary_source_catalog_identity);
    identity = mix(identity,
                   startup_request_boundary_resident_root_identity);
    identity = mix(identity,
                   startup_request_boundary_resident_arena_bytes);
    identity = mix(identity, startup_request_boundary_source_bindings);
    identity = mix(identity, execution_complete_gdn_catalog_identity);
    identity = mix(identity, execution_complete_gdn_binding_count);
    identity = mix(identity, execution_mlp_pair_catalog_identity);
    identity = mix(identity, execution_mlp_pair_binding_count);
    identity = mix(identity, execution_full_attention_catalog_identity);
    identity = mix(identity, execution_full_attention_binding_count);
    identity = mix(identity,
                   retained_complete_gdn_catalog_fold_identity);
    identity = mix(identity, retained_mlp_pair_catalog_fold_identity);
    identity = mix(identity, retained_full_attention_catalog_fold_identity);
    identity = mix(identity, full_attention_resource_bundle_identity);
    identity = mix(identity, execution_request_boundary_catalog_identity);
    identity = mix(identity,
                   retained_request_boundary_catalog_fold_identity);
    identity = mix(identity, request_boundary_source_catalog_identity);
    identity = mix(identity, request_boundary_resource_bundle_identity);
    identity = mix(identity, request_boundary_binding_identity);
    identity = mix(identity, request_boundary_resident_root_identity);
    identity = mix(identity, request_boundary_resident_arena_bytes);
    identity = mix(identity, execution_request_boundary_binding_count);
    identity = mix(identity, execution_request_boundary_resource_queries);
    identity = mix(identity, transient_allocation_identity);
    identity = mix(identity, recurrent_allocation_identity);
    identity = mix(identity, kv_allocation_identity);
    identity = mix(identity, request_state_kv_allocation_identity);
    identity = mix(identity, kv_allocation_begin);
    identity = mix(identity, engine_rope_owner_identity);
    identity = mix(identity, engine_rope_binding_identity);
    identity = mix(identity, engine_rope_allocation_begin);
    identity = mix(identity,
                   static_cast<std::uint64_t>(engine_rope_device_ordinal + 1));
    identity = mix(identity, engine_rope_positions);
    identity = mix(identity, engine_rope_pairs);
    identity = mix(identity, execution_events_owner_identity);
    identity = mix(identity,
                   request_boundary_host_staging_allocation_identity);
    identity = mix(identity, request_boundary_host_staging_begin);
    identity = mix(identity, request_boundary_scratch_alias_identity);
    identity = mix(identity, transient_bytes);
    identity = mix(identity, recurrent_bytes);
    identity = mix(identity, kv_allocation_bytes);
    identity = mix(identity, request_state_kv_allocation_bytes);
    identity = mix(identity, engine_rope_allocation_bytes);
    identity = mix(identity, request_boundary_host_staging_bytes);
    identity = mix(identity, request_boundary_host_owned_bytes);
    identity = mix(identity, request_boundary_scratch_alias_span_bytes);
    identity = mix(identity, request_boundary_host_staging_flags);
    identity = mix(identity, execution_required_device_allocation_bytes);
    identity = mix(identity, execution_minimum_free_bytes_after_create);
    identity = mix(identity, execution_free_bytes_before_allocations);
    identity = mix(identity, execution_free_bytes_after_allocations);
    identity = mix(identity, owned_bytes);
    identity = mix(identity, total_owned_bytes);
    identity = mix(identity, anchored_bytes);
    identity = mix(identity, total_anchored_bytes);
    identity = mix(identity, legacy_request_arena_bytes);
    identity = mix(identity, minimum_free_bytes_after_legacy_create);
    identity = mix(identity, minimum_free_bytes_after_rope_create);
    identity = mix(identity, minimum_free_bytes_after_complete_aot_create);
    identity = mix(identity, request_state_kv_physical_owner_bound);
    identity = mix(identity, execution_aggregate_memory_gate_passed);
    identity = mix(
        identity, startup_request_boundary_normal_resident_authority);
    identity = mix(
        identity, startup_request_boundary_host_test_resident_authority);
    identity = mix(identity, request_boundary_execution_catalog_bound);
    identity = mix(identity,
                   request_boundary_source_private_resource_queries);
    identity = mix(identity, request_boundary_normal_resident_authority);
    identity = mix(identity, request_boundary_host_test_resident_authority);
    identity = mix(identity, request_boundary_synthetic_unbound);
    identity = mix(identity, request_boundary_host_staging_pinned);
    identity = mix(identity,
                   request_boundary_host_staging_construction_zero_initialized);
    identity = mix(identity, request_boundary_scratch_aliases_exact);
    identity = mix(identity, request_boundary_request_selectable);
    identity = mix(identity, request_boundary_launcher_authority);
    identity = mix(identity,
                   request_boundary_production_dispatch_eligible);
    identity = mix(identity, normal_factory_branch);
    identity = mix(identity, synthetic_t1_gdn_layer0_source);
    identity = mix(identity, complete_gdn_layer0_bound);
    identity = mix(identity, whole_model_executor_bound);
    identity = mix(identity, selector_bound);
    identity = mix(identity, api_route_bound);
    identity = mix(identity, default_off);
    identity = mix(identity, production_dispatch_eligible);
    identity = mix(identity, lifetime_chain_sealed);
    return identity == 0U ? 1U : identity;
  }

  [[nodiscard]] constexpr bool valid() const noexcept {
    return lifetime_root_identity != 0U &&
           lifetime_root_identity == compute_lifetime_root_identity() &&
           startup_package_identity != 0U &&
           execution_package_identity != 0U &&
           execution_startup_package_identity == startup_package_identity &&
           startup_owner_identity != 0U &&
           startup_allocation_identity != 0U &&
           startup_device_identity != 0U && startup_device_ordinal >= 0 &&
           execution_device_ordinal == startup_device_ordinal &&
           startup_complete_gdn_source_catalog_identity != 0U &&
           startup_mlp_source_catalog_identity != 0U &&
           startup_full_attention_source_catalog_identity != 0U &&
           startup_request_boundary_source_catalog_identity != 0U &&
           startup_request_boundary_resident_root_identity != 0U &&
           startup_request_boundary_resident_arena_bytes ==
               kSm87MacroFeedV4EngineLifetimeExpectedRequestBoundaryResidentBytes &&
           startup_request_boundary_source_bindings ==
               kSm87MacroFeedV4EngineLifetimeExpectedRequestBoundaryBindings &&
           startup_request_boundary_normal_resident_authority &&
           !startup_request_boundary_host_test_resident_authority &&
           execution_complete_gdn_catalog_identity != 0U &&
           execution_complete_gdn_binding_count ==
               kSm87MacroFeedV4EngineLifetimeExpectedCompleteGdnBindings &&
           execution_mlp_pair_catalog_identity != 0U &&
           execution_mlp_pair_binding_count ==
               kSm87MacroFeedV4EngineLifetimeExpectedMlpPairBindings &&
           execution_full_attention_catalog_identity != 0U &&
           execution_full_attention_binding_count ==
               kSm87MacroFeedV4EngineLifetimeExpectedFullAttentionBindings &&
           retained_complete_gdn_catalog_fold_identity != 0U &&
           retained_mlp_pair_catalog_fold_identity != 0U &&
           retained_full_attention_catalog_fold_identity != 0U &&
           full_attention_resource_bundle_identity != 0U &&
           execution_request_boundary_catalog_identity != 0U &&
           retained_request_boundary_catalog_fold_identity != 0U &&
           request_boundary_source_catalog_identity ==
               startup_request_boundary_source_catalog_identity &&
           request_boundary_resource_bundle_identity != 0U &&
           request_boundary_binding_identity != 0U &&
           request_boundary_resident_root_identity != 0U &&
           request_boundary_resident_root_identity ==
               startup_request_boundary_resident_root_identity &&
           request_boundary_resident_arena_bytes ==
               kSm87MacroFeedV4EngineLifetimeExpectedRequestBoundaryResidentBytes &&
           request_boundary_resident_arena_bytes ==
               startup_request_boundary_resident_arena_bytes &&
           execution_request_boundary_binding_count ==
               kSm87MacroFeedV4EngineLifetimeExpectedRequestBoundaryBindings &&
           execution_request_boundary_resource_queries ==
               kSm87MacroFeedV4EngineLifetimeExpectedRequestBoundaryResourceQueries &&
           transient_allocation_identity != 0U &&
           recurrent_allocation_identity != 0U &&
           kv_allocation_identity != 0U && kv_allocation_begin != 0U &&
           request_state_kv_allocation_identity == kv_allocation_identity &&
           engine_rope_owner_identity != 0U &&
           engine_rope_binding_identity != 0U &&
           engine_rope_allocation_begin != 0U &&
           engine_rope_device_ordinal == execution_device_ordinal &&
           engine_rope_positions ==
               kSm87MacroFeedV4EngineLifetimeExpectedRopePositions &&
           engine_rope_pairs ==
               kSm87MacroFeedV4EngineLifetimeExpectedRopePairs &&
           execution_events_owner_identity != 0U &&
           request_boundary_host_staging_allocation_identity != 0U &&
           request_boundary_host_staging_begin != 0U &&
           request_boundary_scratch_alias_identity != 0U &&
           request_boundary_host_staging_allocation_identity !=
               transient_allocation_identity &&
           request_boundary_host_staging_allocation_identity !=
               recurrent_allocation_identity &&
           request_boundary_host_staging_allocation_identity !=
               kv_allocation_identity &&
           request_boundary_scratch_alias_identity !=
               transient_allocation_identity &&
           request_boundary_scratch_alias_identity !=
               recurrent_allocation_identity &&
           request_boundary_scratch_alias_identity !=
               kv_allocation_identity &&
           request_boundary_scratch_alias_identity !=
               request_boundary_host_staging_allocation_identity &&
           transient_allocation_identity != recurrent_allocation_identity &&
           transient_allocation_identity != kv_allocation_identity &&
           recurrent_allocation_identity != kv_allocation_identity &&
           transient_bytes ==
               kSm87MacroFeedV4EngineLifetimeExpectedTransientBytes &&
           recurrent_bytes ==
               kSm87MacroFeedV4EngineLifetimeMinimumOwnedArenaBytes &&
           kv_allocation_bytes ==
               kSm87MacroFeedV4EngineLifetimeExpectedKvArenaBytes &&
           request_state_kv_allocation_bytes == kv_allocation_bytes &&
           request_state_kv_physical_owner_bound &&
           engine_rope_allocation_bytes ==
               kSm87MacroFeedV4EngineLifetimeExpectedRopeBytes &&
           request_boundary_host_staging_bytes ==
               kSm87MacroFeedV4EngineLifetimeExpectedHostOwnedBytes &&
           request_boundary_host_owned_bytes ==
               request_boundary_host_staging_bytes &&
           request_boundary_scratch_alias_span_bytes ==
               kSm87MacroFeedV4EngineLifetimeExpectedScratchAliasSpanBytes &&
           request_boundary_host_staging_flags ==
               kSm87MacroFeedV4EngineLifetimeExpectedPinnedHostFlags &&
           execution_required_device_allocation_bytes == owned_bytes &&
           execution_aggregate_memory_gate_passed &&
           execution_free_bytes_before_allocations >=
               execution_required_device_allocation_bytes &&
           execution_minimum_free_bytes_after_create <=
               execution_free_bytes_before_allocations -
                   execution_required_device_allocation_bytes &&
           execution_free_bytes_after_allocations >=
               execution_minimum_free_bytes_after_create &&
           owned_bytes == transient_bytes + recurrent_bytes +
                              kv_allocation_bytes &&
           owned_bytes == kSm87MacroFeedV4EngineLifetimeExpectedOwnedBytes &&
           total_owned_bytes == owned_bytes + request_boundary_host_owned_bytes &&
           total_owned_bytes ==
               kSm87MacroFeedV4EngineLifetimeExpectedTotalOwnedBytes &&
           anchored_bytes == owned_bytes + engine_rope_allocation_bytes &&
           anchored_bytes ==
               kSm87MacroFeedV4EngineLifetimeExpectedAnchoredBytes &&
           total_anchored_bytes ==
               total_owned_bytes + engine_rope_allocation_bytes &&
           total_anchored_bytes ==
               kSm87MacroFeedV4EngineLifetimeExpectedTotalAnchoredBytes &&
           legacy_request_arena_bytes ==
               kSm87MacroFeedV4EngineLifetimeLegacyRequestArenaBytes &&
           minimum_free_bytes_after_legacy_create <=
               std::numeric_limits<std::uint64_t>::max() -
                   legacy_request_arena_bytes &&
           execution_minimum_free_bytes_after_create ==
               minimum_free_bytes_after_legacy_create +
                   legacy_request_arena_bytes &&
           execution_minimum_free_bytes_after_create <=
               std::numeric_limits<std::uint64_t>::max() - owned_bytes &&
           minimum_free_bytes_after_rope_create ==
               execution_minimum_free_bytes_after_create + owned_bytes &&
           minimum_free_bytes_after_rope_create <=
               std::numeric_limits<std::uint64_t>::max() -
                   engine_rope_allocation_bytes &&
           minimum_free_bytes_after_complete_aot_create ==
               minimum_free_bytes_after_rope_create +
                   engine_rope_allocation_bytes &&
           request_boundary_execution_catalog_bound &&
           request_boundary_source_private_resource_queries &&
           request_boundary_normal_resident_authority &&
           !request_boundary_host_test_resident_authority &&
           !request_boundary_synthetic_unbound &&
           request_boundary_host_staging_pinned &&
           request_boundary_host_staging_construction_zero_initialized &&
           request_boundary_scratch_aliases_exact &&
           !request_boundary_request_selectable &&
           !request_boundary_launcher_authority &&
           !request_boundary_production_dispatch_eligible &&
           normal_factory_branch && !synthetic_t1_gdn_layer0_source &&
           complete_gdn_layer0_bound && !whole_model_executor_bound &&
           !selector_bound && !api_route_bound && default_off &&
           !production_dispatch_eligible && lifetime_chain_sealed;
  }
};

using Sm87MacroFeedV4EngineLifetimeConstructionSnapshotCallback = void (*)(
    void* context,
    const Sm87MacroFeedV4EngineLifetimeConstructionSnapshot& snapshot)
    noexcept;

struct Sm87MacroFeedV4EngineLifetimeConstructionSnapshotHook final {
  Sm87MacroFeedV4EngineLifetimeConstructionSnapshotCallback callback =
      nullptr;
  void* context = nullptr;

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return callback != nullptr;
  }
};

// Source-private, test-only thread-local hook.  It observes construction but
// has no authority to construct, launch, retain, mutate, or destroy the root.
[[nodiscard]] Sm87MacroFeedV4EngineLifetimeConstructionSnapshotHook
exchange_sm87_macrofeed_v4_engine_lifetime_construction_snapshot_hook(
    Sm87MacroFeedV4EngineLifetimeConstructionSnapshotHook hook) noexcept;

}  // namespace q3x::runtime::sm87_macrofeed_v4_engine_lifetime_probe_detail
