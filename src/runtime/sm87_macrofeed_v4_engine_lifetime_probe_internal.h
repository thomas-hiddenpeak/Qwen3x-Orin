#pragma once

#if !defined(Q3X_ENABLE_SM87_MACROFEED_V3_P40_EXECUTOR_ADMISSION) || \
    !defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
#error "the V4 Engine lifetime probe requires the combined V3/V4 admission"
#endif

#include <cstddef>
#include <cstdint>

namespace q3x::runtime::sm87_macrofeed_v4_engine_lifetime_probe_detail {

inline constexpr std::size_t
    kSm87MacroFeedV4EngineLifetimeExpectedCompleteGdnBindings = 48U;
inline constexpr std::size_t
    kSm87MacroFeedV4EngineLifetimeExpectedMlpPairBindings = 64U;
inline constexpr std::uint64_t
    kSm87MacroFeedV4EngineLifetimeExpectedOwnedBytes = 599'261'184U;
inline constexpr std::uint64_t
    kSm87MacroFeedV4EngineLifetimeExpectedTransientBytes = 442'368'000U;
inline constexpr std::uint64_t
    kSm87MacroFeedV4EngineLifetimeMinimumOwnedArenaBytes = 156'893'184U;

// Capability-free construction snapshot.  It contains no ModelWeights,
// device pointer, CUDA asset, stream/event handle, launch receipt, or route
// selector.  The Engine publishes it only after the normal startup and
// execution packages have been assigned to their final lifetime root.
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
  std::uint64_t execution_complete_gdn_catalog_identity = 0U;
  std::size_t execution_complete_gdn_binding_count = 0U;
  std::uint64_t execution_mlp_pair_catalog_identity = 0U;
  std::size_t execution_mlp_pair_binding_count = 0U;
  std::uint64_t retained_complete_gdn_catalog_fold_identity = 0U;
  std::uint64_t retained_mlp_pair_catalog_fold_identity = 0U;
  std::uint64_t transient_allocation_identity = 0U;
  std::uint64_t recurrent_allocation_identity = 0U;
  std::uint64_t execution_events_owner_identity = 0U;
  std::uint64_t transient_bytes = 0U;
  std::uint64_t recurrent_bytes = 0U;
  std::uint64_t owned_bytes = 0U;
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
    identity = mix(identity, execution_complete_gdn_catalog_identity);
    identity = mix(identity, execution_complete_gdn_binding_count);
    identity = mix(identity, execution_mlp_pair_catalog_identity);
    identity = mix(identity, execution_mlp_pair_binding_count);
    identity = mix(identity,
                   retained_complete_gdn_catalog_fold_identity);
    identity = mix(identity, retained_mlp_pair_catalog_fold_identity);
    identity = mix(identity, transient_allocation_identity);
    identity = mix(identity, recurrent_allocation_identity);
    identity = mix(identity, execution_events_owner_identity);
    identity = mix(identity, transient_bytes);
    identity = mix(identity, recurrent_bytes);
    identity = mix(identity, owned_bytes);
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
           execution_complete_gdn_catalog_identity != 0U &&
           execution_complete_gdn_binding_count ==
               kSm87MacroFeedV4EngineLifetimeExpectedCompleteGdnBindings &&
           execution_mlp_pair_catalog_identity != 0U &&
           execution_mlp_pair_binding_count ==
               kSm87MacroFeedV4EngineLifetimeExpectedMlpPairBindings &&
           retained_complete_gdn_catalog_fold_identity != 0U &&
           retained_mlp_pair_catalog_fold_identity != 0U &&
           transient_allocation_identity != 0U &&
           recurrent_allocation_identity != 0U &&
           execution_events_owner_identity != 0U &&
           transient_allocation_identity != recurrent_allocation_identity &&
           transient_bytes ==
               kSm87MacroFeedV4EngineLifetimeExpectedTransientBytes &&
           recurrent_bytes ==
               kSm87MacroFeedV4EngineLifetimeMinimumOwnedArenaBytes &&
           owned_bytes == transient_bytes + recurrent_bytes &&
           owned_bytes == kSm87MacroFeedV4EngineLifetimeExpectedOwnedBytes &&
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
