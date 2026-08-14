#pragma once

#if !defined(Q3X_ENABLE_SM87_MACROFEED_V3_P40_EXECUTOR_ADMISSION) || \
    !defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
#error "the V4 Engine lifetime probe requires the combined V3/V4 admission"
#endif

#include <cstddef>
#include <cstdint>

namespace q3x::runtime::sm87_macrofeed_v4_engine_lifetime_probe_detail {

inline constexpr std::size_t
    kSm87MacroFeedV4EngineLifetimeExpectedGdnQkvZBindings = 48U;
inline constexpr std::uint64_t
    kSm87MacroFeedV4EngineLifetimeExpectedOwnedBytes = 599'261'184U;
inline constexpr std::uint64_t
    kSm87MacroFeedV4EngineLifetimeMinimumOwnedArenaBytes = 156'893'184U;

// Capability-free construction snapshot.  It contains no ModelWeights,
// device pointer, CUDA asset, stream/event handle, launch receipt, or route
// selector.  The Engine publishes it only after the normal startup and
// execution packages have been assigned to their final lifetime root.
struct Sm87MacroFeedV4EngineLifetimeConstructionSnapshot final {
  std::uint64_t startup_package_identity = 0U;
  std::uint64_t execution_package_identity = 0U;
  std::uint64_t execution_startup_package_identity = 0U;
  std::uint64_t startup_owner_identity = 0U;
  std::uint64_t startup_allocation_identity = 0U;
  std::uint64_t startup_device_identity = 0U;
  std::int32_t startup_device_ordinal = -1;
  std::int32_t execution_device_ordinal = -1;
  std::uint64_t startup_gdn_qkvz_catalog_identity = 0U;
  std::size_t startup_gdn_qkvz_binding_count = 0U;
  std::uint64_t execution_gdn_qkvz_catalog_identity = 0U;
  std::size_t execution_gdn_qkvz_binding_count = 0U;
  std::uint64_t owned_bytes = 0U;
  bool synthetic_t1_gdn_layer0_source = true;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return startup_package_identity != 0U &&
           execution_package_identity != 0U &&
           execution_startup_package_identity == startup_package_identity &&
           startup_owner_identity != 0U &&
           startup_allocation_identity != 0U &&
           startup_device_identity != 0U && startup_device_ordinal >= 0 &&
           execution_device_ordinal == startup_device_ordinal &&
           startup_gdn_qkvz_catalog_identity != 0U &&
           startup_gdn_qkvz_binding_count ==
               kSm87MacroFeedV4EngineLifetimeExpectedGdnQkvZBindings &&
           execution_gdn_qkvz_catalog_identity ==
               startup_gdn_qkvz_catalog_identity &&
           execution_gdn_qkvz_binding_count ==
               kSm87MacroFeedV4EngineLifetimeExpectedGdnQkvZBindings &&
           owned_bytes == kSm87MacroFeedV4EngineLifetimeExpectedOwnedBytes &&
           !synthetic_t1_gdn_layer0_source;
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
