#include "q3x/runtime/reference_engine.h"
#include "reference_engine_decode_sidecar_policy_internal.h"

#include <cstdint>
#include <iostream>
#include <type_traits>

namespace {

using q3x::runtime::reference_engine_detail::DecodeSidecarSelection;
using q3x::runtime::reference_engine_detail::select_decode_sidecar_policy;

constexpr auto kTestingBaseline =
    select_decode_sidecar_policy(true, false, true);
constexpr auto kTestingAdmission =
    select_decode_sidecar_policy(true, true, false);
constexpr auto kOrdinaryProduction =
    select_decode_sidecar_policy(false, false, true);
constexpr auto kIsolatedReleaseRoute =
    select_decode_sidecar_policy(false, true, false);

static_assert(kTestingBaseline.selection == DecodeSidecarSelection::kDisabled);
static_assert(!kTestingBaseline.requested());
static_assert(!kTestingBaseline.production_requested());
static_assert(kTestingAdmission.selection ==
              DecodeSidecarSelection::kTestingAdmission);
static_assert(kTestingAdmission.requested());
static_assert(!kTestingAdmission.production_requested());
static_assert(kOrdinaryProduction.selection ==
              DecodeSidecarSelection::kProductionDefault);
static_assert(kOrdinaryProduction.requested());
static_assert(kOrdinaryProduction.production_requested());
static_assert(kIsolatedReleaseRoute.selection ==
              DecodeSidecarSelection::kDisabled);
static_assert(!kIsolatedReleaseRoute.requested());
static_assert(!kIsolatedReleaseRoute.production_requested());
static_assert(std::is_same_v<
              decltype(q3x::runtime::ReferenceEngineLoadStats{}
                           .nvfp4_down_consumer_order_production_requested),
              bool>);
static_assert(std::is_same_v<
              decltype(q3x::runtime::ReferenceEngineLoadStats{}
                           .nvfp4_gate_up_coupled_feed_production_bytes),
              std::uint64_t>);

}  // namespace

int main() {
  const q3x::runtime::ReferenceEngineLoadStats empty_receipt;
  if (empty_receipt.nvfp4_down_consumer_order_production_requested ||
      empty_receipt.nvfp4_down_consumer_order_production_enabled ||
      empty_receipt.nvfp4_down_consumer_order_production_bytes != 0U ||
      empty_receipt.nvfp4_gate_up_coupled_feed_production_requested ||
      empty_receipt.nvfp4_gate_up_coupled_feed_production_enabled ||
      empty_receipt.nvfp4_gate_up_coupled_feed_production_bytes != 0U) {
    std::cerr << "empty Decode production receipt is not zero-initialized\n";
    return 1;
  }
  std::cout << "reference engine Decode sidecar typed policy: PASS\n";
  return 0;
}
