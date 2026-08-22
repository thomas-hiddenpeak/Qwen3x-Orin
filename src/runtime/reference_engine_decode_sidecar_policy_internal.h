#pragma once

#include <cstdint>

namespace q3x::runtime::reference_engine_detail {

// Decode sidecar selection is a build/route fact. Testing binaries keep a
// same-ELF baseline/admission seam; ordinary release binaries bind the
// measured layouts as the production default without consulting process
// environment state. Non-ordinary release routes remain isolated.
enum class DecodeSidecarSelection : std::uint8_t {
  kDisabled = 0U,
  kTestingAdmission,
  kProductionDefault,
};

struct DecodeSidecarPolicy {
  DecodeSidecarSelection selection = DecodeSidecarSelection::kDisabled;

  [[nodiscard]] constexpr bool requested() const noexcept {
    return selection != DecodeSidecarSelection::kDisabled;
  }

  [[nodiscard]] constexpr bool production_requested() const noexcept {
    return selection == DecodeSidecarSelection::kProductionDefault;
  }
};

[[nodiscard]] constexpr DecodeSidecarPolicy select_decode_sidecar_policy(
    const bool build_testing, const bool testing_admission,
    const bool ordinary_production_route) noexcept {
  if (build_testing) {
    return {testing_admission
                ? DecodeSidecarSelection::kTestingAdmission
                : DecodeSidecarSelection::kDisabled};
  }
  return {ordinary_production_route
              ? DecodeSidecarSelection::kProductionDefault
              : DecodeSidecarSelection::kDisabled};
}

}  // namespace q3x::runtime::reference_engine_detail
