#include "q3x/kernels/sm87_a4w4_gateup_k512_m128n128_projection_serial.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <type_traits>

int main() {
  namespace kernels = q3x::kernels;
  constexpr auto p1853 =
      kernels::sm87_a4w4_gateup_k512_m128n128_projection_serial_plan(
          1'853U, 1'920U, 17'408U, 5'120U, 0U, 12'288U);
  constexpr auto secondary =
      kernels::sm87_a4w4_gateup_k512_m128n128_projection_serial_plan(
          2'048U, 2'048U, 17'408U, 5'120U, 12'288U, 5'120U);
  constexpr auto bad_launch =
      kernels::sm87_a4w4_gateup_k512_m128n128_projection_serial_plan(
          1'853U, 1'856U, 17'408U, 5'120U, 0U, 12'288U);
  constexpr auto noncanonical_padding =
      kernels::sm87_a4w4_gateup_k512_m128n128_projection_serial_plan(
          1U, 256U, 17'408U, 5'120U, 0U, 12'288U);

  using Query = int (*)(
      kernels::Sm87A4W4GateUpK512M128N128ProjectionSerialResources*)
      noexcept;
  using Launch = int (*)(
      const std::uint8_t*, std::size_t, const std::uint16_t*,
      std::size_t, const std::uint8_t*, std::size_t,
      const std::uint16_t*, std::size_t, const std::uint8_t*,
      std::size_t, const std::uint16_t*, std::size_t, std::size_t,
      std::size_t, std::size_t, std::size_t, std::size_t, std::size_t,
      std::uint16_t*, std::size_t, std::size_t, void*) noexcept;

  Query query = &kernels::
      query_sm87_a4w4_gateup_k512_m128n128_projection_serial_resources_cuda;
  Launch launch = &kernels::
      launch_sm87_a4w4_gateup_k512_m128n128_projection_serial_bf16_cuda;
  const bool valid =
      query != nullptr && launch != nullptr &&
      kernels::kSm87A4W4GateUpK512M128N128ProjectionSerialThreads ==
          512U &&
      kernels::kSm87A4W4GateUpK512M128N128ProjectionSerialWarps == 16U &&
      kernels::kSm87A4W4GateUpK512M128N128ProjectionSerialStages == 3U &&
      kernels::
              kSm87A4W4GateUpK512M128N128ProjectionSerialDynamicSharedBytes ==
          165'376U &&
      p1853.m_tiles == 15U && p1853.n_tiles == 96U &&
      p1853.k512_groups == 10U && p1853.launch_ctas == 16U &&
      kernels::
          sm87_a4w4_gateup_k512_m128n128_projection_serial_is_model_plan(
              p1853) &&
      secondary.m_tiles == 16U && secondary.n_tiles == 40U &&
      secondary.launch_ctas == 16U && bad_launch.launch_ctas == 0U &&
      noncanonical_padding.launch_ctas == 16U &&
      !kernels::
          sm87_a4w4_gateup_k512_m128n128_projection_serial_is_model_plan(
              noncanonical_padding) &&
      kernels::
              sm87_a4w4_gateup_k512_m128n128_projection_serial_scale_capacity(
                  1'920U, 5'120U) ==
          19'200U &&
      kernels::
              query_sm87_a4w4_gateup_k512_m128n128_projection_serial_resources_cuda(
                  nullptr) == static_cast<int>(cudaErrorInvalidValue);
  if (!valid) {
    std::cerr << "M128N128 projection-serial Gate+Up contract failed\n";
    return 1;
  }
  std::cout << "M128N128 projection-serial Gate+Up contract passed\n";
  return 0;
}
