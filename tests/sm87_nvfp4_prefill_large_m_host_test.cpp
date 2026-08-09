#include "q3x/kernels/sm87_nvfp4_prefill_large_m.h"

#include <cstddef>
#include <iostream>

namespace {

using q3x::kernels::Sm87NvFp4PrefillLargeMDataflow;
using q3x::kernels::Sm87NvFp4PrefillLargeMRole;
using q3x::kernels::sm87_nvfp4_prefill_large_m_plan;
using q3x::kernels::sm87_nvfp4_prefill_large_m_supports;

[[nodiscard]] bool check(const bool condition, const char* const message) {
  if (!condition) {
    std::cerr << message << '\n';
  }
  return condition;
}

}  // namespace

int main() {
  bool ok = true;
  for (const auto token_count : {std::size_t{8'192U},
                                 std::size_t{7'712U}}) {
    const auto gate = sm87_nvfp4_prefill_large_m_plan(
        Sm87NvFp4PrefillLargeMRole::kGateUp, token_count);
    ok &= check(gate.valid(), "Gate+Up plan must be valid");
    ok &= check(gate.input_features == 5'120U &&
                    gate.output_features == 34'816U,
                "Gate+Up shape mismatch");
    ok &= check(
        gate.dataflow ==
            Sm87NvFp4PrefillLargeMDataflow::kGateUpAStationaryRaster,
        "Gate+Up must use the A-stationary raster");
    ok &= check(gate.grid_n == 136U, "Gate+Up grid-N mismatch");

    const auto down = sm87_nvfp4_prefill_large_m_plan(
        Sm87NvFp4PrefillLargeMRole::kDown, token_count);
    ok &= check(down.valid(), "Down plan must be valid");
    ok &= check(down.input_features == 17'408U &&
                    down.output_features == 5'120U,
                "Down shape mismatch");
    ok &= check(
        down.dataflow ==
            Sm87NvFp4PrefillLargeMDataflow::kDownBStationaryRaster,
        "Down must use the B-stationary raster");
    ok &= check(down.grid_n == 20U, "Down grid-N mismatch");

    const std::size_t expected_grid_m = token_count == 8'192U ? 64U : 61U;
    const std::size_t expected_tail = token_count == 8'192U ? 0U : 32U;
    ok &= check(gate.grid_m == expected_grid_m &&
                    down.grid_m == expected_grid_m,
                "grid-M mismatch");
    ok &= check(gate.tail_rows == expected_tail &&
                    down.tail_rows == expected_tail,
                "tail-row mismatch");
    ok &= check(gate.dynamic_shared_bytes == 82'944U &&
                    down.dynamic_shared_bytes == 82'944U,
                "three-stage shared-memory budget mismatch");
  }

  for (const auto token_count : {std::size_t{0U}, std::size_t{512U},
                                 std::size_t{513U}, std::size_t{7'711U},
                                 std::size_t{8'191U}, std::size_t{8'193U}}) {
    ok &= check(!sm87_nvfp4_prefill_large_m_supports(
                    Sm87NvFp4PrefillLargeMRole::kGateUp, token_count),
                "unsupported Gate+Up M must fail closed");
    ok &= check(!sm87_nvfp4_prefill_large_m_supports(
                    Sm87NvFp4PrefillLargeMRole::kDown, token_count),
                "unsupported Down M must fail closed");
  }
  ok &= check(!sm87_nvfp4_prefill_large_m_plan(
                   static_cast<Sm87NvFp4PrefillLargeMRole>(255U), 8'192U)
                   .valid(),
              "invalid role must fail closed");
  return ok ? 0 : 1;
}
