#include "q3x/kernels/sm87_a4w4_attention_k256_m128n256.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <iostream>

namespace {

struct Test final {
  int failures{};

  void expect(const bool condition, const char* const description) {
    if (!condition) {
      ++failures;
      std::cerr << "FAIL: " << description << '\n';
    }
  }
};

template <std::size_t Projections, std::size_t MaximumPanels>
void audit_topology(
    Test& test,
    const q3x::kernels::Sm87A4W4AttentionK256Topology topology,
    const std::array<std::size_t, Projections>& panel_counts,
    const std::size_t expected_cells) {
  using namespace q3x::kernels;
  std::array<std::array<unsigned int, MaximumPanels>, Projections> visits{};
  test.expect(sm87_a4w4_attention_k256_fixed_cell_count(topology) ==
                  expected_cells,
              "fixed macro-cell count changed");
  for (std::size_t cell = 0U; cell < expected_cells; ++cell) {
    for (std::size_t slot = 0U;
         slot < kSm87A4W4AttentionK256PanelsPerCell; ++slot) {
      const auto descriptor =
          sm87_a4w4_attention_k256_fixed_panel(topology, cell, slot);
      test.expect(descriptor.projection < Projections,
                  "topology projection escaped projection set");
      if (descriptor.projection < Projections) {
        test.expect(descriptor.panel < panel_counts[descriptor.projection],
                    "topology panel escaped projection bounds");
        if (descriptor.panel < panel_counts[descriptor.projection]) {
          ++visits[descriptor.projection][descriptor.panel];
        }
      }
    }
  }
  for (std::size_t projection = 0U; projection < Projections;
       ++projection) {
    for (std::size_t panel = 0U; panel < panel_counts[projection]; ++panel) {
      test.expect(visits[projection][panel] == 1U,
                  "fixed topology must visit every panel exactly once");
    }
  }
}

}  // namespace

int main() {
  using namespace q3x::kernels;
  Test test;

  test.expect(kSm87A4W4AttentionK256TileM == 128U &&
                  kSm87A4W4AttentionK256TileN == 256U &&
                  kSm87A4W4AttentionK256Threads == 512U &&
                  kSm87A4W4AttentionK256Warps == 16U &&
                  kSm87A4W4AttentionK256PersistentCtas == 16U &&
                  kSm87A4W4AttentionK256CtasPerSm == 1U,
              "M128N256/512T/16-warp/16-CTA ownership changed");
  test.expect(kSm87A4W4AttentionK256StageACodeBytes == 16'384U &&
                  kSm87A4W4AttentionK256StageBCodeBytes == 32'768U &&
                  kSm87A4W4AttentionK256StageAScaleBytes == 256U &&
                  kSm87A4W4AttentionK256StageBScaleBytes == 512U &&
                  kSm87A4W4AttentionK256StageBytes == 49'920U &&
                  kSm87A4W4AttentionK256Stages == 3U &&
                  kSm87A4W4AttentionK256SharedBytes == 149'760U,
              "three combined cp.async stages changed");
  test.expect(4U * kSm87A4W4AttentionK256StageBytes == 199'680U,
              "four-stage forbidden-size witness changed");

  audit_topology<2U, 192U>(
      test, Sm87A4W4AttentionK256Topology::kLinearQkvZ,
      {kQwen36AttentionK256LinearQkvPanels,
       kQwen36AttentionK256LinearZPanels},
      64U);
  audit_topology<3U, 192U>(
      test, Sm87A4W4AttentionK256Topology::kFullQkv,
      {kQwen36AttentionK256FullQPanels,
       kQwen36AttentionK256FullKPanels,
       kQwen36AttentionK256FullVPanels},
      56U);
  audit_topology<1U, 192U>(
      test, Sm87A4W4AttentionK256Topology::kAttentionO,
      {kQwen36AttentionK256OPanels}, 20U);

  for (std::size_t cell = 0U; cell < 48U; ++cell) {
    const auto q0 = sm87_a4w4_attention_k256_fixed_panel(
        Sm87A4W4AttentionK256Topology::kLinearQkvZ, cell, 0U);
    const auto q1 = sm87_a4w4_attention_k256_fixed_panel(
        Sm87A4W4AttentionK256Topology::kLinearQkvZ, cell, 1U);
    const auto z0 = sm87_a4w4_attention_k256_fixed_panel(
        Sm87A4W4AttentionK256Topology::kLinearQkvZ, cell, 2U);
    const auto z1 = sm87_a4w4_attention_k256_fixed_panel(
        Sm87A4W4AttentionK256Topology::kLinearQkvZ, cell, 3U);
    test.expect(q0.projection == 0U && q1.projection == 0U &&
                    z0.projection == 1U && z1.projection == 1U,
                "Linear first 48 cells must be 2Q+2Z");
  }
  for (std::size_t cell = 48U; cell < 64U; ++cell) {
    for (std::size_t slot = 0U; slot < 4U; ++slot) {
      test.expect(sm87_a4w4_attention_k256_fixed_panel(
                      Sm87A4W4AttentionK256Topology::kLinearQkvZ,
                      cell, slot)
                          .projection == 0U,
                  "Linear final 16 cells must be 4Q");
    }
  }
  for (std::size_t cell = 0U; cell < 8U; ++cell) {
    test.expect(sm87_a4w4_attention_k256_fixed_panel(
                    Sm87A4W4AttentionK256Topology::kFullQkv, cell, 2U)
                        .projection == 1U,
                "Full first eight cells must be 2Q+2K");
  }
  for (std::size_t cell = 8U; cell < 16U; ++cell) {
    test.expect(sm87_a4w4_attention_k256_fixed_panel(
                    Sm87A4W4AttentionK256Topology::kFullQkv, cell, 2U)
                        .projection == 2U,
                "Full next eight cells must be 2Q+2V");
  }

  constexpr auto k768 =
      sm87_a4w4_attention_k256_test_plan(256U, 768U, 3U);
  test.expect(k768.m_tiles == 2U && k768.macro_cells == 3U &&
                  k768.k256_groups == 3U &&
                  k768.physical_k64_groups == 12U &&
                  k768.work_cells == 6U && k768.launch_ctas == 6U,
              "K768 arbitrary-panel plan changed");
  test.expect(sm87_a4w4_attention_k256_launch_token_count(1U) == 128U &&
                  sm87_a4w4_attention_k256_launch_token_count(128U) ==
                      128U &&
                  sm87_a4w4_attention_k256_launch_token_count(129U) ==
                      256U &&
                  sm87_a4w4_attention_k256_launch_token_count(0U) == 0U,
              "logical/padded M authority changed");
  test.expect(sm87_a4w4_attention_k256_packed_capacity_bytes(129U, 768U) ==
                  73'728U &&
                  sm87_a4w4_attention_k256_scale_capacity_elements(
                      129U, 768U) == 576U,
              "K768 consumer-block capacity/scale wrap changed");

  test.expect(query_sm87_a4w4_attention_k256_m128n256_resources_cuda(
                  nullptr) == static_cast<int>(cudaErrorInvalidValue),
              "resource query must reject null before CUDA state");
  if (test.failures == 0) {
    std::cout << "SM87 Attention K256 M128N256 contract passed\n";
  }
  return test.failures == 0 ? 0 : 1;
}
