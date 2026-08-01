#include "q3x/kernels/sm87_a4w4_attention_o_k512_cell.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

class Test final {
 public:
  void expect(const bool condition, const char* const message) {
    if (!condition) {
      std::cerr << message << '\n';
      ++failures_;
    }
  }

  [[nodiscard]] int result() const noexcept { return failures_ == 0 ? 0 : 1; }

 private:
  int failures_{};
};

}  // namespace

int main() {
  using namespace q3x::kernels;
  Test test;

  test.expect(kSm87A4W4AttentionOK512StageABytes == 16'384U &&
                  kSm87A4W4AttentionOK512StageBBytes == 8'192U &&
                  kSm87A4W4AttentionOK512StageBytes == 24'576U &&
                  kSm87A4W4AttentionOK512SharedBytes == 49'152U,
              "K256 copy-stage/shared-memory contract failed");
  test.expect(kSm87A4W4AttentionOK512SharedBytes * 2U ==
                  96U * 1'024U,
              "two-CTA shared-memory boundary failed");

  constexpr Sm87A4W4AttentionOK512Plan real =
      sm87_a4w4_attention_o_k512_plan(2'048U, 5'120U, 6'144U);
  test.expect(real.m_tiles == 16U && real.n_tiles == 80U &&
                  real.k512_groups == 12U &&
                  real.physical_k256_groups == 24U &&
                  real.physical_k64_groups == 96U &&
                  real.work_tiles == 1'280U && real.launch_ctas == 32U,
              "real Attention-O M2048/N5120/K6144 plan failed");
  test.expect(sm87_a4w4_attention_o_k512_plan(128U, 64U, 512U)
                      .launch_ctas == 1U,
              "minimal complete-cell plan failed");
  test.expect(sm87_a4w4_attention_o_k512_plan(127U, 64U, 512U)
                      .launch_ctas == 0U &&
                  sm87_a4w4_attention_o_k512_plan(128U, 63U, 512U)
                          .launch_ctas == 0U &&
                  sm87_a4w4_attention_o_k512_plan(128U, 64U, 384U)
                          .launch_ctas == 0U,
              "M/N/K tail rejection failed");

  constexpr std::size_t outer = 129U;
  constexpr std::size_t logical_k = 1'024U;
  constexpr std::size_t k64_groups = logical_k / 64U;
  constexpr std::size_t k512_groups = logical_k / 512U;
  const std::size_t packed_capacity =
      sm87_a4w4_attention_o_k512_packed_capacity_bytes(outer, logical_k);
  const std::size_t scale_capacity =
      sm87_a4w4_attention_o_k512_scale_capacity_elements(outer, logical_k);
  test.expect(packed_capacity == 98'304U,
              "packed consumer-block capacity failed");
  test.expect(scale_capacity == 384U,
              "K512 scale capacity failed");

  std::vector<std::uint8_t> packed_seen(packed_capacity, 0U);
  for (std::size_t row = 0U; row < outer; ++row) {
    for (std::size_t group = 0U; group < k64_groups; ++group) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        const std::size_t offset =
            sm87_a4w4_attention_o_k512_packed_offset(
                row, group, byte, k64_groups);
        test.expect(offset < packed_capacity,
                    "packed offset exceeded capacity");
        if (offset < packed_capacity) {
          test.expect(packed_seen[offset] == 0U,
                      "packed offset alias detected");
          packed_seen[offset] = 1U;
        }
      }
    }
  }

  std::vector<std::uint8_t> scale_seen(scale_capacity, 0U);
  for (std::size_t row = 0U; row < outer; ++row) {
    for (std::size_t group = 0U; group < k512_groups; ++group) {
      const std::size_t offset =
          sm87_a4w4_attention_o_k512_scale_offset(
              row, group, k512_groups);
      test.expect(offset < scale_capacity,
                  "scale offset exceeded capacity");
      if (offset < scale_capacity) {
        test.expect(scale_seen[offset] == 0U,
                    "scale offset alias detected");
        scale_seen[offset] = 1U;
      }
    }
  }

  // A logical group owns exactly two physical copy phases and eight K64
  // partials.  This integer reduction boundary is the defining new oracle.
  constexpr std::int32_t k64_partial[8U] = {
      4'096, -4'096, 3'000, -2'000, 1'000, -500, 250, -125};
  std::int32_t k512_partial = 0;
  for (const std::int32_t value : k64_partial) {
    k512_partial += value;
  }
  test.expect(k512_partial == 1'625,
              "one-S32-partial-per-K512 reduction failed");
  test.expect(8 * 4'096 == 32'768,
              "K512 S32 bound changed unexpectedly");

  if (test.result() == 0) {
    std::cout << "SM87 Attention-O K512 standalone contract passed\n";
  }
  return test.result();
}
