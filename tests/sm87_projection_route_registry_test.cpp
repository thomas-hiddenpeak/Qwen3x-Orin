#include "kernels/sm87/projection_route_registry.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>

namespace {

using q3x::kernels::sm87_detail::ProjectionPlan;
using q3x::kernels::sm87_detail::ProjectionQuery;
using q3x::kernels::sm87_detail::ProjectionRoute;
using q3x::kernels::sm87_detail::ProjectionShape;
using q3x::kernels::sm87_detail::WeightEncoding;
using q3x::kernels::sm87_detail::select_projection_plan;

[[nodiscard]] constexpr ProjectionQuery query(
    const WeightEncoding encoding, const std::size_t token_count,
    const std::size_t rows, const std::size_t columns,
    const bool weight_aligned_4 = true,
    const bool weight_aligned_16 = true,
    const bool activation_aligned_8 = true,
    const bool block_scales_aligned_2 = true) noexcept {
  return ProjectionQuery{encoding,
                         token_count,
                         rows,
                         columns,
                         weight_aligned_4,
                         weight_aligned_16,
                         activation_aligned_8,
                         block_scales_aligned_2};
}

[[nodiscard]] constexpr bool matches(
    const ProjectionPlan plan, const ProjectionShape shape,
    const ProjectionRoute route, const unsigned int maximum_blocks = 0U,
    const std::size_t launch_count = 1U) noexcept {
  return plan.shape == shape && plan.route == route &&
         plan.maximum_blocks == maximum_blocks &&
         plan.launch_count == launch_count;
}

static_assert(matches(
    select_projection_plan(query(WeightEncoding::kFp8, 1U, 10'240U, 5'120U)),
    ProjectionShape::kFp8_10240x5120, ProjectionRoute::kFp8M1RowQuad,
    1'536U));
static_assert(matches(
    select_projection_plan(query(WeightEncoding::kFp8, 2U, 1'024U, 5'120U)),
    ProjectionShape::kFp8_1024x5120, ProjectionRoute::kFp8M2RowPair,
    2'048U));
static_assert(matches(
    select_projection_plan(query(WeightEncoding::kFp8, 16U, 1'024U, 5'120U)),
    ProjectionShape::kFp8_1024x5120, ProjectionRoute::kSplitM16IntoM8, 0U,
    2U));
static_assert(matches(
    select_projection_plan(query(WeightEncoding::kFp8, 32U, 10'240U,
                                 5'120U)),
    ProjectionShape::kFp8_10240x5120, ProjectionRoute::kFp8M32Wmma));
static_assert(matches(
    select_projection_plan(query(WeightEncoding::kNvFp4, 32U, 17'408U,
                                 5'120U)),
    ProjectionShape::kNvFp4_17408x5120,
    ProjectionRoute::kNvFp4M32Wmma));
static_assert(matches(
    select_projection_plan(
        query(WeightEncoding::kNvFp4, 1U, 248'320U, 5'120U)),
    ProjectionShape::kNvFp4_248320x5120,
    ProjectionRoute::kNvFp4M1LmHeadActivationStaged));
static_assert(matches(
    select_projection_plan(
        query(WeightEncoding::kNvFp4, 16U, 17'408U, 5'120U)),
    ProjectionShape::kNvFp4_17408x5120, ProjectionRoute::kNvFp4M16Wmma));
static_assert(matches(
    select_projection_plan(query(WeightEncoding::kFp8, 8U, 0U, 5'120U)),
    ProjectionShape::kUnknown, ProjectionRoute::kNoOp, 0U, 0U));
static_assert(matches(
    select_projection_plan(query(WeightEncoding::kNvFp4, 8U, 0U, 1U)),
    ProjectionShape::kUnknown, ProjectionRoute::kInvalid, 0U, 0U));

struct ExpectedRoute {
  std::size_t token_count;
  ProjectionRoute route;
  unsigned int maximum_blocks;
  std::size_t launch_count;
};

struct ExactShapeRoutes {
  WeightEncoding encoding;
  std::size_t rows;
  std::size_t columns;
  ProjectionShape shape;
  std::array<ExpectedRoute, 5U> routes;
};

constexpr std::array<ExactShapeRoutes, 8U> kExactShapes{{
    {WeightEncoding::kFp8,
     10'240U,
     5'120U,
     ProjectionShape::kFp8_10240x5120,
     {{{1U, ProjectionRoute::kFp8M1RowQuad, 1'536U, 1U},
       {2U, ProjectionRoute::kFp8M2RowQuad, 1'536U, 1U},
       {8U, ProjectionRoute::kFp8M8Fixed, 0U, 1U},
       {16U, ProjectionRoute::kFp8M16Wmma, 0U, 1U},
       {32U, ProjectionRoute::kFp8M32Wmma, 0U, 1U}}}},
    {WeightEncoding::kFp8,
     5'120U,
     6'144U,
     ProjectionShape::kFp8_5120x6144,
     {{{1U, ProjectionRoute::kFp8M1RowQuad, 1'280U, 1U},
       {2U, ProjectionRoute::kFp8M2RowQuad, 768U, 1U},
       {8U, ProjectionRoute::kFp8M8Fixed, 0U, 1U},
       {16U, ProjectionRoute::kFp8M16Wmma, 0U, 1U},
       {32U, ProjectionRoute::kFp8M32Wmma, 0U, 1U}}}},
    {WeightEncoding::kFp8,
     6'144U,
     5'120U,
     ProjectionShape::kFp8_6144x5120,
     {{{1U, ProjectionRoute::kFp8M1RowQuad, 768U, 1U},
       {2U, ProjectionRoute::kFp8M2RowQuad, 1'024U, 1U},
       {8U, ProjectionRoute::kFp8M8Fixed, 0U, 1U},
       {16U, ProjectionRoute::kFp8M16Wmma, 0U, 1U},
       {32U, ProjectionRoute::kFp8M32Wmma, 0U, 1U}}}},
    {WeightEncoding::kFp8,
     12'288U,
     5'120U,
     ProjectionShape::kFp8_12288x5120,
     {{{1U, ProjectionRoute::kFp8M1RowQuad, 2'048U, 1U},
       {2U, ProjectionRoute::kFp8M2RowQuad, 2'048U, 1U},
       {8U, ProjectionRoute::kFp8M8Fixed, 0U, 1U},
       {16U, ProjectionRoute::kFp8M16Wmma, 0U, 1U},
       {32U, ProjectionRoute::kFp8M32Wmma, 0U, 1U}}}},
    {WeightEncoding::kFp8,
     1'024U,
     5'120U,
     ProjectionShape::kFp8_1024x5120,
     {{{1U, ProjectionRoute::kFp8M1RowPair, 2'048U, 1U},
       {2U, ProjectionRoute::kFp8M2RowPair, 2'048U, 1U},
       {8U, ProjectionRoute::kFp8M8Fixed, 0U, 1U},
       {16U, ProjectionRoute::kSplitM16IntoM8, 0U, 2U},
       {32U, ProjectionRoute::kSplitM32IntoM16, 0U, 2U}}}},
    {WeightEncoding::kNvFp4,
     17'408U,
     5'120U,
     ProjectionShape::kNvFp4_17408x5120,
     {{{1U, ProjectionRoute::kNvFp4M1GateUpActivationStaged, 0U, 1U},
       {2U, ProjectionRoute::kNvFp4M2RowQuad, 64U, 1U},
       {8U, ProjectionRoute::kNvFp4M8Fixed, 0U, 1U},
       {16U, ProjectionRoute::kNvFp4M16Wmma, 0U, 1U},
       {32U, ProjectionRoute::kNvFp4M32Wmma, 0U, 1U}}}},
    {WeightEncoding::kNvFp4,
     5'120U,
     17'408U,
     ProjectionShape::kNvFp4_5120x17408,
     {{{1U, ProjectionRoute::kNvFp4M1DownActivationStaged, 0U, 1U},
       {2U, ProjectionRoute::kNvFp4M2RowQuad, 64U, 1U},
       {8U, ProjectionRoute::kNvFp4M8Fixed, 0U, 1U},
       {16U, ProjectionRoute::kNvFp4M16Wmma, 0U, 1U},
       {32U, ProjectionRoute::kNvFp4M32Wmma, 0U, 1U}}}},
    {WeightEncoding::kNvFp4,
     248'320U,
     5'120U,
     ProjectionShape::kNvFp4_248320x5120,
     {{{1U, ProjectionRoute::kNvFp4M1LmHeadActivationStaged, 0U, 1U},
       {2U, ProjectionRoute::kNvFp4M2ScaleCodebook, 0U, 1U},
       {8U, ProjectionRoute::kNvFp4M8ScaleCodebook, 0U, 1U},
       {16U, ProjectionRoute::kSplitM16IntoM8, 0U, 2U},
       {32U, ProjectionRoute::kSplitM32IntoM16, 0U, 2U}}}},
}};

int failures = 0;

void expect(const ProjectionQuery& route_query, const ProjectionShape shape,
            const ProjectionRoute route,
            const unsigned int maximum_blocks = 0U,
            const std::size_t launch_count = 1U) {
  const ProjectionPlan actual = select_projection_plan(route_query);
  if (matches(actual, shape, route, maximum_blocks, launch_count)) {
    return;
  }
  std::cerr << "route mismatch: encoding="
            << static_cast<unsigned int>(route_query.encoding)
            << " M=" << route_query.token_count << " N=" << route_query.rows
            << " K=" << route_query.columns << " expected(route="
            << static_cast<unsigned int>(route) << ", shape="
            << static_cast<unsigned int>(shape) << ", cap=" << maximum_blocks
            << ", launches=" << launch_count << ") actual(route="
            << static_cast<unsigned int>(actual.route) << ", shape="
            << static_cast<unsigned int>(actual.shape) << ", cap="
            << actual.maximum_blocks << ", launches=" << actual.launch_count
            << ")\n";
  ++failures;
}

void test_exact_shape_matrix() {
  for (const ExactShapeRoutes& exact : kExactShapes) {
    for (const ExpectedRoute& expected : exact.routes) {
      expect(query(exact.encoding, expected.token_count, exact.rows,
                   exact.columns),
             exact.shape, expected.route, expected.maximum_blocks,
             expected.launch_count);
    }
  }
}

void test_alignment_matrix() {
  for (const ExactShapeRoutes& exact : kExactShapes) {
    for (unsigned int bits = 0U; bits < 16U; ++bits) {
      const bool weight_aligned_4 = (bits & 1U) != 0U;
      const bool weight_aligned_16 = (bits & 2U) != 0U;
      const bool activation_aligned_8 = (bits & 4U) != 0U;
      const bool block_scales_aligned_2 = (bits & 8U) != 0U;
      for (std::size_t token_count = 1U; token_count <= 8U; ++token_count) {
        const ProjectionQuery route_query = query(
            exact.encoding, token_count, exact.rows, exact.columns,
            weight_aligned_4, weight_aligned_16, activation_aligned_8,
            block_scales_aligned_2);
        if (weight_aligned_4 && activation_aligned_8) {
          if (token_count == 1U) {
            expect(route_query, exact.shape, exact.routes[0U].route,
                   exact.routes[0U].maximum_blocks);
          } else if (token_count == 2U) {
            expect(route_query, exact.shape, exact.routes[1U].route,
                   exact.routes[1U].maximum_blocks);
          } else if (token_count == 8U) {
            expect(route_query, exact.shape, exact.routes[2U].route,
                   exact.routes[2U].maximum_blocks);
          } else {
            expect(route_query, exact.shape,
                   exact.encoding == WeightEncoding::kFp8
                       ? ProjectionRoute::kFp8SmallMVector
                       : ProjectionRoute::kNvFp4SmallMVector);
          }
        } else if (token_count == 1U) {
          expect(route_query, exact.shape,
                 exact.encoding == WeightEncoding::kFp8
                     ? ProjectionRoute::kFp8M1Scalar
                     : ProjectionRoute::kNvFp4M1Scalar);
        } else {
          expect(route_query, exact.shape, ProjectionRoute::kSerialM1, 0U,
                 token_count);
        }
      }

      const ProjectionQuery m16_query = query(
          exact.encoding, 16U, exact.rows, exact.columns, weight_aligned_4,
          weight_aligned_16, activation_aligned_8,
          block_scales_aligned_2);
      const bool tensor_eligible =
          weight_aligned_16 && activation_aligned_8 &&
          (exact.encoding == WeightEncoding::kFp8 ||
           block_scales_aligned_2);
      const bool tensor_shape =
          exact.routes[3U].route == ProjectionRoute::kFp8M16Wmma ||
          exact.routes[3U].route == ProjectionRoute::kNvFp4M16Wmma;
      if (tensor_eligible && tensor_shape) {
        expect(m16_query, exact.shape, exact.routes[3U].route);
      } else {
        expect(m16_query, exact.shape, ProjectionRoute::kSplitM16IntoM8, 0U,
               2U);
      }

      const ProjectionQuery m32_query = query(
          exact.encoding, 32U, exact.rows, exact.columns, weight_aligned_4,
          weight_aligned_16, activation_aligned_8,
          block_scales_aligned_2);
      const bool direct_m32_shape =
          exact.routes[4U].route == ProjectionRoute::kFp8M32Wmma ||
          exact.routes[4U].route == ProjectionRoute::kNvFp4M32Wmma;
      if (tensor_eligible && direct_m32_shape) {
        expect(m32_query, exact.shape, exact.routes[4U].route);
      } else {
        expect(m32_query, exact.shape, ProjectionRoute::kSplitM32IntoM16, 0U,
               2U);
      }
    }
  }
}

void test_generic_thresholds_and_fallbacks() {
  expect(query(WeightEncoding::kFp8, 1U, 64U, 1'023U),
         ProjectionShape::kUnknown, ProjectionRoute::kFp8M1Scalar);
  expect(query(WeightEncoding::kFp8, 1U, 64U, 1'024U),
         ProjectionShape::kUnknown, ProjectionRoute::kFp8M1Vector);
  expect(query(WeightEncoding::kFp8, 1U, 64U, 1'025U),
         ProjectionShape::kUnknown, ProjectionRoute::kFp8M1Scalar);
  expect(query(WeightEncoding::kFp8, 1U, 1'023U, 1'024U),
         ProjectionShape::kUnknown, ProjectionRoute::kFp8M1Vector);
  expect(query(WeightEncoding::kFp8, 1U, 1'024U, 1'024U),
         ProjectionShape::kUnknown, ProjectionRoute::kFp8M1VectorGridCap,
         2'048U);
  expect(query(WeightEncoding::kFp8, 2U, 1'024U, 1'024U),
         ProjectionShape::kUnknown, ProjectionRoute::kFp8M2VectorGridCap,
         2'048U);
  expect(query(WeightEncoding::kFp8, 8U, 1'023U, 1'024U),
         ProjectionShape::kUnknown, ProjectionRoute::kFp8SmallMVector);
  expect(query(WeightEncoding::kFp8, 8U, 1'024U, 1'024U),
         ProjectionShape::kUnknown, ProjectionRoute::kFp8M8RowPair);

  expect(query(WeightEncoding::kNvFp4, 1U, 7U, 5'120U),
         ProjectionShape::kUnknown, ProjectionRoute::kNvFp4M1Vector);
  expect(query(WeightEncoding::kNvFp4, 1U, 8U, 5'120U),
         ProjectionShape::kUnknown, ProjectionRoute::kNvFp4M1ScaleCodebook,
         96U);
  expect(query(WeightEncoding::kNvFp4, 2U, 8U, 5'120U),
         ProjectionShape::kUnknown, ProjectionRoute::kNvFp4M2ScaleCodebook);
  expect(query(WeightEncoding::kNvFp4, 1U, 8U, 4'864U),
         ProjectionShape::kUnknown, ProjectionRoute::kNvFp4M1Vector);
  expect(query(WeightEncoding::kNvFp4, 1U, 8U, 240U),
         ProjectionShape::kUnknown, ProjectionRoute::kNvFp4M1Scalar);
  expect(query(WeightEncoding::kNvFp4, 1U, 8U, 256U),
         ProjectionShape::kUnknown, ProjectionRoute::kNvFp4M1Vector);
  expect(query(WeightEncoding::kNvFp4, 1U, 8U, 272U),
         ProjectionShape::kUnknown, ProjectionRoute::kNvFp4M1Scalar);
  expect(query(WeightEncoding::kNvFp4, 8U, 15U, 256U),
         ProjectionShape::kUnknown, ProjectionRoute::kNvFp4SmallMVector);
  expect(query(WeightEncoding::kNvFp4, 8U, 16U, 256U),
         ProjectionShape::kUnknown,
         ProjectionRoute::kNvFp4M8ScaleCodebook);

  for (std::size_t token_count = 2U; token_count <= 8U; ++token_count) {
    expect(query(WeightEncoding::kFp8, token_count, 64U, 1'024U, false,
                 true, true, true),
           ProjectionShape::kUnknown, ProjectionRoute::kSerialM1, 0U,
           token_count);
    expect(query(WeightEncoding::kNvFp4, token_count, 64U, 256U, true,
                 true, false, true),
           ProjectionShape::kUnknown, ProjectionRoute::kSerialM1, 0U,
           token_count);
  }
}

void test_near_misses_and_invalid_queries() {
  for (const ExactShapeRoutes& exact : kExactShapes) {
    const ProjectionRoute generic_m1 =
        exact.encoding == WeightEncoding::kFp8
            ? ProjectionRoute::kFp8M1VectorGridCap
            : ProjectionRoute::kNvFp4M1ScaleCodebook;
    const unsigned int generic_m1_cap =
        exact.encoding == WeightEncoding::kFp8 ? 2'048U : 96U;
    const ProjectionRoute generic_m2 =
        exact.encoding == WeightEncoding::kFp8
            ? ProjectionRoute::kFp8M2VectorGridCap
            : ProjectionRoute::kNvFp4M2ScaleCodebook;
    const unsigned int generic_m2_cap =
        exact.encoding == WeightEncoding::kFp8 ? 2'048U : 0U;
    const ProjectionRoute generic_m8 =
        exact.encoding == WeightEncoding::kFp8
            ? ProjectionRoute::kFp8M8RowPair
            : ProjectionRoute::kNvFp4M8ScaleCodebook;
    for (const std::size_t near_rows : {exact.rows - 1U, exact.rows + 1U}) {
      const bool fp8_below_row_threshold =
          exact.encoding == WeightEncoding::kFp8 && near_rows < 1'024U;
      expect(query(exact.encoding, 1U, near_rows, exact.columns),
             ProjectionShape::kUnknown,
             fp8_below_row_threshold ? ProjectionRoute::kFp8M1Vector
                                      : generic_m1,
             fp8_below_row_threshold ? 0U : generic_m1_cap);
      expect(query(exact.encoding, 2U, near_rows, exact.columns),
             ProjectionShape::kUnknown,
             fp8_below_row_threshold ? ProjectionRoute::kFp8SmallMVector
                                      : generic_m2,
             fp8_below_row_threshold ? 0U : generic_m2_cap);
      expect(query(exact.encoding, 8U, near_rows, exact.columns),
             ProjectionShape::kUnknown,
             fp8_below_row_threshold ? ProjectionRoute::kFp8SmallMVector
                                      : generic_m8);
    }

    expect(query(exact.encoding, 16U, exact.rows - 1U, exact.columns),
           ProjectionShape::kUnknown, ProjectionRoute::kSplitM16IntoM8, 0U,
           2U);
    expect(query(exact.encoding, 16U, exact.rows + 1U, exact.columns),
           ProjectionShape::kUnknown, ProjectionRoute::kSplitM16IntoM8, 0U,
           2U);
    expect(query(exact.encoding, 32U, exact.rows - 1U, exact.columns),
           ProjectionShape::kUnknown, ProjectionRoute::kSplitM32IntoM16, 0U,
           2U);
    expect(query(exact.encoding, 32U, exact.rows + 1U, exact.columns),
           ProjectionShape::kUnknown, ProjectionRoute::kSplitM32IntoM16, 0U,
           2U);
    const std::size_t column_delta =
        exact.encoding == WeightEncoding::kFp8 ? 1U : 16U;
    expect(query(exact.encoding, 16U, exact.rows,
                 exact.columns - column_delta),
           ProjectionShape::kUnknown, ProjectionRoute::kSplitM16IntoM8, 0U,
           2U);
    expect(query(exact.encoding, 16U, exact.rows,
                 exact.columns + column_delta),
           ProjectionShape::kUnknown, ProjectionRoute::kSplitM16IntoM8, 0U,
           2U);
    expect(query(exact.encoding, 32U, exact.rows,
                 exact.columns - column_delta),
           ProjectionShape::kUnknown, ProjectionRoute::kSplitM32IntoM16, 0U,
           2U);
    expect(query(exact.encoding, 32U, exact.rows,
                 exact.columns + column_delta),
           ProjectionShape::kUnknown, ProjectionRoute::kSplitM32IntoM16, 0U,
           2U);
    expect(query(exact.encoding, 1U, exact.rows,
                 exact.columns + column_delta),
           ProjectionShape::kUnknown,
           exact.encoding == WeightEncoding::kFp8
               ? ProjectionRoute::kFp8M1Scalar
               : ProjectionRoute::kNvFp4M1Scalar);
    expect(query(exact.encoding, 2U, exact.rows,
                 exact.columns + column_delta),
           ProjectionShape::kUnknown, ProjectionRoute::kSerialM1, 0U, 2U);
    expect(query(exact.encoding, 8U, exact.rows,
                 exact.columns + column_delta),
           ProjectionShape::kUnknown, ProjectionRoute::kSerialM1, 0U, 8U);
  }

  for (const std::size_t token_count : {0U, 9U, 15U, 17U, 31U, 33U}) {
    expect(query(WeightEncoding::kFp8, token_count, 64U, 1'024U),
           ProjectionShape::kUnknown, ProjectionRoute::kInvalid, 0U, 0U);
  }
  expect(query(static_cast<WeightEncoding>(0xffU), 1U, 64U, 1'024U),
         ProjectionShape::kUnknown, ProjectionRoute::kInvalid, 0U, 0U);
  expect(query(WeightEncoding::kNvFp4, 1U, 64U, 255U),
         ProjectionShape::kUnknown, ProjectionRoute::kInvalid, 0U, 0U);
  expect(query(WeightEncoding::kFp8, 1U, 0U, 3U),
         ProjectionShape::kUnknown, ProjectionRoute::kNoOp, 0U, 0U);
  expect(query(WeightEncoding::kFp8, 8U, 3U, 0U),
         ProjectionShape::kUnknown, ProjectionRoute::kNoOp, 0U, 0U);
  expect(query(WeightEncoding::kNvFp4, 1U, 0U, 16U),
         ProjectionShape::kUnknown, ProjectionRoute::kNoOp, 0U, 0U);
  expect(query(WeightEncoding::kNvFp4, 16U, 3U, 0U),
         ProjectionShape::kUnknown, ProjectionRoute::kNoOp, 0U, 0U);
  expect(query(WeightEncoding::kFp8, 32U, 0U, 3U),
         ProjectionShape::kUnknown, ProjectionRoute::kNoOp, 0U, 0U);
  expect(query(WeightEncoding::kNvFp4, 32U, 3U, 0U),
         ProjectionShape::kUnknown, ProjectionRoute::kNoOp, 0U, 0U);
}

}  // namespace

int main() {
  test_exact_shape_matrix();
  test_alignment_matrix();
  test_generic_thresholds_and_fallbacks();
  test_near_misses_and_invalid_queries();
  if (failures != 0) {
    std::cerr << failures << " SM87 projection registry checks failed\n";
    return 1;
  }
  std::cout << "SM87 projection registry checks passed\n";
  return 0;
}
