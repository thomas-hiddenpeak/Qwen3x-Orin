#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define Q3X_SM87_ROUTE_INLINE __forceinline__
#elif defined(__GNUC__) || defined(__clang__)
#define Q3X_SM87_ROUTE_INLINE inline __attribute__((always_inline))
#else
#define Q3X_SM87_ROUTE_INLINE inline
#endif

// Private, host/device-independent SM87 projection policy.  Callers must run
// the public launcher's argument validation before consuming a plan: this
// registry intentionally describes route selection, not pointer/range safety.
namespace q3x::kernels::sm87_detail {

enum class WeightEncoding : std::uint8_t {
  kFp8,
  kNvFp4,
};

enum class ProjectionShape : std::uint8_t {
  kUnknown,
  kFp8_10240x5120,
  kFp8_5120x6144,
  kFp8_6144x5120,
  kFp8_12288x5120,
  kFp8_1024x5120,
  kNvFp4_17408x5120,
  kNvFp4_5120x17408,
  kNvFp4_248320x5120,
};

// Every value names a production control-flow leaf, except for the two
// explicit recursive fallbacks.  Those fallbacks deliberately re-enter the
// complete M1/M8 dispatcher so offset-pointer eligibility and launch-error
// ordering remain unchanged.
enum class ProjectionRoute : std::uint8_t {
  kInvalid,
  kNoOp,
  kFp8M1Scalar,
  kFp8M1Vector,
  kFp8M1VectorGridCap,
  kFp8M1RowPair,
  kFp8M1RowQuad,
  kNvFp4M1Scalar,
  kNvFp4M1Vector,
  kNvFp4M1ScaleCodebook,
  kNvFp4M1DownActivationStaged,
  kNvFp4M1GateUpActivationStaged,
  kNvFp4M1LmHeadActivationStaged,
  kSerialM1,
  kFp8SmallMVector,
  kFp8M2VectorGridCap,
  kFp8M2RowPair,
  kFp8M2RowQuad,
  kFp8M8Fixed,
  kFp8M8RowPair,
  kNvFp4SmallMVector,
  kNvFp4M2ScaleCodebook,
  kNvFp4M2RowQuad,
  kNvFp4M8Fixed,
  kNvFp4M8ScaleCodebook,
  kFp8M16Wmma,
  kNvFp4M16Wmma,
  kSplitM16IntoM8,
};

struct ProjectionQuery {
  WeightEncoding encoding;
  std::size_t token_count;
  std::size_t rows;
  std::size_t columns;
  bool weight_aligned_4;
  bool weight_aligned_16;
  bool activation_aligned_8;
  bool block_scales_aligned_2;
};

struct ProjectionPlan {
  ProjectionShape shape;
  ProjectionRoute route;
  unsigned int maximum_blocks;
  std::size_t launch_count;
};

inline constexpr std::size_t kFp8VectorColumns = 1'024U;
inline constexpr std::size_t kNvFp4VectorColumns = 256U;
inline constexpr std::size_t kNvFp4GroupSize = 16U;
inline constexpr std::size_t kFp8PersistentMinimumRows = 1'024U;
inline constexpr unsigned int kFp8PersistentMaximumBlocks = 2'048U;
inline constexpr std::size_t kNvFp4M1ScaleCodebookMinimumRows = 8U;
inline constexpr std::size_t kNvFp4M1ScaleCodebookMinimumColumns = 5'120U;
inline constexpr unsigned int kNvFp4M1ScaleCodebookMaximumBlocks = 96U;
inline constexpr std::size_t kNvFp4M8ScaleCodebookMinimumRows = 16U;
inline constexpr unsigned int kNvFp4M2RowQuadMaximumBlocks = 64U;

[[nodiscard]] static Q3X_SM87_ROUTE_INLINE constexpr ProjectionShape
classify_projection_shape(
    const WeightEncoding encoding, const std::size_t rows,
    const std::size_t columns) noexcept {
  switch (encoding) {
    case WeightEncoding::kFp8:
      if (rows == 10'240U && columns == 5'120U) {
        return ProjectionShape::kFp8_10240x5120;
      }
      if (rows == 5'120U && columns == 6'144U) {
        return ProjectionShape::kFp8_5120x6144;
      }
      if (rows == 6'144U && columns == 5'120U) {
        return ProjectionShape::kFp8_6144x5120;
      }
      if (rows == 12'288U && columns == 5'120U) {
        return ProjectionShape::kFp8_12288x5120;
      }
      if (rows == 1'024U && columns == 5'120U) {
        return ProjectionShape::kFp8_1024x5120;
      }
      return ProjectionShape::kUnknown;
    case WeightEncoding::kNvFp4:
      if (rows == 17'408U && columns == 5'120U) {
        return ProjectionShape::kNvFp4_17408x5120;
      }
      if (rows == 5'120U && columns == 17'408U) {
        return ProjectionShape::kNvFp4_5120x17408;
      }
      if (rows == 248'320U && columns == 5'120U) {
        return ProjectionShape::kNvFp4_248320x5120;
      }
      return ProjectionShape::kUnknown;
  }
  return ProjectionShape::kUnknown;
}

[[nodiscard]] static Q3X_SM87_ROUTE_INLINE constexpr bool
is_fp8_checkpoint_shape(
    const ProjectionShape shape) noexcept {
  return shape == ProjectionShape::kFp8_10240x5120 ||
         shape == ProjectionShape::kFp8_5120x6144 ||
         shape == ProjectionShape::kFp8_6144x5120 ||
         shape == ProjectionShape::kFp8_12288x5120 ||
         shape == ProjectionShape::kFp8_1024x5120;
}

[[nodiscard]] static Q3X_SM87_ROUTE_INLINE constexpr bool
is_fp8_m16_wmma_shape(
    const ProjectionShape shape) noexcept {
  // The 1024-row projection intentionally remains two M8 launches: its WMMA
  // candidate regressed in the production gate.
  return shape == ProjectionShape::kFp8_10240x5120 ||
         shape == ProjectionShape::kFp8_5120x6144 ||
         shape == ProjectionShape::kFp8_6144x5120 ||
         shape == ProjectionShape::kFp8_12288x5120;
}

[[nodiscard]] static Q3X_SM87_ROUTE_INLINE constexpr bool is_nvfp4_mlp_shape(
    const ProjectionShape shape) noexcept {
  return shape == ProjectionShape::kNvFp4_17408x5120 ||
         shape == ProjectionShape::kNvFp4_5120x17408;
}

[[nodiscard]] static Q3X_SM87_ROUTE_INLINE constexpr bool
is_nvfp4_checkpoint_shape(
    const ProjectionShape shape) noexcept {
  return is_nvfp4_mlp_shape(shape) ||
         shape == ProjectionShape::kNvFp4_248320x5120;
}

[[nodiscard]] static Q3X_SM87_ROUTE_INLINE constexpr unsigned int
fp8_m1_row_quad_maximum_blocks(
    const ProjectionShape shape) noexcept {
  switch (shape) {
    case ProjectionShape::kFp8_10240x5120:
      return 1'536U;
    case ProjectionShape::kFp8_5120x6144:
      return 1'280U;
    case ProjectionShape::kFp8_6144x5120:
      return 768U;
    case ProjectionShape::kFp8_12288x5120:
      return 2'048U;
    default:
      return 0U;
  }
}

[[nodiscard]] static Q3X_SM87_ROUTE_INLINE constexpr unsigned int
fp8_m2_row_quad_maximum_blocks(
    const ProjectionShape shape) noexcept {
  switch (shape) {
    case ProjectionShape::kFp8_10240x5120:
      return 1'536U;
    case ProjectionShape::kFp8_5120x6144:
      return 768U;
    case ProjectionShape::kFp8_6144x5120:
      return 1'024U;
    case ProjectionShape::kFp8_12288x5120:
      return 2'048U;
    default:
      return 0U;
  }
}

[[nodiscard]] static Q3X_SM87_ROUTE_INLINE constexpr ProjectionPlan
make_projection_plan(
    const ProjectionShape shape, const ProjectionRoute route,
    const unsigned int maximum_blocks = 0U,
    const std::size_t launch_count = 1U) noexcept {
  return ProjectionPlan{shape, route, maximum_blocks, launch_count};
}

[[nodiscard]] static Q3X_SM87_ROUTE_INLINE constexpr ProjectionPlan
select_fp8_projection_plan(
    const ProjectionQuery& query, const ProjectionShape shape) noexcept {
  const bool vector_eligible =
      (query.columns % kFp8VectorColumns) == 0U &&
      query.weight_aligned_4 && query.activation_aligned_8;

  if (query.token_count == 16U) {
    if (is_fp8_m16_wmma_shape(shape) && query.weight_aligned_16 &&
        query.activation_aligned_8) {
      return make_projection_plan(shape, ProjectionRoute::kFp8M16Wmma);
    }
    return make_projection_plan(shape, ProjectionRoute::kSplitM16IntoM8, 0U,
                                2U);
  }

  if (query.token_count == 1U) {
    if (!vector_eligible) {
      return make_projection_plan(shape, ProjectionRoute::kFp8M1Scalar);
    }
    const unsigned int row_quad_blocks =
        fp8_m1_row_quad_maximum_blocks(shape);
    if (row_quad_blocks != 0U) {
      return make_projection_plan(shape, ProjectionRoute::kFp8M1RowQuad,
                                  row_quad_blocks);
    }
    if (shape == ProjectionShape::kFp8_1024x5120) {
      return make_projection_plan(shape, ProjectionRoute::kFp8M1RowPair,
                                  kFp8PersistentMaximumBlocks);
    }
    if (query.rows >= kFp8PersistentMinimumRows) {
      return make_projection_plan(shape,
                                  ProjectionRoute::kFp8M1VectorGridCap,
                                  kFp8PersistentMaximumBlocks);
    }
    return make_projection_plan(shape, ProjectionRoute::kFp8M1Vector);
  }

  if (!vector_eligible) {
    return make_projection_plan(shape, ProjectionRoute::kSerialM1, 0U,
                                query.token_count);
  }

  if (query.token_count == 2U) {
    const unsigned int row_quad_blocks =
        fp8_m2_row_quad_maximum_blocks(shape);
    if (row_quad_blocks != 0U) {
      return make_projection_plan(shape, ProjectionRoute::kFp8M2RowQuad,
                                  row_quad_blocks);
    }
    if (shape == ProjectionShape::kFp8_1024x5120) {
      return make_projection_plan(shape, ProjectionRoute::kFp8M2RowPair,
                                  kFp8PersistentMaximumBlocks);
    }
    if (query.rows >= kFp8PersistentMinimumRows) {
      return make_projection_plan(shape,
                                  ProjectionRoute::kFp8M2VectorGridCap,
                                  kFp8PersistentMaximumBlocks);
    }
    return make_projection_plan(shape, ProjectionRoute::kFp8SmallMVector);
  }

  if (query.token_count == 8U) {
    if (is_fp8_checkpoint_shape(shape)) {
      return make_projection_plan(shape, ProjectionRoute::kFp8M8Fixed);
    }
    if (query.rows >= kFp8PersistentMinimumRows) {
      return make_projection_plan(shape, ProjectionRoute::kFp8M8RowPair);
    }
  }
  return make_projection_plan(shape, ProjectionRoute::kFp8SmallMVector);
}

[[nodiscard]] static Q3X_SM87_ROUTE_INLINE constexpr ProjectionPlan
select_nvfp4_projection_plan(
    const ProjectionQuery& query, const ProjectionShape shape) noexcept {
  const bool vector_eligible =
      (query.columns % kNvFp4VectorColumns) == 0U &&
      query.weight_aligned_4 && query.activation_aligned_8;

  if (query.token_count == 16U) {
    if (is_nvfp4_mlp_shape(shape) && query.weight_aligned_16 &&
        query.activation_aligned_8 && query.block_scales_aligned_2) {
      return make_projection_plan(shape, ProjectionRoute::kNvFp4M16Wmma);
    }
    return make_projection_plan(shape, ProjectionRoute::kSplitM16IntoM8, 0U,
                                2U);
  }

  if (query.token_count == 1U) {
    if (!vector_eligible) {
      return make_projection_plan(shape, ProjectionRoute::kNvFp4M1Scalar);
    }
    if (shape == ProjectionShape::kNvFp4_5120x17408) {
      return make_projection_plan(
          shape, ProjectionRoute::kNvFp4M1DownActivationStaged);
    }
    if (shape == ProjectionShape::kNvFp4_17408x5120) {
      return make_projection_plan(
          shape, ProjectionRoute::kNvFp4M1GateUpActivationStaged);
    }
    if (shape == ProjectionShape::kNvFp4_248320x5120) {
      return make_projection_plan(
          shape, ProjectionRoute::kNvFp4M1LmHeadActivationStaged);
    }
    if (query.rows >= kNvFp4M1ScaleCodebookMinimumRows &&
        query.columns >= kNvFp4M1ScaleCodebookMinimumColumns) {
      return make_projection_plan(
          shape, ProjectionRoute::kNvFp4M1ScaleCodebook,
          kNvFp4M1ScaleCodebookMaximumBlocks);
    }
    return make_projection_plan(shape, ProjectionRoute::kNvFp4M1Vector);
  }

  if (!vector_eligible) {
    return make_projection_plan(shape, ProjectionRoute::kSerialM1, 0U,
                                query.token_count);
  }

  if (query.token_count == 2U) {
    if (is_nvfp4_mlp_shape(shape)) {
      return make_projection_plan(shape, ProjectionRoute::kNvFp4M2RowQuad,
                                  kNvFp4M2RowQuadMaximumBlocks);
    }
    if (query.rows >= kNvFp4M1ScaleCodebookMinimumRows &&
        query.columns >= kNvFp4M1ScaleCodebookMinimumColumns) {
      return make_projection_plan(shape,
                                  ProjectionRoute::kNvFp4M2ScaleCodebook);
    }
    return make_projection_plan(shape, ProjectionRoute::kNvFp4SmallMVector);
  }

  if (query.token_count == 8U) {
    if (is_nvfp4_mlp_shape(shape)) {
      return make_projection_plan(shape, ProjectionRoute::kNvFp4M8Fixed);
    }
    if (query.rows >= kNvFp4M8ScaleCodebookMinimumRows) {
      return make_projection_plan(
          shape, ProjectionRoute::kNvFp4M8ScaleCodebook);
    }
  }
  return make_projection_plan(shape, ProjectionRoute::kNvFp4SmallMVector);
}

[[nodiscard]] static Q3X_SM87_ROUTE_INLINE constexpr ProjectionPlan
select_projection_plan(
    const ProjectionQuery& query) noexcept {
  if (query.token_count == 0U || query.token_count > 16U ||
      (query.token_count > 8U && query.token_count != 16U)) {
    return make_projection_plan(ProjectionShape::kUnknown,
                                ProjectionRoute::kInvalid, 0U, 0U);
  }

  switch (query.encoding) {
    case WeightEncoding::kFp8:
      break;
    case WeightEncoding::kNvFp4:
      if ((query.columns % kNvFp4GroupSize) != 0U) {
        return make_projection_plan(ProjectionShape::kUnknown,
                                    ProjectionRoute::kInvalid, 0U, 0U);
      }
      break;
    default:
      return make_projection_plan(ProjectionShape::kUnknown,
                                  ProjectionRoute::kInvalid, 0U, 0U);
  }

  const ProjectionShape shape =
      classify_projection_shape(query.encoding, query.rows, query.columns);
  if (query.rows == 0U || query.columns == 0U) {
    return make_projection_plan(shape, ProjectionRoute::kNoOp, 0U, 0U);
  }
  if (query.encoding == WeightEncoding::kFp8) {
    return select_fp8_projection_plan(query, shape);
  }
  return select_nvfp4_projection_plan(query, shape);
}

}  // namespace q3x::kernels::sm87_detail

#undef Q3X_SM87_ROUTE_INLINE
