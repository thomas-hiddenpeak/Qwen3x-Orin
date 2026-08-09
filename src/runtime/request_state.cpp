#include "q3x/runtime/request_state.h"

#include "q3x/runtime/prefill_workspace_plan.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace q3x::runtime {
namespace {

constexpr std::uint64_t kBf16Bytes = 2U;
constexpr std::uint64_t kFp32Bytes = 4U;
constexpr std::uint64_t kHiddenElements = 5'120U;
constexpr std::uint64_t kProjectionElements = 17'408U;
constexpr std::uint64_t kLinearScalarElements = 48U;
// The scheduler-wide Marlin admission reuses this buffer for its largest
// 16-SM x M64 x N256 FP32 cross-CTA reduction. Production's vocabulary
// scratch is slightly smaller, so reserve the larger capacity for both.
constexpr std::uint64_t kFp32MinimumElements = 262'144U;
constexpr std::uint64_t kQueryHeadCount = 24U;
constexpr std::uint64_t kKvHeadCount = 4U;
constexpr std::uint64_t kHeadDimension = 256U;
constexpr std::uint64_t kConvChannels = 10'240U;
constexpr std::uint64_t kConvHistory = 3U;
constexpr std::uint64_t kGdnValueHeads = 48U;
constexpr std::uint64_t kRequestGdnHeadDimension = 128U;
constexpr std::uint64_t kRopePairs = 32U;
constexpr float kRopeTheta = 10'000'000.0F;
constexpr std::uint64_t kMaximumConfigurableArenaBytes =
    64ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr std::uint64_t kLayerMajorFamilyPhaseArenaBytes = 855'638'016U;
constexpr std::uint64_t kProjectionTemporaryBytes = 1'048'832U;

constexpr std::uint64_t kGdnZOffset = 167'772'160U;
constexpr std::uint64_t kGdnAOffset = 268'435'456U;
constexpr std::uint64_t kGdnBOffset = 269'221'888U;
constexpr std::uint64_t kGdnCoreOffset = 270'008'320U;
constexpr std::uint64_t kGdnNativeWorkspaceOffset = 370'671'616U;
constexpr std::uint64_t kGdnProjectionTemporaryOffset = 353'894'400U;
constexpr std::uint64_t kOutputProjectionTemporaryOffset = 83'886'080U;

constexpr std::uint64_t kAttentionProcessedQOffset = 201'326'592U;
constexpr std::uint64_t kAttentionPackedGateOffset = 301'989'888U;
constexpr std::uint64_t kAttentionProjectionTemporaryOffset = 285'212'672U;
constexpr std::uint64_t kAttentionBranchOutputOffset = 100'663'296U;
constexpr std::uint64_t kAttentionOutputTemporaryOffset = 184'549'376U;

constexpr std::uint64_t kMlpUpOffset = 285'212'672U;
constexpr std::uint64_t kMlpActivatedOffset = 570'425'344U;
constexpr std::uint64_t kMlpGateUpTemporaryOffset = 654'311'424U;

RequestDiagnostic make_diagnostic(RequestErrorCode code,
                                  std::string message,
                                  std::string context = {},
                                  std::string expected = {},
                                  std::string actual = {},
                                  int cuda_error = 0) {
    RequestDiagnostic diagnostic;
    diagnostic.code = code;
    diagnostic.message = std::move(message);
    diagnostic.context = std::move(context);
    diagnostic.expected = std::move(expected);
    diagnostic.actual = std::move(actual);
    diagnostic.cuda_error = cuda_error;
    return diagnostic;
}

RequestPlanResult plan_failure(RequestDiagnostic diagnostic) {
    RequestPlanResult result;
    result.diagnostic = std::move(diagnostic);
    return result;
}

LayerMajorRequestPlanResult layer_major_plan_failure(
    RequestDiagnostic diagnostic) {
    LayerMajorRequestPlanResult result;
    result.diagnostic = std::move(diagnostic);
    return result;
}

RequestStateResult state_failure(RequestDiagnostic diagnostic) {
    RequestStateResult result;
    result.diagnostic = std::move(diagnostic);
    return result;
}

bool checked_add(std::uint64_t left,
                 std::uint64_t right,
                 std::uint64_t& output) noexcept {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        return false;
    }
    output = left + right;
    return true;
}

bool checked_multiply(std::uint64_t left,
                      std::uint64_t right,
                      std::uint64_t& output) noexcept {
    if (left != 0U &&
        right > std::numeric_limits<std::uint64_t>::max() / left) {
        return false;
    }
    output = left * right;
    return true;
}

bool checked_align(std::uint64_t value, std::uint64_t& output) noexcept {
    constexpr std::uint64_t mask = kRequestArenaAlignment - 1U;
    if (value > std::numeric_limits<std::uint64_t>::max() - mask) {
        return false;
    }
    output = (value + mask) & ~mask;
    return true;
}

class PlanBuilder {
  public:
    [[nodiscard]] bool add(std::uint64_t elements,
                           std::uint32_t element_size,
                           RequestRegion& region) noexcept {
        std::uint64_t bytes = 0U;
        std::uint64_t aligned = 0U;
        std::uint64_t end = 0U;
        if (element_size == 0U ||
            !checked_multiply(elements, element_size, bytes) || bytes == 0U ||
            !checked_align(cursor_, aligned) ||
            !checked_add(aligned, bytes, end)) {
            return false;
        }
        region.arena_offset = aligned;
        region.byte_size = bytes;
        region.element_capacity = elements;
        region.element_size_bytes = element_size;
        cursor_ = end;
        return true;
    }

    [[nodiscard]] bool align() noexcept {
        std::uint64_t aligned = 0U;
        if (!checked_align(cursor_, aligned)) {
            return false;
        }
        cursor_ = aligned;
        return true;
    }

    [[nodiscard]] std::uint64_t cursor() const noexcept { return cursor_; }

  private:
    std::uint64_t cursor_ = 0U;
};

[[nodiscard]] bool add_matrix(PlanBuilder& builder,
                              const std::uint32_t rows,
                              const std::uint32_t columns,
                              const std::uint64_t row_stride_elements,
                              const std::uint32_t element_size,
                              RequestMatrixRegion& matrix) noexcept {
    std::uint64_t elements = 0U;
    if (rows == 0U || columns == 0U || row_stride_elements < columns ||
        !checked_multiply(rows, row_stride_elements, elements) ||
        !builder.add(elements, element_size, matrix.storage)) {
        return false;
    }
    matrix.row_capacity = rows;
    matrix.columns = columns;
    matrix.row_stride_elements = row_stride_elements;
    return true;
}

[[nodiscard]] bool make_subregion_checked(const RequestRegion& aggregate,
                                          const std::uint64_t relative_offset,
                                          const std::uint64_t byte_size,
                                          const std::uint64_t elements,
                                          const std::uint32_t element_size,
                                          RequestRegion& result) noexcept {
    std::uint64_t relative_end = 0U;
    std::uint64_t arena_offset = 0U;
    std::uint64_t typed_bytes = 0U;
    if (byte_size == 0U || elements == 0U || element_size == 0U ||
        !checked_multiply(elements, element_size, typed_bytes) ||
        typed_bytes != byte_size ||
        !checked_add(relative_offset, byte_size, relative_end) ||
        relative_end > aggregate.byte_size ||
        !checked_add(aggregate.arena_offset, relative_offset, arena_offset)) {
        return false;
    }
    result.arena_offset = arena_offset;
    result.byte_size = byte_size;
    result.element_capacity = elements;
    result.element_size_bytes = element_size;
    return true;
}

[[nodiscard]] bool make_byte_subregion_checked(
    const RequestRegion& aggregate,
    const std::uint64_t relative_offset,
    const std::uint64_t byte_size,
    RequestRegion& result) noexcept {
    return make_subregion_checked(aggregate, relative_offset, byte_size,
                                  byte_size, 1U, result);
}

[[nodiscard]] bool make_matrix_subregion_checked(
    const RequestRegion& aggregate,
    const std::uint64_t relative_offset,
    const std::uint32_t rows,
    const std::uint32_t columns,
    const std::uint64_t row_stride_elements,
    const std::uint32_t element_size,
    RequestMatrixRegion& result) noexcept {
    std::uint64_t elements = 0U;
    std::uint64_t bytes = 0U;
    if (rows == 0U || columns == 0U || row_stride_elements < columns ||
        !checked_multiply(rows, row_stride_elements, elements) ||
        !checked_multiply(elements, element_size, bytes) ||
        !make_subregion_checked(aggregate, relative_offset, bytes, elements,
                                element_size, result.storage)) {
        return false;
    }
    result.row_capacity = rows;
    result.columns = columns;
    result.row_stride_elements = row_stride_elements;
    return true;
}

RequestDiagnostic cuda_diagnostic(cudaError_t status, std::string context) {
    const char* const name = cudaGetErrorName(status);
    const char* const description = cudaGetErrorString(status);
    std::string actual = name == nullptr ? "unknown CUDA error" : name;
    if (description != nullptr) {
        actual += ": ";
        actual += description;
    }
    return make_diagnostic(RequestErrorCode::kCudaFailure,
                           "CUDA request-state operation failed",
                           std::move(context),
                           {},
                           std::move(actual),
                           static_cast<int>(status));
}

struct CreateResources {
    void* arena = nullptr;
    cudaStream_t stream = nullptr;

    CreateResources() = default;
    CreateResources(const CreateResources&) = delete;
    CreateResources& operator=(const CreateResources&) = delete;

    ~CreateResources() {
        if (stream != nullptr) {
            (void)cudaStreamSynchronize(stream);
            (void)cudaStreamDestroy(stream);
        }
        if (arena != nullptr) {
            (void)cudaFree(arena);
        }
        (void)cudaGetLastError();
    }
};

RequestViewResult access_failure(RequestAccessError error) noexcept {
    RequestViewResult result;
    result.error = error;
    return result;
}

RequestConstViewResult const_access_failure(RequestAccessError error) noexcept {
    RequestConstViewResult result;
    result.error = error;
    return result;
}

RequestMatrixViewResult matrix_access_failure(
    const RequestAccessError error) noexcept {
    RequestMatrixViewResult result;
    result.error = error;
    return result;
}

template <typename Views>
LayerMajorTypedViewResult<Views> typed_access_failure(
    const RequestAccessError error) noexcept {
    LayerMajorTypedViewResult<Views> result;
    result.error = error;
    return result;
}

RequestRegion subregion(const RequestRegion& aggregate,
                        std::uint64_t relative_offset,
                        std::uint64_t elements,
                        std::uint32_t element_size) noexcept {
    RequestRegion result;
    result.arena_offset = aggregate.arena_offset + relative_offset;
    result.byte_size = elements * element_size;
    result.element_capacity = elements;
    result.element_size_bytes = element_size;
    return result;
}

float bf16_round_trip(const float value) noexcept {
    std::uint32_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t least_significant_retained_bit =
        (bits >> 16U) & 1U;
    bits += 0x7FFFU + least_significant_retained_bit;
    bits &= 0xFFFF0000U;
    float rounded = 0.0F;
    std::memcpy(&rounded, &bits, sizeof(rounded));
    return rounded;
}

}  // namespace

RequestLayerSlotResult map_request_layer(
    const std::size_t layer_index,
    const model::LayerType required_type) noexcept {
    RequestLayerSlotResult result;
    if (layer_index >= kRequestLayerCount) {
        result.error = RequestAccessError::kLayerOutOfRange;
        return result;
    }
    if (required_type != model::LayerType::kLinearAttention &&
        required_type != model::LayerType::kFullAttention) {
        result.error = RequestAccessError::kLayerTypeMismatch;
        return result;
    }
    RequestLayerSlot slot;
    if (((layer_index + 1U) % 4U) == 0U) {
        slot.type = model::LayerType::kFullAttention;
        slot.slot = layer_index / 4U;
    } else {
        slot.type = model::LayerType::kLinearAttention;
        slot.slot = layer_index - layer_index / 4U;
    }
    if (slot.type != required_type) {
        result.error = RequestAccessError::kLayerTypeMismatch;
        return result;
    }
    if ((slot.type == model::LayerType::kLinearAttention &&
         slot.slot >= kRequestLinearLayerCount) ||
        (slot.type == model::LayerType::kFullAttention &&
         slot.slot >= kRequestFullLayerCount)) {
        result.error = RequestAccessError::kSlotOutOfRange;
        return result;
    }
    result.value.emplace(slot);
    return result;
}

RequestAccessError validate_request_memory_profile(
    const RequestMemoryProfile actual,
    const RequestMemoryProfile required) noexcept {
    return actual == required ? RequestAccessError::kNone
                              : RequestAccessError::kMemoryProfileMismatch;
}

RequestPlanResult build_request_memory_plan(
    const RequestMemoryOptions& options) {
    if (options.batch_size != 1U || options.max_sequence_length == 0U ||
        options.prefill_chunk_size == 0U ||
        options.prefill_chunk_size > kMaximumRequestPrefillChunkSize ||
        options.max_arena_bytes == 0U ||
        options.max_arena_bytes > kMaximumConfigurableArenaBytes) {
        return plan_failure(make_diagnostic(
            RequestErrorCode::kInvalidOption,
            "request memory options violate batch, prefill chunk, sequence, or "
            "arena limits",
            "options"));
    }

    RequestMemoryPlan plan;
    PlanBuilder builder;
    std::uint64_t conv_elements = 0U;
    std::uint64_t gdn_elements = 0U;
    std::uint64_t kv_elements = 0U;
    std::uint64_t probability_elements = 0U;
    std::uint64_t rope_elements = 0U;
    std::uint64_t hidden_elements = 0U;
    std::uint64_t projection_elements = 0U;
    std::uint64_t linear_scalar_elements = 0U;
    if (!checked_multiply(kRequestLinearLayerCount,
                          kConvChannels,
                          conv_elements) ||
        !checked_multiply(conv_elements, kConvHistory, conv_elements) ||
        !checked_multiply(kRequestLinearLayerCount,
                          kGdnValueHeads,
                          gdn_elements) ||
        !checked_multiply(gdn_elements,
                          kRequestGdnHeadDimension,
                          gdn_elements) ||
        !checked_multiply(gdn_elements,
                          kRequestGdnHeadDimension,
                          gdn_elements) ||
        !checked_multiply(options.max_sequence_length,
                          kKvHeadCount,
                          kv_elements) ||
        !checked_multiply(kv_elements, kHeadDimension, kv_elements) ||
        !checked_multiply(options.max_sequence_length,
                          kQueryHeadCount,
                          probability_elements) ||
        !checked_multiply(options.max_sequence_length,
                          kRopePairs,
                          rope_elements) ||
        !checked_multiply(kHiddenElements,
                          options.prefill_chunk_size,
                          hidden_elements) ||
        !checked_multiply(kProjectionElements,
                          options.prefill_chunk_size,
                          projection_elements) ||
        !checked_multiply(kLinearScalarElements,
                          options.prefill_chunk_size,
                          linear_scalar_elements)) {
        return plan_failure(make_diagnostic(
            RequestErrorCode::kArithmeticOverflow,
            "request dimensions overflow uint64 during plan construction",
            "max_sequence_length"));
    }
    if (options.max_sequence_length > kAbsoluteRequestMaxSequenceLength) {
        return plan_failure(make_diagnostic(
            RequestErrorCode::kInvalidOption,
            "max_sequence_length exceeds the Qwen3.6-27B absolute limit",
            "max_sequence_length",
            std::to_string(kAbsoluteRequestMaxSequenceLength),
            std::to_string(options.max_sequence_length)));
    }

    plan.batch_size = options.batch_size;
    plan.prefill_chunk_size = options.prefill_chunk_size;
    plan.max_sequence_length =
        static_cast<std::uint32_t>(options.max_sequence_length);
    plan.persistent_offset = 0U;
    if (!builder.add(conv_elements, kBf16Bytes, plan.conv_state) ||
        !builder.add(gdn_elements, kBf16Bytes, plan.gdn_state)) {
        return plan_failure(make_diagnostic(
            RequestErrorCode::kArithmeticOverflow,
            "persistent state layout overflows uint64",
            "persistent"));
    }
    for (std::size_t slot = 0U; slot < kRequestFullLayerCount; ++slot) {
        if (!builder.add(kv_elements, kBf16Bytes, plan.key_cache[slot]) ||
            !builder.add(kv_elements, kBf16Bytes, plan.value_cache[slot])) {
            return plan_failure(make_diagnostic(
                RequestErrorCode::kArithmeticOverflow,
                "KV cache layout overflows uint64",
                "kv_cache"));
        }
    }
    if (!builder.align()) {
        return plan_failure(make_diagnostic(
            RequestErrorCode::kArithmeticOverflow,
            "persistent state alignment overflows uint64",
            "persistent"));
    }
    plan.persistent_bytes = builder.cursor() - plan.persistent_offset;

    plan.workspace_offset = builder.cursor();
    for (RequestRegion& hidden : plan.hidden_bf16) {
        if (!builder.add(hidden_elements, kBf16Bytes, hidden)) {
            return plan_failure(make_diagnostic(
                RequestErrorCode::kArithmeticOverflow,
                "hidden workspace layout overflows uint64",
                "hidden_bf16"));
        }
    }
    for (RequestRegion& projection : plan.projection_bf16) {
        if (!builder.add(projection_elements, kBf16Bytes, projection)) {
            return plan_failure(make_diagnostic(
                RequestErrorCode::kArithmeticOverflow,
                "projection workspace layout overflows uint64",
                "projection_bf16"));
        }
    }
    if (!builder.add(linear_scalar_elements,
                     kBf16Bytes,
                     plan.linear_a_bf16) ||
        !builder.add(linear_scalar_elements,
                     kBf16Bytes,
                     plan.linear_b_bf16)) {
        return plan_failure(make_diagnostic(
            RequestErrorCode::kArithmeticOverflow,
            "linear scalar workspace layout overflows uint64",
            "linear_a_b"));
    }
    const std::uint64_t fp32_capacity =
        std::max(kFp32MinimumElements, probability_elements);
    if (!builder.add(fp32_capacity, kFp32Bytes, plan.fp32_scratch)) {
        return plan_failure(make_diagnostic(
            RequestErrorCode::kArithmeticOverflow,
            "FP32 scratch layout overflows uint64",
            "fp32_scratch"));
    }
    plan.gqa_probability_scratch = plan.fp32_scratch;
    plan.gqa_probability_scratch.byte_size = probability_elements * kFp32Bytes;
    plan.gqa_probability_scratch.element_capacity = probability_elements;
    if (!builder.align()) {
        return plan_failure(make_diagnostic(
            RequestErrorCode::kArithmeticOverflow,
            "workspace alignment overflows uint64",
            "workspace"));
    }
    plan.workspace_bytes = builder.cursor() - plan.workspace_offset;

    plan.rope_offset = builder.cursor();
    if (!builder.add(rope_elements, kFp32Bytes, plan.rope_cos_fp32) ||
        !builder.add(rope_elements, kFp32Bytes, plan.rope_sin_fp32) ||
        !builder.align()) {
        return plan_failure(make_diagnostic(
            RequestErrorCode::kArithmeticOverflow,
            "RoPE cache layout overflows uint64",
            "rope"));
    }
    plan.rope_bytes = builder.cursor() - plan.rope_offset;
    plan.arena_bytes = builder.cursor();

    std::size_t linear_slots = 0U;
    std::size_t full_slots = 0U;
    for (std::size_t layer = 0U; layer < kRequestLayerCount; ++layer) {
        const model::LayerType type =
            ((layer + 1U) % 4U) == 0U
                ? model::LayerType::kFullAttention
                : model::LayerType::kLinearAttention;
        RequestLayerSlotResult mapped = map_request_layer(layer, type);
        if (!mapped) {
            return plan_failure(make_diagnostic(
                RequestErrorCode::kInvalidLayerSchedule,
                "fixed layer schedule could not be mapped",
                std::to_string(layer)));
        }
        plan.layers[layer] = *mapped.value;
        if (type == model::LayerType::kFullAttention) {
            ++full_slots;
        } else {
            ++linear_slots;
        }
    }
    if (linear_slots != kRequestLinearLayerCount ||
        full_slots != kRequestFullLayerCount ||
        plan.conv_state.byte_size != kRequestConvStateBytes ||
        plan.gdn_state.byte_size != kRequestGdnStateBytes) {
        return plan_failure(make_diagnostic(
            RequestErrorCode::kInvalidLayerSchedule,
            "fixed layer schedule or persistent ABI totals are inconsistent",
            "schedule"));
    }
    if (plan.arena_bytes > options.max_arena_bytes) {
        return plan_failure(make_diagnostic(
            RequestErrorCode::kArenaLimitExceeded,
            "request arena exceeds configured max_arena_bytes",
            "max_arena_bytes",
            std::to_string(options.max_arena_bytes),
            std::to_string(plan.arena_bytes)));
    }

    RequestPlanResult result;
    result.value.emplace(plan);
    return result;
}

LayerMajorRequestPlanResult build_layer_major_request_memory_plan(
    const LayerMajorRequestMemoryOptions& options) {
    if (options.batch_size != 1U || options.max_sequence_length == 0U ||
        options.max_sequence_length > kAbsoluteRequestMaxSequenceLength ||
        options.max_arena_bytes == 0U ||
        options.max_arena_bytes > kMaximumConfigurableArenaBytes) {
        return layer_major_plan_failure(make_diagnostic(
            RequestErrorCode::kInvalidOption,
            "layer-major request options violate batch, explicit sequence, "
            "or arena limits",
            "options"));
    }

    LayerMajorPrefillWorkspaceOptions workspace_options;
    workspace_options.sequence_capacity_tokens =
        options.max_sequence_length;
    workspace_options.request_arena_limit_bytes = options.max_arena_bytes;
    workspace_options.hidden_strategy =
        PrefillHiddenStrategy::kSinglePromptWideConditional;
    workspace_options.scratch_strategy = PrefillOperatorScratchStrategy::
        kC8192FamilyOverlayWithDisjointLegacyC512;
    workspace_options.gdn_tactic =
        PrefillGdnPhysicalTactic::kC64NativeInPlaceConv;
    workspace_options.legacy_gdn_tactic =
        PrefillLegacyGdnPhysicalTactic::kC16Composite;
    workspace_options.mlp_tactic =
        PrefillMlpPhysicalTactic::kSeparateGateUpAndSilu;
    const LayerMajorPrefillWorkspacePlanResult workspace_result =
        build_unbound_layer_major_prefill_workspace_plan(workspace_options);
    if (!workspace_result) {
        return layer_major_plan_failure(make_diagnostic(
            RequestErrorCode::kInvalidOption,
            "fixed layer-major workspace strategy was rejected by its planner",
            "prefill_workspace_plan",
            "valid fixed strategy",
            std::to_string(static_cast<unsigned>(workspace_result.error))));
    }
    const LayerMajorPrefillWorkspacePlan& workspace =
        *workspace_result.value;
    if (workspace.selected.required_bytes > options.max_arena_bytes) {
        return layer_major_plan_failure(make_diagnostic(
            RequestErrorCode::kArenaLimitExceeded,
            "layer-major request arena exceeds configured max_arena_bytes",
            "max_arena_bytes",
            std::to_string(options.max_arena_bytes),
            std::to_string(workspace.selected.required_bytes)));
    }

    // Reuse the already qualified legacy arithmetic only as a source of exact
    // common persistent/KV capacities and the fixed layer schedule. Its
    // workspace offsets are not copied into the candidate layout.
    RequestMemoryOptions legacy_shape_options;
    legacy_shape_options.batch_size = 1U;
    legacy_shape_options.prefill_chunk_size =
        kMaximumRequestPrefillChunkSize;
    legacy_shape_options.max_sequence_length = options.max_sequence_length;
    legacy_shape_options.max_arena_bytes = kMaximumRequestArenaBytes;
    RequestPlanResult legacy_shape_result =
        build_request_memory_plan(legacy_shape_options);
    if (!legacy_shape_result) {
        return layer_major_plan_failure(
            std::move(legacy_shape_result.diagnostic));
    }
    const RequestMemoryPlan& legacy_shape = *legacy_shape_result.value;

    LayerMajorRequestMemoryPlan plan;
    RequestMemoryPlan& common = plan.common;
    common.profile = RequestMemoryProfile::kLayerMajorC8192;
    common.batch_size = 1U;
    common.prefill_chunk_size = kMaximumRequestPrefillChunkSize;
    common.max_sequence_length =
        static_cast<std::uint32_t>(options.max_sequence_length);
    common.layers = legacy_shape.layers;
    plan.hidden_strategy = workspace_options.hidden_strategy;
    plan.scratch_strategy = workspace_options.scratch_strategy;
    plan.gdn_tactic = workspace_options.gdn_tactic;
    plan.legacy_gdn_tactic = workspace_options.legacy_gdn_tactic;
    plan.mlp_tactic = workspace_options.mlp_tactic;

    PlanBuilder builder;
    common.persistent_offset = 0U;
    if (!builder.add(legacy_shape.conv_state.element_capacity,
                     legacy_shape.conv_state.element_size_bytes,
                     common.conv_state) ||
        !builder.add(legacy_shape.gdn_state.element_capacity,
                     legacy_shape.gdn_state.element_size_bytes,
                     common.gdn_state)) {
        return layer_major_plan_failure(make_diagnostic(
            RequestErrorCode::kArithmeticOverflow,
            "layer-major persistent state layout overflows uint64",
            "persistent"));
    }
    for (std::size_t slot = 0U; slot < kRequestFullLayerCount; ++slot) {
        if (!builder.add(legacy_shape.key_cache[slot].element_capacity,
                         legacy_shape.key_cache[slot].element_size_bytes,
                         common.key_cache[slot]) ||
            !builder.add(legacy_shape.value_cache[slot].element_capacity,
                         legacy_shape.value_cache[slot].element_size_bytes,
                         common.value_cache[slot])) {
            return layer_major_plan_failure(make_diagnostic(
                RequestErrorCode::kArithmeticOverflow,
                "layer-major KV cache layout overflows uint64",
                "kv_cache"));
        }
    }
    if (!builder.align()) {
        return layer_major_plan_failure(make_diagnostic(
            RequestErrorCode::kArithmeticOverflow,
            "layer-major persistent alignment overflows uint64",
            "persistent"));
    }
    common.persistent_bytes = builder.cursor();
    common.workspace_offset = builder.cursor();

    if (!add_matrix(builder, common.max_sequence_length,
                    static_cast<std::uint32_t>(kHiddenElements),
                    kHiddenElements, kBf16Bytes,
                    plan.prompt_residual_bf16) ||
        !builder.add(kLayerMajorFamilyPhaseArenaBytes, 1U,
                     plan.c8192_family_phase_arena)) {
        return layer_major_plan_failure(make_diagnostic(
            RequestErrorCode::kArithmeticOverflow,
            "layer-major prompt residual or family arena overflows uint64",
            "layer_major_workspace"));
    }

    const RequestRegion& family = plan.c8192_family_phase_arena;
    constexpr std::uint32_t panel =
        kLayerMajorRequestOperatorPanelCapacity;
    if (!make_matrix_subregion_checked(family, 0U, panel, 1U, 1U, 4U,
                                       plan.panel_token_ids_u32) ||
        !make_matrix_subregion_checked(family, 0U, panel, 10'240U,
                                       10'240U, kBf16Bytes,
                                       plan.gdn.qkv_bf16) ||
        !make_matrix_subregion_checked(family, kGdnZOffset, panel, 6'144U,
                                       6'144U, kBf16Bytes,
                                       plan.gdn.z_bf16) ||
        !make_matrix_subregion_checked(family, kGdnAOffset, panel, 48U, 48U,
                                       kBf16Bytes, plan.gdn.a_bf16) ||
        !make_matrix_subregion_checked(family, kGdnBOffset, panel, 48U, 48U,
                                       kBf16Bytes, plan.gdn.b_bf16) ||
        !make_matrix_subregion_checked(family, kGdnCoreOffset, panel,
                                       6'144U, 6'144U, kBf16Bytes,
                                       plan.gdn.recurrent_core_bf16) ||
        !make_byte_subregion_checked(
            family, kGdnNativeWorkspaceOffset,
            kLayerMajorPrefillGdnC64NativeWorkspaceBytes,
            plan.gdn.native_c64_workspace) ||
        !make_matrix_subregion_checked(family, kGdnCoreOffset, panel,
                                       5'120U, 5'120U, kBf16Bytes,
                                       plan.gdn.normalized_input_bf16) ||
        !make_byte_subregion_checked(family,
                                     kGdnProjectionTemporaryOffset,
                                     kProjectionTemporaryBytes,
                                     plan.gdn.input_projection_temporary) ||
        !make_matrix_subregion_checked(family, 0U, panel, 5'120U, 5'120U,
                                       kBf16Bytes,
                                       plan.gdn.branch_output_bf16) ||
        !make_byte_subregion_checked(family,
                                     kOutputProjectionTemporaryOffset,
                                     kProjectionTemporaryBytes,
                                     plan.gdn.output_projection_temporary)) {
        return layer_major_plan_failure(make_diagnostic(
            RequestErrorCode::kInvalidLayerSchedule,
            "fixed GDN phase views exceed the C8192 family arena",
            "gdn_phase_layout"));
    }

    if (!make_matrix_subregion_checked(family, 0U, panel, 12'288U,
                                       12'288U, kBf16Bytes,
                                       plan.attention.raw_q_gate_bf16) ||
        !make_matrix_subregion_checked(
            family, kAttentionProcessedQOffset, panel, 6'144U, 6'144U,
            kBf16Bytes, plan.attention.processed_q_bf16) ||
        !make_matrix_subregion_checked(
            family, kAttentionPackedGateOffset, panel, 6'144U, 6'144U,
            kBf16Bytes, plan.attention.packed_gate_bf16) ||
        !make_matrix_subregion_checked(
            family, kAttentionProcessedQOffset, panel, 5'120U, 5'120U,
            kBf16Bytes, plan.attention.normalized_input_bf16) ||
        !make_byte_subregion_checked(
            family, kAttentionProjectionTemporaryOffset,
            kProjectionTemporaryBytes,
            plan.attention.input_projection_temporary) ||
        !make_matrix_subregion_checked(family, 0U, panel, 6'144U, 6'144U,
                                       kBf16Bytes,
                                       plan.attention.core_output_bf16) ||
        !make_matrix_subregion_checked(
            family, kAttentionBranchOutputOffset, panel, 5'120U, 5'120U,
            kBf16Bytes, plan.attention.branch_output_bf16) ||
        !make_byte_subregion_checked(
            family, kAttentionOutputTemporaryOffset,
            kProjectionTemporaryBytes,
            plan.attention.output_projection_temporary)) {
        return layer_major_plan_failure(make_diagnostic(
            RequestErrorCode::kInvalidLayerSchedule,
            "fixed Attention phase views exceed the C8192 family arena",
            "attention_phase_layout"));
    }

    if (!make_matrix_subregion_checked(family, 0U, panel, 17'408U,
                                       17'408U, kBf16Bytes,
                                       plan.mlp.gate_bf16) ||
        !make_matrix_subregion_checked(family, kMlpUpOffset, panel, 17'408U,
                                       17'408U, kBf16Bytes,
                                       plan.mlp.up_bf16) ||
        !make_matrix_subregion_checked(family, kMlpActivatedOffset, panel,
                                       17'408U, 17'408U, kBf16Bytes,
                                       plan.mlp.activated_bf16) ||
        !make_matrix_subregion_checked(family, kMlpActivatedOffset, panel,
                                       5'120U, 5'120U, kBf16Bytes,
                                       plan.mlp.normalized_input_bf16) ||
        !make_byte_subregion_checked(
            family, kMlpGateUpTemporaryOffset, kProjectionTemporaryBytes,
            plan.mlp.gate_up_projection_temporary) ||
        !make_matrix_subregion_checked(family, kMlpUpOffset, panel, 5'120U,
                                       5'120U, kBf16Bytes,
                                       plan.mlp.branch_output_bf16) ||
        !make_byte_subregion_checked(family, 0U,
                                     kProjectionTemporaryBytes,
                                     plan.mlp.down_projection_temporary)) {
        return layer_major_plan_failure(make_diagnostic(
            RequestErrorCode::kInvalidLayerSchedule,
            "fixed MLP phase views exceed the C8192 family arena",
            "mlp_phase_layout"));
    }

    const std::uint64_t legacy_offset = builder.cursor();
    for (std::size_t index = 0U; index < kRequestHiddenBufferCount; ++index) {
        if (!add_matrix(builder, kMaximumRequestPrefillChunkSize,
                        static_cast<std::uint32_t>(kHiddenElements),
                        kHiddenElements, kBf16Bytes,
                        plan.legacy_c512.hidden_bf16[index])) {
            return layer_major_plan_failure(make_diagnostic(
                RequestErrorCode::kArithmeticOverflow,
                "legacy hidden sublayout overflows uint64",
                "legacy_c512"));
        }
        common.hidden_bf16[index] =
            plan.legacy_c512.hidden_bf16[index].storage;
    }
    for (std::size_t index = 0U; index < kRequestProjectionBufferCount;
         ++index) {
        if (!add_matrix(builder, kMaximumRequestPrefillChunkSize,
                        static_cast<std::uint32_t>(kProjectionElements),
                        kProjectionElements, kBf16Bytes,
                        plan.legacy_c512.projection_bf16[index])) {
            return layer_major_plan_failure(make_diagnostic(
                RequestErrorCode::kArithmeticOverflow,
                "legacy projection sublayout overflows uint64",
                "legacy_c512"));
        }
        common.projection_bf16[index] =
            plan.legacy_c512.projection_bf16[index].storage;
    }
    if (!add_matrix(builder, kMaximumRequestPrefillChunkSize,
                    static_cast<std::uint32_t>(kLinearScalarElements),
                    kLinearScalarElements, kBf16Bytes,
                    plan.legacy_c512.linear_a_bf16) ||
        !add_matrix(builder, kMaximumRequestPrefillChunkSize,
                    static_cast<std::uint32_t>(kLinearScalarElements),
                    kLinearScalarElements, kBf16Bytes,
                    plan.legacy_c512.linear_b_bf16)) {
        return layer_major_plan_failure(make_diagnostic(
            RequestErrorCode::kArithmeticOverflow,
            "legacy linear scalar sublayout overflows uint64",
            "legacy_c512"));
    }
    common.linear_a_bf16 = plan.legacy_c512.linear_a_bf16.storage;
    common.linear_b_bf16 = plan.legacy_c512.linear_b_bf16.storage;
    if (!builder.add(legacy_shape.fp32_scratch.element_capacity,
                     kFp32Bytes, plan.legacy_c512.fp32_scratch)) {
        return layer_major_plan_failure(make_diagnostic(
            RequestErrorCode::kArithmeticOverflow,
            "legacy FP32 sublayout overflows uint64",
            "legacy_c512"));
    }
    plan.legacy_c512.gqa_probability_scratch =
        plan.legacy_c512.fp32_scratch;
    plan.legacy_c512.gqa_probability_scratch.byte_size =
        legacy_shape.gqa_probability_scratch.byte_size;
    plan.legacy_c512.gqa_probability_scratch.element_capacity =
        legacy_shape.gqa_probability_scratch.element_capacity;
    common.fp32_scratch = plan.legacy_c512.fp32_scratch;
    common.gqa_probability_scratch =
        plan.legacy_c512.gqa_probability_scratch;
    if (!builder.align()) {
        return layer_major_plan_failure(make_diagnostic(
            RequestErrorCode::kArithmeticOverflow,
            "legacy C512 sublayout alignment overflows uint64",
            "legacy_c512"));
    }
    const std::uint64_t selected_scratch_bytes =
        builder.cursor() - family.arena_offset;
    if (legacy_offset !=
            family.arena_offset + kLayerMajorFamilyPhaseArenaBytes ||
        selected_scratch_bytes !=
            workspace.operator_scratch.selected.total_required_bytes) {
        return layer_major_plan_failure(make_diagnostic(
            RequestErrorCode::kInvalidLayerSchedule,
            "RequestState scratch layout disagrees with workspace planner",
            "selected_operator_scratch",
            std::to_string(
                workspace.operator_scratch.selected.total_required_bytes),
            std::to_string(selected_scratch_bytes)));
    }

    if (!add_matrix(builder, 1U, static_cast<std::uint32_t>(kHiddenElements),
                    kHiddenElements, kBf16Bytes,
                    plan.final_hidden_bf16) ||
        !builder.align()) {
        return layer_major_plan_failure(make_diagnostic(
            RequestErrorCode::kArithmeticOverflow,
            "final hidden handoff layout overflows uint64",
            "final_hidden"));
    }
    common.workspace_bytes = builder.cursor() - common.workspace_offset;
    common.rope_offset = builder.cursor();
    if (!builder.add(legacy_shape.rope_cos_fp32.element_capacity,
                     kFp32Bytes, common.rope_cos_fp32) ||
        !builder.add(legacy_shape.rope_sin_fp32.element_capacity,
                     kFp32Bytes, common.rope_sin_fp32) ||
        !builder.align()) {
        return layer_major_plan_failure(make_diagnostic(
            RequestErrorCode::kArithmeticOverflow,
            "layer-major RoPE layout overflows uint64",
            "rope"));
    }
    common.rope_bytes = builder.cursor() - common.rope_offset;
    common.arena_bytes = builder.cursor();

    if (common.persistent_bytes !=
            workspace.persistent_state.total_required_bytes ||
        plan.prompt_residual_bf16.storage.byte_size !=
            workspace.prompt_wide_hidden.selected.aggregate_bf16
                .required_bytes ||
        plan.final_hidden_bf16.storage.byte_size !=
            workspace.final_hidden_handoff_bf16.required_bytes ||
        common.rope_bytes != workspace.position_state.total_required_bytes ||
        common.arena_bytes != workspace.selected.required_bytes ||
        common.arena_bytes > options.max_arena_bytes ||
        workspace.executable()) {
        return layer_major_plan_failure(make_diagnostic(
            RequestErrorCode::kInvalidLayerSchedule,
            "RequestState total or fixed profile disagrees with workspace planner",
            "layer_major_total",
            std::to_string(workspace.selected.required_bytes),
            std::to_string(common.arena_bytes)));
    }

    LayerMajorRequestPlanResult result;
    result.value.emplace(std::move(plan));
    return result;
}

RequestState::~RequestState() { release(); }

RequestState::RequestState(RequestState&& other) noexcept
    : arena_(std::exchange(other.arena_, nullptr)),
      plan_(std::exchange(other.plan_, {})),
      sequence_length_(std::exchange(other.sequence_length_, 0U)),
      layer_major_plan_(std::exchange(other.layer_major_plan_, std::nullopt)) {}

RequestState& RequestState::operator=(RequestState&& other) noexcept {
    if (this != &other) {
        release();
        arena_ = std::exchange(other.arena_, nullptr);
        plan_ = std::exchange(other.plan_, {});
        sequence_length_ = std::exchange(other.sequence_length_, 0U);
        layer_major_plan_ =
            std::exchange(other.layer_major_plan_, std::nullopt);
    }
    return *this;
}

void RequestState::release() noexcept {
    if (arena_ != nullptr) {
        (void)cudaFree(arena_);
        arena_ = nullptr;
        (void)cudaGetLastError();
    }
    plan_ = {};
    layer_major_plan_.reset();
    sequence_length_ = 0U;
}

DeviceBufferView RequestState::mutable_view(
    const RequestRegion& region) noexcept {
    DeviceBufferView result;
    result.device_data = static_cast<std::uint8_t*>(arena_) +
                         static_cast<std::size_t>(region.arena_offset);
    result.arena_offset = region.arena_offset;
    result.byte_size = region.byte_size;
    result.element_capacity = region.element_capacity;
    result.element_size_bytes = region.element_size_bytes;
    return result;
}

DeviceMatrixView RequestState::mutable_matrix_view(
    const RequestMatrixRegion& region) noexcept {
    DeviceMatrixView result;
    result.storage = mutable_view(region.storage);
    result.row_capacity = region.row_capacity;
    result.columns = region.columns;
    result.row_stride_elements = region.row_stride_elements;
    return result;
}

ConstDeviceBufferView RequestState::const_view(
    const RequestRegion& region) const noexcept {
    ConstDeviceBufferView result;
    result.device_data = static_cast<const std::uint8_t*>(arena_) +
                         static_cast<std::size_t>(region.arena_offset);
    result.arena_offset = region.arena_offset;
    result.byte_size = region.byte_size;
    result.element_capacity = region.element_capacity;
    result.element_size_bytes = region.element_size_bytes;
    return result;
}

RequestOperationStatus RequestState::commit_token() noexcept {
    if (arena_ == nullptr) {
        return {RequestAccessError::kEmptyState, 0};
    }
    if (sequence_length_ >= common_plan().max_sequence_length) {
        return {RequestAccessError::kCapacityExceeded, 0};
    }
    ++sequence_length_;
    return {};
}

RequestOperationStatus RequestState::set_sequence_length(
    const std::uint32_t length) noexcept {
    if (arena_ == nullptr) {
        return {RequestAccessError::kEmptyState, 0};
    }
    if (length > common_plan().max_sequence_length) {
        return {RequestAccessError::kCapacityExceeded, 0};
    }
    sequence_length_ = length;
    return {};
}

RequestViewResult RequestState::conv_state(const std::size_t layer_index) noexcept {
    if (arena_ == nullptr) {
        return access_failure(RequestAccessError::kEmptyState);
    }
    const RequestLayerSlotResult mapped = map_request_layer(
        layer_index, model::LayerType::kLinearAttention);
    if (!mapped) {
        return access_failure(mapped.error);
    }
    constexpr std::uint64_t elements_per_layer =
        kConvChannels * kConvHistory;
    const RequestRegion region = subregion(
        common_plan().conv_state,
        mapped.value->slot * elements_per_layer * kBf16Bytes,
        elements_per_layer,
        kBf16Bytes);
    RequestViewResult result;
    result.value.emplace(mutable_view(region));
    return result;
}

RequestViewResult RequestState::gdn_state(const std::size_t layer_index) noexcept {
    if (arena_ == nullptr) {
        return access_failure(RequestAccessError::kEmptyState);
    }
    const RequestLayerSlotResult mapped = map_request_layer(
        layer_index, model::LayerType::kLinearAttention);
    if (!mapped) {
        return access_failure(mapped.error);
    }
    constexpr std::uint64_t elements_per_layer =
        kGdnValueHeads * kRequestGdnHeadDimension *
        kRequestGdnHeadDimension;
    const RequestRegion region = subregion(
        common_plan().gdn_state,
        mapped.value->slot * elements_per_layer * kBf16Bytes,
        elements_per_layer,
        kBf16Bytes);
    RequestViewResult result;
    result.value.emplace(mutable_view(region));
    return result;
}

RequestViewResult RequestState::key_cache(const std::size_t layer_index) noexcept {
    if (arena_ == nullptr) {
        return access_failure(RequestAccessError::kEmptyState);
    }
    const RequestLayerSlotResult mapped = map_request_layer(
        layer_index, model::LayerType::kFullAttention);
    if (!mapped) {
        return access_failure(mapped.error);
    }
    RequestViewResult result;
    result.value.emplace(
        mutable_view(common_plan().key_cache[mapped.value->slot]));
    return result;
}

RequestViewResult RequestState::value_cache(const std::size_t layer_index) noexcept {
    if (arena_ == nullptr) {
        return access_failure(RequestAccessError::kEmptyState);
    }
    const RequestLayerSlotResult mapped = map_request_layer(
        layer_index, model::LayerType::kFullAttention);
    if (!mapped) {
        return access_failure(mapped.error);
    }
    RequestViewResult result;
    result.value.emplace(
        mutable_view(common_plan().value_cache[mapped.value->slot]));
    return result;
}

RequestViewResult RequestState::key_position(const std::size_t layer_index,
                                             const std::size_t position) noexcept {
    RequestViewResult cache = key_cache(layer_index);
    if (!cache) {
        return cache;
    }
    if (position >= common_plan().max_sequence_length) {
        return access_failure(RequestAccessError::kPositionOutOfRange);
    }
    constexpr std::uint64_t elements_per_position =
        kKvHeadCount * kHeadDimension;
    RequestRegion region;
    region.arena_offset = cache.value->arena_offset +
                          position * elements_per_position * kBf16Bytes;
    region.byte_size = elements_per_position * kBf16Bytes;
    region.element_capacity = elements_per_position;
    region.element_size_bytes = kBf16Bytes;
    RequestViewResult result;
    result.value.emplace(mutable_view(region));
    return result;
}

RequestViewResult RequestState::value_position(const std::size_t layer_index,
                                               const std::size_t position) noexcept {
    RequestViewResult cache = value_cache(layer_index);
    if (!cache) {
        return cache;
    }
    if (position >= common_plan().max_sequence_length) {
        return access_failure(RequestAccessError::kPositionOutOfRange);
    }
    constexpr std::uint64_t elements_per_position =
        kKvHeadCount * kHeadDimension;
    RequestRegion region;
    region.arena_offset = cache.value->arena_offset +
                          position * elements_per_position * kBf16Bytes;
    region.byte_size = elements_per_position * kBf16Bytes;
    region.element_capacity = elements_per_position;
    region.element_size_bytes = kBf16Bytes;
    RequestViewResult result;
    result.value.emplace(mutable_view(region));
    return result;
}

RequestViewResult RequestState::hidden_buffer(const std::size_t index) noexcept {
    if (arena_ == nullptr) {
        return access_failure(RequestAccessError::kEmptyState);
    }
    const RequestAccessError profile_error = validate_request_memory_profile(
        memory_profile(), RequestMemoryProfile::kLegacyC512);
    if (profile_error != RequestAccessError::kNone) {
        return access_failure(profile_error);
    }
    if (index >= plan_.hidden_bf16.size()) {
        return access_failure(RequestAccessError::kInvalidBufferIndex);
    }
    RequestViewResult result;
    result.value.emplace(mutable_view(plan_.hidden_bf16[index]));
    return result;
}

RequestViewResult RequestState::projection_buffer(
    const std::size_t index) noexcept {
    if (arena_ == nullptr) {
        return access_failure(RequestAccessError::kEmptyState);
    }
    const RequestAccessError profile_error = validate_request_memory_profile(
        memory_profile(), RequestMemoryProfile::kLegacyC512);
    if (profile_error != RequestAccessError::kNone) {
        return access_failure(profile_error);
    }
    if (index >= plan_.projection_bf16.size()) {
        return access_failure(RequestAccessError::kInvalidBufferIndex);
    }
    RequestViewResult result;
    result.value.emplace(mutable_view(plan_.projection_bf16[index]));
    return result;
}

RequestViewResult RequestState::linear_a_buffer() noexcept {
    if (arena_ == nullptr) {
        return access_failure(RequestAccessError::kEmptyState);
    }
    const RequestAccessError profile_error = validate_request_memory_profile(
        memory_profile(), RequestMemoryProfile::kLegacyC512);
    if (profile_error != RequestAccessError::kNone) {
        return access_failure(profile_error);
    }
    RequestViewResult result;
    result.value.emplace(mutable_view(plan_.linear_a_bf16));
    return result;
}

RequestViewResult RequestState::linear_b_buffer() noexcept {
    if (arena_ == nullptr) {
        return access_failure(RequestAccessError::kEmptyState);
    }
    const RequestAccessError profile_error = validate_request_memory_profile(
        memory_profile(), RequestMemoryProfile::kLegacyC512);
    if (profile_error != RequestAccessError::kNone) {
        return access_failure(profile_error);
    }
    RequestViewResult result;
    result.value.emplace(mutable_view(plan_.linear_b_bf16));
    return result;
}

RequestViewResult RequestState::fp32_scratch() noexcept {
    if (arena_ == nullptr) {
        return access_failure(RequestAccessError::kEmptyState);
    }
    const RequestAccessError profile_error = validate_request_memory_profile(
        memory_profile(), RequestMemoryProfile::kLegacyC512);
    if (profile_error != RequestAccessError::kNone) {
        return access_failure(profile_error);
    }
    RequestViewResult result;
    result.value.emplace(mutable_view(plan_.fp32_scratch));
    return result;
}

RequestViewResult RequestState::gqa_probability_scratch() noexcept {
    if (arena_ == nullptr) {
        return access_failure(RequestAccessError::kEmptyState);
    }
    const RequestAccessError profile_error = validate_request_memory_profile(
        memory_profile(), RequestMemoryProfile::kLegacyC512);
    if (profile_error != RequestAccessError::kNone) {
        return access_failure(profile_error);
    }
    RequestViewResult result;
    result.value.emplace(mutable_view(plan_.gqa_probability_scratch));
    return result;
}

RequestMatrixViewResult RequestState::layer_major_prompt_residual() noexcept {
    if (arena_ == nullptr) {
        return matrix_access_failure(RequestAccessError::kEmptyState);
    }
    if (validate_request_memory_profile(
            memory_profile(), RequestMemoryProfile::kLayerMajorC8192) !=
            RequestAccessError::kNone ||
        !layer_major_plan_) {
        return matrix_access_failure(
            RequestAccessError::kMemoryProfileMismatch);
    }
    RequestMatrixViewResult result;
    result.value.emplace(
        mutable_matrix_view(layer_major_plan_->prompt_residual_bf16));
    return result;
}

RequestMatrixViewResult RequestState::layer_major_panel_token_ids() noexcept {
    if (arena_ == nullptr) {
        return matrix_access_failure(RequestAccessError::kEmptyState);
    }
    if (validate_request_memory_profile(
            memory_profile(), RequestMemoryProfile::kLayerMajorC8192) !=
            RequestAccessError::kNone ||
        !layer_major_plan_) {
        return matrix_access_failure(
            RequestAccessError::kMemoryProfileMismatch);
    }
    RequestMatrixViewResult result;
    result.value.emplace(
        mutable_matrix_view(layer_major_plan_->panel_token_ids_u32));
    return result;
}

LayerMajorGdnPhaseViewResult
RequestState::layer_major_gdn_phase_views() noexcept {
    if (arena_ == nullptr) {
        return typed_access_failure<LayerMajorGdnPhaseViews>(
            RequestAccessError::kEmptyState);
    }
    if (validate_request_memory_profile(
            memory_profile(), RequestMemoryProfile::kLayerMajorC8192) !=
            RequestAccessError::kNone ||
        !layer_major_plan_) {
        return typed_access_failure<LayerMajorGdnPhaseViews>(
            RequestAccessError::kMemoryProfileMismatch);
    }
    const LayerMajorGdnPhaseRegions& regions = layer_major_plan_->gdn;
    LayerMajorGdnPhaseViews views;
    views.qkv_bf16 = mutable_matrix_view(regions.qkv_bf16);
    views.z_bf16 = mutable_matrix_view(regions.z_bf16);
    views.a_bf16 = mutable_matrix_view(regions.a_bf16);
    views.b_bf16 = mutable_matrix_view(regions.b_bf16);
    views.recurrent_core_bf16 =
        mutable_matrix_view(regions.recurrent_core_bf16);
    views.native_c64_workspace = mutable_view(regions.native_c64_workspace);
    views.normalized_input_bf16 =
        mutable_matrix_view(regions.normalized_input_bf16);
    views.input_projection_temporary =
        mutable_view(regions.input_projection_temporary);
    views.branch_output_bf16 =
        mutable_matrix_view(regions.branch_output_bf16);
    views.output_projection_temporary =
        mutable_view(regions.output_projection_temporary);
    LayerMajorGdnPhaseViewResult result;
    result.value.emplace(std::move(views));
    return result;
}

LayerMajorAttentionPhaseViewResult
RequestState::layer_major_attention_phase_views() noexcept {
    if (arena_ == nullptr) {
        return typed_access_failure<LayerMajorAttentionPhaseViews>(
            RequestAccessError::kEmptyState);
    }
    if (validate_request_memory_profile(
            memory_profile(), RequestMemoryProfile::kLayerMajorC8192) !=
            RequestAccessError::kNone ||
        !layer_major_plan_) {
        return typed_access_failure<LayerMajorAttentionPhaseViews>(
            RequestAccessError::kMemoryProfileMismatch);
    }
    const LayerMajorAttentionPhaseRegions& regions =
        layer_major_plan_->attention;
    LayerMajorAttentionPhaseViews views;
    views.raw_q_gate_bf16 = mutable_matrix_view(regions.raw_q_gate_bf16);
    views.processed_q_bf16 = mutable_matrix_view(regions.processed_q_bf16);
    views.packed_gate_bf16 = mutable_matrix_view(regions.packed_gate_bf16);
    views.normalized_input_bf16 =
        mutable_matrix_view(regions.normalized_input_bf16);
    views.input_projection_temporary =
        mutable_view(regions.input_projection_temporary);
    views.core_output_bf16 =
        mutable_matrix_view(regions.core_output_bf16);
    views.branch_output_bf16 =
        mutable_matrix_view(regions.branch_output_bf16);
    views.output_projection_temporary =
        mutable_view(regions.output_projection_temporary);
    LayerMajorAttentionPhaseViewResult result;
    result.value.emplace(std::move(views));
    return result;
}

LayerMajorMlpPhaseViewResult
RequestState::layer_major_mlp_phase_views() noexcept {
    if (arena_ == nullptr) {
        return typed_access_failure<LayerMajorMlpPhaseViews>(
            RequestAccessError::kEmptyState);
    }
    if (validate_request_memory_profile(
            memory_profile(), RequestMemoryProfile::kLayerMajorC8192) !=
            RequestAccessError::kNone ||
        !layer_major_plan_) {
        return typed_access_failure<LayerMajorMlpPhaseViews>(
            RequestAccessError::kMemoryProfileMismatch);
    }
    const LayerMajorMlpPhaseRegions& regions = layer_major_plan_->mlp;
    LayerMajorMlpPhaseViews views;
    views.gate_bf16 = mutable_matrix_view(regions.gate_bf16);
    views.up_bf16 = mutable_matrix_view(regions.up_bf16);
    views.activated_bf16 = mutable_matrix_view(regions.activated_bf16);
    views.normalized_input_bf16 =
        mutable_matrix_view(regions.normalized_input_bf16);
    views.gate_up_projection_temporary =
        mutable_view(regions.gate_up_projection_temporary);
    views.branch_output_bf16 =
        mutable_matrix_view(regions.branch_output_bf16);
    views.down_projection_temporary =
        mutable_view(regions.down_projection_temporary);
    LayerMajorMlpPhaseViewResult result;
    result.value.emplace(std::move(views));
    return result;
}

LayerMajorLegacyC512ViewResult
RequestState::layer_major_legacy_c512_views() noexcept {
    if (arena_ == nullptr) {
        return typed_access_failure<LayerMajorLegacyC512Views>(
            RequestAccessError::kEmptyState);
    }
    if (validate_request_memory_profile(
            memory_profile(), RequestMemoryProfile::kLayerMajorC8192) !=
            RequestAccessError::kNone ||
        !layer_major_plan_) {
        return typed_access_failure<LayerMajorLegacyC512Views>(
            RequestAccessError::kMemoryProfileMismatch);
    }
    const LayerMajorLegacyC512Regions& regions =
        layer_major_plan_->legacy_c512;
    LayerMajorLegacyC512Views views;
    for (std::size_t index = 0U; index < views.hidden_bf16.size(); ++index) {
        views.hidden_bf16[index] =
            mutable_matrix_view(regions.hidden_bf16[index]);
    }
    for (std::size_t index = 0U; index < views.projection_bf16.size();
         ++index) {
        views.projection_bf16[index] =
            mutable_matrix_view(regions.projection_bf16[index]);
    }
    views.linear_a_bf16 = mutable_matrix_view(regions.linear_a_bf16);
    views.linear_b_bf16 = mutable_matrix_view(regions.linear_b_bf16);
    views.fp32_scratch = mutable_view(regions.fp32_scratch);
    views.gqa_probability_scratch =
        mutable_view(regions.gqa_probability_scratch);
    LayerMajorLegacyC512ViewResult result;
    result.value.emplace(std::move(views));
    return result;
}

RequestMatrixViewResult RequestState::layer_major_final_hidden() noexcept {
    if (arena_ == nullptr) {
        return matrix_access_failure(RequestAccessError::kEmptyState);
    }
    if (validate_request_memory_profile(
            memory_profile(), RequestMemoryProfile::kLayerMajorC8192) !=
            RequestAccessError::kNone ||
        !layer_major_plan_) {
        return matrix_access_failure(
            RequestAccessError::kMemoryProfileMismatch);
    }
    RequestMatrixViewResult result;
    result.value.emplace(
        mutable_matrix_view(layer_major_plan_->final_hidden_bf16));
    return result;
}

RequestConstViewResult RequestState::rope_cos(
    const std::size_t position) const noexcept {
    if (arena_ == nullptr) {
        return const_access_failure(RequestAccessError::kEmptyState);
    }
    if (position >= common_plan().max_sequence_length) {
        return const_access_failure(RequestAccessError::kPositionOutOfRange);
    }
    const RequestRegion region = subregion(
        common_plan().rope_cos_fp32,
        position * kRopePairs * kFp32Bytes,
        kRopePairs,
        kFp32Bytes);
    RequestConstViewResult result;
    result.value.emplace(const_view(region));
    return result;
}

RequestConstViewResult RequestState::rope_sin(
    const std::size_t position) const noexcept {
    if (arena_ == nullptr) {
        return const_access_failure(RequestAccessError::kEmptyState);
    }
    if (position >= common_plan().max_sequence_length) {
        return const_access_failure(RequestAccessError::kPositionOutOfRange);
    }
    const RequestRegion region = subregion(
        common_plan().rope_sin_fp32,
        position * kRopePairs * kFp32Bytes,
        kRopePairs,
        kFp32Bytes);
    RequestConstViewResult result;
    result.value.emplace(const_view(region));
    return result;
}

RequestConstViewResult RequestState::current_rope_cos() const noexcept {
    return rope_cos(sequence_length_);
}

RequestConstViewResult RequestState::current_rope_sin() const noexcept {
    return rope_sin(sequence_length_);
}

RequestOperationStatus RequestState::reset_async(void* const cuda_stream) noexcept {
    if (arena_ == nullptr) {
        return {RequestAccessError::kEmptyState, 0};
    }
    (void)cudaGetLastError();
    const RequestMemoryPlan& active_plan = common_plan();
    const cudaError_t status = cudaMemsetAsync(
        static_cast<std::uint8_t*>(arena_) +
            static_cast<std::size_t>(active_plan.persistent_offset),
        0,
        static_cast<std::size_t>(active_plan.persistent_bytes),
        reinterpret_cast<cudaStream_t>(cuda_stream));
    if (status != cudaSuccess) {
        return {RequestAccessError::kNone, static_cast<int>(status)};
    }
    sequence_length_ = 0U;
    return {};
}

RequestStateResult create_request_state(const RequestMemoryOptions& options) {
    RequestPlanResult plan_result = build_request_memory_plan(options);
    if (!plan_result) {
        return state_failure(std::move(plan_result.diagnostic));
    }
    try {
        const RequestMemoryPlan& plan = *plan_result.value;
        if (plan.arena_bytes >
                static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
            plan.rope_cos_fp32.element_capacity >
                static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            return state_failure(make_diagnostic(
                RequestErrorCode::kArithmeticOverflow,
                "request plan exceeds host size_t",
                "plan"));
        }
        const std::size_t rope_elements = static_cast<std::size_t>(
            plan.rope_cos_fp32.element_capacity);
        std::vector<float> cosines(rope_elements);
        std::vector<float> sines(rope_elements);
        for (std::size_t position = 0U;
             position < plan.max_sequence_length; ++position) {
            for (std::size_t pair = 0U; pair < kRopePairs; ++pair) {
                const float exponent =
                    2.0F * static_cast<float>(pair) / 64.0F;
                const float inverse_frequency =
                    1.0F / std::pow(kRopeTheta, exponent);
                const float angle =
                    static_cast<float>(position) * inverse_frequency;
                const std::size_t index = position * kRopePairs + pair;
                cosines[index] = bf16_round_trip(
                    std::cos(angle));
                sines[index] = bf16_round_trip(
                    std::sin(angle));
            }
        }

        (void)cudaGetLastError();
        std::size_t free_bytes = 0U;
        std::size_t total_bytes = 0U;
        cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (status != cudaSuccess) {
            return state_failure(cuda_diagnostic(status, "cudaMemGetInfo"));
        }
        (void)total_bytes;
        const std::uint64_t free_u64 = static_cast<std::uint64_t>(free_bytes);
        if (plan.arena_bytes > free_u64 ||
            options.min_free_bytes_after_create >
                free_u64 - plan.arena_bytes) {
            return state_failure(make_diagnostic(
                RequestErrorCode::kInsufficientDeviceMemory,
                "cudaMemGetInfo cannot satisfy request arena plus safety margin",
                "cudaMemGetInfo",
                std::to_string(plan.arena_bytes) + "+" +
                    std::to_string(options.min_free_bytes_after_create),
                std::to_string(free_u64)));
        }

        CreateResources resources;
        status = cudaMalloc(&resources.arena,
                            static_cast<std::size_t>(plan.arena_bytes));
        if (status != cudaSuccess) {
            return state_failure(cuda_diagnostic(status, "cudaMalloc(request arena)"));
        }
        status = cudaStreamCreateWithFlags(&resources.stream,
                                           cudaStreamNonBlocking);
        if (status != cudaSuccess) {
            return state_failure(cuda_diagnostic(status, "cudaStreamCreateWithFlags"));
        }
        auto* const arena = static_cast<std::uint8_t*>(resources.arena);
        status = cudaMemsetAsync(
            arena + static_cast<std::size_t>(plan.persistent_offset),
            0,
            static_cast<std::size_t>(plan.persistent_bytes),
            resources.stream);
        if (status != cudaSuccess) {
            return state_failure(cuda_diagnostic(status, "cudaMemsetAsync(persistent)"));
        }
        status = cudaMemcpyAsync(
            arena + static_cast<std::size_t>(plan.rope_cos_fp32.arena_offset),
            cosines.data(),
            static_cast<std::size_t>(plan.rope_cos_fp32.byte_size),
            cudaMemcpyHostToDevice,
            resources.stream);
        if (status != cudaSuccess) {
            return state_failure(cuda_diagnostic(status, "cudaMemcpyAsync(rope cos)"));
        }
        status = cudaMemcpyAsync(
            arena + static_cast<std::size_t>(plan.rope_sin_fp32.arena_offset),
            sines.data(),
            static_cast<std::size_t>(plan.rope_sin_fp32.byte_size),
            cudaMemcpyHostToDevice,
            resources.stream);
        if (status != cudaSuccess) {
            return state_failure(cuda_diagnostic(status, "cudaMemcpyAsync(rope sin)"));
        }
        status = cudaStreamSynchronize(resources.stream);
        if (status != cudaSuccess) {
            return state_failure(cuda_diagnostic(status, "cudaStreamSynchronize(create)"));
        }
        status = cudaStreamDestroy(resources.stream);
        resources.stream = nullptr;
        if (status != cudaSuccess) {
            return state_failure(cuda_diagnostic(status, "cudaStreamDestroy(create)"));
        }

        RequestState state;
        state.arena_ = resources.arena;
        resources.arena = nullptr;
        state.plan_ = plan;
        state.sequence_length_ = 0U;
        (void)cudaGetLastError();
        RequestStateResult result;
        result.value.emplace(std::move(state));
        return result;
    } catch (const std::bad_alloc&) {
        return state_failure(make_diagnostic(
            RequestErrorCode::kAllocationFailure,
            "host allocation failed while creating request state",
            "rope_cache"));
    } catch (const std::length_error&) {
        return state_failure(make_diagnostic(
            RequestErrorCode::kAllocationFailure,
            "host RoPE cache exceeds container limits",
            "rope_cache"));
    }
}

RequestStateResult create_layer_major_request_state(
    const LayerMajorRequestMemoryOptions& options) {
    LayerMajorRequestPlanResult plan_result =
        build_layer_major_request_memory_plan(options);
    if (!plan_result) {
        return state_failure(std::move(plan_result.diagnostic));
    }
    try {
        LayerMajorRequestMemoryPlan& layer_major_plan = *plan_result.value;
        const RequestMemoryPlan& plan = layer_major_plan.common;
        if (plan.arena_bytes >
                static_cast<std::uint64_t>(
                    std::numeric_limits<std::size_t>::max()) ||
            plan.rope_cos_fp32.element_capacity >
                static_cast<std::uint64_t>(
                    std::numeric_limits<std::size_t>::max())) {
            return state_failure(make_diagnostic(
                RequestErrorCode::kArithmeticOverflow,
                "layer-major request plan exceeds host size_t",
                "plan"));
        }
        const std::size_t rope_elements = static_cast<std::size_t>(
            plan.rope_cos_fp32.element_capacity);
        std::vector<float> cosines(rope_elements);
        std::vector<float> sines(rope_elements);
        for (std::size_t position = 0U;
             position < plan.max_sequence_length; ++position) {
            for (std::size_t pair = 0U; pair < kRopePairs; ++pair) {
                const float exponent =
                    2.0F * static_cast<float>(pair) / 64.0F;
                const float inverse_frequency =
                    1.0F / std::pow(kRopeTheta, exponent);
                const float angle =
                    static_cast<float>(position) * inverse_frequency;
                const std::size_t index = position * kRopePairs + pair;
                cosines[index] = bf16_round_trip(std::cos(angle));
                sines[index] = bf16_round_trip(std::sin(angle));
            }
        }

        (void)cudaGetLastError();
        std::size_t free_bytes = 0U;
        std::size_t total_bytes = 0U;
        cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (status != cudaSuccess) {
            return state_failure(cuda_diagnostic(status, "cudaMemGetInfo"));
        }
        (void)total_bytes;
        const std::uint64_t free_u64 = static_cast<std::uint64_t>(free_bytes);
        if (plan.arena_bytes > free_u64 ||
            options.min_free_bytes_after_create >
                free_u64 - plan.arena_bytes) {
            return state_failure(make_diagnostic(
                RequestErrorCode::kInsufficientDeviceMemory,
                "cudaMemGetInfo cannot satisfy layer-major request arena plus "
                "safety margin",
                "cudaMemGetInfo",
                std::to_string(plan.arena_bytes) + "+" +
                    std::to_string(options.min_free_bytes_after_create),
                std::to_string(free_u64)));
        }

        CreateResources resources;
        status = cudaMalloc(&resources.arena,
                            static_cast<std::size_t>(plan.arena_bytes));
        if (status != cudaSuccess) {
            return state_failure(
                cuda_diagnostic(status,
                                "cudaMalloc(layer-major request arena)"));
        }
        status = cudaStreamCreateWithFlags(&resources.stream,
                                           cudaStreamNonBlocking);
        if (status != cudaSuccess) {
            return state_failure(
                cuda_diagnostic(status, "cudaStreamCreateWithFlags"));
        }
        auto* const arena = static_cast<std::uint8_t*>(resources.arena);
        status = cudaMemsetAsync(
            arena + static_cast<std::size_t>(plan.persistent_offset),
            0,
            static_cast<std::size_t>(plan.persistent_bytes),
            resources.stream);
        if (status != cudaSuccess) {
            return state_failure(
                cuda_diagnostic(status, "cudaMemsetAsync(persistent)"));
        }
        status = cudaMemcpyAsync(
            arena + static_cast<std::size_t>(plan.rope_cos_fp32.arena_offset),
            cosines.data(),
            static_cast<std::size_t>(plan.rope_cos_fp32.byte_size),
            cudaMemcpyHostToDevice,
            resources.stream);
        if (status != cudaSuccess) {
            return state_failure(
                cuda_diagnostic(status, "cudaMemcpyAsync(rope cos)"));
        }
        status = cudaMemcpyAsync(
            arena + static_cast<std::size_t>(plan.rope_sin_fp32.arena_offset),
            sines.data(),
            static_cast<std::size_t>(plan.rope_sin_fp32.byte_size),
            cudaMemcpyHostToDevice,
            resources.stream);
        if (status != cudaSuccess) {
            return state_failure(
                cuda_diagnostic(status, "cudaMemcpyAsync(rope sin)"));
        }
        status = cudaStreamSynchronize(resources.stream);
        if (status != cudaSuccess) {
            return state_failure(
                cuda_diagnostic(status,
                                "cudaStreamSynchronize(layer-major create)"));
        }
        status = cudaStreamDestroy(resources.stream);
        resources.stream = nullptr;
        if (status != cudaSuccess) {
            return state_failure(
                cuda_diagnostic(status,
                                "cudaStreamDestroy(layer-major create)"));
        }

        RequestState state;
        state.arena_ = resources.arena;
        resources.arena = nullptr;
        state.layer_major_plan_.emplace(std::move(layer_major_plan));
        state.sequence_length_ = 0U;
        (void)cudaGetLastError();
        RequestStateResult result;
        result.value.emplace(std::move(state));
        return result;
    } catch (const std::bad_alloc&) {
        return state_failure(make_diagnostic(
            RequestErrorCode::kAllocationFailure,
            "host allocation failed while creating layer-major request state",
            "rope_cache"));
    } catch (const std::length_error&) {
        return state_failure(make_diagnostic(
            RequestErrorCode::kAllocationFailure,
            "host RoPE cache exceeds container limits",
            "rope_cache"));
    }
}

std::string_view to_string(RequestErrorCode code) noexcept {
    switch (code) {
        case RequestErrorCode::kNone:
            return "none";
        case RequestErrorCode::kInvalidOption:
            return "invalid_option";
        case RequestErrorCode::kArithmeticOverflow:
            return "arithmetic_overflow";
        case RequestErrorCode::kInvalidLayerSchedule:
            return "invalid_layer_schedule";
        case RequestErrorCode::kArenaLimitExceeded:
            return "arena_limit_exceeded";
        case RequestErrorCode::kInsufficientDeviceMemory:
            return "insufficient_device_memory";
        case RequestErrorCode::kCudaFailure:
            return "cuda_failure";
        case RequestErrorCode::kAllocationFailure:
            return "allocation_failure";
    }
    return "unknown";
}

std::string_view to_string(RequestAccessError error) noexcept {
    switch (error) {
        case RequestAccessError::kNone:
            return "none";
        case RequestAccessError::kLayerOutOfRange:
            return "layer_out_of_range";
        case RequestAccessError::kLayerTypeMismatch:
            return "layer_type_mismatch";
        case RequestAccessError::kSlotOutOfRange:
            return "slot_out_of_range";
        case RequestAccessError::kPositionOutOfRange:
            return "position_out_of_range";
        case RequestAccessError::kCapacityExceeded:
            return "capacity_exceeded";
        case RequestAccessError::kInvalidBufferIndex:
            return "invalid_buffer_index";
        case RequestAccessError::kMemoryProfileMismatch:
            return "memory_profile_mismatch";
        case RequestAccessError::kEmptyState:
            return "empty_state";
    }
    return "unknown";
}

}  // namespace q3x::runtime
