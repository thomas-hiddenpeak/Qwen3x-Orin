#include "q3x/runtime/request_state.h"

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
constexpr std::uint64_t kFp32MinimumElements = 248'320U;
constexpr std::uint64_t kQueryHeadCount = 24U;
constexpr std::uint64_t kKvHeadCount = 4U;
constexpr std::uint64_t kHeadDimension = 256U;
constexpr std::uint64_t kConvChannels = 10'240U;
constexpr std::uint64_t kConvHistory = 3U;
constexpr std::uint64_t kGdnValueHeads = 48U;
constexpr std::uint64_t kGdnHeadDimension = 128U;
constexpr std::uint64_t kRopePairs = 32U;
constexpr float kRopeTheta = 10'000'000.0F;
constexpr std::uint64_t kMaximumConfigurableArenaBytes =
    64ULL * 1024ULL * 1024ULL * 1024ULL;

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
                          kGdnHeadDimension,
                          gdn_elements) ||
        !checked_multiply(gdn_elements,
                          kGdnHeadDimension,
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
    if (options.prefill_chunk_size >=
            kRequestNvFp4LargeMMinimumPrefillChunkSize &&
        options.max_sequence_length >=
            kRequestNvFp4LargeMMinimumSequenceLength &&
        !builder.add(kRequestNvFp4LargeMWeightBf16ScratchElements,
                     kBf16Bytes,
                     plan.nvfp4_large_m_weight_bf16_scratch)) {
        return plan_failure(make_diagnostic(
            RequestErrorCode::kArithmeticOverflow,
            "NVFP4 large-M BF16 scratch layout overflows uint64",
            "nvfp4_large_m_weight_bf16_scratch"));
    }
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

RequestState::~RequestState() { release(); }

RequestState::RequestState(RequestState&& other) noexcept
    : arena_(std::exchange(other.arena_, nullptr)),
      plan_(std::exchange(other.plan_, {})),
      sequence_length_(std::exchange(other.sequence_length_, 0U)) {}

RequestState& RequestState::operator=(RequestState&& other) noexcept {
    if (this != &other) {
        release();
        arena_ = std::exchange(other.arena_, nullptr);
        plan_ = std::exchange(other.plan_, {});
        sequence_length_ = std::exchange(other.sequence_length_, 0U);
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
    if (sequence_length_ >= plan_.max_sequence_length) {
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
    if (length > plan_.max_sequence_length) {
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
        plan_.conv_state,
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
        kGdnValueHeads * kGdnHeadDimension * kGdnHeadDimension;
    const RequestRegion region = subregion(
        plan_.gdn_state,
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
    result.value.emplace(mutable_view(plan_.key_cache[mapped.value->slot]));
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
    result.value.emplace(mutable_view(plan_.value_cache[mapped.value->slot]));
    return result;
}

RequestViewResult RequestState::key_position(const std::size_t layer_index,
                                             const std::size_t position) noexcept {
    RequestViewResult cache = key_cache(layer_index);
    if (!cache) {
        return cache;
    }
    if (position >= plan_.max_sequence_length) {
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
    if (position >= plan_.max_sequence_length) {
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
    RequestViewResult result;
    result.value.emplace(mutable_view(plan_.linear_a_bf16));
    return result;
}

RequestViewResult RequestState::linear_b_buffer() noexcept {
    if (arena_ == nullptr) {
        return access_failure(RequestAccessError::kEmptyState);
    }
    RequestViewResult result;
    result.value.emplace(mutable_view(plan_.linear_b_bf16));
    return result;
}

RequestViewResult RequestState::fp32_scratch() noexcept {
    if (arena_ == nullptr) {
        return access_failure(RequestAccessError::kEmptyState);
    }
    RequestViewResult result;
    result.value.emplace(mutable_view(plan_.fp32_scratch));
    return result;
}

RequestViewResult RequestState::gqa_probability_scratch() noexcept {
    if (arena_ == nullptr) {
        return access_failure(RequestAccessError::kEmptyState);
    }
    RequestViewResult result;
    result.value.emplace(mutable_view(plan_.gqa_probability_scratch));
    return result;
}

std::uint16_t* RequestState::nvfp4_large_m_weight_bf16_scratch() noexcept {
    const RequestRegion& region =
        plan_.nvfp4_large_m_weight_bf16_scratch;
    if (arena_ == nullptr ||
        region.byte_size != kRequestNvFp4LargeMWeightBf16ScratchBytes ||
        region.element_capacity !=
            kRequestNvFp4LargeMWeightBf16ScratchElements ||
        region.element_size_bytes != kBf16Bytes ||
        (region.arena_offset % kRequestArenaAlignment) != 0U ||
        region.arena_offset > plan_.arena_bytes ||
        region.byte_size > plan_.arena_bytes - region.arena_offset) {
        return nullptr;
    }
    return reinterpret_cast<std::uint16_t*>(
        static_cast<std::uint8_t*>(arena_) +
        static_cast<std::size_t>(region.arena_offset));
}

RequestConstViewResult RequestState::rope_cos(
    const std::size_t position) const noexcept {
    if (arena_ == nullptr) {
        return const_access_failure(RequestAccessError::kEmptyState);
    }
    if (position >= plan_.max_sequence_length) {
        return const_access_failure(RequestAccessError::kPositionOutOfRange);
    }
    const RequestRegion region = subregion(
        plan_.rope_cos_fp32,
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
    if (position >= plan_.max_sequence_length) {
        return const_access_failure(RequestAccessError::kPositionOutOfRange);
    }
    const RequestRegion region = subregion(
        plan_.rope_sin_fp32,
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
    const cudaError_t status = cudaMemsetAsync(
        static_cast<std::uint8_t*>(arena_) +
            static_cast<std::size_t>(plan_.persistent_offset),
        0,
        static_cast<std::size_t>(plan_.persistent_bytes),
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
        case RequestAccessError::kEmptyState:
            return "empty_state";
    }
    return "unknown";
}

}  // namespace q3x::runtime
