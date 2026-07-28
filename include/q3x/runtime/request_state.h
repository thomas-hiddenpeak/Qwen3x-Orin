#pragma once

#include "q3x/model/model_config.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace q3x::runtime {

inline constexpr std::uint64_t kRequestArenaAlignment = 256U;
inline constexpr std::uint64_t kDefaultRequestMaxSequenceLength = 128U;
inline constexpr std::uint64_t kAbsoluteRequestMaxSequenceLength = 262'144U;
inline constexpr std::uint32_t kDefaultRequestPrefillChunkSize = 1U;
inline constexpr std::uint32_t kMaximumRequestPrefillChunkSize = 512U;
inline constexpr std::size_t kRequestLayerCount = 64U;
inline constexpr std::size_t kRequestLinearLayerCount = 48U;
inline constexpr std::size_t kRequestFullLayerCount = 16U;
inline constexpr std::size_t kRequestHiddenBufferCount = 3U;
inline constexpr std::size_t kRequestProjectionBufferCount = 4U;
inline constexpr std::uint32_t
    kRequestNvFp4LargeMMinimumPrefillChunkSize = 512U;
inline constexpr std::uint64_t
    kRequestNvFp4LargeMMinimumSequenceLength = 512U;
inline constexpr std::uint64_t
    kRequestNvFp4LargeMWeightBf16ScratchElements = 17'408U * 5'120U;
inline constexpr std::uint64_t kRequestNvFp4LargeMWeightBf16ScratchBytes =
    kRequestNvFp4LargeMWeightBf16ScratchElements * 2U;

inline constexpr std::uint64_t kRequestConvStateBytes = 2'949'120U;
inline constexpr std::uint64_t kRequestGdnStateBytes = 75'497'472U;
inline constexpr std::uint64_t kRequestKvBytesPerToken = 65'536U;
inline constexpr std::uint64_t kDefaultRequestArenaBytes = 88'031'744U;
inline constexpr std::uint64_t kMaximumRequestArenaBytes = 17'615'978'496ULL;

enum class RequestErrorCode : std::uint8_t {
    kNone,
    kInvalidOption,
    kArithmeticOverflow,
    kInvalidLayerSchedule,
    kArenaLimitExceeded,
    kInsufficientDeviceMemory,
    kCudaFailure,
    kAllocationFailure,
};

struct RequestDiagnostic {
    RequestErrorCode code = RequestErrorCode::kNone;
    std::string message;
    std::string context;
    std::string expected;
    std::string actual;
    int cuda_error = 0;
};

struct RequestMemoryOptions {
    // The runtime is intentionally batch-one. Keeping this explicit makes an
    // accidental future batch>1 caller fail closed instead of underallocating.
    std::uint32_t batch_size = 1U;
    std::uint32_t prefill_chunk_size = kDefaultRequestPrefillChunkSize;
    std::uint64_t max_sequence_length = kDefaultRequestMaxSequenceLength;
    std::uint64_t max_arena_bytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;
    std::uint64_t min_free_bytes_after_create =
        8ULL * 1024ULL * 1024ULL * 1024ULL;
};

struct RequestRegion {
    std::uint64_t arena_offset = 0U;
    std::uint64_t byte_size = 0U;
    std::uint64_t element_capacity = 0U;
    std::uint32_t element_size_bytes = 0U;
};

struct RequestLayerSlot {
    model::LayerType type = model::LayerType::kInvalid;
    std::size_t slot = 0U;
};

enum class RequestAccessError : std::uint8_t {
    kNone,
    kLayerOutOfRange,
    kLayerTypeMismatch,
    kSlotOutOfRange,
    kPositionOutOfRange,
    kCapacityExceeded,
    kInvalidBufferIndex,
    kEmptyState,
};

struct RequestLayerSlotResult {
    std::optional<RequestLayerSlot> value;
    RequestAccessError error = RequestAccessError::kNone;

    [[nodiscard]] bool ok() const noexcept {
        return value.has_value() && error == RequestAccessError::kNone;
    }
    [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Maps the fixed Qwen3.6-27B schedule (layers 3,7,...,63 are full
// attention). required_type must be linear or full; a mismatched or out-of-
// range request is rejected rather than silently returning another slot kind.
[[nodiscard]] RequestLayerSlotResult map_request_layer(
    std::size_t layer_index,
    model::LayerType required_type) noexcept;

struct RequestMemoryPlan {
    std::uint32_t batch_size = 1U;
    std::uint32_t prefill_chunk_size = kDefaultRequestPrefillChunkSize;
    std::uint32_t max_sequence_length = 0U;
    std::uint64_t arena_bytes = 0U;
    std::uint64_t persistent_offset = 0U;
    std::uint64_t persistent_bytes = 0U;
    std::uint64_t workspace_offset = 0U;
    std::uint64_t workspace_bytes = 0U;
    std::uint64_t rope_offset = 0U;
    std::uint64_t rope_bytes = 0U;

    // Aggregate canonical BF16 persistent states. Per-layer views are derived
    // with the checked schedule mapping.
    RequestRegion conv_state;  // [48, 10240, 3]
    RequestRegion gdn_state;   // [48, 48, 128, 128]
    std::array<RequestRegion, kRequestFullLayerCount> key_cache;
    std::array<RequestRegion, kRequestFullLayerCount> value_cache;

    std::array<RequestRegion, kRequestHiddenBufferCount> hidden_bf16;
    std::array<RequestRegion, kRequestProjectionBufferCount> projection_bf16;
    RequestRegion linear_a_bf16;  // [48], independent from projection buffers
    RequestRegion linear_b_bf16;  // [48], independent from projection buffers
    RequestRegion fp32_scratch;
    // Aliases fp32_scratch. element_capacity is the logical minimum required
    // for [24, max_sequence_length] attention probabilities.
    RequestRegion gqa_probability_scratch;
    // One reusable canonical [17408, 5120] BF16 weight scratch for the exact
    // C512 NVFP4 Gate -> Up -> Down serial route. It owns workspace bytes only
    // when both the configured chunk and sequence capacity can reach 512;
    // otherwise all fields remain zero.
    RequestRegion nvfp4_large_m_weight_bf16_scratch;

    // [max_sequence_length, 32]. Values are generated in FP32, rounded to
    // BF16 RNE like the vLLM cache, then expanded back to FP32 storage for the
    // current decode-op float-pointer ABI.
    RequestRegion rope_cos_fp32;
    RequestRegion rope_sin_fp32;

    std::array<RequestLayerSlot, kRequestLayerCount> layers;
};

struct RequestPlanResult {
    std::optional<RequestMemoryPlan> value;
    RequestDiagnostic diagnostic;

    [[nodiscard]] bool ok() const noexcept {
        return value.has_value() && diagnostic.code == RequestErrorCode::kNone;
    }
    [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Pure CPU, deterministic and allocation-free except for the optional result
// object itself. Every region begins at a checked 256-byte-aligned offset.
[[nodiscard]] RequestPlanResult build_request_memory_plan(
    const RequestMemoryOptions& options = {});

struct DeviceBufferView {
    void* device_data = nullptr;
    std::uint64_t arena_offset = 0U;
    std::uint64_t byte_size = 0U;
    std::uint64_t element_capacity = 0U;
    std::uint32_t element_size_bytes = 0U;
};

struct ConstDeviceBufferView {
    const void* device_data = nullptr;
    std::uint64_t arena_offset = 0U;
    std::uint64_t byte_size = 0U;
    std::uint64_t element_capacity = 0U;
    std::uint32_t element_size_bytes = 0U;
};

struct RequestViewResult {
    std::optional<DeviceBufferView> value;
    RequestAccessError error = RequestAccessError::kNone;

    [[nodiscard]] bool ok() const noexcept {
        return value.has_value() && error == RequestAccessError::kNone;
    }
    [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

struct RequestConstViewResult {
    std::optional<ConstDeviceBufferView> value;
    RequestAccessError error = RequestAccessError::kNone;

    [[nodiscard]] bool ok() const noexcept {
        return value.has_value() && error == RequestAccessError::kNone;
    }
    [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

struct RequestOperationStatus {
    RequestAccessError error = RequestAccessError::kNone;
    int cuda_error = 0;

    [[nodiscard]] bool ok() const noexcept {
        return error == RequestAccessError::kNone && cuda_error == 0;
    }
    [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

struct RequestStateResult;

class RequestState {
  public:
    RequestState() noexcept = default;
    ~RequestState();

    RequestState(const RequestState&) = delete;
    RequestState& operator=(const RequestState&) = delete;
    RequestState(RequestState&& other) noexcept;
    RequestState& operator=(RequestState&& other) noexcept;

    [[nodiscard]] explicit operator bool() const noexcept {
        return arena_ != nullptr;
    }
    [[nodiscard]] void* arena_data() noexcept { return arena_; }
    [[nodiscard]] const void* arena_data() const noexcept { return arena_; }
    [[nodiscard]] std::uint64_t arena_bytes() const noexcept {
        return plan_.arena_bytes;
    }
    [[nodiscard]] const RequestMemoryPlan& plan() const noexcept {
        return plan_;
    }

    [[nodiscard]] std::uint32_t sequence_length() const noexcept {
        return sequence_length_;
    }
    [[nodiscard]] std::uint32_t current_position() const noexcept {
        return sequence_length_;
    }
    [[nodiscard]] std::uint32_t max_sequence_length() const noexcept {
        return plan_.max_sequence_length;
    }
    [[nodiscard]] std::uint32_t remaining_capacity() const noexcept {
        return plan_.max_sequence_length >= sequence_length_
                   ? plan_.max_sequence_length - sequence_length_
                   : 0U;
    }
    [[nodiscard]] RequestOperationStatus commit_token() noexcept;
    [[nodiscard]] RequestOperationStatus set_sequence_length(
        std::uint32_t length) noexcept;

    [[nodiscard]] RequestViewResult conv_state(std::size_t layer_index) noexcept;
    [[nodiscard]] RequestViewResult gdn_state(std::size_t layer_index) noexcept;
    [[nodiscard]] RequestViewResult key_cache(std::size_t layer_index) noexcept;
    [[nodiscard]] RequestViewResult value_cache(std::size_t layer_index) noexcept;
    [[nodiscard]] RequestViewResult key_position(std::size_t layer_index,
                                                 std::size_t position) noexcept;
    [[nodiscard]] RequestViewResult value_position(std::size_t layer_index,
                                                   std::size_t position) noexcept;

    [[nodiscard]] RequestViewResult hidden_buffer(std::size_t index) noexcept;
    [[nodiscard]] RequestViewResult projection_buffer(std::size_t index) noexcept;
    [[nodiscard]] RequestViewResult linear_a_buffer() noexcept;
    [[nodiscard]] RequestViewResult linear_b_buffer() noexcept;
    [[nodiscard]] RequestViewResult fp32_scratch() noexcept;
    [[nodiscard]] RequestViewResult gqa_probability_scratch() noexcept;
    // Returns the sole reusable large-M scratch, or null when this state's
    // chunk/sequence plan cannot execute an exact C512 tile.
    [[nodiscard]] std::uint16_t*
    nvfp4_large_m_weight_bf16_scratch() noexcept;

    [[nodiscard]] RequestConstViewResult rope_cos(std::size_t position) const noexcept;
    [[nodiscard]] RequestConstViewResult rope_sin(std::size_t position) const noexcept;
    [[nodiscard]] RequestConstViewResult current_rope_cos() const noexcept;
    [[nodiscard]] RequestConstViewResult current_rope_sin() const noexcept;

    // Enqueues zeroing of all persistent conv/GDN/KV storage and immediately
    // resets host logical length. The caller must order subsequent use on the
    // same stream (or synchronize explicitly). Workspace and RoPE are untouched.
    [[nodiscard]] RequestOperationStatus reset_async(
        void* cuda_stream = nullptr) noexcept;

  private:
    friend RequestStateResult create_request_state(
        const RequestMemoryOptions&);

    void release() noexcept;
    [[nodiscard]] DeviceBufferView mutable_view(
        const RequestRegion& region) noexcept;
    [[nodiscard]] ConstDeviceBufferView const_view(
        const RequestRegion& region) const noexcept;

    void* arena_ = nullptr;
    RequestMemoryPlan plan_;
    std::uint32_t sequence_length_ = 0U;
};

struct RequestStateResult {
    std::optional<RequestState> value;
    RequestDiagnostic diagnostic;

    [[nodiscard]] bool ok() const noexcept {
        return value.has_value() && diagnostic.code == RequestErrorCode::kNone;
    }
    [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Creates one device arena, zeros the complete persistent span, generates the
// fixed theta=1e7 / rotary_dim=64 BF16-rounded-in-FP32 cos/sin cache, copies it
// to the arena, and synchronizes initialization before returning.
[[nodiscard]] RequestStateResult create_request_state(
    const RequestMemoryOptions& options = {});

[[nodiscard]] std::string_view to_string(RequestErrorCode code) noexcept;
[[nodiscard]] std::string_view to_string(RequestAccessError error) noexcept;

}  // namespace q3x::runtime
