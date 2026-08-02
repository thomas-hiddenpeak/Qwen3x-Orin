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
inline constexpr std::size_t kRequestLongPrefillHiddenBufferCount = 2U;
inline constexpr std::size_t kRequestA4PrefillScaleGroupSize = 64U;
inline constexpr std::uint64_t kRequestA4GateUpCtaScratchBytes =
    16ULL * 64ULL * 1024ULL;
inline constexpr std::uint32_t kRequestLongPrefillProjectionSpanAlignment =
    kMaximumRequestPrefillChunkSize;
inline constexpr std::uint32_t kRequestLongPrefillAdmissionMaximumTokens =
    40'960U;
inline constexpr std::uint64_t kRequestLongPrefillPrimaryWidth = 12'288U;
inline constexpr std::uint64_t kRequestLongPrefillSecondaryWidth = 6'144U;
inline constexpr std::uint64_t kRequestConvStateBytes = 2'949'120U;
inline constexpr std::uint64_t kRequestGdnStateBytes = 75'497'472U;
inline constexpr std::uint64_t kRequestKvBytesPerToken = 65'536U;
inline constexpr std::uint64_t kDefaultRequestArenaBytes = 88'087'040U;
inline constexpr std::uint64_t kMaximumRequestArenaBytes = 17'437'720'576ULL;

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
    // Test-only layer-major Prefill admission. Zero preserves the established
    // tile-major arena. A nonzero value reserves two full-length BF16 hidden
    // slabs while the existing C512 projection workspace remains shared.
    std::uint32_t long_prefill_token_capacity = 0U;
    // Optional whole-M projection span. Zero preserves the established
    // layer-major C512 workspace byte-for-byte. A nonzero value requires the
    // long-Prefill slabs above, is measured in tokens, and must be a C512
    // multiple no larger than the full hidden capacity. The final logical
    // span may be a shorter tail when P itself is not C512-aligned.
    std::uint32_t long_prefill_projection_span_capacity = 0U;
    // Gated full-model A4 Prefill admission. Reserves one packed hidden input
    // and one packed intermediate/attention-output input plus their BF16 K64
    // scales. Decode and ordinary Prefill retain the byte-identical arena when
    // this is false.
    bool enable_a4_prefill_workspace = false;
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
    std::uint32_t long_prefill_token_capacity = 0U;
    std::uint32_t long_prefill_projection_span_capacity = 0U;
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
    // Optional full-prompt BF16 ping-pong slabs for the layer-major admission.
    // The regular hidden/projection regions above remain C512 scratch and are
    // reused by every layer/tile instead of scaling projection storage by P.
    std::array<RequestRegion, kRequestLongPrefillHiddenBufferCount>
        long_prefill_hidden_bf16;
    // Optional whole-M projection outputs. They are independent of the four
    // legacy [C,17408] projection buffers and only exist when an explicit
    // projection span capacity S is admitted.
    RequestRegion long_prefill_projection_primary_bf16;    // [S, 12288]
    RequestRegion long_prefill_projection_secondary_bf16;  // [S, 6144]
    std::array<RequestRegion, kRequestProjectionBufferCount> projection_bf16;
    // These use T=C on the legacy route and T=S on the explicit whole-M
    // route. The intermediate allocation also backs the smaller [T,6144]
    // attention-output input because their lifetimes are disjoint.
    RequestRegion prefill_a4_hidden_packed;       // [T, 5120/2] U8
    RequestRegion prefill_a4_hidden_scales_bf16;  // [T, 5120/64]
    RequestRegion prefill_a4_intermediate_packed;  // [T, 17408/2] U8
    RequestRegion prefill_a4_intermediate_scales_bf16;  // [T, 17408/64]
    // Stable per-request backing for at most 16 paired-Gate CTAs. It is
    // reserved with every A4 Prefill workspace so baseline and candidate use
    // the same arena and CUDA Graph capture never observes a lazy allocation.
    RequestRegion prefill_a4_gateup_cta_scratch;  // [16, 64 KiB] U8
    RequestRegion linear_a_bf16;  // [48], independent from projection buffers
    RequestRegion linear_b_bf16;  // [48], independent from projection buffers
    RequestRegion fp32_scratch;
    // Aliases fp32_scratch. element_capacity is the logical minimum required
    // for [24, max_sequence_length] attention probabilities.
    RequestRegion gqa_probability_scratch;
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
    [[nodiscard]] RequestViewResult long_prefill_hidden_buffer(
        std::size_t index) noexcept;
    [[nodiscard]] RequestViewResult
    long_prefill_projection_primary_buffer() noexcept;
    [[nodiscard]] RequestViewResult
    long_prefill_projection_secondary_buffer() noexcept;
    [[nodiscard]] RequestViewResult projection_buffer(std::size_t index) noexcept;
    [[nodiscard]] RequestViewResult prefill_a4_hidden_packed() noexcept;
    [[nodiscard]] RequestViewResult prefill_a4_hidden_scales() noexcept;
    [[nodiscard]] RequestViewResult prefill_a4_intermediate_packed() noexcept;
    [[nodiscard]] RequestViewResult prefill_a4_intermediate_scales() noexcept;
    [[nodiscard]] RequestViewResult prefill_a4_gateup_cta_scratch() noexcept;
    [[nodiscard]] RequestViewResult linear_a_buffer() noexcept;
    [[nodiscard]] RequestViewResult linear_b_buffer() noexcept;
    [[nodiscard]] RequestViewResult fp32_scratch() noexcept;
    [[nodiscard]] RequestViewResult gqa_probability_scratch() noexcept;
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
