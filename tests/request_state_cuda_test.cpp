#include "q3x/runtime/request_state.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace runtime = q3x::runtime;

class TestContext {
  public:
    void expect(bool condition, const char* message) {
        if (!condition) {
            std::cerr << "FAILED: " << message << '\n';
            ++failures_;
        }
    }
    [[nodiscard]] int failures() const noexcept { return failures_; }

  private:
    int failures_ = 0;
};

void print_diagnostic(const runtime::RequestDiagnostic& diagnostic) {
    std::cerr << "request diagnostic: code="
              << runtime::to_string(diagnostic.code)
              << " message=" << diagnostic.message
              << " context=" << diagnostic.context
              << " expected=" << diagnostic.expected
              << " actual=" << diagnostic.actual
              << " cuda=" << diagnostic.cuda_error << '\n';
}

float bf16_round_trip(float value) {
    std::uint32_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    bits &= 0xFFFF0000U;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

bool pointer_matches(const runtime::RequestState& state,
                     const runtime::DeviceBufferView& view) {
    const auto base = reinterpret_cast<std::uintptr_t>(state.arena_data());
    const auto pointer = reinterpret_cast<std::uintptr_t>(view.device_data);
    return pointer == base + view.arena_offset &&
           view.arena_offset <= state.arena_bytes() &&
           view.byte_size <= state.arena_bytes() - view.arena_offset &&
           (view.arena_offset % runtime::kRequestArenaAlignment) == 0U;
}

bool sample_zero(const runtime::DeviceBufferView& view) {
    const std::size_t count = static_cast<std::size_t>(
        std::min<std::uint64_t>(view.byte_size, 64U));
    std::array<std::uint8_t, 64U> bytes{};
    if (cudaMemcpy(bytes.data(),
                   view.device_data,
                   count,
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    for (std::size_t index = 0U; index < count; ++index) {
        if (bytes[index] != 0U) {
            return false;
        }
    }
    return true;
}

void test_create_views_rope_and_reset(TestContext& test) {
    runtime::RequestMemoryOptions options;
    options.max_sequence_length = 4U;
    options.max_arena_bytes = 128U * 1024U * 1024U;
    options.min_free_bytes_after_create = 0U;

    (void)cudaMemcpy(nullptr, nullptr, 1U, cudaMemcpyHostToDevice);
    test.expect(cudaPeekAtLastError() != cudaSuccess,
                "test injected a stale CUDA last error before create");
    runtime::RequestStateResult created = runtime::create_request_state(options);
    if (!created) {
        print_diagnostic(created.diagnostic);
    }
    test.expect(created.ok(), "small true-CUDA request state creates successfully");
    if (!created) {
        return;
    }
    runtime::RequestState& state = *created.value;
    const runtime::RequestPlanResult expected_plan =
        runtime::build_request_memory_plan(options);
    test.expect(expected_plan && state.arena_data() != nullptr &&
                    state.arena_bytes() == expected_plan.value->arena_bytes &&
                    state.sequence_length() == 0U &&
                    state.current_position() == 0U &&
                    state.remaining_capacity() == 4U,
                "request owns one exact arena with zero logical length");
    test.expect((reinterpret_cast<std::uintptr_t>(state.arena_data()) % 256U) == 0U,
                "cudaMalloc request arena is at least 256-byte aligned");

    auto conv0 = state.conv_state(0U);
    auto conv_last = state.conv_state(62U);
    auto gdn0 = state.gdn_state(0U);
    auto key0 = state.key_cache(3U);
    auto value_last = state.value_cache(63U);
    test.expect(conv0 && conv0.value->element_capacity == 30'720U &&
                    conv0.value->byte_size == 61'440U &&
                    pointer_matches(state, *conv0.value) &&
                    conv_last && pointer_matches(state, *conv_last.value),
                "linear conv layer views have exact shape and arena pointers");
    test.expect(gdn0 && gdn0.value->element_capacity == 786'432U &&
                    gdn0.value->byte_size == 1'572'864U &&
                    pointer_matches(state, *gdn0.value),
                "linear GDN layer view has canonical [48,128,128] BF16 size");
    test.expect(key0 && key0.value->element_capacity == 4'096U &&
                    key0.value->byte_size == 8'192U &&
                    pointer_matches(state, *key0.value) &&
                    value_last && pointer_matches(state, *value_last.value),
                "full-attention K/V views have [4,4,256] BF16 capacity");
    test.expect(!state.conv_state(3U) &&
                    state.conv_state(3U).error ==
                        runtime::RequestAccessError::kLayerTypeMismatch &&
                    !state.key_cache(0U) &&
                    state.key_cache(0U).error ==
                        runtime::RequestAccessError::kLayerTypeMismatch &&
                    !state.gdn_state(64U) &&
                    state.gdn_state(64U).error ==
                        runtime::RequestAccessError::kLayerOutOfRange,
                "state accessors reject wrong layer type and range");

    auto key_position = state.key_position(3U, 1U);
    test.expect(key_position && key_position.value->element_capacity == 1'024U &&
                    key_position.value->byte_size == 2'048U &&
                    key_position.value->arena_offset ==
                        key0.value->arena_offset + 2'048U,
                "per-position K view advances by one [4,256] BF16 row");
    test.expect(!state.value_position(3U, 4U) &&
                    state.value_position(3U, 4U).error ==
                        runtime::RequestAccessError::kPositionOutOfRange,
                "K/V position at capacity is rejected");

    auto hidden0 = state.hidden_buffer(0U);
    auto projection3 = state.projection_buffer(3U);
    auto linear_a = state.linear_a_buffer();
    auto linear_b = state.linear_b_buffer();
    auto fp32 = state.fp32_scratch();
    auto probabilities = state.gqa_probability_scratch();
    test.expect(hidden0 && hidden0.value->element_capacity == 5'120U &&
                    projection3 &&
                    projection3.value->element_capacity == 17'408U &&
                    linear_a && linear_b &&
                    linear_a.value->element_capacity == 48U &&
                    linear_b.value->element_capacity == 48U &&
                    linear_a.value->device_data != linear_b.value->device_data,
                "hidden/projection and independent linear a/b views are exact");
    test.expect(fp32 && fp32.value->element_capacity == 262'144U &&
                    probabilities &&
                    probabilities.value->element_capacity == 96U &&
                    fp32.value->device_data == probabilities.value->device_data,
                "small GQA scratch explicitly aliases the 262144 FP32 scratch");
    test.expect(!state.hidden_buffer(3U) &&
                    state.hidden_buffer(3U).error ==
                        runtime::RequestAccessError::kInvalidBufferIndex &&
                    !state.projection_buffer(4U),
                "workspace buffer indices are bounded");
    test.expect(
        !state.layer_major_prompt_residual() &&
            state.layer_major_prompt_residual().error ==
                runtime::RequestAccessError::kMemoryProfileMismatch &&
            !state.layer_major_gdn_phase_views() &&
            state.layer_major_gdn_phase_views().error ==
                runtime::RequestAccessError::kMemoryProfileMismatch,
        "legacy state rejects candidate-only typed workspace access");

    test.expect(sample_zero(*conv0.value) && sample_zero(*conv_last.value) &&
                    sample_zero(*gdn0.value) && sample_zero(*key0.value) &&
                    sample_zero(*value_last.value),
                "new request persistent conv/GDN/KV samples are zero");

    for (std::size_t position : {0U, 1U, 3U}) {
        auto cosine_view = state.rope_cos(position);
        auto sine_view = state.rope_sin(position);
        test.expect(cosine_view && sine_view &&
                        cosine_view.value->element_capacity == 32U &&
                        sine_view.value->element_capacity == 32U,
                    "RoPE position exposes 32 cosine and sine values");
        if (!cosine_view || !sine_view) {
            continue;
        }
        std::array<float, 32U> cosines{};
        std::array<float, 32U> sines{};
        const cudaError_t cos_status = cudaMemcpy(
            cosines.data(), cosine_view.value->device_data,
            sizeof(cosines), cudaMemcpyDeviceToHost);
        const cudaError_t sin_status = cudaMemcpy(
            sines.data(), sine_view.value->device_data,
            sizeof(sines), cudaMemcpyDeviceToHost);
        bool exact = cos_status == cudaSuccess && sin_status == cudaSuccess;
        for (std::size_t pair = 0U; exact && pair < 32U; ++pair) {
            const float exponent =
                2.0F * static_cast<float>(pair) / 64.0F;
            const float inverse_frequency =
                1.0F / std::pow(10'000'000.0F, exponent);
            const float angle = static_cast<float>(position) * inverse_frequency;
            const float expected_cos = bf16_round_trip(
                std::cos(angle));
            const float expected_sin = bf16_round_trip(
                std::sin(angle));
            exact = cosines[pair] == expected_cos && sines[pair] == expected_sin;
        }
        test.expect(exact, "device RoPE matches FP32-generate/BF16-RNE policy exactly");
        if (position == 0U) {
            test.expect(cosines[0] == 1.0F && sines[0] == 0.0F,
                        "RoPE position zero is exact one/zero");
        }
        if (position == 1U) {
            const float raw = static_cast<float>(std::cos(1.0));
            test.expect(cosines[0] == bf16_round_trip(raw) && cosines[0] != raw,
                        "nonzero RoPE value is observably BF16-rounded");
            test.expect(cosines[0] == 0.5390625F &&
                            sines[0] == 0.83984375F &&
                            cosines[1] == 0.82421875F &&
                            sines[1] == 0.56640625F &&
                            sines[31] == 1.6577541828155518e-7F,
                        "position-one RoPE matches fixed PyTorch/vLLM BF16 values");
        }
        if (position == 3U) {
            test.expect(cosines[0] == -0.98828125F &&
                            sines[0] == 0.1416015625F &&
                            cosines[1] == -0.2392578125F &&
                            sines[1] == 0.97265625F &&
                            sines[31] == 4.9546360969543457e-7F,
                        "position-three RoPE matches fixed PyTorch/vLLM BF16 values");
        }
    }
    test.expect(!state.rope_cos(4U) &&
                    state.rope_cos(4U).error ==
                        runtime::RequestAccessError::kPositionOutOfRange,
                "RoPE lookup at max capacity is rejected");

    test.expect(state.commit_token() && state.commit_token() &&
                    state.sequence_length() == 2U &&
                    state.remaining_capacity() == 2U &&
                    state.current_rope_cos(),
                "commit_token advances bounded host position without allocation");
    test.expect(state.set_sequence_length(4U) &&
                    !state.commit_token() &&
                    !state.current_rope_cos() &&
                    !state.set_sequence_length(5U),
                "logical length cannot advance beyond capacity");

    cudaStream_t stream = nullptr;
    test.expect(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) ==
                    cudaSuccess,
                "reset test stream creates");
    if (stream != nullptr) {
        (void)cudaMemsetAsync(conv0.value->device_data,
                              0xAB,
                              static_cast<std::size_t>(conv0.value->byte_size),
                              stream);
        (void)cudaMemsetAsync(gdn0.value->device_data,
                              0xAB,
                              64U,
                              stream);
        (void)cudaMemsetAsync(key0.value->device_data,
                              0xAB,
                              static_cast<std::size_t>(key0.value->byte_size),
                              stream);
        (void)cudaMemsetAsync(value_last.value->device_data,
                              0xAB,
                              64U,
                              stream);
        (void)cudaMemsetAsync(hidden0.value->device_data, 0xCD, 64U, stream);
        (void)cudaMemcpy(nullptr, nullptr, 1U, cudaMemcpyHostToDevice);
        test.expect(cudaPeekAtLastError() != cudaSuccess,
                    "test injected stale CUDA error before reset");
        const runtime::RequestOperationStatus reset =
            state.reset_async(reinterpret_cast<void*>(stream));
        test.expect(reset && state.sequence_length() == 0U,
                    "reset enqueue clears stale error and immediately resets length");
        const cudaError_t sync_status = cudaStreamSynchronize(stream);
        test.expect(sync_status == cudaSuccess && sample_zero(*conv0.value) &&
                        sample_zero(*gdn0.value) && sample_zero(*key0.value) &&
                        sample_zero(*value_last.value),
                    "async reset zeros the entire persistent span");
        std::array<std::uint8_t, 64U> workspace_bytes{};
        const cudaError_t workspace_status = cudaMemcpy(
            workspace_bytes.data(), hidden0.value->device_data,
            workspace_bytes.size(), cudaMemcpyDeviceToHost);
        bool workspace_untouched = workspace_status == cudaSuccess;
        for (const std::uint8_t byte : workspace_bytes) {
            workspace_untouched = workspace_untouched && byte == 0xCDU;
        }
        test.expect(workspace_untouched,
                    "reset leaves reusable workspace untouched");
        (void)cudaStreamDestroy(stream);
    }

    runtime::RequestState moved;
    moved = std::move(state);
    test.expect(static_cast<bool>(moved) && !static_cast<bool>(state) &&
                    moved.conv_state(0U) &&
                    !state.conv_state(0U) &&
                    state.conv_state(0U).error ==
                        runtime::RequestAccessError::kEmptyState,
                "request state move transfers sole arena ownership");
    created.value.reset();
}

void test_memory_gate(TestContext& test) {
    runtime::RequestMemoryOptions options;
    options.max_sequence_length = 1U;
    options.max_arena_bytes = 128U * 1024U * 1024U;
    options.min_free_bytes_after_create =
        std::numeric_limits<std::uint64_t>::max();
    const runtime::RequestStateResult result =
        runtime::create_request_state(options);
    test.expect(!result && result.diagnostic.code ==
                               runtime::RequestErrorCode::kInsufficientDeviceMemory,
                "cudaMemGetInfo safety margin fails before arena allocation");
}

void test_layer_major_create_views_and_profile_gate(TestContext& test) {
    const char* const enabled =
        std::getenv("Q3X_RUN_LAYER_MAJOR_REQUEST_STATE_CUDA");
    if (enabled == nullptr || std::strcmp(enabled, "1") != 0) {
        std::cout << "SKIP: layer-major RequestState CUDA allocation is "
                     "opt-in (~1 GiB)\n";
        return;
    }

    runtime::LayerMajorRequestMemoryOptions options;
    options.max_sequence_length = 1U;
    options.min_free_bytes_after_create = 256U * 1024U * 1024U;
    const runtime::LayerMajorRequestPlanResult expected =
        runtime::build_layer_major_request_memory_plan(options);
    test.expect(static_cast<bool>(expected),
                "minimum layer-major CUDA plan builds before allocation");
    if (!expected) {
        return;
    }

    std::size_t free_bytes = 0U;
    std::size_t total_bytes = 0U;
    const cudaError_t memory_status =
        cudaMemGetInfo(&free_bytes, &total_bytes);
    (void)total_bytes;
    if (memory_status != cudaSuccess ||
        expected.value->common.arena_bytes > free_bytes ||
        options.min_free_bytes_after_create >
            free_bytes - expected.value->common.arena_bytes) {
        std::cout << "SKIP: insufficient free device memory for opt-in "
                     "layer-major RequestState allocation\n";
        (void)cudaGetLastError();
        return;
    }

    runtime::RequestStateResult created =
        runtime::create_layer_major_request_state(options);
    if (!created && created.diagnostic.code ==
                        runtime::RequestErrorCode::kInsufficientDeviceMemory) {
        std::cout << "SKIP: layer-major RequestState lost its free-memory "
                     "reservation before cudaMalloc\n";
        return;
    }
    if (!created) {
        print_diagnostic(created.diagnostic);
    }
    test.expect(created.ok(),
                "opt-in layer-major CUDA request state creates successfully");
    if (!created) {
        return;
    }

    runtime::RequestState& state = *created.value;
    test.expect(
        state.memory_profile() ==
                runtime::RequestMemoryProfile::kLayerMajorC8192 &&
            state.layer_major_plan() != nullptr &&
            state.plan().arena_bytes == expected.value->common.arena_bytes &&
            state.layer_major_plan()->c8192_family_phase_arena.arena_offset ==
                expected.value->c8192_family_phase_arena.arena_offset,
        "RequestState preserves the complete candidate plan without slicing");
    test.expect(
        !state.hidden_buffer(0U) &&
            state.hidden_buffer(0U).error ==
                runtime::RequestAccessError::kMemoryProfileMismatch &&
            !state.projection_buffer(0U) &&
            state.projection_buffer(0U).error ==
                runtime::RequestAccessError::kMemoryProfileMismatch &&
            !state.linear_a_buffer() &&
            state.linear_a_buffer().error ==
                runtime::RequestAccessError::kMemoryProfileMismatch &&
            !state.fp32_scratch() &&
            state.fp32_scratch().error ==
                runtime::RequestAccessError::kMemoryProfileMismatch,
        "flat legacy workspace access fails closed on the layer-major profile");

    const auto residual = state.layer_major_prompt_residual();
    const auto token_ids = state.layer_major_panel_token_ids();
    const auto gdn = state.layer_major_gdn_phase_views();
    const auto attention = state.layer_major_attention_phase_views();
    const auto mlp = state.layer_major_mlp_phase_views();
    const auto legacy = state.layer_major_legacy_c512_views();
    const auto final_hidden = state.layer_major_final_hidden();
    test.expect(
        residual && pointer_matches(state, residual.value->storage) &&
            token_ids && pointer_matches(state, token_ids.value->storage) &&
            gdn && pointer_matches(state, gdn.value->qkv_bf16.storage) &&
            pointer_matches(state, gdn.value->native_c64_workspace) &&
            attention &&
            pointer_matches(state, attention.value->raw_q_gate_bf16.storage) &&
            mlp && pointer_matches(state, mlp.value->gate_bf16.storage) &&
            legacy &&
            pointer_matches(state, legacy.value->hidden_bf16[0U].storage) &&
            final_hidden &&
            pointer_matches(state, final_hidden.value->storage),
        "all layer-major typed device views resolve inside the owned arena");
    test.expect(
        gdn && attention && mlp && legacy &&
            gdn.value->normalized_input_bf16.storage.device_data ==
                gdn.value->recurrent_core_bf16.storage.device_data &&
            attention.value->normalized_input_bf16.storage.device_data ==
                attention.value->processed_q_bf16.storage.device_data &&
            attention.value->core_output_bf16.storage.device_data !=
                attention.value->branch_output_bf16.storage.device_data &&
            mlp.value->normalized_input_bf16.storage.device_data ==
                mlp.value->activated_bf16.storage.device_data &&
            mlp.value->branch_output_bf16.storage.device_data !=
                mlp.value->activated_bf16.storage.device_data &&
            legacy.value->gqa_probability_scratch.device_data ==
                legacy.value->fp32_scratch.device_data,
        "typed views expose only the declared aliases and keep Down output distinct");

    test.expect(state.commit_token() && !state.commit_token() &&
                    state.sequence_length() == 1U &&
                    !state.set_sequence_length(2U),
                "layer-major state retains common sequence bounds");
    runtime::RequestState moved = std::move(state);
    test.expect(
        moved && !state && moved.layer_major_plan() != nullptr &&
            moved.layer_major_final_hidden() &&
            !state.layer_major_final_hidden() &&
            state.layer_major_final_hidden().error ==
                runtime::RequestAccessError::kEmptyState,
        "move transfers complete layer-major ownership and empties the source");
}

}  // namespace

int main() {
    (void)cudaGetLastError();
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0) {
        std::cout << "SKIP: CUDA device unavailable\n";
        (void)cudaGetLastError();
        return 77;
    }
    TestContext test;
    test_create_views_rope_and_reset(test);
    test_memory_gate(test);
    test_layer_major_create_views_and_profile_gate(test);
    if (test.failures() != 0) {
        std::cerr << test.failures() << " request-state CUDA test(s) failed\n";
        return 1;
    }
    std::cout << "All request-state CUDA tests passed\n";
    return 0;
}
