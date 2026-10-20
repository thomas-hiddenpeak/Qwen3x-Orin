#include "mtp_gdn_state_internal.h"
#include "mtp_device_internal.h"
#include "mtp_verify_graph_internal.h"
#include <memory>
#include <new>
#include "q3x/runtime/decode_ops.h"
#include "q3x/runtime/gdn_decode.h"
#if defined(Q3X_ENABLE_FUSED_DECODE)
#include "decode_fused_gqa_internal.h"
#endif
#include <cuda_runtime_api.h>

namespace q3x::runtime::mtp_detail {
int launch_mtp_verify_projection(const std::uint8_t*, const std::uint8_t*, float,
    const std::uint16_t*, unsigned, unsigned, unsigned, std::uint16_t*, void*, const std::uint8_t*, const std::uint8_t*, unsigned) noexcept;
int launch_mtp_verify_attention(const std::uint16_t*, const std::uint16_t*,
    const std::uint16_t*, unsigned, unsigned, void*, std::size_t,
    std::uint16_t*, void*) noexcept;
int launch_mtp_short_attention(const std::uint16_t*, const std::uint16_t*,
    const std::uint16_t*, unsigned, unsigned, std::uint16_t*, void*) noexcept;
bool TargetTransaction::enqueue_gdn_block(unsigned first, unsigned count) noexcept {
  constexpr std::size_t H = 5120;
  auto& v = target_.views_;
  const auto& model = *target_.weights_;
  auto stream = static_cast<cudaStream_t>(target_.stream_);
  const auto ok = [](int status) { return status == cudaSuccess; };
  const auto project = [&](const LinearWeight& w, const std::uint16_t* x,
                            std::uint16_t* y) {
    if (count > 1) {
      if (const auto* f = std::get_if<Fp8LinearWeight>(&w))
        return ok(launch_mtp_verify_projection(f->weight, nullptr, f->weight_scale,
            x, count, f->output_size, f->input_size, y, stream, fp8_input_layout(*f), nullptr, 0));
      if (const auto* q = std::get_if<NvFp4LinearWeight>(&w))
        return ok(launch_mtp_verify_projection(q->packed_weight, q->block_scale, q->weight_scale_2,
            x, count, q->output_size, q->input_size, y, stream,
            q->decode_gate_up_coupled_feed_sidecar ? q->decode_gate_up_coupled_feed_sidecar :
              q->down_consumer_order_weight, q->down_scale6_sidecar, q->down_scale6_base));
    }
    return ok(launch_projection_tile_to_bf16_cuda(ProjectionBackend::kSm87WeightOnly,
        w, x, count, v.fp32_scratch, v.fp32_scratch_elements, y, stream));
  };
  const auto prefix_region = [&](unsigned row, const void* source,
                                 std::size_t bytes) -> void* {
    const auto offset = static_cast<const char*>(source) -
                        static_cast<const char*>(state_.arena_data());
    if (row >= count || offset < 0 || std::uint64_t(offset) < recurrent_offset_ ||
        std::uint64_t(offset) + bytes > recurrent_offset_ + recurrent_bytes_) return nullptr;
    return static_cast<char*>(snapshots_) + (row + 1) * slot_bytes_ +
           offset - recurrent_offset_;
  };
  const auto save_region = [&](unsigned row, const void* source, std::size_t bytes) {
    void* destination = prefix_region(row, source, bytes);
    return destination && ok(cudaMemcpyAsync(destination, source, bytes,
                                             cudaMemcpyDeviceToDevice, stream));
  };
  for (unsigned layer = first; layer < first + 3; ++layer) {
    const auto& w = model.layer(layer);
    const auto* a = std::get_if<LinearAttentionWeights>(&w.attention);
    if (!a) return false;
    if (!project(a->in_proj_qkv, v.hidden[1], v.projection[0]) ||
        !project(a->in_proj_z, v.hidden[1], v.projection[1])) return false;
    // Preserve the incumbent direct BF16 A/B pair arithmetic per row.
    for (unsigned row = 0; row < count; ++row)
      if (!ok(launch_projection_pair_tile_to_bf16_cuda(ProjectionBackend::kSm87WeightOnly,
            a->in_proj_a, a->in_proj_b, v.hidden[1] + row * H, 1,
            v.fp32_scratch, v.fp32_scratch_elements, v.linear_a + row * 48,
            v.linear_b + row * 48, stream))) return false;
    for (unsigned row = 0; row < count; ++row) {
      constexpr std::size_t state_bytes = 48ULL * 128 * 128 * 2;
      auto* state_output = static_cast<std::uint16_t*>(
          prefix_region(row, static_cast<char*>(state_.arena_data()) + state_.plan().gdn_state.arena_offset + (layer - layer / 4) * state_bytes, state_bytes));
      const auto* state_input = row ? static_cast<const std::uint16_t*>(
          prefix_region(row - 1, static_cast<char*>(state_.arena_data()) + state_.plan().gdn_state.arena_offset + (layer - layer / 4) * state_bytes, state_bytes)) : v.gdn_state[layer];
      if (!state_input || !state_output) return false;
      if (!ok(launch_causal_conv1d_silu_update_reference_cuda(v.projection[0] + row * 10240,
            a->conv1d.data, v.conv_state[layer], v.projection[0] + row * 10240, {}, stream)) ||
          !ok(
#if defined(Q3X_ENABLE_MTP_GDN_OWNER)
            gdn_inputs_ && row == 0 ? launch_mtp_gdn_selected_state(
              v.projection[0] + row * 10240, v.linear_a + row * 48, v.linear_b + row * 48,
              a->a_log.data, a->dt_bias.data, gdn_inputs_ + layer - layer / 4,
              state_output, a->norm.data, v.projection[1] + row * 6144,
              v.projection[2] + row * 6144, stream) :
#endif
            launch_gated_delta_net_update_plain_rms_norm_silu_gate_cuda(
            v.projection[0] + row * 10240, v.linear_a + row * 48, v.linear_b + row * 48,
            a->a_log.data, a->dt_bias.data, state_input, state_output,
            1.e-6F, a->norm.data, v.projection[1] + row * 6144, 48, 128, 1.e-6F,
            v.projection[2] + row * 6144, {}, stream)) ||
          !save_region(row, v.conv_state[layer], 10240 * 3 * 2)) return false;
    }
    if (!project(a->out_proj, v.projection[2], v.hidden[1])) return false;
    // One CTA per independent row preserves the original 256-lane tree
    // and BF16 residual publication without a cooperative grid per row.
    if (!ok(launch_residual_add_headwise_centered_rms_norm_prefill_5120_cuda(
          v.hidden[0], v.hidden[1], w.post_attention_layernorm.data,
          count, H, 1.e-6F, v.hidden[2], v.hidden[1], stream))) return false;
    if (!project(w.mlp.gate_proj, v.hidden[1], v.projection[0]) ||
        !project(w.mlp.up_proj, v.hidden[1], v.projection[1]) ||
        !ok(launch_silu_mul_reference_cuda(v.projection[0], v.projection[1],
              count * 17408, v.projection[0], stream)) ||
        !project(w.mlp.down_proj, v.projection[0], v.projection[1])) return false;
    const auto* next_norm = layer == 63 ? model.final_norm().data :
                                         model.layer(layer + 1).input_layernorm.data;
    if (!ok(launch_residual_add_headwise_centered_rms_norm_prefill_5120_cuda(
          v.hidden[2], v.projection[1], next_norm, count, H, 1.e-6F,
          v.hidden[0], v.hidden[1], stream))) return false;
  }
  return true;
}

void TargetTransaction::release_gdn_graphs() noexcept {
  bool any = false;
  for (auto handle : gdn_graphs_) any |= handle != nullptr;
  if (any) (void)cudaStreamSynchronize(static_cast<cudaStream_t>(target_.stream_));
  for (auto& handle : gdn_graphs_) {
    if (handle) (void)cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(handle));
    handle = nullptr;
  }
}

bool TargetTransaction::prepare_gdn_graphs() noexcept {
  auto stream = static_cast<cudaStream_t>(target_.stream_);
  if (cudaStreamSynchronize(stream) != cudaSuccess) return false;
  const auto& model = *target_.weights_;
  for (unsigned group = 0; group < 16; ++group) {
    for (unsigned j = 0; j < 3; ++j)
      if (!std::holds_alternative<LinearAttentionWeights>(model.layer(group * 4 + j).attention))
        return false;
    if (!std::holds_alternative<FullAttentionWeights>(model.layer(group * 4 + 3).attention))
      return false;
  }
  for (unsigned count = 2; count <= 4; ++count) {
    for (unsigned group = 0; group < 16; ++group) {
      if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        release_gdn_graphs(); return false;
      }
      const bool recorded = enqueue_gdn_block(group * 4, count);
      cudaGraph_t graph = nullptr;
      const auto end_status = cudaStreamEndCapture(stream, &graph);
      if (!recorded || end_status != cudaSuccess || !graph) {
        if (graph) (void)cudaGraphDestroy(graph);
        release_gdn_graphs(); return false;
      }
      cudaGraphExec_t executable = nullptr;
      const auto status = cudaGraphInstantiateWithFlags(&executable, graph, 0);
      (void)cudaGraphDestroy(graph);
      if (status != cudaSuccess) {
        if (executable) (void)cudaGraphExecDestroy(executable);
        release_gdn_graphs(); return false;
      }
      gdn_graphs_[(count - 2) * 16 + group] = executable;
      if (cudaGraphUpload(executable, stream) != cudaSuccess) {
        release_gdn_graphs(); return false;
      }
    }
  }
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    release_gdn_graphs(); return false;
  }
  return true;
}

bool TargetTransaction::enqueue_multirow(bool full_graph, bool short_graph) noexcept {
  constexpr std::size_t H = 5120, V = 248320;
  auto& v = target_.views_;
  const auto& model = *target_.weights_;
  auto stream = static_cast<cudaStream_t>(target_.stream_);
  const auto ok = [](int s) { return s == cudaSuccess; };
  const auto project = [&](const LinearWeight& w, const std::uint16_t* x,
                            std::uint16_t* y) {
    if (rows_ > 1) {
      if (const auto* f = std::get_if<Fp8LinearWeight>(&w))
        return ok(launch_mtp_verify_projection(f->weight, nullptr, f->weight_scale,
            x, rows_, f->output_size, f->input_size, y, stream, fp8_input_layout(*f), nullptr, 0));
      if (const auto* q = std::get_if<NvFp4LinearWeight>(&w))
        return ok(launch_mtp_verify_projection(q->packed_weight, q->block_scale, q->weight_scale_2,
            x, rows_, q->output_size, q->input_size, y, stream,
            q->decode_gate_up_coupled_feed_sidecar ? q->decode_gate_up_coupled_feed_sidecar :
              q->down_consumer_order_weight, q->down_scale6_sidecar, q->down_scale6_base));
    }
    return ok(launch_projection_tile_to_bf16_cuda(ProjectionBackend::kSm87WeightOnly,
        w, x, rows_, v.fp32_scratch, v.fp32_scratch_elements, y, stream));
  };
  const auto norm = [&](const std::uint16_t* x, const std::uint16_t* w,
                       std::uint16_t* y) {
    return ok(launch_headwise_centered_rms_norm_reference_cuda(
        x, w, rows_, H, 1.e-6F, y, stream));
  };
  // Each prefix slot is assembled layer by layer. No slot is visible until
  // every layer, normalized hidden and full logits have completed.
  const auto prefix_region = [&](unsigned row, const void* source,
                                 std::size_t bytes) -> void* {
    const auto offset = static_cast<const char*>(source) -
                        static_cast<const char*>(state_.arena_data());
    if (row >= rows_ || offset < 0 || std::uint64_t(offset) < recurrent_offset_ ||
        std::uint64_t(offset) + bytes > recurrent_offset_ + recurrent_bytes_) return nullptr;
    return static_cast<char*>(snapshots_) + (row + 1) * slot_bytes_ +
           offset - recurrent_offset_;
  };
  const auto save_region = [&](unsigned row, const void* source, std::size_t bytes) {
    void* destination = prefix_region(row, source, bytes);
    return destination && ok(cudaMemcpyAsync(destination, source, bytes,
                                             cudaMemcpyDeviceToDevice, stream));
  };
  if (full_graph) {
    if (!ok(launch_mtp_graph_embedding(model.embed_tokens().weight,
          static_cast<const MtpVerifyDeviceInputs*>(graph_inputs_), rows_,
          v.hidden[0], stream))) return false;
  } else {
    for (unsigned row = 0; row < rows_; ++row)
      if (!ok(launch_embedding_gather_reference_cuda(model.embed_tokens().weight,
            V, H, inputs_[row], v.hidden[0] + row * H, stream))) return false;
  }
  if (!norm(v.hidden[0], model.layer(0).input_layernorm.data, v.hidden[1])) return false;
  for (unsigned layer = 0; layer < 64; ++layer) {
    if (rows_ > 1 && layer % 4 == 0) {
      if (full_graph) {
        if (!enqueue_gdn_block(layer, rows_)) return false;
      } else {
        const auto graph = static_cast<cudaGraphExec_t>(gdn_graphs_[(rows_ - 2) * 16 + layer / 4]);
        if (!graph || !ok(cudaGraphLaunch(graph, stream))) return false;
      }
      layer += 2;
      continue;
    }
    const auto& w = model.layer(layer);
    if (const auto* a = std::get_if<LinearAttentionWeights>(&w.attention)) {
      if (!project(a->in_proj_qkv, v.hidden[1], v.projection[0]) ||
          !project(a->in_proj_z, v.hidden[1], v.projection[1])) return false;
      // Preserve the incumbent direct BF16 A/B pair arithmetic per row.
      for (unsigned row = 0; row < rows_; ++row)
        if (!ok(launch_projection_pair_tile_to_bf16_cuda(ProjectionBackend::kSm87WeightOnly,
              a->in_proj_a, a->in_proj_b, v.hidden[1] + row * H, 1,
              v.fp32_scratch, v.fp32_scratch_elements, v.linear_a + row * 48,
              v.linear_b + row * 48, stream))) return false;
      for (unsigned row = 0; row < rows_; ++row) {
        constexpr std::size_t state_bytes = 48ULL * 128 * 128 * 2;
        auto* state_output = static_cast<std::uint16_t*>(
            prefix_region(row, static_cast<char*>(state_.arena_data()) + state_.plan().gdn_state.arena_offset + (layer - layer / 4) * state_bytes, state_bytes));
        const auto* state_input = row ? static_cast<const std::uint16_t*>(
            prefix_region(row - 1, static_cast<char*>(state_.arena_data()) + state_.plan().gdn_state.arena_offset + (layer - layer / 4) * state_bytes, state_bytes)) : v.gdn_state[layer];
        if (!state_input || !state_output) return false;
        if (!ok(launch_causal_conv1d_silu_update_reference_cuda(v.projection[0] + row * 10240,
              a->conv1d.data, v.conv_state[layer], v.projection[0] + row * 10240, {}, stream)) ||
            !ok(
#if defined(Q3X_ENABLE_MTP_GDN_OWNER)
            gdn_inputs_ && row == 0 ? launch_mtp_gdn_selected_state(
              v.projection[0] + row * 10240, v.linear_a + row * 48, v.linear_b + row * 48,
              a->a_log.data, a->dt_bias.data, gdn_inputs_ + layer - layer / 4,
              state_output, a->norm.data, v.projection[1] + row * 6144,
              v.projection[2] + row * 6144, stream) :
#endif
            launch_gated_delta_net_update_plain_rms_norm_silu_gate_cuda(
              v.projection[0] + row * 10240, v.linear_a + row * 48, v.linear_b + row * 48,
              a->a_log.data, a->dt_bias.data, state_input, state_output,
              1.e-6F, a->norm.data, v.projection[1] + row * 6144, 48, 128, 1.e-6F,
              v.projection[2] + row * 6144, {}, stream)) ||
            !save_region(row, v.conv_state[layer], 10240 * 3 * 2)) return false;
      }
      if (!project(a->out_proj, v.projection[2], v.hidden[1])) return false;
    } else if (const auto* a = std::get_if<FullAttentionWeights>(&w.attention)) {
      auto* key = full_graph ? v.projection[1] :
          v.key_cache[layer] + entry_position_ * 1024ULL;
      auto* value = full_graph ? v.projection[1] + rows_ * 1024ULL :
          v.value_cache[layer] + entry_position_ * 1024ULL;
      if (!project(a->q_proj, v.hidden[1], v.projection[0]) ||
          !project(a->k_proj, v.hidden[1], key) ||
          !project(a->v_proj, v.hidden[1], value)) return false;
      if (full_graph) {
        if (!ok(launch_mtp_graph_preprocess(v.projection[0], key, value,
              a->q_norm.data, a->k_norm.data, v.rope_cos, v.rope_sin,
              static_cast<const MtpVerifyDeviceInputs*>(graph_inputs_), rows_,
              v.projection[3], v.key_cache[layer], v.value_cache[layer], stream))) return false;
      } else {
      for (unsigned row = 0; row < rows_; ++row) {
        auto* query = v.projection[3] + row * 12288;
        auto* gate = query + 6144;
        const auto position = entry_position_ + row;
        if (!ok(launch_full_attention_preprocess_24_4_256_64_cuda(
              v.projection[0] + row * 12288, key + row * 1024,
              a->q_norm.data, a->k_norm.data, 1.e-6F, query, gate,
              v.rope_cos, v.rope_sin, position, 1, stream))) return false;
      }
      }
      // Projection 2 is dead throughout full Attention. Bound the probability
      // lifetime to this phase; small scratch plans retain scalar-row dispatch.
      const std::size_t probability_bytes = rows_ * 24ULL *
          ((entry_position_ + rows_ + 3ULL) & ~3ULL) * sizeof(float);
      const std::size_t available_bytes =
          state_.plan().prefill_chunk_size * 17408ULL * sizeof(std::uint16_t);
      const bool batch_attention = full_graph ? !short_graph :
          (rows_ > 1 && entry_position_ + 1 >= 512 && probability_bytes <= available_bytes);
      if (full_graph && !ok(launch_mtp_graph_attention(v.projection[3],
            v.key_cache[layer], v.value_cache[layer],
            static_cast<const MtpVerifyDeviceInputs*>(graph_inputs_),
            short_graph ? 512 - rows_ : state_.max_sequence_length() - rows_ + 1,
            rows_, v.projection[2], available_bytes, v.projection[1], stream))) return false;
      if (!full_graph && batch_attention && !ok(launch_mtp_verify_attention(v.projection[3],
            v.key_cache[layer], v.value_cache[layer], entry_position_ + 1, rows_,
            v.projection[2], available_bytes, v.projection[1], stream))) return false;
      const bool short_attention = full_graph ? short_graph :
          (rows_ > 1 && entry_position_ + 1 >= 65 && entry_position_ + rows_ <= 511);
      if (!full_graph && short_attention && !ok(launch_mtp_short_attention(v.projection[3],
            v.key_cache[layer], v.value_cache[layer], entry_position_ + 1, rows_,
            v.projection[1], stream))) return false;
      for (unsigned row = 0; row < rows_; ++row) {
        auto* query = v.projection[3] + row * 12288;
        auto* gate = query + 6144;
        if (short_attention) continue;
        auto* out = v.projection[1] + row * 6144;
        const auto position = entry_position_ + row;
        if (batch_attention) {
          if (!ok(launch_sigmoid_gate_reference_cuda(out, gate, 6144, out, stream))) return false;
          continue;
        }
        if (position < 64) {
          if (!ok(launch_gqa_attention_sigmoid_gate_24_4_256_cuda(query,
                v.key_cache[layer], v.value_cache[layer], position + 1, .0625F,
                v.fp32_scratch, v.fp32_scratch_elements, gate, out, stream))) return false;
        } else {
#if defined(Q3X_ENABLE_FUSED_DECODE)
          if (position + 1 >= fused_decode::kMinimumSequence) {
            if (!ok(fused_decode::launch(query, v.key_cache[layer], v.value_cache[layer],
                  position + 1, v.fp32_scratch, v.fp32_scratch_elements * sizeof(float),
                  out, stream))) return false;
          } else
#endif
          if (!ok(launch_gqa_attention_reference_cuda(query, v.key_cache[layer],
                v.value_cache[layer], 24, 4, position + 1, 256, .0625F,
                v.fp32_scratch, v.fp32_scratch_elements, out, stream))) return false;
          if (!ok(launch_sigmoid_gate_reference_cuda(out, gate, 6144, out, stream))) return false;
        }
      }
      if (!project(a->o_proj, v.projection[1], v.hidden[1])) return false;
    } else return false;
    // One CTA per independent row preserves the original 256-lane tree
    // and BF16 residual publication without a cooperative grid per row.
    if (!ok(launch_residual_add_headwise_centered_rms_norm_prefill_5120_cuda(
          v.hidden[0], v.hidden[1], w.post_attention_layernorm.data,
          rows_, H, 1.e-6F, v.hidden[2], v.hidden[1], stream))) return false;
    if (!project(w.mlp.gate_proj, v.hidden[1], v.projection[0]) ||
        !project(w.mlp.up_proj, v.hidden[1], v.projection[1]) ||
        !ok(launch_silu_mul_reference_cuda(v.projection[0], v.projection[1],
              rows_ * 17408, v.projection[0], stream)) ||
        !project(w.mlp.down_proj, v.projection[0], v.projection[1])) return false;
    const auto* next_norm = layer == 63 ? model.final_norm().data :
                                         model.layer(layer + 1).input_layernorm.data;
    if (!ok(launch_residual_add_headwise_centered_rms_norm_prefill_5120_cuda(
          v.hidden[2], v.projection[1], next_norm, rows_, H, 1.e-6F,
          v.hidden[0], v.hidden[1], stream))) return false;
  }
  // Projection 0 is dead after the final layer. Service-sized C512 storage can
  // stage all vocabulary rows; smaller oracle arenas keep the scalar boundary.
  const auto* head = std::get_if<NvFp4LinearWeight>(&model.lm_head());
  const bool batch_logits = rows_>1 && head && head->input_size==H &&
      head->output_size==V && state_.plan().prefill_chunk_size*17408ULL>=rows_*V;
  if(batch_logits && !ok(launch_mtp_verify_projection(head->packed_weight,
      head->block_scale,head->weight_scale_2,v.hidden[1],rows_,V,H,
      v.projection[0],stream,nullptr,nullptr,0)))return false;
  auto* argmax = reinterpret_cast<Bf16GreedyArgmaxResult*>(v.fp32_scratch);
  static_assert(kBf16GreedyArgmaxWorkspaceResults * sizeof(*argmax) <= 512);
  auto* result_rows = reinterpret_cast<Bf16GreedyArgmaxResult*>(
      reinterpret_cast<char*>(v.fp32_scratch) + 512);
  for (unsigned row = 0; row < rows_; ++row) {
    auto* slot = static_cast<char*>(snapshots_) + (row + 1) * slot_bytes_ + recurrent_bytes_;
    auto* logits = reinterpret_cast<std::uint16_t*>(slot + 2 * H);
    if (!ok(cudaMemcpyAsync(slot, v.hidden[1] + row * H, 2 * H,
              cudaMemcpyDeviceToDevice, stream)) ||
        !(batch_logits ? ok(cudaMemcpyAsync(logits,v.projection[0]+row*V,2*V,
              cudaMemcpyDeviceToDevice,stream)) :
          ok(launch_projection_to_bf16_cuda(ProjectionBackend::kSm87WeightOnly,
              model.lm_head(),v.hidden[1]+row*H,nullptr,0,logits,stream))) ||
        !ok(launch_bf16_greedy_argmax_cuda(logits, V, argmax, stream)) ||
        !ok(cudaMemcpyAsync(result_rows + row, argmax, sizeof(*argmax),
              cudaMemcpyDeviceToDevice, stream))) return false;
  }
  return true;
}
}  // namespace q3x::runtime::mtp_detail

namespace q3x::runtime::mtp_detail {
namespace {
struct FullVerifierGraph {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  std::array<cudaGraphNode_t, 16> qk_nodes{};
  std::array<cudaKernelNodeParams, 16> qk_parameters{};
};
struct FullVerifierGraphs { std::array<FullVerifierGraph, 6> entries{}; };
}

void TargetTransaction::release_full_graphs() noexcept {
  if (full_graphs_ || graph_inputs_ || graph_host_)
    (void)cudaStreamSynchronize(static_cast<cudaStream_t>(target_.stream_));
  auto* owner = static_cast<FullVerifierGraphs*>(full_graphs_);
  if (owner) {
    for (auto& entry : owner->entries) {
      if (entry.executable) (void)cudaGraphExecDestroy(entry.executable);
      if (entry.graph) (void)cudaGraphDestroy(entry.graph);
    }
    delete owner;
  }
  if (graph_inputs_) (void)cudaFree(graph_inputs_);
  if (graph_host_) (void)cudaFreeHost(graph_host_);
  full_graphs_ = graph_inputs_ = graph_host_ = nullptr;
}

bool TargetTransaction::prepare_full_graphs() noexcept {
  const std::size_t available = state_.plan().prefill_chunk_size * 17408ULL * 2;
  const auto capacity = state_.max_sequence_length();
  if (capacity < 512 || available < 4 * 24ULL * ((capacity + 3ULL) & ~3ULL) * 4 ||
      available < 4 * 248320ULL * 2) return true;  // Explicit small-arena route.
  auto stream = static_cast<cudaStream_t>(target_.stream_);
  if (cudaStreamSynchronize(stream) != cudaSuccess) return false;
  auto* owner = new (std::nothrow) FullVerifierGraphs;
  if (!owner) return false;
  full_graphs_ = owner;
  const auto saved_rows = rows_;
  const auto fail = [&]() noexcept {
    rows_ = saved_rows; release_full_graphs(); return false;
  };
  static_assert(sizeof(MtpVerifyDeviceInputs) == 32);
  static_assert(4 * sizeof(Bf16GreedyArgmaxResult) == 32);
  if (cudaMalloc(&graph_inputs_, 32) != cudaSuccess ||
      cudaHostAlloc(&graph_host_, 64, cudaHostAllocDefault) != cudaSuccess) return fail();
  for (unsigned count = 2; count <= 4; ++count) {
    rows_ = count;
    for (unsigned family = 0; family < 2; ++family) {
      auto& entry = owner->entries[(count - 2) * 2 + family];
      if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess)
        return fail();
      const bool recorded = enqueue_multirow(true, family == 0);
      const auto end_status = cudaStreamEndCapture(stream, &entry.graph);
      if (!recorded || end_status != cudaSuccess || !entry.graph) return fail();
      if (cudaGraphInstantiateWithFlags(&entry.executable, entry.graph, 0) != cudaSuccess)
        return fail();
      std::size_t node_count = 0;
      if (cudaGraphGetNodes(entry.graph, nullptr, &node_count) != cudaSuccess) return fail();
      std::unique_ptr<cudaGraphNode_t[]> nodes(new (std::nothrow) cudaGraphNode_t[node_count]);
      if (!nodes || cudaGraphGetNodes(entry.graph, nodes.get(), &node_count) != cudaSuccess)
        return fail();
      unsigned qk_count = 0;
      for (std::size_t i = 0; i < node_count; ++i) {
        cudaGraphNodeType type;
        if (cudaGraphNodeGetType(nodes[i], &type) != cudaSuccess) return fail();
        if (type != cudaGraphNodeTypeKernel) continue;
        cudaKernelNodeParams parameters{};
        if (cudaGraphKernelNodeGetParams(nodes[i], &parameters) != cudaSuccess) return fail();
        if (!is_mtp_graph_qk_kernel(parameters.func)) continue;
        if (qk_count >= 16) return fail();
        entry.qk_nodes[qk_count] = nodes[i];
        entry.qk_parameters[qk_count++] = parameters;
      }
      if (qk_count != (family ? 16U : 0U) ||
          cudaGraphUpload(entry.executable, stream) != cudaSuccess) return fail();
    }
  }
  rows_ = saved_rows;
  if (cudaStreamSynchronize(stream) != cudaSuccess) return fail();
  return true;
}

bool TargetTransaction::verify_multirow(std::uint32_t* predictions) noexcept {
  constexpr unsigned V = 248320;
  if (!active_ || verified_ || !predictions || rows_ < 1 || rows_ > 4 ||
      state_.current_position() != entry_position_ ||
      std::uint64_t(entry_position_) + rows_ > state_.max_sequence_length()) return false;
  for (unsigned row = 0; row < rows_; ++row) if (inputs_[row] >= V) return false;
  auto stream = static_cast<cudaStream_t>(target_.stream_);
  bool drained = false;
  struct DrainOnFailure {
    cudaStream_t stream;
    bool& drained;
    ~DrainOnFailure() { if (!drained) (void)cudaStreamSynchronize(stream); }
  } drain{stream, drained};
#if defined(Q3X_ENABLE_MTP_GDN_OWNER)
  if (!publish_gdn_inputs()) return false;
#endif
  const bool short_graph = entry_position_ + 1 >= 65 && entry_position_ + rows_ <= 511;
  const bool use_graph = full_graphs_ && rows_ >= 2 &&
      (short_graph || entry_position_ + 1 >= 512);
  std::array<Bf16GreedyArgmaxResult, 4> local_results{};
  auto* results = local_results.data();
  if (use_graph) {
    auto& entry = static_cast<FullVerifierGraphs*>(full_graphs_)->entries[
        (rows_ - 2) * 2 + (short_graph ? 0 : 1)];
    if (!entry.executable || !graph_inputs_ || !graph_host_) return false;
    auto* input = static_cast<MtpVerifyDeviceInputs*>(graph_host_);
    *input = {};
    input->position = entry_position_;
    for (unsigned row = 0; row < rows_; ++row) input->token_ids[row] = inputs_[row];
    results = reinterpret_cast<Bf16GreedyArgmaxResult*>(static_cast<char*>(graph_host_) + 32);
    if (!short_graph) {
      for (unsigned i = 0; i < 16; ++i) {
        entry.qk_parameters[i].gridDim.x = (entry_position_ + rows_ + 127) / 128;
        if (cudaGraphExecKernelNodeSetParams(entry.executable, entry.qk_nodes[i],
              &entry.qk_parameters[i]) != cudaSuccess) return false;
      }
    }
    if (cudaMemcpyAsync(graph_inputs_, input, sizeof(*input), cudaMemcpyHostToDevice,
          stream) != cudaSuccess || cudaGraphLaunch(entry.executable, stream) != cudaSuccess)
      return false;
  } else if (!enqueue_multirow(false, false)) return false;
  if (cudaMemcpyAsync(results, reinterpret_cast<char*>(target_.views_.fp32_scratch) + 512,
        rows_ * sizeof(*results), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) return false;
  drained = true;
  for (unsigned row = 0; row < rows_; ++row) {
    if (results[row].has_nonfinite || results[row].index >= V) return false;
    predictions[row] = predictions_[row] = results[row].index;
  }
  direct_gdn_rows_ += rows_ * 48ULL;
  if (use_graph) {
    ++full_graph_rounds_; full_graph_rows_ += rows_;
    if (!short_graph) graph_qk_updates_ += 16;
  }
  return true;
}
}  // namespace q3x::runtime::mtp_detail
