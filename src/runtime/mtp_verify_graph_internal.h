#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::runtime::mtp_detail {

// Source-private full-verifier prerequisite. The caller validates token IDs,
// position and capacity before upload; the stream owns this immutable capsule
// until the complete Graph and its output read have retired.
struct alignas(16) MtpVerifyDeviceInputs {
  std::uint32_t position;
  std::uint32_t token_ids[4];
};
static_assert(sizeof(MtpVerifyDeviceInputs) == 32);

// A graph family admits either first_sequence >= 512 (long) or
// 65 <= first_sequence <= 512 - rows (short). The actual device value is
// position + 1 and must not exceed maximum_first_sequence. No padding enters
// any arithmetic. Inputs/workspace/output must be disjoint and 16-byte aligned.
// The caller keeps the capsule in separate device storage; this private API
// does not read device values on the host or synchronize to validate them.
int launch_mtp_graph_attention(const std::uint16_t* query,
    const std::uint16_t* key, const std::uint16_t* value,
    const MtpVerifyDeviceInputs* parameters, unsigned maximum_first_sequence,
    unsigned rows, void* workspace, std::size_t workspace_bytes,
    std::uint16_t* output, void* stream) noexcept;

// Fixed-budget alternate-tail topology, rows 3 or 4. The final row is a
// child of seed row 0, logically at position+1, physically at position+rows-1.
// All other rows remain linear. Arithmetic sees committed prefix, seed and
// self only; the final row never reads primary sibling/descendant K/V.
// Same storage/lifetime bounds as the linear entry. Not selected by service.
int launch_mtp_tree_graph_attention(const std::uint16_t* query,
    const std::uint16_t* key, const std::uint16_t* value,
    const MtpVerifyDeviceInputs* parameters, unsigned maximum_first_sequence,
    unsigned rows, void* workspace, std::size_t workspace_bytes,
    std::uint16_t* output, void* stream) noexcept;

// Startup graph helpers. The owning transaction validates actual tokens and
// positions before upload. K/V staging is bounded by rows * 1024 elements;
// query/gate output retains the existing interleaved-row stride of 12288.
int launch_mtp_graph_embedding(const std::uint16_t* embedding,
    const MtpVerifyDeviceInputs* parameters, unsigned rows,
    std::uint16_t* output, void* stream) noexcept;
int launch_mtp_graph_preprocess(const std::uint16_t* q_gate,
    std::uint16_t* key_stage, const std::uint16_t* value_stage,
    const std::uint16_t* q_norm, const std::uint16_t* k_norm,
    const float* cosines, const float* sines,
    const MtpVerifyDeviceInputs* parameters, unsigned rows,
    std::uint16_t* query_gate, std::uint16_t* key_cache,
    std::uint16_t* value_cache, void* stream) noexcept;

// Alternate-tail RoPE uses position+1; physical cache writes stay position+row.
int launch_mtp_tree_graph_preprocess(const std::uint16_t* q_gate,
    std::uint16_t* key_stage, const std::uint16_t* value_stage,
    const std::uint16_t* q_norm, const std::uint16_t* k_norm,
    const float* cosines, const float* sines,
    const MtpVerifyDeviceInputs* parameters, unsigned rows,
    std::uint16_t* query_gate, std::uint16_t* key_cache,
    std::uint16_t* value_cache, void* stream) noexcept;

// Identifies the sole shape-dependent grid in long Attention. A graph owner
// may update gridDim.x = ceil((position + rows) / 128) before replay, preserving
// every other node parameter and the captured graph's lifetime.
bool is_mtp_graph_qk_kernel(const void* function) noexcept;

}  // namespace q3x::runtime::mtp_detail
