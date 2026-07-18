#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::runtime {

enum class LayoutOpStatus : std::uint8_t {
  kSuccess = 0,
  kInvalidArgument,
  kSizeOverflow,
  kOverlappingStorage,
};

[[nodiscard]] const char* layout_op_status_string(
    LayoutOpStatus status) noexcept;

// Split a row laid out as
//
//   [q_head_0, gate_head_0, q_head_1, gate_head_1, ...]
//
// into two contiguous [head_count, head_dimension] arrays. This is the exact
// q_proj ABI used by Qwen3.6 full-attention layers. Input, query_output, and
// gate_output must be pairwise non-overlapping. BF16 values are copied as raw
// bits; this operation performs no arithmetic or conversion.
[[nodiscard]] LayoutOpStatus split_interleaved_q_gate_reference_cpu(
    const std::uint16_t* input, std::size_t head_count,
    std::size_t head_dimension, std::uint16_t* query_output,
    std::uint16_t* gate_output) noexcept;

// Asynchronous CUDA counterpart. All pointers must identify device-accessible
// storage. The launch allocates, copies, and synchronizes nothing. cuda_stream
// is a cudaStream_t represented as void*, with nullptr selecting the legacy
// default stream. Invalid host-visible arguments return cudaErrorInvalidValue.
[[nodiscard]] int launch_split_interleaved_q_gate_reference_cuda(
    const std::uint16_t* input, std::size_t head_count,
    std::size_t head_dimension, std::uint16_t* query_output,
    std::uint16_t* gate_output, void* cuda_stream = nullptr) noexcept;

}  // namespace q3x::runtime
