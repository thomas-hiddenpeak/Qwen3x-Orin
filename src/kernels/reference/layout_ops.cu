#include "q3x/runtime/layout_ops.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::runtime {
namespace {

constexpr unsigned int kThreads = 256U;
constexpr std::size_t kMaximumBlocks = 65535U;

[[nodiscard]] bool multiply_overflows(const std::size_t left,
                                      const std::size_t right) noexcept {
  return right != 0U &&
         left > std::numeric_limits<std::size_t>::max() / right;
}

[[nodiscard]] bool ranges_overlap(const void* const first,
                                  const std::size_t first_elements,
                                  const void* const second,
                                  const std::size_t second_elements) noexcept {
  if (multiply_overflows(first_elements, sizeof(std::uint16_t)) ||
      multiply_overflows(second_elements, sizeof(std::uint16_t))) {
    return true;
  }
  const auto first_begin = reinterpret_cast<std::uintptr_t>(first);
  const auto second_begin = reinterpret_cast<std::uintptr_t>(second);
  const std::size_t first_bytes = first_elements * sizeof(std::uint16_t);
  const std::size_t second_bytes = second_elements * sizeof(std::uint16_t);
  if (first_bytes > std::numeric_limits<std::uintptr_t>::max() - first_begin ||
      second_bytes >
          std::numeric_limits<std::uintptr_t>::max() - second_begin) {
    return true;
  }
  const std::uintptr_t first_end = first_begin + first_bytes;
  const std::uintptr_t second_end = second_begin + second_bytes;
  return first_begin < second_end && second_begin < first_end;
}

__global__ void split_q_gate_kernel(
    const std::uint16_t* const input, const std::size_t head_count,
    const std::size_t head_dimension,
    std::uint16_t* const query_output,
    std::uint16_t* const gate_output) {
  const std::size_t output_elements = head_count * head_dimension;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < output_elements; index += stride) {
    const std::size_t head = index / head_dimension;
    const std::size_t dimension = index - head * head_dimension;
    const std::size_t input_offset = head * 2U * head_dimension + dimension;
    query_output[index] = input[input_offset];
    gate_output[index] = input[input_offset + head_dimension];
  }
}

}  // namespace

int launch_split_interleaved_q_gate_reference_cuda(
    const std::uint16_t* const input, const std::size_t head_count,
    const std::size_t head_dimension,
    std::uint16_t* const query_output,
    std::uint16_t* const gate_output,
    void* const cuda_stream) noexcept {
  if (head_count == 0U || head_dimension == 0U) {
    return static_cast<int>(cudaSuccess);
  }
  if (multiply_overflows(head_count, head_dimension)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t output_elements = head_count * head_dimension;
  if (multiply_overflows(output_elements, 2U) || input == nullptr ||
      query_output == nullptr || gate_output == nullptr ||
      ranges_overlap(input, output_elements * 2U, query_output,
                     output_elements) ||
      ranges_overlap(input, output_elements * 2U, gate_output,
                     output_elements) ||
      ranges_overlap(query_output, output_elements, gate_output,
                     output_elements)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const std::size_t wanted_blocks =
      (output_elements + kThreads - 1U) / kThreads;
  const auto blocks = static_cast<unsigned int>(
      wanted_blocks < kMaximumBlocks ? wanted_blocks : kMaximumBlocks);
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  split_q_gate_kernel<<<blocks, kThreads, 0U, stream>>>(
      input, head_count, head_dimension, query_output, gate_output);
  return static_cast<int>(cudaGetLastError());
}

}  // namespace q3x::runtime
