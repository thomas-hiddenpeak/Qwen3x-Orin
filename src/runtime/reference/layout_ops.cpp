#include "q3x/runtime/layout_ops.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace q3x::runtime {
namespace {

[[nodiscard]] bool multiply_overflows(const std::size_t left,
                                      const std::size_t right) noexcept {
  return right != 0U &&
         left > std::numeric_limits<std::size_t>::max() / right;
}

[[nodiscard]] bool byte_range(const void* const pointer,
                              const std::size_t elements,
                              std::uintptr_t& begin,
                              std::uintptr_t& end) noexcept {
  if (multiply_overflows(elements, sizeof(std::uint16_t))) {
    return false;
  }
  begin = reinterpret_cast<std::uintptr_t>(pointer);
  const std::size_t bytes = elements * sizeof(std::uint16_t);
  if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
    return false;
  }
  end = begin + bytes;
  return true;
}

[[nodiscard]] bool overlaps(const std::uintptr_t first_begin,
                            const std::uintptr_t first_end,
                            const std::uintptr_t second_begin,
                            const std::uintptr_t second_end) noexcept {
  return first_begin < second_end && second_begin < first_end;
}

[[nodiscard]] LayoutOpStatus validate(
    const std::uint16_t* const input, const std::size_t head_count,
    const std::size_t head_dimension,
    std::uint16_t* const query_output,
    std::uint16_t* const gate_output) noexcept {
  if (head_count == 0U || head_dimension == 0U) {
    return LayoutOpStatus::kSuccess;
  }
  if (multiply_overflows(head_count, head_dimension)) {
    return LayoutOpStatus::kSizeOverflow;
  }
  const std::size_t output_elements = head_count * head_dimension;
  if (multiply_overflows(output_elements, 2U)) {
    return LayoutOpStatus::kSizeOverflow;
  }
  if (input == nullptr || query_output == nullptr || gate_output == nullptr) {
    return LayoutOpStatus::kInvalidArgument;
  }

  std::uintptr_t input_begin = 0U;
  std::uintptr_t input_end = 0U;
  std::uintptr_t query_begin = 0U;
  std::uintptr_t query_end = 0U;
  std::uintptr_t gate_begin = 0U;
  std::uintptr_t gate_end = 0U;
  if (!byte_range(input, output_elements * 2U, input_begin, input_end) ||
      !byte_range(query_output, output_elements, query_begin, query_end) ||
      !byte_range(gate_output, output_elements, gate_begin, gate_end)) {
    return LayoutOpStatus::kSizeOverflow;
  }
  if (overlaps(input_begin, input_end, query_begin, query_end) ||
      overlaps(input_begin, input_end, gate_begin, gate_end) ||
      overlaps(query_begin, query_end, gate_begin, gate_end)) {
    return LayoutOpStatus::kOverlappingStorage;
  }
  return LayoutOpStatus::kSuccess;
}

}  // namespace

const char* layout_op_status_string(const LayoutOpStatus status) noexcept {
  switch (status) {
    case LayoutOpStatus::kSuccess:
      return "success";
    case LayoutOpStatus::kInvalidArgument:
      return "invalid argument";
    case LayoutOpStatus::kSizeOverflow:
      return "tensor size overflow";
    case LayoutOpStatus::kOverlappingStorage:
      return "input and output storage overlap";
  }
  return "unknown layout-op status";
}

LayoutOpStatus split_interleaved_q_gate_reference_cpu(
    const std::uint16_t* const input, const std::size_t head_count,
    const std::size_t head_dimension,
    std::uint16_t* const query_output,
    std::uint16_t* const gate_output) noexcept {
  const LayoutOpStatus status =
      validate(input, head_count, head_dimension, query_output, gate_output);
  if (status != LayoutOpStatus::kSuccess || head_count == 0U ||
      head_dimension == 0U) {
    return status;
  }
  for (std::size_t head = 0; head < head_count; ++head) {
    const std::size_t input_offset = head * 2U * head_dimension;
    const std::size_t output_offset = head * head_dimension;
    std::copy_n(input + input_offset, head_dimension,
                query_output + output_offset);
    std::copy_n(input + input_offset + head_dimension, head_dimension,
                gate_output + output_offset);
  }
  return LayoutOpStatus::kSuccess;
}

}  // namespace q3x::runtime
