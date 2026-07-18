#include "q3x/runtime/layout_ops.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>

int main() {
  int failures = 0;
  const auto expect = [&failures](const bool condition, const char* message) {
    if (!condition) {
      ++failures;
      std::cerr << "FAIL: " << message << '\n';
    }
  };

  constexpr std::size_t kHeads = 3U;
  constexpr std::size_t kDimension = 5U;
  std::array<std::uint16_t, kHeads * 2U * kDimension> input{};
  std::array<std::uint16_t, kHeads * kDimension> query{};
  std::array<std::uint16_t, kHeads * kDimension> gate{};
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<std::uint16_t>(index + 1U);
  }
  expect(q3x::runtime::split_interleaved_q_gate_reference_cpu(
             input.data(), kHeads, kDimension, query.data(), gate.data()) ==
             q3x::runtime::LayoutOpStatus::kSuccess,
         "split succeeds");
  for (std::size_t head = 0; head < kHeads; ++head) {
    for (std::size_t dimension = 0; dimension < kDimension; ++dimension) {
      const std::size_t output_index = head * kDimension + dimension;
      const std::size_t input_index = head * 2U * kDimension + dimension;
      expect(query[output_index] == input[input_index], "query bit copy");
      expect(gate[output_index] == input[input_index + kDimension],
             "gate bit copy");
    }
  }

  expect(q3x::runtime::split_interleaved_q_gate_reference_cpu(
             nullptr, 0U, kDimension, nullptr, nullptr) ==
             q3x::runtime::LayoutOpStatus::kSuccess,
         "empty split is a no-op");
  expect(q3x::runtime::split_interleaved_q_gate_reference_cpu(
             input.data(), kHeads, kDimension,
             const_cast<std::uint16_t*>(input.data()), gate.data()) ==
             q3x::runtime::LayoutOpStatus::kOverlappingStorage,
         "input/output overlap is rejected");
  expect(q3x::runtime::split_interleaved_q_gate_reference_cpu(
             input.data(), kHeads, kDimension, query.data(), query.data()) ==
             q3x::runtime::LayoutOpStatus::kOverlappingStorage,
         "query/gate overlap is rejected");
  expect(q3x::runtime::split_interleaved_q_gate_reference_cpu(
             input.data(), std::numeric_limits<std::size_t>::max(), 2U,
             query.data(), gate.data()) ==
             q3x::runtime::LayoutOpStatus::kSizeOverflow,
         "shape overflow is rejected");
  expect(q3x::runtime::split_interleaved_q_gate_reference_cpu(
             nullptr, 1U, 1U, query.data(), gate.data()) ==
             q3x::runtime::LayoutOpStatus::kInvalidArgument,
         "null input is rejected");

  if (failures != 0) {
    return 1;
  }
  std::cout << "layout ops host tests passed\n";
  return 0;
}
