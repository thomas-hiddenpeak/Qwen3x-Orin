#include "sha256_internal.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

volatile std::uint32_t sink = 0U;

template <typename Transform>
double run_once(const std::vector<std::uint8_t>& input,
                Transform&& transform) {
  std::array<std::uint32_t, 8> state =
      q3x::core::internal::kSha256InitialState;
  const auto begin = Clock::now();
  transform(state.data(), input.data(), input.size() / 64U);
  const auto end = Clock::now();
  sink = sink ^ state[0];
  return std::chrono::duration<double>(end - begin).count();
}

double median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2U];
}

}  // namespace

int main(int argc, char** argv) {
  const std::size_t mebibytes =
      argc > 1 ? static_cast<std::size_t>(std::stoull(argv[1])) : 128U;
  const std::size_t rounds =
      argc > 2 ? static_cast<std::size_t>(std::stoull(argv[2])) : 5U;
  if (mebibytes == 0U || rounds == 0U || mebibytes > 4096U) {
    std::cerr << "usage: q3x_sha256_benchmark [MiB=128] [rounds=5]\n";
    return 2;
  }
  if (!q3x::core::internal::sha256_arm_sha2_available()) {
    std::cout << "SHA-256 ARM SHA2 unavailable; benchmark skipped\n";
    return 77;
  }

  const std::size_t bytes = mebibytes * 1024U * 1024U;
  std::vector<std::uint8_t> input(bytes);
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<std::uint8_t>(index * 131U + 17U);
  }

  std::array<std::uint32_t, 8> portable =
      q3x::core::internal::kSha256InitialState;
  std::array<std::uint32_t, 8> accelerated = portable;
  q3x::core::internal::sha256_transform_blocks_portable(
      portable.data(), input.data(), input.size() / 64U);
  if (!q3x::core::internal::sha256_transform_blocks_arm_sha2(
          accelerated.data(), input.data(), input.size() / 64U) ||
      accelerated != portable) {
    std::cerr << "SHA-256 benchmark equivalence precondition failed\n";
    return 1;
  }

  std::vector<double> scalar_samples;
  std::vector<double> sha2_samples;
  scalar_samples.reserve(rounds);
  sha2_samples.reserve(rounds);
  for (std::size_t round = 0; round < rounds; ++round) {
    if ((round & 1U) == 0U) {
      scalar_samples.push_back(run_once(
          input, q3x::core::internal::sha256_transform_blocks_portable));
      sha2_samples.push_back(run_once(
          input, [](std::uint32_t* state, const std::uint8_t* blocks,
                    const std::size_t block_count) {
            static_cast<void>(
                q3x::core::internal::sha256_transform_blocks_arm_sha2(
                    state, blocks, block_count));
          }));
    } else {
      sha2_samples.push_back(run_once(
          input, [](std::uint32_t* state, const std::uint8_t* blocks,
                    const std::size_t block_count) {
            static_cast<void>(
                q3x::core::internal::sha256_transform_blocks_arm_sha2(
                    state, blocks, block_count));
          }));
      scalar_samples.push_back(run_once(
          input, q3x::core::internal::sha256_transform_blocks_portable));
    }
  }

  const double scalar_seconds = median(std::move(scalar_samples));
  const double sha2_seconds = median(std::move(sha2_samples));
  const double scalar_mib_per_second =
      static_cast<double>(mebibytes) / scalar_seconds;
  const double sha2_mib_per_second =
      static_cast<double>(mebibytes) / sha2_seconds;
  std::cout << std::fixed << std::setprecision(3)
            << "bytes=" << bytes << " rounds=" << rounds
            << " scalar_seconds=" << scalar_seconds
            << " scalar_MiB_per_s=" << scalar_mib_per_second
            << " arm_sha2_seconds=" << sha2_seconds
            << " arm_sha2_MiB_per_s=" << sha2_mib_per_second
            << " speedup=" << (scalar_seconds / sha2_seconds) << "x\n";
  return sink == 0xFFFFFFFFU ? EXIT_FAILURE : EXIT_SUCCESS;
}
