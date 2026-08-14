#pragma once

#include "q3x/kernels/sm87_bulk_dataflow_v2_fp8_whole_p40.h"

#include <array>
#include <cstdint>

namespace q3x::kernels::sm87_bulk_v2_fp8_whole_p40_oracle_detail {

// Source-private bounded seam for the synthetic CUDA oracle.  It narrows M/N
// while launching the identical role-specialized cooperative kernel symbols
// as the P40000 entry.  It accepts raw payload bytes only for T1 correctness
// and has no authentication, performance, API, or production authority.
struct RawArguments final {
  std::uint64_t transaction_epoch = 0U;
  Sm87TargetAotProjectionRole role =
      Sm87TargetAotProjectionRole::kInvalid;
  const std::uint16_t* input = nullptr;
  const std::uint8_t* payload = nullptr;
  std::array<std::uint16_t, 3U> compensated_scale_bf16_bits{};
  std::uint16_t* primary_output = nullptr;
  std::uint16_t* secondary_output = nullptr;
  std::uint16_t* tertiary_output = nullptr;
  Sm87BulkV2Fp8WholeP40DeviceControl* device_control = nullptr;
  const std::uint32_t* cancellation_signal = nullptr;
  std::uint32_t m_tiles = 0U;
  std::uint32_t n_tiles = 0U;
  void* cuda_stream = nullptr;
};

[[nodiscard]] int launch_raw(const RawArguments& arguments) noexcept;

}  // namespace q3x::kernels::
   // sm87_bulk_v2_fp8_whole_p40_oracle_detail
