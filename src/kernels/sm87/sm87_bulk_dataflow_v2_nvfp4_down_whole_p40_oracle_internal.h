#pragma once

#include "q3x/kernels/sm87_bulk_dataflow_v2_nvfp4_down_whole_p40.h"

namespace q3x::kernels::sm87_bulk_v2_nvfp4_down_whole_p40_oracle_detail {

// Source-private same-kernel seam for the bounded synthetic differential
// oracle. Production-facing code cannot include this header. The public P40
// launcher always supplies the fixed 625x20 domain; the oracle shortens only
// the M domain while executing the identical kernel symbol and arithmetic.
struct RawArguments final {
  std::uint64_t transaction_epoch = 0U;
  const std::uint16_t* h = nullptr;
  const std::uint8_t* down_payload = nullptr;
  float tensor_scale = 0.0F;
  std::uint16_t* residual = nullptr;
  Sm87BulkV2NvFp4DownWholeP40DeviceControl* device_control = nullptr;
  const std::uint32_t* cancellation_signal = nullptr;
  std::uint32_t m_tiles = 1U;
  std::uint32_t n_tiles = kSm87BulkV2NvFp4DownWholeP40NTiles;
  void* cuda_stream = nullptr;
};

[[nodiscard]] int launch_raw(const RawArguments& arguments) noexcept;

}  // namespace q3x::kernels::sm87_bulk_v2_nvfp4_down_whole_p40_oracle_detail
