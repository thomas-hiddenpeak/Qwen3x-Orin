#pragma once

#include "q3x/kernels/sm87_bulk_dataflow_v2_nvfp4_gate_up_whole_p40.h"

#include <cstdint>

namespace q3x::kernels::sm87_bulk_v2_nvfp4_gate_up_whole_p40_oracle_detail {

// Source-private same-kernel seam for the synthetic exact CUDA oracle.  The
// public enqueue fixes M=40000 and N=17408; this seam narrows the active tile
// domain while launching the identical cooperative kernel symbol from the
// same ELF.  It has no production dispatch authority.
struct RawArguments final {
  std::uint64_t transaction_epoch = 0U;
  const std::uint16_t* normalized_input = nullptr;
  const std::uint8_t* gate_up_payload = nullptr;
  float gate_tensor_scale = 0.0F;
  float up_tensor_scale = 0.0F;
  std::uint16_t* h = nullptr;
  Sm87BulkV2NvFp4GateUpWholeP40DeviceControl* device_control = nullptr;
  const std::uint32_t* cancellation_signal = nullptr;
  std::uint32_t m_tiles = 0U;
  std::uint32_t n_tiles = 0U;
  void* cuda_stream = nullptr;
};

[[nodiscard]] int launch_raw(const RawArguments& arguments) noexcept;

}  // namespace q3x::kernels::
   // sm87_bulk_v2_nvfp4_gate_up_whole_p40_oracle_detail
