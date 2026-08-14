#pragma once

#include "q3x/kernels/sm87_macrofeed_v4_bf16_ab.h"
#include "q3x/kernels/sm87_macrofeed_v4_fp8.h"
#include "q3x/kernels/sm87_macrofeed_v4_norm_residual.h"

#include <cstddef>

namespace q3x::runtime::sm87_macrofeed_v4_execution_events_detail {
class Sm87MacroFeedV4ExecutionEventsOwner;
}

namespace q3x::kernels::sm87_macrofeed_v4_bound_launch_detail {

// Fixed GDN-QKVZ C8000 binding.  There is intentionally no caller-selected
// role, layout, token count, row stride, K/V output, or CUDA stream in this
// schema.  The bound body always consumes contiguous H5120 and publishes the
// 16,384 projected values into the canonical 17,408-wide phase scratch.
struct Sm87MacroFeedV4GdnQkvzC8000Arguments final {
  const std::uint16_t* hidden_input = nullptr;
  Sm87TargetAotFp8CudaAssetView asset{};
  std::uint16_t* phase_scratch = nullptr;
};

// The CUDA stream exists only inside an EventsOwner-locked submission.  The
// execution package cannot construct, copy, move, inspect, or retain this
// capability, so a raw handle never crosses the owner boundary.
class Sm87MacroFeedV4LockedSubmitToken final {
 public:
  Sm87MacroFeedV4LockedSubmitToken() = delete;
  Sm87MacroFeedV4LockedSubmitToken(
      const Sm87MacroFeedV4LockedSubmitToken&) = delete;
  Sm87MacroFeedV4LockedSubmitToken& operator=(
      const Sm87MacroFeedV4LockedSubmitToken&) = delete;
  Sm87MacroFeedV4LockedSubmitToken(
      Sm87MacroFeedV4LockedSubmitToken&&) = delete;
  Sm87MacroFeedV4LockedSubmitToken& operator=(
      Sm87MacroFeedV4LockedSubmitToken&&) = delete;

 private:
  explicit Sm87MacroFeedV4LockedSubmitToken(void* cuda_stream) noexcept
      : cuda_stream_(cuda_stream) {}

  void* cuda_stream_ = nullptr;

  friend class q3x::runtime::
      sm87_macrofeed_v4_execution_events_detail::
          Sm87MacroFeedV4ExecutionEventsOwner;
  friend int enqueue_input_norm_prevalidated(
      const Sm87MacroFeedV4LockedSubmitToken&,
      const Sm87MacroFeedV4InputNormArguments&,
      const Sm87MacroFeedV4NormResidualAdmissionResourceSnapshot&,
      std::size_t*) noexcept;
  friend int enqueue_bf16_ab_prevalidated(
      const Sm87MacroFeedV4LockedSubmitToken&,
      const Sm87MacroFeedV4Bf16AbArguments&,
      const Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot&,
      std::size_t*) noexcept;
  friend int enqueue_gdn_qkvz_c8000_prevalidated(
      const Sm87MacroFeedV4LockedSubmitToken&,
      const Sm87MacroFeedV4GdnQkvzC8000Arguments&,
      const Sm87MacroFeedV4Fp8CudaResources&,
      std::size_t*) noexcept;
};

// Construction-prevalidated V4 launch seams.  These functions intentionally
// perform no device, function, occupancy, stream-device, allocation-range,
// JIT, repack, autotune, or tactic query.  They are source-private and may be
// called only by the event owner while it holds its stream submission lock.
// The execution package must have sealed every pointer/resource fact and own
// the complete CUDA allocation lifetime.
[[nodiscard]] int enqueue_input_norm_prevalidated(
    const Sm87MacroFeedV4LockedSubmitToken& token,
    const Sm87MacroFeedV4InputNormArguments& arguments,
    const Sm87MacroFeedV4NormResidualAdmissionResourceSnapshot& resources,
    std::size_t* submitted_launches) noexcept;

[[nodiscard]] int enqueue_bf16_ab_prevalidated(
    const Sm87MacroFeedV4LockedSubmitToken& token,
    const Sm87MacroFeedV4Bf16AbArguments& arguments,
    const Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot& resources,
    std::size_t* submitted_launches) noexcept;

[[nodiscard]] int enqueue_gdn_qkvz_c8000_prevalidated(
    const Sm87MacroFeedV4LockedSubmitToken& token,
    const Sm87MacroFeedV4GdnQkvzC8000Arguments& arguments,
    const Sm87MacroFeedV4Fp8CudaResources& resources,
    std::size_t* submitted_launches) noexcept;

}  // namespace q3x::kernels::sm87_macrofeed_v4_bound_launch_detail
