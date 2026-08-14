#pragma once

#include "q3x/kernels/sm87_macrofeed_v4_bf16_ab.h"
#include "q3x/kernels/sm87_macrofeed_v4_norm_residual.h"

#include <cstddef>

namespace q3x::runtime::sm87_macrofeed_v4_execution_events_detail {
class Sm87MacroFeedV4ExecutionEventsOwner;
}

namespace q3x::kernels::sm87_macrofeed_v4_bound_launch_detail {

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

}  // namespace q3x::kernels::sm87_macrofeed_v4_bound_launch_detail
