#pragma once

#include <cstdint>

namespace q3x::runtime::mtp_detail {

// Candidate-only source-address indirection for the first row of a verifier.
// state_handle is one aligned device pointer slot. Its pointee is a complete
// [48,128,128] BF16 GDN state and may exactly equal state_output. The caller
// validates the pointee's extent/lifetime before publishing the handle, and
// orders handle updates and all consumers on the same stream. Neither the
// handle nor its pointee may be modified concurrently with this operation.
// The handle itself must be disjoint from all operands. Later verifier rows
// continue to use direct immutable prefix inputs. No allocation/sync/fallback.
// Device-resident pointer contents cannot be validated by this host entry.
int launch_mtp_gdn_selected_state(
    const std::uint16_t* conv_qkv, const std::uint16_t* a,
    const std::uint16_t* b, const std::uint16_t* a_log,
    const std::uint16_t* dt_bias,
    const std::uint16_t* const* state_handle,
    std::uint16_t* state_output, const std::uint16_t* norm_weight,
    const std::uint16_t* silu_gate, std::uint16_t* output,
    void* stream) noexcept;

}  // namespace q3x::runtime::mtp_detail
