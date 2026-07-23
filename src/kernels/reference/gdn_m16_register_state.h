#ifndef Q3X_KERNELS_REFERENCE_GDN_M16_REGISTER_STATE_H_
#define Q3X_KERNELS_REFERENCE_GDN_M16_REGISTER_STATE_H_

#include <cstdint>

namespace q3x::runtime::gdn_decode_detail {

// Internal exact-M16 launch ABI. The validated production dispatcher owns
// shape, pointer, alias, and epsilon validation before calling this entry.
[[nodiscard]] int launch_gated_delta_net_update_m16_register_state_exact_cuda(
    const std::uint16_t* conv_qkv, const std::uint16_t* a,
    const std::uint16_t* b, const std::uint16_t* A_log,
    const std::uint16_t* dt_bias, const std::uint16_t* state_input,
    std::uint16_t* state_output, float l2_epsilon, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

}  // namespace q3x::runtime::gdn_decode_detail

#endif  // Q3X_KERNELS_REFERENCE_GDN_M16_REGISTER_STATE_H_
