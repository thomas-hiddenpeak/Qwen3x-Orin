#pragma once

#include "q3x/kernels/sm87_a4w4_attention_k256_m128n256.h"

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Default-off structural Attention-projection candidate.  It preserves the
// incumbent K256 payload and fixed topology contract while changing only the
// consumer dataflow: 256 threads, M32N128/warp, one register-handoff A slot,
// and four long-lived B slots.
inline constexpr std::size_t
    kSm87A4W4AttentionK256AExchangeB4Threads = 256U;
inline constexpr std::size_t
    kSm87A4W4AttentionK256AExchangeB4Warps = 8U;
inline constexpr std::size_t
    kSm87A4W4AttentionK256AExchangeBytes = 16'640U;
inline constexpr std::size_t
    kSm87A4W4AttentionK256B4SlotBytes = 33'280U;
inline constexpr std::size_t
    kSm87A4W4AttentionK256B4Slots = 4U;
inline constexpr std::size_t
    kSm87A4W4AttentionK256AExchangeB4SharedBytes = 149'760U;
inline constexpr std::size_t
    kSm87A4W4AttentionK256AExchangeB4MaximumRegisters = 255U;

// Queries all three fixed topology specializations and reports their worst
// resource values.  No runtime route is changed by this surface.
[[nodiscard]] int
query_sm87_a4w4_attention_k256_m128n256_a_exchange_b4_resources_cuda(
    Sm87A4W4AttentionK256Resources* resources) noexcept;

// Independent fixed-topology correctness/performance launcher.  The inputs,
// outputs, projection order, and capacity rules are identical to the existing
// K256 M128N256 launcher; only the kernel consumer is different.
[[nodiscard]] int
launch_sm87_a4w4_attention_k256_m128n256_a_exchange_b4_bf16_cuda(
    Sm87A4W4AttentionK256Topology topology,
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_k256_scales_bf16,
    std::size_t a_scale_capacity_elements,
    std::size_t token_count,
    const Sm87A4W4AttentionK256ProjectionView* projections,
    std::size_t projection_count,
    void* cuda_stream = nullptr) noexcept;

static_assert(kSm87A4W4AttentionK256AExchangeB4Threads == 256U);
static_assert(kSm87A4W4AttentionK256AExchangeB4Warps == 8U);
static_assert(kSm87A4W4AttentionK256AExchangeBytes == 16'640U);
static_assert(kSm87A4W4AttentionK256B4SlotBytes == 33'280U);
static_assert(kSm87A4W4AttentionK256AExchangeB4SharedBytes ==
              kSm87A4W4AttentionK256AExchangeBytes +
                  kSm87A4W4AttentionK256B4Slots *
                      kSm87A4W4AttentionK256B4SlotBytes);

}  // namespace q3x::kernels
