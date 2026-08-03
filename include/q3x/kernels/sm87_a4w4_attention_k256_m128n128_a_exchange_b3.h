#pragma once

#include "q3x/kernels/sm87_a4w4_attention_k256_m128n256.h"

#include <cstddef>

namespace q3x::kernels {

// Default-off structural Attention projection candidate.  One 256-thread CTA
// owns an M128N128 cell:
//
//   warp = n64_half * 4 + m32
//
// Each warp retains M32N64 (64 FP32 accumulator registers/thread).  One K256
// A tile is exchanged through a short-lived shared slot and retained in
// registers while three K256 B tiles remain in a long-lived ring.
// This cuts both the accumulator live set and the B footprint in half versus
// the M128N256 A-exchange/B4 production experiment.
inline constexpr std::size_t
    kSm87A4W4AttentionK256M128N128AExchangeThreads = 256U;
inline constexpr std::size_t
    kSm87A4W4AttentionK256M128N128AExchangeWarps = 8U;
inline constexpr std::size_t
    kSm87A4W4AttentionK256M128N128TileM = 128U;
inline constexpr std::size_t
    kSm87A4W4AttentionK256M128N128TileN = 128U;
inline constexpr std::size_t
    kSm87A4W4AttentionK256M128N128WarpTileM = 32U;
inline constexpr std::size_t
    kSm87A4W4AttentionK256M128N128WarpTileN = 64U;
inline constexpr std::size_t
    kSm87A4W4AttentionK256M128N128AExchangeBytes = 16'640U;
inline constexpr std::size_t
    kSm87A4W4AttentionK256M128N128BSlotBytes = 16'640U;
inline constexpr std::size_t
    kSm87A4W4AttentionK256M128N128B3SharedBytes = 66'560U;
inline constexpr std::size_t
    kSm87A4W4AttentionK256M128N128MaximumRegisters = 128U;
inline constexpr std::size_t
    kSm87A4W4AttentionK256M128N128MaximumDynamicSharedBytes =
        82'688U;
inline constexpr std::size_t
    kSm87A4W4AttentionK256M128N128PersistentCtas = 32U;
inline constexpr std::size_t
    kSm87A4W4AttentionK256M128N128RequiredCtasPerSm = 2U;

struct Sm87A4W4AttentionK256M128N128Resources final {
  int registers_per_thread{};
  std::size_t static_shared_bytes{};
  std::size_t dynamic_shared_bytes{};
  std::size_t local_bytes{};
  int active_blocks_per_sm{};
  int maximum_threads_per_block{};
  int device_compute_major{};
  int device_compute_minor{};
  int device_multiprocessor_count{};
};

// Queries the worst of all three exact Qwen3.6 B3 topology specializations.
// Success means <=128 registers, zero compiler local frame, exactly 66,560
// bytes of dynamic shared memory, and two resident CTAs/SM on the exact
// 16-SM SM87 target.
[[nodiscard]] int
query_sm87_a4w4_attention_k256_m128n128_a_exchange_b3_resources_cuda(
    Sm87A4W4AttentionK256M128N128Resources* resources) noexcept;

// Exact-topology B3 launcher.  It consumes the incumbent K256 ABI and
// projection views, doubles the canonical N256 macro-cell count into N128
// cells, and launches a fixed grid of 32 CTAs.
[[nodiscard]] int
launch_sm87_a4w4_attention_k256_m128n128_a_exchange_b3_bf16_cuda(
    Sm87A4W4AttentionK256Topology topology,
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_k256_scales_bf16,
    std::size_t a_scale_capacity_elements,
    std::size_t token_count,
    const Sm87A4W4AttentionK256ProjectionView* projections,
    std::size_t projection_count,
    void* cuda_stream = nullptr) noexcept;

static_assert(kSm87A4W4AttentionK256M128N128AExchangeThreads == 256U);
static_assert(kSm87A4W4AttentionK256M128N128AExchangeWarps == 8U);
static_assert(kSm87A4W4AttentionK256M128N128AExchangeBytes == 16'640U);
static_assert(kSm87A4W4AttentionK256M128N128BSlotBytes == 16'640U);
static_assert(kSm87A4W4AttentionK256M128N128B3SharedBytes ==
              kSm87A4W4AttentionK256M128N128AExchangeBytes +
                  3U * kSm87A4W4AttentionK256M128N128BSlotBytes);
static_assert(kSm87A4W4AttentionK256M128N128B3SharedBytes <=
              kSm87A4W4AttentionK256M128N128MaximumDynamicSharedBytes);
static_assert(kSm87A4W4AttentionK256M128N128PersistentCtas ==
              16U * kSm87A4W4AttentionK256M128N128RequiredCtasPerSm);

}  // namespace q3x::kernels
