#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::runtime {

// Capability-free evidence schema for the physical full-model receipt of
// AC-PREFILL-SM87-MACROFEED-v3.  Public validation proves only that an already
// issued value is structurally coherent.  It cannot by itself prove that CUDA
// work ran, that completion was observed, or that any numerical, performance,
// API, or production qualification exists.  Issuance belongs to a future
// private executor after it consumes real constituent receipts and completion
// events; construction helpers are intentionally absent from this installed
// header.
inline constexpr std::array<std::uint8_t, 8U>
    kSm87MacroFeedV3ReceiptMagic{{'Q', '3', 'X', 'M', 'F', '3', 'R', '1'}};
inline constexpr std::uint16_t kSm87MacroFeedV3ReceiptAbiMajor = 1U;
inline constexpr std::uint16_t kSm87MacroFeedV3ReceiptAbiMinor = 0U;

inline constexpr std::size_t kSm87MacroFeedV3ReceiptLayerCount = 64U;
inline constexpr std::size_t kSm87MacroFeedV3ReceiptGdnLayerCount = 48U;
inline constexpr std::size_t kSm87MacroFeedV3ReceiptAttentionLayerCount = 16U;
inline constexpr std::size_t kSm87MacroFeedV3ReceiptArtifactsPerLayer = 4U;
inline constexpr std::uint64_t kSm87MacroFeedV3ReceiptFp8Launches = 128U;
inline constexpr std::uint64_t kSm87MacroFeedV3ReceiptGateUpLaunches = 64U;
inline constexpr std::uint64_t kSm87MacroFeedV3ReceiptDownLaunches = 64U;
inline constexpr std::uint64_t kSm87MacroFeedV3ReceiptBf16AbLaunches = 48U;
inline constexpr std::uint64_t kSm87MacroFeedV3ReceiptGdnLaunches = 432U;
inline constexpr std::uint64_t kSm87MacroFeedV3ReceiptAttentionCalls = 16U;
inline constexpr std::uint32_t
    kSm87MacroFeedV3ReceiptAttentionPreprocessPerLayer = 5U;
inline constexpr std::uint64_t
    kSm87MacroFeedV3ReceiptAttentionPreprocessPerRequest =
        kSm87MacroFeedV3ReceiptAttentionLayerCount *
        kSm87MacroFeedV3ReceiptAttentionPreprocessPerLayer;

enum class Sm87MacroFeedV3LayerType : std::uint8_t {
  kInvalid = 0U,
  kGdn,
  kFullAttention,
};

[[nodiscard]] constexpr Sm87MacroFeedV3LayerType
sm87_macrofeed_v3_expected_layer_type(const std::size_t layer) noexcept {
  if (layer >= kSm87MacroFeedV3ReceiptLayerCount) {
    return Sm87MacroFeedV3LayerType::kInvalid;
  }
  return ((layer + 1U) % 4U) == 0U
             ? Sm87MacroFeedV3LayerType::kFullAttention
             : Sm87MacroFeedV3LayerType::kGdn;
}

struct Sm87MacroFeedV3PhysicalIdentity final {
  std::uint64_t package_identity = 0U;
  std::uint64_t owner_identity = 0U;
  std::uint64_t allocation_identity = 0U;
  std::uint64_t catalog_identity = 0U;
  std::uint64_t device_identity = 0U;
  std::int32_t device_ordinal = -1;
};

struct Sm87MacroFeedV3ArtifactIdentities final {
  std::uint64_t fp8_input = 0U;
  std::uint64_t fp8_output = 0U;
  std::uint64_t nvfp4_gate_up = 0U;
  std::uint64_t nvfp4_down = 0U;
};

struct Sm87MacroFeedV3LayerPhysicalCounts final {
  std::uint32_t fp8_input = 0U;
  std::uint32_t fp8_output = 0U;
  std::uint32_t nvfp4_gate_up = 0U;
  std::uint32_t nvfp4_down = 0U;
  std::uint32_t bf16_ab = 0U;
  std::uint32_t gdn = 0U;
  std::uint32_t attention_calls = 0U;
  // V3 retains the fixed five-M8000-panel preprocess topology for every
  // full-Attention layer.  This is an observed physical count, not permission
  // for a public caller to forecast or issue a successful receipt.
  std::uint32_t attention_preprocess = 0U;
};

struct Sm87MacroFeedV3LayerReceipt final {
  std::uint64_t receipt_identity = 0U;
  std::uint64_t request_identity = 0U;
  std::size_t layer_index = std::numeric_limits<std::size_t>::max();
  Sm87MacroFeedV3LayerType layer_type = Sm87MacroFeedV3LayerType::kInvalid;
  Sm87MacroFeedV3ArtifactIdentities artifacts{};
  Sm87MacroFeedV3PhysicalIdentity physical_identity{};
  Sm87MacroFeedV3LayerPhysicalCounts physical_counts{};

  bool fill_completed = false;
  bool drain_completed = false;
  bool completion_observed = false;
  bool attention_preprocess_count_observed = false;

  bool used_fallback = false;
  bool used_mtp = false;
  bool used_cublaslt = false;
  bool used_request_jit = false;
  bool used_request_repack = false;
  bool used_request_autotune = false;
};

struct Sm87MacroFeedV3PhysicalAggregate final {
  std::size_t completed_layers = 0U;
  std::size_t completed_gdn_layers = 0U;
  std::size_t completed_attention_layers = 0U;
  std::size_t fill_completed_layers = 0U;
  std::size_t drain_completed_layers = 0U;
  std::size_t completion_observed_layers = 0U;
  std::size_t attention_preprocess_count_observed_layers = 0U;

  std::uint64_t fp8_input = 0U;
  std::uint64_t fp8_output = 0U;
  std::uint64_t fp8 = 0U;
  std::uint64_t nvfp4_gate_up = 0U;
  std::uint64_t nvfp4_down = 0U;
  std::uint64_t bf16_ab = 0U;
  std::uint64_t gdn = 0U;
  std::uint64_t attention_calls = 0U;
  std::uint64_t attention_preprocess = 0U;

  bool used_fallback = false;
  bool used_mtp = false;
  bool used_cublaslt = false;
  bool used_request_jit = false;
  bool used_request_repack = false;
  bool used_request_autotune = false;
};

struct Sm87MacroFeedV3TransactionReceipt final {
  std::uint64_t receipt_identity = 0U;
  std::array<std::uint8_t, 8U> magic{};
  std::uint16_t abi_major = 0U;
  std::uint16_t abi_minor = 0U;
  std::uint64_t request_identity = 0U;
  Sm87MacroFeedV3PhysicalIdentity physical_identity{};
  std::array<Sm87MacroFeedV3LayerReceipt,
             kSm87MacroFeedV3ReceiptLayerCount>
      layers{};
  std::size_t layer_receipt_count = 0U;
  Sm87MacroFeedV3PhysicalAggregate aggregate{};
  bool request_completion_observed = false;
  bool complete = false;
};

[[nodiscard]] bool sm87_macrofeed_v3_layer_receipt_valid(
    const Sm87MacroFeedV3LayerReceipt& receipt) noexcept;

[[nodiscard]] bool sm87_macrofeed_v3_transaction_receipt_valid(
    const Sm87MacroFeedV3TransactionReceipt& transaction) noexcept;

[[nodiscard]] bool sm87_macrofeed_v3_transaction_complete(
    const Sm87MacroFeedV3TransactionReceipt& transaction) noexcept;

static_assert(kSm87MacroFeedV3ReceiptGdnLayerCount * 9U ==
              kSm87MacroFeedV3ReceiptGdnLaunches);
static_assert(kSm87MacroFeedV3ReceiptAttentionPreprocessPerRequest == 80U);
static_assert(kSm87MacroFeedV3ReceiptGdnLayerCount +
                  kSm87MacroFeedV3ReceiptAttentionLayerCount ==
              kSm87MacroFeedV3ReceiptLayerCount);

}  // namespace q3x::runtime
