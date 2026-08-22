#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::runtime::reference_runner_detail {

// Source-private observation point used only by BUILD_TESTING binaries.  The
// view is published after layer-3 full-Attention preprocessing has completed
// enqueue and before any Attention tactic consumes the four read-only inputs.
struct PrefillLayer3AttentionP513OracleView {
  std::size_t layer = 0U;
  std::size_t first_position = 0U;
  std::size_t token_count = 0U;
  const std::uint16_t* processed_query = nullptr;
  const std::uint16_t* key_cache = nullptr;
  const std::uint16_t* value_cache = nullptr;
  const std::uint16_t* packed_gate = nullptr;
  void* cuda_stream = nullptr;
};

using PrefillLayer3AttentionP513OracleCallback = void (*)(
    const PrefillLayer3AttentionP513OracleView& view,
    void* context) noexcept;

struct PrefillLayer3AttentionP513OracleHook {
  PrefillLayer3AttentionP513OracleCallback callback = nullptr;
  void* context = nullptr;
};

[[nodiscard]] PrefillLayer3AttentionP513OracleHook
exchange_prefill_layer3_attention_p513_oracle_hook(
    PrefillLayer3AttentionP513OracleHook hook) noexcept;

}  // namespace q3x::runtime::reference_runner_detail
