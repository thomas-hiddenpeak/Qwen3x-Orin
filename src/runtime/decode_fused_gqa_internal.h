#pragma once
#include <cstddef>
#include <cstdint>

// Internal numerical admission, never the public probability-producing API.
// The runner owns the stream and request arena. The launcher owns no allocation
// or mutable device buffer; scratch is dead after the same-stream consumer.
namespace q3x::runtime::fused_decode {
inline constexpr std::size_t kMinimumSequence = 512;
inline constexpr std::size_t kMaximumSequence = 44095;
inline constexpr std::size_t kWorkspaceBytes = 8 * (24 * 256 * 4 + 24 * 4);
// Called once during runner construction, before readiness or Graph capture.
int prepare() noexcept;
// Fixed Q24/KV4/D256, BF16, scale 1/16, contiguous [S,4,256] KV.
// All five spans must be disjoint and remain alive through stream completion.
// Invalid host arguments fail before enqueue. Device non-finites propagate to
// the runner's finite-logit check; this is not a synchronous tensor validator.
int launch(const std::uint16_t* query, const std::uint16_t* key,
           const std::uint16_t* value, std::size_t sequence,
           void* workspace, std::size_t workspace_bytes,
           std::uint16_t* output, void* stream) noexcept;
// Source-private observation seam; only the non-installable admission links it.
struct Observation {
  const std::uint16_t *query, *key, *value;
  std::size_t sequence, layer;
  float* workspace;
  std::size_t workspace_elements;
  std::uint16_t* output;
  void* stream;
};
// Test-only teacher forcing overrides selected tokens after raw-logit capture.
using PredictionOverride = std::uint32_t (*)(std::uint32_t, std::size_t, void*) noexcept;
void set_prediction_override(PredictionOverride callback, void* context) noexcept;
std::uint32_t prediction_token(std::uint32_t proposed, std::size_t position) noexcept;
using Observer = int (*)(const Observation&, void*) noexcept;
void set_observer(Observer callback, void* context) noexcept;
int observe(const Observation&) noexcept;
}  // namespace q3x::runtime::fused_decode
