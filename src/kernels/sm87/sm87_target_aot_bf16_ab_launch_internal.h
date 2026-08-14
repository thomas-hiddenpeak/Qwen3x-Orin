#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace q3x::kernels::sm87_target_aot_bf16_ab_execution_detail {

inline constexpr std::size_t kInterleavedP40LayerBindings = 48U;

struct InterleavedP40PrevalidatedContract final {
  std::size_t startup_bindings = 0U;
  std::size_t startup_complete_device_ranges_per_binding = 0U;
  std::size_t hot_launches_per_layer = 0U;
  std::size_t hot_static_cuda_queries = 0U;
  bool startup_resource_gate = false;
  bool startup_nonblocking_stream_gate = false;
  bool startup_exact_layer_order_gate = false;
  bool hot_allocation_free = false;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return startup_bindings == kInterleavedP40LayerBindings &&
           startup_complete_device_ranges_per_binding == 4U &&
           hot_launches_per_layer == 1U && hot_static_cuda_queries == 0U &&
           startup_resource_gate && startup_nonblocking_stream_gate &&
           startup_exact_layer_order_gate && hot_allocation_free;
  }
};

[[nodiscard]] constexpr InterleavedP40PrevalidatedContract
interleaved_p40_prevalidated_contract() noexcept {
  return {kInterleavedP40LayerBindings, 4U, 1U, 0U,
          true, true, true, true};
}

struct InterleavedP40Binding final {
  std::uint32_t layer = 0U;
  const std::uint16_t* a_weights = nullptr;
  const std::uint16_t* b_weights = nullptr;
  const std::uint16_t* input = nullptr;
  std::uint16_t* interleaved_ab_output = nullptr;
};

struct SealedInterleavedP40Result;

// Source-private startup capability.  The only constructor is the sealing
// function below, which performs the resource and complete 48-binding checks
// before request admission.  The request hot path can therefore enqueue one
// layer without repeating device, function, occupancy, or pointer queries.
class SealedInterleavedP40Access final {
 public:
  SealedInterleavedP40Access(const SealedInterleavedP40Access&) = default;
  SealedInterleavedP40Access& operator=(
      const SealedInterleavedP40Access&) = default;
  SealedInterleavedP40Access(SealedInterleavedP40Access&&) = default;
  SealedInterleavedP40Access& operator=(
      SealedInterleavedP40Access&&) = default;
  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] std::int32_t device_ordinal() const noexcept {
    return device_ordinal_;
  }
  [[nodiscard]] void* cuda_stream() const noexcept { return cuda_stream_; }

 private:
  SealedInterleavedP40Access(
      std::array<InterleavedP40Binding, kInterleavedP40LayerBindings>
          bindings,
      std::int32_t device_ordinal, void* cuda_stream,
      std::uint64_t seal_nonce) noexcept
      : bindings_(bindings),
        device_ordinal_(device_ordinal),
        cuda_stream_(cuda_stream),
        seal_nonce_(seal_nonce) {}

  std::array<InterleavedP40Binding, kInterleavedP40LayerBindings> bindings_{};
  std::int32_t device_ordinal_ = -1;
  void* cuda_stream_ = nullptr;
  std::uint64_t seal_nonce_ = 0U;

  friend struct SealedInterleavedP40Result;
  friend SealedInterleavedP40Result seal_interleaved_p40(
      const InterleavedP40Binding*, std::size_t, void*) noexcept;
  friend int enqueue_interleaved_p40_prevalidated(
      const SealedInterleavedP40Access&, std::size_t,
      std::size_t*) noexcept;
};

struct SealedInterleavedP40Result final {
  std::optional<SealedInterleavedP40Access> access;
  int cuda_error = 0;

  [[nodiscard]] explicit operator bool() const noexcept {
    return access.has_value() && access->valid() && cuda_error == 0;
  }
};

[[nodiscard]] SealedInterleavedP40Result seal_interleaved_p40(
    const InterleavedP40Binding* bindings, std::size_t binding_count,
    void* cuda_stream) noexcept;

// Enqueues exactly one 625-CTA A/B grid.  submitted_launches is set to zero
// before validation and one immediately after a successful launch submission.
// No CUDA capability/resource/pointer query is reachable from this function.
[[nodiscard]] int enqueue_interleaved_p40_prevalidated(
    const SealedInterleavedP40Access& access, std::size_t gdn_ordinal,
    std::size_t* submitted_launches) noexcept;

[[nodiscard]] int launch_interleaved_p40(
    const std::uint16_t* a_weights,
    const std::uint16_t* b_weights,
    const std::uint16_t* input,
    std::uint16_t* interleaved_ab_output,
    void* cuda_stream) noexcept;

}  // namespace q3x::kernels::sm87_target_aot_bf16_ab_execution_detail
