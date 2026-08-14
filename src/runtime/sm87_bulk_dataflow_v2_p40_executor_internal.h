#pragma once

#include "sm87_bulk_dataflow_v2_p40_composition_internal.h"
#include "sm87_target_aot_engine_rope_internal.h"

#include "q3x/runtime/decode_ops.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace q3x::runtime::sm87_bulk_v2_p40_executor_detail {

namespace owner_detail = sm87_bulk_v2_p40_owner_detail;

struct Sm87BulkV2P40ExecutorCreateResult;

#if defined(Q3X_ENABLE_SM87_BULK_DATAFLOW_V2_P40_EXECUTOR_ADMISSION)
inline constexpr bool kSm87BulkV2P40ExecutorAdmissionCompiled = true;
#else
inline constexpr bool kSm87BulkV2P40ExecutorAdmissionCompiled = false;
#endif

enum class Sm87BulkV2P40ExecutorError : std::uint8_t {
  kNone = 0U,
  kAdmissionDisabled,
  kInvalidComposition,
  kInvalidEngineRope,
  kIncompleteProjectionAssets,
  kInvalidInput,
  kRequestRearm,
  kGdnSessionRearm,
  kTransactionBegin,
  kCudaSubmission,
  kGdnSubmission,
  kOwnerTransaction,
  kInvalidHandoff,
  kCancelled,
  kAllocation,
};

struct Sm87BulkV2P40ExecutorStatus final {
  Sm87BulkV2P40ExecutorError error = Sm87BulkV2P40ExecutorError::kNone;
  const char* context = "none";
  int cuda_error = 0;
  std::size_t layer = kSm87BulkV2P40Layers;
  std::size_t segment = 0U;
  std::size_t constituent = 0U;

  [[nodiscard]] explicit operator bool() const noexcept {
    return error == Sm87BulkV2P40ExecutorError::kNone;
  }
};

using Sm87BulkV2P40CancellationProbe = bool (*)(void*) noexcept;

struct Sm87BulkV2P40ExecutionControl final {
  Sm87BulkV2P40CancellationProbe cancellation_probe = nullptr;
  void* cancellation_context = nullptr;
};

struct Sm87BulkV2P40ExecutionResult final {
  Sm87BulkV2P40ExecutorStatus status{};
  owner_detail::Sm87BulkV2P40OwnerReceipt receipt{};
  std::uint64_t request_epoch = 0U;
  std::uint32_t token_id = 0U;
  std::uint16_t value_bits = 0U;
  bool handoff_complete = false;

  [[nodiscard]] explicit operator bool() const noexcept {
    return static_cast<bool>(status) && request_epoch != 0U &&
           token_id < kSm87BulkV2P40Vocabulary && handoff_complete &&
           receipt.aggregate.lifecycle ==
               Sm87BulkV2P40OwnerLifecycle::kCompleted &&
           receipt.aggregate.state_committed &&
           receipt.aggregate.handoff_observed;
  }
};

class Sm87BulkV2P40Executor final {
 public:
  Sm87BulkV2P40Executor(const Sm87BulkV2P40Executor&) = delete;
  Sm87BulkV2P40Executor& operator=(const Sm87BulkV2P40Executor&) = delete;
  Sm87BulkV2P40Executor(Sm87BulkV2P40Executor&&) = delete;
  Sm87BulkV2P40Executor& operator=(Sm87BulkV2P40Executor&&) = delete;
  ~Sm87BulkV2P40Executor();

  [[nodiscard]] bool ready() const noexcept;
  [[nodiscard]] const sm87_bulk_v2_p40_composition_detail::
      Sm87BulkV2P40CompositionRoot* composition() const noexcept {
    return composition_.get();
  }
  [[nodiscard]] Sm87BulkV2P40ExecutionResult execute(
      const std::uint32_t* host_prompt_token_ids,
      std::size_t prompt_tokens,
      const Sm87BulkV2P40ExecutionControl& control = {}) noexcept;

 private:
  using CompositionRoot = sm87_bulk_v2_p40_composition_detail::
      Sm87BulkV2P40CompositionRoot;

  struct LayerAssets final {
    kernels::Sm87TargetAotFp8CudaAssetView input_fp8{};
    kernels::Sm87TargetAotFp8CudaAssetView output_fp8{};
    kernels::Sm87TargetAotNvFp4CudaAssetView gate_up{};
    kernels::Sm87TargetAotNvFp4CudaAssetView down{};
  };

  Sm87BulkV2P40Executor(
      std::unique_ptr<CompositionRoot> composition,
      sm87_target_aot_p40_executor_detail::Sm87TargetAotP40EngineRope rope,
      const std::array<LayerAssets, kSm87BulkV2P40Layers>& assets) noexcept;

  std::unique_ptr<CompositionRoot> composition_;
  sm87_target_aot_p40_executor_detail::Sm87TargetAotP40EngineRope rope_{};
  std::array<LayerAssets, kSm87BulkV2P40Layers> assets_{};
  std::uint64_t next_request_epoch_ = 1U;
  bool terminal_failure_ = false;

  friend struct Sm87BulkV2P40ExecutorCreateResult;
  friend Sm87BulkV2P40ExecutorCreateResult
  create_sm87_bulk_dataflow_v2_p40_executor(
      std::unique_ptr<CompositionRoot>,
      const sm87_target_aot_p40_executor_detail::
          Sm87TargetAotEngineRopeOwner&) noexcept;
};

struct Sm87BulkV2P40ExecutorCreateResult final {
  std::unique_ptr<Sm87BulkV2P40Executor> executor;
  Sm87BulkV2P40ExecutorStatus status{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return executor != nullptr && executor->ready() &&
           static_cast<bool>(status);
  }
};

// The executor takes ownership of the unique startup composition.  RoPE is
// accepted only through its engine-lifetime owner, never as caller-supplied
// device pointers or readiness claims.
[[nodiscard]] Sm87BulkV2P40ExecutorCreateResult
create_sm87_bulk_dataflow_v2_p40_executor(
    std::unique_ptr<sm87_bulk_v2_p40_composition_detail::
                        Sm87BulkV2P40CompositionRoot>
        composition,
    const sm87_target_aot_p40_executor_detail::
        Sm87TargetAotEngineRopeOwner& rope_owner) noexcept;

[[nodiscard]] const char* to_string(
    Sm87BulkV2P40ExecutorError error) noexcept;

static_assert(sizeof(Bf16GreedyArgmaxResult) ==
              sizeof(owner_detail::Sm87BulkV2P40PinnedHandoff));
static_assert(offsetof(Bf16GreedyArgmaxResult, index) ==
              offsetof(owner_detail::Sm87BulkV2P40PinnedHandoff, token_id));
static_assert(offsetof(Bf16GreedyArgmaxResult, value_bits) ==
              offsetof(owner_detail::Sm87BulkV2P40PinnedHandoff,
                       value_bits));
static_assert(offsetof(Bf16GreedyArgmaxResult, has_nonfinite) ==
              offsetof(owner_detail::Sm87BulkV2P40PinnedHandoff,
                       nonfinite));

}  // namespace q3x::runtime::sm87_bulk_v2_p40_executor_detail
