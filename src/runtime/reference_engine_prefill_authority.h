#pragma once

#include "q3x/runtime/prefill_operator_bindings.h"
#include "q3x/runtime/reference_runner.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>

namespace q3x::runtime::sm87_macrofeed_v3_p40_execution_package_detail {
class Sm87MacroFeedV3P40ExecutionPackage;
}

namespace q3x::runtime::reference_engine_detail {

inline constexpr std::size_t kBoundPrefillSubmissionEventCount = 2U;

// Engine-internal capability boundary for the default-off layer-major
// evaluation route. Public PrefillOperatorBindingSet values remain
// descriptive and can never be exchanged for this authority.
enum class BoundPrefillPlanError : std::uint8_t {
  kNone = 0,
  kInvalidDependency,
  kWrongMemoryProfile,
  kRunnerIdentityMismatch,
  kIncompleteTypedViews,
  kUnsupportedBinary,
  kIncompleteNativeRole,
};

enum class NativePrefillTactic : std::uint8_t {
  kNvfp4GateUpOracleSpanC512 = 0,
  kNvfp4DownOracleSpanC512,
  kFp8OracleSpanC512,
  kNvfp4GateUpSegmentedMarlinOperatorPanel,
  kNvfp4DownSegmentedMarlinOperatorPanel,
  kFp8SegmentedMarlinOperatorPanel,
  kNvfp4GateUpNativeQuantizedLargeMOperatorPanel,
  kNvfp4DownNativeQuantizedLargeMOperatorPanel,
  kFp8NativeQuantizedLargeMOperatorPanel,
  kNvfp4GateUpTrueLargeMOperatorPanel,
  kNvfp4DownTrueLargeMOperatorPanel,
  kNvfp4GateUpG2LargeMOperatorPanel,
  kNvfp4DownD2LargeMOperatorPanel,
  kNvfp4GateUpPersistentP40LayerWide,
  kNvfp4DownResidualPersistentP40LayerWide,
  kFp8Nvfp4TrueLargeMRouteCompanion,
  kFp8PromptWideP40FillDrain,
  kBf16AbOracleSpanEstablishedM32,
  kBf16AbPromptWideP40,
  kExactGdnOracleSpanWholeRawQkvC512,
  kExactGdnPromptWideP40ChunkGraph,
  kExactCausalAttentionOracleSpanC512C16Reference256,
  kNativeCausalAttentionGroupQ64OperatorPanel,
  kNativeCausalAttentionGroupQ128V4OperatorPanel,
  kNativeCausalAttentionFlashInferExactOperatorPanel,
  kNativeCausalAttentionFlashInferExactWholePrompt,
  kResidualOperatorPanel,
  kNormalizationOperatorPanel,
  kEmbeddingOperatorPanel,
  kFinalHandoff,
  kNvfp4GateUpP40ProjectionReset,
  kNvfp4DownResidualP40ProjectionReset,
  kFp8P40ProjectionReset,
  kNvfp4GateUpP40PackedProjection,
  kNvfp4DownResidualP40PackedProjection,
  kFp8P40PackedProjection,
  kNvfp4GateUpP40PackedNvfp4V2,
  kNvfp4DownResidualP40PackedNvfp4V2,
  // Exact stock-vLLM host-dispatch parity keeps the projection publication
  // boundary explicit. Down writes a BF16 branch; residual is the independent
  // kResidualOperatorPanel role and may not be reported as fused here.
  kNvfp4GateUpP40VllmMarlinParity,
  kNvfp4DownBranchP40VllmMarlinParity,
  kNvfp4GateUpP40MacroFeedV3,
  kNvfp4DownResidualP40MacroFeedV3,
  kFp8P40MacroFeedV3RoleSpecialized,
  kExactGdnP40MacroFeedV3,
};

// Compile-inventory fact only; device resources are still queried by bind().
// This remains false unless the independent whole-core admission and every
// required component admission were compiled into the same q3x_core binary.
[[nodiscard]] bool
prompt_wide_p40_whole_core_prefill_authority_enabled() noexcept;

enum class NativePrefillCompletionDomain : std::uint8_t {
  kMainStreamBarrier = 0,
  kAuxiliaryJoinedToMainBarrier,
};

struct NativePrefillRoleReceipt {
  PrefillBindingRole role = PrefillBindingRole::kInvalid;
  NativePrefillTactic tactic =
      NativePrefillTactic::kFinalHandoff;
  NativePrefillCompletionDomain completion =
      NativePrefillCompletionDomain::kMainStreamBarrier;
  const void* artifact_owner = nullptr;
  const void* workspace_owner = nullptr;
  std::uint64_t workspace_bytes = 0U;
  const void* auxiliary_workspace_owner = nullptr;
  std::uint64_t auxiliary_workspace_bytes = 0U;
  std::uint32_t maximum_logical_panel_m = 0U;
  std::uint32_t minimum_physical_m = 0U;
  std::uint32_t maximum_physical_m = 0U;
};

class BoundPrefillExecutionPlan final {
 public:
  BoundPrefillExecutionPlan() = delete;
  BoundPrefillExecutionPlan(const BoundPrefillExecutionPlan&) = delete;
  BoundPrefillExecutionPlan& operator=(
      const BoundPrefillExecutionPlan&) = delete;
  BoundPrefillExecutionPlan(BoundPrefillExecutionPlan&&) = delete;
  BoundPrefillExecutionPlan& operator=(BoundPrefillExecutionPlan&&) = delete;

 private:
  BoundPrefillExecutionPlan(
      const ModelWeights* weights, RequestState* state,
      ReferenceRunner* runner, const void* arena_base,
      std::uint64_t arena_bytes,
      const LayerMajorRequestMemoryPlan* memory_plan,
      const LayerMajorPrefillArithmeticContract* arithmetic_contract,
      bool exact_c512_arithmetic_workspace_bound,
      LayerMajorPrefillProjectionTactic projection_tactic,
      LayerMajorPrefillFullAttentionTactic full_attention_tactic,
      LayerMajorPrefillMlpScheduleTactic mlp_schedule_tactic,
      const void* main_stream, const void* auxiliary_stream,
      std::array<const void*, kBoundPrefillSubmissionEventCount>
          submission_events,
      std::array<NativePrefillRoleReceipt,
                 kLayerMajorPrefillRequiredOperatorRoleCount>
          roles,
      const sm87_macrofeed_v3_p40_execution_package_detail::
          Sm87MacroFeedV3P40ExecutionPackage* macrofeed_v3_package = nullptr,
      std::uint64_t macrofeed_v3_package_identity = 0U) noexcept;

  const ModelWeights* weights_ = nullptr;
  RequestState* state_ = nullptr;
  ReferenceRunner* runner_ = nullptr;
  const void* arena_base_ = nullptr;
  std::uint64_t arena_bytes_ = 0U;
  const LayerMajorRequestMemoryPlan* memory_plan_ = nullptr;
  const LayerMajorPrefillArithmeticContract* arithmetic_contract_ = nullptr;
  // The sealed exact contract uses the authenticated disjoint C512 arithmetic
  // workspace for every oracle span's NVFP4 Gate+Up/SiLU/Down sequence. The
  // M8192 Marlin contract retains that workspace for every partial panel and
  // binds typed reduction/lock workspaces for its full-panel bulk sequence.
  bool exact_c512_arithmetic_workspace_bound_ = false;
  LayerMajorPrefillProjectionTactic projection_tactic_ =
      LayerMajorPrefillProjectionTactic::kExactSegmentedC512;
  LayerMajorPrefillFullAttentionTactic full_attention_tactic_ =
      LayerMajorPrefillFullAttentionTactic::kExactSegmentedC512;
  LayerMajorPrefillMlpScheduleTactic mlp_schedule_tactic_ =
      LayerMajorPrefillMlpScheduleTactic::kPerOperatorPanel;
  const void* main_stream_ = nullptr;
  const void* auxiliary_stream_ = nullptr;
  std::array<const void*, kBoundPrefillSubmissionEventCount>
      submission_events_{};
  std::array<NativePrefillRoleReceipt,
             kLayerMajorPrefillRequiredOperatorRoleCount>
      roles_{};
  const sm87_macrofeed_v3_p40_execution_package_detail::
      Sm87MacroFeedV3P40ExecutionPackage* macrofeed_v3_package_ = nullptr;
  std::uint64_t macrofeed_v3_package_identity_ = 0U;
  // One Engine-bound monotonic request serial. It is consumed only by V3 and
  // never reset or inferred from caller-visible request data.
  mutable std::atomic<std::uint64_t> macrofeed_v3_next_request_identity_{1U};

  friend class ReferenceEnginePrefillPlanFactory;
  friend class ReferenceEnginePrefillExecutor;
};

struct BoundPrefillPlanResult {
  std::unique_ptr<const BoundPrefillExecutionPlan> value;
  BoundPrefillPlanError error = BoundPrefillPlanError::kNone;
  ReferenceRunnerStatus status;

  [[nodiscard]] explicit operator bool() const noexcept {
    return value != nullptr && error == BoundPrefillPlanError::kNone &&
           status.ok();
  }
};

class BoundPrefillRequestReceipt final {
 public:
  BoundPrefillRequestReceipt() = delete;
  BoundPrefillRequestReceipt(const BoundPrefillRequestReceipt&) = delete;
  BoundPrefillRequestReceipt& operator=(
      const BoundPrefillRequestReceipt&) = delete;
  BoundPrefillRequestReceipt(BoundPrefillRequestReceipt&& other) noexcept;
  BoundPrefillRequestReceipt& operator=(
      BoundPrefillRequestReceipt&&) noexcept = delete;

 private:
  enum class Phase : std::uint8_t {
    kAwaitingExecution = 0,
    kAwaitingLogits,
    kAwaitingCommit,
    kConsumed,
    kPoisoned,
  };

  BoundPrefillRequestReceipt(
      const BoundPrefillExecutionPlan* plan,
      ReferenceRunner* runner,
      const PrefillExecutionPlan& geometry,
      std::uint64_t macrofeed_v3_request_identity = 0U) noexcept;

  const BoundPrefillExecutionPlan* plan_ = nullptr;
  ReferenceRunner* runner_ = nullptr;
  PrefillExecutionPlan geometry_{};
  Phase phase_ = Phase::kAwaitingExecution;
  std::uint64_t macrofeed_v3_request_identity_ = 0U;
  std::optional<Sm87MacroFeedV3TransactionReceipt>
      macrofeed_v3_transaction_receipt_;

  friend class ReferenceEnginePrefillExecutor;
};

class ReferenceEnginePrefillPlanFactory final {
 public:
  [[nodiscard]] static BoundPrefillPlanResult bind(
      const ModelWeights* weights, RequestState* state,
      ReferenceRunner* runner,
      LayerMajorPrefillProjectionTactic projection_tactic,
      LayerMajorPrefillFullAttentionTactic full_attention_tactic,
      const sm87_macrofeed_v3_p40_execution_package_detail::
          Sm87MacroFeedV3P40ExecutionPackage* macrofeed_v3_package = nullptr)
      noexcept;
};

// Same-ELF, thread-local correctness oracle selector. Production execution
// never changes this value and always takes the operator-panel route. Tests
// may scope it around one synchronous ReferenceEngine::generate() call to
// compare the retained segmented compatibility core with the bound panel
// executor without creating a second public mode or fallback.
[[nodiscard]] bool
exchange_reference_engine_prefill_compatibility_oracle_for_test(
    bool enabled) noexcept;

class ReferenceEnginePrefillExecutor final {
 public:
  [[nodiscard]] static std::string_view deployment_plan_id(
      const BoundPrefillExecutionPlan& plan) noexcept;

  [[nodiscard]] static ReferenceWholeRequestPrefillOutcome execute(
      const BoundPrefillExecutionPlan& plan, ReferenceRunner& runner,
      const std::uint32_t* input_token_ids, std::size_t token_count,
      const PrefillExecutionPlan& geometry,
      const ReferenceWholeRequestPrefillOptions& options,
      std::optional<BoundPrefillRequestReceipt>& receipt) noexcept;

  [[nodiscard]] static ReferenceStepOutcome finish(
      const BoundPrefillExecutionPlan& plan, ReferenceRunner& runner,
      BoundPrefillRequestReceipt& receipt,
      std::uint32_t input_token_id,
      const ReferenceStepOptions& options) noexcept;

  [[nodiscard]] static ReferenceRunnerStatus commit(
      const BoundPrefillExecutionPlan& plan, ReferenceRunner& runner,
      BoundPrefillRequestReceipt& receipt,
      const PrefillExecutionPlan& geometry,
      const PrefillExecutionProgress& progress) noexcept;

 private:
  [[nodiscard]] static bool plan_matches_runner(
      const BoundPrefillExecutionPlan& plan,
      const ReferenceRunner& runner) noexcept;
  [[nodiscard]] static bool macrofeed_v3_receipt_matches_plan(
      const BoundPrefillExecutionPlan& plan,
      const BoundPrefillRequestReceipt& receipt) noexcept;
};

}  // namespace q3x::runtime::reference_engine_detail
