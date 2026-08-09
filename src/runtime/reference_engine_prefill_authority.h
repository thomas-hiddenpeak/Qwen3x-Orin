#pragma once

#include "q3x/runtime/prefill_operator_bindings.h"
#include "q3x/runtime/reference_runner.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

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
  kNvfp4GateUpCanonicalPanel = 0,
  kNvfp4DownCanonicalPanel,
  kFp8CanonicalPanel,
  kBf16CanonicalPanel,
  kExactGdnCanonicalPanel,
  kExactCausalAttentionCanonicalPanel,
  kResidualCanonicalPanel,
  kNormalizationCanonicalPanel,
  kEmbeddingCanonicalPanel,
  kFinalHandoff,
};

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
  std::uint32_t maximum_logical_panel_m = 0U;
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
      const void* main_stream, const void* auxiliary_stream,
      std::array<const void*, kBoundPrefillSubmissionEventCount>
          submission_events,
      std::array<NativePrefillRoleReceipt,
                 kLayerMajorPrefillRequiredOperatorRoleCount>
          roles) noexcept;

  const ModelWeights* weights_ = nullptr;
  RequestState* state_ = nullptr;
  ReferenceRunner* runner_ = nullptr;
  const void* arena_base_ = nullptr;
  std::uint64_t arena_bytes_ = 0U;
  const LayerMajorRequestMemoryPlan* memory_plan_ = nullptr;
  const void* main_stream_ = nullptr;
  const void* auxiliary_stream_ = nullptr;
  std::array<const void*, kBoundPrefillSubmissionEventCount>
      submission_events_{};
  std::array<NativePrefillRoleReceipt,
             kLayerMajorPrefillRequiredOperatorRoleCount>
      roles_{};

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
      const PrefillExecutionPlan& geometry) noexcept;

  const BoundPrefillExecutionPlan* plan_ = nullptr;
  ReferenceRunner* runner_ = nullptr;
  PrefillExecutionPlan geometry_{};
  Phase phase_ = Phase::kAwaitingExecution;

  friend class ReferenceEnginePrefillExecutor;
};

class ReferenceEnginePrefillPlanFactory final {
 public:
  [[nodiscard]] static BoundPrefillPlanResult bind(
      const ModelWeights* weights, RequestState* state,
      ReferenceRunner* runner) noexcept;
};

class ReferenceEnginePrefillExecutor final {
 public:
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
};

}  // namespace q3x::runtime::reference_engine_detail
