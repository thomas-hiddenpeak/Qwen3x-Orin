#include "q3x/runtime/reference_benchmark.h"
#include "q3x/runtime/reference_engine.h"
#include "q3x/version.h"
#include "q3x/runtime/prefill_workspace_plan.h"
#include "q3x/runtime/whole_core_request_geometry.h"

#include <iostream>
#include <string_view>

int main() {
  const std::string_view error = q3x::runtime::to_string(
      q3x::runtime::ReferenceEngineError::kInvalidArgument);
  const std::string_view stop = q3x::runtime::to_string(
      q3x::runtime::ReferenceStopReason::kImEnd);
  const std::string_view benchmark = q3x::runtime::to_string(
      q3x::runtime::ReferenceBenchmarkError::kRepeatabilityFailure);
  const std::string_view projection = q3x::runtime::to_string(
      q3x::runtime::ProjectionBackend::kSm87WeightOnly);
  q3x::runtime::ReferenceBenchmarkOptions options;
  q3x::runtime::ReferenceEngineOptions engine_options;
  q3x::runtime::ReferenceOneShotOptions one_shot_options;
  q3x::runtime::ReferenceGeneration generation;
  q3x::runtime::RequestStateResetReceipt reset_receipt;
  if (error != "invalid_argument" || stop != "im_end" ||
      benchmark != "repeatability_failure" ||
      projection != "sm87_weight_only" || options.warmup_rounds != 1U ||
      options.measured_rounds != 3U ||
      engine_options.projection_backend !=
          q3x::runtime::ProjectionBackend::kReference ||
      one_shot_options.projection_backend !=
          q3x::runtime::ProjectionBackend::kReference ||
      generation.request_state_reset.has_value() ||
      reset_receipt.mode !=
          q3x::runtime::RequestStateResetMode::kConservativeFull ||
      q3x::runtime::to_string(reset_receipt.mode) != "conservative_full" ||
      Q3X_VERSION_MAJOR != 0 || Q3X_VERSION_MINOR != 8 ||
      Q3X_VERSION_PATCH != 1 ||
      q3x::runtime::kMaximumRequestPrefillChunkSize != 512U) {
    return 1;
  }
#if defined(Q3X_ENABLE_WHOLE_CORE_SERVICE)
  const auto plan = q3x::runtime::build_unbound_layer_major_p40_whole_core_workspace_plan();
  if (!plan || plan.value->required_bytes !=
                   q3x::runtime::kWholeCoreCompiledArenaBytes ||
      q3x::runtime::kWholeCoreCompiledSequenceCapacity != 44095U) return 2;
#endif
  std::cout << "installed q3x::engine consumer linked successfully\n";
  return 0;
}
