#include "q3x/runtime/reference_benchmark.h"
#include "q3x/runtime/reference_engine.h"
#include "q3x/version.h"

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
  if (error != "invalid_argument" || stop != "im_end" ||
      benchmark != "repeatability_failure" ||
      projection != "sm87_weight_only" || options.warmup_rounds != 1U ||
      options.measured_rounds != 3U ||
      engine_options.projection_backend !=
          q3x::runtime::ProjectionBackend::kReference ||
      one_shot_options.projection_backend !=
          q3x::runtime::ProjectionBackend::kReference ||
      Q3X_VERSION_MAJOR != 0 || Q3X_VERSION_MINOR != 4 ||
      Q3X_VERSION_PATCH != 0 ||
      q3x::runtime::kMaximumRequestPrefillChunkSize != 512U) {
    return 1;
  }
  std::cout << "installed q3x::engine consumer linked successfully\n";
  return 0;
}
