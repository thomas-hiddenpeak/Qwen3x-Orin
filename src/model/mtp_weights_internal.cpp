#include "mtp_weights_internal.h"

namespace q3x::model::mtp_detail {

PlanResult plan_weights(const weights::WeightManifest& manifest) noexcept {
  std::size_t count = 0;
  for (const auto& item : manifest.tensors) {
    const bool named_mtp = item.first.compare(0, 4, "mtp.") == 0;
    if (named_mtp != (item.second.category == weights::TensorCategory::kMtp)) {
      return {std::nullopt, PlanError::kCategory, {}};
    }
    count += named_mtp ? 1U : 0U;
  }
  if (count != kTensorSpecs.size()) {
    return {std::nullopt, PlanError::kTensorSet, {}};
  }
  WeightPlan plan;
  for (std::size_t i = 0; i < kTensorSpecs.size(); ++i) {
    const TensorSpec& spec = kTensorSpecs[i];
    const auto found = manifest.tensors.find(spec.name);
    if (found == manifest.tensors.end()) {
      return {std::nullopt, PlanError::kTensorSet, spec.name};
    }
    const auto& locator = found->second;
    if (locator.dtype != io::safetensors::DType::kBf16) {
      return {std::nullopt, PlanError::kDtype, spec.name};
    }
    if (locator.shape.size() != (spec.columns == 0 ? 1U : 2U) ||
        locator.shape[0] != spec.rows ||
        (spec.columns != 0 && locator.shape[1] != spec.columns)) {
      return {std::nullopt, PlanError::kShape, spec.name};
    }
    const std::uint64_t bytes =
        2 * spec.rows * (spec.columns == 0 ? 1 : spec.columns);
    if (locator.byte_size != bytes) {
      return {std::nullopt, PlanError::kBytes, spec.name};
    }
    if (locator.file_end < locator.file_begin ||
        locator.file_end - locator.file_begin != bytes) {
      return {std::nullopt, PlanError::kRange, spec.name};
    }
    // Every pinned extent is already a multiple of the 256-byte alignment.
    // Computing from validated specs avoids trusting unbounded header sizes.
    plan.tensors[i] = {&locator, plan.arena_bytes};
    plan.arena_bytes += bytes;
  }
  if (plan.arena_bytes != weights::kPinnedQwen36_27BMtpBytes) {
    return {std::nullopt, PlanError::kBytes, {}};
  }
  return {plan, PlanError::kNone, {}};
}

}  // namespace q3x::model::mtp_detail
