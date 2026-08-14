#include "../src/runtime/sm87_macrofeed_v4_p40_execution_package_internal.h"
#include "support/sm87_target_aot_complete_host_fixture.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string_view>
#include <type_traits>

namespace q3x::runtime::sm87_macrofeed_v4_p40_execution_detail {

class Sm87MacroFeedV4P40ExecutionPackageCudaTestFixture final {
 public:
  struct Samples final {
    std::uint16_t normalized = 0U;
    std::uint16_t projection_a = 0U;
    std::uint16_t projection_b = 0U;
  };

  struct RequestOutcome final {
    Sm87MacroFeedV4RequestStatePhase phase =
        Sm87MacroFeedV4RequestStatePhase::kInvalid;
    bool canonical_state_published = true;
    bool logical_sequence_fence_published = true;
    bool decode_access_issued = true;
  };

  [[nodiscard]] static Sm87MacroFeedV4GdnLayer0FrontHalfResult execute_once(
      Sm87MacroFeedV4P40ExecutionPackage& package) noexcept {
    return package.execute_gdn_layer0_front_half_once();
  }

  [[nodiscard]] static bool exercise_terminal_poison_drain(
      Sm87MacroFeedV4P40ExecutionPackage& package) noexcept {
    if (package.request_state_ == nullptr || package.events_owner_ == nullptr ||
        package.events_driver_ == nullptr) {
      return false;
    }
    const auto request_access = package.request_state_->issue_sealed_access();
    const auto begin_request = package.events_driver_->begin_request(
        *package.request_state_, request_access);
    if (!begin_request) {
      return false;
    }
    auto panel = package.events_driver_->begin_panel(0U);
    if (!panel) {
      return false;
    }
    const auto injected =
        sm87_macrofeed_v4_execution_events_detail::
            Sm87MacroFeedV4ExecutionEventsCudaTestFixture::
                inject_poison_without_drain(
                    *package.events_owner_,
                    sm87_macrofeed_v4_execution_events_detail::
                        Sm87MacroFeedV4ExecutionError::kCudaSubmission);
    if (injected.error !=
        sm87_macrofeed_v4_execution_events_detail::
            Sm87MacroFeedV4ExecutionError::kCudaSubmission) {
      return false;
    }
    const auto terminalized = package.terminalize_event_failure(
        "test_tail_record_cuda_failure", injected);
    const auto snapshot = package.events_driver_->snapshot();
    const auto aborted = package.abort_request_state();
    return terminalized.error ==
               Sm87MacroFeedV4P40ExecutionPackageError::kExecutionEvent &&
           terminalized.event_status.error ==
               sm87_macrofeed_v4_execution_events_detail::
                   Sm87MacroFeedV4ExecutionError::kCudaSubmission &&
           snapshot.state ==
               sm87_macrofeed_v4_execution_events_detail::
                   Sm87MacroFeedV4ExecutionOwnerState::kPoisoned &&
           snapshot.poison_drain_all_stream_synchronizations_attempted &&
           snapshot.poisoned_terminal_quiescence_attested && aborted;
  }

  [[nodiscard]] static bool seed(
      Sm87MacroFeedV4P40ExecutionPackage& package) noexcept {
    if (package.ping_ == nullptr || package.pong_ == nullptr ||
        package.scratch_ == nullptr) {
      return false;
    }
    if (cudaMemset(package.ping_, 0x3f,
                   kernels::kSm87MacroFeedV4NormResidualHiddenBytes) !=
            cudaSuccess ||
        cudaMemset(package.pong_, 0,
                   kernels::kSm87MacroFeedV4NormResidualHiddenBytes) !=
            cudaSuccess ||
        cudaMemset(package.scratch_, 0,
                   kernels::kSm87MacroFeedV4Bf16AbScratchBytes) !=
            cudaSuccess) {
      return false;
    }
    // The package streams are deliberately non-blocking.  Establish fixture
    // initialization physically before handing execution to those private
    // streams instead of relying on legacy-default-stream ordering.
    return cudaDeviceSynchronize() == cudaSuccess;
  }

  [[nodiscard]] static bool read_samples(
      const Sm87MacroFeedV4P40ExecutionPackage& package,
      Samples* const samples) noexcept {
    if (samples == nullptr || package.pong_ == nullptr ||
        package.scratch_ == nullptr) {
      return false;
    }
    *samples = {};
    return cudaMemcpy(&samples->normalized, package.pong_,
                      sizeof(samples->normalized),
                      cudaMemcpyDeviceToHost) == cudaSuccess &&
           cudaMemcpy(&samples->projection_a,
                      package.scratch_ +
                          kernels::kSm87MacroFeedV4Bf16AbAOffset,
                      sizeof(samples->projection_a),
                      cudaMemcpyDeviceToHost) == cudaSuccess &&
           cudaMemcpy(&samples->projection_b,
                      package.scratch_ +
                          kernels::kSm87MacroFeedV4Bf16AbBOffset,
                      sizeof(samples->projection_b),
                      cudaMemcpyDeviceToHost) == cudaSuccess;
  }

  [[nodiscard]] static bool read_request_outcome(
      const Sm87MacroFeedV4P40ExecutionPackage& package,
      RequestOutcome* const outcome) noexcept {
    if (outcome == nullptr || package.request_state_ == nullptr) {
      return false;
    }
    const auto snapshot = package.request_state_->snapshot();
    outcome->phase = snapshot.phase;
    outcome->canonical_state_published =
        snapshot.canonical_state_published;
    outcome->logical_sequence_fence_published =
        snapshot.logical_sequence_fence_published;
    outcome->decode_access_issued = snapshot.decode_access_issued;
    return true;
  }
};

}  // namespace q3x::runtime::sm87_macrofeed_v4_p40_execution_detail

namespace {

namespace execution =
    q3x::runtime::sm87_macrofeed_v4_p40_execution_detail;
namespace events =
    q3x::runtime::sm87_macrofeed_v4_execution_events_detail;
namespace bound_launch =
    q3x::kernels::sm87_macrofeed_v4_bound_launch_detail;
namespace startup =
    q3x::runtime::sm87_macrofeed_v4_p40_startup_package_detail;
namespace target_aot =
    q3x::runtime::target_aot_complete_execution_detail;
namespace kernels = q3x::kernels;

using Access = target_aot::Sm87TargetAotCompleteProjectionExecutionAccess;
using Bf16AbPair = target_aot::Sm87TargetAotCompleteHostTestBf16AbPair;
using LayerNormPair =
    target_aot::Sm87TargetAotCompleteHostTestLayerNormPair;
using Owner = q3x::runtime::Sm87TargetAotCompleteProjectionDeviceAssets;
using q3x::runtime::Bf16LinearWeight;
using q3x::runtime::Bf16VectorWeight;
using q3x::runtime::ModelWeights;

void require_test(const bool condition, const std::string_view message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

class LiveExecutionWeights final {
 public:
  LiveExecutionWeights() = default;
  LiveExecutionWeights(const LiveExecutionWeights&) = delete;
  LiveExecutionWeights& operator=(const LiveExecutionWeights&) = delete;

  ~LiveExecutionWeights() {
    if (layer_norm_allocation_ != nullptr) {
      (void)cudaFree(layer_norm_allocation_);
      layer_norm_allocation_ = nullptr;
    }
    if (bf16_ab_allocation_ != nullptr) {
      (void)cudaFree(bf16_ab_allocation_);
      bf16_ab_allocation_ = nullptr;
    }
  }

  [[nodiscard]] bool allocate() noexcept {
    if (bf16_ab_allocation_ != nullptr ||
        layer_norm_allocation_ != nullptr) {
      return false;
    }
    if (cudaMalloc(&bf16_ab_allocation_, kBf16AbAllocationBytes) !=
            cudaSuccess ||
        bf16_ab_allocation_ == nullptr) {
      bf16_ab_allocation_ = nullptr;
      return false;
    }
    if (cudaMemset(bf16_ab_allocation_, 0, kBf16AbAllocationBytes) !=
        cudaSuccess) {
      return false;
    }
    if (cudaMalloc(&layer_norm_allocation_, kLayerNormAllocationBytes) !=
            cudaSuccess ||
        layer_norm_allocation_ == nullptr) {
      layer_norm_allocation_ = nullptr;
      return false;
    }
    if (cudaMemset(layer_norm_allocation_, 0,
                   kLayerNormAllocationBytes) != cudaSuccess) {
      return false;
    }

    auto* const bf16_bytes =
        static_cast<std::uint8_t*>(bf16_ab_allocation_);
    for (std::size_t ordinal = 0U; ordinal < bf16_ab_pairs_.size();
         ++ordinal) {
      bf16_ab_pairs_[ordinal].a = Bf16LinearWeight{
          reinterpret_cast<const std::uint16_t*>(
              bf16_bytes + (2U * ordinal) * kBf16AbWeightBytes),
          kernels::kSm87MacroFeedV4Bf16AbRowsPerProjection,
          kernels::kSm87MacroFeedV4Bf16AbInputFeatures};
      bf16_ab_pairs_[ordinal].b = Bf16LinearWeight{
          reinterpret_cast<const std::uint16_t*>(
              bf16_bytes + (2U * ordinal + 1U) * kBf16AbWeightBytes),
          kernels::kSm87MacroFeedV4Bf16AbRowsPerProjection,
          kernels::kSm87MacroFeedV4Bf16AbInputFeatures};
    }

    auto* const norm_bytes =
        static_cast<std::uint8_t*>(layer_norm_allocation_);
    for (std::size_t layer = 0U; layer < layer_norm_pairs_.size(); ++layer) {
      layer_norm_pairs_[layer].input_layernorm = Bf16VectorWeight{
          reinterpret_cast<const std::uint16_t*>(
              norm_bytes + (2U * layer) * kLayerNormWeightBytes),
          kernels::kSm87MacroFeedV4NormResidualHidden};
      layer_norm_pairs_[layer].post_attention_layernorm = Bf16VectorWeight{
          reinterpret_cast<const std::uint16_t*>(
              norm_bytes + (2U * layer + 1U) * kLayerNormWeightBytes),
          kernels::kSm87MacroFeedV4NormResidualHidden};
    }
    return true;
  }

  [[nodiscard]] bool install(ModelWeights& model_weights) const noexcept {
    return bf16_ab_allocation_ != nullptr &&
           layer_norm_allocation_ != nullptr &&
           Access::install_complete_host_test_bf16_ab_pairs(
               model_weights, bf16_ab_pairs_.data(),
               bf16_ab_pairs_.size()) &&
           Access::install_complete_host_test_layer_norm_pairs(
               model_weights, layer_norm_pairs_.data(),
               layer_norm_pairs_.size());
  }

  [[nodiscard]] bool install_one_past_final_post_norm(
      ModelWeights& model_weights) noexcept {
    if (layer_norm_allocation_ == nullptr) {
      return false;
    }
    saved_final_post_norm_ = layer_norm_pairs_.back().post_attention_layernorm;
    layer_norm_pairs_.back().post_attention_layernorm = Bf16VectorWeight{
        reinterpret_cast<const std::uint16_t*>(
            static_cast<const std::uint8_t*>(layer_norm_allocation_) +
            kLayerNormAllocationBytes),
        kernels::kSm87MacroFeedV4NormResidualHidden};
    final_post_norm_poisoned_ = true;
    return Access::install_complete_host_test_layer_norm_pairs(
        model_weights, layer_norm_pairs_.data(), layer_norm_pairs_.size());
  }

  [[nodiscard]] bool restore_final_post_norm(
      ModelWeights& model_weights) noexcept {
    if (!final_post_norm_poisoned_) {
      return false;
    }
    layer_norm_pairs_.back().post_attention_layernorm = saved_final_post_norm_;
    saved_final_post_norm_ = {};
    final_post_norm_poisoned_ = false;
    return Access::install_complete_host_test_layer_norm_pairs(
        model_weights, layer_norm_pairs_.data(), layer_norm_pairs_.size());
  }

 private:
  static constexpr std::size_t kBf16AbWeightBytes =
      kernels::kSm87MacroFeedV4Bf16AbWeightBytes;
  static constexpr std::size_t kBf16AbAllocationBytes =
      2U * q3x::runtime::kQwen36LinearAttentionLayerCount *
      kBf16AbWeightBytes;
  static constexpr std::size_t kLayerNormWeightBytes =
      kernels::kSm87MacroFeedV4NormResidualWeightBytes;
  static constexpr std::size_t kLayerNormAllocationBytes =
      2U * q3x::runtime::kQwen36DenseLayerCount *
      kLayerNormWeightBytes;

  static_assert(kBf16AbAllocationBytes == 47'185'920U);
  static_assert(kLayerNormAllocationBytes == 1'310'720U);
  static_assert(kBf16AbWeightBytes % 256U == 0U);
  static_assert(kLayerNormWeightBytes % 256U == 0U);

  void* bf16_ab_allocation_ = nullptr;
  void* layer_norm_allocation_ = nullptr;
  std::array<Bf16AbPair, q3x::runtime::kQwen36LinearAttentionLayerCount>
      bf16_ab_pairs_{};
  std::array<LayerNormPair, q3x::runtime::kQwen36DenseLayerCount>
      layer_norm_pairs_{};
  Bf16VectorWeight saved_final_post_norm_{};
  bool final_post_norm_poisoned_ = false;
};

[[nodiscard]] bool exact_sm87_device_available() noexcept {
  int device = -1;
  cudaDeviceProp properties{};
  return cudaGetDevice(&device) == cudaSuccess && device == 0 &&
         cudaGetDeviceProperties(&properties, device) == cudaSuccess &&
         properties.major == 8 && properties.minor == 7 &&
         properties.multiProcessorCount == 16;
}

[[nodiscard]] bool clear_fixture(std::optional<ModelWeights>& model_weights,
                                 Owner& owner) noexcept {
  model_weights.reset();
  return Access::clear_host_test_fixture(owner) && owner.empty();
}

void test_real_cuda_front_half() {
  static_assert(
      execution::kSm87MacroFeedV4P40ExecutionPackageCompiled);
  static_assert(startup::kSm87MacroFeedV4P40StartupPackageCompiled);
  static_assert(!std::is_default_constructible_v<
                execution::Sm87MacroFeedV4P40ExecutionPackage>);
  static_assert(!std::is_copy_constructible_v<
                execution::Sm87MacroFeedV4P40ExecutionPackage>);
  static_assert(!std::is_move_constructible_v<
                execution::Sm87MacroFeedV4P40ExecutionPackage>);
  static_assert(!std::is_default_constructible_v<
                bound_launch::Sm87MacroFeedV4LockedSubmitToken>);
  static_assert(!std::is_copy_constructible_v<
                bound_launch::Sm87MacroFeedV4LockedSubmitToken>);
  static_assert(!std::is_move_constructible_v<
                bound_launch::Sm87MacroFeedV4LockedSubmitToken>);
  static_assert(!std::is_default_constructible_v<
                events::Sm87MacroFeedV4ExecutionEventsDriver>);
  static_assert(!std::is_copy_constructible_v<
                events::Sm87MacroFeedV4ExecutionEventsDriver>);
  static_assert(!std::is_move_constructible_v<
                events::Sm87MacroFeedV4ExecutionEventsDriver>);

  Owner owner;
  LiveExecutionWeights live_weights;
  require_test(live_weights.allocate(),
               "could not allocate real BF16/LayerNorm CUDA fixtures");
  std::optional<ModelWeights> model_weights =
      Access::make_complete_host_test_fixture(owner);
  require_test(model_weights.has_value(),
               "could not construct complete target-AOT fixture");
  require_test(live_weights.install(*model_weights),
               "could not install real execution weights");

  require_test(
      live_weights.install_one_past_final_post_norm(*model_weights),
      "could not install one-past LayerNorm negative fixture");
  auto invalid_startup =
      startup::Sm87MacroFeedV4P40StartupPackage::create(*model_weights);
  require_test(static_cast<bool>(invalid_startup),
               "negative fixture could not create its startup package");
  auto invalid_execution =
      execution::Sm87MacroFeedV4P40ExecutionPackage::create(
          *invalid_startup.package);
  require_test(
      !invalid_execution && invalid_execution.package == nullptr &&
          invalid_execution.status.error ==
              execution::Sm87MacroFeedV4P40ExecutionPackageError::
                  kLayerNormCatalog &&
          invalid_execution.status.layer ==
              q3x::runtime::kQwen36DenseLayerCount - 1U &&
          invalid_execution.status.post_attention_norm &&
          invalid_execution.status.cuda_error != 0,
      "one-past LayerNorm device range did not fail closed");
  invalid_startup.package.reset();
  require_test(live_weights.restore_final_post_norm(*model_weights),
               "could not restore final LayerNorm binding");

  auto startup_created =
      startup::Sm87MacroFeedV4P40StartupPackage::create(*model_weights);
  if (!startup_created) {
    std::cerr << "startup package error="
              << static_cast<unsigned>(startup_created.status.error)
              << " context=" << startup_created.status.context
              << " cuda=" << startup_created.status.cuda_error << '\n';
  }
  require_test(static_cast<bool>(startup_created),
               "actual-kernel startup package creation failed");

  auto terminal_drain_created =
      execution::Sm87MacroFeedV4P40ExecutionPackage::create(
          *startup_created.package);
  require_test(static_cast<bool>(terminal_drain_created),
               "terminal-drain execution package creation failed");
  require_test(
      execution::Sm87MacroFeedV4P40ExecutionPackageCudaTestFixture::
          exercise_terminal_poison_drain(*terminal_drain_created.package),
      "tail-stage CUDA poison did not immediately drain all private streams");
  terminal_drain_created.package.reset();

  auto execution_created =
      execution::Sm87MacroFeedV4P40ExecutionPackage::create(
          *startup_created.package);
  if (!execution_created) {
    std::cerr << "execution package error="
              << static_cast<unsigned>(execution_created.status.error)
              << " context=" << execution_created.status.context
              << " layer=" << execution_created.status.layer
              << " post_norm="
              << execution_created.status.post_attention_norm
              << " cuda=" << execution_created.status.cuda_error << '\n';
  }
  require_test(static_cast<bool>(execution_created),
               "real CUDA execution package creation failed");

  const auto& audit = execution_created.audit;
  require_test(
      audit.valid() && audit.fixed_gdn_layer0_front_half_bound &&
          !audit.whole_layer_executor_bound &&
          !audit.whole_model_executor_bound && !audit.selector_bound &&
          !audit.api_route_bound && audit.default_off &&
          !audit.jit_present && !audit.request_time_repack_present &&
          !audit.request_time_autotune_present && !audit.fallback_present &&
          !audit.cublaslt_present && !audit.mtp_present &&
          !audit.production_dispatch_eligible,
      "execution audit overstated the layer-0 front-half slice");

  require_test(
      execution::Sm87MacroFeedV4P40ExecutionPackageCudaTestFixture::seed(
          *execution_created.package),
      "could not seed private execution-package buffers");
  const auto front_half =
      execution::Sm87MacroFeedV4P40ExecutionPackageCudaTestFixture::
          execute_once(*execution_created.package);
  if (!front_half) {
    std::cerr << "front-half error="
              << static_cast<unsigned>(front_half.status.error)
              << " context=" << front_half.status.context
              << " cuda=" << front_half.status.cuda_error << '\n';
  }
  require_test(static_cast<bool>(front_half) && front_half.receipt.valid(),
               "real CUDA front-half execution did not close its receipt");
  require_test(
      front_half.receipt.input_norm_launches == 1U &&
          front_half.receipt.bf16_ab_launches == 1U &&
          front_half.receipt.bound_kernel_submissions == 2U &&
          front_half.receipt.physical_completion_receipts == 1U &&
          front_half.receipt.norm_ready_recorded &&
          front_half.receipt.norm_ready_waited_by_ab &&
          front_half.receipt.ab_ready_recorded &&
          !front_half.receipt.ab_ready_waited_by_main &&
          front_half.receipt.owner_drained_physically &&
          front_half.receipt.request_discarded_without_publication &&
          front_half.receipt.gdn_layer0_front_half_only &&
          !front_half.receipt.layer_complete &&
          !front_half.receipt.panel_complete &&
          !front_half.receipt.model_complete &&
          !front_half.receipt.production_dispatch_eligible,
      "front-half receipt overstated ordering, completion, or authority");

  execution::Sm87MacroFeedV4P40ExecutionPackageCudaTestFixture::Samples
      samples{};
  require_test(
      execution::Sm87MacroFeedV4P40ExecutionPackageCudaTestFixture::
          read_samples(*execution_created.package, &samples),
      "could not read physically drained CUDA samples");
  require_test(samples.normalized != 0U,
               "input normalization did not publish a nonzero sample");
  require_test(samples.projection_a == 0U && samples.projection_b == 0U,
               "zero BF16 A/B weights produced a nonzero projection");

  execution::Sm87MacroFeedV4P40ExecutionPackageCudaTestFixture::
      RequestOutcome request_outcome{};
  require_test(
      execution::Sm87MacroFeedV4P40ExecutionPackageCudaTestFixture::
          read_request_outcome(*execution_created.package,
                               &request_outcome),
      "could not inspect the private request-state outcome");
  require_test(
      request_outcome.phase ==
              q3x::runtime::Sm87MacroFeedV4RequestStatePhase::kFailed &&
          !request_outcome.canonical_state_published &&
          !request_outcome.logical_sequence_fence_published &&
          !request_outcome.decode_access_issued,
      "front-half admission published canonical or Decode-visible state");

  const auto repeated =
      execution::Sm87MacroFeedV4P40ExecutionPackageCudaTestFixture::
          execute_once(*execution_created.package);
  require_test(
      !repeated &&
          repeated.status.error ==
              execution::Sm87MacroFeedV4P40ExecutionPackageError::
                  kAlreadyExecuted &&
          !repeated.receipt.valid(),
      "one-shot front-half package accepted a repeated execution");

  execution_created.package.reset();
  auto startup_independent_snapshot =
      execution::Sm87MacroFeedV4P40ExecutionPackage::create(
          *startup_created.package);
  require_test(static_cast<bool>(startup_independent_snapshot),
               "startup-independent projection snapshot creation failed");
  startup_created.package.reset();
  require_test(startup_independent_snapshot.package->valid(),
               "execution package retained startup-owned catalog pointers");
  startup_independent_snapshot.package.reset();
  require_test(clear_fixture(model_weights, owner),
               "complete target-AOT fixture cleanup failed");
}

}  // namespace

int main() {
  if (!exact_sm87_device_available()) {
    std::cout << "sm87_macrofeed_v4_p40_execution_package_cuda_test: "
                 "SKIP (requires device 0 with SM87 and exactly 16 SMs)\n";
    return 77;
  }
  test_real_cuda_front_half();
  std::cout <<
      "sm87_macrofeed_v4_p40_execution_package_cuda_test: PASS\n";
  return 0;
}
