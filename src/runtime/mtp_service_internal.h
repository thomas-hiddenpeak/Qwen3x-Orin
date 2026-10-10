#pragma once
#include "mtp_device_internal.h"
#include "q3x/runtime/reference_engine.h"
#include <chrono>
#include <string>

namespace q3x::runtime::mtp_detail {
inline constexpr std::uint32_t kServicePromptLimit = 44095;
inline constexpr std::uint32_t kServiceOutputLimit = 4096;
struct ServiceReport {
  std::array<std::uint64_t, 4> proposal_histogram{};
  std::uint32_t draft_length = 0, prompt_rows = 0, draft_rows = 0, rounds = 0;
  std::uint32_t proposed = 0, accepted = 0, verified = 0, committed = 0;
  std::array<std::uint32_t, 3> accepted_by_position{};
  double initialization_ms = 0, draft_ms = 0, verify_ms = 0, reconcile_ms = 0;
  bool initialized = false, cancelled = false, failed = false;
};
class Service {
 public:
  Service(const std::filesystem::path&, const ModelWeights&, ReferenceRunner&, RequestState&);
  void reset_report(bool (*cancel)(void*) noexcept, void* context) noexcept;
  ReferenceRunnerStatus initialize(const std::vector<std::uint32_t>&) noexcept;
  ReferenceRunnerStatus decode(reference_engine_detail::GenerationControl&,
                              const reference_engine_detail::GenerationControlOptions&);
  const ServiceReport& report() const noexcept { return report_; }
  std::string report_json() const;
 private:
  Weights weights_;
  Draft draft_;
  TargetTransaction transaction_;
  RequestState& state_;
  std::uint32_t length_;
  std::size_t startup_free_bytes_ = 0;
  ServiceReport report_{};
  bool (*cancel_)(void*) noexcept = nullptr;
  void* cancel_context_ = nullptr;
};
// The owning engine installs this only around its serialized control call.
Service* active_service() noexcept;
class ServiceScope {
 public:
  explicit ServiceScope(Service*) noexcept;
  ~ServiceScope();
 private:
  Service* previous_;
};
}
