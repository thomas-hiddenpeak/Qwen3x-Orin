#include "mtp_service_internal.h"
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <sstream>
#include <iomanip>

namespace q3x::runtime::mtp_detail {
namespace {
using Clock = std::chrono::steady_clock;
thread_local Service* current = nullptr;
double elapsed(Clock::time_point start) noexcept {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
std::uint32_t configured_length() {
  const char* value = std::getenv("Q3X_MTP_DRAFT_LENGTH");
  if (!value || (std::string(value) != "2" && std::string(value) != "3"))
    throw std::runtime_error("MTP admission requires Q3X_MTP_DRAFT_LENGTH=2 or 3");
  return static_cast<std::uint32_t>(value[0] - '0');
}
ReferenceRunnerStatus failure(bool cancelled) noexcept {
  ReferenceRunnerStatus status;
  status.error = cancelled ? ReferenceRunnerError::kCancelled : ReferenceRunnerError::kPoisoned;
  status.operation = cancelled ? "mtp_cancelled" : "mtp_transaction_failure";
  return status;
}
struct TimedBackend final : RoundBackend {
  RoundBackend& inner;
  ServiceReport& report;
  TimedBackend(RoundBackend& b, ServiceReport& r) : inner(b), report(r) {}
  bool begin(std::uint32_t rows) noexcept override { return inner.begin(rows); }
  bool propose(std::uint32_t seed, std::uint32_t count, std::uint32_t* ids) noexcept override {
    const auto start = Clock::now(); const bool ok = inner.propose(seed, count, ids);
    report.draft_ms += elapsed(start); return ok;
  }
  bool verify(std::uint32_t seed, const std::uint32_t* ids, std::uint32_t count,
              std::uint32_t* predictions) noexcept override {
    const auto start = Clock::now(); const bool ok = inner.verify(seed, ids, count, predictions);
    report.verify_ms += elapsed(start); return ok;
  }
  bool commit_prefix(std::uint32_t rows, std::uint32_t pending) noexcept override {
    const auto start = Clock::now(); const bool ok = inner.commit_prefix(rows, pending);
    report.reconcile_ms += elapsed(start); return ok;
  }
  bool finish() noexcept override { return inner.finish(); }
  bool abort() noexcept override { return inner.abort(); }
};
}
Service* active_service() noexcept { return current; }
ServiceScope::ServiceScope(Service* value) noexcept : previous_(current) { current = value; }
ServiceScope::~ServiceScope() { current = previous_; }
Service::Service(const std::filesystem::path& path, const ModelWeights& model,
                 ReferenceRunner& runner, RequestState& state)
    : weights_(path), draft_(weights_, model, state.max_sequence_length(),
        TargetTransaction::cosines(runner), TargetTransaction::sines(runner)),
      transaction_(runner, state, draft_, true), state_(state), length_(configured_length()) {
  std::size_t total = 0;
  if (cudaMemGetInfo(&startup_free_bytes_, &total) != cudaSuccess ||
      startup_free_bytes_ < 8ULL * 1024 * 1024 * 1024)
    throw std::runtime_error("MTP composed startup free-memory reserve");
}
void Service::reset_report(bool (*cancel)(void*) noexcept, void* context) noexcept {
  report_ = {}; report_.draft_length = length_; cancel_ = cancel; cancel_context_ = context;
}
ReferenceRunnerStatus Service::initialize(const std::vector<std::uint32_t>& prompt) noexcept {
  const auto start = Clock::now();
  const bool ok = transaction_.initialize_whole_core_prefill(
      prompt.data(), static_cast<std::uint32_t>(prompt.size()), cancel_, cancel_context_, true);
  report_.initialization_ms = elapsed(start);
  report_.prompt_rows = static_cast<std::uint32_t>(prompt.size());
  report_.draft_rows = draft_.position();
  report_.initialized = ok;
  if (!ok) {
    report_.cancelled = cancel_ && cancel_(cancel_context_);
    report_.failed = !report_.cancelled;
    return failure(report_.cancelled);
  }
  return {};
}
ReferenceRunnerStatus Service::decode(reference_engine_detail::GenerationControl& control,
    const reference_engine_detail::GenerationControlOptions& options) {
  const auto decode_start = Clock::now();
  control.timing.subsequent_token_milliseconds.reserve(options.max_new_tokens);
  struct Observer {
    Service& service;
    reference_engine_detail::GenerationControl& control;
    const reference_engine_detail::GenerationControlOptions& options;
    Clock::time_point last;
    static bool emit(void* opaque, std::uint32_t token) noexcept {
      auto& self = *static_cast<Observer*>(opaque);
      auto& c = self.control;
      const auto now = Clock::now();
      const double ms = std::chrono::duration<double, std::milli>(now - self.last).count();
      self.last = now;
      ReferenceStepResult step;
      step.position = self.service.state_.current_position() - 1;
      step.input_token_id = c.generated_token_ids.back();
      step.prediction = ReferenceStepPrediction{token};
      step.timing = ReferenceStepTiming{ms};
      c.steps.push_back(step);  // All vectors are reserved before entry.
      const auto index = c.generated_token_ids.size();
      c.generated_token_ids.push_back(token);
      c.timing.subsequent_token_milliseconds.push_back(ms);
      c.timing.decode_after_first_milliseconds += ms;
      c.timing.total_generation_milliseconds += ms;
      ++self.service.report_.committed;
      return !self.options.committed_token || self.options.committed_token(
          self.options.committed_token_context, token, index, ms);
    }
  } observer{*this, control, options, Clock::now()};
  TimedBackend backend(transaction_, report_);
  while (control.generated_token_ids.size() < options.max_new_tokens) {
    if (cancel_ && cancel_(cancel_context_)) {
      control.stop_reason = ReferenceStopReason::kCancelled; report_.cancelled = true; break;
    }
    RoundOptions round;
    round.draft_length = length_; round.seed_token = control.generated_token_ids.back();
    round.remaining_output = options.max_new_tokens - control.generated_token_ids.size();
    round.available_target_rows = state_.max_sequence_length() - state_.current_position();
    round.vocabulary_size = kReferenceVocabularySize; round.stop_token = options.stop_token_id;
    const auto result = run_round(round, backend, Observer::emit, &observer);
    ++report_.rounds; report_.proposed += result.proposed_tokens;
    report_.accepted += result.accepted_tokens; report_.verified += result.verified_rows;
    for (std::uint32_t i = 0; i < result.accepted_tokens; ++i) ++report_.accepted_by_position[i];
    if (!result.ok()) { report_.failed = true; return failure(false); }
    if (result.status == RoundStatus::kStop) { control.stop_reason = ReferenceStopReason::kImEnd; break; }
    if (result.status == RoundStatus::kCancelled) {
      control.stop_reason = ReferenceStopReason::kCancelled; report_.cancelled = true; break;
    }
  }
  control.timing.decode_after_first_milliseconds = elapsed(decode_start);
  control.timing.total_generation_milliseconds = control.timing.prompt_prefill_milliseconds +
      control.timing.decode_after_first_milliseconds;
  return {};
}
std::string Service::report_json() const {
  const auto& r = report_;
  std::ostringstream out; out << std::setprecision(17) << std::boolalpha;
  out << "{\"enabled\":true,\"verifier\":\"multirow-nv-live-reduction-v31\",\"draft_length\":" << r.draft_length
      << ",\"startup_free_bytes\":" << startup_free_bytes_
      << ",\"prompt_rows\":" << r.prompt_rows << ",\"draft_prefill_rows\":" << r.draft_rows
      << ",\"initialized\":" << r.initialized << ",\"rounds\":" << r.rounds
      << ",\"proposed\":" << r.proposed << ",\"accepted\":" << r.accepted
      << ",\"verified_rows\":" << r.verified << ",\"committed_decode_tokens\":" << r.committed
      << ",\"accepted_by_position\":[" << r.accepted_by_position[0] << ','
      << r.accepted_by_position[1] << ',' << r.accepted_by_position[2] << ']'
      << ",\"draft_ordered_attention_steps\":" << draft_.ordered_attention_steps()
      << ",\"draft_initialization_ms\":" << r.initialization_ms
      << ",\"draft_ms\":" << r.draft_ms << ",\"verify_ms\":" << r.verify_ms
      << ",\"reconcile_ms\":" << r.reconcile_ms
      << ",\"cancelled\":" << r.cancelled << ",\"failed\":" << r.failed << '}';
  return out.str();
}
}
