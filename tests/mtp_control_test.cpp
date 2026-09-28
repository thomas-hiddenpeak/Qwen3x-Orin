#include "runtime/mtp_control.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {
namespace mtp = q3x::runtime::mtp_detail;
int checks = 0;
void require(bool condition) {
  ++checks;
  if (!condition) {
    std::cerr << "MTP controller assertion " << checks << " failed\n";
    std::exit(1);
  }
}

// A stateful scalar oracle, deliberately including recurrent, convolution and
// history dependencies. Speculative execution mutates all three, so merely
// restoring a length cannot pass the subsequent-round oracle.
struct State {
  std::uint64_t recurrent = 123;
  std::array<std::uint32_t, 3> conv{3, 5, 7};
  std::vector<std::uint32_t> kv{42, 31, 97};
  bool operator==(const State& other) const {
    return recurrent == other.recurrent && conv == other.conv && kv == other.kv;
  }
};

void consume(State& state, std::uint32_t token) {
  state.recurrent = state.recurrent * 6364136223846793005ULL + token +
                    state.conv[0] * 31ULL + state.kv[state.kv.size() / 2];
  state.conv = {state.conv[1], state.conv[2], token};
  state.kv.push_back(token);
}
std::uint32_t predict(const State& state) {
  return static_cast<std::uint32_t>((state.recurrent >> 16) % 65535);
}

enum class Fault { kNone, kBegin, kPropose, kVerify, kCommit, kFinish,
                   kDraftId, kTargetId, kAbort };

class Backend final : public mtp::RoundBackend {
 public:
  State state;
  State draft_state;
  State entry;
  std::array<State, 4> prefixes;
  std::uint32_t mismatch = 3;
  std::uint32_t pending = 0;
  std::uint32_t staged_rows = 0;
  std::uint32_t commit_calls = 0;
  std::uint32_t fail_commit_at = 1;
  unsigned begin_calls = 0;
  unsigned propose_calls = 0;
  unsigned abort_calls = 0;
  bool active = false;
  bool poisoned = false;
  Fault fault = Fault::kNone;

  bool begin(std::uint32_t rows) noexcept override {
    ++begin_calls;
    require(!active && !poisoned && rows >= 1 && rows <= 4);
    active = true;
    entry = state;
    commit_calls = 0;
    staged_rows = rows;
    return fault != Fault::kBegin;
  }
  bool propose(std::uint32_t seed, std::uint32_t count,
               std::uint32_t* draft) noexcept override {
    ++propose_calls;
    if (fault == Fault::kPropose) return false;
    State cursor = entry;
    for (std::uint32_t i = 0; i < count; ++i) {
      consume(cursor, i == 0 ? seed : draft[i - 1]);
      draft[i] = predict(cursor);
      if (i == mismatch) draft[i] = (draft[i] + 1) % 65535;
    }
    if (fault == Fault::kDraftId) draft[count - 1] = 65536;
    draft_state = cursor;  // Deliberately dirty until commit reconciliation.
    return true;
  }
  bool verify(std::uint32_t seed, const std::uint32_t* draft,
              std::uint32_t count, std::uint32_t* predictions) noexcept override {
    State cursor = entry;
    for (std::uint32_t i = 0; i <= count; ++i) {
      consume(cursor, i == 0 ? seed : draft[i - 1]);
      prefixes[i] = cursor;
      predictions[i] = predict(cursor);
    }
    state = cursor;  // Mutated rejected state must actually be restored.
    if (fault == Fault::kTargetId) predictions[count] = 65536;
    return fault != Fault::kVerify && fault != Fault::kAbort;
  }
  bool commit_prefix(std::uint32_t rows,
                     std::uint32_t pending_token) noexcept override {
    ++commit_calls;
    if (fault == Fault::kCommit && commit_calls == fail_commit_at) return false;
    require(rows == commit_calls && rows <= staged_rows);
    state = prefixes[rows - 1];
    draft_state = state;
    pending = pending_token;
    require(pending == predict(state));
    return true;
  }
  bool finish() noexcept override {
    require(active);
    if (fault == Fault::kFinish) return false;
    active = false;
    return true;
  }
  bool abort() noexcept override {
    ++abort_calls;
    active = false;
    poisoned = true;
    return fault != Fault::kAbort;
  }
};

mtp::RoundOptions options(const Backend& backend, std::uint32_t length,
                          std::uint32_t remaining) {
  return {length, predict(backend.state), remaining, remaining, 65536, 65535};
}

struct Observer {
  Backend* backend;
  std::uint32_t seen = 0;
  std::uint32_t cancel_after = 100;
  static bool call(void* ptr, std::uint32_t token) noexcept {
    auto& self = *static_cast<Observer*>(ptr);
    ++self.seen;
    require(self.backend->pending == token);
    require(self.backend->state.kv.size() ==
            self.backend->entry.kv.size() + self.seen);
    require(self.backend->state == self.backend->draft_state);
    return self.seen < self.cancel_after;
  }
};

void replay() {
  for (std::uint32_t length : {2U, 3U}) {
    for (std::uint32_t mismatch = 0; mismatch <= length; ++mismatch) {
      for (std::uint32_t total = 1; total <= 65; ++total) {
        Backend backend;
        backend.mismatch = mismatch;
        State scalar = backend.state;
        std::uint32_t seed = predict(scalar);
        std::uint32_t remaining = total;
        while (remaining != 0) {
          const auto opts = options(backend, length, remaining);
          Observer observer{&backend};
          const auto result = mtp::run_round(opts, backend, Observer::call, &observer);
          require(result.ok() && result.output_count > 0);
          require(result.output_count == observer.seen);
          require(result.output_count == result.committed_target_rows);
          require(result.verified_rows == result.proposed_tokens + 1);
          const auto accepted = std::min(mismatch, result.proposed_tokens);
          require(result.accepted_tokens == accepted);
          require(result.output_count == accepted + 1);
          for (std::uint32_t i = 0; i < result.output_count; ++i) {
            consume(scalar, seed);
            seed = predict(scalar);
            require(seed == result.output[i]);
          }
          require(backend.state == scalar && backend.draft_state == scalar);
          remaining -= result.output_count;
          require(result.status == (remaining == 0 ? mtp::RoundStatus::kLength
                                                    : mtp::RoundStatus::kContinue));
          require(!backend.active && !backend.poisoned);
        }
      }
    }
  }
}

void termination() {
  for (std::uint32_t length : {2U, 3U}) {
    for (std::uint32_t mismatch = 0; mismatch <= length; ++mismatch) {
      for (std::uint32_t at = 1; at <= mismatch + 1; ++at) {
        for (const bool eos : {false, true}) {
          Backend backend;
          backend.mismatch = mismatch;
          auto opts = options(backend, length, 10);
          State scalar = backend.state;
          std::uint32_t seed = opts.seed_token;
          for (std::uint32_t i = 0; i < at; ++i) {
            consume(scalar, seed);
            seed = predict(scalar);
          }
          if (eos) opts.stop_token = seed;
          // False on EOS verifies that stop takes precedence over cancellation.
          Observer observer{&backend, 0, at};
          const auto result = mtp::run_round(opts, backend, Observer::call, &observer);
          require(result.ok() && result.output_count == at);
          require(result.status == (eos ? mtp::RoundStatus::kStop
                                        : mtp::RoundStatus::kCancelled));
          require(backend.state == scalar && backend.draft_state == scalar);
          require(backend.pending == seed && !backend.active);
          require(result.accepted_tokens == std::min(at, mismatch));
        }
      }
    }
  }
}

void failures() {
  for (Fault fault : {Fault::kBegin, Fault::kPropose, Fault::kVerify,
                      Fault::kCommit, Fault::kFinish, Fault::kDraftId,
                      Fault::kTargetId, Fault::kAbort}) {
    for (std::uint32_t commit = 1; commit <= 4; ++commit) {
      Backend backend;
      backend.fault = fault;
      backend.fail_commit_at = commit;
      Observer observer{&backend};
      auto result = mtp::run_round(options(backend, 3, 10), backend,
                                    Observer::call, &observer);
      require(!result.ok() && backend.poisoned && !backend.active);
      require(backend.abort_calls == 1);
      require(result.abort_succeeded == (fault != Fault::kAbort));
      require(observer.seen == result.output_count);
      if (fault == Fault::kCommit) require(observer.seen == commit - 1);
      if (fault == Fault::kFinish) require(observer.seen == 4);
    }
  }
  Backend tail;
  auto result = mtp::run_round(options(tail, 3, 1), tail);
  require(result.ok() && tail.propose_calls == 0 && result.proposed_tokens == 0);
  for (unsigned invalid = 0; invalid < 9; ++invalid) {
    Backend backend;
    auto opts = options(backend, 2, 10);
    switch (invalid) {
      case 0: opts.draft_length = 0; break;
      case 1: opts.draft_length = 1; break;
      case 2: opts.draft_length = 4; break;
      case 3: opts.vocabulary_size = 0; break;
      case 4: opts.seed_token = 65536; break;
      case 5: opts.stop_token = 65536; break;
      case 6: opts.seed_token = opts.stop_token; break;
      case 7: opts.remaining_output = 0; break;
      case 8: opts.available_target_rows = 9; break;
    }
    result = mtp::run_round(opts, backend);
    require(result.status == mtp::RoundStatus::kInvalidOptions);
    require(backend.begin_calls == 0 && backend.abort_calls == 0);
  }
  Backend backend;
  result = mtp::run_round(options(backend, 2, 10), backend, nullptr, &backend);
  require(result.status == mtp::RoundStatus::kInvalidOptions);
  require(backend.begin_calls == 0);
}
}  // namespace

int main() {
  replay();
  termination();
  failures();
  std::cout << "mtp_control_checks=" << checks << "\n";
}
