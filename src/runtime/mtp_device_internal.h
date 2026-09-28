#pragma once

#include "mtp_control.h"
#include "q3x/runtime/model_weights.h"
#include "q3x/runtime/reference_runner.h"

#include <array>
#include <filesystem>
#include <memory>

namespace q3x::runtime::mtp_detail {

// All definitions live exclusively in the non-installable test target.
class Weights {
 public:
  explicit Weights(const std::filesystem::path& model_directory);
  ~Weights();
  Weights(const Weights&) = delete;
  Weights& operator=(const Weights&) = delete;
  const std::uint16_t* tensor(std::size_t index) const noexcept;
 private:
  void* arena_ = nullptr;
  std::array<std::uint64_t, 15> offsets_{};
};

class Draft {
 public:
  Draft(const Weights&, const ModelWeights&, std::uint32_t capacity,
        const float* cosines, const float* sines);
  ~Draft();
  Draft(const Draft&) = delete;
  Draft& operator=(const Draft&) = delete;
  bool step(std::uint32_t token, const std::uint16_t* hidden,
            bool logits, std::uint32_t& prediction) noexcept;
  // Initialize only live K/V from shifted tokens and independent target
  // hidden rows. Requires empty state. No final hidden/logits are produced.
  // Cancellation poisons and drains; successful reset is required for reuse.
  bool initialize_kv(const std::uint32_t* shifted_tokens,
                     const std::uint16_t* target_hidden, std::uint32_t rows,
                     bool (*cancel)(void*) noexcept = nullptr,
                     void* cancel_context = nullptr) noexcept;
  bool rewind(std::uint32_t position) noexcept;
  bool reset() noexcept;
  bool poison() noexcept;
  std::uint32_t position() const noexcept;
  const std::uint16_t* hidden() const noexcept;
  const std::uint16_t* keys() const noexcept;
  const std::uint16_t* values() const noexcept;
  int error() const noexcept;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Isolated transaction with explicit scalar oracle or multi-row verifier.
// Retains every complete prefix and repairs draft KV using target hidden.
// Neither local mode alone has API acceleration or production authority.
class TargetTransaction final : public RoundBackend {
 public:
  TargetTransaction(ReferenceRunner&, RequestState&, Draft&, bool multirow = false);
  ~TargetTransaction();
  TargetTransaction(const TargetTransaction&) = delete;
  TargetTransaction& operator=(const TargetTransaction&) = delete;
  static const std::uint16_t* hidden(const ReferenceRunner&) noexcept;
  static const std::uint16_t* logits(const ReferenceRunner&) noexcept;
  static const float* cosines(const ReferenceRunner&) noexcept;
  static const float* sines(const ReferenceRunner&) noexcept;
  // After successful whole-core O1 generation, before any Decode. Captures
  // every normalized prompt row and builds shifted draft KV through P-2.
  // Caller must supply the exact prompt just consumed by the engine.
  bool initialize_whole_core_prefill(const std::uint32_t* prompt,
                                    std::uint32_t count,
                                    bool (*cancel)(void*) noexcept = nullptr,
                                    void* cancel_context = nullptr) noexcept;
  const std::uint16_t* prompt_hidden() const noexcept { return prompt_hidden_; }
  bool begin(std::uint32_t verify_rows) noexcept override;
  bool propose(std::uint32_t seed, std::uint32_t count,
               std::uint32_t* draft) noexcept override;
  bool verify(std::uint32_t seed, const std::uint32_t* draft,
              std::uint32_t count, std::uint32_t* predictions) noexcept override;
  bool commit_prefix(std::uint32_t rows, std::uint32_t pending) noexcept override;
  bool finish() noexcept override;
  bool abort() noexcept override;
 private:
  bool verify_multirow(std::uint32_t* predictions) noexcept;
  bool multirow_ = false;
  bool snapshot(std::uint32_t slot) noexcept;
  ReferenceRunner& target_;
  RequestState& state_;
  Draft& draft_;
  void* snapshots_ = nullptr;
  std::uint16_t* prompt_hidden_ = nullptr;
  std::uint64_t recurrent_offset_ = 0;
  std::uint64_t recurrent_bytes_ = 0;
  std::uint64_t slot_bytes_ = 0;
  std::uint32_t entry_position_ = 0;
  std::uint32_t rows_ = 0;
  std::uint32_t committed_ = 0;
  std::array<std::uint32_t, 4> inputs_{};
  std::array<std::uint32_t, 4> predictions_{};
  bool active_ = false;
  bool verified_ = false;
};

}  // namespace q3x::runtime::mtp_detail
