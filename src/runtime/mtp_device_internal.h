#pragma once

#include "mtp_tree_control.h"
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
  const std::uint8_t* mlp_codes(std::size_t index) const noexcept;
  const std::uint16_t* mlp_scales(std::size_t index) const noexcept;
 private:
  void* packed_mlp_ = nullptr;
  void* arena_ = nullptr;
  std::array<std::uint64_t, 15> offsets_{};
};

class Draft {
 public:
  Draft(const Weights&, const ModelWeights&, std::uint32_t capacity,
        const float* cosines, const float* sines,
        bool use_ordered_attention = true);
  ~Draft();
  Draft(const Draft&) = delete;
  Draft& operator=(const Draft&) = delete;
  bool step(std::uint32_t token, const std::uint16_t* hidden,
            bool logits, std::uint32_t& prediction, float* confidence = nullptr) noexcept;
  // Initialize only live K/V from shifted tokens and independent target
  // hidden rows. Requires empty state. No final hidden/logits are produced.
  // Cancellation poisons and drains; successful reset is required for reuse.
  bool initialize_kv(const std::uint32_t* shifted_tokens,
                     const std::uint16_t* target_hidden, std::uint32_t rows,
                     bool (*cancel)(void*) noexcept = nullptr,
                     void* cancel_context = nullptr) noexcept;
  // Append one target-conditioned cache row; no valid draft hidden is produced.
  // The next proposal must begin with the selected target hidden.
  bool append_kv(std::uint32_t token, const std::uint16_t* target_hidden) noexcept;
  // Physical preparation does not publish a logical cache length.
  bool prepare_kv_prefix(const std::uint32_t* tokens,
                         const std::uint16_t* const* hidden,
                         std::uint32_t rows) noexcept;
  bool commit_prepared_kv(std::uint32_t rows) noexcept;
  void discard_prepared_kv() noexcept;
  bool rewind(std::uint32_t position) noexcept;
  bool reset() noexcept;
  bool poison() noexcept;
  std::uint32_t position() const noexcept;
  const std::uint16_t* hidden() const noexcept;
  const std::uint16_t* keys() const noexcept;
  const std::uint16_t* values() const noexcept;
  int error() const noexcept;
  std::uint64_t ordered_attention_steps() const noexcept;
  std::uint64_t integer_head_steps() const noexcept;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Isolated transaction with explicit scalar oracle or multi-row verifier.
// Retains every complete prefix and repairs draft KV using target hidden.
// Neither local mode alone has API acceleration or production authority.
class TargetTransaction final : public TreeRoundBackend {
 public:
  TargetTransaction(ReferenceRunner&, RequestState&, Draft&, bool multirow = false,
                    bool tree_graphs = false);
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
                                    void* cancel_context = nullptr,
                                    bool committed_service_handoff = false) noexcept;
  const std::uint16_t* prompt_hidden() const noexcept { return prompt_hidden_; }
  std::uint64_t entry_snapshots() const noexcept { return entry_snapshots_; }
  std::uint64_t full_graph_rounds() const noexcept { return full_graph_rounds_; }
  std::uint64_t full_graph_rows() const noexcept { return full_graph_rows_; }
  std::uint64_t graph_qk_updates() const noexcept { return graph_qk_updates_; }
  std::uint64_t scalar_verify_rows() const noexcept { return scalar_verify_rows_; }
  std::uint64_t direct_gdn_rows() const noexcept { return direct_gdn_rows_; }
  std::uint64_t seed_kv_reuses() const noexcept { return seed_kv_reuses_; }
  std::uint64_t tail_kv_batches() const noexcept { return tail_kv_batches_; }
  std::uint64_t tail_kv_rows() const noexcept { return tail_kv_rows_; }
  bool begin(std::uint32_t verify_rows) noexcept override;
  bool propose(std::uint32_t seed, std::uint32_t count,
               std::uint32_t* draft) noexcept override;
  bool propose_bounded(std::uint32_t seed, std::uint32_t maximum,
      std::uint32_t* draft, std::uint32_t& actual) noexcept override;
  bool verify(std::uint32_t seed, const std::uint32_t* draft,
              std::uint32_t count, std::uint32_t* predictions) noexcept override;
  // Explicit native tree oracle entry; not selected by proposal/service yet.
  bool verify_tree(std::uint32_t seed, const std::uint32_t* draft,
                   std::uint32_t count, std::uint32_t* predictions) noexcept;
  bool proposal_parents(std::uint32_t count, TreeParents&) const noexcept override;
  bool commit_node(std::uint32_t node, std::uint32_t depth,
                   std::uint32_t pending) noexcept override;
  bool commit_prefix(std::uint32_t rows, std::uint32_t pending) noexcept override;
  bool finish() noexcept override;
  bool abort() noexcept override;
 private:
  bool verify_multirow(std::uint32_t* predictions) noexcept;
  bool enqueue_multirow(bool full_graph, bool short_graph) noexcept;
  bool prepare_full_graphs() noexcept;
  void release_full_graphs() noexcept;
  void* full_graphs_ = nullptr;
  void* graph_inputs_ = nullptr;
  void* graph_host_ = nullptr;
  std::uint64_t full_graph_rounds_ = 0, full_graph_rows_ = 0, graph_qk_updates_ = 0;
  bool enqueue_gdn_block(unsigned first_layer, unsigned rows) noexcept;
  bool prepare_gdn_graphs() noexcept;
  void release_gdn_graphs() noexcept;
  std::array<void*, 48> gdn_graphs_{};
  bool multirow_ = false;
  bool tree_graphs_ = false;
  bool snapshot(std::uint32_t slot) noexcept;
  ReferenceRunner& target_;
  RequestState& state_;
  Draft& draft_;
  void* snapshots_ = nullptr;
  bool owns_snapshots_ = false;
  std::uint16_t* prompt_hidden_ = nullptr;
  std::uint64_t recurrent_offset_ = 0;
  std::uint64_t recurrent_bytes_ = 0;
  std::uint64_t slot_bytes_ = 0;
  std::uint32_t entry_position_ = 0;
  std::uint32_t rows_ = 0;
  std::uint32_t committed_ = 0;
  std::uint32_t committed_node_ = kTreeEntryParent;
  bool alternate_tail_ = false;
  std::array<std::uint32_t, 4> inputs_{};
  std::array<std::uint32_t, 4> predictions_{};
  std::uint64_t entry_snapshots_ = 0, direct_gdn_rows_ = 0, seed_kv_reuses_ = 0;
  std::uint64_t tail_kv_batches_ = 0, tail_kv_rows_ = 0;
  std::uint64_t scalar_verify_rows_ = 0;
  std::uint32_t prepared_tail_rows_ = 0;
  std::uint32_t cached_seed_ = 0;
  bool seed_cache_valid_ = false;
  bool active_ = false;
  bool verified_ = false;
};

}  // namespace q3x::runtime::mtp_detail
