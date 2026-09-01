#pragma once

#include "q3x/runtime/reference_engine.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

// Source-private evidence and access seam for the real-checkpoint layer-0
// target-AOT oracle. Nothing in this file is installed. The records contain
// identities and digests only: they never publish a device pointer, asset
// descriptor, receipt object, or launcher capability.
namespace q3x::runtime::reference_engine_test_detail {

struct Sm87TargetAotLayer0M192KernelResourceEvidence final {
  std::uint8_t role = 0U;
  int binary_version = 0;
  int registers_per_thread = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  int maximum_threads_per_block = 0;
  int active_blocks_per_sm = 0;
  int physical_ctas = 0;
  int device_ordinal = -1;
  int device_major = 0;
  int device_minor = 0;
  int device_sm_count = 0;

  std::uint32_t block_m = 0U;
  std::uint32_t block_n = 0U;
  std::uint32_t block_k = 0U;
  std::uint32_t pipeline_stages = 0U;
  std::uint32_t cta_threads = 0U;
  std::uint32_t grid_m = 0U;
  std::uint32_t grid_n = 0U;
  std::uint32_t full_logical_tasks = 0U;
  std::uint32_t tail_logical_tasks = 0U;
  std::uint32_t total_logical_tasks = 0U;
  std::uint32_t n_halves_per_logical_task = 0U;
  std::uint32_t n_half_features = 0U;
  std::uint32_t raster_group_m = 0U;
  bool n_stationary = false;
  std::uint32_t logical_tasks_per_cta_min = 0U;
  std::uint32_t logical_tasks_per_cta_max = 0U;
  bool exact_geometry_gate = false;
  bool exact_resource_gate = false;
};

struct Sm87TargetAotLayer0M192BoundaryEvidence final {
  std::uint64_t elements = 0U;
  std::uint64_t full_elements = 0U;
  std::uint64_t tail_elements = 0U;
  std::uint64_t baseline_candidate_full_mismatches = 0U;
  std::uint64_t baseline_candidate_tail_mismatches = 0U;
  std::uint64_t baseline_replay_full_mismatches = 0U;
  std::uint64_t baseline_replay_tail_mismatches = 0U;
  std::uint64_t candidate_replay_full_mismatches = 0U;
  std::uint64_t candidate_replay_tail_mismatches = 0U;
  bool first_mismatch_present = false;
  std::uint64_t first_mismatch_index = 0U;
  std::uint32_t first_mismatch_row = 0U;
  std::uint32_t first_mismatch_column = 0U;
  std::uint16_t first_mismatch_expected = 0U;
  std::uint16_t first_mismatch_actual = 0U;
  std::string first_mismatch_pair;
  std::string baseline_sha256;
  std::string candidate_sha256;
  std::string replay_sha256;
  bool baseline_all_finite = false;
  bool candidate_all_finite = false;
  bool replay_all_finite = false;
  bool baseline_complete_write = false;
  bool candidate_complete_write = false;
  bool replay_complete_write = false;
  bool baseline_guards_intact = false;
  bool candidate_guards_intact = false;
  bool replay_guards_intact = false;
  bool baseline_candidate_bitwise_exact = false;
  bool baseline_replay_bitwise_exact = false;
  bool candidate_replay_bitwise_exact = false;
  bool bitwise_exact = false;
};

struct Sm87TargetAotLayer0M192FixtureEvidence final {
  std::string generator;
  std::string dtype;
  std::uint32_t rows = 0U;
  std::uint32_t columns = 0U;
  std::uint64_t elements = 0U;
  std::uint64_t bytes = 0U;
  std::string sha256;
};

struct Sm87TargetAotLayer0M192ArtifactEvidence final {
  std::size_t layer_index = 0U;
  std::uint8_t role = 0U;
  std::uint64_t artifact_identity = 0U;
  std::uint64_t source_inventory_identity = 0U;
  std::uint64_t upload_receipt_identity = 0U;
  std::uint16_t plan_identity = 0U;
  std::uint16_t layout_identity = 0U;
  std::uint16_t transform_identity = 0U;
  std::uint64_t device_arena_offset = 0U;
  std::uint64_t payload_bytes = 0U;
  std::string payload_sha256;
  std::uint64_t manifest_seal = 0U;
  std::uint64_t allocation_owner_identity = 0U;
  std::uint64_t allocation_identity = 0U;
  int device_ordinal = -1;
  bool exact = false;
};

struct Sm87TargetAotLayer0M192ReceiptEvidence final {
  std::string baseline_route;
  std::string candidate_plan;
  std::string candidate_layout;
  std::string candidate_publication;
  std::uint64_t owner_identity = 0U;
  std::uint64_t allocation_identity = 0U;
  std::uint64_t arena_bytes = 0U;
  int device_ordinal = -1;
  std::size_t artifact_count = 0U;
  std::string verified_payload_catalog_sha256;
  bool attachment_exact = false;
  bool complete_owner_allocation_ranges_checked = false;
  bool complete_owner_allocation_exact_cover = false;
  bool complete_owner_allocation_nonoverlap = false;
  bool oracle_allocations_disjoint_from_owner = false;
  Sm87TargetAotLayer0M192ArtifactEvidence gate_up;
  Sm87TargetAotLayer0M192ArtifactEvidence down;
};

struct Sm87TargetAotLayer0M192CleanupEvidence final {
  bool synchronize_attempted = false;
  int synchronize_cuda_error = 0;
  std::size_t device_frees_attempted = 0U;
  std::size_t device_frees_succeeded = 0U;
  int first_device_free_cuda_error = 0;
  bool stream_destroy_attempted = false;
  int stream_destroy_cuda_error = 0;
  bool passed = false;
};

struct Sm87TargetAotP40000KillTestEvidence final {
  std::string authority;
  std::string kernel_skeleton;
  std::string decision;
  std::size_t token_count = 0U;
  std::size_t model_layers = 0U;
  std::size_t full_prompt_mlp_layers = 0U;
  std::size_t warmup_pairs = 0U;
  std::size_t measured_pairs = 0U;
  double whole_product_projection_budget_seconds = 0.0;
  std::array<double, 2U> gate_up_milliseconds{};
  std::array<double, 2U> down_milliseconds{};
  std::array<double, 2U> pair_milliseconds{};
  double optimistic_pair_milliseconds = 0.0;
  double optimistic_full_prompt_mlp_seconds = 0.0;
  bool zero_activation_fixture = false;
  bool zero_residual_fixture = false;
  bool inputs_remained_zero = false;
  bool outputs_all_zero = false;
  bool guards_intact = false;
  bool timing_events_destroyed = false;
  bool attempted = false;
  bool completed = false;
  bool budget_exceeded = false;
};

struct Sm87TargetAotLayer0M192OracleResult final {
  std::size_t layer_index = 0U;
  std::size_t token_count = 0U;
  std::size_t full_token_count = 0U;
  std::size_t tail_token_count = 0U;
  std::size_t baseline_gate_launches = 0U;
  std::size_t baseline_up_launches = 0U;
  std::size_t baseline_down_launches = 0U;
  std::size_t candidate_gate_up_launches = 0U;
  std::size_t candidate_down_launches = 0U;
  std::size_t replay_gate_up_launches = 0U;
  std::size_t replay_down_launches = 0U;
  bool owner_attachment_authenticated = false;
  bool canonical_layer0_weights = false;
  bool activation_preserved_after_candidate = false;
  bool residual_preserved_after_candidate = false;
  bool activation_preserved_after_replay = false;
  bool residual_preserved_after_replay = false;
  Sm87TargetAotLayer0M192FixtureEvidence activation_fixture;
  Sm87TargetAotLayer0M192FixtureEvidence residual_fixture;
  Sm87TargetAotLayer0M192ReceiptEvidence receipt;
  Sm87TargetAotLayer0M192KernelResourceEvidence gate_up_resources;
  Sm87TargetAotLayer0M192KernelResourceEvidence down_resources;
  Sm87TargetAotLayer0M192BoundaryEvidence gate_up;
  Sm87TargetAotLayer0M192BoundaryEvidence down_residual;
  Sm87TargetAotP40000KillTestEvidence p40000_kill_test;
  Sm87TargetAotLayer0M192CleanupEvidence cleanup;
  bool passed = false;
};

struct Sm87TargetAotLayer0M192OracleOutcome final {
  std::optional<Sm87TargetAotLayer0M192OracleResult> value;
  ReferenceEngineDiagnostic diagnostic;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && value->passed &&
           diagnostic.code == ReferenceEngineError::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

class Sm87TargetAotLayer0M192OracleAccess final {
 public:
  // Test-only, fixed-M192, real-weight correctness/resource screen. The
  // definition exists only in an explicit admission build and never installs
  // a runner selector or public kernel launch.
  [[nodiscard]] static Sm87TargetAotLayer0M192OracleOutcome screen(
      ReferenceEngine& engine, bool run_p40000_kill_test = false);
};

}  // namespace q3x::runtime::reference_engine_test_detail
