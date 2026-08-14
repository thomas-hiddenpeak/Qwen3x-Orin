#include "q3x/kernels/sm87_bulk_dataflow_v2_nvfp4_projection.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace kernels = q3x::kernels;

namespace {

using PayloadRole = kernels::Sm87BulkV2NvFp4LogicalPayloadRole;
using ProjectionRole = kernels::Sm87TargetAotProjectionRole;

static_assert(kernels::kSm87BulkV2NvFp4P40Tokens == 40'000U);
static_assert(kernels::kSm87BulkV2NvFp4SegmentsPerLayer == 40U);
static_assert(kernels::kSm87BulkV2NvFp4PhysicalMacroLaunches == 2'560U);
static_assert(kernels::kSm87BulkV2NvFp4RoleCount == 128U);
static_assert(kernels::kSm87BulkV2NvFp4PersistentCtas == 32U);
static_assert(kernels::kSm87BulkV2NvFp4DownOwnerCtas == 20U);
static_assert(kernels::kSm87BulkV2NvFp4DedicatedProducerCtas == 12U);
static_assert(kernels::kSm87BulkV2NvFp4ReadinessSlots == 32U);
static_assert(kernels::kSm87BulkV2NvFp4DynamicSharedBytes == 52'224U);
static_assert(kernels::kSm87BulkV2NvFp4MaximumRegisters == 128U);
static_assert(kernels::kSm87BulkV2NvFp4GroupScratchBytes == 8'912'896ULL);
static_assert(kernels::kSm87BulkV2NvFp4HotReadyWindowBytes ==
              1'048'576ULL);
static_assert(kernels::kSm87BulkV2NvFp4AggregateL2BudgetBytes ==
              3'670'016ULL);
static_assert(kernels::kSm87BulkV2NvFp4FamilyPayloadBytes ==
              9'625'927'680ULL);
static_assert(kernels::sm87_bulk_v2_nvfp4_manifest_valid(
    kernels::kSm87BulkV2NvFp4FrozenManifest));

class TestContext final {
 public:
  void expect(const bool condition, const char* const message) {
    if (!condition) {
      ++failures_;
      std::cerr << "FAIL: " << message << '\n';
    }
  }

  [[nodiscard]] int failures() const noexcept { return failures_; }

 private:
  int failures_ = 0;
};

void test_segment_and_family_manifest(TestContext& test) {
  std::uint32_t next_token = 0U;
  std::uint32_t total_tokens = 0U;
  std::uint64_t gate_tasks = 0U;
  std::uint64_t down_tasks = 0U;
  for (std::uint32_t segment = 0U;
       segment < kernels::kSm87BulkV2NvFp4SegmentsPerLayer; ++segment) {
    const auto plan = kernels::sm87_bulk_v2_nvfp4_segment_plan(segment);
    test.expect(plan.valid && plan.segment == segment,
                "every P40 segment is represented exactly once");
    test.expect(plan.first_token == next_token,
                "macrosegments are contiguous and non-overlapping");
    test.expect(plan.m_tiles * kernels::kSm87BulkV2NvFp4TileM ==
                    plan.token_count,
                "every macrosegment is an exact M64 decomposition");
    if (segment < 39U) {
      test.expect(plan.token_count == 1'024U && plan.group_count == 4U &&
                      !plan.tail,
                  "the first 39 macrosegments are M1024/M256x4");
    } else {
      test.expect(plan.token_count == 64U && plan.group_count == 1U &&
                      plan.tail,
                  "the terminal macrosegment is the exact M64 tail");
    }
    next_token += plan.token_count;
    total_tokens += plan.token_count;
    gate_tasks += plan.gate_tasks;
    down_tasks += plan.down_tasks;
  }
  test.expect(total_tokens == 40'000U && next_token == 40'000U,
              "39x1024+64 closes the P40 token interval");
  test.expect(gate_tasks == 170'000U && down_tasks == 12'500U,
              "P40 GateUp and Down logical tile counts are exact");
  test.expect(!kernels::sm87_bulk_v2_nvfp4_segment_plan(40U).valid,
              "there is no generic segment beyond the frozen P40 plan");

  const auto& manifest = kernels::kSm87BulkV2NvFp4FrozenManifest;
  std::uint32_t gate_roles = 0U;
  std::uint32_t down_roles = 0U;
  std::uint32_t launches = 0U;
  std::uint64_t payload = 0U;
  for (std::uint32_t ordinal = 0U; ordinal < manifest.role_count;
       ++ordinal) {
    const auto& role = manifest.roles[ordinal];
    test.expect(role.ordinal == ordinal && role.layer == ordinal / 2U,
                "128 logical roles remain layer-major and paired");
    test.expect(role.joint_with_adjacent_role,
                "GateUp and Down expose only a joint macro receipt");
    launches += role.owned_physical_launches;
    payload += role.payload_bytes;
    if (role.role == ProjectionRole::kNvFp4GateUp) {
      ++gate_roles;
      test.expect(role.owns_joint_launch,
                  "GateUp descriptor owns the pair's physical launch count");
    } else if (role.role == ProjectionRole::kNvFp4Down) {
      ++down_roles;
      test.expect(!role.owns_joint_launch &&
                      role.owned_physical_launches == 0U,
                  "Down shares rather than double-counts joint launches");
    } else {
      test.expect(false, "the NVFP4 family contains no unrelated role");
    }
  }
  test.expect(gate_roles == 64U && down_roles == 64U,
              "all 64 layers contain exactly one GateUp/Down pair");
  test.expect(launches == 2'560U,
              "64 layers x 40 joint macrosegments yields 2560 launches");
  test.expect(payload == kernels::kSm87BulkV2NvFp4FamilyPayloadBytes,
              "manifest payload accounting retains all authenticated bytes");
  test.expect(!manifest.request_time_repack && !manifest.request_time_jit &&
                  manifest.execution_identity ==
                      kernels::Sm87BulkV2NvFp4ExecutionIdentity::
                          kExactControlSteppingStoneM256J64N256K64NoResidencyV2 &&
                  !manifest.numerical_contract_qualified &&
                  !manifest.production_dispatch_eligible,
              "the manifest names the default-off no-residency stepping stone");

  auto tampered = manifest;
  ++tampered.roles[9U].owned_physical_launches;
  test.expect(!kernels::sm87_bulk_v2_nvfp4_manifest_valid(tampered),
              "the deterministic manifest seal detects role substitution");
}

void test_exhaustive_task_mapping(TestContext& test) {
  std::uint64_t observed_gate = 0U;
  std::uint64_t observed_down = 0U;
  for (std::uint32_t segment = 0U;
       segment < kernels::kSm87BulkV2NvFp4SegmentsPerLayer; ++segment) {
    const auto macro = kernels::sm87_bulk_v2_nvfp4_segment_plan(segment);
    std::vector<std::uint8_t> seen_gate(
        static_cast<std::size_t>(macro.m_tiles) *
            kernels::kSm87BulkV2NvFp4GateNTiles,
        0U);
    std::vector<std::uint8_t> seen_down(
        static_cast<std::size_t>(macro.m_tiles) *
            kernels::kSm87BulkV2NvFp4DownNTiles,
        0U);
    std::uint32_t gate_base = 0U;
    std::uint32_t down_base = 0U;
    for (std::uint32_t group_index = 0U;
         group_index < macro.group_count; ++group_index) {
      const auto group = kernels::sm87_bulk_v2_nvfp4_group_plan(
          segment, group_index);
      test.expect(group.valid && group.first_m_tile == 4U * group_index,
                  "M256 group ownership is contiguous");
      for (std::uint32_t ordinal = 0U; ordinal < group.gate_tasks;
           ++ordinal) {
        const auto task =
            kernels::sm87_bulk_v2_nvfp4_gate_task(group, ordinal);
        const std::uint32_t local =
            task.segment_m_tile * kernels::kSm87BulkV2NvFp4GateNTiles +
            task.q;
        test.expect(task.valid && task.full_k_single_cta &&
                        task.k_tiles == 80U &&
                        task.first_output == task.q * 64U &&
                        local < seen_gate.size(),
                    "every paired GateUp task owns one M64xJ64 full-K tile");
        if (local < seen_gate.size()) {
          test.expect(seen_gate[local] == 0U,
                      "GateUp task mapping has no duplicate");
          ++seen_gate[local];
        }
        ++observed_gate;
      }
      for (std::uint32_t row = 0U; row < group.active_rows; ++row) {
        for (std::uint32_t owner = 0U;
             owner < kernels::kSm87BulkV2NvFp4DownOwnerCtas; ++owner) {
          const auto task = kernels::sm87_bulk_v2_nvfp4_down_task(
              group, owner, row);
          const std::uint32_t local =
              task.segment_m_tile *
                  kernels::kSm87BulkV2NvFp4DownNTiles +
              task.n_tile;
          test.expect(task.valid && task.owner_cta == owner &&
                          task.n_tile == owner &&
                          task.q_steps == 272U &&
                          task.q_strictly_ascending &&
                          task.full_k_single_cta &&
                          task.first_output == owner * 256U &&
                          local < seen_down.size(),
                      "each of 20 consumers owns one M64xN256 full-K tile");
          if (local < seen_down.size()) {
            test.expect(seen_down[local] == 0U,
                        "Down task mapping has no duplicate");
            ++seen_down[local];
          }
          ++observed_down;
        }
      }
      gate_base += group.gate_tasks;
      down_base += group.down_tasks;
    }
    test.expect(gate_base == macro.gate_tasks &&
                    down_base == macro.down_tasks,
                "group tasks sum exactly to the enclosing macrosegment");
    for (const auto count : seen_gate) {
      test.expect(count == 1U,
                  "GateUp exhaustive domain has complete unique coverage");
    }
    for (const auto count : seen_down) {
      test.expect(count == 1U,
                  "Down exhaustive domain has complete unique coverage");
    }
  }
  test.expect(observed_gate == 170'000U && observed_down == 12'500U,
              "all P40 logical projection work was enumerated once");

  std::uint32_t previous_k = 0U;
  bool first = true;
  for (std::uint32_t q = 0U; q < 272U; ++q) {
    for (std::uint32_t k16 = 0U; k16 < 4U; ++k16) {
      const std::uint32_t k = q * 64U + k16 * 16U;
      test.expect(first || k > previous_k,
                  "Down traverses J64 then K16 in strict ascending order");
      first = false;
      previous_k = k;
    }
  }
  test.expect(previous_k == 17'392U,
              "Down's single owner reaches the terminal K16 exactly");
}

struct WorkStealSimulation final {
  std::array<std::uint32_t, kernels::kSm87BulkV2NvFp4RowsPerGroup>
      next_q{};
  std::array<std::uint32_t, kernels::kSm87BulkV2NvFp4RowsPerGroup>
      retired_q{};
  std::array<std::array<std::uint32_t,
                        kernels::kSm87BulkV2NvFp4ReadinessSlots>,
             kernels::kSm87BulkV2NvFp4RowsPerGroup>
      ready{};
  std::array<std::array<std::uint32_t,
                        kernels::kSm87BulkV2NvFp4ReadinessSlots>,
             kernels::kSm87BulkV2NvFp4RowsPerGroup>
      readers{};
  std::array<std::uint32_t, kernels::kSm87BulkV2NvFp4DownOwnerCtas>
      consumer_row{};
  std::array<std::uint32_t, kernels::kSm87BulkV2NvFp4DownOwnerCtas>
      consumer_q{};
  std::vector<std::uint8_t> claimed;
  std::uint32_t active_rows = 0U;
  std::uint32_t consumer_claims = 0U;
  std::uint32_t producer_claims = 0U;
  std::uint32_t maximum_credit = 0U;

  explicit WorkStealSimulation(const std::uint32_t rows)
      : claimed(static_cast<std::size_t>(rows) *
                    kernels::kSm87BulkV2NvFp4GateNTiles,
                0U),
        active_rows(rows) {}

  bool claim(const std::uint32_t seed, const bool consumer,
             TestContext& test) {
    const std::uint32_t row = kernels::sm87_bulk_v2_nvfp4_claimable_row(
        next_q, retired_q, active_rows, seed);
    if (row >= active_rows) {
      return false;
    }
    const std::uint32_t q = next_q[row]++;
    const std::uint32_t ordinal =
        row * kernels::kSm87BulkV2NvFp4GateNTiles + q;
    test.expect(ordinal < claimed.size() && claimed[ordinal] == 0U,
                "atomic claim cursor grants every GateUp task once");
    if (ordinal < claimed.size()) {
      ++claimed[ordinal];
    }
    const std::uint32_t slot =
        kernels::sm87_bulk_v2_nvfp4_readiness_slot(q);
    test.expect(readers[row][slot] == 0U,
                "credit prevents readiness-slot overwrite before retire");
    readers[row][slot] = kernels::kSm87BulkV2NvFp4DownOwnerCtas;
    ready[row][slot] =
        kernels::sm87_bulk_v2_nvfp4_readiness_generation(q);
    const std::uint32_t credit = next_q[row] - retired_q[row];
    if (credit > maximum_credit) {
      maximum_credit = credit;
    }
    test.expect(credit <= kernels::kSm87BulkV2NvFp4ReadinessSlots,
                "producer lead never exceeds the 32-slot credit");
    if (consumer) {
      ++consumer_claims;
    } else {
      ++producer_claims;
    }
    return true;
  }

  bool consume(const std::uint32_t owner, TestContext& test) {
    const std::uint32_t row = consumer_row[owner];
    if (row >= active_rows) {
      return false;
    }
    const std::uint32_t q = consumer_q[owner];
    const std::uint32_t slot =
        kernels::sm87_bulk_v2_nvfp4_readiness_slot(q);
    const std::uint32_t generation =
        kernels::sm87_bulk_v2_nvfp4_readiness_generation(q);
    if (ready[row][slot] != generation) {
      return false;
    }
    test.expect(readers[row][slot] > 0U,
                "a ready H tile retains one credit per Down owner");
    if (readers[row][slot] > 0U && --readers[row][slot] == 0U) {
      retired_q[row] = q + 1U;
    }
    if (++consumer_q[owner] == kernels::kSm87BulkV2NvFp4GateNTiles) {
      consumer_q[owner] = 0U;
      ++consumer_row[owner];
    }
    return true;
  }

  [[nodiscard]] bool complete() const noexcept {
    for (std::uint32_t row = 0U; row < active_rows; ++row) {
      if (next_q[row] != kernels::kSm87BulkV2NvFp4GateNTiles ||
          retired_q[row] != kernels::kSm87BulkV2NvFp4GateNTiles) {
        return false;
      }
    }
    for (const auto row : consumer_row) {
      if (row != active_rows) {
        return false;
      }
    }
    return true;
  }
};

void run_work_steal_simulation(const std::uint32_t active_rows,
                               TestContext& test) {
  WorkStealSimulation state(active_rows);
  constexpr std::uint32_t kStepLimit = 2'000'000U;
  std::uint32_t step = 0U;
  for (; step < kStepLimit && !state.complete(); ++step) {
    for (std::uint32_t owner = 0U;
         owner < kernels::kSm87BulkV2NvFp4DownOwnerCtas; ++owner) {
      if (!state.consume(owner, test)) {
        (void)state.claim(step + owner, true, test);
      }
    }
    for (std::uint32_t producer = 0U;
         producer < kernels::kSm87BulkV2NvFp4DedicatedProducerCtas;
         ++producer) {
      (void)state.claim(step + producer +
                            kernels::kSm87BulkV2NvFp4DownOwnerCtas,
                        false, test);
    }
  }
  test.expect(step < kStepLimit && state.complete(),
              "work-conserving producer/consumer schedule terminates");
  for (const auto count : state.claimed) {
    test.expect(count == 1U,
                "dedicated producers and stealing consumers jointly cover GateUp once");
  }
  test.expect(state.consumer_claims > 0U && state.producer_claims > 0U,
              "both CTA classes execute producer work in the simulation");
  test.expect(state.maximum_credit ==
                  kernels::kSm87BulkV2NvFp4ReadinessSlots,
              "the executable schedule exercises the full 32-slot window");
}

void test_work_stealing_and_balance(TestContext& test) {
  run_work_steal_simulation(4U, test);
  run_work_steal_simulation(1U, test);

  const std::uint64_t gate =
      kernels::kSm87BulkV2NvFp4FullGroupGateWork;
  const std::uint64_t down =
      kernels::kSm87BulkV2NvFp4FullGroupDownWork;
  test.expect(gate == 2U * down,
              "Gate+Up work is exactly twice Down work for every M256 group");
  const std::uint64_t denominator =
      kernels::kSm87BulkV2NvFp4StealFractionDenominator;
  const std::uint64_t numerator =
      kernels::kSm87BulkV2NvFp4StealFractionNumerator;
  const std::uint64_t gate_worker_numerators =
      kernels::kSm87BulkV2NvFp4DedicatedProducerCtas * denominator +
      kernels::kSm87BulkV2NvFp4DownOwnerCtas * numerator;
  const std::uint64_t down_worker_numerators =
      kernels::kSm87BulkV2NvFp4DownOwnerCtas *
      (denominator - numerator);
  test.expect(numerator == 7U && denominator == 15U &&
                  gate_worker_numerators == 320U &&
                  down_worker_numerators == 160U,
              "7/15 stealing produces a 2:1 effective worker split");
  test.expect(gate * down_worker_numerators ==
                  down * gate_worker_numerators,
              "the ideal work-conserving GateUp and Down completion times match");
  test.expect((gate + down) % kernels::kSm87BulkV2NvFp4PersistentCtas ==
                  0U &&
                  (gate + down) /
                          kernels::kSm87BulkV2NvFp4PersistentCtas ==
                      2'139'095'040ULL,
              "ideal aggregate work divides exactly across 32 resident CTAs");
}

void accumulate_payload_role_bijection(
    const PayloadRole role, std::vector<std::uint8_t>& seen,
    TestContext& test) {
  const bool gate_or_up = role == PayloadRole::kGate ||
                          role == PayloadRole::kUp;
  const auto projection_role =
      gate_or_up ? ProjectionRole::kNvFp4GateUp
                 : ProjectionRole::kNvFp4Down;
  const std::uint32_t partition = role == PayloadRole::kUp ? 1U : 0U;
  const auto layout =
      kernels::sm87_target_aot_projection_packed_layout(projection_role);
  const auto& source = layout.partitions[partition];
  const std::uint64_t partition_fragments =
      static_cast<std::uint64_t>(source.n_tiles) * source.k_tiles *
      4U * 4U * 8U;
  const std::uint64_t role_fragments =
      layout.payload_bytes /
      (64U + 8U);
  test.expect(seen.size() == role_fragments,
              "the joint seen-domain spans the complete role payload");
  const std::uint32_t n_tiles =
      gate_or_up ? kernels::kSm87BulkV2NvFp4GateNTiles
                 : kernels::kSm87BulkV2NvFp4DownNTiles;
  const std::uint32_t k_tiles =
      gate_or_up ? kernels::kSm87BulkV2NvFp4GateKTiles
                 : kernels::kSm87BulkV2NvFp4DownKTiles;
  const std::uint32_t n8_panels = gate_or_up ? 8U : 32U;
  std::uint64_t observed_fragments = 0U;
  std::uint64_t observed_payload_bytes = 0U;
  for (std::uint32_t n_tile = 0U; n_tile < n_tiles; ++n_tile) {
    for (std::uint32_t k_tile = 0U; k_tile < k_tiles; ++k_tile) {
      for (std::uint32_t k16 = 0U; k16 < 4U; ++k16) {
        for (std::uint32_t n8 = 0U; n8 < n8_panels; ++n8) {
          const auto view = kernels::sm87_bulk_v2_nvfp4_payload_fragment(
              role, n_tile, k_tile, k16, n8);
          const std::uint64_t source_cell =
              source.payload_offset +
              (static_cast<std::uint64_t>(view.source_n_tile) *
                   source.k_tiles +
               k_tile) *
                  source.cell_bytes;
          const std::uint64_t source_panel =
              (static_cast<std::uint64_t>(k16) * 4U +
               view.source_n_warp) *
                  8U +
              view.source_n8_panel;
          const std::uint64_t expected_weight_offset =
              source_cell + source_panel * 64U;
          const std::uint64_t expected_scale_offset =
              source_cell + source.weight_bytes_per_cell +
              source_panel * 8U;
          test.expect(view.valid && view.same_authenticated_bytes &&
                          view.partition_index == partition &&
                          view.authenticated.weight_bytes == 64U &&
                          view.authenticated.block_scale_bytes == 8U &&
                          view.authenticated.weight_offset ==
                              expected_weight_offset &&
                          view.authenticated.block_scale_offset ==
                              expected_scale_offset &&
                          view.source_fragment_ordinal < seen.size(),
                      "every v2 fragment is a direct authenticated byte view");
          if (view.source_fragment_ordinal < seen.size()) {
            test.expect(seen[view.source_fragment_ordinal] == 0U,
                        "v2 fragment view never aliases a source fragment");
            ++seen[view.source_fragment_ordinal];
          }
          if (gate_or_up) {
            test.expect(view.source_n_tile == n_tile / 4U &&
                            view.source_n_warp == n_tile % 4U &&
                            view.source_n8_panel == n8,
                        "Gate/Up J64 selects one quarter of an N256 cell");
          } else {
            test.expect(view.source_n_tile == n_tile &&
                            view.source_n_warp == n8 / 8U &&
                            view.source_n8_panel == n8 % 8U,
                        "Down N256 consumes all four authenticated N64 warps");
          }
          ++observed_fragments;
          observed_payload_bytes +=
              view.authenticated.weight_bytes +
              view.authenticated.block_scale_bytes;
        }
      }
    }
  }
  test.expect(observed_fragments == partition_fragments,
              "v2 partition and target-AOT fragment cardinalities are identical");
  test.expect(observed_payload_bytes == source.payload_bytes &&
                  source.payload_bytes ==
                      kernels::kSm87BulkV2NvFp4GatePartitionPayloadBytes,
              "fragment bijection covers every weight and scale byte once");
}

void test_payload_bijection(TestContext& test) {
  const auto gate_up_layout =
      kernels::sm87_target_aot_projection_packed_layout(
          ProjectionRole::kNvFp4GateUp);
  std::vector<std::uint8_t> gate_up_seen(
      gate_up_layout.payload_bytes / (64U + 8U), 0U);
  accumulate_payload_role_bijection(PayloadRole::kGate, gate_up_seen,
                                    test);
  accumulate_payload_role_bijection(PayloadRole::kUp, gate_up_seen, test);
  for (const auto count : gate_up_seen) {
    test.expect(count == 1U,
                "Gate and Up jointly cover the authenticated role once");
  }

  const auto down_layout =
      kernels::sm87_target_aot_projection_packed_layout(
          ProjectionRole::kNvFp4Down);
  std::vector<std::uint8_t> down_seen(
      down_layout.payload_bytes / (64U + 8U), 0U);
  accumulate_payload_role_bijection(PayloadRole::kDown, down_seen, test);
  for (const auto count : down_seen) {
    test.expect(count == 1U,
                "Down covers the authenticated role exactly once");
  }
  test.expect(!kernels::sm87_bulk_v2_nvfp4_payload_fragment(
                   PayloadRole::kGate, 272U, 0U, 0U, 0U)
                   .valid &&
                  !kernels::sm87_bulk_v2_nvfp4_payload_fragment(
                       PayloadRole::kDown, 20U, 0U, 0U, 0U)
                       .valid,
              "out-of-domain payload coordinates fail closed");
}

void test_joint_receipt_state_machine(TestContext& test) {
  for (std::uint32_t segment = 0U;
       segment < kernels::kSm87BulkV2NvFp4SegmentsPerLayer; ++segment) {
    auto receipt = kernels::sm87_bulk_v2_nvfp4_make_joint_receipt(
        19U, 7U, segment);
    test.expect(
        kernels::sm87_bulk_v2_nvfp4_joint_receipt_valid(receipt) &&
            receipt.execution_identity ==
                kernels::Sm87BulkV2NvFp4ExecutionIdentity::
                    kExactControlSteppingStoneM256J64N256K64NoResidencyV2 &&
            receipt.exact_control_stepping_stone &&
            !receipt.cross_group_weight_residency_qualified &&
            !receipt.p40_hot_path_qualified,
        "joint receipt begins as an exact-control no-residency stepping stone");
    receipt = kernels::sm87_bulk_v2_nvfp4_advance_joint_receipt(
        receipt, kernels::Sm87BulkV2NvFp4JointEvent::kEnqueueMacro);
    const auto macro = kernels::sm87_bulk_v2_nvfp4_segment_plan(segment);
    for (std::uint32_t group_index = 0U;
         group_index < macro.group_count; ++group_index) {
      receipt = kernels::sm87_bulk_v2_nvfp4_advance_joint_receipt(
          receipt, kernels::Sm87BulkV2NvFp4JointEvent::kStartGroup);
      const auto group = kernels::sm87_bulk_v2_nvfp4_group_plan(
          segment, group_index);
      receipt = kernels::sm87_bulk_v2_nvfp4_advance_joint_receipt(
          receipt, kernels::Sm87BulkV2NvFp4JointEvent::kCompleteGroup,
          group.gate_tasks, group.down_tasks);
      test.expect(kernels::sm87_bulk_v2_nvfp4_joint_receipt_valid(receipt) &&
                      receipt.completed_groups == group_index + 1U,
                  "one joint group receipt accounts for both projection roles");
    }
    receipt = kernels::sm87_bulk_v2_nvfp4_advance_joint_receipt(
        receipt, kernels::Sm87BulkV2NvFp4JointEvent::kCompleteMacro);
    test.expect(kernels::sm87_bulk_v2_nvfp4_joint_receipt_valid(receipt) &&
                    receipt.phase ==
                        kernels::Sm87BulkV2NvFp4JointPhase::kMacroComplete,
                "macro completion requires every M256 group receipt");
  }

  auto wrong = kernels::sm87_bulk_v2_nvfp4_make_joint_receipt(3U, 0U, 0U);
  wrong = kernels::sm87_bulk_v2_nvfp4_advance_joint_receipt(
      wrong, kernels::Sm87BulkV2NvFp4JointEvent::kEnqueueMacro);
  wrong = kernels::sm87_bulk_v2_nvfp4_advance_joint_receipt(
      wrong, kernels::Sm87BulkV2NvFp4JointEvent::kStartGroup);
  wrong = kernels::sm87_bulk_v2_nvfp4_advance_joint_receipt(
      wrong, kernels::Sm87BulkV2NvFp4JointEvent::kCompleteGroup,
      1U, 1U);
  test.expect(!kernels::sm87_bulk_v2_nvfp4_joint_receipt_valid(wrong),
              "partial or role-local task receipts cannot complete a group");

  auto overstated =
      kernels::sm87_bulk_v2_nvfp4_make_joint_receipt(4U, 0U, 0U);
  overstated.cross_group_weight_residency_qualified = true;
  test.expect(!kernels::sm87_bulk_v2_nvfp4_joint_receipt_valid(overstated),
              "an exact-control receipt cannot claim unimplemented residency");

  auto cancelled =
      kernels::sm87_bulk_v2_nvfp4_make_joint_receipt(5U, 1U, 39U);
  cancelled = kernels::sm87_bulk_v2_nvfp4_advance_joint_receipt(
      cancelled, kernels::Sm87BulkV2NvFp4JointEvent::kEnqueueMacro);
  cancelled = kernels::sm87_bulk_v2_nvfp4_advance_joint_receipt(
      cancelled, kernels::Sm87BulkV2NvFp4JointEvent::kStartGroup);
  cancelled = kernels::sm87_bulk_v2_nvfp4_advance_joint_receipt(
      cancelled,
      kernels::Sm87BulkV2NvFp4JointEvent::kObserveCancellation);
  test.expect(kernels::sm87_bulk_v2_nvfp4_joint_receipt_valid(cancelled) &&
                  cancelled.phase ==
                      kernels::Sm87BulkV2NvFp4JointPhase::kCancelled &&
                  cancelled.cancellation_word_observed &&
                  cancelled.unpublished_state_discarded,
              "cancellation is terminal and discards private partial state");
}

[[nodiscard]] kernels::Sm87BulkV2NvFp4CodeEvidence
valid_code_evidence() noexcept {
  kernels::Sm87BulkV2NvFp4CodeEvidence result;
  result.elf_identity = 1U;
  result.canonical_sass_hash = 2U;
  result.instruction_rows = 2'048U;
  result.text_bytes = 64U * 1'024U;
  result.k_dependent_unrolled_body_copies = 1U;
  result.local_load_store_rows = 0U;
  result.launch_bounds_256_2 = true;
  result.cooperative_grid_sync_present = true;
  result.atomic_cas_work_claim_present = true;
  result.cp_async_ca_activation_present = true;
  result.cp_async_cg_payload_present = true;
  return result;
}

[[nodiscard]] kernels::Sm87BulkV2NvFp4KernelResources
valid_resources() noexcept {
  kernels::Sm87BulkV2NvFp4KernelResources result;
  result.binary_version = 87;
  result.registers_per_thread = 128;
  result.static_shared_bytes = 0U;
  result.dynamic_shared_bytes =
      kernels::kSm87BulkV2NvFp4DynamicSharedBytes;
  result.local_bytes = 0U;
  result.maximum_threads_per_block = 256;
  result.active_blocks_per_sm = 2;
  result.cooperative_grid_capacity = 32;
  result.code = valid_code_evidence();
  result.kernel_compiled = true;
  result.cooperative_launch_supported = true;
  result.resource_and_code_gate_passed = true;
  result.numerical_contract_qualified = false;
  result.production_dispatch_eligible = false;
  return result;
}

void test_resource_and_code_gate(TestContext& test) {
  const auto accepted = valid_resources();
  test.expect(kernels::sm87_bulk_v2_nvfp4_resources_valid(accepted),
              "the complete two-CTA/SM plus compact-SASS gate is coherent");

  auto rejected = accepted;
  rejected.registers_per_thread = 129;
  test.expect(!kernels::sm87_bulk_v2_nvfp4_resources_valid(rejected),
              "129 registers fails rather than silently becoming one CTA/SM");
  rejected = accepted;
  rejected.active_blocks_per_sm = 1;
  test.expect(!kernels::sm87_bulk_v2_nvfp4_resources_valid(rejected),
              "one-CTA target-AOT serialization is rejected statically");
  rejected = accepted;
  rejected.code.instruction_rows = 26'944U;
  test.expect(!kernels::sm87_bulk_v2_nvfp4_resources_valid(rejected),
              "the historical target-AOT Gate SASS footprint fails the v2 gate");
  rejected = accepted;
  rejected.code.k_dependent_unrolled_body_copies = 80U;
  test.expect(!kernels::sm87_bulk_v2_nvfp4_resources_valid(rejected),
              "K-proportional mainloop expansion is forbidden");
  rejected = accepted;
  rejected.code.local_load_store_rows = 1U;
  test.expect(!kernels::sm87_bulk_v2_nvfp4_resources_valid(rejected),
              "any local-memory instruction row fails the static gate");
  rejected = accepted;
  rejected.numerical_contract_qualified = true;
  test.expect(!kernels::sm87_bulk_v2_nvfp4_resources_valid(rejected),
              "compile-only evidence cannot claim numerical qualification");
  rejected = accepted;
  rejected.production_dispatch_eligible = true;
  test.expect(!kernels::sm87_bulk_v2_nvfp4_resources_valid(rejected),
              "first-stage resources cannot open a production selector");
}

}  // namespace

int main() {
  static_assert(alignof(kernels::Sm87BulkV2NvFp4DeviceControl) == 64U);
  static_assert(sizeof(kernels::Sm87BulkV2NvFp4DeviceControl) == 1'152U);
  TestContext test;
  test_segment_and_family_manifest(test);
  test_exhaustive_task_mapping(test);
  test_work_stealing_and_balance(test);
  test_payload_bijection(test);
  test_joint_receipt_state_machine(test);
  test_resource_and_code_gate(test);
  if (test.failures() != 0) {
    std::cerr << test.failures() << " checks failed\n";
    return 1;
  }
  std::cout << "SM87 bulk-dataflow-v2 NVFP4 host/static contract passed\n";
  return 0;
}
