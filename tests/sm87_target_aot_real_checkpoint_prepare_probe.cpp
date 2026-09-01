#include "q3x/core/sha256.h"
#include "q3x/runtime/reference_engine.h"
#include "q3x/runtime/sm87_target_aot_projection_device_assets.h"
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION)
#include "sm87_target_aot_layer0_m192_oracle_evidence_internal.h"
#endif
#include "sm87_target_aot_prepare_provenance.h"

#include <cuda_runtime.h>

#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <cctype>
#include <cstdint>
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION)
#include <ctime>
#endif
#include <filesystem>
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION)
#include <fstream>
#endif
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>

namespace core = q3x::core;
namespace runtime = q3x::runtime;
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION)
namespace oracle = q3x::runtime::reference_engine_test_detail;
#endif
namespace provenance = q3x::test::sm87_target_aot_prepare_provenance;

namespace {

constexpr std::uint64_t kDestroyRecoveryToleranceBytes =
    256ULL * 1024ULL * 1024ULL;
constexpr std::uint64_t kC512RequestArenaBytes = 200'310'784ULL;

void write_json_string(std::ostream& output, const std::string_view value) {
  output.put('"');
  for (const unsigned char byte : value) {
    switch (byte) {
      case '"':
        output << "\\\"";
        break;
      case '\\':
        output << "\\\\";
        break;
      case '\b':
        output << "\\b";
        break;
      case '\f':
        output << "\\f";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        if (byte < 0x20U) {
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<unsigned int>(byte) << std::dec
                 << std::setfill(' ');
        } else {
          output.put(static_cast<char>(byte));
        }
        break;
    }
  }
  output.put('"');
}

void write_boolean(std::ostream& output, const bool value) {
  output << (value ? "true" : "false");
}

[[nodiscard]] bool is_lower_hex(const std::string_view value,
                                const std::size_t expected_size) {
  if (value.size() != expected_size) {
    return false;
  }
  for (const unsigned char byte : value) {
    if (!std::isdigit(byte) && !(byte >= 'a' && byte <= 'f')) {
      return false;
    }
  }
  return true;
}

#if defined(Q3X_ENABLE_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION)
[[nodiscard]] std::string utc_now() {
  const auto now = std::chrono::system_clock::now();
  const auto seconds = std::chrono::time_point_cast<std::chrono::seconds>(now);
  const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
      now - seconds).count();
  const std::time_t time = std::chrono::system_clock::to_time_t(now);
  std::tm utc{};
  if (gmtime_r(&time, &utc) == nullptr) {
    return {};
  }
  char date[32U]{};
  if (std::strftime(date, sizeof(date), "%Y-%m-%dT%H:%M:%S", &utc) == 0U) {
    return {};
  }
  std::ostringstream output;
  output << date << '.' << std::setw(3) << std::setfill('0')
         << milliseconds << 'Z';
  return output.str();
}

[[nodiscard]] std::string read_boot_id() {
  std::ifstream input("/proc/sys/kernel/random/boot_id");
  std::string value;
  if (!std::getline(input, value)) {
    return {};
  }
  return value;
}
#endif

struct ProbeEvidence final {
  bool passed = false;
  bool p40000_quick_kill_requested = false;
  std::filesystem::path model_directory;
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION)
  std::string child_started_at_utc;
  std::string child_evidence_finished_at_utc;
  std::string boot_id;
  std::uint64_t child_pid = 0U;
  bool execution_identity_valid = false;
#endif

  bool source_build_provenance_valid = false;
  bool binary_provenance_valid = false;
  std::filesystem::path executable_path;
  std::uint64_t executable_bytes = 0U;
  std::string executable_sha256;

  int device_ordinal = -1;
  bool device_query_attempted = false;
  int device_query_cuda_error = 0;
  bool initial_mem_info_attempted = false;
  int initial_mem_info_cuda_error = 0;
  bool engine_create_attempted = false;
  bool during_mem_info_attempted = false;
  int during_mem_info_cuda_error = 0;
  bool during_total_matches_initial = false;
  bool synchronize_attempted = false;
  int synchronize_cuda_error = 0;
  bool engine_destroy_attempted = false;
  bool engine_destroy_completed = false;
  bool final_mem_info_attempted = false;
  int final_mem_info_cuda_error = 0;
  bool final_total_matches_initial = false;
  bool memory_recovery_audit_attempted = false;

  std::uint64_t device_total_before_bytes = 0U;
  std::uint64_t device_total_during_bytes = 0U;
  std::uint64_t device_total_after_destroy_bytes = 0U;
  std::uint64_t device_free_before_bytes = 0U;
  std::uint64_t device_free_during_bytes = 0U;
  std::uint64_t device_free_after_destroy_bytes = 0U;

  std::uint64_t wall_milliseconds = 0U;
  bool engine_created = false;
  bool c512_competition_path_exercised = false;
  bool exact_target_inventory = false;
  bool mutually_exclusive_prefill_sidecars_empty = false;
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION)
  bool layer0_m192_oracle_attempted = false;
#endif
  bool memory_recovered_after_destroy = false;
  runtime::ReferenceEngineLoadStats load;
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION)
  oracle::Sm87TargetAotLayer0M192OracleOutcome layer0_m192_oracle;
#endif
  runtime::ReferenceEngineDiagnostic diagnostic;
};

#if defined(Q3X_ENABLE_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION)
void write_kernel_resources(
    std::ostream& output,
    const oracle::Sm87TargetAotLayer0M192KernelResourceEvidence& value) {
  output << "{\"role\": " << static_cast<unsigned int>(value.role)
         << ", \"binary_version\": " << value.binary_version
         << ", \"registers_per_thread\": " << value.registers_per_thread
         << ", \"static_shared_bytes\": " << value.static_shared_bytes
         << ", \"dynamic_shared_bytes\": " << value.dynamic_shared_bytes
         << ", \"local_bytes\": " << value.local_bytes
         << ", \"maximum_threads_per_block\": "
         << value.maximum_threads_per_block
         << ", \"active_blocks_per_sm\": " << value.active_blocks_per_sm
         << ", \"physical_ctas\": " << value.physical_ctas
         << ", \"device_ordinal\": " << value.device_ordinal
         << ", \"device_major\": " << value.device_major
         << ", \"device_minor\": " << value.device_minor
         << ", \"device_sm_count\": " << value.device_sm_count
         << ", \"block_m\": " << value.block_m
         << ", \"block_n\": " << value.block_n
         << ", \"block_k\": " << value.block_k
         << ", \"pipeline_stages\": " << value.pipeline_stages
         << ", \"cta_threads\": " << value.cta_threads
         << ", \"grid_m\": " << value.grid_m
         << ", \"grid_n\": " << value.grid_n
         << ", \"full_logical_tasks\": " << value.full_logical_tasks
         << ", \"tail_logical_tasks\": " << value.tail_logical_tasks
         << ", \"total_logical_tasks\": " << value.total_logical_tasks
         << ", \"n_halves_per_logical_task\": "
         << value.n_halves_per_logical_task
         << ", \"n_half_features\": " << value.n_half_features
         << ", \"raster_group_m\": " << value.raster_group_m
         << ", \"n_stationary\": ";
  write_boolean(output, value.n_stationary);
  output << ", \"logical_tasks_per_cta_min\": "
         << value.logical_tasks_per_cta_min
         << ", \"logical_tasks_per_cta_max\": "
         << value.logical_tasks_per_cta_max
         << ", \"exact_geometry_gate\": ";
  write_boolean(output, value.exact_geometry_gate);
  output
         << ", \"exact_resource_gate\": ";
  write_boolean(output, value.exact_resource_gate);
  output << '}';
}

void write_boundary(
    std::ostream& output,
    const oracle::Sm87TargetAotLayer0M192BoundaryEvidence& value) {
  output << "{\"elements\": " << value.elements
         << ", \"full_elements\": " << value.full_elements
         << ", \"tail_elements\": " << value.tail_elements
         << ", \"baseline_candidate_full_mismatches\": "
         << value.baseline_candidate_full_mismatches
         << ", \"baseline_candidate_tail_mismatches\": "
         << value.baseline_candidate_tail_mismatches
         << ", \"baseline_replay_full_mismatches\": "
         << value.baseline_replay_full_mismatches
         << ", \"baseline_replay_tail_mismatches\": "
         << value.baseline_replay_tail_mismatches
         << ", \"candidate_replay_full_mismatches\": "
         << value.candidate_replay_full_mismatches
         << ", \"candidate_replay_tail_mismatches\": "
         << value.candidate_replay_tail_mismatches
         << ", \"first_mismatch_present\": ";
  write_boolean(output, value.first_mismatch_present);
  output << ", \"first_mismatch_index\": " << value.first_mismatch_index
         << ", \"first_mismatch_row\": " << value.first_mismatch_row
         << ", \"first_mismatch_column\": "
         << value.first_mismatch_column
         << ", \"first_mismatch_expected\": "
         << value.first_mismatch_expected
         << ", \"first_mismatch_actual\": " << value.first_mismatch_actual
         << ", \"first_mismatch_pair\": ";
  write_json_string(output, value.first_mismatch_pair);
  output << ", \"baseline_sha256\": ";
  write_json_string(output, value.baseline_sha256);
  output << ", \"candidate_sha256\": ";
  write_json_string(output, value.candidate_sha256);
  output << ", \"replay_sha256\": ";
  write_json_string(output, value.replay_sha256);
  output << ", \"baseline_all_finite\": ";
  write_boolean(output, value.baseline_all_finite);
  output << ", \"candidate_all_finite\": ";
  write_boolean(output, value.candidate_all_finite);
  output << ", \"replay_all_finite\": ";
  write_boolean(output, value.replay_all_finite);
  output << ", \"baseline_complete_write\": ";
  write_boolean(output, value.baseline_complete_write);
  output << ", \"candidate_complete_write\": ";
  write_boolean(output, value.candidate_complete_write);
  output << ", \"replay_complete_write\": ";
  write_boolean(output, value.replay_complete_write);
  output << ", \"baseline_guards_intact\": ";
  write_boolean(output, value.baseline_guards_intact);
  output << ", \"candidate_guards_intact\": ";
  write_boolean(output, value.candidate_guards_intact);
  output << ", \"replay_guards_intact\": ";
  write_boolean(output, value.replay_guards_intact);
  output << ", \"baseline_candidate_bitwise_exact\": ";
  write_boolean(output, value.baseline_candidate_bitwise_exact);
  output << ", \"baseline_replay_bitwise_exact\": ";
  write_boolean(output, value.baseline_replay_bitwise_exact);
  output << ", \"candidate_replay_bitwise_exact\": ";
  write_boolean(output, value.candidate_replay_bitwise_exact);
  output << ", \"bitwise_exact\": ";
  write_boolean(output, value.bitwise_exact);
  output << '}';
}

void write_fixture(
    std::ostream& output,
    const oracle::Sm87TargetAotLayer0M192FixtureEvidence& value) {
  output << "{\"generator\": ";
  write_json_string(output, value.generator);
  output << ", \"dtype\": ";
  write_json_string(output, value.dtype);
  output << ", \"shape\": [" << value.rows << ", " << value.columns
         << "], \"elements\": " << value.elements << ", \"bytes\": "
         << value.bytes << ", \"sha256\": ";
  write_json_string(output, value.sha256);
  output << '}';
}

void write_artifact(
    std::ostream& output,
    const oracle::Sm87TargetAotLayer0M192ArtifactEvidence& value) {
  output << "{\"layer_index\": " << value.layer_index
         << ", \"role\": " << static_cast<unsigned int>(value.role)
         << ", \"artifact_identity\": " << value.artifact_identity
         << ", \"source_inventory_identity\": "
         << value.source_inventory_identity
         << ", \"upload_receipt_identity\": "
         << value.upload_receipt_identity << ", \"plan_identity\": "
         << value.plan_identity << ", \"layout_identity\": "
         << value.layout_identity << ", \"transform_identity\": "
         << value.transform_identity << ", \"device_arena_offset\": "
         << value.device_arena_offset << ", \"payload_bytes\": "
         << value.payload_bytes << ", \"payload_sha256\": ";
  write_json_string(output, value.payload_sha256);
  output << ", \"manifest_seal\": " << value.manifest_seal
         << ", \"allocation_owner_identity\": "
         << value.allocation_owner_identity << ", \"allocation_identity\": "
         << value.allocation_identity << ", \"device_ordinal\": "
         << value.device_ordinal << ", \"exact\": ";
  write_boolean(output, value.exact);
  output << '}';
}

void write_p40000_kill_test(
    std::ostream& output,
    const oracle::Sm87TargetAotP40000KillTestEvidence& value) {
  output << "{\n        \"authority\": ";
  write_json_string(output, value.authority);
  output << ",\n        \"kernel_skeleton\": ";
  write_json_string(output, value.kernel_skeleton);
  output << ",\n        \"decision\": ";
  write_json_string(output, value.decision);
  output << ",\n        \"token_count\": " << value.token_count
         << ",\n        \"model_layers\": " << value.model_layers
         << ",\n        \"full_prompt_mlp_layers\": "
         << value.full_prompt_mlp_layers
         << ",\n        \"warmup_pairs\": " << value.warmup_pairs
         << ",\n        \"measured_pairs\": " << value.measured_pairs
         << ",\n        \"samples\": [";
  for (std::size_t index = 0U; index < value.pair_milliseconds.size();
       ++index) {
    output << (index == 0U ? "\n" : ",\n")
           << "          {\"index\": " << index
           << ", \"gate_up_milliseconds\": " << std::setprecision(17)
           << value.gate_up_milliseconds[index]
           << ", \"down_milliseconds\": "
           << value.down_milliseconds[index]
           << ", \"pair_milliseconds\": "
           << value.pair_milliseconds[index] << '}';
  }
  output << "\n        ],\n        \"budget\": {"
            "\"whole_product_projection_seconds\": "
         << value.whole_product_projection_budget_seconds
         << ", \"optimistic_pair_milliseconds\": "
         << value.optimistic_pair_milliseconds
         << ", \"optimistic_full_prompt_mlp_seconds\": "
         << value.optimistic_full_prompt_mlp_seconds
         << ", \"exceeded\": ";
  write_boolean(output, value.budget_exceeded);
  output << "},\n        \"zero\": {\"activation_fixture\": ";
  write_boolean(output, value.zero_activation_fixture);
  output << ", \"residual_fixture\": ";
  write_boolean(output, value.zero_residual_fixture);
  output << ", \"inputs_remained_zero\": ";
  write_boolean(output, value.inputs_remained_zero);
  output << ", \"outputs_all_zero\": ";
  write_boolean(output, value.outputs_all_zero);
  output << "},\n        \"guard\": {\"intact\": ";
  write_boolean(output, value.guards_intact);
  output << "},\n        \"events\": {\"destroyed\": ";
  write_boolean(output, value.timing_events_destroyed);
  output << "},\n        \"attempted\": ";
  write_boolean(output, value.attempted);
  output << ",\n        \"completed\": ";
  write_boolean(output, value.completed);
  output << "\n      }";
}
#endif

[[nodiscard]] std::string_view quick_kill_status(
    const ProbeEvidence& evidence) {
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION)
  if (!evidence.p40000_quick_kill_requested ||
      !evidence.layer0_m192_oracle.value.has_value()) {
    return "invalid";
  }
  const auto& result = *evidence.layer0_m192_oracle.value;
  const auto& kill_test = result.p40000_kill_test;
  const bool valid =
      evidence.execution_identity_valid &&
      evidence.source_build_provenance_valid &&
      evidence.binary_provenance_valid && evidence.exact_target_inventory &&
      evidence.mutually_exclusive_prefill_sidecars_empty &&
      evidence.layer0_m192_oracle.ok() && evidence.synchronize_attempted &&
      evidence.synchronize_cuda_error == 0 &&
      evidence.engine_destroy_completed && kill_test.attempted &&
      kill_test.completed &&
      kill_test.token_count == 40'000U && kill_test.model_layers == 64U &&
      kill_test.full_prompt_mlp_layers == 63U &&
      kill_test.warmup_pairs != 0U && kill_test.measured_pairs == 2U &&
      kill_test.zero_activation_fixture && kill_test.zero_residual_fixture &&
      kill_test.inputs_remained_zero && kill_test.outputs_all_zero &&
      kill_test.guards_intact && kill_test.timing_events_destroyed;
  if (!valid) {
    return "invalid";
  }
  if (kill_test.decision ==
          "reject-current-nvfp4-bf16-hmma-kernel-skeleton" &&
      kill_test.budget_exceeded) {
    return "reject";
  }
  if (kill_test.decision == "continue-to-broader-composed-validation" &&
      !kill_test.budget_exceeded) {
    return "continue";
  }
#else
  (void)evidence;
#endif
  return "invalid";
}

[[nodiscard]] bool capture_provenance(ProbeEvidence& evidence) {
  std::error_code error;
  evidence.executable_path =
      std::filesystem::canonical("/proc/self/exe", error);
  if (error) {
    evidence.executable_path.clear();
  } else {
    evidence.executable_bytes = static_cast<std::uint64_t>(
        std::filesystem::file_size(evidence.executable_path, error));
  }
  const core::Sha256FileResult binary =
      core::sha256_file("/proc/self/exe");
  if (binary.ok()) {
    evidence.executable_sha256 = binary.digest->hex();
  }
  evidence.binary_provenance_valid =
      !error && !evidence.executable_path.empty() &&
      evidence.executable_bytes != 0U &&
      is_lower_hex(evidence.executable_sha256, 64U);

  evidence.source_build_provenance_valid =
      provenance::kGitClean &&
      is_lower_hex(provenance::kGitCommit, 40U) &&
      is_lower_hex(provenance::kGitTree, 40U) &&
      is_lower_hex(provenance::kBuildReceiptSha256, 64U) &&
      is_lower_hex(provenance::kCxxCompilerSha256, 64U) &&
      is_lower_hex(provenance::kCudaCompilerSha256, 64U) &&
      std::string_view(provenance::kBuildType) == "Release" &&
      std::string_view(provenance::kCxxGlobalFlags).empty() &&
      std::string_view(provenance::kCudaGlobalFlags).empty() &&
      std::string_view(provenance::kCxxReleaseFlags) == "-O3 -DNDEBUG" &&
      std::string_view(provenance::kCudaReleaseFlags) == "-O3 -DNDEBUG" &&
      provenance::kBuildTesting &&
      std::string_view(provenance::kCxxCompilerId) == "GNU" &&
      std::string_view(provenance::kCudaCompilerId) == "NVIDIA" &&
      std::string_view(provenance::kCudaToolkitVersion) == "13.3.73" &&
      std::string_view(provenance::kQ3xCudaArchitectures) == "87" &&
      std::string_view(provenance::kEffectiveCudaArchitectures) == "87" &&
      provenance::kAotSystemAdmission &&
      provenance::kTargetProjectionAdmission &&
      provenance::kTargetDeviceAssetsAdmission &&
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION)
      provenance::kTargetLayer0M192OracleAdmission &&
#endif
      !provenance::kFp8MarlinPrefillAdmission &&
      !provenance::kNvFp4MarlinPrefillAdmission &&
      !provenance::kP40PackedProjectionAdmission &&
      !provenance::kP40PackedNvFp4V2Admission &&
      !provenance::kP40VllmMarlinParityAdmission;
  return evidence.source_build_provenance_valid &&
         evidence.binary_provenance_valid;
}

[[nodiscard]] bool publish_create_only(
    const std::filesystem::path& path, const std::string_view payload) {
  const std::filesystem::path temporary =
      path.string() + ".tmp." + std::to_string(getpid());
  const int descriptor = open(temporary.c_str(),
                              O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
  if (descriptor < 0) {
    return false;
  }
  bool ok = true;
  std::size_t written = 0U;
  while (written < payload.size()) {
    const ssize_t count =
        write(descriptor, payload.data() + written, payload.size() - written);
    if (count <= 0) {
      ok = false;
      break;
    }
    written += static_cast<std::size_t>(count);
  }
  if (ok && fsync(descriptor) != 0) {
    ok = false;
  }
  if (close(descriptor) != 0) {
    ok = false;
  }
  if (ok && link(temporary.c_str(), path.c_str()) != 0) {
    ok = false;
  }
  (void)unlink(temporary.c_str());
  return ok;
}

[[nodiscard]] bool write_evidence(const std::filesystem::path& path,
                                  const ProbeEvidence& evidence) {
  std::ostringstream output;
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION)
  if (evidence.p40000_quick_kill_requested) {
    output << "{\n  \"schema_version\": 1,\n  \"artifact\": "
              "\"q3x_sm87_target_aot_p40000_quick_kill\",\n"
              "  \"status\": \""
           << quick_kill_status(evidence)
           << "\",\n  \"claim_boundary\": "
              "\"checkpoint-weight-only optimistic early-stop authority for "
              "the current NVFP4 M128N256K64 persistent16 BF16-HMMA "
              "GateUp+SiLU and Down+residual kernel skeleton only; no real-"
              "activation, FP8-projection, Attention, GDN, whole-runner, "
              "generation, public-launcher, production-route, hardware-bound, "
              "or AOT-class authority\",\n"
              "  \"model_directory\": ";
  } else {
    output << "{\n  \"schema_version\": 3,\n  \"artifact\": "
              "\"q3x_sm87_target_aot_real_checkpoint_preparation\",\n"
              "  \"status\": \""
           << (evidence.passed ? "pass" : "fail")
           << "\",\n  \"claim_boundary\": "
              "\"prepare, authenticated device readback, private attachment, "
              "fixed-M192 layer-0 numerical/resource oracle, and destruction "
              "only; no generation, timing/performance, runner, public-launcher, "
              "or production-route authority\",\n"
              "  \"model_directory\": ";
  }
  write_json_string(output, evidence.model_directory.string());
  output << ",\n  \"execution_identity\": {\n    \"boot_id\": ";
  write_json_string(output, evidence.boot_id);
  output << ",\n    \"child_pid\": " << evidence.child_pid
         << ",\n    \"child_started_at_utc\": ";
  write_json_string(output, evidence.child_started_at_utc);
  output << ",\n    \"child_evidence_finished_at_utc\": ";
  write_json_string(output, evidence.child_evidence_finished_at_utc);
  output << ",\n    \"valid\": ";
  write_boolean(output, evidence.execution_identity_valid);
  output << "\n  },\n  \"source\": {\n    \"git_commit\": ";
#else
  if (evidence.p40000_quick_kill_requested) {
    output << "{\n  \"schema_version\": 1,\n  \"artifact\": "
              "\"q3x_sm87_target_aot_p40000_quick_kill\",\n"
              "  \"status\": \"invalid\",\n"
              "  \"claim_boundary\": "
              "\"checkpoint-weight-only optimistic early-stop authority for "
              "the current NVFP4 M128N256K64 persistent16 BF16-HMMA "
              "GateUp+SiLU and Down+residual kernel skeleton only; no real-"
              "activation, FP8-projection, Attention, GDN, whole-runner, "
              "generation, public-launcher, production-route, hardware-bound, "
              "or AOT-class authority\",\n"
              "  \"model_directory\": ";
  } else {
    output << "{\n  \"schema_version\": 2,\n  \"artifact\": "
              "\"q3x_sm87_target_aot_real_checkpoint_preparation\",\n"
              "  \"status\": \""
           << (evidence.passed ? "pass" : "fail")
           << "\",\n  \"claim_boundary\": "
              "\"prepare, authenticated device readback, private attachment, "
              "and destruction only; no launcher, generation, numerical, "
              "performance, or production-route authority\",\n"
              "  \"model_directory\": ";
  }
  write_json_string(output, evidence.model_directory.string());
  output << ",\n  \"source\": {\n    \"git_commit\": ";
#endif
  write_json_string(output, provenance::kGitCommit);
  output << ",\n    \"git_tree\": ";
  write_json_string(output, provenance::kGitTree);
  output << ",\n    \"clean_at_build\": ";
  write_boolean(output, provenance::kGitClean);
  output << "\n  },\n  \"binary\": {\n    \"path\": ";
  write_json_string(output, evidence.executable_path.string());
  output << ",\n    \"bytes\": " << evidence.executable_bytes
         << ",\n    \"self_sha256\": ";
  write_json_string(output, evidence.executable_sha256);
  output << "\n  },\n  \"build\": {\n    \"receipt_sha256\": ";
  write_json_string(output, provenance::kBuildReceiptSha256);
  output << ",\n    \"cmake_version\": ";
  write_json_string(output, provenance::kCmakeVersion);
  output << ",\n    \"generator\": ";
  write_json_string(output, provenance::kGenerator);
  output << ",\n    \"build_type\": ";
  write_json_string(output, provenance::kBuildType);
  output << ",\n    \"build_testing\": ";
  write_boolean(output, provenance::kBuildTesting);
  output << ",\n    \"q3x_cuda_architectures\": ";
  write_json_string(output, provenance::kQ3xCudaArchitectures);
  output << ",\n    \"effective_cuda_architectures\": ";
  write_json_string(output, provenance::kEffectiveCudaArchitectures);
  output << ",\n    \"cxx_compiler\": {\n      \"path\": ";
  write_json_string(output, provenance::kCxxCompiler);
  output << ",\n      \"id\": ";
  write_json_string(output, provenance::kCxxCompilerId);
  output << ",\n      \"version\": ";
  write_json_string(output, provenance::kCxxCompilerVersion);
  output << ",\n      \"sha256\": ";
  write_json_string(output, provenance::kCxxCompilerSha256);
  output << "\n    },\n    \"cuda_compiler\": {\n      \"path\": ";
  write_json_string(output, provenance::kCudaCompiler);
  output << ",\n      \"id\": ";
  write_json_string(output, provenance::kCudaCompilerId);
  output << ",\n      \"version\": ";
  write_json_string(output, provenance::kCudaCompilerVersion);
  output << ",\n      \"sha256\": ";
  write_json_string(output, provenance::kCudaCompilerSha256);
  output << "\n    },\n    \"cuda_toolkit_version\": ";
  write_json_string(output, provenance::kCudaToolkitVersion);
  output << ",\n    \"cxx_release_flags\": ";
  write_json_string(output, provenance::kCxxReleaseFlags);
  output << ",\n    \"cuda_release_flags\": ";
  write_json_string(output, provenance::kCudaReleaseFlags);
  output << ",\n    \"cxx_global_flags\": ";
  write_json_string(output, provenance::kCxxGlobalFlags);
  output << ",\n    \"cuda_global_flags\": ";
  write_json_string(output, provenance::kCudaGlobalFlags);
  output << ",\n    \"admissions\": {\n"
            "      \"aot_system\": ";
  write_boolean(output, provenance::kAotSystemAdmission);
  output << ",\n      \"target_projection\": ";
  write_boolean(output, provenance::kTargetProjectionAdmission);
  output << ",\n      \"target_device_assets\": ";
  write_boolean(output, provenance::kTargetDeviceAssetsAdmission);
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION)
  output << ",\n      \"target_layer0_m192_oracle\": ";
  write_boolean(output, provenance::kTargetLayer0M192OracleAdmission);
#endif
  output << ",\n      \"fp8_marlin_prefill\": ";
  write_boolean(output, provenance::kFp8MarlinPrefillAdmission);
  output << ",\n      \"nvfp4_marlin_prefill\": ";
  write_boolean(output, provenance::kNvFp4MarlinPrefillAdmission);
  output << ",\n      \"p40_packed_projection\": ";
  write_boolean(output, provenance::kP40PackedProjectionAdmission);
  output << ",\n      \"p40_packed_nvfp4_v2\": ";
  write_boolean(output, provenance::kP40PackedNvFp4V2Admission);
  output << ",\n      \"p40_vllm_marlin_parity\": ";
  write_boolean(output, provenance::kP40VllmMarlinParityAdmission);
  output << "\n    }\n  },\n  \"device\": {\n    \"ordinal\": "
         << evidence.device_ordinal
         << ",\n    \"total_before_bytes\": "
         << evidence.device_total_before_bytes
         << ",\n    \"total_during_bytes\": "
         << evidence.device_total_during_bytes
         << ",\n    \"total_after_destroy_bytes\": "
         << evidence.device_total_after_destroy_bytes
         << ",\n    \"free_before_bytes\": "
         << evidence.device_free_before_bytes
         << ",\n    \"free_during_bytes\": "
         << evidence.device_free_during_bytes
         << ",\n    \"free_after_destroy_bytes\": "
         << evidence.device_free_after_destroy_bytes
         << ",\n    \"destroy_recovery_tolerance_bytes\": "
         << kDestroyRecoveryToleranceBytes
         << "\n  },\n  \"cuda_errors\": {\n    \"device_query\": "
         << evidence.device_query_cuda_error
         << ",\n    \"initial_mem_info\": "
         << evidence.initial_mem_info_cuda_error
         << ",\n    \"during_mem_info\": "
         << evidence.during_mem_info_cuda_error
         << ",\n    \"synchronize_before_destroy\": "
         << evidence.synchronize_cuda_error
         << ",\n    \"final_mem_info\": "
         << evidence.final_mem_info_cuda_error
         << "\n  },\n  \"attempted\": {\n    \"device_query\": ";
  write_boolean(output, evidence.device_query_attempted);
  output << ",\n    \"initial_mem_info\": ";
  write_boolean(output, evidence.initial_mem_info_attempted);
  output << ",\n    \"engine_create\": ";
  write_boolean(output, evidence.engine_create_attempted);
  output << ",\n    \"during_mem_info\": ";
  write_boolean(output, evidence.during_mem_info_attempted);
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION)
  output << ",\n    \"layer0_m192_oracle\": ";
  write_boolean(output, evidence.layer0_m192_oracle_attempted);
#endif
  output << ",\n    \"synchronize_before_destroy\": ";
  write_boolean(output, evidence.synchronize_attempted);
  output << ",\n    \"engine_destroy\": ";
  write_boolean(output, evidence.engine_destroy_attempted);
  output << ",\n    \"final_mem_info\": ";
  write_boolean(output, evidence.final_mem_info_attempted);
  output << ",\n    \"memory_recovery_audit\": ";
  write_boolean(output, evidence.memory_recovery_audit_attempted);
  output << "\n  },\n  \"checks\": {\n    \"source_build_provenance_valid\": ";
  write_boolean(output, evidence.source_build_provenance_valid);
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION)
  output << ",\n    \"execution_identity_valid\": ";
  write_boolean(output, evidence.execution_identity_valid);
#endif
  output << ",\n    \"binary_provenance_valid\": ";
  write_boolean(output, evidence.binary_provenance_valid);
  output << ",\n    \"engine_created\": ";
  write_boolean(output, evidence.engine_created);
  output << ",\n    \"c512_competition_path_exercised\": ";
  write_boolean(output, evidence.c512_competition_path_exercised);
  output << ",\n    \"during_total_matches_initial\": ";
  write_boolean(output, evidence.during_total_matches_initial);
  output << ",\n    \"exact_target_inventory\": ";
  write_boolean(output, evidence.exact_target_inventory);
  output << ",\n    \"mutually_exclusive_prefill_sidecars_empty\": ";
  write_boolean(output, evidence.mutually_exclusive_prefill_sidecars_empty);
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION)
  output << ",\n    \"layer0_m192_oracle_passed\": ";
  write_boolean(output, evidence.layer0_m192_oracle.ok());
#endif
  output << ",\n    \"engine_destroy_completed\": ";
  write_boolean(output, evidence.engine_destroy_completed);
  output << ",\n    \"final_total_matches_initial\": ";
  write_boolean(output, evidence.final_total_matches_initial);
  output << ",\n    \"memory_recovered_after_destroy\": ";
  write_boolean(output, evidence.memory_recovered_after_destroy);
  output << "\n  },\n  \"timing\": {\n    \"probe_wall_milliseconds\": "
         << evidence.wall_milliseconds
         << ",\n    \"engine_total_milliseconds\": "
         << std::setprecision(17) << evidence.load.total_milliseconds
         << ",\n    \"resident_load_milliseconds\": "
         << evidence.load.resident_load_milliseconds
         << ",\n    \"weight_bind_milliseconds\": "
         << evidence.load.weight_bind_milliseconds
         << ",\n    \"request_state_milliseconds\": "
         << evidence.load.request_state_milliseconds
         << ",\n    \"target_prepare_attach_milliseconds\": "
         << evidence.load.target_aot_projection_device_asset_milliseconds
         << ",\n    \"runner_factory_milliseconds\": "
         << evidence.load.runner_factory_milliseconds
         << "\n  },\n  \"resident\": {\n    \"bytes_read\": "
         << evidence.load.resident.bytes_read
         << ",\n    \"bytes_copied\": "
         << evidence.load.resident.bytes_copied
         << ",\n    \"bytes_skipped\": "
         << evidence.load.resident.bytes_skipped
         << ",\n    \"shard_workers\": "
         << evidence.load.resident.shard_workers
         << ",\n    \"pinned_staging_bytes\": "
         << evidence.load.resident.pinned_staging_bytes
         << ",\n    \"shards\": [";
  for (std::size_t index = 0U; index < evidence.load.resident.shards.size();
       ++index) {
    const auto& shard = evidence.load.resident.shards[index];
    output << (index == 0U ? "\n" : ",\n") << "      {\"filename\": ";
    write_json_string(output, shard.filename);
    output << ", \"sha256\": ";
    write_json_string(output, shard.sha256);
    output << ", \"bytes_read\": " << shard.bytes_read
           << ", \"bytes_copied\": " << shard.bytes_copied << "}";
  }
  if (!evidence.load.resident.shards.empty()) {
    output << '\n';
  }
  output << "    ]\n  },\n  \"request\": {\n    \"arena_bytes\": "
         << evidence.load.request_arena_bytes
         << ",\n    \"max_sequence_length\": "
         << evidence.load.request_max_sequence_length
         << ",\n    \"prefill_chunk_size\": "
         << evidence.load.request_prefill_chunk_size
         << "\n  },\n  \"target_aot\": {\n    \"requested\": ";
  write_boolean(output,
                evidence.load.target_aot_projection_device_assets_requested);
  output << ",\n    \"enabled\": ";
  write_boolean(output,
                evidence.load.target_aot_projection_device_assets_enabled);
  output << ",\n    \"attached\": ";
  write_boolean(output,
                evidence.load.target_aot_projection_device_assets_attached);
  output << ",\n    \"artifacts\": "
         << evidence.load.target_aot_projection_device_asset_artifacts
         << ",\n    \"sources\": "
         << evidence.load.target_aot_projection_device_asset_sources
         << ",\n    \"arena_bytes\": "
         << evidence.load.target_aot_projection_device_asset_bytes
         << ",\n    \"host_staging_peak_bytes\": "
         << evidence.load.target_aot_projection_host_staging_peak_bytes
         << ",\n    \"source_d2h_bytes\": "
         << evidence.load.target_aot_projection_source_d2h_bytes
         << ",\n    \"payload_h2d_bytes\": "
         << evidence.load.target_aot_projection_payload_h2d_bytes
         << ",\n    \"verification_d2h_bytes\": "
         << evidence.load.target_aot_projection_verification_d2h_bytes
         << ",\n    \"verified_payload_catalog_sha256\": ";
  write_json_string(
      output,
      evidence.load.target_aot_projection_verified_payload_catalog_sha256);
  output << ",\n    \"owner_identity\": "
         << evidence.load.target_aot_projection_owner_identity
         << ",\n    \"allocation_identity\": "
         << evidence.load.target_aot_projection_allocation_identity
         << ",\n    \"device_ordinal\": "
         << evidence.load.target_aot_projection_device_ordinal
         << "\n  },\n  \"mutually_exclusive_prefill_sidecars\": {\n"
            "    \"fp8_qkv_enabled\": ";
  write_boolean(output, evidence.load.fp8_prefill_qkv_sidecars_enabled);
  output << ",\n    \"fp8_supermatrix_enabled\": ";
  write_boolean(output, evidence.load.fp8_prefill_supermatrix_sidecars_enabled);
  output << ",\n    \"fp8_marlin_enabled\": ";
  write_boolean(output, evidence.load.fp8_marlin_prefill_sidecars_enabled);
  output << ",\n    \"nvfp4_marlin_enabled\": ";
  write_boolean(output, evidence.load.nvfp4_marlin_prefill_sidecars_enabled);
  output << ",\n    \"p40_packed_enabled\": ";
  write_boolean(output, evidence.load.p40_packed_projection_assets_enabled);
  output << ",\n    \"p40_vllm_parity_enabled\": ";
  write_boolean(output,
                evidence.load.nvfp4_marlin_p40_parity_sidecars_enabled);
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION)
  output << "\n  },\n  \"layer0_m192_oracle\": {\n    \"attempted\": ";
  write_boolean(output, evidence.layer0_m192_oracle_attempted);
  output << ",\n    \"result_available\": ";
  write_boolean(output, evidence.layer0_m192_oracle.value.has_value());
  output << ",\n    \"passed\": ";
  write_boolean(output, evidence.layer0_m192_oracle.ok());
  output << ",\n    \"result\": ";
  if (!evidence.layer0_m192_oracle.value.has_value()) {
    output << "null";
  } else {
    const auto& value = *evidence.layer0_m192_oracle.value;
    output << "{\n      \"layer_index\": " << value.layer_index
           << ",\n      \"token_count\": " << value.token_count
           << ",\n      \"full_token_count\": " << value.full_token_count
           << ",\n      \"tail_token_count\": " << value.tail_token_count
           << ",\n      \"baseline_gate_launches\": "
           << value.baseline_gate_launches
           << ",\n      \"baseline_up_launches\": "
           << value.baseline_up_launches
           << ",\n      \"baseline_down_launches\": "
           << value.baseline_down_launches
           << ",\n      \"candidate_gate_up_launches\": "
           << value.candidate_gate_up_launches
           << ",\n      \"candidate_down_launches\": "
           << value.candidate_down_launches
           << ",\n      \"replay_gate_up_launches\": "
           << value.replay_gate_up_launches
           << ",\n      \"replay_down_launches\": "
           << value.replay_down_launches
           << ",\n      \"owner_attachment_authenticated\": ";
    write_boolean(output, value.owner_attachment_authenticated);
    output << ",\n      \"canonical_layer0_weights\": ";
    write_boolean(output, value.canonical_layer0_weights);
    output << ",\n      \"activation_preserved_after_candidate\": ";
    write_boolean(output, value.activation_preserved_after_candidate);
    output << ",\n      \"residual_preserved_after_candidate\": ";
    write_boolean(output, value.residual_preserved_after_candidate);
    output << ",\n      \"activation_preserved_after_replay\": ";
    write_boolean(output, value.activation_preserved_after_replay);
    output << ",\n      \"residual_preserved_after_replay\": ";
    write_boolean(output, value.residual_preserved_after_replay);
    output << ",\n      \"activation_fixture\": ";
    write_fixture(output, value.activation_fixture);
    output << ",\n      \"residual_fixture\": ";
    write_fixture(output, value.residual_fixture);
    output << ",\n      \"receipt\": {\n        \"baseline_route\": ";
    write_json_string(output, value.receipt.baseline_route);
    output << ",\n        \"candidate_plan\": ";
    write_json_string(output, value.receipt.candidate_plan);
    output << ",\n        \"candidate_layout\": ";
    write_json_string(output, value.receipt.candidate_layout);
    output << ",\n        \"candidate_publication\": ";
    write_json_string(output, value.receipt.candidate_publication);
    output << ",\n        \"owner_identity\": "
           << value.receipt.owner_identity
           << ",\n        \"allocation_identity\": "
           << value.receipt.allocation_identity
           << ",\n        \"arena_bytes\": " << value.receipt.arena_bytes
           << ",\n        \"device_ordinal\": "
           << value.receipt.device_ordinal
           << ",\n        \"artifact_count\": "
           << value.receipt.artifact_count
           << ",\n        \"verified_payload_catalog_sha256\": ";
    write_json_string(output,
                      value.receipt.verified_payload_catalog_sha256);
    output << ",\n        \"attachment_exact\": ";
    write_boolean(output, value.receipt.attachment_exact);
    output << ",\n        \"complete_owner_allocation_ranges_checked\": ";
    write_boolean(output,
                  value.receipt.complete_owner_allocation_ranges_checked);
    output << ",\n        \"complete_owner_allocation_exact_cover\": ";
    write_boolean(output,
                  value.receipt.complete_owner_allocation_exact_cover);
    output << ",\n        \"complete_owner_allocation_nonoverlap\": ";
    write_boolean(output,
                  value.receipt.complete_owner_allocation_nonoverlap);
    output << ",\n        \"oracle_allocations_disjoint_from_owner\": ";
    write_boolean(output,
                  value.receipt.oracle_allocations_disjoint_from_owner);
    output << ",\n        \"gate_up\": ";
    write_artifact(output, value.receipt.gate_up);
    output << ",\n        \"down\": ";
    write_artifact(output, value.receipt.down);
    output << "\n      }";
    output << ",\n      \"gate_up_resources\": ";
    write_kernel_resources(output, value.gate_up_resources);
    output << ",\n      \"down_resources\": ";
    write_kernel_resources(output, value.down_resources);
    output << ",\n      \"gate_up\": ";
    write_boundary(output, value.gate_up);
    output << ",\n      \"down_residual\": ";
    write_boundary(output, value.down_residual);
    if (evidence.p40000_quick_kill_requested) {
      output << ",\n      \"p40000_kill_test\": ";
      write_p40000_kill_test(output, value.p40000_kill_test);
    }
    output << ",\n      \"cleanup\": {\"synchronize_attempted\": ";
    write_boolean(output, value.cleanup.synchronize_attempted);
    output << ", \"synchronize_cuda_error\": "
           << value.cleanup.synchronize_cuda_error
           << ", \"device_frees_attempted\": "
           << value.cleanup.device_frees_attempted
           << ", \"device_frees_succeeded\": "
           << value.cleanup.device_frees_succeeded
           << ", \"first_device_free_cuda_error\": "
           << value.cleanup.first_device_free_cuda_error
           << ", \"stream_destroy_attempted\": ";
    write_boolean(output, value.cleanup.stream_destroy_attempted);
    output << ", \"stream_destroy_cuda_error\": "
           << value.cleanup.stream_destroy_cuda_error << ", \"passed\": ";
    write_boolean(output, value.cleanup.passed);
    output << '}';
    output << ",\n      \"passed\": ";
    write_boolean(output, value.passed);
    output << "\n    }";
  }
  output << ",\n    \"diagnostic\": {\n      \"code\": "
         << static_cast<unsigned int>(
                evidence.layer0_m192_oracle.diagnostic.code)
         << ",\n      \"stage\": ";
  write_json_string(output, evidence.layer0_m192_oracle.diagnostic.stage);
  output << ",\n      \"message\": ";
  write_json_string(output, evidence.layer0_m192_oracle.diagnostic.message);
  output << ",\n      \"context\": ";
  write_json_string(output, evidence.layer0_m192_oracle.diagnostic.context);
  output << ",\n      \"cuda_error\": "
         << evidence.layer0_m192_oracle.diagnostic.cuda_error
         << "\n    }\n  }";
#else
  output << "\n  }";
#endif
  output << ",\n  \"diagnostic\": {\n    \"code\": "
         << static_cast<unsigned int>(evidence.diagnostic.code)
         << ",\n    \"stage\": ";
  write_json_string(output, evidence.diagnostic.stage);
  output << ",\n    \"message\": ";
  write_json_string(output, evidence.diagnostic.message);
  output << ",\n    \"context\": ";
  write_json_string(output, evidence.diagnostic.context);
  output << ",\n    \"cuda_error\": " << evidence.diagnostic.cuda_error
         << "\n  }\n}\n";
  return publish_create_only(path, output.str());
}

}  // namespace

int main(const int argc, char** argv) {
  const bool p40000_quick_kill_requested =
      argc == 4 && std::string_view(argv[3]) == "--p40000-quick-kill";
  if (argc != 3 && !p40000_quick_kill_requested) {
    std::cerr << "usage: " << argv[0]
              << " MODEL_DIRECTORY EVIDENCE_JSON "
                 "[--p40000-quick-kill]\n";
    return 2;
  }

  static_assert(
      runtime::sm87_target_aot_projection_device_assets_compiled());
  ProbeEvidence evidence;
  evidence.p40000_quick_kill_requested = p40000_quick_kill_requested;
  evidence.model_directory = argv[1];
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION)
  evidence.child_started_at_utc = utc_now();
  evidence.boot_id = read_boot_id();
  evidence.child_pid = static_cast<std::uint64_t>(getpid());
  evidence.execution_identity_valid =
      !evidence.child_started_at_utc.empty() && !evidence.boot_id.empty() &&
      evidence.child_pid != 0U;
#endif
  const std::filesystem::path evidence_path = argv[2];
  const auto started = std::chrono::steady_clock::now();
  (void)capture_provenance(evidence);
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION)
  if (!evidence.execution_identity_valid ||
      !evidence.source_build_provenance_valid ||
      !evidence.binary_provenance_valid) {
    evidence.diagnostic.code = runtime::ReferenceEngineError::kInvalidArgument;
    evidence.diagnostic.stage = "provenance_gate";
    evidence.diagnostic.message =
        "real-checkpoint preparation requires boot/time/process execution "
        "identity, a clean frozen Release SM87 admission build, and a "
        "self-hashed running ELF";
  }
#else
  if (!evidence.source_build_provenance_valid ||
      !evidence.binary_provenance_valid) {
    evidence.diagnostic.code = runtime::ReferenceEngineError::kInvalidArgument;
    evidence.diagnostic.stage = "provenance_gate";
    evidence.diagnostic.message =
        "real-checkpoint preparation requires a clean, frozen Release SM87 "
        "admission build and a self-hashed running ELF";
  }
#endif

  evidence.device_query_attempted = true;
  cudaError_t status = cudaGetDevice(&evidence.device_ordinal);
  evidence.device_query_cuda_error = static_cast<int>(status);
  if (status == cudaSuccess) {
    std::size_t free_before = 0U;
    std::size_t total_before = 0U;
    evidence.initial_mem_info_attempted = true;
    status = cudaMemGetInfo(&free_before, &total_before);
    evidence.initial_mem_info_cuda_error = static_cast<int>(status);
    evidence.device_free_before_bytes = free_before;
    evidence.device_total_before_bytes = total_before;
  }

  runtime::ReferenceEngineCreateResult created;
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION)
  if (evidence.execution_identity_valid &&
      evidence.source_build_provenance_valid &&
      evidence.binary_provenance_valid &&
      evidence.device_query_cuda_error == 0 &&
      evidence.initial_mem_info_attempted &&
      evidence.initial_mem_info_cuda_error == 0) {
#else
  if (evidence.source_build_provenance_valid &&
      evidence.binary_provenance_valid &&
      evidence.device_query_cuda_error == 0 &&
      evidence.initial_mem_info_attempted &&
      evidence.initial_mem_info_cuda_error == 0) {
#endif
    runtime::ReferenceEngineOptions options;
    options.projection_backend = runtime::ProjectionBackend::kSm87WeightOnly;
    options.prepare_sm87_target_aot_projection_device_assets = true;
    options.request_options.max_sequence_length = 512U;
    options.request_options.prefill_chunk_size =
        runtime::kMaximumRequestPrefillChunkSize;
    options.request_options.max_arena_bytes = 512ULL * 1024ULL * 1024ULL;

    evidence.engine_create_attempted = true;
    created = runtime::create_reference_engine(evidence.model_directory,
                                               options);
    evidence.diagnostic = created.diagnostic;
    evidence.engine_created = static_cast<bool>(created);
    if (created) {
      evidence.load = created.value->load_stats();
      evidence.during_mem_info_attempted = true;
      std::size_t free_during = 0U;
      std::size_t total_during = 0U;
      status = cudaMemGetInfo(&free_during, &total_during);
      evidence.during_mem_info_cuda_error = static_cast<int>(status);
      evidence.device_free_during_bytes = free_during;
      evidence.device_total_during_bytes = total_during;
      evidence.during_total_matches_initial =
          status == cudaSuccess &&
          total_during == evidence.device_total_before_bytes;

      evidence.c512_competition_path_exercised =
          evidence.load.request_max_sequence_length == 512U &&
          evidence.load.request_prefill_chunk_size ==
              runtime::kMaximumRequestPrefillChunkSize &&
          evidence.load.request_arena_bytes == kC512RequestArenaBytes;
      evidence.exact_target_inventory =
          evidence.load.target_aot_projection_device_assets_requested &&
          evidence.load.target_aot_projection_device_assets_enabled &&
          evidence.load.target_aot_projection_device_assets_attached &&
          evidence.load.target_aot_projection_device_asset_artifacts ==
              runtime::kSm87TargetAotProjectionDeviceArtifactCount &&
          evidence.load.target_aot_projection_device_asset_sources ==
              runtime::kSm87TargetAotProjectionDeviceSourceCount &&
          evidence.load.target_aot_projection_device_asset_bytes ==
              runtime::kSm87TargetAotProjectionDeviceArenaBytes &&
          evidence.load.target_aot_projection_host_staging_peak_bytes ==
              runtime::kSm87TargetAotProjectionMaximumHostStagingBytes &&
          evidence.load.target_aot_projection_source_d2h_bytes ==
              runtime::kSm87TargetAotProjectionCanonicalSourceD2hBytes &&
          evidence.load.target_aot_projection_payload_h2d_bytes ==
              runtime::kSm87TargetAotProjectionDeviceArenaBytes &&
          evidence.load.target_aot_projection_verification_d2h_bytes ==
              runtime::kSm87TargetAotProjectionDeviceArenaBytes &&
          is_lower_hex(
              evidence.load
                  .target_aot_projection_verified_payload_catalog_sha256,
              64U) &&
          evidence.load.target_aot_projection_owner_identity != 0U &&
          evidence.load.target_aot_projection_allocation_identity != 0U &&
          evidence.load.target_aot_projection_device_ordinal ==
              evidence.device_ordinal;
      evidence.mutually_exclusive_prefill_sidecars_empty =
          evidence.c512_competition_path_exercised &&
          !evidence.load.fp8_prefill_qkv_sidecars_enabled &&
          !evidence.load.fp8_prefill_supermatrix_sidecars_enabled &&
          !evidence.load.fp8_marlin_prefill_sidecars_enabled &&
          !evidence.load.nvfp4_marlin_prefill_sidecars_enabled &&
          !evidence.load.p40_packed_projection_assets_enabled &&
          !evidence.load.nvfp4_marlin_p40_parity_sidecars_enabled;
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION)
      if (evidence.exact_target_inventory &&
          evidence.mutually_exclusive_prefill_sidecars_empty) {
        evidence.layer0_m192_oracle_attempted = true;
        if (evidence.p40000_quick_kill_requested) {
          evidence.layer0_m192_oracle =
              oracle::Sm87TargetAotLayer0M192OracleAccess::screen(
                  *created.value, true);
        } else {
          evidence.layer0_m192_oracle =
              oracle::Sm87TargetAotLayer0M192OracleAccess::screen(
                  *created.value, false);
        }
        if (!evidence.layer0_m192_oracle) {
          evidence.diagnostic = evidence.layer0_m192_oracle.diagnostic;
        }
      }
#endif
    }
  }

  evidence.synchronize_attempted = true;
  evidence.synchronize_cuda_error =
      static_cast<int>(cudaDeviceSynchronize());

  if (created.value.has_value()) {
    evidence.engine_destroy_attempted = true;
    created.value.reset();
    evidence.engine_destroy_completed = !created.value.has_value();
  }

  evidence.final_mem_info_attempted = true;
  std::size_t free_after = 0U;
  std::size_t total_after = 0U;
  const cudaError_t final_status = cudaMemGetInfo(&free_after, &total_after);
  evidence.final_mem_info_cuda_error = static_cast<int>(final_status);
  evidence.device_free_after_destroy_bytes = free_after;
  evidence.device_total_after_destroy_bytes = total_after;
  evidence.final_total_matches_initial =
      final_status == cudaSuccess &&
      evidence.initial_mem_info_attempted &&
      evidence.initial_mem_info_cuda_error == 0 &&
      total_after == evidence.device_total_before_bytes;
  evidence.memory_recovery_audit_attempted =
      evidence.initial_mem_info_attempted &&
      evidence.initial_mem_info_cuda_error == 0 &&
      evidence.final_mem_info_attempted &&
      evidence.final_mem_info_cuda_error == 0;
  evidence.memory_recovered_after_destroy =
      evidence.memory_recovery_audit_attempted &&
      evidence.final_total_matches_initial &&
      (free_after >= evidence.device_free_before_bytes ||
       evidence.device_free_before_bytes - free_after <=
           kDestroyRecoveryToleranceBytes);

  evidence.wall_milliseconds = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - started)
          .count());
  evidence.passed =
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION)
      evidence.execution_identity_valid &&
#endif
      evidence.source_build_provenance_valid &&
      evidence.binary_provenance_valid && evidence.device_query_attempted &&
      evidence.device_query_cuda_error == 0 &&
      evidence.initial_mem_info_attempted &&
      evidence.initial_mem_info_cuda_error == 0 &&
      evidence.engine_create_attempted && evidence.engine_created &&
      evidence.c512_competition_path_exercised &&
      evidence.during_mem_info_attempted &&
      evidence.during_mem_info_cuda_error == 0 &&
      evidence.during_total_matches_initial &&
      evidence.exact_target_inventory &&
      evidence.mutually_exclusive_prefill_sidecars_empty &&
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION)
      evidence.layer0_m192_oracle_attempted &&
      evidence.layer0_m192_oracle.ok() &&
#endif
      evidence.synchronize_attempted && evidence.synchronize_cuda_error == 0 &&
      evidence.engine_destroy_attempted &&
      evidence.engine_destroy_completed && evidence.final_mem_info_attempted &&
      evidence.final_mem_info_cuda_error == 0 &&
      evidence.final_total_matches_initial &&
      evidence.memory_recovery_audit_attempted &&
      evidence.memory_recovered_after_destroy;

#if defined(Q3X_ENABLE_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION)
  evidence.child_evidence_finished_at_utc = utc_now();
  evidence.execution_identity_valid =
      evidence.execution_identity_valid &&
      !evidence.child_evidence_finished_at_utc.empty();
  evidence.passed = evidence.passed && evidence.execution_identity_valid;
#endif

  if (!write_evidence(evidence_path, evidence)) {
    std::cerr << "failed to publish create-only evidence: " << evidence_path
              << '\n';
    return evidence.p40000_quick_kill_requested ? 4 : 3;
  }
  if (evidence.p40000_quick_kill_requested) {
    const std::string_view status = quick_kill_status(evidence);
    std::cout << "target_aot_p40000_quick_kill=" << status << '\n'
              << "evidence=" << evidence_path << '\n';
    if (status == "continue") {
      return 0;
    }
    if (status == "reject") {
      return 3;
    }
    return 1;
  }
  std::cout << "target_aot_real_checkpoint_prepare="
            << (evidence.passed ? "pass" : "fail") << '\n'
            << "evidence=" << evidence_path << '\n';
  return evidence.passed ? 0 : 1;
}
