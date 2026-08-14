#include "q3x/core/sha256.h"
#include "q3x/runtime/prefill_execution_plan.h"
#include "q3x/runtime/reference_engine.h"
#include "q3x/runtime/sm87_target_aot_projection_complete_device_assets.h"
#include "sm87_macrofeed_v4_engine_lifetime_probe_internal.h"

#include <cuda_runtime.h>

#include <fcntl.h>
#include <unistd.h>

#include <cctype>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>

#if !defined(Q3X_MACROFEED_V4_LIFETIME_PROBE_BUILD_TESTING) || \
    !defined(Q3X_MACROFEED_V4_LIFETIME_PROBE_BUILD_TYPE) || \
    !defined(Q3X_MACROFEED_V4_LIFETIME_PROBE_Q3X_CUDA_ARCHITECTURES) || \
    !defined(Q3X_MACROFEED_V4_LIFETIME_PROBE_EFFECTIVE_CUDA_ARCHITECTURES) || \
    !defined(Q3X_MACROFEED_V4_LIFETIME_PROBE_REPOSITORY_ROOT)
#error "the V4 Engine lifetime probe requires explicit build configuration"
#endif

namespace core = q3x::core;
namespace runtime = q3x::runtime;
namespace lifetime_probe =
    q3x::runtime::sm87_macrofeed_v4_engine_lifetime_probe_detail;

namespace {

constexpr std::uint64_t kDestroyRecoveryToleranceBytes =
    64ULL * 1024ULL * 1024ULL;
constexpr std::uint64_t kRetainedFreeBytes =
    4ULL * 1024ULL * 1024ULL * 1024ULL;

struct ProbeEvidence final {
  bool passed = false;
  std::filesystem::path model_directory;
  std::filesystem::path repository_work_root;
  std::filesystem::path evidence_output_path;
  bool evidence_output_repository_local = false;
  bool binary_provenance_valid = false;
  std::filesystem::path executable_path;
  std::uint64_t executable_bytes = 0U;
  std::string executable_sha256;
  std::string build_type = Q3X_MACROFEED_V4_LIFETIME_PROBE_BUILD_TYPE;
  std::string q3x_cuda_architectures =
      Q3X_MACROFEED_V4_LIFETIME_PROBE_Q3X_CUDA_ARCHITECTURES;
  std::string effective_cuda_architectures =
      Q3X_MACROFEED_V4_LIFETIME_PROBE_EFFECTIVE_CUDA_ARCHITECTURES;
  bool build_testing =
      Q3X_MACROFEED_V4_LIFETIME_PROBE_BUILD_TESTING != 0;
  bool macrofeed_v3_admission = false;
  bool macrofeed_v4_admission = false;
  bool ndebug = false;
  bool build_configuration_exact = false;
  int device_ordinal = -1;
  int device_query_cuda_error = 0;
  int initial_mem_info_cuda_error = 0;
  int during_mem_info_cuda_error = 0;
  int pre_destroy_sync_cuda_error = 0;
  int post_destroy_sync_cuda_error = 0;
  int final_mem_info_cuda_error = 0;
  std::uint64_t free_before_bytes = 0U;
  std::uint64_t free_during_bytes = 0U;
  std::uint64_t free_after_destroy_bytes = 0U;
  std::uint64_t total_before_bytes = 0U;
  std::uint64_t total_during_bytes = 0U;
  std::uint64_t total_after_destroy_bytes = 0U;
  bool engine_create_attempted = false;
  bool engine_created = false;
  bool engine_valid = false;
  bool complete_aot_inventory_exact = false;
  bool macrofeed_v3_engine_contract_exact = false;
  lifetime_probe::Sm87MacroFeedV4EngineLifetimeConstructionSnapshot
      construction_snapshot;
  std::size_t construction_snapshot_count = 0U;
  bool construction_snapshot_hook_exact = false;
  bool construction_snapshot_valid = false;
  bool snapshot_matches_complete_aot_load_stats = false;
  bool live_engine_reduced_free_memory = false;
  bool pre_destroy_synchronized = false;
  bool engine_destroy_attempted = false;
  bool engine_destroyed = false;
  bool post_destroy_synchronized = false;
  bool memory_recovered_after_destroy = false;
  runtime::ReferenceEngineLoadStats load;
  runtime::ReferenceEngineDiagnostic diagnostic;
};

struct EvidencePathValidation final {
  bool valid = false;
  std::filesystem::path repository_work_root;
  std::filesystem::path output_path;
  std::string error;
};

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
          output << "\\u" << std::hex << std::setw(4)
                 << std::setfill('0') << static_cast<unsigned int>(byte)
                 << std::dec << std::setfill(' ');
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

[[nodiscard]] bool is_nonzero_lower_hex(const std::string_view value,
                                        const std::size_t size) {
  if (value.size() != size) {
    return false;
  }
  bool nonzero = false;
  for (const unsigned char byte : value) {
    if (!std::isdigit(byte) && !(byte >= 'a' && byte <= 'f')) {
      return false;
    }
    nonzero = nonzero || byte != '0';
  }
  return nonzero;
}

[[nodiscard]] bool is_strictly_within(
    const std::filesystem::path& child,
    const std::filesystem::path& parent) {
  auto child_part = child.begin();
  for (auto parent_part = parent.begin(); parent_part != parent.end();
       ++parent_part, ++child_part) {
    if (child_part == child.end() || *child_part != *parent_part) {
      return false;
    }
  }
  return child_part != child.end();
}

[[nodiscard]] EvidencePathValidation validate_evidence_path(
    const std::filesystem::path& requested_path) {
  EvidencePathValidation validation;
  std::error_code error;
  const std::filesystem::path repository_root = std::filesystem::canonical(
      Q3X_MACROFEED_V4_LIFETIME_PROBE_REPOSITORY_ROOT, error);
  if (error || repository_root.empty()) {
    validation.error = "repository root is not canonical";
    return validation;
  }
  const std::filesystem::path repository_work_root =
      repository_root / ".q3x-work";
  const std::filesystem::file_status repository_work_status =
      std::filesystem::symlink_status(repository_work_root, error);
  if (error || repository_work_status.type() !=
                   std::filesystem::file_type::directory) {
    validation.error =
        "repository .q3x-work must be a real directory, not a symlink";
    return validation;
  }
  validation.repository_work_root =
      std::filesystem::canonical(repository_work_root, error);
  if (error || validation.repository_work_root.empty() ||
      !std::filesystem::is_directory(validation.repository_work_root, error) ||
      error) {
    validation.error = "repository .q3x-work directory is unavailable";
    return validation;
  }
  if (requested_path.empty() || requested_path.filename().empty() ||
      requested_path.extension() != ".json") {
    validation.error = "evidence output must name one .json file";
    return validation;
  }
  const std::filesystem::path absolute_path =
      requested_path.is_absolute()
          ? requested_path
          : std::filesystem::absolute(requested_path, error);
  if (error || absolute_path.empty()) {
    validation.error = "evidence output path is not absolute";
    return validation;
  }
  const std::filesystem::path canonical_parent =
      std::filesystem::canonical(absolute_path.parent_path(), error);
  if (error || canonical_parent.empty() ||
      !std::filesystem::is_directory(canonical_parent, error) || error) {
    validation.error = "evidence output parent is not a canonical directory";
    return validation;
  }
  validation.output_path = canonical_parent / absolute_path.filename();
  if (!is_strictly_within(validation.output_path,
                          validation.repository_work_root)) {
    validation.error =
        "evidence output must remain below repository .q3x-work";
    return validation;
  }
  const std::filesystem::file_status output_status =
      std::filesystem::symlink_status(validation.output_path, error);
  if (!error) {
    if (output_status.type() != std::filesystem::file_type::not_found) {
      validation.error = "evidence output already exists";
      return validation;
    }
  } else if (error != std::errc::no_such_file_or_directory) {
    validation.error = "evidence output status cannot be established";
    return validation;
  }
  validation.valid = true;
  return validation;
}

void capture_construction_snapshot(
    void* const context,
    const lifetime_probe::
        Sm87MacroFeedV4EngineLifetimeConstructionSnapshot& snapshot)
    noexcept {
  if (context == nullptr) {
    return;
  }
  auto& evidence = *static_cast<ProbeEvidence*>(context);
  evidence.construction_snapshot = snapshot;
  ++evidence.construction_snapshot_count;
}

void capture_binary_and_build_configuration(ProbeEvidence& evidence) {
  std::error_code canonical_error;
  evidence.executable_path =
      std::filesystem::canonical("/proc/self/exe", canonical_error);
  std::error_code size_error;
  if (!canonical_error) {
    evidence.executable_bytes = static_cast<std::uint64_t>(
        std::filesystem::file_size(evidence.executable_path, size_error));
  }
  const core::Sha256FileResult executable =
      core::sha256_file("/proc/self/exe");
  if (executable.ok()) {
    evidence.executable_sha256 = executable.digest->hex();
  }
  evidence.binary_provenance_valid =
      !canonical_error && !size_error && !evidence.executable_path.empty() &&
      evidence.executable_bytes != 0U &&
      is_nonzero_lower_hex(evidence.executable_sha256, 64U);

#if defined(Q3X_ENABLE_SM87_MACROFEED_V3_P40_EXECUTOR_ADMISSION)
  evidence.macrofeed_v3_admission = true;
#endif
#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
  evidence.macrofeed_v4_admission = true;
#endif
#if defined(NDEBUG)
  evidence.ndebug = true;
#endif
  evidence.build_configuration_exact =
      evidence.build_testing && evidence.macrofeed_v3_admission &&
      evidence.macrofeed_v4_admission && evidence.ndebug &&
      evidence.build_type == "Release" &&
      evidence.q3x_cuda_architectures == "87" &&
      evidence.effective_cuda_architectures == "87";
}

[[nodiscard]] bool publish_create_only(
    const std::filesystem::path& path, const std::string_view payload) {
  const std::string output_name = path.filename().string();
  const std::string temporary_name =
      output_name + ".tmp." + std::to_string(getpid());
  const int directory_descriptor =
      open(path.parent_path().c_str(),
           O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
  if (directory_descriptor < 0) {
    return false;
  }
  const int descriptor =
      openat(directory_descriptor, temporary_name.c_str(),
             O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
  if (descriptor < 0) {
    (void)close(directory_descriptor);
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
  if (ok && linkat(directory_descriptor, temporary_name.c_str(),
                   directory_descriptor, output_name.c_str(), 0) != 0) {
    ok = false;
  }
  if (unlinkat(directory_descriptor, temporary_name.c_str(), 0) != 0) {
    ok = false;
  }
  if (fsync(directory_descriptor) != 0) {
    ok = false;
  }
  if (close(directory_descriptor) != 0) {
    ok = false;
  }
  return ok;
}

[[nodiscard]] bool write_evidence(const std::filesystem::path& path,
                                  const ProbeEvidence& evidence) {
  std::ostringstream output;
  output << "{\n  \"schema_version\": 2,\n"
            "  \"artifact\": "
            "\"q3x_sm87_macrofeed_v4_real_checkpoint_engine_lifetime\",\n"
            "  \"status\": \""
         << (evidence.passed ? "pass" : "fail")
         << "\",\n  \"claim_boundary\": ";
  write_json_string(
      output,
      "real-checkpoint Engine construction, normal 48-GDN-QKVZ factory "
      "postcondition, CUDA quiescence, destruction, and free-memory recovery "
      "within the declared 64 MiB tolerance only; the recovery gate does not "
      "exclude smaller retained resources and grants no generation, "
      "numerical, performance, selector, API-route, or production authority");
  output << ",\n  \"model_directory\": ";
  write_json_string(output, evidence.model_directory.string());
  output << ",\n  \"evidence_output\": {\n    \"path\": ";
  write_json_string(output, evidence.evidence_output_path.string());
  output << ",\n    \"repository_work_root\": ";
  write_json_string(output, evidence.repository_work_root.string());
  output << ",\n    \"repository_local\": ";
  write_boolean(output, evidence.evidence_output_repository_local);
  output << "\n  }";
  output << ",\n  \"binary\": {\n    \"path\": ";
  write_json_string(output, evidence.executable_path.string());
  output << ",\n    \"bytes\": " << evidence.executable_bytes
         << ",\n    \"sha256\": ";
  write_json_string(output, evidence.executable_sha256);
  output << ",\n    \"valid\": ";
  write_boolean(output, evidence.binary_provenance_valid);
  output << "\n  },\n  \"build_configuration\": {\n"
         << "    \"build_type\": ";
  write_json_string(output, evidence.build_type);
  output << ",\n    \"q3x_cuda_architectures\": ";
  write_json_string(output, evidence.q3x_cuda_architectures);
  output << ",\n    \"effective_cuda_architectures\": ";
  write_json_string(output, evidence.effective_cuda_architectures);
  output << ",\n    \"build_testing\": ";
  write_boolean(output, evidence.build_testing);
  output << ",\n    \"macrofeed_v3_admission\": ";
  write_boolean(output, evidence.macrofeed_v3_admission);
  output << ",\n    \"macrofeed_v4_admission\": ";
  write_boolean(output, evidence.macrofeed_v4_admission);
  output << ",\n    \"ndebug\": ";
  write_boolean(output, evidence.ndebug);
  output << ",\n    \"exact\": ";
  write_boolean(output, evidence.build_configuration_exact);
  output << "\n  },\n  \"device\": {\n    \"ordinal\": "
         << evidence.device_ordinal
         << ",\n    \"query_cuda_error\": "
         << evidence.device_query_cuda_error << "\n  },\n"
         << "  \"engine\": {\n    \"create_attempted\": ";
  write_boolean(output, evidence.engine_create_attempted);
  output << ",\n    \"created\": ";
  write_boolean(output, evidence.engine_created);
  output << ",\n    \"valid\": ";
  write_boolean(output, evidence.engine_valid);
  output << ",\n    \"complete_aot_inventory_exact\": ";
  write_boolean(output, evidence.complete_aot_inventory_exact);
  output << ",\n    \"macrofeed_v3_engine_contract_exact\": ";
  write_boolean(output, evidence.macrofeed_v3_engine_contract_exact);
  output << ",\n    \"construction_snapshot_count\": "
         << evidence.construction_snapshot_count
         << ",\n    \"construction_snapshot_hook_exact\": ";
  write_boolean(output, evidence.construction_snapshot_hook_exact);
  output << ",\n    \"construction_snapshot_valid\": ";
  write_boolean(output, evidence.construction_snapshot_valid);
  output << ",\n    \"snapshot_matches_complete_aot_load_stats\": ";
  write_boolean(output, evidence.snapshot_matches_complete_aot_load_stats);
  output << ",\n    \"destroy_attempted\": ";
  write_boolean(output, evidence.engine_destroy_attempted);
  output << ",\n    \"destroyed\": ";
  write_boolean(output, evidence.engine_destroyed);
  output << "\n  },\n  \"complete_aot\": {\n"
         << "    \"artifacts\": "
         << evidence.load.target_aot_complete_projection_artifacts
         << ",\n    \"sources\": "
         << evidence.load.target_aot_complete_projection_sources
         << ",\n    \"bytes\": "
         << evidence.load.target_aot_complete_projection_bytes
         << ",\n    \"catalog_sha256\": ";
  write_json_string(
      output, evidence.load.target_aot_complete_projection_catalog_sha256);
  output << ",\n    \"owner_identity\": "
         << evidence.load.target_aot_complete_projection_owner_identity
         << ",\n    \"allocation_identity\": "
         << evidence.load.target_aot_complete_projection_allocation_identity
         << ",\n    \"device_ordinal\": "
         << evidence.load.target_aot_complete_projection_device_ordinal
         << "\n  },\n  \"macrofeed_v4_construction_snapshot\": {\n"
         << "    \"startup_package_identity\": "
         << evidence.construction_snapshot.startup_package_identity
         << ",\n    \"execution_package_identity\": "
         << evidence.construction_snapshot.execution_package_identity
         << ",\n    \"execution_startup_package_identity\": "
         << evidence.construction_snapshot.execution_startup_package_identity
         << ",\n    \"startup_owner_identity\": "
         << evidence.construction_snapshot.startup_owner_identity
         << ",\n    \"startup_allocation_identity\": "
         << evidence.construction_snapshot.startup_allocation_identity
         << ",\n    \"startup_device_identity\": "
         << evidence.construction_snapshot.startup_device_identity
         << ",\n    \"startup_device_ordinal\": "
         << evidence.construction_snapshot.startup_device_ordinal
         << ",\n    \"execution_device_ordinal\": "
         << evidence.construction_snapshot.execution_device_ordinal
         << ",\n    \"startup_gdn_qkvz_catalog_identity\": "
         << evidence.construction_snapshot.startup_gdn_qkvz_catalog_identity
         << ",\n    \"startup_gdn_qkvz_binding_count\": "
         << evidence.construction_snapshot.startup_gdn_qkvz_binding_count
         << ",\n    \"execution_gdn_qkvz_catalog_identity\": "
         << evidence.construction_snapshot.execution_gdn_qkvz_catalog_identity
         << ",\n    \"execution_gdn_qkvz_binding_count\": "
         << evidence.construction_snapshot.execution_gdn_qkvz_binding_count
         << ",\n    \"synthetic_t1_gdn_layer0_source\": ";
  write_boolean(
      output,
      evidence.construction_snapshot.synthetic_t1_gdn_layer0_source);
  output << ",\n    \"owned_bytes\": "
         << evidence.construction_snapshot.owned_bytes
         << ",\n    \"valid\": ";
  write_boolean(output, evidence.construction_snapshot_valid);
  output << "\n  },\n  \"memory\": {\n"
         << "    \"initial_mem_info_cuda_error\": "
         << evidence.initial_mem_info_cuda_error
         << ",\n    \"during_mem_info_cuda_error\": "
         << evidence.during_mem_info_cuda_error
         << ",\n    \"final_mem_info_cuda_error\": "
         << evidence.final_mem_info_cuda_error
         << ",\n    \"free_before_bytes\": "
         << evidence.free_before_bytes
         << ",\n    \"free_during_bytes\": "
         << evidence.free_during_bytes
         << ",\n    \"free_after_destroy_bytes\": "
         << evidence.free_after_destroy_bytes
         << ",\n    \"total_before_bytes\": "
         << evidence.total_before_bytes
         << ",\n    \"total_during_bytes\": "
         << evidence.total_during_bytes
         << ",\n    \"total_after_destroy_bytes\": "
         << evidence.total_after_destroy_bytes
         << ",\n    \"destroy_recovery_tolerance_bytes\": "
         << kDestroyRecoveryToleranceBytes
         << ",\n    \"live_engine_reduced_free_memory\": ";
  write_boolean(output, evidence.live_engine_reduced_free_memory);
  output << ",\n    \"pre_destroy_sync_cuda_error\": "
         << evidence.pre_destroy_sync_cuda_error
         << ",\n    \"post_destroy_sync_cuda_error\": "
         << evidence.post_destroy_sync_cuda_error
         << ",\n    \"recovered_after_destroy\": ";
  write_boolean(output, evidence.memory_recovered_after_destroy);
  output << "\n  },\n  \"diagnostic\": {\n    \"code\": "
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

[[nodiscard]] runtime::ReferenceEngineOptions make_engine_options() {
  runtime::ReferenceEngineOptions options;
  options.projection_backend = runtime::ProjectionBackend::kSm87WeightOnly;
  options.generation_route = runtime::ReferenceGenerationRoute::kReference;
  options.enable_trace = false;
  options.decode_graph_cache_policy =
      runtime::ReferenceDecodeGraphCachePolicy::kDisabled;
  options.prefill_execution_mode =
      runtime::ReferencePrefillExecutionMode::kWholeRequestLayerMajor;
  options.prefill_full_attention_tactic =
      runtime::LayerMajorPrefillFullAttentionTactic::
          kNativeFlashInferExactWholePrompt;
  options.prefill_projection_tactic =
      runtime::LayerMajorPrefillProjectionTactic::
          kNativePromptWideP40MacroFeedV3;
  options.resident_options.min_free_bytes_after_load = kRetainedFreeBytes;
  options.request_options.batch_size = 1U;
  options.request_options.max_sequence_length =
      runtime::kLayerMajorPrefillPromptWideP40RequestCapacityTokens;
  options.request_options.prefill_chunk_size =
      runtime::kMaximumRequestPrefillChunkSize;
  options.request_options.max_arena_bytes =
      runtime::kLayerMajorPrefillPromptWideP40RequestArenaBytes;
  options.request_options.min_free_bytes_after_create = kRetainedFreeBytes;
  return options;
}

}  // namespace

int main(const int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0]
              << " MODEL_DIRECTORY EVIDENCE_JSON\n";
    return 2;
  }

  static_assert(runtime::kSm87TargetAotCompleteProjectionDeviceArtifactCount ==
                256U);
  static_assert(runtime::kSm87TargetAotCompleteProjectionDeviceSourceCount ==
                400U);
  static_assert(runtime::kLayerMajorPrefillPromptWideP40RequestCapacityTokens ==
                40'001U);
  static_assert(runtime::kLayerMajorPrefillPromptWideP40RequestArenaBytes ==
                8'640'542'976ULL);
  static_assert(
      kDestroyRecoveryToleranceBytes <
      lifetime_probe::kSm87MacroFeedV4EngineLifetimeMinimumOwnedArenaBytes);

  ProbeEvidence evidence;
  evidence.model_directory = argv[1];
  const EvidencePathValidation evidence_path = validate_evidence_path(argv[2]);
  if (!evidence_path.valid) {
    std::cerr << "invalid evidence output: " << evidence_path.error << '\n';
    return 2;
  }
  evidence.repository_work_root = evidence_path.repository_work_root;
  evidence.evidence_output_path = evidence_path.output_path;
  evidence.evidence_output_repository_local = true;
  capture_binary_and_build_configuration(evidence);

  cudaError_t cuda_status = cudaGetDevice(&evidence.device_ordinal);
  evidence.device_query_cuda_error = static_cast<int>(cuda_status);
  std::size_t free_before = 0U;
  std::size_t total_before = 0U;
  if (cuda_status == cudaSuccess) {
    cuda_status = cudaMemGetInfo(&free_before, &total_before);
    evidence.initial_mem_info_cuda_error = static_cast<int>(cuda_status);
    evidence.free_before_bytes = free_before;
    evidence.total_before_bytes = total_before;
  } else {
    evidence.initial_mem_info_cuda_error = static_cast<int>(cuda_status);
  }

  runtime::ReferenceEngineCreateResult created;
  if (!evidence.model_directory.empty() &&
      evidence.device_query_cuda_error == 0 &&
      evidence.initial_mem_info_cuda_error == 0) {
    evidence.engine_create_attempted = true;
    const lifetime_probe::
        Sm87MacroFeedV4EngineLifetimeConstructionSnapshotHook requested_hook{
            capture_construction_snapshot, &evidence};
    const auto previous_hook = lifetime_probe::
        exchange_sm87_macrofeed_v4_engine_lifetime_construction_snapshot_hook(
            requested_hook);
    created = runtime::create_reference_engine(evidence.model_directory,
                                               make_engine_options());
    const auto installed_hook = lifetime_probe::
        exchange_sm87_macrofeed_v4_engine_lifetime_construction_snapshot_hook(
            previous_hook);
    evidence.construction_snapshot_hook_exact =
        !previous_hook && installed_hook.callback == requested_hook.callback &&
        installed_hook.context == requested_hook.context;
    evidence.construction_snapshot_valid =
        evidence.construction_snapshot_count == 1U &&
        evidence.construction_snapshot.valid();
    evidence.diagnostic = created.diagnostic;
    evidence.engine_created = static_cast<bool>(created);
    if (created) {
      evidence.engine_valid = static_cast<bool>(*created.value);
      evidence.load = created.value->load_stats();
      const auto& load = evidence.load;
      evidence.complete_aot_inventory_exact =
          load.target_aot_complete_projection_device_assets_enabled &&
          load.target_aot_complete_projection_device_assets_attached &&
          load.target_aot_complete_projection_artifacts ==
              runtime::kSm87TargetAotCompleteProjectionDeviceArtifactCount &&
          load.target_aot_complete_projection_sources ==
              runtime::kSm87TargetAotCompleteProjectionDeviceSourceCount &&
          load.target_aot_complete_projection_bytes ==
              runtime::kSm87TargetAotCompleteProjectionDeviceArenaBytes &&
          load.target_aot_complete_projection_source_d2h_bytes ==
              runtime::kSm87TargetAotCompleteProjectionCanonicalSourceD2hBytes &&
          load.target_aot_complete_projection_payload_h2d_bytes ==
              runtime::kSm87TargetAotCompleteProjectionDeviceArenaBytes &&
          load.target_aot_complete_projection_verification_d2h_bytes ==
              runtime::kSm87TargetAotCompleteProjectionDeviceArenaBytes &&
          is_nonzero_lower_hex(
              load.target_aot_complete_projection_catalog_sha256, 64U) &&
          load.target_aot_complete_projection_owner_identity != 0U &&
          load.target_aot_complete_projection_allocation_identity != 0U &&
          load.target_aot_complete_projection_device_ordinal ==
              evidence.device_ordinal;
      evidence.macrofeed_v3_engine_contract_exact =
          load.projection_backend == runtime::ProjectionBackend::kSm87WeightOnly &&
          load.generation_route == runtime::ReferenceGenerationRoute::kReference &&
          load.decode_graph_cache_requested_policy ==
              runtime::ReferenceDecodeGraphCachePolicy::kDisabled &&
          load.decode_graph_cache_effective_policy ==
              runtime::ReferenceDecodeGraphCachePolicy::kDisabled &&
          load.request_arena_bytes ==
              runtime::kLayerMajorPrefillPromptWideP40RequestArenaBytes &&
          load.request_max_sequence_length ==
              runtime::kLayerMajorPrefillPromptWideP40RequestCapacityTokens &&
          load.request_prefill_chunk_size ==
              runtime::kMaximumRequestPrefillChunkSize &&
          load.request_memory_profile ==
              runtime::RequestMemoryProfile::kLayerMajorP40WholeCore &&
          !load.target_aot_projection_device_assets_requested &&
          !load.target_aot_projection_device_assets_enabled &&
          !load.target_aot_projection_device_assets_attached;
      evidence.snapshot_matches_complete_aot_load_stats =
          evidence.construction_snapshot_valid &&
          evidence.construction_snapshot.startup_owner_identity ==
              load.target_aot_complete_projection_owner_identity &&
          evidence.construction_snapshot.startup_allocation_identity ==
              load.target_aot_complete_projection_allocation_identity &&
          evidence.construction_snapshot.startup_device_identity != 0U &&
          evidence.construction_snapshot.startup_device_ordinal ==
              load.target_aot_complete_projection_device_ordinal &&
          evidence.construction_snapshot.execution_device_ordinal ==
              load.target_aot_complete_projection_device_ordinal &&
          evidence.construction_snapshot.startup_device_ordinal ==
              evidence.device_ordinal;

      std::size_t free_during = 0U;
      std::size_t total_during = 0U;
      cuda_status = cudaMemGetInfo(&free_during, &total_during);
      evidence.during_mem_info_cuda_error = static_cast<int>(cuda_status);
      evidence.free_during_bytes = free_during;
      evidence.total_during_bytes = total_during;
      evidence.live_engine_reduced_free_memory =
          cuda_status == cudaSuccess && total_during == total_before &&
          free_during < free_before;
    }
  }

  cuda_status = cudaDeviceSynchronize();
  evidence.pre_destroy_sync_cuda_error = static_cast<int>(cuda_status);
  evidence.pre_destroy_synchronized = cuda_status == cudaSuccess;

  if (created.value.has_value()) {
    evidence.engine_destroy_attempted = true;
    created.value.reset();
    evidence.engine_destroyed = !created.value.has_value();
  }

  cuda_status = cudaDeviceSynchronize();
  evidence.post_destroy_sync_cuda_error = static_cast<int>(cuda_status);
  evidence.post_destroy_synchronized = cuda_status == cudaSuccess;

  std::size_t free_after = 0U;
  std::size_t total_after = 0U;
  cuda_status = cudaMemGetInfo(&free_after, &total_after);
  evidence.final_mem_info_cuda_error = static_cast<int>(cuda_status);
  evidence.free_after_destroy_bytes = free_after;
  evidence.total_after_destroy_bytes = total_after;
  evidence.memory_recovered_after_destroy =
      cuda_status == cudaSuccess && evidence.initial_mem_info_cuda_error == 0 &&
      total_after == total_before &&
      (free_after >= free_before ||
       free_before - free_after <= kDestroyRecoveryToleranceBytes);

  evidence.passed =
      evidence.binary_provenance_valid &&
      evidence.build_configuration_exact &&
      evidence.device_query_cuda_error == 0 &&
      evidence.initial_mem_info_cuda_error == 0 &&
      evidence.engine_create_attempted && evidence.engine_created &&
      evidence.engine_valid && evidence.complete_aot_inventory_exact &&
      evidence.macrofeed_v3_engine_contract_exact &&
      evidence.construction_snapshot_hook_exact &&
      evidence.construction_snapshot_count == 1U &&
      evidence.construction_snapshot_valid &&
      evidence.snapshot_matches_complete_aot_load_stats &&
      evidence.during_mem_info_cuda_error == 0 &&
      evidence.live_engine_reduced_free_memory &&
      evidence.pre_destroy_synchronized &&
      evidence.engine_destroy_attempted && evidence.engine_destroyed &&
      evidence.post_destroy_synchronized &&
      evidence.final_mem_info_cuda_error == 0 &&
      evidence.memory_recovered_after_destroy;

  if (!write_evidence(evidence_path.output_path, evidence)) {
    std::cerr << "failed to publish create-only evidence: "
              << evidence_path.output_path
              << '\n';
    return 1;
  }
  std::cout << "macrofeed_v4_real_checkpoint_engine_lifetime="
            << (evidence.passed ? "pass" : "fail") << '\n'
            << "evidence=" << evidence_path.output_path << '\n';
  return evidence.passed ? 0 : 1;
}
