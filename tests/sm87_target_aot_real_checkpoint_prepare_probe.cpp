#include "q3x/core/sha256.h"
#include "q3x/runtime/reference_engine.h"
#include "q3x/runtime/sm87_target_aot_projection_device_assets.h"
#include "sm87_target_aot_prepare_provenance.h"

#include <cuda_runtime.h>

#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>

namespace core = q3x::core;
namespace runtime = q3x::runtime;
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

struct ProbeEvidence final {
  bool passed = false;
  std::filesystem::path model_directory;

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
  bool memory_recovered_after_destroy = false;
  runtime::ReferenceEngineLoadStats load;
  runtime::ReferenceEngineDiagnostic diagnostic;
};

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
      std::string_view(provenance::kCudaToolkitVersion) == "13.3.33" &&
      std::string_view(provenance::kQ3xCudaArchitectures) == "87" &&
      std::string_view(provenance::kEffectiveCudaArchitectures) == "87" &&
      provenance::kAotSystemAdmission &&
      provenance::kTargetProjectionAdmission &&
      provenance::kTargetDeviceAssetsAdmission &&
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
  output << "{\n  \"schema_version\": 2,\n  \"artifact\": "
            "\"q3x_sm87_target_aot_real_checkpoint_preparation\",\n"
            "  \"status\": \""
         << (evidence.passed ? "pass" : "fail")
         << "\",\n  \"claim_boundary\": "
            "\"prepare, authenticated device readback, private attachment, "
            "and destruction only; no launcher, generation, numerical, "
            "performance, or production-route authority\",\n"
            "  \"model_directory\": ";
  write_json_string(output, evidence.model_directory.string());
  output << ",\n  \"source\": {\n    \"git_commit\": ";
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

}  // namespace

int main(const int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0]
              << " MODEL_DIRECTORY EVIDENCE_JSON\n";
    return 2;
  }

  static_assert(
      runtime::sm87_target_aot_projection_device_assets_compiled());
  ProbeEvidence evidence;
  evidence.model_directory = argv[1];
  const std::filesystem::path evidence_path = argv[2];
  const auto started = std::chrono::steady_clock::now();
  (void)capture_provenance(evidence);
  if (!evidence.source_build_provenance_valid ||
      !evidence.binary_provenance_valid) {
    evidence.diagnostic.code = runtime::ReferenceEngineError::kInvalidArgument;
    evidence.diagnostic.stage = "provenance_gate";
    evidence.diagnostic.message =
        "real-checkpoint preparation requires a clean, frozen Release SM87 "
        "admission build and a self-hashed running ELF";
  }

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
  if (evidence.source_build_provenance_valid &&
      evidence.binary_provenance_valid &&
      evidence.device_query_cuda_error == 0 &&
      evidence.initial_mem_info_attempted &&
      evidence.initial_mem_info_cuda_error == 0) {
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
      evidence.synchronize_attempted && evidence.synchronize_cuda_error == 0 &&
      evidence.engine_destroy_attempted &&
      evidence.engine_destroy_completed && evidence.final_mem_info_attempted &&
      evidence.final_mem_info_cuda_error == 0 &&
      evidence.final_total_matches_initial &&
      evidence.memory_recovery_audit_attempted &&
      evidence.memory_recovered_after_destroy;

  if (!write_evidence(evidence_path, evidence)) {
    std::cerr << "failed to publish create-only evidence: " << evidence_path
              << '\n';
    return 3;
  }
  std::cout << "target_aot_real_checkpoint_prepare="
            << (evidence.passed ? "pass" : "fail") << '\n'
            << "evidence=" << evidence_path << '\n';
  return evidence.passed ? 0 : 1;
}
