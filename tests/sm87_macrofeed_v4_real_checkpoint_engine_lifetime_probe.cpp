#include "q3x/core/sha256.h"
#include "q3x/runtime/prefill_execution_plan.h"
#include "q3x/runtime/reference_engine.h"
#include "q3x/runtime/sm87_target_aot_projection_complete_device_assets.h"
#include "sm87_macrofeed_v4_engine_lifetime_probe_internal.h"
#include "sm87_macrofeed_v4_lifetime_probe_provenance.h"

#include <cuda_runtime.h>

#include <fcntl.h>
#include <unistd.h>

#include <array>
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
namespace provenance =
    q3x::test::sm87_macrofeed_v4_lifetime_probe_provenance;

namespace {

constexpr std::uint64_t kDestroyRecoveryToleranceBytes =
    32ULL * 1024ULL * 1024ULL;
constexpr std::uint64_t kRetainedFreeBytes =
    4ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr std::string_view kPinnedModelIdentity =
    "nvidia/Qwen3.6-27B-NVFP4";
constexpr std::string_view kPinnedModelRevision =
    "0893e1606ff3d5f97a441f405d5fc541a6bdf404";

struct PinnedMetadataFile final {
  std::string_view filename;
  std::uint64_t bytes = 0U;
  std::string_view sha256;
};

constexpr std::array<PinnedMetadataFile, 3U> kPinnedMetadataFiles{{
    {"config.json", 88'567U,
     "c04a19ba293737ad7be4f6e96d6666cb7e479cbe19ecc0c289fad267135b0338"},
    {"hf_quant_config.json", 54'902U,
     "fd7200cd8bca2a8a5d777061521abf83e2deb97ab6bc2f04e7a0a3d3f8ecd5c1"},
    {"model.safetensors.index.json", 214'866U,
     "7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2"},
}};

struct ProbeEvidence final {
  bool passed = false;
  std::filesystem::path model_directory;
  std::filesystem::path canonical_model_directory;
  std::array<std::uint64_t, kPinnedMetadataFiles.size()> metadata_file_bytes{};
  std::array<std::string, kPinnedMetadataFiles.size()> metadata_file_sha256{};
  bool checkpoint_metadata_exact = false;
  bool resident_shard_manifest_exact = false;
  std::filesystem::path repository_work_root;
  std::filesystem::path evidence_output_path;
  bool evidence_output_repository_local = false;
  bool source_build_provenance_valid = false;
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
  bool macrofeed_v4_full_attention_preprocess_admission = false;
  bool macrofeed_v4_attention_c8000_admission = false;
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
  bool normal_catalogs_exact = false;
  bool full_attention_ownership_exact = false;
  bool reserve_chain_exact = false;
  bool owner_allocation_device_lifetime_chain_exact = false;
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

void capture_checkpoint_metadata(ProbeEvidence& evidence) {
  std::error_code error;
  evidence.canonical_model_directory =
      std::filesystem::canonical(evidence.model_directory, error);
  if (error || evidence.canonical_model_directory.empty() ||
      !std::filesystem::is_directory(evidence.canonical_model_directory,
                                     error) ||
      error) {
    return;
  }
  bool exact = true;
  for (std::size_t index = 0U; index < kPinnedMetadataFiles.size(); ++index) {
    const auto& expected = kPinnedMetadataFiles[index];
    const std::filesystem::path path =
        evidence.canonical_model_directory / expected.filename;
    const std::filesystem::file_status status =
        std::filesystem::symlink_status(path, error);
    if (error || status.type() != std::filesystem::file_type::regular) {
      exact = false;
      error.clear();
      continue;
    }
    const std::uint64_t bytes = static_cast<std::uint64_t>(
        std::filesystem::file_size(path, error));
    if (error) {
      exact = false;
      error.clear();
      continue;
    }
    evidence.metadata_file_bytes[index] = bytes;
    const core::Sha256FileResult hashed = core::sha256_file(path.string());
    if (!hashed.ok()) {
      exact = false;
      continue;
    }
    evidence.metadata_file_sha256[index] = hashed.digest->hex();
    exact = exact && bytes == expected.bytes &&
            evidence.metadata_file_sha256[index] == expected.sha256;
  }
  evidence.checkpoint_metadata_exact = exact;
}

[[nodiscard]] bool resident_shard_manifest_exact(
    const runtime::ResidentLoadStats& resident) {
  const auto& expected = runtime::pinned_qwen36_27b_shards();
  if (resident.shards.size() != expected.size() || expected.size() != 3U) {
    return false;
  }
  for (const auto& pinned : expected) {
    bool found = false;
    for (const auto& observed : resident.shards) {
      if (observed.filename == pinned.filename) {
        found = observed.bytes_read == pinned.file_size &&
                observed.sha256 == pinned.sha256 &&
                is_nonzero_lower_hex(observed.sha256, 64U);
        break;
      }
    }
    if (!found) {
      return false;
    }
  }
  return true;
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

  evidence.source_build_provenance_valid =
      is_nonzero_lower_hex(provenance::kGitCommit, 40U) &&
      is_nonzero_lower_hex(provenance::kGitTree, 40U) &&
      provenance::kGitClean &&
      is_nonzero_lower_hex(provenance::kBuildReceiptSha256, 64U) &&
      !std::string_view(provenance::kCmakeVersion).empty() &&
      !std::string_view(provenance::kGenerator).empty() &&
      !std::string_view(provenance::kCxxCompiler).empty() &&
      !std::string_view(provenance::kCxxCompilerId).empty() &&
      !std::string_view(provenance::kCxxCompilerVersion).empty() &&
      !std::string_view(provenance::kCudaCompiler).empty() &&
      !std::string_view(provenance::kCudaCompilerId).empty() &&
      !std::string_view(provenance::kCudaCompilerVersion).empty() &&
      !std::string_view(provenance::kCudaToolkitVersion).empty() &&
      std::string_view(provenance::kBuildReceiptSchema) ==
          "q3x.sm87.macrofeed-v4.engine-lifetime-build.v1" &&
      is_nonzero_lower_hex(provenance::kCxxCompilerSha256, 64U) &&
      is_nonzero_lower_hex(provenance::kCudaCompilerSha256, 64U);

#if defined(Q3X_ENABLE_SM87_MACROFEED_V3_P40_EXECUTOR_ADMISSION)
  evidence.macrofeed_v3_admission = true;
#endif
#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
  evidence.macrofeed_v4_admission = true;
#endif
#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_FULL_ATTENTION_PREPROCESS_ADMISSION)
  evidence.macrofeed_v4_full_attention_preprocess_admission = true;
#endif
#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_ATTENTION_C8000_ADMISSION)
  evidence.macrofeed_v4_attention_c8000_admission = true;
#endif
#if defined(NDEBUG)
  evidence.ndebug = true;
#endif
  evidence.build_configuration_exact =
      evidence.source_build_provenance_valid &&
      evidence.build_testing && evidence.macrofeed_v3_admission &&
      evidence.macrofeed_v4_admission &&
      evidence.macrofeed_v4_full_attention_preprocess_admission &&
      evidence.macrofeed_v4_attention_c8000_admission && evidence.ndebug &&
      evidence.build_type == "Release" &&
      evidence.q3x_cuda_architectures == "87" &&
      evidence.effective_cuda_architectures == "87" &&
      provenance::kBuildType == evidence.build_type &&
      provenance::kBuildTesting == evidence.build_testing &&
      provenance::kQ3xCudaArchitectures ==
          evidence.q3x_cuda_architectures &&
      provenance::kEffectiveCudaArchitectures ==
          evidence.effective_cuda_architectures &&
      provenance::kMacroFeedV3P40ExecutorAdmission &&
      provenance::kMacroFeedV4StartupPackageAdmission &&
      provenance::kMacroFeedV4ExecutionEventsAdmission &&
      provenance::kMacroFeedV4ExecutionPackageAdmission &&
      provenance::kMacroFeedV4NormResidualAdmission &&
      provenance::kMacroFeedV4Bf16AbAdmission &&
      provenance::kMacroFeedV4Fp8Admission &&
      provenance::kMacroFeedV4FullAttentionPreprocessAdmission &&
      provenance::kMacroFeedV4AttentionC8000Admission &&
      provenance::kMacroFeedV4GdnC8000Admission &&
      provenance::kMacroFeedV4NvFp4GateUpAdmission &&
      provenance::kMacroFeedV4NvFp4DownAdmission &&
      provenance::kTargetAotCompleteDeviceAssetsV2Admission;
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
  output << "{\n  \"schema_version\": 4,\n"
            "  \"artifact\": "
            "\"q3x_sm87_macrofeed_v4_real_checkpoint_engine_lifetime\",\n"
            "  \"status\": \""
         << (evidence.passed ? "pass" : "fail")
         << "\",\n  \"claim_boundary\": ";
  write_json_string(
      output,
      "pinned-metadata and three-resident-shard real-checkpoint Engine "
      "construction, normal 48-complete-GDN, 64-MLP-pair, and 16-Full-"
      "Attention catalog identities/folds, exact V4 KV and shared Engine "
      "RoPE ownership, staged aggregate reserve arguments, target-AOT "
      "owner/allocation/device lifetime chain, "
      "CUDA quiescence, destruction, and free-memory "
      "recovery within the declared 32 MiB tolerance only; the recovery gate "
      "does not exclude smaller retained resources and grants no generation, "
      "numerical, performance, selector, API-route, or production authority");
  output << ",\n  \"model_directory\": ";
  write_json_string(output, evidence.model_directory.string());
  output << ",\n  \"checkpoint_identity\": {\n    \"model\": ";
  write_json_string(output, kPinnedModelIdentity);
  output << ",\n    \"revision\": ";
  write_json_string(output, kPinnedModelRevision);
  output << ",\n    \"canonical_directory\": ";
  write_json_string(output, evidence.canonical_model_directory.string());
  output << ",\n    \"metadata_files\": [";
  for (std::size_t index = 0U; index < kPinnedMetadataFiles.size(); ++index) {
    const auto& expected = kPinnedMetadataFiles[index];
    output << (index == 0U ? "\n" : ",\n") << "      {\"filename\": ";
    write_json_string(output, expected.filename);
    output << ", \"bytes\": " << evidence.metadata_file_bytes[index]
           << ", \"sha256\": ";
    write_json_string(output, evidence.metadata_file_sha256[index]);
    output << ", \"expected_bytes\": " << expected.bytes
           << ", \"expected_sha256\": ";
    write_json_string(output, expected.sha256);
    output << ", \"exact\": ";
    write_boolean(output,
                  evidence.metadata_file_bytes[index] == expected.bytes &&
                      evidence.metadata_file_sha256[index] == expected.sha256);
    output << "}";
  }
  output << "\n    ],\n    \"metadata_exact\": ";
  write_boolean(output, evidence.checkpoint_metadata_exact);
  output << ",\n    \"resident_shards\": [";
  const auto& pinned_shards = runtime::pinned_qwen36_27b_shards();
  for (std::size_t index = 0U; index < pinned_shards.size(); ++index) {
    const auto& expected = pinned_shards[index];
    const runtime::ShardLoadStats* observed = nullptr;
    for (const auto& candidate : evidence.load.resident.shards) {
      if (candidate.filename == expected.filename) {
        observed = &candidate;
        break;
      }
    }
    output << (index == 0U ? "\n" : ",\n") << "      {\"filename\": ";
    write_json_string(output, expected.filename);
    output << ", \"bytes\": "
           << (observed == nullptr ? 0U : observed->bytes_read)
           << ", \"sha256\": ";
    write_json_string(output,
                      observed == nullptr ? std::string_view{}
                                          : observed->sha256);
    output << ", \"expected_bytes\": " << expected.file_size
           << ", \"expected_sha256\": ";
    write_json_string(output, expected.sha256);
    output << ", \"exact\": ";
    write_boolean(output,
                  observed != nullptr &&
                      observed->bytes_read == expected.file_size &&
                      observed->sha256 == expected.sha256);
    output << "}";
  }
  output << "\n    ],\n    \"resident_shards_exact\": ";
  write_boolean(output, evidence.resident_shard_manifest_exact);
  output << "\n  },\n  \"evidence_output\": {\n    \"path\": ";
  write_json_string(output, evidence.evidence_output_path.string());
  output << ",\n    \"repository_work_root\": ";
  write_json_string(output, evidence.repository_work_root.string());
  output << ",\n    \"repository_local\": ";
  write_boolean(output, evidence.evidence_output_repository_local);
  output << "\n  }";
  output << ",\n  \"source\": {\n    \"git_commit\": ";
  write_json_string(output, provenance::kGitCommit);
  output << ",\n    \"git_tree\": ";
  write_json_string(output, provenance::kGitTree);
  output << ",\n    \"clean_at_build\": ";
  write_boolean(output, provenance::kGitClean);
  output << ",\n    \"valid\": ";
  write_boolean(output, evidence.source_build_provenance_valid);
  output << "\n  },\n  \"binary\": {\n    \"path\": ";
  write_json_string(output, evidence.executable_path.string());
  output << ",\n    \"bytes\": " << evidence.executable_bytes
         << ",\n    \"self_sha256\": ";
  write_json_string(output, evidence.executable_sha256);
  output << ",\n    \"valid\": ";
  write_boolean(output, evidence.binary_provenance_valid);
  output << "\n  },\n  \"build_configuration\": {\n"
         << "    \"receipt_schema\": ";
  write_json_string(output, provenance::kBuildReceiptSchema);
  output << ",\n    \"receipt_sha256\": ";
  write_json_string(output, provenance::kBuildReceiptSha256);
  output << ",\n    \"cmake_version\": ";
  write_json_string(output, provenance::kCmakeVersion);
  output << ",\n    \"generator\": ";
  write_json_string(output, provenance::kGenerator);
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
  output << ",\n    \"flags\": {\n      \"cxx_release\": ";
  write_json_string(output, provenance::kCxxReleaseFlags);
  output << ",\n      \"cuda_release\": ";
  write_json_string(output, provenance::kCudaReleaseFlags);
  output << ",\n      \"cxx_global\": ";
  write_json_string(output, provenance::kCxxGlobalFlags);
  output << ",\n      \"cuda_global\": ";
  write_json_string(output, provenance::kCudaGlobalFlags);
  output << "\n    }";
  output << ",\n    \"build_type\": ";
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
  output << ",\n    \"macrofeed_v4_full_attention_preprocess_admission\": ";
  write_boolean(
      output, evidence.macrofeed_v4_full_attention_preprocess_admission);
  output << ",\n    \"macrofeed_v4_attention_c8000_admission\": ";
  write_boolean(output, evidence.macrofeed_v4_attention_c8000_admission);
  output << ",\n    \"required_admissions_exact\": ";
  write_boolean(
      output,
      provenance::kMacroFeedV3P40ExecutorAdmission &&
          provenance::kMacroFeedV4StartupPackageAdmission &&
          provenance::kMacroFeedV4ExecutionEventsAdmission &&
          provenance::kMacroFeedV4ExecutionPackageAdmission &&
          provenance::kMacroFeedV4NormResidualAdmission &&
          provenance::kMacroFeedV4Bf16AbAdmission &&
          provenance::kMacroFeedV4Fp8Admission &&
          provenance::kMacroFeedV4FullAttentionPreprocessAdmission &&
          provenance::kMacroFeedV4AttentionC8000Admission &&
          provenance::kMacroFeedV4GdnC8000Admission &&
          provenance::kMacroFeedV4NvFp4GateUpAdmission &&
          provenance::kMacroFeedV4NvFp4DownAdmission &&
          provenance::kTargetAotCompleteDeviceAssetsV2Admission);
  output << ",\n    \"admissions\": {\n"
         << "      \"macrofeed_v3_p40_executor\": ";
  write_boolean(output, provenance::kMacroFeedV3P40ExecutorAdmission);
  output << ",\n      \"macrofeed_v4_startup_package\": ";
  write_boolean(output, provenance::kMacroFeedV4StartupPackageAdmission);
  output << ",\n      \"macrofeed_v4_execution_events\": ";
  write_boolean(output, provenance::kMacroFeedV4ExecutionEventsAdmission);
  output << ",\n      \"macrofeed_v4_execution_package\": ";
  write_boolean(output, provenance::kMacroFeedV4ExecutionPackageAdmission);
  output << ",\n      \"macrofeed_v4_norm_residual\": ";
  write_boolean(output, provenance::kMacroFeedV4NormResidualAdmission);
  output << ",\n      \"macrofeed_v4_bf16_ab\": ";
  write_boolean(output, provenance::kMacroFeedV4Bf16AbAdmission);
  output << ",\n      \"macrofeed_v4_fp8\": ";
  write_boolean(output, provenance::kMacroFeedV4Fp8Admission);
  output << ",\n      \"macrofeed_v4_full_attention_preprocess\": ";
  write_boolean(
      output, provenance::kMacroFeedV4FullAttentionPreprocessAdmission);
  output << ",\n      \"macrofeed_v4_attention_c8000\": ";
  write_boolean(output, provenance::kMacroFeedV4AttentionC8000Admission);
  output << ",\n      \"macrofeed_v4_gdn_c8000\": ";
  write_boolean(output, provenance::kMacroFeedV4GdnC8000Admission);
  output << ",\n      \"macrofeed_v4_nvfp4_gate_up\": ";
  write_boolean(output, provenance::kMacroFeedV4NvFp4GateUpAdmission);
  output << ",\n      \"macrofeed_v4_nvfp4_down\": ";
  write_boolean(output, provenance::kMacroFeedV4NvFp4DownAdmission);
  output << ",\n      \"target_aot_complete_device_assets_v2\": ";
  write_boolean(output,
                provenance::kTargetAotCompleteDeviceAssetsV2Admission);
  output << "\n    }";
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
         << "\n  },\n  \"macrofeed_v4_normal_catalogs\": {\n"
         << "    \"normal_factory_branch\": ";
  write_boolean(output, evidence.construction_snapshot.normal_factory_branch);
  output << ",\n    \"startup_full_attention_source_catalog_identity\": "
         << evidence.construction_snapshot
                .startup_full_attention_source_catalog_identity
         << ",\n    \"complete_gdn_catalog_identity\": "
         << evidence.construction_snapshot
                .execution_complete_gdn_catalog_identity
         << ",\n    \"complete_gdn_binding_count\": "
         << evidence.construction_snapshot
                .execution_complete_gdn_binding_count
         << ",\n    \"mlp_pair_catalog_identity\": "
         << evidence.construction_snapshot.execution_mlp_pair_catalog_identity
         << ",\n    \"mlp_pair_binding_count\": "
         << evidence.construction_snapshot.execution_mlp_pair_binding_count
         << ",\n    \"full_attention_catalog_identity\": "
         << evidence.construction_snapshot
                .execution_full_attention_catalog_identity
         << ",\n    \"full_attention_binding_count\": "
         << evidence.construction_snapshot
                .execution_full_attention_binding_count
         << ",\n    \"retained_complete_gdn_catalog_fold_identity\": "
         << evidence.construction_snapshot
                .retained_complete_gdn_catalog_fold_identity
         << ",\n    \"retained_mlp_pair_catalog_fold_identity\": "
         << evidence.construction_snapshot
                .retained_mlp_pair_catalog_fold_identity
         << ",\n    \"retained_full_attention_catalog_fold_identity\": "
         << evidence.construction_snapshot
                .retained_full_attention_catalog_fold_identity
         << ",\n    \"full_attention_resource_bundle_identity\": "
         << evidence.construction_snapshot
                .full_attention_resource_bundle_identity
         << ",\n    \"exact\": ";
  write_boolean(output, evidence.normal_catalogs_exact);
  output << "\n  },\n  \"full_attention_owner\": {\n"
         << "    \"kv_allocation_identity\": "
         << evidence.construction_snapshot.kv_allocation_identity
         << ",\n    \"request_state_kv_allocation_identity\": "
         << evidence.construction_snapshot
                .request_state_kv_allocation_identity
         << ",\n    \"kv_allocation_begin\": "
         << evidence.construction_snapshot.kv_allocation_begin
         << ",\n    \"kv_allocation_bytes\": "
         << evidence.construction_snapshot.kv_allocation_bytes
         << ",\n    \"request_state_kv_allocation_bytes\": "
         << evidence.construction_snapshot
                .request_state_kv_allocation_bytes
         << ",\n    \"request_state_kv_physical_owner_bound\": ";
  write_boolean(
      output,
      evidence.construction_snapshot.request_state_kv_physical_owner_bound);
  output << ",\n    \"engine_rope_owner_identity\": "
         << evidence.construction_snapshot.engine_rope_owner_identity
         << ",\n    \"engine_rope_binding_identity\": "
         << evidence.construction_snapshot.engine_rope_binding_identity
         << ",\n    \"engine_rope_allocation_begin\": "
         << evidence.construction_snapshot.engine_rope_allocation_begin
         << ",\n    \"engine_rope_device_ordinal\": "
         << evidence.construction_snapshot.engine_rope_device_ordinal
         << ",\n    \"engine_rope_positions\": "
         << evidence.construction_snapshot.engine_rope_positions
         << ",\n    \"engine_rope_pairs\": "
         << evidence.construction_snapshot.engine_rope_pairs
         << ",\n    \"engine_rope_allocation_bytes\": "
         << evidence.construction_snapshot.engine_rope_allocation_bytes
         << ",\n    \"execution_owned_bytes\": "
         << evidence.construction_snapshot.owned_bytes
         << ",\n    \"rope_anchored_bytes\": "
         << evidence.construction_snapshot.anchored_bytes
         << ",\n    \"exact\": ";
  write_boolean(output, evidence.full_attention_ownership_exact);
  output << "\n  },\n  \"aggregate_reserve_chain\": {\n"
         << "    \"legacy_request_arena_bytes\": "
         << evidence.construction_snapshot.legacy_request_arena_bytes
         << ",\n    \"minimum_free_bytes_after_legacy_create\": "
         << evidence.construction_snapshot
                .minimum_free_bytes_after_legacy_create
         << ",\n    \"minimum_free_bytes_after_execution_create\": "
         << evidence.construction_snapshot
                .execution_minimum_free_bytes_after_create
         << ",\n    \"minimum_free_bytes_after_rope_create\": "
         << evidence.construction_snapshot
                .minimum_free_bytes_after_rope_create
         << ",\n    \"minimum_free_bytes_after_complete_aot_create\": "
         << evidence.construction_snapshot
                .minimum_free_bytes_after_complete_aot_create
         << ",\n    \"execution_required_device_allocation_bytes\": "
         << evidence.construction_snapshot
                .execution_required_device_allocation_bytes
         << ",\n    \"execution_free_bytes_before_allocations\": "
         << evidence.construction_snapshot
                .execution_free_bytes_before_allocations
         << ",\n    \"execution_free_bytes_after_allocations\": "
         << evidence.construction_snapshot
                .execution_free_bytes_after_allocations
         << ",\n    \"execution_aggregate_memory_gate_passed\": ";
  write_boolean(
      output,
      evidence.construction_snapshot.execution_aggregate_memory_gate_passed);
  output << ",\n    \"future_bytes_after_execution_create\": "
         << lifetime_probe::
                kSm87MacroFeedV4EngineLifetimeFutureAfterExecutionBytes
         << ",\n    \"future_bytes_after_rope_create\": "
         << lifetime_probe::kSm87MacroFeedV4EngineLifetimeFutureAfterRopeBytes
         << ",\n    \"future_bytes_after_complete_aot_create\": "
         << lifetime_probe::
                kSm87MacroFeedV4EngineLifetimeFutureAfterCompleteAotBytes
         << ",\n    \"exact\": ";
  write_boolean(output, evidence.reserve_chain_exact);
  output << "\n  },\n  \"owner_allocation_device_lifetime_chain\": {\n"
         << "    \"lifetime_root_identity\": "
         << evidence.construction_snapshot.lifetime_root_identity
         << ",\n    \"startup_package_identity\": "
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
         << ",\n    \"startup_complete_gdn_source_catalog_identity\": "
         << evidence.construction_snapshot
                .startup_complete_gdn_source_catalog_identity
         << ",\n    \"startup_mlp_source_catalog_identity\": "
         << evidence.construction_snapshot.startup_mlp_source_catalog_identity
         << ",\n    \"transient_allocation_identity\": "
         << evidence.construction_snapshot.transient_allocation_identity
         << ",\n    \"recurrent_allocation_identity\": "
         << evidence.construction_snapshot.recurrent_allocation_identity
         << ",\n    \"execution_events_owner_identity\": "
         << evidence.construction_snapshot.execution_events_owner_identity
         << ",\n    \"transient_bytes\": "
         << evidence.construction_snapshot.transient_bytes
         << ",\n    \"recurrent_bytes\": "
         << evidence.construction_snapshot.recurrent_bytes
         << ",\n    \"owned_bytes\": "
         << evidence.construction_snapshot.owned_bytes
         << ",\n    \"lifetime_chain_sealed\": ";
  write_boolean(output, evidence.construction_snapshot.lifetime_chain_sealed);
  output << ",\n    \"exact\": ";
  write_boolean(output,
                evidence.owner_allocation_device_lifetime_chain_exact);
  output << "\n  },\n  \"macrofeed_v4_construction_snapshot\": {\n"
         << "    \"synthetic_t1_gdn_layer0_source\": ";
  write_boolean(
      output,
      evidence.construction_snapshot.synthetic_t1_gdn_layer0_source);
  output << ",\n    \"complete_gdn_layer0_bound\": ";
  write_boolean(output,
                evidence.construction_snapshot.complete_gdn_layer0_bound);
  output << ",\n    \"whole_model_executor_bound\": ";
  write_boolean(output,
                evidence.construction_snapshot.whole_model_executor_bound);
  output << ",\n    \"selector_bound\": ";
  write_boolean(output, evidence.construction_snapshot.selector_bound);
  output << ",\n    \"api_route_bound\": ";
  write_boolean(output, evidence.construction_snapshot.api_route_bound);
  output << ",\n    \"default_off\": ";
  write_boolean(output, evidence.construction_snapshot.default_off);
  output << ",\n    \"production_dispatch_eligible\": ";
  write_boolean(
      output, evidence.construction_snapshot.production_dispatch_eligible);
  output << ",\n    \"valid\": ";
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
  static_assert(
      lifetime_probe::
          kSm87MacroFeedV4EngineLifetimeExpectedCompleteGdnBindings == 48U);
  static_assert(
      lifetime_probe::
          kSm87MacroFeedV4EngineLifetimeExpectedMlpPairBindings == 64U);
  static_assert(
      lifetime_probe::
          kSm87MacroFeedV4EngineLifetimeExpectedFullAttentionBindings == 16U);
  static_assert(runtime::kLayerMajorPrefillPromptWideP40RequestCapacityTokens ==
                40'001U);
  static_assert(runtime::kLayerMajorPrefillPromptWideP40RequestArenaBytes ==
                8'640'542'976ULL);
  static_assert(
      lifetime_probe::kSm87MacroFeedV4EngineLifetimeExpectedOwnedBytes ==
      3'220'701'184ULL);
  static_assert(
      lifetime_probe::kSm87MacroFeedV4EngineLifetimeExpectedRopeBytes ==
      67'108'864ULL);
  static_assert(
      lifetime_probe::kSm87MacroFeedV4EngineLifetimeExpectedAnchoredBytes ==
      3'287'810'048ULL);
  static_assert(
      lifetime_probe::
          kSm87MacroFeedV4EngineLifetimeFutureAfterCompleteAotBytes ==
      11'928'353'024ULL);
  static_assert(
      kDestroyRecoveryToleranceBytes <
      lifetime_probe::kSm87MacroFeedV4EngineLifetimeExpectedRopeBytes);

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
  if (!evidence.model_directory.empty()) {
    capture_checkpoint_metadata(evidence);
  }

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
      evidence.checkpoint_metadata_exact &&
      evidence.device_query_cuda_error == 0 &&
      evidence.initial_mem_info_cuda_error == 0) {
    evidence.engine_create_attempted = true;
    const lifetime_probe::
        Sm87MacroFeedV4EngineLifetimeConstructionSnapshotHook requested_hook{
            capture_construction_snapshot, &evidence};
    const auto previous_hook = lifetime_probe::
        exchange_sm87_macrofeed_v4_engine_lifetime_construction_snapshot_hook(
            requested_hook);
    // Metadata and the Engine must consume the same canonical checkpoint
    // path.  Do not re-resolve the caller spelling after the pinned metadata
    // gate, because an owner-local symlink substitution between those two
    // operations would otherwise weaken the evidence snapshot.
    created = runtime::create_reference_engine(
        evidence.canonical_model_directory, make_engine_options());
    const auto installed_hook = lifetime_probe::
        exchange_sm87_macrofeed_v4_engine_lifetime_construction_snapshot_hook(
            previous_hook);
    evidence.construction_snapshot_hook_exact =
        !previous_hook && installed_hook.callback == requested_hook.callback &&
        installed_hook.context == requested_hook.context;
    evidence.construction_snapshot_valid =
        evidence.construction_snapshot_count == 1U &&
        evidence.construction_snapshot.valid();
    evidence.normal_catalogs_exact =
        evidence.construction_snapshot_valid &&
        evidence.construction_snapshot.normal_factory_branch &&
        !evidence.construction_snapshot.synthetic_t1_gdn_layer0_source &&
        evidence.construction_snapshot
                .startup_complete_gdn_source_catalog_identity != 0U &&
        evidence.construction_snapshot.startup_mlp_source_catalog_identity !=
            0U &&
        evidence.construction_snapshot
                .execution_complete_gdn_catalog_identity != 0U &&
        evidence.construction_snapshot.execution_complete_gdn_binding_count ==
            lifetime_probe::
                kSm87MacroFeedV4EngineLifetimeExpectedCompleteGdnBindings &&
        evidence.construction_snapshot.execution_mlp_pair_catalog_identity !=
            0U &&
        evidence.construction_snapshot.execution_mlp_pair_binding_count ==
            lifetime_probe::
                kSm87MacroFeedV4EngineLifetimeExpectedMlpPairBindings &&
        evidence.construction_snapshot
                .retained_complete_gdn_catalog_fold_identity != 0U &&
        evidence.construction_snapshot
                .retained_mlp_pair_catalog_fold_identity != 0U &&
        evidence.construction_snapshot
                .startup_full_attention_source_catalog_identity != 0U &&
        evidence.construction_snapshot
                .execution_full_attention_catalog_identity != 0U &&
        evidence.construction_snapshot.execution_full_attention_binding_count ==
            lifetime_probe::
                kSm87MacroFeedV4EngineLifetimeExpectedFullAttentionBindings &&
        evidence.construction_snapshot
                .retained_full_attention_catalog_fold_identity != 0U &&
        evidence.construction_snapshot
                .full_attention_resource_bundle_identity != 0U;
    evidence.full_attention_ownership_exact =
        evidence.construction_snapshot_valid &&
        evidence.construction_snapshot.kv_allocation_identity != 0U &&
        evidence.construction_snapshot.kv_allocation_begin != 0U &&
        evidence.construction_snapshot.request_state_kv_allocation_identity ==
            evidence.construction_snapshot.kv_allocation_identity &&
        evidence.construction_snapshot.kv_allocation_bytes ==
            lifetime_probe::
                kSm87MacroFeedV4EngineLifetimeExpectedKvArenaBytes &&
        evidence.construction_snapshot.request_state_kv_allocation_bytes ==
            evidence.construction_snapshot.kv_allocation_bytes &&
        evidence.construction_snapshot.request_state_kv_physical_owner_bound &&
        evidence.construction_snapshot.engine_rope_owner_identity != 0U &&
        evidence.construction_snapshot.engine_rope_binding_identity != 0U &&
        evidence.construction_snapshot.engine_rope_allocation_begin != 0U &&
        evidence.construction_snapshot.engine_rope_device_ordinal ==
            evidence.device_ordinal &&
        evidence.construction_snapshot.engine_rope_positions ==
            lifetime_probe::
                kSm87MacroFeedV4EngineLifetimeExpectedRopePositions &&
        evidence.construction_snapshot.engine_rope_pairs ==
            lifetime_probe::kSm87MacroFeedV4EngineLifetimeExpectedRopePairs &&
        evidence.construction_snapshot.engine_rope_allocation_bytes ==
            lifetime_probe::kSm87MacroFeedV4EngineLifetimeExpectedRopeBytes &&
        evidence.construction_snapshot.owned_bytes ==
            lifetime_probe::kSm87MacroFeedV4EngineLifetimeExpectedOwnedBytes &&
        evidence.construction_snapshot.anchored_bytes ==
            lifetime_probe::
                kSm87MacroFeedV4EngineLifetimeExpectedAnchoredBytes;
    evidence.reserve_chain_exact =
        evidence.construction_snapshot_valid &&
        evidence.construction_snapshot.legacy_request_arena_bytes ==
            runtime::kLayerMajorPrefillPromptWideP40RequestArenaBytes &&
        evidence.construction_snapshot.minimum_free_bytes_after_legacy_create ==
            kRetainedFreeBytes &&
        evidence.construction_snapshot
                .execution_minimum_free_bytes_after_create ==
            kRetainedFreeBytes +
                runtime::kLayerMajorPrefillPromptWideP40RequestArenaBytes &&
        evidence.construction_snapshot.minimum_free_bytes_after_rope_create ==
            kRetainedFreeBytes +
                lifetime_probe::
                    kSm87MacroFeedV4EngineLifetimeFutureAfterRopeBytes &&
        evidence.construction_snapshot
                .minimum_free_bytes_after_complete_aot_create ==
            kRetainedFreeBytes +
                lifetime_probe::
                    kSm87MacroFeedV4EngineLifetimeFutureAfterCompleteAotBytes &&
        evidence.construction_snapshot
                .execution_required_device_allocation_bytes ==
            lifetime_probe::kSm87MacroFeedV4EngineLifetimeExpectedOwnedBytes &&
        evidence.construction_snapshot.execution_aggregate_memory_gate_passed &&
        evidence.construction_snapshot.execution_free_bytes_before_allocations >=
            evidence.construction_snapshot
                .execution_required_device_allocation_bytes &&
        evidence.construction_snapshot
                .execution_minimum_free_bytes_after_create <=
            evidence.construction_snapshot.execution_free_bytes_before_allocations -
                evidence.construction_snapshot
                    .execution_required_device_allocation_bytes &&
        evidence.construction_snapshot.execution_free_bytes_after_allocations >=
            evidence.construction_snapshot
                .execution_minimum_free_bytes_after_create;
    evidence.diagnostic = created.diagnostic;
    evidence.engine_created = static_cast<bool>(created);
    if (created) {
      evidence.engine_valid = static_cast<bool>(*created.value);
      evidence.load = created.value->load_stats();
      const auto& load = evidence.load;
      evidence.resident_shard_manifest_exact =
          resident_shard_manifest_exact(load.resident);
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
          load.target_aot_engine_rope_ready &&
          load.target_aot_engine_rope_bytes ==
              lifetime_probe::
                  kSm87MacroFeedV4EngineLifetimeExpectedRopeBytes &&
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
      evidence.owner_allocation_device_lifetime_chain_exact =
          evidence.snapshot_matches_complete_aot_load_stats &&
          evidence.normal_catalogs_exact &&
          evidence.full_attention_ownership_exact &&
          evidence.reserve_chain_exact &&
          evidence.construction_snapshot.lifetime_root_identity ==
              evidence.construction_snapshot
                  .compute_lifetime_root_identity() &&
          evidence.construction_snapshot.execution_startup_package_identity ==
              evidence.construction_snapshot.startup_package_identity &&
          evidence.construction_snapshot.transient_allocation_identity != 0U &&
          evidence.construction_snapshot.recurrent_allocation_identity != 0U &&
          evidence.construction_snapshot.transient_allocation_identity !=
              evidence.construction_snapshot.recurrent_allocation_identity &&
          evidence.construction_snapshot.execution_events_owner_identity !=
              0U &&
          evidence.construction_snapshot.transient_bytes ==
              lifetime_probe::
                  kSm87MacroFeedV4EngineLifetimeExpectedTransientBytes &&
          evidence.construction_snapshot.recurrent_bytes ==
              lifetime_probe::
                  kSm87MacroFeedV4EngineLifetimeMinimumOwnedArenaBytes &&
          evidence.construction_snapshot.owned_bytes ==
              lifetime_probe::kSm87MacroFeedV4EngineLifetimeExpectedOwnedBytes &&
          evidence.construction_snapshot.lifetime_chain_sealed;

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
      evidence.source_build_provenance_valid &&
      evidence.binary_provenance_valid &&
      evidence.build_configuration_exact &&
      evidence.device_query_cuda_error == 0 &&
      evidence.initial_mem_info_cuda_error == 0 &&
      evidence.engine_create_attempted && evidence.engine_created &&
      evidence.engine_valid && evidence.complete_aot_inventory_exact &&
      evidence.checkpoint_metadata_exact &&
      evidence.resident_shard_manifest_exact &&
      evidence.macrofeed_v3_engine_contract_exact &&
      evidence.construction_snapshot_hook_exact &&
      evidence.construction_snapshot_count == 1U &&
      evidence.construction_snapshot_valid &&
      evidence.normal_catalogs_exact &&
      evidence.full_attention_ownership_exact &&
      evidence.reserve_chain_exact &&
      evidence.snapshot_matches_complete_aot_load_stats &&
      evidence.owner_allocation_device_lifetime_chain_exact &&
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
