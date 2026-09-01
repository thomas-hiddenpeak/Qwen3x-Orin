#include "q3x/core/sha256.h"
#include "q3x/io/json.h"
#include "q3x/model/weight_manifest.h"
#include "q3x/runtime/reference_engine.h"
#include "sm87_aot_real_p40_arithmetic_witness_provenance.h"
#include "support/sm87_aot_system_v1_p40_projection_catalog_internal.h"
#include "support/sm87_aot_system_v1_real_p40_active_cell_witness_internal.h"

#include <cuda_runtime_api.h>

#include <elf.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

extern char** environ;

namespace core = q3x::core;
namespace json = q3x::io::json;
namespace runtime = q3x::runtime;
namespace active_cell = q3x::test::sm87_aot_real_p40_active_cell;
namespace lower_bound = q3x::test::sm87_aot_bmma_lower_bound;
namespace projection_catalog =
    q3x::test::sm87_aot_p40_projection_catalog;
namespace provenance =
    q3x::test::sm87_aot_real_p40_arithmetic_witness_provenance;
namespace witness_hook = q3x::runtime::reference_runner_detail;
namespace fs = std::filesystem;

namespace {

inline constexpr std::size_t kPromptTokens = 40'000U;
inline constexpr std::uint32_t kMaxNewTokens = 1U;
inline constexpr std::uint32_t kPrefillChunkSize = 512U;
inline constexpr std::uint64_t kRequestMaxArenaBytes =
    4ULL * 1024ULL * 1024ULL * 1024ULL;
inline constexpr std::uint64_t kMaximumCorpusBytes = 1ULL * 1024ULL * 1024ULL;
inline constexpr std::uint64_t kMaximumPreflightBytes =
    1ULL * 1024ULL * 1024ULL;
inline constexpr std::uint64_t kMaximumExecutableBytes =
    512ULL * 1024ULL * 1024ULL;
inline constexpr std::uint64_t kFrozenMaximumGpuFrequencyHz =
    1'300'500'000ULL;
inline constexpr std::string_view kDecisionUnit =
    "bmma-static-support-k16-parent-zero-fill-v2";
inline constexpr std::string_view kWorkPackageId = "P3";
inline constexpr std::string_view kArchitectureCandidateId =
    "AC-PREFILL-SM87-AOT-SYSTEM-v1";
inline constexpr std::string_view kExpectedEnabledQ3xBuildOptions =
    "Q3X_BUILD_SM87_AOT_SYSTEM_V1_ADMISSION,"
    "Q3X_BUILD_SM87_AOT_SYSTEM_V1_ARITHMETIC_WITNESS";
inline constexpr std::string_view kPreflightSchema =
    "q3x.sm87.aot-system-v1.real-p40-arithmetic-witness.preflight.v1";
inline constexpr std::string_view kGpuMaxFrequencyPath =
    "/sys/class/devfreq/17000000.gpu/max_freq";
inline constexpr std::string_view kGpuAvailableFrequenciesPath =
    "/sys/class/devfreq/17000000.gpu/available_frequencies";
inline constexpr std::string_view kCorpusFileSha256 =
    "8970ac50693f49d1b27d35a0610ecbe5072594330d69b301f4dab731789b6844";
inline constexpr std::string_view kPromptU32LeSha256 =
    "76cd23e2d60a9473af0ff24767b1b7ca614b36857697b420246731165156f78f";
inline constexpr std::string_view kModelRepository =
    "nvidia/Qwen3.6-27B-NVFP4";
inline constexpr std::string_view kModelRevision =
    "0893e1606ff3d5f97a441f405d5fc541a6bdf404";
inline constexpr std::array<std::string_view, 3U> kModelMetadataNames{{
    "config.json",
    "hf_quant_config.json",
    "model.safetensors.index.json",
}};
inline constexpr std::array<std::string_view, 3U> kModelMetadataSha256{{
    "c04a19ba293737ad7be4f6e96d6666cb7e479cbe19ecc0c289fad267135b0338",
    "fd7200cd8bca2a8a5d777061521abf83e2deb97ab6bc2f04e7a0a3d3f8ecd5c1",
    "7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2",
}};
inline constexpr std::string_view kRouteIdentity =
    "q3x.sm87.real-p40.active-cell-witness.route.v4;"
    "api=ReferenceEngine.generate_prompt_token_ids;"
    "runner=ordinary-legacy-enqueue_prefill_layer_segment;"
    "backend=sm87-weight-only;profile=legacy-c512;"
    "max-sequence-length=40000;prefill-chunk-size=512;"
    "max-new-tokens=1;logits=predicted-token-only;"
    "execution=legacy-c512-tiled;target-aot-assets=false;"
    "ambient-q3x-environment=empty;"
    "build=clean-release-exact-two-admissions;"
    "binary=self-fd-sha256-and-gnu-build-id;preflight=sha256-bound;"
    "device=cc8.7-sm16;maximum-gpu-frequency-hz=1300500000;"
    "mapping=bmma-static-support-k16-parent-zero-fill-v2;"
    "checkpoint=400-module-1392-tensor-payload-catalog-sha256;"
    "capture=source-local-five-role-canonical-u16-support-mask-chain-v2;"
    "prefix=[0,39999);tiles=79;terminal-scalar=[39999,40000)-free";
inline constexpr std::string_view kShardManifestDomain =
    "q3x.sm87.real-p40.loaded-pinned-shard-manifest.v1";
inline constexpr std::string_view kRouteDigestDomain =
    "q3x.sm87.real-p40.active-cell-witness.route-digest.v4";

struct PreflightEvidence final {
  fs::path requested_path;
  fs::path canonical_path;
  std::uint64_t file_bytes = 0U;
  core::Sha256Digest file_sha256{};
  std::uint64_t launcher_pid = 0U;
  std::uint64_t created_at_unix_ns = 0U;
  std::uint64_t freshness_limit_seconds = 0U;
  std::string hostname;
  std::string machine;
  std::string producer_path;
  std::string producer_sha256;
  std::string source_preflight_producer_path;
  std::string source_preflight_producer_sha256;
  std::string source_preflight_report_sha256;
  std::string probe_sha256;
  fs::path gpu_max_frequency_canonical_path;
  fs::path gpu_available_frequencies_canonical_path;
  std::uint64_t gpu_max_frequency_hz = 0U;
  std::uint64_t gpu_available_max_frequency_hz = 0U;
  bool schema_authenticated = false;
  bool decision_identity_authenticated = false;
  bool hard_stop_clear = false;
  bool fan_fields_sanitized = false;
  bool cache_drop_attempted = false;
  bool cache_drop_succeeded = false;
  bool cpu_affinity_recorded = false;
  bool nvpmodel_recorded = false;
  bool device_clocks_recorded = false;
  bool temperature_envelope_recorded = false;
  bool authenticated = false;
};

struct BinaryIdentity final {
  bool attempted = false;
  bool regular_file = false;
  bool stable_fd_identity = false;
  bool sha256_complete = false;
  bool gnu_build_id_present = false;
  bool authenticated = false;
  std::uint64_t file_bytes = 0U;
  std::uint64_t device = 0U;
  std::uint64_t inode = 0U;
  core::Sha256Digest sha256{};
  std::string gnu_build_id;
};

struct ModelMetadataFileEvidence final {
  std::string filename;
  fs::path canonical_path;
  std::uint64_t file_bytes = 0U;
  core::Sha256Digest sha256{};
  bool authenticated = false;
};

struct CorpusEvidence final {
  fs::path requested_path;
  fs::path canonical_path;
  std::uint64_t file_bytes = 0U;
  core::Sha256Digest file_sha256{};
  core::Sha256Digest prompt_u32le_sha256{};
  std::vector<std::uint32_t> prompt_token_ids;
  std::uint32_t minimum_token_id = 0U;
  std::uint32_t maximum_token_id = 0U;
  bool single_line_jsonl = false;
  bool file_sha256_authenticated = false;
  bool prompt_length_authenticated = false;
  bool prompt_token_range_authenticated = false;
  bool prompt_u32le_sha256_authenticated = false;
};

struct ProbeEvidence final {
  fs::path model_requested_path;
  fs::path model_canonical_path;
  PreflightEvidence preflight;
  fs::path evidence_requested_path;
  fs::path evidence_canonical_path;
  CorpusEvidence corpus;
  std::array<ModelMetadataFileEvidence, kModelMetadataNames.size()>
      model_metadata{};
  bool model_metadata_authenticated = false;
  bool projection_manifest_build_attempted = false;
  bool projection_manifest_authenticated = false;
  bool projection_catalog_build_attempted = false;
  bool projection_catalog_authenticated = false;
  projection_catalog::Catalog projection_tensors{};

  BinaryIdentity binary;
  bool build_provenance_authenticated = false;

  runtime::ReferenceEngineLoadStats load{};
  bool engine_create_attempted = false;
  bool engine_created = false;
  bool load_captured = false;
  bool legacy_c512_route_authenticated = false;
  bool target_aot_assets_absent = false;
  bool q3x_environment_checked = false;
  bool q3x_environment_clean = false;
  std::string ambient_q3x_variable_name;
  bool loaded_shards_match_pins = false;
  lower_bound::Sha256Digest loaded_shard_manifest_sha256{};
  bool device_query_attempted = false;
  int device_ordinal = -1;
  int device_query_cuda_error = 0;
  int device_properties_cuda_error = 0;
  std::string device_name;
  int device_compute_major = 0;
  int device_compute_minor = 0;
  int device_sm_count = 0;
  std::uint64_t device_total_memory_bytes = 0U;
  std::string device_uuid_hex;
  int cuda_driver_version = 0;
  int cuda_runtime_version = 0;
  fs::path gpu_max_frequency_canonical_path;
  fs::path gpu_available_frequencies_canonical_path;
  std::uint64_t gpu_max_frequency_hz = 0U;
  std::uint64_t gpu_available_max_frequency_hz = 0U;
  bool device_envelope_authenticated = false;
  lower_bound::Sha256Digest real_p40_route_sha256{};
  bool route_identity_authenticated = false;

  bool collector_prepare_attempted = false;
  bool collector_valid_at_prepare = false;
  bool hook_installed_exclusively = false;
  bool hook_restored = false;
  bool collector_early_reject_before_finalize = false;
  bool collector_finalized = false;
  active_cell::CollectorResult collector{};

  bool generation_attempted = false;
  bool generation_completed = false;
  bool generation_value_present = false;
  bool expected_runner_sentinel_failure = false;
  runtime::ReferenceEngineDiagnostic generation_diagnostic{};

  bool valid_mapping_reject = false;
  bool valid_mapping_inconclusive = false;
  int exit_code = 1;
  std::string diagnostic_stage;
  std::string diagnostic_message;
  std::uint64_t wall_milliseconds = 0U;
};

[[nodiscard]] bool digest_nonzero(
    const lower_bound::Sha256Digest& digest) noexcept {
  return std::any_of(digest.begin(), digest.end(),
                     [](const std::uint8_t byte) { return byte != 0U; });
}

[[nodiscard]] lower_bound::Sha256Digest lower_digest(
    const core::Sha256Digest& digest) noexcept {
  return digest.bytes;
}

[[nodiscard]] std::string digest_hex(
    const lower_bound::Sha256Digest& digest) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string result(digest.size() * 2U, '0');
  for (std::size_t index = 0U; index < digest.size(); ++index) {
    result[index * 2U] = kHex[digest[index] >> 4U];
    result[index * 2U + 1U] = kHex[digest[index] & 0x0fU];
  }
  return result;
}

[[nodiscard]] bool hash_u64(core::Sha256& hash,
                            const std::uint64_t value) noexcept {
  std::array<std::uint8_t, 8U> encoded{};
  for (std::size_t index = 0U; index < encoded.size(); ++index) {
    encoded[index] =
        static_cast<std::uint8_t>((value >> (index * 8U)) & 0xffU);
  }
  return hash.update(encoded.data(), encoded.size());
}

[[nodiscard]] bool hash_string(core::Sha256& hash,
                               const std::string_view value) noexcept {
  return hash_u64(hash, static_cast<std::uint64_t>(value.size())) &&
         hash.update(value.data(), value.size());
}

[[nodiscard]] bool hash_digest(
    core::Sha256& hash,
    const lower_bound::Sha256Digest& digest) noexcept {
  return hash.update(digest.data(), digest.size());
}

void set_failure(ProbeEvidence& evidence, const std::string_view stage,
                 const std::string_view message) {
  if (evidence.diagnostic_stage.empty()) {
    evidence.diagnostic_stage.assign(stage);
    evidence.diagnostic_message.assign(message);
  }
  evidence.valid_mapping_reject = false;
  evidence.valid_mapping_inconclusive = false;
  evidence.exit_code = 1;
}

void write_json_string(std::ostream& output, const std::string_view value) {
  output.put('"');
  for (const char raw : value) {
    const unsigned char byte = static_cast<unsigned char>(raw);
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
          output.put(raw);
        }
        break;
    }
  }
  output.put('"');
}

void write_boolean(std::ostream& output, const bool value) {
  output << (value ? "true" : "false");
}

[[nodiscard]] bool path_is_within(const fs::path& root,
                                  const fs::path& candidate) {
  const fs::path relative = candidate.lexically_relative(root);
  if (relative.empty() || relative.is_absolute()) {
    return false;
  }
  for (const fs::path& component : relative) {
    if (component == "..") {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool lowercase_hex_string(const std::string_view value,
                                        const std::size_t expected_size) {
  return value.size() == expected_size &&
         std::all_of(value.begin(), value.end(), [](const char byte) {
           return (byte >= '0' && byte <= '9') ||
                  (byte >= 'a' && byte <= 'f');
         });
}

[[nodiscard]] bool authenticate_build_provenance(ProbeEvidence& evidence,
                                                 std::string& error) {
  evidence.build_provenance_authenticated = false;
  constexpr std::string_view kEmptySha256 =
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
  const bool valid =
      lowercase_hex_string(provenance::kGitCommit, 40U) &&
      lowercase_hex_string(provenance::kGitTree, 40U) &&
      provenance::kGitClean &&
      std::string_view(provenance::kGitStatusSha256) == kEmptySha256 &&
      lowercase_hex_string(provenance::kBuildReceiptSha256, 64U) &&
      lowercase_hex_string(provenance::kCxxCompilerSha256, 64U) &&
      lowercase_hex_string(provenance::kCudaCompilerSha256, 64U) &&
      lowercase_hex_string(provenance::kLauncherSourceSha256, 64U) &&
      lowercase_hex_string(provenance::kPreflightSourceSha256, 64U) &&
      std::string_view(provenance::kBuildType) == "Release" &&
      provenance::kBuildTesting && provenance::kAotSystemAdmission &&
      provenance::kArithmeticWitnessAdmission &&
      std::string_view(provenance::kEnabledQ3xBuildOptions) ==
          kExpectedEnabledQ3xBuildOptions &&
      std::string_view(provenance::kEffectiveCudaArchitectures) == "87" &&
      !std::string_view(provenance::kCmakeVersion).empty() &&
      !std::string_view(provenance::kGenerator).empty() &&
      !std::string_view(provenance::kCxxCompilerId).empty() &&
      !std::string_view(provenance::kCxxCompilerVersion).empty() &&
      !std::string_view(provenance::kCudaCompilerId).empty() &&
      !std::string_view(provenance::kCudaCompilerVersion).empty() &&
      !std::string_view(provenance::kCudaToolkitVersion).empty();
  if (!valid) {
    error = "binary was not built from a clean Release tree with exactly the "
            "two required default-off witness admissions and SM87 codegen";
    return false;
  }
  evidence.build_provenance_authenticated = true;
  return true;
}

[[nodiscard]] bool same_file_identity(const struct stat& left,
                                      const struct stat& right) noexcept {
  return left.st_dev == right.st_dev && left.st_ino == right.st_ino &&
         left.st_mode == right.st_mode && left.st_size == right.st_size &&
         left.st_mtim.tv_sec == right.st_mtim.tv_sec &&
         left.st_mtim.tv_nsec == right.st_mtim.tv_nsec &&
         left.st_ctim.tv_sec == right.st_ctim.tv_sec &&
         left.st_ctim.tv_nsec == right.st_ctim.tv_nsec;
}

[[nodiscard]] std::size_t align_note_field(const std::size_t value,
                                           bool& valid) noexcept {
  if (value > std::numeric_limits<std::size_t>::max() - 3U) {
    valid = false;
    return 0U;
  }
  return (value + 3U) & ~std::size_t{3U};
}

[[nodiscard]] std::string byte_span_hex(const std::uint8_t* const bytes,
                                        const std::size_t size) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string result(size * 2U, '0');
  for (std::size_t index = 0U; index < size; ++index) {
    result[index * 2U] = kHex[bytes[index] >> 4U];
    result[index * 2U + 1U] = kHex[bytes[index] & 0x0fU];
  }
  return result;
}

[[nodiscard]] bool extract_gnu_build_id(
    const std::vector<std::uint8_t>& image, std::string& build_id,
    std::string& error) {
  build_id.clear();
  if (image.size() < sizeof(Elf64_Ehdr)) {
    error = "self executable is smaller than an ELF64 header";
    return false;
  }
  Elf64_Ehdr header{};
  std::memcpy(&header, image.data(), sizeof(header));
  if (std::memcmp(header.e_ident, ELFMAG, SELFMAG) != 0 ||
      header.e_ident[EI_CLASS] != ELFCLASS64 ||
      header.e_ident[EI_DATA] != ELFDATA2LSB ||
      header.e_machine != EM_AARCH64 ||
      header.e_phentsize != sizeof(Elf64_Phdr) || header.e_phnum == 0U ||
      header.e_phnum == PN_XNUM) {
    error = "self executable is not the expected AArch64 ELF64 image";
    return false;
  }
  const std::uint64_t table_bytes =
      static_cast<std::uint64_t>(header.e_phnum) * sizeof(Elf64_Phdr);
  if (header.e_phoff > image.size() || table_bytes > image.size() ||
      header.e_phoff > image.size() - table_bytes) {
    error = "self executable program-header table is out of range";
    return false;
  }

  for (std::size_t index = 0U; index < header.e_phnum; ++index) {
    Elf64_Phdr program{};
    const std::size_t offset = static_cast<std::size_t>(header.e_phoff) +
                               index * sizeof(Elf64_Phdr);
    std::memcpy(&program, image.data() + offset, sizeof(program));
    if (program.p_type != PT_NOTE) {
      continue;
    }
    if (program.p_offset > image.size() || program.p_filesz > image.size() ||
        program.p_offset > image.size() - program.p_filesz) {
      error = "self executable PT_NOTE segment is out of range";
      return false;
    }
    std::size_t cursor = static_cast<std::size_t>(program.p_offset);
    const std::size_t end = cursor + static_cast<std::size_t>(program.p_filesz);
    while (cursor < end) {
      if (end - cursor < sizeof(Elf64_Nhdr)) {
        error = "self executable contains a truncated ELF note";
        return false;
      }
      Elf64_Nhdr note{};
      std::memcpy(&note, image.data() + cursor, sizeof(note));
      cursor += sizeof(note);
      bool valid_alignment = true;
      const std::size_t name_bytes = align_note_field(note.n_namesz,
                                                       valid_alignment);
      const std::size_t descriptor_bytes = align_note_field(
          note.n_descsz, valid_alignment);
      if (!valid_alignment || name_bytes > end - cursor) {
        error = "self executable ELF note name is out of range";
        return false;
      }
      const std::size_t name_offset = cursor;
      cursor += name_bytes;
      if (descriptor_bytes > end - cursor || note.n_descsz == 0U) {
        error = "self executable ELF note descriptor is out of range";
        return false;
      }
      const std::size_t descriptor_offset = cursor;
      cursor += descriptor_bytes;
      const bool is_gnu_build_id =
          note.n_type == NT_GNU_BUILD_ID && note.n_namesz == 4U &&
          std::memcmp(image.data() + name_offset, "GNU\0", 4U) == 0;
      if (!is_gnu_build_id) {
        continue;
      }
      const std::string observed = byte_span_hex(
          image.data() + descriptor_offset, note.n_descsz);
      if (!build_id.empty() && build_id != observed) {
        error = "self executable contains conflicting GNU build IDs";
        return false;
      }
      build_id = observed;
    }
  }
  if (build_id.empty()) {
    error = "self executable has no GNU build ID";
    return false;
  }
  return true;
}

[[nodiscard]] bool authenticate_self_binary(ProbeEvidence& evidence,
                                            std::string& error) {
  BinaryIdentity& identity = evidence.binary;
  identity = {};
  identity.attempted = true;
  const int descriptor = ::open("/proc/self/exe", O_RDONLY | O_CLOEXEC);
  if (descriptor < 0) {
    error = "open(/proc/self/exe) failed: " +
            std::string(std::strerror(errno));
    return false;
  }
  struct stat before {};
  if (::fstat(descriptor, &before) != 0 || !S_ISREG(before.st_mode) ||
      before.st_size <= 0 ||
      static_cast<std::uint64_t>(before.st_size) > kMaximumExecutableBytes ||
      static_cast<std::uint64_t>(before.st_size) >
          static_cast<std::uint64_t>(
              std::numeric_limits<std::size_t>::max())) {
    error = "self executable fd is not a bounded non-empty regular file";
    (void)::close(descriptor);
    return false;
  }
  identity.regular_file = true;
  identity.file_bytes = static_cast<std::uint64_t>(before.st_size);
  identity.device = static_cast<std::uint64_t>(before.st_dev);
  identity.inode = static_cast<std::uint64_t>(before.st_ino);

  std::vector<std::uint8_t> image(
      static_cast<std::size_t>(before.st_size));
  std::size_t consumed = 0U;
  while (consumed < image.size()) {
    const ssize_t count =
        ::read(descriptor, image.data() + consumed, image.size() - consumed);
    if (count < 0 && errno == EINTR) {
      continue;
    }
    if (count <= 0) {
      error = "failed to read the complete self executable from one fd";
      (void)::close(descriptor);
      return false;
    }
    consumed += static_cast<std::size_t>(count);
  }
  char trailing = 0;
  ssize_t trailing_count = 0;
  do {
    trailing_count = ::read(descriptor, &trailing, 1U);
  } while (trailing_count < 0 && errno == EINTR);
  struct stat after {};
  const bool post_stat_ok = ::fstat(descriptor, &after) == 0;
  const int close_status = ::close(descriptor);
  identity.stable_fd_identity =
      trailing_count == 0 && post_stat_ok && same_file_identity(before, after);
  if (!identity.stable_fd_identity || close_status != 0) {
    error = "self executable changed while its single-fd identity was read";
    return false;
  }

  core::Sha256 hash;
  if (!hash.update(image.data(), image.size())) {
    error = "self executable SHA-256 update failed";
    return false;
  }
  identity.sha256 = hash.finalize();
  identity.sha256_complete =
      identity.sha256.hex() != std::string(64U, '0');
  if (!identity.sha256_complete ||
      !extract_gnu_build_id(image, identity.gnu_build_id, error)) {
    return false;
  }
  identity.gnu_build_id_present = true;
  identity.authenticated = true;
  return true;
}

[[nodiscard]] bool authenticate_clean_q3x_environment(
    ProbeEvidence& evidence, std::string& error) {
  evidence.q3x_environment_checked = true;
  evidence.q3x_environment_clean = false;
  evidence.ambient_q3x_variable_name.clear();
  if (::environ == nullptr) {
    error = "process environment is unavailable";
    return false;
  }
  for (char** cursor = ::environ; *cursor != nullptr; ++cursor) {
    const std::string_view entry(*cursor);
    const std::size_t separator = entry.find('=');
    const std::string_view name = entry.substr(0U, separator);
    if (name.rfind("Q3X_", 0U) == 0U) {
      evidence.ambient_q3x_variable_name.assign(
          name.substr(0U, std::min<std::size_t>(name.size(), 255U)));
      error = "ambient Q3X_* environment selector is forbidden: " +
              evidence.ambient_q3x_variable_name;
      return false;
    }
  }
  evidence.q3x_environment_clean = true;
  return true;
}

[[nodiscard]] bool resolve_evidence_path(
    const fs::path& requested, fs::path& canonical_output,
    std::string& error) {
  if (requested.empty() || requested.filename().empty() ||
      requested.filename() == "." || requested.filename() == "..") {
    error = "evidence path must name a new JSON file";
    return false;
  }

  std::error_code filesystem_error;
  fs::path absolute = fs::absolute(requested, filesystem_error);
  if (filesystem_error) {
    error = "failed to make evidence path absolute: " +
            filesystem_error.message();
    return false;
  }
  const fs::path parent = fs::canonical(absolute.parent_path(),
                                        filesystem_error);
  if (filesystem_error || parent.empty() ||
      !fs::is_directory(parent, filesystem_error)) {
    error = "evidence parent is not an existing canonical directory";
    return false;
  }
  canonical_output = parent / absolute.filename();

  const fs::file_status target_status =
      fs::symlink_status(canonical_output, filesystem_error);
  if (!filesystem_error && fs::exists(target_status)) {
    error = "evidence path already exists";
    return false;
  }
  filesystem_error.clear();

  fs::path q3x_work_root;
  for (fs::path cursor = parent; !cursor.empty();) {
    if (cursor.filename() == ".q3x-work") {
      const fs::path repository = cursor.parent_path();
      if (fs::exists(repository / ".git", filesystem_error) &&
          !filesystem_error &&
          fs::is_regular_file(repository / "docs" / "README.md",
                              filesystem_error) &&
          !filesystem_error) {
        q3x_work_root = cursor;
        break;
      }
      filesystem_error.clear();
    }
    const fs::path next = cursor.parent_path();
    if (next == cursor) {
      break;
    }
    cursor = next;
  }
  if (q3x_work_root.empty() ||
      !path_is_within(q3x_work_root, canonical_output)) {
    error = "evidence path must be inside a repository .q3x-work tree";
    return false;
  }
  return true;
}

class CreateOnlyEvidenceFile final {
 public:
  CreateOnlyEvidenceFile() = default;
  ~CreateOnlyEvidenceFile() {
    if (descriptor_ >= 0) {
      (void)::close(descriptor_);
    }
  }

  CreateOnlyEvidenceFile(const CreateOnlyEvidenceFile&) = delete;
  CreateOnlyEvidenceFile& operator=(const CreateOnlyEvidenceFile&) = delete;

  [[nodiscard]] bool create(const fs::path& path, std::string& error) {
    path_ = path;
    descriptor_ = ::open(path.c_str(),
                         O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (descriptor_ < 0) {
      error = "create-only evidence open failed: " +
              std::string(std::strerror(errno));
      return false;
    }
    return true;
  }

  [[nodiscard]] bool publish(const std::string_view payload,
                             std::string& error) {
    if (descriptor_ < 0) {
      error = "evidence descriptor is not open";
      return false;
    }
    bool ok = true;
    std::size_t written = 0U;
    while (written < payload.size()) {
      const ssize_t count =
          ::write(descriptor_, payload.data() + written,
                  payload.size() - written);
      if (count < 0 && errno == EINTR) {
        continue;
      }
      if (count <= 0) {
        error = "evidence write failed: " +
                std::string(std::strerror(errno));
        ok = false;
        break;
      }
      written += static_cast<std::size_t>(count);
    }
    if (ok && ::fsync(descriptor_) != 0) {
      error = "evidence fsync failed: " +
              std::string(std::strerror(errno));
      ok = false;
    }
    if (::close(descriptor_) != 0 && ok) {
      error = "evidence close failed: " +
              std::string(std::strerror(errno));
      ok = false;
    }
    descriptor_ = -1;
    if (!ok) {
      // A partial JSON file has no evidence authority and must not block a
      // clean create-only retry under the same predeclared path.
      (void)::unlink(path_.c_str());
    }
    return ok;
  }

 private:
  fs::path path_;
  int descriptor_ = -1;
};

[[nodiscard]] bool canonical_directory(const fs::path& requested,
                                       fs::path& canonical,
                                       std::string& error) {
  std::error_code filesystem_error;
  canonical = fs::canonical(requested, filesystem_error);
  if (filesystem_error || canonical.empty() ||
      !fs::is_directory(canonical, filesystem_error)) {
    error = "model directory is not an existing canonical directory";
    return false;
  }
  return true;
}

[[nodiscard]] bool read_small_canonical_file(const fs::path& requested,
                                             fs::path& canonical,
                                             std::string& text,
                                             std::string& error) {
  std::error_code filesystem_error;
  canonical = fs::canonical(requested, filesystem_error);
  if (filesystem_error || canonical.empty()) {
    error = "failed to canonicalize " + requested.string();
    return false;
  }
  const int descriptor =
      ::open(canonical.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (descriptor < 0) {
    error = "failed to open canonical " + requested.string() + ": " +
            std::string(std::strerror(errno));
    return false;
  }
  text.clear();
  std::array<char, 1024U> buffer{};
  bool ok = true;
  while (true) {
    const ssize_t count = ::read(descriptor, buffer.data(), buffer.size());
    if (count < 0 && errno == EINTR) {
      continue;
    }
    if (count < 0) {
      error = "failed to read canonical " + requested.string() + ": " +
              std::string(std::strerror(errno));
      ok = false;
      break;
    }
    if (count == 0) {
      break;
    }
    const std::size_t size = static_cast<std::size_t>(count);
    if (text.size() > 16U * 1024U - size) {
      error = "sysfs frequency file exceeds the bounded read limit";
      ok = false;
      break;
    }
    text.append(buffer.data(), size);
  }
  if (::close(descriptor) != 0 && ok) {
    error = "failed to close canonical " + requested.string();
    ok = false;
  }
  if (ok && text.empty()) {
    error = "canonical sysfs frequency file is empty";
    ok = false;
  }
  return ok;
}

[[nodiscard]] bool parse_frequency_values(
    const std::string_view text, std::vector<std::uint64_t>& values,
    std::string& error) {
  values.clear();
  std::size_t cursor = 0U;
  while (cursor < text.size()) {
    while (cursor < text.size() &&
           (text[cursor] == ' ' || text[cursor] == '\t' ||
            text[cursor] == '\n' || text[cursor] == '\r')) {
      ++cursor;
    }
    if (cursor == text.size()) {
      break;
    }
    std::uint64_t value = 0U;
    std::size_t digits = 0U;
    while (cursor < text.size() && text[cursor] >= '0' &&
           text[cursor] <= '9') {
      const std::uint64_t digit =
          static_cast<std::uint64_t>(text[cursor] - '0');
      if (value > (std::numeric_limits<std::uint64_t>::max() - digit) /
                      10U) {
        error = "sysfs frequency value overflows uint64";
        return false;
      }
      value = value * 10U + digit;
      ++cursor;
      ++digits;
    }
    if (digits == 0U || value == 0U ||
        (cursor < text.size() && text[cursor] != ' ' &&
         text[cursor] != '\t' && text[cursor] != '\n' &&
         text[cursor] != '\r')) {
      error = "sysfs frequency file is not whitespace-separated decimal Hz";
      return false;
    }
    values.push_back(value);
  }
  if (values.empty()) {
    error = "sysfs frequency file contains no positive frequency";
    return false;
  }
  return true;
}

[[nodiscard]] bool authenticate_device_envelope(ProbeEvidence& evidence,
                                                std::string& error) {
  evidence.device_query_attempted = true;
  cudaError_t status = cudaGetDevice(&evidence.device_ordinal);
  evidence.device_query_cuda_error = static_cast<int>(status);
  if (status != cudaSuccess || evidence.device_ordinal < 0) {
    error = "cudaGetDevice failed before witness-hook installation";
    return false;
  }
  cudaDeviceProp properties{};
  status = cudaGetDeviceProperties(&properties, evidence.device_ordinal);
  evidence.device_properties_cuda_error = static_cast<int>(status);
  if (status != cudaSuccess) {
    error = "cudaGetDeviceProperties failed before witness-hook installation";
    return false;
  }
  evidence.device_name = properties.name;
  evidence.device_compute_major = properties.major;
  evidence.device_compute_minor = properties.minor;
  evidence.device_sm_count = properties.multiProcessorCount;
  evidence.device_total_memory_bytes =
      static_cast<std::uint64_t>(properties.totalGlobalMem);
  evidence.device_uuid_hex = byte_span_hex(
      reinterpret_cast<const std::uint8_t*>(properties.uuid.bytes),
      sizeof(properties.uuid.bytes));
  status = cudaDriverGetVersion(&evidence.cuda_driver_version);
  if (status != cudaSuccess || evidence.cuda_driver_version <= 0) {
    error = "cudaDriverGetVersion failed before witness-hook installation";
    return false;
  }
  status = cudaRuntimeGetVersion(&evidence.cuda_runtime_version);
  if (status != cudaSuccess || evidence.cuda_runtime_version <= 0) {
    error = "cudaRuntimeGetVersion failed before witness-hook installation";
    return false;
  }
  if (properties.major != 8 || properties.minor != 7 ||
      properties.multiProcessorCount != 16 ||
      evidence.device_total_memory_bytes == 0U ||
      evidence.device_uuid_hex.size() != 32U) {
    error = "witness requires the exact CC 8.7, 16-SM device";
    return false;
  }

  std::string max_frequency_text;
  if (!read_small_canonical_file(
          fs::path(kGpuMaxFrequencyPath),
          evidence.gpu_max_frequency_canonical_path, max_frequency_text,
          error)) {
    return false;
  }
  std::vector<std::uint64_t> frequencies;
  if (!parse_frequency_values(max_frequency_text, frequencies, error) ||
      frequencies.size() != 1U) {
    if (error.empty()) {
      error = "max_freq must contain exactly one decimal frequency";
    }
    return false;
  }
  evidence.gpu_max_frequency_hz = frequencies.front();

  std::string available_frequency_text;
  if (!read_small_canonical_file(
          fs::path(kGpuAvailableFrequenciesPath),
          evidence.gpu_available_frequencies_canonical_path,
          available_frequency_text, error) ||
      !parse_frequency_values(available_frequency_text, frequencies, error)) {
    return false;
  }
  evidence.gpu_available_max_frequency_hz =
      *std::max_element(frequencies.begin(), frequencies.end());
  if (evidence.gpu_max_frequency_hz != kFrozenMaximumGpuFrequencyHz ||
      evidence.gpu_available_max_frequency_hz >
          kFrozenMaximumGpuFrequencyHz ||
      evidence.gpu_available_max_frequency_hz <
          evidence.gpu_max_frequency_hz) {
    error = "GPU max_freq must equal 1300500000 Hz and the maximum "
            "available frequency must be no higher or internally lower";
    return false;
  }
  evidence.device_envelope_authenticated = true;
  return true;
}

[[nodiscard]] bool read_regular_file(const fs::path& requested,
                                     fs::path& canonical,
                                     std::string& bytes,
                                     std::string& error) {
  std::error_code filesystem_error;
  canonical = fs::canonical(requested, filesystem_error);
  if (filesystem_error || canonical.empty()) {
    error = "failed to canonicalize corpus file";
    return false;
  }
  const int descriptor =
      ::open(canonical.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (descriptor < 0) {
    error = "failed to open canonical corpus file: " +
            std::string(std::strerror(errno));
    return false;
  }
  struct stat status {};
  if (::fstat(descriptor, &status) != 0 || !S_ISREG(status.st_mode) ||
      status.st_size <= 0 ||
      static_cast<std::uint64_t>(status.st_size) > kMaximumCorpusBytes ||
      static_cast<std::uint64_t>(status.st_size) >
          static_cast<std::uint64_t>(
              std::numeric_limits<std::size_t>::max())) {
    error = "corpus must be a non-empty regular file no larger than 1 MiB";
    (void)::close(descriptor);
    return false;
  }

  bytes.assign(static_cast<std::size_t>(status.st_size), '\0');
  std::size_t consumed = 0U;
  while (consumed < bytes.size()) {
    const ssize_t count =
        ::read(descriptor, bytes.data() + consumed, bytes.size() - consumed);
    if (count < 0 && errno == EINTR) {
      continue;
    }
    if (count <= 0) {
      error = "failed to read the complete corpus file";
      (void)::close(descriptor);
      return false;
    }
    consumed += static_cast<std::size_t>(count);
  }
  char trailing = 0;
  ssize_t trailing_count = 0;
  do {
    trailing_count = ::read(descriptor, &trailing, 1U);
  } while (trailing_count < 0 && errno == EINTR);
  struct stat final_status {};
  const bool final_identity_ok =
      ::fstat(descriptor, &final_status) == 0 &&
      same_file_identity(status, final_status);
  const int close_status = ::close(descriptor);
  if (trailing_count != 0 || !final_identity_ok || close_status != 0) {
    error = "corpus changed while it was being read";
    return false;
  }
  return true;
}

[[nodiscard]] bool hash_direct_regular_file(const fs::path& requested,
                                            fs::path& canonical,
                                            std::string& sha256,
                                            std::string& error) {
  std::error_code filesystem_error;
  const fs::file_status status =
      fs::symlink_status(requested, filesystem_error);
  if (filesystem_error || fs::is_symlink(status) ||
      !fs::is_regular_file(status)) {
    error = "hashed source is not a direct regular file: " +
            requested.string();
    return false;
  }
  std::string bytes;
  if (!read_regular_file(requested, canonical, bytes, error)) {
    return false;
  }
  core::Sha256 hash;
  if (!hash.update(bytes.data(), bytes.size())) {
    error = "source SHA-256 update failed: " + requested.string();
    return false;
  }
  sha256 = hash.finalize().hex();
  return lowercase_hex_string(sha256, 64U);
}

[[nodiscard]] fs::path repository_q3x_work_ancestor(
    const fs::path& path) {
  std::error_code filesystem_error;
  for (fs::path cursor = path.parent_path(); !cursor.empty();) {
    if (cursor.filename() == ".q3x-work") {
      const fs::path repository = cursor.parent_path();
      if (fs::exists(repository / ".git", filesystem_error) &&
          !filesystem_error &&
          fs::is_regular_file(repository / "docs" / "README.md",
                              filesystem_error) &&
          !filesystem_error) {
        return cursor;
      }
      filesystem_error.clear();
    }
    const fs::path next = cursor.parent_path();
    if (next == cursor) {
      break;
    }
    cursor = next;
  }
  return {};
}

[[nodiscard]] bool authenticate_model_metadata(ProbeEvidence& evidence,
                                               std::string& error) {
  evidence.model_metadata_authenticated = false;
  for (std::size_t index = 0U; index < kModelMetadataNames.size(); ++index) {
    ModelMetadataFileEvidence& file = evidence.model_metadata[index];
    file.filename = std::string(kModelMetadataNames[index]);
    const fs::path requested = evidence.model_canonical_path / file.filename;
    std::error_code filesystem_error;
    const fs::file_status requested_status =
        fs::symlink_status(requested, filesystem_error);
    if (filesystem_error || fs::is_symlink(requested_status) ||
        !fs::is_regular_file(requested_status)) {
      error = "model metadata is not a direct regular file: " + file.filename;
      return false;
    }
    std::string bytes;
    if (!read_regular_file(requested, file.canonical_path, bytes, error)) {
      error = "model metadata read failed for " + file.filename + ": " + error;
      return false;
    }
    if (file.canonical_path.parent_path() != evidence.model_canonical_path ||
        !path_is_within(evidence.model_canonical_path,
                        file.canonical_path)) {
      error = "model metadata escapes the canonical read-only model root: " +
              file.filename;
      return false;
    }
    core::Sha256 hash;
    if (!hash.update(bytes.data(), bytes.size())) {
      error = "model metadata SHA-256 update failed for " + file.filename;
      return false;
    }
    file.file_bytes = static_cast<std::uint64_t>(bytes.size());
    file.sha256 = hash.finalize();
    file.authenticated =
        file.sha256.hex() == kModelMetadataSha256[index];
    if (!file.authenticated) {
      error = "model metadata SHA-256 mismatch for " + file.filename;
      return false;
    }
  }
  evidence.model_metadata_authenticated = true;
  return true;
}

[[nodiscard]] bool authenticate_projection_tensor_catalog(
    ProbeEvidence& evidence, std::string& error) {
  evidence.projection_manifest_build_attempted = true;
  q3x::model::weights::ManifestResult manifest =
      q3x::model::weights::build_qwen36_27b_text_manifest(
          evidence.model_canonical_path);
  if (!manifest || !manifest.value.has_value()) {
    error = "pinned Qwen3.6 text manifest build failed";
    if (!manifest.diagnostics.empty()) {
      const q3x::model::weights::ManifestDiagnostic& diagnostic =
          manifest.diagnostics.front();
      error += ": " +
               std::string(q3x::model::weights::to_string(diagnostic.code)) +
               ":" + diagnostic.context + ":" + diagnostic.message;
    }
    return false;
  }
  evidence.projection_manifest_authenticated = true;
  evidence.projection_catalog_build_attempted = true;
  projection_catalog::Result built =
      projection_catalog::build_p40_projection_tensor_catalog(
          evidence.model_canonical_path, *manifest.value);
  if (!built || !built.value.has_value()) {
    error = "projection tensor catalog build failed: " +
            std::string(projection_catalog::to_string(built.error.code)) +
            ":" + built.error.context;
    if (!built.error.reader_error.ok()) {
      error += ":reader=" + std::string(
          q3x::test::sm87_aot_checkpoint_reader::to_string(
              built.error.reader_error.code));
    }
    return false;
  }
  projection_catalog::Catalog catalog = std::move(*built.value);
  if (catalog.modules.size() !=
          projection_catalog::kP40ProjectionModuleCount ||
      catalog.tensors.size() !=
          projection_catalog::kP40ProjectionTensorCount ||
      catalog.shard_receipts.size() != 3U ||
      catalog.catalog_sha256.hex() == std::string(64U, '0') ||
      !std::all_of(
          catalog.tensors.begin(), catalog.tensors.end(),
          [](const projection_catalog::TensorRecord& tensor) {
            const bool scalar_kind =
                tensor.kind == projection_catalog::TensorKind::kWeightScale2 ||
                tensor.kind == projection_catalog::TensorKind::kInputScale ||
                (tensor.kind == projection_catalog::TensorKind::kWeightScale &&
                 tensor.dtype == q3x::io::safetensors::DType::kF32);
            return !tensor.name.empty() && tensor.layer < 64U &&
                   tensor.role != projection_catalog::ProjectionRole::kInvalid &&
                   !tensor.shard.empty() && tensor.file.is_absolute() &&
                   tensor.file_begin < tensor.file_end &&
                   tensor.payload_sha256.hex() != std::string(64U, '0') &&
                   tensor.scalar_scale_bits_present == scalar_kind;
          })) {
    error = "projection tensor catalog failed final authentication";
    return false;
  }
  evidence.projection_tensors = std::move(catalog);
  evidence.projection_catalog_authenticated = true;
  return true;
}

[[nodiscard]] bool json_string_field_equals(const json::Value& root,
                                            const std::string_view key,
                                            const std::string_view expected) {
  const json::Value* const value = root.find(key);
  const std::string* const text = value == nullptr ? nullptr : value->as_string();
  return text != nullptr && *text == expected;
}

[[nodiscard]] bool json_string_field(const json::Value& root,
                                     const std::string_view key,
                                     std::string& output) {
  const json::Value* const value = root.find(key);
  const std::string* const text = value == nullptr ? nullptr : value->as_string();
  if (text == nullptr) {
    return false;
  }
  output = *text;
  return true;
}

[[nodiscard]] bool json_boolean_field(const json::Value& root,
                                      const std::string_view key,
                                      bool& output) {
  const json::Value* const value = root.find(key);
  const bool* const boolean = value == nullptr ? nullptr : value->as_bool();
  if (boolean == nullptr) {
    return false;
  }
  output = *boolean;
  return true;
}

[[nodiscard]] bool json_uint64_field(const json::Value& root,
                                     const std::string_view key,
                                     std::uint64_t& output) {
  const json::Value* const value = root.find(key);
  const json::Number* const number =
      value == nullptr ? nullptr : value->as_number();
  return number != nullptr && number->to_uint64(output);
}

[[nodiscard]] bool authenticate_preflight_record(
    PreflightEvidence& evidence, const fs::path& evidence_output,
    const std::string_view expected_probe_sha256, std::string& error) {
  std::string bytes;
  if (!read_regular_file(evidence.requested_path, evidence.canonical_path,
                         bytes, error)) {
    error = "preflight record read failed: " + error;
    return false;
  }
  if (bytes.size() > kMaximumPreflightBytes) {
    error = "preflight record exceeds 1 MiB";
    return false;
  }
  const fs::path preflight_q3x_work =
      repository_q3x_work_ancestor(evidence.canonical_path);
  const fs::path output_q3x_work =
      repository_q3x_work_ancestor(evidence_output);
  if (preflight_q3x_work.empty() || output_q3x_work.empty() ||
      preflight_q3x_work != output_q3x_work ||
      !path_is_within(preflight_q3x_work, evidence.canonical_path)) {
    error = "preflight and evidence must share one repository .q3x-work tree";
    return false;
  }

  core::Sha256 hash;
  if (!hash.update(bytes.data(), bytes.size())) {
    error = "preflight SHA-256 update failed";
    return false;
  }
  evidence.file_bytes = static_cast<std::uint64_t>(bytes.size());
  evidence.file_sha256 = hash.finalize();

  json::ParseOptions options;
  options.max_input_bytes = static_cast<std::size_t>(kMaximumPreflightBytes);
  options.max_nesting_depth = 32U;
  options.max_values = 10'000U;
  options.max_container_items = 10'000U;
  const json::ParseResult parsed = json::parse(bytes, options);
  if (!parsed || parsed.value->as_object() == nullptr) {
    error = "preflight JSON is not one bounded object";
    return false;
  }
  const json::Value& root = *parsed.value;
  evidence.schema_authenticated =
      json_string_field_equals(root, "schema", kPreflightSchema);
  evidence.decision_identity_authenticated =
      json_string_field_equals(root, "decision_unit", kDecisionUnit) &&
      json_string_field_equals(root, "work_package_id", kWorkPackageId) &&
      json_string_field_equals(root, "architecture_candidate_id",
                               kArchitectureCandidateId);
  std::string gpu_max_frequency_path;
  std::string gpu_available_frequencies_path;
  bool material_contention = true;
  bool safety_stop = true;
  bool temperature_stop = true;
  bool sync_completed = false;
  std::uint64_t unowned_handles = 1U;
  const bool fields_present =
      json_uint64_field(root, "launcher_pid", evidence.launcher_pid) &&
      json_uint64_field(root, "created_at_unix_ns",
                        evidence.created_at_unix_ns) &&
      json_uint64_field(root, "freshness_limit_seconds",
                        evidence.freshness_limit_seconds) &&
      json_string_field(root, "hostname", evidence.hostname) &&
      json_string_field(root, "machine", evidence.machine) &&
      json_string_field(root, "producer_path", evidence.producer_path) &&
      json_string_field(root, "producer_sha256",
                        evidence.producer_sha256) &&
      json_string_field(root, "source_preflight_producer_path",
                        evidence.source_preflight_producer_path) &&
      json_string_field(root, "source_preflight_producer_sha256",
                        evidence.source_preflight_producer_sha256) &&
      json_string_field(root, "source_preflight_report_sha256",
                        evidence.source_preflight_report_sha256) &&
      json_string_field(root, "probe_sha256", evidence.probe_sha256) &&
      json_string_field(root, "gpu_max_frequency_canonical_path",
                        gpu_max_frequency_path) &&
      json_uint64_field(root, "gpu_max_frequency_hz",
                        evidence.gpu_max_frequency_hz) &&
      json_string_field(root, "gpu_available_frequencies_canonical_path",
                        gpu_available_frequencies_path) &&
      json_uint64_field(root, "gpu_available_max_frequency_hz",
                        evidence.gpu_available_max_frequency_hz) &&
      json_boolean_field(root, "hard_stop_clear", evidence.hard_stop_clear) &&
      json_boolean_field(root, "fan_fields_sanitized",
                         evidence.fan_fields_sanitized) &&
      json_boolean_field(root, "material_contention_detected",
                         material_contention) &&
      json_boolean_field(root, "safety_stop", safety_stop) &&
      json_boolean_field(root, "temperature_operational_stop",
                         temperature_stop) &&
      json_boolean_field(root, "sync_completed", sync_completed) &&
      json_boolean_field(root, "cache_drop_attempted",
                         evidence.cache_drop_attempted) &&
      json_boolean_field(root, "cache_drop_succeeded",
                         evidence.cache_drop_succeeded) &&
      json_boolean_field(root, "cpu_affinity_recorded",
                         evidence.cpu_affinity_recorded) &&
      json_boolean_field(root, "nvpmodel_recorded",
                         evidence.nvpmodel_recorded) &&
      json_boolean_field(root, "device_clocks_recorded",
                         evidence.device_clocks_recorded) &&
      json_boolean_field(root, "temperature_envelope_recorded",
                         evidence.temperature_envelope_recorded) &&
      json_uint64_field(root, "unowned_gpu_handle_count", unowned_handles);
  evidence.gpu_max_frequency_canonical_path = gpu_max_frequency_path;
  evidence.gpu_available_frequencies_canonical_path =
      gpu_available_frequencies_path;

  struct utsname host_identity {};
  const bool uname_ok = ::uname(&host_identity) == 0;
  const auto now_duration =
      std::chrono::system_clock::now().time_since_epoch();
  const auto now_ns_signed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(now_duration)
          .count();
  const std::uint64_t now_ns =
      now_ns_signed < 0 ? 0U : static_cast<std::uint64_t>(now_ns_signed);
  constexpr std::uint64_t kNanosecondsPerSecond = 1'000'000'000ULL;
  const bool fresh =
      evidence.freshness_limit_seconds == 120U &&
      evidence.created_at_unix_ns != 0U &&
      evidence.created_at_unix_ns <= now_ns &&
      now_ns - evidence.created_at_unix_ns <=
          evidence.freshness_limit_seconds * kNanosecondsPerSecond;
  const fs::path repository_root = preflight_q3x_work.parent_path();
  std::error_code producer_error;
  const fs::path expected_producer = fs::canonical(
      repository_root / "tools" / "evaluation" /
          "run_sm87_aot_real_p40_arithmetic_witness.py",
      producer_error);
  if (producer_error) {
    error = "cannot canonicalize the compiled-in witness launcher source";
    return false;
  }
  const fs::path observed_producer =
      fs::path(evidence.producer_path).is_absolute()
          ? fs::path(evidence.producer_path)
          : repository_root / evidence.producer_path;
  const fs::path canonical_producer =
      fs::canonical(observed_producer, producer_error);
  if (producer_error) {
    error = "cannot canonicalize the preflight producer path";
    return false;
  }
  const fs::path expected_source_preflight = fs::canonical(
      repository_root / "tools" / "evaluation" / "orin_perf_preflight.py",
      producer_error);
  if (producer_error) {
    error = "cannot canonicalize the source preflight path";
    return false;
  }
  const fs::path observed_source_preflight =
      fs::path(evidence.source_preflight_producer_path).is_absolute()
          ? fs::path(evidence.source_preflight_producer_path)
          : repository_root / evidence.source_preflight_producer_path;
  const fs::path canonical_source_preflight =
      fs::canonical(observed_source_preflight, producer_error);
  if (producer_error) {
    error = "cannot canonicalize the reported source preflight path";
    return false;
  }
  fs::path hashed_producer_path;
  fs::path hashed_source_preflight_path;
  std::string actual_producer_sha256;
  std::string actual_source_preflight_sha256;
  if (!hash_direct_regular_file(expected_producer, hashed_producer_path,
                                actual_producer_sha256, error) ||
      !hash_direct_regular_file(expected_source_preflight,
                                hashed_source_preflight_path,
                                actual_source_preflight_sha256, error)) {
    return false;
  }
  const bool producer_authenticated =
      canonical_producer == expected_producer &&
      canonical_source_preflight == expected_source_preflight &&
      hashed_producer_path == expected_producer &&
      hashed_source_preflight_path == expected_source_preflight &&
      actual_producer_sha256 == provenance::kLauncherSourceSha256 &&
      actual_source_preflight_sha256 == provenance::kPreflightSourceSha256 &&
      evidence.producer_sha256 == provenance::kLauncherSourceSha256 &&
      evidence.source_preflight_producer_sha256 ==
          provenance::kPreflightSourceSha256 &&
      evidence.probe_sha256 == expected_probe_sha256 &&
      lowercase_hex_string(evidence.producer_sha256, 64U) &&
      lowercase_hex_string(evidence.source_preflight_producer_sha256, 64U) &&
      lowercase_hex_string(evidence.source_preflight_report_sha256, 64U) &&
      lowercase_hex_string(evidence.probe_sha256, 64U);
  const bool launcher_authenticated =
      evidence.launcher_pid == static_cast<std::uint64_t>(::getppid()) &&
      fresh && uname_ok && evidence.hostname == host_identity.nodename &&
      evidence.machine == host_identity.machine;
  std::error_code frequency_path_error;
  const fs::path expected_max_frequency_path =
      fs::canonical(fs::path(kGpuMaxFrequencyPath), frequency_path_error);
  if (frequency_path_error) {
    error = "cannot canonicalize the GPU max-frequency path";
    return false;
  }
  const fs::path expected_available_frequencies_path = fs::canonical(
      fs::path(kGpuAvailableFrequenciesPath), frequency_path_error);
  if (frequency_path_error) {
    error = "cannot canonicalize the GPU available-frequencies path";
    return false;
  }
  const bool frequency_record_authenticated =
      evidence.gpu_max_frequency_canonical_path ==
          expected_max_frequency_path &&
      evidence.gpu_available_frequencies_canonical_path ==
          expected_available_frequencies_path &&
      evidence.gpu_max_frequency_hz == kFrozenMaximumGpuFrequencyHz &&
      evidence.gpu_available_max_frequency_hz <=
          kFrozenMaximumGpuFrequencyHz &&
      evidence.gpu_available_max_frequency_hz >=
          evidence.gpu_max_frequency_hz;
  evidence.authenticated =
      evidence.schema_authenticated &&
      evidence.decision_identity_authenticated && fields_present &&
      producer_authenticated && launcher_authenticated &&
      frequency_record_authenticated &&
      evidence.hard_stop_clear && evidence.fan_fields_sanitized &&
      !material_contention && !safety_stop && !temperature_stop &&
      sync_completed && evidence.cache_drop_attempted &&
      unowned_handles == 0U &&
      evidence.file_sha256.hex() != std::string(64U, '0');
  if (!evidence.authenticated) {
    error = "preflight record did not authenticate the clean real-model lane";
    return false;
  }
  return true;
}

[[nodiscard]] bool load_corpus(CorpusEvidence& evidence,
                               std::string& error) {
  std::string bytes;
  if (!read_regular_file(evidence.requested_path, evidence.canonical_path,
                         bytes, error)) {
    return false;
  }
  evidence.file_bytes = static_cast<std::uint64_t>(bytes.size());
  evidence.single_line_jsonl =
      !bytes.empty() && bytes.back() == '\n' &&
      std::count(bytes.begin(), bytes.end(), '\n') == 1 &&
      std::find(bytes.begin(), bytes.end(), '\r') == bytes.end();
  if (!evidence.single_line_jsonl) {
    error = "corpus is not exactly one newline-terminated JSONL record";
    return false;
  }

  core::Sha256 file_hash;
  if (!file_hash.update(bytes.data(), bytes.size())) {
    error = "corpus SHA-256 update failed";
    return false;
  }
  evidence.file_sha256 = file_hash.finalize();
  evidence.file_sha256_authenticated =
      evidence.file_sha256.hex() == kCorpusFileSha256;
  if (!evidence.file_sha256_authenticated) {
    error = "corpus file SHA-256 differs from the pinned P40000 record";
    return false;
  }

  json::ParseOptions options;
  options.max_input_bytes = static_cast<std::size_t>(kMaximumCorpusBytes);
  options.max_nesting_depth = 16U;
  options.max_values = 50'100U;
  options.max_container_items = 50'100U;
  const json::ParseResult parsed = json::parse(bytes, options);
  if (!parsed) {
    error = "corpus JSON parse failed at offset " +
            std::to_string(parsed.error.offset) + ": " +
            std::string(parsed.error.message());
    return false;
  }
  const json::Value* const prompt = parsed.value->find("prompt");
  const json::Value::Array* const token_values =
      prompt == nullptr ? nullptr : prompt->as_array();
  if (token_values == nullptr || token_values->size() != kPromptTokens) {
    error = "corpus prompt is not the exact 40000-token array";
    return false;
  }
  evidence.prompt_length_authenticated = true;
  evidence.prompt_token_ids.clear();
  evidence.prompt_token_ids.reserve(token_values->size());
  evidence.minimum_token_id = std::numeric_limits<std::uint32_t>::max();
  evidence.maximum_token_id = 0U;
  core::Sha256 prompt_hash;
  for (const json::Value& item : *token_values) {
    const json::Number* const number = item.as_number();
    std::uint64_t raw = 0U;
    if (number == nullptr || !number->to_uint64(raw) ||
        raw >= runtime::kReferenceVocabularySize) {
      error = "corpus contains a token outside the pinned vocabulary";
      return false;
    }
    const std::uint32_t token = static_cast<std::uint32_t>(raw);
    evidence.minimum_token_id = std::min(evidence.minimum_token_id, token);
    evidence.maximum_token_id = std::max(evidence.maximum_token_id, token);
    evidence.prompt_token_ids.push_back(token);
    const std::array<std::uint8_t, 4U> encoded{{
        static_cast<std::uint8_t>(token & 0xffU),
        static_cast<std::uint8_t>((token >> 8U) & 0xffU),
        static_cast<std::uint8_t>((token >> 16U) & 0xffU),
        static_cast<std::uint8_t>((token >> 24U) & 0xffU),
    }};
    if (!prompt_hash.update(encoded.data(), encoded.size())) {
      error = "prompt u32le SHA-256 update failed";
      return false;
    }
  }
  evidence.prompt_token_range_authenticated = true;
  evidence.prompt_u32le_sha256 = prompt_hash.finalize();
  evidence.prompt_u32le_sha256_authenticated =
      evidence.prompt_u32le_sha256.hex() == kPromptU32LeSha256;
  if (!evidence.prompt_u32le_sha256_authenticated) {
    error = "prompt u32le SHA-256 differs from the pinned P40000 record";
    return false;
  }
  return true;
}

[[nodiscard]] bool target_aot_assets_are_absent(
    const runtime::ReferenceEngineLoadStats& load) noexcept {
  return !load.target_aot_projection_device_assets_requested &&
         !load.target_aot_projection_device_assets_enabled &&
         !load.target_aot_projection_device_assets_attached &&
         load.target_aot_projection_device_asset_artifacts == 0U &&
         load.target_aot_projection_device_asset_sources == 0U &&
         load.target_aot_projection_device_asset_bytes == 0U &&
         load.target_aot_projection_host_staging_peak_bytes == 0U &&
         load.target_aot_projection_source_d2h_bytes == 0U &&
         load.target_aot_projection_payload_h2d_bytes == 0U &&
         load.target_aot_projection_verification_d2h_bytes == 0U &&
         load.target_aot_projection_verified_payload_catalog_sha256.empty() &&
         load.target_aot_projection_owner_identity == 0U &&
         load.target_aot_projection_allocation_identity == 0U &&
         load.target_aot_projection_device_ordinal == -1;
}

[[nodiscard]] bool authenticate_loaded_shards(
    const runtime::ResidentLoadStats& observed,
    lower_bound::Sha256Digest& manifest_sha256, std::string& error) {
  const auto& expected = runtime::pinned_qwen36_27b_shards();
  if (expected.empty() || observed.shards.size() != expected.size()) {
    error = "resident load did not report exactly the pinned shard count";
    return false;
  }
  core::Sha256 hash;
  if (!hash_string(hash, kShardManifestDomain) ||
      !hash_u64(hash, static_cast<std::uint64_t>(expected.size()))) {
    error = "loaded-shard manifest SHA-256 initialization failed";
    return false;
  }
  std::uint64_t expected_total = 0U;
  for (std::size_t index = 0U; index < expected.size(); ++index) {
    const runtime::ShardIdentity& pin = expected[index];
    const runtime::ShardLoadStats& load = observed.shards[index];
    if (load.filename != pin.filename || load.bytes_read != pin.file_size ||
        load.sha256 != pin.sha256) {
      error = "resident shard identity differs from pinned_qwen36_27b_shards "
              "at index " +
              std::to_string(index);
      return false;
    }
    if (expected_total >
        std::numeric_limits<std::uint64_t>::max() - pin.file_size) {
      error = "pinned shard byte total overflowed";
      return false;
    }
    expected_total += pin.file_size;
    if (!hash_u64(hash, index) || !hash_string(hash, pin.filename) ||
        !hash_u64(hash, pin.file_size) || !hash_string(hash, pin.sha256)) {
      error = "loaded-shard manifest SHA-256 update failed";
      return false;
    }
  }
  if (observed.bytes_read != expected_total) {
    error = "resident aggregate bytes_read differs from pinned shard bytes";
    return false;
  }
  manifest_sha256 = lower_digest(hash.finalize());
  if (!digest_nonzero(manifest_sha256)) {
    error = "loaded-shard manifest SHA-256 finalized to zero";
    return false;
  }
  return true;
}

[[nodiscard]] bool authenticate_engine_route(
    const runtime::ReferenceEngine& engine,
    const runtime::ReferenceEngineLoadStats& load,
    const CorpusEvidence& corpus,
    const ProbeEvidence& device_evidence,
    const lower_bound::Sha256Digest& shard_manifest_sha256,
    lower_bound::Sha256Digest& route_sha256, std::string& error) {
  const lower_bound::MappingSpec& mapping =
      lower_bound::frozen_mapping_spec();
  const bool legacy =
      load.projection_backend == runtime::ProjectionBackend::kSm87WeightOnly &&
      load.request_memory_profile == runtime::RequestMemoryProfile::kLegacyC512 &&
      load.request_max_sequence_length == kPromptTokens &&
      engine.max_sequence_length() == kPromptTokens &&
      load.request_prefill_chunk_size == kPrefillChunkSize &&
      load.request_arena_bytes != 0U &&
      load.request_arena_bytes <= kRequestMaxArenaBytes &&
      !load.p40_packed_projection_assets_enabled &&
      !load.nvfp4_marlin_p40_parity_sidecars_enabled;
  if (!legacy) {
    error = "created engine is not the ordinary Legacy-C512/SM87 P40000 route";
    return false;
  }
  if (!target_aot_assets_are_absent(load)) {
    error = "target-AOT projection assets were requested, enabled, or retained";
    return false;
  }
  if (!device_evidence.build_provenance_authenticated ||
      !device_evidence.binary.authenticated ||
      !device_evidence.binary.regular_file ||
      !device_evidence.binary.stable_fd_identity ||
      !device_evidence.binary.sha256_complete ||
      !device_evidence.binary.gnu_build_id_present ||
      !device_evidence.preflight.authenticated ||
      !device_evidence.model_metadata_authenticated ||
      !device_evidence.projection_manifest_authenticated ||
      !device_evidence.projection_catalog_authenticated ||
      device_evidence.projection_tensors.modules.size() !=
          projection_catalog::kP40ProjectionModuleCount ||
      device_evidence.projection_tensors.tensors.size() !=
          projection_catalog::kP40ProjectionTensorCount ||
      device_evidence.projection_tensors.catalog_sha256.hex() ==
          std::string(64U, '0')) {
    error = "build, binary, preflight, model, or tensor catalog identity is not authenticated";
    return false;
  }
  if (!device_evidence.q3x_environment_checked ||
      !device_evidence.q3x_environment_clean ||
      !device_evidence.ambient_q3x_variable_name.empty()) {
    error = "ambient Q3X_* environment state is not authenticated empty";
    return false;
  }
  if (!device_evidence.device_envelope_authenticated ||
      device_evidence.device_compute_major != 8 ||
      device_evidence.device_compute_minor != 7 ||
      device_evidence.device_sm_count != 16 ||
      device_evidence.device_total_memory_bytes == 0U ||
      device_evidence.device_uuid_hex.size() != 32U ||
      device_evidence.cuda_driver_version <= 0 ||
      device_evidence.cuda_runtime_version <= 0 ||
      device_evidence.preflight.gpu_max_frequency_canonical_path !=
          device_evidence.gpu_max_frequency_canonical_path ||
      device_evidence.preflight.gpu_available_frequencies_canonical_path !=
          device_evidence.gpu_available_frequencies_canonical_path ||
      device_evidence.preflight.gpu_max_frequency_hz !=
          device_evidence.gpu_max_frequency_hz ||
      device_evidence.preflight.gpu_available_max_frequency_hz !=
          device_evidence.gpu_available_max_frequency_hz ||
      device_evidence.gpu_max_frequency_hz !=
          kFrozenMaximumGpuFrequencyHz ||
      device_evidence.gpu_available_max_frequency_hz >
          kFrozenMaximumGpuFrequencyHz) {
    error = "device envelope is not authenticated for the frozen mapping";
    return false;
  }
  if (mapping.identity.empty() || mapping.instruction.empty() ||
      mapping.support_active_parent_predicate.empty() ||
      mapping.exact_arithmetic_pass_floor.empty() || mapping.mma_m != 16U ||
      mapping.mma_n != 8U || mapping.bmma_k != 256U ||
      mapping.parent_k != 16U || mapping.zero_fill_factor != 16U ||
      mapping.physical_instructions_per_active_joint_k16_cell != 1U ||
      mapping.maximum_warp_instructions_per_sm_cycle != 4U ||
      mapping.sm_count != 16U ||
      mapping.clock_hz != kFrozenMaximumGpuFrequencyHz ||
      mapping.projection_budget_seconds != 5U ||
      mapping.p40_prompt_rows != kPromptTokens ||
      mapping.production_prefix_rows !=
          witness_hook::kAotArithmeticWitnessP40PrefixRows ||
      mapping.terminal_scalar_rows != 1U ||
      !mapping.partial_m16_tail_rows_charged_free ||
      !mapping.one_parent_per_instruction ||
      !mapping.cross_parent_packing_forbidden ||
      !mapping.support_active_parent_must_issue ||
      !mapping.result_aware_parent_elision_forbidden ||
      !mapping.cross_parent_common_subexpression_elimination_forbidden ||
      !mapping.additional_exactness_passes_charged_free) {
    error = "frozen static-schedule arithmetic mapping contract is invalid";
    return false;
  }

  core::Sha256 hash;
  const lower_bound::Sha256Digest corpus_sha =
      lower_digest(corpus.file_sha256);
  const lower_bound::Sha256Digest prompt_sha =
      lower_digest(corpus.prompt_u32le_sha256);
  const lower_bound::Sha256Digest binary_sha =
      lower_digest(device_evidence.binary.sha256);
  const lower_bound::Sha256Digest preflight_sha =
      lower_digest(device_evidence.preflight.file_sha256);
  const lower_bound::Sha256Digest projection_catalog_sha =
      lower_digest(device_evidence.projection_tensors.catalog_sha256);
  if (!hash_string(hash, kRouteDigestDomain) ||
      !hash_string(hash, kRouteIdentity) ||
      !hash_string(hash, kDecisionUnit) ||
      !hash_string(hash, kWorkPackageId) ||
      !hash_string(hash, kArchitectureCandidateId) ||
      !hash_string(hash, kModelRepository) ||
      !hash_string(hash, kModelRevision) ||
      !hash_string(hash, provenance::kGitCommit) ||
      !hash_string(hash, provenance::kGitTree) ||
      !hash_u64(hash, provenance::kGitClean ? 1U : 0U) ||
      !hash_string(hash, provenance::kGitStatusSha256) ||
      !hash_string(hash, provenance::kBuildReceiptSha256) ||
      !hash_string(hash, provenance::kCmakeVersion) ||
      !hash_string(hash, provenance::kGenerator) ||
      !hash_string(hash, provenance::kBuildType) ||
      !hash_u64(hash, provenance::kBuildTesting ? 1U : 0U) ||
      !hash_string(hash, provenance::kEffectiveCudaArchitectures) ||
      !hash_string(hash, provenance::kEnabledQ3xBuildOptions) ||
      !hash_string(hash, provenance::kCxxCompilerId) ||
      !hash_string(hash, provenance::kCxxCompilerVersion) ||
      !hash_string(hash, provenance::kCxxCompilerSha256) ||
      !hash_string(hash, provenance::kCudaCompilerId) ||
      !hash_string(hash, provenance::kCudaCompilerVersion) ||
      !hash_string(hash, provenance::kCudaCompilerSha256) ||
      !hash_string(hash, provenance::kCudaToolkitVersion) ||
      !hash_string(hash, provenance::kLauncherSourceSha256) ||
      !hash_string(hash, provenance::kPreflightSourceSha256) ||
      !hash_u64(hash, device_evidence.binary.file_bytes) ||
      !hash_u64(hash, device_evidence.binary.device) ||
      !hash_u64(hash, device_evidence.binary.inode) ||
      !hash_digest(hash, binary_sha) ||
      !hash_string(hash, device_evidence.binary.gnu_build_id) ||
      !hash_digest(hash, preflight_sha) ||
      !hash_u64(hash, device_evidence.q3x_environment_clean ? 1U : 0U) ||
      !hash_digest(hash, corpus_sha) || !hash_digest(hash, prompt_sha) ||
      !hash_digest(hash, shard_manifest_sha256) ||
      !hash_string(hash, projection_catalog::kP40ProjectionCatalogDomain) ||
      !hash_u64(hash, device_evidence.projection_tensors.modules.size()) ||
      !hash_u64(hash, device_evidence.projection_tensors.tensors.size()) ||
      !hash_digest(hash, projection_catalog_sha) ||
      !hash_string(hash, active_cell::kActivationCaptureChainDomain) ||
      !hash_string(hash, active_cell::kActivationCallDomain) ||
      !hash_string(hash, active_cell::kActivationSourceDtype) ||
      !hash_string(hash, active_cell::kActivationPayloadDtype) ||
      !hash_string(hash, mapping.identity) ||
      !hash_string(hash, mapping.instruction) ||
      !hash_string(hash, mapping.support_active_parent_predicate) ||
      !hash_string(hash, mapping.exact_arithmetic_pass_floor) ||
      !hash_u64(hash, mapping.mma_m) || !hash_u64(hash, mapping.mma_n) ||
      !hash_u64(hash, mapping.bmma_k) || !hash_u64(hash, mapping.parent_k) ||
      !hash_u64(hash, mapping.zero_fill_factor) ||
      !hash_u64(hash,
                mapping.physical_instructions_per_active_joint_k16_cell) ||
      !hash_u64(hash, mapping.maximum_warp_instructions_per_sm_cycle) ||
      !hash_u64(hash, mapping.sm_count) || !hash_u64(hash, mapping.clock_hz) ||
      !hash_u64(hash, mapping.projection_budget_seconds) ||
      !hash_u64(hash, mapping.p40_prompt_rows) ||
      !hash_u64(hash, mapping.production_prefix_rows) ||
      !hash_u64(hash, mapping.terminal_scalar_rows) ||
      !hash_u64(hash,
                mapping.partial_m16_tail_rows_charged_free ? 1U : 0U) ||
      !hash_u64(hash, mapping.one_parent_per_instruction ? 1U : 0U) ||
      !hash_u64(hash, mapping.cross_parent_packing_forbidden ? 1U : 0U) ||
      !hash_u64(hash, mapping.support_active_parent_must_issue ? 1U : 0U) ||
      !hash_u64(hash,
                mapping.result_aware_parent_elision_forbidden ? 1U : 0U) ||
      !hash_u64(
          hash,
          mapping.cross_parent_common_subexpression_elimination_forbidden
              ? 1U
              : 0U) ||
      !hash_u64(hash,
                mapping.additional_exactness_passes_charged_free ? 1U : 0U) ||
      !hash_u64(hash, load.request_arena_bytes) ||
      !hash_u64(hash, load.request_max_sequence_length) ||
      !hash_u64(hash, load.request_prefill_chunk_size) ||
      !hash_u64(hash,
                static_cast<std::uint64_t>(
                    device_evidence.device_compute_major)) ||
      !hash_u64(hash,
                static_cast<std::uint64_t>(
                    device_evidence.device_compute_minor)) ||
      !hash_u64(hash,
                static_cast<std::uint64_t>(
                    device_evidence.device_sm_count)) ||
      !hash_u64(hash, device_evidence.device_total_memory_bytes) ||
      !hash_string(hash, device_evidence.device_name) ||
      !hash_string(hash, device_evidence.device_uuid_hex) ||
      !hash_u64(hash,
                static_cast<std::uint64_t>(
                    device_evidence.cuda_driver_version)) ||
      !hash_u64(hash,
                static_cast<std::uint64_t>(
                    device_evidence.cuda_runtime_version)) ||
      !hash_u64(hash, device_evidence.gpu_max_frequency_hz) ||
      !hash_u64(hash,
                device_evidence.gpu_available_max_frequency_hz)) {
    error = "real-P40 route SHA-256 derivation failed";
    return false;
  }
  for (std::size_t index = 0U; index < kModelMetadataNames.size(); ++index) {
    const ModelMetadataFileEvidence& file =
        device_evidence.model_metadata[index];
    const lower_bound::Sha256Digest digest = lower_digest(file.sha256);
    if (!file.authenticated || file.filename != kModelMetadataNames[index] ||
        file.sha256.hex() != kModelMetadataSha256[index] ||
        !hash_u64(hash, index) || !hash_string(hash, file.filename) ||
        !hash_u64(hash, file.file_bytes) || !hash_digest(hash, digest)) {
      error = "model metadata identity could not enter the route digest";
      return false;
    }
  }
  route_sha256 = lower_digest(hash.finalize());
  if (!digest_nonzero(route_sha256)) {
    error = "real-P40 route SHA-256 finalized to zero";
    return false;
  }
  return true;
}

class ScopedAOperandHook final {
 public:
  ScopedAOperandHook() = default;
  ~ScopedAOperandHook() {
    if (active_) {
      (void)witness_hook::exchange_aot_arithmetic_witness_a_operand_hook(
          previous_);
      restored_ = true;
    }
  }

  ScopedAOperandHook(const ScopedAOperandHook&) = delete;
  ScopedAOperandHook& operator=(const ScopedAOperandHook&) = delete;

  [[nodiscard]] bool install(
      const witness_hook::AotArithmeticWitnessAOperandHook hook) noexcept {
    if (active_ || hook.callback == nullptr || hook.context == nullptr) {
      return false;
    }
    previous_ =
        witness_hook::exchange_aot_arithmetic_witness_a_operand_hook(hook);
    if (previous_.callback != nullptr || previous_.context != nullptr) {
      (void)witness_hook::exchange_aot_arithmetic_witness_a_operand_hook(
          previous_);
      previous_ = {};
      return false;
    }
    active_ = true;
    return true;
  }

  void restore() noexcept {
    if (active_) {
      (void)witness_hook::exchange_aot_arithmetic_witness_a_operand_hook(
          previous_);
      active_ = false;
      restored_ = true;
    }
  }

  [[nodiscard]] bool restored() const noexcept { return restored_; }

 private:
  witness_hook::AotArithmeticWitnessAOperandHook previous_{};
  bool active_ = false;
  bool restored_ = false;
};

[[nodiscard]] bool expected_sentinel_failure(
    const runtime::ReferenceGenerateResult& generation) noexcept {
  const runtime::ReferenceEngineDiagnostic& diagnostic =
      generation.diagnostic;
  const bool down_operation =
      diagnostic.operation ==
          "prefill_aot_arithmetic_witness_marlin_mlp_down" ||
      diagnostic.operation ==
          "prefill_aot_arithmetic_witness_generic_mlp_down";
  return !generation.ok() && !generation.value.has_value() &&
         diagnostic.code == runtime::ReferenceEngineError::kRunnerStepFailure &&
         diagnostic.stage == "generation_control" &&
         diagnostic.dependency_error ==
             static_cast<int>(runtime::ReferenceRunnerError::kCudaFailure) &&
         diagnostic.cuda_error == static_cast<int>(
                                      active_cell::RealP40ActiveCellWitnessCollector::
                                          early_reject_sentinel()) &&
         diagnostic.layer == runtime::kReferenceDecoderLayerCount - 1U &&
         down_operation;
}

[[nodiscard]] std::uint32_t expected_partition_count(
    const lower_bound::ProjectionRole role) noexcept {
  switch (role) {
    case lower_bound::ProjectionRole::kNvFp4GateUp:
      return 128U;
    case lower_bound::ProjectionRole::kNvFp4Down:
      return 64U;
    case lower_bound::ProjectionRole::kFp8GdnQkvZ:
      return 96U;
    case lower_bound::ProjectionRole::kFp8FullQkv:
      return 48U;
    case lower_bound::ProjectionRole::kFp8AttentionOutput:
      return 64U;
    case lower_bound::ProjectionRole::kInvalid:
    case lower_bound::ProjectionRole::kCount:
      return 0U;
  }
  return 0U;
}

[[nodiscard]] std::uint64_t expected_activation_calls_per_segment(
    const lower_bound::ProjectionRole role) noexcept {
  switch (role) {
    case lower_bound::ProjectionRole::kNvFp4GateUp:
    case lower_bound::ProjectionRole::kNvFp4Down:
    case lower_bound::ProjectionRole::kFp8AttentionOutput:
      return 64U;
    case lower_bound::ProjectionRole::kFp8GdnQkvZ:
      return 48U;
    case lower_bound::ProjectionRole::kFp8FullQkv:
      return 16U;
    case lower_bound::ProjectionRole::kInvalid:
    case lower_bound::ProjectionRole::kCount:
      return 0U;
  }
  return 0U;
}

[[nodiscard]] std::uint64_t expected_activation_k16_groups(
    const lower_bound::ProjectionRole role) noexcept {
  switch (role) {
    case lower_bound::ProjectionRole::kNvFp4Down:
      return 1'088U;
    case lower_bound::ProjectionRole::kFp8AttentionOutput:
      return 384U;
    case lower_bound::ProjectionRole::kNvFp4GateUp:
    case lower_bound::ProjectionRole::kFp8GdnQkvZ:
    case lower_bound::ProjectionRole::kFp8FullQkv:
      return 320U;
    case lower_bound::ProjectionRole::kInvalid:
    case lower_bound::ProjectionRole::kCount:
      return 0U;
  }
  return 0U;
}

[[nodiscard]] std::uint64_t expected_activation_mask_elements(
    const lower_bound::ProjectionRole role,
    const std::uint32_t completed_segments) noexcept {
  if (completed_segments == 0U ||
      completed_segments > active_cell::kRealP40SegmentCount) {
    return 0U;
  }
  const std::uint64_t full_segments =
      std::min<std::uint32_t>(completed_segments, 78U);
  const std::uint64_t m16_tiles =
      full_segments * 32U + (completed_segments == 79U ? 4U : 0U);
  return m16_tiles * expected_activation_calls_per_segment(role) *
         expected_activation_k16_groups(role);
}

[[nodiscard]] bool validate_strict_reject(const ProbeEvidence& evidence,
                                          std::string& error) {
  const active_cell::CollectorResult& collected = evidence.collector;
  if (!evidence.projection_manifest_authenticated ||
      !evidence.projection_catalog_authenticated ||
      !evidence.route_identity_authenticated ||
      !evidence.generation_attempted || evidence.generation_completed ||
      !evidence.expected_runner_sentinel_failure ||
      !evidence.collector_early_reject_before_finalize ||
      !collected.early_reject_requested) {
    error = "runner did not stop only through the collector's stable sentinel";
    return false;
  }
  if (!evidence.hook_installed_exclusively || !evidence.hook_restored) {
    error = "source-local witness hook was not installed and restored exclusively";
    return false;
  }
  if (collected.failure_code != active_cell::CollectorFailureCode::kNone ||
      collected.cuda_status != static_cast<int>(cudaSuccess) ||
      collected.exceptional_flag_value != 0U ||
      collected.exceptional_operand_detected ||
      collected.completed_segments == 0U ||
      collected.completed_segments > active_cell::kRealP40SegmentCount ||
      !collected.checkpoint_inventory_complete ||
      !collected.checkpoint_manifest_authenticated ||
      !digest_nonzero(collected.checkpoint_manifest_sha256)) {
    error = "collector did not finalize a complete authenticated processed subset";
    return false;
  }
  const std::uint32_t expected_rows =
      collected.completed_segments < active_cell::kRealP40SegmentCount
          ? collected.completed_segments * kPrefillChunkSize
          : witness_hook::kAotArithmeticWitnessP40PrefixRows;
  if (collected.covered_prompt_rows != expected_rows ||
      collected.covered_prompt_rows == 0U ||
      collected.covered_prompt_rows >
          witness_hook::kAotArithmeticWitnessP40PrefixRows) {
    error = "collector row coverage does not match its complete segment count";
    return false;
  }
  if (!collected.lower_bound_receipt.has_value()) {
    error = "collector did not return a lower-bound receipt";
    return false;
  }

  const lower_bound::DecisionReceipt& receipt =
      *collected.lower_bound_receipt;
  if (receipt.decision() != lower_bound::Decision::kReject ||
      !receipt.privately_issued() ||
      receipt.all_expected_real_p40_cells_complete() ||
      !receipt.all_processed_subsets_complete() ||
      !receipt.all_processed_evidence_authenticated() ||
      receipt.issues() != lower_bound::kLowerBoundIssuePartialP40Coverage ||
      receipt.proven_warp_instruction_lower_bound() <=
          receipt.five_second_absolute_instruction_capacity() ||
      receipt.optimistic_seconds_lower_bound() <=
          lower_bound::frozen_mapping_spec().projection_budget_seconds) {
    error = "lower-bound receipt is not the strict authenticated partial-P40 REJECT";
    return false;
  }

  std::uint64_t expected_total = 0U;
  std::uint64_t observed_total = 0U;
  std::uint64_t active_total = 0U;
  for (std::size_t index = 0U;
       index < lower_bound::kProjectionRoleCount; ++index) {
    const lower_bound::ProjectionRole role =
        static_cast<lower_bound::ProjectionRole>(index + 1U);
    const active_cell::RoleSummary& summary = collected.roles[index];
    const lower_bound::RoleLowerBound& issued = receipt.roles()[index];
    const std::uint64_t frozen_expected =
        lower_bound::frozen_expected_joint_k16_cells(role);
    const std::uint64_t frozen_observed =
        lower_bound::frozen_geometric_joint_k16_cells_for_prefix_rows(
            role, collected.covered_prompt_rows);
    if (summary.role != role || issued.role != role ||
        !summary.weight_inventory_complete ||
        summary.weight_partition_count != expected_partition_count(role) ||
        !summary.activation_inventory_authenticated ||
        !summary.joint_enumeration_authenticated ||
        !digest_nonzero(summary.activation_inventory_sha256) ||
        !digest_nonzero(summary.joint_enumeration_sha256) ||
        summary.activation_call_count !=
            expected_activation_calls_per_segment(role) *
                collected.completed_segments ||
        summary.activation_mask_element_count !=
            expected_activation_mask_elements(role,
                                              collected.completed_segments) ||
        summary.activation_payload_bytes !=
            expected_activation_mask_elements(role,
                                              collected.completed_segments) *
                sizeof(std::uint16_t) ||
        summary.observed_geometric_joint_k16_cells != frozen_observed ||
        summary.proven_active_joint_k16_cells > frozen_observed ||
        issued.expected_joint_k16_cells != frozen_expected ||
        issued.reported_expected_joint_k16_cells != frozen_expected ||
        issued.observed_geometric_joint_k16_cells !=
            summary.observed_geometric_joint_k16_cells ||
        issued.proven_active_joint_k16_cells !=
            summary.proven_active_joint_k16_cells ||
        issued.covered_prompt_rows != collected.covered_prompt_rows ||
        issued.proven_warp_instruction_lower_bound !=
            summary.proven_active_joint_k16_cells ||
        !issued.processed_subset_complete || issued.complete_real_p40_coverage ||
        !issued.evidence_authenticated) {
      error = "one role differs from the authenticated collector/issuer contract";
      return false;
    }
    if (expected_total >
            std::numeric_limits<std::uint64_t>::max() - frozen_expected ||
        observed_total > std::numeric_limits<std::uint64_t>::max() -
                             summary.observed_geometric_joint_k16_cells ||
        active_total > std::numeric_limits<std::uint64_t>::max() -
                           summary.proven_active_joint_k16_cells) {
      error = "receipt aggregate count overflow";
      return false;
    }
    expected_total += frozen_expected;
    observed_total += summary.observed_geometric_joint_k16_cells;
    active_total += summary.proven_active_joint_k16_cells;
  }
  if (receipt.expected_joint_k16_cells() != expected_total ||
      receipt.reported_expected_joint_k16_cells() != expected_total ||
      receipt.observed_geometric_joint_k16_cells() != observed_total ||
      receipt.proven_active_joint_k16_cells() != active_total ||
      receipt.proven_warp_instruction_lower_bound() != active_total) {
    error = "receipt aggregate counts differ from its five role receipts";
    return false;
  }
  return true;
}

[[nodiscard]] bool validate_strict_inconclusive(
    const ProbeEvidence& evidence, std::string& error) {
  const active_cell::CollectorResult& collected = evidence.collector;
  if (!evidence.projection_manifest_authenticated ||
      !evidence.projection_catalog_authenticated ||
      !evidence.route_identity_authenticated ||
      !evidence.generation_attempted || !evidence.generation_completed ||
      !evidence.generation_value_present ||
      evidence.expected_runner_sentinel_failure ||
      evidence.collector_early_reject_before_finalize ||
      collected.early_reject_requested ||
      evidence.generation_diagnostic.code !=
          runtime::ReferenceEngineError::kNone) {
    error = "runner did not complete normally after the mapping class survived";
    return false;
  }
  if (!evidence.hook_installed_exclusively || !evidence.hook_restored) {
    error = "source-local witness hook was not installed and restored exclusively";
    return false;
  }
  if (collected.failure_code != active_cell::CollectorFailureCode::kNone ||
      collected.cuda_status != static_cast<int>(cudaSuccess) ||
      collected.exceptional_flag_value != 0U ||
      collected.exceptional_operand_detected ||
      collected.completed_segments != active_cell::kRealP40SegmentCount ||
      collected.covered_prompt_rows !=
          witness_hook::kAotArithmeticWitnessP40PrefixRows ||
      !collected.checkpoint_inventory_complete ||
      !collected.checkpoint_manifest_authenticated ||
      !digest_nonzero(collected.checkpoint_manifest_sha256) ||
      !collected.lower_bound_receipt.has_value()) {
    error = "collector did not finalize the complete authenticated P39999 prefix";
    return false;
  }

  const lower_bound::DecisionReceipt& receipt =
      *collected.lower_bound_receipt;
  constexpr std::uint32_t kExpectedIssues =
      lower_bound::kLowerBoundIssuePartialP40Coverage |
      lower_bound::kLowerBoundIssueFiveSecondBudgetNotExceeded;
  if (receipt.decision() != lower_bound::Decision::kInconclusive ||
      !receipt.privately_issued() ||
      receipt.all_expected_real_p40_cells_complete() ||
      !receipt.all_processed_subsets_complete() ||
      !receipt.all_processed_evidence_authenticated() ||
      receipt.issues() != kExpectedIssues ||
      receipt.proven_warp_instruction_lower_bound() >
          receipt.five_second_absolute_instruction_capacity() ||
      receipt.optimistic_seconds_lower_bound() >
          lower_bound::frozen_mapping_spec().projection_budget_seconds) {
    error = "lower-bound receipt is not the strict authenticated valid INCONCLUSIVE";
    return false;
  }

  std::uint64_t expected_total = 0U;
  std::uint64_t observed_total = 0U;
  std::uint64_t active_total = 0U;
  for (std::size_t index = 0U;
       index < lower_bound::kProjectionRoleCount; ++index) {
    const lower_bound::ProjectionRole role =
        static_cast<lower_bound::ProjectionRole>(index + 1U);
    const active_cell::RoleSummary& summary = collected.roles[index];
    const lower_bound::RoleLowerBound& issued = receipt.roles()[index];
    const std::uint64_t frozen_expected =
        lower_bound::frozen_expected_joint_k16_cells(role);
    const std::uint64_t frozen_observed =
        lower_bound::frozen_geometric_joint_k16_cells_for_prefix_rows(
            role, witness_hook::kAotArithmeticWitnessP40PrefixRows);
    if (summary.role != role || issued.role != role ||
        !summary.weight_inventory_complete ||
        summary.weight_partition_count != expected_partition_count(role) ||
        !summary.activation_inventory_authenticated ||
        !summary.joint_enumeration_authenticated ||
        !digest_nonzero(summary.activation_inventory_sha256) ||
        !digest_nonzero(summary.joint_enumeration_sha256) ||
        summary.activation_call_count !=
            expected_activation_calls_per_segment(role) *
                collected.completed_segments ||
        summary.activation_mask_element_count !=
            expected_activation_mask_elements(role,
                                              collected.completed_segments) ||
        summary.activation_payload_bytes !=
            expected_activation_mask_elements(role,
                                              collected.completed_segments) *
                sizeof(std::uint16_t) ||
        summary.observed_geometric_joint_k16_cells != frozen_observed ||
        summary.proven_active_joint_k16_cells > frozen_observed ||
        issued.expected_joint_k16_cells != frozen_expected ||
        issued.reported_expected_joint_k16_cells != frozen_expected ||
        issued.observed_geometric_joint_k16_cells !=
            summary.observed_geometric_joint_k16_cells ||
        issued.proven_active_joint_k16_cells !=
            summary.proven_active_joint_k16_cells ||
        issued.covered_prompt_rows !=
            witness_hook::kAotArithmeticWitnessP40PrefixRows ||
        issued.proven_warp_instruction_lower_bound !=
            summary.proven_active_joint_k16_cells ||
        !issued.processed_subset_complete || issued.complete_real_p40_coverage ||
        !issued.evidence_authenticated) {
      error = "one role differs from the complete-prefix collector/issuer contract";
      return false;
    }
    if (expected_total >
            std::numeric_limits<std::uint64_t>::max() - frozen_expected ||
        observed_total > std::numeric_limits<std::uint64_t>::max() -
                             summary.observed_geometric_joint_k16_cells ||
        active_total > std::numeric_limits<std::uint64_t>::max() -
                           summary.proven_active_joint_k16_cells) {
      error = "valid-INCONCLUSIVE receipt aggregate count overflow";
      return false;
    }
    expected_total += frozen_expected;
    observed_total += summary.observed_geometric_joint_k16_cells;
    active_total += summary.proven_active_joint_k16_cells;
  }
  if (receipt.expected_joint_k16_cells() != expected_total ||
      receipt.reported_expected_joint_k16_cells() != expected_total ||
      receipt.observed_geometric_joint_k16_cells() != observed_total ||
      receipt.proven_active_joint_k16_cells() != active_total ||
      receipt.proven_warp_instruction_lower_bound() != active_total) {
    error = "valid-INCONCLUSIVE aggregates differ from the five role receipts";
    return false;
  }
  return true;
}

[[nodiscard]] bool run_decision_validator_self_test() {
  const auto make_evidence = [](const std::uint64_t requested_active) {
    ProbeEvidence evidence;
    evidence.projection_manifest_authenticated = true;
    evidence.projection_catalog_authenticated = true;
    evidence.route_identity_authenticated = true;
    evidence.generation_attempted = true;
    evidence.hook_installed_exclusively = true;
    evidence.hook_restored = true;
    evidence.collector_finalized = true;
    evidence.collector.completed_segments = active_cell::kRealP40SegmentCount;
    evidence.collector.covered_prompt_rows =
        witness_hook::kAotArithmeticWitnessP40PrefixRows;
    evidence.collector.checkpoint_inventory_complete = true;
    evidence.collector.checkpoint_manifest_authenticated = true;
    evidence.collector.checkpoint_manifest_sha256.fill(0x6dU);

    lower_bound::LowerBoundIssuer issuer;
    std::uint64_t remaining_active = requested_active;
    for (std::size_t index = 0U;
         index < lower_bound::kProjectionRoleCount; ++index) {
      const lower_bound::ProjectionRole role =
          static_cast<lower_bound::ProjectionRole>(index + 1U);
      const std::uint64_t observed =
          lower_bound::frozen_geometric_joint_k16_cells_for_prefix_rows(
              role, witness_hook::kAotArithmeticWitnessP40PrefixRows);
      const std::uint64_t active = std::min(observed, remaining_active);
      remaining_active -= active;

      lower_bound::RoleEvidence role_evidence;
      role_evidence.role = role;
      role_evidence.expected_joint_k16_cells =
          lower_bound::frozen_expected_joint_k16_cells(role);
      role_evidence.observed_joint_k16_cells = observed;
      role_evidence.active_joint_k16_cells = active;
      role_evidence.covered_prompt_rows =
          witness_hook::kAotArithmeticWitnessP40PrefixRows;
      role_evidence.real_p40_route_sha256.fill(0x4aU);
      role_evidence.activation_inventory_sha256.fill(
          static_cast<std::uint8_t>(index + 1U));
      role_evidence.checkpoint_manifest_sha256.fill(0x6dU);
      role_evidence.joint_enumeration_sha256.fill(
          static_cast<std::uint8_t>(index + 17U));
      role_evidence.real_p40_route_authenticated = true;
      role_evidence.activation_inventory_authenticated = true;
      role_evidence.checkpoint_manifest_authenticated = true;
      role_evidence.joint_enumeration_authenticated = true;
      role_evidence.covered_rows_are_contiguous_prefix = true;
      role_evidence.observed_subset_complete = true;
      if (!issuer.add(role_evidence)) {
        return ProbeEvidence{};
      }

      active_cell::RoleSummary& summary = evidence.collector.roles[index];
      summary.role = role;
      summary.weight_partition_count = expected_partition_count(role);
      summary.weight_inventory_complete = true;
      summary.observed_geometric_joint_k16_cells = observed;
      summary.proven_active_joint_k16_cells = active;
      summary.activation_inventory_sha256 =
          role_evidence.activation_inventory_sha256;
      summary.joint_enumeration_sha256 =
          role_evidence.joint_enumeration_sha256;
      summary.activation_call_count =
          expected_activation_calls_per_segment(role) *
          active_cell::kRealP40SegmentCount;
      summary.activation_mask_element_count =
          expected_activation_mask_elements(
              role, active_cell::kRealP40SegmentCount);
      summary.activation_payload_bytes =
          summary.activation_mask_element_count * sizeof(std::uint16_t);
      summary.activation_inventory_authenticated = true;
      summary.joint_enumeration_authenticated = true;
    }
    if (remaining_active != 0U) {
      return ProbeEvidence{};
    }
    evidence.collector.lower_bound_receipt = issuer.finalize();
    return evidence;
  };

  ProbeEvidence inconclusive = make_evidence(0U);
  inconclusive.generation_completed = true;
  inconclusive.generation_value_present = true;
  std::string error;
  if (!validate_strict_inconclusive(inconclusive, error) ||
      validate_strict_reject(inconclusive, error)) {
    return false;
  }

  const std::uint64_t reject_active =
      static_cast<std::uint64_t>(
          lower_bound::frozen_mapping_spec()
              .maximum_warp_instructions_per_sm_cycle) *
          lower_bound::frozen_mapping_spec().sm_count *
          lower_bound::frozen_mapping_spec().clock_hz *
          lower_bound::frozen_mapping_spec().projection_budget_seconds +
      1U;
  ProbeEvidence reject = make_evidence(reject_active);
  reject.expected_runner_sentinel_failure = true;
  reject.collector_early_reject_before_finalize = true;
  reject.collector.early_reject_requested = true;
  return validate_strict_reject(reject, error) &&
         !validate_strict_inconclusive(reject, error);
}

[[nodiscard]] std::string_view collector_failure_name(
    const active_cell::CollectorFailureCode code) noexcept {
  switch (code) {
    case active_cell::CollectorFailureCode::kNone:
      return "none";
    case active_cell::CollectorFailureCode::kHostAllocation:
      return "host_allocation";
    case active_cell::CollectorFailureCode::kCudaAllocation:
      return "cuda_allocation";
    case active_cell::CollectorFailureCode::kInvalidIdentity:
      return "invalid_identity";
    case active_cell::CollectorFailureCode::kProtocolOrder:
      return "protocol_order";
    case active_cell::CollectorFailureCode::kInvalidOperandView:
      return "invalid_operand_view";
    case active_cell::CollectorFailureCode::kWeightInventoryMismatch:
      return "weight_inventory_mismatch";
    case active_cell::CollectorFailureCode::kCudaMaskLaunch:
      return "cuda_mask_launch";
    case active_cell::CollectorFailureCode::kCudaCopy:
      return "cuda_copy";
    case active_cell::CollectorFailureCode::kCudaSynchronize:
      return "cuda_synchronize";
    case active_cell::CollectorFailureCode::kArithmeticOverflow:
      return "arithmetic_overflow";
    case active_cell::CollectorFailureCode::kDigestFailure:
      return "digest_failure";
    case active_cell::CollectorFailureCode::kLowerBoundIssuerFailure:
      return "lower_bound_issuer_failure";
    case active_cell::CollectorFailureCode::kIncompleteSegment:
      return "incomplete_segment";
    case active_cell::CollectorFailureCode::kAlreadyFinalized:
      return "already_finalized";
    case active_cell::CollectorFailureCode::kExceptionalOperand:
      return "exceptional_operand";
  }
  return "unknown";
}

[[nodiscard]] std::string bounded_failure_text(
    const std::array<char, 256U>& text) {
  const auto end = std::find(text.begin(), text.end(), '\0');
  return std::string(text.begin(), end);
}

void write_mapping(std::ostream& output) {
  const lower_bound::MappingSpec& mapping =
      lower_bound::frozen_mapping_spec();
  output << "{\"identity\":";
  write_json_string(output, mapping.identity);
  output << ",\"instruction\":";
  write_json_string(output, mapping.instruction);
  output << ",\"support_active_parent_predicate\":";
  write_json_string(output, mapping.support_active_parent_predicate);
  output << ",\"exact_arithmetic_pass_floor\":";
  write_json_string(output, mapping.exact_arithmetic_pass_floor);
  output << ",\"mma_m\":" << mapping.mma_m
         << ",\"mma_n\":" << mapping.mma_n
         << ",\"bmma_k\":" << mapping.bmma_k
         << ",\"parent_k\":" << mapping.parent_k
         << ",\"zero_fill_factor\":" << mapping.zero_fill_factor
         << ",\"physical_instructions_per_active_joint_k16_cell\":"
         << mapping.physical_instructions_per_active_joint_k16_cell
         << ",\"maximum_warp_instructions_per_sm_cycle\":"
         << mapping.maximum_warp_instructions_per_sm_cycle
         << ",\"sm_count\":" << mapping.sm_count
         << ",\"clock_hz\":" << mapping.clock_hz
         << ",\"projection_budget_seconds\":"
         << mapping.projection_budget_seconds
         << ",\"p40_prompt_rows\":" << mapping.p40_prompt_rows
         << ",\"production_prefix_rows\":"
         << mapping.production_prefix_rows
         << ",\"terminal_scalar_rows\":" << mapping.terminal_scalar_rows
         << ",\"partial_m16_tail_rows_charged_free\":";
  write_boolean(output, mapping.partial_m16_tail_rows_charged_free);
  output << ",\"one_parent_per_instruction\":";
  write_boolean(output, mapping.one_parent_per_instruction);
  output << ",\"cross_parent_packing_forbidden\":";
  write_boolean(output, mapping.cross_parent_packing_forbidden);
  output << ",\"support_active_parent_must_issue\":";
  write_boolean(output, mapping.support_active_parent_must_issue);
  output << ",\"result_aware_parent_elision_forbidden\":";
  write_boolean(output, mapping.result_aware_parent_elision_forbidden);
  output << ",\"cross_parent_common_subexpression_elimination_forbidden\":";
  write_boolean(
      output,
      mapping.cross_parent_common_subexpression_elimination_forbidden);
  output << ",\"additional_exactness_passes_charged_free\":";
  write_boolean(output, mapping.additional_exactness_passes_charged_free);
  output << '}';
}

[[nodiscard]] std::string reader_receipt_filename(
    const q3x::test::sm87_aot_checkpoint_reader::ReaderReceipt& receipt) {
  const std::size_t size =
      std::min(receipt.shard_filename_size, receipt.shard_filename.size());
  return std::string(receipt.shard_filename.data(), size);
}

void write_projection_tensor_catalog(std::ostream& output,
                                     const ProbeEvidence& evidence) {
  const projection_catalog::Catalog& catalog = evidence.projection_tensors;
  output << "{\"manifest_build_attempted\":";
  write_boolean(output, evidence.projection_manifest_build_attempted);
  output << ",\"manifest_authenticated\":";
  write_boolean(output, evidence.projection_manifest_authenticated);
  output << ",\"catalog_build_attempted\":";
  write_boolean(output, evidence.projection_catalog_build_attempted);
  output << ",\"catalog_authenticated\":";
  write_boolean(output, evidence.projection_catalog_authenticated);
  output << ",\"catalog_domain\":";
  write_json_string(output, projection_catalog::kP40ProjectionCatalogDomain);
  output << ",\"catalog_sha256\":";
  write_json_string(output, catalog.catalog_sha256.hex());
  output << ",\"module_count\":" << catalog.modules.size()
         << ",\"tensor_count\":" << catalog.tensors.size()
         << ",\"shard_receipts\":[";
  for (std::size_t index = 0U; index < catalog.shard_receipts.size();
       ++index) {
    if (index != 0U) {
      output.put(',');
    }
    const q3x::test::sm87_aot_checkpoint_reader::ReaderReceipt& receipt =
        catalog.shard_receipts[index];
    output << "{\"shard\":";
    write_json_string(output, reader_receipt_filename(receipt));
    output << ",\"expected_file_size\":" << receipt.expected_file_size
           << ",\"observed_file_size\":" << receipt.file_size
           << ",\"bytes_read\":" << receipt.bytes_read
           << ",\"selected_bytes\":" << receipt.selected_bytes
           << ",\"selected_range_count\":"
           << receipt.selected_range_count
           << ",\"sequential_read_calls\":"
           << receipt.sequential_read_calls
           << ",\"sink_delivery_calls\":" << receipt.sink_delivery_calls
           << ",\"expected_full_sha256\":";
    write_json_string(output, receipt.expected_full_sha256.hex());
    output << ",\"observed_full_sha256\":";
    write_json_string(output, receipt.full_sha256.hex());
    output << ",\"range_inventory_sha256\":";
    write_json_string(output, receipt.range_inventory_sha256.hex());
    output << ",\"receipt_sha256\":";
    write_json_string(output, receipt.receipt_sha256.hex());
    output << ",\"committed\":";
    write_boolean(output, receipt.committed);
    output.put('}');
  }
  output << "],\"modules\":[";
  for (std::size_t index = 0U; index < catalog.modules.size(); ++index) {
    if (index != 0U) {
      output.put(',');
    }
    const projection_catalog::ModuleRecord& module = catalog.modules[index];
    output << "{\"name\":";
    write_json_string(output, module.name);
    output << ",\"layer\":" << module.layer << ",\"role\":";
    write_json_string(output, projection_catalog::to_string(module.role));
    output << ",\"tensor_indices\":[";
    for (std::size_t tensor = 0U; tensor < module.tensor_indices.size();
         ++tensor) {
      if (tensor != 0U) {
        output.put(',');
      }
      output << module.tensor_indices[tensor];
    }
    output << "]}";
  }
  output << "],\"tensors\":[";
  for (std::size_t index = 0U; index < catalog.tensors.size(); ++index) {
    if (index != 0U) {
      output.put(',');
    }
    const projection_catalog::TensorRecord& tensor = catalog.tensors[index];
    output << "{\"index\":" << index << ",\"name\":";
    write_json_string(output, tensor.name);
    output << ",\"layer\":" << tensor.layer << ",\"role\":";
    write_json_string(output, projection_catalog::to_string(tensor.role));
    output << ",\"kind\":";
    write_json_string(output, projection_catalog::to_string(tensor.kind));
    output << ",\"dtype\":";
    write_json_string(output, q3x::io::safetensors::to_string(tensor.dtype));
    output << ",\"shape\":[";
    for (std::size_t dimension = 0U; dimension < tensor.shape.size();
         ++dimension) {
      if (dimension != 0U) {
        output.put(',');
      }
      output << tensor.shape[dimension];
    }
    output << "],\"shard\":";
    write_json_string(output, tensor.shard);
    output << ",\"file\":";
    write_json_string(output, tensor.file.string());
    output << ",\"file_begin\":" << tensor.file_begin
           << ",\"file_end\":" << tensor.file_end
           << ",\"payload_bytes\":"
           << tensor.file_end - tensor.file_begin
           << ",\"payload_sha256\":";
    write_json_string(output, tensor.payload_sha256.hex());
    output << ",\"scalar_scale_bits_present\":";
    write_boolean(output, tensor.scalar_scale_bits_present);
    output << ",\"scalar_scale_bits\":";
    if (tensor.scalar_scale_bits_present) {
      output << tensor.scalar_scale_bits;
    } else {
      output << "null";
    }
    output << ",\"scalar_scale_bits_hex\":";
    if (tensor.scalar_scale_bits_present) {
      std::ostringstream encoded;
      encoded << "0x" << std::hex << std::setw(8) << std::setfill('0')
              << tensor.scalar_scale_bits;
      write_json_string(output, encoded.str());
    } else {
      output << "null";
    }
    output.put('}');
  }
  output << "]}";
}

void write_receipt(std::ostream& output,
                   const active_cell::CollectorResult& collected) {
  if (!collected.lower_bound_receipt.has_value()) {
    output << "null";
    return;
  }
  const lower_bound::DecisionReceipt& receipt =
      *collected.lower_bound_receipt;
  output << "{\"decision\":";
  write_json_string(output, lower_bound::to_string(receipt.decision()));
  output << ",\"issues\":" << receipt.issues()
         << ",\"privately_issued\":";
  write_boolean(output, receipt.privately_issued());
  output << ",\"all_expected_real_p40_cells_complete\":";
  write_boolean(output, receipt.all_expected_real_p40_cells_complete());
  output << ",\"all_processed_subsets_complete\":";
  write_boolean(output, receipt.all_processed_subsets_complete());
  output << ",\"all_processed_evidence_authenticated\":";
  write_boolean(output, receipt.all_processed_evidence_authenticated());
  output << ",\"expected_joint_k16_cells\":"
         << receipt.expected_joint_k16_cells()
         << ",\"reported_expected_joint_k16_cells\":"
         << receipt.reported_expected_joint_k16_cells()
         << ",\"observed_geometric_joint_k16_cells\":"
         << receipt.observed_geometric_joint_k16_cells()
         << ",\"proven_active_joint_k16_cells\":"
         << receipt.proven_active_joint_k16_cells()
         << ",\"proven_warp_instruction_lower_bound\":"
         << receipt.proven_warp_instruction_lower_bound()
         << ",\"absolute_warp_instructions_per_second\":"
         << receipt.absolute_warp_instructions_per_second()
         << ",\"five_second_absolute_instruction_capacity\":"
         << receipt.five_second_absolute_instruction_capacity()
         << ",\"optimistic_seconds_lower_bound\":"
         << std::setprecision(17) << receipt.optimistic_seconds_lower_bound()
         << ",\"roles\":[";
  for (std::size_t index = 0U; index < receipt.roles().size(); ++index) {
    if (index != 0U) {
      output.put(',');
    }
    const lower_bound::RoleLowerBound& role = receipt.roles()[index];
    output << "{\"role\":";
    write_json_string(output, lower_bound::to_string(role.role));
    output << ",\"expected_joint_k16_cells\":"
           << role.expected_joint_k16_cells
           << ",\"reported_expected_joint_k16_cells\":"
           << role.reported_expected_joint_k16_cells
           << ",\"observed_geometric_joint_k16_cells\":"
           << role.observed_geometric_joint_k16_cells
           << ",\"proven_active_joint_k16_cells\":"
           << role.proven_active_joint_k16_cells
           << ",\"covered_prompt_rows\":" << role.covered_prompt_rows
           << ",\"proven_warp_instruction_lower_bound\":"
           << role.proven_warp_instruction_lower_bound
           << ",\"optimistic_seconds_lower_bound\":"
           << std::setprecision(17) << role.optimistic_seconds_lower_bound
           << ",\"processed_subset_complete\":";
    write_boolean(output, role.processed_subset_complete);
    output << ",\"complete_real_p40_coverage\":";
    write_boolean(output, role.complete_real_p40_coverage);
    output << ",\"evidence_authenticated\":";
    write_boolean(output, role.evidence_authenticated);
    output.put('}');
  }
  output << "]}";
}

[[nodiscard]] std::string serialize_evidence(const ProbeEvidence& evidence) {
  std::ostringstream output;
  output << "{\n  \"schema\":";
  write_json_string(
      output, "q3x.sm87.aot-system-v1.real-p40-arithmetic-witness.v1");
  output << ",\n  \"status\":";
  write_json_string(
      output,
      evidence.valid_mapping_reject
          ? "valid_mapping_class_reject"
          : (evidence.valid_mapping_inconclusive
                 ? "valid_mapping_class_inconclusive"
                 : "infrastructure_failure"));
  output << ",\n  \"exit_code\":" << evidence.exit_code
         << ",\n  \"decision_scope\":{\"decision_unit\":";
  write_json_string(output, kDecisionUnit);
  output << ",\"work_package_id\":";
  write_json_string(output, kWorkPackageId);
  output << ",\"architecture_candidate_id\":";
  write_json_string(output, kArchitectureCandidateId);
  output << ",\"valid_mapping_reject\":";
  write_boolean(output, evidence.valid_mapping_reject);
  output << ",\"valid_mapping_inconclusive\":";
  write_boolean(output, evidence.valid_mapping_inconclusive);
  output << "}"
         << ",\n  \"claim_boundary\":";
  if (evidence.valid_mapping_reject) {
    write_json_string(
        output,
        "The ordinary Legacy-C512 runner was observed only until the private "
        "collector issued a strict monotone arithmetic mapping-class REJECT. "
        "The terminal scalar and every incomplete M16 tail are charged as "
        "free. An interrupted generation is not a completed request and "
        "grants no output, timing, API, release, or production authority. The "
        "REJECT applies only to the frozen static-support K16-parent to "
        "zero-filled K256 schedule; it does not reject result-aware parent "
        "elision, cross-parent packing/CSE, other exact arithmetic/dataflow "
        "classes, AOT loading, or AOT as a whole. Activation evidence "
        "authenticates the canonical uint16 support-mask payload consumed by "
        "this predicate for every processed call; full raw BF16 payloads are "
        "not retained or authenticated and cannot support another predicate.");
  } else if (evidence.valid_mapping_inconclusive) {
    write_json_string(
        output,
        "The ordinary Legacy-C512 runner completed after the authenticated "
        "P39999 static-support witness stayed at or below the frozen five-second "
        "absolute issue capacity. This valid INCONCLUSIVE means only that the "
        "narrow K16-parent to zero-filled K256 mapping class survives this "
        "quick reject. It grants no performance, accuracy-comparison, release, "
        "or production-promotion authority. Activation evidence authenticates "
        "the canonical uint16 support-mask payload consumed by this predicate "
        "for every call; full raw BF16 payloads are not retained or "
        "authenticated and cannot support another predicate.");
  } else {
    write_json_string(
        output,
        "Infrastructure failure grants no arithmetic, output, timing, API, "
        "release, or production authority.");
  }
  output << ",\n  \"paths\":{\"model_requested\":";
  write_json_string(output, evidence.model_requested_path.string());
  output << ",\"model_canonical\":";
  write_json_string(output, evidence.model_canonical_path.string());
  output << ",\"corpus_requested\":";
  write_json_string(output, evidence.corpus.requested_path.string());
  output << ",\"corpus_canonical\":";
  write_json_string(output, evidence.corpus.canonical_path.string());
  output << ",\"preflight_requested\":";
  write_json_string(output, evidence.preflight.requested_path.string());
  output << ",\"preflight_canonical\":";
  write_json_string(output, evidence.preflight.canonical_path.string());
  output << ",\"evidence_requested\":";
  write_json_string(output, evidence.evidence_requested_path.string());
  output << ",\"evidence_canonical\":";
  write_json_string(output, evidence.evidence_canonical_path.string());
  output << "},\n  \"provenance\":{\"authenticated\":";
  write_boolean(output, evidence.build_provenance_authenticated &&
                            evidence.binary.authenticated &&
                            evidence.preflight.authenticated);
  output << ",\"git\":{\"commit\":";
  write_json_string(output, provenance::kGitCommit);
  output << ",\"tree\":";
  write_json_string(output, provenance::kGitTree);
  output << ",\"clean_at_build\":";
  write_boolean(output, provenance::kGitClean);
  output << ",\"status_porcelain_sha256\":";
  write_json_string(output, provenance::kGitStatusSha256);
  output << "},\"build\":{\"receipt_sha256\":";
  write_json_string(output, provenance::kBuildReceiptSha256);
  output << ",\"cmake_version\":";
  write_json_string(output, provenance::kCmakeVersion);
  output << ",\"generator\":";
  write_json_string(output, provenance::kGenerator);
  output << ",\"build_type\":";
  write_json_string(output, provenance::kBuildType);
  output << ",\"build_testing\":";
  write_boolean(output, provenance::kBuildTesting);
  output << ",\"effective_cuda_architectures\":";
  write_json_string(output, provenance::kEffectiveCudaArchitectures);
  output << ",\"enabled_q3x_build_options\":";
  write_json_string(output, provenance::kEnabledQ3xBuildOptions);
  output << ",\"cxx_compiler_id\":";
  write_json_string(output, provenance::kCxxCompilerId);
  output << ",\"cxx_compiler_version\":";
  write_json_string(output, provenance::kCxxCompilerVersion);
  output << ",\"cxx_compiler_sha256\":";
  write_json_string(output, provenance::kCxxCompilerSha256);
  output << ",\"cuda_compiler_id\":";
  write_json_string(output, provenance::kCudaCompilerId);
  output << ",\"cuda_compiler_version\":";
  write_json_string(output, provenance::kCudaCompilerVersion);
  output << ",\"cuda_compiler_sha256\":";
  write_json_string(output, provenance::kCudaCompilerSha256);
  output << ",\"cuda_toolkit_version\":";
  write_json_string(output, provenance::kCudaToolkitVersion);
  output << ",\"launcher_source_sha256\":";
  write_json_string(output, provenance::kLauncherSourceSha256);
  output << ",\"source_preflight_sha256\":";
  write_json_string(output, provenance::kPreflightSourceSha256);
  output << "},\"binary\":{\"opened_path\":\"/proc/self/exe\","
            "\"attempted\":";
  write_boolean(output, evidence.binary.attempted);
  output << ",\"regular_file\":";
  write_boolean(output, evidence.binary.regular_file);
  output << ",\"stable_single_fd_identity\":";
  write_boolean(output, evidence.binary.stable_fd_identity);
  output << ",\"file_bytes\":" << evidence.binary.file_bytes
         << ",\"device\":" << evidence.binary.device
         << ",\"inode\":" << evidence.binary.inode
         << ",\"sha256\":";
  write_json_string(output, evidence.binary.sha256.hex());
  output << ",\"sha256_complete\":";
  write_boolean(output, evidence.binary.sha256_complete);
  output << ",\"gnu_build_id\":";
  write_json_string(output, evidence.binary.gnu_build_id);
  output << ",\"gnu_build_id_present\":";
  write_boolean(output, evidence.binary.gnu_build_id_present);
  output << ",\"authenticated\":";
  write_boolean(output, evidence.binary.authenticated);
  output << "},\"preflight\":{\"schema\":";
  write_json_string(output, kPreflightSchema);
  output << ",\"file_bytes\":" << evidence.preflight.file_bytes
         << ",\"sha256\":";
  write_json_string(output, evidence.preflight.file_sha256.hex());
  output << ",\"launcher_pid\":" << evidence.preflight.launcher_pid
         << ",\"created_at_unix_ns\":"
         << evidence.preflight.created_at_unix_ns
         << ",\"freshness_limit_seconds\":"
         << evidence.preflight.freshness_limit_seconds
         << ",\"hostname\":";
  write_json_string(output, evidence.preflight.hostname);
  output << ",\"machine\":";
  write_json_string(output, evidence.preflight.machine);
  output << ",\"producer_path\":";
  write_json_string(output, evidence.preflight.producer_path);
  output << ",\"producer_sha256\":";
  write_json_string(output, evidence.preflight.producer_sha256);
  output << ",\"source_preflight_producer_path\":";
  write_json_string(output,
                    evidence.preflight.source_preflight_producer_path);
  output << ",\"source_preflight_producer_sha256\":";
  write_json_string(
      output, evidence.preflight.source_preflight_producer_sha256);
  output << ",\"source_preflight_report_sha256\":";
  write_json_string(output,
                    evidence.preflight.source_preflight_report_sha256);
  output << ",\"probe_sha256\":";
  write_json_string(output, evidence.preflight.probe_sha256);
  output << ",\"gpu_max_frequency_canonical_path\":";
  write_json_string(
      output, evidence.preflight.gpu_max_frequency_canonical_path.string());
  output << ",\"gpu_max_frequency_hz\":"
         << evidence.preflight.gpu_max_frequency_hz
         << ",\"gpu_available_frequencies_canonical_path\":";
  write_json_string(
      output,
      evidence.preflight.gpu_available_frequencies_canonical_path.string());
  output << ",\"gpu_available_max_frequency_hz\":"
         << evidence.preflight.gpu_available_max_frequency_hz;
  output << ",\"schema_authenticated\":";
  write_boolean(output, evidence.preflight.schema_authenticated);
  output << ",\"decision_identity_authenticated\":";
  write_boolean(output, evidence.preflight.decision_identity_authenticated);
  output << ",\"hard_stop_clear\":";
  write_boolean(output, evidence.preflight.hard_stop_clear);
  output << ",\"fan_fields_sanitized\":";
  write_boolean(output, evidence.preflight.fan_fields_sanitized);
  output << ",\"cache_drop_attempted\":";
  write_boolean(output, evidence.preflight.cache_drop_attempted);
  output << ",\"cache_drop_succeeded\":";
  write_boolean(output, evidence.preflight.cache_drop_succeeded);
  output << ",\"cpu_affinity_recorded\":";
  write_boolean(output, evidence.preflight.cpu_affinity_recorded);
  output << ",\"nvpmodel_recorded\":";
  write_boolean(output, evidence.preflight.nvpmodel_recorded);
  output << ",\"device_clocks_recorded\":";
  write_boolean(output, evidence.preflight.device_clocks_recorded);
  output << ",\"temperature_envelope_recorded\":";
  write_boolean(output, evidence.preflight.temperature_envelope_recorded);
  output << ",\"auxiliary_observations_complete\":";
  write_boolean(output, evidence.preflight.cpu_affinity_recorded &&
                            evidence.preflight.nvpmodel_recorded &&
                            evidence.preflight.device_clocks_recorded &&
                            evidence.preflight.temperature_envelope_recorded);
  output << ",\"authenticated\":";
  write_boolean(output, evidence.preflight.authenticated);
  output << "}},\n  \"real_path_boundary\":{\"api\":";
  write_json_string(output, "ReferenceEngine::generate_prompt_token_ids");
  output << ",\"execution_mode\":";
  write_json_string(output, "LegacyC512Tiled");
  output << ",\"runner_boundary\":";
  write_json_string(output,
                    "ReferenceRunner::enqueue_prefill_layer_segment");
  output << ",\"hook_scope\":";
  write_json_string(
      output, "legacy_prefix_layer_segment_final_scalar_excluded");
  output << ",\"route_identity\":";
  write_json_string(output, kRouteIdentity);
  output << ",\"source_local_default_off\":true},\n  \"ambient_q3x_environment\":{"
            "\"checked\":";
  write_boolean(output, evidence.q3x_environment_checked);
  output << ",\"clean\":";
  write_boolean(output, evidence.q3x_environment_clean);
  output << ",\"first_forbidden_variable_name\":";
  write_json_string(output, evidence.ambient_q3x_variable_name);
  output << "},\n  \"model_identity\":{\"repository\":";
  write_json_string(output, kModelRepository);
  output << ",\"revision\":";
  write_json_string(output, kModelRevision);
  output << ",\"metadata_authenticated\":";
  write_boolean(output, evidence.model_metadata_authenticated);
  output << ",\"metadata\":[";
  for (std::size_t index = 0U; index < evidence.model_metadata.size();
       ++index) {
    if (index != 0U) {
      output.put(',');
    }
    const ModelMetadataFileEvidence& file = evidence.model_metadata[index];
    output << "{\"filename\":";
    write_json_string(output, file.filename);
    output << ",\"canonical_path\":";
    write_json_string(output, file.canonical_path.string());
    output << ",\"file_bytes\":" << file.file_bytes
           << ",\"expected_sha256\":";
    write_json_string(output, kModelMetadataSha256[index]);
    output << ",\"observed_sha256\":";
    write_json_string(output, file.sha256.hex());
    output << ",\"authenticated\":";
    write_boolean(output, file.authenticated);
    output.put('}');
  }
  output << "]},\n  \"corpus\":{"
         << "\"file_bytes\":" << evidence.corpus.file_bytes
         << ",\"expected_file_sha256\":";
  write_json_string(output, kCorpusFileSha256);
  output << ",\"observed_file_sha256\":";
  write_json_string(output, evidence.corpus.file_sha256.hex());
  output << ",\"file_sha256_authenticated\":";
  write_boolean(output, evidence.corpus.file_sha256_authenticated);
  output << ",\"single_line_jsonl\":";
  write_boolean(output, evidence.corpus.single_line_jsonl);
  output << ",\"prompt_tokens\":"
         << evidence.corpus.prompt_token_ids.size()
         << ",\"expected_prompt_u32le_sha256\":";
  write_json_string(output, kPromptU32LeSha256);
  output << ",\"observed_prompt_u32le_sha256\":";
  write_json_string(output, evidence.corpus.prompt_u32le_sha256.hex());
  output << ",\"prompt_u32le_sha256_authenticated\":";
  write_boolean(output, evidence.corpus.prompt_u32le_sha256_authenticated);
  output << ",\"prompt_length_authenticated\":";
  write_boolean(output, evidence.corpus.prompt_length_authenticated);
  output << ",\"prompt_token_range_authenticated\":";
  write_boolean(output, evidence.corpus.prompt_token_range_authenticated);
  output << ",\"minimum_token_id\":" << evidence.corpus.minimum_token_id
         << ",\"maximum_token_id\":" << evidence.corpus.maximum_token_id
         << ",\"vocabulary_size\":" << runtime::kReferenceVocabularySize
         << "},\n  \"engine\":{\"create_attempted\":";
  write_boolean(output, evidence.engine_create_attempted);
  output << ",\"created\":";
  write_boolean(output, evidence.engine_created);
  output << ",\"backend\":";
  write_json_string(output, "sm87_weight_only");
  output << ",\"memory_profile\":";
  write_json_string(output, "legacy_c512");
  output << ",\"configured_max_sequence_length\":" << kPromptTokens
         << ",\"loaded_max_sequence_length\":"
         << evidence.load.request_max_sequence_length
         << ",\"configured_prefill_chunk_size\":" << kPrefillChunkSize
         << ",\"loaded_prefill_chunk_size\":"
         << evidence.load.request_prefill_chunk_size
         << ",\"configured_max_arena_bytes\":" << kRequestMaxArenaBytes
         << ",\"loaded_request_arena_bytes\":"
         << evidence.load.request_arena_bytes
         << ",\"legacy_c512_route_authenticated\":";
  write_boolean(output, evidence.legacy_c512_route_authenticated);
  output << ",\"target_aot_assets_configured\":false,"
            "\"target_aot_assets_requested\":";
  write_boolean(output,
                evidence.load.target_aot_projection_device_assets_requested);
  output << ",\"target_aot_assets_enabled\":";
  write_boolean(output,
                evidence.load.target_aot_projection_device_assets_enabled);
  output << ",\"target_aot_assets_attached\":";
  write_boolean(output,
                evidence.load.target_aot_projection_device_assets_attached);
  output << ",\"target_aot_assets_absent\":";
  write_boolean(output, evidence.target_aot_assets_absent);
  output << "},\n  \"device_envelope\":{\"query_attempted\":";
  write_boolean(output, evidence.device_query_attempted);
  output << ",\"device_ordinal\":" << evidence.device_ordinal
         << ",\"device_query_cuda_error\":"
         << evidence.device_query_cuda_error
         << ",\"device_properties_cuda_error\":"
         << evidence.device_properties_cuda_error << ",\"device_name\":";
  write_json_string(output, evidence.device_name);
  output << ",\"compute_capability_major\":"
         << evidence.device_compute_major
         << ",\"compute_capability_minor\":"
         << evidence.device_compute_minor << ",\"sm_count\":"
         << evidence.device_sm_count << ",\"total_memory_bytes\":"
         << evidence.device_total_memory_bytes << ",\"uuid_hex\":";
  write_json_string(output, evidence.device_uuid_hex);
  output << ",\"cuda_driver_version\":" << evidence.cuda_driver_version
         << ",\"cuda_runtime_version\":" << evidence.cuda_runtime_version
         << ",\"max_freq_requested_path\":";
  write_json_string(output, kGpuMaxFrequencyPath);
  output << ",\"max_freq_canonical_path\":";
  write_json_string(output,
                    evidence.gpu_max_frequency_canonical_path.string());
  output << ",\"max_freq_hz\":" << evidence.gpu_max_frequency_hz
         << ",\"available_frequencies_requested_path\":";
  write_json_string(output, kGpuAvailableFrequenciesPath);
  output << ",\"available_frequencies_canonical_path\":";
  write_json_string(
      output, evidence.gpu_available_frequencies_canonical_path.string());
  output << ",\"available_max_frequency_hz\":"
         << evidence.gpu_available_max_frequency_hz
         << ",\"frozen_maximum_frequency_hz\":"
         << kFrozenMaximumGpuFrequencyHz << ",\"authenticated\":";
  write_boolean(output, evidence.device_envelope_authenticated);
  output << "},\n  \"loaded_shards\":{\"authenticated\":";
  write_boolean(output, evidence.loaded_shards_match_pins);
  output << ",\"manifest_domain\":";
  write_json_string(output, kShardManifestDomain);
  output << ",\"manifest_sha256\":";
  write_json_string(output,
                    digest_hex(evidence.loaded_shard_manifest_sha256));
  output << ",\"expected\":[";
  const auto& pinned = runtime::pinned_qwen36_27b_shards();
  for (std::size_t index = 0U; index < pinned.size(); ++index) {
    if (index != 0U) {
      output.put(',');
    }
    output << "{\"filename\":";
    write_json_string(output, pinned[index].filename);
    output << ",\"file_size\":" << pinned[index].file_size
           << ",\"sha256\":";
    write_json_string(output, pinned[index].sha256);
    output.put('}');
  }
  output << "],\"observed\":[";
  for (std::size_t index = 0U;
       index < evidence.load.resident.shards.size(); ++index) {
    if (index != 0U) {
      output.put(',');
    }
    const runtime::ShardLoadStats& shard =
        evidence.load.resident.shards[index];
    output << "{\"filename\":";
    write_json_string(output, shard.filename);
    output << ",\"bytes_read\":" << shard.bytes_read
           << ",\"bytes_copied\":" << shard.bytes_copied
           << ",\"sha256\":";
    write_json_string(output, shard.sha256);
    output.put('}');
  }
  output << "]},\n  \"projection_tensor_catalog\":";
  write_projection_tensor_catalog(output, evidence);
  output << ",\n  \"identities\":{\"route_digest_domain\":";
  write_json_string(output, kRouteDigestDomain);
  output << ",\"real_p40_route_sha256\":";
  write_json_string(output, digest_hex(evidence.real_p40_route_sha256));
  output << ",\"route_identity_authenticated\":";
  write_boolean(output, evidence.route_identity_authenticated);
  output << "},\n  \"generation_attempt\":{\"attempted\":";
  write_boolean(output, evidence.generation_attempted);
  output << ",\"completed\":";
  write_boolean(output, evidence.generation_completed);
  output << ",\"value_present\":";
  write_boolean(output, evidence.generation_value_present);
  output << ",\"expected_runner_sentinel_failure\":";
  write_boolean(output, evidence.expected_runner_sentinel_failure);
  output << ",\"diagnostic\":{\"code\":";
  write_json_string(output,
                    runtime::to_string(evidence.generation_diagnostic.code));
  output << ",\"stage\":";
  write_json_string(output, evidence.generation_diagnostic.stage);
  output << ",\"message\":";
  write_json_string(output, evidence.generation_diagnostic.message);
  output << ",\"context\":";
  write_json_string(output, evidence.generation_diagnostic.context);
  output << ",\"dependency_error\":"
         << evidence.generation_diagnostic.dependency_error
         << ",\"cuda_error\":"
         << evidence.generation_diagnostic.cuda_error
         << ",\"layer\":" << evidence.generation_diagnostic.layer
         << ",\"operation\":";
  write_json_string(output, evidence.generation_diagnostic.operation);
  output << ",\"retired_prefill_quanta\":"
         << evidence.generation_diagnostic.retired_prefill_quanta
         << "}},\n  \"collector\":{\"prepare_attempted\":";
  write_boolean(output, evidence.collector_prepare_attempted);
  output << ",\"valid_at_prepare\":";
  write_boolean(output, evidence.collector_valid_at_prepare);
  output << ",\"hook_installed_exclusively\":";
  write_boolean(output, evidence.hook_installed_exclusively);
  output << ",\"hook_restored\":";
  write_boolean(output, evidence.hook_restored);
  output << ",\"finalized\":";
  write_boolean(output, evidence.collector_finalized);
  output << ",\"early_reject_before_finalize\":";
  write_boolean(output, evidence.collector_early_reject_before_finalize);
  output << ",\"early_reject_requested\":";
  write_boolean(output, evidence.collector.early_reject_requested);
  output << ",\"stable_early_reject_cuda_status\":"
         << static_cast<int>(
                active_cell::RealP40ActiveCellWitnessCollector::
                    early_reject_sentinel())
         << ",\"failure_code\":";
  write_json_string(output,
                    collector_failure_name(evidence.collector.failure_code));
  output << ",\"failure_code_raw\":"
         << static_cast<unsigned int>(evidence.collector.failure_code)
         << ",\"cuda_status\":" << evidence.collector.cuda_status
         << ",\"exceptional_flag_value\":"
         << evidence.collector.exceptional_flag_value
         << ",\"exceptional_operand_detected\":";
  write_boolean(output, evidence.collector.exceptional_operand_detected);
  output << ",\"failure_text\":";
  write_json_string(output,
                    bounded_failure_text(evidence.collector.failure_text));
  output << ",\"completed_segments\":"
         << evidence.collector.completed_segments
         << ",\"covered_prompt_rows\":"
         << evidence.collector.covered_prompt_rows
         << ",\"frozen_prefix_segments\":"
         << active_cell::kRealP40SegmentCount
         << ",\"frozen_prefix_rows\":"
         << witness_hook::kAotArithmeticWitnessP40PrefixRows
         << ",\"terminal_scalar_rows_excluded\":1,"
            "\"terminal_scalar_treated_as_free\":true,"
            "\"partial_m16_tail_rows_charged_free\":true,"
            "\"checkpoint_inventory_complete\":";
  write_boolean(output, evidence.collector.checkpoint_inventory_complete);
  output << ",\"checkpoint_manifest_authenticated\":";
  write_boolean(output,
                evidence.collector.checkpoint_manifest_authenticated);
  output << ",\"checkpoint_inventory_sha256\":";
  write_json_string(output,
                    digest_hex(evidence.collector.checkpoint_manifest_sha256));
  output << ",\"activation_capture_contract\":{\"chain_domain\":";
  write_json_string(output, active_cell::kActivationCaptureChainDomain);
  output << ",\"call_domain\":";
  write_json_string(output, active_cell::kActivationCallDomain);
  output << ",\"capture_point\":";
  write_json_string(
      output, "ReferenceRunner::enqueue_prefill_layer_segment A operand");
  output << ",\"source_dtype\":";
  write_json_string(output, active_cell::kActivationSourceDtype);
  output << ",\"payload_dtype\":";
  write_json_string(output, active_cell::kActivationPayloadDtype);
  output << ",\"source_shape_per_call\":";
  write_json_string(output, "[token_count,k] with row_stride_elements=k");
  output << ",\"payload_shape_per_call\":";
  write_json_string(output, "[ceil(token_count/16),k/16]");
  output << ",\"payload_definition\":";
  write_json_string(
      output,
      "one uint16 little-endian nonzero/finite support mask per M16xK16 "
      "source cell; this is the exact sufficient statistic consumed by the "
      "frozen static-support predicate");
  output << ",\"per_call_binding\":";
  write_json_string(
      output,
      "ordered chain binds segment, first_position, token_count, call, layer, "
      "role, scope, source dtype/shape/stride, payload dtype/shape/bytes, "
      "exceptional flag, partition count, and raw payload SHA256");
  output << ",\"raw_bf16_activation_payload_authenticated\":false,"
            "\"all_completed_call_payloads_authenticated\":";
  write_boolean(
      output,
      evidence.collector.completed_segments != 0U &&
          std::all_of(
              evidence.collector.roles.begin(), evidence.collector.roles.end(),
              [](const active_cell::RoleSummary& role) {
                return role.activation_inventory_authenticated &&
                       role.activation_call_count != 0U &&
                       role.activation_mask_element_count != 0U &&
                       role.activation_payload_bytes %
                               sizeof(std::uint16_t) ==
                           0U &&
                       role.activation_payload_bytes /
                               sizeof(std::uint16_t) ==
                           role.activation_mask_element_count;
              }));
  output << "},\"roles\":[";
  for (std::size_t index = 0U; index < evidence.collector.roles.size();
       ++index) {
    if (index != 0U) {
      output.put(',');
    }
    const active_cell::RoleSummary& role = evidence.collector.roles[index];
    output << "{\"role\":";
    write_json_string(output, lower_bound::to_string(role.role));
    output << ",\"weight_partition_count\":"
           << role.weight_partition_count
           << ",\"weight_inventory_complete\":";
    write_boolean(output, role.weight_inventory_complete);
    output << ",\"observed_geometric_joint_k16_cells\":"
           << role.observed_geometric_joint_k16_cells
           << ",\"proven_active_joint_k16_cells\":"
           << role.proven_active_joint_k16_cells
           << ",\"activation_inventory_sha256\":";
    write_json_string(output, digest_hex(role.activation_inventory_sha256));
    output << ",\"activation_inventory_authenticated\":";
    write_boolean(output, role.activation_inventory_authenticated);
    output << ",\"activation_call_count\":"
           << role.activation_call_count
           << ",\"activation_mask_element_count\":"
           << role.activation_mask_element_count
           << ",\"activation_payload_bytes\":"
           << role.activation_payload_bytes;
    output << ",\"joint_enumeration_sha256\":";
    write_json_string(output, digest_hex(role.joint_enumeration_sha256));
    output << ",\"joint_enumeration_authenticated\":";
    write_boolean(output, role.joint_enumeration_authenticated);
    output.put('}');
  }
  output << "]},\n  \"mapping\":";
  write_mapping(output);
  output << ",\n  \"lower_bound_receipt\":";
  write_receipt(output, evidence.collector);
  output << ",\n  \"diagnostic\":{\"stage\":";
  write_json_string(output, evidence.diagnostic_stage);
  output << ",\"message\":";
  write_json_string(output, evidence.diagnostic_message);
  output << "},\n  \"timing\":{\"wall_milliseconds\":"
         << evidence.wall_milliseconds
         << ",\"diagnostic_only\":true}\n}\n";
  return output.str();
}

[[nodiscard]] int run_self_test() {
  ProbeEvidence evidence;
  std::string binary_error;
  const bool binary_ok = authenticate_self_binary(evidence, binary_error);
  const bool provenance_shape_ok =
      lowercase_hex_string(provenance::kGitCommit, 40U) &&
      lowercase_hex_string(provenance::kGitTree, 40U) &&
      lowercase_hex_string(provenance::kBuildReceiptSha256, 64U) &&
      std::string_view(provenance::kBuildType) == "Release" &&
      provenance::kBuildTesting && provenance::kAotSystemAdmission &&
      provenance::kArithmeticWitnessAdmission &&
      std::string_view(provenance::kEnabledQ3xBuildOptions) ==
          kExpectedEnabledQ3xBuildOptions &&
      std::string_view(provenance::kEffectiveCudaArchitectures) == "87";
  const bool passed =
      active_cell::run_real_p40_active_cell_protocol_self_test() &&
      projection_catalog::run_p40_projection_catalog_protocol_self_test() &&
      run_decision_validator_self_test() && provenance_shape_ok && binary_ok &&
      evidence.binary.authenticated &&
      evidence.binary.sha256_complete &&
      evidence.binary.gnu_build_id.size() >= 32U;
  std::cout << "{\"schema\":\"q3x.sm87.aot-system-v1.real-p40-"
               "arithmetic-witness.protocol-self-test.v1\",\"passed\":"
            << (passed ? "true" : "false")
            << ",\"binary_identity_checked\":"
            << (binary_ok ? "true" : "false") << "}\n";
  if (!binary_ok) {
    std::cerr << binary_error << '\n';
  }
  return passed ? 0 : 1;
}

}  // namespace

int main(const int argc, char** argv) {
  if (argc == 2 && argv[1] != nullptr &&
      std::string_view(argv[1]) == "--self-test") {
    return run_self_test();
  }
  if (argc != 5) {
    std::cerr << "usage: " << argv[0]
              << " MODEL_DIRECTORY CORPUS_JSONL PREFLIGHT_JSON EVIDENCE_JSON\n"
                 "       "
              << argv[0] << " --self-test\n";
    return 2;
  }

  ProbeEvidence evidence;
  evidence.model_requested_path = argv[1];
  evidence.corpus.requested_path = argv[2];
  evidence.preflight.requested_path = argv[3];
  evidence.evidence_requested_path = argv[4];
  const auto started = std::chrono::steady_clock::now();

  std::string error;
  if (!resolve_evidence_path(evidence.evidence_requested_path,
                             evidence.evidence_canonical_path, error)) {
    std::cerr << "evidence path rejected: " << error << '\n';
    return 1;
  }
  CreateOnlyEvidenceFile evidence_file;
  if (!evidence_file.create(evidence.evidence_canonical_path, error)) {
    std::cerr << error << '\n';
    return 1;
  }

  try {
    if (!authenticate_clean_q3x_environment(evidence, error)) {
      set_failure(evidence, "ambient_environment", error);
    } else if (!authenticate_build_provenance(evidence, error)) {
      set_failure(evidence, "build_provenance", error);
    } else if (!authenticate_self_binary(evidence, error)) {
      set_failure(evidence, "self_binary_identity", error);
    } else if (!authenticate_preflight_record(
                   evidence.preflight, evidence.evidence_canonical_path,
                   evidence.binary.sha256.hex(), error)) {
      set_failure(evidence, "preflight_authentication", error);
    } else if (!canonical_directory(evidence.model_requested_path,
                                    evidence.model_canonical_path, error)) {
      set_failure(evidence, "model_path", error);
    } else if (!authenticate_model_metadata(evidence, error)) {
      set_failure(evidence, "model_metadata_authentication", error);
    } else if (!load_corpus(evidence.corpus, error)) {
      set_failure(evidence, "corpus_authentication", error);
    } else if (!authenticate_device_envelope(evidence, error)) {
      set_failure(evidence, "device_envelope_authentication", error);
    } else if (!authenticate_projection_tensor_catalog(evidence, error)) {
      set_failure(evidence, "projection_tensor_catalog", error);
    } else {
      runtime::ReferenceEngineOptions options;
      options.projection_backend = runtime::ProjectionBackend::kSm87WeightOnly;
      options.prefill_execution_mode =
          runtime::ReferencePrefillExecutionMode::kLegacyC512Tiled;
      options.prepare_sm87_target_aot_projection_device_assets = false;
      options.request_options.batch_size = 1U;
      options.request_options.max_sequence_length = kPromptTokens;
      options.request_options.prefill_chunk_size = kPrefillChunkSize;
      options.request_options.max_arena_bytes = kRequestMaxArenaBytes;

      evidence.engine_create_attempted = true;
      runtime::ReferenceEngineCreateResult created =
          runtime::create_reference_engine(evidence.model_canonical_path,
                                           options);
      evidence.engine_created = created.ok();
      if (!created) {
        evidence.generation_diagnostic = created.diagnostic;
        set_failure(evidence, "engine_create",
                    "Legacy-C512/SM87 engine creation failed: " +
                        created.diagnostic.message);
      } else {
        runtime::ReferenceEngine& engine = *created.value;
        evidence.load = engine.load_stats();
        evidence.load_captured = true;
        evidence.target_aot_assets_absent =
            target_aot_assets_are_absent(evidence.load);

        if (!authenticate_loaded_shards(
                evidence.load.resident,
                evidence.loaded_shard_manifest_sha256, error)) {
          set_failure(evidence, "loaded_shard_authentication", error);
        } else {
          evidence.loaded_shards_match_pins = true;
          if (!authenticate_engine_route(
                  engine, evidence.load, evidence.corpus, evidence,
                  evidence.loaded_shard_manifest_sha256,
                  evidence.real_p40_route_sha256, error)) {
            set_failure(evidence, "route_authentication", error);
          } else {
            evidence.legacy_c512_route_authenticated = true;
            evidence.route_identity_authenticated = true;
            active_cell::CollectorIdentity identity;
            identity.real_p40_route_sha256 =
                evidence.real_p40_route_sha256;
            identity.authenticated_shard_manifest_sha256 =
                evidence.loaded_shard_manifest_sha256;
            identity.real_p40_route_authenticated = true;
            identity.shard_manifest_authenticated = true;

            evidence.collector_prepare_attempted = true;
            active_cell::RealP40ActiveCellWitnessCollector collector(identity);
            evidence.collector_valid_at_prepare = collector.valid();
            if (!collector.valid()) {
              evidence.collector = collector.finalize();
              evidence.collector_finalized = true;
              set_failure(evidence, "collector_prepare",
                          "real-P40 active-cell collector preparation failed");
            } else {
              runtime::ReferenceGenerateResult generation;
              try {
                ScopedAOperandHook scoped_hook;
                evidence.hook_installed_exclusively =
                    scoped_hook.install(collector.hook());
                if (!evidence.hook_installed_exclusively) {
                  set_failure(evidence, "hook_install",
                              "source-local A-operand hook was already occupied");
                } else {
                  runtime::ReferenceGenerateOptions generate_options;
                  generate_options.max_new_tokens = kMaxNewTokens;
                  generate_options.prefill_chunk_size = kPrefillChunkSize;
                  generate_options.logits_mode =
                      runtime::ReferenceLogitsMode::kPredictedTokenOnly;
                  generate_options.prefill_execution_mode =
                      runtime::ReferencePrefillExecutionMode::kLegacyC512Tiled;
                  evidence.generation_attempted = true;
                  generation = engine.generate_prompt_token_ids(
                      evidence.corpus.prompt_token_ids, generate_options);
                  evidence.generation_completed = generation.ok();
                  evidence.generation_value_present =
                      generation.value.has_value();
                  evidence.generation_diagnostic = generation.diagnostic;
                  evidence.expected_runner_sentinel_failure =
                      expected_sentinel_failure(generation);
                  evidence.collector_early_reject_before_finalize =
                      collector.early_reject_requested();
                }
                scoped_hook.restore();
                evidence.hook_restored = scoped_hook.restored();
              } catch (const std::exception& exception) {
                set_failure(evidence, "generation_exception", exception.what());
              } catch (...) {
                set_failure(evidence, "generation_exception",
                            "unknown exception during real-P40 generation attempt");
              }

              evidence.collector = collector.finalize();
              evidence.collector_finalized = true;
              std::string reject_error;
              std::string inconclusive_error;
              if (validate_strict_reject(evidence, reject_error)) {
                evidence.valid_mapping_reject = true;
                evidence.valid_mapping_inconclusive = false;
                evidence.exit_code = 3;
                evidence.diagnostic_stage = "mapping_class_decision";
                evidence.diagnostic_message =
                    "strict authenticated real-P40 arithmetic mapping-class REJECT";
              } else if (validate_strict_inconclusive(
                             evidence, inconclusive_error)) {
                evidence.valid_mapping_reject = false;
                evidence.valid_mapping_inconclusive = true;
                evidence.exit_code = 0;
                evidence.diagnostic_stage = "mapping_class_decision";
                evidence.diagnostic_message =
                    "strict authenticated real-P40 arithmetic mapping-class "
                    "INCONCLUSIVE; direction survives this quick reject";
              } else {
                set_failure(
                    evidence, "mapping_decision_validation",
                    "REJECT validation: " + reject_error +
                        "; INCONCLUSIVE validation: " + inconclusive_error);
              }
            }
          }
        }
      }
    }
  } catch (const std::exception& exception) {
    set_failure(evidence, "probe_exception", exception.what());
  } catch (...) {
    set_failure(evidence, "probe_exception", "unknown probe exception");
  }

  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - started);
  evidence.wall_milliseconds =
      elapsed.count() < 0 ? 0U : static_cast<std::uint64_t>(elapsed.count());
  if (evidence.diagnostic_stage.empty() ||
      (!evidence.valid_mapping_reject &&
       !evidence.valid_mapping_inconclusive && evidence.exit_code != 1)) {
    set_failure(evidence, "probe_state",
                "probe ended without a strict decision or an explicit failure");
  }

  const std::string payload = serialize_evidence(evidence);
  if (!evidence_file.publish(payload, error)) {
    std::cerr << error << '\n';
    return 1;
  }
  std::cout << (evidence.valid_mapping_reject
                    ? "valid real-P40 arithmetic mapping-class REJECT"
                    : (evidence.valid_mapping_inconclusive
                           ? "valid real-P40 arithmetic mapping-class "
                             "INCONCLUSIVE"
                           : "real-P40 arithmetic witness infrastructure "
                             "failure"))
            << " evidence=" << evidence.evidence_canonical_path << '\n';
  return evidence.exit_code;
}
