#include "q3x/runtime/prefill_mlp_k512_fragment_native_overlay.h"

#include "q3x/core/sha256.h"
#include "q3x/io/json.h"
#include "q3x/kernels/sm87_a4w4_down_k512_fragment_native.h"
#include "q3x/kernels/sm87_a4w4_gateup_k512_fragment_native.h"
#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"
#include "q3x/runtime/prefill_quantized_contract.h"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <new>
#include <sstream>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace q3x::runtime {
namespace {

namespace fs = std::filesystem;
namespace json = q3x::io::json;
namespace kernels = q3x::kernels;

constexpr std::string_view kReceiptSchema =
    "q3x.prefill.mlp-k512.fragment-native.publication-receipt";

[[nodiscard]] PrefillMLPK512OverlayDiagnostic make_diagnostic(
    const PrefillMLPK512OverlayErrorCode code, std::string context,
    std::string message, std::string expected = {}, std::string actual = {},
    const int system_error = 0) {
  PrefillMLPK512OverlayDiagnostic result;
  result.code = code;
  result.context = std::move(context);
  result.message = std::move(message);
  result.expected = std::move(expected);
  result.actual = std::move(actual);
  result.system_error = system_error;
  return result;
}

[[nodiscard]] bool lower_sha256(const std::string_view value) noexcept {
  return value.size() == 64U &&
         std::all_of(value.begin(), value.end(), [](const char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

[[nodiscard]] bool product_fits(const std::size_t first,
                                const std::size_t second) noexcept {
  return first == 0U ||
         second <= std::numeric_limits<std::size_t>::max() / first;
}

[[nodiscard]] bool valid_base(
    const PrefillMLPK512BaseBinding& base) noexcept {
  return base.physical_layout == kPrefillA4K128PhysicalLayout &&
         lower_sha256(base.manifest_sha256) &&
         lower_sha256(base.policy_sha256) &&
         lower_sha256(base.payload_sha256);
}

[[nodiscard]] bool same_base(const PrefillMLPK512BaseBinding& first,
                             const PrefillMLPK512BaseBinding& second) noexcept {
  return first.physical_layout == second.physical_layout &&
         first.manifest_sha256 == second.manifest_sha256 &&
         first.policy_sha256 == second.policy_sha256 &&
         first.payload_sha256 == second.payload_sha256;
}

[[nodiscard]] bool valid_source_binding(
    const PrefillMLPK512FragmentNativeSourceBinding& source) noexcept {
  return source.physical_layout == kPrefillMLPK512OverlayLayout &&
         lower_sha256(source.receipt_sha256) &&
         lower_sha256(source.manifest_sha256) &&
         lower_sha256(source.policy_sha256) && source.policy_bytes != 0U &&
         lower_sha256(source.payload_sha256) &&
         source.payload_bytes == kPrefillMLPK512OverlayPayloadBytes;
}

[[nodiscard]] bool byte_ranges_overlap(
    const void* const first, const std::size_t first_bytes,
    const void* const second, const std::size_t second_bytes) noexcept {
  if (first == nullptr || second == nullptr || first_bytes == 0U ||
      second_bytes == 0U) {
    return false;
  }
  const std::uintptr_t first_begin =
      reinterpret_cast<std::uintptr_t>(first);
  const std::uintptr_t second_begin =
      reinterpret_cast<std::uintptr_t>(second);
  constexpr std::uintptr_t maximum =
      std::numeric_limits<std::uintptr_t>::max();
  if (first_bytes > maximum - first_begin ||
      second_bytes > maximum - second_begin) {
    return true;
  }
  return first_begin < second_begin + second_bytes &&
         second_begin < first_begin + first_bytes;
}

[[nodiscard]] bool gateup_ranges_are_disjoint(
    const std::uint8_t* const first_codes,
    const std::size_t first_code_bytes,
    const std::uint8_t* const first_scales,
    const std::size_t first_scale_bytes,
    const std::uint8_t* const second_codes,
    const std::size_t second_code_bytes,
    const std::uint8_t* const second_scales,
    const std::size_t second_scale_bytes,
    const std::uint8_t* const output_codes,
    const std::size_t output_code_bytes,
    const std::uint8_t* const output_scales,
    const std::size_t output_scale_bytes) noexcept {
  return !byte_ranges_overlap(first_codes, first_code_bytes, output_codes,
                              output_code_bytes) &&
         !byte_ranges_overlap(first_codes, first_code_bytes, output_scales,
                              output_scale_bytes) &&
         !byte_ranges_overlap(first_scales, first_scale_bytes, output_codes,
                              output_code_bytes) &&
         !byte_ranges_overlap(first_scales, first_scale_bytes, output_scales,
                              output_scale_bytes) &&
         !byte_ranges_overlap(second_codes, second_code_bytes, output_codes,
                              output_code_bytes) &&
         !byte_ranges_overlap(second_codes, second_code_bytes, output_scales,
                              output_scale_bytes) &&
         !byte_ranges_overlap(second_scales, second_scale_bytes, output_codes,
                              output_code_bytes) &&
         !byte_ranges_overlap(second_scales, second_scale_bytes,
                              output_scales, output_scale_bytes) &&
         !byte_ranges_overlap(output_codes, output_code_bytes, output_scales,
                              output_scale_bytes);
}

[[nodiscard]] std::size_t canonical_code_offset(
    const std::size_t outer, const std::size_t k64_group,
    const std::size_t byte_in_k64,
    const std::size_t k64_group_count) noexcept {
  return kernels::sm87_a4w4_consumer_packed_offset(
      outer, k64_group, byte_in_k64, k64_group_count);
}

[[nodiscard]] std::size_t canonical_scale_byte_offset(
    const std::size_t outer, const std::size_t k512_group,
    const std::size_t k512_group_count) noexcept {
  return 2U * (((outer / 64U) * k512_group_count + k512_group) * 64U +
               outer % 64U);
}

void write_quoted(std::ostream& output, const std::string_view value) {
  output << '"';
  for (const unsigned char character : value) {
    switch (character) {
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
        if (character < 0x20U) {
          output << "\\u00" << std::hex << std::setw(2)
                 << std::setfill('0') << static_cast<unsigned int>(character)
                 << std::dec << std::setfill(' ');
        } else {
          output << static_cast<char>(character);
        }
        break;
    }
  }
  output << '"';
}

void write_base(std::ostream& output,
                const PrefillMLPK512BaseBinding& base) {
  output << "{\"sidecar_kind\":\"a4_k128\",\"physical_layout\":";
  write_quoted(output, base.physical_layout);
  output << ",\"manifest_sha256\":";
  write_quoted(output, base.manifest_sha256);
  output << ",\"policy_sha256\":";
  write_quoted(output, base.policy_sha256);
  output << ",\"payload_sha256\":";
  write_quoted(output, base.payload_sha256);
  output << '}';
}

void write_source(std::ostream& output,
                  const PrefillMLPK512FragmentNativeSourceBinding& source) {
  output << "{\"physical_layout\":";
  write_quoted(output, source.physical_layout);
  output << ",\"receipt_sha256\":";
  write_quoted(output, source.receipt_sha256);
  output << ",\"manifest_sha256\":";
  write_quoted(output, source.manifest_sha256);
  output << ",\"policy_sha256\":";
  write_quoted(output, source.policy_sha256);
  output << ",\"policy_bytes\":" << source.policy_bytes
         << ",\"payload_sha256\":";
  write_quoted(output, source.payload_sha256);
  output << ",\"payload_bytes\":" << source.payload_bytes << '}';
}

[[nodiscard]] std::string sha256_text(const std::string_view text) {
  core::Sha256 hash;
  if (!hash.update(text.data(), text.size())) {
    return {};
  }
  return hash.finalize().hex();
}

[[nodiscard]] std::string manifest_body(
    const PrefillMLPK512FragmentNativeManifest& manifest) {
  std::ostringstream output;
  output << "schema=q3x.prefill.mlp-k512.fragment-native-manifest\nversion="
         << manifest.version_major << '.' << manifest.version_minor
         << "\nlayout=" << manifest.physical_layout
         << "\ncheckpoint=" << manifest.source_checkpoint_id
         << "\nconfig=" << manifest.source_config_sha256
         << "\nindex=" << manifest.source_index_sha256 << "\nbase="
         << manifest.required_base.physical_layout << ':'
         << manifest.required_base.manifest_sha256 << ':'
         << manifest.required_base.policy_sha256 << ':'
         << manifest.required_base.payload_sha256 << "\nsource_v1="
         << manifest.source_v1.physical_layout << ':'
         << manifest.source_v1.receipt_sha256 << ':'
         << manifest.source_v1.manifest_sha256 << ':'
         << manifest.source_v1.policy_sha256 << ':'
         << manifest.source_v1.policy_bytes << ':'
         << manifest.source_v1.payload_sha256 << ':'
         << manifest.source_v1.payload_bytes << "\npayload="
         << manifest.payload_bytes << "\nlayers=" << manifest.layer_count
         << '\n';
  for (std::size_t layer = 0U;
       layer < kPrefillMLPK512FragmentNativeLayerCount; ++layer) {
    const auto view = prefill_mlp_k512_fragment_native_layer_view(layer);
    output << view.layer_index << ':' << view.layer_offset << ':'
           << view.gateup_code_offset << ':' << view.gateup_code_bytes << ':'
           << view.gateup_scale_offset << ':' << view.gateup_scale_bytes << ':'
           << view.down_code_offset << ':' << view.down_code_bytes << ':'
           << view.down_scale_offset << ':' << view.down_scale_bytes << '\n';
  }
  return output.str();
}

[[nodiscard]] std::string serialize_receipt(
    const PrefillMLPK512FragmentNativeReceipt& receipt) {
  std::ostringstream output;
  output << "{\"schema\":\"" << kReceiptSchema
         << "\",\"version\":{\"major\":1,\"minor\":0},"
            "\"mode\":\"lossless_permutation\","
            "\"production_residency_eligible\":true,"
            "\"physical_layout\":";
  write_quoted(output, receipt.physical_layout);
  output << ",\"source_checkpoint_id\":";
  write_quoted(output, receipt.source_checkpoint_id);
  output << ",\"source_config_sha256\":";
  write_quoted(output, receipt.source_config_sha256);
  output << ",\"source_index_sha256\":";
  write_quoted(output, receipt.source_index_sha256);
  output << ",\"required_base\":";
  write_base(output, receipt.required_base);
  output << ",\"source_v1\":";
  write_source(output, receipt.source_v1);
  output << ",\"manifest_sha256\":";
  write_quoted(output, receipt.manifest_sha256);
  output << ",\"payload_sha256\":";
  write_quoted(output, receipt.payload_sha256);
  output << ",\"payload_bytes\":" << receipt.payload_bytes
         << ",\"layer_count\":" << receipt.layer_count << "}\n";
  return output.str();
}

[[nodiscard]] bool exact_keys(
    const json::Value::Object& object,
    const std::initializer_list<std::string_view> keys) {
  if (object.size() != keys.size()) {
    return false;
  }
  return std::all_of(keys.begin(), keys.end(), [&](const auto key) {
    return object.find(key) != object.end();
  });
}

[[nodiscard]] bool json_string(const json::Value::Object& object,
                               const std::string_view key,
                               std::string& value) {
  const auto found = object.find(key);
  const std::string* string =
      found == object.end() ? nullptr : found->second.as_string();
  if (string == nullptr) {
    return false;
  }
  value = *string;
  return true;
}

[[nodiscard]] bool json_uint(const json::Value::Object& object,
                             const std::string_view key,
                             std::uint64_t& value) {
  const auto found = object.find(key);
  if (found == object.end()) {
    return false;
  }
  const json::Number* number = found->second.as_number();
  return number != nullptr && number->to_uint64(value);
}

[[nodiscard]] bool parse_version(const json::Value& value,
                                 std::uint32_t& major,
                                 std::uint32_t& minor) {
  const auto* object = value.as_object();
  std::uint64_t parsed_major = 0U;
  std::uint64_t parsed_minor = 0U;
  if (object == nullptr || !exact_keys(*object, {"major", "minor"}) ||
      !json_uint(*object, "major", parsed_major) ||
      !json_uint(*object, "minor", parsed_minor) ||
      parsed_major > std::numeric_limits<std::uint32_t>::max() ||
      parsed_minor > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  major = static_cast<std::uint32_t>(parsed_major);
  minor = static_cast<std::uint32_t>(parsed_minor);
  return true;
}

[[nodiscard]] bool parse_base(const json::Value& value,
                              PrefillMLPK512BaseBinding& base) {
  const auto* object = value.as_object();
  std::string kind;
  return object != nullptr &&
         exact_keys(*object,
                    {"sidecar_kind", "physical_layout", "manifest_sha256",
                     "policy_sha256", "payload_sha256"}) &&
         json_string(*object, "sidecar_kind", kind) && kind == "a4_k128" &&
         json_string(*object, "physical_layout", base.physical_layout) &&
         json_string(*object, "manifest_sha256", base.manifest_sha256) &&
         json_string(*object, "policy_sha256", base.policy_sha256) &&
         json_string(*object, "payload_sha256", base.payload_sha256) &&
         valid_base(base);
}

[[nodiscard]] bool parse_source(
    const json::Value& value,
    PrefillMLPK512FragmentNativeSourceBinding& source) {
  const auto* object = value.as_object();
  return object != nullptr &&
         exact_keys(*object,
                    {"physical_layout", "receipt_sha256", "manifest_sha256",
                     "policy_sha256", "policy_bytes", "payload_sha256",
                     "payload_bytes"}) &&
         json_string(*object, "physical_layout", source.physical_layout) &&
         json_string(*object, "receipt_sha256", source.receipt_sha256) &&
         json_string(*object, "manifest_sha256", source.manifest_sha256) &&
         json_string(*object, "policy_sha256", source.policy_sha256) &&
         json_uint(*object, "policy_bytes", source.policy_bytes) &&
         json_string(*object, "payload_sha256", source.payload_sha256) &&
         json_uint(*object, "payload_bytes", source.payload_bytes) &&
         valid_source_binding(source);
}

class UniqueFd final {
 public:
  UniqueFd() noexcept = default;
  explicit UniqueFd(const int fd) noexcept : fd_(fd) {}
  ~UniqueFd() {
    if (fd_ >= 0) {
      (void)::close(fd_);
    }
  }
  UniqueFd(const UniqueFd&) = delete;
  UniqueFd& operator=(const UniqueFd&) = delete;
  UniqueFd(UniqueFd&& other) noexcept : fd_(other.fd_) { other.fd_ = -1; }
  UniqueFd& operator=(UniqueFd&& other) noexcept {
    if (this != &other) {
      if (fd_ >= 0) {
        (void)::close(fd_);
      }
      fd_ = other.fd_;
      other.fd_ = -1;
    }
    return *this;
  }
  [[nodiscard]] int get() const noexcept { return fd_; }
  [[nodiscard]] explicit operator bool() const noexcept { return fd_ >= 0; }

 private:
  int fd_ = -1;
};

[[nodiscard]] bool pread_exact(const int fd, void* const destination,
                               const std::size_t bytes,
                               const std::uint64_t offset,
                               int& error) noexcept {
  if (offset >
      static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
    error = EOVERFLOW;
    return false;
  }
  auto* output = static_cast<std::uint8_t*>(destination);
  std::size_t complete = 0U;
  while (complete < bytes) {
    const std::uint64_t position = offset + complete;
    if (position < offset ||
        position >
            static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
      error = EOVERFLOW;
      return false;
    }
    const ssize_t count = ::pread(fd, output + complete, bytes - complete,
                                  static_cast<off_t>(position));
    if (count < 0) {
      if (errno == EINTR) {
        continue;
      }
      error = errno;
      return false;
    }
    if (count == 0) {
      error = EIO;
      return false;
    }
    complete += static_cast<std::size_t>(count);
  }
  return true;
}

[[nodiscard]] bool pwrite_exact(const int fd, const void* const source,
                                const std::size_t bytes,
                                const std::uint64_t offset,
                                int& error) noexcept {
  if (offset >
      static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
    error = EOVERFLOW;
    return false;
  }
  const auto* input = static_cast<const std::uint8_t*>(source);
  std::size_t complete = 0U;
  while (complete < bytes) {
    const std::uint64_t position = offset + complete;
    if (position < offset ||
        position >
            static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
      error = EOVERFLOW;
      return false;
    }
    const ssize_t count = ::pwrite(fd, input + complete, bytes - complete,
                                   static_cast<off_t>(position));
    if (count < 0) {
      if (errno == EINTR) {
        continue;
      }
      error = errno;
      return false;
    }
    if (count == 0) {
      error = EIO;
      return false;
    }
    complete += static_cast<std::size_t>(count);
  }
  return true;
}

struct FileSnapshot final {
  dev_t device{};
  ino_t inode{};
  off_t size{};
  timespec modified{};
  timespec changed{};
};

[[nodiscard]] bool snapshot_fd(const int fd, FileSnapshot& snapshot,
                               int& error) noexcept {
  struct stat status {};
  if (::fstat(fd, &status) != 0) {
    error = errno;
    return false;
  }
  if (!S_ISREG(status.st_mode)) {
    error = EINVAL;
    return false;
  }
  snapshot.device = status.st_dev;
  snapshot.inode = status.st_ino;
  snapshot.size = status.st_size;
  snapshot.modified = status.st_mtim;
  snapshot.changed = status.st_ctim;
  return true;
}

[[nodiscard]] bool same_snapshot(const FileSnapshot& first,
                                 const FileSnapshot& second) noexcept {
  return first.device == second.device && first.inode == second.inode &&
         first.size == second.size &&
         first.modified.tv_sec == second.modified.tv_sec &&
         first.modified.tv_nsec == second.modified.tv_nsec &&
         first.changed.tv_sec == second.changed.tv_sec &&
         first.changed.tv_nsec == second.changed.tv_nsec;
}

[[nodiscard]] PrefillMLPK512OverlayDiagnostic hash_fd(
    const int fd, const std::uint64_t bytes, std::string& digest,
    std::uint64_t* const bytes_read = nullptr) {
  try {
    constexpr std::size_t kChunk = 8U * 1024U * 1024U;
    std::vector<std::uint8_t> buffer(static_cast<std::size_t>(
        std::min<std::uint64_t>(bytes, kChunk)));
    core::Sha256 hash;
    std::uint64_t offset = 0U;
    while (offset < bytes) {
      const std::size_t count = static_cast<std::size_t>(
          std::min<std::uint64_t>(buffer.size(), bytes - offset));
      int error = 0;
      if (!pread_exact(fd, buffer.data(), count, offset, error)) {
        return make_diagnostic(PrefillMLPK512OverlayErrorCode::kIoFailure,
                               "sha256", "file read failed while hashing",
                               {}, {}, error);
      }
      if (!hash.update(buffer.data(), count)) {
        return make_diagnostic(
            PrefillMLPK512OverlayErrorCode::kArithmeticOverflow, "sha256",
            "SHA-256 byte count overflowed");
      }
      offset += count;
    }
    if (bytes_read != nullptr) {
      *bytes_read += bytes;
    }
    digest = hash.finalize().hex();
    return {};
  } catch (const std::bad_alloc&) {
    return make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kAllocationFailure, "sha256",
        "hash buffer allocation failed");
  }
}

[[nodiscard]] PrefillMLPK512OverlayDiagnostic read_bounded_file(
    const fs::path& path, const std::uint64_t maximum_bytes,
    std::string& document, std::uint64_t* const bytes_read = nullptr) {
  UniqueFd input(::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  if (!input) {
    return make_diagnostic(PrefillMLPK512OverlayErrorCode::kIoFailure,
                           path.string(), "failed to open bounded input", {},
                           {}, errno);
  }
  FileSnapshot snapshot;
  int error = 0;
  if (!snapshot_fd(input.get(), snapshot, error) || snapshot.size < 0 ||
      static_cast<std::uint64_t>(snapshot.size) > maximum_bytes) {
    return make_diagnostic(PrefillMLPK512OverlayErrorCode::kIoFailure,
                           path.string(), "bounded input size is invalid", {},
                           {}, error);
  }
  try {
    document.resize(static_cast<std::size_t>(snapshot.size));
  } catch (const std::bad_alloc&) {
    return make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kAllocationFailure, path.string(),
        "bounded input allocation failed");
  }
  if (!document.empty() &&
      !pread_exact(input.get(), document.data(), document.size(), 0U, error)) {
    return make_diagnostic(PrefillMLPK512OverlayErrorCode::kIoFailure,
                           path.string(), "bounded input read failed", {}, {},
                           error);
  }
  FileSnapshot after;
  if (!snapshot_fd(input.get(), after, error) ||
      !same_snapshot(snapshot, after)) {
    return make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kSourceAuthenticationFailed,
        path.string(), "bounded input changed while being read", {}, {},
        error);
  }
  if (bytes_read != nullptr) {
    *bytes_read += document.size();
  }
  return {};
}

[[nodiscard]] PrefillMLPK512OverlayDiagnostic publish_document_no_replace(
    const fs::path& target, const std::string_view document) {
  if (target.empty()) {
    return make_diagnostic(PrefillMLPK512OverlayErrorCode::kInvalidOption,
                           "receipt", "receipt output path is empty");
  }
  struct stat status {};
  errno = 0;
  if (::lstat(target.c_str(), &status) == 0 || errno != ENOENT) {
    return make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kPublicationConflict, target.string(),
        "publication target already exists", {}, {}, errno);
  }
  const fs::path temporary =
      fs::path(target.string() + ".tmp." + std::to_string(::getpid()));
  UniqueFd output(::open(temporary.c_str(),
                         O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
                         S_IRUSR | S_IWUSR));
  if (!output) {
    return make_diagnostic(PrefillMLPK512OverlayErrorCode::kIoFailure,
                           temporary.string(),
                           "failed to create temporary publication", {}, {},
                           errno);
  }
  int error = 0;
  if (!pwrite_exact(output.get(), document.data(), document.size(), 0U,
                    error) ||
      ::fchmod(output.get(), S_IRUSR) != 0 || ::fsync(output.get()) != 0) {
    const int saved = error != 0 ? error : errno;
    (void)::unlink(temporary.c_str());
    return make_diagnostic(PrefillMLPK512OverlayErrorCode::kIoFailure,
                           temporary.string(),
                           "failed to seal temporary publication", {}, {},
                           saved);
  }
  if (::link(temporary.c_str(), target.c_str()) != 0) {
    const int saved = errno;
    (void)::unlink(temporary.c_str());
    return make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kPublicationConflict, target.string(),
        "atomic no-replace publication failed", {}, {}, saved);
  }
  if (::unlink(temporary.c_str()) != 0) {
    const int saved = errno;
    (void)::unlink(target.c_str());
    return make_diagnostic(PrefillMLPK512OverlayErrorCode::kIoFailure,
                           target.string(),
                           "temporary unlink failed; publication rolled back",
                           {}, {}, saved);
  }
  const fs::path parent =
      target.parent_path().empty() ? fs::path(".") : target.parent_path();
  UniqueFd directory(::open(parent.c_str(),
                            O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
  if (!directory || ::fsync(directory.get()) != 0) {
    const int saved = errno;
    (void)::unlink(target.c_str());
    return make_diagnostic(PrefillMLPK512OverlayErrorCode::kIoFailure,
                           parent.string(),
                           "directory sync failed; publication rolled back",
                           {}, {}, saved);
  }
  return {};
}

}  // namespace

PrefillMLPK512FragmentNativeManifestResult
build_prefill_mlp_k512_fragment_native_manifest(
    const PrefillMLPK512OverlayReceipt& source,
    const std::string_view source_receipt_sha256) {
  PrefillMLPK512FragmentNativeManifestResult result;
  if (source.version_major != kPrefillMLPK512OverlayVersionMajor ||
      source.version_minor != kPrefillMLPK512OverlayVersionMinor ||
      !source.production_residency_eligible ||
      source.physical_layout != kPrefillMLPK512OverlayLayout ||
      source.source_checkpoint_id.empty() ||
      !lower_sha256(source.source_config_sha256) ||
      !lower_sha256(source.source_index_sha256) ||
      !lower_sha256(source.manifest_sha256) ||
      !lower_sha256(source.policy_sha256) || source.policy_bytes == 0U ||
      !valid_base(source.required_base) ||
      !lower_sha256(source.payload_sha256) ||
      source.payload_bytes != kPrefillMLPK512OverlayPayloadBytes ||
      source.projection_count != kPrefillMLPK512OverlayProjectionCount ||
      !lower_sha256(source_receipt_sha256)) {
    result.diagnostic = make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kInvalidReceipt, "source_v1",
        "source is not a complete authenticated MLP K512 v1 publication");
    return result;
  }
  PrefillMLPK512FragmentNativeManifest manifest;
  manifest.physical_layout =
      std::string(kPrefillMLPK512FragmentNativeLayout);
  manifest.source_checkpoint_id = source.source_checkpoint_id;
  manifest.source_config_sha256 = source.source_config_sha256;
  manifest.source_index_sha256 = source.source_index_sha256;
  manifest.required_base = source.required_base;
  manifest.source_v1.physical_layout = source.physical_layout;
  manifest.source_v1.receipt_sha256 = source_receipt_sha256;
  manifest.source_v1.manifest_sha256 = source.manifest_sha256;
  manifest.source_v1.policy_sha256 = source.policy_sha256;
  manifest.source_v1.policy_bytes = source.policy_bytes;
  manifest.source_v1.payload_sha256 = source.payload_sha256;
  manifest.source_v1.payload_bytes = source.payload_bytes;
  manifest.payload_bytes = kPrefillMLPK512FragmentNativePayloadBytes;
  manifest.layer_count = kPrefillMLPK512FragmentNativeLayerCount;
  manifest.manifest_sha256 = sha256_text(manifest_body(manifest));
  result.diagnostic =
      validate_prefill_mlp_k512_fragment_native_manifest(manifest);
  if (!result.diagnostic) {
    return result;
  }
  result.value.emplace(std::move(manifest));
  return result;
}

PrefillMLPK512OverlayDiagnostic
validate_prefill_mlp_k512_fragment_native_manifest(
    const PrefillMLPK512FragmentNativeManifest& manifest) {
  if (manifest.version_major !=
          kPrefillMLPK512FragmentNativeVersionMajor ||
      manifest.version_minor !=
          kPrefillMLPK512FragmentNativeVersionMinor ||
      manifest.physical_layout != kPrefillMLPK512FragmentNativeLayout ||
      manifest.source_checkpoint_id.empty() ||
      !lower_sha256(manifest.source_config_sha256) ||
      !lower_sha256(manifest.source_index_sha256) ||
      !valid_base(manifest.required_base) ||
      !valid_source_binding(manifest.source_v1) ||
      manifest.payload_bytes != kPrefillMLPK512FragmentNativePayloadBytes ||
      manifest.layer_count != kPrefillMLPK512FragmentNativeLayerCount ||
      !lower_sha256(manifest.manifest_sha256)) {
    return make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kInvalidManifest, "manifest",
        "fragment-native MLP K512 manifest header is invalid");
  }
  std::uint64_t expected_offset = 0U;
  for (std::size_t layer = 0U;
       layer < kPrefillMLPK512FragmentNativeLayerCount; ++layer) {
    const auto view = prefill_mlp_k512_fragment_native_layer_view(layer);
    if (!view.valid || view.layer_index != layer ||
        view.layer_offset != expected_offset ||
        view.gateup_code_offset != expected_offset ||
        view.gateup_code_bytes !=
            kPrefillMLPK512FragmentNativeGateUpCodeBytes ||
        view.gateup_scale_offset !=
            view.gateup_code_offset + view.gateup_code_bytes ||
        view.gateup_scale_bytes !=
            kPrefillMLPK512FragmentNativeGateUpScaleBytes ||
        view.down_code_offset !=
            view.gateup_scale_offset + view.gateup_scale_bytes ||
        view.down_code_bytes !=
            kPrefillMLPK512FragmentNativeDownCodeBytes ||
        view.down_scale_offset !=
            view.down_code_offset + view.down_code_bytes ||
        view.down_scale_bytes !=
            kPrefillMLPK512FragmentNativeDownScaleBytes) {
      return make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kInvalidManifest,
          "manifest.layers[" + std::to_string(layer) + "]",
          "layer-major composite offsets are not contiguous");
    }
    expected_offset += kPrefillMLPK512FragmentNativeLayerBytes;
  }
  if (expected_offset != manifest.payload_bytes) {
    return make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kInvalidManifest, "manifest.layers",
        "layer inventory does not span the complete payload");
  }
  const std::string digest = sha256_text(manifest_body(manifest));
  if (digest != manifest.manifest_sha256) {
    return make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kDigestMismatch,
        "manifest.manifest_sha256", "manifest digest mismatch",
        manifest.manifest_sha256, digest);
  }
  return {};
}

PrefillMLPK512OverlayDiagnostic
permute_prefill_mlp_k512_gateup_fragment_native(
    const std::uint8_t* const gate_codes,
    const std::size_t gate_code_bytes,
    const std::uint8_t* const gate_scales,
    const std::size_t gate_scale_bytes,
    const std::uint8_t* const up_codes, const std::size_t up_code_bytes,
    const std::uint8_t* const up_scales,
    const std::size_t up_scale_bytes, const std::size_t output_size,
    const std::size_t input_size, std::uint8_t* const paired_codes,
    const std::size_t paired_code_bytes,
    std::uint8_t* const paired_scales,
    const std::size_t paired_scale_bytes) {
  if (gate_codes == nullptr || gate_scales == nullptr || up_codes == nullptr ||
      up_scales == nullptr || paired_codes == nullptr ||
      paired_scales == nullptr || output_size == 0U ||
      output_size % 64U != 0U || input_size == 0U ||
      input_size % 512U != 0U || !product_fits(output_size, input_size)) {
    return make_diagnostic(PrefillMLPK512OverlayErrorCode::kInvalidOption,
                           "gateup_permute",
                           "complete N64/K512 buffers are required");
  }
  const std::size_t elements = output_size * input_size;
  const std::size_t projection_codes = elements / 2U;
  const std::size_t projection_scales = elements / 512U * 2U;
  if (gate_code_bytes != projection_codes ||
      up_code_bytes != projection_codes ||
      gate_scale_bytes != projection_scales ||
      up_scale_bytes != projection_scales ||
      paired_code_bytes != 2U * projection_codes ||
      paired_scale_bytes != 2U * projection_scales ||
      !gateup_ranges_are_disjoint(
          gate_codes, gate_code_bytes, gate_scales, gate_scale_bytes, up_codes,
          up_code_bytes, up_scales, up_scale_bytes, paired_codes,
          paired_code_bytes, paired_scales, paired_scale_bytes)) {
    return make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kInvalidOption, "gateup_permute",
        "capacities differ from equal-byte ABI or ranges overlap");
  }
  const std::size_t k512_groups = input_size / 512U;
  const std::size_t k64_groups = input_size / 64U;
  for (std::size_t block = 0U; block < output_size / 64U; ++block) {
    for (std::size_t group = 0U; group < k512_groups; ++group) {
      for (std::size_t phase = 0U; phase < 8U; ++phase) {
        for (std::size_t fragment = 0U; fragment < 8U; ++fragment) {
          const std::size_t fragment_n = block * 64U + fragment * 8U;
          for (std::size_t lane = 0U; lane < 32U; ++lane) {
            const std::size_t row = fragment_n + lane / 4U;
            const std::size_t source0 = canonical_code_offset(
                row, group * 8U + phase, 4U * (lane % 4U), k64_groups);
            const std::size_t source1 = canonical_code_offset(
                row, group * 8U + phase, 16U + 4U * (lane % 4U),
                k64_groups);
            const std::size_t destination =
                kernels::
                    sm87_a4w4_gateup_k512_fragment_native_code_slot_offset(
                        fragment_n, group, phase, lane, k512_groups);
            std::memcpy(paired_codes + destination, gate_codes + source0, 4U);
            std::memcpy(paired_codes + destination + 4U,
                        gate_codes + source1, 4U);
            std::memcpy(paired_codes + destination + 8U, up_codes + source0,
                        4U);
            std::memcpy(paired_codes + destination + 12U, up_codes + source1,
                        4U);
          }
        }
      }
      for (std::size_t row = block * 64U; row < (block + 1U) * 64U;
           ++row) {
        const std::size_t source =
            canonical_scale_byte_offset(row, group, k512_groups);
        const std::size_t destination =
            2U * kernels::
                       sm87_a4w4_gateup_k512_fragment_native_scale_pair_offset(
                           row, group, k512_groups);
        std::memcpy(paired_scales + destination, gate_scales + source, 2U);
        std::memcpy(paired_scales + destination + 2U, up_scales + source, 2U);
      }
    }
  }
  return {};
}

PrefillMLPK512OverlayDiagnostic
unpermute_prefill_mlp_k512_gateup_fragment_native(
    const std::uint8_t* const paired_codes,
    const std::size_t paired_code_bytes,
    const std::uint8_t* const paired_scales,
    const std::size_t paired_scale_bytes, const std::size_t output_size,
    const std::size_t input_size, std::uint8_t* const gate_codes,
    const std::size_t gate_code_bytes, std::uint8_t* const gate_scales,
    const std::size_t gate_scale_bytes, std::uint8_t* const up_codes,
    const std::size_t up_code_bytes, std::uint8_t* const up_scales,
    const std::size_t up_scale_bytes) {
  if (gate_codes == nullptr || gate_scales == nullptr || up_codes == nullptr ||
      up_scales == nullptr || paired_codes == nullptr ||
      paired_scales == nullptr || output_size == 0U ||
      output_size % 64U != 0U || input_size == 0U ||
      input_size % 512U != 0U || !product_fits(output_size, input_size)) {
    return make_diagnostic(PrefillMLPK512OverlayErrorCode::kInvalidOption,
                           "gateup_unpermute",
                           "complete N64/K512 buffers are required");
  }
  const std::size_t elements = output_size * input_size;
  const std::size_t projection_codes = elements / 2U;
  const std::size_t projection_scales = elements / 512U * 2U;
  if (gate_code_bytes != projection_codes ||
      up_code_bytes != projection_codes ||
      gate_scale_bytes != projection_scales ||
      up_scale_bytes != projection_scales ||
      paired_code_bytes != 2U * projection_codes ||
      paired_scale_bytes != 2U * projection_scales ||
      !gateup_ranges_are_disjoint(
          gate_codes, gate_code_bytes, gate_scales, gate_scale_bytes, up_codes,
          up_code_bytes, up_scales, up_scale_bytes, paired_codes,
          paired_code_bytes, paired_scales, paired_scale_bytes)) {
    return make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kInvalidOption, "gateup_unpermute",
        "capacities differ from equal-byte ABI or ranges overlap");
  }
  const std::size_t k512_groups = input_size / 512U;
  const std::size_t k64_groups = input_size / 64U;
  for (std::size_t block = 0U; block < output_size / 64U; ++block) {
    for (std::size_t group = 0U; group < k512_groups; ++group) {
      for (std::size_t phase = 0U; phase < 8U; ++phase) {
        for (std::size_t fragment = 0U; fragment < 8U; ++fragment) {
          const std::size_t fragment_n = block * 64U + fragment * 8U;
          for (std::size_t lane = 0U; lane < 32U; ++lane) {
            const std::size_t row = fragment_n + lane / 4U;
            const std::size_t destination0 = canonical_code_offset(
                row, group * 8U + phase, 4U * (lane % 4U), k64_groups);
            const std::size_t destination1 = canonical_code_offset(
                row, group * 8U + phase, 16U + 4U * (lane % 4U),
                k64_groups);
            const std::size_t source =
                kernels::
                    sm87_a4w4_gateup_k512_fragment_native_code_slot_offset(
                        fragment_n, group, phase, lane, k512_groups);
            std::memcpy(gate_codes + destination0, paired_codes + source, 4U);
            std::memcpy(gate_codes + destination1, paired_codes + source + 4U,
                        4U);
            std::memcpy(up_codes + destination0, paired_codes + source + 8U,
                        4U);
            std::memcpy(up_codes + destination1, paired_codes + source + 12U,
                        4U);
          }
        }
      }
      for (std::size_t row = block * 64U; row < (block + 1U) * 64U;
           ++row) {
        const std::size_t destination =
            canonical_scale_byte_offset(row, group, k512_groups);
        const std::size_t source =
            2U * kernels::
                       sm87_a4w4_gateup_k512_fragment_native_scale_pair_offset(
                           row, group, k512_groups);
        std::memcpy(gate_scales + destination, paired_scales + source, 2U);
        std::memcpy(up_scales + destination, paired_scales + source + 2U, 2U);
      }
    }
  }
  return {};
}

PrefillMLPK512OverlayDiagnostic
permute_prefill_mlp_k512_down_fragment_native(
    const std::uint8_t* const canonical_codes,
    const std::size_t canonical_code_bytes,
    const std::uint8_t* const canonical_scales,
    const std::size_t canonical_scale_bytes, const std::size_t output_size,
    const std::size_t input_size, std::uint8_t* const fragment_codes,
    const std::size_t fragment_code_bytes,
    std::uint8_t* const fragment_scales,
    const std::size_t fragment_scale_bytes) {
  if (canonical_codes == nullptr || canonical_scales == nullptr ||
      fragment_codes == nullptr || fragment_scales == nullptr ||
      output_size == 0U || output_size % 128U != 0U || input_size == 0U ||
      input_size % 512U != 0U || !product_fits(output_size, input_size)) {
    return make_diagnostic(PrefillMLPK512OverlayErrorCode::kInvalidOption,
                           "down_permute",
                           "complete N128/K512 buffers are required");
  }
  const std::size_t elements = output_size * input_size;
  const std::size_t expected_codes = elements / 2U;
  const std::size_t expected_scales = elements / 512U * 2U;
  if (canonical_code_bytes != expected_codes ||
      fragment_code_bytes != expected_codes ||
      canonical_scale_bytes != expected_scales ||
      fragment_scale_bytes != expected_scales ||
      byte_ranges_overlap(canonical_codes, canonical_code_bytes,
                          fragment_codes, fragment_code_bytes) ||
      byte_ranges_overlap(canonical_codes, canonical_code_bytes,
                          fragment_scales, fragment_scale_bytes) ||
      byte_ranges_overlap(canonical_scales, canonical_scale_bytes,
                          fragment_codes, fragment_code_bytes) ||
      byte_ranges_overlap(canonical_scales, canonical_scale_bytes,
                          fragment_scales, fragment_scale_bytes) ||
      byte_ranges_overlap(fragment_codes, fragment_code_bytes,
                          fragment_scales, fragment_scale_bytes)) {
    return make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kInvalidOption, "down_permute",
        "capacities differ from equal-byte ABI or ranges overlap");
  }
  const std::size_t k512_groups = input_size / 512U;
  const std::size_t k64_groups = input_size / 64U;
  for (std::size_t panel = 0U; panel < output_size / 128U; ++panel) {
    for (std::size_t group = 0U; group < k512_groups; ++group) {
      for (std::size_t phase = 0U; phase < 8U; ++phase) {
        for (std::size_t warp = 0U; warp < 8U; ++warp) {
          for (std::size_t lane = 0U; lane < 32U; ++lane) {
            const std::size_t destination =
                kernels::sm87_a4w4_down_k512_fragment_b_vector_offset(
                    panel, group, phase, warp, lane, k512_groups);
            for (std::size_t word = 0U; word < 4U; ++word) {
              const auto coordinate =
                  kernels::sm87_a4w4_down_k512_fragment_b_word_coordinate(
                      warp, lane, word);
              const std::size_t source = canonical_code_offset(
                  panel * 128U + coordinate.n, group * 8U + phase,
                  coordinate.byte_in_k64, k64_groups);
              std::memcpy(fragment_codes + destination + word * 4U,
                          canonical_codes + source, 4U);
            }
          }
        }
      }
    }
  }
  // Down scale layout remains canonical by design.
  std::memcpy(fragment_scales, canonical_scales, canonical_scale_bytes);
  return {};
}

PrefillMLPK512OverlayDiagnostic
unpermute_prefill_mlp_k512_down_fragment_native(
    const std::uint8_t* const fragment_codes,
    const std::size_t fragment_code_bytes,
    const std::uint8_t* const fragment_scales,
    const std::size_t fragment_scale_bytes, const std::size_t output_size,
    const std::size_t input_size, std::uint8_t* const canonical_codes,
    const std::size_t canonical_code_bytes,
    std::uint8_t* const canonical_scales,
    const std::size_t canonical_scale_bytes) {
  if (canonical_codes == nullptr || canonical_scales == nullptr ||
      fragment_codes == nullptr || fragment_scales == nullptr ||
      output_size == 0U || output_size % 128U != 0U || input_size == 0U ||
      input_size % 512U != 0U || !product_fits(output_size, input_size)) {
    return make_diagnostic(PrefillMLPK512OverlayErrorCode::kInvalidOption,
                           "down_unpermute",
                           "complete N128/K512 buffers are required");
  }
  const std::size_t elements = output_size * input_size;
  const std::size_t expected_codes = elements / 2U;
  const std::size_t expected_scales = elements / 512U * 2U;
  if (canonical_code_bytes != expected_codes ||
      fragment_code_bytes != expected_codes ||
      canonical_scale_bytes != expected_scales ||
      fragment_scale_bytes != expected_scales ||
      byte_ranges_overlap(canonical_codes, canonical_code_bytes,
                          fragment_codes, fragment_code_bytes) ||
      byte_ranges_overlap(canonical_codes, canonical_code_bytes,
                          fragment_scales, fragment_scale_bytes) ||
      byte_ranges_overlap(canonical_scales, canonical_scale_bytes,
                          fragment_codes, fragment_code_bytes) ||
      byte_ranges_overlap(canonical_scales, canonical_scale_bytes,
                          fragment_scales, fragment_scale_bytes) ||
      byte_ranges_overlap(canonical_codes, canonical_code_bytes,
                          canonical_scales, canonical_scale_bytes)) {
    return make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kInvalidOption, "down_unpermute",
        "capacities differ from equal-byte ABI or ranges overlap");
  }
  const std::size_t k512_groups = input_size / 512U;
  const std::size_t k64_groups = input_size / 64U;
  for (std::size_t panel = 0U; panel < output_size / 128U; ++panel) {
    for (std::size_t group = 0U; group < k512_groups; ++group) {
      for (std::size_t phase = 0U; phase < 8U; ++phase) {
        for (std::size_t warp = 0U; warp < 8U; ++warp) {
          for (std::size_t lane = 0U; lane < 32U; ++lane) {
            const std::size_t source =
                kernels::sm87_a4w4_down_k512_fragment_b_vector_offset(
                    panel, group, phase, warp, lane, k512_groups);
            for (std::size_t word = 0U; word < 4U; ++word) {
              const auto coordinate =
                  kernels::sm87_a4w4_down_k512_fragment_b_word_coordinate(
                      warp, lane, word);
              const std::size_t destination = canonical_code_offset(
                  panel * 128U + coordinate.n, group * 8U + phase,
                  coordinate.byte_in_k64, k64_groups);
              std::memcpy(canonical_codes + destination,
                          fragment_codes + source + word * 4U, 4U);
            }
          }
        }
      }
    }
  }
  std::memcpy(canonical_scales, fragment_scales, fragment_scale_bytes);
  return {};
}

std::optional<PrefillMLPK512FragmentNativeReceipt>
parse_prefill_mlp_k512_fragment_native_receipt(
    const std::string_view document,
    PrefillMLPK512OverlayDiagnostic& diagnostic) {
  diagnostic = {};
  try {
    json::ParseOptions options;
    options.max_input_bytes = 1U * 1024U * 1024U;
    options.max_nesting_depth = 8U;
    options.max_values = 128U;
    options.max_container_items = 128U;
    const json::ParseResult parsed = json::parse(document, options);
    const auto* root = parsed ? parsed.value->as_object() : nullptr;
    if (root == nullptr ||
        !exact_keys(*root,
                    {"schema", "version", "mode",
                     "production_residency_eligible", "physical_layout",
                     "source_checkpoint_id", "source_config_sha256",
                     "source_index_sha256", "required_base", "source_v1",
                     "manifest_sha256", "payload_sha256", "payload_bytes",
                     "layer_count"})) {
      diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kInvalidReceipt, "receipt",
          "strict fragment-native receipt JSON schema mismatch");
      return std::nullopt;
    }
    PrefillMLPK512FragmentNativeReceipt receipt;
    std::string schema;
    std::string mode;
    const bool* eligible =
        root->at("production_residency_eligible").as_bool();
    if (!json_string(*root, "schema", schema) || schema != kReceiptSchema ||
        !parse_version(root->at("version"), receipt.version_major,
                       receipt.version_minor) ||
        receipt.version_major !=
            kPrefillMLPK512FragmentNativeVersionMajor ||
        receipt.version_minor !=
            kPrefillMLPK512FragmentNativeVersionMinor ||
        !json_string(*root, "mode", mode) ||
        mode != "lossless_permutation" || eligible == nullptr || !*eligible ||
        !json_string(*root, "physical_layout", receipt.physical_layout) ||
        receipt.physical_layout != kPrefillMLPK512FragmentNativeLayout ||
        !json_string(*root, "source_checkpoint_id",
                     receipt.source_checkpoint_id) ||
        receipt.source_checkpoint_id.empty() ||
        !json_string(*root, "source_config_sha256",
                     receipt.source_config_sha256) ||
        !lower_sha256(receipt.source_config_sha256) ||
        !json_string(*root, "source_index_sha256",
                     receipt.source_index_sha256) ||
        !lower_sha256(receipt.source_index_sha256) ||
        !parse_base(root->at("required_base"), receipt.required_base) ||
        !parse_source(root->at("source_v1"), receipt.source_v1) ||
        !json_string(*root, "manifest_sha256", receipt.manifest_sha256) ||
        !lower_sha256(receipt.manifest_sha256) ||
        !json_string(*root, "payload_sha256", receipt.payload_sha256) ||
        !lower_sha256(receipt.payload_sha256) ||
        !json_uint(*root, "payload_bytes", receipt.payload_bytes) ||
        receipt.payload_bytes != kPrefillMLPK512FragmentNativePayloadBytes ||
        !json_uint(*root, "layer_count", receipt.layer_count) ||
        receipt.layer_count != kPrefillMLPK512FragmentNativeLayerCount) {
      diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kInvalidReceipt, "receipt",
          "fragment-native receipt identity or fixed ABI is invalid");
      return std::nullopt;
    }
    PrefillMLPK512OverlayReceipt source;
    source.production_residency_eligible = true;
    source.physical_layout = receipt.source_v1.physical_layout;
    source.source_checkpoint_id = receipt.source_checkpoint_id;
    source.source_config_sha256 = receipt.source_config_sha256;
    source.source_index_sha256 = receipt.source_index_sha256;
    source.manifest_sha256 = receipt.source_v1.manifest_sha256;
    source.policy_sha256 = receipt.source_v1.policy_sha256;
    source.policy_bytes = receipt.source_v1.policy_bytes;
    source.required_base = receipt.required_base;
    source.payload_sha256 = receipt.source_v1.payload_sha256;
    source.payload_bytes = receipt.source_v1.payload_bytes;
    source.projection_count = kPrefillMLPK512OverlayProjectionCount;
    const auto manifest = build_prefill_mlp_k512_fragment_native_manifest(
        source, receipt.source_v1.receipt_sha256);
    if (!manifest || manifest.value->manifest_sha256 !=
                         receipt.manifest_sha256 ||
        !same_base(manifest.value->required_base, receipt.required_base)) {
      diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kInvalidReceipt,
          "receipt.manifest_sha256",
          "receipt does not bind the deterministic fragment-native manifest",
          manifest ? manifest.value->manifest_sha256 : std::string{},
          receipt.manifest_sha256);
      return std::nullopt;
    }
    receipt.production_residency_eligible = true;
    receipt.receipt_sha256 = sha256_text(document);
    return receipt;
  } catch (const std::bad_alloc&) {
    diagnostic = make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kAllocationFailure, "receipt",
        "receipt allocation failed");
    return std::nullopt;
  } catch (...) {
    diagnostic = make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kInvalidReceipt, "receipt",
        "unexpected fragment-native receipt parse failure");
    return std::nullopt;
  }
}

PrefillMLPK512OverlayDiagnostic
write_prefill_mlp_k512_fragment_native_receipt_no_replace(
    const PrefillMLPK512FragmentNativeReceipt& receipt,
    const fs::path& output_path) {
  if (receipt.version_major !=
          kPrefillMLPK512FragmentNativeVersionMajor ||
      receipt.version_minor !=
          kPrefillMLPK512FragmentNativeVersionMinor ||
      !receipt.production_residency_eligible ||
      receipt.physical_layout != kPrefillMLPK512FragmentNativeLayout ||
      receipt.source_checkpoint_id.empty() ||
      !lower_sha256(receipt.source_config_sha256) ||
      !lower_sha256(receipt.source_index_sha256) ||
      !valid_base(receipt.required_base) ||
      !valid_source_binding(receipt.source_v1) ||
      !lower_sha256(receipt.manifest_sha256) ||
      !lower_sha256(receipt.payload_sha256) ||
      receipt.payload_bytes != kPrefillMLPK512FragmentNativePayloadBytes ||
      receipt.layer_count != kPrefillMLPK512FragmentNativeLayerCount) {
    return make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kInvalidReceipt, "receipt",
        "receipt object is not eligible for publication");
  }
  const std::string document = serialize_receipt(receipt);
  PrefillMLPK512OverlayDiagnostic diagnostic;
  const auto parsed = parse_prefill_mlp_k512_fragment_native_receipt(
      document, diagnostic);
  if (!parsed || parsed->source_checkpoint_id != receipt.source_checkpoint_id ||
      parsed->source_config_sha256 != receipt.source_config_sha256 ||
      parsed->source_index_sha256 != receipt.source_index_sha256 ||
      !same_base(parsed->required_base, receipt.required_base) ||
      parsed->source_v1.receipt_sha256 != receipt.source_v1.receipt_sha256 ||
      parsed->source_v1.manifest_sha256 != receipt.source_v1.manifest_sha256 ||
      parsed->source_v1.policy_sha256 != receipt.source_v1.policy_sha256 ||
      parsed->source_v1.policy_bytes != receipt.source_v1.policy_bytes ||
      parsed->source_v1.payload_sha256 != receipt.source_v1.payload_sha256 ||
      parsed->manifest_sha256 != receipt.manifest_sha256 ||
      parsed->payload_sha256 != receipt.payload_sha256) {
    return diagnostic.ok()
               ? make_diagnostic(
                     PrefillMLPK512OverlayErrorCode::kInvalidReceipt,
                     "receipt", "receipt failed canonical round trip")
               : diagnostic;
  }
  return publish_document_no_replace(output_path, document);
}

PrefillMLPK512FragmentNativeConversionResult
convert_authenticated_prefill_mlp_k512_to_fragment_native(
    const PrefillMLPK512FragmentNativeConversionOptions& options) {
  PrefillMLPK512FragmentNativeConversionResult result;
  fs::path payload_temporary;
  fs::path receipt_temporary;
  struct Cleanup final {
    fs::path* payload;
    fs::path* receipt;
    ~Cleanup() {
      if (payload != nullptr && !payload->empty()) {
        (void)::unlink(payload->c_str());
      }
      if (receipt != nullptr && !receipt->empty()) {
        (void)::unlink(receipt->c_str());
      }
    }
  } cleanup{&payload_temporary, &receipt_temporary};

  try {
    if (options.source_v1_payload_path.empty() ||
        options.source_v1_receipt_path.empty() ||
        options.source_v1_policy_path.empty() || options.output_path.empty() ||
        !lower_sha256(options.expected_source_v1_receipt_sha256) ||
        options.outer_chunk_rows == 0U ||
        options.outer_chunk_rows % 512U != 0U ||
        options.outer_chunk_rows > 1'024U ||
        options.max_receipt_bytes == 0U || options.max_policy_bytes == 0U) {
      result.diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kInvalidOption, "convert",
          "authenticated v1 paths, pinned receipt SHA, output, and bounded "
          "N512 chunk are required");
      return result;
    }
    const fs::path output_receipt_path =
        fs::path(options.output_path.string() + ".receipt.json");
    struct stat path_status {};
    errno = 0;
    if (::lstat(options.output_path.c_str(), &path_status) == 0 ||
        errno != ENOENT) {
      result.diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kPublicationConflict,
          options.output_path.string(), "payload target already exists", {},
          {}, errno);
      return result;
    }
    errno = 0;
    if (::lstat(output_receipt_path.c_str(), &path_status) == 0 ||
        errno != ENOENT) {
      result.diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kPublicationConflict,
          output_receipt_path.string(), "receipt target already exists", {},
          {}, errno);
      return result;
    }

    std::string source_receipt_document;
    result.diagnostic = read_bounded_file(
        options.source_v1_receipt_path, options.max_receipt_bytes,
        source_receipt_document, &result.stats.source_bytes_read);
    if (!result.diagnostic) {
      return result;
    }
    const std::string source_receipt_sha256 =
        sha256_text(source_receipt_document);
    if (source_receipt_sha256 !=
        options.expected_source_v1_receipt_sha256) {
      result.diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kDigestMismatch,
          options.source_v1_receipt_path.string(),
          "v1 receipt digest differs from the required trust anchor",
          options.expected_source_v1_receipt_sha256, source_receipt_sha256);
      return result;
    }
    PrefillMLPK512OverlayDiagnostic source_receipt_diagnostic;
    const auto source_receipt = parse_prefill_mlp_k512_overlay_receipt(
        source_receipt_document, source_receipt_diagnostic);
    if (!source_receipt) {
      result.diagnostic = source_receipt_diagnostic;
      return result;
    }
    const auto manifest = build_prefill_mlp_k512_fragment_native_manifest(
        *source_receipt, source_receipt_sha256);
    if (!manifest) {
      result.diagnostic = manifest.diagnostic;
      return result;
    }

    UniqueFd policy(::open(options.source_v1_policy_path.c_str(),
                           O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
    FileSnapshot policy_snapshot;
    int error = 0;
    if (!policy ||
        !snapshot_fd(policy.get(), policy_snapshot, error) ||
        policy_snapshot.size < 0 ||
        static_cast<std::uint64_t>(policy_snapshot.size) !=
            source_receipt->policy_bytes ||
        static_cast<std::uint64_t>(policy_snapshot.size) >
            options.max_policy_bytes) {
      result.diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kSourceAuthenticationFailed,
          options.source_v1_policy_path.string(),
          "v1 policy is not the exact bounded receipt-bound file", {}, {},
          error);
      return result;
    }
    std::string policy_sha256;
    result.diagnostic = hash_fd(
        policy.get(), static_cast<std::uint64_t>(policy_snapshot.size),
        policy_sha256, &result.stats.source_bytes_read);
    if (!result.diagnostic) {
      return result;
    }
    FileSnapshot policy_after;
    if (policy_sha256 != source_receipt->policy_sha256 ||
        !snapshot_fd(policy.get(), policy_after, error) ||
        !same_snapshot(policy_snapshot, policy_after)) {
      result.diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kSourceAuthenticationFailed,
          options.source_v1_policy_path.string(),
          "v1 policy digest or immutable snapshot differs from receipt",
          source_receipt->policy_sha256, policy_sha256, error);
      return result;
    }

    UniqueFd source_payload(::open(options.source_v1_payload_path.c_str(),
                                   O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
    FileSnapshot source_snapshot;
    if (!source_payload ||
        !snapshot_fd(source_payload.get(), source_snapshot, error) ||
        source_snapshot.size < 0 ||
        static_cast<std::uint64_t>(source_snapshot.size) !=
            kPrefillMLPK512OverlayPayloadBytes) {
      result.diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kSourceAuthenticationFailed,
          options.source_v1_payload_path.string(),
          "v1 payload is not the exact regular-file ABI", {}, {}, error);
      return result;
    }
    std::string authenticated_source_payload_sha256;
    result.diagnostic = hash_fd(
        source_payload.get(), kPrefillMLPK512OverlayPayloadBytes,
        authenticated_source_payload_sha256, &result.stats.source_bytes_read);
    if (!result.diagnostic) {
      return result;
    }
    FileSnapshot source_after_hash;
    if (authenticated_source_payload_sha256 !=
            source_receipt->payload_sha256 ||
        !snapshot_fd(source_payload.get(), source_after_hash, error) ||
        !same_snapshot(source_snapshot, source_after_hash)) {
      result.diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kSourceAuthenticationFailed,
          options.source_v1_payload_path.string(),
          "v1 payload digest or immutable snapshot differs from receipt",
          source_receipt->payload_sha256,
          authenticated_source_payload_sha256, error);
      return result;
    }

    const fs::path parent = options.output_path.parent_path().empty()
                                ? fs::path(".")
                                : options.output_path.parent_path();
    payload_temporary = fs::path(options.output_path.string() + ".tmp." +
                                 std::to_string(::getpid()));
    receipt_temporary =
        fs::path(output_receipt_path.string() + ".tmp." +
                 std::to_string(::getpid()));
    UniqueFd output(::open(payload_temporary.c_str(),
                           O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
                           S_IRUSR | S_IWUSR));
    if (!output) {
      result.diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kIoFailure,
          payload_temporary.string(), "failed to create temporary payload",
          {}, {}, errno);
      return result;
    }
    if (kPrefillMLPK512FragmentNativePayloadBytes >
            static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) ||
        ::ftruncate(
            output.get(),
            static_cast<off_t>(kPrefillMLPK512FragmentNativePayloadBytes)) !=
            0) {
      result.diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kIoFailure,
          payload_temporary.string(), "failed to size temporary payload", {},
          {}, errno);
      return result;
    }
    if (options.preallocate_output) {
      const int allocation_error = ::posix_fallocate(
          output.get(), 0,
          static_cast<off_t>(kPrefillMLPK512FragmentNativePayloadBytes));
      if (allocation_error != 0) {
        result.diagnostic = make_diagnostic(
            PrefillMLPK512OverlayErrorCode::kIoFailure,
            payload_temporary.string(), "payload preallocation failed", {},
            {}, allocation_error);
        return result;
      }
    }

    auto read_payload = [&](void* destination, const std::size_t bytes,
                            const std::uint64_t offset,
                            const std::string_view context) -> bool {
      int io_error = 0;
      if (!pread_exact(source_payload.get(), destination, bytes, offset,
                       io_error)) {
        result.diagnostic = make_diagnostic(
            PrefillMLPK512OverlayErrorCode::kIoFailure,
            std::string(context), "v1 payload chunk read failed", {}, {},
            io_error);
        return false;
      }
      result.stats.source_bytes_read += bytes;
      return true;
    };
    auto write_payload = [&](const void* source, const std::size_t bytes,
                             const std::uint64_t offset,
                             const std::string_view context) -> bool {
      int io_error = 0;
      if (!pwrite_exact(output.get(), source, bytes, offset, io_error)) {
        result.diagnostic = make_diagnostic(
            PrefillMLPK512OverlayErrorCode::kIoFailure,
            std::string(context), "v2 payload chunk write failed", {}, {},
            io_error);
        return false;
      }
      result.stats.output_bytes_written += bytes;
      return true;
    };

    const std::size_t gateup_rows = options.outer_chunk_rows;
    const std::size_t gateup_k =
        static_cast<std::size_t>(kPrefillMLPK512OverlayGateUpInputSize);
    const std::size_t gateup_projection_code_bytes =
        gateup_rows * gateup_k / 2U;
    const std::size_t gateup_projection_scale_bytes =
        gateup_rows * (gateup_k / 512U) * 2U;
    std::vector<std::uint8_t> gate_codes(gateup_projection_code_bytes);
    std::vector<std::uint8_t> gate_scales(gateup_projection_scale_bytes);
    std::vector<std::uint8_t> up_codes(gateup_projection_code_bytes);
    std::vector<std::uint8_t> up_scales(gateup_projection_scale_bytes);
    std::vector<std::uint8_t> paired_codes(
        2U * gateup_projection_code_bytes);
    std::vector<std::uint8_t> paired_scales(
        2U * gateup_projection_scale_bytes);
    const std::uint64_t gate_peak =
        gate_codes.capacity() + gate_scales.capacity() + up_codes.capacity() +
        up_scales.capacity() + paired_codes.capacity() +
        paired_scales.capacity();
    result.stats.peak_working_bytes =
        std::max(result.stats.peak_working_bytes, gate_peak);

    const std::size_t down_rows = options.outer_chunk_rows;
    const std::size_t down_k =
        static_cast<std::size_t>(kPrefillMLPK512OverlayDownInputSize);
    const std::size_t down_code_bytes = down_rows * down_k / 2U;
    const std::size_t down_scale_bytes =
        down_rows * (down_k / 512U) * 2U;
    std::vector<std::uint8_t> canonical_down_codes(down_code_bytes);
    std::vector<std::uint8_t> canonical_down_scales(down_scale_bytes);
    std::vector<std::uint8_t> fragment_down_codes(down_code_bytes);
    std::vector<std::uint8_t> fragment_down_scales(down_scale_bytes);
    const std::uint64_t total_peak =
        gate_peak + canonical_down_codes.capacity() +
        canonical_down_scales.capacity() + fragment_down_codes.capacity() +
        fragment_down_scales.capacity();
    result.stats.peak_working_bytes =
        std::max(result.stats.peak_working_bytes, total_peak);

    for (std::size_t layer = 0U;
         layer < kPrefillMLPK512FragmentNativeLayerCount; ++layer) {
      const std::uint64_t gate_v1_offset =
          (3U * layer) * kPrefillMLPK512OverlayProjectionBytes;
      const std::uint64_t up_v1_offset =
          gate_v1_offset + kPrefillMLPK512OverlayProjectionBytes;
      const std::uint64_t down_v1_offset =
          up_v1_offset + kPrefillMLPK512OverlayProjectionBytes;
      const auto view = prefill_mlp_k512_fragment_native_layer_view(layer);

      for (std::size_t row = 0U;
           row < kPrefillMLPK512OverlayGateUpOutputSize;
           row += gateup_rows) {
        const std::uint64_t source_code_delta = row * gateup_k / 2U;
        const std::uint64_t source_scale_delta =
            row * (gateup_k / 512U) * 2U;
        if (!read_payload(gate_codes.data(), gate_codes.size(),
                          gate_v1_offset + source_code_delta,
                          "gate_codes") ||
            !read_payload(gate_scales.data(), gate_scales.size(),
                          gate_v1_offset +
                              kPrefillMLPK512OverlayProjectionWeightBytes +
                              source_scale_delta,
                          "gate_scales") ||
            !read_payload(up_codes.data(), up_codes.size(),
                          up_v1_offset + source_code_delta, "up_codes") ||
            !read_payload(up_scales.data(), up_scales.size(),
                          up_v1_offset +
                              kPrefillMLPK512OverlayProjectionWeightBytes +
                              source_scale_delta,
                          "up_scales")) {
          return result;
        }
        result.diagnostic = permute_prefill_mlp_k512_gateup_fragment_native(
            gate_codes.data(), gate_codes.size(), gate_scales.data(),
            gate_scales.size(), up_codes.data(), up_codes.size(),
            up_scales.data(), up_scales.size(), gateup_rows, gateup_k,
            paired_codes.data(), paired_codes.size(), paired_scales.data(),
            paired_scales.size());
        if (!result.diagnostic) {
          result.diagnostic.context =
              "layer[" + std::to_string(layer) + "]:" +
              result.diagnostic.context;
          return result;
        }
        if (!write_payload(paired_codes.data(), paired_codes.size(),
                           view.gateup_code_offset + row * gateup_k,
                           "paired_gateup_codes") ||
            !write_payload(paired_scales.data(), paired_scales.size(),
                           view.gateup_scale_offset +
                               row * (gateup_k / 512U) * 4U,
                           "paired_gateup_scales")) {
          return result;
        }
      }

      for (std::size_t row = 0U;
           row < kPrefillMLPK512OverlayDownOutputSize; row += down_rows) {
        const std::uint64_t source_code_delta = row * down_k / 2U;
        const std::uint64_t source_scale_delta =
            row * (down_k / 512U) * 2U;
        if (!read_payload(canonical_down_codes.data(),
                          canonical_down_codes.size(),
                          down_v1_offset + source_code_delta,
                          "down_codes") ||
            !read_payload(canonical_down_scales.data(),
                          canonical_down_scales.size(),
                          down_v1_offset +
                              kPrefillMLPK512OverlayProjectionWeightBytes +
                              source_scale_delta,
                          "down_scales")) {
          return result;
        }
        result.diagnostic = permute_prefill_mlp_k512_down_fragment_native(
            canonical_down_codes.data(), canonical_down_codes.size(),
            canonical_down_scales.data(), canonical_down_scales.size(),
            down_rows, down_k, fragment_down_codes.data(),
            fragment_down_codes.size(), fragment_down_scales.data(),
            fragment_down_scales.size());
        if (!result.diagnostic) {
          result.diagnostic.context =
              "layer[" + std::to_string(layer) + "]:" +
              result.diagnostic.context;
          return result;
        }
        if (!write_payload(fragment_down_codes.data(),
                           fragment_down_codes.size(),
                           view.down_code_offset + source_code_delta,
                           "fragment_down_codes") ||
            !write_payload(fragment_down_scales.data(),
                           fragment_down_scales.size(),
                           view.down_scale_offset + source_scale_delta,
                           "fragment_down_scales")) {
          return result;
        }
      }
      ++result.stats.layers_permuted;
    }

    if (result.stats.layers_permuted !=
            kPrefillMLPK512FragmentNativeLayerCount ||
        result.stats.output_bytes_written !=
            kPrefillMLPK512FragmentNativePayloadBytes) {
      result.diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kIoFailure,
          payload_temporary.string(),
          "permutation did not write the complete 64-layer payload");
      return result;
    }
    if (::fsync(output.get()) != 0) {
      result.diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kIoFailure,
          payload_temporary.string(), "payload fsync failed", {}, {}, errno);
      return result;
    }

    std::string final_source_payload_sha256;
    result.diagnostic = hash_fd(
        source_payload.get(), kPrefillMLPK512OverlayPayloadBytes,
        final_source_payload_sha256, &result.stats.source_bytes_read);
    if (!result.diagnostic) {
      return result;
    }
    FileSnapshot source_final;
    if (final_source_payload_sha256 != source_receipt->payload_sha256 ||
        !snapshot_fd(source_payload.get(), source_final, error) ||
        !same_snapshot(source_snapshot, source_final)) {
      result.diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kSourceAuthenticationFailed,
          options.source_v1_payload_path.string(),
          "v1 payload changed during permutation",
          source_receipt->payload_sha256, final_source_payload_sha256, error);
      return result;
    }

    std::string output_payload_sha256;
    result.diagnostic = hash_fd(
        output.get(), kPrefillMLPK512FragmentNativePayloadBytes,
        output_payload_sha256);
    if (!result.diagnostic) {
      return result;
    }
    if (::fchmod(output.get(), S_IRUSR) != 0 || ::fsync(output.get()) != 0) {
      result.diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kIoFailure,
          payload_temporary.string(), "failed to seal payload read-only", {},
          {}, errno);
      return result;
    }

    PrefillMLPK512FragmentNativeReceipt receipt;
    receipt.production_residency_eligible = true;
    receipt.physical_layout = manifest.value->physical_layout;
    receipt.source_checkpoint_id = manifest.value->source_checkpoint_id;
    receipt.source_config_sha256 = manifest.value->source_config_sha256;
    receipt.source_index_sha256 = manifest.value->source_index_sha256;
    receipt.required_base = manifest.value->required_base;
    receipt.source_v1 = manifest.value->source_v1;
    receipt.manifest_sha256 = manifest.value->manifest_sha256;
    receipt.payload_sha256 = output_payload_sha256;
    receipt.payload_bytes = kPrefillMLPK512FragmentNativePayloadBytes;
    receipt.layer_count = kPrefillMLPK512FragmentNativeLayerCount;
    const std::string receipt_document = serialize_receipt(receipt);
    PrefillMLPK512OverlayDiagnostic receipt_diagnostic;
    const auto parsed_receipt =
        parse_prefill_mlp_k512_fragment_native_receipt(receipt_document,
                                                       receipt_diagnostic);
    if (!parsed_receipt) {
      result.diagnostic = receipt_diagnostic;
      return result;
    }
    UniqueFd receipt_output(::open(
        receipt_temporary.c_str(),
        O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
        S_IRUSR | S_IWUSR));
    if (!receipt_output) {
      result.diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kIoFailure,
          receipt_temporary.string(), "failed to create temporary receipt",
          {}, {}, errno);
      return result;
    }
    int receipt_error = 0;
    if (!pwrite_exact(receipt_output.get(), receipt_document.data(),
                      receipt_document.size(), 0U, receipt_error) ||
        ::fchmod(receipt_output.get(), S_IRUSR) != 0 ||
        ::fsync(receipt_output.get()) != 0) {
      result.diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kIoFailure,
          receipt_temporary.string(), "failed to seal temporary receipt", {},
          {}, receipt_error != 0 ? receipt_error : errno);
      return result;
    }

    if (::link(payload_temporary.c_str(), options.output_path.c_str()) != 0) {
      result.diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kPublicationConflict,
          options.output_path.string(), "payload no-replace link failed", {},
          {}, errno);
      return result;
    }
    if (::link(receipt_temporary.c_str(), output_receipt_path.c_str()) != 0) {
      const int saved = errno;
      (void)::unlink(options.output_path.c_str());
      result.diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kPublicationConflict,
          output_receipt_path.string(),
          "receipt no-replace link failed; rolled back", {}, {}, saved);
      return result;
    }
    if (::unlink(payload_temporary.c_str()) != 0 ||
        ::unlink(receipt_temporary.c_str()) != 0) {
      const int saved = errno;
      (void)::unlink(output_receipt_path.c_str());
      (void)::unlink(options.output_path.c_str());
      result.diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kIoFailure, parent.string(),
          "temporary cleanup failed; publication rolled back", {}, {}, saved);
      return result;
    }
    payload_temporary.clear();
    receipt_temporary.clear();
    UniqueFd directory(::open(parent.c_str(), O_RDONLY | O_DIRECTORY |
                                                  O_CLOEXEC | O_NOFOLLOW));
    if (!directory || ::fsync(directory.get()) != 0) {
      const int saved = errno;
      (void)::unlink(output_receipt_path.c_str());
      (void)::unlink(options.output_path.c_str());
      result.diagnostic = make_diagnostic(
          PrefillMLPK512OverlayErrorCode::kIoFailure, parent.string(),
          "publication directory sync failed; rolled back", {}, {}, saved);
      return result;
    }
    result.receipt.emplace(*parsed_receipt);
    result.diagnostic = {};
    return result;
  } catch (const std::bad_alloc&) {
    result.diagnostic = make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kAllocationFailure, "convert",
        "fragment-native conversion allocation failed");
    return result;
  } catch (...) {
    result.diagnostic = make_diagnostic(
        PrefillMLPK512OverlayErrorCode::kIoFailure, "convert",
        "unexpected fragment-native conversion failure");
    return result;
  }
}

}  // namespace q3x::runtime
