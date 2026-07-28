#include "pinned_checkpoint.h"

#include "q3x/core/sha256.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <new>
#include <set>
#include <sstream>
#include <system_error>
#include <utility>

namespace q3x::test::support {
namespace {

namespace fs = std::filesystem;
namespace st = q3x::io::safetensors;

constexpr std::string_view kQwen36Index =
    "model.safetensors.index.json";
constexpr std::string_view kQwen36Shard1 =
    "model-00001-of-00003.safetensors";

struct ResolvedFile {
  fs::path path;
  std::uint64_t bytes = 0U;
};

[[nodiscard]] PinnedBundleLoadResult fail(
    const PinnedCheckpointErrorCode code, std::string stage,
    std::string message, std::string path = {}, std::string tensor = {},
    std::string expected = {}, std::string actual = {}) {
  PinnedCheckpointError error;
  error.code = code;
  error.stage = std::move(stage);
  error.path = std::move(path);
  error.tensor = std::move(tensor);
  error.expected = std::move(expected);
  error.actual = std::move(actual);
  error.message = std::move(message);
  return PinnedBundleLoadResult{std::nullopt, std::move(error)};
}

[[nodiscard]] bool is_lower_hex_sha256(const std::string_view digest) {
  return digest.size() == 64U &&
         std::all_of(digest.begin(), digest.end(), [](const char byte) {
           return (byte >= '0' && byte <= '9') ||
                  (byte >= 'a' && byte <= 'f');
         });
}

[[nodiscard]] bool is_safe_relative_file_path(
    const std::string_view path) {
  if (path.empty() || path.front() == '/' || path.back() == '/' ||
      path.find('\\') != std::string_view::npos ||
      path.find("//") != std::string_view::npos) {
    return false;
  }
  for (const char byte : path) {
    if (static_cast<unsigned char>(byte) < 0x20U) {
      return false;
    }
  }
  const fs::path candidate(path);
  if (candidate.is_absolute() || candidate.has_root_name() ||
      candidate.has_root_directory()) {
    return false;
  }
  for (const fs::path& component : candidate) {
    if (component.empty() || component == "." || component == "..") {
      return false;
    }
  }
  return candidate.generic_string() == path;
}

[[nodiscard]] bool path_is_strictly_within(
    const fs::path& directory, const fs::path& candidate) noexcept {
  auto directory_component = directory.begin();
  auto candidate_component = candidate.begin();
  while (directory_component != directory.end() &&
         candidate_component != candidate.end()) {
    if (*directory_component != *candidate_component) {
      return false;
    }
    ++directory_component;
    ++candidate_component;
  }
  return directory_component == directory.end() &&
         candidate_component != candidate.end();
}

[[nodiscard]] std::string describe_safetensors_error(
    const st::Error& error) {
  std::string description =
      std::string(st::to_string(error.code)) + ": " +
      std::string(error.message());
  if (error.offset != st::kUnknownOffset) {
    description += " offset=" + std::to_string(error.offset);
  }
  if (!error.context.empty()) {
    description += " context=" + error.context;
  }
  if (!error.expected.empty()) {
    description += " expected=" + error.expected;
  }
  if (!error.actual.empty()) {
    description += " actual=" + error.actual;
  }
  return description;
}

[[nodiscard]] std::string shape_string(
    const std::vector<std::uint64_t>& shape) {
  std::ostringstream output;
  output << '[';
  for (std::size_t index = 0U; index < shape.size(); ++index) {
    if (index != 0U) {
      output << ',';
    }
    output << shape[index];
  }
  output << ']';
  return output.str();
}

[[nodiscard]] std::string tensor_contract_string(
    const PinnedTensor& tensor) {
  return "dtype=" + std::string(st::to_string(tensor.dtype)) +
         " shape=" + shape_string(tensor.shape) + " range=" +
         std::to_string(tensor.file_begin) + ':' +
         std::to_string(tensor.file_end);
}

[[nodiscard]] std::string tensor_contract_string(
    const st::TensorInfo& tensor) {
  return "dtype=" + std::string(st::to_string(tensor.dtype)) +
         " shape=" + shape_string(tensor.shape) + " range=" +
         std::to_string(tensor.file_begin) + ':' +
         std::to_string(tensor.file_end);
}

[[nodiscard]] const PinnedShard* find_shard(
    const PinnedModelRevision& model, const std::string_view path) noexcept {
  const auto found =
      std::find_if(model.shards.begin(), model.shards.end(),
                   [path](const PinnedShard& shard) {
                     return shard.relative_path == path;
                   });
  return found == model.shards.end() ? nullptr : &*found;
}

[[nodiscard]] std::optional<PinnedCheckpointError> resolve_regular_file(
    const fs::path& directory, const std::string& relative_path,
    const std::string& stage, ResolvedFile& resolved) {
  if (!is_safe_relative_file_path(relative_path)) {
    return PinnedCheckpointError{
        PinnedCheckpointErrorCode::kInvalidDescriptor,
        stage,
        relative_path,
        {},
        "safe relative path",
        relative_path,
        "pinned file path is not a safe canonical relative path"};
  }

  std::error_code filesystem_error;
  const fs::path requested = directory / fs::path(relative_path);
  const fs::file_status status =
      fs::symlink_status(requested, filesystem_error);
  if (filesystem_error || !fs::exists(status)) {
    return PinnedCheckpointError{
        PinnedCheckpointErrorCode::kPathMissing,
        stage,
        requested.string(),
        {},
        "existing regular non-symlink file",
        filesystem_error ? filesystem_error.message() : "missing",
        "pinned file does not exist"};
  }
  if (fs::is_symlink(status)) {
    return PinnedCheckpointError{
        PinnedCheckpointErrorCode::kSymlinkRejected,
        stage,
        requested.string(),
        {},
        "regular non-symlink file",
        "symlink",
        "pinned file must not be a symlink"};
  }
  if (!fs::is_regular_file(status)) {
    return PinnedCheckpointError{
        PinnedCheckpointErrorCode::kPathTypeMismatch,
        stage,
        requested.string(),
        {},
        "regular file",
        "non-regular file",
        "pinned path is not a regular file"};
  }

  const fs::path canonical = fs::canonical(requested, filesystem_error);
  if (filesystem_error || canonical.empty()) {
    return PinnedCheckpointError{
        PinnedCheckpointErrorCode::kCanonicalizationFailed,
        stage,
        requested.string(),
        {},
        "canonical file path",
        filesystem_error.message(),
        "failed to canonicalize pinned file"};
  }
  const fs::path expected = (directory / fs::path(relative_path)).lexically_normal();
  if (!path_is_strictly_within(directory, canonical) || canonical != expected) {
    return PinnedCheckpointError{
        PinnedCheckpointErrorCode::kPathEscape,
        stage,
        requested.string(),
        {},
        expected.string(),
        canonical.string(),
        "pinned file canonical path escaped or changed spelling"};
  }

  const std::uintmax_t observed_bytes =
      fs::file_size(canonical, filesystem_error);
  if (filesystem_error ||
      observed_bytes > std::numeric_limits<std::uint64_t>::max()) {
    return PinnedCheckpointError{
        PinnedCheckpointErrorCode::kFileSizeMismatch,
        stage,
        canonical.string(),
        {},
        "representable file size",
        filesystem_error ? filesystem_error.message()
                         : std::to_string(observed_bytes),
        "failed to obtain a representable pinned file size"};
  }
  resolved.path = canonical;
  resolved.bytes = static_cast<std::uint64_t>(observed_bytes);
  return std::nullopt;
}

[[nodiscard]] std::optional<PinnedCheckpointError> validate_descriptor(
    const PinnedBundleDescriptor& descriptor) {
  if (descriptor.id.empty() || descriptor.model == nullptr ||
      descriptor.model->id.empty() || descriptor.model->repository.empty() ||
      descriptor.model->revision.empty() ||
      descriptor.model->index_file.empty() || descriptor.tensors.empty()) {
    return PinnedCheckpointError{
        PinnedCheckpointErrorCode::kInvalidDescriptor,
        "descriptor",
        {},
        {},
        "non-empty bundle/model/revision/index/tensor contract",
        descriptor.id,
        "pinned bundle descriptor is incomplete"};
  }

  const PinnedModelRevision& model = *descriptor.model;
  std::set<std::string, std::less<>> metadata_names;
  bool index_pinned = false;
  for (const PinnedFile& file : model.metadata_files) {
    if (!is_safe_relative_file_path(file.relative_path) ||
        !is_lower_hex_sha256(file.sha256) ||
        !metadata_names.insert(file.relative_path).second) {
      return PinnedCheckpointError{
          PinnedCheckpointErrorCode::kInvalidDescriptor,
          "descriptor.metadata",
          file.relative_path,
          {},
          "unique safe path and lowercase SHA256",
          file.sha256,
          "invalid pinned metadata file descriptor"};
    }
    index_pinned = index_pinned || file.relative_path == model.index_file;
  }
  if (!index_pinned) {
    return PinnedCheckpointError{
        PinnedCheckpointErrorCode::kInvalidDescriptor,
        "descriptor.index",
        model.index_file,
        {},
        "index file present in metadata SHA pins",
        "missing",
        "checkpoint index must have an exact SHA pin"};
  }

  std::set<std::string, std::less<>> shard_names;
  for (const PinnedShard& shard : model.shards) {
    if (!st::is_safe_relative_shard_path(shard.relative_path) ||
        shard.file_bytes == 0U || shard.data_offset < 8U ||
        shard.header_bytes + 8U != shard.data_offset ||
        !is_lower_hex_sha256(shard.sha256) ||
        !shard_names.insert(shard.relative_path).second) {
      return PinnedCheckpointError{
          PinnedCheckpointErrorCode::kInvalidDescriptor,
          "descriptor.shard",
          shard.relative_path,
          {},
          "unique safe shard with size/header/data-offset/SHA pins",
          shard.sha256,
          "invalid pinned shard descriptor"};
    }
  }

  std::set<std::string, std::less<>> tensor_names;
  std::string single_shard;
  for (const PinnedTensor& tensor : descriptor.tensors) {
    const PinnedShard* const shard = find_shard(model, tensor.shard);
    const bool range_valid = tensor.file_begin <= tensor.file_end &&
                             tensor.file_end <=
                                 (shard == nullptr ? 0U : shard->file_bytes) &&
                             tensor.file_end != tensor.file_begin;
    const bool scalar_pin_valid =
        !tensor.exact_f32_bits.has_value() ||
        (tensor.dtype == st::DType::kF32 && tensor.shape.empty() &&
         tensor.file_end - tensor.file_begin == sizeof(float));
    if (tensor.name.empty() || shard == nullptr || !range_valid ||
        !is_lower_hex_sha256(tensor.sha256) || !scalar_pin_valid ||
        !tensor_names.insert(tensor.name).second) {
      return PinnedCheckpointError{
          PinnedCheckpointErrorCode::kInvalidDescriptor,
          "descriptor.tensor",
          tensor.shard,
          tensor.name,
          "unique tensor with known shard/range/SHA/scalar contract",
          tensor_contract_string(tensor),
          "invalid pinned tensor descriptor"};
    }
    if (descriptor.require_single_shard) {
      if (single_shard.empty()) {
        single_shard = tensor.shard;
      } else if (single_shard != tensor.shard) {
        return PinnedCheckpointError{
            PinnedCheckpointErrorCode::kInvalidDescriptor,
            "descriptor.tensor",
            tensor.shard,
            tensor.name,
            single_shard,
            tensor.shard,
            "bundle requires every tensor to share one shard"};
      }
    }
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<PinnedCheckpointError> validate_directory(
    const fs::path& requested, fs::path& canonical) {
  std::error_code filesystem_error;
  const fs::file_status status =
      fs::symlink_status(requested, filesystem_error);
  if (filesystem_error || !fs::exists(status)) {
    return PinnedCheckpointError{
        PinnedCheckpointErrorCode::kPathMissing,
        "checkpoint.directory",
        requested.string(),
        {},
        "existing non-symlink directory",
        filesystem_error ? filesystem_error.message() : "missing",
        "checkpoint directory does not exist"};
  }
  if (fs::is_symlink(status)) {
    return PinnedCheckpointError{
        PinnedCheckpointErrorCode::kSymlinkRejected,
        "checkpoint.directory",
        requested.string(),
        {},
        "non-symlink directory",
        "symlink",
        "checkpoint directory must not be a symlink"};
  }
  if (!fs::is_directory(status)) {
    return PinnedCheckpointError{
        PinnedCheckpointErrorCode::kPathTypeMismatch,
        "checkpoint.directory",
        requested.string(),
        {},
        "directory",
        "non-directory",
        "checkpoint path is not a directory"};
  }
  canonical = fs::canonical(requested, filesystem_error);
  if (filesystem_error || canonical.empty()) {
    return PinnedCheckpointError{
        PinnedCheckpointErrorCode::kCanonicalizationFailed,
        "checkpoint.directory",
        requested.string(),
        {},
        "canonical directory",
        filesystem_error.message(),
        "failed to canonicalize checkpoint directory"};
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<PinnedCheckpointError> verify_file_hash(
    const ResolvedFile& file, const std::string& expected,
    const std::string& stage, std::string& observed) {
  const core::Sha256FileResult hash = core::sha256_file(file.path.string());
  if (!hash.ok()) {
    return PinnedCheckpointError{
        PinnedCheckpointErrorCode::kFileHashFailure,
        stage,
        file.path.string(),
        {},
        expected,
        hash.error,
        "failed to hash pinned file through a read-only stream"};
  }
  observed = hash.digest->hex();
  if (observed != expected) {
    return PinnedCheckpointError{
        PinnedCheckpointErrorCode::kFileHashMismatch,
        stage,
        file.path.string(),
        {},
        expected,
        observed,
        "pinned file SHA256 mismatch"};
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<PinnedCheckpointError> check_stable_size(
    const ResolvedFile& file, const std::string& stage) {
  std::error_code filesystem_error;
  const std::uintmax_t current = fs::file_size(file.path, filesystem_error);
  if (filesystem_error || current != file.bytes) {
    return PinnedCheckpointError{
        PinnedCheckpointErrorCode::kFileSizeMismatch,
        stage,
        file.path.string(),
        {},
        std::to_string(file.bytes),
        filesystem_error ? filesystem_error.message()
                         : std::to_string(current),
        "pinned file size changed while it was being inspected"};
  }
  return std::nullopt;
}

[[nodiscard]] std::uint32_t decode_little_endian_u32(
    const std::uint8_t* const bytes) noexcept {
  return static_cast<std::uint32_t>(bytes[0]) |
         (static_cast<std::uint32_t>(bytes[1]) << 8U) |
         (static_cast<std::uint32_t>(bytes[2]) << 16U) |
         (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

[[nodiscard]] PinnedBundleLoadResult load_impl(
    const fs::path& checkpoint_directory,
    const PinnedBundleDescriptor& descriptor,
    const PinnedBundleLoadOptions& options) {
  if (const auto error = validate_descriptor(descriptor)) {
    return PinnedBundleLoadResult{std::nullopt, *error};
  }
  if (options.maximum_total_payload_bytes == 0U) {
    return fail(PinnedCheckpointErrorCode::kInvalidRequest,
                "request", "maximum payload byte limit must be nonzero");
  }

  fs::path directory;
  if (const auto error = validate_directory(checkpoint_directory, directory)) {
    return PinnedBundleLoadResult{std::nullopt, *error};
  }

  std::vector<const PinnedTensor*> selected;
  std::set<std::string, std::less<>> selected_names;
  if (options.tensor_names.empty()) {
    selected.reserve(descriptor.tensors.size());
    for (const PinnedTensor& tensor : descriptor.tensors) {
      selected.push_back(&tensor);
      selected_names.insert(tensor.name);
    }
  } else {
    selected.reserve(options.tensor_names.size());
    for (const std::string& name : options.tensor_names) {
      const PinnedTensor* const tensor = find_pinned_tensor(descriptor, name);
      if (tensor == nullptr) {
        return fail(PinnedCheckpointErrorCode::kInvalidRequest,
                    "request.tensor", "requested tensor is not pinned",
                    {}, name, "tensor present in bundle descriptor", "missing");
      }
      if (!selected_names.insert(name).second) {
        return fail(PinnedCheckpointErrorCode::kInvalidRequest,
                    "request.tensor", "requested tensor is duplicated",
                    {}, name, "unique requested tensor", "duplicate");
      }
      selected.push_back(tensor);
    }
  }

  std::uint64_t total_payload_bytes = 0U;
  for (const PinnedTensor* const tensor : selected) {
    const std::uint64_t bytes = tensor->file_end - tensor->file_begin;
    if (bytes > options.maximum_total_payload_bytes - total_payload_bytes) {
      return fail(PinnedCheckpointErrorCode::kInvalidRequest,
                  "request.payload_bytes",
                  "selected payloads exceed the configured byte limit", {},
                  tensor->name,
                  std::to_string(options.maximum_total_payload_bytes),
                  std::to_string(total_payload_bytes + bytes));
    }
    total_payload_bytes += bytes;
  }

  LoadedPinnedBundle loaded;
  loaded.descriptor_id = descriptor.id;
  loaded.model_id = descriptor.model->id;
  loaded.repository = descriptor.model->repository;
  loaded.revision = descriptor.model->revision;
  loaded.canonical_directory = directory.string();

  ResolvedFile index_file;
  for (const PinnedFile& pin : descriptor.model->metadata_files) {
    ResolvedFile file;
    if (const auto error = resolve_regular_file(
            directory, pin.relative_path, "metadata.resolve", file)) {
      return PinnedBundleLoadResult{std::nullopt, *error};
    }
    if (pin.file_bytes != 0U && file.bytes != pin.file_bytes) {
      return fail(PinnedCheckpointErrorCode::kFileSizeMismatch,
                  "metadata.size", "pinned metadata file size mismatch",
                  file.path.string(), {}, std::to_string(pin.file_bytes),
                  std::to_string(file.bytes));
    }
    std::string observed_sha256;
    if (const auto error = verify_file_hash(
            file, pin.sha256, "metadata.sha256", observed_sha256)) {
      return PinnedBundleLoadResult{std::nullopt, *error};
    }
    if (const auto error = check_stable_size(file, "metadata.stability")) {
      return PinnedBundleLoadResult{std::nullopt, *error};
    }
    loaded.metadata_files.push_back(PinnedFileEvidence{
        pin.relative_path, file.path.string(), file.bytes,
        std::move(observed_sha256), true});
    if (pin.relative_path == descriptor.model->index_file) {
      index_file = file;
    }
  }

  const st::Result<st::Index> index = st::read_index(index_file.path.string());
  if (!index) {
    return fail(PinnedCheckpointErrorCode::kIndexReadFailed,
                "index.read", "failed to parse pinned safetensors index",
                index_file.path.string(), {}, "valid safetensors index",
                describe_safetensors_error(index.error));
  }
  if (const auto error = check_stable_size(index_file, "index.stability")) {
    return PinnedBundleLoadResult{std::nullopt, *error};
  }

  std::vector<const PinnedShard*> selected_shards;
  std::set<std::string, std::less<>> selected_shard_names;
  for (const PinnedTensor* const tensor : selected) {
    const std::string* const indexed_shard =
        index.value->shard_for(tensor->name);
    if (indexed_shard == nullptr || *indexed_shard != tensor->shard ||
        !st::is_safe_relative_shard_path(
            indexed_shard == nullptr ? std::string_view{}
                                     : std::string_view(*indexed_shard))) {
      return fail(PinnedCheckpointErrorCode::kIndexMappingMismatch,
                  "index.mapping", "tensor-to-shard index pin mismatch",
                  index_file.path.string(), tensor->name, tensor->shard,
                  indexed_shard == nullptr ? "missing" : *indexed_shard);
    }
    if (selected_shard_names.insert(tensor->shard).second) {
      selected_shards.push_back(find_shard(*descriptor.model, tensor->shard));
    }
  }

  loaded.tensors.reserve(selected.size());
  loaded.shards.reserve(selected_shards.size());
  for (const PinnedShard* const shard_pin : selected_shards) {
    if (shard_pin == nullptr) {
      return fail(PinnedCheckpointErrorCode::kInvalidDescriptor,
                  "descriptor.shard", "selected tensor references no shard");
    }
    ResolvedFile shard_file;
    if (const auto error = resolve_regular_file(
            directory, shard_pin->relative_path, "shard.resolve", shard_file)) {
      return PinnedBundleLoadResult{std::nullopt, *error};
    }
    if (shard_file.bytes != shard_pin->file_bytes) {
      return fail(PinnedCheckpointErrorCode::kFileSizeMismatch,
                  "shard.size", "pinned checkpoint shard size mismatch",
                  shard_file.path.string(), {},
                  std::to_string(shard_pin->file_bytes),
                  std::to_string(shard_file.bytes));
    }

    const st::Result<st::Header> header =
        st::read_header(shard_file.path.string());
    if (!header) {
      return fail(PinnedCheckpointErrorCode::kShardHeaderReadFailed,
                  "shard.header", "failed to read pinned shard header",
                  shard_file.path.string(), {}, "valid safetensors header",
                  describe_safetensors_error(header.error));
    }
    if (header.value->file_size != shard_pin->file_bytes ||
        header.value->header_size != shard_pin->header_bytes ||
        header.value->data_offset != shard_pin->data_offset) {
      return fail(
          PinnedCheckpointErrorCode::kShardHeaderMismatch,
          "shard.header", "pinned shard header contract mismatch",
          shard_file.path.string(), {},
          "bytes=" + std::to_string(shard_pin->file_bytes) +
              " header=" + std::to_string(shard_pin->header_bytes) +
              " data_offset=" + std::to_string(shard_pin->data_offset),
          "bytes=" + std::to_string(header.value->file_size) +
              " header=" + std::to_string(header.value->header_size) +
              " data_offset=" + std::to_string(header.value->data_offset));
    }

    PinnedShardEvidence shard_evidence;
    shard_evidence.relative_path = shard_pin->relative_path;
    shard_evidence.canonical_path = shard_file.path.string();
    shard_evidence.file_bytes = shard_file.bytes;
    shard_evidence.header_bytes = header.value->header_size;
    shard_evidence.data_offset = header.value->data_offset;
    shard_evidence.expected_sha256 = shard_pin->sha256;
    shard_evidence.opened_read_only = true;
    if (options.verify_full_shard_sha256) {
      if (const auto error = verify_file_hash(
              shard_file, shard_pin->sha256, "shard.sha256",
              shard_evidence.observed_sha256)) {
        return PinnedBundleLoadResult{std::nullopt, *error};
      }
      shard_evidence.full_sha256_verified = true;
    }

    std::ifstream input(shard_file.path, std::ios::binary | std::ios::in);
    if (!input) {
      return fail(PinnedCheckpointErrorCode::kPayloadReadFailed,
                  "shard.open", "failed to open checkpoint shard read-only",
                  shard_file.path.string());
    }
    const std::uint64_t maximum_streamoff = static_cast<std::uint64_t>(
        std::numeric_limits<std::streamoff>::max());
    const std::uint64_t maximum_streamsize = static_cast<std::uint64_t>(
        std::numeric_limits<std::streamsize>::max());
    const std::uint64_t maximum_size_t = static_cast<std::uint64_t>(
        std::numeric_limits<std::size_t>::max());

    for (const PinnedTensor* const tensor_pin : selected) {
      if (tensor_pin->shard != shard_pin->relative_path) {
        continue;
      }
      const st::TensorInfo* const tensor =
          header.value->find_tensor(tensor_pin->name);
      if (tensor == nullptr) {
        return fail(PinnedCheckpointErrorCode::kTensorMissing,
                    "tensor.header", "pinned tensor is absent from shard",
                    shard_file.path.string(), tensor_pin->name,
                    tensor_contract_string(*tensor_pin), "missing");
      }
      const std::uint64_t expected_bytes =
          tensor_pin->file_end - tensor_pin->file_begin;
      const bool metadata_exact =
          tensor->dtype == tensor_pin->dtype &&
          tensor->shape == tensor_pin->shape &&
          tensor->byte_size == expected_bytes &&
          tensor->file_begin == tensor_pin->file_begin &&
          tensor->file_end == tensor_pin->file_end &&
          tensor->file_end <= header.value->file_size;
      if (!metadata_exact) {
        return fail(PinnedCheckpointErrorCode::kTensorMetadataMismatch,
                    "tensor.header", "pinned tensor metadata mismatch",
                    shard_file.path.string(), tensor_pin->name,
                    tensor_contract_string(*tensor_pin),
                    tensor_contract_string(*tensor));
      }
      if (tensor->file_begin > maximum_streamoff ||
          tensor->file_end > maximum_streamoff ||
          tensor->byte_size > maximum_streamsize ||
          tensor->byte_size > maximum_size_t) {
        return fail(PinnedCheckpointErrorCode::kRangeNotRepresentable,
                    "tensor.range",
                    "pinned tensor range is not stream/size_t representable",
                    shard_file.path.string(), tensor_pin->name,
                    "representable bounded range",
                    tensor_contract_string(*tensor));
      }

      PinnedTensorPayload payload;
      payload.name = tensor_pin->name;
      payload.shard = tensor_pin->shard;
      payload.dtype = tensor_pin->dtype;
      payload.shape = tensor_pin->shape;
      payload.file_begin = tensor_pin->file_begin;
      payload.file_end = tensor_pin->file_end;
      payload.bytes.resize(static_cast<std::size_t>(tensor->byte_size));

      input.clear();
      input.seekg(static_cast<std::streamoff>(tensor->file_begin),
                  std::ios::beg);
      if (!input) {
        return fail(PinnedCheckpointErrorCode::kPayloadReadFailed,
                    "tensor.seek", "failed to seek to pinned tensor payload",
                    shard_file.path.string(), tensor_pin->name,
                    std::to_string(tensor->file_begin), "seek failed");
      }
      input.read(reinterpret_cast<char*>(payload.bytes.data()),
                 static_cast<std::streamsize>(payload.bytes.size()));
      if (input.gcount() !=
          static_cast<std::streamsize>(payload.bytes.size())) {
        return fail(PinnedCheckpointErrorCode::kPayloadReadFailed,
                    "tensor.read", "short read of pinned tensor payload",
                    shard_file.path.string(), tensor_pin->name,
                    std::to_string(payload.bytes.size()),
                    std::to_string(input.gcount()));
      }
      payload.sha256 = core::sha256(std::string_view(
          reinterpret_cast<const char*>(payload.bytes.data()),
          payload.bytes.size())).hex();
      if (payload.sha256 != tensor_pin->sha256) {
        return fail(PinnedCheckpointErrorCode::kPayloadHashMismatch,
                    "tensor.sha256", "pinned tensor SHA256 mismatch",
                    shard_file.path.string(), tensor_pin->name,
                    tensor_pin->sha256, payload.sha256);
      }

      if (tensor_pin->exact_f32_bits.has_value() ||
          tensor_pin->require_finite_positive_f32) {
        if (payload.bytes.size() != sizeof(float)) {
          return fail(PinnedCheckpointErrorCode::kScalarMismatch,
                      "tensor.scalar", "F32 scalar payload has wrong size",
                      shard_file.path.string(), tensor_pin->name,
                      std::to_string(sizeof(float)),
                      std::to_string(payload.bytes.size()));
        }
        const std::uint32_t bits =
            decode_little_endian_u32(payload.bytes.data());
        float value = 0.0F;
        std::memcpy(&value, &bits, sizeof(value));
        payload.f32_bits = bits;
        payload.f32_value = value;
        const bool bits_exact = !tensor_pin->exact_f32_bits.has_value() ||
                                bits == *tensor_pin->exact_f32_bits;
        const bool value_valid =
            !tensor_pin->require_finite_positive_f32 ||
            (std::isfinite(value) && value > 0.0F);
        if (!bits_exact || !value_valid) {
          return fail(PinnedCheckpointErrorCode::kScalarMismatch,
                      "tensor.scalar", "pinned F32 scalar contract mismatch",
                      shard_file.path.string(), tensor_pin->name,
                      tensor_pin->exact_f32_bits.has_value()
                          ? std::to_string(*tensor_pin->exact_f32_bits)
                          : "finite-positive",
                      std::to_string(bits));
        }
      }
      loaded.tensors.push_back(std::move(payload));
    }

    input.clear();
    input.seekg(0, std::ios::end);
    const std::streamoff observed_end = input.tellg();
    if (!input || observed_end < 0 ||
        static_cast<std::uint64_t>(observed_end) != shard_file.bytes) {
      return fail(PinnedCheckpointErrorCode::kFileSizeMismatch,
                  "shard.stability",
                  "checkpoint shard size changed while payloads were read",
                  shard_file.path.string(), {},
                  std::to_string(shard_file.bytes),
                  observed_end < 0 ? "tellg failed"
                                   : std::to_string(observed_end));
    }
    if (const auto error = check_stable_size(shard_file, "shard.stability")) {
      return PinnedBundleLoadResult{std::nullopt, *error};
    }
    loaded.shards.push_back(std::move(shard_evidence));
  }

  if (loaded.tensors.size() != selected.size()) {
    return fail(PinnedCheckpointErrorCode::kPayloadReadFailed,
                "bundle.complete", "not every selected tensor was loaded",
                {}, {}, std::to_string(selected.size()),
                std::to_string(loaded.tensors.size()));
  }
  return PinnedBundleLoadResult{std::move(loaded), std::nullopt};
}

}  // namespace

const PinnedTensorPayload* LoadedPinnedBundle::find_tensor(
    const std::string_view name) const noexcept {
  const auto found =
      std::find_if(tensors.begin(), tensors.end(),
                   [name](const PinnedTensorPayload& tensor) {
                     return tensor.name == name;
                   });
  return found == tensors.end() ? nullptr : &*found;
}

PinnedBundleLoadResult load_pinned_checkpoint_bundle(
    const fs::path& checkpoint_directory,
    const PinnedBundleDescriptor& descriptor,
    const PinnedBundleLoadOptions& options) {
  try {
    return load_impl(checkpoint_directory, descriptor, options);
  } catch (const std::bad_alloc&) {
    return fail(PinnedCheckpointErrorCode::kAllocationFailure,
                "allocation", "allocation failed while loading pinned bundle");
  } catch (const std::length_error& error) {
    return fail(PinnedCheckpointErrorCode::kAllocationFailure,
                "allocation", error.what());
  }
}

const PinnedTensor* find_pinned_tensor(
    const PinnedBundleDescriptor& descriptor,
    const std::string_view name) noexcept {
  const auto found =
      std::find_if(descriptor.tensors.begin(), descriptor.tensors.end(),
                   [name](const PinnedTensor& tensor) {
                     return tensor.name == name;
                   });
  return found == descriptor.tensors.end() ? nullptr : &*found;
}

std::string_view to_string(const PinnedCheckpointErrorCode code) noexcept {
  switch (code) {
    case PinnedCheckpointErrorCode::kNone:
      return "none";
    case PinnedCheckpointErrorCode::kInvalidDescriptor:
      return "invalid-descriptor";
    case PinnedCheckpointErrorCode::kInvalidRequest:
      return "invalid-request";
    case PinnedCheckpointErrorCode::kPathMissing:
      return "path-missing";
    case PinnedCheckpointErrorCode::kPathTypeMismatch:
      return "path-type-mismatch";
    case PinnedCheckpointErrorCode::kSymlinkRejected:
      return "symlink-rejected";
    case PinnedCheckpointErrorCode::kCanonicalizationFailed:
      return "canonicalization-failed";
    case PinnedCheckpointErrorCode::kPathEscape:
      return "path-escape";
    case PinnedCheckpointErrorCode::kFileSizeMismatch:
      return "file-size-mismatch";
    case PinnedCheckpointErrorCode::kFileHashFailure:
      return "file-hash-failure";
    case PinnedCheckpointErrorCode::kFileHashMismatch:
      return "file-hash-mismatch";
    case PinnedCheckpointErrorCode::kIndexReadFailed:
      return "index-read-failed";
    case PinnedCheckpointErrorCode::kIndexMappingMismatch:
      return "index-mapping-mismatch";
    case PinnedCheckpointErrorCode::kShardHeaderReadFailed:
      return "shard-header-read-failed";
    case PinnedCheckpointErrorCode::kShardHeaderMismatch:
      return "shard-header-mismatch";
    case PinnedCheckpointErrorCode::kTensorMissing:
      return "tensor-missing";
    case PinnedCheckpointErrorCode::kTensorMetadataMismatch:
      return "tensor-metadata-mismatch";
    case PinnedCheckpointErrorCode::kRangeNotRepresentable:
      return "range-not-representable";
    case PinnedCheckpointErrorCode::kPayloadReadFailed:
      return "payload-read-failed";
    case PinnedCheckpointErrorCode::kPayloadHashMismatch:
      return "payload-hash-mismatch";
    case PinnedCheckpointErrorCode::kScalarMismatch:
      return "scalar-mismatch";
    case PinnedCheckpointErrorCode::kAllocationFailure:
      return "allocation-failure";
  }
  return "unknown";
}

std::string describe_pinned_checkpoint_error(
    const PinnedCheckpointError& error) {
  std::string description = std::string(to_string(error.code));
  if (!error.stage.empty()) {
    description += " stage=" + error.stage;
  }
  if (!error.path.empty()) {
    description += " path=" + error.path;
  }
  if (!error.tensor.empty()) {
    description += " tensor=" + error.tensor;
  }
  if (!error.expected.empty()) {
    description += " expected=" + error.expected;
  }
  if (!error.actual.empty()) {
    description += " actual=" + error.actual;
  }
  if (!error.message.empty()) {
    description += " message=" + error.message;
  }
  return description;
}

const PinnedModelRevision& qwen36_27b_nvfp4_model_revision() {
  static const PinnedModelRevision revision{
      "nvidia-qwen3.6-27b-nvfp4@0893e160",
      "nvidia/Qwen3.6-27B-NVFP4",
      "0893e1606ff3d5f97a441f405d5fc541a6bdf404",
      std::string(kQwen36Index),
      {
          PinnedFile{
              "config.json",
              "c04a19ba293737ad7be4f6e96d6666cb7e479cbe19ecc0c289fad267135b0338",
              88'567U},
          PinnedFile{
              "hf_quant_config.json",
              "fd7200cd8bca2a8a5d777061521abf83e2deb97ab6bc2f04e7a0a3d3f8ecd5c1",
              54'902U},
          PinnedFile{
              std::string(kQwen36Index),
              "7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2",
              214'866U},
      },
      {
          PinnedShard{
              std::string(kQwen36Shard1),
              9'965'652'512ULL,
              126'504U,
              126'512U,
              "b4a0d9a57ff1859dac1144b53ca285011db072737d8813fc16d8d1e07ecae17d"},
      }};
  return revision;
}

const PinnedBundleDescriptor& qwen36_27b_nvfp4_layer0_mlp_bundle() {
  static const PinnedBundleDescriptor descriptor{
      "qwen36-27b-nvfp4-layer0-mlp",
      &qwen36_27b_nvfp4_model_revision(),
      {
          PinnedTensor{
              std::string(kQwen36Layer0GateWeight),
              std::string(kQwen36Shard1),
              st::DType::kU8,
              {17'408U, 2'560U},
              6'757'009'952ULL,
              6'801'574'432ULL,
              "e9e2d70cef19e52d65a0f7917ea6d936c172809ed247b350443b4344297159d8",
              std::nullopt,
              false},
          PinnedTensor{
              std::string(kQwen36Layer0GateBlockScale),
              std::string(kQwen36Shard1),
              st::DType::kF8E4M3,
              {17'408U, 320U},
              3'606'039'072ULL,
              3'611'609'632ULL,
              "6eeaaa3bf8605b1d85252e13e6c495f6cf1b06e7fee8f27ea6367abbbb8fde0e",
              std::nullopt,
              false},
          PinnedTensor{
              std::string(kQwen36Layer0GateWeightScale2),
              std::string(kQwen36Shard1),
              st::DType::kF32,
              {},
              126'548U,
              126'552U,
              "10f036efcb439d7571cc2c35568c9786ae3315b169f3972ff1c2aae782131f91",
              0x391e'79e8U,
              true},
          PinnedTensor{
              std::string(kQwen36Layer0UpWeight),
              std::string(kQwen36Shard1),
              st::DType::kU8,
              {17'408U, 2'560U},
              6'801'574'432ULL,
              6'846'138'912ULL,
              "e604b0b18206afe695a191ecf77a6aaf4dfbc0f7e93f1f9789d9b579aed6215f",
              std::nullopt,
              false},
          PinnedTensor{
              std::string(kQwen36Layer0UpBlockScale),
              std::string(kQwen36Shard1),
              st::DType::kF8E4M3,
              {17'408U, 320U},
              3'611'609'632ULL,
              3'617'180'192ULL,
              "ba393d3f9d25a1f4decba80715c6079d27d9de1d059a038a2f3f3f0932870947",
              std::nullopt,
              false},
          PinnedTensor{
              std::string(kQwen36Layer0UpWeightScale2),
              std::string(kQwen36Shard1),
              st::DType::kF32,
              {},
              126'556U,
              126'560U,
              "10f036efcb439d7571cc2c35568c9786ae3315b169f3972ff1c2aae782131f91",
              0x391e'79e8U,
              true},
          PinnedTensor{
              std::string(kQwen36Layer0DownWeight),
              std::string(kQwen36Shard1),
              st::DType::kU8,
              {5'120U, 8'704U},
              6'712'445'472ULL,
              6'757'009'952ULL,
              "bc1b428661d3cf657a4d69ff8d7e482b8125ef0f323e6df29b153a22fa2b6daf",
              std::nullopt,
              false},
          PinnedTensor{
              std::string(kQwen36Layer0DownBlockScale),
              std::string(kQwen36Shard1),
              st::DType::kF8E4M3,
              {5'120U, 1'088U},
              3'600'468'512ULL,
              3'606'039'072ULL,
              "7943b475b23f75886309e93bf673aacc22c699e19ff400ef85607ab1a4006019",
              std::nullopt,
              false},
          PinnedTensor{
              std::string(kQwen36Layer0DownWeightScale2),
              std::string(kQwen36Shard1),
              st::DType::kF32,
              {},
              126'540U,
              126'544U,
              "face5c84f9ca43eb34d792708f1c0d4bdc532c91d6419e65123a5d560245dd72",
              0x39b6'1862U,
              true},
      },
      true};
  return descriptor;
}

}  // namespace q3x::test::support
