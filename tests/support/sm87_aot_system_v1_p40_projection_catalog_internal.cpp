#include "sm87_aot_system_v1_p40_projection_catalog_internal.h"

#include "q3x/model/model_config.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <map>
#include <new>
#include <set>
#include <utility>

namespace q3x::test::sm87_aot_p40_projection_catalog {
namespace {

namespace reader = sm87_aot_checkpoint_reader;
namespace weights = model::weights;
namespace st = io::safetensors;

struct ModuleSpec {
  std::string name;
  std::uint32_t layer = 0U;
  ProjectionRole role = ProjectionRole::kInvalid;
};

struct SelectedRange {
  reader::TensorRangeRequest request;
  std::size_t tensor_index = 0U;
  bool scalar = false;
};

[[nodiscard]] Result failure(const ErrorCode code, std::string context = {},
                             const reader::ReaderError& source = {}) noexcept {
  Result result;
  result.error.code = code;
  result.error.context = std::move(context);
  result.error.reader_error = source;
  return result;
}

void hash_u64(core::Sha256& hash, const std::uint64_t value) noexcept {
  std::array<std::uint8_t, 8U> bytes{};
  for (std::size_t index = 0U; index < bytes.size(); ++index) {
    bytes[index] = static_cast<std::uint8_t>(value >> (index * 8U));
  }
  (void)hash.update(bytes.data(), bytes.size());
}

void hash_string(core::Sha256& hash, const std::string_view value) noexcept {
  hash_u64(hash, value.size());
  (void)hash.update(value.data(), value.size());
}

[[nodiscard]] std::vector<ModuleSpec> frozen_modules() {
  const model::ModelConfig* const config =
      model::find_known_model(model::KnownModel::kQwen36_27B);
  if (config == nullptr || config->num_hidden_layers != 64U) {
    return {};
  }
  std::vector<ModuleSpec> modules;
  modules.reserve(kP40ProjectionModuleCount);
  for (std::uint32_t layer = 0U; layer < config->num_hidden_layers; ++layer) {
    const model::LayerType layer_type = config->layer_type(layer);
    const model::LayerType expected_type =
        layer % 4U == 3U ? model::LayerType::kFullAttention
                         : model::LayerType::kLinearAttention;
    if (layer_type != expected_type) {
      return {};
    }
    const std::string prefix =
        "model.language_model.layers." + std::to_string(layer) + ".";
    modules.push_back({prefix + "mlp.gate_proj", layer,
                       ProjectionRole::kNvFp4GateUp});
    modules.push_back({prefix + "mlp.up_proj", layer,
                       ProjectionRole::kNvFp4GateUp});
    modules.push_back({prefix + "mlp.down_proj", layer,
                       ProjectionRole::kNvFp4Down});
    if (layer_type == model::LayerType::kLinearAttention) {
      modules.push_back({prefix + "linear_attn.in_proj_qkv", layer,
                         ProjectionRole::kFp8GdnQkvZ});
      modules.push_back({prefix + "linear_attn.in_proj_z", layer,
                         ProjectionRole::kFp8GdnQkvZ});
      modules.push_back({prefix + "linear_attn.out_proj", layer,
                         ProjectionRole::kFp8AttentionOutput});
    } else if (layer_type == model::LayerType::kFullAttention) {
      for (const std::string_view projection : {"q_proj", "k_proj", "v_proj"}) {
        modules.push_back({prefix + "self_attn." + std::string(projection),
                           layer, ProjectionRole::kFp8FullQkv});
      }
      modules.push_back({prefix + "self_attn.o_proj", layer,
                         ProjectionRole::kFp8AttentionOutput});
    } else {
      return {};
    }
  }
  return modules;
}

[[nodiscard]] bool append_tensor(const weights::WeightManifest& manifest,
                                 const ModuleSpec& module,
                                 const std::string_view suffix,
                                 const TensorKind kind,
                                 const st::DType dtype,
                                 const bool scalar,
                                 Catalog& catalog,
                                 ModuleRecord& output_module,
                                 std::set<std::string, std::less<>>& names,
                                 Error& error) {
  const std::string name = module.name + std::string(suffix);
  const weights::TensorLocator* const locator = manifest.find(name);
  if (locator == nullptr) {
    error = {ErrorCode::kMissingModuleTensor, name, {}};
    return false;
  }
  if (locator->category != weights::TensorCategory::kText ||
      locator->dtype != dtype || locator->file_begin >= locator->file_end ||
      locator->byte_size != locator->file_end - locator->file_begin ||
      (scalar && (!locator->shape.empty() || locator->byte_size != 4U)) ||
      (!scalar && locator->shape.empty())) {
    error = {ErrorCode::kTensorContractMismatch, name, {}};
    return false;
  }
  if (!names.emplace(name).second) {
    error = {ErrorCode::kDuplicateTensor, name, {}};
    return false;
  }
  TensorRecord tensor;
  tensor.name = name;
  tensor.layer = module.layer;
  tensor.role = module.role;
  tensor.kind = kind;
  tensor.dtype = locator->dtype;
  tensor.shape = locator->shape;
  tensor.shard = locator->shard;
  tensor.file = locator->file;
  tensor.file_begin = locator->file_begin;
  tensor.file_end = locator->file_end;
  tensor.scalar_scale_bits_present = scalar;
  output_module.tensor_indices.push_back(catalog.tensors.size());
  catalog.tensors.emplace_back(std::move(tensor));
  return true;
}

class PayloadSink final : public reader::ProvisionalRangeSink {
 public:
  PayloadSink(const std::vector<SelectedRange>& selected, Catalog& catalog)
      : selected_(selected), catalog_(catalog), hashes_(selected.size()),
        delivered_(selected.size(), 0U), scalar_bytes_(selected.size()) {}

  [[nodiscard]] bool begin(const reader::SinkTransaction& transaction) override {
    begun_ = transaction.expected_range_count == selected_.size();
    return begun_;
  }

  [[nodiscard]] reader::SinkConsumeResult consume(
      const std::size_t range_index, const std::uint64_t absolute_file_offset,
      const std::uint64_t tensor_byte_offset, const std::uint8_t* const bytes,
      const std::size_t byte_count) override {
    if (!begun_ || range_index >= selected_.size() ||
        tensor_byte_offset != delivered_[range_index] ||
        absolute_file_offset !=
            selected_[range_index].request.file_begin + tensor_byte_offset ||
        bytes == nullptr || !hashes_[range_index].update(bytes, byte_count)) {
      return {};
    }
    if (selected_[range_index].scalar) {
      if (tensor_byte_offset > 4U || byte_count > 4U - tensor_byte_offset) {
        return {};
      }
      std::memcpy(scalar_bytes_[range_index].data() + tensor_byte_offset,
                  bytes, byte_count);
    }
    delivered_[range_index] += byte_count;
    return {true, byte_count};
  }

  [[nodiscard]] bool commit() override {
    if (!begun_) {
      return false;
    }
    for (std::size_t index = 0U; index < selected_.size(); ++index) {
      const SelectedRange& selected = selected_[index];
      if (delivered_[index] !=
          selected.request.file_end - selected.request.file_begin) {
        return false;
      }
      TensorRecord& tensor = catalog_.tensors[selected.tensor_index];
      tensor.payload_sha256 = hashes_[index].finalize();
      if (tensor.payload_sha256.hex() == std::string(64U, '0')) {
        return false;
      }
      if (selected.scalar) {
        tensor.scalar_scale_bits =
            static_cast<std::uint32_t>(scalar_bytes_[index][0]) |
            (static_cast<std::uint32_t>(scalar_bytes_[index][1]) << 8U) |
            (static_cast<std::uint32_t>(scalar_bytes_[index][2]) << 16U) |
            (static_cast<std::uint32_t>(scalar_bytes_[index][3]) << 24U);
      }
    }
    committed_ = true;
    return true;
  }

  void abort() noexcept override { aborted_ = true; }

 private:
  const std::vector<SelectedRange>& selected_;
  Catalog& catalog_;
  std::vector<core::Sha256> hashes_;
  std::vector<std::uint64_t> delivered_;
  std::vector<std::array<std::uint8_t, 4U>> scalar_bytes_;
  bool begun_ = false;
  bool committed_ = false;
  bool aborted_ = false;
};

[[nodiscard]] core::Sha256Digest catalog_digest(const Catalog& catalog) noexcept {
  core::Sha256 hash;
  hash_string(hash, kP40ProjectionCatalogDomain);
  hash_u64(hash, catalog.modules.size());
  hash_u64(hash, catalog.tensors.size());
  for (const ModuleRecord& module : catalog.modules) {
    hash_string(hash, module.name);
    hash_u64(hash, module.layer);
    hash_u64(hash, static_cast<std::uint64_t>(module.role));
    hash_u64(hash, module.tensor_indices.size());
    for (const std::size_t index : module.tensor_indices) hash_u64(hash, index);
  }
  for (const TensorRecord& tensor : catalog.tensors) {
    hash_string(hash, tensor.name);
    hash_u64(hash, tensor.layer);
    hash_u64(hash, static_cast<std::uint64_t>(tensor.role));
    hash_u64(hash, static_cast<std::uint64_t>(tensor.kind));
    hash_u64(hash, static_cast<std::uint64_t>(tensor.dtype));
    hash_u64(hash, tensor.shape.size());
    for (const std::uint64_t dimension : tensor.shape) hash_u64(hash, dimension);
    hash_string(hash, tensor.shard);
    hash_string(hash, tensor.file.string());
    hash_u64(hash, tensor.file_begin);
    hash_u64(hash, tensor.file_end);
    (void)hash.update(tensor.payload_sha256.bytes.data(),
                      tensor.payload_sha256.bytes.size());
    hash_u64(hash, tensor.scalar_scale_bits_present ? 1U : 0U);
    hash_u64(hash, tensor.scalar_scale_bits);
  }
  hash_u64(hash, catalog.shard_receipts.size());
  for (const reader::ReaderReceipt& receipt : catalog.shard_receipts) {
    (void)hash.update(receipt.receipt_sha256.bytes.data(),
                      receipt.receipt_sha256.bytes.size());
  }
  return hash.finalize();
}

}  // namespace

Result build_p40_projection_tensor_catalog(
    const std::filesystem::path& checkpoint_root,
    const weights::WeightManifest& manifest,
    const reader::ReaderOptions& reader_options) {
  try {
    if (manifest.checkpoint.model != model::KnownModel::kQwen36_27B ||
        manifest.summary.text_tensor_count !=
            weights::kPinnedQwen36_27BTextTensorCount ||
        manifest.summary.shard_count != 3U ||
        manifest.summary.fp8_module_count != 208U ||
        manifest.summary.nvfp4_module_count != 193U) {
      return failure(ErrorCode::kInvalidManifest, "manifest identity");
    }
    const std::vector<ModuleSpec> modules = frozen_modules();
    if (modules.size() != kP40ProjectionModuleCount) {
      return failure(ErrorCode::kInternalFailure, "frozen module schedule");
    }

    Catalog catalog;
    catalog.modules.reserve(modules.size());
    catalog.tensors.reserve(1'600U);
    std::set<std::string, std::less<>> names;
    for (const ModuleSpec& module : modules) {
      ModuleRecord output_module{module.name, module.layer, module.role, {}};
      const weights::TensorLocator* const weight =
          manifest.find(module.name + ".weight");
      if (weight == nullptr) {
        return failure(ErrorCode::kMissingModuleTensor,
                       module.name + ".weight");
      }
      Error error;
      if (weight->dtype == st::DType::kU8) {
        if (module.role != ProjectionRole::kNvFp4GateUp &&
            module.role != ProjectionRole::kNvFp4Down) {
          return failure(ErrorCode::kTensorContractMismatch, module.name);
        }
        if (!append_tensor(manifest, module, ".weight", TensorKind::kWeight,
                           st::DType::kU8, false, catalog, output_module, names,
                           error) ||
            !append_tensor(manifest, module, ".weight_scale",
                           TensorKind::kWeightScale, st::DType::kF8E4M3, false,
                           catalog, output_module, names, error) ||
            !append_tensor(manifest, module, ".weight_scale_2",
                           TensorKind::kWeightScale2, st::DType::kF32, true,
                           catalog, output_module, names, error) ||
            !append_tensor(manifest, module, ".input_scale",
                           TensorKind::kInputScale, st::DType::kF32, true,
                           catalog, output_module, names, error)) {
          return failure(error.code, error.context);
        }
      } else if (weight->dtype == st::DType::kF8E4M3) {
        if (module.role == ProjectionRole::kNvFp4GateUp ||
            module.role == ProjectionRole::kNvFp4Down) {
          return failure(ErrorCode::kTensorContractMismatch, module.name);
        }
        if (!append_tensor(manifest, module, ".weight", TensorKind::kWeight,
                           st::DType::kF8E4M3, false, catalog, output_module,
                           names, error) ||
            !append_tensor(manifest, module, ".weight_scale",
                           TensorKind::kWeightScale, st::DType::kF32, true,
                           catalog, output_module, names, error) ||
            !append_tensor(manifest, module, ".input_scale",
                           TensorKind::kInputScale, st::DType::kF32, true,
                           catalog, output_module, names, error)) {
          return failure(error.code, error.context);
        }
      } else {
        return failure(ErrorCode::kTensorContractMismatch, module.name);
      }
      catalog.modules.emplace_back(std::move(output_module));
    }

    std::error_code path_error;
    const std::filesystem::path absolute_root =
        std::filesystem::absolute(checkpoint_root, path_error).lexically_normal();
    if (path_error || !absolute_root.is_absolute()) {
      return failure(ErrorCode::kShardContractMismatch, "checkpoint root");
    }
    std::map<std::string, std::vector<SelectedRange>, std::less<>> by_shard;
    for (std::size_t index = 0U; index < catalog.tensors.size(); ++index) {
      const TensorRecord& tensor = catalog.tensors[index];
      if (tensor.file != (absolute_root / tensor.shard).lexically_normal()) {
        return failure(ErrorCode::kShardContractMismatch, tensor.name);
      }
      SelectedRange selected;
      selected.request = {tensor.name, tensor.dtype, tensor.shape,
                          tensor.file_begin, tensor.file_end};
      selected.tensor_index = index;
      selected.scalar = tensor.scalar_scale_bits_present;
      by_shard[tensor.shard].emplace_back(std::move(selected));
    }
    if (by_shard.size() != 3U) {
      return failure(ErrorCode::kShardContractMismatch, "selected shard count");
    }
    catalog.shard_receipts.reserve(by_shard.size());
    for (auto& shard : by_shard) {
      std::vector<SelectedRange>& selected = shard.second;
      std::sort(selected.begin(), selected.end(), [](const SelectedRange& a,
                                                     const SelectedRange& b) {
        return a.request.file_begin < b.request.file_begin;
      });
      std::vector<reader::TensorRangeRequest> requests;
      requests.reserve(selected.size());
      for (const SelectedRange& item : selected) requests.push_back(item.request);
      PayloadSink sink(selected, catalog);
      reader::ReaderResult read = reader::read_pinned_shard_ranges(
          checkpoint_root, shard.first, requests, sink, reader_options);
      if (!read) {
        return failure(ErrorCode::kReaderFailure, shard.first, read.error);
      }
      catalog.shard_receipts.push_back(*read.receipt);
    }
    if (catalog.modules.size() != kP40ProjectionModuleCount ||
        catalog.tensors.size() != kP40ProjectionTensorCount ||
        catalog.shard_receipts.size() != 3U) {
      return failure(ErrorCode::kCountMismatch, "catalog totals");
    }
    catalog.catalog_sha256 = catalog_digest(catalog);
    Result result;
    result.value.emplace(std::move(catalog));
    return result;
  } catch (const std::bad_alloc&) {
    return failure(ErrorCode::kAllocationFailure, "catalog allocation");
  } catch (...) {
    return failure(ErrorCode::kInternalFailure, "catalog exception");
  }
}

bool run_p40_projection_catalog_protocol_self_test() noexcept {
  try {
    const std::vector<ModuleSpec> modules = frozen_modules();
    if (modules.size() != kP40ProjectionModuleCount) {
      return false;
    }
    std::array<std::size_t, 6U> counts{};
    std::set<std::string, std::less<>> names;
    std::size_t implied_tensor_count = 0U;
    for (const ModuleSpec& module : modules) {
      const std::size_t role = static_cast<std::size_t>(module.role);
      if (module.name.empty() || module.layer >= 64U || role == 0U ||
          role >= counts.size() || !names.emplace(module.name).second) {
        return false;
      }
      ++counts[role];
      implied_tensor_count +=
          module.role == ProjectionRole::kNvFp4GateUp ||
                  module.role == ProjectionRole::kNvFp4Down
              ? 4U
              : 3U;
    }
    return counts[static_cast<std::size_t>(ProjectionRole::kNvFp4GateUp)] ==
               128U &&
           counts[static_cast<std::size_t>(ProjectionRole::kNvFp4Down)] ==
               64U &&
           counts[static_cast<std::size_t>(ProjectionRole::kFp8GdnQkvZ)] ==
               96U &&
           counts[static_cast<std::size_t>(ProjectionRole::kFp8FullQkv)] ==
               48U &&
           counts[static_cast<std::size_t>(
               ProjectionRole::kFp8AttentionOutput)] == 64U &&
           implied_tensor_count == kP40ProjectionTensorCount;
  } catch (...) {
    return false;
  }
}

std::string_view to_string(const ProjectionRole role) noexcept {
  switch (role) {
    case ProjectionRole::kInvalid: return "invalid";
    case ProjectionRole::kNvFp4GateUp: return "nvfp4_gate_up";
    case ProjectionRole::kNvFp4Down: return "nvfp4_down";
    case ProjectionRole::kFp8GdnQkvZ: return "fp8_gdn_qkv_z";
    case ProjectionRole::kFp8FullQkv: return "fp8_full_qkv";
    case ProjectionRole::kFp8AttentionOutput: return "fp8_attention_output";
  }
  return "invalid";
}

std::string_view to_string(const TensorKind kind) noexcept {
  switch (kind) {
    case TensorKind::kWeight: return "weight";
    case TensorKind::kWeightScale: return "weight_scale";
    case TensorKind::kWeightScale2: return "weight_scale_2";
    case TensorKind::kInputScale: return "input_scale";
  }
  return "weight";
}

std::string_view to_string(const ErrorCode code) noexcept {
  switch (code) {
    case ErrorCode::kNone: return "none";
    case ErrorCode::kInvalidManifest: return "invalid manifest";
    case ErrorCode::kMissingModuleTensor: return "missing module tensor";
    case ErrorCode::kTensorContractMismatch: return "tensor contract mismatch";
    case ErrorCode::kDuplicateTensor: return "duplicate tensor";
    case ErrorCode::kShardContractMismatch: return "shard contract mismatch";
    case ErrorCode::kReaderFailure: return "reader failure";
    case ErrorCode::kPayloadFailure: return "payload failure";
    case ErrorCode::kCountMismatch: return "count mismatch";
    case ErrorCode::kArithmeticOverflow: return "arithmetic overflow";
    case ErrorCode::kAllocationFailure: return "allocation failure";
    case ErrorCode::kInternalFailure: return "internal failure";
  }
  return "internal failure";
}

}  // namespace q3x::test::sm87_aot_p40_projection_catalog
