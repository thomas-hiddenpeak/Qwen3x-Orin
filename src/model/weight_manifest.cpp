#include "q3x/model/weight_manifest.h"

#include "q3x/model/model_config.h"

#include <algorithm>
#include <limits>
#include <new>
#include <set>
#include <sstream>
#include <system_error>
#include <utility>

namespace q3x::model::weights {
namespace {

namespace st = io::safetensors;

struct TensorRequirement {
    st::DType dtype = st::DType::kBool;
    std::vector<std::uint64_t> shape;
};

using RequirementMap =
    std::map<std::string, TensorRequirement, std::less<>>;

ManifestDiagnostic make_diagnostic(ManifestErrorCode code,
                                   std::string context,
                                   std::string message,
                                   std::string expected = {},
                                   std::string actual = {}) {
    ManifestDiagnostic diagnostic;
    diagnostic.code = code;
    diagnostic.context = std::move(context);
    diagnostic.expected = std::move(expected);
    diagnostic.actual = std::move(actual);
    diagnostic.message = std::move(message);
    return diagnostic;
}

ManifestResult failure(ManifestDiagnostic diagnostic) {
    ManifestResult result;
    result.diagnostics.emplace_back(std::move(diagnostic));
    return result;
}

bool checked_add(std::uint64_t left,
                 std::uint64_t right,
                 std::uint64_t& output) noexcept {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        return false;
    }
    output = left + right;
    return true;
}

bool checked_multiply(std::uint64_t left,
                      std::uint64_t right,
                      std::uint64_t& output) noexcept {
    if (left != 0U &&
        right > std::numeric_limits<std::uint64_t>::max() / left) {
        return false;
    }
    output = left * right;
    return true;
}

bool is_power_of_two(std::uint64_t value) noexcept {
    return value != 0U && (value & (value - 1U)) == 0U;
}

bool checked_align_up(std::uint64_t value,
                      std::uint64_t alignment,
                      std::uint64_t& output) noexcept {
    const std::uint64_t mask = alignment - 1U;
    if (value > std::numeric_limits<std::uint64_t>::max() - mask) {
        return false;
    }
    output = (value + mask) & ~mask;
    return true;
}

bool compute_byte_size(st::DType dtype,
                       const std::vector<std::uint64_t>& shape,
                       std::uint64_t& output) noexcept {
    std::uint64_t elements = 1;
    for (const std::uint64_t dimension : shape) {
        if (!checked_multiply(elements, dimension, elements)) {
            return false;
        }
    }
    std::uint64_t bits = 0;
    if (!checked_multiply(elements,
                          static_cast<std::uint64_t>(st::bit_width(dtype)),
                          bits) ||
        (bits % 8U) != 0U) {
        return false;
    }
    output = bits / 8U;
    return true;
}

std::string shape_string(const std::vector<std::uint64_t>& shape) {
    std::ostringstream output;
    output << '[';
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index != 0U) {
            output << ',';
        }
        output << shape[index];
    }
    output << ']';
    return output.str();
}

bool add_requirement(RequirementMap& requirements,
                     std::string name,
                     st::DType dtype,
                     std::vector<std::uint64_t> shape) {
    return requirements
        .emplace(std::move(name),
                 TensorRequirement{dtype, std::move(shape)})
        .second;
}

std::optional<ManifestDiagnostic> add_quantized_requirement(
    RequirementMap& requirements,
    const checkpoint::ModelOptSummary& quantization,
    const std::string& module,
    std::uint64_t output_size,
    std::uint64_t input_size) {
    const auto selected = quantization.quantized_modules.find(module);
    if (selected == quantization.quantized_modules.end()) {
        return make_diagnostic(
            ManifestErrorCode::kQuantizationMismatch,
            module,
            "runtime projection has no ModelOpt per-module record");
    }

    bool inserted = true;
    if (selected->second.algorithm == checkpoint::QuantAlgorithm::kFp8) {
        inserted &= add_requirement(requirements,
                                    module + ".weight",
                                    st::DType::kF8E4M3,
                                    {output_size, input_size});
        inserted &= add_requirement(requirements,
                                    module + ".weight_scale",
                                    st::DType::kF32,
                                    {});
        inserted &= add_requirement(requirements,
                                    module + ".input_scale",
                                    st::DType::kF32,
                                    {});
    } else {
        if (selected->second.algorithm !=
                checkpoint::QuantAlgorithm::kW4A16Nvfp4 ||
            selected->second.group_size != 16U || (input_size % 16U) != 0U) {
            return make_diagnostic(
                ManifestErrorCode::kQuantizationMismatch,
                module,
                "runtime projection has an unsupported NVFP4 record",
                "W4A16_NVFP4 group_size=16");
        }
        inserted &= add_requirement(requirements,
                                    module + ".weight",
                                    st::DType::kU8,
                                    {output_size, input_size / 2U});
        inserted &= add_requirement(requirements,
                                    module + ".weight_scale",
                                    st::DType::kF8E4M3,
                                    {output_size, input_size / 16U});
        inserted &= add_requirement(requirements,
                                    module + ".weight_scale_2",
                                    st::DType::kF32,
                                    {});
        inserted &= add_requirement(requirements,
                                    module + ".input_scale",
                                    st::DType::kF32,
                                    {});
    }
    if (!inserted) {
        return make_diagnostic(ManifestErrorCode::kQuantizationMismatch,
                               module,
                               "duplicate generated tensor requirement");
    }
    return std::nullopt;
}

const checkpoint::KnownCheckpointDescriptor* dense_descriptor() noexcept {
    for (const auto& descriptor : checkpoint::known_checkpoint_catalog()) {
        if (descriptor.model == KnownModel::kQwen36_27B) {
            return &descriptor;
        }
    }
    return nullptr;
}

struct RequirementResult {
    std::optional<RequirementMap> value;
    ManifestDiagnostic diagnostic;
};

RequirementResult make_requirements(
    const checkpoint::ModelOptSummary& quantization) {
    RequirementResult result;
    const checkpoint::KnownCheckpointDescriptor* descriptor =
        dense_descriptor();
    if (descriptor == nullptr) {
        result.diagnostic = make_diagnostic(
            ManifestErrorCode::kUnsupportedCheckpoint,
            "checkpoint catalog",
            "Qwen3.6-27B pinned descriptor is unavailable");
        return result;
    }
    const std::vector<checkpoint::Diagnostic> quantization_errors =
        checkpoint::validate_modelopt_quantization(quantization, *descriptor);
    if (!quantization_errors.empty()) {
        const checkpoint::Diagnostic& error = quantization_errors.front();
        result.diagnostic = make_diagnostic(
            ManifestErrorCode::kQuantizationMismatch,
            error.json_pointer,
            error.message,
            error.expected,
            error.actual);
        return result;
    }

    const ModelConfig* config = find_known_model(KnownModel::kQwen36_27B);
    if (config == nullptr || config->num_hidden_layers != 64U ||
        config->hidden_size != 5120U || config->intermediate_size != 17408U) {
        result.diagnostic = make_diagnostic(
            ManifestErrorCode::kUnsupportedCheckpoint,
            "model catalog",
            "Qwen3.6-27B architecture contract is unavailable");
        return result;
    }

    RequirementMap requirements;
    add_requirement(requirements,
                    "model.language_model.embed_tokens.weight",
                    st::DType::kBf16,
                    {config->vocab_size, config->hidden_size});
    add_requirement(requirements,
                    "model.language_model.norm.weight",
                    st::DType::kBf16,
                    {config->hidden_size});

    for (std::uint32_t layer = 0; layer < config->num_hidden_layers; ++layer) {
        const std::string prefix =
            "model.language_model.layers." + std::to_string(layer) + ".";
        add_requirement(requirements,
                        prefix + "input_layernorm.weight",
                        st::DType::kBf16,
                        {config->hidden_size});
        add_requirement(requirements,
                        prefix + "post_attention_layernorm.weight",
                        st::DType::kBf16,
                        {config->hidden_size});

        for (const auto& projection : {
                 std::pair<std::string_view, std::pair<std::uint64_t,
                                                       std::uint64_t>>{
                     "mlp.gate_proj",
                     {config->intermediate_size, config->hidden_size}},
                 {"mlp.up_proj",
                  {config->intermediate_size, config->hidden_size}},
                 {"mlp.down_proj",
                  {config->hidden_size, config->intermediate_size}},
             }) {
            if (auto error = add_quantized_requirement(
                    requirements,
                    quantization,
                    prefix + std::string(projection.first),
                    projection.second.first,
                    projection.second.second)) {
                result.diagnostic = std::move(*error);
                return result;
            }
        }

        if (config->layer_type(layer) == LayerType::kLinearAttention) {
            add_requirement(requirements,
                            prefix + "linear_attn.in_proj_a.weight",
                            st::DType::kBf16,
                            {48U, config->hidden_size});
            add_requirement(requirements,
                            prefix + "linear_attn.in_proj_b.weight",
                            st::DType::kBf16,
                            {48U, config->hidden_size});
            add_requirement(requirements,
                            prefix + "linear_attn.conv1d.weight",
                            st::DType::kBf16,
                            {10240U, 1U, 4U});
            add_requirement(requirements,
                            prefix + "linear_attn.A_log",
                            st::DType::kBf16,
                            {48U});
            add_requirement(requirements,
                            prefix + "linear_attn.dt_bias",
                            st::DType::kBf16,
                            {48U});
            add_requirement(requirements,
                            prefix + "linear_attn.norm.weight",
                            st::DType::kBf16,
                            {128U});
            for (const auto& projection : {
                     std::pair<std::string_view, std::pair<std::uint64_t,
                                                           std::uint64_t>>{
                         "linear_attn.in_proj_qkv", {10240U, 5120U}},
                     {"linear_attn.in_proj_z", {6144U, 5120U}},
                     {"linear_attn.out_proj", {5120U, 6144U}},
                 }) {
                if (auto error = add_quantized_requirement(
                        requirements,
                        quantization,
                        prefix + std::string(projection.first),
                        projection.second.first,
                        projection.second.second)) {
                    result.diagnostic = std::move(*error);
                    return result;
                }
            }
        } else if (config->layer_type(layer) == LayerType::kFullAttention) {
            add_requirement(requirements,
                            prefix + "self_attn.q_norm.weight",
                            st::DType::kBf16,
                            {config->head_dim});
            add_requirement(requirements,
                            prefix + "self_attn.k_norm.weight",
                            st::DType::kBf16,
                            {config->head_dim});
            for (const auto& projection : {
                     std::pair<std::string_view, std::pair<std::uint64_t,
                                                           std::uint64_t>>{
                         "self_attn.q_proj", {12288U, 5120U}},
                     {"self_attn.k_proj", {1024U, 5120U}},
                     {"self_attn.v_proj", {1024U, 5120U}},
                     {"self_attn.o_proj", {5120U, 6144U}},
                 }) {
                if (auto error = add_quantized_requirement(
                        requirements,
                        quantization,
                        prefix + std::string(projection.first),
                        projection.second.first,
                        projection.second.second)) {
                    result.diagnostic = std::move(*error);
                    return result;
                }
            }
        } else {
            result.diagnostic = make_diagnostic(
                ManifestErrorCode::kUnsupportedCheckpoint,
                prefix,
                "invalid hybrid-attention layer schedule");
            return result;
        }
    }

    if (auto error = add_quantized_requirement(requirements,
                                                quantization,
                                                "lm_head",
                                                config->vocab_size,
                                                config->hidden_size)) {
        result.diagnostic = std::move(*error);
        return result;
    }
    if (requirements.size() != kPinnedQwen36_27BTextTensorCount) {
        result.diagnostic = make_diagnostic(
            ManifestErrorCode::kCountMismatch,
            "generated text ABI",
            "generated Qwen3.6-27B tensor ABI has an unexpected size",
            std::to_string(kPinnedQwen36_27BTextTensorCount),
            std::to_string(requirements.size()));
        return result;
    }
    result.value.emplace(std::move(requirements));
    return result;
}

std::optional<ManifestDiagnostic> validate_locator_storage(
    const std::string& name,
    const TensorLocator& locator) {
    if (!st::is_safe_relative_shard_path(locator.shard)) {
        return make_diagnostic(
            ManifestErrorCode::kUnsafeShardPath,
            name,
            "tensor locator contains an unsafe shard path",
            "safe relative *.safetensors path",
            locator.shard);
    }
    if (locator.file.empty() || !locator.file.is_absolute()) {
        return make_diagnostic(
            ManifestErrorCode::kUnsafeShardPath,
            name,
            "tensor locator file path must be absolute",
            "absolute path",
            locator.file.string());
    }
    std::uint64_t computed_size = 0;
    if (!compute_byte_size(locator.dtype, locator.shape, computed_size)) {
        return make_diagnostic(
            ManifestErrorCode::kArithmeticOverflow,
            name,
            "tensor shape and dtype overflow a byte-size calculation");
    }
    if (locator.byte_size != computed_size) {
        return make_diagnostic(ManifestErrorCode::kByteSizeMismatch,
                               name,
                               "locator byte size disagrees with dtype and shape",
                               std::to_string(computed_size),
                               std::to_string(locator.byte_size));
    }
    if (locator.file_end < locator.file_begin ||
        locator.file_end - locator.file_begin != locator.byte_size) {
        return make_diagnostic(
            ManifestErrorCode::kOffsetMismatch,
            name,
            "absolute file offsets disagree with tensor byte size",
            std::to_string(locator.byte_size),
            "[" + std::to_string(locator.file_begin) + "," +
                std::to_string(locator.file_end) + ")");
    }
    return std::nullopt;
}

ManifestDiagnostic from_checkpoint_error(
    const checkpoint::InspectionResult& inspection) {
    if (inspection.diagnostics.empty()) {
        return make_diagnostic(ManifestErrorCode::kCheckpointRejected,
                               "checkpoint",
                               "pinned checkpoint inspection failed");
    }
    const checkpoint::Diagnostic& source = inspection.diagnostics.front();
    return make_diagnostic(ManifestErrorCode::kCheckpointRejected,
                           source.source + source.json_pointer,
                           source.message,
                           source.expected,
                           source.actual);
}

ManifestErrorCode code_for_safetensors_error(st::ErrorCode code,
                                             ManifestErrorCode fallback) {
    if (code == st::ErrorCode::kAllocationFailure) {
        return ManifestErrorCode::kAllocationFailure;
    }
    if (code == st::ErrorCode::kOpenFailed ||
        code == st::ErrorCode::kIoFailure ||
        code == st::ErrorCode::kShardMissing ||
        code == st::ErrorCode::kShardNotRegular) {
        return ManifestErrorCode::kIoFailure;
    }
    if (code == st::ErrorCode::kUnsafeShardPath) {
        return ManifestErrorCode::kUnsafeShardPath;
    }
    return fallback;
}

ManifestDiagnostic from_safetensors_error(const st::Error& error,
                                          ManifestErrorCode fallback) {
    return make_diagnostic(code_for_safetensors_error(error.code, fallback),
                           error.context,
                           std::string(st::to_string(error.code)),
                           error.expected,
                           error.actual);
}

std::optional<ManifestDiagnostic> require_pinned_artifact_totals(
    const WeightManifestSummary& summary) {
    constexpr std::size_t kExpectedTotalTensors =
        kPinnedQwen36_27BTextTensorCount +
        kPinnedQwen36_27BVisionTensorCount +
        kPinnedQwen36_27BMtpTensorCount;
    if (summary.shard_count != 3U ||
        summary.tensor_count != kExpectedTotalTensors ||
        summary.text_tensor_count != kPinnedQwen36_27BTextTensorCount ||
        summary.vision_tensor_count != kPinnedQwen36_27BVisionTensorCount ||
        summary.mtp_tensor_count != kPinnedQwen36_27BMtpTensorCount) {
        return make_diagnostic(
            ManifestErrorCode::kCountMismatch,
            "pinned artifact",
            "tensor category or shard counts differ from the pinned artifact",
            "shards=3,total=2194,text=1846,vision=333,mtp=15",
            "shards=" + std::to_string(summary.shard_count) +
                ",total=" + std::to_string(summary.tensor_count) +
                ",text=" + std::to_string(summary.text_tensor_count) +
                ",vision=" + std::to_string(summary.vision_tensor_count) +
                ",mtp=" + std::to_string(summary.mtp_tensor_count));
    }
    if (summary.raw_text_bytes != kPinnedQwen36_27BTextBytes ||
        summary.vision_bytes != kPinnedQwen36_27BVisionBytes ||
        summary.mtp_bytes != kPinnedQwen36_27BMtpBytes) {
        return make_diagnostic(
            ManifestErrorCode::kSizeMismatch,
            "pinned artifact",
            "tensor category bytes differ from the pinned artifact",
            "text=20150569096,vision=921460192,mtp=849398784",
            "text=" + std::to_string(summary.raw_text_bytes) +
                ",vision=" + std::to_string(summary.vision_bytes) +
                ",mtp=" + std::to_string(summary.mtp_bytes));
    }
    return std::nullopt;
}

}  // namespace

const TensorLocator* WeightManifest::find(std::string_view name) const noexcept {
    const auto iterator = tensors.find(name);
    return iterator == tensors.end() ? nullptr : &iterator->second;
}

TensorCategory classify_tensor(std::string_view name) noexcept {
    if (name.rfind("model.language_model.", 0) == 0 ||
        name.rfind("lm_head.", 0) == 0) {
        return TensorCategory::kText;
    }
    if (name.rfind("model.visual.", 0) == 0) {
        return TensorCategory::kVision;
    }
    if (name.rfind("mtp.", 0) == 0) {
        return TensorCategory::kMtp;
    }
    return TensorCategory::kUnknown;
}

ManifestResult validate_qwen36_27b_text_manifest(
    std::map<std::string, TensorLocator, std::less<>> tensors,
    const checkpoint::ModelOptSummary& quantization,
    std::uint64_t arena_alignment) {
    if (!is_power_of_two(arena_alignment)) {
        return failure(make_diagnostic(
            ManifestErrorCode::kInvalidOption,
            "arena_alignment",
            "arena alignment must be a non-zero power of two",
            "power of two",
            std::to_string(arena_alignment)));
    }
    try {
        RequirementResult requirement_result = make_requirements(quantization);
        if (!requirement_result.value) {
            return failure(std::move(requirement_result.diagnostic));
        }
        const RequirementMap& requirements = *requirement_result.value;

        WeightManifest manifest;
        manifest.summary.tensor_count = tensors.size();
        manifest.summary.fp8_module_count = quantization.fp8_count;
        manifest.summary.nvfp4_module_count = quantization.nvfp4_count;
        manifest.summary.arena_alignment = arena_alignment;
        std::set<std::string, std::less<>> shards;

        for (auto& tensor : tensors) {
            const std::string& name = tensor.first;
            TensorLocator& locator = tensor.second;
            if (auto error = validate_locator_storage(name, locator)) {
                return failure(std::move(*error));
            }
            locator.category = classify_tensor(name);
            shards.emplace(locator.shard);

            switch (locator.category) {
                case TensorCategory::kText: {
                    ++manifest.summary.text_tensor_count;
                    const auto expected = requirements.find(name);
                    if (expected == requirements.end()) {
                        return failure(make_diagnostic(
                            ManifestErrorCode::kUnexpectedTextTensor,
                            name,
                            "tensor is not part of the Qwen3.6-27B text ABI"));
                    }
                    if (locator.dtype != expected->second.dtype) {
                        return failure(make_diagnostic(
                            ManifestErrorCode::kDTypeMismatch,
                            name,
                            "tensor dtype differs from the runtime ABI",
                            std::string(st::to_string(expected->second.dtype)),
                            std::string(st::to_string(locator.dtype))));
                    }
                    if (locator.shape != expected->second.shape) {
                        return failure(make_diagnostic(
                            ManifestErrorCode::kShapeMismatch,
                            name,
                            "tensor shape differs from the runtime ABI",
                            shape_string(expected->second.shape),
                            shape_string(locator.shape)));
                    }
                    if (!checked_add(manifest.summary.raw_text_bytes,
                                     locator.byte_size,
                                     manifest.summary.raw_text_bytes)) {
                        return failure(make_diagnostic(
                            ManifestErrorCode::kArithmeticOverflow,
                            name,
                            "raw text byte total overflows uint64"));
                    }
                    std::uint64_t allocation_bytes = 0;
                    if (!checked_align_up(locator.byte_size,
                                          arena_alignment,
                                          allocation_bytes) ||
                        !checked_add(
                            manifest.summary.estimated_text_arena_bytes,
                            allocation_bytes,
                            manifest.summary.estimated_text_arena_bytes)) {
                        return failure(make_diagnostic(
                            ManifestErrorCode::kArithmeticOverflow,
                            name,
                            "aligned text arena size overflows uint64"));
                    }
                    break;
                }
                case TensorCategory::kVision:
                    ++manifest.summary.vision_tensor_count;
                    if (!checked_add(manifest.summary.vision_bytes,
                                     locator.byte_size,
                                     manifest.summary.vision_bytes)) {
                        return failure(make_diagnostic(
                            ManifestErrorCode::kArithmeticOverflow,
                            name,
                            "vision byte total overflows uint64"));
                    }
                    break;
                case TensorCategory::kMtp:
                    ++manifest.summary.mtp_tensor_count;
                    if (!checked_add(manifest.summary.mtp_bytes,
                                     locator.byte_size,
                                     manifest.summary.mtp_bytes)) {
                        return failure(make_diagnostic(
                            ManifestErrorCode::kArithmeticOverflow,
                            name,
                            "MTP byte total overflows uint64"));
                    }
                    break;
                case TensorCategory::kUnknown:
                    return failure(make_diagnostic(
                        ManifestErrorCode::kUnknownTensor,
                        name,
                        "tensor is outside text, vision, and MTP namespaces"));
            }
        }

        for (const auto& requirement : requirements) {
            if (tensors.find(requirement.first) == tensors.end()) {
                return failure(make_diagnostic(
                    ManifestErrorCode::kMissingTensor,
                    requirement.first,
                    "required Qwen3.6-27B text tensor is missing"));
            }
        }
        if (manifest.summary.text_tensor_count != requirements.size()) {
            return failure(make_diagnostic(
                ManifestErrorCode::kCountMismatch,
                "text tensors",
                "text tensor count differs from the complete runtime ABI",
                std::to_string(requirements.size()),
                std::to_string(manifest.summary.text_tensor_count)));
        }
        if (manifest.summary.raw_text_bytes != kPinnedQwen36_27BTextBytes) {
            return failure(make_diagnostic(
                ManifestErrorCode::kSizeMismatch,
                "text tensors",
                "validated text bytes differ from the pinned runtime ABI",
                std::to_string(kPinnedQwen36_27BTextBytes),
                std::to_string(manifest.summary.raw_text_bytes)));
        }
        if (!checked_add(manifest.summary.vision_bytes,
                         manifest.summary.mtp_bytes,
                         manifest.summary.skipped_bytes)) {
            return failure(make_diagnostic(
                ManifestErrorCode::kArithmeticOverflow,
                "skipped tensors",
                "skipped byte total overflows uint64"));
        }
        manifest.summary.shard_count = shards.size();
        manifest.tensors = std::move(tensors);

        ManifestResult result;
        result.value.emplace(std::move(manifest));
        return result;
    } catch (const std::bad_alloc&) {
        return failure(make_diagnostic(ManifestErrorCode::kAllocationFailure,
                                       "manifest",
                                       "allocation failed during ABI validation"));
    } catch (const std::length_error&) {
        return failure(make_diagnostic(ManifestErrorCode::kAllocationFailure,
                                       "manifest",
                                       "container size exceeded during ABI validation"));
    }
}

ManifestResult build_qwen36_27b_text_manifest(
    const std::filesystem::path& directory,
    const ManifestOptions& options) {
    if (!is_power_of_two(options.arena_alignment) ||
        options.index_options.max_file_bytes == 0U ||
        options.index_options.max_tensors == 0U ||
        options.index_options.max_shards == 0U ||
        options.shard_options.max_shards == 0U ||
        options.shard_options.max_tensors == 0U) {
        return failure(make_diagnostic(
            ManifestErrorCode::kInvalidOption,
            "manifest options",
            "manifest limits must be non-zero and alignment a power of two"));
    }
    try {
        checkpoint::InspectionOptions inspection_options;
        inspection_options.max_index_bytes =
            options.index_options.max_file_bytes;
        inspection_options.max_index_tensors =
            options.index_options.max_tensors;
        inspection_options.max_index_shards =
            options.index_options.max_shards;
        const checkpoint::InspectionResult inspection =
            checkpoint::inspect_directory(directory, inspection_options);
        if (!inspection) {
            return failure(from_checkpoint_error(inspection));
        }
        if (inspection.report->checkpoint.model != KnownModel::kQwen36_27B) {
            return failure(make_diagnostic(
                ManifestErrorCode::kUnsupportedCheckpoint,
                inspection.report->checkpoint.id,
                "weight manifest currently supports only pinned Qwen3.6-27B",
                std::string(to_string(KnownModel::kQwen36_27B)),
                std::string(to_string(inspection.report->checkpoint.model))));
        }

        const std::filesystem::path index_path =
            directory / "model.safetensors.index.json";
        st::Result<st::Index> index =
            st::read_index(index_path.string(), options.index_options);
        if (!index) {
            return failure(from_safetensors_error(
                index.error, ManifestErrorCode::kIndexRejected));
        }
        st::Result<st::ShardValidationSummary> shard_validation =
            st::validate_index_shards(
                directory, *index.value, options.shard_options);
        if (!shard_validation) {
            return failure(from_safetensors_error(
                shard_validation.error, ManifestErrorCode::kShardRejected));
        }

        std::map<std::string, TensorLocator, std::less<>> locators;
        for (const std::string& shard : index.value->shards) {
            if (!st::is_safe_relative_shard_path(shard)) {
                return failure(make_diagnostic(
                    ManifestErrorCode::kUnsafeShardPath,
                    shard,
                    "index contains an unsafe shard path"));
            }
            const std::filesystem::path relative_path(shard);
            const std::filesystem::path shard_path = directory / relative_path;
            std::error_code filesystem_error;
            std::filesystem::path absolute_path =
                std::filesystem::absolute(shard_path, filesystem_error)
                    .lexically_normal();
            if (filesystem_error || !absolute_path.is_absolute()) {
                return failure(make_diagnostic(
                    ManifestErrorCode::kIoFailure,
                    shard,
                    "could not resolve an absolute shard path",
                    {},
                    filesystem_error.message()));
            }
            st::Result<st::Header> header = st::read_header(
                shard_path.string(), options.shard_options.header_options);
            if (!header) {
                return failure(from_safetensors_error(
                    header.error, ManifestErrorCode::kShardRejected));
            }
            for (const auto& tensor : header.value->tensors) {
                const std::string* indexed_shard =
                    index.value->shard_for(tensor.first);
                if (indexed_shard == nullptr || *indexed_shard != shard) {
                    return failure(make_diagnostic(
                        ManifestErrorCode::kIndexRejected,
                        tensor.first,
                        "tensor locator disagrees with index shard ownership",
                        indexed_shard == nullptr ? "present in index"
                                                 : *indexed_shard,
                        shard));
                }
                TensorLocator locator;
                locator.shard = shard;
                locator.file = absolute_path;
                locator.file_begin = tensor.second.file_begin;
                locator.file_end = tensor.second.file_end;
                locator.byte_size = tensor.second.byte_size;
                locator.dtype = tensor.second.dtype;
                locator.shape = tensor.second.shape;
                if (!locators.emplace(tensor.first, std::move(locator)).second) {
                    return failure(make_diagnostic(
                        ManifestErrorCode::kDuplicateTensor,
                        tensor.first,
                        "tensor occurs in more than one shard"));
                }
            }
        }
        if (locators.size() != index.value->weight_map.size()) {
            return failure(make_diagnostic(
                ManifestErrorCode::kCountMismatch,
                "index",
                "locator count differs from index weight_map",
                std::to_string(index.value->weight_map.size()),
                std::to_string(locators.size())));
        }

        ManifestResult result = validate_qwen36_27b_text_manifest(
            std::move(locators),
            inspection.report->quantization,
            options.arena_alignment);
        if (!result) {
            return result;
        }
        result.value->checkpoint = inspection.report->checkpoint;
        if (auto error = require_pinned_artifact_totals(result.value->summary)) {
            return failure(std::move(*error));
        }
        if (result.value->summary.tensor_count !=
                shard_validation.value->validated_tensors ||
            result.value->summary.shard_count !=
                shard_validation.value->validated_shards) {
            return failure(make_diagnostic(
                ManifestErrorCode::kCountMismatch,
                "shard validation",
                "manifest and strict shard validator counts disagree"));
        }
        std::uint64_t classified_bytes = 0;
        if (!checked_add(result.value->summary.raw_text_bytes,
                         result.value->summary.skipped_bytes,
                         classified_bytes) ||
            classified_bytes != shard_validation.value->payload_bytes) {
            return failure(make_diagnostic(
                ManifestErrorCode::kSizeMismatch,
                "shard validation",
                "classified tensor bytes disagree with shard payload bytes",
                std::to_string(shard_validation.value->payload_bytes),
                std::to_string(classified_bytes)));
        }
        return result;
    } catch (const std::bad_alloc&) {
        return failure(make_diagnostic(ManifestErrorCode::kAllocationFailure,
                                       "manifest",
                                       "allocation failed while reading headers"));
    } catch (const std::length_error&) {
        return failure(make_diagnostic(ManifestErrorCode::kAllocationFailure,
                                       "manifest",
                                       "container size exceeded while reading headers"));
    } catch (const std::filesystem::filesystem_error& error) {
        return failure(make_diagnostic(ManifestErrorCode::kIoFailure,
                                       directory.string(),
                                       "filesystem operation failed",
                                       {},
                                       error.what()));
    }
}

std::string_view to_string(TensorCategory category) noexcept {
    switch (category) {
        case TensorCategory::kText:
            return "text";
        case TensorCategory::kVision:
            return "vision";
        case TensorCategory::kMtp:
            return "mtp";
        case TensorCategory::kUnknown:
            return "unknown";
    }
    return "unknown";
}

std::string_view to_string(ManifestErrorCode code) noexcept {
    switch (code) {
        case ManifestErrorCode::kNone:
            return "none";
        case ManifestErrorCode::kInvalidOption:
            return "invalid-option";
        case ManifestErrorCode::kCheckpointRejected:
            return "checkpoint-rejected";
        case ManifestErrorCode::kUnsupportedCheckpoint:
            return "unsupported-checkpoint";
        case ManifestErrorCode::kIndexRejected:
            return "index-rejected";
        case ManifestErrorCode::kShardRejected:
            return "shard-rejected";
        case ManifestErrorCode::kUnsafeShardPath:
            return "unsafe-shard-path";
        case ManifestErrorCode::kDuplicateTensor:
            return "duplicate-tensor";
        case ManifestErrorCode::kUnknownTensor:
            return "unknown-tensor";
        case ManifestErrorCode::kUnexpectedTextTensor:
            return "unexpected-text-tensor";
        case ManifestErrorCode::kMissingTensor:
            return "missing-tensor";
        case ManifestErrorCode::kDTypeMismatch:
            return "dtype-mismatch";
        case ManifestErrorCode::kShapeMismatch:
            return "shape-mismatch";
        case ManifestErrorCode::kByteSizeMismatch:
            return "byte-size-mismatch";
        case ManifestErrorCode::kOffsetMismatch:
            return "offset-mismatch";
        case ManifestErrorCode::kQuantizationMismatch:
            return "quantization-mismatch";
        case ManifestErrorCode::kCountMismatch:
            return "count-mismatch";
        case ManifestErrorCode::kSizeMismatch:
            return "size-mismatch";
        case ManifestErrorCode::kArithmeticOverflow:
            return "arithmetic-overflow";
        case ManifestErrorCode::kIoFailure:
            return "io-failure";
        case ManifestErrorCode::kAllocationFailure:
            return "allocation-failure";
    }
    return "unknown";
}

}  // namespace q3x::model::weights
