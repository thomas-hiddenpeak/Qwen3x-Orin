#include "q3x/model/weight_manifest.h"

#include "q3x/model/checkpoint_metadata.h"
#include "q3x/model/model_config.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace checkpoint = q3x::model::checkpoint;
namespace weights = q3x::model::weights;
namespace st = q3x::io::safetensors;

class TestContext {
public:
    void expect(bool condition, std::string_view message) {
        if (!condition) {
            ++failures_;
            std::cerr << "FAILED: " << message << '\n';
        }
    }

    [[nodiscard]] int failures() const noexcept { return failures_; }

private:
    int failures_ = 0;
};

bool has_code(const weights::ManifestResult& result,
              weights::ManifestErrorCode code) {
    for (const auto& diagnostic : result.diagnostics) {
        if (diagnostic.code == code) {
            return true;
        }
    }
    return false;
}

checkpoint::ModelOptSummary make_quantization() {
    const auto& descriptor = checkpoint::known_checkpoint_catalog()[0];
    checkpoint::ModelOptSummary summary;
    summary.producer_name = "modelopt";
    summary.producer_version = descriptor.modelopt_version;
    summary.quantization_algorithm = "MIXED_PRECISION";
    summary.kv_cache_quantization_algorithm = "FP8";
    summary.exclude_modules = {"mtp*", "mtp.layers.0*"};
    summary.quantized_modules = checkpoint::expected_quantized_modules(
        q3x::model::KnownModel::kQwen36_27B);
    for (const auto& item : summary.quantized_modules) {
        if (item.second.algorithm == checkpoint::QuantAlgorithm::kFp8) {
            ++summary.fp8_count;
        } else {
            ++summary.nvfp4_count;
        }
    }
    return summary;
}

std::uint64_t byte_size(st::DType dtype,
                        const std::vector<std::uint64_t>& shape) {
    std::uint64_t elements = 1;
    for (const std::uint64_t dimension : shape) {
        if (dimension != 0U &&
            elements > std::numeric_limits<std::uint64_t>::max() / dimension) {
            throw std::runtime_error("synthetic tensor size overflow");
        }
        elements *= dimension;
    }
    const std::uint64_t bits =
        elements * static_cast<std::uint64_t>(st::bit_width(dtype));
    return bits / 8U;
}

void reset_storage(weights::TensorLocator& locator,
                   std::uint64_t file_begin) {
    locator.byte_size = byte_size(locator.dtype, locator.shape);
    locator.file_begin = file_begin;
    locator.file_end = file_begin + locator.byte_size;
}

void add_tensor(
    std::map<std::string, weights::TensorLocator, std::less<>>& tensors,
    std::uint64_t& cursor,
    std::string name,
    st::DType dtype,
    std::vector<std::uint64_t> shape) {
    weights::TensorLocator locator;
    locator.shard = "synthetic.safetensors";
    locator.file = "/synthetic.safetensors";
    locator.dtype = dtype;
    locator.shape = std::move(shape);
    reset_storage(locator, cursor);
    cursor = locator.file_end;
    if (!tensors.emplace(std::move(name), std::move(locator)).second) {
        throw std::runtime_error("duplicate synthetic tensor");
    }
}

void add_quantized(
    std::map<std::string, weights::TensorLocator, std::less<>>& tensors,
    std::uint64_t& cursor,
    const checkpoint::ModelOptSummary& quantization,
    const std::string& module,
    std::uint64_t output_size,
    std::uint64_t input_size) {
    const auto selected = quantization.quantized_modules.find(module);
    if (selected == quantization.quantized_modules.end()) {
        throw std::runtime_error("synthetic module is absent from ModelOpt map");
    }
    if (selected->second.algorithm == checkpoint::QuantAlgorithm::kFp8) {
        add_tensor(tensors,
                   cursor,
                   module + ".weight",
                   st::DType::kF8E4M3,
                   {output_size, input_size});
        add_tensor(tensors,
                   cursor,
                   module + ".weight_scale",
                   st::DType::kF32,
                   {});
        add_tensor(tensors,
                   cursor,
                   module + ".input_scale",
                   st::DType::kF32,
                   {});
    } else {
        add_tensor(tensors,
                   cursor,
                   module + ".weight",
                   st::DType::kU8,
                   {output_size, input_size / 2U});
        add_tensor(tensors,
                   cursor,
                   module + ".weight_scale",
                   st::DType::kF8E4M3,
                   {output_size, input_size / 16U});
        add_tensor(tensors,
                   cursor,
                   module + ".weight_scale_2",
                   st::DType::kF32,
                   {});
        add_tensor(tensors,
                   cursor,
                   module + ".input_scale",
                   st::DType::kF32,
                   {});
    }
}

std::map<std::string, weights::TensorLocator, std::less<>>
make_text_locators(const checkpoint::ModelOptSummary& quantization) {
    const q3x::model::ModelConfig* config = q3x::model::find_known_model(
        q3x::model::KnownModel::kQwen36_27B);
    if (config == nullptr) {
        throw std::runtime_error("Qwen3.6-27B model config is unavailable");
    }
    std::map<std::string, weights::TensorLocator, std::less<>> tensors;
    std::uint64_t cursor = 4096;
    add_tensor(tensors,
               cursor,
               "model.language_model.embed_tokens.weight",
               st::DType::kBf16,
               {config->vocab_size, config->hidden_size});
    add_tensor(tensors,
               cursor,
               "model.language_model.norm.weight",
               st::DType::kBf16,
               {config->hidden_size});

    for (std::uint32_t layer = 0; layer < config->num_hidden_layers; ++layer) {
        const std::string prefix =
            "model.language_model.layers." + std::to_string(layer) + ".";
        add_tensor(tensors,
                   cursor,
                   prefix + "input_layernorm.weight",
                   st::DType::kBf16,
                   {config->hidden_size});
        add_tensor(tensors,
                   cursor,
                   prefix + "post_attention_layernorm.weight",
                   st::DType::kBf16,
                   {config->hidden_size});
        add_quantized(tensors,
                      cursor,
                      quantization,
                      prefix + "mlp.gate_proj",
                      config->intermediate_size,
                      config->hidden_size);
        add_quantized(tensors,
                      cursor,
                      quantization,
                      prefix + "mlp.up_proj",
                      config->intermediate_size,
                      config->hidden_size);
        add_quantized(tensors,
                      cursor,
                      quantization,
                      prefix + "mlp.down_proj",
                      config->hidden_size,
                      config->intermediate_size);

        if (config->layer_type(layer) ==
            q3x::model::LayerType::kLinearAttention) {
            add_tensor(tensors,
                       cursor,
                       prefix + "linear_attn.in_proj_a.weight",
                       st::DType::kBf16,
                       {48U, config->hidden_size});
            add_tensor(tensors,
                       cursor,
                       prefix + "linear_attn.in_proj_b.weight",
                       st::DType::kBf16,
                       {48U, config->hidden_size});
            add_tensor(tensors,
                       cursor,
                       prefix + "linear_attn.conv1d.weight",
                       st::DType::kBf16,
                       {10240U, 1U, 4U});
            add_tensor(tensors,
                       cursor,
                       prefix + "linear_attn.A_log",
                       st::DType::kBf16,
                       {48U});
            add_tensor(tensors,
                       cursor,
                       prefix + "linear_attn.dt_bias",
                       st::DType::kBf16,
                       {48U});
            add_tensor(tensors,
                       cursor,
                       prefix + "linear_attn.norm.weight",
                       st::DType::kBf16,
                       {128U});
            add_quantized(tensors,
                          cursor,
                          quantization,
                          prefix + "linear_attn.in_proj_qkv",
                          10240U,
                          5120U);
            add_quantized(tensors,
                          cursor,
                          quantization,
                          prefix + "linear_attn.in_proj_z",
                          6144U,
                          5120U);
            add_quantized(tensors,
                          cursor,
                          quantization,
                          prefix + "linear_attn.out_proj",
                          5120U,
                          6144U);
        } else {
            add_tensor(tensors,
                       cursor,
                       prefix + "self_attn.q_norm.weight",
                       st::DType::kBf16,
                       {256U});
            add_tensor(tensors,
                       cursor,
                       prefix + "self_attn.k_norm.weight",
                       st::DType::kBf16,
                       {256U});
            add_quantized(tensors,
                          cursor,
                          quantization,
                          prefix + "self_attn.q_proj",
                          12288U,
                          5120U);
            add_quantized(tensors,
                          cursor,
                          quantization,
                          prefix + "self_attn.k_proj",
                          1024U,
                          5120U);
            add_quantized(tensors,
                          cursor,
                          quantization,
                          prefix + "self_attn.v_proj",
                          1024U,
                          5120U);
            add_quantized(tensors,
                          cursor,
                          quantization,
                          prefix + "self_attn.o_proj",
                          5120U,
                          6144U);
        }
    }
    add_quantized(tensors,
                  cursor,
                  quantization,
                  "lm_head",
                  config->vocab_size,
                  config->hidden_size);
    return tensors;
}

void test_valid_synthetic_manifest(TestContext& test) {
    const checkpoint::ModelOptSummary quantization = make_quantization();
    const auto locators = make_text_locators(quantization);
    test.expect(locators.size() ==
                    weights::kPinnedQwen36_27BTextTensorCount,
                "synthetic ABI contains all 1846 text tensors");
    const weights::ManifestResult result =
        weights::validate_qwen36_27b_text_manifest(locators, quantization);
    test.expect(result.ok(), "complete synthetic text ABI validates");
    if (!result) {
        return;
    }
    test.expect(result.value->summary.text_tensor_count == 1846 &&
                    result.value->summary.fp8_module_count == 208 &&
                    result.value->summary.nvfp4_module_count == 193,
                "manifest reports exact tensor and quantization counts");
    test.expect(result.value->summary.raw_text_bytes == 20150569096ULL,
                "manifest reports exact raw text bytes");
    test.expect(result.value->summary.arena_alignment == 256 &&
                    result.value->summary.estimated_text_arena_bytes ==
                        20150786560ULL,
                "manifest reports checked 256-byte arena estimate");
    const weights::TensorLocator* qkv = result.value->find(
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight");
    test.expect(qkv != nullptr && qkv->category == weights::TensorCategory::kText &&
                    qkv->dtype == st::DType::kF8E4M3 &&
                    qkv->shape == std::vector<std::uint64_t>({10240U, 5120U}) &&
                    qkv->file_end > qkv->file_begin,
                "representative FP8 locator retains ABI and absolute offsets");
}

void test_shape_dtype_and_companion_failures(TestContext& test) {
    const checkpoint::ModelOptSummary quantization = make_quantization();
    const auto source = make_text_locators(quantization);

    auto wrong_shape = source;
    auto& embedding =
        wrong_shape.at("model.language_model.embed_tokens.weight");
    embedding.shape[0] -= 1U;
    reset_storage(embedding, embedding.file_begin);
    auto result = weights::validate_qwen36_27b_text_manifest(
        std::move(wrong_shape), quantization);
    test.expect(!result && has_code(result, weights::ManifestErrorCode::kShapeMismatch),
                "incorrect mandatory tensor shape is rejected");

    auto wrong_dtype = source;
    auto& a_log = wrong_dtype.at(
        "model.language_model.layers.0.linear_attn.A_log");
    a_log.dtype = st::DType::kF32;
    reset_storage(a_log, a_log.file_begin);
    result = weights::validate_qwen36_27b_text_manifest(
        std::move(wrong_dtype), quantization);
    test.expect(!result && has_code(result, weights::ManifestErrorCode::kDTypeMismatch),
                "pinned BF16 A_log dtype is enforced");

    auto missing_scale = source;
    missing_scale.erase("lm_head.input_scale");
    result = weights::validate_qwen36_27b_text_manifest(
        std::move(missing_scale), quantization);
    test.expect(!result && has_code(result, weights::ManifestErrorCode::kMissingTensor),
                "missing quantization companion is rejected");
}

void test_extra_unknown_and_overflow_failures(TestContext& test) {
    const checkpoint::ModelOptSummary quantization = make_quantization();
    const auto source = make_text_locators(quantization);

    auto extra_text = source;
    std::uint64_t cursor = 1;
    add_tensor(extra_text,
               cursor,
               "model.language_model.extra.weight",
               st::DType::kBf16,
               {1U});
    auto result = weights::validate_qwen36_27b_text_manifest(
        std::move(extra_text), quantization);
    test.expect(!result &&
                    has_code(result,
                             weights::ManifestErrorCode::kUnexpectedTextTensor),
                "additional text tensor is rejected");

    auto unknown = source;
    cursor = 1;
    add_tensor(unknown,
               cursor,
               "unscoped.weight",
               st::DType::kBf16,
               {1U});
    result = weights::validate_qwen36_27b_text_manifest(
        std::move(unknown), quantization);
    test.expect(!result && has_code(result, weights::ManifestErrorCode::kUnknownTensor),
                "tensor outside explicit namespaces is rejected");

    auto overflow = source;
    auto& embedding = overflow.at("model.language_model.embed_tokens.weight");
    embedding.shape = {std::numeric_limits<std::uint64_t>::max(),
                       std::numeric_limits<std::uint64_t>::max()};
    result = weights::validate_qwen36_27b_text_manifest(
        std::move(overflow), quantization);
    test.expect(!result &&
                    has_code(result,
                             weights::ManifestErrorCode::kArithmeticOverflow),
                "malicious tensor size overflow is rejected before ABI comparison");
}

void test_skip_categories_and_options(TestContext& test) {
    const checkpoint::ModelOptSummary quantization = make_quantization();
    auto locators = make_text_locators(quantization);
    std::uint64_t cursor = 100;
    add_tensor(locators,
               cursor,
               "model.visual.synthetic.weight",
               st::DType::kBf16,
               {2U});
    add_tensor(locators,
               cursor,
               "mtp.synthetic.weight",
               st::DType::kBf16,
               {3U});
    auto result = weights::validate_qwen36_27b_text_manifest(
        std::move(locators), quantization);
    test.expect(result.ok() && result.value->summary.vision_tensor_count == 1 &&
                    result.value->summary.mtp_tensor_count == 1 &&
                    result.value->summary.skipped_bytes == 10,
                "vision and MTP tensors are classified and skipped explicitly");

    auto unsafe = make_text_locators(quantization);
    unsafe.begin()->second.shard = "../escape.safetensors";
    result = weights::validate_qwen36_27b_text_manifest(
        std::move(unsafe), quantization);
    test.expect(!result &&
                    has_code(result,
                             weights::ManifestErrorCode::kUnsafeShardPath),
                "unsafe synthetic shard path is rejected");

    result = weights::validate_qwen36_27b_text_manifest(
        make_text_locators(quantization), quantization, 3U);
    test.expect(!result && has_code(result, weights::ManifestErrorCode::kInvalidOption),
                "non-power-of-two arena alignment is rejected");

    checkpoint::ModelOptSummary wrong_quantization = quantization;
    wrong_quantization.producer_version = "0.0.0";
    result = weights::validate_qwen36_27b_text_manifest(
        make_text_locators(quantization), wrong_quantization);
    test.expect(!result &&
                    has_code(result,
                             weights::ManifestErrorCode::kQuantizationMismatch),
                "manifest refuses a non-pinned ModelOpt table");
}

void test_classification(TestContext& test) {
    test.expect(weights::classify_tensor("model.language_model.norm.weight") ==
                    weights::TensorCategory::kText &&
                    weights::classify_tensor("lm_head.weight") ==
                        weights::TensorCategory::kText,
                "language model and lm_head classify as text");
    test.expect(weights::classify_tensor("model.visual.patch_embed.weight") ==
                    weights::TensorCategory::kVision &&
                    weights::classify_tensor("mtp.layers.0.norm.weight") ==
                        weights::TensorCategory::kMtp,
                "vision and MTP namespaces classify explicitly");
    test.expect(weights::classify_tensor("model.unknown.weight") ==
                    weights::TensorCategory::kUnknown,
                "unrecognized namespace remains unknown");
}

void test_official_checkpoint_if_available(TestContext& test) {
    const char* root = std::getenv("Q3X_OFFICIAL_CHECKPOINT_ROOT");
    if (root == nullptr || *root == '\0') {
        return;
    }
    const weights::ManifestResult result =
        weights::build_qwen36_27b_text_manifest(root);
    test.expect(result.ok(), "official pinned checkpoint manifest validates");
    if (!result) {
        for (const auto& diagnostic : result.diagnostics) {
            std::cerr << "official diagnostic: "
                      << weights::to_string(diagnostic.code) << " context="
                      << diagnostic.context << " message=" << diagnostic.message
                      << " expected=" << diagnostic.expected
                      << " actual=" << diagnostic.actual << '\n';
        }
        return;
    }
    const auto& summary = result.value->summary;
    test.expect(summary.shard_count == 3 && summary.tensor_count == 2194 &&
                    summary.text_tensor_count == 1846 &&
                    summary.vision_tensor_count == 333 &&
                    summary.mtp_tensor_count == 15,
                "official category and shard counts match pinned evidence");
    test.expect(summary.raw_text_bytes == 20150569096ULL &&
                    summary.vision_bytes == 921460192ULL &&
                    summary.mtp_bytes == 849398784ULL &&
                    summary.skipped_bytes == 1770858976ULL,
                "official category byte totals match pinned evidence");
    test.expect(summary.fp8_module_count == 208 &&
                    summary.nvfp4_module_count == 193 &&
                    summary.estimated_text_arena_bytes == 20150786560ULL,
                "official quantization counts and arena estimate match");
    const weights::TensorLocator* embedding = result.value->find(
        "model.language_model.embed_tokens.weight");
    test.expect(embedding != nullptr && embedding->file.is_absolute() &&
                    embedding->file_begin > 8U &&
                    embedding->file_end - embedding->file_begin ==
                        embedding->byte_size,
                "official locator exposes an absolute file range");
}

}  // namespace

int main() {
    TestContext test;
    try {
        test_valid_synthetic_manifest(test);
        test_shape_dtype_and_companion_failures(test);
        test_extra_unknown_and_overflow_failures(test);
        test_skip_categories_and_options(test);
        test_classification(test);
        test_official_checkpoint_if_available(test);
    } catch (const std::exception& error) {
        std::cerr << "FAILED: unexpected exception: " << error.what() << '\n';
        return 1;
    }
    if (test.failures() != 0) {
        std::cerr << test.failures() << " weight manifest test(s) failed\n";
        return 1;
    }
    std::cout << "All weight manifest tests passed\n";
    return 0;
}
