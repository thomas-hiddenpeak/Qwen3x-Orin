#include "q3x/model/checkpoint_metadata.h"

#include "q3x/io/json.h"
#include "q3x/io/safetensors.h"
#include "q3x/model/model_config.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

namespace checkpoint = q3x::model::checkpoint;

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

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        const auto stamp = std::chrono::steady_clock::now()
                               .time_since_epoch()
                               .count();
        const std::filesystem::path base =
            std::filesystem::temp_directory_path();
        for (std::size_t attempt = 0; attempt < 100; ++attempt) {
            path_ = base / ("q3x-checkpoint-metadata-test-" +
                            std::to_string(stamp) + "-" +
                            std::to_string(attempt));
            std::error_code error;
            if (std::filesystem::create_directory(path_, error)) {
                return;
            }
        }
        throw std::runtime_error("could not create test directory");
    }

    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

bool write_file(const std::filesystem::path& path, std::string_view contents) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    return static_cast<bool>(output);
}

std::string safetensors_prefix(const std::uint64_t header_size) {
    std::string prefix(8, '\0');
    for (std::size_t index = 0; index < prefix.size(); ++index) {
        prefix[index] = static_cast<char>(
            (header_size >> (index * 8U)) & UINT64_C(0xFF));
    }
    return prefix;
}

void append_json_string(std::string& output, const std::string_view value) {
    constexpr char kHex[] = "0123456789abcdef";
    output.push_back('"');
    for (const char character : value) {
        const auto byte = static_cast<unsigned char>(character);
        switch (byte) {
            case '"':
                output += "\\\"";
                break;
            case '\\':
                output += "\\\\";
                break;
            case '\b':
                output += "\\b";
                break;
            case '\f':
                output += "\\f";
                break;
            case '\n':
                output += "\\n";
                break;
            case '\r':
                output += "\\r";
                break;
            case '\t':
                output += "\\t";
                break;
            default:
                if (byte < 0x20U) {
                    output += "\\u00";
                    output.push_back(kHex[(byte >> 4U) & 0x0FU]);
                    output.push_back(kHex[byte & 0x0FU]);
                } else {
                    output.push_back(character);
                }
                break;
        }
    }
    output.push_back('"');
}

bool write_sparse_safetensors(const std::filesystem::path& path,
                              const std::string_view header,
                              const std::uint64_t payload_bytes) {
    const auto header_bytes = static_cast<std::uint64_t>(header.size());
    if (header_bytes > std::numeric_limits<std::uint64_t>::max() - 8U ||
        payload_bytes > std::numeric_limits<std::uint64_t>::max() - 8U -
                            header_bytes) {
        return false;
    }
    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
    if (error) {
        return false;
    }
    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        if (!output) {
            return false;
        }
        const std::string prefix = safetensors_prefix(header_bytes);
        output.write(prefix.data(), static_cast<std::streamsize>(prefix.size()));
        output.write(header.data(), static_cast<std::streamsize>(header.size()));
        if (!output) {
            return false;
        }
    }
    std::filesystem::resize_file(
        path,
        static_cast<std::uintmax_t>(8U + header_bytes + payload_bytes),
        error);
    return !error;
}

struct SyntheticShardOptions {
    std::optional<std::string> omitted_tensor;
    std::optional<std::pair<std::string, std::string>> moved_tensor;
    bool add_extra_tensor = false;
    std::optional<std::uint64_t> payload_bytes;
};

bool write_synthetic_shards(
    const std::filesystem::path& directory,
    const q3x::io::safetensors::Index& index,
    const SyntheticShardOptions& options = {}) {
    if (index.weight_map.empty() || index.shards.empty() ||
        !index.total_size.has_value()) {
        return false;
    }

    const std::string& payload_tensor = index.weight_map.begin()->first;
    const std::string& payload_shard = index.weight_map.begin()->second;
    if ((options.omitted_tensor &&
         *options.omitted_tensor == payload_tensor) ||
        (options.moved_tensor &&
         options.moved_tensor->first == payload_tensor)) {
        return false;
    }

    std::map<std::string, std::vector<std::string>, std::less<>> shard_tensors;
    for (const std::string& shard : index.shards) {
        shard_tensors.try_emplace(shard);
    }
    for (const auto& mapping : index.weight_map) {
        if (options.omitted_tensor &&
            mapping.first == *options.omitted_tensor) {
            continue;
        }
        const std::string* actual_shard = &mapping.second;
        if (options.moved_tensor &&
            mapping.first == options.moved_tensor->first) {
            actual_shard = &options.moved_tensor->second;
        }
        const auto shard = shard_tensors.find(*actual_shard);
        if (shard == shard_tensors.end()) {
            return false;
        }
        shard->second.emplace_back(mapping.first);
    }

    constexpr std::string_view kExtraTensor =
        "__q3x_strict_contract_test_extra__";
    if (options.add_extra_tensor) {
        if (index.weight_map.find(kExtraTensor) != index.weight_map.end()) {
            return false;
        }
        shard_tensors.begin()->second.emplace_back(kExtraTensor);
    }

    const std::uint64_t payload_bytes =
        options.payload_bytes.value_or(*index.total_size);
    for (const std::string& shard : index.shards) {
        const auto tensors = shard_tensors.find(shard);
        if (tensors == shard_tensors.end()) {
            return false;
        }
        std::string header;
        header.reserve(tensors->second.size() * 96U + 2U);
        header.push_back('{');
        bool first = true;
        for (const std::string& tensor : tensors->second) {
            if (!first) {
                header.push_back(',');
            }
            first = false;
            append_json_string(header, tensor);
            const bool owns_payload = tensor == payload_tensor;
            header += ":{\"dtype\":\"U8\",\"shape\":[";
            header += owns_payload ? std::to_string(payload_bytes) : "0";
            header += "],\"data_offsets\":[0,";
            header += owns_payload ? std::to_string(payload_bytes) : "0";
            header += "]}";
        }
        header.push_back('}');
        const std::uint64_t shard_payload =
            shard == payload_shard ? payload_bytes : 0U;
        if (!write_sparse_safetensors(directory / shard,
                                      header,
                                      shard_payload)) {
            return false;
        }
    }
    return true;
}

bool copy_metadata_files(const std::filesystem::path& source,
                         const std::filesystem::path& destination) {
    for (const std::string_view name : {
             "config.json",
             "hf_quant_config.json",
             "model.safetensors.index.json"}) {
        std::error_code error;
        std::filesystem::copy_file(source / name,
                                   destination / name,
                                   std::filesystem::copy_options::overwrite_existing,
                                   error);
        if (error) {
            return false;
        }
    }
    return true;
}

std::optional<q3x::io::json::Value> parse_document(std::string_view text,
                                                    TestContext& test,
                                                    std::string_view label) {
    q3x::io::json::ParseResult parsed = q3x::io::json::parse(text);
    test.expect(parsed.ok(), label);
    if (!parsed) {
        return std::nullopt;
    }
    return std::move(parsed.value);
}

bool has_code(const std::vector<checkpoint::Diagnostic>& diagnostics,
              checkpoint::DiagnosticCode code) {
    for (const checkpoint::Diagnostic& diagnostic : diagnostics) {
        if (diagnostic.code == code) {
            return true;
        }
    }
    return false;
}

std::string make_hf_config(q3x::model::KnownModel model,
                           std::optional<std::string_view> quant_method =
                               std::nullopt) {
    const q3x::model::ModelConfig* config = q3x::model::find_known_model(model);
    if (config == nullptr) {
        throw std::runtime_error("unknown model in HF config fixture");
    }

    std::ostringstream json;
    json << "{\"architectures\":[\"" << config->hf_architecture
         << "\"],\"model_type\":\"" << config->hf_model_type
         << "\",\"text_config\":{"
         << "\"model_type\":\"" << config->hf_text_model_type
         << "\",\"dtype\":\"bfloat16\",\"hidden_act\":\"silu\""
         << ",\"vocab_size\":" << config->vocab_size
         << ",\"hidden_size\":" << config->hidden_size
         << ",\"num_hidden_layers\":" << config->num_hidden_layers
         << ",\"max_position_embeddings\":"
         << config->max_position_embeddings
         << ",\"rms_norm_eps\":" << config->rms_norm_eps
         << ",\"num_attention_heads\":" << config->num_attention_heads
         << ",\"num_key_value_heads\":" << config->num_key_value_heads
         << ",\"head_dim\":" << config->head_dim
         << ",\"full_attention_interval\":"
         << config->full_attention_interval
         << ",\"attention_bias\":false,\"attn_output_gate\":"
         << (config->attention_output_gate ? "true" : "false")
         << ",\"linear_num_key_heads\":" << config->linear_num_key_heads
         << ",\"linear_key_head_dim\":" << config->linear_key_head_dim
         << ",\"linear_num_value_heads\":" << config->linear_num_value_heads
         << ",\"linear_value_head_dim\":" << config->linear_value_head_dim
         << ",\"linear_conv_kernel_dim\":"
         << config->linear_conv_kernel_dim
         << ",\"mtp_num_hidden_layers\":" << config->mtp_num_hidden_layers
         << ",\"mtp_use_dedicated_embeddings\":"
         << (config->mtp_uses_dedicated_embeddings ? "true" : "false");

    if (config->is_moe()) {
        json << ",\"num_experts\":" << config->num_experts
             << ",\"num_experts_per_tok\":"
             << config->num_experts_per_token
             << ",\"moe_intermediate_size\":"
             << config->moe_intermediate_size
             << ",\"shared_expert_intermediate_size\":"
             << config->shared_expert_intermediate_size;
    } else {
        json << ",\"intermediate_size\":" << config->intermediate_size;
    }

    json << ",\"partial_rotary_factor\":0.25,\"rope_parameters\":{"
         << "\"partial_rotary_factor\":0.25,\"rope_theta\":"
         << config->rope_theta << "},\"layer_types\":[";
    for (std::size_t layer = 0; layer < config->num_hidden_layers; ++layer) {
        if (layer != 0) {
            json << ',';
        }
        const bool full =
            config->layer_type(layer) == q3x::model::LayerType::kFullAttention;
        json << (full ? "\"full_attention\"" : "\"linear_attention\"");
    }
    const q3x::model::VisionConfig& vision = config->vision;
    json << "]},\"vision_config\":{"
         << "\"depth\":" << vision.depth
         << ",\"hidden_size\":" << vision.hidden_size
         << ",\"intermediate_size\":" << vision.intermediate_size
         << ",\"num_heads\":" << vision.num_heads
         << ",\"out_hidden_size\":" << vision.out_hidden_size
         << ",\"num_position_embeddings\":"
         << vision.num_position_embeddings
         << ",\"patch_size\":" << vision.patch_size
         << ",\"spatial_merge_size\":" << vision.spatial_merge_size
         << ",\"temporal_patch_size\":" << vision.temporal_patch_size << '}';
    if (quant_method) {
        json << ",\"quantization_config\":{\"quant_method\":\""
             << *quant_method << "\"}";
    }
    json << '}';
    return json.str();
}

checkpoint::ModelOptSummary make_valid_quantization(
    const checkpoint::KnownCheckpointDescriptor& descriptor) {
    checkpoint::ModelOptSummary summary;
    summary.producer_name = "modelopt";
    summary.producer_version = descriptor.modelopt_version;
    summary.quantization_algorithm = "MIXED_PRECISION";
    summary.kv_cache_quantization_algorithm = "FP8";
    summary.exclude_modules = {"mtp*", "mtp.layers.0*"};
    summary.quantized_modules =
        checkpoint::expected_quantized_modules(descriptor.model);
    for (const auto& item : summary.quantized_modules) {
        if (item.second.algorithm == checkpoint::QuantAlgorithm::kFp8) {
            ++summary.fp8_count;
        } else {
            ++summary.nvfp4_count;
        }
    }
    return summary;
}

void test_known_checkpoint_catalog(TestContext& test) {
    const auto& catalog = checkpoint::known_checkpoint_catalog();
    test.expect(catalog.size() == 2,
                "checkpoint catalog contains exactly the two NVIDIA artifacts");

    const auto& dense = catalog[0];
    test.expect(dense.model == q3x::model::KnownModel::kQwen36_27B,
                "first checkpoint is Qwen3.6-27B");
    test.expect(dense.repository == "nvidia/Qwen3.6-27B-NVFP4",
                "dense repository is pinned");
    test.expect(dense.config_sha256 ==
                    "c04a19ba293737ad7be4f6e96d6666cb7e479cbe19ecc0c289fad267135b0338",
                "dense config digest is pinned");
    test.expect(dense.quant_config_sha256 ==
                    "fd7200cd8bca2a8a5d777061521abf83e2deb97ab6bc2f04e7a0a3d3f8ecd5c1",
                "dense quantization digest is pinned");
    test.expect(dense.index_sha256 ==
                    "7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2",
                "dense index digest is pinned");
    test.expect(dense.expected_fp8_modules == 208 &&
                    dense.expected_nvfp4_modules == 193 &&
                    dense.expected_index_tensors == 2194 &&
                    dense.expected_index_shards == 3,
                "dense descriptor records observed artifact statistics");

    const auto& moe = catalog[1];
    test.expect(moe.model == q3x::model::KnownModel::kQwen36_35BA3B,
                "second checkpoint is Qwen3.6-35B-A3B");
    test.expect(moe.config_sha256 ==
                    "58aefa1c9eff7989f431d748f2ddec39446cb1fd2a69acc46e285c6a37b0cecc",
                "MoE config digest is pinned");
    test.expect(moe.quant_config_sha256 ==
                    "75fe7cc8d5836b58734e05ee67423a4ce91d602aaad45c8173a1b7597cd57663",
                "MoE quantization digest is pinned");
    test.expect(moe.index_sha256 ==
                    "d67403a4e9793c0ba8a136baf14b3b76ec7b32c822267978084895e07ebd8a3e",
                "MoE index digest is pinned");
    test.expect(moe.expected_fp8_modules == 130 &&
                    moe.expected_nvfp4_modules == 161 &&
                    moe.expected_index_tensors == 124468 &&
                    moe.expected_index_shards == 3,
                "MoE descriptor records observed artifact statistics");
    test.expect(!moe.known_metadata_quirk.empty() &&
                    moe.external_quantization_authoritative,
                "MoE stale embedded metadata quirk is explicit");

    test.expect(checkpoint::find_known_checkpoint_by_config_sha256(
                    dense.config_sha256) == &dense,
                "config digest lookup resolves dense checkpoint");
    test.expect(checkpoint::find_known_checkpoint_by_config_sha256(
                    moe.config_sha256) == &moe,
                "config digest lookup resolves MoE checkpoint");
    test.expect(checkpoint::find_known_checkpoint_by_config_sha256(
                    std::string(64, '0')) == nullptr,
                "unknown config digest does not resolve");
}

void test_hf_config_shape_parsing(TestContext& test) {
    using q3x::model::KnownModel;
    for (const KnownModel model : {KnownModel::kQwen36_27B,
                                   KnownModel::kQwen36_35BA3B}) {
        auto document =
            parse_document(make_hf_config(model), test, "HF fixture JSON parses");
        if (!document) {
            continue;
        }
        auto parsed = checkpoint::parse_hf_config(*document);
        test.expect(parsed.ok(), "supported HF config is semantically valid");
        if (!parsed) {
            continue;
        }
        test.expect(parsed.value->shape_candidates.size() == 2,
                    "shared Qwen3.5/Qwen3.6 shape has two release candidates");
        test.expect(parsed.value->shape_candidates[0] ==
                        (model == KnownModel::kQwen36_27B
                             ? KnownModel::kQwen35_27B
                             : KnownModel::kQwen35_35BA3B),
                    "shape matching preserves explicit release ambiguity");
        test.expect(parsed.value->shape_candidates[1] == model,
                    "Qwen3.6 shape candidate is retained");
    }

    auto dflash = parse_document(
        R"({"architectures":["DFlashDraftModel"],"model_type":"qwen3"})",
        test,
        "DFlash fixture JSON parses");
    if (dflash) {
        std::string source = "owned-source.json";
        auto parsed = checkpoint::parse_hf_config(*dflash, source);
        source.assign("mutated");
        test.expect(!parsed && has_code(parsed.diagnostics,
                                       checkpoint::DiagnosticCode::
                                           kUnsupportedArchitecture),
                    "DFlash draft architecture is rejected explicitly");
        test.expect(!parsed.diagnostics.empty() &&
                        parsed.diagnostics[0].source == "owned-source.json",
                    "diagnostic source strings own their storage");
    }
}

void test_modelopt_parser(TestContext& test) {
    constexpr std::string_view kValid = R"({
      "producer":{"name":"modelopt","version":"0.45.0"},
      "quantization":{
        "quant_algo":"MIXED_PRECISION",
        "kv_cache_quant_algo":"FP8",
        "exclude_modules":["mtp*","mtp.layers.0*"],
        "quantized_layers":{
          "linear":{"quant_algo":"FP8"},
          "mlp":{"quant_algo":"W4A16_NVFP4","group_size":16}
        }
      }
    })";
    auto document = parse_document(kValid, test, "valid ModelOpt JSON parses");
    if (document) {
        auto parsed = checkpoint::parse_modelopt_quant_config(*document);
        test.expect(parsed.ok(), "strict ModelOpt schema accepts observed fields");
        if (parsed) {
            test.expect(parsed.value->fp8_count == 1 &&
                            parsed.value->nvfp4_count == 1,
                        "ModelOpt parser counts algorithms");
            const auto fp8 = parsed.value->quantized_modules.find("linear");
            const auto nvfp4 = parsed.value->quantized_modules.find("mlp");
            test.expect(fp8 != parsed.value->quantized_modules.end() &&
                            fp8->second.algorithm ==
                                checkpoint::QuantAlgorithm::kFp8 &&
                            !fp8->second.group_size,
                        "FP8 module has no group size");
            test.expect(nvfp4 != parsed.value->quantized_modules.end() &&
                            nvfp4->second.algorithm ==
                                checkpoint::QuantAlgorithm::kW4A16Nvfp4 &&
                            nvfp4->second.group_size == 16,
                        "NVFP4 module retains group size 16");
        }
    }

    constexpr std::string_view kUnknownField = R"({
      "producer":{"name":"modelopt","version":"0.45.0"},
      "quantization":{
        "quant_algo":"MIXED_PRECISION","kv_cache_quant_algo":"FP8",
        "exclude_modules":[],
        "quantized_layers":{"linear":{"quant_algo":"FP8","group_size":16}}
      }
    })";
    auto unknown =
        parse_document(kUnknownField, test, "unknown-field fixture parses");
    if (unknown) {
        auto parsed = checkpoint::parse_modelopt_quant_config(*unknown);
        test.expect(!parsed && has_code(parsed.diagnostics,
                                       checkpoint::DiagnosticCode::kUnknownField),
                    "strict FP8 record rejects an extra group_size field");
    }

    constexpr std::string_view kBadGroup = R"({
      "producer":{"name":"modelopt","version":"0.45.0"},
      "quantization":{
        "quant_algo":"MIXED_PRECISION","kv_cache_quant_algo":"FP8",
        "exclude_modules":[],
        "quantized_layers":{"mlp":{"quant_algo":"W4A16_NVFP4","group_size":32}}
      }
    })";
    auto bad_group =
        parse_document(kBadGroup, test, "bad-group fixture parses");
    if (bad_group) {
        auto parsed = checkpoint::parse_modelopt_quant_config(*bad_group);
        test.expect(!parsed &&
                        has_code(parsed.diagnostics,
                                 checkpoint::DiagnosticCode::
                                     kUnsupportedQuantization),
                    "unsupported NVFP4 group size is rejected");
    }
}

void test_expected_quantized_module_sets(TestContext& test) {
    using q3x::model::KnownModel;
    const auto dense =
        checkpoint::expected_quantized_modules(KnownModel::kQwen36_27B);
    std::size_t dense_fp8 = 0;
    std::size_t dense_nvfp4 = 0;
    for (const auto& item : dense) {
        if (item.second.algorithm == checkpoint::QuantAlgorithm::kFp8) {
            ++dense_fp8;
        } else {
            ++dense_nvfp4;
        }
    }
    test.expect(dense.size() == 401 && dense_fp8 == 208 &&
                    dense_nvfp4 == 193,
                "dense expected set has the observed 208/193 schedule");
    test.expect(dense.count(
                    "model.language_model.layers.0.linear_attn.in_proj_qkv") ==
                    1 &&
                    dense.count(
                        "model.language_model.layers.3.self_attn.q_proj") == 1 &&
                    dense.count("model.language_model.layers.0.mlp.gate_proj") ==
                        1 &&
                    dense.count("lm_head") == 1,
                "dense expected set contains representative exact paths");
    test.expect(dense.count(
                    "model.language_model.layers.0.self_attn.q_proj") == 0 &&
                    dense.count(
                        "model.language_model.layers.3.linear_attn.in_proj_qkv") ==
                        0,
                "dense expected set enforces the hybrid attention schedule");

    const auto moe = checkpoint::expected_quantized_modules(
        KnownModel::kQwen36_35BA3B);
    std::size_t moe_fp8 = 0;
    std::size_t moe_nvfp4 = 0;
    for (const auto& item : moe) {
        if (item.second.algorithm == checkpoint::QuantAlgorithm::kFp8) {
            ++moe_fp8;
        } else {
            ++moe_nvfp4;
        }
    }
    test.expect(moe.size() == 291 && moe_fp8 == 130 && moe_nvfp4 == 161,
                "MoE expected set has the observed 130/161 schedule");
    test.expect(moe.count("model.language_model.layers.0.mlp.experts") == 1 &&
                    moe.count("model.language_model.layers.0.mlp.shared_expert."
                              "down_proj") == 1 &&
                    moe.count("lm_head") == 1,
                "MoE expected set contains packed experts and shared expert");
}

void test_quantization_validation(TestContext& test) {
    for (const auto& descriptor : checkpoint::known_checkpoint_catalog()) {
        checkpoint::ModelOptSummary summary =
            make_valid_quantization(descriptor);
        test.expect(
            checkpoint::validate_modelopt_quantization(summary, descriptor)
                .empty(),
            "exact generated quantization set validates against descriptor");

        auto missing = summary;
        missing.quantized_modules.erase(missing.quantized_modules.begin());
        auto diagnostics =
            checkpoint::validate_modelopt_quantization(missing, descriptor);
        test.expect(has_code(diagnostics,
                             checkpoint::DiagnosticCode::kMissingQuantizedModule),
                    "missing quantized module is diagnosed by exact path");

        auto unexpected = summary;
        unexpected.quantized_modules.emplace(
            "model.language_model.not_a_real_module",
            checkpoint::QuantizedModule{checkpoint::QuantAlgorithm::kFp8,
                                        std::nullopt});
        diagnostics =
            checkpoint::validate_modelopt_quantization(unexpected, descriptor);
        test.expect(
            has_code(diagnostics,
                     checkpoint::DiagnosticCode::kUnexpectedQuantizedModule),
            "unexpected quantized module is diagnosed by exact path");

        auto wrong_producer = summary;
        wrong_producer.producer_version = "0.0.0";
        diagnostics = checkpoint::validate_modelopt_quantization(
            wrong_producer, descriptor);
        test.expect(has_code(diagnostics,
                             checkpoint::DiagnosticCode::kProducerMismatch),
                    "producer version mismatch is explicit");
    }
}

void test_shard_path_policy(TestContext& test) {
    test.expect(checkpoint::is_safe_relative_shard_path(
                    "model-00001-of-00003.safetensors"),
                "ordinary shard basename is safe");
    test.expect(checkpoint::is_safe_relative_shard_path(
                    "weights/model-00001-of-00003.safetensors"),
                "nested relative shard path is safe");
    for (const std::string_view unsafe : {
             "", "/tmp/model.safetensors", "../model.safetensors",
             "weights/../model.safetensors", "./model.safetensors",
             "weights\\model.safetensors", "C:model.safetensors",
             "model.bin", "weights//model.safetensors"}) {
        test.expect(!checkpoint::is_safe_relative_shard_path(unsafe),
                    "unsafe or non-safetensors shard path is rejected");
    }
}

void test_inspection_early_rejections(TestContext& test) {
    TemporaryDirectory directory;
    for (const std::string_view method : {"awq", "gptq",
                                          "compressed-tensors"}) {
        test.expect(write_file(directory.path() / "config.json",
                               make_hf_config(
                                   q3x::model::KnownModel::kQwen36_27B,
                                   method)),
                    "unsupported quantization fixture is written");
        const checkpoint::InspectionResult inspected =
            checkpoint::inspect_directory(directory.path());
        test.expect(inspected.status ==
                        checkpoint::InspectionStatus::kUnsupportedQuantization &&
                        has_code(inspected.diagnostics,
                                 checkpoint::DiagnosticCode::
                                     kUnsupportedQuantization),
                    "AWQ/GPTQ/compressed-tensors fails before shard access");
    }

    test.expect(write_file(
                    directory.path() / "config.json",
                    R"({"architectures":["DFlashDraftModel"],"model_type":"qwen3"})"),
                "DFlash fixture is written");
    checkpoint::InspectionResult inspected =
        checkpoint::inspect_directory(directory.path());
    test.expect(inspected.status ==
                    checkpoint::InspectionStatus::kUnsupportedArchitecture &&
                    has_code(inspected.diagnostics,
                             checkpoint::DiagnosticCode::
                                 kUnsupportedArchitecture),
                "DFlash draft fails before unrelated metadata is requested");

    test.expect(write_file(directory.path() / "config.json",
                           make_hf_config(
                               q3x::model::KnownModel::kQwen36_27B,
                               "modelopt")),
                "known-shape unknown-revision fixture is written");
    inspected = checkpoint::inspect_directory(directory.path());
    test.expect(inspected.status ==
                    checkpoint::InspectionStatus::kUnknownRevision &&
                    has_code(inspected.diagnostics,
                             checkpoint::DiagnosticCode::kUnknownRevision),
                "known shape with an unpinned config digest is unknown revision");
}

void test_invalid_inspection_options(TestContext& test) {
    checkpoint::InspectionOptions options;
    options.max_index_tensors = 0;
    const checkpoint::InspectionResult inspected =
        checkpoint::inspect_directory("does-not-matter", options);
    test.expect(inspected.status ==
                    checkpoint::InspectionStatus::kInvalidMetadata &&
                    has_code(inspected.diagnostics,
                             checkpoint::DiagnosticCode::kInvalidValue),
                "zero defensive limit is rejected before filesystem access");
}

void test_official_metadata_if_available(TestContext& test) {
    const char* root_text = std::getenv("Q3X_OFFICIAL_METADATA_ROOT");
    if (root_text == nullptr || *root_text == '\0') {
        return;
    }
    const std::filesystem::path root(root_text);
    const std::vector<std::pair<std::string_view, q3x::model::KnownModel>> cases = {
        {"27", q3x::model::KnownModel::kQwen36_27B},
        {"35", q3x::model::KnownModel::kQwen36_35BA3B},
    };
    for (const auto& item : cases) {
        const std::filesystem::path directory = root / item.first;
        const checkpoint::InspectionResult inspected =
            checkpoint::inspect_directory(directory);
        test.expect(inspected.ok(),
                    "official pinned metadata passes without local shards");
        if (inspected.report) {
            test.expect(inspected.report->checkpoint.model == item.second,
                        "official metadata resolves the intended descriptor");
            test.expect(inspected.report->present_shards == 0 &&
                            inspected.report->missing_shards == 3,
                        "metadata-only inspection records all missing shards");
            test.expect(inspected.report->diagnostics.size() == 3,
                        "each absent official shard produces one warning");
        }

        checkpoint::InspectionOptions require_shards;
        require_shards.require_shards = true;
        const checkpoint::InspectionResult strict =
            checkpoint::inspect_directory(directory, require_shards);
        test.expect(strict.status == checkpoint::InspectionStatus::kMissingShard &&
                        has_code(strict.diagnostics,
                                 checkpoint::DiagnosticCode::kShardMissing),
                    "require_shards strengthens missing shard to an error");
    }
}

void test_strict_shard_contract_if_available(TestContext& test) {
    namespace safetensors = q3x::io::safetensors;

    const char* root_text = std::getenv("Q3X_OFFICIAL_METADATA_ROOT");
    if (root_text == nullptr || *root_text == '\0') {
        return;
    }

    // Never create or mutate shards below Q3X_OFFICIAL_METADATA_ROOT: callers
    // may point it at a real checkpoint. Exact byte copies preserve the pinned
    // metadata hashes while all destructive fixture mutations stay temporary.
    TemporaryDirectory directory;
    const std::filesystem::path source =
        std::filesystem::path(root_text) / "27";
    test.expect(copy_metadata_files(source, directory.path()),
                "official dense metadata is copied into an isolated fixture");

    const safetensors::Result<safetensors::Index> index =
        safetensors::read_index(
            (directory.path() / "model.safetensors.index.json").string());
    test.expect(index.ok(), "copied dense index parses for shard synthesis");
    if (!index || !index.value->total_size || index.value->shards.size() < 2U) {
        return;
    }

    const std::string& payload_tensor = index.value->weight_map.begin()->first;
    std::string mutation_tensor;
    for (const auto& mapping : index.value->weight_map) {
        if (mapping.first != payload_tensor) {
            mutation_tensor = mapping.first;
            break;
        }
    }
    test.expect(!mutation_tensor.empty(),
                "dense index has a non-payload tensor for mutations");
    if (mutation_tensor.empty()) {
        return;
    }
    const std::string& declared_shard =
        index.value->weight_map.at(mutation_tensor);
    const auto target = std::find_if(
        index.value->shards.begin(),
        index.value->shards.end(),
        [&declared_shard](const std::string& shard) {
            return shard != declared_shard;
        });
    test.expect(target != index.value->shards.end(),
                "dense index has another shard for ownership mutation");
    if (target == index.value->shards.end()) {
        return;
    }

    checkpoint::InspectionOptions strict_options;
    strict_options.require_shards = true;
    auto inspect_strict = [&]() {
        return checkpoint::inspect_directory(directory.path(), strict_options);
    };

    test.expect(write_synthetic_shards(directory.path(), *index.value),
                "valid sparse shard contract is synthesized");
    checkpoint::InspectionResult inspected = inspect_strict();
    test.expect(inspected.ok(),
                "strict checkpoint inspection accepts the complete contract");
    if (inspected.report) {
        test.expect(
            inspected.report->shard_contract_validated &&
                inspected.report->validated_shards ==
                    index.value->shards.size() &&
                inspected.report->validated_tensors ==
                    index.value->weight_map.size() &&
                inspected.report->validated_payload_bytes ==
                    *index.value->total_size &&
                inspected.report->present_shards ==
                    index.value->shards.size() &&
                inspected.report->missing_shards == 0U,
            "strict report exposes validated shard, tensor, and payload totals");
    }

    const std::filesystem::path first_shard =
        directory.path() / index.value->shards.front();
    constexpr std::string_view kLfsPointer =
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
        "size 21921428072\n";
    test.expect(write_file(first_shard, kLfsPointer),
                "temporary shard is replaced by a Git LFS pointer");
    inspected = inspect_strict();
    test.expect(
        inspected.status == checkpoint::InspectionStatus::kInvalidMetadata &&
            has_code(inspected.diagnostics,
                     checkpoint::DiagnosticCode::kShardHeaderInvalid),
        "strict checkpoint inspection rejects a Git LFS pointer shard");

    test.expect(write_file(first_shard, safetensors_prefix(2U) + "[]"),
                "temporary shard is replaced by a malformed header");
    inspected = inspect_strict();
    test.expect(
        inspected.status == checkpoint::InspectionStatus::kInvalidMetadata &&
            has_code(inspected.diagnostics,
                     checkpoint::DiagnosticCode::kShardHeaderInvalid),
        "strict checkpoint inspection maps malformed shard headers");

    SyntheticShardOptions wrong_owner;
    wrong_owner.moved_tensor =
        std::make_pair(mutation_tensor, std::string(*target));
    test.expect(write_synthetic_shards(directory.path(),
                                       *index.value,
                                       wrong_owner),
                "wrong-owner shard contract is synthesized");
    inspected = inspect_strict();
    test.expect(
        inspected.status == checkpoint::InspectionStatus::kInvalidMetadata &&
            has_code(inspected.diagnostics,
                     checkpoint::DiagnosticCode::kWrongShardTensor),
        "strict checkpoint inspection maps wrong shard ownership");

    SyntheticShardOptions missing;
    missing.omitted_tensor = mutation_tensor;
    test.expect(write_synthetic_shards(directory.path(), *index.value, missing),
                "missing-tensor shard contract is synthesized");
    inspected = inspect_strict();
    test.expect(
        inspected.status == checkpoint::InspectionStatus::kInvalidMetadata &&
            has_code(inspected.diagnostics,
                     checkpoint::DiagnosticCode::kMissingShardTensor),
        "strict checkpoint inspection maps an indexed tensor missing on disk");

    SyntheticShardOptions extra;
    extra.add_extra_tensor = true;
    test.expect(write_synthetic_shards(directory.path(), *index.value, extra),
                "extra-tensor shard contract is synthesized");
    inspected = inspect_strict();
    test.expect(
        inspected.status == checkpoint::InspectionStatus::kInvalidMetadata &&
            has_code(inspected.diagnostics,
                     checkpoint::DiagnosticCode::kUnexpectedShardTensor),
        "strict checkpoint inspection maps a tensor absent from the index");

    SyntheticShardOptions wrong_total;
    wrong_total.payload_bytes = *index.value->total_size - 1U;
    test.expect(write_synthetic_shards(directory.path(),
                                       *index.value,
                                       wrong_total),
                "payload-total mismatch shard contract is synthesized");
    inspected = inspect_strict();
    test.expect(
        inspected.status == checkpoint::InspectionStatus::kInvalidMetadata &&
            has_code(inspected.diagnostics,
                     checkpoint::DiagnosticCode::kShardPayloadSizeMismatch),
        "strict checkpoint inspection maps aggregate payload mismatch");
}

}  // namespace

int main() {
    TestContext test;
    try {
        test_known_checkpoint_catalog(test);
        test_hf_config_shape_parsing(test);
        test_modelopt_parser(test);
        test_expected_quantized_module_sets(test);
        test_quantization_validation(test);
        test_shard_path_policy(test);
        test_inspection_early_rejections(test);
        test_invalid_inspection_options(test);
        test_official_metadata_if_available(test);
        test_strict_shard_contract_if_available(test);
    } catch (const std::exception& error) {
        std::cerr << "FAILED: unexpected test exception: " << error.what() << '\n';
        return 1;
    }

    if (test.failures() != 0) {
        std::cerr << test.failures() << " checkpoint metadata test(s) failed\n";
        return 1;
    }
    std::cout << "All checkpoint metadata tests passed\n";
    return 0;
}
