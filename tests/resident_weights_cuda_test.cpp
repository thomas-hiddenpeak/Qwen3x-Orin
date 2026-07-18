#include "q3x/core/sha256.h"
#include "q3x/model/weight_manifest.h"
#include "q3x/runtime/resident_weights.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

namespace runtime = q3x::runtime;
namespace weights = q3x::model::weights;
namespace st = q3x::io::safetensors;

class TestContext {
  public:
    void expect(bool condition, const char* message) {
        if (!condition) {
            std::cerr << "FAILED: " << message << '\n';
            ++failures_;
        }
    }
    [[nodiscard]] int failures() const noexcept { return failures_; }

  private:
    int failures_ = 0;
};

class TempDirectory {
  public:
    TempDirectory() {
        std::string pattern = "/tmp/q3x-resident-loader-XXXXXX";
        std::vector<char> writable(pattern.begin(), pattern.end());
        writable.push_back('\0');
        char* const created = ::mkdtemp(writable.data());
        if (created != nullptr) {
            path_ = created;
        }
    }
    ~TempDirectory() {
        if (!path_.empty()) {
            std::error_code error;
            (void)std::filesystem::remove_all(path_, error);
        }
    }

    TempDirectory(const TempDirectory&) = delete;
    TempDirectory& operator=(const TempDirectory&) = delete;
    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }
    [[nodiscard]] explicit operator bool() const noexcept {
        return !path_.empty();
    }

  private:
    std::filesystem::path path_;
};

bool write_file(const std::filesystem::path& path, std::string_view bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    return static_cast<bool>(output);
}

weights::TensorLocator locator(std::string shard,
                               std::uint64_t begin,
                               std::uint64_t bytes,
                               weights::TensorCategory category) {
    weights::TensorLocator result;
    result.category = category;
    result.shard = std::move(shard);
    result.file_begin = begin;
    result.file_end = begin + bytes;
    result.byte_size = bytes;
    result.dtype = st::DType::kU8;
    result.shape = {bytes};
    return result;
}

weights::WeightManifest make_manifest(const std::string& shard) {
    weights::WeightManifest manifest;
    manifest.tensors.emplace(
        "model.language_model.test_a.weight",
        locator(shard, 5U, 12U, weights::TensorCategory::kText));
    manifest.tensors.emplace(
        "model.visual.skip.weight",
        locator(shard, 17U, 8U, weights::TensorCategory::kVision));
    manifest.tensors.emplace(
        "model.language_model.test_b.weight",
        locator(shard, 25U, 24U, weights::TensorCategory::kText));
    manifest.tensors.emplace(
        "mtp.skip.weight",
        locator(shard, 49U, 6U, weights::TensorCategory::kMtp));
    manifest.tensors.emplace(
        "lm_head.test_c",
        locator(shard, 60U, 4U, weights::TensorCategory::kText));
    manifest.summary.shard_count = 1U;
    manifest.summary.tensor_count = 5U;
    manifest.summary.text_tensor_count = 3U;
    manifest.summary.raw_text_bytes = 40U;
    manifest.summary.arena_alignment = runtime::kResidentTensorAlignment;
    manifest.summary.estimated_text_arena_bytes = 768U;
    return manifest;
}

runtime::ShardIdentity identity(std::string filename,
                                std::string_view data,
                                std::uint64_t size_override = 0U) {
    runtime::ShardIdentity result;
    result.filename = std::move(filename);
    result.file_size = size_override == 0U ? data.size() : size_override;
    result.sha256 = q3x::core::sha256(data).hex();
    return result;
}

runtime::ResidentLoadOptions tiny_options() {
    runtime::ResidentLoadOptions options;
    options.chunk_bytes = 10U;
    options.min_free_bytes_after_load = 0U;
    options.max_arena_bytes = 1024U * 1024U;
    options.max_tensors = 100U;
    options.max_shards = 4U;
    options.max_memcpy_operations = 100U;
    return options;
}

void print_diagnostic(const runtime::ResidentLoadDiagnostic& diagnostic) {
    std::cerr << "loader diagnostic: code="
              << runtime::to_string(diagnostic.code)
              << " message=" << diagnostic.message
              << " context=" << diagnostic.context
              << " shard=" << diagnostic.shard
              << " offset=" << diagnostic.offset
              << " errno=" << diagnostic.system_error
              << " cuda=" << diagnostic.cuda_error
              << " expected=" << diagnostic.expected
              << " actual=" << diagnostic.actual << '\n';
}

void expect_tensor_bytes(TestContext& test,
                         const runtime::ResidentWeights& resident,
                         std::string_view name,
                         std::string_view expected,
                         const char* message) {
    const runtime::DeviceTensorView* view = resident.find(name);
    if (view == nullptr || view->byte_size != expected.size()) {
        test.expect(false, message);
        return;
    }
    std::string actual(expected.size(), '\0');
    const cudaError_t status = cudaMemcpy(actual.data(),
                                          view->device_data,
                                          actual.size(),
                                          cudaMemcpyDeviceToHost);
    test.expect(status == cudaSuccess && actual == expected, message);
}

void test_success_security_and_failures(TestContext& test) {
    TempDirectory temporary;
    test.expect(static_cast<bool>(temporary), "temporary directory is created");
    if (!temporary) {
        return;
    }
    std::string data;
    data.reserve(80U);
    for (std::size_t index = 0U; index < 80U; ++index) {
        data.push_back(static_cast<char>(index));
    }
    const std::string filename = "tiny.safetensors";
    test.expect(write_file(temporary.path() / filename, data),
                "synthetic shard is written");
    const weights::WeightManifest manifest = make_manifest(filename);
    const runtime::ShardIdentity good = identity(filename, data);

    // Leave a caller-owned CUDA error uncleared. The loader must isolate it.
    (void)cudaMemcpy(nullptr, nullptr, 1U, cudaMemcpyHostToDevice);
    test.expect(cudaPeekAtLastError() != cudaSuccess,
                "test injected a stale CUDA last error");

    runtime::ResidentLoadOptions portable_options = tiny_options();
    portable_options.sha256_backend =
        runtime::ResidentSha256Backend::kPortable;
    runtime::ResidentLoadResult loaded = runtime::load_resident_weights(
        temporary.path(), manifest, {good}, portable_options);
    if (!loaded) {
        print_diagnostic(loaded.diagnostic);
    }
    test.expect(loaded.ok(), "tiny true-CUDA resident load succeeds");
    if (!loaded) {
        return;
    }
    test.expect(loaded.value->size_bytes() == 768U &&
                    loaded.value->tensor_count() == 3U &&
                    loaded.value->arena_data() != nullptr,
                "resident result owns one exact arena and three text views");
    const runtime::ResidentLoadStats& stats = loaded.value->stats();
    test.expect(stats.bytes_read == 80U && stats.bytes_copied == 40U &&
                    stats.bytes_skipped == 40U && stats.chunks == 8U &&
                    stats.memcpy_operations == 6U,
                "loader reports exact read/copy/skip/chunk/scatter statistics");
    test.expect(stats.shards.size() == 1U &&
                    stats.shards[0].sha256 == good.sha256 &&
                    stats.shards[0].bytes_read == 80U &&
                    stats.shards[0].bytes_copied == 40U &&
                    stats.sha256_backend ==
                        runtime::ResidentSha256Backend::kPortable,
                "portable full-file SHA and byte statistics are exposed");

    runtime::ResidentLoadOptions accelerated_options = tiny_options();
    accelerated_options.sha256_backend =
        runtime::ResidentSha256Backend::kLinuxAfAlg;
    runtime::ResidentLoadResult accelerated = runtime::load_resident_weights(
        temporary.path(), manifest, {good}, accelerated_options);
    const bool af_alg_available = accelerated.ok();
    if (!accelerated &&
        accelerated.diagnostic.code !=
            runtime::ResidentLoadErrorCode::kSha256BackendUnavailable) {
        print_diagnostic(accelerated.diagnostic);
    }
    test.expect(
        accelerated.ok() ||
            accelerated.diagnostic.code ==
                runtime::ResidentLoadErrorCode::kSha256BackendUnavailable,
        "forced Linux AF_ALG either succeeds or reports unavailable");
    if (accelerated) {
        const runtime::ResidentLoadStats& accelerated_stats =
            accelerated.value->stats();
        test.expect(accelerated_stats.sha256_backend ==
                            runtime::ResidentSha256Backend::kLinuxAfAlg &&
                        accelerated_stats.bytes_read == stats.bytes_read &&
                        accelerated_stats.bytes_copied == stats.bytes_copied &&
                        accelerated_stats.chunks == stats.chunks &&
                        accelerated_stats.shards.size() == 1U &&
                        accelerated_stats.shards[0].sha256 == good.sha256,
                    "AF_ALG and portable loaders report identical authenticated bytes");
        expect_tensor_bytes(test,
                            *accelerated.value,
                            "model.language_model.test_b.weight",
                            std::string_view(data).substr(25U, 24U),
                            "AF_ALG load preserves cross-chunk device bytes");
    }

    runtime::ResidentLoadResult automatic = runtime::load_resident_weights(
        temporary.path(), manifest, {good}, tiny_options());
    test.expect(automatic.ok() &&
                    automatic.value->stats().sha256_backend ==
                        (af_alg_available
                             ? runtime::ResidentSha256Backend::kLinuxAfAlg
                             : runtime::ResidentSha256Backend::kPortable),
                "default auto policy selects AF_ALG when available and otherwise falls back");

    runtime::ResidentLoadOptions invalid_backend_options = tiny_options();
    invalid_backend_options.sha256_backend =
        static_cast<runtime::ResidentSha256Backend>(255U);
    runtime::ResidentLoadResult invalid_backend =
        runtime::load_resident_weights(temporary.path(),
                                       manifest,
                                       {good},
                                       invalid_backend_options);
    test.expect(!invalid_backend &&
                    invalid_backend.diagnostic.code ==
                        runtime::ResidentLoadErrorCode::kInvalidOption,
                "unknown SHA-256 backend is rejected before loading");

    runtime::ResidentWeights moved;
    moved = std::move(*loaded.value);
    test.expect(!static_cast<bool>(*loaded.value) &&
                    static_cast<bool>(moved) && moved.tensor_count() == 3U,
                "resident arena is move-only and source becomes empty");
    loaded.value.reset();
    expect_tensor_bytes(test,
                        moved,
                        "model.language_model.test_a.weight",
                        std::string_view(data).substr(5U, 12U),
                        "first text tensor round-trips from device");
    expect_tensor_bytes(test,
                        moved,
                        "model.language_model.test_b.weight",
                        std::string_view(data).substr(25U, 24U),
                        "cross-chunk text tensor round-trips from device");
    expect_tensor_bytes(test,
                        moved,
                        "lm_head.test_c",
                        std::string_view(data).substr(60U, 4U),
                        "small scalar-like text tensor round-trips from device");
    const runtime::DeviceTensorView* second =
        moved.find("model.language_model.test_b.weight");
    test.expect(second != nullptr && second->arena_offset == 256U &&
                    reinterpret_cast<std::uintptr_t>(second->device_data) % 256U == 0U,
                "device tensor view preserves 256-byte alignment");

    runtime::ShardIdentity wrong_hash = good;
    wrong_hash.sha256 = std::string(64U, '0');
    runtime::ResidentLoadResult failed = runtime::load_resident_weights(
        temporary.path(), manifest, {wrong_hash}, tiny_options());
    test.expect(!failed && failed.diagnostic.code ==
                               runtime::ResidentLoadErrorCode::kSha256Mismatch,
                "wrong full-file SHA fails and releases the arena");

    const std::string truncated_name = "truncated.safetensors";
    test.expect(write_file(temporary.path() / truncated_name,
                           std::string_view(data).substr(0U, 79U)),
                "truncated shard is written");
    failed = runtime::load_resident_weights(
        temporary.path(),
        make_manifest(truncated_name),
        {identity(truncated_name, data, 80U)},
        tiny_options());
    test.expect(!failed && failed.diagnostic.code ==
                               runtime::ResidentLoadErrorCode::kShardSizeMismatch,
                "truncated shard fails before allocation");

    const std::string link_name = "link.safetensors";
    std::error_code symlink_error;
    std::filesystem::create_symlink(filename,
                                    temporary.path() / link_name,
                                    symlink_error);
    test.expect(!symlink_error, "synthetic shard symlink is created");
    if (!symlink_error) {
        failed = runtime::load_resident_weights(
            temporary.path(),
            make_manifest(link_name),
            {identity(link_name, data)},
            tiny_options());
        test.expect(!failed &&
                        failed.diagnostic.code ==
                            runtime::ResidentLoadErrorCode::kOpenShardFailed &&
                        failed.diagnostic.system_error != 0,
                    "O_NOFOLLOW rejects a shard symlink with errno context");
    }
}

std::vector<char> read_source_sample(const weights::TensorLocator& locator,
                                     std::size_t bytes) {
    bytes = std::min<std::size_t>(
        bytes, static_cast<std::size_t>(locator.byte_size));
    std::vector<char> result(bytes);
    std::ifstream input(locator.file, std::ios::binary);
    input.seekg(static_cast<std::streamoff>(locator.file_begin));
    input.read(result.data(), static_cast<std::streamsize>(result.size()));
    if (!input) {
        result.clear();
    }
    return result;
}

void test_official_checkpoint_if_requested(TestContext& test) {
    const char* const root = std::getenv("Q3X_OFFICIAL_27B_ROOT");
    if (root == nullptr || *root == '\0') {
        return;
    }
    const weights::ManifestResult manifest =
        weights::build_qwen36_27b_text_manifest(root);
    test.expect(manifest.ok(), "official manifest validates before resident load");
    if (!manifest) {
        return;
    }

    const auto started = std::chrono::steady_clock::now();
    runtime::ResidentLoadResult loaded =
        runtime::load_pinned_qwen36_27b(root);
    const double seconds = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - started)
                               .count();
    if (!loaded) {
        print_diagnostic(loaded.diagnostic);
    }
    test.expect(loaded.ok(), "official 27B authenticated resident load succeeds");
    if (!loaded) {
        return;
    }
    const runtime::ResidentLoadStats& stats = loaded.value->stats();
    test.expect(loaded.value->size_bytes() == 20'150'786'560ULL &&
                    loaded.value->tensor_count() == 1846U,
                "official loader owns exact 20.15GB arena and 1846 views");
    test.expect(stats.bytes_read == 21'921'697'184ULL &&
                    stats.bytes_copied == 20'150'569'096ULL &&
                    stats.bytes_skipped == 1'771'128'088ULL &&
                    stats.chunks == 328U,
                "official loader reports exact full-file and text-only totals");
    const auto& identities = runtime::pinned_qwen36_27b_shards();
    bool hashes_match = stats.shards.size() == identities.size();
    for (std::size_t index = 0U;
         hashes_match && index < identities.size(); ++index) {
        hashes_match = stats.shards[index].filename == identities[index].filename &&
                       stats.shards[index].sha256 == identities[index].sha256 &&
                       stats.shards[index].bytes_read == identities[index].file_size;
    }
    test.expect(hashes_match, "all official full-shard SHA identities match");

    for (const std::string_view name : {
             std::string_view("model.language_model.embed_tokens.weight"),
             std::string_view(
                 "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"),
             std::string_view("lm_head.weight_scale_2")}) {
        const weights::TensorLocator* source = manifest.value->find(name);
        const runtime::DeviceTensorView* device = loaded.value->find(name);
        if (source == nullptr || device == nullptr) {
            test.expect(false, "official sample tensor locator exists");
            continue;
        }
        const std::vector<char> expected = read_source_sample(*source, 32U);
        std::vector<char> actual(expected.size());
        const cudaError_t status = cudaMemcpy(actual.data(),
                                              device->device_data,
                                              actual.size(),
                                              cudaMemcpyDeviceToHost);
        test.expect(!expected.empty() && status == cudaSuccess &&
                        actual == expected,
                    "official sampled tensor bytes round-trip from device");
    }
    std::cout << "Official resident load: seconds=" << seconds
              << " bytes_read=" << stats.bytes_read
              << " bytes_copied=" << stats.bytes_copied
              << " chunks=" << stats.chunks
              << " memcpy_ops=" << stats.memcpy_operations
              << " sha256_backend="
              << runtime::to_string(stats.sha256_backend) << '\n';
}

}  // namespace

int main() {
    (void)cudaGetLastError();
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0) {
        std::cout << "SKIP: CUDA device unavailable\n";
        (void)cudaGetLastError();
        return 77;
    }
    TestContext test;
    test.expect(runtime::to_string(runtime::ResidentSha256Backend::kAuto) ==
                        "auto" &&
                    runtime::to_string(
                        runtime::ResidentSha256Backend::kPortable) ==
                        "portable" &&
                    runtime::to_string(
                        runtime::ResidentSha256Backend::kLinuxAfAlg) ==
                        "linux_af_alg" &&
                    runtime::to_string(
                        runtime::ResidentLoadErrorCode::kSha256Failure) ==
                        "sha256_failure",
                "SHA-256 backend and failure names are stable");
    test_success_security_and_failures(test);
    test_official_checkpoint_if_requested(test);
    if (test.failures() != 0) {
        std::cerr << test.failures() << " resident loader CUDA test(s) failed\n";
        return 1;
    }
    std::cout << "All resident loader CUDA tests passed\n";
    return 0;
}
