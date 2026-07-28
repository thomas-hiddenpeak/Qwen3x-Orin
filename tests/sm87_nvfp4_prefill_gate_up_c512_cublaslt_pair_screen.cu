#include "q3x/kernels/sm87_weight_only_gemv.h"
#include "q3x/core/sha256.h"
#include "q3x/io/safetensors.h"
#include "pinned_checkpoint.h"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace q3x::kernels {

// Test-only ABI for the admitted native C512 M64N256 pair-lookup control and
// the complete horizontal topology comparisons selected by this screen.
// Keep it out of the installed header until the production route is selected.
[[nodiscard]] int
launch_sm87_nvfp4_w4a16_gate_c512_m64_n256_k64_cp_async_capairlookup_test_cuda(
    const std::uint8_t* packed_weights, const std::uint8_t* block_scales,
    float weight_scale_2, const std::uint16_t* activations,
    std::size_t token_count, std::size_t rows, std::size_t columns,
    std::uint16_t* output, void* stream) noexcept;

[[nodiscard]] int
query_sm87_nvfp4_w4a16_gate_c512_m64_n256_k64_cp_async_capairlookup_resources_test_cuda(
    std::size_t token_count, std::size_t rows, std::size_t columns,
    int* registers_per_thread, std::size_t* static_shared_bytes,
    std::size_t* dynamic_shared_bytes, std::size_t* local_bytes,
    int* maximum_threads_per_block, int* active_blocks_per_sm) noexcept;

[[nodiscard]] int
launch_sm87_nvfp4_w4a16_gate_c512_m128_n256_horizontal_p0_test_cuda(
    const std::uint8_t* packed_weights, const std::uint8_t* block_scales,
    float weight_scale_2, const std::uint16_t* activations,
    std::size_t token_count, std::size_t rows, std::size_t columns,
    std::uint16_t* output, void* stream) noexcept;

[[nodiscard]] int
query_sm87_nvfp4_w4a16_gate_c512_m128_n256_horizontal_p0_resources_test_cuda(
    std::size_t token_count, std::size_t rows, std::size_t columns,
    int* registers_per_thread, std::size_t* static_shared_bytes,
    std::size_t* dynamic_shared_bytes, std::size_t* local_bytes,
    int* maximum_threads_per_block, int* active_blocks_per_sm) noexcept;

[[nodiscard]] int
launch_sm87_nvfp4_w4a16_gate_c512_m128_n256_horizontal_named_p1_test_cuda(
    const std::uint8_t* packed_weights, const std::uint8_t* block_scales,
    float weight_scale_2, const std::uint16_t* activations,
    std::size_t token_count, std::size_t rows, std::size_t columns,
    std::uint16_t* output, void* stream) noexcept;

[[nodiscard]] int
query_sm87_nvfp4_w4a16_gate_c512_m128_n256_horizontal_named_p1_resources_test_cuda(
    std::size_t token_count, std::size_t rows, std::size_t columns,
    int* registers_per_thread, std::size_t* static_shared_bytes,
    std::size_t* dynamic_shared_bytes, std::size_t* local_bytes,
    int* maximum_threads_per_block, int* active_blocks_per_sm) noexcept;

}  // namespace q3x::kernels

namespace {

constexpr std::size_t kM = 512U;
constexpr std::size_t kK = 5'120U;
constexpr std::size_t kN = 17'408U;
constexpr std::size_t kAElements = kM * kK;
constexpr std::size_t kBElements = kN * kK;
constexpr std::size_t kCElements = kM * kN;
constexpr std::size_t kPackedWeightBytes = kN * (kK / 2U);
constexpr std::size_t kBlockScaleBytes = kN * (kK / 16U);
constexpr std::size_t kGuardElements = 256U;
constexpr std::size_t kGuardedOutputElements =
    kGuardElements + kCElements + kGuardElements;
constexpr int kMaximumHeuristics = 16;
constexpr int kSelectionWarmups = 2;
constexpr int kSelectionIterations = 4;
constexpr int kWarmups = 2;
constexpr int kIterations = 8;
constexpr int kRounds = 6;
constexpr double kRequiredSerialVsProduction = 1.22;
constexpr double kRequiredTwoScratchVsSerial = 1.03;
constexpr float kSyntheticGateWeightScale2 = 1.25F;
constexpr float kSyntheticUpWeightScale2 = 0.75F;
constexpr std::uint8_t kGuardPoisonByte = 0x3cU;
constexpr std::uint16_t kGuardPoison = 0x3c3cU;

constexpr std::string_view kCheckpointIndex =
    "model.safetensors.index.json";
constexpr std::string_view kGateWeightTensor =
    "model.language_model.layers.0.mlp.gate_proj.weight";
constexpr std::string_view kGateBlockScaleTensor =
    "model.language_model.layers.0.mlp.gate_proj.weight_scale";
constexpr std::string_view kGateWeightScale2Tensor =
    "model.language_model.layers.0.mlp.gate_proj.weight_scale_2";
constexpr std::string_view kUpWeightTensor =
    "model.language_model.layers.0.mlp.up_proj.weight";
constexpr std::string_view kUpBlockScaleTensor =
    "model.language_model.layers.0.mlp.up_proj.weight_scale";
constexpr std::string_view kUpWeightScale2Tensor =
    "model.language_model.layers.0.mlp.up_proj.weight_scale_2";
constexpr std::string_view kPinnedGateWeightSha256 =
    "e9e2d70cef19e52d65a0f7917ea6d936c172809ed247b350443b4344297159d8";
constexpr std::string_view kPinnedGateBlockScaleSha256 =
    "6eeaaa3bf8605b1d85252e13e6c495f6cf1b06e7fee8f27ea6367abbbb8fde0e";
constexpr std::string_view kPinnedGateWeightScale2Sha256 =
    "10f036efcb439d7571cc2c35568c9786ae3315b169f3972ff1c2aae782131f91";
constexpr std::string_view kPinnedUpWeightSha256 =
    "e604b0b18206afe695a191ecf77a6aaf4dfbc0f7e93f1f9789d9b579aed6215f";
constexpr std::string_view kPinnedUpBlockScaleSha256 =
    "ba393d3f9d25a1f4decba80715c6079d27d9de1d059a038a2f3f3f0932870947";
constexpr std::string_view kPinnedUpWeightScale2Sha256 =
    "10f036efcb439d7571cc2c35568c9786ae3315b169f3972ff1c2aae782131f91";

static_assert(kAElements == 2'621'440U);
static_assert(kBElements == 89'128'960U);
static_assert(kCElements == 8'912'896U);
static_assert(kPackedWeightBytes == 44'564'480U);
static_assert(kBlockScaleBytes == 5'570'560U);

class TestContext {
 public:
  void expect(const bool condition, const std::string& message) {
    if (!condition) {
      ++failures_;
      std::cerr << "FAIL: " << message << '\n';
    }
  }

  [[nodiscard]] bool cuda_ok(const cudaError_t status,
                             const std::string& operation) {
    expect(status == cudaSuccess,
           operation + ": " + cudaGetErrorString(status));
    return status == cudaSuccess;
  }

  [[nodiscard]] bool lt_ok(const cublasStatus_t status,
                           const std::string& operation) {
    expect(status == CUBLAS_STATUS_SUCCESS,
           operation + ": cuBLAS status " +
               std::to_string(static_cast<int>(status)));
    return status == CUBLAS_STATUS_SUCCESS;
  }

  [[nodiscard]] int failures() const noexcept { return failures_; }

 private:
  int failures_ = 0;
};

struct PayloadBytes {
  std::vector<std::uint8_t> values;
  std::uint64_t file_begin = 0U;
  std::uint64_t file_end = 0U;
  std::string sha256;
};

struct PayloadScalar {
  float value = 0.0F;
  std::uint32_t little_endian_bits = 0U;
  std::uint64_t file_begin = 0U;
  std::uint64_t file_end = 0U;
  std::string sha256;
};

struct CheckpointPayload {
  PayloadBytes gate_weight;
  PayloadBytes gate_block_scale;
  PayloadScalar gate_weight_scale_2;
  PayloadBytes up_weight;
  PayloadBytes up_block_scale;
  PayloadScalar up_weight_scale_2;
  std::string canonical_directory;
  std::string shard;
  std::uint64_t shard_file_bytes = 0U;
  std::uint64_t header_bytes = 0U;
  std::uint64_t data_offset = 0U;
};

[[nodiscard]] std::string describe_safetensors_error(
    const q3x::io::safetensors::Error& error) {
  std::string description =
      std::string(q3x::io::safetensors::to_string(error.code)) + ": " +
      std::string(error.message());
  if (error.offset != q3x::io::safetensors::kUnknownOffset) {
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

[[nodiscard]] bool path_is_strictly_within(
    const std::filesystem::path& directory,
    const std::filesystem::path& candidate) noexcept {
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

[[nodiscard]] std::string payload_sha256(const void* const data,
                                         const std::size_t bytes) {
  return q3x::core::sha256(std::string_view(
                               static_cast<const char*>(data), bytes))
      .hex();
}

[[nodiscard]] std::uint32_t float_bits(const float value) noexcept {
  std::uint32_t bits = 0U;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

[[nodiscard]] std::string hex_u32(const std::uint32_t value) {
  constexpr char kHex[] = "0123456789abcdef";
  std::string result(8U, '0');
  for (std::size_t index = 0U; index < result.size(); ++index) {
    const unsigned int shift =
        static_cast<unsigned int>((result.size() - 1U - index) * 4U);
    result[index] = kHex[(value >> shift) & 0x0fU];
  }
  return result;
}

[[nodiscard]] bool load_checkpoint_payload(
    TestContext& test, const std::string& checkpoint_directory,
    CheckpointPayload& payload) {
  namespace fs = std::filesystem;
  namespace st = q3x::io::safetensors;

  payload = CheckpointPayload{};
  std::error_code filesystem_error;
  const fs::path requested_directory(checkpoint_directory);
  const fs::file_status directory_link_status =
      fs::symlink_status(requested_directory, filesystem_error);
  if (filesystem_error || !fs::is_directory(directory_link_status) ||
      fs::is_symlink(directory_link_status)) {
    test.expect(false, "checkpoint path is an accessible non-symlink "
                       "directory: " +
                           checkpoint_directory);
    return false;
  }
  const fs::path directory =
      fs::canonical(requested_directory, filesystem_error);
  if (filesystem_error || directory.empty()) {
    test.expect(false, "checkpoint directory canonicalization succeeds");
    return false;
  }

  const fs::path requested_index =
      requested_directory / fs::path(kCheckpointIndex);
  const fs::file_status index_link_status =
      fs::symlink_status(requested_index, filesystem_error);
  if (filesystem_error || !fs::is_regular_file(index_link_status) ||
      fs::is_symlink(index_link_status)) {
    test.expect(false,
                "checkpoint index is a regular non-symlink file");
    return false;
  }
  const fs::path index_path = fs::canonical(requested_index, filesystem_error);
  if (filesystem_error || !path_is_strictly_within(directory, index_path) ||
      index_path != directory / fs::path(kCheckpointIndex)) {
    test.expect(false,
                "checkpoint index resolves inside checkpoint directory");
    return false;
  }
  const st::Result<st::Index> index = st::read_index(index_path.string());
  if (!index) {
    test.expect(false, "read checkpoint safetensors index: " +
                           describe_safetensors_error(index.error));
    return false;
  }

  constexpr std::array<std::string_view, 6U> kRequiredTensors{{
      kGateWeightTensor, kGateBlockScaleTensor, kGateWeightScale2Tensor,
      kUpWeightTensor, kUpBlockScaleTensor, kUpWeightScale2Tensor,
  }};
  std::array<const std::string*, kRequiredTensors.size()> shards{};
  for (std::size_t tensor = 0U; tensor < kRequiredTensors.size(); ++tensor) {
    shards[tensor] = index.value->shard_for(kRequiredTensors[tensor]);
  }
  const bool all_indexed =
      std::all_of(shards.begin(), shards.end(),
                  [](const std::string* const shard) {
                    return shard != nullptr;
                  });
  bool one_safe_shard = all_indexed;
  if (one_safe_shard) {
    one_safe_shard = st::is_safe_relative_shard_path(*shards[0]);
    for (std::size_t tensor = 1U; tensor < shards.size(); ++tensor) {
      one_safe_shard = one_safe_shard && *shards[tensor] == *shards[0];
    }
  }
  test.expect(one_safe_shard,
              "checkpoint index pins all six Gate/Up payloads to one safe "
              "relative shard");
  if (!one_safe_shard) {
    return false;
  }

  const fs::path requested_shard =
      requested_directory / fs::path(*shards[0]);
  const fs::file_status shard_link_status =
      fs::symlink_status(requested_shard, filesystem_error);
  if (filesystem_error || !fs::is_regular_file(shard_link_status) ||
      fs::is_symlink(shard_link_status)) {
    test.expect(false,
                "checkpoint shard is a regular non-symlink file");
    return false;
  }
  const fs::path shard_path = fs::canonical(requested_shard, filesystem_error);
  if (filesystem_error || !path_is_strictly_within(directory, shard_path) ||
      shard_path != directory / fs::path(*shards[0])) {
    test.expect(false,
                "checkpoint shard resolves inside checkpoint directory");
    return false;
  }
  const st::Result<st::Header> header = st::read_header(shard_path.string());
  if (!header) {
    test.expect(false, "read checkpoint safetensors shard header: " +
                           describe_safetensors_error(header.error));
    return false;
  }

  const st::TensorInfo* const gate_weight =
      header.value->find_tensor(kGateWeightTensor);
  const st::TensorInfo* const gate_block_scale =
      header.value->find_tensor(kGateBlockScaleTensor);
  const st::TensorInfo* const gate_weight_scale_2 =
      header.value->find_tensor(kGateWeightScale2Tensor);
  const st::TensorInfo* const up_weight =
      header.value->find_tensor(kUpWeightTensor);
  const st::TensorInfo* const up_block_scale =
      header.value->find_tensor(kUpBlockScaleTensor);
  const st::TensorInfo* const up_weight_scale_2 =
      header.value->find_tensor(kUpWeightScale2Tensor);
  const auto matrix_exact = [](const st::TensorInfo* const tensor,
                               const st::DType dtype,
                               const std::uint64_t rows,
                               const std::uint64_t columns,
                               const std::uint64_t bytes) {
    return tensor != nullptr && tensor->dtype == dtype &&
           tensor->shape.size() == 2U && tensor->shape[0] == rows &&
           tensor->shape[1] == columns && tensor->byte_size == bytes &&
           tensor->file_begin <= tensor->file_end &&
           tensor->file_end - tensor->file_begin == bytes;
  };
  const auto scalar_exact = [](const st::TensorInfo* const tensor) {
    return tensor != nullptr && tensor->dtype == st::DType::kF32 &&
           tensor->shape.empty() && tensor->element_count == 1U &&
           tensor->byte_size == sizeof(float) &&
           tensor->file_begin <= tensor->file_end &&
           tensor->file_end - tensor->file_begin == sizeof(float);
  };
  const bool shapes_exact =
      matrix_exact(gate_weight, st::DType::kU8, kN, kK / 2U,
                   kPackedWeightBytes) &&
      matrix_exact(up_weight, st::DType::kU8, kN, kK / 2U,
                   kPackedWeightBytes) &&
      matrix_exact(gate_block_scale, st::DType::kF8E4M3, kN, kK / 16U,
                   kBlockScaleBytes) &&
      matrix_exact(up_block_scale, st::DType::kF8E4M3, kN, kK / 16U,
                   kBlockScaleBytes) &&
      scalar_exact(gate_weight_scale_2) && scalar_exact(up_weight_scale_2);
  test.expect(shapes_exact,
              "checkpoint layer-0 Gate/Up tensors have exact canonical "
              "NVFP4 packed, block-scale, and scalar shapes");
  if (!shapes_exact) {
    return false;
  }

  const std::array<const st::TensorInfo*, 6U> tensors{{
      gate_weight, gate_block_scale, gate_weight_scale_2, up_weight,
      up_block_scale, up_weight_scale_2,
  }};
  const std::uint64_t maximum_streamoff = static_cast<std::uint64_t>(
      std::numeric_limits<std::streamoff>::max());
  const std::uint64_t maximum_streamsize = static_cast<std::uint64_t>(
      std::numeric_limits<std::streamsize>::max());
  const std::uint64_t shard_file_bytes = header.value->file_size;
  const bool stream_ranges_representable =
      std::all_of(tensors.begin(), tensors.end(),
                  [maximum_streamoff, maximum_streamsize,
                   shard_file_bytes](const st::TensorInfo* const tensor) {
                    return tensor->file_begin <= maximum_streamoff &&
                           tensor->file_end <= maximum_streamoff &&
                           tensor->file_end <= shard_file_bytes &&
                           tensor->byte_size <= maximum_streamsize;
                  });
  test.expect(stream_ranges_representable,
              "checkpoint Gate/Up tensor ranges are stream-representable");
  if (!stream_ranges_representable) {
    return false;
  }

  std::ifstream input(shard_path, std::ios::binary | std::ios::in);
  if (!input) {
    test.expect(false, "open checkpoint Gate/Up shard read-only");
    return false;
  }
  input.seekg(0, std::ios::end);
  const std::streamoff observed_file_size = input.tellg();
  const bool file_size_pinned =
      input && observed_file_size >= 0 &&
      static_cast<std::uint64_t>(observed_file_size) ==
          header.value->file_size;
  test.expect(file_size_pinned,
              "checkpoint shard size remains pinned after header read");
  if (!file_size_pinned) {
    return false;
  }

  payload.gate_weight.values.resize(kPackedWeightBytes);
  payload.gate_block_scale.values.resize(kBlockScaleBytes);
  payload.up_weight.values.resize(kPackedWeightBytes);
  payload.up_block_scale.values.resize(kBlockScaleBytes);
  const auto read_exact =
      [&input](const st::TensorInfo& tensor, void* const destination) {
        input.clear();
        input.seekg(static_cast<std::streamoff>(tensor.file_begin),
                    std::ios::beg);
        if (!input) {
          return false;
        }
        input.read(static_cast<char*>(destination),
                   static_cast<std::streamsize>(tensor.byte_size));
        return input.gcount() ==
               static_cast<std::streamsize>(tensor.byte_size);
      };
  const bool payloads_read =
      read_exact(*gate_weight, payload.gate_weight.values.data()) &&
      read_exact(*gate_block_scale, payload.gate_block_scale.values.data()) &&
      read_exact(*gate_weight_scale_2, &payload.gate_weight_scale_2.value) &&
      read_exact(*up_weight, payload.up_weight.values.data()) &&
      read_exact(*up_block_scale, payload.up_block_scale.values.data()) &&
      read_exact(*up_weight_scale_2, &payload.up_weight_scale_2.value);
  const bool scalars_valid =
      payloads_read && std::isfinite(payload.gate_weight_scale_2.value) &&
      payload.gate_weight_scale_2.value > 0.0F &&
      std::isfinite(payload.up_weight_scale_2.value) &&
      payload.up_weight_scale_2.value > 0.0F;
  test.expect(payloads_read,
              "read all six complete bounded checkpoint Gate/Up payloads");
  test.expect(scalars_valid,
              "checkpoint Gate/Up weight_scale_2 scalars are finite and "
              "positive");
  if (!payloads_read || !scalars_valid) {
    payload = CheckpointPayload{};
    return false;
  }

  payload.gate_weight.sha256 =
      payload_sha256(payload.gate_weight.values.data(),
                     payload.gate_weight.values.size());
  payload.gate_block_scale.sha256 =
      payload_sha256(payload.gate_block_scale.values.data(),
                     payload.gate_block_scale.values.size());
  payload.gate_weight_scale_2.sha256 =
      payload_sha256(&payload.gate_weight_scale_2.value, sizeof(float));
  payload.up_weight.sha256 =
      payload_sha256(payload.up_weight.values.data(),
                     payload.up_weight.values.size());
  payload.up_block_scale.sha256 =
      payload_sha256(payload.up_block_scale.values.data(),
                     payload.up_block_scale.values.size());
  payload.up_weight_scale_2.sha256 =
      payload_sha256(&payload.up_weight_scale_2.value, sizeof(float));
  const bool hashes_pinned =
      payload.gate_weight.sha256 == kPinnedGateWeightSha256 &&
      payload.gate_block_scale.sha256 == kPinnedGateBlockScaleSha256 &&
      payload.gate_weight_scale_2.sha256 ==
          kPinnedGateWeightScale2Sha256 &&
      payload.up_weight.sha256 == kPinnedUpWeightSha256 &&
      payload.up_block_scale.sha256 == kPinnedUpBlockScaleSha256 &&
      payload.up_weight_scale_2.sha256 == kPinnedUpWeightScale2Sha256;
  test.expect(hashes_pinned,
              "checkpoint Gate/Up payload SHA256 values match all six pins");
  if (!hashes_pinned) {
    payload = CheckpointPayload{};
    return false;
  }

  payload.gate_weight_scale_2.little_endian_bits =
      float_bits(payload.gate_weight_scale_2.value);
  payload.up_weight_scale_2.little_endian_bits =
      float_bits(payload.up_weight_scale_2.value);
  const auto capture_range = [](const st::TensorInfo& tensor,
                                auto& destination) {
    destination.file_begin = tensor.file_begin;
    destination.file_end = tensor.file_end;
  };
  capture_range(*gate_weight, payload.gate_weight);
  capture_range(*gate_block_scale, payload.gate_block_scale);
  capture_range(*gate_weight_scale_2, payload.gate_weight_scale_2);
  capture_range(*up_weight, payload.up_weight);
  capture_range(*up_block_scale, payload.up_block_scale);
  capture_range(*up_weight_scale_2, payload.up_weight_scale_2);
  payload.canonical_directory = directory.string();
  payload.shard = *shards[0];
  payload.shard_file_bytes = header.value->file_size;
  payload.header_bytes = header.value->header_size;
  payload.data_offset = header.value->data_offset;
  return true;
}

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      (void)cudaFree(data_);
    }
  }

  [[nodiscard]] bool allocate(TestContext& test, const std::size_t count,
                              const std::string& label) {
    if (count == 0U ||
        count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      test.expect(false, label + " count is representable and nonzero");
      return false;
    }
    count_ = count;
    const cudaError_t status =
        cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T));
    if (!test.cuda_ok(status, label)) {
      count_ = 0U;
      return false;
    }
    return true;
  }

  [[nodiscard]] T* get() noexcept { return data_; }
  [[nodiscard]] const T* get() const noexcept { return data_; }
  [[nodiscard]] std::size_t count() const noexcept { return count_; }
  [[nodiscard]] std::size_t bytes() const noexcept {
    return count_ * sizeof(T);
  }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0U;
};

class LtObjects {
 public:
  LtObjects() = default;
  LtObjects(const LtObjects&) = delete;
  LtObjects& operator=(const LtObjects&) = delete;

  ~LtObjects() {
    if (preference_ != nullptr) {
      (void)cublasLtMatmulPreferenceDestroy(preference_);
    }
    if (output_layout_ != nullptr) {
      (void)cublasLtMatrixLayoutDestroy(output_layout_);
    }
    if (activation_layout_ != nullptr) {
      (void)cublasLtMatrixLayoutDestroy(activation_layout_);
    }
    if (weight_layout_ != nullptr) {
      (void)cublasLtMatrixLayoutDestroy(weight_layout_);
    }
    if (operation_ != nullptr) {
      (void)cublasLtMatmulDescDestroy(operation_);
    }
    if (handle_ != nullptr) {
      (void)cublasLtDestroy(handle_);
    }
  }

  [[nodiscard]] bool create(TestContext& test, const std::string& label) {
    bool ready =
        test.lt_ok(cublasLtCreate(&handle_), label + " create handle");
    ready = ready && test.lt_ok(
                           cublasLtMatmulDescCreate(
                               &operation_, CUBLAS_COMPUTE_32F, CUDA_R_32F),
                           label + " create operation");
    const cublasOperation_t transpose_weight = CUBLAS_OP_T;
    ready = ready && test.lt_ok(
                           cublasLtMatmulDescSetAttribute(
                               operation_, CUBLASLT_MATMUL_DESC_TRANSA,
                               &transpose_weight, sizeof(transpose_weight)),
                           label + " set transpose A");

    // Row-major [N,K] weight storage is column-major [K,N]. Row-major [M,K]
    // activation storage is column-major [K,M]. Compute C^T = W A^T so the
    // visible allocation remains row-major [M,N].
    ready = ready && test.lt_ok(
                           cublasLtMatrixLayoutCreate(
                               &weight_layout_, CUDA_R_16BF, kK, kN, kK),
                           label + " create weight layout");
    ready = ready && test.lt_ok(
                           cublasLtMatrixLayoutCreate(
                               &activation_layout_, CUDA_R_16BF, kK, kM, kK),
                           label + " create activation layout");
    ready = ready && test.lt_ok(
                           cublasLtMatrixLayoutCreate(
                               &output_layout_, CUDA_R_16BF, kN, kM, kN),
                           label + " create output layout");
    ready = ready && test.lt_ok(
                           cublasLtMatmulPreferenceCreate(&preference_),
                           label + " create preference");
    std::uint64_t zero_workspace = 0U;
    ready = ready && test.lt_ok(
                           cublasLtMatmulPreferenceSetAttribute(
                               preference_,
                               CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                               &zero_workspace, sizeof(zero_workspace)),
                           label + " require zero workspace");
    return ready;
  }

  [[nodiscard]] cublasLtHandle_t handle() const noexcept { return handle_; }
  [[nodiscard]] cublasLtMatmulDesc_t operation() const noexcept {
    return operation_;
  }
  [[nodiscard]] cublasLtMatrixLayout_t weight_layout() const noexcept {
    return weight_layout_;
  }
  [[nodiscard]] cublasLtMatrixLayout_t activation_layout() const noexcept {
    return activation_layout_;
  }
  [[nodiscard]] cublasLtMatrixLayout_t output_layout() const noexcept {
    return output_layout_;
  }
  [[nodiscard]] cublasLtMatmulPreference_t preference() const noexcept {
    return preference_;
  }

 private:
  cublasLtHandle_t handle_ = nullptr;
  cublasLtMatmulDesc_t operation_ = nullptr;
  cublasLtMatrixLayout_t weight_layout_ = nullptr;
  cublasLtMatrixLayout_t activation_layout_ = nullptr;
  cublasLtMatrixLayout_t output_layout_ = nullptr;
  cublasLtMatmulPreference_t preference_ = nullptr;
};

class Execution {
 public:
  Execution() = default;
  Execution(const Execution&) = delete;
  Execution& operator=(const Execution&) = delete;

  ~Execution() {
    if (stop_ != nullptr) {
      (void)cudaEventDestroy(stop_);
    }
    if (start_ != nullptr) {
      (void)cudaEventDestroy(start_);
    }
    if (done_ != nullptr) {
      (void)cudaEventDestroy(done_);
    }
    if (fork_ != nullptr) {
      (void)cudaEventDestroy(fork_);
    }
    if (auxiliary_ != nullptr) {
      (void)cudaStreamDestroy(auxiliary_);
    }
    if (main_ != nullptr) {
      (void)cudaStreamDestroy(main_);
    }
  }

  [[nodiscard]] bool create(TestContext& test) {
    bool ready = test.cuda_ok(
        cudaStreamCreateWithFlags(&main_, cudaStreamNonBlocking),
        "create main stream");
    ready = ready && test.cuda_ok(
                           cudaStreamCreateWithFlags(&auxiliary_,
                                                     cudaStreamNonBlocking),
                           "create auxiliary stream");
    ready = ready && test.cuda_ok(
                           cudaEventCreateWithFlags(&fork_,
                                                    cudaEventDisableTiming),
                           "create fork event");
    ready = ready && test.cuda_ok(
                           cudaEventCreateWithFlags(&done_,
                                                    cudaEventDisableTiming),
                           "create done event");
    ready = ready &&
            test.cuda_ok(cudaEventCreate(&start_), "create timing start");
    ready = ready &&
            test.cuda_ok(cudaEventCreate(&stop_), "create timing stop");
    return ready;
  }

  [[nodiscard]] cudaStream_t main() const noexcept { return main_; }
  [[nodiscard]] cudaStream_t auxiliary() const noexcept { return auxiliary_; }
  [[nodiscard]] cudaEvent_t fork() const noexcept { return fork_; }
  [[nodiscard]] cudaEvent_t done() const noexcept { return done_; }
  [[nodiscard]] cudaEvent_t start() const noexcept { return start_; }
  [[nodiscard]] cudaEvent_t stop() const noexcept { return stop_; }

 private:
  cudaStream_t main_ = nullptr;
  cudaStream_t auxiliary_ = nullptr;
  cudaEvent_t fork_ = nullptr;
  cudaEvent_t done_ = nullptr;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

enum class NativeKernel : std::uint8_t {
  kM64N256PairLookup,
  kM128N256HorizontalP0,
  kM128N256HorizontalNamedP1,
};

[[nodiscard]] const char* native_kernel_name(
    const NativeKernel kernel) noexcept {
  switch (kernel) {
    case NativeKernel::kM64N256PairLookup:
      return "m64n256_pairlookup";
    case NativeKernel::kM128N256HorizontalP0:
      return "m128n256_horizontal_p0";
    case NativeKernel::kM128N256HorizontalNamedP1:
      return "m128n256_horizontal_named_p1";
  }
  return "unknown";
}

struct Fixture {
  DeviceBuffer<__nv_bfloat16> activation;
  DeviceBuffer<std::uint8_t> gate_packed;
  DeviceBuffer<std::uint8_t> gate_scales;
  DeviceBuffer<std::uint8_t> up_packed;
  DeviceBuffer<std::uint8_t> up_scales;
  DeviceBuffer<__nv_bfloat16> scratch0;
  DeviceBuffer<__nv_bfloat16> scratch1;
  DeviceBuffer<std::uint16_t> reference_gate_store;
  DeviceBuffer<std::uint16_t> reference_up_store;
  DeviceBuffer<std::uint16_t> candidate_gate_store;
  DeviceBuffer<std::uint16_t> candidate_up_store;
  DeviceBuffer<unsigned long long> validation;
  float gate_weight_scale_2 = kSyntheticGateWeightScale2;
  float up_weight_scale_2 = kSyntheticUpWeightScale2;
  bool checkpoint_payload = false;
  NativeKernel native_kernel = NativeKernel::kM64N256PairLookup;

  [[nodiscard]] std::uint16_t* reference_gate() noexcept {
    return reference_gate_store.get() + kGuardElements;
  }
  [[nodiscard]] std::uint16_t* reference_up() noexcept {
    return reference_up_store.get() + kGuardElements;
  }
  [[nodiscard]] std::uint16_t* candidate_gate() noexcept {
    return candidate_gate_store.get() + kGuardElements;
  }
  [[nodiscard]] std::uint16_t* candidate_up() noexcept {
    return candidate_up_store.get() + kGuardElements;
  }

  [[nodiscard]] std::uint64_t planned_bytes() const noexcept {
    return static_cast<std::uint64_t>(activation.bytes()) +
           gate_packed.bytes() + gate_scales.bytes() + up_packed.bytes() +
           up_scales.bytes() + scratch0.bytes() + scratch1.bytes() +
           reference_gate_store.bytes() + reference_up_store.bytes() +
           candidate_gate_store.bytes() + candidate_up_store.bytes() +
           validation.bytes();
  }
};

struct SelectedAlgorithm {
  cublasLtMatmulAlgo_t value{};
  int heuristic_index = -1;
  double milliseconds = std::numeric_limits<double>::quiet_NaN();
};

enum class Variant : std::uint8_t {
  kProduction,
  kSerialOneScratch,
  kNaiveDual,
  kStaggeredDual,
  kNativeSerial,
  kNativeDual,
};

[[nodiscard]] const char* variant_name(const Variant variant) noexcept {
  switch (variant) {
    case Variant::kProduction:
      return "production_m128_fork_join";
    case Variant::kSerialOneScratch:
      return "A_serial_one_handle_one_scratch";
    case Variant::kNaiveDual:
      return "B_naive_two_handle_two_scratch";
    case Variant::kStaggeredDual:
      return "C_staggered_two_handle_two_scratch";
    case Variant::kNativeSerial:
      return "D_native_selected_serial";
    case Variant::kNativeDual:
      return "E_native_selected_dual";
  }
  return "unknown";
}

__global__ void fill_deterministic_bf16_kernel(__nv_bfloat16* const values,
                                                const std::size_t count,
                                                const std::uint32_t salt,
                                                const float scale) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  std::uint32_t code = static_cast<std::uint32_t>(index) ^ salt;
  code ^= code >> 16U;
  code *= 0x7feb'352dU;
  code ^= code >> 15U;
  code *= 0x846c'a68bU;
  code ^= code >> 16U;
  const int centered = static_cast<int>(code % 17U) - 8;
  values[index] = __float2bfloat16_rn(static_cast<float>(centered) * scale);
}

__device__ __forceinline__ std::uint32_t mix_u32(std::uint32_t code) {
  code ^= code >> 16U;
  code *= 0x7feb'352dU;
  code ^= code >> 15U;
  code *= 0x846c'a68bU;
  code ^= code >> 16U;
  return code;
}

__global__ void fill_canonical_nvfp4_kernel(
    std::uint8_t* const values, const std::size_t count,
    const std::uint32_t salt, const bool block_scales) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const std::uint32_t code =
      mix_u32(static_cast<std::uint32_t>(index) ^ salt);
  values[index] =
      block_scales ? static_cast<std::uint8_t>(0x50U + code % 24U)
                   : static_cast<std::uint8_t>(code);
}

__device__ __forceinline__ float decode_e4m3fn_device(
    const std::uint8_t bits) {
  const unsigned int sign =
      static_cast<unsigned int>(bits & 0x80U) << 24U;
  const unsigned int magnitude = static_cast<unsigned int>(bits & 0x7fU);
  const unsigned int exponent = magnitude >> 3U;
  const unsigned int mantissa = magnitude & 0x07U;
  if (magnitude == 0x7fU) {
    return __uint_as_float(sign | 0x7fc0'0000U);
  }
  if (exponent == 0U) {
    if (mantissa == 0U) {
      return __uint_as_float(sign);
    }
    const unsigned int leading =
        mantissa >= 4U ? 2U : (mantissa >= 2U ? 1U : 0U);
    const unsigned int fp32_exponent = 118U + leading;
    const unsigned int fp32_mantissa =
        (mantissa - (1U << leading)) << (23U - leading);
    return __uint_as_float(sign | (fp32_exponent << 23U) | fp32_mantissa);
  }
  return __uint_as_float(sign | ((120U + exponent) << 23U) |
                         (mantissa << 20U));
}

__device__ __forceinline__ float decode_e2m1_device(
    const std::uint8_t nibble) {
  const unsigned int sign =
      static_cast<unsigned int>(nibble & 0x08U) << 28U;
  const unsigned int magnitude = static_cast<unsigned int>(nibble & 0x07U);
  const unsigned int nonzero_mask =
      0U - static_cast<unsigned int>(magnitude != 0U);
  const unsigned int mantissa =
      ((magnitude & 1U) & static_cast<unsigned int>(magnitude > 1U)) << 22U;
  const unsigned int finite_bits =
      ((126U + (magnitude >> 1U)) << 23U) | mantissa;
  return __uint_as_float(sign | (finite_bits & nonzero_mask));
}

__global__ __launch_bounds__(256, 4)
void dequantize_nvfp4_contiguous_kernel(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales,
    __nv_bfloat16* const canonical_bf16) {
  constexpr unsigned int kPackedPerRow = kK / 2U;
  constexpr unsigned int kScalesPerRow = kK / 16U;
  constexpr unsigned int kThreads = 256U;
  constexpr unsigned int kPasses = kPackedPerRow / kThreads;
  static_assert(kPackedPerRow == kPasses * kThreads);
  const unsigned int n = blockIdx.x;
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
  const std::size_t packed_base =
      static_cast<std::size_t>(n) * kPackedPerRow;
  const std::size_t scale_base =
      static_cast<std::size_t>(n) * kScalesPerRow;
  auto* const output_pairs = reinterpret_cast<std::uint32_t*>(canonical_bf16);

  std::uint8_t packed_values[kPasses];
  std::uint32_t scale_words[kPasses];
#pragma unroll
  for (unsigned int pass = 0U; pass < kPasses; ++pass) {
    const unsigned int packed_k = threadIdx.x + pass * kThreads;
    packed_values[pass] = packed_weights[packed_base + packed_k];
    scale_words[pass] = 0U;
    if (lane == 0U) {
      const std::size_t word_index =
          scale_base + pass * (kThreads / 8U) + warp * 4U;
      scale_words[pass] = *reinterpret_cast<const std::uint32_t*>(
          block_scales + word_index);
    }
  }

#pragma unroll
  for (unsigned int pass = 0U; pass < kPasses; ++pass) {
    const unsigned int packed_k = threadIdx.x + pass * kThreads;
    const std::uint32_t scale_word =
        __shfl_sync(0xffff'ffffU, scale_words[pass], 0);
    const std::uint8_t scale_code = static_cast<std::uint8_t>(
        scale_word >> ((lane >> 3U) * 8U));
    const float scale = decode_e4m3fn_device(scale_code);
    const std::uint8_t packed = packed_values[pass];
    const __nv_bfloat16 low = __float2bfloat16_rn(
        decode_e2m1_device(packed & 0x0fU) * scale);
    const __nv_bfloat16 high = __float2bfloat16_rn(
        decode_e2m1_device(packed >> 4U) * scale);
    output_pairs[packed_base + packed_k] =
        static_cast<std::uint32_t>(__bfloat16_as_ushort(low)) |
        (static_cast<std::uint32_t>(__bfloat16_as_ushort(high)) << 16U);
  }
}

__global__ void validate_pair_kernel(
    const std::uint16_t* const gate,
    const std::uint16_t* const gate_reference,
    const std::uint16_t* const up, const std::uint16_t* const up_reference,
    unsigned long long* const statistics) {
  unsigned long long gate_mismatch = 0U;
  unsigned long long gate_nonfinite = 0U;
  unsigned long long gate_sum = 0U;
  unsigned long long up_mismatch = 0U;
  unsigned long long up_nonfinite = 0U;
  unsigned long long up_sum = 0U;
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < kCElements; index += stride) {
    const std::uint16_t gate_value = gate[index];
    const std::uint16_t up_value = up[index];
    gate_mismatch += gate_value != gate_reference[index] ? 1U : 0U;
    up_mismatch += up_value != up_reference[index] ? 1U : 0U;
    gate_nonfinite +=
        isfinite(__bfloat162float(__ushort_as_bfloat16(gate_value))) ? 0U : 1U;
    up_nonfinite +=
        isfinite(__bfloat162float(__ushort_as_bfloat16(up_value))) ? 0U : 1U;
    gate_sum += gate_value;
    up_sum += up_value;
  }
  if (gate_mismatch != 0U) {
    atomicAdd(statistics + 0U, gate_mismatch);
  }
  if (gate_nonfinite != 0U) {
    atomicAdd(statistics + 1U, gate_nonfinite);
  }
  atomicAdd(statistics + 2U, gate_sum);
  if (up_mismatch != 0U) {
    atomicAdd(statistics + 3U, up_mismatch);
  }
  if (up_nonfinite != 0U) {
    atomicAdd(statistics + 4U, up_nonfinite);
  }
  atomicAdd(statistics + 5U, up_sum);
}

[[nodiscard]] bool read_u64(const std::string& path,
                            std::uint64_t& value) {
  std::ifstream input(path);
  return static_cast<bool>(input >> value);
}

struct ClockState {
  std::uint64_t gpu_min = 0U;
  std::uint64_t gpu_current = 0U;
  std::uint64_t gpu_max = 0U;
  std::uint64_t emc_min = 0U;
  std::uint64_t emc_current = 0U;
  std::uint64_t emc_max = 0U;
};

[[nodiscard]] std::optional<ClockState> read_clock_state() {
  constexpr const char* kGpuRoot =
      "/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/";
  constexpr const char* kEmcRoot =
      "/sys/devices/platform/bwmgr/devfreq/bwmgr/";
  ClockState state;
  if (!read_u64(std::string(kGpuRoot) + "min_freq", state.gpu_min) ||
      !read_u64(std::string(kGpuRoot) + "cur_freq", state.gpu_current) ||
      !read_u64(std::string(kGpuRoot) + "max_freq", state.gpu_max) ||
      !read_u64(std::string(kEmcRoot) + "min_freq", state.emc_min) ||
      !read_u64(std::string(kEmcRoot) + "cur_freq", state.emc_current) ||
      !read_u64(std::string(kEmcRoot) + "max_freq", state.emc_max)) {
    return std::nullopt;
  }
  return state;
}

[[nodiscard]] bool clocks_are_fixed(const ClockState& state) noexcept {
  return state.gpu_min == state.gpu_max &&
         state.gpu_current == state.gpu_max && state.emc_min == state.emc_max &&
         state.emc_current == state.emc_max;
}

[[nodiscard]] bool launch_dequantize(const std::uint8_t* const packed,
                                     const std::uint8_t* const scales,
                                     __nv_bfloat16* const output,
                                     const cudaStream_t stream) {
  dequantize_nvfp4_contiguous_kernel<<<static_cast<unsigned int>(kN), 256U, 0U,
                                       stream>>>(packed, scales, output);
  return cudaGetLastError() == cudaSuccess;
}

[[nodiscard]] bool launch_lt(const LtObjects& lt,
                             const cublasLtMatmulAlgo_t& algorithm,
                             const float alpha,
                             const __nv_bfloat16* const weight,
                             const __nv_bfloat16* const activation,
                             std::uint16_t* const output,
                             const cudaStream_t stream) {
  constexpr float kBeta = 0.0F;
  return cublasLtMatmul(
             lt.handle(), lt.operation(), &alpha, weight, lt.weight_layout(),
             activation, lt.activation_layout(), &kBeta, output,
             lt.output_layout(), output, lt.output_layout(), &algorithm,
             nullptr, 0U, stream) == CUBLAS_STATUS_SUCCESS;
}

[[nodiscard]] double measure_lt_only(
    TestContext& test, const LtObjects& lt,
    const cublasLtMatmulAlgo_t& algorithm,
    const __nv_bfloat16* const weight,
    const __nv_bfloat16* const activation, std::uint16_t* const output,
    const cudaStream_t stream, const std::string& label) {
  bool ready = true;
  for (int warmup = 0; warmup < kSelectionWarmups; ++warmup) {
    ready = launch_lt(lt, algorithm, 1.0F, weight, activation, output, stream) &&
            ready;
  }
  ready = test.cuda_ok(cudaStreamSynchronize(stream),
                       label + " warmup synchronize") &&
          ready;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  ready = test.cuda_ok(cudaEventCreate(&start), label + " create start") &&
          ready;
  ready = test.cuda_ok(cudaEventCreate(&stop), label + " create stop") && ready;
  ready = test.cuda_ok(cudaEventRecord(start, stream), label + " record start") &&
          ready;
  for (int iteration = 0; iteration < kSelectionIterations; ++iteration) {
    ready = launch_lt(lt, algorithm, 1.0F, weight, activation, output, stream) &&
            ready;
  }
  ready = test.cuda_ok(cudaEventRecord(stop, stream), label + " record stop") &&
          ready;
  ready = test.cuda_ok(cudaEventSynchronize(stop), label + " synchronize stop") &&
          ready;
  float total = 0.0F;
  ready = test.cuda_ok(cudaEventElapsedTime(&total, start, stop),
                       label + " elapsed") &&
          ready;
  if (stop != nullptr) {
    (void)cudaEventDestroy(stop);
  }
  if (start != nullptr) {
    (void)cudaEventDestroy(start);
  }
  if (!ready) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return static_cast<double>(total) / kSelectionIterations;
}

[[nodiscard]] std::optional<SelectedAlgorithm> select_algorithm(
    TestContext& test, const LtObjects& lt,
    const __nv_bfloat16* const weight,
    const __nv_bfloat16* const activation, std::uint16_t* const output,
    const cudaStream_t stream) {
  std::array<cublasLtMatmulHeuristicResult_t, kMaximumHeuristics> heuristics{};
  int returned = 0;
  if (!test.lt_ok(cublasLtMatmulAlgoGetHeuristic(
                      lt.handle(), lt.operation(), lt.weight_layout(),
                      lt.activation_layout(), lt.output_layout(),
                      lt.output_layout(), lt.preference(), kMaximumHeuristics,
                      heuristics.data(), &returned),
                  "query zero-workspace algorithms")) {
    return std::nullopt;
  }
  test.expect(returned > 0, "zero-workspace heuristic list is nonempty");
  SelectedAlgorithm selected;
  selected.milliseconds = std::numeric_limits<double>::infinity();
  for (int index = 0; index < returned; ++index) {
    const auto& result = heuristics[static_cast<std::size_t>(index)];
    if (result.state != CUBLAS_STATUS_SUCCESS || result.workspaceSize != 0U) {
      continue;
    }
    const double milliseconds = measure_lt_only(
        test, lt, result.algo, weight, activation, output, stream,
        "select zero-workspace algorithm " + std::to_string(index));
    std::cout << "NVFP4_PAIR_LT_HEURISTIC: index=" << index
              << " workspace_bytes=" << result.workspaceSize
              << " milliseconds=" << milliseconds << '\n';
    if (std::isfinite(milliseconds) && milliseconds < selected.milliseconds) {
      selected.value = result.algo;
      selected.heuristic_index = index;
      selected.milliseconds = milliseconds;
    }
  }
  if (selected.heuristic_index < 0) {
    test.expect(false, "at least one zero-workspace algorithm executes");
    return std::nullopt;
  }
  return selected;
}

[[nodiscard]] bool launch_production_branch(
    const std::uint8_t* const packed, const std::uint8_t* const scales,
    const float weight_scale_2, const __nv_bfloat16* const activation,
    std::uint16_t* const output, const cudaStream_t stream) {
  return q3x::kernels::
             launch_sm87_nvfp4_w4a16_whole_chunk_gate_up_branch_gemm_bf16_cuda(
                 packed, scales, weight_scale_2,
                 reinterpret_cast<const std::uint16_t*>(activation), kM, kN,
                 kK, output, static_cast<void*>(stream)) ==
         static_cast<int>(cudaSuccess);
}

[[nodiscard]] int launch_native_branch_status(
    const NativeKernel native_kernel,
    const std::uint8_t* const packed, const std::uint8_t* const scales,
    const float weight_scale_2, const __nv_bfloat16* const activation,
    const std::size_t tokens, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    const cudaStream_t stream) {
  int status = static_cast<int>(cudaErrorInvalidValue);
  if (native_kernel == NativeKernel::kM64N256PairLookup) {
    status = q3x::kernels::
        launch_sm87_nvfp4_w4a16_gate_c512_m64_n256_k64_cp_async_capairlookup_test_cuda(
            packed, scales, weight_scale_2,
            reinterpret_cast<const std::uint16_t*>(activation), tokens, rows,
            columns, output, static_cast<void*>(stream));
  } else if (native_kernel == NativeKernel::kM128N256HorizontalP0) {
    status = q3x::kernels::
        launch_sm87_nvfp4_w4a16_gate_c512_m128_n256_horizontal_p0_test_cuda(
            packed, scales, weight_scale_2,
            reinterpret_cast<const std::uint16_t*>(activation), tokens, rows,
            columns, output, static_cast<void*>(stream));
  } else {
    status = q3x::kernels::
        launch_sm87_nvfp4_w4a16_gate_c512_m128_n256_horizontal_named_p1_test_cuda(
            packed, scales, weight_scale_2,
            reinterpret_cast<const std::uint16_t*>(activation), tokens, rows,
            columns, output, static_cast<void*>(stream));
  }
  return status;
}

[[nodiscard]] bool launch_native_branch(
    const NativeKernel native_kernel,
    const std::uint8_t* const packed, const std::uint8_t* const scales,
    const float weight_scale_2, const __nv_bfloat16* const activation,
    std::uint16_t* const output, const cudaStream_t stream) {
  return launch_native_branch_status(native_kernel, packed, scales,
                                     weight_scale_2, activation, kM, kN, kK,
                                     output, stream) ==
         static_cast<int>(cudaSuccess);
}

[[nodiscard]] bool launch_variant(
    const Variant variant, Fixture& fixture, const Execution& execution,
    const LtObjects& main_lt, const LtObjects& auxiliary_lt,
    const SelectedAlgorithm& selected) {
  const auto main = execution.main();
  const auto auxiliary = execution.auxiliary();
  if (variant == Variant::kProduction) {
    bool ready = cudaEventRecord(execution.fork(), main) == cudaSuccess;
    ready = ready &&
            cudaStreamWaitEvent(auxiliary, execution.fork(), 0U) == cudaSuccess;
    ready = ready && launch_production_branch(
                         fixture.gate_packed.get(), fixture.gate_scales.get(),
                         fixture.gate_weight_scale_2, fixture.activation.get(),
                         fixture.candidate_gate(), main);
    ready = ready && launch_production_branch(
                         fixture.up_packed.get(), fixture.up_scales.get(),
                         fixture.up_weight_scale_2, fixture.activation.get(),
                         fixture.candidate_up(), auxiliary);
    ready = ready && cudaEventRecord(execution.done(), auxiliary) == cudaSuccess;
    ready = ready &&
            cudaStreamWaitEvent(main, execution.done(), 0U) == cudaSuccess;
    return ready;
  }

  if (variant == Variant::kSerialOneScratch) {
    return launch_dequantize(fixture.gate_packed.get(),
                             fixture.gate_scales.get(), fixture.scratch0.get(),
                             main) &&
           launch_lt(main_lt, selected.value, fixture.gate_weight_scale_2,
                     fixture.scratch0.get(), fixture.activation.get(),
                     fixture.candidate_gate(), main) &&
           launch_dequantize(fixture.up_packed.get(), fixture.up_scales.get(),
                             fixture.scratch0.get(), main) &&
           launch_lt(main_lt, selected.value, fixture.up_weight_scale_2,
                     fixture.scratch0.get(), fixture.activation.get(),
                     fixture.candidate_up(), main);
  }

  if (variant == Variant::kNativeSerial) {
    return launch_native_branch(
               fixture.native_kernel,
               fixture.gate_packed.get(), fixture.gate_scales.get(),
               fixture.gate_weight_scale_2, fixture.activation.get(),
               fixture.candidate_gate(), main) &&
           launch_native_branch(
               fixture.native_kernel,
               fixture.up_packed.get(), fixture.up_scales.get(),
               fixture.up_weight_scale_2, fixture.activation.get(),
               fixture.candidate_up(), main);
  }

  if (variant == Variant::kNativeDual) {
    bool ready = cudaEventRecord(execution.fork(), main) == cudaSuccess;
    ready = ready &&
            cudaStreamWaitEvent(auxiliary, execution.fork(), 0U) == cudaSuccess;
    ready = ready && launch_native_branch(
                         fixture.native_kernel,
                         fixture.gate_packed.get(), fixture.gate_scales.get(),
                         fixture.gate_weight_scale_2, fixture.activation.get(),
                         fixture.candidate_gate(), main);
    ready = ready && launch_native_branch(
                         fixture.native_kernel,
                         fixture.up_packed.get(), fixture.up_scales.get(),
                         fixture.up_weight_scale_2, fixture.activation.get(),
                         fixture.candidate_up(), auxiliary);
    ready = ready && cudaEventRecord(execution.done(), auxiliary) == cudaSuccess;
    ready = ready &&
            cudaStreamWaitEvent(main, execution.done(), 0U) == cudaSuccess;
    return ready;
  }

  if (variant == Variant::kNaiveDual) {
    bool ready = cudaEventRecord(execution.fork(), main) == cudaSuccess;
    ready = ready &&
            cudaStreamWaitEvent(auxiliary, execution.fork(), 0U) == cudaSuccess;
    ready = ready && launch_dequantize(
                         fixture.gate_packed.get(), fixture.gate_scales.get(),
                         fixture.scratch0.get(), main);
    ready = ready && launch_lt(
                         main_lt, selected.value, fixture.gate_weight_scale_2,
                         fixture.scratch0.get(), fixture.activation.get(),
                         fixture.candidate_gate(), main);
    ready = ready && launch_dequantize(
                         fixture.up_packed.get(), fixture.up_scales.get(),
                         fixture.scratch1.get(), auxiliary);
    ready = ready && launch_lt(
                         auxiliary_lt, selected.value,
                         fixture.up_weight_scale_2,
                         fixture.scratch1.get(), fixture.activation.get(),
                         fixture.candidate_up(), auxiliary);
    ready = ready && cudaEventRecord(execution.done(), auxiliary) == cudaSuccess;
    ready = ready &&
            cudaStreamWaitEvent(main, execution.done(), 0U) == cudaSuccess;
    return ready;
  }

  // Gate dequantization is deliberately completed before the auxiliary branch
  // joins. Gate Lt on main can then overlap Up dequantization on auxiliary;
  // Up Lt follows its dequantization in stream order before the final join.
  bool ready = launch_dequantize(
      fixture.gate_packed.get(), fixture.gate_scales.get(),
      fixture.scratch0.get(), main);
  ready = ready && cudaEventRecord(execution.fork(), main) == cudaSuccess;
  ready = ready &&
          cudaStreamWaitEvent(auxiliary, execution.fork(), 0U) == cudaSuccess;
  ready = ready && launch_lt(main_lt, selected.value,
                             fixture.gate_weight_scale_2,
                             fixture.scratch0.get(), fixture.activation.get(),
                             fixture.candidate_gate(), main);
  ready = ready && launch_dequantize(
                       fixture.up_packed.get(), fixture.up_scales.get(),
                       fixture.scratch1.get(), auxiliary);
  ready = ready && launch_lt(auxiliary_lt, selected.value,
                             fixture.up_weight_scale_2,
                             fixture.scratch1.get(), fixture.activation.get(),
                             fixture.candidate_up(), auxiliary);
  ready = ready && cudaEventRecord(execution.done(), auxiliary) == cudaSuccess;
  ready = ready &&
          cudaStreamWaitEvent(main, execution.done(), 0U) == cudaSuccess;
  return ready;
}

[[nodiscard]] bool poison_outputs(TestContext& test, Fixture& fixture,
                                  const Execution& execution,
                                  const std::string& label) {
  bool ready = test.cuda_ok(
      cudaMemsetAsync(fixture.candidate_gate_store.get(), kGuardPoisonByte,
                      fixture.candidate_gate_store.bytes(), execution.main()),
      label + " poison Gate output");
  ready = ready && test.cuda_ok(
                       cudaMemsetAsync(fixture.candidate_up_store.get(),
                                       kGuardPoisonByte,
                                       fixture.candidate_up_store.bytes(),
                                       execution.main()),
                       label + " poison Up output");
  return ready;
}

[[nodiscard]] bool check_one_guard(TestContext& test,
                                   const DeviceBuffer<std::uint16_t>& buffer,
                                   const cudaStream_t stream,
                                   const std::string& label) {
  std::array<std::uint16_t, kGuardElements> prefix{};
  std::array<std::uint16_t, kGuardElements> suffix{};
  bool ready = test.cuda_ok(
      cudaMemcpyAsync(prefix.data(), buffer.get(), sizeof(prefix),
                      cudaMemcpyDeviceToHost, stream),
      label + " copy prefix guard");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(suffix.data(),
                                       buffer.get() + kGuardElements +
                                           kCElements,
                                       sizeof(suffix), cudaMemcpyDeviceToHost,
                                       stream),
                       label + " copy suffix guard");
  ready = ready &&
          test.cuda_ok(cudaStreamSynchronize(stream), label + " guard sync");
  const bool intact =
      std::all_of(prefix.begin(), prefix.end(), [](const std::uint16_t value) {
        return value == kGuardPoison;
      }) &&
      std::all_of(suffix.begin(), suffix.end(), [](const std::uint16_t value) {
        return value == kGuardPoison;
      });
  test.expect(intact, label + " guards remain intact");
  return ready && intact;
}

struct PairValidation {
  std::array<unsigned long long, 6U> values{};

  [[nodiscard]] bool exact_finite_nonzero() const noexcept {
    return values[0] == 0U && values[1] == 0U && values[2] != 0U &&
           values[3] == 0U && values[4] == 0U && values[5] != 0U;
  }
};

[[nodiscard]] bool validate_pair(TestContext& test, Fixture& fixture,
                                 const Execution& execution,
                                 const std::string& label,
                                 PairValidation& result) {
  bool ready = test.cuda_ok(
      cudaMemsetAsync(fixture.validation.get(), 0, fixture.validation.bytes(),
                      execution.main()),
      label + " zero validation");
  validate_pair_kernel<<<256U, 256U, 0U, execution.main()>>>(
      fixture.candidate_gate(), fixture.reference_gate(),
      fixture.candidate_up(), fixture.reference_up(), fixture.validation.get());
  ready = test.cuda_ok(cudaGetLastError(), label + " launch validation") &&
          ready;
  ready = test.cuda_ok(
              cudaMemcpyAsync(result.values.data(), fixture.validation.get(),
                              sizeof(result.values), cudaMemcpyDeviceToHost,
                              execution.main()),
              label + " copy validation") &&
          ready;
  ready =
      test.cuda_ok(cudaStreamSynchronize(execution.main()), label + " sync") &&
      ready;
  const bool guards =
      check_one_guard(test, fixture.candidate_gate_store, execution.main(),
                      label + " Gate") &&
      check_one_guard(test, fixture.candidate_up_store, execution.main(),
                      label + " Up");
  const bool exact = result.exact_finite_nonzero();
  test.expect(exact, label + " matches production and is finite");
  std::cout << "NVFP4_PAIR_VALIDATION: label=" << label
            << " gate_mismatches=" << result.values[0] << '/' << kCElements
            << " gate_nonfinite=" << result.values[1]
            << " gate_encoded_sum=" << result.values[2]
            << " up_mismatches=" << result.values[3] << '/' << kCElements
            << " up_nonfinite=" << result.values[4]
            << " up_encoded_sum=" << result.values[5]
            << " guards=" << (guards ? "intact" : "BAD")
            << " gate=" << (ready && guards && exact ? "PASS" : "FAIL")
            << '\n';
  return ready && guards && exact;
}

[[nodiscard]] bool generate_production_reference(
    TestContext& test, Fixture& fixture, const Execution& execution) {
  bool ready = test.cuda_ok(
      cudaMemsetAsync(fixture.reference_gate_store.get(), kGuardPoisonByte,
                      fixture.reference_gate_store.bytes(), execution.main()),
      "poison reference Gate");
  ready = ready && test.cuda_ok(
                       cudaMemsetAsync(fixture.reference_up_store.get(),
                                       kGuardPoisonByte,
                                       fixture.reference_up_store.bytes(),
                                       execution.main()),
                       "poison reference Up");
  ready = ready &&
          cudaEventRecord(execution.fork(), execution.main()) == cudaSuccess;
  ready = ready && cudaStreamWaitEvent(execution.auxiliary(), execution.fork(),
                                       0U) == cudaSuccess;
  ready = ready && launch_production_branch(
                       fixture.gate_packed.get(), fixture.gate_scales.get(),
                       fixture.gate_weight_scale_2, fixture.activation.get(),
                       fixture.reference_gate(), execution.main());
  ready = ready && launch_production_branch(
                       fixture.up_packed.get(), fixture.up_scales.get(),
                       fixture.up_weight_scale_2, fixture.activation.get(),
                       fixture.reference_up(), execution.auxiliary());
  ready = ready &&
          cudaEventRecord(execution.done(), execution.auxiliary()) == cudaSuccess;
  ready = ready && cudaStreamWaitEvent(execution.main(), execution.done(), 0U) ==
                       cudaSuccess;
  ready = test.cuda_ok(cudaStreamSynchronize(execution.main()),
                       "production reference synchronize") &&
          ready;
  const bool guards =
      check_one_guard(test, fixture.reference_gate_store, execution.main(),
                      "reference Gate") &&
      check_one_guard(test, fixture.reference_up_store, execution.main(),
                      "reference Up");
  return ready && guards;
}

[[nodiscard]] bool run_eager_correctness(
    TestContext& test, Fixture& fixture, const Execution& execution,
    const LtObjects& main_lt, const LtObjects& auxiliary_lt,
    const SelectedAlgorithm& selected, const Variant variant) {
  bool ready = true;
  for (int replay = 0; replay < 2; ++replay) {
    const std::string label = std::string(variant_name(variant)) +
                              "_eager_replay_" + std::to_string(replay + 1);
    ready = poison_outputs(test, fixture, execution, label) && ready;
    ready = launch_variant(variant, fixture, execution, main_lt, auxiliary_lt,
                           selected) &&
            ready;
    PairValidation validation;
    ready = validate_pair(test, fixture, execution, label, validation) && ready;
  }
  return ready;
}

struct GraphCounts {
  std::size_t total = 0U;
  std::size_t kernels = 0U;
  std::size_t event_records = 0U;
  std::size_t event_waits = 0U;
  std::size_t other = 0U;
};

[[nodiscard]] bool query_graph_counts(TestContext& test, cudaGraph_t graph,
                                      GraphCounts& counts,
                                      const std::string& label) {
  bool ready = test.cuda_ok(cudaGraphGetNodes(graph, nullptr, &counts.total),
                            label + " query node count");
  std::vector<cudaGraphNode_t> nodes(counts.total);
  std::size_t returned = counts.total;
  ready = test.cuda_ok(cudaGraphGetNodes(graph, nodes.data(), &returned),
                       label + " query nodes") &&
          ready;
  ready = ready && returned == counts.total;
  for (const cudaGraphNode_t node : nodes) {
    cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
    ready = test.cuda_ok(cudaGraphNodeGetType(node, &type),
                         label + " query node type") &&
            ready;
    if (type == cudaGraphNodeTypeKernel) {
      ++counts.kernels;
    } else if (type == cudaGraphNodeTypeEventRecord) {
      ++counts.event_records;
    } else if (type == cudaGraphNodeTypeWaitEvent) {
      ++counts.event_waits;
    } else {
      ++counts.other;
    }
  }
  return ready;
}

[[nodiscard]] bool run_graph_correctness(
    TestContext& test, Fixture& fixture, const Execution& execution,
    const LtObjects& main_lt, const LtObjects& auxiliary_lt,
    const SelectedAlgorithm& selected, const Variant variant) {
  const std::string label = variant_name(variant);
  bool ready =
      test.cuda_ok(cudaStreamSynchronize(execution.main()),
                   label + " pre-capture main sync");
  ready = test.cuda_ok(cudaStreamSynchronize(execution.auxiliary()),
                       label + " pre-capture auxiliary sync") &&
          ready;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  ready = test.cuda_ok(
              cudaStreamBeginCapture(execution.main(),
                                     cudaStreamCaptureModeThreadLocal),
              label + " begin capture") &&
          ready;
  const bool launched =
      launch_variant(variant, fixture, execution, main_lt, auxiliary_lt,
                     selected);
  const cudaError_t end_status =
      cudaStreamEndCapture(execution.main(), &graph);
  ready = test.cuda_ok(end_status, label + " end capture") && launched && ready;
  GraphCounts counts;
  if (ready) {
    ready = query_graph_counts(test, graph, counts, label) && ready;
    const bool native_or_production =
        variant == Variant::kProduction ||
        variant == Variant::kNativeSerial ||
        variant == Variant::kNativeDual;
    const std::size_t minimum_kernels = native_or_production ? 2U : 4U;
    test.expect(counts.kernels >= minimum_kernels,
                label + " graph contains expected compute nodes");
    ready = ready && counts.kernels >= minimum_kernels;
  }
  if (ready) {
    ready = test.cuda_ok(
                cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0U),
                label + " instantiate") &&
            ready;
  }
  for (int replay = 0; ready && replay < 2; ++replay) {
    const std::string replay_label =
        label + "_graph_replay_" + std::to_string(replay + 1);
    ready = poison_outputs(test, fixture, execution, replay_label) && ready;
    ready = test.cuda_ok(cudaGraphLaunch(executable, execution.main()),
                         replay_label + " launch") &&
            ready;
    PairValidation validation;
    ready = validate_pair(test, fixture, execution, replay_label, validation) &&
            ready;
  }
  if (executable != nullptr) {
    ready = test.cuda_ok(cudaGraphExecDestroy(executable),
                         label + " destroy executable") &&
            ready;
  }
  if (graph != nullptr) {
    ready = test.cuda_ok(cudaGraphDestroy(graph), label + " destroy graph") &&
            ready;
  }
  std::cout << "NVFP4_PAIR_GRAPH: variant=" << label
            << " total_nodes=" << counts.total
            << " kernel_nodes=" << counts.kernels
            << " event_record_nodes=" << counts.event_records
            << " event_wait_nodes=" << counts.event_waits
            << " other_nodes=" << counts.other
            << " cold_and_warm_replays=2 gate="
            << (ready ? "PASS" : "FAIL") << '\n';
  return ready;
}

[[nodiscard]] double measure_variant(
    TestContext& test, Fixture& fixture, const Execution& execution,
    const LtObjects& main_lt, const LtObjects& auxiliary_lt,
    const SelectedAlgorithm& selected, const Variant variant,
    const std::string& label) {
  bool ready = true;
  for (int warmup = 0; warmup < kWarmups; ++warmup) {
    ready = launch_variant(variant, fixture, execution, main_lt, auxiliary_lt,
                           selected) &&
            ready;
  }
  ready = test.cuda_ok(cudaStreamSynchronize(execution.main()),
                       label + " warmup synchronize") &&
          ready;
  ready = test.cuda_ok(cudaEventRecord(execution.start(), execution.main()),
                       label + " record start") &&
          ready;
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    ready = launch_variant(variant, fixture, execution, main_lt, auxiliary_lt,
                           selected) &&
            ready;
  }
  ready = test.cuda_ok(cudaEventRecord(execution.stop(), execution.main()),
                       label + " record stop") &&
          ready;
  ready = test.cuda_ok(cudaEventSynchronize(execution.stop()),
                       label + " synchronize stop") &&
          ready;
  float total = 0.0F;
  ready = test.cuda_ok(cudaEventElapsedTime(&total, execution.start(),
                                            execution.stop()),
                       label + " elapsed") &&
          ready;
  if (!ready) {
    test.expect(false, label + " measurement completes");
    return std::numeric_limits<double>::quiet_NaN();
  }
  return static_cast<double>(total) / kIterations;
}

struct ComparisonResult {
  double baseline_milliseconds = std::numeric_limits<double>::quiet_NaN();
  double candidate_milliseconds = std::numeric_limits<double>::quiet_NaN();
  double speedup = std::numeric_limits<double>::quiet_NaN();
  bool every_round_positive = false;
};

[[nodiscard]] ComparisonResult run_bccb(
    TestContext& test, Fixture& fixture, const Execution& execution,
    const LtObjects& main_lt, const LtObjects& auxiliary_lt,
    const SelectedAlgorithm& selected, const Variant baseline,
    const Variant candidate, const std::string& comparison,
    const bool admission_enabled) {
  double baseline_sum = 0.0;
  double candidate_sum = 0.0;
  bool every_round_positive = true;
  for (int round = 0; round < kRounds; ++round) {
    const std::string prefix =
        comparison + "_round_" + std::to_string(round + 1) + '_';
    const double b1 = measure_variant(test, fixture, execution, main_lt,
                                      auxiliary_lt, selected, baseline,
                                      prefix + "B1");
    const double c1 = measure_variant(test, fixture, execution, main_lt,
                                      auxiliary_lt, selected, candidate,
                                      prefix + "C1");
    const double c2 = measure_variant(test, fixture, execution, main_lt,
                                      auxiliary_lt, selected, candidate,
                                      prefix + "C2");
    const double b2 = measure_variant(test, fixture, execution, main_lt,
                                      auxiliary_lt, selected, baseline,
                                      prefix + "B2");
    const bool finite = std::isfinite(b1) && std::isfinite(c1) &&
                        std::isfinite(c2) && std::isfinite(b2) && b1 > 0.0 &&
                        c1 > 0.0 && c2 > 0.0 && b2 > 0.0;
    const double speedup =
        finite ? (b1 + b2) / (c1 + c2)
               : std::numeric_limits<double>::quiet_NaN();
    every_round_positive =
        every_round_positive && finite && speedup > 1.0;
    if (finite) {
      baseline_sum += b1 + b2;
      candidate_sum += c1 + c2;
    }
    std::cout << "PERF_NVFP4_PAIR_ROUND: comparison=" << comparison
              << " round=" << round + 1 << " order=B-C-C-B"
              << " B1_ms=" << b1 << " C1_ms=" << c1
              << " C2_ms=" << c2 << " B2_ms=" << b2
              << " speedup=" << speedup
              << " strict_positive_gate="
              << (admission_enabled
                      ? (finite && speedup > 1.0 ? "PASS" : "FAIL")
                      : "NOT_RUN")
              << " admission="
              << (admission_enabled ? "DIAGNOSTIC_SUBGATE" : "NOT_RUN")
              << '\n';
  }
  ComparisonResult result;
  result.baseline_milliseconds = baseline_sum / (2.0 * kRounds);
  result.candidate_milliseconds = candidate_sum / (2.0 * kRounds);
  result.speedup = baseline_sum / candidate_sum;
  result.every_round_positive = every_round_positive;
  std::cout << "PERF_NVFP4_PAIR_AGGREGATE: comparison=" << comparison
            << " baseline_variant=" << variant_name(baseline)
            << " candidate_variant=" << variant_name(candidate)
            << " baseline_pair_ms=" << result.baseline_milliseconds
            << " candidate_pair_ms=" << result.candidate_milliseconds
            << " speedup=" << result.speedup
            << " every_round_strict_positive="
            << (result.every_round_positive ? "true" : "false")
            << " rounds=" << kRounds << " iterations=" << kIterations
            << " fixed_clock_required="
            << (admission_enabled ? "true" : "false")
            << " admission="
            << (admission_enabled ? "DIAGNOSTIC_SUBGATE" : "NOT_RUN")
            << '\n';
  return result;
}

[[nodiscard]] bool allocate_fixture(TestContext& test, Fixture& fixture) {
  bool ready = fixture.activation.allocate(test, kAElements,
                                           "allocate shared activation");
  ready = ready && fixture.gate_packed.allocate(
                       test, kPackedWeightBytes, "allocate Gate packed weight");
  ready = ready && fixture.gate_scales.allocate(
                       test, kBlockScaleBytes, "allocate Gate block scales");
  ready = ready && fixture.up_packed.allocate(
                       test, kPackedWeightBytes, "allocate Up packed weight");
  ready = ready && fixture.up_scales.allocate(
                       test, kBlockScaleBytes, "allocate Up block scales");
  ready = ready && fixture.scratch0.allocate(
                       test, kBElements, "allocate BF16 scratch0");
  ready = ready && fixture.scratch1.allocate(
                       test, kBElements, "allocate BF16 scratch1");
  ready = ready && fixture.reference_gate_store.allocate(
                       test, kGuardedOutputElements,
                       "allocate guarded reference Gate output");
  ready = ready && fixture.reference_up_store.allocate(
                       test, kGuardedOutputElements,
                       "allocate guarded reference Up output");
  ready = ready && fixture.candidate_gate_store.allocate(
                       test, kGuardedOutputElements,
                       "allocate guarded candidate Gate output");
  ready = ready && fixture.candidate_up_store.allocate(
                       test, kGuardedOutputElements,
                       "allocate guarded candidate Up output");
  ready = ready &&
          fixture.validation.allocate(test, 6U, "allocate validation counters");
  return ready;
}

[[nodiscard]] bool initialize_fixture(TestContext& test, Fixture& fixture,
                                      const Execution& execution,
                                      const CheckpointPayload* const
                                          checkpoint) {
  constexpr unsigned int kThreads = 256U;
  const auto blocks = [](const std::size_t count) {
    return static_cast<unsigned int>((count + kThreads - 1U) / kThreads);
  };
  fill_deterministic_bf16_kernel<<<blocks(kAElements), kThreads, 0U,
                                   execution.main()>>>(
      fixture.activation.get(), kAElements, 0x1234'5678U, 1.0F / 64.0F);
  bool ready = test.cuda_ok(cudaGetLastError(), "fill shared activation");
  fixture.checkpoint_payload = checkpoint != nullptr;
  if (checkpoint != nullptr) {
    const bool payload_exact =
        checkpoint->gate_weight.values.size() == kPackedWeightBytes &&
        checkpoint->gate_block_scale.values.size() == kBlockScaleBytes &&
        checkpoint->up_weight.values.size() == kPackedWeightBytes &&
        checkpoint->up_block_scale.values.size() == kBlockScaleBytes &&
        std::isfinite(checkpoint->gate_weight_scale_2.value) &&
        checkpoint->gate_weight_scale_2.value > 0.0F &&
        std::isfinite(checkpoint->up_weight_scale_2.value) &&
        checkpoint->up_weight_scale_2.value > 0.0F;
    test.expect(payload_exact,
                "prepared checkpoint Gate/Up payload remains exact");
    if (!payload_exact) {
      return false;
    }
    fixture.gate_weight_scale_2 = checkpoint->gate_weight_scale_2.value;
    fixture.up_weight_scale_2 = checkpoint->up_weight_scale_2.value;
    ready = test.cuda_ok(
                cudaMemcpyAsync(fixture.gate_packed.get(),
                                checkpoint->gate_weight.values.data(),
                                kPackedWeightBytes, cudaMemcpyHostToDevice,
                                execution.main()),
                "copy checkpoint Gate packed weight") &&
            ready;
    ready = test.cuda_ok(
                cudaMemcpyAsync(fixture.gate_scales.get(),
                                checkpoint->gate_block_scale.values.data(),
                                kBlockScaleBytes, cudaMemcpyHostToDevice,
                                execution.main()),
                "copy checkpoint Gate block scales") &&
            ready;
    ready = test.cuda_ok(
                cudaMemcpyAsync(fixture.up_packed.get(),
                                checkpoint->up_weight.values.data(),
                                kPackedWeightBytes, cudaMemcpyHostToDevice,
                                execution.main()),
                "copy checkpoint Up packed weight") &&
            ready;
    ready = test.cuda_ok(
                cudaMemcpyAsync(fixture.up_scales.get(),
                                checkpoint->up_block_scale.values.data(),
                                kBlockScaleBytes, cudaMemcpyHostToDevice,
                                execution.main()),
                "copy checkpoint Up block scales") &&
            ready;
  } else {
    fixture.gate_weight_scale_2 = kSyntheticGateWeightScale2;
    fixture.up_weight_scale_2 = kSyntheticUpWeightScale2;
    fill_canonical_nvfp4_kernel<<<blocks(kPackedWeightBytes), kThreads, 0U,
                                  execution.main()>>>(
        fixture.gate_packed.get(), kPackedWeightBytes, 0x6a09'e667U, false);
    ready = test.cuda_ok(cudaGetLastError(), "fill Gate packed weight") && ready;
    fill_canonical_nvfp4_kernel<<<blocks(kBlockScaleBytes), kThreads, 0U,
                                  execution.main()>>>(
        fixture.gate_scales.get(), kBlockScaleBytes, 0xbb67'ae85U, true);
    ready =
        test.cuda_ok(cudaGetLastError(), "fill Gate block scales") && ready;
    fill_canonical_nvfp4_kernel<<<blocks(kPackedWeightBytes), kThreads, 0U,
                                  execution.main()>>>(
        fixture.up_packed.get(), kPackedWeightBytes, 0x3c6e'f372U, false);
    ready =
        test.cuda_ok(cudaGetLastError(), "fill distinct Up packed weight") &&
        ready;
    fill_canonical_nvfp4_kernel<<<blocks(kBlockScaleBytes), kThreads, 0U,
                                  execution.main()>>>(
        fixture.up_scales.get(), kBlockScaleBytes, 0xa54f'f53aU, true);
    ready =
        test.cuda_ok(cudaGetLastError(), "fill distinct Up block scales") &&
        ready;
  }
  ready = test.cuda_ok(cudaStreamSynchronize(execution.main()),
                       "initialize fixture synchronize") &&
          ready;
  return ready;
}

[[nodiscard]] bool run_native_resource_and_status_gate(
    TestContext& test, Fixture& fixture, const Execution& execution) {
  int registers = -1;
  std::size_t static_shared = std::numeric_limits<std::size_t>::max();
  std::size_t dynamic_shared = std::numeric_limits<std::size_t>::max();
  std::size_t local = std::numeric_limits<std::size_t>::max();
  int threads = -1;
  int active = -1;
  int resource_status = static_cast<int>(cudaErrorInvalidValue);
  if (fixture.native_kernel == NativeKernel::kM64N256PairLookup) {
    resource_status = q3x::kernels::
        query_sm87_nvfp4_w4a16_gate_c512_m64_n256_k64_cp_async_capairlookup_resources_test_cuda(
            kM, kN, kK, &registers, &static_shared, &dynamic_shared, &local,
            &threads, &active);
  } else if (fixture.native_kernel ==
             NativeKernel::kM128N256HorizontalP0) {
    resource_status = q3x::kernels::
        query_sm87_nvfp4_w4a16_gate_c512_m128_n256_horizontal_p0_resources_test_cuda(
            kM, kN, kK, &registers, &static_shared, &dynamic_shared, &local,
            &threads, &active);
  } else {
    resource_status = q3x::kernels::
        query_sm87_nvfp4_w4a16_gate_c512_m128_n256_horizontal_named_p1_resources_test_cuda(
            kM, kN, kK, &registers, &static_shared, &dynamic_shared, &local,
            &threads, &active);
  }
  const bool pair_lookup =
      fixture.native_kernel == NativeKernel::kM64N256PairLookup;
  const std::size_t expected_static_shared = pair_lookup ? 1'536U : 512U;
  const std::size_t expected_dynamic_shared =
      fixture.native_kernel == NativeKernel::kM128N256HorizontalP0
          ? 61'440U
          : (fixture.native_kernel ==
                     NativeKernel::kM128N256HorizontalNamedP1
                 ? 61'536U
                 : 43'008U);
  const int expected_threads = pair_lookup ? 256 : 512;
  const int minimum_active_blocks = pair_lookup ? 2 : 1;
  const bool resource_gate =
      resource_status == static_cast<int>(cudaSuccess) && registers <= 128 &&
      static_shared == expected_static_shared &&
      dynamic_shared == expected_dynamic_shared && local == 0U &&
      threads == expected_threads && active >= minimum_active_blocks;
  test.expect(resource_gate,
              "selected native candidate clears frozen resources");
  std::cout << "NVFP4_PAIR_NATIVE_RESOURCES: native_kernel="
            << native_kernel_name(fixture.native_kernel)
            << " status=" << resource_status
            << " registers=" << registers
            << " static_shared_bytes=" << static_shared
            << " dynamic_shared_bytes=" << dynamic_shared
            << " local_bytes=" << local << " threads=" << threads
            << " active_blocks_per_sm=" << active
            << " gate=" << (resource_gate ? "PASS" : "FAIL") << '\n';

  bool ready = test.cuda_ok(cudaStreamSynchronize(execution.main()),
                            "native invalid pre-capture synchronize");
  cudaGraph_t graph = nullptr;
  ready = test.cuda_ok(
              cudaStreamBeginCapture(execution.main(),
                                     cudaStreamCaptureModeThreadLocal),
              "native invalid begin capture") &&
          ready;
  std::array<int, 3U> invalid_statuses{};
  if (ready) {
    const auto launch_invalid = [&](const std::size_t tokens,
                                    const std::size_t rows,
                                    const std::size_t columns) {
      return launch_native_branch_status(
          fixture.native_kernel, fixture.gate_packed.get(),
          fixture.gate_scales.get(), fixture.gate_weight_scale_2,
          fixture.activation.get(), tokens, rows, columns,
          fixture.candidate_gate(), execution.main());
    };
    invalid_statuses[0] = launch_invalid(kM - 1U, kN, kK);
    invalid_statuses[1] = launch_invalid(kM, kN - 1U, kK);
    invalid_statuses[2] = launch_invalid(kM, kN, kK - 1U);
    ready = test.cuda_ok(cudaStreamEndCapture(execution.main(), &graph),
                         "native invalid end capture") &&
            ready;
  }
  std::size_t graph_nodes = 0U;
  if (ready && graph != nullptr) {
    ready = test.cuda_ok(cudaGraphGetNodes(graph, nullptr, &graph_nodes),
                         "native invalid query graph nodes") &&
            ready;
  }
  if (graph != nullptr) {
    ready = test.cuda_ok(cudaGraphDestroy(graph),
                         "native invalid destroy graph") &&
            ready;
  }
  const std::size_t invalid_count = static_cast<std::size_t>(std::count(
      invalid_statuses.begin(), invalid_statuses.end(),
      static_cast<int>(cudaErrorInvalidValue)));
  const bool status_gate =
      ready && invalid_count == invalid_statuses.size() && graph_nodes == 0U;
  test.expect(status_gate,
              "selected native near misses enqueue zero graph nodes");
  std::cout << "NVFP4_PAIR_NATIVE_STATUS: invalid_statuses="
            << invalid_count << '/' << invalid_statuses.size()
            << " graph_nodes=" << graph_nodes
            << " gate=" << (status_gate ? "PASS" : "FAIL") << '\n';
  return resource_gate && status_gate;
}

enum class Mode {
  kSmoke,
  kPerformanceCheckpoint,
};

[[nodiscard]] constexpr const char* mode_name(const Mode mode) noexcept {
  return mode == Mode::kSmoke ? "smoke" : "performance-checkpoint";
}

struct Options {
  std::string checkpoint_directory;
  NativeKernel native_kernel = NativeKernel::kM64N256PairLookup;
  Mode mode = Mode::kSmoke;
};

[[nodiscard]] bool parse_native_kernel(const std::string& value,
                                       NativeKernel& kernel) {
  if (value == "m64n256pairlookup" || value == "m64n256_pairlookup") {
    kernel = NativeKernel::kM64N256PairLookup;
    return true;
  }
  if (value == "m128n256horizontalp0" ||
      value == "m128n256_horizontal_p0") {
    kernel = NativeKernel::kM128N256HorizontalP0;
    return true;
  }
  if (value == "m128n256horizontalnamedp1" ||
      value == "m128n256_horizontal_named_p1") {
    kernel = NativeKernel::kM128N256HorizontalNamedP1;
    return true;
  }
  return false;
}

[[nodiscard]] bool parse_options(const int argc, char** const argv,
                                 Options& options) {
  bool checkpoint_seen = false;
  bool native_seen = false;
  bool mode_seen = false;
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    if (argument == "--mode" || argument.rfind("--mode=", 0U) == 0U) {
      if (mode_seen) {
        std::cerr << "duplicate --mode argument\n";
        return false;
      }
      mode_seen = true;
      if (argument == "--mode" && index + 1 >= argc) {
        std::cerr << "--mode requires smoke or performance-checkpoint\n";
        return false;
      }
      const std::string value =
          argument == "--mode"
              ? std::string(argv[++index])
              : argument.substr(std::string("--mode=").size());
      if (value == "smoke") {
        options.mode = Mode::kSmoke;
      } else if (value == "performance-checkpoint") {
        options.mode = Mode::kPerformanceCheckpoint;
      } else {
        std::cerr << "unknown --mode value: " << value << '\n';
        return false;
      }
    } else if (argument == "--checkpoint") {
      if (checkpoint_seen) {
        std::cerr << "duplicate --checkpoint argument\n";
        return false;
      }
      if (index + 1 >= argc) {
        std::cerr << "--checkpoint requires a non-empty directory\n";
        return false;
      }
      checkpoint_seen = true;
      options.checkpoint_directory = argv[++index];
      if (options.checkpoint_directory.empty()) {
        std::cerr << "--checkpoint requires a non-empty directory\n";
        return false;
      }
    } else if (argument.rfind("--checkpoint=", 0U) == 0U) {
      if (checkpoint_seen) {
        std::cerr << "duplicate --checkpoint argument\n";
        return false;
      }
      checkpoint_seen = true;
      options.checkpoint_directory =
          argument.substr(std::string("--checkpoint=").size());
      if (options.checkpoint_directory.empty()) {
        std::cerr << "--checkpoint requires a non-empty directory\n";
        return false;
      }
    } else if (argument == "--native" ||
               argument.rfind("--native=", 0U) == 0U) {
      if (native_seen) {
        std::cerr << "duplicate --native argument\n";
        return false;
      }
      native_seen = true;
      if (argument == "--native" && index + 1 >= argc) {
        std::cerr << "--native requires a value\n";
        return false;
      }
      const std::string value = argument == "--native"
                                    ? std::string(argv[++index])
                                    : argument.substr(
                                          std::string("--native=").size());
      if (!parse_native_kernel(value, options.native_kernel)) {
        std::cerr << "unknown --native value: " << value << '\n';
        return false;
      }
    } else {
      std::cerr << "unknown argument: " << argument << '\n';
      return false;
    }
  }
  return true;
}

void print_payload_provenance(
    const CheckpointPayload* const checkpoint) {
  if (checkpoint == nullptr) {
    std::cout << "NVFP4_PAIR_PAYLOAD: payload=synthetic"
              << " activation=deterministic_bf16"
              << " activation_seed=0x12345678"
              << " activation_scale=0.015625"
              << " gate_weight_seed=0x6a09e667"
              << " gate_block_scale_seed=0xbb67ae85"
              << " up_weight_seed=0x3c6ef372"
              << " up_block_scale_seed=0xa54ff53a"
              << " gate_weight_scale_2=" << kSyntheticGateWeightScale2
              << " up_weight_scale_2=" << kSyntheticUpWeightScale2 << '\n';
    return;
  }

  std::cout << std::setprecision(std::numeric_limits<float>::max_digits10)
            << "NVFP4_PAIR_PAYLOAD: payload=checkpoint"
            << " canonical_directory=" << checkpoint->canonical_directory
            << " index=" << kCheckpointIndex
            << " shard=" << checkpoint->shard
            << " shard_file_bytes=" << checkpoint->shard_file_bytes
            << " header_bytes=" << checkpoint->header_bytes
            << " data_offset=" << checkpoint->data_offset
            << " gate_weight_tensor=" << kGateWeightTensor
            << " gate_weight_dtype=U8"
            << " gate_weight_shape=17408x2560"
            << " gate_weight_begin=" << checkpoint->gate_weight.file_begin
            << " gate_weight_end=" << checkpoint->gate_weight.file_end
            << " gate_weight_bytes=" << checkpoint->gate_weight.values.size()
            << " gate_weight_sha256=" << checkpoint->gate_weight.sha256
            << " gate_block_scale_tensor=" << kGateBlockScaleTensor
            << " gate_block_scale_dtype=F8_E4M3"
            << " gate_block_scale_shape=17408x320"
            << " gate_block_scale_begin="
            << checkpoint->gate_block_scale.file_begin
            << " gate_block_scale_end="
            << checkpoint->gate_block_scale.file_end
            << " gate_block_scale_bytes="
            << checkpoint->gate_block_scale.values.size()
            << " gate_block_scale_sha256="
            << checkpoint->gate_block_scale.sha256
            << " gate_weight_scale_2_tensor=" << kGateWeightScale2Tensor
            << " gate_weight_scale_2_dtype=F32"
            << " gate_weight_scale_2_shape=scalar"
            << " gate_weight_scale_2_begin="
            << checkpoint->gate_weight_scale_2.file_begin
            << " gate_weight_scale_2_end="
            << checkpoint->gate_weight_scale_2.file_end
            << " gate_weight_scale_2="
            << checkpoint->gate_weight_scale_2.value
            << " gate_weight_scale_2_bits=0x"
            << hex_u32(checkpoint->gate_weight_scale_2.little_endian_bits)
            << " gate_weight_scale_2_sha256="
            << checkpoint->gate_weight_scale_2.sha256
            << " up_weight_tensor=" << kUpWeightTensor
            << " up_weight_dtype=U8"
            << " up_weight_shape=17408x2560"
            << " up_weight_begin=" << checkpoint->up_weight.file_begin
            << " up_weight_end=" << checkpoint->up_weight.file_end
            << " up_weight_bytes=" << checkpoint->up_weight.values.size()
            << " up_weight_sha256=" << checkpoint->up_weight.sha256
            << " up_block_scale_tensor=" << kUpBlockScaleTensor
            << " up_block_scale_dtype=F8_E4M3"
            << " up_block_scale_shape=17408x320"
            << " up_block_scale_begin="
            << checkpoint->up_block_scale.file_begin
            << " up_block_scale_end=" << checkpoint->up_block_scale.file_end
            << " up_block_scale_bytes="
            << checkpoint->up_block_scale.values.size()
            << " up_block_scale_sha256="
            << checkpoint->up_block_scale.sha256
            << " up_weight_scale_2_tensor=" << kUpWeightScale2Tensor
            << " up_weight_scale_2_dtype=F32"
            << " up_weight_scale_2_shape=scalar"
            << " up_weight_scale_2_begin="
            << checkpoint->up_weight_scale_2.file_begin
            << " up_weight_scale_2_end="
            << checkpoint->up_weight_scale_2.file_end
            << " up_weight_scale_2=" << checkpoint->up_weight_scale_2.value
            << " up_weight_scale_2_bits=0x"
            << hex_u32(checkpoint->up_weight_scale_2.little_endian_bits)
            << " up_weight_scale_2_sha256="
            << checkpoint->up_weight_scale_2.sha256
            << " activation=deterministic_bf16"
            << " activation_seed=0x12345678"
            << " activation_scale=0.015625"
            << " payload_sha256_pins=true"
            << " checkpoint_read_only=true\n";
}

}  // namespace

int main(const int argc, char** const argv) {
  Options options;
  if (!parse_options(argc, argv, options)) {
    return 2;
  }
  const bool performance_checkpoint =
      options.mode == Mode::kPerformanceCheckpoint;
  if (performance_checkpoint && options.checkpoint_directory.empty()) {
    std::cerr << "--mode=performance-checkpoint requires --checkpoint DIR\n";
    return 2;
  }
  if (!performance_checkpoint && !options.checkpoint_directory.empty()) {
    std::cerr << "--checkpoint is only valid with "
                 "--mode=performance-checkpoint; no implicit performance "
                 "mode is selected\n";
    return 2;
  }
  TestContext test;
  CheckpointPayload checkpoint_payload;
  const CheckpointPayload* selected_checkpoint = nullptr;
  if (performance_checkpoint) {
    q3x::test::support::PinnedBundleLoadOptions pinned_options;
    pinned_options.tensor_names = {
        std::string(q3x::test::support::kQwen36Layer0GateWeight),
        std::string(q3x::test::support::kQwen36Layer0GateBlockScale),
        std::string(q3x::test::support::kQwen36Layer0GateWeightScale2),
        std::string(q3x::test::support::kQwen36Layer0UpWeight),
        std::string(q3x::test::support::kQwen36Layer0UpBlockScale),
        std::string(q3x::test::support::kQwen36Layer0UpWeightScale2),
    };
    constexpr std::uint64_t kPinnedPayloadBytes =
        2U * (kPackedWeightBytes + kBlockScaleBytes + sizeof(float));
    pinned_options.maximum_total_payload_bytes = kPinnedPayloadBytes;
    const q3x::test::support::PinnedBundleLoadResult pinned_bundle =
        q3x::test::support::load_pinned_checkpoint_bundle(
            options.checkpoint_directory,
            q3x::test::support::qwen36_27b_nvfp4_layer0_mlp_bundle(),
            pinned_options);
    if (!pinned_bundle) {
      std::cerr << "NVFP4_PAIR_PINNED_BUNDLE_ERROR: ";
      if (pinned_bundle.error.has_value()) {
        std::cerr << q3x::test::support::describe_pinned_checkpoint_error(
            *pinned_bundle.error);
      } else {
        std::cerr << "loader failed without a structured error";
      }
      std::cerr << '\n';
      return 1;
    }
    const q3x::test::support::LoadedPinnedBundle& loaded =
        *pinned_bundle.value;
    if (loaded.tensors.size() != pinned_options.tensor_names.size()) {
      std::cerr << "NVFP4_PAIR_PINNED_BUNDLE_ERROR: expected "
                << pinned_options.tensor_names.size() << " tensors, loaded "
                << loaded.tensors.size() << '\n';
      return 1;
    }
    std::cout << "NVFP4_PAIR_PINNED_BUNDLE: descriptor="
              << loaded.descriptor_id << " model=" << loaded.model_id
              << " repository=" << loaded.repository
              << " revision=" << loaded.revision
              << " tensor_count=" << loaded.tensors.size()
              << " maximum_total_payload_bytes=" << kPinnedPayloadBytes
              << " checkpoint_read_only=true\n";
  }
  if (!options.checkpoint_directory.empty()) {
    if (!load_checkpoint_payload(test, options.checkpoint_directory,
                                 checkpoint_payload)) {
      return 1;
    }
    selected_checkpoint = &checkpoint_payload;
  }
  print_payload_provenance(selected_checkpoint);
  std::cout << "NVFP4_PAIR_MODE: mode=" << mode_name(options.mode)
            << " admission="
            << (performance_checkpoint ? "PENDING" : "NOT_RUN")
            << " evidence="
            << (performance_checkpoint ? "checkpoint_weight_only"
                                       : "synthetic_smoke")
            << '\n';
  int device = 0;
  if (!test.cuda_ok(cudaGetDevice(&device), "get active CUDA device")) {
    return 1;
  }
  cudaDeviceProp properties{};
  if (!test.cuda_ok(cudaGetDeviceProperties(&properties, device),
                    "query active CUDA device")) {
    return 1;
  }
  if (properties.major != 8 || properties.minor != 7) {
    std::cout << "SKIP: Gate/Up cuBLASLt pair screen requires SM87; found "
              << properties.major << '.' << properties.minor << '\n';
    return 77;
  }

  const std::optional<ClockState> clocks = read_clock_state();
  if (performance_checkpoint &&
      (!clocks.has_value() || !clocks_are_fixed(*clocks))) {
    std::cout << "SKIP: Gate/Up pair performance screen requires fixed GPU and "
                 "EMC clocks"
              << " clock_state_available="
              << (clocks.has_value() ? "true" : "false") << '\n';
    return 77;
  }
  const ClockState observed_clocks = clocks.value_or(ClockState{});
  const bool fixed_clocks =
      clocks.has_value() && clocks_are_fixed(observed_clocks);

  std::size_t free_before = 0U;
  std::size_t total_before = 0U;
  if (!test.cuda_ok(cudaMemGetInfo(&free_before, &total_before),
                    "query memory before fixture")) {
    return 1;
  }

  std::cout << std::fixed << std::setprecision(6)
            << "NVFP4_PAIR_PROTOCOL: device=" << properties.name
            << " cc=" << properties.major << '.' << properties.minor
            << " M=" << kM << " N=" << kN << " K=" << kK
            << " payload="
            << (selected_checkpoint == nullptr ? "synthetic" : "checkpoint")
            << " gate_weight_scale_2="
            << (selected_checkpoint == nullptr
                    ? kSyntheticGateWeightScale2
                    : selected_checkpoint->gate_weight_scale_2.value)
            << " up_weight_scale_2="
            << (selected_checkpoint == nullptr
                    ? kSyntheticUpWeightScale2
                    : selected_checkpoint->up_weight_scale_2.value)
            << " native_kernel=" << native_kernel_name(options.native_kernel)
            << " workspace_bytes=0"
            << " clock_state_available="
            << (clocks.has_value() ? "true" : "false")
            << " gpu_min_hz=" << observed_clocks.gpu_min
            << " gpu_current_hz=" << observed_clocks.gpu_current
            << " gpu_max_hz=" << observed_clocks.gpu_max
            << " emc_min_hz=" << observed_clocks.emc_min
            << " emc_current_hz=" << observed_clocks.emc_current
            << " emc_max_hz=" << observed_clocks.emc_max
            << " fixed_clocks=" << (fixed_clocks ? "true" : "false")
            << '\n';

  Fixture fixture;
  fixture.native_kernel = options.native_kernel;
  Execution execution;
  LtObjects main_lt;
  LtObjects auxiliary_lt;
  bool ready = allocate_fixture(test, fixture);
  ready = ready && execution.create(test);
  ready = ready && main_lt.create(test, "main Lt");
  ready = ready && auxiliary_lt.create(test, "auxiliary Lt");
  if (!ready) {
    return 1;
  }

  std::size_t free_after = 0U;
  std::size_t total_after = 0U;
  ready = test.cuda_ok(cudaMemGetInfo(&free_after, &total_after),
                       "query memory after fixture");
  const std::uint64_t observed_drop =
      free_before >= free_after ? free_before - free_after : 0U;
  constexpr std::uint64_t kPerScratchBytes =
      kBElements * sizeof(__nv_bfloat16);
  constexpr std::uint64_t kTwoScratchBytes = 2U * kPerScratchBytes;
  const std::uint64_t two_scratch_fixture_bytes = fixture.planned_bytes();
  const std::uint64_t one_scratch_fixture_bytes =
      two_scratch_fixture_bytes - kPerScratchBytes;
  std::cout << "NVFP4_PAIR_MEMORY: total_device_bytes=" << total_after
            << " free_before_bytes=" << free_before
            << " free_after_bytes=" << free_after
            << " observed_free_drop_bytes=" << observed_drop
            << " two_scratch_test_fixture_bytes="
            << two_scratch_fixture_bytes
            << " one_scratch_test_fixture_bytes="
            << one_scratch_fixture_bytes
            << " one_scratch_incremental_bytes=" << kPerScratchBytes
            << " two_scratch_incremental_bytes=" << kTwoScratchBytes
            << " per_scratch_bytes=" << kPerScratchBytes
            << " lt_workspace_bytes=0" << '\n';

  ready = initialize_fixture(test, fixture, execution, selected_checkpoint) &&
          ready;
  ready = launch_dequantize(fixture.gate_packed.get(),
                            fixture.gate_scales.get(), fixture.scratch0.get(),
                            execution.main()) &&
          ready;
  ready = test.cuda_ok(cudaStreamSynchronize(execution.main()),
                       "prepare algorithm-selection weight") &&
          ready;
  const std::optional<SelectedAlgorithm> selected = select_algorithm(
      test, main_lt, fixture.scratch0.get(), fixture.activation.get(),
      fixture.candidate_gate(), execution.main());
  if (!ready || !selected.has_value()) {
    return 1;
  }

  cublasLtMatmulHeuristicResult_t auxiliary_check{};
  ready = test.lt_ok(
              cublasLtMatmulAlgoCheck(
                  auxiliary_lt.handle(), auxiliary_lt.operation(),
                  auxiliary_lt.weight_layout(),
                  auxiliary_lt.activation_layout(),
                  auxiliary_lt.output_layout(), auxiliary_lt.output_layout(),
                  &selected->value, &auxiliary_check),
              "validate selected algorithm on auxiliary handle") &&
          ready;
  test.expect(auxiliary_check.workspaceSize == 0U,
              "auxiliary handle keeps selected algorithm zero-workspace");
  ready = ready && auxiliary_check.workspaceSize == 0U;
  std::cout << "NVFP4_PAIR_LT_SELECTED: heuristic_index="
            << selected->heuristic_index
            << " selection_milliseconds=" << selected->milliseconds
            << " main_workspace_bytes=0"
            << " auxiliary_workspace_bytes=" << auxiliary_check.workspaceSize
            << " gate=PASS\n";

  ready = run_native_resource_and_status_gate(test, fixture, execution) &&
          ready;
  ready = generate_production_reference(test, fixture, execution) && ready;
  for (const Variant variant :
       {Variant::kProduction, Variant::kSerialOneScratch,
        Variant::kNaiveDual, Variant::kStaggeredDual,
        Variant::kNativeSerial, Variant::kNativeDual}) {
    ready = run_eager_correctness(test, fixture, execution, main_lt,
                                  auxiliary_lt, *selected, variant) &&
            ready;
    ready = run_graph_correctness(test, fixture, execution, main_lt,
                                  auxiliary_lt, *selected, variant) &&
            ready;
  }

  const ComparisonResult production_vs_a = run_bccb(
      test, fixture, execution, main_lt, auxiliary_lt, *selected,
      Variant::kProduction, Variant::kSerialOneScratch, "production_vs_A",
      performance_checkpoint);
  const ComparisonResult a_vs_b = run_bccb(
      test, fixture, execution, main_lt, auxiliary_lt, *selected,
      Variant::kSerialOneScratch, Variant::kNaiveDual, "A_vs_B",
      performance_checkpoint);
  const ComparisonResult a_vs_c = run_bccb(
      test, fixture, execution, main_lt, auxiliary_lt, *selected,
      Variant::kSerialOneScratch, Variant::kStaggeredDual, "A_vs_C",
      performance_checkpoint);
  const ComparisonResult bridge_vs_native_serial = run_bccb(
      test, fixture, execution, main_lt, auxiliary_lt, *selected,
      Variant::kSerialOneScratch, Variant::kNativeSerial,
      "cublaslt_bridge_vs_native_serial", performance_checkpoint);
  const ComparisonResult bridge_vs_native_dual = run_bccb(
      test, fixture, execution, main_lt, auxiliary_lt, *selected,
      Variant::kSerialOneScratch, Variant::kNativeDual,
      "cublaslt_bridge_vs_native_dual", performance_checkpoint);
  const ComparisonResult native_serial_vs_dual = run_bccb(
      test, fixture, execution, main_lt, auxiliary_lt, *selected,
      Variant::kNativeSerial, Variant::kNativeDual,
      "native_serial_vs_native_dual", performance_checkpoint);

  const bool serial_gate = production_vs_a.every_round_positive &&
                           std::isfinite(production_vs_a.speedup) &&
                           production_vs_a.speedup >=
                               kRequiredSerialVsProduction;
  const bool naive_recommendation =
      a_vs_b.every_round_positive && std::isfinite(a_vs_b.speedup) &&
      a_vs_b.speedup >= kRequiredTwoScratchVsSerial;
  const bool staggered_recommendation =
      a_vs_c.every_round_positive && std::isfinite(a_vs_c.speedup) &&
      a_vs_c.speedup >= kRequiredTwoScratchVsSerial;
  const Variant recommendation =
      staggered_recommendation &&
              (!naive_recommendation || a_vs_c.speedup >= a_vs_b.speedup)
          ? Variant::kStaggeredDual
          : (naive_recommendation ? Variant::kNaiveDual
                                  : Variant::kSerialOneScratch);
  const bool native_timing_finite =
      std::isfinite(bridge_vs_native_serial.baseline_milliseconds) &&
      std::isfinite(bridge_vs_native_serial.candidate_milliseconds) &&
      std::isfinite(bridge_vs_native_serial.speedup) &&
      std::isfinite(bridge_vs_native_dual.baseline_milliseconds) &&
      std::isfinite(bridge_vs_native_dual.candidate_milliseconds) &&
      std::isfinite(bridge_vs_native_dual.speedup) &&
      std::isfinite(native_serial_vs_dual.baseline_milliseconds) &&
      std::isfinite(native_serial_vs_dual.candidate_milliseconds) &&
      std::isfinite(native_serial_vs_dual.speedup);
  test.expect(native_timing_finite,
              "native and cuBLASLt head-to-head timings are finite");
  // Select the native schedule only from its direct paired comparison.  The
  // two bridge-vs-native blocks execute at different points in the process,
  // so comparing their absolute candidate times would admit clock/thermal
  // drift into what is already measured by native_serial_vs_dual.
  const bool native_dual_wins_direct_pair =
      native_serial_vs_dual.every_round_positive &&
      native_serial_vs_dual.speedup > 1.0;
  const Variant native_recommendation =
      native_dual_wins_direct_pair ? Variant::kNativeDual
                                   : Variant::kNativeSerial;
  const ComparisonResult& bridge_vs_best_native =
      native_recommendation == Variant::kNativeDual
          ? bridge_vs_native_dual
          : bridge_vs_native_serial;
  const bool native_clears_bridge =
      bridge_vs_best_native.every_round_positive &&
      std::isfinite(bridge_vs_best_native.speedup) &&
      bridge_vs_best_native.speedup > 1.0;
  const bool performance_admission = serial_gate && native_clears_bridge;
  if (performance_checkpoint) {
    std::cout << "NVFP4_PAIR_FINAL: production_vs_A_speedup="
              << production_vs_a.speedup
              << " production_vs_A_required=" << kRequiredSerialVsProduction
              << " production_vs_A_every_round_positive="
              << (production_vs_a.every_round_positive ? "true" : "false")
              << " A_vs_B_speedup=" << a_vs_b.speedup
              << " A_vs_B_required=" << kRequiredTwoScratchVsSerial
              << " A_vs_B_every_round_positive="
              << (a_vs_b.every_round_positive ? "true" : "false")
              << " A_vs_B_recommend="
              << (naive_recommendation ? "true" : "false")
              << " A_vs_C_speedup=" << a_vs_c.speedup
              << " A_vs_C_required=" << kRequiredTwoScratchVsSerial
              << " A_vs_C_every_round_positive="
              << (a_vs_c.every_round_positive ? "true" : "false")
              << " A_vs_C_recommend="
              << (staggered_recommendation ? "true" : "false")
              << " recommended_variant=" << variant_name(recommendation)
              << " recommended_scratch_count="
              << (recommendation == Variant::kSerialOneScratch ? 1 : 2)
              << " hard_gate=" << (serial_gate ? "PASS" : "FAIL") << '\n';
    std::cout << "NVFP4_PAIR_NATIVE_FINAL: native_kernel="
              << native_kernel_name(fixture.native_kernel)
              << " bridge_serial_ms="
              << bridge_vs_native_serial.baseline_milliseconds
              << " native_serial_ms="
              << bridge_vs_native_serial.candidate_milliseconds
              << " bridge_vs_native_serial_speedup="
              << bridge_vs_native_serial.speedup
              << " bridge_vs_native_serial_every_round_positive="
              << (bridge_vs_native_serial.every_round_positive ? "true"
                                                                : "false")
              << " native_dual_ms="
              << bridge_vs_native_dual.candidate_milliseconds
              << " bridge_vs_native_dual_speedup="
              << bridge_vs_native_dual.speedup
              << " bridge_vs_native_dual_every_round_positive="
              << (bridge_vs_native_dual.every_round_positive ? "true"
                                                              : "false")
              << " native_serial_vs_dual_speedup="
              << native_serial_vs_dual.speedup
              << " native_serial_vs_dual_every_round_positive="
              << (native_serial_vs_dual.every_round_positive ? "true"
                                                              : "false")
              << " recommended_native_schedule="
              << variant_name(native_recommendation)
              << " native_clears_bridge="
              << (native_clears_bridge ? "true" : "false")
              << " correctness_graph_resource_status_gate="
              << (ready && native_timing_finite ? "PASS" : "FAIL") << '\n';
  } else {
    std::cout << "NVFP4_PAIR_SMOKE_TIMING: production_vs_A_speedup="
              << production_vs_a.speedup
              << " A_vs_B_speedup=" << a_vs_b.speedup
              << " A_vs_C_speedup=" << a_vs_c.speedup
              << " bridge_vs_native_serial_speedup="
              << bridge_vs_native_serial.speedup
              << " bridge_vs_native_dual_speedup="
              << bridge_vs_native_dual.speedup
              << " native_serial_vs_dual_speedup="
              << native_serial_vs_dual.speedup
              << " admission=NOT_RUN evidence=synthetic_smoke\n";
  }

  ready = test.cuda_ok(cudaStreamSynchronize(execution.main()),
                       "final main synchronize") &&
          ready;
  ready = test.cuda_ok(cudaStreamSynchronize(execution.auxiliary()),
                       "final auxiliary synchronize") &&
          ready;
  if (!ready) {
    test.expect(false, "Gate/Up pair screen completed all CUDA work");
  }
  if (test.failures() != 0) {
    std::cerr << test.failures()
              << " Gate/Up C512 cuBLASLt pair assertion(s) failed\n";
    return 1;
  }
  if (!performance_checkpoint) {
    std::cout << "NVFP4_PAIR_ADMISSION: mode=smoke admission=NOT_RUN"
              << " evidence=synthetic_smoke status=PASS\n";
    std::cout << "Gate/Up C512 cuBLASLt pair smoke passed\n";
    return 0;
  }
  std::cout << "NVFP4_PAIR_ADMISSION: mode=performance-checkpoint"
            << " admission=" << (performance_admission ? "PASS" : "REJECT")
            << " evidence=checkpoint_weight_only"
            << " candidate=" << native_kernel_name(fixture.native_kernel)
            << " bridge_gate=" << (serial_gate ? "PASS" : "REJECT")
            << " native_gate="
            << (native_clears_bridge ? "PASS" : "REJECT")
            << " status=" << (performance_admission ? "PASS" : "REJECT")
            << '\n';
  std::cout << "Gate/Up C512 cuBLASLt pair performance checkpoint "
            << (performance_admission ? "passed"
                                      : "did not clear admission")
            << '\n';
  if (!performance_admission) {
    return 3;
  }
  return 0;
}
