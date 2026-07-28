#include "q3x/core/sha256.h"
#include "q3x/io/safetensors.h"
#include "q3x/kernels/sm87_nvfp4_prefill_cublaslt.h"
#include "q3x/kernels/sm87_weight_only_gemv.h"
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

namespace {

constexpr std::size_t kM = 512U;
constexpr std::size_t kK = 17'408U;
constexpr std::size_t kN = 5'120U;
constexpr std::size_t kAElements = kM * kK;
constexpr std::size_t kBElements = kK * kN;
constexpr std::size_t kCElements = kM * kN;
constexpr std::size_t kPackedWeightBytes = kN * (kK / 2U);
constexpr std::size_t kBlockScaleBytes = kN * (kK / 16U);
constexpr std::size_t kWorkspaceBytes = 0U;
constexpr int kMaximumHeuristics = 16;
constexpr int kSelectionWarmups = 2;
constexpr int kSelectionIterations = 4;
constexpr int kFormalWarmups = 10;
constexpr int kFormalIterations = 24;
constexpr int kFormalRounds = 6;
constexpr double kRequiredInclusiveSpeedup = 1.22;
constexpr double kMaximumProductionNrmse = 1.0e-2;
constexpr double kMinimumProductionCosine = 0.9999;
constexpr double kRelativeFloor = 1.0e-2;
constexpr float kWeightScale2 = 1.25F;
constexpr std::string_view kCheckpointWeightTensor =
    "model.language_model.layers.0.mlp.down_proj.weight";
constexpr std::string_view kCheckpointBlockScaleTensor =
    "model.language_model.layers.0.mlp.down_proj.weight_scale";
constexpr std::string_view kCheckpointWeightScale2Tensor =
    "model.language_model.layers.0.mlp.down_proj.weight_scale_2";
constexpr std::string_view kPinnedCheckpointWeightSha256 =
    "bc1b428661d3cf657a4d69ff8d7e482b8125ef0f323e6df29b153a22fa2b6daf";
constexpr std::string_view kPinnedCheckpointBlockScaleSha256 =
    "7943b475b23f75886309e93bf673aacc22c699e19ff400ef85607ab1a4006019";
constexpr std::string_view kPinnedCheckpointWeightScale2Sha256 =
    "face5c84f9ca43eb34d792708f1c0d4bdc532c91d6419e65123a5d560245dd72";
constexpr std::uint64_t kPinnedCheckpointShardBytes = 9'965'652'512ULL;
constexpr std::uint64_t kPinnedCheckpointWeightBegin = 6'712'445'472ULL;
constexpr std::uint64_t kPinnedCheckpointWeightEnd =
    kPinnedCheckpointWeightBegin + kPackedWeightBytes;
constexpr std::uint64_t kPinnedCheckpointBlockScaleBegin = 3'600'468'512ULL;
constexpr std::uint64_t kPinnedCheckpointBlockScaleEnd =
    kPinnedCheckpointBlockScaleBegin + kBlockScaleBytes;
constexpr std::uint64_t kPinnedCheckpointWeightScale2Begin = 126'540ULL;
constexpr std::uint64_t kPinnedCheckpointWeightScale2End =
    kPinnedCheckpointWeightScale2Begin + sizeof(float);
constexpr std::uint32_t kPinnedCheckpointWeightScale2Bits = 0x39b6'1862U;
constexpr std::size_t kGuardElements = 256U;
constexpr std::uint16_t kGuardBits = 0xa5a5U;
constexpr double kUsefulFlops =
    2.0 * static_cast<double>(kM) * static_cast<double>(kN) *
    static_cast<double>(kK);

static_assert(kAElements == 8'912'896U);
static_assert(kBElements == 89'128'960U);
static_assert(kCElements == 2'621'440U);
static_assert(kPackedWeightBytes == 44'564'480U);
static_assert(kBlockScaleBytes == 5'570'560U);
static_assert(kPinnedCheckpointWeightEnd == 6'757'009'952ULL);
static_assert(kPinnedCheckpointBlockScaleEnd == 3'606'039'072ULL);
static_assert(kPinnedCheckpointWeightScale2End == 126'544ULL);

struct NumericalMetrics {
  std::size_t bitwise_mismatches = 0U;
  std::size_t candidate_nonfinite = 0U;
  std::size_t production_nonfinite = 0U;
  double maximum_absolute = 0.0;
  double maximum_relative = 0.0;
  double squared_error = 0.0;
  double squared_candidate = 0.0;
  double squared_production = 0.0;
  double dot = 0.0;
  double nrmse = 0.0;
  double cosine = 0.0;
};

[[nodiscard]] float decode_bf16_host(const std::uint16_t encoded) {
  const std::uint32_t bits = static_cast<std::uint32_t>(encoded) << 16U;
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

[[nodiscard]] NumericalMetrics compare_bf16_outputs(
    const std::vector<std::uint16_t>& candidate,
    const std::vector<std::uint16_t>& production) {
  NumericalMetrics metrics{};
  const std::size_t count = std::min(candidate.size(), production.size());
  for (std::size_t index = 0U; index < count; ++index) {
    metrics.bitwise_mismatches +=
        candidate[index] != production[index] ? 1U : 0U;
    const double candidate_value = decode_bf16_host(candidate[index]);
    const double production_value = decode_bf16_host(production[index]);
    if (!std::isfinite(candidate_value)) {
      ++metrics.candidate_nonfinite;
    }
    if (!std::isfinite(production_value)) {
      ++metrics.production_nonfinite;
    }
    if (!std::isfinite(candidate_value) ||
        !std::isfinite(production_value)) {
      continue;
    }
    const double absolute =
        std::abs(candidate_value - production_value);
    const double relative =
        absolute / std::max(std::abs(production_value), kRelativeFloor);
    metrics.maximum_absolute =
        std::max(metrics.maximum_absolute, absolute);
    metrics.maximum_relative =
        std::max(metrics.maximum_relative, relative);
    metrics.squared_error += absolute * absolute;
    metrics.squared_candidate += candidate_value * candidate_value;
    metrics.squared_production += production_value * production_value;
    metrics.dot += candidate_value * production_value;
  }
  metrics.nrmse = std::sqrt(
      metrics.squared_error /
      std::max(metrics.squared_production,
               std::numeric_limits<double>::min()));
  if (metrics.squared_candidate == 0.0 &&
      metrics.squared_production == 0.0 && metrics.squared_error == 0.0) {
    metrics.cosine = 1.0;
  } else {
    metrics.cosine =
        metrics.dot /
        std::sqrt(std::max(metrics.squared_candidate *
                               metrics.squared_production,
                           std::numeric_limits<double>::min()));
  }
  return metrics;
}

[[nodiscard]] bool select_hybrid_for_scale(const float weight_scale_2) {
  // Positive finite scales are the only region admitted to the new route.
  // Zero keeps the existing production path to preserve signed-zero details;
  // negative/non-finite values retain the public launcher's invalid contract.
  return std::isfinite(weight_scale_2) && weight_scale_2 > 0.0F;
}

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

struct CheckpointPayload {
  std::vector<std::uint8_t> packed_weights;
  std::vector<std::uint8_t> block_scales;
  std::array<std::uint8_t, sizeof(float)> weight_scale_2_bytes{};
  float weight_scale_2 = 0.0F;
  std::uint32_t weight_scale_2_bits = 0U;
  std::string weight_tensor;
  std::string block_scale_tensor;
  std::string weight_scale_2_tensor;
  std::string shard;
  std::string weight_sha256;
  std::string block_scale_sha256;
  std::string weight_scale_2_sha256;
};

[[nodiscard]] std::string describe_safetensors_error(
    const q3x::io::safetensors::Error& error) {
  std::string description =
      std::string(q3x::io::safetensors::to_string(error.code)) + ": " +
      std::string(error.message());
  if (!error.context.empty()) {
    description += " context=" + error.context;
  }
  if (!error.expected.empty()) {
    description += " expected=" + error.expected;
  }
  if (!error.actual.empty()) {
    description += " actual=" + error.actual;
  }
  if (error.offset != q3x::io::safetensors::kUnknownOffset) {
    description += " offset=" + std::to_string(error.offset);
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

[[nodiscard]] bool load_checkpoint_payload(
    TestContext& test, const std::string& checkpoint_directory,
    CheckpointPayload& payload) {
  namespace fs = std::filesystem;
  namespace st = q3x::io::safetensors;

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
      requested_directory / "model.safetensors.index.json";
  const fs::file_status index_link_status =
      fs::symlink_status(requested_index, filesystem_error);
  if (filesystem_error || !fs::is_regular_file(index_link_status) ||
      fs::is_symlink(index_link_status)) {
    test.expect(false, "checkpoint index is a regular non-symlink file");
    return false;
  }
  const fs::path index_path = fs::canonical(requested_index, filesystem_error);
  if (filesystem_error || !path_is_strictly_within(directory, index_path) ||
      index_path != directory / "model.safetensors.index.json") {
    test.expect(false, "checkpoint index resolves inside checkpoint directory");
    return false;
  }
  const st::Result<st::Index> index = st::read_index(index_path.string());
  if (!index) {
    test.expect(false, "read checkpoint safetensors index: " +
                           describe_safetensors_error(index.error));
    return false;
  }

  const std::string* const weight_shard =
      index.value->shard_for(kCheckpointWeightTensor);
  const std::string* const block_scale_shard =
      index.value->shard_for(kCheckpointBlockScaleTensor);
  const std::string* const weight_scale_2_shard =
      index.value->shard_for(kCheckpointWeightScale2Tensor);
  const bool shard_mapping_exact =
      weight_shard != nullptr && block_scale_shard != nullptr &&
      weight_scale_2_shard != nullptr && *weight_shard == *block_scale_shard &&
      *weight_shard == *weight_scale_2_shard &&
      st::is_safe_relative_shard_path(*weight_shard);
  test.expect(shard_mapping_exact,
              "checkpoint index pins layer-0 Down payloads to one safe shard");
  if (!shard_mapping_exact) {
    return false;
  }

  const fs::path requested_shard =
      requested_directory / fs::path(*weight_shard);
  const fs::file_status shard_link_status =
      fs::symlink_status(requested_shard, filesystem_error);
  if (filesystem_error || !fs::is_regular_file(shard_link_status) ||
      fs::is_symlink(shard_link_status)) {
    test.expect(false, "checkpoint shard is a regular non-symlink file");
    return false;
  }
  const fs::path shard_path = fs::canonical(requested_shard, filesystem_error);
  if (filesystem_error || !path_is_strictly_within(directory, shard_path) ||
      shard_path != directory / fs::path(*weight_shard)) {
    test.expect(false, "checkpoint shard resolves inside checkpoint directory");
    return false;
  }
  const st::Result<st::Header> header = st::read_header(shard_path.string());
  if (!header) {
    test.expect(false, "read checkpoint safetensors shard header: " +
                           describe_safetensors_error(header.error));
    return false;
  }
  const bool shard_size_pinned =
      header.value->file_size == kPinnedCheckpointShardBytes;
  test.expect(shard_size_pinned,
              "checkpoint layer-0 Down shard has pinned byte size");

  const st::TensorInfo* const weight =
      header.value->find_tensor(kCheckpointWeightTensor);
  const st::TensorInfo* const block_scale =
      header.value->find_tensor(kCheckpointBlockScaleTensor);
  const st::TensorInfo* const weight_scale_2 =
      header.value->find_tensor(kCheckpointWeightScale2Tensor);
  const bool weight_exact =
      weight != nullptr && weight->dtype == st::DType::kU8 &&
      weight->shape.size() == 2U && weight->shape[0] == kN &&
      weight->shape[1] == kK / 2U &&
      weight->element_count == kPackedWeightBytes &&
      weight->byte_size == kPackedWeightBytes &&
      weight->file_begin == kPinnedCheckpointWeightBegin &&
      weight->file_end == kPinnedCheckpointWeightEnd &&
      weight->file_end <= header.value->file_size;
  test.expect(weight_exact,
              "checkpoint Down weight is pinned U8 [5120,8704] / "
              "44564480B range");
  const bool block_scale_exact =
      block_scale != nullptr && block_scale->dtype == st::DType::kF8E4M3 &&
      block_scale->shape.size() == 2U && block_scale->shape[0] == kN &&
      block_scale->shape[1] == kK / 16U &&
      block_scale->element_count == kBlockScaleBytes &&
      block_scale->byte_size == kBlockScaleBytes &&
      block_scale->file_begin == kPinnedCheckpointBlockScaleBegin &&
      block_scale->file_end == kPinnedCheckpointBlockScaleEnd &&
      block_scale->file_end <= header.value->file_size;
  test.expect(block_scale_exact,
              "checkpoint Down block scale is pinned F8_E4M3 "
              "[5120,1088] / 5570560B range");
  const bool weight_scale_2_exact =
      weight_scale_2 != nullptr && weight_scale_2->dtype == st::DType::kF32 &&
      weight_scale_2->shape.empty() && weight_scale_2->element_count == 1U &&
      weight_scale_2->byte_size == sizeof(float) &&
      weight_scale_2->file_begin == kPinnedCheckpointWeightScale2Begin &&
      weight_scale_2->file_end == kPinnedCheckpointWeightScale2End &&
      weight_scale_2->file_end <= header.value->file_size;
  test.expect(weight_scale_2_exact,
              "checkpoint Down weight_scale_2 is one pinned F32 scalar range");
  if (!shard_size_pinned || !weight_exact || !block_scale_exact ||
      !weight_scale_2_exact) {
    return false;
  }

  const std::uint64_t maximum_streamoff = static_cast<std::uint64_t>(
      std::numeric_limits<std::streamoff>::max());
  const std::uint64_t maximum_streamsize = static_cast<std::uint64_t>(
      std::numeric_limits<std::streamsize>::max());
  const bool ranges_representable =
      weight->file_begin <= maximum_streamoff &&
      weight->file_end <= maximum_streamoff &&
      block_scale->file_begin <= maximum_streamoff &&
      block_scale->file_end <= maximum_streamoff &&
      weight_scale_2->file_begin <= maximum_streamoff &&
      weight_scale_2->file_end <= maximum_streamoff &&
      weight->byte_size <= maximum_streamsize &&
      block_scale->byte_size <= maximum_streamsize &&
      weight_scale_2->byte_size <= maximum_streamsize;
  test.expect(ranges_representable,
              "checkpoint Down tensor ranges are stream-representable");
  if (!ranges_representable) {
    return false;
  }

  std::ifstream input(shard_path, std::ios::binary);
  if (!input) {
    test.expect(false, "open checkpoint Down shard read-only");
    return false;
  }
  input.seekg(0, std::ios::end);
  const std::streamoff observed_file_size = input.tellg();
  const bool file_size_stable =
      input && observed_file_size >= 0 &&
      static_cast<std::uint64_t>(observed_file_size) ==
          kPinnedCheckpointShardBytes &&
      static_cast<std::uint64_t>(observed_file_size) ==
          header.value->file_size;
  test.expect(file_size_stable,
              "checkpoint Down shard size remains pinned after header read");
  if (!file_size_stable) {
    return false;
  }

  payload.packed_weights.resize(kPackedWeightBytes);
  payload.block_scales.resize(kBlockScaleBytes);
  const auto read_exact =
      [&input](const std::uint64_t offset, void* const destination,
               const std::size_t bytes) {
        input.clear();
        input.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        if (!input) {
          return false;
        }
        input.read(static_cast<char*>(destination),
                   static_cast<std::streamsize>(bytes));
        return input.gcount() == static_cast<std::streamsize>(bytes);
      };
  const bool weight_read =
      read_exact(weight->file_begin, payload.packed_weights.data(),
                 payload.packed_weights.size());
  const bool block_scale_read =
      read_exact(block_scale->file_begin, payload.block_scales.data(),
                 payload.block_scales.size());
  const bool weight_scale_2_read =
      read_exact(weight_scale_2->file_begin,
                 payload.weight_scale_2_bytes.data(),
                 payload.weight_scale_2_bytes.size());
  if (weight_scale_2_read) {
    std::memcpy(&payload.weight_scale_2,
                payload.weight_scale_2_bytes.data(), sizeof(float));
    std::memcpy(&payload.weight_scale_2_bits,
                payload.weight_scale_2_bytes.data(), sizeof(std::uint32_t));
  }
  const bool scale_value_exact =
      weight_scale_2_read && std::isfinite(payload.weight_scale_2) &&
      payload.weight_scale_2 > 0.0F &&
      payload.weight_scale_2_bits == kPinnedCheckpointWeightScale2Bits;
  test.expect(weight_read, "read complete bounded checkpoint Down weight");
  test.expect(block_scale_read,
              "read complete bounded checkpoint Down block scale");
  test.expect(scale_value_exact,
              "read finite positive pinned checkpoint Down weight_scale_2");
  if (!weight_read || !block_scale_read || !scale_value_exact) {
    payload.packed_weights.clear();
    payload.block_scales.clear();
    return false;
  }

  payload.weight_sha256 = q3x::core::sha256(std::string_view(
      reinterpret_cast<const char*>(payload.packed_weights.data()),
      payload.packed_weights.size())).hex();
  payload.block_scale_sha256 = q3x::core::sha256(std::string_view(
      reinterpret_cast<const char*>(payload.block_scales.data()),
      payload.block_scales.size())).hex();
  payload.weight_scale_2_sha256 = q3x::core::sha256(std::string_view(
      reinterpret_cast<const char*>(payload.weight_scale_2_bytes.data()),
      payload.weight_scale_2_bytes.size())).hex();
  const bool hashes_pinned =
      payload.weight_sha256 == kPinnedCheckpointWeightSha256 &&
      payload.block_scale_sha256 == kPinnedCheckpointBlockScaleSha256 &&
      payload.weight_scale_2_sha256 ==
          kPinnedCheckpointWeightScale2Sha256;
  test.expect(hashes_pinned,
              "checkpoint Down weight/scale/scale_2 SHA256 match pinned "
              "payloads");
  if (!hashes_pinned) {
    payload.packed_weights.clear();
    payload.block_scales.clear();
    return false;
  }

  payload.weight_tensor = std::string(kCheckpointWeightTensor);
  payload.block_scale_tensor = std::string(kCheckpointBlockScaleTensor);
  payload.weight_scale_2_tensor =
      std::string(kCheckpointWeightScale2Tensor);
  payload.shard = *weight_shard;
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
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      test.expect(false, label + " allocation is representable");
      return false;
    }
    return test.cuda_ok(
        cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)),
        label);
  }

  [[nodiscard]] T* get() noexcept { return data_; }
  [[nodiscard]] const T* get() const noexcept { return data_; }

 private:
  T* data_ = nullptr;
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

  [[nodiscard]] bool create(TestContext& test) {
    bool ready = test.lt_ok(cublasLtCreate(&handle_), "create cuBLASLt");
    ready = ready && test.lt_ok(
                           cublasLtMatmulDescCreate(
                               &operation_, CUBLAS_COMPUTE_32F, CUDA_R_32F),
                           "create BF16 matmul descriptor");

    const cublasOperation_t transpose_weight = CUBLAS_OP_T;
    ready = ready && test.lt_ok(
                           cublasLtMatmulDescSetAttribute(
                               operation_, CUBLASLT_MATMUL_DESC_TRANSA,
                               &transpose_weight,
                               sizeof(transpose_weight)),
                           "transpose canonical BF16 weight operand");

    // A row-major [M,K] allocation is column-major [K,M].  The persistent
    // BF16 weight allocation is canonical row-major [N,K], hence
    // column-major [K,N].  Transpose that first operand to compute
    // C^T = B A^T while the visible output remains row-major [M,N].
    ready = ready && test.lt_ok(
                           cublasLtMatrixLayoutCreate(
                               &weight_layout_, CUDA_R_16BF, kK, kN, kK),
                           "create canonical BF16 weight layout [K,N]");
    ready = ready && test.lt_ok(
                           cublasLtMatrixLayoutCreate(
                               &activation_layout_, CUDA_R_16BF, kK, kM, kK),
                           "create BF16 activation layout [K,M]");
    ready = ready && test.lt_ok(
                           cublasLtMatrixLayoutCreate(
                               &output_layout_, CUDA_R_16BF, kN, kM, kN),
                           "create BF16 output layout [N,M]");
    ready = ready && test.lt_ok(
                           cublasLtMatmulPreferenceCreate(&preference_),
                           "create cuBLASLt preference");
    std::size_t workspace_bytes = kWorkspaceBytes;
    ready = ready && test.lt_ok(
                           cublasLtMatmulPreferenceSetAttribute(
                               preference_,
                               CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                               &workspace_bytes, sizeof(workspace_bytes)),
                           "set cuBLASLt workspace preference");
    constexpr std::uint32_t kMinimumOperandAlignmentBytes = 16U;
    ready = ready && test.lt_ok(
                           cublasLtMatmulPreferenceSetAttribute(
                               preference_,
                               CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                               &kMinimumOperandAlignmentBytes,
                               sizeof(kMinimumOperandAlignmentBytes)),
                           "set cuBLASLt A minimum alignment");
    ready = ready && test.lt_ok(
                           cublasLtMatmulPreferenceSetAttribute(
                               preference_,
                               CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                               &kMinimumOperandAlignmentBytes,
                               sizeof(kMinimumOperandAlignmentBytes)),
                           "set cuBLASLt B minimum alignment");
    ready = ready && test.lt_ok(
                           cublasLtMatmulPreferenceSetAttribute(
                               preference_,
                               CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                               &kMinimumOperandAlignmentBytes,
                               sizeof(kMinimumOperandAlignmentBytes)),
                           "set cuBLASLt C minimum alignment");
    ready = ready && test.lt_ok(
                           cublasLtMatmulPreferenceSetAttribute(
                               preference_,
                               CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
                               &kMinimumOperandAlignmentBytes,
                               sizeof(kMinimumOperandAlignmentBytes)),
                           "set cuBLASLt D minimum alignment");
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

// Trusted scalar decoder with canonical contiguous [N,K] output and the exact
// E2M1/E4M3FN-to-BF16 boundary as production.  It is intentionally slow and
// runs only outside timing to independently validate the optimized route.
__global__ void dequantize_nvfp4_contiguous_reference_kernel(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales,
    __nv_bfloat16* const canonical_bf16) {
  const std::size_t count = kN * kK;
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += stride) {
    const std::size_t n = index / kK;
    const std::size_t k = index - n * kK;
    const std::uint8_t packed =
        packed_weights[n * (kK / 2U) + k / 2U];
    const std::uint8_t nibble =
        (k & 1U) != 0U ? static_cast<std::uint8_t>(packed >> 4U)
                       : static_cast<std::uint8_t>(packed & 0x0fU);
    const std::uint8_t scale_code =
        block_scales[n * (kK / 16U) + k / 16U];
    const float value =
        decode_e2m1_device(nibble) * decode_e4m3fn_device(scale_code);
    canonical_bf16[index] = __float2bfloat16_rn(value);
  }
}

// Production-shaped large-M staging route: one CTA owns one canonical N row.
// Every warp prefetches ten 32-byte packed spans plus aligned four-scale words
// before decoding, then writes adjacent uint32 BF16 pairs.  The cuBLASLt weight
// operand is transposed by its descriptor, so no physical [N,K] -> [K,N]
// transpose is required.
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

// Minimal stop-loss candidate for the Down staging path.  Unlike the locked
// 34-pass prefetch control above, this keeps only the current packed byte and
// scale word live: each pass loads, decodes, and stores before the next pass.
// The non-unrolled loop is deliberate so the compiler cannot materialize a
// 34-element local array behind the source-level scalar form.
__global__ __launch_bounds__(256, 4)
void dequantize_nvfp4_contiguous_sequential_kernel(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales,
    __nv_bfloat16* const canonical_bf16) {
  constexpr unsigned int kPackedPerRow = kK / 2U;
  constexpr unsigned int kScalesPerRow = kK / 16U;
  constexpr unsigned int kThreads = 256U;
  constexpr unsigned int kPasses = kPackedPerRow / kThreads;
  static_assert(kPasses == 34U);
  static_assert(kPackedPerRow == kPasses * kThreads);
  const unsigned int n = blockIdx.x;
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
  const std::size_t packed_base =
      static_cast<std::size_t>(n) * kPackedPerRow;
  const std::size_t scale_base =
      static_cast<std::size_t>(n) * kScalesPerRow;
  auto* const output_pairs = reinterpret_cast<std::uint32_t*>(canonical_bf16);

#pragma unroll 1
  for (unsigned int pass = 0U; pass < kPasses; ++pass) {
    const unsigned int packed_k = threadIdx.x + pass * kThreads;
    const std::uint8_t packed = packed_weights[packed_base + packed_k];
    std::uint32_t scale_word = 0U;
    if (lane == 0U) {
      const std::size_t word_index =
          scale_base + pass * (kThreads / 8U) + warp * 4U;
      scale_word = *reinterpret_cast<const std::uint32_t*>(
          block_scales + word_index);
    }
    scale_word = __shfl_sync(0xffff'ffffU, scale_word, 0);
    const std::uint8_t scale_code = static_cast<std::uint8_t>(
        scale_word >> ((lane >> 3U) * 8U));
    const float scale = decode_e4m3fn_device(scale_code);
    const __nv_bfloat16 low = __float2bfloat16_rn(
        decode_e2m1_device(packed & 0x0fU) * scale);
    const __nv_bfloat16 high = __float2bfloat16_rn(
        decode_e2m1_device(packed >> 4U) * scale);
    output_pairs[packed_base + packed_k] =
        static_cast<std::uint32_t>(__bfloat16_as_ushort(low)) |
        (static_cast<std::uint32_t>(__bfloat16_as_ushort(high)) << 16U);
  }
}

template <unsigned int kPassBase, unsigned int kWindowPasses>
__device__ __forceinline__ void dequantize_nvfp4_contiguous_window(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales,
    std::uint32_t* const output_pairs, const std::size_t packed_base,
    const std::size_t scale_base, const unsigned int lane,
    const unsigned int warp) {
  constexpr unsigned int kThreads = 256U;
  static_assert(kWindowPasses == 8U || kWindowPasses == 2U);
  static_assert(kPassBase + kWindowPasses <= 34U);
  std::uint8_t packed_values[kWindowPasses];
  std::uint32_t scale_words[kWindowPasses];
#pragma unroll
  for (unsigned int slot = 0U; slot < kWindowPasses; ++slot) {
    constexpr unsigned int kScaleSpanPerPass = kThreads / 8U;
    const unsigned int pass = kPassBase + slot;
    const unsigned int packed_k = threadIdx.x + pass * kThreads;
    packed_values[slot] = packed_weights[packed_base + packed_k];
    scale_words[slot] = 0U;
    if (lane == 0U) {
      const std::size_t word_index =
          scale_base + pass * kScaleSpanPerPass + warp * 4U;
      scale_words[slot] = *reinterpret_cast<const std::uint32_t*>(
          block_scales + word_index);
    }
  }

#pragma unroll
  for (unsigned int slot = 0U; slot < kWindowPasses; ++slot) {
    const unsigned int pass = kPassBase + slot;
    const unsigned int packed_k = threadIdx.x + pass * kThreads;
    const std::uint32_t scale_word =
        __shfl_sync(0xffff'ffffU, scale_words[slot], 0);
    const std::uint8_t scale_code = static_cast<std::uint8_t>(
        scale_word >> ((lane >> 3U) * 8U));
    const float scale = decode_e4m3fn_device(scale_code);
    const std::uint8_t packed = packed_values[slot];
    const __nv_bfloat16 low = __float2bfloat16_rn(
        decode_e2m1_device(packed & 0x0fU) * scale);
    const __nv_bfloat16 high = __float2bfloat16_rn(
        decode_e2m1_device(packed >> 4U) * scale);
    output_pairs[packed_base + packed_k] =
        static_cast<std::uint32_t>(__bfloat16_as_ushort(low)) |
        (static_cast<std::uint32_t>(__bfloat16_as_ushort(high)) << 16U);
  }
}

// Bounded latency-hiding candidate: four complete eight-pass prefetch/decode
// windows followed by one compile-time two-pass tail.  Every window is fully
// unrolled, preserving cross-pass memory-level parallelism without the
// locked baseline's 34-pass live range.
__global__ __launch_bounds__(256, 4)
void dequantize_nvfp4_contiguous_window8_kernel(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales,
    __nv_bfloat16* const canonical_bf16) {
  constexpr unsigned int kPackedPerRow = kK / 2U;
  constexpr unsigned int kScalesPerRow = kK / 16U;
  static_assert(kPackedPerRow == 34U * 256U);
  const unsigned int n = blockIdx.x;
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
  const std::size_t packed_base =
      static_cast<std::size_t>(n) * kPackedPerRow;
  const std::size_t scale_base =
      static_cast<std::size_t>(n) * kScalesPerRow;
  auto* const output_pairs = reinterpret_cast<std::uint32_t*>(canonical_bf16);
  dequantize_nvfp4_contiguous_window<0U, 8U>(
      packed_weights, block_scales, output_pairs, packed_base, scale_base,
      lane, warp);
  dequantize_nvfp4_contiguous_window<8U, 8U>(
      packed_weights, block_scales, output_pairs, packed_base, scale_base,
      lane, warp);
  dequantize_nvfp4_contiguous_window<16U, 8U>(
      packed_weights, block_scales, output_pairs, packed_base, scale_base,
      lane, warp);
  dequantize_nvfp4_contiguous_window<24U, 8U>(
      packed_weights, block_scales, output_pairs, packed_base, scale_base,
      lane, warp);
  dequantize_nvfp4_contiguous_window<32U, 2U>(
      packed_weights, block_scales, output_pairs, packed_base, scale_base,
      lane, warp);
}

__global__ void validate_bf16_replay_kernel(
    const __nv_bfloat16* const output,
    const __nv_bfloat16* const replay_reference, const std::size_t count,
    unsigned long long* const mismatch_count,
    unsigned long long* const nonfinite_count,
    unsigned long long* const encoded_sum) {
  unsigned long long local_mismatch = 0U;
  unsigned long long local_nonfinite = 0U;
  unsigned long long local_sum = 0U;
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += stride) {
    const std::uint16_t encoded =
        reinterpret_cast<const std::uint16_t*>(output)[index];
    const std::uint16_t replay_encoded =
        reinterpret_cast<const std::uint16_t*>(replay_reference)[index];
    local_mismatch += encoded != replay_encoded ? 1U : 0U;
    local_nonfinite +=
        isfinite(__bfloat162float(output[index])) ? 0U : 1U;
    local_sum += encoded;
  }
  if (local_mismatch != 0U) {
    atomicAdd(mismatch_count, local_mismatch);
  }
  if (local_nonfinite != 0U) {
    atomicAdd(nonfinite_count, local_nonfinite);
  }
  atomicAdd(encoded_sum, local_sum);
}

__global__ void validate_bytes_immutable_kernel(
    const std::uint8_t* const values,
    const std::uint8_t* const snapshot, const std::size_t count,
    unsigned long long* const mismatch_count,
    unsigned long long* const value_sum,
    unsigned long long* const snapshot_sum) {
  unsigned long long local_mismatch = 0U;
  unsigned long long local_value_sum = 0U;
  unsigned long long local_snapshot_sum = 0U;
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += stride) {
    const std::uint8_t value = values[index];
    const std::uint8_t reference = snapshot[index];
    local_mismatch += value != reference ? 1U : 0U;
    local_value_sum += value;
    local_snapshot_sum += reference;
  }
  if (local_mismatch != 0U) {
    atomicAdd(mismatch_count, local_mismatch);
  }
  atomicAdd(value_sum, local_value_sum);
  atomicAdd(snapshot_sum, local_snapshot_sum);
}

__global__ void validate_bf16_guards_kernel(
    const __nv_bfloat16* const guarded, const std::size_t payload_count,
    const std::size_t guard_count, const std::uint16_t guard_bits,
    unsigned long long* const mismatch_count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= guard_count) {
    return;
  }
  const auto* const encoded =
      reinterpret_cast<const std::uint16_t*>(guarded);
  const bool prefix_mismatch = encoded[index] != guard_bits;
  const bool suffix_mismatch =
      encoded[guard_count + payload_count + index] != guard_bits;
  if (prefix_mismatch || suffix_mismatch) {
    atomicAdd(mismatch_count,
              static_cast<unsigned long long>(prefix_mismatch) +
                  static_cast<unsigned long long>(suffix_mismatch));
  }
}

[[nodiscard]] bool launch_lt(
    TestContext& test, const LtObjects& lt,
    const cublasLtMatmulAlgo_t& algorithm, const float alpha,
    const __nv_bfloat16* const persistent_weight,
    const __nv_bfloat16* const activation, __nv_bfloat16* const output,
    void* const workspace, const std::size_t workspace_bytes,
    const cudaStream_t stream, const std::string& label) {
  constexpr float kBeta = 0.0F;
  return test.lt_ok(
      cublasLtMatmul(
          lt.handle(), lt.operation(), &alpha, persistent_weight,
          lt.weight_layout(), activation, lt.activation_layout(), &kBeta,
          output, lt.output_layout(), output, lt.output_layout(), &algorithm,
          workspace, workspace_bytes, stream),
      label);
}

[[nodiscard]] bool launch_dequantize_contiguous(
    TestContext& test, const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales,
    __nv_bfloat16* const transient_weight, const cudaStream_t stream,
    const std::string& label) {
  constexpr unsigned int kThreads = 256U;
  dequantize_nvfp4_contiguous_kernel<<<static_cast<unsigned int>(kN),
                                       kThreads, 0, stream>>>(
      packed_weights, block_scales, transient_weight);
  return test.cuda_ok(cudaGetLastError(), label);
}

[[nodiscard]] bool launch_dequantize_contiguous_sequential(
    TestContext& test, const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales,
    __nv_bfloat16* const transient_weight, const cudaStream_t stream,
    const std::string& label) {
  constexpr unsigned int kThreads = 256U;
  dequantize_nvfp4_contiguous_sequential_kernel<<<
      static_cast<unsigned int>(kN), kThreads, 0, stream>>>(
      packed_weights, block_scales, transient_weight);
  return test.cuda_ok(cudaGetLastError(), label);
}

[[nodiscard]] bool launch_dequantize_contiguous_window8(
    TestContext& test, const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales,
    __nv_bfloat16* const transient_weight, const cudaStream_t stream,
    const std::string& label) {
  constexpr unsigned int kThreads = 256U;
  dequantize_nvfp4_contiguous_window8_kernel<<<
      static_cast<unsigned int>(kN), kThreads, 0, stream>>>(
      packed_weights, block_scales, transient_weight);
  return test.cuda_ok(cudaGetLastError(), label);
}

[[nodiscard]] double measure_algorithm(
    TestContext& test, const LtObjects& lt,
    const cublasLtMatmulAlgo_t& algorithm,
    const __nv_bfloat16* const persistent_weight,
    const __nv_bfloat16* const activation, __nv_bfloat16* const output,
    void* const workspace, const std::size_t workspace_bytes,
    const cudaStream_t stream, const int warmups, const int iterations,
    const std::string& label) {
  for (int warmup = 0; warmup < warmups; ++warmup) {
    if (!launch_lt(test, lt, algorithm, 1.0F, persistent_weight, activation,
                   output, workspace, workspace_bytes, stream,
                   label + " warmup " + std::to_string(warmup))) {
      return std::numeric_limits<double>::quiet_NaN();
    }
  }
  if (!test.cuda_ok(cudaStreamSynchronize(stream), label + " warmup sync")) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  bool ready = test.cuda_ok(cudaEventCreate(&start), label + " create start");
  ready = ready &&
          test.cuda_ok(cudaEventCreate(&stop), label + " create stop");
  ready = ready &&
          test.cuda_ok(cudaEventRecord(start, stream), label + " record start");
  for (int iteration = 0; ready && iteration < iterations; ++iteration) {
    ready = launch_lt(test, lt, algorithm, 1.0F, persistent_weight,
                      activation, output, workspace, workspace_bytes, stream,
                      label + " measured " + std::to_string(iteration));
  }
  ready = ready &&
          test.cuda_ok(cudaEventRecord(stop, stream), label + " record stop");
  ready = ready &&
          test.cuda_ok(cudaEventSynchronize(stop), label + " stop sync");
  float total_milliseconds = 0.0F;
  ready = ready && test.cuda_ok(
                       cudaEventElapsedTime(&total_milliseconds, start, stop),
                       label + " elapsed time");
  if (stop != nullptr) {
    (void)cudaEventDestroy(stop);
  }
  if (start != nullptr) {
    (void)cudaEventDestroy(start);
  }
  if (!ready) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return static_cast<double>(total_milliseconds) /
         static_cast<double>(iterations);
}

[[nodiscard]] double measure_dequantize(
    TestContext& test, const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales,
    __nv_bfloat16* const transient_weight, const cudaStream_t stream,
    const int warmups, const int iterations, const std::string& label) {
  for (int warmup = 0; warmup < warmups; ++warmup) {
    if (!launch_dequantize_contiguous(
            test, packed_weights, block_scales, transient_weight, stream,
            label + " warmup " + std::to_string(warmup))) {
      return std::numeric_limits<double>::quiet_NaN();
    }
  }
  if (!test.cuda_ok(cudaStreamSynchronize(stream), label + " warmup sync")) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  bool ready = test.cuda_ok(cudaEventCreate(&start), label + " create start");
  ready = ready &&
          test.cuda_ok(cudaEventCreate(&stop), label + " create stop");
  ready = ready &&
          test.cuda_ok(cudaEventRecord(start, stream), label + " record start");
  for (int iteration = 0; ready && iteration < iterations; ++iteration) {
    ready = launch_dequantize_contiguous(
        test, packed_weights, block_scales, transient_weight, stream,
        label + " measured " + std::to_string(iteration));
  }
  ready = ready &&
          test.cuda_ok(cudaEventRecord(stop, stream), label + " record stop");
  ready = ready &&
          test.cuda_ok(cudaEventSynchronize(stop), label + " stop sync");
  float total_milliseconds = 0.0F;
  ready = ready && test.cuda_ok(
                       cudaEventElapsedTime(&total_milliseconds, start, stop),
                       label + " elapsed time");
  if (stop != nullptr) {
    (void)cudaEventDestroy(stop);
  }
  if (start != nullptr) {
    (void)cudaEventDestroy(start);
  }
  if (!ready) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return static_cast<double>(total_milliseconds) /
         static_cast<double>(iterations);
}

[[nodiscard]] double measure_dequantize_sequential(
    TestContext& test, const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales,
    __nv_bfloat16* const transient_weight, const cudaStream_t stream,
    const int warmups, const int iterations, const std::string& label) {
  for (int warmup = 0; warmup < warmups; ++warmup) {
    if (!launch_dequantize_contiguous_sequential(
            test, packed_weights, block_scales, transient_weight, stream,
            label + " warmup " + std::to_string(warmup))) {
      return std::numeric_limits<double>::quiet_NaN();
    }
  }
  if (!test.cuda_ok(cudaStreamSynchronize(stream), label + " warmup sync")) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  bool ready = test.cuda_ok(cudaEventCreate(&start), label + " create start");
  ready = ready &&
          test.cuda_ok(cudaEventCreate(&stop), label + " create stop");
  ready = ready &&
          test.cuda_ok(cudaEventRecord(start, stream), label + " record start");
  for (int iteration = 0; ready && iteration < iterations; ++iteration) {
    ready = launch_dequantize_contiguous_sequential(
        test, packed_weights, block_scales, transient_weight, stream,
        label + " measured " + std::to_string(iteration));
  }
  ready = ready &&
          test.cuda_ok(cudaEventRecord(stop, stream), label + " record stop");
  ready = ready &&
          test.cuda_ok(cudaEventSynchronize(stop), label + " stop sync");
  float total_milliseconds = 0.0F;
  ready = ready && test.cuda_ok(
                       cudaEventElapsedTime(&total_milliseconds, start, stop),
                       label + " elapsed time");
  if (stop != nullptr) {
    (void)cudaEventDestroy(stop);
  }
  if (start != nullptr) {
    (void)cudaEventDestroy(start);
  }
  if (!ready) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return static_cast<double>(total_milliseconds) /
         static_cast<double>(iterations);
}


[[nodiscard]] double measure_dequantize_window8(
    TestContext& test, const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales,
    __nv_bfloat16* const transient_weight, const cudaStream_t stream,
    const int warmups, const int iterations, const std::string& label) {
  for (int warmup = 0; warmup < warmups; ++warmup) {
    if (!launch_dequantize_contiguous_window8(
            test, packed_weights, block_scales, transient_weight, stream,
            label + " warmup " + std::to_string(warmup))) {
      return std::numeric_limits<double>::quiet_NaN();
    }
  }
  if (!test.cuda_ok(cudaStreamSynchronize(stream), label + " warmup sync")) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  bool ready = test.cuda_ok(cudaEventCreate(&start), label + " create start");
  ready = ready &&
          test.cuda_ok(cudaEventCreate(&stop), label + " create stop");
  ready = ready &&
          test.cuda_ok(cudaEventRecord(start, stream), label + " record start");
  for (int iteration = 0; ready && iteration < iterations; ++iteration) {
    ready = launch_dequantize_contiguous_window8(
        test, packed_weights, block_scales, transient_weight, stream,
        label + " measured " + std::to_string(iteration));
  }
  ready = ready &&
          test.cuda_ok(cudaEventRecord(stop, stream), label + " record stop");
  ready = ready &&
          test.cuda_ok(cudaEventSynchronize(stop), label + " stop sync");
  float total_milliseconds = 0.0F;
  ready = ready && test.cuda_ok(
                       cudaEventElapsedTime(&total_milliseconds, start, stop),
                       label + " elapsed time");
  if (stop != nullptr) {
    (void)cudaEventDestroy(stop);
  }
  if (start != nullptr) {
    (void)cudaEventDestroy(start);
  }
  if (!ready) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return static_cast<double>(total_milliseconds) /
         static_cast<double>(iterations);
}

[[nodiscard]] double measure_inclusive_window8(
    TestContext& test, const LtObjects& lt,
    const cublasLtMatmulAlgo_t& algorithm,
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales,
    __nv_bfloat16* const transient_weight,
    const __nv_bfloat16* const activation, __nv_bfloat16* const output,
    void* const workspace, const std::size_t workspace_bytes,
    const cudaStream_t stream, const float weight_scale_2,
    const int warmups, const int iterations, const std::string& label) {
  const auto launch_chain = [&](const std::string& iteration_label) {
    bool launched = launch_dequantize_contiguous_window8(
        test, packed_weights, block_scales, transient_weight, stream,
        iteration_label + " dequantize");
    launched = launched && launch_lt(
                               test, lt, algorithm, weight_scale_2,
                               transient_weight,
                               activation, output, workspace, workspace_bytes,
                               stream, iteration_label + " cuBLASLt");
    return launched;
  };
  for (int warmup = 0; warmup < warmups; ++warmup) {
    if (!launch_chain(label + " warmup " + std::to_string(warmup))) {
      return std::numeric_limits<double>::quiet_NaN();
    }
  }
  if (!test.cuda_ok(cudaStreamSynchronize(stream), label + " warmup sync")) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  bool ready = test.cuda_ok(cudaEventCreate(&start), label + " create start");
  ready = ready &&
          test.cuda_ok(cudaEventCreate(&stop), label + " create stop");
  ready = ready &&
          test.cuda_ok(cudaEventRecord(start, stream), label + " record start");
  for (int iteration = 0; ready && iteration < iterations; ++iteration) {
    ready = launch_chain(label + " measured " + std::to_string(iteration));
  }
  ready = ready &&
          test.cuda_ok(cudaEventRecord(stop, stream), label + " record stop");
  ready = ready &&
          test.cuda_ok(cudaEventSynchronize(stop), label + " stop sync");
  float total_milliseconds = 0.0F;
  ready = ready && test.cuda_ok(
                       cudaEventElapsedTime(&total_milliseconds, start, stop),
                       label + " elapsed time");
  if (stop != nullptr) {
    (void)cudaEventDestroy(stop);
  }
  if (start != nullptr) {
    (void)cudaEventDestroy(start);
  }
  if (!ready) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return static_cast<double>(total_milliseconds) /
         static_cast<double>(iterations);
}

[[nodiscard]] bool launch_production_down(
    TestContext& test, const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales,
    const __nv_bfloat16* const activation, __nv_bfloat16* const output,
    const float weight_scale_2, const cudaStream_t stream,
    const std::string& label) {
  const int status = q3x::kernels::
      launch_sm87_nvfp4_w4a16_whole_chunk_down_gemm_bf16_cuda(
          packed_weights, block_scales, weight_scale_2,
          reinterpret_cast<const std::uint16_t*>(activation), kM, kN, kK,
          reinterpret_cast<std::uint16_t*>(output),
          static_cast<void*>(stream));
  return test.cuda_ok(static_cast<cudaError_t>(status), label);
}

[[nodiscard]] double measure_production_down(
    TestContext& test, const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales,
    const __nv_bfloat16* const activation, __nv_bfloat16* const output,
    const float weight_scale_2, const cudaStream_t stream, const int warmups,
    const int iterations, const std::string& label) {
  for (int warmup = 0; warmup < warmups; ++warmup) {
    if (!launch_production_down(
            test, packed_weights, block_scales, activation, output,
            weight_scale_2, stream,
            label + " warmup " + std::to_string(warmup))) {
      return std::numeric_limits<double>::quiet_NaN();
    }
  }
  if (!test.cuda_ok(cudaStreamSynchronize(stream), label + " warmup sync")) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  bool ready = test.cuda_ok(cudaEventCreate(&start), label + " create start");
  ready = ready &&
          test.cuda_ok(cudaEventCreate(&stop), label + " create stop");
  ready = ready &&
          test.cuda_ok(cudaEventRecord(start, stream), label + " record start");
  for (int iteration = 0; ready && iteration < iterations; ++iteration) {
    ready = launch_production_down(
        test, packed_weights, block_scales, activation, output,
        weight_scale_2, stream,
        label + " measured " + std::to_string(iteration));
  }
  ready = ready &&
          test.cuda_ok(cudaEventRecord(stop, stream), label + " record stop");
  ready = ready &&
          test.cuda_ok(cudaEventSynchronize(stop), label + " stop sync");
  float total_milliseconds = 0.0F;
  ready = ready && test.cuda_ok(
                       cudaEventElapsedTime(&total_milliseconds, start, stop),
                       label + " elapsed time");
  if (stop != nullptr) {
    (void)cudaEventDestroy(stop);
  }
  if (start != nullptr) {
    (void)cudaEventDestroy(start);
  }
  if (!ready) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return static_cast<double>(total_milliseconds) /
         static_cast<double>(iterations);
}

[[nodiscard]] bool launch_public_down_module(
    TestContext& test,
    q3x::kernels::Sm87Nvfp4PrefillDownCublasLtContext* const context,
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales,
    const __nv_bfloat16* const activation,
    __nv_bfloat16* const transient_weight, __nv_bfloat16* const output,
    const float weight_scale_2, const cudaStream_t stream,
    const std::string& label) {
  const int status =
      q3x::kernels::launch_sm87_nvfp4_prefill_cublaslt_down_c512(
          context, packed_weights, block_scales, weight_scale_2,
          reinterpret_cast<const std::uint16_t*>(activation), kM, kN, kK,
          reinterpret_cast<std::uint16_t*>(transient_weight),
          kBElements * sizeof(__nv_bfloat16),
          reinterpret_cast<std::uint16_t*>(output),
          static_cast<void*>(stream));
  return test.cuda_ok(static_cast<cudaError_t>(status), label);
}

[[nodiscard]] double measure_public_down_module(
    TestContext& test,
    q3x::kernels::Sm87Nvfp4PrefillDownCublasLtContext* const context,
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales,
    const __nv_bfloat16* const activation,
    __nv_bfloat16* const transient_weight, __nv_bfloat16* const output,
    const float weight_scale_2, const cudaStream_t stream, const int warmups,
    const int iterations, const std::string& label) {
  for (int warmup = 0; warmup < warmups; ++warmup) {
    if (!launch_public_down_module(
            test, context, packed_weights, block_scales, activation,
            transient_weight, output, weight_scale_2, stream,
            label + " warmup " + std::to_string(warmup))) {
      return std::numeric_limits<double>::quiet_NaN();
    }
  }
  if (!test.cuda_ok(cudaStreamSynchronize(stream), label + " warmup sync")) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  bool ready = test.cuda_ok(cudaEventCreate(&start), label + " create start");
  ready = ready &&
          test.cuda_ok(cudaEventCreate(&stop), label + " create stop");
  ready = ready &&
          test.cuda_ok(cudaEventRecord(start, stream), label + " record start");
  for (int iteration = 0; ready && iteration < iterations; ++iteration) {
    ready = launch_public_down_module(
        test, context, packed_weights, block_scales, activation,
        transient_weight, output, weight_scale_2, stream,
        label + " measured " + std::to_string(iteration));
  }
  ready = ready &&
          test.cuda_ok(cudaEventRecord(stop, stream), label + " record stop");
  ready = ready &&
          test.cuda_ok(cudaEventSynchronize(stop), label + " stop sync");
  float total_milliseconds = 0.0F;
  ready = ready && test.cuda_ok(
                       cudaEventElapsedTime(&total_milliseconds, start, stop),
                       label + " elapsed time");
  if (stop != nullptr) {
    (void)cudaEventDestroy(stop);
  }
  if (start != nullptr) {
    (void)cudaEventDestroy(start);
  }
  if (!ready) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return static_cast<double>(total_milliseconds) /
         static_cast<double>(iterations);
}

[[nodiscard]] double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2U;
  if ((values.size() & 1U) != 0U) {
    return values[middle];
  }
  return 0.5 * (values[middle - 1U] + values[middle]);
}

[[nodiscard]] std::size_t pointer_alignment_bytes(const void* const pointer) {
  const auto address = reinterpret_cast<std::uintptr_t>(pointer);
  if (address == 0U) {
    return 0U;
  }
  return static_cast<std::size_t>(address & (~address + 1U));
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

enum class Mode {
  kSmoke,
  kPerformanceCheckpoint,
};

[[nodiscard]] constexpr const char* mode_name(const Mode mode) noexcept {
  return mode == Mode::kSmoke ? "smoke" : "performance-checkpoint";
}

struct Options {
  std::string checkpoint_directory;
  Mode mode = Mode::kSmoke;
};

[[nodiscard]] bool parse_options(const int argc, char** argv,
                                 Options& options) {
  bool checkpoint_seen = false;
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
    } else if (argument == "--checkpoint") {
      if (checkpoint_seen) {
        std::cerr << "duplicate --checkpoint argument\n";
        return false;
      }
      if (index + 1 >= argc) {
        std::cerr << "--checkpoint requires a directory\n";
        return false;
      }
      checkpoint_seen = true;
      options.checkpoint_directory = argv[++index];
      if (options.checkpoint_directory.empty()) {
        std::cerr << "--checkpoint requires a non-empty directory\n";
        return false;
      }
    } else {
      std::cerr << "unknown argument: " << argument << '\n';
      return false;
    }
  }
  return true;
}

}  // namespace

int main(const int argc, char** argv) {
  Options options{};
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
        std::string(q3x::test::support::kQwen36Layer0DownWeight),
        std::string(q3x::test::support::kQwen36Layer0DownBlockScale),
        std::string(q3x::test::support::kQwen36Layer0DownWeightScale2),
    };
    constexpr std::uint64_t kPinnedPayloadBytes =
        kPackedWeightBytes + kBlockScaleBytes + sizeof(float);
    pinned_options.maximum_total_payload_bytes = kPinnedPayloadBytes;
    const q3x::test::support::PinnedBundleLoadResult pinned_bundle =
        q3x::test::support::load_pinned_checkpoint_bundle(
            options.checkpoint_directory,
            q3x::test::support::qwen36_27b_nvfp4_layer0_mlp_bundle(),
            pinned_options);
    if (!pinned_bundle) {
      std::cerr << "NVFP4_DOWN_C512_PINNED_BUNDLE_ERROR: ";
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
      std::cerr << "NVFP4_DOWN_C512_PINNED_BUNDLE_ERROR: expected "
                << pinned_options.tensor_names.size() << " tensors, loaded "
                << loaded.tensors.size() << '\n';
      return 1;
    }
    std::cout << "NVFP4_DOWN_C512_PINNED_BUNDLE: descriptor="
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
    std::cout << std::fixed << std::setprecision(9)
              << "NVFP4_DOWN_C512_PAYLOAD: payload=checkpoint"
              << " weight_tensor=" << checkpoint_payload.weight_tensor
              << " block_scale_tensor="
              << checkpoint_payload.block_scale_tensor
              << " weight_scale_2_tensor="
              << checkpoint_payload.weight_scale_2_tensor
              << " shard=" << checkpoint_payload.shard
              << " weight_bytes=" << checkpoint_payload.packed_weights.size()
              << " block_scale_bytes="
              << checkpoint_payload.block_scales.size()
              << " weight_scale_2=" << checkpoint_payload.weight_scale_2
              << " weight_scale_2_bits=0x" << std::hex
              << checkpoint_payload.weight_scale_2_bits << std::dec
              << " weight_range=" << kPinnedCheckpointWeightBegin << ':'
              << kPinnedCheckpointWeightEnd
              << " block_scale_range=" << kPinnedCheckpointBlockScaleBegin
              << ':' << kPinnedCheckpointBlockScaleEnd
              << " weight_scale_2_range="
              << kPinnedCheckpointWeightScale2Begin << ':'
              << kPinnedCheckpointWeightScale2End
              << " weight_sha256=" << checkpoint_payload.weight_sha256
              << " block_scale_sha256="
              << checkpoint_payload.block_scale_sha256
              << " weight_scale_2_sha256="
              << checkpoint_payload.weight_scale_2_sha256
              << " checkpoint_read_only=true\n";
  } else {
    std::cout << "NVFP4_DOWN_C512_PAYLOAD: payload=synthetic"
              << " weight_tensor=none block_scale_tensor=none"
              << " weight_scale_2_tensor=none"
              << " weight_bytes=" << kPackedWeightBytes
              << " block_scale_bytes=" << kBlockScaleBytes << '\n';
  }
  std::cout << "NVFP4_DOWN_C512_MODE: mode=" << mode_name(options.mode)
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
    std::cout << "SKIP: persistent-BF16 Down ceiling requires SM87; found "
              << properties.major << '.' << properties.minor << '\n';
    return 77;
  }

  const std::optional<ClockState> clocks = read_clock_state();
  if (performance_checkpoint &&
      (!clocks.has_value() || !clocks_are_fixed(*clocks))) {
    std::cout << "SKIP: real-checkpoint public Down module proof requires "
                 "fixed GPU and EMC clocks"
              << " clock_state_available="
              << (clocks.has_value() ? "true" : "false") << '\n';
    return 77;
  }

  cudaFuncAttributes dequant_attributes{};
  if (!test.cuda_ok(
          cudaFuncGetAttributes(&dequant_attributes,
                                dequantize_nvfp4_contiguous_kernel),
          "query Down direct-dequant kernel resources")) {
    return 1;
  }
  int active_dequant_blocks_per_sm = 0;
  if (!test.cuda_ok(
          cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &active_dequant_blocks_per_sm,
              dequantize_nvfp4_contiguous_kernel, 256, 0U),
          "query Down direct-dequant occupancy")) {
    return 1;
  }
  cudaFuncAttributes sequential_dequant_attributes{};
  if (!test.cuda_ok(
          cudaFuncGetAttributes(
              &sequential_dequant_attributes,
              dequantize_nvfp4_contiguous_sequential_kernel),
          "query sequential Down direct-dequant resources")) {
    return 1;
  }
  int active_sequential_dequant_blocks_per_sm = 0;
  if (!test.cuda_ok(
          cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &active_sequential_dequant_blocks_per_sm,
              dequantize_nvfp4_contiguous_sequential_kernel, 256, 0U),
          "query sequential Down direct-dequant occupancy")) {
    return 1;
  }
  cudaFuncAttributes window8_dequant_attributes{};
  if (!test.cuda_ok(
          cudaFuncGetAttributes(&window8_dequant_attributes,
                                dequantize_nvfp4_contiguous_window8_kernel),
          "query window8 Down direct-dequant resources")) {
    return 1;
  }
  int active_window8_dequant_blocks_per_sm = 0;
  if (!test.cuda_ok(
          cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &active_window8_dequant_blocks_per_sm,
              dequantize_nvfp4_contiguous_window8_kernel, 256, 0U),
          "query window8 Down direct-dequant occupancy")) {
    return 1;
  }

  std::cout << std::fixed << std::setprecision(6)
            << "CUBLASLT_DOWN_C512_PROTOCOL: device=" << properties.name
            << " cc=" << properties.major << '.' << properties.minor
            << " M=" << kM << " N=" << kN << " K=" << kK
            << " persistent_B_bytes="
            << kBElements * sizeof(__nv_bfloat16)
            << " control_dequantization_timed=false workspace_bytes="
            << kWorkspaceBytes << " useful_GFLOP=" << kUsefulFlops / 1.0e9
            << " fixed_clocks="
            << (clocks.has_value() && clocks_are_fixed(*clocks) ? "true"
                                                                 : "false");
  if (clocks.has_value()) {
    std::cout << " gpu_min_hz=" << clocks->gpu_min
              << " gpu_current_hz=" << clocks->gpu_current
              << " gpu_max_hz=" << clocks->gpu_max
              << " emc_min_hz=" << clocks->emc_min
              << " emc_current_hz=" << clocks->emc_current
              << " emc_max_hz=" << clocks->emc_max;
  }
  std::cout
            << '\n';
  std::cout << "NVFP4_DEQUANT_DOWN_C512_RESOURCES: threads=256 blocks="
            << kN << " registers_per_thread=" << dequant_attributes.numRegs
            << " static_shared_bytes=" << dequant_attributes.sharedSizeBytes
            << " local_bytes_per_thread=" << dequant_attributes.localSizeBytes
            << " max_dynamic_shared_bytes="
            << dequant_attributes.maxDynamicSharedSizeBytes
            << " active_blocks_per_sm=" << active_dequant_blocks_per_sm
            << '\n';
  std::cout << "NVFP4_DEQUANT_DOWN_C512_SEQUENTIAL_RESOURCES: threads=256"
            << " blocks=" << kN
            << " registers_per_thread="
            << sequential_dequant_attributes.numRegs
            << " static_shared_bytes="
            << sequential_dequant_attributes.sharedSizeBytes
            << " local_bytes_per_thread="
            << sequential_dequant_attributes.localSizeBytes
            << " max_dynamic_shared_bytes="
            << sequential_dequant_attributes.maxDynamicSharedSizeBytes
            << " active_blocks_per_sm="
            << active_sequential_dequant_blocks_per_sm << '\n';
  std::cout << "NVFP4_DEQUANT_DOWN_C512_WINDOW8_RESOURCES: threads=256"
            << " blocks=" << kN
            << " registers_per_thread=" << window8_dequant_attributes.numRegs
            << " static_shared_bytes="
            << window8_dequant_attributes.sharedSizeBytes
            << " local_bytes_per_thread="
            << window8_dequant_attributes.localSizeBytes
            << " max_dynamic_shared_bytes="
            << window8_dequant_attributes.maxDynamicSharedSizeBytes
            << " active_blocks_per_sm="
            << active_window8_dequant_blocks_per_sm << '\n';

  DeviceBuffer<__nv_bfloat16> activation;
  DeviceBuffer<__nv_bfloat16> persistent_weight;
  DeviceBuffer<__nv_bfloat16> output;
  DeviceBuffer<__nv_bfloat16> replay_reference;
  DeviceBuffer<__nv_bfloat16> production_output;
  DeviceBuffer<__nv_bfloat16> dequant_reference;
  DeviceBuffer<__nv_bfloat16> sequential_dequant_output;
  DeviceBuffer<__nv_bfloat16> window8_dequant_output;
  DeviceBuffer<std::uint8_t> canonical_packed_weight;
  DeviceBuffer<std::uint8_t> canonical_block_scale;
  DeviceBuffer<__nv_bfloat16> activation_snapshot;
  DeviceBuffer<std::uint8_t> canonical_packed_weight_snapshot;
  DeviceBuffer<std::uint8_t> canonical_block_scale_snapshot;
  DeviceBuffer<std::uint8_t> workspace;
  DeviceBuffer<unsigned long long> validation;
  bool ready = activation.allocate(test, kAElements, "allocate BF16 A");
  ready = ready && persistent_weight.allocate(test, kBElements,
                                               "allocate persistent BF16 B");
  ready = ready && output.allocate(test, kCElements + 2U * kGuardElements,
                                   "allocate guarded BF16 C");
  ready = ready && replay_reference.allocate(
                       test, kCElements + 2U * kGuardElements,
                       "allocate guarded BF16 replay C");
  ready = ready && production_output.allocate(
                       test, kCElements + 2U * kGuardElements,
                       "allocate guarded production M128 C");
  ready = ready && dequant_reference.allocate(
                       test, kBElements, "allocate BF16 dequant reference");
  ready = ready && sequential_dequant_output.allocate(
                       test, kBElements + 2U * kGuardElements,
                       "allocate guarded sequential BF16 dequant output");
  ready = ready && window8_dequant_output.allocate(
                       test, kBElements + 2U * kGuardElements,
                       "allocate guarded window8 BF16 dequant output");
  ready = ready && canonical_packed_weight.allocate(
                       test, kPackedWeightBytes,
                       "allocate canonical NVFP4 packed weight");
  ready = ready && canonical_block_scale.allocate(
                       test, kBlockScaleBytes,
                       "allocate canonical NVFP4 block scales");
  ready = ready && activation_snapshot.allocate(
                       test, kAElements, "allocate activation snapshot");
  ready = ready && canonical_packed_weight_snapshot.allocate(
                       test, kPackedWeightBytes,
                       "allocate packed-weight snapshot");
  ready = ready && canonical_block_scale_snapshot.allocate(
                       test, kBlockScaleBytes,
                       "allocate block-scale snapshot");
  ready = ready && validation.allocate(test, 3U, "allocate validation counts");

  cudaStream_t stream = nullptr;
  ready = ready && test.cuda_ok(
                       cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                       "create nonblocking stream");
  LtObjects lt;
  ready = ready && lt.create(test);
  if (!ready) {
    if (stream != nullptr) {
      (void)cudaStreamDestroy(stream);
    }
    return 1;
  }

  __nv_bfloat16* const candidate_output = output.get() + kGuardElements;
  __nv_bfloat16* const trusted_output =
      replay_reference.get() + kGuardElements;
  __nv_bfloat16* const exact_production_output =
      production_output.get() + kGuardElements;
  __nv_bfloat16* const sequential_dequant_payload =
      sequential_dequant_output.get() + kGuardElements;
  __nv_bfloat16* const window8_dequant_payload =
      window8_dequant_output.get() + kGuardElements;
  const std::size_t weight_alignment =
      pointer_alignment_bytes(persistent_weight.get());
  const std::size_t activation_alignment =
      pointer_alignment_bytes(activation.get());
  const std::size_t output_alignment =
      pointer_alignment_bytes(candidate_output);
  constexpr std::size_t kRequiredLtPointerAlignment = 16U;
  const bool lt_pointer_alignment_gate =
      weight_alignment >= kRequiredLtPointerAlignment &&
      activation_alignment >= kRequiredLtPointerAlignment &&
      output_alignment >= kRequiredLtPointerAlignment;
  test.expect(lt_pointer_alignment_gate,
              "cuBLASLt operands satisfy advertised 16-byte alignment");
  std::cout << "CUBLASLT_DOWN_C512_POINTER_ALIGNMENT: preference_bytes="
            << kRequiredLtPointerAlignment
            << " A_weight_actual_bytes=" << weight_alignment
            << " B_activation_actual_bytes=" << activation_alignment
            << " C_output_actual_bytes=" << output_alignment
            << " D_output_actual_bytes=" << output_alignment
            << " gate=" << (lt_pointer_alignment_gate ? "PASS" : "FAIL")
            << '\n';
  const std::size_t guarded_output_bytes =
      (kCElements + 2U * kGuardElements) * sizeof(__nv_bfloat16);
  ready = test.cuda_ok(
      cudaMemsetAsync(output.get(), static_cast<int>(kGuardBits & 0xffU),
                      guarded_output_bytes, stream),
      "initialize candidate output guards");
  ready = ready && test.cuda_ok(
                       cudaMemsetAsync(replay_reference.get(),
                                       static_cast<int>(kGuardBits & 0xffU),
                                       guarded_output_bytes, stream),
                       "initialize trusted output guards");
  ready = ready && test.cuda_ok(
                       cudaMemsetAsync(production_output.get(),
                                       static_cast<int>(kGuardBits & 0xffU),
                                       guarded_output_bytes, stream),
                       "initialize production output guards");
  const std::size_t guarded_dequant_bytes =
      (kBElements + 2U * kGuardElements) * sizeof(__nv_bfloat16);
  ready = ready && test.cuda_ok(
                       cudaMemsetAsync(
                           sequential_dequant_output.get(),
                           static_cast<int>(kGuardBits & 0xffU),
                           guarded_dequant_bytes, stream),
                       "initialize sequential dequant output guards");
  ready = ready && test.cuda_ok(
                       cudaMemsetAsync(
                           window8_dequant_output.get(),
                           static_cast<int>(kGuardBits & 0xffU),
                           guarded_dequant_bytes, stream),
                       "initialize window8 dequant output guards");

  constexpr unsigned int kFillThreads = 256U;
  const auto fill = [&](DeviceBuffer<__nv_bfloat16>& buffer,
                        const std::size_t count, const std::uint32_t salt,
                        const float scale) {
    const std::size_t blocks =
        (count + kFillThreads - 1U) / kFillThreads;
    fill_deterministic_bf16_kernel<<<static_cast<unsigned int>(blocks),
                                     kFillThreads, 0, stream>>>(
        buffer.get(), count, salt, scale);
    return test.cuda_ok(cudaGetLastError(), "launch deterministic BF16 fill");
  };
  ready = fill(activation, kAElements, 0x1234'5678U, 1.0F / 64.0F);
  ready = ready && fill(persistent_weight, kBElements, 0x9abc'def0U,
                        1.0F / 128.0F);
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           activation_snapshot.get(), activation.get(),
                           kAElements * sizeof(__nv_bfloat16),
                           cudaMemcpyDeviceToDevice, stream),
                       "snapshot BF16 activations");
  ready = ready &&
          test.cuda_ok(cudaStreamSynchronize(stream), "finish BF16 fills");

  std::array<cublasLtMatmulHeuristicResult_t, kMaximumHeuristics> heuristics{};
  int returned_algorithms = 0;
  ready = ready && test.lt_ok(
                       cublasLtMatmulAlgoGetHeuristic(
                           lt.handle(), lt.operation(), lt.weight_layout(),
                           lt.activation_layout(), lt.output_layout(),
                           lt.output_layout(), lt.preference(),
                           kMaximumHeuristics, heuristics.data(),
                           &returned_algorithms),
                       "query cuBLASLt algorithms");
  test.expect(returned_algorithms > 0,
              "cuBLASLt returns at least one BF16 algorithm");
  if (!ready || returned_algorithms <= 0) {
    (void)cudaStreamDestroy(stream);
    return 1;
  }

  int selected_index = -1;
  double selected_milliseconds = std::numeric_limits<double>::infinity();
  const auto read_algorithm_config = [](
                                         const cublasLtMatmulAlgo_t& algorithm,
                                         const cublasLtMatmulAlgoConfigAttributes_t attribute,
                                         auto* const value) {
    std::size_t written = 0U;
    return value != nullptr &&
           cublasLtMatmulAlgoConfigGetAttribute(
               &algorithm, attribute, value, sizeof(*value), &written) ==
               CUBLAS_STATUS_SUCCESS &&
           written == sizeof(*value);
  };
  for (int index = 0; index < returned_algorithms; ++index) {
    if (heuristics[static_cast<std::size_t>(index)].state !=
            CUBLAS_STATUS_SUCCESS ||
        heuristics[static_cast<std::size_t>(index)].workspaceSize >
            kWorkspaceBytes) {
      continue;
    }
    const double milliseconds = measure_algorithm(
        test, lt, heuristics[static_cast<std::size_t>(index)].algo,
        persistent_weight.get(), activation.get(), candidate_output,
        workspace.get(), kWorkspaceBytes, stream, kSelectionWarmups,
        kSelectionIterations, "select algorithm " + std::to_string(index));
    const auto& algorithm =
        heuristics[static_cast<std::size_t>(index)].algo;
    std::int32_t algorithm_id = -1;
    std::uint32_t tile_id = 0U;
    std::int32_t split_k = 0;
    std::uint32_t reduction_scheme = 0U;
    std::uint32_t cta_swizzle = 0U;
    std::uint32_t custom_option = 0U;
    std::uint32_t stages_id = 0U;
    const bool config_available =
        read_algorithm_config(algorithm, CUBLASLT_ALGO_CONFIG_ID,
                              &algorithm_id) &&
        read_algorithm_config(algorithm, CUBLASLT_ALGO_CONFIG_TILE_ID,
                              &tile_id) &&
        read_algorithm_config(algorithm, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                              &split_k) &&
        read_algorithm_config(
            algorithm, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
            &reduction_scheme) &&
        read_algorithm_config(algorithm,
                              CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
                              &cta_swizzle) &&
        read_algorithm_config(algorithm, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
                              &custom_option) &&
        read_algorithm_config(algorithm, CUBLASLT_ALGO_CONFIG_STAGES_ID,
                              &stages_id);
    const double tflops = kUsefulFlops / (milliseconds * 1.0e9);
    std::cout << "CUBLASLT_DOWN_C512_HEURISTIC: index=" << index
              << " workspace_bytes="
              << heuristics[static_cast<std::size_t>(index)].workspaceSize
              << " config_available="
              << (config_available ? "true" : "false")
              << " algorithm_id=" << algorithm_id
              << " tile_id=" << tile_id
              << " split_k=" << split_k
              << " reduction_scheme=" << reduction_scheme
              << " cta_swizzle=" << cta_swizzle
              << " custom_option=" << custom_option
              << " stages_id=" << stages_id
              << " milliseconds=" << milliseconds
              << " TFLOP_per_s=" << tflops << '\n';
    if (std::isfinite(milliseconds) && milliseconds < selected_milliseconds) {
      selected_milliseconds = milliseconds;
      selected_index = index;
    }
  }
  test.expect(selected_index >= 0,
              "at least one cuBLASLt BF16 algorithm executes");
  if (selected_index >= 0) {
    test.expect(
        heuristics[static_cast<std::size_t>(selected_index)].workspaceSize ==
            0U,
        "selected cuBLASLt algorithm requires zero workspace");
  }
  if (selected_index < 0) {
    (void)cudaStreamDestroy(stream);
    return 1;
  }

  const auto& selected =
      heuristics[static_cast<std::size_t>(selected_index)].algo;
  ready = launch_lt(test, lt, selected, 1.0F, persistent_weight.get(),
                    activation.get(), candidate_output, workspace.get(),
                    kWorkspaceBytes, stream, "validation reference");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(trusted_output, candidate_output,
                                       kCElements * sizeof(__nv_bfloat16),
                                       cudaMemcpyDeviceToDevice, stream),
                       "copy replay reference");
  ready = ready && launch_lt(test, lt, selected, 1.0F,
                             persistent_weight.get(), activation.get(),
                             candidate_output, workspace.get(), kWorkspaceBytes,
                             stream, "validation replay");
  ready = ready && test.cuda_ok(
                       cudaMemsetAsync(validation.get(), 0,
                                       3U * sizeof(unsigned long long), stream),
                       "zero validation counts");
  validate_bf16_replay_kernel<<<256U, 256U, 0, stream>>>(
      candidate_output, trusted_output, kCElements, validation.get(),
      validation.get() + 1U, validation.get() + 2U);
  ready = ready &&
          test.cuda_ok(cudaGetLastError(), "launch replay validation");
  std::array<unsigned long long, 3U> host_validation{};
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(host_validation.data(), validation.get(),
                                       sizeof(host_validation),
                                       cudaMemcpyDeviceToHost, stream),
                       "copy validation counts");
  ready = ready &&
          test.cuda_ok(cudaStreamSynchronize(stream), "validation sync");
  test.expect(host_validation[0] == 0U, "BF16 output replay is bit exact");
  test.expect(host_validation[1] == 0U, "every BF16 output is finite");
  test.expect(host_validation[2] != 0U, "BF16 encoded checksum is nonzero");
  std::cout << "CUBLASLT_DOWN_C512_VALIDATION: replay_mismatches="
            << host_validation[0] << '/' << kCElements
            << " nonfinite=" << host_validation[1]
            << " encoded_sum=" << host_validation[2]
            << " gate="
            << ((host_validation[0] == 0U && host_validation[1] == 0U &&
                 host_validation[2] != 0U)
                    ? "PASS"
                    : "FAIL")
            << '\n';

  std::vector<double> round_milliseconds;
  round_milliseconds.reserve(kFormalRounds);
  double persistent_median_milliseconds =
      std::numeric_limits<double>::quiet_NaN();
  for (int round = 0; ready && round < kFormalRounds; ++round) {
    const double milliseconds = measure_algorithm(
        test, lt, selected, persistent_weight.get(), activation.get(),
        candidate_output, workspace.get(), kWorkspaceBytes, stream,
        kFormalWarmups, kFormalIterations,
        "formal round " + std::to_string(round + 1));
    ready = ready && std::isfinite(milliseconds) && milliseconds > 0.0;
    test.expect(ready, "formal BF16 timing is finite and positive");
    if (ready) {
      round_milliseconds.push_back(milliseconds);
      std::cout << "CUBLASLT_DOWN_C512_ROUND: round=" << round + 1
                << " iterations=" << kFormalIterations
                << " milliseconds=" << milliseconds
                << " TFLOP_per_s="
                << kUsefulFlops / (milliseconds * 1.0e9) << '\n';
    }
  }

  if (round_milliseconds.size() ==
      static_cast<std::size_t>(kFormalRounds)) {
    persistent_median_milliseconds = median(round_milliseconds);
    const auto [minimum, maximum] =
        std::minmax_element(round_milliseconds.begin(),
                            round_milliseconds.end());
    std::cout << "CUBLASLT_DOWN_C512_FINAL: selected_index=" << selected_index
              << " rounds=" << kFormalRounds
              << " selected_workspace_bytes="
              << heuristics[static_cast<std::size_t>(selected_index)]
                     .workspaceSize
              << " median_milliseconds=" << persistent_median_milliseconds
              << " minimum_milliseconds=" << *minimum
              << " maximum_milliseconds=" << *maximum
              << " median_TFLOP_per_s="
              << kUsefulFlops /
                     (persistent_median_milliseconds * 1.0e9)
              << " comparison_scope=absolute_persistent_BF16_control"
              << " control_dequantization_timed=false gate=PASS\n";
  }

  // Build a canonical production-shaped NVFP4 Down matrix in [N,K] order,
  // decode it through both the trusted scalar route and the optimized direct
  // route, then time the entire transient-dequantization + cuBLASLt chain.
  const auto fill_canonical = [&](DeviceBuffer<std::uint8_t>& buffer,
                                  const std::size_t count,
                                  const std::uint32_t salt,
                                  const bool block_scales,
                                  const std::string& label) {
    const std::size_t blocks =
        (count + kFillThreads - 1U) / kFillThreads;
    fill_canonical_nvfp4_kernel<<<static_cast<unsigned int>(blocks),
                                  kFillThreads, 0, stream>>>(
        buffer.get(), count, salt, block_scales);
    return test.cuda_ok(cudaGetLastError(), label);
  };
  const float nvfp4_weight_scale_2 =
      selected_checkpoint != nullptr ? selected_checkpoint->weight_scale_2
                                     : kWeightScale2;
  if (selected_checkpoint != nullptr) {
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             canonical_packed_weight.get(),
                             selected_checkpoint->packed_weights.data(),
                             kPackedWeightBytes, cudaMemcpyHostToDevice,
                             stream),
                         "upload pinned checkpoint Down weight");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             canonical_block_scale.get(),
                             selected_checkpoint->block_scales.data(),
                             kBlockScaleBytes, cudaMemcpyHostToDevice, stream),
                         "upload pinned checkpoint Down block scale");
  } else {
    ready = ready && fill_canonical(
                         canonical_packed_weight, kPackedWeightBytes,
                         0x6a09'e667U, false,
                         "launch canonical packed NVFP4 fill");
    ready = ready && fill_canonical(
                         canonical_block_scale, kBlockScaleBytes,
                         0xbb67'ae85U, true,
                         "launch canonical E4M3FN scale fill");
  }
  test.expect(select_hybrid_for_scale(nvfp4_weight_scale_2),
              "selected Down weight_scale_2 is finite and positive");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           canonical_packed_weight_snapshot.get(),
                           canonical_packed_weight.get(), kPackedWeightBytes,
                           cudaMemcpyDeviceToDevice, stream),
                       "snapshot canonical packed weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           canonical_block_scale_snapshot.get(),
                           canonical_block_scale.get(), kBlockScaleBytes,
                           cudaMemcpyDeviceToDevice, stream),
                       "snapshot canonical block scales");
  constexpr unsigned int kReferenceBlocks = 65'535U;
  dequantize_nvfp4_contiguous_reference_kernel<<<kReferenceBlocks,
                                                 kFillThreads, 0, stream>>>(
      canonical_packed_weight.get(), canonical_block_scale.get(),
      dequant_reference.get());
  ready = ready && test.cuda_ok(cudaGetLastError(),
                                "launch trusted NVFP4 decoder");
  ready = ready && launch_dequantize_contiguous(
                       test, canonical_packed_weight.get(),
                       canonical_block_scale.get(), persistent_weight.get(),
                       stream, "launch contiguous NVFP4 decoder validation");
  ready = ready && test.cuda_ok(
                       cudaMemsetAsync(validation.get(), 0,
                                       3U * sizeof(unsigned long long), stream),
                       "zero NVFP4 decode validation counts");
  validate_bf16_replay_kernel<<<256U, 256U, 0, stream>>>(
      persistent_weight.get(), dequant_reference.get(), kBElements,
      validation.get(), validation.get() + 1U, validation.get() + 2U);
  ready = ready && test.cuda_ok(cudaGetLastError(),
                                "launch NVFP4 decode validation");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(host_validation.data(), validation.get(),
                                       sizeof(host_validation),
                                       cudaMemcpyDeviceToHost, stream),
                       "copy NVFP4 decode validation counts");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                "NVFP4 decode validation sync");
  test.expect(host_validation[0] == 0U,
              "contiguous NVFP4 decoder is bit exact versus trusted decoder");
  test.expect(host_validation[1] == 0U,
              "every contiguous NVFP4 decoded BF16 value is finite");
  test.expect(host_validation[2] != 0U,
              "contiguous NVFP4 decoded checksum is nonzero");
  std::cout << "NVFP4_DEQUANT_C512_VALIDATION: mismatches="
            << host_validation[0] << '/' << kBElements
            << " nonfinite=" << host_validation[1]
            << " encoded_sum=" << host_validation[2]
            << " comparison=trusted_scalar_exact_bf16"
            << " gate="
            << ((host_validation[0] == 0U && host_validation[1] == 0U &&
                 host_validation[2] != 0U)
                    ? "PASS"
                    : "FAIL")
            << '\n';

  // A second full decode must replay the same trusted BF16 matrix.  This also
  // rules out stale shared-memory or incomplete-grid behavior before timing.
  ready = ready && launch_dequantize_contiguous(
                       test, canonical_packed_weight.get(),
                       canonical_block_scale.get(), persistent_weight.get(),
                       stream, "launch contiguous NVFP4 decoder replay");
  ready = ready && test.cuda_ok(
                       cudaMemsetAsync(validation.get(), 0,
                                       3U * sizeof(unsigned long long), stream),
                       "zero NVFP4 decode replay counts");
  validate_bf16_replay_kernel<<<256U, 256U, 0, stream>>>(
      persistent_weight.get(), dequant_reference.get(), kBElements,
      validation.get(), validation.get() + 1U, validation.get() + 2U);
  ready = ready && test.cuda_ok(cudaGetLastError(),
                                "launch NVFP4 decode replay validation");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(host_validation.data(), validation.get(),
                                       sizeof(host_validation),
                                       cudaMemcpyDeviceToHost, stream),
                       "copy NVFP4 decode replay counts");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                "NVFP4 decode replay sync");
  test.expect(host_validation[0] == 0U,
              "contiguous NVFP4 decoder replay remains bit exact");
  test.expect(host_validation[1] == 0U,
              "every replayed NVFP4 decoded BF16 value is finite");
  std::cout << "NVFP4_DEQUANT_C512_REPLAY: mismatches="
            << host_validation[0] << '/' << kBElements
            << " nonfinite=" << host_validation[1]
            << " encoded_sum=" << host_validation[2]
            << " gate="
            << ((host_validation[0] == 0U && host_validation[1] == 0U)
                    ? "PASS"
                    : "FAIL")
            << '\n';

  // Minimal dequant stop-loss: compare the locked 34-pass prefetch control
  // with a scalar-live sequential loop.  Both eager and replay launches are
  // checked over the complete 89,128,960-element canonical BF16 matrix.
  const auto validate_experimental_dequant_output =
      [&](const __nv_bfloat16* const experimental_output,
          const std::string& variant, const std::string& phase,
          std::array<unsigned long long, 3U>& counts) {
        bool validation_ready = test.cuda_ok(
            cudaMemsetAsync(validation.get(), 0,
                            3U * sizeof(unsigned long long), stream),
            phase + " zero validation counts");
        validate_bf16_replay_kernel<<<256U, 256U, 0, stream>>>(
            experimental_output, dequant_reference.get(), kBElements,
            validation.get(), validation.get() + 1U, validation.get() + 2U);
        validation_ready =
            validation_ready &&
            test.cuda_ok(cudaGetLastError(), phase + " launch validation");
        validation_ready =
            validation_ready &&
            test.cuda_ok(cudaMemcpyAsync(
                             counts.data(), validation.get(), sizeof(counts),
                             cudaMemcpyDeviceToHost, stream),
                         phase + " copy validation counts");
        validation_ready =
            validation_ready &&
            test.cuda_ok(cudaStreamSynchronize(stream),
                         phase + " synchronize validation");
        const bool exact = validation_ready && counts[0] == 0U &&
                           counts[1] == 0U && counts[2] != 0U;
        test.expect(exact,
                    phase + " is bit exact versus trusted scalar decode");
        std::cout << "NVFP4_DEQUANT_DOWN_C512_EXPERIMENTAL_VALIDATION: variant="
                  << variant << " phase=" << phase
                  << " mismatches=" << counts[0] << '/'
                  << kBElements << " nonfinite=" << counts[1]
                  << " encoded_sum=" << counts[2]
                  << " gate=" << (exact ? "PASS" : "FAIL") << '\n';
        return exact;
      };

  ready = ready && launch_dequantize_contiguous_sequential(
                       test, canonical_packed_weight.get(),
                       canonical_block_scale.get(), sequential_dequant_payload,
                       stream, "launch sequential dequant validation");
  std::array<unsigned long long, 3U> sequential_eager_counts{};
  const bool sequential_eager_exact =
      validate_experimental_dequant_output(
          sequential_dequant_payload, "sequential", "eager",
          sequential_eager_counts);
  ready = ready && sequential_eager_exact;

  ready = ready && launch_dequantize_contiguous_sequential(
                       test, canonical_packed_weight.get(),
                       canonical_block_scale.get(), sequential_dequant_payload,
                       stream, "launch sequential dequant replay");
  std::array<unsigned long long, 3U> sequential_replay_counts{};
  const bool sequential_replay_exact =
      validate_experimental_dequant_output(
          sequential_dequant_payload, "sequential", "replay",
          sequential_replay_counts);
  const bool sequential_correctness_gate =
      sequential_eager_exact && sequential_replay_exact &&
      sequential_eager_counts[2] == sequential_replay_counts[2];
  test.expect(sequential_correctness_gate,
              "sequential eager and replay checksums are identical");
  ready = ready && sequential_correctness_gate;

  constexpr int kSequentialScreenWarmups = 10;
  constexpr int kSequentialScreenIterations = 24;
  constexpr int kSequentialScreenRounds = 6;
  constexpr double kSequentialRequiredSpeedup = 1.03;
  std::vector<double> sequential_baseline_rounds;
  std::vector<double> sequential_candidate_rounds;
  sequential_baseline_rounds.reserve(kSequentialScreenRounds);
  sequential_candidate_rounds.reserve(kSequentialScreenRounds);
  for (int round = 0; ready && round < kSequentialScreenRounds; ++round) {
    double baseline_milliseconds = 0.0;
    double candidate_milliseconds = 0.0;
    if ((round & 1) == 0) {
      baseline_milliseconds = measure_dequantize(
          test, canonical_packed_weight.get(), canonical_block_scale.get(),
          persistent_weight.get(), stream, kSequentialScreenWarmups,
          kSequentialScreenIterations,
          "sequential screen baseline round " + std::to_string(round + 1));
      candidate_milliseconds = measure_dequantize_sequential(
          test, canonical_packed_weight.get(), canonical_block_scale.get(),
          sequential_dequant_payload, stream, kSequentialScreenWarmups,
          kSequentialScreenIterations,
          "sequential screen candidate round " + std::to_string(round + 1));
    } else {
      candidate_milliseconds = measure_dequantize_sequential(
          test, canonical_packed_weight.get(), canonical_block_scale.get(),
          sequential_dequant_payload, stream, kSequentialScreenWarmups,
          kSequentialScreenIterations,
          "sequential screen candidate round " + std::to_string(round + 1));
      baseline_milliseconds = measure_dequantize(
          test, canonical_packed_weight.get(), canonical_block_scale.get(),
          persistent_weight.get(), stream, kSequentialScreenWarmups,
          kSequentialScreenIterations,
          "sequential screen baseline round " + std::to_string(round + 1));
    }
    const bool timing_ok = std::isfinite(baseline_milliseconds) &&
                           baseline_milliseconds > 0.0 &&
                           std::isfinite(candidate_milliseconds) &&
                           candidate_milliseconds > 0.0;
    test.expect(timing_ok,
                "sequential dequant screen timing is finite and positive");
    ready = ready && timing_ok;
    if (timing_ok) {
      sequential_baseline_rounds.push_back(baseline_milliseconds);
      sequential_candidate_rounds.push_back(candidate_milliseconds);
      std::cout << "NVFP4_DEQUANT_DOWN_C512_SEQUENTIAL_ROUND: round="
                << round + 1 << " order="
                << (((round & 1) == 0) ? "B-C" : "C-B")
                << " warmups=" << kSequentialScreenWarmups
                << " iterations=" << kSequentialScreenIterations
                << " baseline_milliseconds=" << baseline_milliseconds
                << " candidate_milliseconds=" << candidate_milliseconds
                << " speedup="
                << baseline_milliseconds / candidate_milliseconds << '\n';
    }
  }

  unsigned long long sequential_guard_mismatches = 0U;
  bool sequential_guard_gate = test.cuda_ok(
      cudaMemsetAsync(validation.get(), 0, sizeof(unsigned long long), stream),
      "sequential dequant zero guard count");
  validate_bf16_guards_kernel<<<1U, 256U, 0, stream>>>(
      sequential_dequant_output.get(), kBElements, kGuardElements, kGuardBits,
      validation.get());
  sequential_guard_gate =
      sequential_guard_gate &&
      test.cuda_ok(cudaGetLastError(),
                   "launch sequential dequant guard validation");
  sequential_guard_gate =
      sequential_guard_gate &&
      test.cuda_ok(cudaMemcpyAsync(
                       &sequential_guard_mismatches, validation.get(),
                       sizeof(sequential_guard_mismatches),
                       cudaMemcpyDeviceToHost, stream),
                   "copy sequential dequant guard count");
  sequential_guard_gate =
      sequential_guard_gate &&
      test.cuda_ok(cudaStreamSynchronize(stream),
                   "synchronize sequential dequant guard validation") &&
      sequential_guard_mismatches == 0U;
  test.expect(sequential_guard_gate,
              "sequential dequant prefix/suffix guards are intact");
  ready = ready && sequential_guard_gate;
  std::cout << "NVFP4_DEQUANT_DOWN_C512_EXPERIMENTAL_GUARDS: variant="
            << "sequential mismatches=" << sequential_guard_mismatches << '/'
            << 2U * kGuardElements
            << " gate=" << (sequential_guard_gate ? "PASS" : "FAIL")
            << '\n';

  bool sequential_selected = false;
  if (sequential_baseline_rounds.size() ==
          static_cast<std::size_t>(kSequentialScreenRounds) &&
      sequential_candidate_rounds.size() ==
          static_cast<std::size_t>(kSequentialScreenRounds)) {
    const double baseline_median = median(sequential_baseline_rounds);
    const double candidate_median = median(sequential_candidate_rounds);
    const double median_speedup = baseline_median / candidate_median;
    double minimum_round_speedup = std::numeric_limits<double>::infinity();
    bool all_round_positive = true;
    for (std::size_t round = 0U;
         round < sequential_baseline_rounds.size(); ++round) {
      const double speedup =
          sequential_baseline_rounds[round] /
          sequential_candidate_rounds[round];
      minimum_round_speedup = std::min(minimum_round_speedup, speedup);
      all_round_positive = all_round_positive && speedup > 1.0;
    }
    const bool local_memory_gate =
        sequential_dequant_attributes.localSizeBytes == 0U;
    sequential_selected =
        sequential_correctness_gate && sequential_guard_gate &&
        local_memory_gate && all_round_positive &&
        median_speedup >= kSequentialRequiredSpeedup;
    std::cout << "NVFP4_DEQUANT_DOWN_C512_SEQUENTIAL_FINAL: rounds="
              << kSequentialScreenRounds
              << " baseline_median_milliseconds=" << baseline_median
              << " candidate_median_milliseconds=" << candidate_median
              << " median_speedup=" << median_speedup
              << " minimum_round_speedup=" << minimum_round_speedup
              << " all_round_positive="
              << (all_round_positive ? "true" : "false")
              << " required_speedup=" << kSequentialRequiredSpeedup
              << " registers_per_thread="
              << sequential_dequant_attributes.numRegs
              << " local_bytes_per_thread="
              << sequential_dequant_attributes.localSizeBytes
              << " active_blocks_per_sm="
              << active_sequential_dequant_blocks_per_sm
              << " correctness="
              << (sequential_correctness_gate ? "true" : "false")
              << " guards_intact="
              << (sequential_guard_gate ? "true" : "false")
              << " action="
              << (performance_checkpoint
                      ? (sequential_selected ? "SELECT" : "REJECT")
                      : "NOT_EVALUATED")
              << " gate="
              << (performance_checkpoint
                      ? (sequential_selected ? "PASS" : "FAIL")
                      : "NOT_RUN")
              << " admission="
              << (performance_checkpoint ? "DIAGNOSTIC_SUBGATE"
                                         : "NOT_RUN")
              << '\n';
  }

  ready = ready && launch_dequantize_contiguous_window8(
                       test, canonical_packed_weight.get(),
                       canonical_block_scale.get(), window8_dequant_payload,
                       stream, "launch window8 dequant validation");
  std::array<unsigned long long, 3U> window8_eager_counts{};
  const bool window8_eager_exact =
      validate_experimental_dequant_output(
          window8_dequant_payload, "window8", "eager", window8_eager_counts);
  ready = ready && window8_eager_exact;

  ready = ready && launch_dequantize_contiguous_window8(
                       test, canonical_packed_weight.get(),
                       canonical_block_scale.get(), window8_dequant_payload,
                       stream, "launch window8 dequant replay");
  std::array<unsigned long long, 3U> window8_replay_counts{};
  const bool window8_replay_exact =
      validate_experimental_dequant_output(
          window8_dequant_payload, "window8", "replay",
          window8_replay_counts);
  const bool window8_correctness_gate =
      window8_eager_exact && window8_replay_exact &&
      window8_eager_counts[2] == window8_replay_counts[2];
  test.expect(window8_correctness_gate,
              "window8 eager and replay checksums are identical");
  ready = ready && window8_correctness_gate;

  constexpr int kWindow8ScreenWarmups = 10;
  constexpr int kWindow8ScreenIterations = 24;
  constexpr int kWindow8ScreenRounds = 6;
  constexpr double kWindow8RequiredSpeedup = 1.03;
  std::vector<double> window8_baseline_rounds;
  std::vector<double> window8_candidate_rounds;
  window8_baseline_rounds.reserve(kWindow8ScreenRounds);
  window8_candidate_rounds.reserve(kWindow8ScreenRounds);
  for (int round = 0; ready && round < kWindow8ScreenRounds; ++round) {
    double baseline_milliseconds = 0.0;
    double candidate_milliseconds = 0.0;
    if ((round & 1) == 0) {
      baseline_milliseconds = measure_dequantize(
          test, canonical_packed_weight.get(), canonical_block_scale.get(),
          persistent_weight.get(), stream, kWindow8ScreenWarmups,
          kWindow8ScreenIterations,
          "window8 screen baseline round " + std::to_string(round + 1));
      candidate_milliseconds = measure_dequantize_window8(
          test, canonical_packed_weight.get(), canonical_block_scale.get(),
          window8_dequant_payload, stream, kWindow8ScreenWarmups,
          kWindow8ScreenIterations,
          "window8 screen candidate round " + std::to_string(round + 1));
    } else {
      candidate_milliseconds = measure_dequantize_window8(
          test, canonical_packed_weight.get(), canonical_block_scale.get(),
          window8_dequant_payload, stream, kWindow8ScreenWarmups,
          kWindow8ScreenIterations,
          "window8 screen candidate round " + std::to_string(round + 1));
      baseline_milliseconds = measure_dequantize(
          test, canonical_packed_weight.get(), canonical_block_scale.get(),
          persistent_weight.get(), stream, kWindow8ScreenWarmups,
          kWindow8ScreenIterations,
          "window8 screen baseline round " + std::to_string(round + 1));
    }
    const bool timing_ok = std::isfinite(baseline_milliseconds) &&
                           baseline_milliseconds > 0.0 &&
                           std::isfinite(candidate_milliseconds) &&
                           candidate_milliseconds > 0.0;
    test.expect(timing_ok,
                "window8 dequant screen timing is finite and positive");
    ready = ready && timing_ok;
    if (timing_ok) {
      window8_baseline_rounds.push_back(baseline_milliseconds);
      window8_candidate_rounds.push_back(candidate_milliseconds);
      std::cout << "NVFP4_DEQUANT_DOWN_C512_WINDOW8_ROUND: round="
                << round + 1 << " order="
                << (((round & 1) == 0) ? "B-C" : "C-B")
                << " warmups=" << kWindow8ScreenWarmups
                << " iterations=" << kWindow8ScreenIterations
                << " baseline_milliseconds=" << baseline_milliseconds
                << " candidate_milliseconds=" << candidate_milliseconds
                << " speedup="
                << baseline_milliseconds / candidate_milliseconds << '\n';
    }
  }

  unsigned long long window8_guard_mismatches = 0U;
  bool window8_guard_gate = test.cuda_ok(
      cudaMemsetAsync(validation.get(), 0, sizeof(unsigned long long), stream),
      "window8 dequant zero guard count");
  validate_bf16_guards_kernel<<<1U, 256U, 0, stream>>>(
      window8_dequant_output.get(), kBElements, kGuardElements, kGuardBits,
      validation.get());
  window8_guard_gate =
      window8_guard_gate &&
      test.cuda_ok(cudaGetLastError(),
                   "launch window8 dequant guard validation");
  window8_guard_gate =
      window8_guard_gate &&
      test.cuda_ok(cudaMemcpyAsync(
                       &window8_guard_mismatches, validation.get(),
                       sizeof(window8_guard_mismatches), cudaMemcpyDeviceToHost,
                       stream),
                   "copy window8 dequant guard count");
  window8_guard_gate =
      window8_guard_gate &&
      test.cuda_ok(cudaStreamSynchronize(stream),
                   "synchronize window8 dequant guard validation") &&
      window8_guard_mismatches == 0U;
  test.expect(window8_guard_gate,
              "window8 dequant prefix/suffix guards are intact");
  ready = ready && window8_guard_gate;
  std::cout << "NVFP4_DEQUANT_DOWN_C512_EXPERIMENTAL_GUARDS: variant=window8"
            << " mismatches=" << window8_guard_mismatches << '/'
            << 2U * kGuardElements
            << " gate=" << (window8_guard_gate ? "PASS" : "FAIL") << '\n';

  bool window8_selected = false;
  if (window8_baseline_rounds.size() ==
          static_cast<std::size_t>(kWindow8ScreenRounds) &&
      window8_candidate_rounds.size() ==
          static_cast<std::size_t>(kWindow8ScreenRounds)) {
    const double baseline_median = median(window8_baseline_rounds);
    const double candidate_median = median(window8_candidate_rounds);
    const double median_speedup = baseline_median / candidate_median;
    double minimum_round_speedup = std::numeric_limits<double>::infinity();
    bool all_round_positive = true;
    for (std::size_t round = 0U; round < window8_baseline_rounds.size();
         ++round) {
      const double speedup =
          window8_baseline_rounds[round] / window8_candidate_rounds[round];
      minimum_round_speedup = std::min(minimum_round_speedup, speedup);
      all_round_positive = all_round_positive && speedup > 1.0;
    }
    const bool local_memory_gate =
        window8_dequant_attributes.localSizeBytes == 0U;
    window8_selected =
        window8_correctness_gate && window8_guard_gate && local_memory_gate &&
        all_round_positive && median_speedup >= kWindow8RequiredSpeedup;
    std::cout << "NVFP4_DEQUANT_DOWN_C512_WINDOW8_FINAL: rounds="
              << kWindow8ScreenRounds
              << " baseline_median_milliseconds=" << baseline_median
              << " candidate_median_milliseconds=" << candidate_median
              << " median_speedup=" << median_speedup
              << " minimum_round_speedup=" << minimum_round_speedup
              << " all_round_positive="
              << (all_round_positive ? "true" : "false")
              << " required_speedup=" << kWindow8RequiredSpeedup
              << " registers_per_thread=" << window8_dequant_attributes.numRegs
              << " local_bytes_per_thread="
              << window8_dequant_attributes.localSizeBytes
              << " active_blocks_per_sm="
              << active_window8_dequant_blocks_per_sm
              << " correctness="
              << (window8_correctness_gate ? "true" : "false")
              << " guards_intact="
              << (window8_guard_gate ? "true" : "false")
              << " action="
              << (performance_checkpoint
                      ? (window8_selected ? "SELECT" : "REJECT")
                      : "NOT_EVALUATED")
              << " gate="
              << (performance_checkpoint
                      ? (window8_selected ? "PASS" : "FAIL")
                      : "NOT_RUN")
              << " admission="
              << (performance_checkpoint ? "DIAGNOSTIC_SUBGATE"
                                         : "NOT_RUN")
              << '\n';
  }

  // Validate the exact timed chain against the trusted scalar-dequantized
  // matrix.  Production applies the non-unit global weight scale as Lt alpha;
  // it is deliberately not rounded into the transient BF16 weights.
  ready = ready && launch_dequantize_contiguous_window8(
                       test, canonical_packed_weight.get(),
                       canonical_block_scale.get(), persistent_weight.get(),
                       stream,
                       "launch inclusive window8 candidate dequant");
  ready = ready && launch_lt(
                       test, lt, selected, nvfp4_weight_scale_2,
                       persistent_weight.get(), activation.get(),
                       candidate_output,
                       workspace.get(), kWorkspaceBytes, stream,
                       "launch inclusive candidate GEMM");
  ready = ready && launch_lt(
                       test, lt, selected, nvfp4_weight_scale_2,
                       dequant_reference.get(), activation.get(),
                       trusted_output, workspace.get(),
                       kWorkspaceBytes, stream,
                       "launch trusted-dequant inclusive reference GEMM");
  ready = ready && test.cuda_ok(
                       cudaMemsetAsync(validation.get(), 0,
                                       3U * sizeof(unsigned long long), stream),
                       "zero inclusive validation counts");
  validate_bf16_replay_kernel<<<256U, 256U, 0, stream>>>(
      candidate_output, trusted_output, kCElements, validation.get(),
      validation.get() + 1U, validation.get() + 2U);
  ready = ready && test.cuda_ok(cudaGetLastError(),
                                "launch inclusive replay validation");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(host_validation.data(), validation.get(),
                                       sizeof(host_validation),
                                       cudaMemcpyDeviceToHost, stream),
                       "copy inclusive validation counts");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                "inclusive validation sync");
  test.expect(host_validation[0] == 0U,
              "inclusive output matches trusted scalar dequantization");
  test.expect(host_validation[1] == 0U,
              "every inclusive NVFP4 + cuBLASLt output is finite");
  test.expect(host_validation[2] != 0U,
              "inclusive NVFP4 + cuBLASLt checksum is nonzero");
  std::cout << "NVFP4_CUBLASLT_DOWN_C512_VALIDATION: reference_mismatches="
            << host_validation[0] << '/' << kCElements
            << " selected_dequant=window8"
            << " nonfinite=" << host_validation[1]
            << " encoded_sum=" << host_validation[2]
            << " weight_scale_2=" << nvfp4_weight_scale_2
            << " scale_application=cuBLASLt_alpha_after_BF16_dequant"
            << " production_M128_bitwise_compared=deferred"
            << " gate="
            << ((host_validation[0] == 0U && host_validation[1] == 0U &&
                 host_validation[2] != 0U)
                    ? "PASS"
                    : "FAIL")
            << '\n';

  // The production module must remain Graph-safe even though the enclosing
  // production Prefill loop is currently eager. Capture the exact
  // dequantize + selected-Lt chain, instantiate it, then require both the
  // first (cold) and second (warm) graph replays to match the trusted eager
  // output bit for bit.
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  bool graph_instantiated = false;
  bool graph_ready = test.cuda_ok(
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
      "begin NVFP4 inclusive graph capture");
  if (graph_ready) {
    const bool captured_dequant = launch_dequantize_contiguous_window8(
        test, canonical_packed_weight.get(), canonical_block_scale.get(),
        persistent_weight.get(), stream, "capture window8 NVFP4 dequant");
    const bool captured_lt =
        captured_dequant &&
        launch_lt(test, lt, selected, nvfp4_weight_scale_2,
                  persistent_weight.get(), activation.get(), candidate_output,
                  workspace.get(), kWorkspaceBytes, stream,
                  "capture selected cuBLASLt GEMM");
    graph_ready = captured_dequant && captured_lt;
    const cudaError_t end_capture_status =
        cudaStreamEndCapture(stream, &graph);
    graph_ready = test.cuda_ok(end_capture_status,
                               "end NVFP4 inclusive graph capture") &&
                  graph_ready;
  }

  std::size_t graph_node_count = 0U;
  if (graph_ready) {
    graph_ready = test.cuda_ok(
        cudaGraphGetNodes(graph, nullptr, &graph_node_count),
        "query NVFP4 inclusive graph nodes");
    test.expect(graph_node_count == 2U,
                "inclusive graph contains exactly dequant and GEMM nodes");
    graph_ready = graph_ready && graph_node_count == 2U;
  }
  if (graph_ready) {
    graph_ready = test.cuda_ok(
        cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0U),
        "instantiate NVFP4 inclusive graph");
    graph_instantiated = graph_ready && graph_exec != nullptr;
  }

  const auto validate_graph_replay =
      [&](const std::string& label,
          std::array<unsigned long long, 3U>& graph_validation) {
        bool replay_ready = test.cuda_ok(
            cudaGraphLaunch(graph_exec, stream), label + " launch");
        replay_ready =
            replay_ready &&
            test.cuda_ok(cudaMemsetAsync(
                             validation.get(), 0,
                             3U * sizeof(unsigned long long), stream),
                         label + " zero validation counts");
        validate_bf16_replay_kernel<<<256U, 256U, 0, stream>>>(
            candidate_output, trusted_output, kCElements,
            validation.get(), validation.get() + 1U, validation.get() + 2U);
        replay_ready =
            replay_ready &&
            test.cuda_ok(cudaGetLastError(), label + " validate output");
        replay_ready =
            replay_ready &&
            test.cuda_ok(cudaMemcpyAsync(
                             graph_validation.data(), validation.get(),
                             sizeof(graph_validation), cudaMemcpyDeviceToHost,
                             stream),
                         label + " copy validation counts");
        replay_ready = replay_ready && test.cuda_ok(
                                           cudaStreamSynchronize(stream),
                                           label + " synchronize");
        return replay_ready;
      };

  std::array<unsigned long long, 3U> cold_graph_validation{};
  std::array<unsigned long long, 3U> warm_graph_validation{};
  if (graph_ready) {
    graph_ready = validate_graph_replay("cold graph replay",
                                        cold_graph_validation);
  }
  if (graph_ready) {
    graph_ready = validate_graph_replay("warm graph replay",
                                        warm_graph_validation);
  }
  if (graph_exec != nullptr) {
    graph_ready = test.cuda_ok(cudaGraphExecDestroy(graph_exec),
                               "destroy NVFP4 inclusive graph executable") &&
                  graph_ready;
  }
  if (graph != nullptr) {
    graph_ready = test.cuda_ok(cudaGraphDestroy(graph),
                               "destroy NVFP4 inclusive graph") &&
                  graph_ready;
  }
  ready = ready && graph_ready;
  test.expect(cold_graph_validation[0] == 0U,
              "cold graph output matches trusted eager output");
  test.expect(cold_graph_validation[1] == 0U,
              "cold graph output is finite");
  test.expect(cold_graph_validation[2] != 0U,
              "cold graph output checksum is nonzero");
  test.expect(warm_graph_validation[0] == 0U,
              "warm graph output matches trusted eager output");
  test.expect(warm_graph_validation[1] == 0U,
              "warm graph output is finite");
  test.expect(warm_graph_validation[2] == cold_graph_validation[2],
              "cold and warm graph checksums are identical");
  std::cout << "NVFP4_CUBLASLT_DOWN_C512_GRAPH: nodes="
            << graph_node_count
            << " selected_dequant=window8"
            << " instantiated=" << (graph_instantiated ? "true" : "false")
            << " cold_reference_mismatches=" << cold_graph_validation[0]
            << '/' << kCElements
            << " cold_nonfinite=" << cold_graph_validation[1]
            << " cold_encoded_sum=" << cold_graph_validation[2]
            << " warm_reference_mismatches=" << warm_graph_validation[0]
            << '/' << kCElements
            << " warm_nonfinite=" << warm_graph_validation[1]
            << " warm_encoded_sum=" << warm_graph_validation[2]
            << " weight_scale_2=" << nvfp4_weight_scale_2
            << " production_M128_bitwise_compared=deferred"
            << " gate="
            << ((graph_ready && cold_graph_validation[0] == 0U &&
                 cold_graph_validation[1] == 0U &&
                 cold_graph_validation[2] != 0U &&
                 warm_graph_validation[0] == 0U &&
                 warm_graph_validation[1] == 0U &&
                 warm_graph_validation[2] == cold_graph_validation[2])
                    ? "PASS"
                    : "FAIL")
            << '\n';

  // P0 production equivalence: the candidate and the public exact-C512 M128
  // dispatcher consume the same canonical weights, scales, activations, and
  // non-unit global scale.  Different tensor-core reduction orders are
  // allowed, so bitwise mismatch is reported while finite/NRMSE/cosine form
  // the numerical admission gate.
  const int production_status = q3x::kernels::
      launch_sm87_nvfp4_w4a16_whole_chunk_down_gemm_bf16_cuda(
          canonical_packed_weight.get(), canonical_block_scale.get(),
          nvfp4_weight_scale_2,
          reinterpret_cast<const std::uint16_t*>(activation.get()), kM, kN,
          kK, reinterpret_cast<std::uint16_t*>(exact_production_output),
          static_cast<void*>(stream));
  ready = ready && test.cuda_ok(static_cast<cudaError_t>(production_status),
                                "launch production exact-C512 M128 Down");
  ready = ready && test.cuda_ok(
                       cudaStreamSynchronize(stream),
                       "synchronize production exact-C512 M128 Down");

  std::vector<std::uint16_t> candidate_host(kCElements);
  std::vector<std::uint16_t> production_host(kCElements);
  ready = ready && test.cuda_ok(
                       cudaMemcpy(candidate_host.data(), candidate_output,
                                  kCElements * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),
                       "copy candidate C512 output");
  ready = ready && test.cuda_ok(
                       cudaMemcpy(production_host.data(),
                                  exact_production_output,
                                  kCElements * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),
                       "copy production M128 C512 output");
  const NumericalMetrics production_metrics =
      compare_bf16_outputs(candidate_host, production_host);
  const bool production_numerical_gate =
      production_metrics.candidate_nonfinite == 0U &&
      production_metrics.production_nonfinite == 0U &&
      production_metrics.nrmse <= kMaximumProductionNrmse &&
      production_metrics.cosine >= kMinimumProductionCosine;
  test.expect(production_numerical_gate,
              "hybrid output passes production M128 numerical gate");
  std::cout << "NVFP4_CUBLASLT_DOWN_C512_PRODUCTION_EQUIVALENCE: status="
            << production_status
            << " selected_dequant=window8"
            << " bitwise_mismatches="
            << production_metrics.bitwise_mismatches << '/' << kCElements
            << " mismatch_fraction="
            << static_cast<double>(production_metrics.bitwise_mismatches) /
                   static_cast<double>(kCElements)
            << " candidate_nonfinite="
            << production_metrics.candidate_nonfinite
            << " production_nonfinite="
            << production_metrics.production_nonfinite
            << " max_abs=" << production_metrics.maximum_absolute
            << " max_rel_floor_1e-2="
            << production_metrics.maximum_relative
            << " nrmse=" << production_metrics.nrmse
            << " cosine=" << production_metrics.cosine
            << " max_nrmse=" << kMaximumProductionNrmse
            << " min_cosine=" << kMinimumProductionCosine
            << " bitwise_required=false"
            << " gate="
            << (production_numerical_gate ? "PASS" : "FAIL") << '\n';

  // The ceiling's local LtObjects selection is not production evidence by
  // itself.  For a pinned checkpoint, launch the separately compiled public
  // module over the exact same operands and require its Window8 scratch and
  // BF16 output to agree bit-for-bit with the trusted decoder, this TU's
  // inclusive chain, and the existing production M128 kernel.
  q3x::kernels::Sm87Nvfp4PrefillDownCublasLtContext*
      public_down_context = nullptr;
  std::size_t public_scratch_bytes = 0U;
  std::size_t public_workspace_bytes = std::numeric_limits<std::size_t>::max();
  int public_heuristic_rank = -1;
  bool public_module_evidence_gate = true;
  if (selected_checkpoint != nullptr) {
    const int public_create_status = q3x::kernels::
        create_sm87_nvfp4_prefill_down_cublaslt_context(&public_down_context);
    bool public_ready =
        test.cuda_ok(static_cast<cudaError_t>(public_create_status),
                     "create public production Down context") &&
        public_down_context != nullptr;
    test.expect(public_down_context != nullptr,
                "public production Down context is non-null");
    int public_query_status = static_cast<int>(cudaErrorInvalidValue);
    if (public_ready) {
      public_query_status = q3x::kernels::
          query_sm87_nvfp4_prefill_down_cublaslt_context(
              public_down_context, &public_scratch_bytes,
              &public_workspace_bytes, &public_heuristic_rank);
      public_ready =
          test.cuda_ok(static_cast<cudaError_t>(public_query_status),
                       "query public production Down context") &&
          public_ready;
    }
    const bool public_contract_exact =
        public_ready &&
        public_scratch_bytes == kBElements * sizeof(__nv_bfloat16) &&
        public_workspace_bytes == 0U && public_heuristic_rank >= 0;
    test.expect(public_contract_exact,
                "public Down context reports exact scratch and zero workspace");
    public_ready = public_ready && public_contract_exact;

    if (public_ready) {
      public_ready = launch_public_down_module(
                         test, public_down_context,
                         canonical_packed_weight.get(),
                         canonical_block_scale.get(), activation.get(),
                         persistent_weight.get(), trusted_output,
                         nvfp4_weight_scale_2, stream,
                         "launch public production Down module") &&
                     public_ready;
      public_ready =
          test.cuda_ok(cudaMemsetAsync(validation.get(), 0,
                                       3U * sizeof(unsigned long long), stream),
                       "zero public module scratch validation") &&
          public_ready;
      validate_bf16_replay_kernel<<<256U, 256U, 0, stream>>>(
          persistent_weight.get(), dequant_reference.get(), kBElements,
          validation.get(), validation.get() + 1U, validation.get() + 2U);
      public_ready =
          test.cuda_ok(cudaGetLastError(),
                       "launch public module scratch validation") &&
          public_ready;
    }

    std::array<unsigned long long, 3U> public_scratch_validation{};
    if (public_ready) {
      public_ready =
          test.cuda_ok(cudaMemcpyAsync(
                           public_scratch_validation.data(), validation.get(),
                           sizeof(public_scratch_validation),
                           cudaMemcpyDeviceToHost, stream),
                       "copy public module scratch validation") &&
          public_ready;
      public_ready =
          test.cuda_ok(cudaStreamSynchronize(stream),
                       "synchronize public production Down module") &&
          public_ready;
    }

    std::vector<std::uint16_t> public_module_host(kCElements);
    if (public_ready) {
      public_ready =
          test.cuda_ok(cudaMemcpy(public_module_host.data(), trusted_output,
                                  kCElements * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),
                       "copy public production Down output") &&
          public_ready;
    }
    const NumericalMetrics public_vs_internal =
        compare_bf16_outputs(public_module_host, candidate_host);
    const NumericalMetrics public_vs_production =
        compare_bf16_outputs(public_module_host, production_host);
    public_module_evidence_gate =
        public_ready && public_scratch_validation[0] == 0U &&
        public_scratch_validation[1] == 0U &&
        public_scratch_validation[2] != 0U &&
        public_vs_internal.bitwise_mismatches == 0U &&
        public_vs_internal.candidate_nonfinite == 0U &&
        public_vs_internal.production_nonfinite == 0U &&
        public_vs_production.bitwise_mismatches == 0U &&
        public_vs_production.candidate_nonfinite == 0U &&
        public_vs_production.production_nonfinite == 0U;
    test.expect(public_module_evidence_gate,
                "public Down module is bitwise exact on pinned checkpoint");
    ready = ready && public_module_evidence_gate;
    std::cout << "NVFP4_CUBLASLT_DOWN_C512_PUBLIC_MODULE_EXACT:"
              << " create_status=" << public_create_status
              << " query_status=" << public_query_status
              << " heuristic_rank=" << public_heuristic_rank
              << " scratch_bytes=" << public_scratch_bytes
              << " workspace_bytes=" << public_workspace_bytes
              << " scratch_reference_mismatches="
              << public_scratch_validation[0] << '/' << kBElements
              << " scratch_nonfinite=" << public_scratch_validation[1]
              << " scratch_encoded_sum=" << public_scratch_validation[2]
              << " output_vs_test_TU_mismatches="
              << public_vs_internal.bitwise_mismatches << '/' << kCElements
              << " output_vs_production_M128_mismatches="
              << public_vs_production.bitwise_mismatches << '/' << kCElements
              << " checkpoint=true gate="
              << (public_module_evidence_gate ? "PASS" : "FAIL") << '\n';
  }

  std::vector<double> dequant_round_milliseconds;
  std::vector<double> inclusive_round_milliseconds;
  std::vector<double> production_round_milliseconds;
  dequant_round_milliseconds.reserve(kFormalRounds);
  inclusive_round_milliseconds.reserve(kFormalRounds);
  production_round_milliseconds.reserve(kFormalRounds);
  for (int round = 0; ready && round < kFormalRounds; ++round) {
    double dequant_milliseconds = 0.0;
    double inclusive_milliseconds = 0.0;
    double production_milliseconds = 0.0;
    // Alternate order across rounds to keep either measurement from always
    // receiving the first thermal/frequency position.
    if ((round & 1) == 0) {
      production_milliseconds = measure_production_down(
          test, canonical_packed_weight.get(), canonical_block_scale.get(),
          activation.get(), exact_production_output, nvfp4_weight_scale_2,
          stream,
          kFormalWarmups, kFormalIterations,
          "production M128 round " + std::to_string(round + 1));
      dequant_milliseconds = measure_dequantize_window8(
          test, canonical_packed_weight.get(), canonical_block_scale.get(),
          persistent_weight.get(), stream, kFormalWarmups, kFormalIterations,
          "NVFP4 window8 dequant round " + std::to_string(round + 1));
      inclusive_milliseconds = measure_inclusive_window8(
          test, lt, selected, canonical_packed_weight.get(),
          canonical_block_scale.get(), persistent_weight.get(),
          activation.get(), candidate_output, workspace.get(), kWorkspaceBytes,
          stream, nvfp4_weight_scale_2, kFormalWarmups, kFormalIterations,
          "NVFP4 inclusive round " + std::to_string(round + 1));
    } else {
      inclusive_milliseconds = measure_inclusive_window8(
          test, lt, selected, canonical_packed_weight.get(),
          canonical_block_scale.get(), persistent_weight.get(),
          activation.get(), candidate_output, workspace.get(), kWorkspaceBytes,
          stream, nvfp4_weight_scale_2, kFormalWarmups, kFormalIterations,
          "NVFP4 inclusive round " + std::to_string(round + 1));
      dequant_milliseconds = measure_dequantize_window8(
          test, canonical_packed_weight.get(), canonical_block_scale.get(),
          persistent_weight.get(), stream, kFormalWarmups, kFormalIterations,
          "NVFP4 window8 dequant round " + std::to_string(round + 1));
      production_milliseconds = measure_production_down(
          test, canonical_packed_weight.get(), canonical_block_scale.get(),
          activation.get(), exact_production_output, nvfp4_weight_scale_2,
          stream,
          kFormalWarmups, kFormalIterations,
          "production M128 round " + std::to_string(round + 1));
    }
    const bool timing_ok = std::isfinite(dequant_milliseconds) &&
                           dequant_milliseconds > 0.0 &&
                           std::isfinite(inclusive_milliseconds) &&
                           inclusive_milliseconds > 0.0 &&
                           std::isfinite(production_milliseconds) &&
                           production_milliseconds > 0.0;
    ready = ready && timing_ok;
    test.expect(timing_ok,
                "inclusive NVFP4 timing is finite and positive");
    if (ready) {
      dequant_round_milliseconds.push_back(dequant_milliseconds);
      inclusive_round_milliseconds.push_back(inclusive_milliseconds);
      production_round_milliseconds.push_back(production_milliseconds);
      std::cout << "NVFP4_CUBLASLT_DOWN_C512_ROUND: round=" << round + 1
                << " iterations=" << kFormalIterations
                << " selected_dequant=window8"
                << " production_M128_milliseconds="
                << production_milliseconds
                << " dequant_milliseconds=" << dequant_milliseconds
                << " persistent_cuBLASLt_median_milliseconds="
                << persistent_median_milliseconds
                << " inclusive_milliseconds=" << inclusive_milliseconds
                << " inclusive_speedup_vs_fresh_production_M128="
                << production_milliseconds / inclusive_milliseconds
                << '\n';
    }
  }

  bool inclusive_speed_gate = false;
  if (inclusive_round_milliseconds.size() ==
          static_cast<std::size_t>(kFormalRounds) &&
      dequant_round_milliseconds.size() ==
          static_cast<std::size_t>(kFormalRounds) &&
      production_round_milliseconds.size() ==
          static_cast<std::size_t>(kFormalRounds)) {
    const double dequant_median = median(dequant_round_milliseconds);
    const double inclusive_median = median(inclusive_round_milliseconds);
    const double production_median = median(production_round_milliseconds);
    const auto [dequant_minimum, dequant_maximum] =
        std::minmax_element(dequant_round_milliseconds.begin(),
                            dequant_round_milliseconds.end());
    const auto [inclusive_minimum, inclusive_maximum] =
        std::minmax_element(inclusive_round_milliseconds.begin(),
                            inclusive_round_milliseconds.end());
    const auto [production_minimum, production_maximum] =
        std::minmax_element(production_round_milliseconds.begin(),
                            production_round_milliseconds.end());
    const double inclusive_speedup = production_median / inclusive_median;
    double minimum_round_speedup = std::numeric_limits<double>::infinity();
    bool all_round_positive = true;
    for (std::size_t round = 0U;
         round < production_round_milliseconds.size(); ++round) {
      const double round_speedup =
          production_round_milliseconds[round] /
          inclusive_round_milliseconds[round];
      minimum_round_speedup = std::min(minimum_round_speedup, round_speedup);
      all_round_positive = all_round_positive && round_speedup > 1.0;
    }
    constexpr double kMinimumDequantBytes =
        static_cast<double>(kPackedWeightBytes + kBlockScaleBytes +
                            kBElements * sizeof(__nv_bfloat16));
    const double dequant_effective_gigabytes_per_second =
        kMinimumDequantBytes / (dequant_median * 1.0e6);
    inclusive_speed_gate =
        inclusive_speedup >= kRequiredInclusiveSpeedup && all_round_positive;
    std::cout << "NVFP4_CUBLASLT_DOWN_C512_FINAL: selected_index="
              << selected_index << " rounds=" << kFormalRounds
              << " selected_dequant=window8"
              << " selected_workspace_bytes="
              << heuristics[static_cast<std::size_t>(selected_index)]
                     .workspaceSize
              << " dequant_median_milliseconds=" << dequant_median
              << " dequant_minimum_milliseconds=" << *dequant_minimum
              << " dequant_maximum_milliseconds=" << *dequant_maximum
              << " dequant_minimum_traffic_GB_per_s="
              << dequant_effective_gigabytes_per_second
              << " production_M128_median_milliseconds="
              << production_median
              << " production_M128_minimum_milliseconds="
              << *production_minimum
              << " production_M128_maximum_milliseconds="
              << *production_maximum
              << " persistent_cuBLASLt_median_milliseconds="
              << persistent_median_milliseconds
              << " inclusive_median_milliseconds=" << inclusive_median
              << " inclusive_minimum_milliseconds=" << *inclusive_minimum
              << " inclusive_maximum_milliseconds=" << *inclusive_maximum
              << " inclusive_TFLOP_per_s="
              << kUsefulFlops / (inclusive_median * 1.0e9)
              << " inclusive_speedup_vs_fresh_production_M128="
              << inclusive_speedup
              << " minimum_round_speedup=" << minimum_round_speedup
              << " all_round_positive="
              << (all_round_positive ? "true" : "false")
              << " required_speedup=" << kRequiredInclusiveSpeedup
              << " comparison_scope=canonical_NVFP4_dequant_plus_cuBLASLt"
              << " weight_scale_2=" << nvfp4_weight_scale_2
              << " scale_application=cuBLASLt_alpha_after_BF16_dequant"
              << " production_M128_bitwise_compared=true"
              << " production_M128_bitwise_mismatches="
              << production_metrics.bitwise_mismatches
              << " dequantization_timed=true gate="
              << (performance_checkpoint
                      ? (inclusive_speed_gate ? "PASS" : "FAIL")
                      : "NOT_RUN")
              << " admission="
              << (performance_checkpoint ? "DIAGNOSTIC_SUBGATE"
                                         : "NOT_RUN")
              << '\n';
  }

  // A same-process B-C-C-B comparison determines whether the local ceiling
  // chain can be extrapolated to the separately compiled public module.  B is
  // this TU's Window8 + selected Lt chain and C is the public production
  // launcher, including its independently autotuned zero-workspace Lt choice.
  // A negative extrapolation result is valid evidence, not a test failure;
  // only missing clocks, launch failures, or non-finite timings fail the gate.
  bool public_module_timing_gate = selected_checkpoint == nullptr;
  bool public_module_extrapolation_supported = false;
  if (selected_checkpoint != nullptr && public_module_evidence_gate) {
    constexpr double kMaximumExtrapolationDeltaFraction = 0.03;
    double test_tu_sum = 0.0;
    double public_module_sum = 0.0;
    bool every_round_within_band = true;
    bool every_timing_finite = true;
    for (int round = 0; round < kFormalRounds; ++round) {
      const std::string prefix =
          "public module BCCB round " + std::to_string(round + 1) + ' ';
      const double b1 = measure_inclusive_window8(
          test, lt, selected, canonical_packed_weight.get(),
          canonical_block_scale.get(), persistent_weight.get(),
          activation.get(), candidate_output, workspace.get(), kWorkspaceBytes,
          stream, nvfp4_weight_scale_2, kFormalWarmups, kFormalIterations,
          prefix + "B1 test TU");
      const double c1 = measure_public_down_module(
          test, public_down_context, canonical_packed_weight.get(),
          canonical_block_scale.get(), activation.get(),
          persistent_weight.get(), trusted_output, nvfp4_weight_scale_2, stream,
          kFormalWarmups, kFormalIterations, prefix + "C1 public module");
      const double c2 = measure_public_down_module(
          test, public_down_context, canonical_packed_weight.get(),
          canonical_block_scale.get(), activation.get(),
          persistent_weight.get(), trusted_output, nvfp4_weight_scale_2, stream,
          kFormalWarmups, kFormalIterations, prefix + "C2 public module");
      const double b2 = measure_inclusive_window8(
          test, lt, selected, canonical_packed_weight.get(),
          canonical_block_scale.get(), persistent_weight.get(),
          activation.get(), candidate_output, workspace.get(), kWorkspaceBytes,
          stream, nvfp4_weight_scale_2, kFormalWarmups, kFormalIterations,
          prefix + "B2 test TU");
      const bool finite =
          std::isfinite(b1) && b1 > 0.0 && std::isfinite(c1) && c1 > 0.0 &&
          std::isfinite(c2) && c2 > 0.0 && std::isfinite(b2) && b2 > 0.0;
      const double test_tu_pair =
          finite ? 0.5 * (b1 + b2)
                 : std::numeric_limits<double>::quiet_NaN();
      const double public_module_pair =
          finite ? 0.5 * (c1 + c2)
                 : std::numeric_limits<double>::quiet_NaN();
      const double delta_fraction =
          finite ? std::abs(public_module_pair - test_tu_pair) / test_tu_pair
                 : std::numeric_limits<double>::quiet_NaN();
      const bool within_band =
          finite && delta_fraction <= kMaximumExtrapolationDeltaFraction;
      every_timing_finite = every_timing_finite && finite;
      every_round_within_band = every_round_within_band && within_band;
      if (finite) {
        test_tu_sum += b1 + b2;
        public_module_sum += c1 + c2;
      }
      std::cout << "NVFP4_CUBLASLT_DOWN_C512_PUBLIC_MODULE_ROUND: round="
                << round + 1 << " order=B-C-C-B"
                << " B1_test_TU_ms=" << b1
                << " C1_public_module_ms=" << c1
                << " C2_public_module_ms=" << c2
                << " B2_test_TU_ms=" << b2
                << " test_TU_pair_ms=" << test_tu_pair
                << " public_module_pair_ms=" << public_module_pair
                << " public_over_test_TU="
                << public_module_pair / test_tu_pair
                << " absolute_delta_fraction=" << delta_fraction
                << " within_3pct=" << (within_band ? "true" : "false")
                << '\n';
    }
    const double denominator = 2.0 * static_cast<double>(kFormalRounds);
    const double test_tu_aggregate = test_tu_sum / denominator;
    const double public_module_aggregate = public_module_sum / denominator;
    const double aggregate_delta_fraction =
        std::abs(public_module_aggregate - test_tu_aggregate) /
        test_tu_aggregate;
    public_module_timing_gate =
        every_timing_finite && clocks.has_value() && clocks_are_fixed(*clocks);
    public_module_extrapolation_supported =
        public_module_timing_gate && every_round_within_band &&
        aggregate_delta_fraction <= kMaximumExtrapolationDeltaFraction;
    test.expect(public_module_timing_gate,
                "public Down module B-C-C-B timing protocol completes");
    ready = ready && public_module_timing_gate;
    std::cout << "NVFP4_CUBLASLT_DOWN_C512_PUBLIC_MODULE_FINAL:"
              << " rounds=" << kFormalRounds
              << " iterations=" << kFormalIterations
              << " order=B-C-C-B"
              << " test_TU_inclusive_milliseconds=" << test_tu_aggregate
              << " public_module_inclusive_milliseconds="
              << public_module_aggregate
              << " public_over_test_TU="
              << public_module_aggregate / test_tu_aggregate
              << " absolute_delta_fraction=" << aggregate_delta_fraction
              << " allowed_delta_fraction="
              << kMaximumExtrapolationDeltaFraction
              << " every_round_within_band="
              << (every_round_within_band ? "true" : "false")
              << " heuristic_rank=" << public_heuristic_rank
              << " workspace_bytes=" << public_workspace_bytes
              << " fixed_clocks=true"
              << " extrapolation="
              << (public_module_extrapolation_supported ? "SUPPORTED"
                                                        : "NOT_SUPPORTED")
              << " timing_gate="
              << (public_module_timing_gate ? "PASS" : "FAIL") << '\n';
  }
  public_module_evidence_gate =
      public_module_evidence_gate && public_module_timing_gate;

  // Probe the scale boundary without admitting it to the hybrid selector.
  // Both raw routes are executed at zero solely to characterize signed-zero
  // behavior; production remains the selected implementation for scale=0.
  ready = ready && test.cuda_ok(
                       cudaMemsetAsync(candidate_output, 0x7f,
                                       kCElements * sizeof(__nv_bfloat16),
                                       stream),
                       "poison zero-scale candidate output");
  ready = ready && test.cuda_ok(
                       cudaMemsetAsync(exact_production_output, 0x7f,
                                       kCElements * sizeof(__nv_bfloat16),
                                       stream),
                       "poison zero-scale production output");
  ready = ready && launch_dequantize_contiguous_window8(
                       test, canonical_packed_weight.get(),
                       canonical_block_scale.get(), persistent_weight.get(),
                       stream, "launch zero-scale window8 candidate dequant");
  ready = ready && launch_lt(
                       test, lt, selected, 0.0F, persistent_weight.get(),
                       activation.get(), candidate_output, workspace.get(),
                       kWorkspaceBytes, stream,
                       "launch zero-scale raw cuBLASLt diagnostic");
  const int zero_production_status = q3x::kernels::
      launch_sm87_nvfp4_w4a16_whole_chunk_down_gemm_bf16_cuda(
          canonical_packed_weight.get(), canonical_block_scale.get(), 0.0F,
          reinterpret_cast<const std::uint16_t*>(activation.get()), kM, kN,
          kK, reinterpret_cast<std::uint16_t*>(exact_production_output),
          static_cast<void*>(stream));
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(zero_production_status),
                       "launch zero-scale production M128 diagnostic");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                "synchronize zero-scale diagnostics");
  ready = ready && test.cuda_ok(
                       cudaMemcpy(candidate_host.data(), candidate_output,
                                  kCElements * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),
                       "copy zero-scale candidate output");
  ready = ready && test.cuda_ok(
                       cudaMemcpy(production_host.data(),
                                  exact_production_output,
                                  kCElements * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),
                       "copy zero-scale production output");
  const NumericalMetrics zero_metrics =
      compare_bf16_outputs(candidate_host, production_host);
  const auto count_bits = [](const std::vector<std::uint16_t>& values,
                             const std::uint16_t bits) {
    return static_cast<std::size_t>(std::count(values.begin(), values.end(),
                                               bits));
  };
  const std::size_t candidate_positive_zero =
      count_bits(candidate_host, 0x0000U);
  const std::size_t candidate_negative_zero =
      count_bits(candidate_host, 0x8000U);
  const std::size_t production_positive_zero =
      count_bits(production_host, 0x0000U);
  const std::size_t production_negative_zero =
      count_bits(production_host, 0x8000U);
  const bool zero_numeric_gate =
      zero_production_status == static_cast<int>(cudaSuccess) &&
      zero_metrics.candidate_nonfinite == 0U &&
      zero_metrics.production_nonfinite == 0U &&
      zero_metrics.maximum_absolute == 0.0 && zero_metrics.nrmse == 0.0 &&
      candidate_positive_zero + candidate_negative_zero == kCElements &&
      production_positive_zero + production_negative_zero == kCElements;
  test.expect(zero_numeric_gate,
              "zero-scale raw routes are finite numerical zeros");
  test.expect(!select_hybrid_for_scale(0.0F),
              "hybrid selector routes zero scale to production");
  std::cout << "NVFP4_CUBLASLT_DOWN_C512_SCALE_ZERO: production_status="
            << zero_production_status
            << " selected_dequant=window8"
            << " selector_admitted="
            << (select_hybrid_for_scale(0.0F) ? "true" : "false")
            << " selector_action=route_existing_production"
            << " bitwise_mismatches=" << zero_metrics.bitwise_mismatches
            << '/' << kCElements
            << " candidate_positive_zero=" << candidate_positive_zero
            << " candidate_negative_zero=" << candidate_negative_zero
            << " production_positive_zero=" << production_positive_zero
            << " production_negative_zero=" << production_negative_zero
            << " max_abs=" << zero_metrics.maximum_absolute
            << " nrmse=" << zero_metrics.nrmse
            << " cosine=" << zero_metrics.cosine
            << " gate=" << (zero_numeric_gate ? "PASS" : "FAIL") << '\n';

  // The public production dispatcher rejects NaN before enqueue.  Capture
  // proves the rejection contributes no graph nodes; the hybrid selector must
  // perform the same host-side fail-closed check and never call cuBLASLt.
  cudaGraph_t nan_graph = nullptr;
  bool nan_capture_ready = test.cuda_ok(
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
      "begin NaN-scale production capture");
  int nan_production_status = static_cast<int>(cudaErrorUnknown);
  if (nan_capture_ready) {
    nan_production_status = q3x::kernels::
        launch_sm87_nvfp4_w4a16_whole_chunk_down_gemm_bf16_cuda(
            canonical_packed_weight.get(), canonical_block_scale.get(),
            std::numeric_limits<float>::quiet_NaN(),
            reinterpret_cast<const std::uint16_t*>(activation.get()), kM, kN,
            kK, reinterpret_cast<std::uint16_t*>(exact_production_output),
            static_cast<void*>(stream));
    nan_capture_ready =
        test.cuda_ok(cudaStreamEndCapture(stream, &nan_graph),
                     "end NaN-scale production capture") &&
        nan_capture_ready;
  }
  std::size_t nan_graph_nodes = 0U;
  if (nan_capture_ready && nan_graph != nullptr) {
    nan_capture_ready =
        test.cuda_ok(cudaGraphGetNodes(nan_graph, nullptr, &nan_graph_nodes),
                     "count NaN-scale production graph nodes") &&
        nan_capture_ready;
  }
  if (nan_graph != nullptr) {
    nan_capture_ready =
        test.cuda_ok(cudaGraphDestroy(nan_graph),
                     "destroy NaN-scale production graph") &&
        nan_capture_ready;
  }
  const bool nan_selector_admitted = select_hybrid_for_scale(
      std::numeric_limits<float>::quiet_NaN());
  const bool negative_selector_admitted = select_hybrid_for_scale(-1.0F);
  const bool nan_gate =
      nan_capture_ready &&
      nan_production_status == static_cast<int>(cudaErrorInvalidValue) &&
      nan_graph_nodes == 0U && !nan_selector_admitted &&
      !negative_selector_admitted &&
      select_hybrid_for_scale(nvfp4_weight_scale_2);
  test.expect(nan_gate,
              "hybrid selector fail-closes NaN/negative before enqueue");
  ready = ready && nan_capture_ready;
  std::cout << "NVFP4_CUBLASLT_DOWN_C512_SCALE_SELECTOR: positive_finite="
            << (select_hybrid_for_scale(nvfp4_weight_scale_2) ? "hybrid"
                                                             : "reject")
            << " zero=route_existing_production"
            << " nan=reject_invalid negative=reject_invalid"
            << " production_nan_status=" << nan_production_status
            << " production_nan_graph_nodes=" << nan_graph_nodes
            << " fail_closed_condition=isfinite(scale)&&scale>0"
            << " gate=" << (nan_gate ? "PASS" : "FAIL") << '\n';

  // Full post-run immutability checks cover all eager, graph, production, and
  // special-scale executions above.
  const auto validate_immutable =
      [&](const std::uint8_t* const values,
          const std::uint8_t* const snapshot, const std::size_t count,
          const std::string& label) {
        std::array<unsigned long long, 3U> host_counts{};
        bool immutable_ready = test.cuda_ok(
            cudaMemsetAsync(validation.get(), 0,
                            3U * sizeof(unsigned long long), stream),
            label + " zero validation counts");
        validate_bytes_immutable_kernel<<<256U, 256U, 0, stream>>>(
            values, snapshot, count, validation.get(), validation.get() + 1U,
            validation.get() + 2U);
        immutable_ready =
            immutable_ready &&
            test.cuda_ok(cudaGetLastError(), label + " launch validation");
        immutable_ready =
            immutable_ready &&
            test.cuda_ok(cudaMemcpyAsync(
                             host_counts.data(), validation.get(),
                             sizeof(host_counts), cudaMemcpyDeviceToHost,
                             stream),
                         label + " copy validation counts");
        immutable_ready = immutable_ready && test.cuda_ok(
                                                   cudaStreamSynchronize(stream),
                                                   label + " synchronize");
        const bool immutable =
            immutable_ready && host_counts[0] == 0U &&
            host_counts[1] == host_counts[2] && host_counts[1] != 0U;
        test.expect(immutable, label + " remains bit exact");
        std::cout << "NVFP4_CUBLASLT_DOWN_C512_INPUT_IMMUTABLE: input="
                  << label << " mismatches=" << host_counts[0] << '/' << count
                  << " value_sum=" << host_counts[1]
                  << " snapshot_sum=" << host_counts[2]
                  << " gate=" << (immutable ? "PASS" : "FAIL") << '\n';
        return immutable;
      };
  bool immutable_gate = validate_immutable(
      reinterpret_cast<const std::uint8_t*>(activation.get()),
      reinterpret_cast<const std::uint8_t*>(activation_snapshot.get()),
      kAElements * sizeof(__nv_bfloat16), "activation");
  immutable_gate =
      validate_immutable(canonical_packed_weight.get(),
                         canonical_packed_weight_snapshot.get(),
                         kPackedWeightBytes, "packed_weight") &&
      immutable_gate;
  immutable_gate =
      validate_immutable(canonical_block_scale.get(),
                         canonical_block_scale_snapshot.get(),
                         kBlockScaleBytes, "block_scale") &&
      immutable_gate;
  ready = ready && immutable_gate;

  const auto validate_guards =
      [&](const DeviceBuffer<__nv_bfloat16>& guarded,
          const std::string& label) {
        unsigned long long host_mismatches = 0U;
        bool guard_ready = test.cuda_ok(
            cudaMemsetAsync(validation.get(), 0,
                            sizeof(unsigned long long), stream),
            label + " zero guard count");
        constexpr unsigned int kGuardThreads = 256U;
        constexpr unsigned int kGuardBlocks =
            static_cast<unsigned int>((kGuardElements + kGuardThreads - 1U) /
                                      kGuardThreads);
        validate_bf16_guards_kernel<<<kGuardBlocks, kGuardThreads, 0, stream>>>(
            guarded.get(), kCElements, kGuardElements, kGuardBits,
            validation.get());
        guard_ready =
            guard_ready &&
            test.cuda_ok(cudaGetLastError(), label + " launch guard check");
        guard_ready =
            guard_ready &&
            test.cuda_ok(cudaMemcpyAsync(
                             &host_mismatches, validation.get(),
                             sizeof(host_mismatches), cudaMemcpyDeviceToHost,
                             stream),
                         label + " copy guard count");
        guard_ready = guard_ready && test.cuda_ok(
                                           cudaStreamSynchronize(stream),
                                           label + " synchronize guard check");
        const bool guards_intact = guard_ready && host_mismatches == 0U;
        test.expect(guards_intact, label + " prefix/suffix guards are intact");
        std::cout << "NVFP4_CUBLASLT_DOWN_C512_GUARDS: output=" << label
                  << " mismatches=" << host_mismatches << '/'
                  << 2U * kGuardElements
                  << " gate=" << (guards_intact ? "PASS" : "FAIL") << '\n';
        return guards_intact;
      };
  bool guard_gate = validate_guards(output, "candidate");
  guard_gate = validate_guards(replay_reference, "trusted_reference") &&
               guard_gate;
  guard_gate =
      validate_guards(production_output, "production_M128") && guard_gate;
  ready = ready && guard_gate;

  const bool retain_candidate =
      ready && inclusive_speed_gate && production_numerical_gate &&
      graph_ready && zero_numeric_gate && nan_gate && immutable_gate &&
      guard_gate && lt_pointer_alignment_gate && window8_selected &&
      public_module_evidence_gate &&
      heuristics[static_cast<std::size_t>(selected_index)].workspaceSize == 0U;
  if (performance_checkpoint) {
    std::cout << "NVFP4_CUBLASLT_DOWN_C512_P0_RECOMMENDATION: action="
              << (retain_candidate
                      ? "retain_existing_guarded_production_route"
                      : "reject_or_continue_test_only")
              << " admitted_shape=exact_C512_Down"
              << " selector_condition=shape_exact&&aligned&&isfinite(scale)&&scale>0"
              << " scale_zero_action=route_existing_production"
              << " scale_nonfinite_or_negative_action=reject_before_enqueue"
              << " workspace_bytes=0"
              << " selected_dequant=window8"
              << " dequant_selection_gate="
              << (window8_selected ? "true" : "false")
              << " production_bitwise_mismatches="
              << production_metrics.bitwise_mismatches << '/' << kCElements
              << " graph_nodes=" << graph_node_count
              << " lt_pointer_alignment="
              << (lt_pointer_alignment_gate ? "true" : "false")
              << " input_immutable=" << (immutable_gate ? "true" : "false")
              << " guards_intact=" << (guard_gate ? "true" : "false")
              << " public_module_evidence="
              << (public_module_evidence_gate ? "pass" : "fail")
              << " public_module_extrapolation="
              << (public_module_extrapolation_supported ? "supported"
                                                        : "not_supported")
              << " gate=" << (retain_candidate ? "PASS" : "FAIL") << '\n';
  }

  q3x::kernels::destroy_sm87_nvfp4_prefill_down_cublaslt_context(
      public_down_context);
  if (stream != nullptr) {
    (void)test.cuda_ok(cudaStreamDestroy(stream), "destroy stream");
  }
  if (!ready) {
    test.expect(false, "NVFP4 inclusive ceiling completed");
  }
  if (test.failures() != 0) {
    std::cerr << test.failures()
              << " NVFP4 Down ceiling assertion(s) failed\n";
    return 1;
  }
  if (!performance_checkpoint) {
    std::cout << "NVFP4_CUBLASLT_DOWN_C512_ADMISSION: mode=smoke"
              << " admission=NOT_RUN evidence=synthetic_smoke status=PASS\n";
    std::cout << "Canonical-NVFP4 inclusive Down C512 smoke completed\n";
    return 0;
  }
  std::cout << "NVFP4_CUBLASLT_DOWN_C512_ADMISSION:"
            << " mode=performance-checkpoint"
            << " admission=" << (retain_candidate ? "PASS" : "REJECT")
            << " evidence=checkpoint_weight_only"
            << " status=" << (retain_candidate ? "PASS" : "REJECT")
            << '\n';
  std::cout << "Canonical-NVFP4 inclusive Down C512 performance checkpoint "
            << (retain_candidate ? "passed" : "did not clear admission")
            << '\n';
  if (!retain_candidate) {
    return 3;
  }
  return 0;
}
