#include "q3x/kernels/sm87_weight_only_gemv.h"
#include "q3x/io/safetensors.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace q3x::kernels {

[[nodiscard]] int
launch_sm87_fp8_w8a16_whole_chunk_qkv_m128_cp_async_register_feed_test_cuda(
    const std::uint8_t* weights, float weight_scale,
    const std::uint16_t* activations, std::size_t token_count,
    std::size_t rows, std::size_t columns, std::uint16_t* output,
    void* cuda_stream) noexcept;

[[nodiscard]] int
query_sm87_fp8_w8a16_whole_chunk_qkv_m128_cp_async_register_feed_resources_test_cuda(
    std::size_t token_count, std::size_t rows, std::size_t columns,
    int* registers_per_thread, std::size_t* static_shared_bytes,
    std::size_t* dynamic_shared_bytes, std::size_t* local_bytes,
    int* maximum_threads_per_block, int* active_blocks_per_sm) noexcept;

[[nodiscard]] int
launch_sm87_fp8_w8a16_whole_chunk_qkv_m128_cp_async_canonical_register_feed_p0_test_cuda(
    const std::uint8_t* weights, float weight_scale,
    const std::uint16_t* activations, std::size_t token_count,
    std::size_t rows, std::size_t columns, std::uint16_t* output,
    void* cuda_stream) noexcept;

[[nodiscard]] int
launch_sm87_fp8_w8a16_whole_chunk_qkv_m128_cp_async_canonical_register_feed_p1_test_cuda(
    const std::uint8_t* weights, float weight_scale,
    const std::uint16_t* activations, std::size_t token_count,
    std::size_t rows, std::size_t columns, std::uint16_t* output,
    void* cuda_stream) noexcept;

[[nodiscard]] int
query_sm87_fp8_w8a16_whole_chunk_qkv_m128_cp_async_canonical_register_feed_p0_resources_test_cuda(
    std::size_t token_count, std::size_t rows, std::size_t columns,
    int* registers_per_thread, std::size_t* static_shared_bytes,
    std::size_t* dynamic_shared_bytes, std::size_t* local_bytes,
    int* maximum_threads_per_block, int* active_blocks_per_sm) noexcept;

[[nodiscard]] int
query_sm87_fp8_w8a16_whole_chunk_qkv_m128_cp_async_canonical_register_feed_p1_resources_test_cuda(
    std::size_t token_count, std::size_t rows, std::size_t columns,
    int* registers_per_thread, std::size_t* static_shared_bytes,
    std::size_t* dynamic_shared_bytes, std::size_t* local_bytes,
    int* maximum_threads_per_block, int* active_blocks_per_sm) noexcept;

}  // namespace q3x::kernels

namespace {

constexpr std::size_t kTokens = 512U;
constexpr std::size_t kRows = 10'240U;
constexpr std::size_t kColumns = 5'120U;
constexpr std::size_t kWeightBytes = kRows * kColumns;
constexpr std::size_t kNBlockRows = 128U;
constexpr std::size_t kKStageColumns = 64U;
constexpr std::size_t kNBlocks = kRows / kNBlockRows;
constexpr std::size_t kKStages = kColumns / kKStageColumns;
constexpr std::size_t kThreads = 256U;
constexpr std::size_t kK16Groups = 4U;
constexpr std::size_t kFragmentSlots = 8U;
constexpr std::size_t kBytesPerThreadPerStage =
    kK16Groups * kFragmentSlots;
constexpr std::size_t kBytesPerTile =
    kThreads * kBytesPerThreadPerStage;
constexpr std::size_t kSidecarBytes =
    kNBlocks * kKStages * kBytesPerTile;
constexpr std::size_t kActivationElements = kTokens * kColumns;
constexpr std::size_t kOutputElements = kTokens * kRows;
constexpr std::size_t kGuardElements = 64U;
constexpr std::size_t kScrubBytes = 32U * 1024U * 1024U;
constexpr float kWeightScale = 0.00100708F;
constexpr std::string_view kDefaultCheckpointTensor =
    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
constexpr int kWarmups = 10;
constexpr int kIterations = 24;
constexpr int kRounds = 6;
constexpr double kRequiredAggregateSpeedup = 1.20;
constexpr double kRequiredRoundSpeedup = 1.15;
constexpr std::array<std::uint8_t, 16U> kCheckpointCodes{{
    0x00U, 0x80U, 0x18U, 0x20U, 0x28U, 0x30U, 0x38U, 0x3cU,
    0xb8U, 0x40U, 0xc0U, 0x48U, 0x50U, 0xd0U, 0x70U, 0xf0U,
}};

static_assert(kNBlocks == 80U);
static_assert(kKStages == 80U);
static_assert(kBytesPerTile == 8'192U);
static_assert(kWeightBytes == 52'428'800U);
static_assert(kSidecarBytes == kWeightBytes);

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

  [[nodiscard]] int failures() const noexcept { return failures_; }

 private:
  int failures_ = 0;
};

struct CheckpointPayload {
  std::vector<std::uint8_t> weights;
  std::string tensor;
  std::string shard;
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
  const fs::file_status directory_status =
      fs::status(requested_directory, filesystem_error);
  if (filesystem_error || !fs::is_directory(directory_status)) {
    test.expect(false,
                "checkpoint path is an accessible directory: " +
                    checkpoint_directory +
                    (filesystem_error ? " (" + filesystem_error.message() +
                                            ")"
                                      : ""));
    return false;
  }
  const fs::path directory = fs::canonical(requested_directory,
                                            filesystem_error);
  if (filesystem_error || directory.empty()) {
    test.expect(false,
                "checkpoint directory canonicalization succeeds: " +
                    checkpoint_directory +
                    (filesystem_error ? " (" + filesystem_error.message() +
                                            ")"
                                      : ""));
    return false;
  }

  const fs::path requested_index =
      requested_directory / "model.safetensors.index.json";
  const fs::file_status index_link_status =
      fs::symlink_status(requested_index, filesystem_error);
  if (filesystem_error || !fs::is_regular_file(index_link_status)) {
    test.expect(false,
                "checkpoint index is a regular non-symlink file: " +
                    requested_index.string() +
                    (filesystem_error ? " (" + filesystem_error.message() +
                                            ")"
                                      : ""));
    return false;
  }
  const fs::path index_path = fs::canonical(requested_index, filesystem_error);
  if (filesystem_error || !path_is_strictly_within(directory, index_path)) {
    test.expect(false,
                "checkpoint index resolves inside checkpoint directory: " +
                    requested_index.string() +
                    (filesystem_error ? " (" + filesystem_error.message() +
                                            ")"
                                      : ""));
    return false;
  }

  const st::Result<st::Index> index = st::read_index(index_path.string());
  if (!index) {
    test.expect(false,
                "read checkpoint safetensors index: " +
                    describe_safetensors_error(index.error));
    return false;
  }
  const std::string* const shard =
      index.value->shard_for(kDefaultCheckpointTensor);
  if (shard == nullptr) {
    test.expect(false,
                "checkpoint index contains tensor " +
                    std::string(kDefaultCheckpointTensor));
    return false;
  }
  if (!st::is_safe_relative_shard_path(*shard)) {
    test.expect(false,
                "checkpoint tensor has a safe relative safetensors shard: " +
                    *shard);
    return false;
  }

  const fs::path requested_shard = requested_directory / fs::path(*shard);
  const fs::file_status shard_link_status =
      fs::symlink_status(requested_shard, filesystem_error);
  if (filesystem_error || !fs::is_regular_file(shard_link_status)) {
    test.expect(false,
                "checkpoint shard is a regular non-symlink file: " +
                    requested_shard.string() +
                    (filesystem_error ? " (" + filesystem_error.message() +
                                            ")"
                                      : ""));
    return false;
  }
  const fs::path shard_path = fs::canonical(requested_shard, filesystem_error);
  if (filesystem_error || !path_is_strictly_within(directory, shard_path)) {
    test.expect(false,
                "checkpoint shard resolves inside checkpoint directory: " +
                    requested_shard.string() +
                    (filesystem_error ? " (" + filesystem_error.message() +
                                            ")"
                                      : ""));
    return false;
  }

  const st::Result<st::Header> header = st::read_header(shard_path.string());
  if (!header) {
    test.expect(false,
                "read checkpoint safetensors shard header: " +
                    describe_safetensors_error(header.error));
    return false;
  }
  const st::TensorInfo* const tensor =
      header.value->find_tensor(kDefaultCheckpointTensor);
  if (tensor == nullptr) {
    test.expect(false,
                "indexed checkpoint tensor exists in its declared shard: " +
                    std::string(kDefaultCheckpointTensor));
    return false;
  }
  if (tensor->dtype != st::DType::kF8E4M3) {
    test.expect(false,
                "checkpoint tensor dtype is F8_E4M3; got " +
                    std::string(st::to_string(tensor->dtype)));
    return false;
  }
  const bool shape_exact = tensor->shape.size() == 2U &&
                           tensor->shape[0] == kRows &&
                           tensor->shape[1] == kColumns;
  if (!shape_exact) {
    std::string actual = "[";
    for (std::size_t dimension = 0U; dimension < tensor->shape.size();
         ++dimension) {
      actual += (dimension == 0U ? "" : ",") +
                std::to_string(tensor->shape[dimension]);
    }
    actual += ']';
    test.expect(false,
                "checkpoint tensor shape is [10240,5120]; got " + actual);
    return false;
  }
  const bool bounds_exact =
      tensor->byte_size == kWeightBytes &&
      tensor->file_begin <= tensor->file_end &&
      tensor->file_end - tensor->file_begin == tensor->byte_size &&
      tensor->file_end <= header.value->file_size;
  if (!bounds_exact) {
    test.expect(false,
                "checkpoint tensor byte range is exactly 52428800 bytes; "
                "byte_size=" +
                    std::to_string(tensor->byte_size) +
                    " file_begin=" + std::to_string(tensor->file_begin) +
                    " file_end=" + std::to_string(tensor->file_end) +
                    " file_size=" + std::to_string(header.value->file_size));
    return false;
  }
  if (tensor->file_begin >
          static_cast<std::uint64_t>(
              std::numeric_limits<std::streamoff>::max()) ||
      tensor->byte_size >
          static_cast<std::uint64_t>(
              std::numeric_limits<std::streamsize>::max())) {
    test.expect(false,
                "checkpoint tensor byte range is representable by ifstream");
    return false;
  }

  std::ifstream input(shard_path, std::ios::binary);
  if (!input) {
    test.expect(false,
                "open checkpoint shard payload read-only: " +
                    shard_path.string());
    return false;
  }
  input.seekg(0, std::ios::end);
  const std::streamoff observed_file_size = input.tellg();
  if (!input || observed_file_size < 0 ||
      static_cast<std::uint64_t>(observed_file_size) !=
          header.value->file_size) {
    test.expect(false,
                "checkpoint shard size remains pinned between header and "
                "payload read");
    return false;
  }
  input.seekg(static_cast<std::streamoff>(tensor->file_begin),
              std::ios::beg);
  if (!input) {
    test.expect(false, "seek to bounded checkpoint tensor payload");
    return false;
  }
  payload.weights.resize(kWeightBytes);
  input.read(reinterpret_cast<char*>(payload.weights.data()),
             static_cast<std::streamsize>(tensor->byte_size));
  if (input.gcount() != static_cast<std::streamsize>(tensor->byte_size)) {
    test.expect(false,
                "read complete bounded checkpoint tensor payload; got " +
                    std::to_string(input.gcount()) + " bytes");
    payload.weights.clear();
    return false;
  }
  payload.tensor = std::string(kDefaultCheckpointTensor);
  payload.shard = *shard;
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
      test.expect(false, label + " size is representable");
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
    if (stream_ != nullptr) {
      (void)cudaStreamDestroy(stream_);
    }
  }

  [[nodiscard]] bool create(TestContext& test) {
    bool ready = test.cuda_ok(
        cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
        "create nonblocking stream");
    ready = ready && test.cuda_ok(cudaEventCreate(&start_),
                                  "create timing start");
    ready = ready &&
            test.cuda_ok(cudaEventCreate(&stop_), "create timing stop");
    return ready;
  }

  [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }
  [[nodiscard]] cudaEvent_t start() const noexcept { return start_; }
  [[nodiscard]] cudaEvent_t stop() const noexcept { return stop_; }

 private:
  cudaStream_t stream_ = nullptr;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

class CapturedGraph {
 public:
  CapturedGraph() = default;
  CapturedGraph(const CapturedGraph&) = delete;
  CapturedGraph& operator=(const CapturedGraph&) = delete;
  ~CapturedGraph() {
    if (executable_ != nullptr) {
      (void)cudaGraphExecDestroy(executable_);
    }
    if (graph_ != nullptr) {
      (void)cudaGraphDestroy(graph_);
    }
  }

  [[nodiscard]] cudaGraph_t* graph_address() noexcept { return &graph_; }
  [[nodiscard]] cudaGraphExec_t* executable_address() noexcept {
    return &executable_;
  }
  [[nodiscard]] cudaGraph_t graph() const noexcept { return graph_; }
  [[nodiscard]] cudaGraphExec_t executable() const noexcept {
    return executable_;
  }

 private:
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t executable_ = nullptr;
};

[[nodiscard]] std::uint16_t encode_bf16(const float value) noexcept {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t magnitude = bits & 0x7fff'ffffU;
  if (magnitude > 0x7f80'0000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  const std::uint32_t upper = bits >> 16U;
  const std::uint32_t rounding_bias = 0x7fffU + (upper & 1U);
  return static_cast<std::uint16_t>((bits + rounding_bias) >> 16U);
}

[[nodiscard]] bool is_bf16_nan(const std::uint16_t bits) noexcept {
  return (bits & 0x7f80U) == 0x7f80U && (bits & 0x007fU) != 0U;
}

[[nodiscard]] bool is_bf16_finite(const std::uint16_t bits) noexcept {
  return (bits & 0x7f80U) != 0x7f80U;
}

__global__ void scrub_l2_kernel(std::uint32_t* const words,
                                const std::size_t count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t current = index; current < count; current += stride) {
    words[current] = words[current] * 1'664'525U + 1'013'904'223U;
  }
}

[[nodiscard]] bool build_fragment_native_sidecar(
    TestContext& test, const std::vector<std::uint8_t>& canonical,
    std::vector<std::uint8_t>& sidecar) {
  if (canonical.size() != kWeightBytes) {
    test.expect(false, "canonical FP8 QKV byte count is exact");
    return false;
  }

  // SM87 BF16 matrix-B fragment sentinel. Within each stage, K16 is the
  // outer dimension so both the uint4 cp.async passes and the uint2
  // per-fragment register loads stay coalesced across the 256 threads.
  std::array<bool, kBytesPerTile> local_seen{};
  bool local_bijection = true;
  for (std::size_t thread = 0U; thread < kThreads; ++thread) {
    const std::size_t warp = thread / 32U;
    const std::size_t lane = thread % 32U;
    for (std::size_t k16 = 0U; k16 < kK16Groups; ++k16) {
      for (std::size_t slot = 0U; slot < kFragmentSlots; ++slot) {
        const std::size_t nlocal =
            warp * 16U + lane / 4U + 8U * (slot / 4U);
        const std::size_t klocal =
            k16 * 16U + 2U * (lane % 4U) + (slot % 2U) +
            8U * ((slot / 2U) % 2U);
        const std::size_t logical = nlocal * kKStageColumns + klocal;
        local_bijection = local_bijection && nlocal < kNBlockRows &&
                          klocal < kKStageColumns &&
                          logical < local_seen.size() &&
                          !local_seen[logical];
        if (logical < local_seen.size()) {
          local_seen[logical] = true;
        }
      }
    }
  }
  local_bijection =
      local_bijection &&
      std::all_of(local_seen.begin(), local_seen.end(),
                  [](const bool visited) { return visited; });
  test.expect(local_bijection,
              "fragment-native sidecar map is a complete 128x64 bijection");
  if (!local_bijection) {
    return false;
  }

  sidecar.resize(kSidecarBytes);
  bool exact_repack = true;
  for (std::size_t nblock = 0U; nblock < kNBlocks; ++nblock) {
    for (std::size_t kstage = 0U; kstage < kKStages; ++kstage) {
      for (std::size_t thread = 0U; thread < kThreads; ++thread) {
        const std::size_t warp = thread / 32U;
        const std::size_t lane = thread % 32U;
        for (std::size_t k16 = 0U; k16 < kK16Groups; ++k16) {
          for (std::size_t slot = 0U; slot < kFragmentSlots; ++slot) {
            const std::size_t nlocal =
                warp * 16U + lane / 4U + 8U * (slot / 4U);
            const std::size_t klocal =
                k16 * 16U + 2U * (lane % 4U) + (slot % 2U) +
                8U * ((slot / 2U) % 2U);
            const std::size_t sidecar_index =
                ((((nblock * kKStages + kstage) * kK16Groups + k16) *
                       kThreads +
                   thread) *
                      kFragmentSlots +
                  slot);
            const std::size_t canonical_index =
                (nblock * kNBlockRows + nlocal) * kColumns +
                kstage * kKStageColumns + klocal;
            sidecar[sidecar_index] = canonical[canonical_index];
            exact_repack =
                exact_repack && sidecar[sidecar_index] == canonical[canonical_index];
          }
        }
      }
    }
  }
  const bool gate = exact_repack && sidecar.size() == canonical.size() &&
                    sidecar.size() == kSidecarBytes;
  test.expect(gate,
              "fragment-native sidecar preserves every canonical weight byte");
  return gate;
}

struct Fixture {
  DeviceBuffer<std::uint8_t> weights;
  DeviceBuffer<std::uint8_t> sidecar;
  DeviceBuffer<std::uint16_t> activations;
  DeviceBuffer<std::uint16_t> output_store;
  DeviceBuffer<std::uint32_t> scrub;
  std::vector<std::uint8_t> host_weights;
  std::vector<std::uint8_t> host_sidecar;
  std::vector<std::uint16_t> host_activations;
  bool checkpoint_payload = false;

  [[nodiscard]] std::uint16_t* output() noexcept {
    return output_store.get() + kGuardElements;
  }
  [[nodiscard]] std::size_t guarded_output_elements() const noexcept {
    return kOutputElements + 2U * kGuardElements;
  }

  [[nodiscard]] bool initialize(TestContext& test,
                                const cudaStream_t stream,
                                const CheckpointPayload* const checkpoint) {
    checkpoint_payload = checkpoint != nullptr;
    if (checkpoint_payload) {
      if (checkpoint->weights.size() != kWeightBytes) {
        test.expect(false,
                    "prepared checkpoint payload has exact FP8 QKV size");
        return false;
      }
      host_weights = checkpoint->weights;
    } else {
      host_weights.resize(kWeightBytes);
    }
    host_activations.resize(kActivationElements);
    if (!checkpoint_payload) {
      for (std::size_t index = 0U; index < host_weights.size(); ++index) {
        const std::size_t row = index / kColumns;
        const std::size_t column = index - row * kColumns;
        host_weights[index] = kCheckpointCodes[
            (index * 5U + row * 11U + (column >> 4U)) %
            kCheckpointCodes.size()];
      }
      // The first 1,024 bytes cover all 256 E4M3FN codes in each of the four
      // byte positions of an aligned uint32 word. The remaining payload stays
      // checkpoint-like and finite for the timing cell.
      for (std::size_t code = 0U; code < 256U; ++code) {
        for (std::size_t position = 0U; position < 4U; ++position) {
          host_weights[code * 4U + position] =
              static_cast<std::uint8_t>(code);
        }
      }
    }
    if (!build_fragment_native_sidecar(test, host_weights, host_sidecar)) {
      return false;
    }
    for (std::size_t index = 0U; index < host_activations.size(); ++index) {
      const int centered = static_cast<int>(index % 31U) - 15;
      host_activations[index] =
          encode_bf16(static_cast<float>(centered) / 16.0F);
    }

    bool ready = weights.allocate(test, kWeightBytes,
                                  "allocate FP8 QKV weights");
    ready = ready && sidecar.allocate(test, kSidecarBytes,
                                      "allocate fragment-native sidecar");
    ready = ready && activations.allocate(test, kActivationElements,
                                           "allocate BF16 activations");
    ready = ready && output_store.allocate(test, guarded_output_elements(),
                                            "allocate guarded output");
    ready = ready && scrub.allocate(test, kScrubBytes / sizeof(std::uint32_t),
                                    "allocate L2 scrub");
    if (!ready) {
      return false;
    }
    ready = test.cuda_ok(
        cudaMemcpyAsync(weights.get(), host_weights.data(), kWeightBytes,
                        cudaMemcpyHostToDevice, stream),
        "upload FP8 QKV weights");
    ready = ready && test.cuda_ok(
                         cudaMemsetAsync(sidecar.get(), 0xcd, kSidecarBytes,
                                         stream),
                         "poison GPU-built fragment-native sidecar");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             activations.get(), host_activations.data(),
                             kActivationElements * sizeof(std::uint16_t),
                             cudaMemcpyHostToDevice, stream),
                         "upload BF16 activations");
    ready = ready && test.cuda_ok(
                         cudaMemsetAsync(scrub.get(), 0, kScrubBytes, stream),
                         "initialize L2 scrub");
    ready = ready && test.cuda_ok(
                         static_cast<cudaError_t>(q3x::kernels::
                             launch_sm87_fp8_w8a16_whole_chunk_qkv_register_feed_pack_cuda(
                                 weights.get(), sidecar.get(), kRows,
                                 kColumns, static_cast<void*>(stream))),
                         "launch production QKV register-feed GPU pack");
    ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                  "fixture upload synchronize");
    std::vector<std::uint8_t> gpu_built_sidecar(kSidecarBytes);
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(gpu_built_sidecar.data(),
                                         sidecar.get(), kSidecarBytes,
                                         cudaMemcpyDeviceToHost, stream),
                         "copy GPU-built fragment-native sidecar");
    ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                  "GPU-built sidecar copy synchronize");
    const bool exact_pack = ready && gpu_built_sidecar == host_sidecar;
    test.expect(exact_pack,
                "production GPU pack exactly matches the host sidecar oracle");
    return exact_pack;
  }
};

enum class Variant {
  kBaseline,
  kCandidate,
  kCanonicalP0,
  kCanonicalP1,
};

[[nodiscard]] const char* variant_name(const Variant variant) noexcept {
  switch (variant) {
    case Variant::kBaseline:
      return "baseline";
    case Variant::kCandidate:
      return "candidate";
    case Variant::kCanonicalP0:
      return "canonical_p0";
    case Variant::kCanonicalP1:
      return "canonical_p1_xor";
  }
  return "unknown";
}

[[nodiscard]] constexpr bool variant_is_test_only(
    const Variant variant) noexcept {
  return variant == Variant::kCanonicalP0 ||
         variant == Variant::kCanonicalP1;
}

[[nodiscard]] cudaError_t launch_variant(Fixture& fixture,
                                         const cudaStream_t stream,
                                         const Variant variant) noexcept {
  int status = static_cast<int>(cudaErrorInvalidValue);
  switch (variant) {
    case Variant::kBaseline:
      status = q3x::kernels::
          launch_sm87_fp8_w8a16_whole_chunk_gemm_bf16_cuda(
              fixture.weights.get(), kWeightScale, fixture.activations.get(),
              kTokens, kRows, kColumns, fixture.output(),
              static_cast<void*>(stream));
      break;
    case Variant::kCandidate:
      status = q3x::kernels::
          launch_sm87_fp8_w8a16_whole_chunk_qkv_register_feed_gemm_bf16_cuda(
              fixture.sidecar.get(), kWeightScale,
              fixture.activations.get(), kTokens, kRows, kColumns,
              fixture.output(), static_cast<void*>(stream));
      break;
    case Variant::kCanonicalP0:
      status = q3x::kernels::
          launch_sm87_fp8_w8a16_whole_chunk_qkv_m128_cp_async_canonical_register_feed_p0_test_cuda(
              fixture.weights.get(), kWeightScale,
              fixture.activations.get(), kTokens, kRows, kColumns,
              fixture.output(), static_cast<void*>(stream));
      break;
    case Variant::kCanonicalP1:
      status = q3x::kernels::
          launch_sm87_fp8_w8a16_whole_chunk_qkv_m128_cp_async_canonical_register_feed_p1_test_cuda(
              fixture.weights.get(), kWeightScale,
              fixture.activations.get(), kTokens, kRows, kColumns,
              fixture.output(), static_cast<void*>(stream));
      break;
  }
  return static_cast<cudaError_t>(status);
}

[[nodiscard]] bool run_resource_gate(TestContext& test,
                                     const Variant variant,
                                     const bool assert_gate = true) {
  int registers = -1;
  std::size_t static_shared = std::numeric_limits<std::size_t>::max();
  std::size_t dynamic_shared = std::numeric_limits<std::size_t>::max();
  std::size_t local = std::numeric_limits<std::size_t>::max();
  int threads = -1;
  int active = -1;
  int status = static_cast<int>(cudaErrorInvalidValue);
  if (variant == Variant::kCandidate) {
    status = q3x::kernels::
        query_sm87_fp8_w8a16_whole_chunk_qkv_m128_cp_async_register_feed_resources_test_cuda(
            kTokens, kRows, kColumns, &registers, &static_shared,
            &dynamic_shared, &local, &threads, &active);
  } else if (variant == Variant::kCanonicalP0) {
    status = q3x::kernels::
        query_sm87_fp8_w8a16_whole_chunk_qkv_m128_cp_async_canonical_register_feed_p0_resources_test_cuda(
            kTokens, kRows, kColumns, &registers, &static_shared,
            &dynamic_shared, &local, &threads, &active);
  } else if (variant == Variant::kCanonicalP1) {
    status = q3x::kernels::
        query_sm87_fp8_w8a16_whole_chunk_qkv_m128_cp_async_canonical_register_feed_p1_resources_test_cuda(
            kTokens, kRows, kColumns, &registers, &static_shared,
            &dynamic_shared, &local, &threads, &active);
  }
  const bool gate =
      status == static_cast<int>(cudaSuccess) && registers > 0 &&
      registers <= 128 && static_shared == 512U &&
      dynamic_shared == 79'872U &&
      static_shared + dynamic_shared == 80'384U && local == 0U &&
      threads == 256 && active == 2;
  std::cout << "FP8_REGISTER_FEED_RESOURCES:";
  if (variant_is_test_only(variant)) {
    std::cout << " variant=" << variant_name(variant)
              << " test_only=true";
  }
  std::cout << " status=" << status
            << " registers=" << registers
            << " static_shared_bytes=" << static_shared
            << " dynamic_shared_bytes=" << dynamic_shared
            << " total_shared_bytes=" << static_shared + dynamic_shared
            << " local_bytes=" << local << " threads=" << threads
            << " active_blocks_per_sm=" << active
            << " resident_warps_per_sm=" << active * threads / 32
            << " gate=" << (gate ? "PASS" : "FAIL") << '\n';
  if (assert_gate) {
    test.expect(
        gate,
        std::string(variant_name(variant)) +
            " clears the strict two-CTA resource gate");
  }
  return gate;
}

[[nodiscard]] bool run_invalid_graph_gate(TestContext& test,
                                          const cudaStream_t stream,
                                          const Variant variant,
                                          const bool assert_gate = true) {
  constexpr std::uintptr_t kWeightAddress = 0x1'0000'0000ULL;
  constexpr std::uintptr_t kActivationAddress = 0x2'0000'0000ULL;
  constexpr std::uintptr_t kOutputAddress = 0x3'0000'0000ULL;
  constexpr std::uintptr_t kMaximum =
      std::numeric_limits<std::uintptr_t>::max();
  const auto* const weights =
      reinterpret_cast<const std::uint8_t*>(kWeightAddress);
  const auto* const activations =
      reinterpret_cast<const std::uint16_t*>(kActivationAddress);
  auto* const output = reinterpret_cast<std::uint16_t*>(kOutputAddress);
  const auto launch =
      [variant, stream](const std::uint8_t* const w, const float scale,
          const std::uint16_t* const a, const std::size_t tokens,
          const std::size_t rows, const std::size_t columns,
          std::uint16_t* const o) noexcept {
        if (variant == Variant::kCandidate) {
          return q3x::kernels::
              launch_sm87_fp8_w8a16_whole_chunk_qkv_register_feed_gemm_bf16_cuda(
                  w, scale, a, tokens, rows, columns, o,
                  static_cast<void*>(stream));
        }
        if (variant == Variant::kCanonicalP0) {
          return q3x::kernels::
              launch_sm87_fp8_w8a16_whole_chunk_qkv_m128_cp_async_canonical_register_feed_p0_test_cuda(
                  w, scale, a, tokens, rows, columns, o,
                  static_cast<void*>(stream));
        }
        return q3x::kernels::
            launch_sm87_fp8_w8a16_whole_chunk_qkv_m128_cp_async_canonical_register_feed_p1_test_cuda(
                w, scale, a, tokens, rows, columns, o,
                static_cast<void*>(stream));
      };

  cudaGraph_t graph = nullptr;
  bool ready = test.cuda_ok(
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
      "invalid begin capture");
  constexpr std::size_t kProductionInvalidCount = 20U;
  constexpr std::size_t kCanonicalInvalidCount = 23U;
  const std::size_t expected_invalid_count =
      variant_is_test_only(variant) ? kCanonicalInvalidCount
                                    : kProductionInvalidCount;
  std::array<int, kCanonicalInvalidCount> statuses{};
  statuses.fill(static_cast<int>(cudaErrorUnknown));
  if (ready) {
    statuses[0] = launch(nullptr, kWeightScale, activations, kTokens, kRows,
                         kColumns, output);
    statuses[1] = launch(weights,
                         std::numeric_limits<float>::quiet_NaN(), activations,
                         kTokens, kRows, kColumns, output);
    statuses[2] = launch(weights, -1.0F, activations, kTokens, kRows,
                         kColumns, output);
    statuses[3] = launch(weights, kWeightScale, nullptr, kTokens, kRows,
                         kColumns, output);
    statuses[4] = launch(weights, kWeightScale, activations, kTokens, kRows,
                         kColumns, nullptr);
    statuses[5] = launch(weights, kWeightScale, activations, 256U, kRows,
                         kColumns, output);
    statuses[6] = launch(weights, kWeightScale, activations, 513U, kRows,
                         kColumns, output);
    statuses[7] = launch(weights, kWeightScale, activations, kTokens,
                         kRows - 1U, kColumns, output);
    statuses[8] = launch(weights, kWeightScale, activations, kTokens, kRows,
                         kColumns - 1U, output);
    statuses[9] = launch(weights + 1U, kWeightScale, activations, kTokens,
                         kRows, kColumns, output);
    statuses[10] = launch(
        weights, kWeightScale,
        reinterpret_cast<const std::uint16_t*>(kActivationAddress + 2U),
        kTokens, kRows, kColumns, output);
    statuses[11] = launch(
        weights, kWeightScale, activations, kTokens, kRows, kColumns,
        reinterpret_cast<std::uint16_t*>(kOutputAddress + 1U));
    statuses[12] = launch(
        weights, kWeightScale, activations, kTokens, kRows, kColumns,
        reinterpret_cast<std::uint16_t*>(kActivationAddress + 128U));
    statuses[13] = launch(
        weights, kWeightScale, activations, kTokens, kRows, kColumns,
        reinterpret_cast<std::uint16_t*>(kWeightAddress + 128U));
    statuses[14] = launch(
        reinterpret_cast<const std::uint8_t*>(kMaximum - 15U), kWeightScale,
        activations, kTokens, kRows, kColumns, output);
    statuses[15] = launch(
        weights, kWeightScale,
        reinterpret_cast<const std::uint16_t*>(kMaximum - 7U), kTokens,
        kRows, kColumns, output);
    statuses[16] = launch(
        weights, kWeightScale, activations, kTokens, kRows, kColumns,
        reinterpret_cast<std::uint16_t*>(kMaximum - 1U));
    statuses[17] = launch(weights, kWeightScale, activations, 0U, kRows,
                          kColumns, output);
    statuses[18] = launch(weights, kWeightScale, activations, kTokens, 0U,
                          kColumns, output);
    statuses[19] = launch(
        weights, std::numeric_limits<float>::infinity(), activations,
        kTokens, kRows, kColumns, output);
    if (variant_is_test_only(variant)) {
      statuses[20] = launch(weights + 8U, kWeightScale, activations,
                            kTokens, kRows, kColumns, output);
      statuses[21] = launch(
          weights, kWeightScale,
          reinterpret_cast<const std::uint16_t*>(kActivationAddress + 8U),
          kTokens, kRows, kColumns, output);
      statuses[22] = launch(
          weights, kWeightScale,
          reinterpret_cast<const std::uint16_t*>(kWeightAddress), kTokens,
          kRows, kColumns, output);
    }
    ready = test.cuda_ok(cudaStreamEndCapture(stream, &graph),
                         "invalid end capture") &&
            ready;
  }
  std::size_t nodes = 0U;
  if (ready && graph != nullptr) {
    ready = test.cuda_ok(cudaGraphGetNodes(graph, nullptr, &nodes),
                         "invalid count graph nodes") &&
            ready;
  }
  if (graph != nullptr) {
    ready = test.cuda_ok(cudaGraphDestroy(graph), "invalid destroy graph") &&
            ready;
  }
  const std::size_t invalid_count = static_cast<std::size_t>(std::count(
      statuses.begin(), statuses.begin() + expected_invalid_count,
      static_cast<int>(cudaErrorInvalidValue)));
  const bool gate = ready && invalid_count == expected_invalid_count &&
                    nodes == 0U;
  std::cout << "FP8_REGISTER_FEED_INVALID_GRAPH:";
  if (variant_is_test_only(variant)) {
    std::cout << " variant=" << variant_name(variant)
              << " test_only=true";
  }
  std::cout << " invalid_statuses="
            << invalid_count << '/' << expected_invalid_count
            << " graph_nodes=" << nodes
            << " gate=" << (gate ? "PASS" : "FAIL") << '\n';
  if (assert_gate) {
    test.expect(gate, std::string("all invalid ") + variant_name(variant) +
                          " calls enqueue zero nodes");
  }
  return gate;
}

[[nodiscard]] bool run_pack_invalid_graph_gate(
    TestContext& test, const cudaStream_t stream) {
  constexpr std::uintptr_t kCanonicalAddress = 0x1'0000'0000ULL;
  constexpr std::uintptr_t kSidecarAddress = 0x2'0000'0000ULL;
  constexpr std::uintptr_t kMaximum =
      std::numeric_limits<std::uintptr_t>::max();
  const auto* const canonical =
      reinterpret_cast<const std::uint8_t*>(kCanonicalAddress);
  auto* const sidecar = reinterpret_cast<std::uint8_t*>(kSidecarAddress);
  const auto launch = [&](const std::uint8_t* const source,
                          std::uint8_t* const destination,
                          const std::size_t rows,
                          const std::size_t columns) noexcept {
    return q3x::kernels::
        launch_sm87_fp8_w8a16_whole_chunk_qkv_register_feed_pack_cuda(
            source, destination, rows, columns,
            static_cast<void*>(stream));
  };

  cudaGraph_t graph = nullptr;
  bool ready = test.cuda_ok(
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
      "invalid pack begin capture");
  std::array<int, 10U> statuses{};
  statuses.fill(static_cast<int>(cudaErrorUnknown));
  if (ready) {
    statuses[0] = launch(nullptr, sidecar, kRows, kColumns);
    statuses[1] = launch(canonical, nullptr, kRows, kColumns);
    statuses[2] = launch(canonical, sidecar, kRows - 1U, kColumns);
    statuses[3] = launch(canonical, sidecar, kRows, kColumns - 1U);
    statuses[4] = launch(canonical + 1U, sidecar, kRows, kColumns);
    statuses[5] = launch(canonical, sidecar + 1U, kRows, kColumns);
    statuses[6] = launch(
        canonical, reinterpret_cast<std::uint8_t*>(kCanonicalAddress),
        kRows, kColumns);
    statuses[7] = launch(
        canonical, reinterpret_cast<std::uint8_t*>(kCanonicalAddress + 16U),
        kRows, kColumns);
    statuses[8] = launch(
        reinterpret_cast<const std::uint8_t*>(kMaximum - 15U), sidecar,
        kRows, kColumns);
    statuses[9] = launch(
        canonical, reinterpret_cast<std::uint8_t*>(kMaximum - 15U), kRows,
        kColumns);
    ready = test.cuda_ok(cudaStreamEndCapture(stream, &graph),
                         "invalid pack end capture") &&
            ready;
  }
  std::size_t nodes = 0U;
  if (ready && graph != nullptr) {
    ready = test.cuda_ok(cudaGraphGetNodes(graph, nullptr, &nodes),
                         "invalid pack count graph nodes") &&
            ready;
  }
  if (graph != nullptr) {
    ready = test.cuda_ok(cudaGraphDestroy(graph),
                         "invalid pack destroy graph") &&
            ready;
  }
  const std::size_t invalid_count = static_cast<std::size_t>(std::count(
      statuses.begin(), statuses.end(),
      static_cast<int>(cudaErrorInvalidValue)));
  const bool gate = ready && invalid_count == statuses.size() && nodes == 0U;
  std::cout << "FP8_REGISTER_FEED_PACK_INVALID_GRAPH: invalid_statuses="
            << invalid_count << '/' << statuses.size()
            << " graph_nodes=" << nodes
            << " gate=" << (gate ? "PASS" : "FAIL") << '\n';
  test.expect(gate, "all invalid production pack calls enqueue zero nodes");
  return gate;
}

[[nodiscard]] bool poison_output(TestContext& test, Fixture& fixture,
                                 const cudaStream_t stream, const int byte,
                                 const std::string& label) {
  bool ready = test.cuda_ok(
      cudaMemsetAsync(fixture.output_store.get(), byte,
                      fixture.guarded_output_elements() *
                          sizeof(std::uint16_t),
                      stream),
      label + " poison output");
  ready = ready &&
          test.cuda_ok(cudaStreamSynchronize(stream),
                       label + " poison synchronize");
  return ready;
}

[[nodiscard]] bool copy_output(TestContext& test, const Fixture& fixture,
                               const cudaStream_t stream,
                               std::vector<std::uint16_t>& host,
                               const std::string& label) {
  host.resize(fixture.guarded_output_elements());
  bool ready = test.cuda_ok(
      cudaMemcpyAsync(host.data(), fixture.output_store.get(),
                      host.size() * sizeof(std::uint16_t),
                      cudaMemcpyDeviceToHost, stream),
      label + " copy output");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                label + " copy synchronize");
  return ready;
}

struct GraphIdentity {
  cudaKernelNodeParams parameters{};
  std::size_t node_count = 0U;
  bool valid = false;
};

[[nodiscard]] bool capture_variant(TestContext& test, Fixture& fixture,
                                   const cudaStream_t stream,
                                   const Variant variant,
                                   CapturedGraph& captured,
                                   GraphIdentity& identity) {
  bool ready = test.cuda_ok(
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
      std::string("begin ") + variant_name(variant) + " capture");
  const bool capture_started = ready;
  if (capture_started) {
    ready = test.cuda_ok(launch_variant(fixture, stream, variant),
                         std::string("launch ") + variant_name(variant) +
                             " capture") &&
            ready;
    // End every successfully started capture even when the launcher fails.
    // CUDA may report an invalidated capture here, but the call still restores
    // the stream so later diagnostics do not inherit capture state.
    const bool capture_ended = test.cuda_ok(
        cudaStreamEndCapture(stream, captured.graph_address()),
        std::string("end ") + variant_name(variant) + " capture");
    ready = capture_ended && ready;
  }
  std::array<cudaGraphNode_t, 2U> nodes{};
  std::size_t capacity = nodes.size();
  cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
  if (ready) {
    ready = test.cuda_ok(
                cudaGraphGetNodes(captured.graph(), nodes.data(), &capacity),
                std::string("get ") + variant_name(variant) +
                    " graph nodes") &&
            ready;
    identity.node_count = capacity;
  }
  if (ready && capacity == 1U) {
    ready = test.cuda_ok(cudaGraphNodeGetType(nodes[0], &type),
                         std::string("get ") + variant_name(variant) +
                             " node type") &&
            ready;
    if (ready && type == cudaGraphNodeTypeKernel) {
      ready = test.cuda_ok(
                  cudaGraphKernelNodeGetParams(nodes[0],
                                               &identity.parameters),
                  std::string("get ") + variant_name(variant) +
                      " kernel params") &&
              ready;
    }
  }
  if (ready) {
    ready = test.cuda_ok(
                cudaGraphInstantiate(captured.executable_address(),
                                     captured.graph(), nullptr, nullptr, 0U),
                std::string("instantiate ") + variant_name(variant) +
                    " graph") &&
            ready;
  }
  const unsigned int expected_block = 256U;
  const unsigned int expected_dynamic =
      variant == Variant::kBaseline ? 0U : 79'872U;
  identity.valid =
      ready && capacity == 1U && type == cudaGraphNodeTypeKernel &&
      identity.parameters.gridDim.x == 320U &&
      identity.parameters.gridDim.y == 1U &&
      identity.parameters.gridDim.z == 1U &&
      identity.parameters.blockDim.x == expected_block &&
      identity.parameters.blockDim.y == 1U &&
      identity.parameters.blockDim.z == 1U &&
      identity.parameters.sharedMemBytes == expected_dynamic &&
      identity.parameters.func != nullptr;
  return identity.valid;
}

[[nodiscard]] bool run_canonical_function_identity_gate(
    TestContext& test, Fixture& fixture, const cudaStream_t stream) {
  CapturedGraph p0_graph;
  CapturedGraph p1_graph;
  GraphIdentity p0_identity;
  GraphIdentity p1_identity;
  bool ready = capture_variant(test, fixture, stream, Variant::kCanonicalP0,
                               p0_graph, p0_identity);
  ready = capture_variant(test, fixture, stream, Variant::kCanonicalP1,
                          p1_graph, p1_identity) &&
          ready;
  const bool gate = ready && p0_identity.valid && p1_identity.valid &&
                    p0_identity.parameters.func !=
                        p1_identity.parameters.func;
  std::cout << "FP8_REGISTER_FEED_CANONICAL_FUNCTION_IDENTITY: "
               "p0_nodes="
            << p0_identity.node_count << " p1_nodes="
            << p1_identity.node_count
            << " functions_distinct="
            << (p0_identity.parameters.func != p1_identity.parameters.func
                    ? "true"
                    : "false")
            << " test_only=true gate=" << (gate ? "PASS" : "FAIL")
            << '\n';
  test.expect(gate,
              "canonical P0 and P1 capture distinct template kernels");
  return gate;
}

[[nodiscard]] bool verify_immutable(TestContext& test,
                                    const Fixture& fixture,
                                    const cudaStream_t stream) {
  std::vector<std::uint8_t> weights(kWeightBytes);
  std::vector<std::uint8_t> sidecar(kSidecarBytes);
  std::vector<std::uint16_t> activations(kActivationElements);
  bool ready = test.cuda_ok(
      cudaMemcpyAsync(weights.data(), fixture.weights.get(), kWeightBytes,
                      cudaMemcpyDeviceToHost, stream),
      "copy immutable weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(sidecar.data(), fixture.sidecar.get(),
                                       kSidecarBytes, cudaMemcpyDeviceToHost,
                                       stream),
                       "copy immutable fragment-native sidecar");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           activations.data(), fixture.activations.get(),
                           kActivationElements * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, stream),
                       "copy immutable activations");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                "immutable copy synchronize");
  const bool gate = ready && weights == fixture.host_weights &&
                    sidecar == fixture.host_sidecar &&
                    activations == fixture.host_activations;
  test.expect(gate,
              "weights, sidecar, and activations remain immutable");
  return gate;
}

[[nodiscard]] bool run_correctness(TestContext& test, Fixture& fixture,
                                   const cudaStream_t stream,
                                   const Variant candidate_variant,
                                   const bool assert_gate = true) {
  std::vector<std::uint16_t> baseline;
  std::vector<std::uint16_t> candidate;
  std::vector<std::uint16_t> replay1;
  std::vector<std::uint16_t> replay2;
  bool ready = poison_output(test, fixture, stream, 0x3c, "baseline");
  ready = ready && test.cuda_ok(
                       launch_variant(fixture, stream, Variant::kBaseline),
                       "baseline direct launch");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                "baseline synchronize");
  ready = ready && copy_output(test, fixture, stream, baseline, "baseline");

  const std::string candidate_label = variant_name(candidate_variant);
  ready = ready && poison_output(test, fixture, stream, 0xa5,
                                 candidate_label);
  ready = ready && test.cuda_ok(
                       launch_variant(fixture, stream, candidate_variant),
                       candidate_label + " direct launch");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                candidate_label + " synchronize");
  ready = ready && copy_output(test, fixture, stream, candidate,
                               candidate_label);

  CapturedGraph baseline_graph;
  CapturedGraph candidate_graph;
  GraphIdentity baseline_identity;
  GraphIdentity candidate_identity;
  ready = ready && capture_variant(test, fixture, stream, Variant::kBaseline,
                                   baseline_graph, baseline_identity);
  ready = ready && capture_variant(test, fixture, stream,
                                   candidate_variant, candidate_graph,
                                   candidate_identity);
  const bool graph_identity =
      ready && baseline_identity.valid && candidate_identity.valid &&
      baseline_identity.parameters.func != candidate_identity.parameters.func;
  std::cout << "FP8_REGISTER_FEED_VALID_GRAPH:";
  if (variant_is_test_only(candidate_variant)) {
    std::cout << " variant=" << candidate_label << " test_only=true";
  }
  std::cout << " baseline_nodes="
            << baseline_identity.node_count
            << " candidate_nodes=" << candidate_identity.node_count
            << " baseline_grid_x=" << baseline_identity.parameters.gridDim.x
            << " candidate_grid_x="
            << candidate_identity.parameters.gridDim.x
            << " baseline_block_x="
            << baseline_identity.parameters.blockDim.x
            << " candidate_block_x="
            << candidate_identity.parameters.blockDim.x
            << " candidate_dynamic_shared_bytes="
            << candidate_identity.parameters.sharedMemBytes
            << " functions_distinct="
            << (baseline_identity.parameters.func !=
                        candidate_identity.parameters.func
                    ? "true"
                    : "false")
            << " gate=" << (graph_identity ? "PASS" : "FAIL") << '\n';
  test.expect(graph_identity,
              "baseline and candidate graph identities are exact");

  ready = ready && poison_output(test, fixture, stream, 0x5a, "replay1");
  ready = ready && test.cuda_ok(
                       cudaGraphLaunch(candidate_graph.executable(), stream),
                       "candidate graph replay1");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                "candidate replay1 synchronize");
  ready = ready && copy_output(test, fixture, stream, replay1, "replay1");
  ready = ready && poison_output(test, fixture, stream, 0x69, "replay2");
  ready = ready && test.cuda_ok(
                       cudaGraphLaunch(candidate_graph.executable(), stream),
                       "candidate graph replay2");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                "candidate replay2 synchronize");
  ready = ready && copy_output(test, fixture, stream, replay2, "replay2");

  std::size_t candidate_mismatches = 0U;
  std::size_t replay1_mismatches = 0U;
  std::size_t replay2_mismatches = 0U;
  std::size_t nan_outputs = 0U;
  std::size_t nan_class_or_sign_mismatches = 0U;
  std::size_t unexpected_nonfinite = 0U;
  bool guards = ready;
  if (ready) {
    for (std::size_t index = 0U; index < kOutputElements; ++index) {
      const std::size_t guarded = kGuardElements + index;
      const std::uint16_t oracle = baseline[guarded];
      candidate_mismatches += candidate[guarded] != oracle ? 1U : 0U;
      replay1_mismatches += replay1[guarded] != oracle ? 1U : 0U;
      replay2_mismatches += replay2[guarded] != oracle ? 1U : 0U;
      const bool oracle_nan = is_bf16_nan(oracle);
      const auto preserves_nan_class_and_sign =
          [oracle, oracle_nan](const std::uint16_t actual) noexcept {
            const bool actual_nan = is_bf16_nan(actual);
            return actual_nan == oracle_nan &&
                   (!oracle_nan || ((actual ^ oracle) & 0x8000U) == 0U);
          };
      const bool nan_class_and_sign =
          preserves_nan_class_and_sign(candidate[guarded]) &&
          preserves_nan_class_and_sign(replay1[guarded]) &&
          preserves_nan_class_and_sign(replay2[guarded]);
      nan_class_or_sign_mismatches += nan_class_and_sign ? 0U : 1U;
      if (oracle_nan) {
        ++nan_outputs;
      } else {
        unexpected_nonfinite +=
            !is_bf16_finite(oracle) ||
                    !is_bf16_finite(candidate[guarded]) ||
                    !is_bf16_finite(replay1[guarded]) ||
                    !is_bf16_finite(replay2[guarded])
                ? 1U
                : 0U;
      }
    }
    for (std::size_t index = 0U; index < kGuardElements; ++index) {
      const std::size_t tail = kGuardElements + kOutputElements + index;
      guards = guards && baseline[index] == 0x3c3cU &&
               baseline[tail] == 0x3c3cU &&
               candidate[index] == 0xa5a5U &&
               candidate[tail] == 0xa5a5U &&
               replay1[index] == 0x5a5aU &&
               replay1[tail] == 0x5a5aU &&
               replay2[index] == 0x6969U &&
               replay2[tail] == 0x6969U;
    }
  }
  const bool raw_code_coverage_required = !fixture.checkpoint_payload;
  bool raw_code_coverage = true;
  if (raw_code_coverage_required) {
    for (std::size_t code = 0U; code < 256U; ++code) {
      for (std::size_t position = 0U; position < 4U; ++position) {
        raw_code_coverage =
            raw_code_coverage &&
            fixture.host_weights[code * 4U + position] ==
                static_cast<std::uint8_t>(code);
      }
    }
  }
  const bool immutable = ready && verify_immutable(test, fixture, stream);
  const bool payload_specific_gate =
      fixture.checkpoint_payload
          ? true
          : raw_code_coverage && nan_outputs == kTokens &&
                unexpected_nonfinite == 0U;
  const bool gate =
      ready && graph_identity && payload_specific_gate &&
      candidate_mismatches == 0U && replay1_mismatches == 0U &&
      replay2_mismatches == 0U && nan_class_or_sign_mismatches == 0U &&
      guards && immutable;
  std::cout << "FP8_REGISTER_FEED_CORRECTNESS:";
  if (variant_is_test_only(candidate_variant)) {
    std::cout << " variant=" << candidate_label << " test_only=true";
  }
  std::cout << " candidate_mismatches="
            << candidate_mismatches << '/' << kOutputElements
            << " replay1_mismatches=" << replay1_mismatches << '/'
            << kOutputElements
            << " replay2_mismatches=" << replay2_mismatches << '/'
            << kOutputElements
            << " raw_codes_per_byte_position=256 byte_positions=4"
            << " raw_code_coverage_required="
            << (raw_code_coverage_required ? "true" : "false")
            << " raw_code_coverage="
            << (raw_code_coverage ? "true" : "false")
            << " classified_nan_outputs=" << nan_outputs
            << " expected_nan_policy="
            << (fixture.checkpoint_payload ? "checkpoint_unconstrained"
                                           : "synthetic_exact_512")
            << " nan_class_or_sign_mismatches="
            << nan_class_or_sign_mismatches
            << " unexpected_nonfinite=" << unexpected_nonfinite
            << " guards=" << (guards ? "intact" : "BAD")
            << " immutable=" << (immutable ? "true" : "false")
            << " gate=" << (gate ? "PASS" : "FAIL") << '\n';
  if (assert_gate) {
    test.expect(gate, candidate_label +
                          " is exact across direct and Graph replay");
  }
  return gate;
}

[[nodiscard]] bool scrub_l2(TestContext& test, Fixture& fixture,
                            const cudaStream_t stream,
                            const std::string& label) {
  scrub_l2_kernel<<<256U, 256U, 0U, stream>>>(
      fixture.scrub.get(), kScrubBytes / sizeof(std::uint32_t));
  bool ready = test.cuda_ok(cudaGetLastError(), label + " scrub launch");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                label + " scrub synchronize");
  return ready;
}

[[nodiscard]] float measure_pass(TestContext& test, Fixture& fixture,
                                 const Execution& execution,
                                 const Variant variant,
                                 const std::string& label,
                                 const int warmups = kWarmups,
                                 const int iterations = kIterations,
                                 const bool profiler_range = false) {
  const cudaStream_t stream = execution.stream();
  bool ready = scrub_l2(test, fixture, stream, label);
  for (int warmup = 0; ready && warmup < warmups; ++warmup) {
    ready = test.cuda_ok(launch_variant(fixture, stream, variant),
                         label + " warmup") &&
            ready;
  }
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                label + " warmup synchronize");
  if (ready && profiler_range) {
    ready = test.cuda_ok(cudaProfilerStart(), label + " profiler start");
  }
  const auto wall_start = std::chrono::steady_clock::now();
  ready = ready && test.cuda_ok(cudaEventRecord(execution.start(), stream),
                                label + " record start");
  for (int iteration = 0; ready && iteration < iterations; ++iteration) {
    ready = test.cuda_ok(launch_variant(fixture, stream, variant),
                         label + " measured launch") &&
            ready;
  }
  ready = ready && test.cuda_ok(cudaEventRecord(execution.stop(), stream),
                                label + " record stop");
  ready = ready && test.cuda_ok(cudaEventSynchronize(execution.stop()),
                                label + " stop synchronize");
  const auto wall_stop = std::chrono::steady_clock::now();
  if (profiler_range) {
    ready = test.cuda_ok(cudaProfilerStop(), label + " profiler stop") &&
            ready;
  }
  float total_ms = std::numeric_limits<float>::quiet_NaN();
  ready = ready && test.cuda_ok(
                       cudaEventElapsedTime(&total_ms, execution.start(),
                                            execution.stop()),
                       label + " elapsed");
  const float average =
      ready && iterations > 0
          ? total_ms / static_cast<float>(iterations)
          : std::numeric_limits<float>::quiet_NaN();
  const double wall_ms =
      std::chrono::duration<double, std::milli>(wall_stop - wall_start)
          .count();
  std::cout << "FP8_REGISTER_FEED_PASS: label=" << label
            << " variant=" << variant_name(variant)
            << " warmups=" << warmups << " iterations=" << iterations
            << " average_ms=" << average
            << " host_wall_average_ms="
            << (iterations > 0 ? wall_ms / static_cast<double>(iterations)
                               : std::numeric_limits<double>::quiet_NaN())
            << " l2_scrub_bytes_outside_timing=" << kScrubBytes
            << " profiler_range=" << (profiler_range ? "true" : "false")
            << '\n';
  return average;
}

[[nodiscard]] bool run_screen(TestContext& test, Fixture& fixture,
                              const Execution& execution,
                              const Variant candidate_variant,
                              const bool assert_gate = true) {
  double baseline_sum = 0.0;
  double candidate_sum = 0.0;
  bool every_round = true;
  for (int round = 0; round < kRounds; ++round) {
    const std::string prefix =
        "round_" + std::to_string(round + 1) + '_';
    const float b1 = measure_pass(test, fixture, execution,
                                  Variant::kBaseline, prefix + "B1");
    const float c1 = measure_pass(test, fixture, execution,
                                  candidate_variant, prefix + "C1");
    const float c2 = measure_pass(test, fixture, execution,
                                  candidate_variant, prefix + "C2");
    const float b2 = measure_pass(test, fixture, execution,
                                  Variant::kBaseline, prefix + "B2");
    const bool finite = std::isfinite(b1) && std::isfinite(c1) &&
                        std::isfinite(c2) && std::isfinite(b2) && b1 > 0.0F &&
                        c1 > 0.0F && c2 > 0.0F && b2 > 0.0F;
    const double speedup =
        finite ? static_cast<double>(b1 + b2) /
                     static_cast<double>(c1 + c2)
               : std::numeric_limits<double>::quiet_NaN();
    every_round =
        every_round && finite && speedup >= kRequiredRoundSpeedup;
    if (finite) {
      baseline_sum += static_cast<double>(b1 + b2);
      candidate_sum += static_cast<double>(c1 + c2);
    }
    std::cout << "PERF_FP8_REGISTER_FEED_ROUND:";
    if (variant_is_test_only(candidate_variant)) {
      std::cout << " variant=" << variant_name(candidate_variant)
                << " test_only=true";
    }
    std::cout << " round=" << round + 1
              << " order=B-C-C-B B1_ms=" << b1 << " C1_ms=" << c1
              << " C2_ms=" << c2 << " B2_ms=" << b2
              << " speedup=" << speedup
              << " required_round_speedup=" << kRequiredRoundSpeedup
              << " gate="
              << (finite && speedup >= kRequiredRoundSpeedup ? "PASS"
                                                            : "FAIL")
              << '\n';
  }
  const bool sums_valid = baseline_sum > 0.0 && candidate_sum > 0.0;
  const double baseline_ms =
      sums_valid ? baseline_sum / (2.0 * kRounds)
                 : std::numeric_limits<double>::quiet_NaN();
  const double candidate_ms =
      sums_valid ? candidate_sum / (2.0 * kRounds)
                 : std::numeric_limits<double>::quiet_NaN();
  const double speedup =
      sums_valid ? baseline_sum / candidate_sum
                 : std::numeric_limits<double>::quiet_NaN();
  const bool gate = every_round && std::isfinite(speedup) &&
                    speedup >= kRequiredAggregateSpeedup;
  std::cout << "PERF_FP8_REGISTER_FEED_AGGREGATE:";
  if (variant_is_test_only(candidate_variant)) {
    std::cout << " variant=" << variant_name(candidate_variant)
              << " test_only=true";
  }
  std::cout << " baseline_ms=" << baseline_ms
            << " candidate_ms=" << candidate_ms
            << " speedup=" << speedup
            << " required_aggregate_speedup="
            << kRequiredAggregateSpeedup
            << " required_minimum_round_speedup="
            << kRequiredRoundSpeedup
            << " every_round_gate=" << (every_round ? "true" : "false")
            << " rounds=" << kRounds
            << " warmups_per_pass=" << kWarmups
            << " iterations_per_pass=" << kIterations
            << " order=B-C-C-B"
            << " logical_tensor_bytes_and_hmma_unchanged=true"
            << " measured_global_requests_may_change=true"
            << " fragment_native_sidecar_equal_bytes="
            << (candidate_variant == Variant::kCandidate ? "true" : "false")
            << " canonical_weight_no_sidecar="
            << (candidate_variant == Variant::kCandidate ? "false" : "true")
            << " triple_cp_async_pipeline=true"
            << " decoded_B_shared_round_trip_removed=true"
            << " register_fragment_feed=true"
            << " gate=" << (gate ? "PASS" : "FAIL") << '\n';
  if (assert_gate) {
    test.expect(gate,
                std::string(variant_name(candidate_variant)) +
                    " clears frozen C512 QKV performance gate");
  }
  return gate;
}

enum class Mode {
  kValidate,
  kValidateCanonical,
  kScreen,
  kScreenCanonical,
  kMeasureBaseline,
  kMeasureCandidate,
  kProfileBaseline,
  kProfileCandidate,
};

struct Options {
  Mode mode = Mode::kValidate;
  std::string checkpoint_directory;
};

[[nodiscard]] bool parse_options(const int argc, char** argv,
                                 Options& options) {
  bool checkpoint_seen = false;
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    if (argument == "--mode=validate") {
      options.mode = Mode::kValidate;
    } else if (argument == "--mode=validate-canonical") {
      options.mode = Mode::kValidateCanonical;
    } else if (argument == "--mode=screen") {
      options.mode = Mode::kScreen;
    } else if (argument == "--mode=screen-canonical") {
      options.mode = Mode::kScreenCanonical;
    } else if (argument == "--mode=measure-baseline") {
      options.mode = Mode::kMeasureBaseline;
    } else if (argument == "--mode=measure-candidate") {
      options.mode = Mode::kMeasureCandidate;
    } else if (argument == "--mode=profile-baseline") {
      options.mode = Mode::kProfileBaseline;
    } else if (argument == "--mode=profile-candidate") {
      options.mode = Mode::kProfileCandidate;
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
    } else {
      std::cerr << "unknown argument: " << argument << '\n';
      return false;
    }
  }
  return true;
}

[[nodiscard]] const char* mode_name(const Mode mode) noexcept {
  switch (mode) {
    case Mode::kValidate:
      return "validate";
    case Mode::kValidateCanonical:
      return "validate_canonical";
    case Mode::kScreen:
      return "screen";
    case Mode::kScreenCanonical:
      return "screen_canonical";
    case Mode::kMeasureBaseline:
      return "measure_baseline";
    case Mode::kMeasureCandidate:
      return "measure_candidate";
    case Mode::kProfileBaseline:
      return "profile_baseline";
    case Mode::kProfileCandidate:
      return "profile_candidate";
  }
  return "unknown";
}

}  // namespace

int main(const int argc, char** argv) {
  Options options{};
  if (!parse_options(argc, argv, options)) {
    return 2;
  }
  TestContext test;
  CheckpointPayload checkpoint_payload;
  const CheckpointPayload* selected_checkpoint = nullptr;
  if (!options.checkpoint_directory.empty()) {
    if (!load_checkpoint_payload(test, options.checkpoint_directory,
                                 checkpoint_payload)) {
      return 1;
    }
    selected_checkpoint = &checkpoint_payload;
    std::cout << "FP8_REGISTER_FEED_PAYLOAD: payload=checkpoint"
              << " tensor=" << checkpoint_payload.tensor
              << " shard=" << checkpoint_payload.shard
              << " bytes=" << checkpoint_payload.weights.size()
              << " checkpoint_read_only=true\n";
  } else {
    std::cout << "FP8_REGISTER_FEED_PAYLOAD: payload=synthetic"
              << " tensor=none shard=none bytes=" << kWeightBytes << '\n';
  }
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status != cudaSuccess || device_count == 0) {
    std::cout << "SKIP: FP8 register-feed screen requires CUDA\n";
    (void)cudaGetLastError();
    return 77;
  }
  cudaDeviceProp properties{};
  if (!test.cuda_ok(cudaGetDeviceProperties(&properties, 0),
                    "read CUDA device properties")) {
    return 1;
  }
  if (properties.major != 8 || properties.minor != 7) {
    std::cout << "SKIP: FP8 register-feed requires SM87; got sm_"
              << properties.major << properties.minor << '\n';
    return 77;
  }
  std::cout << std::fixed << std::setprecision(6)
            << "FP8_REGISTER_FEED_DEVICE: name=" << properties.name
            << " sm=" << properties.major << properties.minor
            << " sm_count=" << properties.multiProcessorCount
            << " shared_per_sm=" << properties.sharedMemPerMultiprocessor
            << " mode=" << mode_name(options.mode)
            << " payload="
            << (selected_checkpoint == nullptr ? "synthetic" : "checkpoint")
            << " tokens=" << kTokens << " rows=" << kRows
            << " columns=" << kColumns
            << " production_launcher=true"
            << " production_gpu_pack=true\n";

  Execution execution;
  if (!execution.create(test)) {
    return 1;
  }
  bool ready = run_resource_gate(test, Variant::kCandidate);
  ready = run_invalid_graph_gate(test, execution.stream(),
                                 Variant::kCandidate) &&
          ready;
  ready = run_pack_invalid_graph_gate(test, execution.stream()) && ready;
  const bool canonical_mode = options.mode == Mode::kValidateCanonical ||
                              options.mode == Mode::kScreenCanonical;
  if (canonical_mode) {
    std::cout << "FP8_REGISTER_FEED_CANONICAL_TEST_SCOPE: variants="
                 "canonical_p0,canonical_p1_xor test_only=true "
                 "p1_delta=xor_only_u16_gather "
                 "production_dispatch_unchanged=true\n";
    const bool p0_resource =
        run_resource_gate(test, Variant::kCanonicalP0);
    const bool p1_resource =
        run_resource_gate(test, Variant::kCanonicalP1);
    const bool p0_invalid = run_invalid_graph_gate(
        test, execution.stream(), Variant::kCanonicalP0);
    const bool p1_invalid = run_invalid_graph_gate(
        test, execution.stream(), Variant::kCanonicalP1);
    ready = p0_resource && p1_resource && p0_invalid && p1_invalid && ready;
  }
  if (!ready) {
    return 1;
  }
  Fixture fixture;
  if (!fixture.initialize(test, execution.stream(), selected_checkpoint)) {
    return 1;
  }
  const bool correct = run_correctness(test, fixture, execution.stream(),
                                       Variant::kCandidate);
  bool canonical_functions_distinct = true;
  bool p0_correct = true;
  bool p1_correct = true;
  if (canonical_mode) {
    canonical_functions_distinct = run_canonical_function_identity_gate(
        test, fixture, execution.stream());
    p0_correct = run_correctness(test, fixture, execution.stream(),
                                 Variant::kCanonicalP0);
    p1_correct = run_correctness(test, fixture, execution.stream(),
                                 Variant::kCanonicalP1);
  }
  if (correct && canonical_functions_distinct && p0_correct && p1_correct) {
    switch (options.mode) {
      case Mode::kValidate:
      case Mode::kValidateCanonical:
        break;
      case Mode::kScreen:
        (void)run_screen(test, fixture, execution, Variant::kCandidate);
        break;
      case Mode::kScreenCanonical: {
        const bool p0_gate = run_screen(test, fixture, execution,
                                        Variant::kCanonicalP0, false);
        std::cout
            << "PERF_FP8_REGISTER_FEED_DISPOSITION: variant="
            << variant_name(Variant::kCanonicalP0)
            << " test_only=true promotion=REJECTED "
               "performance_gate_asserted=false observed_gate="
            << (p0_gate ? "PASS" : "FAIL") << '\n';
        const bool p1_gate = run_screen(test, fixture, execution,
                                        Variant::kCanonicalP1);
        std::cout
            << "PERF_FP8_REGISTER_FEED_DISPOSITION: variant="
            << variant_name(Variant::kCanonicalP1)
            << " test_only=true promotion="
            << (p1_gate ? "BASELINE_GATE_ADMITTED" : "REJECTED")
            << " performance_gate_asserted=true observed_gate="
            << (p1_gate ? "PASS" : "FAIL") << '\n';
        break;
      }
      case Mode::kMeasureBaseline:
        (void)measure_pass(test, fixture, execution, Variant::kBaseline,
                           "standalone_baseline");
        break;
      case Mode::kMeasureCandidate:
        (void)measure_pass(test, fixture, execution, Variant::kCandidate,
                           "standalone_candidate");
        break;
      case Mode::kProfileBaseline:
      case Mode::kProfileCandidate: {
        const Variant variant = options.mode == Mode::kProfileBaseline
                                    ? Variant::kBaseline
                                    : Variant::kCandidate;
        const float milliseconds = measure_pass(
            test, fixture, execution, variant, "single_profile", 0, 1, true);
        std::cout << "FP8_REGISTER_FEED_PROFILE_MARKER: mode="
                  << mode_name(options.mode)
                  << " milliseconds=" << milliseconds
                  << " profiler_range_kernel_launches=1"
                  << " scrub_launches_in_range=0\n";
        break;
      }
    }
  }
  if (test.failures() != 0) {
    std::cerr << test.failures()
              << " FP8 register-feed assertion(s) failed\n";
    return 1;
  }
  std::cout << "FP8 QKV register-feed SM87 screen passed\n";
  return 0;
}
