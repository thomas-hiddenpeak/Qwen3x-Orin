#include "q3x/kernels/sm87_fp8_prefill_supermatrix.h"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kTokens = 512U;
constexpr std::size_t kRows = 5'120U;
constexpr std::size_t kColumns = 6'144U;
constexpr std::size_t kActiveColumnsPerToken = 12U;
constexpr std::size_t kWeightBytes = kRows * kColumns;
constexpr std::size_t kActivationElements = kTokens * kColumns;
constexpr std::size_t kOutputElements = kTokens * kRows;
constexpr std::size_t kByteGuard = 64U;
constexpr std::size_t kOutputGuard = 32U;
constexpr std::uint8_t kByteGuardValue = 0xcdU;
constexpr std::uint16_t kOutputGuardValue = 0xa5a5U;
constexpr float kWeightScale = 0.5F;

static_assert(kActiveColumnsPerToken * kTokens == kColumns);
static_assert((kRows % 256U) == 0U);
static_assert((kColumns % 64U) == 0U);

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

  void expect_invalid(const int status, const std::string& operation) {
    expect(status == static_cast<int>(cudaErrorInvalidValue),
           operation + " returns cudaErrorInvalidValue (actual=" +
               std::to_string(status) + ")");
  }

  [[nodiscard]] int failures() const noexcept { return failures_; }

 private:
  int failures_ = 0;
};

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

  [[nodiscard]] cudaError_t allocate(const std::size_t count) {
    return cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T));
  }

  [[nodiscard]] T* get() noexcept { return data_; }
  [[nodiscard]] const T* get() const noexcept { return data_; }

 private:
  T* data_ = nullptr;
};

class StreamOwner {
 public:
  StreamOwner() = default;
  StreamOwner(const StreamOwner&) = delete;
  StreamOwner& operator=(const StreamOwner&) = delete;

  ~StreamOwner() {
    if (stream_ != nullptr) {
      (void)cudaStreamDestroy(stream_);
    }
  }

  [[nodiscard]] cudaError_t create() {
    return cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
  }

  [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

[[nodiscard]] std::uint32_t mix32(std::uint32_t value) noexcept {
  value ^= value >> 16U;
  value *= 0x7feb'352dU;
  value ^= value >> 15U;
  value *= 0x846c'a68bU;
  value ^= value >> 16U;
  return value;
}

[[nodiscard]] std::uint32_t weight_hash(const std::size_t row,
                                        const std::size_t column) noexcept {
  return mix32(static_cast<std::uint32_t>(row) * 0x9e37'79b9U ^
               static_cast<std::uint32_t>(column) * 0x85eb'ca6bU ^
               0xc2b2'ae35U);
}

[[nodiscard]] std::uint8_t weight_code(const std::size_t row,
                                       const std::size_t column) noexcept {
  const std::uint32_t hash = weight_hash(row, column);
  const std::uint8_t magnitude = static_cast<std::uint8_t>(
      0x30U + 8U * ((hash >> 1U) & 3U));
  return static_cast<std::uint8_t>(magnitude |
                                   ((hash & 1U) != 0U ? 0x80U : 0U));
}

[[nodiscard]] float exact_weight_value(const std::uint8_t code) noexcept {
  const unsigned int level = (static_cast<unsigned int>(code & 0x78U) -
                              0x30U) /
                             8U;
  const float magnitude = std::ldexp(0.5F, static_cast<int>(level));
  return (code & 0x80U) != 0U ? -magnitude : magnitude;
}

[[nodiscard]] float activation_value(const std::size_t token,
                                     const std::size_t group) noexcept {
  const std::uint32_t hash = mix32(
      static_cast<std::uint32_t>(token) * 0x27d4'eb2dU ^
      static_cast<std::uint32_t>(group) * 0x1656'67b1U ^ 0xd3a2'646cU);
  return (hash & 1U) != 0U ? -1.0F : 1.0F;
}

[[nodiscard]] std::uint16_t bf16_bits(const float value) noexcept {
  std::uint32_t bits = 0U;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t magnitude = bits & 0x7fff'ffffU;
  if (magnitude > 0x7f80'0000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] bool device_is_sm87(TestContext& test) {
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status == cudaErrorNoDevice ||
      count_status == cudaErrorInsufficientDriver || device_count == 0) {
    (void)cudaGetLastError();
    std::cout << "SKIP: no CUDA device is available\n";
    return false;
  }
  if (!test.cuda_ok(count_status, "cudaGetDeviceCount")) {
    return false;
  }
  int device = 0;
  if (!test.cuda_ok(cudaGetDevice(&device), "cudaGetDevice")) {
    return false;
  }
  cudaDeviceProp properties{};
  if (!test.cuda_ok(cudaGetDeviceProperties(&properties, device),
                    "cudaGetDeviceProperties")) {
    return false;
  }
  if (properties.major != 8 || properties.minor != 7) {
    std::cout << "SKIP: test requires SM87, found sm" << properties.major
              << properties.minor << '\n';
    return false;
  }
  return true;
}

[[nodiscard]] bool run_exact_c512_projection(TestContext& test) {
  std::vector<std::uint8_t> host_weights(kWeightBytes);
  for (std::size_t row = 0U; row < kRows; ++row) {
    for (std::size_t column = 0U; column < kColumns; ++column) {
      host_weights[row * kColumns + column] = weight_code(row, column);
    }
  }

  std::vector<std::uint16_t> host_activations(kActivationElements, 0U);
  for (std::size_t token = 0U; token < kTokens; ++token) {
    for (std::size_t group = 0U; group < kActiveColumnsPerToken; ++group) {
      const std::size_t column = group * kTokens + token;
      host_activations[token * kColumns + column] =
          bf16_bits(activation_value(token, group));
    }
  }

  DeviceBuffer<std::uint8_t> canonical_storage;
  DeviceBuffer<std::uint8_t> sidecar_storage;
  DeviceBuffer<std::uint16_t> activation_storage;
  DeviceBuffer<std::uint16_t> output_storage;
  StreamOwner stream;
  if (!test.cuda_ok(canonical_storage.allocate(kWeightBytes +
                                                2U * kByteGuard),
                    "allocate canonical weights") ||
      !test.cuda_ok(sidecar_storage.allocate(kWeightBytes +
                                              2U * kByteGuard),
                    "allocate sidecar weights") ||
      !test.cuda_ok(activation_storage.allocate(kActivationElements +
                                                 2U * kOutputGuard),
                    "allocate activations") ||
      !test.cuda_ok(output_storage.allocate(kOutputElements +
                                             2U * kOutputGuard),
                    "allocate output") ||
      !test.cuda_ok(stream.create(), "create nonblocking stream")) {
    return false;
  }

  std::uint8_t* const canonical = canonical_storage.get() + kByteGuard;
  std::uint8_t* const sidecar = sidecar_storage.get() + kByteGuard;
  std::uint16_t* const activations =
      activation_storage.get() + kOutputGuard;
  std::uint16_t* const output = output_storage.get() + kOutputGuard;

  if (!test.cuda_ok(cudaMemsetAsync(canonical_storage.get(),
                                    kByteGuardValue,
                                    kWeightBytes + 2U * kByteGuard,
                                    stream.get()),
                    "initialize canonical guards") ||
      !test.cuda_ok(cudaMemsetAsync(sidecar_storage.get(), kByteGuardValue,
                                    kWeightBytes + 2U * kByteGuard,
                                    stream.get()),
                    "initialize sidecar guards") ||
      !test.cuda_ok(cudaMemsetAsync(output_storage.get(), 0xa5,
                                    (kOutputElements + 2U * kOutputGuard) *
                                        sizeof(std::uint16_t),
                                    stream.get()),
                    "initialize output guards") ||
      !test.cuda_ok(cudaMemcpyAsync(canonical, host_weights.data(),
                                    kWeightBytes, cudaMemcpyHostToDevice,
                                    stream.get()),
                    "copy canonical weights") ||
      !test.cuda_ok(cudaMemcpyAsync(activations, host_activations.data(),
                                    kActivationElements *
                                        sizeof(std::uint16_t),
                                    cudaMemcpyHostToDevice, stream.get()),
                    "copy activations")) {
    return false;
  }

  const int pack_status =
      q3x::kernels::launch_sm87_fp8_prefill_supermatrix_pack_cuda(
          canonical, sidecar, kRows, kColumns, stream.get());
  test.expect(pack_status == static_cast<int>(cudaSuccess),
              "supermatrix pack launch succeeds (actual=" +
                  std::to_string(pack_status) + ")");
  if (pack_status != static_cast<int>(cudaSuccess)) {
    return false;
  }

  const q3x::kernels::Sm87Fp8PrefillSupermatrixPartition partition{
      sidecar, kWeightScale, kRows, output};
  const int gemm_status =
      q3x::kernels::launch_sm87_fp8_prefill_supermatrix_gemm_bf16_cuda(
          &partition, 1U, activations, kTokens, kColumns, stream.get());
  test.expect(gemm_status == static_cast<int>(cudaSuccess),
              "supermatrix GEMM launch succeeds (actual=" +
                  std::to_string(gemm_status) + ")");
  if (gemm_status != static_cast<int>(cudaSuccess)) {
    return false;
  }

  std::vector<std::uint16_t> host_output(kOutputElements);
  std::array<std::uint8_t, kByteGuard> sidecar_prefix{};
  std::array<std::uint8_t, kByteGuard> sidecar_suffix{};
  std::array<std::uint16_t, kOutputGuard> output_prefix{};
  std::array<std::uint16_t, kOutputGuard> output_suffix{};
  if (!test.cuda_ok(cudaMemcpyAsync(host_output.data(), output,
                                    kOutputElements * sizeof(std::uint16_t),
                                    cudaMemcpyDeviceToHost, stream.get()),
                    "copy GEMM output") ||
      !test.cuda_ok(cudaMemcpyAsync(sidecar_prefix.data(),
                                    sidecar_storage.get(), kByteGuard,
                                    cudaMemcpyDeviceToHost, stream.get()),
                    "copy sidecar prefix guard") ||
      !test.cuda_ok(cudaMemcpyAsync(sidecar_suffix.data(),
                                    sidecar + kWeightBytes, kByteGuard,
                                    cudaMemcpyDeviceToHost, stream.get()),
                    "copy sidecar suffix guard") ||
      !test.cuda_ok(cudaMemcpyAsync(output_prefix.data(),
                                    output_storage.get(),
                                    kOutputGuard * sizeof(std::uint16_t),
                                    cudaMemcpyDeviceToHost, stream.get()),
                    "copy output prefix guard") ||
      !test.cuda_ok(cudaMemcpyAsync(output_suffix.data(),
                                    output + kOutputElements,
                                    kOutputGuard * sizeof(std::uint16_t),
                                    cudaMemcpyDeviceToHost, stream.get()),
                    "copy output suffix guard") ||
      !test.cuda_ok(cudaStreamSynchronize(stream.get()),
                    "synchronize pack and GEMM")) {
    return false;
  }

  for (const std::uint8_t value : sidecar_prefix) {
    test.expect(value == kByteGuardValue,
                "pack preserves sidecar prefix guard");
  }
  for (const std::uint8_t value : sidecar_suffix) {
    test.expect(value == kByteGuardValue,
                "pack preserves sidecar suffix guard");
  }
  for (const std::uint16_t value : output_prefix) {
    test.expect(value == kOutputGuardValue,
                "GEMM preserves output prefix guard");
  }
  for (const std::uint16_t value : output_suffix) {
    test.expect(value == kOutputGuardValue,
                "GEMM preserves output suffix guard");
  }

  std::size_t mismatch_count = 0U;
  constexpr std::size_t kReportedMismatches = 12U;
  for (std::size_t token = 0U; token < kTokens; ++token) {
    for (std::size_t row = 0U; row < kRows; ++row) {
      float dot = 0.0F;
      for (std::size_t group = 0U; group < kActiveColumnsPerToken;
           ++group) {
        const std::size_t column = group * kTokens + token;
        dot += activation_value(token, group) *
               exact_weight_value(
                   host_weights[row * kColumns + column]);
      }
      const std::uint16_t expected = bf16_bits(dot * kWeightScale);
      const std::uint16_t actual = host_output[token * kRows + row];
      if (actual != expected) {
        if (mismatch_count < kReportedMismatches) {
          std::cerr << "MISMATCH: token=" << token << " row=" << row
                    << " expected_bits=" << expected
                    << " actual_bits=" << actual << '\n';
        }
        ++mismatch_count;
      }
    }
  }
  test.expect(mismatch_count == 0U,
              "complete C512 K6144/N5120 output is bitwise exact; " +
                  std::to_string(mismatch_count) + " mismatches");

  // Keep guard probes after the positive path. Every pointer used by an
  // overlap case still denotes enough allocated storage should a regression
  // accidentally enqueue work, making failures deterministic rather than
  // relying on fabricated addresses.
  test.expect_invalid(
      q3x::kernels::launch_sm87_fp8_prefill_supermatrix_pack_cuda(
          nullptr, sidecar, kRows, kColumns, stream.get()),
      "pack rejects null canonical weights");
  test.expect_invalid(
      q3x::kernels::launch_sm87_fp8_prefill_supermatrix_pack_cuda(
          canonical, nullptr, kRows, kColumns, stream.get()),
      "pack rejects null sidecar weights");
  test.expect_invalid(
      q3x::kernels::launch_sm87_fp8_prefill_supermatrix_pack_cuda(
          canonical, sidecar, kRows - 1U, kColumns, stream.get()),
      "pack rejects non-N256 rows");
  test.expect_invalid(
      q3x::kernels::launch_sm87_fp8_prefill_supermatrix_pack_cuda(
          canonical, sidecar, kRows, kColumns - 1U, stream.get()),
      "pack rejects unsupported columns");
  test.expect_invalid(
      q3x::kernels::launch_sm87_fp8_prefill_supermatrix_pack_cuda(
          canonical + 1U, sidecar, kRows, kColumns, stream.get()),
      "pack rejects misaligned canonical weights");
  test.expect_invalid(
      q3x::kernels::launch_sm87_fp8_prefill_supermatrix_pack_cuda(
          canonical, sidecar + 1U, kRows, kColumns, stream.get()),
      "pack rejects misaligned sidecar weights");
  test.expect_invalid(
      q3x::kernels::launch_sm87_fp8_prefill_supermatrix_pack_cuda(
          canonical, canonical, kRows, kColumns, stream.get()),
      "pack rejects source/destination overlap");
  const std::size_t overflowing_rows =
      ((std::numeric_limits<std::size_t>::max() / kColumns) / 256U + 1U) *
      256U;
  const auto* const fake_canonical =
      reinterpret_cast<const std::uint8_t*>(std::uintptr_t{0x1000'0000'0000U});
  auto* const fake_sidecar =
      reinterpret_cast<std::uint8_t*>(std::uintptr_t{0x2000'0000'0000U});
  test.expect_invalid(
      q3x::kernels::launch_sm87_fp8_prefill_supermatrix_pack_cuda(
          fake_canonical, fake_sidecar, overflowing_rows, kColumns,
          stream.get()),
      "pack rejects rows*columns overflow");
  const std::size_t rows_beyond_uint =
      (static_cast<std::size_t>(
           std::numeric_limits<unsigned int>::max()) /
           256U +
       1U) *
      256U;
  test.expect_invalid(
      q3x::kernels::launch_sm87_fp8_prefill_supermatrix_pack_cuda(
          fake_canonical, fake_sidecar, rows_beyond_uint, kColumns,
          stream.get()),
      "pack rejects rows beyond the unsigned launch/index domain");

  std::array<q3x::kernels::Sm87Fp8PrefillSupermatrixPartition, 4U>
      partitions{{partition, partition, partition, partition}};
  test.expect_invalid(
      q3x::kernels::launch_sm87_fp8_prefill_supermatrix_gemm_bf16_cuda(
          nullptr, 1U, activations, kTokens, kColumns, stream.get()),
      "GEMM rejects null descriptor array");
  test.expect_invalid(
      q3x::kernels::launch_sm87_fp8_prefill_supermatrix_gemm_bf16_cuda(
          partitions.data(), 0U, activations, kTokens, kColumns,
          stream.get()),
      "GEMM rejects zero partitions");
  test.expect_invalid(
      q3x::kernels::launch_sm87_fp8_prefill_supermatrix_gemm_bf16_cuda(
          partitions.data(), 4U, activations, kTokens, kColumns,
          stream.get()),
      "GEMM rejects more than three partitions");
  test.expect_invalid(
      q3x::kernels::launch_sm87_fp8_prefill_supermatrix_gemm_bf16_cuda(
          partitions.data(), 1U, activations, kTokens - 1U, kColumns,
          stream.get()),
      "GEMM rejects non-C512 token count");
  test.expect_invalid(
      q3x::kernels::launch_sm87_fp8_prefill_supermatrix_gemm_bf16_cuda(
          partitions.data(), 1U, activations, kTokens, kColumns - 1U,
          stream.get()),
      "GEMM rejects unsupported columns");

  auto invalid_partition = partition;
  invalid_partition.rows = kRows - 256U;
  test.expect_invalid(
      q3x::kernels::launch_sm87_fp8_prefill_supermatrix_gemm_bf16_cuda(
          &invalid_partition, 1U, activations, kTokens, kColumns,
          stream.get()),
      "GEMM rejects unsupported partition topology");
  invalid_partition = partition;
  invalid_partition.register_feed_sidecar = nullptr;
  test.expect_invalid(
      q3x::kernels::launch_sm87_fp8_prefill_supermatrix_gemm_bf16_cuda(
          &invalid_partition, 1U, activations, kTokens, kColumns,
          stream.get()),
      "GEMM rejects null sidecar");
  invalid_partition = partition;
  invalid_partition.output = nullptr;
  test.expect_invalid(
      q3x::kernels::launch_sm87_fp8_prefill_supermatrix_gemm_bf16_cuda(
          &invalid_partition, 1U, activations, kTokens, kColumns,
          stream.get()),
      "GEMM rejects null output");
  invalid_partition = partition;
  invalid_partition.weight_scale = -1.0F;
  test.expect_invalid(
      q3x::kernels::launch_sm87_fp8_prefill_supermatrix_gemm_bf16_cuda(
          &invalid_partition, 1U, activations, kTokens, kColumns,
          stream.get()),
      "GEMM rejects negative scale");
  invalid_partition = partition;
  invalid_partition.weight_scale =
      std::numeric_limits<float>::quiet_NaN();
  test.expect_invalid(
      q3x::kernels::launch_sm87_fp8_prefill_supermatrix_gemm_bf16_cuda(
          &invalid_partition, 1U, activations, kTokens, kColumns,
          stream.get()),
      "GEMM rejects non-finite scale");
  invalid_partition = partition;
  invalid_partition.register_feed_sidecar = sidecar + 1U;
  test.expect_invalid(
      q3x::kernels::launch_sm87_fp8_prefill_supermatrix_gemm_bf16_cuda(
          &invalid_partition, 1U, activations, kTokens, kColumns,
          stream.get()),
      "GEMM rejects misaligned sidecar");
  invalid_partition = partition;
  invalid_partition.output = output + 1U;
  test.expect_invalid(
      q3x::kernels::launch_sm87_fp8_prefill_supermatrix_gemm_bf16_cuda(
          &invalid_partition, 1U, activations, kTokens, kColumns,
          stream.get()),
      "GEMM rejects misaligned output");
  test.expect_invalid(
      q3x::kernels::launch_sm87_fp8_prefill_supermatrix_gemm_bf16_cuda(
          &partition, 1U, activations + 1U, kTokens, kColumns,
          stream.get()),
      "GEMM rejects misaligned activations");

  test.expect_invalid(
      q3x::kernels::launch_sm87_fp8_prefill_supermatrix_gemm_bf16_cuda(
          &partition, 1U,
          reinterpret_cast<const std::uint16_t*>(sidecar), kTokens,
          kColumns, stream.get()),
      "GEMM rejects sidecar/activation overlap");
  invalid_partition = partition;
  invalid_partition.output = reinterpret_cast<std::uint16_t*>(sidecar);
  test.expect_invalid(
      q3x::kernels::launch_sm87_fp8_prefill_supermatrix_gemm_bf16_cuda(
          &invalid_partition, 1U, activations, kTokens, kColumns,
          stream.get()),
      "GEMM rejects sidecar/output overlap");
  invalid_partition = partition;
  invalid_partition.output = activations;
  test.expect_invalid(
      q3x::kernels::launch_sm87_fp8_prefill_supermatrix_gemm_bf16_cuda(
          &invalid_partition, 1U, activations, kTokens, kColumns,
          stream.get()),
      "GEMM rejects activation/output overlap");

  // Exercise both cross-partition sidecar/output alias directions with a
  // supported K5120 two-partition topology. These aligned sentinel addresses
  // are range-checked entirely on the host and must never reach a CUDA launch.
  const auto* const fake_activations = reinterpret_cast<const std::uint16_t*>(
      std::uintptr_t{0x3000'0000'0000U});
  const auto* const first_sidecar = reinterpret_cast<const std::uint8_t*>(
      std::uintptr_t{0x4000'0000'0000U});
  auto* const first_output = reinterpret_cast<std::uint16_t*>(
      std::uintptr_t{0x5000'0000'0000U});
  const auto* const second_sidecar = reinterpret_cast<const std::uint8_t*>(
      std::uintptr_t{0x6000'0000'0000U});
  auto* const second_output = reinterpret_cast<std::uint16_t*>(
      std::uintptr_t{0x7000'0000'0000U});
  std::array<q3x::kernels::Sm87Fp8PrefillSupermatrixPartition, 2U>
      cross_partitions{{
          {first_sidecar, 1.0F, 10'240U, first_output},
          {reinterpret_cast<const std::uint8_t*>(first_output), 1.0F,
           6'144U, second_output},
      }};
  test.expect_invalid(
      q3x::kernels::launch_sm87_fp8_prefill_supermatrix_gemm_bf16_cuda(
          cross_partitions.data(), cross_partitions.size(),
          fake_activations, kTokens, 5'120U, stream.get()),
      "GEMM rejects later sidecar overlapping an earlier output");
  cross_partitions[1].register_feed_sidecar = second_sidecar;
  cross_partitions[1].output =
      reinterpret_cast<std::uint16_t*>(
          const_cast<std::uint8_t*>(first_sidecar));
  test.expect_invalid(
      q3x::kernels::launch_sm87_fp8_prefill_supermatrix_gemm_bf16_cuda(
          cross_partitions.data(), cross_partitions.size(),
          fake_activations, kTokens, 5'120U, stream.get()),
      "GEMM rejects later output overlapping an earlier sidecar");

  return true;
}

}  // namespace

int main() {
  TestContext test;
  if (!device_is_sm87(test)) {
    return test.failures() == 0 ? 77 : 1;
  }
  (void)run_exact_c512_projection(test);
  if (test.failures() != 0) {
    std::cerr << test.failures()
              << " FP8 supermatrix v2 test assertion(s) failed\n";
    return 1;
  }
  std::cout << "FP8 supermatrix v2 CUDA correctness/guard test passed\n";
  return 0;
}
