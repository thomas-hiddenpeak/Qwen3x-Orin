#include "q3x/kernels/sm87_a4w4_attention_k256_m128n256.h"
#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace kernels = q3x::kernels;

[[nodiscard]] std::uint32_t float_bits(const float value) noexcept {
  std::uint32_t bits{};
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

[[nodiscard]] float bits_float(const std::uint32_t bits) noexcept {
  float value{};
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

[[nodiscard]] std::uint16_t encode_bf16(const float value) noexcept {
  std::uint32_t bits = float_bits(value);
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] float decode_bf16(const std::uint16_t bits) noexcept {
  return bits_float(static_cast<std::uint32_t>(bits) << 16U);
}

[[nodiscard]] float multiply_rn(const float first,
                                const float second) noexcept {
  volatile float result = first * second;
  return result;
}

[[nodiscard]] float divide_rn(const float first,
                              const float second) noexcept {
  volatile float result = first / second;
  return result;
}

[[nodiscard]] bool cuda_ok(const cudaError_t status,
                           const char* const operation) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << operation << " failed: " << cudaGetErrorString(status)
            << '\n';
  return false;
}

template <class T>
class DeviceBuffer final {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() { cudaFree(pointer_); }

  [[nodiscard]] bool allocate(const std::size_t count) {
    count_ = count;
    return cuda_ok(cudaMalloc(reinterpret_cast<void**>(&pointer_),
                              count * sizeof(T)),
                   "cudaMalloc");
  }

  [[nodiscard]] T* get() noexcept { return pointer_; }
  [[nodiscard]] const T* get() const noexcept { return pointer_; }
  [[nodiscard]] std::size_t size() const noexcept { return count_; }

 private:
  T* pointer_{};
  std::size_t count_{};
};

struct QuantizedA final {
  std::vector<std::uint8_t> packed;
  std::vector<std::uint16_t> scales;
  std::vector<std::int8_t> codes;
};

[[nodiscard]] std::int8_t clamp_code(const float value,
                                     const float scale) noexcept {
  const float quotient = divide_rn(value, scale);
  const int rounded = static_cast<int>(std::nearbyint(quotient));
  return static_cast<std::int8_t>(
      rounded < -7 ? -7 : rounded > 7 ? 7 : rounded);
}

[[nodiscard]] QuantizedA make_quantizer_oracle(
    const std::vector<std::uint16_t>& input,
    const std::size_t input_stride,
    const std::size_t logical_m,
    const std::size_t launch_m,
    const std::size_t k,
    const float clip_ratio) {
  const std::size_t groups = k / 256U;
  const std::size_t physical_groups = k / 64U;
  QuantizedA result;
  result.packed.assign(
      kernels::sm87_a4w4_attention_k256_packed_capacity_bytes(launch_m, k),
      0U);
  result.scales.assign(
      kernels::sm87_a4w4_attention_k256_scale_capacity_elements(launch_m, k),
      0U);
  result.codes.assign(launch_m * k, 0);
  for (std::size_t row = 0U; row < launch_m; ++row) {
    for (std::size_t group = 0U; group < groups; ++group) {
      float maximum = 0.0F;
      if (row < logical_m) {
        for (std::size_t inner = 0U; inner < 256U; ++inner) {
          maximum = std::fmax(
              maximum,
              std::fabs(decode_bf16(
                  input[row * input_stride + group * 256U + inner])));
        }
      }
      const float clipped_maximum = multiply_rn(maximum, clip_ratio);
      std::uint16_t scale_bits = encode_bf16(
          maximum == 0.0F ? 1.0F : divide_rn(clipped_maximum, 7.0F));
      float stored_scale = decode_bf16(scale_bits);
      if (maximum != 0.0F && stored_scale == 0.0F) {
        scale_bits = 1U;
        stored_scale = decode_bf16(scale_bits);
      }
      result.scales[kernels::sm87_a4w4_attention_k256_scale_offset(
          row, group, groups)] = scale_bits;
      for (std::size_t inner = 0U; inner < 256U; ++inner) {
        float value = row < logical_m
                          ? decode_bf16(input[row * input_stride +
                                              group * 256U + inner])
                          : 0.0F;
        value = std::fmin(std::fmax(value, -clipped_maximum),
                          clipped_maximum);
        result.codes[row * k + group * 256U + inner] =
            clamp_code(value, stored_scale);
      }
    }
    for (std::size_t physical = 0U; physical < physical_groups;
         ++physical) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        const std::size_t inner = physical * 64U + 2U * byte;
        const auto even = result.codes[row * k + inner];
        const auto odd = result.codes[row * k + inner + 1U];
        result.packed[kernels::sm87_a4w4_attention_k256_packed_offset(
            row, physical, byte, physical_groups)] =
            kernels::sm87_a4w4_pack_signed_pair(even, odd);
      }
    }
  }
  return result;
}

[[nodiscard]] std::int8_t b_code(const std::size_t projection,
                                 const std::size_t row,
                                 const std::size_t inner) noexcept {
  std::uint32_t mixed =
      static_cast<std::uint32_t>((projection + 1U) * 0x9e3779b9U) ^
      static_cast<std::uint32_t>(row * 0x85ebca6bU) ^
      static_cast<std::uint32_t>(inner * 0xc2b2ae35U + 0x27d4eb2dU);
  mixed ^= mixed >> 16U;
  mixed ^= mixed >> 7U;
  return static_cast<std::int8_t>(static_cast<int>(mixed & 15U) - 8);
}

struct HostProjection final {
  std::vector<std::uint8_t> packed;
  std::vector<std::uint16_t> scales;
  std::vector<std::uint16_t> output_initial;
  std::vector<std::uint16_t> expected;
};

[[nodiscard]] HostProjection make_projection(
    const std::size_t projection,
    const std::size_t m,
    const std::size_t n,
    const std::size_t k,
    const std::size_t output_stride,
    const QuantizedA& a,
    const std::uint16_t output_sentinel) {
  const std::size_t k256_groups = k / 256U;
  const std::size_t physical_groups = k / 64U;
  HostProjection result;
  result.packed.assign(
      kernels::sm87_a4w4_attention_k256_packed_capacity_bytes(n, k), 0U);
  result.scales.assign(
      kernels::sm87_a4w4_attention_k256_scale_capacity_elements(n, k), 0U);
  result.output_initial.assign(m * output_stride, output_sentinel);
  result.expected = result.output_initial;
  for (std::size_t row = 0U; row < n; ++row) {
    for (std::size_t group = 0U; group < k256_groups; ++group) {
      const float scale_value =
          0.0017F * static_cast<float>(
                        5U + ((projection * 13U + row * 7U + group * 11U) %
                              29U));
      result.scales[kernels::sm87_a4w4_attention_k256_scale_offset(
          row, group, k256_groups)] = encode_bf16(scale_value);
    }
    for (std::size_t physical = 0U; physical < physical_groups;
         ++physical) {
      for (std::size_t byte = 0U; byte < 32U; ++byte) {
        const std::size_t inner = physical * 64U + 2U * byte;
        result.packed[kernels::sm87_a4w4_attention_k256_packed_offset(
            row, physical, byte, physical_groups)] =
            kernels::sm87_a4w4_pack_signed_pair(
                b_code(projection, row, inner),
                b_code(projection, row, inner + 1U));
      }
    }
  }
  for (std::size_t row_m = 0U; row_m < m; ++row_m) {
    for (std::size_t row_n = 0U; row_n < n; ++row_n) {
      float accumulator = 0.0F;
      for (std::size_t group = 0U; group < k256_groups; ++group) {
        std::int32_t partial = 0;
        for (std::size_t inner = 0U; inner < 256U; ++inner) {
          partial += static_cast<std::int32_t>(
                         a.codes[row_m * k + group * 256U + inner]) *
                     static_cast<std::int32_t>(
                         b_code(projection, row_n,
                                group * 256U + inner));
        }
        const float a_scale = decode_bf16(
            a.scales[kernels::sm87_a4w4_attention_k256_scale_offset(
                row_m, group, k256_groups)]);
        const float b_scale = decode_bf16(
            result.scales[kernels::sm87_a4w4_attention_k256_scale_offset(
                row_n, group, k256_groups)]);
        accumulator = std::fma(
            static_cast<float>(partial), multiply_rn(a_scale, b_scale),
            accumulator);
      }
      result.expected[row_m * output_stride + row_n] =
          encode_bf16(accumulator);
    }
  }
  return result;
}

[[nodiscard]] bool all_equal(const std::vector<std::uint8_t>& storage,
                             const std::size_t begin,
                             const std::size_t end,
                             const std::uint8_t value) {
  return std::all_of(storage.begin() + static_cast<std::ptrdiff_t>(begin),
                     storage.begin() + static_cast<std::ptrdiff_t>(end),
                     [value](const std::uint8_t element) {
                       return element == value;
                     });
}

[[nodiscard]] bool run_quantizer_and_graph(
    const std::vector<std::uint16_t>& input_storage,
    const std::size_t input_guard,
    const std::size_t input_stride,
    const std::size_t logical_m,
    const std::size_t launch_m,
    const std::size_t k,
    const float clip_ratio,
    const QuantizedA& oracle,
    DeviceBuffer<std::uint16_t>& device_input_storage,
    DeviceBuffer<std::uint8_t>& device_packed_storage,
    DeviceBuffer<std::uint16_t>& device_scale_storage,
    const std::size_t packed_guard,
    const std::size_t scale_guard) {
  constexpr std::uint8_t kPackedSentinel = 0xa5U;
  constexpr std::uint16_t kScaleSentinel = 0x7fc1U;
  std::vector<std::uint8_t> packed_initial(
      2U * packed_guard + oracle.packed.size(), kPackedSentinel);
  std::vector<std::uint16_t> scale_initial(
      2U * scale_guard + oracle.scales.size(), kScaleSentinel);
  if (!device_input_storage.allocate(input_storage.size()) ||
      !device_packed_storage.allocate(packed_initial.size()) ||
      !device_scale_storage.allocate(scale_initial.size()) ||
      !cuda_ok(cudaMemcpy(device_input_storage.get(), input_storage.data(),
                          input_storage.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "copy quantizer input") ||
      !cuda_ok(cudaMemcpy(device_packed_storage.get(), packed_initial.data(),
                          packed_initial.size(), cudaMemcpyHostToDevice),
               "initialize packed guards") ||
      !cuda_ok(cudaMemcpy(device_scale_storage.get(), scale_initial.data(),
                          scale_initial.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "initialize scale guards")) {
    return false;
  }
  std::uint16_t* const device_input =
      device_input_storage.get() + input_guard;
  std::uint8_t* const device_packed =
      device_packed_storage.get() + packed_guard;
  std::uint16_t* const device_scales =
      device_scale_storage.get() + scale_guard;
  const int invalid = kernels::launch_sm87_a4_quantize_bf16_k256_cuda(
      device_input, input_stride, logical_m, launch_m - 128U, k,
      clip_ratio, device_packed, oracle.packed.size(), device_scales,
      oracle.scales.size());
  if (invalid != static_cast<int>(cudaErrorInvalidValue)) {
    std::cerr << "K256 quantizer accepted invalid padded M\n";
    return false;
  }
  const int status = kernels::launch_sm87_a4_quantize_bf16_k256_cuda(
      device_input, input_stride, logical_m, launch_m, k, clip_ratio,
      device_packed, oracle.packed.size(), device_scales,
      oracle.scales.size());
  if (!cuda_ok(static_cast<cudaError_t>(status), "launch K256 quantizer") ||
      !cuda_ok(cudaDeviceSynchronize(), "sync K256 quantizer")) {
    return false;
  }

  // Capture only after one warm launch, then replay the exact producer twice.
  cudaStream_t stream{};
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  bool graph_ok = cuda_ok(cudaStreamCreate(&stream), "create graph stream") &&
                  cuda_ok(cudaStreamBeginCapture(
                              stream, cudaStreamCaptureModeThreadLocal),
                          "begin K256 quantizer capture");
  if (graph_ok) {
    const int captured = kernels::launch_sm87_a4_quantize_bf16_k256_cuda(
        device_input, input_stride, logical_m, launch_m, k, clip_ratio,
        device_packed, oracle.packed.size(), device_scales,
        oracle.scales.size(), stream);
    graph_ok = cuda_ok(static_cast<cudaError_t>(captured),
                       "capture K256 quantizer launch") &&
               cuda_ok(cudaStreamEndCapture(stream, &graph),
                       "end K256 quantizer capture") &&
               cuda_ok(cudaGraphInstantiate(&executable, graph, nullptr,
                                            nullptr, 0U),
                       "instantiate K256 quantizer graph") &&
               cuda_ok(cudaGraphLaunch(executable, stream),
                       "first K256 quantizer replay") &&
               cuda_ok(cudaGraphLaunch(executable, stream),
                       "second K256 quantizer replay") &&
               cuda_ok(cudaStreamSynchronize(stream),
                       "sync K256 quantizer graph");
  }
  cudaGraphExecDestroy(executable);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
  if (!graph_ok) {
    return false;
  }

  std::vector<std::uint16_t> input_after(input_storage.size());
  std::vector<std::uint8_t> packed_after(packed_initial.size());
  std::vector<std::uint16_t> scale_after(scale_initial.size());
  if (!cuda_ok(cudaMemcpy(input_after.data(), device_input_storage.get(),
                          input_after.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "copy immutable quantizer input") ||
      !cuda_ok(cudaMemcpy(packed_after.data(), device_packed_storage.get(),
                          packed_after.size(), cudaMemcpyDeviceToHost),
               "copy quantizer packed result") ||
      !cuda_ok(cudaMemcpy(scale_after.data(), device_scale_storage.get(),
                          scale_after.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "copy quantizer scale result")) {
    return false;
  }
  const bool guards =
      all_equal(packed_after, 0U, packed_guard, kPackedSentinel) &&
      all_equal(packed_after, packed_guard + oracle.packed.size(),
                packed_after.size(), kPackedSentinel) &&
      std::all_of(scale_after.begin(),
                  scale_after.begin() +
                      static_cast<std::ptrdiff_t>(scale_guard),
                  [](const std::uint16_t value) {
                    return value == kScaleSentinel;
                  }) &&
      std::all_of(scale_after.begin() + static_cast<std::ptrdiff_t>(
                                         scale_guard + oracle.scales.size()),
                  scale_after.end(), [](const std::uint16_t value) {
                    return value == kScaleSentinel;
                  });
  const bool payload =
      std::equal(oracle.packed.begin(), oracle.packed.end(),
                 packed_after.begin() +
                     static_cast<std::ptrdiff_t>(packed_guard)) &&
      std::equal(oracle.scales.begin(), oracle.scales.end(),
                 scale_after.begin() +
                     static_cast<std::ptrdiff_t>(scale_guard));
  if (input_after != input_storage || !guards || !payload) {
    std::cerr << "K256 quantizer oracle/guard/immutable check failed\n";
    return false;
  }
  return true;
}

[[nodiscard]] bool run_fixed_zero_smoke(
    const kernels::Sm87A4W4AttentionK256Topology topology,
    const char* const label) {
  constexpr std::size_t kM = 128U;
  const kernels::Sm87A4W4AttentionK256Plan plan =
      kernels::sm87_a4w4_attention_k256_fixed_plan(topology, kM);
  const std::size_t projection_count =
      kernels::sm87_a4w4_attention_k256_fixed_projection_count(topology);
  if (plan.launch_ctas == 0U || projection_count == 0U) {
    std::cerr << label << ": invalid fixed plan\n";
    return false;
  }
  std::vector<std::uint8_t> host_a(
      kernels::sm87_a4w4_attention_k256_packed_capacity_bytes(
          kM, plan.input_size),
      0U);
  std::vector<std::uint16_t> host_a_scales(
      kernels::sm87_a4w4_attention_k256_scale_capacity_elements(
          kM, plan.input_size),
      encode_bf16(1.0F));
  DeviceBuffer<std::uint8_t> device_a;
  DeviceBuffer<std::uint16_t> device_a_scales;
  if (!device_a.allocate(host_a.size()) ||
      !device_a_scales.allocate(host_a_scales.size()) ||
      !cuda_ok(cudaMemcpy(device_a.get(), host_a.data(), host_a.size(),
                          cudaMemcpyHostToDevice),
               "copy fixed zero A") ||
      !cuda_ok(cudaMemcpy(device_a_scales.get(), host_a_scales.data(),
                          host_a_scales.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "copy fixed unit A scales")) {
    return false;
  }
  std::array<std::vector<std::uint8_t>, 3U> host_b;
  std::array<std::vector<std::uint16_t>, 3U> host_b_scales;
  std::array<std::vector<std::uint16_t>, 3U> host_output;
  std::array<DeviceBuffer<std::uint8_t>, 3U> device_b;
  std::array<DeviceBuffer<std::uint16_t>, 3U> device_b_scales;
  std::array<DeviceBuffer<std::uint16_t>, 3U> device_output;
  std::array<kernels::Sm87A4W4AttentionK256ProjectionView, 3U> views{};
  for (std::size_t projection = 0U; projection < projection_count;
       ++projection) {
    const std::size_t n =
        kernels::sm87_a4w4_attention_k256_fixed_projection_panels(
            topology, projection) *
        kernels::kSm87A4W4AttentionK256PanelN;
    const std::size_t stride = n + 2U;
    host_b[projection].assign(
        kernels::sm87_a4w4_attention_k256_packed_capacity_bytes(
            n, plan.input_size),
        0U);
    host_b_scales[projection].assign(
        kernels::sm87_a4w4_attention_k256_scale_capacity_elements(
            n, plan.input_size),
        encode_bf16(1.0F));
    host_output[projection].assign(kM * stride, 0x7fc1U);
    if (!device_b[projection].allocate(host_b[projection].size()) ||
        !device_b_scales[projection].allocate(
            host_b_scales[projection].size()) ||
        !device_output[projection].allocate(host_output[projection].size()) ||
        !cuda_ok(cudaMemcpy(device_b[projection].get(),
                            host_b[projection].data(),
                            host_b[projection].size(),
                            cudaMemcpyHostToDevice),
                 "copy fixed zero B") ||
        !cuda_ok(cudaMemcpy(device_b_scales[projection].get(),
                            host_b_scales[projection].data(),
                            host_b_scales[projection].size() *
                                sizeof(std::uint16_t),
                            cudaMemcpyHostToDevice),
                 "copy fixed unit B scales") ||
        !cuda_ok(cudaMemcpy(device_output[projection].get(),
                            host_output[projection].data(),
                            host_output[projection].size() *
                                sizeof(std::uint16_t),
                            cudaMemcpyHostToDevice),
                 "initialize fixed output")) {
      return false;
    }
    views[projection] = {
        device_b[projection].get(),
        host_b[projection].size(),
        device_b_scales[projection].get(),
        host_b_scales[projection].size(),
        n,
        device_output[projection].get(),
        stride,
        host_output[projection].size()};
  }
  const int status =
      kernels::launch_sm87_a4w4_attention_k256_m128n256_bf16_cuda(
          topology, device_a.get(), host_a.size(), device_a_scales.get(),
          host_a_scales.size(), kM, views.data(), projection_count);
  if (!cuda_ok(static_cast<cudaError_t>(status), label) ||
      !cuda_ok(cudaDeviceSynchronize(), "sync fixed topology")) {
    return false;
  }
  for (std::size_t projection = 0U; projection < projection_count;
       ++projection) {
    if (!cuda_ok(cudaMemcpy(host_output[projection].data(),
                            device_output[projection].get(),
                            host_output[projection].size() *
                                sizeof(std::uint16_t),
                            cudaMemcpyDeviceToHost),
                 "copy fixed output")) {
      return false;
    }
    const std::size_t n = views[projection].output_size;
    const std::size_t stride = views[projection].output_row_stride_elements;
    for (std::size_t row = 0U; row < kM; ++row) {
      for (std::size_t column = 0U; column < n; ++column) {
        if (host_output[projection][row * stride + column] != 0U) {
          std::cerr << label << ": missing/incorrect fixed panel at projection "
                    << projection << " coordinate (" << row << ',' << column
                    << ")\n";
          return false;
        }
      }
      for (std::size_t column = n; column < stride; ++column) {
        if (host_output[projection][row * stride + column] != 0x7fc1U) {
          std::cerr << label << ": fixed row guard overwritten\n";
          return false;
        }
      }
    }
  }
  std::cout << label << " fixed topology smoke passed\n";
  return true;
}

}  // namespace

int main() {
  constexpr std::size_t kLogicalM = 129U;
  constexpr std::size_t kLaunchM = 256U;
  constexpr std::size_t kAttentionM = 128U;
  constexpr std::size_t kK = 768U;
  constexpr std::size_t kInputStride = 773U;
  constexpr float kClipRatio = 0.9375F;
  constexpr std::size_t kInputGuard = 16U;
  constexpr std::size_t kPackedGuard = 32U;
  constexpr std::size_t kScaleGuard = 16U;
  constexpr std::size_t kProjectionN = 256U;
  constexpr std::size_t kOutputStride = 258U;
  constexpr std::uint16_t kOutputSentinel = 0x7fc1U;

  kernels::Sm87A4W4AttentionK256Resources resources{};
  const int resource_status =
      kernels::query_sm87_a4w4_attention_k256_m128n256_resources_cuda(
          &resources);
  if (resource_status != static_cast<int>(cudaSuccess)) {
    std::cerr << "resource gate failed: "
              << cudaGetErrorString(
                     static_cast<cudaError_t>(resource_status))
              << '\n';
    return 1;
  }
  std::cout << "resources: regs=" << resources.registers_per_thread
            << " dynamic=" << resources.dynamic_shared_bytes
            << " local=" << resources.local_bytes
            << " active=" << resources.active_blocks_per_sm << '\n';

  std::vector<std::uint16_t> input_storage(
      2U * kInputGuard + kLogicalM * kInputStride, 0x3555U);
  std::vector<std::uint16_t> input(kLogicalM * kInputStride, 0U);
  for (std::size_t row = 0U; row < kLogicalM; ++row) {
    for (std::size_t inner = 0U; inner < kK; ++inner) {
      const int pattern = static_cast<int>(
                              (row * 131U + inner * 17U +
                               (row ^ inner) * 7U) %
                              401U) -
                          200;
      input[row * kInputStride + inner] =
          encode_bf16(static_cast<float>(pattern) * 0.0137F);
    }
    for (std::size_t inner = kK; inner < kInputStride; ++inner) {
      input[row * kInputStride + inner] = 0x3f80U;
    }
  }
  // Exercise the all-zero group and non-power-of-two extrema explicitly.
  std::fill(input.begin() + 256U, input.begin() + 512U, 0U);
  input[1U * kInputStride + 37U] = encode_bf16(3.1415927F);
  std::copy(input.begin(), input.end(),
            input_storage.begin() +
                static_cast<std::ptrdiff_t>(kInputGuard));
  const QuantizedA a_oracle = make_quantizer_oracle(
      input, kInputStride, kLogicalM, kLaunchM, kK, kClipRatio);

  DeviceBuffer<std::uint16_t> device_input_storage;
  DeviceBuffer<std::uint8_t> device_a_packed_storage;
  DeviceBuffer<std::uint16_t> device_a_scale_storage;
  if (!run_quantizer_and_graph(
          input_storage, kInputGuard, kInputStride, kLogicalM, kLaunchM,
          kK, kClipRatio, a_oracle, device_input_storage,
          device_a_packed_storage, device_a_scale_storage, kPackedGuard,
          kScaleGuard)) {
    return 1;
  }
  std::uint8_t* const device_a_packed =
      device_a_packed_storage.get() + kPackedGuard;
  std::uint16_t* const device_a_scales =
      device_a_scale_storage.get() + kScaleGuard;

  std::array<HostProjection, 3U> host_projections;
  std::array<DeviceBuffer<std::uint8_t>, 3U> device_b;
  std::array<DeviceBuffer<std::uint16_t>, 3U> device_b_scales;
  std::array<DeviceBuffer<std::uint16_t>, 3U> device_output_storage;
  std::array<kernels::Sm87A4W4AttentionK256ProjectionView, 3U> views{};
  constexpr std::size_t kOutputGuard = 16U;
  for (std::size_t projection = 0U; projection < 3U; ++projection) {
    host_projections[projection] = make_projection(
        projection, kAttentionM, kProjectionN, kK, kOutputStride,
        a_oracle, kOutputSentinel);
    const auto& host = host_projections[projection];
    if (!device_b[projection].allocate(host.packed.size()) ||
        !device_b_scales[projection].allocate(host.scales.size()) ||
        !device_output_storage[projection].allocate(
            2U * kOutputGuard + host.output_initial.size()) ||
        !cuda_ok(cudaMemcpy(device_b[projection].get(), host.packed.data(),
                            host.packed.size(), cudaMemcpyHostToDevice),
                 "copy B codes") ||
        !cuda_ok(cudaMemcpy(device_b_scales[projection].get(),
                            host.scales.data(),
                            host.scales.size() * sizeof(std::uint16_t),
                            cudaMemcpyHostToDevice),
                 "copy B scales") ||
        !cuda_ok(cudaMemset(device_output_storage[projection].get(), 0xc1,
                            kOutputGuard * sizeof(std::uint16_t)),
                 "initialize output prefix guard") ||
        !cuda_ok(cudaMemcpy(
                     device_output_storage[projection].get() + kOutputGuard,
                     host.output_initial.data(),
                     host.output_initial.size() * sizeof(std::uint16_t),
                     cudaMemcpyHostToDevice),
                 "initialize output payload") ||
        !cuda_ok(cudaMemset(
                     device_output_storage[projection].get() +
                         kOutputGuard + host.output_initial.size(),
                     0xc1, kOutputGuard * sizeof(std::uint16_t)),
                 "initialize output suffix guard")) {
      return 1;
    }
    views[projection] = {
        device_b[projection].get(),
        host.packed.size(),
        device_b_scales[projection].get(),
        host.scales.size(),
        kProjectionN,
        device_output_storage[projection].get() + kOutputGuard,
        kOutputStride,
        host.output_initial.size()};
  }

  // Three arbitrary macro-cells interleave all three projections.  With a
  // two-CTA cap one CTA must execute two persistent iterations.
  constexpr std::array<kernels::Sm87A4W4AttentionK256PanelDescriptor, 12U>
      descriptors{{
          {0U, 0U}, {1U, 0U}, {2U, 0U}, {0U, 1U},
          {1U, 1U}, {2U, 1U}, {0U, 2U}, {1U, 2U},
          {2U, 2U}, {0U, 3U}, {1U, 3U}, {2U, 3U},
      }};
  const int attention_status =
      kernels::launch_sm87_a4w4_attention_k256_m128n256_test_bf16_cuda(
          device_a_packed, a_oracle.packed.size(), device_a_scales,
          a_oracle.scales.size(), kAttentionM, kK, views.data(),
          views.size(), descriptors.data(), 3U, 2U);
  if (!cuda_ok(static_cast<cudaError_t>(attention_status),
               "launch arbitrary Attention K256")) {
    return 1;
  }

  for (std::size_t projection = 0U; projection < 3U; ++projection) {
    std::vector<std::uint8_t> b_after(host_projections[projection].packed.size());
    std::vector<std::uint16_t> scales_after(
        host_projections[projection].scales.size());
    std::vector<std::uint16_t> output_after(
        2U * kOutputGuard +
        host_projections[projection].output_initial.size());
    if (!cuda_ok(cudaMemcpy(b_after.data(), device_b[projection].get(),
                            b_after.size(), cudaMemcpyDeviceToHost),
                 "copy immutable B codes") ||
        !cuda_ok(cudaMemcpy(scales_after.data(),
                            device_b_scales[projection].get(),
                            scales_after.size() * sizeof(std::uint16_t),
                            cudaMemcpyDeviceToHost),
                 "copy immutable B scales") ||
        !cuda_ok(cudaMemcpy(output_after.data(),
                            device_output_storage[projection].get(),
                            output_after.size() * sizeof(std::uint16_t),
                            cudaMemcpyDeviceToHost),
                 "copy guarded output")) {
      return 1;
    }
    const auto& host = host_projections[projection];
    const bool prefix_guard = std::all_of(
        output_after.begin(),
        output_after.begin() + static_cast<std::ptrdiff_t>(kOutputGuard),
        [](const std::uint16_t value) { return value == 0xc1c1U; });
    const bool suffix_guard = std::all_of(
        output_after.begin() + static_cast<std::ptrdiff_t>(
                                   kOutputGuard + host.expected.size()),
        output_after.end(),
        [](const std::uint16_t value) { return value == 0xc1c1U; });
    const bool output_exact = std::equal(
        host.expected.begin(), host.expected.end(),
        output_after.begin() + static_cast<std::ptrdiff_t>(kOutputGuard));
    if (b_after != host.packed || scales_after != host.scales ||
        !prefix_guard || !suffix_guard || !output_exact) {
      std::cerr << "projection " << projection
                << " oracle/guard/immutable check failed\n";
      if (!output_exact) {
        for (std::size_t index = 0U; index < host.expected.size(); ++index) {
          const std::uint16_t actual = output_after[kOutputGuard + index];
          if (actual != host.expected[index]) {
            std::cerr << "first mismatch index=" << index << " expected=0x"
                      << std::hex << host.expected[index] << " actual=0x"
                      << actual << std::dec << '\n';
            break;
          }
        }
      }
      return 1;
    }
  }

  std::vector<std::uint8_t> a_packed_after(a_oracle.packed.size());
  std::vector<std::uint16_t> a_scales_after(a_oracle.scales.size());
  if (!cuda_ok(cudaMemcpy(a_packed_after.data(), device_a_packed,
                          a_packed_after.size(), cudaMemcpyDeviceToHost),
               "copy immutable A codes") ||
      !cuda_ok(cudaMemcpy(a_scales_after.data(), device_a_scales,
                          a_scales_after.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "copy immutable A scales") ||
      a_packed_after != a_oracle.packed ||
      a_scales_after != a_oracle.scales) {
    std::cerr << "Attention K256 modified A inputs\n";
    return 1;
  }

  if (!run_fixed_zero_smoke(
          kernels::Sm87A4W4AttentionK256Topology::kLinearQkvZ,
          "Linear 2Q2Z/4Q") ||
      !run_fixed_zero_smoke(
          kernels::Sm87A4W4AttentionK256Topology::kFullQkv,
          "Full 2Q2K/2Q2V/4Q") ||
      !run_fixed_zero_smoke(
          kernels::Sm87A4W4AttentionK256Topology::kAttentionO,
          "O adjacent 4O")) {
    return 1;
  }

  std::cout << "SM87 Attention K256 M128N256 bit-exact correctness passed: "
               "K768, mixed panels, guards, immutable inputs, persistent "
               "iterations, Graph replay\n";
  return 0;
}
