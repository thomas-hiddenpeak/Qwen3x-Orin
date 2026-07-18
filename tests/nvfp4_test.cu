#include "q3x/quantization/nvfp4.h"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

class TestContext {
public:
    void expect_true(const bool condition, const std::string& message) {
        if (!condition) {
            ++failures_;
            std::cerr << "FAIL: " << message << '\n';
        }
    }

    void expect_near(
        const float actual,
        const float expected,
        const float tolerance,
        const std::string& message) {
        if (!(std::fabs(actual - expected) <= tolerance)) {
            ++failures_;
            std::cerr << "FAIL: " << message << ": expected " << expected
                      << ", got " << actual << '\n';
        }
    }

    [[nodiscard]] int failures() const noexcept { return failures_; }

private:
    int failures_ = 0;
};

void test_all_e2m1_encodings(TestContext& test) {
    constexpr std::array<float, 16> expected = {
        0.0F,  0.5F,  1.0F,  1.5F,  2.0F,  3.0F,  4.0F,  6.0F,
        -0.0F, -0.5F, -1.0F, -1.5F, -2.0F, -3.0F, -4.0F, -6.0F,
    };

    for (std::size_t code = 0; code < expected.size(); ++code) {
        const float actual = q3x::quantization::decode_e2m1(
            static_cast<std::uint8_t>(code));
        test.expect_true(
            actual == expected[code],
            "E2M1 code " + std::to_string(code));
    }
    test.expect_true(
        !std::signbit(q3x::quantization::decode_e2m1(0x0U)),
        "E2M1 0x0 is positive zero");
    test.expect_true(
        std::signbit(q3x::quantization::decode_e2m1(0x8U)),
        "E2M1 0x8 preserves negative zero");

    test.expect_true(
        q3x::quantization::unpack_e2m1_nibble(0xa5U, false) == 0x5U,
        "low nibble is the even-K value");
    test.expect_true(
        q3x::quantization::unpack_e2m1_nibble(0xa5U, true) == 0xaU,
        "high nibble is the odd-K value");
}

void test_e4m3fn_key_encodings(TestContext& test) {
    using q3x::quantization::decode_e4m3fn;

    struct Case {
        std::uint8_t bits;
        float value;
        const char* label;
    };
    constexpr std::array<Case, 12> cases = {{
        {0x00U, 0.0F, "+zero"},
        {0x01U, 1.0F / 512.0F, "smallest subnormal"},
        {0x07U, 7.0F / 512.0F, "largest subnormal"},
        {0x08U, 1.0F / 64.0F, "smallest normal"},
        {0x38U, 1.0F, "one"},
        {0x3cU, 1.5F, "one and a half"},
        {0x40U, 2.0F, "two"},
        {0x70U, 128.0F, "largest ordinary exponent base"},
        {0x78U, 256.0F, "finite-only extended exponent"},
        {0x7eU, 448.0F, "maximum finite"},
        {0xb8U, -1.0F, "negative one"},
        {0xfeU, -448.0F, "minimum finite"},
    }};

    for (const Case& value : cases) {
        test.expect_near(
            decode_e4m3fn(value.bits), value.value, 0.0F,
            std::string("E4M3FN ") + value.label);
    }
    test.expect_true(
        !std::signbit(decode_e4m3fn(0x00U)),
        "E4M3FN positive zero sign");
    test.expect_true(
        std::signbit(decode_e4m3fn(0x80U)),
        "E4M3FN negative zero sign");
    test.expect_true(std::isnan(decode_e4m3fn(0x7fU)), "E4M3FN positive NaN");
    test.expect_true(std::isnan(decode_e4m3fn(0xffU)), "E4M3FN negative NaN");
}

void fill_reference_fixture(
    std::vector<std::uint8_t>& packed,
    std::vector<std::uint8_t>& scales) {
    // Each row contains all 16 E2M1 encodings twice. The second row reverses
    // nibble order to catch row-stride and low/high-nibble mistakes.
    for (std::size_t row = 0; row < 2; ++row) {
        for (std::size_t column = 0; column < 32; column += 2) {
            const std::uint8_t low = static_cast<std::uint8_t>(
                row == 0 ? column % 16 : 15 - (column % 16));
            const std::uint8_t high = static_cast<std::uint8_t>(
                row == 0 ? (column + 1) % 16 : 15 - ((column + 1) % 16));
            packed[row * 16 + column / 2] =
                static_cast<std::uint8_t>(low | (high << 4U));
        }
    }

    // [rows, columns / 16]: 1.0, 2.0, 0.5, 3.0.
    scales = {0x38U, 0x40U, 0x30U, 0x44U};
}

void test_group16_and_global_scale(TestContext& test) {
    constexpr std::size_t rows = 2;
    constexpr std::size_t columns = 32;
    constexpr float weight_scale_2 = 0.25F;
    std::vector<std::uint8_t> packed(rows * columns / 2);
    std::vector<std::uint8_t> scales;
    std::vector<float> output(rows * columns);
    fill_reference_fixture(packed, scales);

    const auto status = q3x::quantization::dequantize_nvfp4_reference(
        packed.data(), scales.data(), weight_scale_2, rows, columns, output.data());
    test.expect_true(
        status == q3x::quantization::NvFp4Status::kSuccess,
        "CPU dequantization succeeds");

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            const std::uint8_t byte = packed[row * (columns / 2) + column / 2];
            const std::uint8_t nibble = (column & 1U) != 0U
                                            ? static_cast<std::uint8_t>(byte >> 4U)
                                            : static_cast<std::uint8_t>(byte & 0x0fU);
            const std::uint8_t scale_bits =
                scales[row * (columns / 16) + column / 16];
            const float expected = q3x::quantization::decode_e2m1(nibble) *
                                   q3x::quantization::decode_e4m3fn(scale_bits) *
                                   weight_scale_2;
            test.expect_near(
                output[row * columns + column], expected, 0.0F,
                "group-16/global-scale formula at element " +
                    std::to_string(row * columns + column));
        }
    }

    test.expect_near(output[14], -4.0F * 1.0F * weight_scale_2, 0.0F,
                     "penultimate value before first group boundary");
    test.expect_near(output[15], -6.0F * 1.0F * weight_scale_2, 0.0F,
                     "last value in first group");
    test.expect_true(!std::signbit(output[16]), "positive zero starts second group");
    test.expect_near(output[17], 0.5F * 2.0F * weight_scale_2, 0.0F,
                     "second group uses its own scale");

    test.expect_true(
        q3x::quantization::dequantize_nvfp4_reference(
            packed.data(), scales.data(), weight_scale_2, 1, 17, output.data()) ==
            q3x::quantization::NvFp4Status::kInvalidColumnCount,
        "non-group-aligned columns are rejected");
}

void test_invalid_arguments_and_size_boundaries(TestContext& test) {
    using q3x::quantization::NvFp4Status;
    using q3x::quantization::dequantize_nvfp4_reference;
    using q3x::quantization::launch_nvfp4_reference_cuda;

    std::uint8_t packed = 0;
    std::uint8_t scale = 0x38U;
    float output = 0.0F;
    test.expect_true(
        dequantize_nvfp4_reference(
            nullptr, nullptr, 1.0F, 0, 16, nullptr) == NvFp4Status::kSuccess,
        "zero-row CPU input accepts empty buffers");
    test.expect_true(
        dequantize_nvfp4_reference(
            nullptr, &scale, 1.0F, 1, 16, &output) ==
            NvFp4Status::kInvalidArgument,
        "non-empty CPU input rejects null packed weights");
    test.expect_true(
        dequantize_nvfp4_reference(
            &packed, &scale, std::numeric_limits<float>::infinity(), 1, 16,
            &output) == NvFp4Status::kInvalidArgument,
        "CPU input rejects a non-finite tensor scale");

    constexpr std::size_t kColumns = 16;
    const std::size_t overflowing_rows =
        std::numeric_limits<std::size_t>::max() / kColumns + 1U;
    test.expect_true(
        dequantize_nvfp4_reference(
            &packed, &scale, 1.0F, overflowing_rows, kColumns, &output) ==
            NvFp4Status::kSizeOverflow,
        "CPU input rejects an overflowing element count");

    test.expect_true(
        static_cast<cudaError_t>(launch_nvfp4_reference_cuda(
            nullptr, nullptr, 1.0F, 0, kColumns, nullptr)) == cudaSuccess,
        "zero-row CUDA input succeeds without a device launch");
    test.expect_true(
        static_cast<cudaError_t>(launch_nvfp4_reference_cuda(
            &packed, &scale, 1.0F, overflowing_rows, kColumns, &output)) ==
            cudaErrorInvalidValue,
        "CUDA input rejects an overflowing element count before launch");
}

void test_cuda_consistency(TestContext& test) {
    int device_count = 0;
    const cudaError_t device_status = cudaGetDeviceCount(&device_count);
    if (device_status != cudaSuccess || device_count == 0) {
        std::cout << "SKIP: CUDA NVFP4 consistency test (no CUDA device)\n";
        (void)cudaGetLastError();
        return;
    }

    constexpr std::size_t rows = 2;
    constexpr std::size_t columns = 32;
    constexpr float weight_scale_2 = 0.25F;
    std::vector<std::uint8_t> packed(rows * columns / 2);
    std::vector<std::uint8_t> scales;
    std::vector<float> cpu(rows * columns);
    std::vector<float> gpu(rows * columns);
    fill_reference_fixture(packed, scales);
    (void)q3x::quantization::dequantize_nvfp4_reference(
        packed.data(), scales.data(), weight_scale_2, rows, columns, cpu.data());

    std::uint8_t* device_packed = nullptr;
    std::uint8_t* device_scales = nullptr;
    float* device_output = nullptr;

    auto check_cuda = [&test](const cudaError_t status, const char* operation) {
        test.expect_true(
            status == cudaSuccess,
            std::string(operation) + ": " + cudaGetErrorString(status));
        return status == cudaSuccess;
    };

    bool ready = check_cuda(
        cudaMalloc(reinterpret_cast<void**>(&device_packed), packed.size()),
        "cudaMalloc packed");
    ready = ready && check_cuda(
        cudaMalloc(reinterpret_cast<void**>(&device_scales), scales.size()),
        "cudaMalloc scales");
    ready = ready && check_cuda(
        cudaMalloc(reinterpret_cast<void**>(&device_output), gpu.size() * sizeof(float)),
        "cudaMalloc output");
    ready = ready && check_cuda(
        cudaMemcpy(device_packed, packed.data(), packed.size(), cudaMemcpyHostToDevice),
        "copy packed");
    ready = ready && check_cuda(
        cudaMemcpy(device_scales, scales.data(), scales.size(), cudaMemcpyHostToDevice),
        "copy scales");

    if (ready) {
        const cudaError_t stale_status =
            cudaMemcpy(nullptr, nullptr, 1, cudaMemcpyHostToDevice);
        test.expect_true(
            stale_status == cudaErrorInvalidValue,
            "invalid CUDA call seeds the stale-error regression");
        const int launch_status = q3x::quantization::launch_nvfp4_reference_cuda(
            device_packed, device_scales, weight_scale_2, rows, columns, device_output);
        ready = check_cuda(static_cast<cudaError_t>(launch_status),
                           "launch reference ignores an unrelated stale error");
    }
    if (ready) {
        ready = check_cuda(cudaDeviceSynchronize(), "synchronize reference");
    }
    if (ready) {
        ready = check_cuda(
            cudaMemcpy(gpu.data(), device_output, gpu.size() * sizeof(float),
                       cudaMemcpyDeviceToHost),
            "copy output");
    }
    if (ready) {
        for (std::size_t index = 0; index < gpu.size(); ++index) {
            test.expect_near(
                gpu[index], cpu[index], 0.0F,
                "CUDA agrees with CPU at element " + std::to_string(index));
            if (cpu[index] == 0.0F) {
                test.expect_true(
                    std::signbit(gpu[index]) == std::signbit(cpu[index]),
                    "CUDA preserves zero sign at element " + std::to_string(index));
            }
        }
    }

    if (device_output != nullptr) {
        (void)cudaFree(device_output);
    }
    if (device_scales != nullptr) {
        (void)cudaFree(device_scales);
    }
    if (device_packed != nullptr) {
        (void)cudaFree(device_packed);
    }
}

}  // namespace

int main() {
    TestContext test;
    test_all_e2m1_encodings(test);
    test_e4m3fn_key_encodings(test);
    test_group16_and_global_scale(test);
    test_invalid_arguments_and_size_boundaries(test);
    test_cuda_consistency(test);

    if (test.failures() != 0) {
        std::cerr << test.failures() << " NVFP4 test assertion(s) failed\n";
        return 1;
    }
    std::cout << "NVFP4 reference tests passed\n";
    return 0;
}
