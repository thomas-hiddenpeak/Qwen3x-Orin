#include "sm87_aot_system_v1_active_cell_cuda.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace active_cell = q3x::test::sm87_aot_active_cell;

namespace {

class TestContext final {
 public:
  void expect(const bool condition, const std::string& message) {
    if (!condition) {
      ++failures_;
      std::cerr << "FAIL: " << message << '\n';
    }
  }

  [[nodiscard]] bool check_cuda(const cudaError_t status,
                                const std::string& operation) {
    const bool passed = status == cudaSuccess;
    expect(passed, operation + ": " + cudaGetErrorString(status));
    return passed;
  }

  [[nodiscard]] int failures() const noexcept { return failures_; }

 private:
  int failures_ = 0;
};

template <typename T>
void release_device(TestContext& test, T*& pointer,
                    const std::string& name) {
  if (pointer == nullptr) {
    return;
  }
  const cudaError_t status = cudaFree(pointer);
  test.expect(status == cudaSuccess,
              "cudaFree " + name + ": " + cudaGetErrorString(status));
  pointer = nullptr;
}

template <typename T>
void expect_guards(TestContext& test, const std::vector<T>& storage,
                   const std::size_t guard_elements,
                   const std::size_t logical_elements, const T sentinel,
                   const std::string& name) {
  for (std::size_t index = 0U; index < guard_elements; ++index) {
    test.expect(storage[index] == sentinel,
                name + " prefix guard " + std::to_string(index));
  }
  const std::size_t suffix = guard_elements + logical_elements;
  for (std::size_t index = 0U; index < guard_elements; ++index) {
    test.expect(storage[suffix + index] == sentinel,
                name + " suffix guard " + std::to_string(index));
  }
}

void expect_masks(TestContext& test,
                  const std::vector<std::uint16_t>& storage,
                  const std::size_t guard_elements,
                  const std::array<std::uint16_t, 4U>& expected,
                  const std::string& name) {
  for (std::size_t index = 0U; index < expected.size(); ++index) {
    test.expect(storage[guard_elements + index] == expected[index],
                name + " mask " + std::to_string(index));
  }
}

void test_layout_contract(TestContext& test) {
  active_cell::MaskLayout layout{91U, 92U, 93U};
  test.expect(!active_cell::bf16_a_mask_layout(1U, 16U, 16U, nullptr),
              "BF16 layout rejects null output");
  test.expect(!active_cell::bf16_a_mask_layout(0U, 16U, 16U, &layout) &&
                  layout.outer_tiles == 0U && layout.k16_groups == 0U &&
                  layout.mask_count == 0U,
              "BF16 layout rejects zero rows and clears output");
  test.expect(!active_cell::bf16_a_mask_layout(1U, 0U, 16U, &layout),
              "BF16 layout rejects zero columns");
  test.expect(!active_cell::bf16_a_mask_layout(1U, 15U, 15U, &layout),
              "BF16 layout rejects a partial K16");
  test.expect(!active_cell::bf16_a_mask_layout(1U, 32U, 31U, &layout),
              "BF16 layout rejects a short row stride");
  test.expect(!active_cell::bf16_a_mask_layout(
                  std::numeric_limits<std::size_t>::max(), 16U, 16U,
                  &layout),
              "BF16 layout rejects addressed-span overflow");
  test.expect(active_cell::bf16_a_mask_layout(18U, 32U, 35U, &layout) &&
                  layout.outer_tiles == 2U && layout.k16_groups == 2U &&
                  layout.mask_count == 4U,
              "BF16 layout covers one M16 tile and an M tail");

  layout = {91U, 92U, 93U};
  test.expect(!active_cell::canonical_b_mask_layout(8U, 16U, nullptr),
              "canonical B layout rejects null output");
  test.expect(!active_cell::canonical_b_mask_layout(0U, 16U, &layout) &&
                  layout.outer_tiles == 0U && layout.k16_groups == 0U &&
                  layout.mask_count == 0U,
              "canonical B layout rejects zero N and clears output");
  test.expect(!active_cell::canonical_b_mask_layout(8U, 0U, &layout),
              "canonical B layout rejects zero K");
  test.expect(!active_cell::canonical_b_mask_layout(8U, 15U, &layout),
              "canonical B layout rejects a partial K16");
  test.expect(!active_cell::canonical_b_mask_layout(
                  std::numeric_limits<std::size_t>::max(), 16U, &layout),
              "canonical B layout rejects logical-size overflow");
  test.expect(active_cell::canonical_b_mask_layout(10U, 32U, &layout) &&
                  layout.outer_tiles == 2U && layout.k16_groups == 2U &&
                  layout.mask_count == 4U,
              "canonical B layout covers one N8 tile and an N tail");
}

void test_bf16_a_masks(TestContext& test, const cudaStream_t stream) {
  constexpr std::size_t kRows = 18U;
  constexpr std::size_t kColumns = 32U;
  constexpr std::size_t kRowStride = 35U;
  constexpr std::size_t kLogicalInputElements = kRows * kRowStride;
  constexpr std::size_t kMaskCount = 4U;
  constexpr std::size_t kGuardElements = 8U;
  constexpr std::size_t kFlagGuards = 4U;
  constexpr std::uint16_t kInputGuard = 0xd69bU;
  constexpr std::uint16_t kMaskGuard = 0xa55aU;
  constexpr std::uint32_t kFlagGuard = 0xd15ea5eU;

  std::vector<std::uint16_t> input_storage(
      kGuardElements + kLogicalInputElements + kGuardElements, kInputGuard);
  std::uint16_t* const input = input_storage.data() + kGuardElements;
  for (std::size_t row = 0U; row < kRows; ++row) {
    for (std::size_t column = 0U; column < kColumns; ++column) {
      input[row * kRowStride + column] = 0x0000U;
    }
    // Nonzero row padding must not affect any K support mask.
    for (std::size_t column = kColumns; column < kRowStride; ++column) {
      input[row * kRowStride + column] = 0x3f80U;
    }
  }

  input[0U * kRowStride + 0U] = 0x0000U;   // +0
  input[1U * kRowStride + 1U] = 0x8000U;   // -0
  input[2U * kRowStride + 2U] = 0x3f80U;   // +1
  input[3U * kRowStride + 3U] = 0xbf80U;   // -1
  input[4U * kRowStride + 4U] = 0x7f80U;   // +Inf
  input[5U * kRowStride + 5U] = 0xff80U;   // -Inf
  input[6U * kRowStride + 6U] = 0x7fc1U;   // NaN
  input[7U * kRowStride + 7U] = 0xffc1U;   // signed NaN
  input[1U * kRowStride + 17U] = 0x7fc1U;  // group 1, local K1
  input[15U * kRowStride + 31U] = 0x3f80U;

  input[16U * kRowStride + 0U] = 0xbf80U;
  input[17U * kRowStride + 15U] = 0x7f80U;
  input[16U * kRowStride + 16U] = 0x8000U;
  input[17U * kRowStride + 18U] = 0x7fc1U;
  input[16U * kRowStride + 20U] = 0x3f80U;

  const std::vector<std::uint16_t> expected_input = input_storage;
  std::vector<std::uint16_t> finite_input_storage = input_storage;
  std::uint16_t* const finite_input =
      finite_input_storage.data() + kGuardElements;
  finite_input[4U * kRowStride + 4U] = 0x3f80U;
  finite_input[5U * kRowStride + 5U] = 0xbf80U;
  finite_input[6U * kRowStride + 6U] = 0x3f80U;
  finite_input[7U * kRowStride + 7U] = 0xbf80U;
  finite_input[1U * kRowStride + 17U] = 0x3f80U;
  finite_input[17U * kRowStride + 15U] = 0x3f80U;
  finite_input[17U * kRowStride + 18U] = 0x3f80U;
  std::vector<std::uint16_t> mask_storage(
      kGuardElements + kMaskCount + kGuardElements, kMaskGuard);
  std::vector<std::uint32_t> flag_storage(
      kFlagGuards + 1U + kFlagGuards, kFlagGuard);
  flag_storage[kFlagGuards] = 0U;
  std::vector<std::uint16_t> copied_input(input_storage.size(), 0U);
  std::vector<std::uint16_t> copied_masks(mask_storage.size(), 0U);
  std::vector<std::uint32_t> copied_flag(flag_storage.size(), 0U);

  std::uint16_t* device_input_storage = nullptr;
  std::uint16_t* device_mask_storage = nullptr;
  std::uint32_t* device_flag_storage = nullptr;
  bool ready = test.check_cuda(
      cudaMalloc(reinterpret_cast<void**>(&device_input_storage),
                 input_storage.size() * sizeof(std::uint16_t)),
      "cudaMalloc BF16 input with guards");
  ready = ready &&
          test.check_cuda(
              cudaMalloc(reinterpret_cast<void**>(&device_mask_storage),
                         mask_storage.size() * sizeof(std::uint16_t)),
              "cudaMalloc BF16 masks with guards");
  ready = ready &&
          test.check_cuda(
              cudaMalloc(reinterpret_cast<void**>(&device_flag_storage),
                         flag_storage.size() * sizeof(std::uint32_t)),
              "cudaMalloc BF16 exceptional flag with guards");

  std::uint16_t* const device_input =
      device_input_storage == nullptr ? nullptr
                                      : device_input_storage + kGuardElements;
  std::uint16_t* const device_masks =
      device_mask_storage == nullptr ? nullptr
                                     : device_mask_storage + kGuardElements;
  std::uint32_t* const device_flag =
      device_flag_storage == nullptr ? nullptr
                                     : device_flag_storage + kFlagGuards;
  if (ready) {
    ready = test.check_cuda(
        cudaMemcpyAsync(device_input_storage, finite_input_storage.data(),
                        finite_input_storage.size() * sizeof(std::uint16_t),
                        cudaMemcpyHostToDevice, stream),
        "copy finite BF16 input and guards") &&
            test.check_cuda(
                cudaMemcpyAsync(device_mask_storage, mask_storage.data(),
                                mask_storage.size() * sizeof(std::uint16_t),
                                cudaMemcpyHostToDevice, stream),
                "copy BF16 mask guards") &&
            test.check_cuda(
                cudaMemcpyAsync(device_flag_storage, flag_storage.data(),
                                flag_storage.size() * sizeof(std::uint32_t),
                                cudaMemcpyHostToDevice, stream),
                "clear BF16 exceptional flag and copy guards");
  }

  if (ready) {
    test.expect(active_cell::launch_bf16_a_k_support_masks(
                    nullptr, kRows, kColumns, kRowStride, device_masks,
                    kMaskCount, device_flag, stream) == cudaErrorInvalidValue,
                "BF16 launch rejects null input");
    test.expect(active_cell::launch_bf16_a_k_support_masks(
                    device_input, kRows, kColumns, kRowStride, nullptr,
                    kMaskCount, device_flag, stream) == cudaErrorInvalidValue,
                "BF16 launch rejects null masks");
    test.expect(active_cell::launch_bf16_a_k_support_masks(
                    device_input, kRows, kColumns, kRowStride, device_masks,
                    kMaskCount - 1U, device_flag, stream) ==
                    cudaErrorInvalidValue,
                "BF16 launch rejects wrong mask count");
    test.expect(active_cell::launch_bf16_a_k_support_masks(
                    device_input, kRows, kColumns, kRowStride, device_masks,
                    kMaskCount, nullptr, stream) == cudaErrorInvalidValue,
                "BF16 launch rejects null exceptional flag");
    const auto* const misaligned_input =
        reinterpret_cast<const std::uint16_t*>(
            reinterpret_cast<const std::uint8_t*>(device_input) + 1U);
    auto* const misaligned_masks = reinterpret_cast<std::uint16_t*>(
        reinterpret_cast<std::uint8_t*>(device_masks) + 1U);
    test.expect(active_cell::launch_bf16_a_k_support_masks(
                    misaligned_input, kRows, kColumns, kRowStride,
                    device_masks, kMaskCount, device_flag, stream) ==
                    cudaErrorInvalidValue,
                "BF16 launch rejects misaligned input");
    test.expect(active_cell::launch_bf16_a_k_support_masks(
                    device_input, kRows, kColumns, kRowStride,
                    misaligned_masks, kMaskCount, device_flag, stream) ==
                    cudaErrorInvalidValue,
                "BF16 launch rejects misaligned masks");
    auto* const misaligned_flag = reinterpret_cast<std::uint32_t*>(
        reinterpret_cast<std::uint8_t*>(device_flag) + 1U);
    test.expect(active_cell::launch_bf16_a_k_support_masks(
                    device_input, kRows, kColumns, kRowStride, device_masks,
                    kMaskCount, misaligned_flag, stream) ==
                    cudaErrorInvalidValue,
                "BF16 launch rejects misaligned exceptional flag");
    test.expect(active_cell::launch_bf16_a_k_support_masks(
                    device_input, std::numeric_limits<std::size_t>::max(),
                    16U, 16U, device_masks, 1U, device_flag, stream) ==
                    cudaErrorInvalidValue,
                "BF16 launch rejects addressed-span overflow");

    (void)cudaGetLastError();
    ready = test.check_cuda(active_cell::launch_bf16_a_k_support_masks(
                                device_input, kRows, kColumns, kRowStride,
                                device_masks, kMaskCount, device_flag, stream),
                            "launch finite BF16 active-cell masks");
  }
  if (ready) {
    ready = test.check_cuda(
                cudaMemcpyAsync(copied_input.data(), device_input_storage,
                                copied_input.size() * sizeof(std::uint16_t),
                                cudaMemcpyDeviceToHost, stream),
                "copy back BF16 input and guards") &&
            test.check_cuda(
                cudaMemcpyAsync(copied_masks.data(), device_mask_storage,
                                copied_masks.size() * sizeof(std::uint16_t),
                                cudaMemcpyDeviceToHost, stream),
                "copy back finite BF16 masks and guards") &&
            test.check_cuda(
                cudaMemcpyAsync(copied_flag.data(), device_flag_storage,
                                copied_flag.size() * sizeof(std::uint32_t),
                                cudaMemcpyDeviceToHost, stream),
                "copy back finite BF16 exceptional flag and guards");
  }
  if (ready) {
    ready = test.check_cuda(cudaStreamSynchronize(stream),
                            "synchronize BF16 active-cell test");
  } else {
    (void)cudaStreamSynchronize(stream);
  }
  if (ready) {
    constexpr std::array<std::uint16_t, 4U> kExpectedMasks{
        0x00fcU, 0x8002U, 0x8001U, 0x0014U};
    test.expect(copied_input == finite_input_storage,
                "finite BF16 input and its guards remain unchanged");
    expect_guards(test, copied_masks, kGuardElements, kMaskCount, kMaskGuard,
                  "BF16 masks");
    expect_masks(test, copied_masks, kGuardElements, kExpectedMasks,
                 "finite BF16");
    expect_guards(test, copied_flag, kFlagGuards, 1U, kFlagGuard,
                  "finite BF16 exceptional flag");
    test.expect(copied_flag[kFlagGuards] == 0U,
                "finite BF16 input keeps exceptional flag clear");
  }

  if (ready) {
    ready = test.check_cuda(
                cudaMemcpyAsync(device_input_storage, input_storage.data(),
                                input_storage.size() * sizeof(std::uint16_t),
                                cudaMemcpyHostToDevice, stream),
                "copy exceptional BF16 input and guards") &&
            test.check_cuda(
                cudaMemcpyAsync(device_mask_storage, mask_storage.data(),
                                mask_storage.size() * sizeof(std::uint16_t),
                                cudaMemcpyHostToDevice, stream),
                "reset exceptional BF16 mask guards") &&
            test.check_cuda(
                cudaMemcpyAsync(device_flag_storage, flag_storage.data(),
                                flag_storage.size() * sizeof(std::uint32_t),
                                cudaMemcpyHostToDevice, stream),
                "clear exceptional BF16 flag and reset guards");
  }
  if (ready) {
    (void)cudaGetLastError();
    ready = test.check_cuda(active_cell::launch_bf16_a_k_support_masks(
                                device_input, kRows, kColumns, kRowStride,
                                device_masks, kMaskCount, device_flag, stream),
                            "launch exceptional BF16 active-cell masks");
  }
  if (ready) {
    ready = test.check_cuda(
                cudaMemcpyAsync(copied_input.data(), device_input_storage,
                                copied_input.size() * sizeof(std::uint16_t),
                                cudaMemcpyDeviceToHost, stream),
                "copy back exceptional BF16 input and guards") &&
            test.check_cuda(
                cudaMemcpyAsync(copied_masks.data(), device_mask_storage,
                                copied_masks.size() * sizeof(std::uint16_t),
                                cudaMemcpyDeviceToHost, stream),
                "copy back exceptional BF16 masks and guards") &&
            test.check_cuda(
                cudaMemcpyAsync(copied_flag.data(), device_flag_storage,
                                copied_flag.size() * sizeof(std::uint32_t),
                                cudaMemcpyDeviceToHost, stream),
                "copy back exceptional BF16 flag and guards");
  }
  if (ready) {
    ready = test.check_cuda(cudaStreamSynchronize(stream),
                            "synchronize exceptional BF16 test");
  } else {
    (void)cudaStreamSynchronize(stream);
  }
  if (ready) {
    constexpr std::array<std::uint16_t, 4U> kExpectedMasks{
        0x00fcU, 0x8002U, 0x8001U, 0x0014U};
    test.expect(copied_input == expected_input,
                "exceptional BF16 input and guards remain unchanged");
    expect_guards(test, copied_masks, kGuardElements, kMaskCount, kMaskGuard,
                  "exceptional BF16 masks");
    expect_masks(test, copied_masks, kGuardElements, kExpectedMasks,
                 "exceptional BF16");
    expect_guards(test, copied_flag, kFlagGuards, 1U, kFlagGuard,
                  "exceptional BF16 flag");
    test.expect(copied_flag[kFlagGuards] == 1U,
                "BF16 NaN or infinity sets exceptional flag");
  }

  release_device(test, device_flag_storage, "BF16 exceptional flag");
  release_device(test, device_mask_storage, "BF16 masks");
  release_device(test, device_input_storage, "BF16 input");
}

void test_fp8_b_masks(TestContext& test, const cudaStream_t stream) {
  constexpr std::size_t kOutputFeatures = 10U;
  constexpr std::size_t kInputFeatures = 32U;
  constexpr std::size_t kLogicalInputBytes =
      kOutputFeatures * kInputFeatures;
  constexpr std::size_t kMaskCount = 4U;
  constexpr std::size_t kByteGuards = 16U;
  constexpr std::size_t kMaskGuards = 8U;
  constexpr std::size_t kFlagGuards = 4U;
  constexpr std::uint8_t kInputGuard = 0xd3U;
  constexpr std::uint16_t kMaskGuard = 0x6db6U;
  constexpr std::uint32_t kFlagGuard = 0xc0decafeU;

  std::vector<std::uint8_t> input_storage(
      kByteGuards + kLogicalInputBytes + kByteGuards, kInputGuard);
  std::uint8_t* const input = input_storage.data() + kByteGuards;
  for (std::size_t index = 0U; index < kLogicalInputBytes; ++index) {
    input[index] = 0x00U;
  }
  const auto set = [input](const std::size_t row, const std::size_t column,
                           const std::uint8_t bits) {
    input[row * kInputFeatures + column] = bits;
  };

  set(0U, 0U, 0x00U);   // +0
  set(0U, 1U, 0x80U);   // -0
  set(0U, 2U, 0x38U);   // ordinary positive
  set(1U, 3U, 0x7fU);   // terminal positive code
  set(2U, 4U, 0xb8U);   // ordinary negative
  set(3U, 5U, 0xffU);   // terminal negative code
  set(6U, 17U, 0x01U);
  set(7U, 16U, 0x80U);  // signed zero remains inactive
  set(5U, 31U, 0x7fU);

  set(8U, 0U, 0x80U);
  set(9U, 1U, 0x38U);
  set(8U, 15U, 0x7fU);
  set(8U, 16U, 0xb8U);
  set(9U, 18U, 0xffU);

  const std::vector<std::uint8_t> expected_input = input_storage;
  std::vector<std::uint8_t> finite_input_storage = input_storage;
  std::uint8_t* const finite_input =
      finite_input_storage.data() + kByteGuards;
  const auto set_finite =
      [finite_input](const std::size_t row, const std::size_t column,
                     const std::uint8_t bits) {
        finite_input[row * kInputFeatures + column] = bits;
      };
  set_finite(1U, 3U, 0x38U);
  set_finite(3U, 5U, 0xb8U);
  set_finite(5U, 31U, 0x38U);
  set_finite(8U, 15U, 0x38U);
  set_finite(9U, 18U, 0xb8U);
  std::vector<std::uint16_t> mask_storage(
      kMaskGuards + kMaskCount + kMaskGuards, kMaskGuard);
  std::vector<std::uint32_t> flag_storage(
      kFlagGuards + 1U + kFlagGuards, kFlagGuard);
  flag_storage[kFlagGuards] = 0U;
  std::vector<std::uint8_t> copied_input(input_storage.size(), 0U);
  std::vector<std::uint16_t> copied_masks(mask_storage.size(), 0U);
  std::vector<std::uint32_t> copied_flag(flag_storage.size(), 0U);

  std::uint8_t* device_input_storage = nullptr;
  std::uint16_t* device_mask_storage = nullptr;
  std::uint32_t* device_flag_storage = nullptr;
  bool ready = test.check_cuda(
      cudaMalloc(reinterpret_cast<void**>(&device_input_storage),
                 input_storage.size()),
      "cudaMalloc FP8 input with guards");
  ready = ready &&
          test.check_cuda(
              cudaMalloc(reinterpret_cast<void**>(&device_mask_storage),
                         mask_storage.size() * sizeof(std::uint16_t)),
              "cudaMalloc FP8 masks with guards");
  ready = ready &&
          test.check_cuda(
              cudaMalloc(reinterpret_cast<void**>(&device_flag_storage),
                         flag_storage.size() * sizeof(std::uint32_t)),
              "cudaMalloc FP8 exceptional flag with guards");
  std::uint8_t* const device_input =
      device_input_storage == nullptr ? nullptr
                                      : device_input_storage + kByteGuards;
  std::uint16_t* const device_masks =
      device_mask_storage == nullptr ? nullptr
                                     : device_mask_storage + kMaskGuards;
  std::uint32_t* const device_flag =
      device_flag_storage == nullptr ? nullptr
                                     : device_flag_storage + kFlagGuards;
  if (ready) {
    ready = test.check_cuda(
                cudaMemcpyAsync(device_input_storage,
                                finite_input_storage.data(),
                                finite_input_storage.size(),
                                cudaMemcpyHostToDevice, stream),
                "copy finite FP8 input and guards") &&
            test.check_cuda(
                cudaMemcpyAsync(device_mask_storage, mask_storage.data(),
                                mask_storage.size() * sizeof(std::uint16_t),
                                cudaMemcpyHostToDevice, stream),
                "copy FP8 mask guards") &&
            test.check_cuda(
                cudaMemcpyAsync(device_flag_storage, flag_storage.data(),
                                flag_storage.size() * sizeof(std::uint32_t),
                                cudaMemcpyHostToDevice, stream),
                "clear FP8 exceptional flag and copy guards");
  }

  if (ready) {
    test.expect(active_cell::launch_fp8_b_k_support_masks(
                    nullptr, kOutputFeatures, kInputFeatures, 1.0F,
                    device_masks, kMaskCount, device_flag, stream) ==
                    cudaErrorInvalidValue,
                "FP8 launch rejects null input");
    test.expect(active_cell::launch_fp8_b_k_support_masks(
                    device_input, kOutputFeatures, kInputFeatures, 1.0F,
                    nullptr, kMaskCount, device_flag, stream) ==
                    cudaErrorInvalidValue,
                "FP8 launch rejects null masks");
    test.expect(active_cell::launch_fp8_b_k_support_masks(
                    device_input, kOutputFeatures, kInputFeatures, 1.0F,
                    device_masks, kMaskCount - 1U, device_flag, stream) ==
                    cudaErrorInvalidValue,
                "FP8 launch rejects wrong mask count");
    test.expect(active_cell::launch_fp8_b_k_support_masks(
                    device_input, kOutputFeatures, kInputFeatures, 1.0F,
                    device_masks, kMaskCount, nullptr, stream) ==
                    cudaErrorInvalidValue,
                "FP8 launch rejects null exceptional flag");
    auto* const misaligned_masks = reinterpret_cast<std::uint16_t*>(
        reinterpret_cast<std::uint8_t*>(device_masks) + 1U);
    test.expect(active_cell::launch_fp8_b_k_support_masks(
                    device_input, kOutputFeatures, kInputFeatures, 1.0F,
                    misaligned_masks, kMaskCount, device_flag, stream) ==
                    cudaErrorInvalidValue,
                "FP8 launch rejects misaligned masks");
    auto* const misaligned_flag = reinterpret_cast<std::uint32_t*>(
        reinterpret_cast<std::uint8_t*>(device_flag) + 1U);
    test.expect(active_cell::launch_fp8_b_k_support_masks(
                    device_input, kOutputFeatures, kInputFeatures, 1.0F,
                    device_masks, kMaskCount, misaligned_flag, stream) ==
                    cudaErrorInvalidValue,
                "FP8 launch rejects misaligned exceptional flag");
    test.expect(active_cell::launch_fp8_b_k_support_masks(
                    device_input, std::numeric_limits<std::size_t>::max(),
                    16U, 1.0F, device_masks, 1U, device_flag, stream) ==
                    cudaErrorInvalidValue,
                "FP8 launch rejects logical-size overflow");

    const std::array<float, 5U> invalid_scales{
        0.0F, -1.0F, std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN()};
    for (std::size_t index = 0U; index < invalid_scales.size(); ++index) {
      test.expect(active_cell::launch_fp8_b_k_support_masks(
                      device_input, kOutputFeatures, kInputFeatures,
                      invalid_scales[index], device_masks, kMaskCount,
                      device_flag, stream) == cudaErrorInvalidValue,
                  "FP8 rejects invalid tensor scale " +
                      std::to_string(index));
    }

    (void)cudaGetLastError();
    ready = test.check_cuda(active_cell::launch_fp8_b_k_support_masks(
                                device_input, kOutputFeatures, kInputFeatures,
                                1.0F, device_masks, kMaskCount, device_flag,
                                stream),
                            "launch finite FP8 active-cell masks");
  }
  if (ready) {
    ready = test.check_cuda(
                cudaMemcpyAsync(copied_input.data(), device_input_storage,
                                copied_input.size(), cudaMemcpyDeviceToHost,
                                stream),
                "copy back FP8 input and guards") &&
            test.check_cuda(
                cudaMemcpyAsync(copied_masks.data(), device_mask_storage,
                                copied_masks.size() * sizeof(std::uint16_t),
                                cudaMemcpyDeviceToHost, stream),
                "copy back finite FP8 masks and guards") &&
            test.check_cuda(
                cudaMemcpyAsync(copied_flag.data(), device_flag_storage,
                                copied_flag.size() * sizeof(std::uint32_t),
                                cudaMemcpyDeviceToHost, stream),
                "copy back finite FP8 exceptional flag and guards");
  }
  if (ready) {
    ready = test.check_cuda(cudaStreamSynchronize(stream),
                            "synchronize FP8 active-cell test");
  } else {
    (void)cudaStreamSynchronize(stream);
  }
  if (ready) {
    constexpr std::array<std::uint16_t, 4U> kExpectedMasks{
        0x003cU, 0x8002U, 0x8002U, 0x0005U};
    test.expect(copied_input == finite_input_storage,
                "finite FP8 input and its guards remain unchanged");
    expect_guards(test, copied_masks, kMaskGuards, kMaskCount, kMaskGuard,
                  "FP8 masks");
    expect_masks(test, copied_masks, kMaskGuards, kExpectedMasks,
                 "finite FP8");
    expect_guards(test, copied_flag, kFlagGuards, 1U, kFlagGuard,
                  "finite FP8 exceptional flag");
    test.expect(copied_flag[kFlagGuards] == 0U,
                "finite FP8 input keeps exceptional flag clear");
  }

  if (ready) {
    ready = test.check_cuda(
                cudaMemcpyAsync(device_input_storage, input_storage.data(),
                                input_storage.size(), cudaMemcpyHostToDevice,
                                stream),
                "copy exceptional FP8 input and guards") &&
            test.check_cuda(
                cudaMemcpyAsync(device_mask_storage, mask_storage.data(),
                                mask_storage.size() * sizeof(std::uint16_t),
                                cudaMemcpyHostToDevice, stream),
                "reset exceptional FP8 mask guards") &&
            test.check_cuda(
                cudaMemcpyAsync(device_flag_storage, flag_storage.data(),
                                flag_storage.size() * sizeof(std::uint32_t),
                                cudaMemcpyHostToDevice, stream),
                "clear exceptional FP8 flag and reset guards");
  }
  if (ready) {
    (void)cudaGetLastError();
    ready = test.check_cuda(active_cell::launch_fp8_b_k_support_masks(
                                device_input, kOutputFeatures, kInputFeatures,
                                1.0F, device_masks, kMaskCount, device_flag,
                                stream),
                            "launch exceptional FP8 active-cell masks");
  }
  if (ready) {
    ready = test.check_cuda(
                cudaMemcpyAsync(copied_input.data(), device_input_storage,
                                copied_input.size(), cudaMemcpyDeviceToHost,
                                stream),
                "copy back exceptional FP8 input and guards") &&
            test.check_cuda(
                cudaMemcpyAsync(copied_masks.data(), device_mask_storage,
                                copied_masks.size() * sizeof(std::uint16_t),
                                cudaMemcpyDeviceToHost, stream),
                "copy back exceptional FP8 masks and guards") &&
            test.check_cuda(
                cudaMemcpyAsync(copied_flag.data(), device_flag_storage,
                                copied_flag.size() * sizeof(std::uint32_t),
                                cudaMemcpyDeviceToHost, stream),
                "copy back exceptional FP8 flag and guards");
  }
  if (ready) {
    ready = test.check_cuda(cudaStreamSynchronize(stream),
                            "synchronize exceptional FP8 test");
  } else {
    (void)cudaStreamSynchronize(stream);
  }
  if (ready) {
    constexpr std::array<std::uint16_t, 4U> kExpectedMasks{
        0x003cU, 0x8002U, 0x8002U, 0x0005U};
    test.expect(copied_input == expected_input,
                "exceptional FP8 input and guards remain unchanged");
    expect_guards(test, copied_masks, kMaskGuards, kMaskCount, kMaskGuard,
                  "exceptional FP8 masks");
    expect_masks(test, copied_masks, kMaskGuards, kExpectedMasks,
                 "exceptional FP8");
    expect_guards(test, copied_flag, kFlagGuards, 1U, kFlagGuard,
                  "exceptional FP8 flag");
    test.expect(copied_flag[kFlagGuards] == 1U,
                "FP8 terminal code sets exceptional flag");
  }

  release_device(test, device_flag_storage, "FP8 exceptional flag");
  release_device(test, device_mask_storage, "FP8 masks");
  release_device(test, device_input_storage, "FP8 input");
}

void set_nvfp4_nibble(std::uint8_t* const packed,
                      const std::size_t input_features,
                      const std::size_t row, const std::size_t column,
                      const std::uint8_t nibble) {
  const std::size_t index = row * (input_features / 2U) + column / 2U;
  const unsigned int shift = 4U * static_cast<unsigned int>(column & 1U);
  const std::uint8_t cleared = static_cast<std::uint8_t>(
      packed[index] & static_cast<std::uint8_t>(~(0x0fU << shift)));
  packed[index] = static_cast<std::uint8_t>(
      cleared | static_cast<std::uint8_t>((nibble & 0x0fU) << shift));
}

void test_nvfp4_b_masks(TestContext& test, const cudaStream_t stream) {
  constexpr std::size_t kOutputFeatures = 10U;
  constexpr std::size_t kInputFeatures = 32U;
  constexpr std::size_t kPackedRowBytes = kInputFeatures / 2U;
  constexpr std::size_t kScaleRowBytes = kInputFeatures / 16U;
  constexpr std::size_t kLogicalPackedBytes =
      kOutputFeatures * kPackedRowBytes;
  constexpr std::size_t kLogicalScaleBytes =
      kOutputFeatures * kScaleRowBytes;
  constexpr std::size_t kMaskCount = 4U;
  constexpr std::size_t kByteGuards = 16U;
  constexpr std::size_t kMaskGuards = 8U;
  constexpr std::size_t kFlagGuards = 4U;
  constexpr std::uint8_t kPackedGuard = 0xc3U;
  constexpr std::uint8_t kScaleGuard = 0x5aU;
  constexpr std::uint16_t kMaskGuard = 0xb66bU;
  constexpr std::uint32_t kFlagGuard = 0x51a1c0deU;

  std::vector<std::uint8_t> packed_storage(
      kByteGuards + kLogicalPackedBytes + kByteGuards, kPackedGuard);
  std::vector<std::uint8_t> scale_storage(
      kByteGuards + kLogicalScaleBytes + kByteGuards, kScaleGuard);
  std::uint8_t* const packed = packed_storage.data() + kByteGuards;
  std::uint8_t* const scales = scale_storage.data() + kByteGuards;
  for (std::size_t index = 0U; index < kLogicalPackedBytes; ++index) {
    packed[index] = 0x00U;
  }
  for (std::size_t index = 0U; index < kLogicalScaleBytes; ++index) {
    scales[index] = 0x38U;  // positive finite E4M3FN
  }
  const auto set_scale = [scales](const std::size_t row,
                                  const std::size_t group,
                                  const std::uint8_t bits) {
    scales[row * kScaleRowBytes + group] = bits;
  };

  // Zero and signed-zero scales suppress otherwise nonzero weights.
  set_scale(0U, 0U, 0x00U);
  set_scale(1U, 0U, 0x80U);
  for (std::size_t column = 0U; column < 16U; ++column) {
    set_nvfp4_nibble(packed, kInputFeatures, 0U, column, 0x1U);
    set_nvfp4_nibble(packed, kInputFeatures, 1U, column, 0x9U);
  }

  // Positive scales expose both low and high nibbles. Signed E2M1 zero does
  // not become active, whereas either-sign nonzero magnitudes do.
  set_nvfp4_nibble(packed, kInputFeatures, 2U, 2U, 0x1U);
  set_nvfp4_nibble(packed, kInputFeatures, 2U, 3U, 0x8U);
  set_nvfp4_nibble(packed, kInputFeatures, 2U, 4U, 0x9U);
  set_nvfp4_nibble(packed, kInputFeatures, 3U, 15U, 0x7U);

  // A negative nonzero block scale makes the complete N8/K16 mask active.
  set_scale(6U, 1U, 0xb8U);
  for (std::size_t column = 16U; column < 32U; ++column) {
    set_nvfp4_nibble(packed, kInputFeatures, 6U, column, 0x1U);
  }

  // The N8 tail contains only rows 8 and 9. A terminal scale fails closed to
  // all-active in group 0. Group 1 remains selective; row 9 has a zero scale.
  set_scale(8U, 0U, 0x7fU);
  for (std::size_t column = 0U; column < 16U; ++column) {
    set_nvfp4_nibble(packed, kInputFeatures, 8U, column, 0x1U);
  }
  set_nvfp4_nibble(packed, kInputFeatures, 8U, 18U, 0x2U);
  set_nvfp4_nibble(packed, kInputFeatures, 8U, 31U, 0xaU);
  set_nvfp4_nibble(packed, kInputFeatures, 8U, 17U, 0x8U);
  set_scale(9U, 1U, 0x00U);
  set_nvfp4_nibble(packed, kInputFeatures, 9U, 16U, 0x1U);
  set_nvfp4_nibble(packed, kInputFeatures, 9U, 17U, 0x9U);

  const std::vector<std::uint8_t> expected_packed = packed_storage;
  const std::vector<std::uint8_t> expected_scales = scale_storage;
  std::vector<std::uint8_t> finite_scale_storage = scale_storage;
  std::uint8_t* const finite_scales =
      finite_scale_storage.data() + kByteGuards;
  finite_scales[6U * kScaleRowBytes + 1U] = 0x38U;
  finite_scales[8U * kScaleRowBytes + 0U] = 0x38U;
  std::vector<std::uint16_t> mask_storage(
      kMaskGuards + kMaskCount + kMaskGuards, kMaskGuard);
  std::vector<std::uint32_t> flag_storage(
      kFlagGuards + 1U + kFlagGuards, kFlagGuard);
  flag_storage[kFlagGuards] = 0U;
  std::vector<std::uint8_t> copied_packed(packed_storage.size(), 0U);
  std::vector<std::uint8_t> copied_scales(scale_storage.size(), 0U);
  std::vector<std::uint16_t> copied_masks(mask_storage.size(), 0U);
  std::vector<std::uint32_t> copied_flag(flag_storage.size(), 0U);

  std::uint8_t* device_packed_storage = nullptr;
  std::uint8_t* device_scale_storage = nullptr;
  std::uint16_t* device_mask_storage = nullptr;
  std::uint32_t* device_flag_storage = nullptr;
  bool ready = test.check_cuda(
      cudaMalloc(reinterpret_cast<void**>(&device_packed_storage),
                 packed_storage.size()),
      "cudaMalloc NVFP4 packed input with guards");
  ready = ready &&
          test.check_cuda(
              cudaMalloc(reinterpret_cast<void**>(&device_scale_storage),
                         scale_storage.size()),
              "cudaMalloc NVFP4 scale input with guards");
  ready = ready &&
          test.check_cuda(
              cudaMalloc(reinterpret_cast<void**>(&device_mask_storage),
                         mask_storage.size() * sizeof(std::uint16_t)),
              "cudaMalloc NVFP4 masks with guards");
  ready = ready &&
          test.check_cuda(
              cudaMalloc(reinterpret_cast<void**>(&device_flag_storage),
                         flag_storage.size() * sizeof(std::uint32_t)),
              "cudaMalloc NVFP4 exceptional flag with guards");
  std::uint8_t* const device_packed =
      device_packed_storage == nullptr
          ? nullptr
          : device_packed_storage + kByteGuards;
  std::uint8_t* const device_scales =
      device_scale_storage == nullptr
          ? nullptr
          : device_scale_storage + kByteGuards;
  std::uint16_t* const device_masks =
      device_mask_storage == nullptr ? nullptr
                                     : device_mask_storage + kMaskGuards;
  std::uint32_t* const device_flag =
      device_flag_storage == nullptr ? nullptr
                                     : device_flag_storage + kFlagGuards;
  if (ready) {
    ready = test.check_cuda(
                cudaMemcpyAsync(device_packed_storage, packed_storage.data(),
                                packed_storage.size(), cudaMemcpyHostToDevice,
                                stream),
                "copy NVFP4 packed input and guards") &&
            test.check_cuda(
                cudaMemcpyAsync(device_scale_storage,
                                finite_scale_storage.data(),
                                finite_scale_storage.size(),
                                cudaMemcpyHostToDevice, stream),
                "copy finite NVFP4 scale input and guards") &&
            test.check_cuda(
                cudaMemcpyAsync(device_mask_storage, mask_storage.data(),
                                mask_storage.size() * sizeof(std::uint16_t),
                                cudaMemcpyHostToDevice, stream),
                "copy NVFP4 mask guards") &&
            test.check_cuda(
                cudaMemcpyAsync(device_flag_storage, flag_storage.data(),
                                flag_storage.size() * sizeof(std::uint32_t),
                                cudaMemcpyHostToDevice, stream),
                "clear NVFP4 exceptional flag and copy guards");
  }

  if (ready) {
    test.expect(active_cell::launch_nvfp4_b_k_support_masks(
                    nullptr, device_scales, kOutputFeatures, kInputFeatures,
                    1.0F, device_masks, kMaskCount, device_flag, stream) ==
                    cudaErrorInvalidValue,
                "NVFP4 launch rejects null packed weights");
    test.expect(active_cell::launch_nvfp4_b_k_support_masks(
                    device_packed, nullptr, kOutputFeatures, kInputFeatures,
                    1.0F, device_masks, kMaskCount, device_flag, stream) ==
                    cudaErrorInvalidValue,
                "NVFP4 launch rejects null block scales");
    test.expect(active_cell::launch_nvfp4_b_k_support_masks(
                    device_packed, device_scales, kOutputFeatures,
                    kInputFeatures, 1.0F, nullptr, kMaskCount, device_flag,
                    stream) ==
                    cudaErrorInvalidValue,
                "NVFP4 launch rejects null masks");
    test.expect(active_cell::launch_nvfp4_b_k_support_masks(
                    device_packed, device_scales, kOutputFeatures,
                    kInputFeatures, 1.0F, device_masks, kMaskCount - 1U,
                    device_flag, stream) == cudaErrorInvalidValue,
                "NVFP4 launch rejects wrong mask count");
    test.expect(active_cell::launch_nvfp4_b_k_support_masks(
                    device_packed, device_scales, kOutputFeatures,
                    kInputFeatures, 1.0F, device_masks, kMaskCount, nullptr,
                    stream) == cudaErrorInvalidValue,
                "NVFP4 launch rejects null exceptional flag");
    auto* const misaligned_masks = reinterpret_cast<std::uint16_t*>(
        reinterpret_cast<std::uint8_t*>(device_masks) + 1U);
    test.expect(active_cell::launch_nvfp4_b_k_support_masks(
                    device_packed, device_scales, kOutputFeatures,
                    kInputFeatures, 1.0F, misaligned_masks, kMaskCount,
                    device_flag, stream) == cudaErrorInvalidValue,
                "NVFP4 launch rejects misaligned masks");
    auto* const misaligned_flag = reinterpret_cast<std::uint32_t*>(
        reinterpret_cast<std::uint8_t*>(device_flag) + 1U);
    test.expect(active_cell::launch_nvfp4_b_k_support_masks(
                    device_packed, device_scales, kOutputFeatures,
                    kInputFeatures, 1.0F, device_masks, kMaskCount,
                    misaligned_flag, stream) == cudaErrorInvalidValue,
                "NVFP4 launch rejects misaligned exceptional flag");
    test.expect(active_cell::launch_nvfp4_b_k_support_masks(
                    device_packed, device_scales,
                    std::numeric_limits<std::size_t>::max(), 16U, 1.0F,
                    device_masks, 1U, device_flag, stream) ==
                    cudaErrorInvalidValue,
                "NVFP4 launch rejects logical-size overflow");

    const std::array<float, 5U> invalid_scales{
        0.0F, -1.0F, std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN()};
    for (std::size_t index = 0U; index < invalid_scales.size(); ++index) {
      test.expect(active_cell::launch_nvfp4_b_k_support_masks(
                      device_packed, device_scales, kOutputFeatures,
                      kInputFeatures, invalid_scales[index], device_masks,
                      kMaskCount, device_flag, stream) ==
                      cudaErrorInvalidValue,
                  "NVFP4 rejects invalid global scale " +
                      std::to_string(index));
    }

    (void)cudaGetLastError();
    ready = test.check_cuda(active_cell::launch_nvfp4_b_k_support_masks(
                                device_packed, device_scales, kOutputFeatures,
                                kInputFeatures, 1.0F, device_masks, kMaskCount,
                                device_flag, stream),
                            "launch finite NVFP4 active-cell masks");
  }
  if (ready) {
    ready = test.check_cuda(
                cudaMemcpyAsync(copied_packed.data(), device_packed_storage,
                                copied_packed.size(), cudaMemcpyDeviceToHost,
                                stream),
                "copy back NVFP4 packed input and guards") &&
            test.check_cuda(
                cudaMemcpyAsync(copied_scales.data(), device_scale_storage,
                                copied_scales.size(), cudaMemcpyDeviceToHost,
                                stream),
                "copy back NVFP4 scales and guards") &&
            test.check_cuda(
                cudaMemcpyAsync(copied_masks.data(), device_mask_storage,
                                copied_masks.size() * sizeof(std::uint16_t),
                                cudaMemcpyDeviceToHost, stream),
                "copy back finite NVFP4 masks and guards") &&
            test.check_cuda(
                cudaMemcpyAsync(copied_flag.data(), device_flag_storage,
                                copied_flag.size() * sizeof(std::uint32_t),
                                cudaMemcpyDeviceToHost, stream),
                "copy back finite NVFP4 exceptional flag and guards");
  }
  if (ready) {
    ready = test.check_cuda(cudaStreamSynchronize(stream),
                            "synchronize NVFP4 active-cell test");
  } else {
    (void)cudaStreamSynchronize(stream);
  }
  if (ready) {
    constexpr std::array<std::uint16_t, 4U> kExpectedMasks{
        0x8014U, 0xffffU, 0xffffU, 0x8004U};
    test.expect(copied_packed == expected_packed,
                "NVFP4 packed input and guards remain unchanged");
    test.expect(copied_scales == finite_scale_storage,
                "finite NVFP4 scales and guards remain unchanged");
    expect_guards(test, copied_masks, kMaskGuards, kMaskCount, kMaskGuard,
                  "NVFP4 masks");
    expect_masks(test, copied_masks, kMaskGuards, kExpectedMasks,
                 "finite NVFP4");
    expect_guards(test, copied_flag, kFlagGuards, 1U, kFlagGuard,
                  "finite NVFP4 exceptional flag");
    test.expect(copied_flag[kFlagGuards] == 0U,
                "finite NVFP4 scales keep exceptional flag clear");
  }

  if (ready) {
    ready = test.check_cuda(
                cudaMemcpyAsync(device_packed_storage, packed_storage.data(),
                                packed_storage.size(), cudaMemcpyHostToDevice,
                                stream),
                "reset exceptional NVFP4 packed input and guards") &&
            test.check_cuda(
                cudaMemcpyAsync(device_scale_storage, scale_storage.data(),
                                scale_storage.size(), cudaMemcpyHostToDevice,
                                stream),
                "copy exceptional NVFP4 scales and guards") &&
            test.check_cuda(
                cudaMemcpyAsync(device_mask_storage, mask_storage.data(),
                                mask_storage.size() * sizeof(std::uint16_t),
                                cudaMemcpyHostToDevice, stream),
                "reset exceptional NVFP4 mask guards") &&
            test.check_cuda(
                cudaMemcpyAsync(device_flag_storage, flag_storage.data(),
                                flag_storage.size() * sizeof(std::uint32_t),
                                cudaMemcpyHostToDevice, stream),
                "clear exceptional NVFP4 flag and reset guards");
  }
  if (ready) {
    (void)cudaGetLastError();
    ready = test.check_cuda(active_cell::launch_nvfp4_b_k_support_masks(
                                device_packed, device_scales, kOutputFeatures,
                                kInputFeatures, 1.0F, device_masks, kMaskCount,
                                device_flag, stream),
                            "launch exceptional NVFP4 active-cell masks");
  }
  if (ready) {
    ready = test.check_cuda(
                cudaMemcpyAsync(copied_packed.data(), device_packed_storage,
                                copied_packed.size(), cudaMemcpyDeviceToHost,
                                stream),
                "copy back exceptional NVFP4 packed input and guards") &&
            test.check_cuda(
                cudaMemcpyAsync(copied_scales.data(), device_scale_storage,
                                copied_scales.size(), cudaMemcpyDeviceToHost,
                                stream),
                "copy back exceptional NVFP4 scales and guards") &&
            test.check_cuda(
                cudaMemcpyAsync(copied_masks.data(), device_mask_storage,
                                copied_masks.size() * sizeof(std::uint16_t),
                                cudaMemcpyDeviceToHost, stream),
                "copy back exceptional NVFP4 masks and guards") &&
            test.check_cuda(
                cudaMemcpyAsync(copied_flag.data(), device_flag_storage,
                                copied_flag.size() * sizeof(std::uint32_t),
                                cudaMemcpyDeviceToHost, stream),
                "copy back exceptional NVFP4 flag and guards");
  }
  if (ready) {
    ready = test.check_cuda(cudaStreamSynchronize(stream),
                            "synchronize exceptional NVFP4 test");
  } else {
    (void)cudaStreamSynchronize(stream);
  }
  if (ready) {
    constexpr std::array<std::uint16_t, 4U> kExpectedMasks{
        0x8014U, 0xffffU, 0xffffU, 0x8004U};
    test.expect(copied_packed == expected_packed,
                "exceptional NVFP4 packed input and guards remain unchanged");
    test.expect(copied_scales == expected_scales,
                "exceptional NVFP4 scales and guards remain unchanged");
    expect_guards(test, copied_masks, kMaskGuards, kMaskCount, kMaskGuard,
                  "exceptional NVFP4 masks");
    expect_masks(test, copied_masks, kMaskGuards, kExpectedMasks,
                 "exceptional NVFP4");
    expect_guards(test, copied_flag, kFlagGuards, 1U, kFlagGuard,
                  "exceptional NVFP4 flag");
    test.expect(copied_flag[kFlagGuards] == 1U,
                "invalid NVFP4 block scale sets exceptional flag");
  }

  release_device(test, device_flag_storage, "NVFP4 exceptional flag");
  release_device(test, device_mask_storage, "NVFP4 masks");
  release_device(test, device_scale_storage, "NVFP4 scales");
  release_device(test, device_packed_storage, "NVFP4 packed weights");
}

}  // namespace

int main() {
  int device_count = 0;
  const cudaError_t device_status = cudaGetDeviceCount(&device_count);
  if (device_status == cudaErrorNoDevice ||
      (device_status == cudaSuccess && device_count == 0)) {
    std::cout << "SKIP: CUDA device unavailable\n";
    (void)cudaGetLastError();
    return 77;
  }
  if (device_status != cudaSuccess) {
    std::cerr << "cudaGetDeviceCount failed: "
              << cudaGetErrorString(device_status) << '\n';
    return 1;
  }
  int device = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    std::cerr << "cudaGetDevice failed: " << cudaGetErrorString(status)
              << '\n';
    return 1;
  }
  cudaDeviceProp properties{};
  status = cudaGetDeviceProperties(&properties, device);
  if (status != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties failed: "
              << cudaGetErrorString(status) << '\n';
    return 1;
  }
  if (properties.major != 8 || properties.minor != 7) {
    std::cout << "SKIP: exact SM87 device required\n";
    return 77;
  }

  TestContext test;
  test_layout_contract(test);

  cudaStream_t stream = nullptr;
  const bool stream_ready = test.check_cuda(
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
      "create active-cell test stream");
  if (stream_ready) {
    test_bf16_a_masks(test, stream);
    test_fp8_b_masks(test, stream);
    test_nvfp4_b_masks(test, stream);
    (void)cudaStreamSynchronize(stream);
  }
  if (stream != nullptr) {
    const cudaError_t destroy_status = cudaStreamDestroy(stream);
    test.expect(destroy_status == cudaSuccess,
                std::string("destroy active-cell test stream: ") +
                    cudaGetErrorString(destroy_status));
  }

  if (test.failures() != 0) {
    std::cerr << test.failures()
              << " active-cell CUDA contract assertion(s) failed\n";
    return 1;
  }
  std::cout << "SM87 AOT active-cell CUDA numerical contract passed\n";
  return 0;
}
