#include "sm87_target_aot_engine_rope_internal.h"

#include <cuda_runtime_api.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <vector>

namespace q3x::runtime::sm87_target_aot_p40_executor_detail {
namespace {

inline constexpr float kRopeTheta = 10'000'000.0F;
std::atomic<std::uint64_t> g_engine_rope_identity{1U};

[[nodiscard]] float bf16_round_trip(const float value) noexcept {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t least_significant_retained_bit =
      (bits >> 16U) & 1U;
  bits += 0x7FFFU + least_significant_retained_bit;
  bits &= 0xFFFF0000U;
  float rounded = 0.0F;
  std::memcpy(&rounded, &bits, sizeof(rounded));
  return rounded;
}

[[nodiscard]] Sm87TargetAotEngineRopeStatus failure(
    const Sm87TargetAotEngineRopeError code, const char* const context,
    const cudaError_t cuda_error = cudaSuccess) noexcept {
  return {code, context, static_cast<int>(cuda_error)};
}

struct PendingCudaResources final {
  float* allocation = nullptr;
  cudaStream_t stream = nullptr;

  PendingCudaResources() noexcept = default;
  PendingCudaResources(const PendingCudaResources&) = delete;
  PendingCudaResources& operator=(const PendingCudaResources&) = delete;
  ~PendingCudaResources() {
    if (stream != nullptr) {
      (void)cudaStreamDestroy(stream);
    }
    if (allocation != nullptr) {
      (void)cudaFree(allocation);
    }
    (void)cudaGetLastError();
  }
};

}  // namespace

Sm87TargetAotEngineRopeOwner::~Sm87TargetAotEngineRopeOwner() {
  if (allocation_ != nullptr) {
    (void)cudaFree(allocation_);
    allocation_ = nullptr;
  }
  identity_ = 0U;
  device_ordinal_ = -1;
  (void)cudaGetLastError();
}

Sm87TargetAotP40EngineRope
Sm87TargetAotEngineRopeOwner::view() const noexcept {
  if (allocation_ == nullptr || identity_ == 0U || device_ordinal_ < 0) {
    return {};
  }
  constexpr std::size_t kTableElements =
      kSm87TargetAotEngineRopePositions * kSm87TargetAotEngineRopePairs;
  return {allocation_, allocation_ + kTableElements,
          kSm87TargetAotEngineRopePositions,
          kSm87TargetAotEngineRopePairs, identity_};
}

Sm87TargetAotEngineRopeCreateResult create_sm87_target_aot_engine_rope(
    const std::uint64_t minimum_free_bytes_after_create) noexcept {
  Sm87TargetAotEngineRopeCreateResult result;
#if !defined(Q3X_ENABLE_SM87_TARGET_AOT_P40_EXECUTOR_V1_ADMISSION)
  (void)minimum_free_bytes_after_create;
  result.status = failure(Sm87TargetAotEngineRopeError::kAdmissionDisabled,
                          "target_aot_p40_executor_admission");
  return result;
#else
  try {
    int device = -1;
    cudaError_t cuda_status = cudaGetDevice(&device);
    if (cuda_status != cudaSuccess || device < 0) {
      result.status = failure(Sm87TargetAotEngineRopeError::kCudaFailure,
                              "cudaGetDevice(engine_rope)", cuda_status);
      return result;
    }
    std::size_t free_bytes = 0U;
    std::size_t total_bytes = 0U;
    cuda_status = cudaMemGetInfo(&free_bytes, &total_bytes);
    (void)total_bytes;
    if (cuda_status != cudaSuccess ||
        kSm87TargetAotEngineRopeAllocationBytes > free_bytes ||
        minimum_free_bytes_after_create >
            static_cast<std::uint64_t>(free_bytes) -
                kSm87TargetAotEngineRopeAllocationBytes) {
      result.status = failure(
          Sm87TargetAotEngineRopeError::kInsufficientDeviceMemory,
          "cudaMemGetInfo(engine_rope)", cuda_status);
      return result;
    }

    constexpr std::size_t kTableElements =
        kSm87TargetAotEngineRopePositions * kSm87TargetAotEngineRopePairs;
    std::vector<float> host_tables(2U * kTableElements);
    float* const cosines = host_tables.data();
    float* const sines = host_tables.data() + kTableElements;
    std::array<float, kSm87TargetAotEngineRopePairs> inverse_frequency{};
    for (std::size_t pair = 0U; pair < inverse_frequency.size(); ++pair) {
      const float exponent =
          2.0F * static_cast<float>(pair) / 64.0F;
      inverse_frequency[pair] =
          1.0F / std::pow(kRopeTheta, exponent);
    }
    for (std::size_t position = 0U;
         position < kSm87TargetAotEngineRopePositions; ++position) {
      for (std::size_t pair = 0U;
           pair < kSm87TargetAotEngineRopePairs; ++pair) {
        const float angle =
            static_cast<float>(position) * inverse_frequency[pair];
        const std::size_t index =
            position * kSm87TargetAotEngineRopePairs + pair;
        cosines[index] = bf16_round_trip(std::cos(angle));
        sines[index] = bf16_round_trip(std::sin(angle));
      }
    }

    PendingCudaResources pending;
    void* allocation = nullptr;
    cuda_status = cudaMalloc(
        &allocation,
        static_cast<std::size_t>(kSm87TargetAotEngineRopeAllocationBytes));
    if (cuda_status != cudaSuccess || allocation == nullptr) {
      result.status = failure(Sm87TargetAotEngineRopeError::kCudaFailure,
                              "cudaMalloc(engine_rope)", cuda_status);
      return result;
    }
    pending.allocation = static_cast<float*>(allocation);
    cuda_status = cudaStreamCreateWithFlags(&pending.stream,
                                            cudaStreamNonBlocking);
    if (cuda_status != cudaSuccess) {
      result.status = failure(Sm87TargetAotEngineRopeError::kCudaFailure,
                              "cudaStreamCreate(engine_rope)", cuda_status);
      return result;
    }
    cuda_status = cudaMemcpyAsync(
        pending.allocation, cosines,
        static_cast<std::size_t>(kSm87TargetAotEngineRopeTableBytes),
        cudaMemcpyHostToDevice, pending.stream);
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaMemcpyAsync(
          pending.allocation + kTableElements, sines,
          static_cast<std::size_t>(kSm87TargetAotEngineRopeTableBytes),
          cudaMemcpyHostToDevice, pending.stream);
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaStreamSynchronize(pending.stream);
    }
    if (cuda_status != cudaSuccess) {
      result.status = failure(Sm87TargetAotEngineRopeError::kCudaFailure,
                              "upload(engine_rope)", cuda_status);
      return result;
    }
    cudaPointerAttributes attributes{};
    cuda_status = cudaPointerGetAttributes(&attributes, pending.allocation);
    if (cuda_status != cudaSuccess ||
        attributes.type != cudaMemoryTypeDevice ||
        attributes.device != device ||
        reinterpret_cast<std::uintptr_t>(pending.allocation) % 16U != 0U) {
      result.status = failure(Sm87TargetAotEngineRopeError::kCudaFailure,
                              "validate(engine_rope)", cuda_status);
      return result;
    }
    std::uint64_t identity =
        g_engine_rope_identity.fetch_add(1U, std::memory_order_relaxed);
    if (identity == 0U) {
      identity =
          g_engine_rope_identity.fetch_add(1U, std::memory_order_relaxed);
    }
    if (identity == 0U) {
      result.status = failure(Sm87TargetAotEngineRopeError::kIdentityFailure,
                              "identity(engine_rope)");
      return result;
    }
    cuda_status = cudaStreamDestroy(pending.stream);
    if (cuda_status != cudaSuccess) {
      result.status = failure(Sm87TargetAotEngineRopeError::kCudaFailure,
                              "cudaStreamDestroy(engine_rope)", cuda_status);
      return result;
    }
    pending.stream = nullptr;
    result.owner.reset(new (std::nothrow) Sm87TargetAotEngineRopeOwner(
        pending.allocation, identity, device));
    if (result.owner == nullptr) {
      result.status = failure(
          Sm87TargetAotEngineRopeError::kHostAllocationFailure,
          "owner(engine_rope)");
      return result;
    }
    pending.allocation = nullptr;
    return result;
  } catch (const std::bad_alloc&) {
    result.status = failure(
        Sm87TargetAotEngineRopeError::kHostAllocationFailure,
        "host_tables(engine_rope)");
    return result;
  } catch (...) {
    result.status = failure(Sm87TargetAotEngineRopeError::kInvalidArgument,
                            "exception(engine_rope)");
    return result;
  }
#endif
}

const char* to_string(const Sm87TargetAotEngineRopeError error) noexcept {
  switch (error) {
    case Sm87TargetAotEngineRopeError::kNone:
      return "none";
    case Sm87TargetAotEngineRopeError::kAdmissionDisabled:
      return "admission_disabled";
    case Sm87TargetAotEngineRopeError::kInvalidArgument:
      return "invalid_argument";
    case Sm87TargetAotEngineRopeError::kInsufficientDeviceMemory:
      return "insufficient_device_memory";
    case Sm87TargetAotEngineRopeError::kHostAllocationFailure:
      return "host_allocation_failure";
    case Sm87TargetAotEngineRopeError::kCudaFailure:
      return "cuda_failure";
    case Sm87TargetAotEngineRopeError::kIdentityFailure:
      return "identity_failure";
  }
  return "unknown";
}

}  // namespace q3x::runtime::sm87_target_aot_p40_executor_detail
