#include "q3x/kernels/sm87_macrofeed_v4_fp8.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace kernels = q3x::kernels;
using Role = kernels::Sm87TargetAotProjectionRole;
using InputLayout = kernels::Sm87MacroFeedV4Fp8InputLayout;

constexpr std::size_t kRows = kernels::kSm87MacroFeedV4Fp8BlockM;
constexpr std::size_t kColumns = kernels::kSm87MacroFeedV4Fp8BlockN;
constexpr std::size_t kInputFeatures =
    kernels::kSm87MacroFeedV4Fp8TestInputFeatures;
constexpr std::size_t kGuardBytes = 64U;
constexpr std::uint8_t kGuardByte = 0xa5U;
constexpr std::uint16_t kPoisonBits = 0x7fc1U;
constexpr std::uint16_t kGateSentinelBits = 0x3f80U;
constexpr std::uint16_t kScratchGapSentinelBits = 0xc000U;
constexpr std::uint16_t kScaleBits =
    kernels::sm87_target_aot_fp8_compensated_tensor_scale_bf16_bits(
        0x3f80'0000U);
static_assert(kScaleBits != 0U);

bool expect(const bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
  }
  return condition;
}

[[nodiscard]] std::uint16_t encode_bf16_rne(const float value) noexcept {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t magnitude = bits & 0x7fff'ffffU;
  if (magnitude > 0x7f80'0000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

class GuardedDeviceBuffer final {
 public:
  GuardedDeviceBuffer() = default;
  GuardedDeviceBuffer(const GuardedDeviceBuffer&) = delete;
  GuardedDeviceBuffer& operator=(const GuardedDeviceBuffer&) = delete;

  ~GuardedDeviceBuffer() {
    if (allocation_ != nullptr) {
      (void)cudaFree(allocation_);
    }
  }

  [[nodiscard]] bool allocate(const std::size_t bytes) noexcept {
    if (bytes == 0U || allocation_ != nullptr ||
        cudaMalloc(reinterpret_cast<void**>(&allocation_),
                   bytes + 2U * kGuardBytes) != cudaSuccess) {
      return false;
    }
    bytes_ = bytes;
    pointer_ = allocation_ + kGuardBytes;
    return true;
  }

  [[nodiscard]] bool upload(const void* const source,
                            const cudaStream_t stream) noexcept {
    return source != nullptr &&
           cudaMemsetAsync(allocation_, kGuardByte,
                           bytes_ + 2U * kGuardBytes, stream) == cudaSuccess &&
           cudaMemcpyAsync(pointer_, source, bytes_, cudaMemcpyHostToDevice,
                           stream) == cudaSuccess;
  }

  [[nodiscard]] bool download(void* const destination,
                              const cudaStream_t stream) const noexcept {
    return destination != nullptr &&
           cudaMemcpyAsync(destination, pointer_, bytes_,
                           cudaMemcpyDeviceToHost, stream) == cudaSuccess;
  }

  [[nodiscard]] bool guards_unchanged() const noexcept {
    std::array<std::uint8_t, kGuardBytes> prefix{};
    std::array<std::uint8_t, kGuardBytes> suffix{};
    if (cudaMemcpy(prefix.data(), allocation_, prefix.size(),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(suffix.data(), pointer_ + bytes_, suffix.size(),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      return false;
    }
    return std::all_of(prefix.begin(), prefix.end(), [](const auto byte) {
             return byte == kGuardByte;
           }) &&
           std::all_of(suffix.begin(), suffix.end(), [](const auto byte) {
             return byte == kGuardByte;
           });
  }

  [[nodiscard]] std::uint8_t* data() noexcept { return pointer_; }

 private:
  std::uint8_t* allocation_ = nullptr;
  std::uint8_t* pointer_ = nullptr;
  std::size_t bytes_ = 0U;
};

class RawDeviceAllocation final {
 public:
  RawDeviceAllocation() = default;
  RawDeviceAllocation(const RawDeviceAllocation&) = delete;
  RawDeviceAllocation& operator=(const RawDeviceAllocation&) = delete;

  ~RawDeviceAllocation() {
    if (pointer_ != nullptr) {
      (void)cudaFree(pointer_);
    }
  }

  [[nodiscard]] bool allocate(const std::size_t bytes) noexcept {
    if (bytes == 0U || pointer_ != nullptr ||
        cudaMalloc(&pointer_, bytes) != cudaSuccess) {
      return false;
    }
    bytes_ = bytes;
    return true;
  }

  [[nodiscard]] void* data() noexcept { return pointer_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

 private:
  void* pointer_ = nullptr;
  std::size_t bytes_ = 0U;
};

[[nodiscard]] kernels::Sm87TargetAotProjectionSha256Digest make_digest(
    const std::uint64_t seed) noexcept {
  kernels::Sm87TargetAotProjectionSha256Digest digest;
  std::uint64_t state = seed | 1U;
  for (std::size_t index = 0U; index < digest.bytes.size(); ++index) {
    state ^= state << 13U;
    state ^= state >> 7U;
    state ^= state << 17U;
    digest.bytes[index] = static_cast<std::uint8_t>(state >> 24U);
  }
  digest.bytes[0U] = 0x7fU;
  digest.bytes[1U] = 0xffU;
  return digest;
}

[[nodiscard]] kernels::Sm87TargetAotProjectionPackedSourceInventory
make_inventory(
    const kernels::Sm87TargetAotProjectionPackedLayout& layout) noexcept {
  kernels::Sm87TargetAotProjectionPackedSourceInventory inventory;
  inventory.identity = 0x5133'4650'3849'4e56ULL;
  inventory.role = layout.role;
  inventory.source_count = layout.partition_count;
  for (std::size_t index = 0U; index < layout.partition_count; ++index) {
    inventory.sources[index] =
        kernels::sm87_target_aot_projection_packed_source_binding(
            layout, index, 0x5133'4650'3853'0001ULL + index,
            make_digest(2U * index + 1U),
            make_digest(2U * index + 2U),
            0x0380'0000U + static_cast<std::uint32_t>(index) * 0x0001'0000U);
  }
  return inventory;
}

[[nodiscard]] kernels::Sm87TargetAotProjectionPackedTransformReceipt
make_transform_receipt(
    const kernels::Sm87TargetAotProjectionPackedLayout& layout,
    const kernels::Sm87TargetAotProjectionPackedSourceInventory& inventory,
    const kernels::Sm87TargetAotProjectionPackedManifest& manifest) noexcept {
  kernels::Sm87TargetAotProjectionPackedTransformReceipt receipt;
  receipt.artifact_identity = manifest.artifact_identity;
  receipt.source_inventory_identity = inventory.identity;
  receipt.role = layout.role;
  receipt.plan_identity = layout.plan_identity;
  receipt.layout_identity = layout.layout_identity;
  receipt.encoding = layout.encoding;
  receipt.transform_identity = kernels::
      Sm87TargetAotProjectionPackedTransformIdentity::
          kCanonicalNkToConsumerN64K16LaneComponentV1;
  receipt.partition_count = layout.partition_count;
  receipt.payload = {manifest.artifact_identity,
                     manifest.payload_offset,
                     manifest.payload_bytes,
                     manifest.payload_digest,
                     true};
  receipt.deterministic_transform = true;
  receipt.no_arithmetic_conversion = true;
  receipt.no_request_time_repacking = true;
  for (std::size_t index = 0U; index < layout.partition_count; ++index) {
    const auto& partition = layout.partitions[index];
    const auto& source = inventory.sources[index];
    auto& observed = receipt.partitions[index];
    const std::uint64_t values =
        static_cast<std::uint64_t>(partition.output_features) *
        partition.input_features;
    observed.logical_role = partition.logical_role;
    observed.partition_index = static_cast<std::uint32_t>(index);
    observed.tensor_identity = source.tensor_identity;
    observed.observed_source_weight_digest = source.weight_digest;
    observed.observed_source_scale_digest = source.scale_digest;
    observed.source_weight_bytes_hashed = values;
    observed.source_scale_bytes_hashed = sizeof(std::uint32_t);
    observed.repacked_weight_values = values;
    observed.payload_offset = partition.payload_offset;
    observed.payload_bytes = partition.payload_bytes;
    observed.source_digests_computed_from_tensor_bytes = true;
    observed.canonical_address_bijection_applied = true;
    observed.bit_exact_weight_permutation = true;
    observed.tensor_scale_kept_external = true;
  }
  return receipt;
}

struct LiveFp8AssetFixture final {
  RawDeviceAllocation payload_allocation;
  kernels::Sm87TargetAotProjectionPackedSourceInventory inventory{};
  kernels::Sm87TargetAotProjectionPackedManifest manifest{};
  kernels::Sm87TargetAotProjectionPackedTransformReceipt transform{};
  kernels::Sm87TargetAotFp8CudaDeviceUploadReceipt upload{};
  kernels::Sm87TargetAotFp8CudaAssetView asset{};

  [[nodiscard]] bool initialize(const Role role,
                                const int device_ordinal) noexcept {
    const auto layout =
        kernels::sm87_target_aot_projection_packed_layout(role);
    if (!layout.valid() || device_ordinal < 0 ||
        !payload_allocation.allocate(layout.payload_bytes)) {
      return false;
    }
    inventory = make_inventory(layout);
    manifest = kernels::sm87_target_aot_projection_make_packed_manifest(
        role, 0x5133'4650'3841'5353ULL, inventory,
        make_digest(0x5133U));
    transform = make_transform_receipt(layout, inventory, manifest);
    if (!inventory.valid(layout) ||
        !kernels::sm87_target_aot_projection_validate_packed_manifest(
            manifest, inventory) ||
        !kernels::sm87_target_aot_projection_validate_transform_receipt(
            manifest, inventory, transform)) {
      return false;
    }

    upload.artifact_identity = manifest.artifact_identity;
    upload.source_inventory_identity = manifest.source_inventory_identity;
    upload.role = role;
    upload.plan_identity = layout.plan_identity;
    upload.layout_identity = layout.layout_identity;
    upload.transform_identity = transform.transform_identity;
    upload.host_payload_offset = manifest.payload_offset;
    upload.host_payload_bytes = manifest.payload_bytes;
    upload.host_payload_digest = manifest.payload_digest;
    upload.host_manifest_seal = manifest.seal;
    upload.tensor_scale_count = manifest.source_count;
    for (std::size_t index = 0U; index < manifest.source_count; ++index) {
      upload.tensor_scale_bits[index] =
          manifest.sources[index].tensor_scale_bits;
      upload.compensated_tensor_scale_bf16_bits[index] =
          kernels::sm87_target_aot_fp8_compensated_tensor_scale_bf16_bits(
              upload.tensor_scale_bits[index]);
    }
    upload.device_allocation_identity = 0x5133'4650'3841'4c4cULL;
    upload.device_allocation_owner_identity = 0x5133'4650'384f'574eULL;
    upload.device_ordinal = device_ordinal;
    upload.device_allocation_begin = reinterpret_cast<std::uintptr_t>(
        payload_allocation.data());
    upload.device_allocation_bytes = payload_allocation.bytes();
    upload.device_allocation_end =
        upload.device_allocation_begin + upload.device_allocation_bytes;
    upload.device_payload_begin = upload.device_allocation_begin;
    upload.device_payload_bytes = manifest.payload_bytes;
    upload.device_payload_end =
        upload.device_payload_begin + upload.device_payload_bytes;
    upload.upload_stream_owner_identity =
        upload.device_allocation_owner_identity;
    upload.upload_stream_identity = 0x5133'4650'3855'5053ULL;
    upload.upload_completion_event_identity = 0x5133'4650'3855'5045ULL;
    upload.verification_stream_owner_identity =
        upload.device_allocation_owner_identity;
    upload.verification_stream_identity = 0x5133'4650'3856'5353ULL;
    upload.verification_completion_event_identity =
        0x5133'4650'3856'4556ULL;
    upload.verification_readback_bytes = manifest.payload_bytes;
    upload.verification_readback_digest = manifest.payload_digest;
    upload.host_payload_digest_verified_before_copy = true;
    upload.host_payload_immutable_until_completion = true;
    upload.copy_enqueued_to_exact_payload_range = true;
    upload.completion_event_recorded_after_copy = true;
    upload.completion_event_observed = true;
    upload.upload_completed = true;
    upload.verification_copy_enqueued_from_exact_payload_range = true;
    upload.verification_event_recorded_after_copy = true;
    upload.verification_event_observed = true;
    upload.verification_completed = true;
    upload.device_payload_matches_host_payload = true;
    upload.allocation_retained_for_asset_lifetime = true;
    upload.receipt_identity =
        kernels::sm87_target_aot_fp8_cuda_compute_upload_receipt_identity(
            upload);
    asset = kernels::sm87_target_aot_bind_fp8_cuda_asset(
        manifest, inventory, transform, upload);
    return kernels::sm87_target_aot_fp8_cuda_asset_valid(asset);
  }
};

[[nodiscard]] __device__ __forceinline__ float decode_bf16(
    const std::uint16_t bits) noexcept {
  return __uint_as_float(static_cast<std::uint32_t>(bits) << 16U);
}

[[nodiscard]] __device__ __forceinline__ std::uint16_t device_encode_bf16(
    const float value) noexcept {
  unsigned int bits = __float_as_uint(value);
  const unsigned int magnitude = bits & 0x7fff'ffffU;
  if (magnitude > 0x7f80'0000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] __device__ __forceinline__ std::uint8_t reference_raw_code(
    const std::uint8_t* const payload, const unsigned int k,
    const unsigned int column, const unsigned int n128_half) noexcept {
  const unsigned int k_tile = k / 64U;
  const unsigned int k16 = (k % 64U) / 16U;
  const unsigned int element = k % 16U;
  const unsigned int canonical_column = n128_half * 128U + column;
  const unsigned int local_column = canonical_column % 128U;
  const unsigned int n8 = local_column / 8U;
  const unsigned int lane =
      (local_column % 8U) * 4U + (element % 8U) / 2U;
  const unsigned int component =
      element < 8U ? ((element & 1U) == 0U ? 0U : 2U)
                   : ((element & 1U) == 0U ? 1U : 3U);
  const std::size_t offset =
      static_cast<std::size_t>(k_tile) * 16'384U +
      static_cast<std::size_t>(k16) * 4'096U +
      static_cast<std::size_t>(n128_half) * 2'048U +
      static_cast<std::size_t>(n8) * 128U +
      static_cast<std::size_t>(lane) * 4U + component;
  return payload[offset];
}

[[nodiscard]] __device__ __forceinline__ float reference_dot(
    const Role role, const InputLayout input_layout,
    const std::uint16_t* const input,
    const std::size_t input_row_stride, const std::uint8_t* const payload,
    const unsigned int row, const unsigned int column,
    const unsigned int n128_half,
    const unsigned int logical_input_first_k) noexcept {
  float accumulator = 0.0F;
#pragma unroll 1
  for (unsigned int k = 0U; k < kInputFeatures; ++k) {
    const std::uint8_t code =
        reference_raw_code(payload, k, column, n128_half);
    const std::uint16_t biased_bits = static_cast<std::uint16_t>(
        (static_cast<std::uint16_t>(code & 0x80U) << 8U) |
        (static_cast<std::uint16_t>(code & 0x7fU) << 4U));
    const unsigned int logical_k = logical_input_first_k + k;
    unsigned int physical_k = logical_k;
    if (input_layout == InputLayout::kFullAttentionInterleavedQScratchV1) {
      physical_k = (logical_k / 256U) * 512U + logical_k % 256U;
    } else if (input_layout == InputLayout::kGdnContiguousVScratchV1) {
      physical_k = static_cast<unsigned int>(
                       kernels::
                           kSm87MacroFeedV4Fp8GdnAttentionOutputPhysicalOffset) +
                   logical_k;
    }
    accumulator = fmaf(
        decode_bf16(input[static_cast<std::size_t>(row) * input_row_stride +
                          physical_k]),
        decode_bf16(biased_bits), accumulator);
  }
  return accumulator;
}

__global__ void fp8_reference_kernel(
    const Role role, const InputLayout input_layout,
    const unsigned int partition,
    const unsigned int partition_n256_tile,
    const std::uint16_t* const input, const std::size_t input_row_stride,
    const std::uint8_t* const payload,
    const std::uint16_t compensated_scale_bits,
    const unsigned int valid_rows, const unsigned int n128_half,
    const unsigned int logical_input_first_k,
    std::uint16_t* const primary_output,
    const std::size_t primary_output_row_stride,
    std::uint16_t* const key_output,
    const std::size_t key_output_row_stride,
    std::uint16_t* const value_output,
    const std::size_t value_output_row_stride) {
  const unsigned int linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= kRows * kColumns) {
    return;
  }
  const unsigned int row = linear / kColumns;
  const unsigned int column = linear % kColumns;
  if (row >= valid_rows) {
    return;
  }
  const float accumulator =
      reference_dot(role, input_layout, input, input_row_stride, payload, row,
                    column, n128_half, logical_input_first_k);
  // MMA starts from +0 and produces +0 for an exactly zero accumulated dot
  // under round-to-nearest. Scalar fmaf can retain a negative underflow zero
  // from the final isolated product, so model the pinned MMA zero sign here.
  const float mma_accumulator = accumulator == 0.0F ? 0.0F : accumulator;
  const std::uint16_t bits = device_encode_bf16(
      mma_accumulator * decode_bf16(compensated_scale_bits));

  std::uint16_t* destination = primary_output;
  std::size_t destination_row_stride = primary_output_row_stride;
  const unsigned int partition_local_column =
      partition_n256_tile * 256U + n128_half * kColumns + column;
  unsigned int destination_column = partition_local_column;
  if (role == Role::kFp8GdnQkvZ && partition == 1U) {
    destination_column =
        kernels::kSm87MacroFeedV4Fp8GdnZOffset +
        partition_local_column;
  } else if (role == Role::kFp8FullQkv && partition == 0U) {
    const bool gate =
        partition_local_column >=
        kernels::kSm87MacroFeedV4Fp8FullQFeatures;
    const unsigned int local =
        gate ? partition_local_column -
                   kernels::kSm87MacroFeedV4Fp8FullQFeatures
             : partition_local_column;
    destination_column =
        (local / kernels::kSm87MacroFeedV4Fp8AttentionHeadFeatures) *
            kernels::kSm87MacroFeedV4Fp8QGateHeadStride +
        local % kernels::kSm87MacroFeedV4Fp8AttentionHeadFeatures +
        (gate ? kernels::kSm87MacroFeedV4Fp8AttentionHeadFeatures : 0U);
  } else if (role == Role::kFp8FullQkv && partition == 1U) {
    destination = key_output;
    destination_row_stride = key_output_row_stride;
  } else if (role == Role::kFp8FullQkv && partition == 2U) {
    destination = value_output;
    destination_row_stride = value_output_row_stride;
  }
  destination[static_cast<std::size_t>(row) * destination_row_stride +
              destination_column] = bits;
}

void build_input(std::vector<std::uint16_t>& input, const Role role,
                 const InputLayout input_layout,
                 const std::size_t row_stride,
                 const std::size_t logical_input_first_k) {
  std::fill(input.begin(), input.end(),
            role == Role::kFp8AttentionOutput ? kScratchGapSentinelBits : 0U);
  for (std::size_t row = 0U; row < kRows; ++row) {
    if (input_layout == InputLayout::kFullAttentionInterleavedQScratchV1) {
      for (std::size_t head = 0U;
           head < kernels::kSm87MacroFeedV4Fp8AttentionHeads; ++head) {
        const std::size_t head_base =
            row * row_stride +
            head * kernels::kSm87MacroFeedV4Fp8QGateHeadStride;
        std::fill_n(input.begin() + static_cast<std::ptrdiff_t>(head_base),
                    kernels::kSm87MacroFeedV4Fp8AttentionHeadFeatures, 0U);
        std::fill_n(
            input.begin() + static_cast<std::ptrdiff_t>(
                                head_base +
                                kernels::kSm87MacroFeedV4Fp8AttentionHeadFeatures),
            kernels::kSm87MacroFeedV4Fp8AttentionHeadFeatures,
            kGateSentinelBits);
      }
    } else if (input_layout == InputLayout::kGdnContiguousVScratchV1) {
      const std::size_t v_begin =
          row * row_stride +
          kernels::kSm87MacroFeedV4Fp8GdnAttentionOutputPhysicalOffset;
      std::fill_n(
          input.begin() + static_cast<std::ptrdiff_t>(v_begin),
          kernels::kSm87MacroFeedV4Fp8AttentionOutputInputFeatures, 0U);
    }
    // One nonzero BF16 operand makes the independent scalar oracle immune to
    // a different zero-sign or reassociation choice while rows collectively
    // cover every K64 pipeline stage and all K16 fragments.
    const std::size_t k = (row * 37U + 11U) % kInputFeatures;
    const std::size_t logical_k = logical_input_first_k + k;
    std::size_t physical_k = logical_k;
    if (input_layout == InputLayout::kFullAttentionInterleavedQScratchV1) {
      physical_k =
          (logical_k /
           kernels::kSm87MacroFeedV4Fp8AttentionHeadFeatures) *
              kernels::kSm87MacroFeedV4Fp8QGateHeadStride +
          logical_k % kernels::kSm87MacroFeedV4Fp8AttentionHeadFeatures;
    } else if (input_layout == InputLayout::kGdnContiguousVScratchV1) {
      physical_k =
          kernels::kSm87MacroFeedV4Fp8GdnAttentionOutputPhysicalOffset +
          logical_k;
    }
    const float sign = (row & 1U) == 0U ? 1.0F : -1.0F;
    input[row * row_stride + physical_k] = encode_bf16_rne(sign);
  }
}

[[nodiscard]] bool attention_gate_and_gap_unchanged(
    const std::vector<std::uint16_t>& before,
    const std::vector<std::uint16_t>& after,
    const std::size_t row_stride) {
  for (std::size_t row = 0U; row < kRows; ++row) {
    const std::size_t row_base = row * row_stride;
    for (std::size_t head = 0U;
         head < kernels::kSm87MacroFeedV4Fp8AttentionHeads; ++head) {
      const std::size_t gate_begin =
          row_base + head * kernels::kSm87MacroFeedV4Fp8QGateHeadStride +
          kernels::kSm87MacroFeedV4Fp8AttentionHeadFeatures;
      for (std::size_t lane = 0U;
           lane < kernels::kSm87MacroFeedV4Fp8AttentionHeadFeatures; ++lane) {
        const std::size_t index = gate_begin + lane;
        if (before[index] != kGateSentinelBits ||
            after[index] != before[index]) {
          return false;
        }
      }
    }
    for (std::size_t physical =
             kernels::kSm87MacroFeedV4Fp8FullQGateFeatures;
         physical < row_stride; ++physical) {
      const std::size_t index = row_base + physical;
      if (before[index] != kScratchGapSentinelBits ||
          after[index] != before[index]) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] bool gdn_non_v_scratch_unchanged(
    const std::vector<std::uint16_t>& before,
    const std::vector<std::uint16_t>& after,
    const std::size_t row_stride) {
  const std::size_t v_begin =
      kernels::kSm87MacroFeedV4Fp8GdnAttentionOutputPhysicalOffset;
  const std::size_t v_end =
      v_begin + kernels::kSm87MacroFeedV4Fp8AttentionOutputInputFeatures;
  for (std::size_t row = 0U; row < kRows; ++row) {
    const std::size_t row_base = row * row_stride;
    for (std::size_t physical = 0U; physical < row_stride; ++physical) {
      if (physical >= v_begin && physical < v_end) {
        continue;
      }
      const std::size_t index = row_base + physical;
      if (before[index] != kScratchGapSentinelBits ||
          after[index] != before[index]) {
        return false;
      }
    }
  }
  return true;
}

void build_payload(std::vector<std::uint8_t>& payload) {
  for (std::size_t index = 0U; index < payload.size(); ++index) {
    std::uint8_t code = static_cast<std::uint8_t>(
        index * 29U + (index / 256U) * 17U + (index / 4'096U) * 73U);
    if ((code & 0x7fU) == 0U) {
      code = static_cast<std::uint8_t>(code + 1U);
    }
    payload[index] = code;
  }
}

[[nodiscard]] const char* role_name(const Role role) noexcept {
  if (role == Role::kFp8GdnQkvZ) {
    return "GDN-QKVZ";
  }
  if (role == Role::kFp8FullQkv) {
    return "Full-QKV";
  }
  return "Attention-O";
}

struct HostOutputs final {
  std::vector<std::uint16_t> primary;
  std::vector<std::uint16_t> key;
  std::vector<std::uint16_t> value;
};

struct DeviceOutputs final {
  GuardedDeviceBuffer primary;
  GuardedDeviceBuffer key;
  GuardedDeviceBuffer value;
};

[[nodiscard]] bool allocate_outputs(const Role role, HostOutputs& host,
                                    DeviceOutputs& device) {
  const std::size_t primary_stride =
      role == Role::kFp8AttentionOutput
          ? kernels::kSm87MacroFeedV4Fp8HiddenRowStride
          : kernels::kSm87MacroFeedV4Fp8ScratchRowStride;
  host.primary.assign(kRows * primary_stride, kPoisonBits);
  if (!device.primary.allocate(host.primary.size() * sizeof(std::uint16_t))) {
    return false;
  }
  if (role == Role::kFp8FullQkv) {
    host.key.assign(kRows * kernels::kSm87MacroFeedV4Fp8KvNhdRowStride,
                    kPoisonBits);
    host.value.assign(kRows * kernels::kSm87MacroFeedV4Fp8KvNhdRowStride,
                      kPoisonBits);
    return device.key.allocate(host.key.size() * sizeof(std::uint16_t)) &&
           device.value.allocate(host.value.size() * sizeof(std::uint16_t));
  }
  return true;
}

[[nodiscard]] bool upload_outputs(const HostOutputs& host,
                                  DeviceOutputs& device,
                                  const cudaStream_t stream) {
  return device.primary.upload(host.primary.data(), stream) &&
         (host.key.empty() || device.key.upload(host.key.data(), stream)) &&
         (host.value.empty() ||
          device.value.upload(host.value.data(), stream));
}

[[nodiscard]] bool download_outputs(HostOutputs& host,
                                    const DeviceOutputs& device,
                                    const cudaStream_t stream) {
  return device.primary.download(host.primary.data(), stream) &&
         (host.key.empty() || device.key.download(host.key.data(), stream)) &&
         (host.value.empty() ||
          device.value.download(host.value.data(), stream));
}

[[nodiscard]] std::uint16_t* optional_pointer(GuardedDeviceBuffer& buffer,
                                              const bool live) noexcept {
  return live ? reinterpret_cast<std::uint16_t*>(buffer.data()) : nullptr;
}

[[nodiscard]] bool output_guards_unchanged(
    const Role role, const DeviceOutputs& outputs) noexcept {
  return outputs.primary.guards_unchanged() &&
         (role != Role::kFp8FullQkv ||
          (outputs.key.guards_unchanged() &&
           outputs.value.guards_unchanged()));
}

[[nodiscard]] bool compare_outputs(const HostOutputs& candidate,
                                   const HostOutputs& reference,
                                   const std::string& label) {
  const auto compare = [&label](const std::vector<std::uint16_t>& left,
                                const std::vector<std::uint16_t>& right,
                                const char* const surface) {
    if (left == right) {
      return true;
    }
    const auto mismatch = std::mismatch(left.begin(), left.end(), right.begin());
    std::cerr << "FAIL: " << label << ' ' << surface << " mismatch at "
              << std::distance(left.begin(), mismatch.first) << " candidate=0x"
              << std::hex << *mismatch.first << " reference=0x"
              << *mismatch.second << std::dec << '\n';
    return false;
  };
  return compare(candidate.primary, reference.primary, "primary") &&
         compare(candidate.key, reference.key, "key") &&
         compare(candidate.value, reference.value, "value");
}

[[nodiscard]] bool run_oracle_case(const Role role,
                                   const InputLayout input_layout,
                                   const std::size_t partition,
                                   const std::size_t partition_n256_tile,
                                   const std::size_t n128_half,
                                   const std::size_t valid_rows,
                                   const cudaStream_t stream,
                                   const std::size_t logical_input_first_k_override =
                                       static_cast<std::size_t>(-1)) {
  const auto plan = kernels::sm87_macrofeed_v4_fp8_plan(
      role, kernels::kSm87MacroFeedV4Fp8Tokens, input_layout);
  if (!plan.valid() || partition >= plan.partition_count || n128_half > 1U ||
      partition_n256_tile >= plan.partition_features[partition] / 256U ||
      valid_rows == 0U || valid_rows > kRows) {
    return false;
  }
  const std::size_t input_stride =
      role == Role::kFp8AttentionOutput
          ? kernels::kSm87MacroFeedV4Fp8ScratchRowStride
          : kernels::kSm87MacroFeedV4Fp8HiddenRowStride;
  const std::size_t default_logical_input_first_k =
      role == Role::kFp8AttentionOutput
          ? (input_layout == InputLayout::kGdnContiguousVScratchV1
                 ? kernels::
                       kSm87MacroFeedV4Fp8TestGdnAttentionOutputLogicalFirstK
                 : kernels::
                       kSm87MacroFeedV4Fp8TestAttentionOutputLogicalFirstK)
          : 0U;
  const std::size_t logical_input_first_k =
      logical_input_first_k_override == static_cast<std::size_t>(-1)
          ? default_logical_input_first_k
          : logical_input_first_k_override;
  std::vector<std::uint16_t> input(kRows * input_stride, 0U);
  std::vector<std::uint16_t> input_after(input.size(), 0U);
  std::vector<std::uint8_t> payload(
      kernels::kSm87MacroFeedV4Fp8TestPayloadBytes, 0U);
  std::vector<std::uint8_t> payload_after(payload.size(), 0U);
  build_input(input, role, input_layout, input_stride,
              logical_input_first_k);
  build_payload(payload);

  GuardedDeviceBuffer device_input;
  GuardedDeviceBuffer device_payload;
  HostOutputs candidate;
  HostOutputs reference;
  DeviceOutputs device_candidate;
  DeviceOutputs device_reference;
  if (!device_input.allocate(input.size() * sizeof(std::uint16_t)) ||
      !device_payload.allocate(payload.size()) ||
      !allocate_outputs(role, candidate, device_candidate) ||
      !allocate_outputs(role, reference, device_reference) ||
      !device_input.upload(input.data(), stream) ||
      !device_payload.upload(payload.data(), stream) ||
      !upload_outputs(candidate, device_candidate, stream) ||
      !upload_outputs(reference, device_reference, stream)) {
    return false;
  }

  const std::size_t primary_stride =
      role == Role::kFp8AttentionOutput
          ? kernels::kSm87MacroFeedV4Fp8HiddenRowStride
          : kernels::kSm87MacroFeedV4Fp8ScratchRowStride;
  kernels::Sm87MacroFeedV4Fp8TileTestArguments arguments;
  arguments.role = role;
  arguments.partition_index = partition;
  arguments.partition_n256_tile = partition_n256_tile;
  arguments.input =
      reinterpret_cast<const std::uint16_t*>(device_input.data());
  arguments.input_row_stride = input_stride;
  arguments.logical_input_first_k = logical_input_first_k;
  arguments.canonical_payload_four_k64_cells = device_payload.data();
  arguments.compensated_scale_bf16_bits = kScaleBits;
  arguments.valid_rows = valid_rows;
  arguments.canonical_n128_half = n128_half;
  arguments.primary_output =
      reinterpret_cast<std::uint16_t*>(device_candidate.primary.data());
  arguments.primary_output_row_stride = primary_stride;
  arguments.key_output = optional_pointer(device_candidate.key,
                                          role == Role::kFp8FullQkv);
  arguments.key_output_row_stride =
      role == Role::kFp8FullQkv
          ? kernels::kSm87MacroFeedV4Fp8KvNhdRowStride
          : 0U;
  arguments.value_output = optional_pointer(device_candidate.value,
                                            role == Role::kFp8FullQkv);
  arguments.value_output_row_stride = arguments.key_output_row_stride;
  arguments.cuda_stream = reinterpret_cast<void*>(stream);
  arguments.input_layout = input_layout;

  const int candidate_status =
      kernels::launch_sm87_macrofeed_v4_fp8_tile_test_cuda(arguments);
  constexpr unsigned int kReferenceThreads = 256U;
  constexpr unsigned int kReferenceBlocks =
      static_cast<unsigned int>((kRows * kColumns + kReferenceThreads - 1U) /
                                kReferenceThreads);
  fp8_reference_kernel<<<kReferenceBlocks, kReferenceThreads, 0U, stream>>>(
      role, input_layout, static_cast<unsigned int>(partition),
      static_cast<unsigned int>(partition_n256_tile), arguments.input,
      input_stride, device_payload.data(), kScaleBits,
      static_cast<unsigned int>(valid_rows),
      static_cast<unsigned int>(n128_half),
      static_cast<unsigned int>(logical_input_first_k),
      reinterpret_cast<std::uint16_t*>(device_reference.primary.data()),
      primary_stride,
      optional_pointer(device_reference.key, role == Role::kFp8FullQkv),
      arguments.key_output_row_stride,
      optional_pointer(device_reference.value, role == Role::kFp8FullQkv),
      arguments.value_output_row_stride);
  const cudaError_t reference_status = cudaPeekAtLastError();
  if (candidate_status != static_cast<int>(cudaSuccess) ||
      reference_status != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess ||
      !download_outputs(candidate, device_candidate, stream) ||
      !download_outputs(reference, device_reference, stream) ||
      !device_input.download(input_after.data(), stream) ||
      !device_payload.download(payload_after.data(), stream) ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    std::cerr << "FAIL: launch/download " << role_name(role)
              << " partition=" << partition
              << " n256=" << partition_n256_tile << " half=" << n128_half
              << " rows=" << valid_rows << " candidate_status="
              << candidate_status << " reference_status="
              << static_cast<int>(reference_status) << '\n';
    return false;
  }

  const std::string label =
      std::string(role_name(role)) + " p=" + std::to_string(partition) +
      " n256=" + std::to_string(partition_n256_tile) +
      " half=" + std::to_string(n128_half) +
      " rows=" + std::to_string(valid_rows) +
      " k=" + std::to_string(logical_input_first_k);
  bool ok = compare_outputs(candidate, reference, label);
  if (input_after != input) {
    const auto mismatch =
        std::mismatch(input.begin(), input.end(), input_after.begin());
    std::cerr << "FAIL: " << label << " input mismatch at "
              << std::distance(input.begin(), mismatch.first) << " expected=0x"
              << std::hex << *mismatch.first << " observed=0x"
              << *mismatch.second << std::dec << '\n';
  }
  ok &= expect(input_after == input, label + " input mutated");
  if (role == Role::kFp8AttentionOutput) {
    if (input_layout == InputLayout::kGdnContiguousVScratchV1) {
      ok &= expect(gdn_non_v_scratch_unchanged(input, input_after,
                                                input_stride),
                   label + " non-V scratch sentinel changed");
    } else {
      ok &= expect(attention_gate_and_gap_unchanged(input, input_after,
                                                     input_stride),
                   label + " Gate/gap storage changed");
    }
  }
  ok &= expect(payload_after == payload, label + " payload mutated");
  ok &= expect(device_input.guards_unchanged(),
               label + " input redzone changed");
  ok &= expect(device_payload.guards_unchanged(),
               label + " payload redzone changed");
  ok &= expect(output_guards_unchanged(role, device_candidate),
               label + " candidate output redzone changed");
  ok &= expect(output_guards_unchanged(role, device_reference),
               label + " reference output redzone changed");

  // Boundary rejection is checked with otherwise valid live pointers.
  if (partition == 0U && partition_n256_tile == 0U && n128_half == 0U &&
      valid_rows == kRows) {
    auto invalid = arguments;
    invalid.valid_rows = 0U;
    ok &= expect(kernels::launch_sm87_macrofeed_v4_fp8_tile_test_cuda(
                     invalid) == static_cast<int>(cudaErrorInvalidValue),
                 label + " zero-row boundary must fail closed");
    invalid = arguments;
    invalid.valid_rows = kRows + 1U;
    ok &= expect(kernels::launch_sm87_macrofeed_v4_fp8_tile_test_cuda(
                     invalid) == static_cast<int>(cudaErrorInvalidValue),
                 label + " M65 boundary must fail closed");
    invalid = arguments;
    invalid.canonical_n128_half = 2U;
    ok &= expect(kernels::launch_sm87_macrofeed_v4_fp8_tile_test_cuda(
                     invalid) == static_cast<int>(cudaErrorInvalidValue),
                 label + " N128-half overflow must fail closed");
    invalid = arguments;
    invalid.partition_index = plan.partition_count;
    ok &= expect(kernels::launch_sm87_macrofeed_v4_fp8_tile_test_cuda(
                     invalid) == static_cast<int>(cudaErrorInvalidValue),
                 label + " partition overflow must fail closed");
    invalid = arguments;
    invalid.partition_n256_tile =
        plan.partition_features[partition] / 256U;
    ok &= expect(kernels::launch_sm87_macrofeed_v4_fp8_tile_test_cuda(
                     invalid) == static_cast<int>(cudaErrorInvalidValue),
                 label + " N256 tile overflow must fail closed");
    invalid = arguments;
    invalid.logical_input_first_k += 64U;
    ok &= expect(kernels::launch_sm87_macrofeed_v4_fp8_tile_test_cuda(
                     invalid) == static_cast<int>(cudaErrorInvalidValue),
                 label + " noncanonical logical K window must fail closed");
  }
  return ok;
}

[[nodiscard]] bool host_contract_test() {
  bool ok = true;
  struct PlanCase final {
    Role role;
    InputLayout input_layout;
  };
  constexpr std::array<PlanCase, 4U> kCases{{
      {Role::kFp8GdnQkvZ, InputLayout::kHiddenContiguousH5120V1},
      {Role::kFp8FullQkv, InputLayout::kHiddenContiguousH5120V1},
      {Role::kFp8AttentionOutput,
       InputLayout::kFullAttentionInterleavedQScratchV1},
      {Role::kFp8AttentionOutput,
       InputLayout::kGdnContiguousVScratchV1},
  }};
  for (const auto& test_case : kCases) {
    const Role role = test_case.role;
    const auto plan = kernels::sm87_macrofeed_v4_fp8_plan(
        role, kernels::kSm87MacroFeedV4Fp8Tokens,
        test_case.input_layout);
    ok &= expect(plan.valid(), std::string(role_name(role)) +
                                  " C8000 plan must validate");
    ok &= expect(plan.grid_m == 125U &&
                     plan.logical_tasks == plan.grid_m * plan.grid_n &&
                     plan.ordinary_full_grid && plan.m_major_n_adjacent &&
                     plan.role_specific_direct_scatter &&
                     plan.exact_fp8_marlin_semantics &&
                     plan.authenticated_asset_zero_copy &&
                     !plan.fallback_permitted && !plan.selector_present &&
                     !plan.request_jit_repack_or_autotune &&
                     !plan.numerical_contract_qualified &&
                     !plan.production_dispatch_eligible,
                 std::string(role_name(role)) +
                     " default-off ordinary-grid contract changed");
    if (role == Role::kFp8FullQkv) {
      ok &= expect(plan.full_q_gate_head_interleaved &&
                       !plan.attention_output_gathers_interleaved_q,
                   "Full Q/G must publish [24,2,256] interleaved scratch");
    } else if (role == Role::kFp8AttentionOutput &&
               test_case.input_layout ==
                   InputLayout::kFullAttentionInterleavedQScratchV1) {
      ok &= expect(
          plan.input_features == 6'144U &&
              plan.input_physical_offset == 0U &&
              plan.input_physical_span == 12'288U &&
              plan.input_row_stride == 17'408U &&
              plan.attention_output_gathers_interleaved_q &&
              plan.attention_gate_and_gap_preserved &&
              !plan.input_base_offset_permitted,
          "O must gather Q slots while preserving Gate/gap storage");
    } else if (role == Role::kFp8AttentionOutput) {
      ok &= expect(
          plan.identity == kernels::Sm87MacroFeedV4Fp8Identity::
                               kGdnAttentionOutputM64N128K64OrdinaryGridV1 &&
              plan.input_physical_offset == 4'096U &&
              plan.input_physical_span == 6'144U &&
              plan.input_row_stride == 17'408U &&
              plan.attention_output_reads_gdn_contiguous_v &&
              !plan.attention_output_gathers_interleaved_q &&
              !plan.input_base_offset_permitted,
          "GDN O must read only fixed scratch V[4096,10240)");
    }

    kernels::Sm87MacroFeedV4Fp8T1AdmissionLaunchReceipt receipt{};
    receipt.identity = plan.identity;
    receipt.role = role;
    receipt.input_layout = test_case.input_layout;
    receipt.artifact_identity = 1U;
    receipt.device_ordinal = 0;
    receipt.token_count = plan.token_count;
    receipt.logical_tasks = plan.logical_tasks;
    receipt.physical_kernel_launches = 1U;
    receipt.fallback_launches = 0U;
    receipt.ordinary_full_grid = true;
    receipt.role_specific_direct_scatter = true;
    receipt.private_nhd_kv = plan.private_nhd_kv;
    receipt.authenticated_asset_zero_copy = true;
    receipt.launch_enqueued = true;
    receipt.completion_observed = false;
    receipt.admission_only = true;
    receipt.caller_constructible_snapshot = true;
    receipt.startup_package_bound = false;
    receipt.execution_capability = false;
    receipt.current_device_matches_snapshot = true;
    receipt.asset_upload_device_matches_current = true;
    receipt.live_resource_snapshot_verified = true;
    receipt.caller_stream_non_null = true;
    receipt.stream_owner_verified = false;
    receipt.live_cuda_ranges_verified = true;
    receipt.production_dispatch_eligible = false;
    ok &= expect(receipt.valid_t1_admission_enqueue_receipt(),
                 std::string(role_name(role)) +
                     " T1 admission receipt must disclose no authority");
    auto elevated_receipt = receipt;
    elevated_receipt.startup_package_bound = true;
    ok &= expect(!elevated_receipt.valid_t1_admission_enqueue_receipt(),
                 std::string(role_name(role)) +
                     " package-bound receipt claim must fail closed");
    elevated_receipt = receipt;
    elevated_receipt.execution_capability = true;
    ok &= expect(!elevated_receipt.valid_t1_admission_enqueue_receipt(),
                 std::string(role_name(role)) +
                     " capability-bearing receipt claim must fail closed");
    elevated_receipt = receipt;
    elevated_receipt.admission_only = false;
    ok &= expect(!elevated_receipt.valid_t1_admission_enqueue_receipt(),
                 std::string(role_name(role)) +
                     " non-admission receipt claim must fail closed");
    elevated_receipt = receipt;
    elevated_receipt.live_resource_snapshot_verified = false;
    ok &= expect(!elevated_receipt.valid_t1_admission_enqueue_receipt(),
                 std::string(role_name(role)) +
                     " unverified live-resource receipt must fail closed");
    elevated_receipt = receipt;
    elevated_receipt.input_layout =
        test_case.input_layout == InputLayout::kGdnContiguousVScratchV1
            ? InputLayout::kFullAttentionInterleavedQScratchV1
            : InputLayout::kGdnContiguousVScratchV1;
    ok &= expect(!elevated_receipt.valid_t1_admission_enqueue_receipt(),
                 std::string(role_name(role)) +
                     " layout-mismatched receipt must fail closed");
  }
  ok &= expect(!kernels::sm87_macrofeed_v4_fp8_plan(
                    Role::kFp8FullQkv, 7'999U)
                    .valid(),
               "non-C8000 production shape must fail closed");

  const auto* const input = reinterpret_cast<const std::uint16_t*>(
      static_cast<std::uintptr_t>(0x1'0000'0000ULL));
  auto* const primary = reinterpret_cast<std::uint16_t*>(
      static_cast<std::uintptr_t>(0x2'0000'0000ULL));
  auto* const key = reinterpret_cast<std::uint16_t*>(
      static_cast<std::uintptr_t>(0x3'0000'0000ULL));
  auto* const value = reinterpret_cast<std::uint16_t*>(
      static_cast<std::uintptr_t>(0x4'0000'0000ULL));
  kernels::Sm87MacroFeedV4Fp8LayoutBinding full{
      Role::kFp8FullQkv,
      kernels::kSm87MacroFeedV4Fp8Tokens,
      input,
      kernels::kSm87MacroFeedV4Fp8HiddenRowStride,
      primary,
      kernels::kSm87MacroFeedV4Fp8ScratchRowStride,
      key,
      kernels::kSm87MacroFeedV4Fp8KvNhdRowStride,
      value,
      kernels::kSm87MacroFeedV4Fp8KvNhdRowStride};
  ok &= expect(kernels::sm87_macrofeed_v4_fp8_layout_valid(full),
               "Full interleaved Q/G + private NHD K/V binding must validate");
  auto invalid_full = full;
  invalid_full.key_output = nullptr;
  ok &= expect(!kernels::sm87_macrofeed_v4_fp8_layout_valid(invalid_full),
               "missing private K owner must fail closed");
  invalid_full = full;
  invalid_full.primary_output_row_stride = 12'288U;
  ok &= expect(!kernels::sm87_macrofeed_v4_fp8_layout_valid(invalid_full),
               "compact Full-QKV scratch stride must fail V4 layout");
  invalid_full = full;
  invalid_full.value_output = invalid_full.key_output;
  ok &= expect(!kernels::sm87_macrofeed_v4_fp8_layout_valid(invalid_full),
               "aliased private K/V ranges must fail closed");

  kernels::Sm87MacroFeedV4Fp8LayoutBinding gdn{
      Role::kFp8GdnQkvZ,
      kernels::kSm87MacroFeedV4Fp8Tokens,
      input,
      kernels::kSm87MacroFeedV4Fp8HiddenRowStride,
      primary,
      kernels::kSm87MacroFeedV4Fp8ScratchRowStride,
      nullptr,
      0U,
      nullptr,
      0U};
  ok &= expect(kernels::sm87_macrofeed_v4_fp8_layout_valid(gdn),
               "GDN QKV[0,10240)/Z[10240,16384) scratch binding must pass");

  kernels::Sm87MacroFeedV4Fp8LayoutBinding output{
      Role::kFp8AttentionOutput,
      kernels::kSm87MacroFeedV4Fp8Tokens,
      input,
      kernels::kSm87MacroFeedV4Fp8ScratchRowStride,
      primary,
      kernels::kSm87MacroFeedV4Fp8HiddenRowStride,
      nullptr,
      0U,
      nullptr,
      0U};
  output.input_layout =
      InputLayout::kFullAttentionInterleavedQScratchV1;
  ok &= expect(kernels::sm87_macrofeed_v4_fp8_layout_valid(output),
               "O logical-K6144 interleaved-Q scratch binding must pass");
  auto invalid_output = output;
  invalid_output.input_row_stride =
      kernels::kSm87MacroFeedV4Fp8AttentionOutputInputFeatures;
  ok &= expect(!kernels::sm87_macrofeed_v4_fp8_layout_valid(invalid_output),
               "compact O input must not impersonate V4 scratch layout");
  auto gdn_output = output;
  gdn_output.input_layout = InputLayout::kGdnContiguousVScratchV1;
  ok &= expect(kernels::sm87_macrofeed_v4_fp8_layout_valid(gdn_output),
               "GDN O fixed contiguous V scratch binding must pass");
  invalid_output = gdn_output;
  invalid_output.input_row_stride =
      kernels::kSm87MacroFeedV4Fp8AttentionOutputInputFeatures;
  ok &= expect(!kernels::sm87_macrofeed_v4_fp8_layout_valid(invalid_output),
               "compact GDN V pointer must not impersonate scratch base");
  invalid_output = gdn_output;
  invalid_output.input_layout = InputLayout::kHiddenContiguousH5120V1;
  ok &= expect(!kernels::sm87_macrofeed_v4_fp8_layout_valid(invalid_output),
               "unsupported O input layout must fail closed");
  ok &= expect(
      kernels::sm87_macrofeed_v4_fp8_interleaved_q_physical_offset(255U) ==
              255U &&
          kernels::sm87_macrofeed_v4_fp8_interleaved_q_physical_offset(
              256U) == 512U &&
          kernels::sm87_macrofeed_v4_fp8_interleaved_q_gate_physical_offset(
              6'144U) == 256U,
      "Q/G head-interleaved physical mapping changed");
  return ok;
}

[[nodiscard]] bool live_t1_device_binding_negative_test(
    const kernels::Sm87MacroFeedV4Fp8T1AdmissionSnapshot& snapshot,
    const InputLayout input_layout,
    const cudaStream_t stream) {
  constexpr Role kRole = Role::kFp8AttentionOutput;
  const auto plan = kernels::sm87_macrofeed_v4_fp8_plan(
      kRole, kernels::kSm87MacroFeedV4Fp8Tokens, input_layout);
  int current_device = -1;
  bool ok = expect(plan.valid() && stream != nullptr &&
                       cudaGetDevice(&current_device) == cudaSuccess &&
                       current_device >= 0,
                   "live T1 rejection fixture prerequisites failed");
  if (!ok) {
    return false;
  }

  LiveFp8AssetFixture fixture;
  RawDeviceAllocation input;
  RawDeviceAllocation output;
  RawDeviceAllocation undersized_input;
  RawDeviceAllocation undersized_output;
  // The production binding is the phase-scratch base, not a compact pointer
  // to either logical input surface.
  const std::size_t input_elements =
      plan.token_count * plan.input_row_stride;
  const std::size_t output_elements =
      (plan.token_count - 1U) * plan.primary_output_row_stride +
      plan.primary_output_features;
  constexpr std::size_t kPointerAlignmentElements =
      16U / sizeof(std::uint16_t);
  const std::size_t output_padding_elements =
      input_layout == InputLayout::kGdnContiguousVScratchV1
          ? kPointerAlignmentElements
          : 0U;
  if (!fixture.initialize(kRole, current_device) ||
      !input.allocate(input_elements * sizeof(std::uint16_t)) ||
      !output.allocate((output_elements + output_padding_elements) *
                       sizeof(std::uint16_t)) ||
      !undersized_input.allocate(256U) ||
      !undersized_output.allocate(256U)) {
    return expect(false,
                  "live T1 rejection fixture device allocation failed");
  }

  kernels::Sm87MacroFeedV4Fp8Arguments base{
      kRole,
      reinterpret_cast<const std::uint16_t*>(input.data()),
      plan.input_row_stride,
      fixture.asset,
      plan.token_count,
      reinterpret_cast<std::uint16_t*>(output.data()),
      plan.primary_output_row_stride,
      nullptr,
      0U,
      nullptr,
      0U,
      stream,
      input_layout};
  ok &= expect(kernels::sm87_macrofeed_v4_fp8_arguments_valid(base),
               "live T1 rejection base arguments must be structurally valid");
  if (!ok) {
    return false;
  }

  const auto expect_rejected = [&ok](
                                   const char* const label,
                                   const kernels::Sm87MacroFeedV4Fp8Arguments&
                                       arguments,
                                   const kernels::
                                       Sm87MacroFeedV4Fp8T1AdmissionSnapshot&
                                           candidate_snapshot) {
    kernels::Sm87MacroFeedV4Fp8T1AdmissionLaunchReceipt receipt{};
    const int status =
        kernels::launch_sm87_macrofeed_v4_fp8_t1_admission_cuda(
            arguments, candidate_snapshot, &receipt);
    const bool rejected =
        status == static_cast<int>(cudaErrorInvalidValue) &&
        !receipt.launch_enqueued && receipt.physical_kernel_launches == 0U &&
        !receipt.valid_t1_admission_enqueue_receipt();
    ok &= expect(rejected, std::string(label) +
                               " must fail before kernel enqueue");
    (void)cudaGetLastError();
  };

  const InputLayout other_output_layout =
      input_layout == InputLayout::kGdnContiguousVScratchV1
          ? InputLayout::kFullAttentionInterleavedQScratchV1
          : InputLayout::kGdnContiguousVScratchV1;
  auto wrong_layout_snapshot = snapshot;
  wrong_layout_snapshot.resources.input_layout = other_output_layout;
  wrong_layout_snapshot.resources.identity =
      kernels::sm87_macrofeed_v4_fp8_identity(kRole, other_output_layout);
  wrong_layout_snapshot.snapshot_identity = kernels::
      sm87_macrofeed_v4_fp8_compute_t1_admission_snapshot_identity(
          wrong_layout_snapshot);
  ok &= expect(
      kernels::sm87_macrofeed_v4_fp8_t1_admission_snapshot_valid(
          wrong_layout_snapshot),
      "alternate O-layout snapshot must remain structurally coherent");
  expect_rejected("snapshot with the other O input layout", base,
                  wrong_layout_snapshot);
  auto wrong_layout_arguments = base;
  wrong_layout_arguments.input_layout = other_output_layout;
  ok &= expect(
      kernels::sm87_macrofeed_v4_fp8_arguments_valid(wrong_layout_arguments),
      "alternate O-layout arguments must remain structurally coherent");
  expect_rejected("arguments with the other O input layout",
                  wrong_layout_arguments, snapshot);

  alignas(256) std::array<std::uint16_t, 128U> host_input{};
  alignas(256) std::array<std::uint16_t, 128U> host_output{};
  alignas(256) std::array<std::uint8_t, 256U> host_payload{};
  constexpr std::uintptr_t kFakeInput = 0x0000'1000'0000'0000ULL;
  constexpr std::uintptr_t kFakeOutput = 0x0000'2000'0000'0000ULL;
  constexpr std::uintptr_t kFakePayload = 0x0000'3000'0000'0000ULL;

  auto candidate = base;
  candidate.input = host_input.data();
  ok &= expect(kernels::sm87_macrofeed_v4_fp8_arguments_valid(candidate),
               "host-input negative must reach live CUDA range validation");
  expect_rejected("host input with otherwise live device ranges", candidate,
                  snapshot);

  candidate = base;
  candidate.primary_output = host_output.data();
  ok &= expect(kernels::sm87_macrofeed_v4_fp8_arguments_valid(candidate),
               "host-output negative must reach live CUDA range validation");
  expect_rejected("host output with otherwise live device ranges", candidate,
                  snapshot);

  candidate = base;
  candidate.input =
      reinterpret_cast<const std::uint16_t*>(kFakeInput);
  ok &= expect(kernels::sm87_macrofeed_v4_fp8_arguments_valid(candidate),
               "fake-input negative must reach live CUDA range validation");
  expect_rejected("fake input with otherwise live device ranges", candidate,
                  snapshot);

  candidate = base;
  candidate.primary_output =
      reinterpret_cast<std::uint16_t*>(kFakeOutput);
  ok &= expect(kernels::sm87_macrofeed_v4_fp8_arguments_valid(candidate),
               "fake-output negative must reach live CUDA range validation");
  expect_rejected("fake output with otherwise live device ranges", candidate,
                  snapshot);

  candidate = base;
  candidate.input = reinterpret_cast<const std::uint16_t*>(
      undersized_input.data());
  ok &= expect(kernels::sm87_macrofeed_v4_fp8_arguments_valid(candidate),
               "undersized-device-input negative must reach live range validation");
  expect_rejected("device input allocation shorter than declared C8000 span",
                  candidate, snapshot);

  if (input_layout == InputLayout::kGdnContiguousVScratchV1) {
    // Shift the input view so an incorrectly offset-free live range still
    // ends inside the existing allocation, while the real +4096-element V
    // offset ends 16 bytes beyond it. The output's 16-byte padding keeps its
    // shifted exact range adjacent and structurally disjoint, without another
    // large allocation.
    const std::size_t internal_shift =
        plan.input_row_stride - plan.input_physical_span -
        plan.input_physical_offset + kPointerAlignmentElements;
    const std::size_t end_without_physical_offset =
        internal_shift + (plan.token_count - 1U) * plan.input_row_stride +
        plan.input_physical_span;
    const std::size_t end_with_physical_offset =
        end_without_physical_offset + plan.input_physical_offset;
    ok &= expect(
        internal_shift % 8U == 0U &&
            end_without_physical_offset <= input_elements &&
            end_with_physical_offset ==
                input_elements + kPointerAlignmentElements,
        "GDN O offset-sensitive input range fixture must straddle allocation end");
    candidate = base;
    candidate.input =
        reinterpret_cast<const std::uint16_t*>(input.data()) + internal_shift;
    candidate.primary_output =
        reinterpret_cast<std::uint16_t*>(output.data()) +
        kPointerAlignmentElements;
    ok &= expect(
        kernels::sm87_macrofeed_v4_fp8_arguments_valid(candidate),
        "GDN O offset-sensitive input must reach live range validation");
    expect_rejected("GDN O input whose range fits only when 4096 offset is omitted",
                    candidate, snapshot);
  }

  candidate = base;
  candidate.primary_output =
      reinterpret_cast<std::uint16_t*>(undersized_output.data());
  ok &= expect(kernels::sm87_macrofeed_v4_fp8_arguments_valid(candidate),
               "undersized-device-output negative must reach live range validation");
  expect_rejected(
      "device output allocation shorter than declared C8000 span", candidate,
      snapshot);

  const auto retarget_asset = [](kernels::Sm87TargetAotFp8CudaAssetView asset,
                                 const std::uintptr_t begin) {
    const auto bytes = asset.payload.bytes;
    asset.payload.begin = begin;
    asset.payload.end = begin + static_cast<std::uintptr_t>(bytes);
    auto& upload = asset.device_upload_receipt;
    upload.device_allocation_begin = begin;
    upload.device_allocation_bytes = bytes;
    upload.device_allocation_end =
        begin + static_cast<std::uintptr_t>(bytes);
    upload.device_payload_begin = begin;
    upload.device_payload_bytes = bytes;
    upload.device_payload_end = begin + static_cast<std::uintptr_t>(bytes);
    upload.receipt_identity =
        kernels::sm87_target_aot_fp8_cuda_compute_upload_receipt_identity(
            upload);
    return asset;
  };

  candidate = base;
  candidate.asset = retarget_asset(
      fixture.asset, reinterpret_cast<std::uintptr_t>(host_payload.data()));
  ok &= expect(kernels::sm87_target_aot_fp8_cuda_asset_valid(candidate.asset) &&
                   kernels::sm87_macrofeed_v4_fp8_arguments_valid(candidate),
               "host-payload negative must remain structurally coherent");
  expect_rejected("host payload with otherwise live device ranges", candidate,
                  snapshot);

  candidate = base;
  candidate.asset = retarget_asset(fixture.asset, kFakePayload);
  ok &= expect(kernels::sm87_target_aot_fp8_cuda_asset_valid(candidate.asset) &&
                   kernels::sm87_macrofeed_v4_fp8_arguments_valid(candidate),
               "fake-payload negative must remain structurally coherent");
  expect_rejected("fake payload with otherwise live device ranges", candidate,
                  snapshot);

  auto mismatched_snapshot = snapshot;
  mismatched_snapshot.resources.device_ordinal = current_device + 1;
  mismatched_snapshot.snapshot_identity = kernels::
      sm87_macrofeed_v4_fp8_compute_t1_admission_snapshot_identity(
          mismatched_snapshot);
  ok &= expect(kernels::sm87_macrofeed_v4_fp8_t1_admission_snapshot_valid(
                   mismatched_snapshot),
               "ordinal-mismatched snapshot must remain structurally valid");
  expect_rejected("snapshot/current-device ordinal mismatch", base,
                  mismatched_snapshot);

  auto forged_resources_snapshot = snapshot;
  forged_resources_snapshot.resources.registers_per_thread +=
      forged_resources_snapshot.resources.registers_per_thread <
              static_cast<int>(
                  kernels::kSm87MacroFeedV4Fp8MaximumRegisters)
          ? 1
          : -1;
  forged_resources_snapshot.snapshot_identity = kernels::
      sm87_macrofeed_v4_fp8_compute_t1_admission_snapshot_identity(
          forged_resources_snapshot);
  ok &= expect(kernels::sm87_macrofeed_v4_fp8_t1_admission_snapshot_valid(
                   forged_resources_snapshot),
               "forged-register snapshot must remain structurally valid");
  expect_rejected("forged register resource observation", base,
                  forged_resources_snapshot);

  forged_resources_snapshot = snapshot;
  ++forged_resources_snapshot.resources.active_blocks_per_sm;
  forged_resources_snapshot.snapshot_identity = kernels::
      sm87_macrofeed_v4_fp8_compute_t1_admission_snapshot_identity(
          forged_resources_snapshot);
  ok &= expect(kernels::sm87_macrofeed_v4_fp8_t1_admission_snapshot_valid(
                   forged_resources_snapshot),
               "forged-occupancy snapshot must remain structurally valid");
  expect_rejected("forged active-CTA resource observation", base,
                  forged_resources_snapshot);

  forged_resources_snapshot = snapshot;
  forged_resources_snapshot.resources.maximum_threads_per_block += 32;
  forged_resources_snapshot.snapshot_identity = kernels::
      sm87_macrofeed_v4_fp8_compute_t1_admission_snapshot_identity(
          forged_resources_snapshot);
  ok &= expect(kernels::sm87_macrofeed_v4_fp8_t1_admission_snapshot_valid(
                   forged_resources_snapshot),
               "forged-unhashed resource snapshot must remain structurally valid");
  expect_rejected("forged maximum-threads resource observation", base,
                  forged_resources_snapshot);

  candidate = base;
  candidate.asset.device_upload_receipt.device_ordinal = current_device + 1;
  candidate.asset.device_upload_receipt.receipt_identity =
      kernels::sm87_target_aot_fp8_cuda_compute_upload_receipt_identity(
          candidate.asset.device_upload_receipt);
  ok &= expect(kernels::sm87_target_aot_fp8_cuda_asset_valid(candidate.asset) &&
                   kernels::sm87_macrofeed_v4_fp8_arguments_valid(candidate),
               "ordinal-mismatched asset must remain structurally valid");
  expect_rejected("asset-upload/current-device ordinal mismatch", candidate,
                  snapshot);
  return ok;
}

}  // namespace

int main() {
  bool ok = host_contract_test();
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::cout << "sm87_macrofeed_v4_fp8_cuda_test: SKIP (CUDA unavailable)\n";
    return ok ? 0 : 1;
  }
  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess ||
      properties.major != 8 || properties.minor != 7 ||
      properties.multiProcessorCount !=
          static_cast<int>(kernels::kSm87MacroFeedV4Fp8SmCount)) {
    std::cout << "sm87_macrofeed_v4_fp8_cuda_test: SKIP (requires SM87/16SM)\n";
    return ok ? 0 : 1;
  }

  cudaStream_t stream = nullptr;
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) !=
      cudaSuccess) {
    return 1;
  }
  constexpr std::array<Role, 3U> kRoles{{
      Role::kFp8GdnQkvZ,
      Role::kFp8FullQkv,
      Role::kFp8AttentionOutput,
  }};
  for (const Role role : kRoles) {
    kernels::Sm87MacroFeedV4Fp8CudaResources resources{};
    kernels::Sm87MacroFeedV4Fp8T1AdmissionSnapshot snapshot{};
    const int query_status =
        kernels::query_sm87_macrofeed_v4_fp8_cuda_resources(role,
                                                             &resources);
    const int snapshot_status =
        kernels::capture_sm87_macrofeed_v4_fp8_t1_admission_snapshot_cuda(
            role, &snapshot);
    ok &= expect(query_status == static_cast<int>(cudaSuccess) &&
                     kernels::sm87_macrofeed_v4_fp8_resource_gate(resources),
                 std::string(role_name(role)) +
                     " two-CTA/SM resource admission failed");
    ok &= expect(
        snapshot_status == static_cast<int>(cudaSuccess) &&
            kernels::sm87_macrofeed_v4_fp8_t1_admission_snapshot_valid(
                snapshot) &&
            snapshot.caller_constructible &&
            !snapshot.startup_package_bound &&
            !snapshot.execution_capability && snapshot.admission_only &&
            snapshot.default_off && !snapshot.production_dispatch_eligible,
                 std::string(role_name(role)) +
                     " T1 admission snapshot boundary failed");
    auto changed = snapshot;
    changed.selector_present = true;
    changed.snapshot_identity =
        kernels::sm87_macrofeed_v4_fp8_compute_t1_admission_snapshot_identity(
            changed);
    ok &= expect(
        !kernels::sm87_macrofeed_v4_fp8_t1_admission_snapshot_valid(changed),
                 std::string(role_name(role)) +
                     " selector-bearing T1 snapshot must fail closed");
    changed = snapshot;
    changed.caller_constructible = false;
    changed.snapshot_identity =
        kernels::sm87_macrofeed_v4_fp8_compute_t1_admission_snapshot_identity(
            changed);
    ok &= expect(
        !kernels::sm87_macrofeed_v4_fp8_t1_admission_snapshot_valid(changed),
        std::string(role_name(role)) +
            " T1 snapshot must disclose caller construction");
    changed = snapshot;
    changed.startup_package_bound = true;
    changed.snapshot_identity =
        kernels::sm87_macrofeed_v4_fp8_compute_t1_admission_snapshot_identity(
            changed);
    ok &= expect(
        !kernels::sm87_macrofeed_v4_fp8_t1_admission_snapshot_valid(changed),
        std::string(role_name(role)) +
            " package-bound claim must fail before private binding");
    changed = snapshot;
    changed.execution_capability = true;
    changed.snapshot_identity =
        kernels::sm87_macrofeed_v4_fp8_compute_t1_admission_snapshot_identity(
            changed);
    ok &= expect(
        !kernels::sm87_macrofeed_v4_fp8_t1_admission_snapshot_valid(changed),
        std::string(role_name(role)) +
            " execution-capability claim must fail closed");
    std::cout << role_name(role) << " resources: regs="
              << resources.registers_per_thread << " shared="
              << resources.dynamic_shared_bytes << " local="
              << resources.local_bytes << " active_cta_per_sm="
              << resources.active_blocks_per_sm << '\n';

    if (role == Role::kFp8AttentionOutput) {
      ok &= live_t1_device_binding_negative_test(
          snapshot, InputLayout::kFullAttentionInterleavedQScratchV1,
          stream);
    }

    const auto input_layout =
        kernels::sm87_macrofeed_v4_fp8_default_input_layout(role);
    const auto plan = kernels::sm87_macrofeed_v4_fp8_plan(
        role, kernels::kSm87MacroFeedV4Fp8Tokens, input_layout);
    for (std::size_t partition = 0U; partition < plan.partition_count;
         ++partition) {
      ok &= run_oracle_case(role, input_layout, partition, 0U, 0U, kRows,
                            stream);
      ok &= run_oracle_case(role, input_layout, partition, 0U, 1U, 37U,
                            stream);
    }
    if (role == Role::kFp8FullQkv) {
      // Partition 0 is logically Q[0,6144),Gate[6144,12288).  These probes
      // hit Q head 1 and both ends of the physically interleaved Gate plane.
      ok &= run_oracle_case(role, input_layout, 0U, 1U, 0U, kRows,
                            stream);
      ok &= run_oracle_case(role, input_layout, 0U, 24U, 0U, kRows,
                            stream);
      ok &= run_oracle_case(role, input_layout, 0U, 47U, 1U, 37U,
                            stream);
    }
  }

  constexpr InputLayout kGdnOLayout =
      InputLayout::kGdnContiguousVScratchV1;
  kernels::Sm87MacroFeedV4Fp8CudaResources gdn_o_resources{};
  kernels::Sm87MacroFeedV4Fp8T1AdmissionSnapshot gdn_o_snapshot{};
  const int gdn_o_query_status =
      kernels::query_sm87_macrofeed_v4_fp8_cuda_resources(
          Role::kFp8AttentionOutput, kGdnOLayout, &gdn_o_resources);
  const int gdn_o_snapshot_status =
      kernels::capture_sm87_macrofeed_v4_fp8_t1_admission_snapshot_cuda(
          Role::kFp8AttentionOutput, kGdnOLayout, &gdn_o_snapshot);
  ok &= expect(
      gdn_o_query_status == static_cast<int>(cudaSuccess) &&
          kernels::sm87_macrofeed_v4_fp8_resource_gate(gdn_o_resources) &&
          gdn_o_resources.input_layout == kGdnOLayout &&
          gdn_o_resources.identity ==
              kernels::Sm87MacroFeedV4Fp8Identity::
                  kGdnAttentionOutputM64N128K64OrdinaryGridV1,
      "GDN-contiguous O resource identity/admission failed");
  ok &= expect(
      gdn_o_snapshot_status == static_cast<int>(cudaSuccess) &&
          kernels::sm87_macrofeed_v4_fp8_t1_admission_snapshot_valid(
              gdn_o_snapshot),
      "GDN-contiguous O T1 snapshot failed");
  std::cout << "GDN-contiguous Attention-O resources: regs="
            << gdn_o_resources.registers_per_thread << " shared="
            << gdn_o_resources.dynamic_shared_bytes << " local="
            << gdn_o_resources.local_bytes << " active_cta_per_sm="
            << gdn_o_resources.active_blocks_per_sm << '\n';
  ok &= live_t1_device_binding_negative_test(gdn_o_snapshot, kGdnOLayout,
                                              stream);
  constexpr std::array<std::size_t, 2U> kGdnORows{{kRows, 37U}};
  constexpr std::array<std::size_t, 2U> kGdnON128Halves{{0U, 1U}};
  constexpr std::array<std::size_t, 2U> kGdnOLogicalFirstKs{{
      kernels::kSm87MacroFeedV4Fp8TestGdnAttentionOutputLogicalFirstK,
      kernels::kSm87MacroFeedV4Fp8TestGdnAttentionOutputLogicalTailFirstK,
  }};
  for (const std::size_t logical_first_k : kGdnOLogicalFirstKs) {
    for (const std::size_t rows : kGdnORows) {
      for (const std::size_t half : kGdnON128Halves) {
        ok &= run_oracle_case(Role::kFp8AttentionOutput, kGdnOLayout, 0U,
                              0U, half, rows, stream, logical_first_k);
      }
    }
  }
  ok &= expect(cudaStreamDestroy(stream) == cudaSuccess,
               "stream destruction failed");
  if (ok) {
    std::cout << "sm87_macrofeed_v4_fp8_cuda_test: PASS\n";
  }
  return ok ? 0 : 1;
}
