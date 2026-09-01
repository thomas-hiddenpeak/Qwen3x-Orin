#include "tests/support/sm87_aot_system_v1_real_p40_active_cell_witness_internal.h"

#include "q3x/core/sha256.h"
#include "tests/support/sm87_aot_system_v1_active_cell_cuda.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <type_traits>
#include <variant>

namespace q3x::test::sm87_aot_real_p40_active_cell {
namespace {

namespace active_cell = q3x::test::sm87_aot_active_cell;
using HookRole = witness_hook::AotArithmeticWitnessProjectionRole;
using HookScope = witness_hook::AotArithmeticWitnessCaptureScope;
using LinearWeight = q3x::runtime::LinearWeight;
using LinearWeightKind = q3x::runtime::LinearWeightKind;

inline constexpr std::size_t kFullSegmentRows = 512U;
inline constexpr std::size_t kTailSegmentRows = 63U;
inline constexpr std::size_t kFullSegmentCount = 78U;
inline constexpr std::size_t kMaximumM16Tiles = 32U;
inline constexpr std::size_t kMaximumK16Groups = 1'088U;
inline constexpr std::size_t kMaximumBMaskElements = 696'320U;
inline constexpr std::size_t kExpectedWeightPartitionCount = 400U;
inline constexpr std::size_t kMaximumPartitionsPerCall = 3U;
inline constexpr std::uint64_t kStrictFiveSecondCellThreshold =
    416'160'000'000ULL;

static_assert(kRealP40CallsPerSegment == 256U);
static_assert(kFullSegmentCount * kFullSegmentRows + kTailSegmentRows ==
              witness_hook::kAotArithmeticWitnessP40PrefixRows);
static_assert(kRealP40SegmentCount ==
              witness_hook::kAotArithmeticWitnessP40PrefixTileCount);

[[nodiscard]] bool checked_add(const std::size_t left,
                               const std::size_t right,
                               std::size_t* const result) noexcept {
  if (result == nullptr ||
      left > std::numeric_limits<std::size_t>::max() - right) {
    return false;
  }
  *result = left + right;
  return true;
}

[[nodiscard]] bool checked_add_u64(std::uint64_t* const value,
                                   const std::uint64_t increment) noexcept {
  if (value == nullptr ||
      *value > std::numeric_limits<std::uint64_t>::max() - increment) {
    return false;
  }
  *value += increment;
  return true;
}

[[nodiscard]] bool checked_multiply_u64(const std::uint64_t left,
                                        const std::uint64_t right,
                                        std::uint64_t* const result) noexcept {
  if (result == nullptr ||
      (left != 0U &&
       right > std::numeric_limits<std::uint64_t>::max() / left)) {
    return false;
  }
  *result = left * right;
  return true;
}

[[nodiscard]] bool digest_is_nonzero(
    const lower_bound::Sha256Digest& digest) noexcept {
  for (const std::uint8_t byte : digest) {
    if (byte != 0U) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] lower_bound::Sha256Digest lower_digest(
    const q3x::core::Sha256Digest& digest) noexcept {
  return digest.bytes;
}

[[nodiscard]] bool hash_bytes(q3x::core::Sha256* const hash,
                              const void* const data,
                              const std::size_t bytes) noexcept {
  return hash != nullptr && hash->update(data, bytes);
}

[[nodiscard]] bool hash_u64(q3x::core::Sha256* const hash,
                            const std::uint64_t value) noexcept {
  std::array<std::uint8_t, 8U> encoded{};
  for (std::size_t index = 0U; index < encoded.size(); ++index) {
    encoded[index] =
        static_cast<std::uint8_t>((value >> (index * 8U)) & 0xffU);
  }
  return hash_bytes(hash, encoded.data(), encoded.size());
}

[[nodiscard]] bool hash_digest(q3x::core::Sha256* const hash,
                               const lower_bound::Sha256Digest& digest) noexcept {
  return hash_bytes(hash, digest.data(), digest.size());
}

[[nodiscard]] bool hash_string(q3x::core::Sha256* const hash,
                               const std::string_view value) noexcept {
  return hash_u64(hash, static_cast<std::uint64_t>(value.size())) &&
         hash_bytes(hash, value.data(), value.size());
}

[[nodiscard]] bool hash_masks(q3x::core::Sha256* const hash,
                              const std::uint16_t* const masks,
                              const std::size_t count) noexcept {
  if (hash == nullptr || (masks == nullptr && count != 0U)) {
    return false;
  }
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return hash_bytes(hash, masks, count * sizeof(std::uint16_t));
#else
  std::array<std::uint8_t, 512U> encoded{};
  std::size_t offset = 0U;
  while (offset < count) {
    const std::size_t chunk = std::min<std::size_t>(
        count - offset, encoded.size() / sizeof(std::uint16_t));
    for (std::size_t index = 0U; index < chunk; ++index) {
      const std::uint16_t value = masks[offset + index];
      encoded[index * 2U] = static_cast<std::uint8_t>(value & 0xffU);
      encoded[index * 2U + 1U] =
          static_cast<std::uint8_t>((value >> 8U) & 0xffU);
    }
    if (!hash_bytes(hash, encoded.data(), chunk * 2U)) {
      return false;
    }
    offset += chunk;
  }
  return true;
#endif
}

[[nodiscard]] std::uint32_t float_bits(const float value) noexcept {
  std::uint32_t result = 0U;
  static_assert(sizeof(result) == sizeof(value));
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

[[nodiscard]] lower_bound::ProjectionRole lower_role(
    const HookRole role) noexcept {
  switch (role) {
    case HookRole::kNvFp4GateUp:
      return lower_bound::ProjectionRole::kNvFp4GateUp;
    case HookRole::kNvFp4Down:
      return lower_bound::ProjectionRole::kNvFp4Down;
    case HookRole::kFp8GdnQkvZ:
      return lower_bound::ProjectionRole::kFp8GdnQkvZ;
    case HookRole::kFp8FullQkv:
      return lower_bound::ProjectionRole::kFp8FullQkv;
    case HookRole::kFp8AttentionOutput:
      return lower_bound::ProjectionRole::kFp8AttentionOutput;
    case HookRole::kInvalid:
      return lower_bound::ProjectionRole::kInvalid;
  }
  return lower_bound::ProjectionRole::kInvalid;
}

[[nodiscard]] std::size_t role_index(
    const lower_bound::ProjectionRole role) noexcept {
  const auto raw = static_cast<std::size_t>(role);
  return raw == 0U ? lower_bound::kProjectionRoleCount : raw - 1U;
}

[[nodiscard]] HookRole expected_role(const std::size_t layer,
                                     const std::size_t phase) noexcept {
  switch (phase) {
    case 0U:
      return layer % 4U == 3U ? HookRole::kFp8FullQkv
                              : HookRole::kFp8GdnQkvZ;
    case 1U:
      return HookRole::kFp8AttentionOutput;
    case 2U:
      return HookRole::kNvFp4GateUp;
    case 3U:
      return HookRole::kNvFp4Down;
    default:
      return HookRole::kInvalid;
  }
}

[[nodiscard]] std::size_t expected_k(const HookRole role) noexcept {
  switch (role) {
    case HookRole::kNvFp4Down:
      return 17'408U;
    case HookRole::kFp8AttentionOutput:
      return 6'144U;
    case HookRole::kNvFp4GateUp:
    case HookRole::kFp8GdnQkvZ:
    case HookRole::kFp8FullQkv:
      return 5'120U;
    case HookRole::kInvalid:
      return 0U;
  }
  return 0U;
}

[[nodiscard]] std::size_t expected_partition_inventory(
    const lower_bound::ProjectionRole role) noexcept {
  switch (role) {
    case lower_bound::ProjectionRole::kNvFp4GateUp:
      return 128U;
    case lower_bound::ProjectionRole::kNvFp4Down:
      return 64U;
    case lower_bound::ProjectionRole::kFp8GdnQkvZ:
      return 96U;
    case lower_bound::ProjectionRole::kFp8FullQkv:
      return 48U;
    case lower_bound::ProjectionRole::kFp8AttentionOutput:
      return 64U;
    case lower_bound::ProjectionRole::kInvalid:
    case lower_bound::ProjectionRole::kCount:
      return 0U;
  }
  return 0U;
}

[[nodiscard]] std::size_t expected_calls_per_segment(
    const lower_bound::ProjectionRole role) noexcept {
  switch (role) {
    case lower_bound::ProjectionRole::kNvFp4GateUp:
    case lower_bound::ProjectionRole::kNvFp4Down:
    case lower_bound::ProjectionRole::kFp8AttentionOutput:
      return 64U;
    case lower_bound::ProjectionRole::kFp8GdnQkvZ:
      return 48U;
    case lower_bound::ProjectionRole::kFp8FullQkv:
      return 16U;
    case lower_bound::ProjectionRole::kInvalid:
    case lower_bound::ProjectionRole::kCount:
      return 0U;
  }
  return 0U;
}

[[nodiscard]] std::size_t segment_rows(const std::size_t segment) noexcept {
  return segment < kFullSegmentCount
             ? kFullSegmentRows
             : (segment == kFullSegmentCount ? kTailSegmentRows : 0U);
}

[[nodiscard]] std::uint32_t segment_first_position(
    const std::size_t segment) noexcept {
  return static_cast<std::uint32_t>(
      std::min(segment, kFullSegmentCount) * kFullSegmentRows);
}

struct ActivationCallSlot final {
  std::size_t host_mask_offset = 0U;
  std::size_t maximum_mask_count = 0U;
  std::size_t current_mask_count = 0U;
  std::size_t current_m16_tiles = 0U;
  std::size_t current_k16_groups = 0U;
  HookRole role = HookRole::kInvalid;
  std::size_t layer = 0U;
  std::size_t partition_count = 0U;
  std::array<std::size_t, kMaximumPartitionsPerCall> weight_records{};
};

struct WeightPartitionRecord final {
  HookRole hook_role = HookRole::kInvalid;
  lower_bound::ProjectionRole role = lower_bound::ProjectionRole::kInvalid;
  std::size_t layer = 0U;
  std::size_t partition = 0U;
  std::size_t output_features = 0U;
  std::size_t input_features = 0U;
  std::size_t n8_tiles = 0U;
  std::size_t k16_groups = 0U;
  LinearWeightKind kind = LinearWeightKind::kBf16;
  const LinearWeight* wrapper = nullptr;
  const std::uint8_t* weight_data = nullptr;
  const std::uint8_t* scale_data = nullptr;
  const float* global_scale_device = nullptr;
  std::uint32_t global_scale_bits = 0U;
  lower_bound::Sha256Digest mask_sha256{};
  std::array<std::uint32_t, kMaximumK16Groups> nonzero_n8_tiles_by_k16{};
  const std::uint16_t* pending_host_masks = nullptr;
  std::size_t pending_mask_count = 0U;
  bool host_job_ready = false;
  bool host_job_ok = false;
};

[[nodiscard]] bool same_weight_identity(
    const WeightPartitionRecord& record,
    const LinearWeight& weight) noexcept {
  if (record.wrapper != &weight ||
      q3x::runtime::linear_weight_kind(weight) != record.kind ||
      q3x::runtime::linear_output_size(weight) != record.output_features ||
      q3x::runtime::linear_input_size(weight) != record.input_features) {
    return false;
  }
  if (const auto* fp8 = std::get_if<q3x::runtime::Fp8LinearWeight>(&weight)) {
    return record.kind == LinearWeightKind::kFp8 &&
           record.weight_data == fp8->weight &&
           record.global_scale_device == fp8->weight_scale_device &&
           record.global_scale_bits == float_bits(fp8->weight_scale);
  }
  if (const auto* nvfp4 =
          std::get_if<q3x::runtime::NvFp4LinearWeight>(&weight)) {
    return record.kind == LinearWeightKind::kNvFp4 &&
           record.weight_data == nvfp4->packed_weight &&
           record.scale_data == nvfp4->block_scale &&
           record.global_scale_device == nvfp4->weight_scale_2_device &&
           record.global_scale_bits == float_bits(nvfp4->weight_scale_2);
  }
  return false;
}

}  // namespace

struct RealP40ActiveCellWitnessCollector::Impl final {
  explicit Impl(const CollectorIdentity input_identity) noexcept
      : identity(input_identity) {
    if (!identity.real_p40_route_authenticated ||
        !identity.shard_manifest_authenticated ||
        !digest_is_nonzero(identity.real_p40_route_sha256) ||
        !digest_is_nonzero(identity.authenticated_shard_manifest_sha256)) {
      fail(CollectorFailureCode::kInvalidIdentity, cudaErrorInvalidValue,
           "route and shard-manifest identities must be authenticated and nonzero");
      return;
    }
    if (!prepare_activation_slots()) {
      fail(CollectorFailureCode::kArithmeticOverflow, cudaErrorInvalidValue,
           "activation slot layout exceeds the fixed pinned arena");
      return;
    }

    cudaError_t status = cudaMalloc(
        reinterpret_cast<void**>(&device_masks),
        kMaximumBMaskElements * sizeof(std::uint16_t));
    if (status != cudaSuccess) {
      fail(CollectorFailureCode::kCudaAllocation, status,
           "cudaMalloc for reusable mask buffer failed");
      return;
    }
    status = cudaMalloc(reinterpret_cast<void**>(&device_exceptional_flag),
                        sizeof(std::uint32_t));
    if (status != cudaSuccess) {
      fail(CollectorFailureCode::kCudaAllocation, status,
           "cudaMalloc for exceptional-operand flag failed");
      return;
    }
    status = cudaHostAlloc(reinterpret_cast<void**>(&pinned_b_masks),
                           kMaximumBMaskElements * sizeof(std::uint16_t),
                           cudaHostAllocDefault);
    if (status != cudaSuccess) {
      fail(CollectorFailureCode::kCudaAllocation, status,
           "cudaHostAlloc for B-mask scratch failed");
      return;
    }
    status = cudaHostAlloc(reinterpret_cast<void**>(&pinned_a_masks),
                           kPinnedActivationSlotBytes,
                           cudaHostAllocDefault);
    if (status != cudaSuccess) {
      fail(CollectorFailureCode::kCudaAllocation, status,
           "cudaHostAlloc for per-segment A-mask slots failed");
      return;
    }
    status = cudaHostAlloc(
        reinterpret_cast<void**>(&pinned_exceptional_flag),
        sizeof(std::uint32_t), cudaHostAllocDefault);
    if (status != cudaSuccess) {
      fail(CollectorFailureCode::kCudaAllocation, status,
           "cudaHostAlloc for exceptional-operand flag failed");
      return;
    }
    *pinned_exceptional_flag = 0U;
  }

  ~Impl() {
    if (pending_stream_work) {
      (void)(current_stream_valid ? cudaStreamSynchronize(current_stream)
                                  : cudaDeviceSynchronize());
    }
    if (device_masks != nullptr) {
      (void)cudaFree(device_masks);
    }
    if (device_exceptional_flag != nullptr) {
      (void)cudaFree(device_exceptional_flag);
    }
    if (pinned_b_masks != nullptr) {
      (void)cudaFreeHost(pinned_b_masks);
    }
    if (pinned_a_masks != nullptr) {
      (void)cudaFreeHost(pinned_a_masks);
    }
    if (pinned_exceptional_flag != nullptr) {
      (void)cudaFreeHost(pinned_exceptional_flag);
    }
  }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;

  [[nodiscard]] bool prepare_activation_slots() noexcept {
    std::size_t next_offset = 0U;
    for (std::size_t call = 0U; call < calls.size(); ++call) {
      const std::size_t layer = call / kRealP40CallsPerLayer;
      const std::size_t phase = call % kRealP40CallsPerLayer;
      ActivationCallSlot& slot = calls[call];
      slot.layer = layer;
      slot.role = expected_role(layer, phase);
      slot.host_mask_offset = next_offset;
      const std::size_t k16_groups = expected_k(slot.role) / 16U;
      slot.maximum_mask_count = kMaximumM16Tiles * k16_groups;
      if (!checked_add(next_offset, slot.maximum_mask_count, &next_offset)) {
        return false;
      }
    }
    activation_slot_elements = next_offset;
    return activation_slot_elements <=
           kPinnedActivationSlotBytes / sizeof(std::uint16_t);
  }

  void fail(const CollectorFailureCode code, const cudaError_t status,
            const char* const text) noexcept {
    if (failure_code != CollectorFailureCode::kNone) {
      return;
    }
    failure_code = code;
    cuda_status = static_cast<int>(status);
    if (text != nullptr) {
      (void)std::snprintf(failure_text.data(), failure_text.size(), "%s", text);
    }
  }

  [[nodiscard]] int failure_status() const noexcept {
    return cuda_status == static_cast<int>(cudaSuccess)
               ? static_cast<int>(cudaErrorInvalidValue)
               : cuda_status;
  }

  [[nodiscard]] std::size_t find_weight_record(
      const HookRole role, const std::size_t layer,
      const std::size_t partition) const noexcept {
    for (std::size_t index = 0U; index < weight_record_count; ++index) {
      const WeightPartitionRecord& record = weight_records[index];
      if (record.hook_role == role && record.layer == layer &&
          record.partition == partition) {
        return index;
      }
    }
    return kExpectedWeightPartitionCount;
  }

  [[nodiscard]] static bool hash_weight_masks(
      WeightPartitionRecord* const record,
      const std::uint16_t* const masks,
      const std::size_t mask_count) noexcept {
    q3x::core::Sha256 hash;
    constexpr std::string_view domain =
        "q3x.sm87.aot.real-p40.weight-k16-support.v1";
    if (record == nullptr || !hash_string(&hash, domain) ||
        !hash_u64(&hash, static_cast<std::uint64_t>(record->role)) ||
        !hash_u64(&hash, record->layer) ||
        !hash_u64(&hash, record->partition) ||
        !hash_u64(&hash, static_cast<std::uint64_t>(record->kind)) ||
        !hash_u64(&hash, record->output_features) ||
        !hash_u64(&hash, record->input_features) ||
        !hash_u64(&hash, record->n8_tiles) ||
        !hash_u64(&hash, record->k16_groups) ||
        !hash_u64(&hash, record->global_scale_bits) ||
        !hash_masks(&hash, masks, mask_count)) {
      return false;
    }
    record->mask_sha256 = lower_digest(hash.finalize());
    return digest_is_nonzero(record->mask_sha256);
  }

  static void CUDART_CB complete_b_mask_host(void* const context) noexcept {
    auto* const record = static_cast<WeightPartitionRecord*>(context);
    if (record == nullptr || record->pending_host_masks == nullptr ||
        record->pending_mask_count !=
            record->n8_tiles * record->k16_groups) {
      if (record != nullptr) {
        record->host_job_ready = true;
        record->host_job_ok = false;
      }
      return;
    }
    for (std::size_t k16 = 0U; k16 < record->k16_groups; ++k16) {
      std::uint32_t count = 0U;
      for (std::size_t n8 = 0U; n8 < record->n8_tiles; ++n8) {
        if (record->pending_host_masks[n8 * record->k16_groups + k16] !=
            0U) {
          ++count;
        }
      }
      record->nonzero_n8_tiles_by_k16[k16] = count;
    }
    record->host_job_ok =
        hash_weight_masks(record, record->pending_host_masks,
                          record->pending_mask_count);
    record->host_job_ready = true;
  }

  [[nodiscard]] bool create_weight_record(
      const witness_hook::AotArithmeticWitnessAOperandView& view,
      const std::size_t partition, const cudaStream_t stream,
      std::size_t* const result_index) noexcept {
    if (result_index == nullptr || partition >= view.partition_count ||
        weight_record_count >= weight_records.size()) {
      fail(CollectorFailureCode::kWeightInventoryMismatch,
           cudaErrorInvalidValue, "weight partition inventory overflow");
      return false;
    }

    const LinearWeight& weight = *view.partitions[partition];
    WeightPartitionRecord record;
    record.hook_role = view.role;
    record.role = lower_role(view.role);
    record.layer = view.layer;
    record.partition = partition;
    record.output_features = q3x::runtime::linear_output_size(weight);
    record.input_features = q3x::runtime::linear_input_size(weight);
    record.kind = q3x::runtime::linear_weight_kind(weight);
    record.wrapper = &weight;

    active_cell::MaskLayout layout;
    if (!active_cell::canonical_b_mask_layout(
            record.output_features, record.input_features, &layout) ||
        layout.mask_count > kMaximumBMaskElements ||
        layout.k16_groups > kMaximumK16Groups) {
      fail(CollectorFailureCode::kWeightInventoryMismatch,
           cudaErrorInvalidValue, "weight mask geometry is outside P40 bounds");
      return false;
    }
    record.n8_tiles = layout.outer_tiles;
    record.k16_groups = layout.k16_groups;

    cudaError_t status = cudaErrorInvalidValue;
    if (const auto* fp8 =
            std::get_if<q3x::runtime::Fp8LinearWeight>(&weight)) {
      record.weight_data = fp8->weight;
      record.global_scale_device = fp8->weight_scale_device;
      record.global_scale_bits = float_bits(fp8->weight_scale);
      status = active_cell::launch_fp8_b_k_support_masks(
          fp8->weight, fp8->output_size, fp8->input_size,
          fp8->weight_scale, device_masks, layout.mask_count,
          device_exceptional_flag, stream);
    } else if (const auto* nvfp4 =
                   std::get_if<q3x::runtime::NvFp4LinearWeight>(&weight)) {
      record.weight_data = nvfp4->packed_weight;
      record.scale_data = nvfp4->block_scale;
      record.global_scale_device = nvfp4->weight_scale_2_device;
      record.global_scale_bits = float_bits(nvfp4->weight_scale_2);
      status = active_cell::launch_nvfp4_b_k_support_masks(
          nvfp4->packed_weight, nvfp4->block_scale, nvfp4->output_size,
          nvfp4->input_size, nvfp4->weight_scale_2, device_masks,
          layout.mask_count, device_exceptional_flag, stream);
    }
    if (status != cudaSuccess) {
      fail(CollectorFailureCode::kCudaMaskLaunch, status,
           "checkpoint B-mask launch failed");
      return false;
    }
    pending_stream_work = true;
    status = cudaMemcpyAsync(pinned_b_masks, device_masks,
                             layout.mask_count * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost, stream);
    if (status != cudaSuccess) {
      fail(CollectorFailureCode::kCudaCopy, status,
           "checkpoint B-mask D2H copy failed");
      return false;
    }

    const std::size_t index = weight_record_count;
    record.pending_host_masks = pinned_b_masks;
    record.pending_mask_count = layout.mask_count;
    weight_records[index] = record;
    // The single pinned scratch is safe without a per-weight synchronize:
    // stream order is B kernel -> D2H -> host count/SHA, and the following
    // weight's D2H cannot overwrite the scratch until this host function has
    // returned.  The host function performs no CUDA call or allocation.
    status = cudaLaunchHostFunc(stream, &Impl::complete_b_mask_host,
                                &weight_records[index]);
    if (status != cudaSuccess) {
      fail(CollectorFailureCode::kCudaCopy, status,
           "checkpoint B-mask host completion enqueue failed");
      return false;
    }
    ++weight_record_count;
    const std::size_t index_by_role = role_index(record.role);
    if (index_by_role >= weight_record_counts_by_role.size() ||
        weight_record_counts_by_role[index_by_role] ==
            std::numeric_limits<std::uint32_t>::max()) {
      fail(CollectorFailureCode::kArithmeticOverflow, cudaErrorInvalidValue,
           "weight partition role count overflow");
      return false;
    }
    ++weight_record_counts_by_role[index_by_role];
    *result_index = index;
    return true;
  }

  [[nodiscard]] bool get_weight_record(
      const witness_hook::AotArithmeticWitnessAOperandView& view,
      const std::size_t partition, const cudaStream_t stream,
      std::size_t* const result_index) noexcept {
    const std::size_t found =
        find_weight_record(view.role, view.layer, partition);
    if (found == kExpectedWeightPartitionCount) {
      return create_weight_record(view, partition, stream, result_index);
    }
    if (result_index == nullptr ||
        !same_weight_identity(weight_records[found],
                              *view.partitions[partition])) {
      fail(CollectorFailureCode::kWeightInventoryMismatch,
           cudaErrorInvalidValue,
           "repeated role/layer/partition changed pointer, shape, or scale");
      return false;
    }
    *result_index = found;
    return true;
  }

  [[nodiscard]] bool complete_weight_inventory() const noexcept {
    if (weight_record_count != kExpectedWeightPartitionCount) {
      return false;
    }
    for (std::size_t index = 0U;
         index < lower_bound::kProjectionRoleCount; ++index) {
      const auto role = static_cast<lower_bound::ProjectionRole>(index + 1U);
      if (weight_record_counts_by_role[index] !=
          expected_partition_inventory(role)) {
        return false;
      }
    }
    for (std::size_t index = 0U; index < weight_record_count; ++index) {
      if (!weight_records[index].host_job_ready ||
          !weight_records[index].host_job_ok ||
          !digest_is_nonzero(weight_records[index].mask_sha256)) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool build_checkpoint_digest() noexcept {
    if (checkpoint_digest_ready) {
      return true;
    }
    if (!complete_weight_inventory()) {
      fail(CollectorFailureCode::kWeightInventoryMismatch,
           cudaErrorInvalidValue,
           "complete segment did not finish the frozen 400-partition inventory");
      return false;
    }
    q3x::core::Sha256 hash;
    constexpr std::string_view domain =
        "q3x.sm87.aot.real-p40.checkpoint-inventory.v1";
    if (!hash_string(&hash, domain) ||
        !hash_digest(&hash, identity.authenticated_shard_manifest_sha256) ||
        !hash_u64(&hash, weight_record_count)) {
      fail(CollectorFailureCode::kDigestFailure, cudaErrorInvalidValue,
           "checkpoint inventory digest initialization failed");
      return false;
    }
    for (std::size_t index = 0U; index < weight_record_count; ++index) {
      const WeightPartitionRecord& record = weight_records[index];
      if (!hash_u64(&hash, index) ||
          !hash_u64(&hash, static_cast<std::uint64_t>(record.role)) ||
          !hash_u64(&hash, record.layer) ||
          !hash_u64(&hash, record.partition) ||
          !hash_u64(&hash, static_cast<std::uint64_t>(record.kind)) ||
          !hash_u64(&hash, record.output_features) ||
          !hash_u64(&hash, record.input_features) ||
          !hash_u64(&hash, record.n8_tiles) ||
          !hash_u64(&hash, record.k16_groups) ||
          !hash_u64(&hash, record.global_scale_bits) ||
          !hash_digest(&hash, record.mask_sha256)) {
        fail(CollectorFailureCode::kDigestFailure, cudaErrorInvalidValue,
             "checkpoint inventory digest update failed");
        return false;
      }
    }
    checkpoint_manifest_sha256 = lower_digest(hash.finalize());
    checkpoint_digest_ready = digest_is_nonzero(checkpoint_manifest_sha256);
    if (!checkpoint_digest_ready) {
      fail(CollectorFailureCode::kDigestFailure, cudaErrorInvalidValue,
           "checkpoint inventory digest finalized to zero");
    }
    return checkpoint_digest_ready;
  }

  [[nodiscard]] bool initialize_segment_hashes(
      std::array<q3x::core::Sha256, lower_bound::kProjectionRoleCount>* const
          activation_hashes,
      std::array<q3x::core::Sha256, lower_bound::kProjectionRoleCount>* const
          joint_hashes) noexcept {
    if (activation_hashes == nullptr || joint_hashes == nullptr) {
      return false;
    }
    const auto& mapping = lower_bound::frozen_mapping_spec();
    for (std::size_t index = 0U;
         index < lower_bound::kProjectionRoleCount; ++index) {
      const auto role = static_cast<lower_bound::ProjectionRole>(index + 1U);
      auto& activation = (*activation_hashes)[index];
      auto& joint = (*joint_hashes)[index];
      constexpr std::string_view joint_domain =
          "q3x.sm87.aot.real-p40.joint-enumeration-chain.v1";
      if (!hash_string(&activation, kActivationCaptureChainDomain) ||
          !hash_digest(&activation, activation_inventory_sha256[index]) ||
          !hash_u64(&activation, completed_segments) ||
          !hash_u64(&activation, covered_prompt_rows) ||
          !hash_u64(&activation, segment_rows(completed_segments)) ||
          !hash_u64(&activation, static_cast<std::uint64_t>(role)) ||
          !hash_u64(&activation, activation_call_counts[index]) ||
          !hash_u64(&activation, activation_mask_element_counts[index]) ||
          !hash_u64(&activation, activation_payload_bytes[index]) ||
          !hash_string(&joint, joint_domain) ||
          !hash_digest(&joint, joint_enumeration_sha256[index]) ||
          !hash_u64(&joint, completed_segments) ||
          !hash_u64(&joint, covered_prompt_rows) ||
          !hash_u64(&joint, segment_rows(completed_segments)) ||
          !hash_u64(&joint, static_cast<std::uint64_t>(role)) ||
          !hash_string(&joint, mapping.identity) ||
          !hash_string(&joint, mapping.instruction) ||
          !hash_string(&joint, mapping.support_active_parent_predicate) ||
          !hash_string(&joint, mapping.exact_arithmetic_pass_floor) ||
          !hash_u64(&joint, mapping.mma_m) ||
          !hash_u64(&joint, mapping.mma_n) ||
          !hash_u64(&joint, mapping.bmma_k) ||
          !hash_u64(&joint, mapping.parent_k) ||
          !hash_u64(&joint, mapping.zero_fill_factor) ||
          !hash_u64(&joint,
                    mapping.physical_instructions_per_active_joint_k16_cell) ||
          !hash_u64(&joint,
                    mapping.maximum_warp_instructions_per_sm_cycle) ||
          !hash_u64(&joint, mapping.sm_count) ||
          !hash_u64(&joint, mapping.clock_hz) ||
          !hash_u64(&joint, mapping.projection_budget_seconds) ||
          !hash_u64(&joint, mapping.p40_prompt_rows) ||
          !hash_u64(&joint, mapping.production_prefix_rows) ||
          !hash_u64(&joint, mapping.terminal_scalar_rows) ||
          !hash_u64(&joint,
                    mapping.partial_m16_tail_rows_charged_free ? 1U : 0U) ||
          !hash_u64(&joint, mapping.one_parent_per_instruction ? 1U : 0U) ||
          !hash_u64(&joint, mapping.cross_parent_packing_forbidden ? 1U : 0U) ||
          !hash_u64(&joint,
                    mapping.support_active_parent_must_issue ? 1U : 0U) ||
          !hash_u64(&joint,
                    mapping.result_aware_parent_elision_forbidden ? 1U : 0U) ||
          !hash_u64(
              &joint,
              mapping.cross_parent_common_subexpression_elimination_forbidden
                  ? 1U
                  : 0U) ||
          !hash_u64(&joint,
                    mapping.additional_exactness_passes_charged_free ? 1U
                                                                     : 0U)) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool process_complete_segment() noexcept {
    std::array<std::uint64_t, lower_bound::kProjectionRoleCount>
        segment_observed{};
    std::array<std::uint64_t, lower_bound::kProjectionRoleCount>
        segment_active{};
    std::array<std::uint64_t, lower_bound::kProjectionRoleCount>
        segment_activation_calls{};
    std::array<std::uint64_t, lower_bound::kProjectionRoleCount>
        segment_activation_mask_elements{};
    std::array<std::uint64_t, lower_bound::kProjectionRoleCount>
        segment_activation_payload_bytes{};
    std::array<q3x::core::Sha256, lower_bound::kProjectionRoleCount>
        activation_hashes{};
    std::array<q3x::core::Sha256, lower_bound::kProjectionRoleCount>
        joint_hashes{};
    if (!initialize_segment_hashes(&activation_hashes, &joint_hashes)) {
      fail(CollectorFailureCode::kDigestFailure, cudaErrorInvalidValue,
           "segment digest initialization failed");
      return false;
    }

    for (std::size_t call = 0U; call < calls.size(); ++call) {
      const ActivationCallSlot& slot = calls[call];
      const lower_bound::ProjectionRole role = lower_role(slot.role);
      const std::size_t index = role_index(role);
      const std::size_t counted_m16_tiles =
          segment_rows(completed_segments) /
          active_cell::kActivationRowsPerMask;
      if (index >= lower_bound::kProjectionRoleCount ||
          slot.current_mask_count == 0U ||
          slot.partition_count == 0U ||
          counted_m16_tiles > slot.current_m16_tiles) {
        fail(CollectorFailureCode::kProtocolOrder, cudaErrorInvalidValue,
             "complete segment contains an unpopulated call slot");
        return false;
      }
      q3x::core::Sha256& activation_hash = activation_hashes[index];
      const std::uint16_t* const a_masks =
          pinned_a_masks + slot.host_mask_offset;
      q3x::core::Sha256 payload_hash;
      if (!hash_masks(&payload_hash, a_masks, slot.current_mask_count)) {
        fail(CollectorFailureCode::kDigestFailure, cudaErrorInvalidValue,
             "activation support-mask payload SHA-256 failed");
        return false;
      }
      const lower_bound::Sha256Digest payload_sha256 =
          lower_digest(payload_hash.finalize());
      std::uint64_t payload_bytes = 0U;
      if (!digest_is_nonzero(payload_sha256) ||
          !checked_multiply_u64(slot.current_mask_count,
                                sizeof(std::uint16_t), &payload_bytes) ||
          !checked_add_u64(&segment_activation_calls[index], 1U) ||
          !checked_add_u64(&segment_activation_mask_elements[index],
                           slot.current_mask_count) ||
          !checked_add_u64(&segment_activation_payload_bytes[index],
                           payload_bytes) ||
          !hash_string(&activation_hash, kActivationCallDomain) ||
          !hash_u64(&activation_hash, completed_segments) ||
          !hash_u64(&activation_hash, segment_first_position(completed_segments)) ||
          !hash_u64(&activation_hash, segment_rows(completed_segments)) ||
          !hash_u64(&activation_hash, call) ||
          !hash_u64(&activation_hash, slot.layer) ||
          !hash_u64(&activation_hash, static_cast<std::uint64_t>(slot.role)) ||
          !hash_u64(&activation_hash,
                    static_cast<std::uint64_t>(HookScope::
                        kLegacyPrefixLayerSegmentFinalScalarExcluded)) ||
          !hash_string(&activation_hash, kActivationSourceDtype) ||
          !hash_u64(&activation_hash, segment_rows(completed_segments)) ||
          !hash_u64(&activation_hash, slot.current_k16_groups * 16U) ||
          !hash_u64(&activation_hash, slot.current_k16_groups * 16U) ||
          !hash_string(&activation_hash, kActivationPayloadDtype) ||
          !hash_u64(&activation_hash, slot.current_m16_tiles) ||
          !hash_u64(&activation_hash, counted_m16_tiles) ||
          !hash_u64(&activation_hash, slot.current_k16_groups) ||
          !hash_u64(&activation_hash, slot.current_mask_count) ||
          !hash_u64(&activation_hash, payload_bytes) ||
          !hash_u64(&activation_hash, 0U) ||
          !hash_u64(&activation_hash, slot.partition_count) ||
          !hash_digest(&activation_hash, payload_sha256)) {
        fail(CollectorFailureCode::kDigestFailure, cudaErrorInvalidValue,
             "activation inventory digest update failed");
        return false;
      }

      q3x::core::Sha256& joint_hash = joint_hashes[index];
      for (std::size_t partition = 0U;
           partition < slot.partition_count; ++partition) {
        const std::size_t record_index = slot.weight_records[partition];
        if (record_index >= weight_record_count) {
          fail(CollectorFailureCode::kWeightInventoryMismatch,
               cudaErrorInvalidValue, "call references an unknown weight record");
          return false;
        }
        const WeightPartitionRecord& record = weight_records[record_index];
        if (record.k16_groups != slot.current_k16_groups) {
          fail(CollectorFailureCode::kWeightInventoryMismatch,
               cudaErrorInvalidValue, "A/B K16 geometry changed within a call");
          return false;
        }

        std::uint64_t geometric = 0U;
        if (!checked_multiply_u64(counted_m16_tiles,
                                  record.k16_groups, &geometric) ||
            !checked_multiply_u64(geometric, record.n8_tiles, &geometric) ||
            !checked_add_u64(&segment_observed[index], geometric)) {
          fail(CollectorFailureCode::kArithmeticOverflow,
               cudaErrorInvalidValue, "observed joint-cell count overflow");
          return false;
        }

        std::uint64_t partition_active = 0U;
        for (std::size_t m16 = 0U; m16 < counted_m16_tiles; ++m16) {
          for (std::size_t k16 = 0U; k16 < slot.current_k16_groups; ++k16) {
            // Frozen static-schedule predicate: a partially supported A K16
            // cell is deliberately free, even if fifteen of its sixteen bits
            // are set. A full A support mask plus a nonzero B support mask is
            // the only parent class that the v2 mapping requires to issue;
            // result-aware/cancellation pruning is explicitly outside it.
            if (a_masks[m16 * slot.current_k16_groups + k16] == 0xffffU &&
                !checked_add_u64(
                    &partition_active,
                    record.nonzero_n8_tiles_by_k16[k16])) {
              fail(CollectorFailureCode::kArithmeticOverflow,
                   cudaErrorInvalidValue, "active joint-cell count overflow");
              return false;
            }
          }
        }
        if (!checked_add_u64(&segment_active[index], partition_active) ||
            !hash_u64(&joint_hash, call) ||
            !hash_u64(&joint_hash, slot.layer) ||
            !hash_u64(&joint_hash, partition) ||
            !hash_u64(&joint_hash, record_index) ||
            !hash_u64(&joint_hash, record.output_features) ||
            !hash_u64(&joint_hash, record.input_features) ||
            !hash_digest(&joint_hash, record.mask_sha256) ||
            !hash_u64(&joint_hash, partition_active)) {
          fail(CollectorFailureCode::kDigestFailure, cudaErrorInvalidValue,
               "joint enumeration digest update failed");
          return false;
        }
      }
    }

    const std::uint32_t rows = static_cast<std::uint32_t>(
        segment_rows(completed_segments));
    if (rows == 0U ||
        covered_prompt_rows >
            std::numeric_limits<std::uint32_t>::max() - rows) {
      fail(CollectorFailureCode::kArithmeticOverflow, cudaErrorInvalidValue,
           "covered prompt row count overflow");
      return false;
    }
    const std::uint32_t candidate_covered = covered_prompt_rows + rows;
    std::array<std::uint64_t, lower_bound::kProjectionRoleCount>
        candidate_observed{};
    std::array<std::uint64_t, lower_bound::kProjectionRoleCount>
        candidate_active{};
    std::array<std::uint64_t, lower_bound::kProjectionRoleCount>
        candidate_activation_calls{};
    std::array<std::uint64_t, lower_bound::kProjectionRoleCount>
        candidate_activation_mask_elements{};
    std::array<std::uint64_t, lower_bound::kProjectionRoleCount>
        candidate_activation_payload_bytes{};
    std::array<lower_bound::Sha256Digest,
               lower_bound::kProjectionRoleCount>
        candidate_activation_digests{};
    std::array<lower_bound::Sha256Digest,
               lower_bound::kProjectionRoleCount>
        candidate_joint_digests{};
    for (std::size_t index = 0U;
         index < lower_bound::kProjectionRoleCount; ++index) {
      const auto role = static_cast<lower_bound::ProjectionRole>(index + 1U);
      candidate_observed[index] = observed_cells[index];
      candidate_active[index] = active_cells[index];
      candidate_activation_calls[index] = activation_call_counts[index];
      candidate_activation_mask_elements[index] =
          activation_mask_element_counts[index];
      candidate_activation_payload_bytes[index] =
          activation_payload_bytes[index];
      if (!checked_add_u64(&candidate_observed[index],
                           segment_observed[index]) ||
          !checked_add_u64(&candidate_active[index], segment_active[index]) ||
          !checked_add_u64(&candidate_activation_calls[index],
                           segment_activation_calls[index]) ||
          !checked_add_u64(&candidate_activation_mask_elements[index],
                           segment_activation_mask_elements[index]) ||
          !checked_add_u64(&candidate_activation_payload_bytes[index],
                           segment_activation_payload_bytes[index]) ||
          candidate_activation_calls[index] !=
              expected_calls_per_segment(role) *
                  static_cast<std::uint64_t>(completed_segments + 1U) ||
          candidate_active[index] > candidate_observed[index] ||
          candidate_observed[index] !=
              lower_bound::frozen_geometric_joint_k16_cells_for_prefix_rows(
                  role, candidate_covered)) {
        fail(CollectorFailureCode::kArithmeticOverflow,
             cudaErrorInvalidValue,
             "segment counts do not match frozen prefix geometry");
        return false;
      }
      candidate_activation_digests[index] =
          lower_digest(activation_hashes[index].finalize());
      if (!digest_is_nonzero(candidate_activation_digests[index]) ||
          !hash_digest(&joint_hashes[index],
                       candidate_activation_digests[index]) ||
          !hash_u64(&joint_hashes[index], candidate_observed[index]) ||
          !hash_u64(&joint_hashes[index], candidate_active[index])) {
        fail(CollectorFailureCode::kDigestFailure, cudaErrorInvalidValue,
             "segment activation/joint digest finalization failed");
        return false;
      }
      candidate_joint_digests[index] =
          lower_digest(joint_hashes[index].finalize());
      if (!digest_is_nonzero(candidate_joint_digests[index])) {
        fail(CollectorFailureCode::kDigestFailure, cudaErrorInvalidValue,
             "joint enumeration digest finalized to zero");
        return false;
      }
    }

    // Commit only after the entire segment has been checked and hashed.
    observed_cells = candidate_observed;
    active_cells = candidate_active;
    activation_call_counts = candidate_activation_calls;
    activation_mask_element_counts = candidate_activation_mask_elements;
    activation_payload_bytes = candidate_activation_payload_bytes;
    activation_inventory_sha256 = candidate_activation_digests;
    joint_enumeration_sha256 = candidate_joint_digests;
    covered_prompt_rows = candidate_covered;
    ++completed_segments;
    expected_call = 0U;
    current_stream_valid = false;
    return true;
  }

  [[nodiscard]] bool issue_lower_bound(
      std::optional<lower_bound::DecisionReceipt>* const output) noexcept {
    if (output == nullptr) {
      return false;
    }
    lower_bound::LowerBoundIssuer issuer;
    if (completed_segments != 0U) {
      for (std::size_t index = 0U;
           index < lower_bound::kProjectionRoleCount; ++index) {
        const auto role = static_cast<lower_bound::ProjectionRole>(index + 1U);
        lower_bound::RoleEvidence evidence;
        evidence.role = role;
        evidence.expected_joint_k16_cells =
            lower_bound::frozen_expected_joint_k16_cells(role);
        evidence.observed_joint_k16_cells = observed_cells[index];
        evidence.active_joint_k16_cells = active_cells[index];
        evidence.covered_prompt_rows = covered_prompt_rows;
        evidence.real_p40_route_sha256 = identity.real_p40_route_sha256;
        evidence.activation_inventory_sha256 =
            activation_inventory_sha256[index];
        evidence.checkpoint_manifest_sha256 = checkpoint_manifest_sha256;
        evidence.joint_enumeration_sha256 = joint_enumeration_sha256[index];
        evidence.real_p40_route_authenticated =
            identity.real_p40_route_authenticated;
        evidence.activation_inventory_authenticated =
            digest_is_nonzero(activation_inventory_sha256[index]);
        evidence.checkpoint_manifest_authenticated =
            checkpoint_digest_ready && identity.shard_manifest_authenticated;
        evidence.joint_enumeration_authenticated =
            digest_is_nonzero(joint_enumeration_sha256[index]);
        evidence.covered_rows_are_contiguous_prefix = true;
        evidence.observed_subset_complete = true;
        if (!issuer.add(evidence)) {
          fail(CollectorFailureCode::kLowerBoundIssuerFailure,
               cudaErrorInvalidValue,
               "lower-bound issuer rejected collector evidence");
          return false;
        }
      }
    }
    output->emplace(issuer.finalize());
    return output->has_value();
  }

  [[nodiscard]] int finish_segment(const cudaStream_t stream) noexcept {
    // The flag copy joins the same unique segment boundary as all 256 fixed A
    // slots.  It observes every preceding A/B kernel on the runner stream.
    cudaError_t status = cudaMemcpyAsync(
        pinned_exceptional_flag, device_exceptional_flag,
        sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream);
    if (status != cudaSuccess) {
      fail(CollectorFailureCode::kCudaCopy, status,
           "exceptional-operand flag D2H copy failed");
      return failure_status();
    }
    status = cudaStreamSynchronize(stream);
    pending_stream_work = false;
    if (status != cudaSuccess) {
      fail(CollectorFailureCode::kCudaSynchronize, status,
           "complete-segment A-mask stream synchronization failed");
      return failure_status();
    }
    exceptional_flag_value = *pinned_exceptional_flag;
    if (exceptional_flag_value != 0U) {
      exceptional_operand_detected = true;
      fail(CollectorFailureCode::kExceptionalOperand,
           cudaErrorInvalidValue,
           "exceptional operand excludes strict active-cell evidence");
      return failure_status();
    }
    if (!build_checkpoint_digest() || !process_complete_segment()) {
      return failure_status();
    }

    std::optional<lower_bound::DecisionReceipt> receipt;
    if (!issue_lower_bound(&receipt) || !receipt.has_value()) {
      return failure_status();
    }
    last_receipt = receipt;
    if (receipt->decision() == lower_bound::Decision::kReject) {
      // The issuer owns the strict integral comparison.  Keep a redundant
      // invariant check so a future receipt change cannot weaken this sentinel.
      if (receipt->proven_warp_instruction_lower_bound() <=
              receipt->five_second_absolute_instruction_capacity() ||
          receipt->proven_warp_instruction_lower_bound() <=
              kStrictFiveSecondCellThreshold) {
        fail(CollectorFailureCode::kLowerBoundIssuerFailure,
             cudaErrorInvalidValue,
             "REJECT receipt did not exceed the strict frozen threshold");
        return failure_status();
      }
      early_reject = true;
      return static_cast<int>(kEarlyRejectCudaStatus);
    }
    return static_cast<int>(cudaSuccess);
  }

  [[nodiscard]] int observe(
      const witness_hook::AotArithmeticWitnessAOperandView& view) noexcept {
    if (early_reject) {
      return static_cast<int>(kEarlyRejectCudaStatus);
    }
    if (failure_code != CollectorFailureCode::kNone) {
      return failure_status();
    }
    if (finalized) {
      fail(CollectorFailureCode::kAlreadyFinalized, cudaErrorInvalidValue,
           "callback invoked after collector finalize");
      return failure_status();
    }
    if (completed_segments >= kRealP40SegmentCount ||
        expected_call >= kRealP40CallsPerSegment) {
      fail(CollectorFailureCode::kProtocolOrder, cudaErrorInvalidValue,
           "callback exceeds the Legacy P40 prefix schedule");
      return failure_status();
    }

    const std::size_t expected_layer =
        expected_call / kRealP40CallsPerLayer;
    const std::size_t phase = expected_call % kRealP40CallsPerLayer;
    const HookRole scheduled_role = expected_role(expected_layer, phase);
    const std::size_t scheduled_rows = segment_rows(completed_segments);
    if (!witness_hook::is_valid_aot_arithmetic_witness_a_operand_view(view)) {
      fail(CollectorFailureCode::kInvalidOperandView, cudaErrorInvalidValue,
           "hook supplied an invalid operand view");
      return failure_status();
    }
    if (view.layer != expected_layer || view.role != scheduled_role ||
        view.scope != HookScope::
                          kLegacyPrefixLayerSegmentFinalScalarExcluded ||
        !view.ordinary_sm87_legacy_c512_scope ||
        !view.production_final_scalar_excluded ||
        view.first_position != covered_prompt_rows ||
        view.first_position != segment_first_position(completed_segments) ||
        view.token_count != scheduled_rows || view.k != expected_k(view.role) ||
        view.activation_row_stride_elements != view.k) {
      fail(CollectorFailureCode::kProtocolOrder, cudaErrorInvalidValue,
           "hook order/segment/scope differs from the frozen Legacy P40 schedule");
      return failure_status();
    }

    const cudaStream_t stream = static_cast<cudaStream_t>(view.cuda_stream);
    if (expected_call == 0U) {
      current_stream = stream;
      current_stream_valid = true;
      const cudaError_t status = cudaMemsetAsync(
          device_exceptional_flag, 0, sizeof(std::uint32_t), stream);
      if (status != cudaSuccess) {
        fail(CollectorFailureCode::kCudaCopy, status,
             "exceptional-operand flag clear failed");
        return failure_status();
      }
      pending_stream_work = true;
    } else if (!current_stream_valid || current_stream != stream) {
      fail(CollectorFailureCode::kProtocolOrder, cudaErrorInvalidValue,
           "one segment used more than one CUDA stream");
      return failure_status();
    }

    ActivationCallSlot& slot = calls[expected_call];
    active_cell::MaskLayout layout;
    if (!active_cell::bf16_a_mask_layout(
            view.token_count, view.k, view.activation_row_stride_elements,
            &layout) ||
        layout.mask_count > slot.maximum_mask_count ||
        slot.host_mask_offset + layout.mask_count > activation_slot_elements) {
      fail(CollectorFailureCode::kInvalidOperandView, cudaErrorInvalidValue,
           "activation mask geometry exceeds its fixed call slot");
      return failure_status();
    }
    cudaError_t status = active_cell::launch_bf16_a_k_support_masks(
        view.activation_bf16, view.token_count, view.k,
        view.activation_row_stride_elements, device_masks, layout.mask_count,
        device_exceptional_flag, stream);
    if (status != cudaSuccess) {
      fail(CollectorFailureCode::kCudaMaskLaunch, status,
           "activation A-mask launch failed");
      return failure_status();
    }
    pending_stream_work = true;
    status = cudaMemcpyAsync(pinned_a_masks + slot.host_mask_offset,
                             device_masks,
                             layout.mask_count * sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost, stream);
    if (status != cudaSuccess) {
      fail(CollectorFailureCode::kCudaCopy, status,
           "activation A-mask D2H copy failed");
      return failure_status();
    }
    slot.current_mask_count = layout.mask_count;
    slot.current_m16_tiles = layout.outer_tiles;
    slot.current_k16_groups = layout.k16_groups;
    slot.partition_count = view.partition_count;
    for (std::size_t partition = 0U; partition < view.partition_count;
         ++partition) {
      if (!get_weight_record(view, partition, stream,
                             &slot.weight_records[partition])) {
        return failure_status();
      }
    }

    ++expected_call;
    if (expected_call == kRealP40CallsPerSegment) {
      return finish_segment(stream);
    }
    return static_cast<int>(cudaSuccess);
  }

  CollectorIdentity identity{};
  std::array<ActivationCallSlot, kRealP40CallsPerSegment> calls{};
  std::array<WeightPartitionRecord, kExpectedWeightPartitionCount>
      weight_records{};
  std::array<std::uint32_t, lower_bound::kProjectionRoleCount>
      weight_record_counts_by_role{};
  std::array<std::uint64_t, lower_bound::kProjectionRoleCount>
      observed_cells{};
  std::array<std::uint64_t, lower_bound::kProjectionRoleCount> active_cells{};
  std::array<std::uint64_t, lower_bound::kProjectionRoleCount>
      activation_call_counts{};
  std::array<std::uint64_t, lower_bound::kProjectionRoleCount>
      activation_mask_element_counts{};
  std::array<std::uint64_t, lower_bound::kProjectionRoleCount>
      activation_payload_bytes{};
  std::array<lower_bound::Sha256Digest,
             lower_bound::kProjectionRoleCount>
      activation_inventory_sha256{};
  std::array<lower_bound::Sha256Digest,
             lower_bound::kProjectionRoleCount>
      joint_enumeration_sha256{};
  lower_bound::Sha256Digest checkpoint_manifest_sha256{};
  std::optional<lower_bound::DecisionReceipt> last_receipt;
  std::array<char, 256U> failure_text{};
  std::uint16_t* device_masks = nullptr;
  std::uint32_t* device_exceptional_flag = nullptr;
  std::uint16_t* pinned_b_masks = nullptr;
  std::uint16_t* pinned_a_masks = nullptr;
  std::uint32_t* pinned_exceptional_flag = nullptr;
  cudaStream_t current_stream = nullptr;
  std::size_t activation_slot_elements = 0U;
  std::size_t weight_record_count = 0U;
  std::size_t expected_call = 0U;
  std::uint32_t covered_prompt_rows = 0U;
  std::uint32_t completed_segments = 0U;
  CollectorFailureCode failure_code = CollectorFailureCode::kNone;
  int cuda_status = static_cast<int>(cudaSuccess);
  std::uint32_t exceptional_flag_value = 0U;
  bool current_stream_valid = false;
  bool pending_stream_work = false;
  bool checkpoint_digest_ready = false;
  bool early_reject = false;
  bool exceptional_operand_detected = false;
  bool finalized = false;
};

RealP40ActiveCellWitnessCollector::RealP40ActiveCellWitnessCollector(
    const CollectorIdentity identity) noexcept
    : impl_(new (std::nothrow) Impl(identity)) {}

RealP40ActiveCellWitnessCollector::~RealP40ActiveCellWitnessCollector() {
  delete impl_;
}

bool RealP40ActiveCellWitnessCollector::valid() const noexcept {
  return impl_ != nullptr &&
         impl_->failure_code == CollectorFailureCode::kNone &&
         impl_->device_masks != nullptr && impl_->pinned_b_masks != nullptr &&
         impl_->pinned_a_masks != nullptr &&
         impl_->device_exceptional_flag != nullptr &&
         impl_->pinned_exceptional_flag != nullptr;
}

bool RealP40ActiveCellWitnessCollector::early_reject_requested() const noexcept {
  return impl_ != nullptr && impl_->early_reject;
}

witness_hook::AotArithmeticWitnessAOperandHook
RealP40ActiveCellWitnessCollector::hook() noexcept {
  return witness_hook::AotArithmeticWitnessAOperandHook{
      &RealP40ActiveCellWitnessCollector::a_operand_callback, this};
}

int RealP40ActiveCellWitnessCollector::a_operand_callback(
    const witness_hook::AotArithmeticWitnessAOperandView& view,
    void* const context) noexcept {
  if (context == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  auto* const collector =
      static_cast<RealP40ActiveCellWitnessCollector*>(context);
  if (collector->impl_ == nullptr) {
    return static_cast<int>(cudaErrorMemoryAllocation);
  }
  return collector->impl_->observe(view);
}

CollectorResult RealP40ActiveCellWitnessCollector::finalize() noexcept {
  CollectorResult result;
  if (impl_ == nullptr) {
    result.failure_code = CollectorFailureCode::kHostAllocation;
    result.cuda_status = static_cast<int>(cudaErrorMemoryAllocation);
    (void)std::snprintf(result.failure_text.data(), result.failure_text.size(),
                        "%s", "host allocation for collector state failed");
    return result;
  }

  // A callback may fail after enqueueing mask copies or B-mask host
  // functions but before reaching the normal segment-boundary synchronize.
  // Join that work before reading any record written by a CUDA host function;
  // otherwise error finalization would race host_job_ready/mask_sha256.
  if (!impl_->finalized && impl_->pending_stream_work) {
    const bool stream_identity_valid = impl_->current_stream_valid;
    const cudaError_t status =
        stream_identity_valid ? cudaStreamSynchronize(impl_->current_stream)
                              : cudaDeviceSynchronize();
    impl_->pending_stream_work = false;
    if (status != cudaSuccess) {
      impl_->fail(CollectorFailureCode::kCudaSynchronize, status,
                  "finalize failed to join pending witness stream work");
    } else if (!stream_identity_valid) {
      impl_->fail(CollectorFailureCode::kProtocolOrder,
                  cudaErrorInvalidValue,
                  "pending witness work had no owning CUDA stream");
    }
  }
  if (!impl_->finalized && impl_->expected_call != 0U &&
      impl_->failure_code == CollectorFailureCode::kNone) {
    impl_->fail(CollectorFailureCode::kIncompleteSegment,
                cudaErrorInvalidValue,
                "finalize excluded an incomplete in-flight segment");
  }
  impl_->finalized = true;
  // A segment boundary already owns a privately issued immutable receipt.
  // Reuse it (especially after the sentinel REJECT) instead of issuing twice.
  // With zero complete segments there is no cached receipt, so issue the sole
  // missing-role INCONCLUSIVE receipt now.
  if (impl_->last_receipt.has_value()) {
    result.lower_bound_receipt = impl_->last_receipt;
  } else if (!impl_->issue_lower_bound(&result.lower_bound_receipt) &&
             impl_->failure_code == CollectorFailureCode::kNone) {
      impl_->fail(CollectorFailureCode::kLowerBoundIssuerFailure,
                  cudaErrorInvalidValue, "final lower-bound issuance failed");
  }
  result.covered_prompt_rows = impl_->covered_prompt_rows;
  result.completed_segments = impl_->completed_segments;
  result.failure_code = impl_->failure_code;
  result.cuda_status = impl_->cuda_status;
  result.failure_text = impl_->failure_text;
  result.checkpoint_manifest_sha256 = impl_->checkpoint_manifest_sha256;
  result.checkpoint_inventory_complete = impl_->complete_weight_inventory();
  result.checkpoint_manifest_authenticated =
      impl_->checkpoint_digest_ready &&
      impl_->identity.shard_manifest_authenticated;
  result.early_reject_requested = impl_->early_reject;
  result.exceptional_flag_value = impl_->exceptional_flag_value;
  result.exceptional_operand_detected =
      impl_->exceptional_operand_detected;
  for (std::size_t index = 0U;
       index < lower_bound::kProjectionRoleCount; ++index) {
    RoleSummary& role = result.roles[index];
    role.role = static_cast<lower_bound::ProjectionRole>(index + 1U);
    role.observed_geometric_joint_k16_cells = impl_->observed_cells[index];
    role.proven_active_joint_k16_cells = impl_->active_cells[index];
    role.activation_inventory_sha256 =
        impl_->activation_inventory_sha256[index];
    role.joint_enumeration_sha256 = impl_->joint_enumeration_sha256[index];
    role.activation_call_count = impl_->activation_call_counts[index];
    role.activation_mask_element_count =
        impl_->activation_mask_element_counts[index];
    role.activation_payload_bytes = impl_->activation_payload_bytes[index];
    role.weight_partition_count =
        impl_->weight_record_counts_by_role[index];
    role.weight_inventory_complete =
        role.weight_partition_count == expected_partition_inventory(role.role);
    role.activation_inventory_authenticated =
        impl_->completed_segments != 0U &&
        digest_is_nonzero(role.activation_inventory_sha256);
    role.joint_enumeration_authenticated =
        impl_->completed_segments != 0U &&
        digest_is_nonzero(role.joint_enumeration_sha256);
  }
  return result;
}

bool run_real_p40_active_cell_protocol_self_test() noexcept {
  bool ok = true;
  const CollectorResult default_result;
  ok = ok &&
       static_cast<std::uint8_t>(
           CollectorFailureCode::kExceptionalOperand) == 15U &&
       default_result.exceptional_flag_value == 0U &&
       !default_result.exceptional_operand_detected;
  std::size_t first_position = 0U;
  std::size_t slot_elements = 0U;
  std::array<std::size_t, lower_bound::kProjectionRoleCount>
      partition_inventory{};
  std::array<std::size_t, lower_bound::kProjectionRoleCount> call_inventory{};
  for (std::size_t segment = 0U; segment < kRealP40SegmentCount; ++segment) {
    ok = ok && segment_first_position(segment) == first_position;
    first_position += segment_rows(segment);
  }
  ok = ok &&
       first_position == witness_hook::kAotArithmeticWitnessP40PrefixRows;

  for (std::size_t call = 0U; call < kRealP40CallsPerSegment; ++call) {
    const std::size_t layer = call / kRealP40CallsPerLayer;
    const std::size_t phase = call % kRealP40CallsPerLayer;
    const HookRole role = expected_role(layer, phase);
    const std::size_t index = role_index(lower_role(role));
    ok = ok && role != HookRole::kInvalid &&
         index < lower_bound::kProjectionRoleCount;
    ++call_inventory[index];
    slot_elements += kMaximumM16Tiles * (expected_k(role) / 16U);
    if (phase == 0U) {
      partition_inventory[index] += layer % 4U == 3U ? 3U : 2U;
    } else {
      partition_inventory[index] += phase == 2U ? 2U : 1U;
    }
  }
  ok = ok && slot_elements * sizeof(std::uint16_t) == 8'650'752U &&
       slot_elements * sizeof(std::uint16_t) <=
           kPinnedActivationSlotBytes;

  std::size_t total_partitions = 0U;
  for (std::size_t index = 0U;
       index < lower_bound::kProjectionRoleCount; ++index) {
    const auto role = static_cast<lower_bound::ProjectionRole>(index + 1U);
    ok = ok && call_inventory[index] == expected_calls_per_segment(role);
    ok = ok && partition_inventory[index] ==
                   expected_partition_inventory(role);
    total_partitions += partition_inventory[index];
    const std::uint64_t prefix_cells =
        lower_bound::frozen_geometric_joint_k16_cells_for_prefix_rows(
            role, witness_hook::kAotArithmeticWitnessP40PrefixRows);
    const std::uint64_t complete_cells =
        lower_bound::frozen_expected_joint_k16_cells(role);
    ok = ok && prefix_cells < complete_cells;
  }
  ok = ok && total_partitions == kExpectedWeightPartitionCount;
  const auto& mapping = lower_bound::frozen_mapping_spec();
  ok = ok && mapping.p40_prompt_rows == 40'000U &&
       mapping.production_prefix_rows ==
           witness_hook::kAotArithmeticWitnessP40PrefixRows &&
       mapping.terminal_scalar_rows == 1U &&
       mapping.partial_m16_tail_rows_charged_free &&
       mapping.physical_instructions_per_active_joint_k16_cell == 1U &&
       !mapping.support_active_parent_predicate.empty() &&
       !mapping.exact_arithmetic_pass_floor.empty() &&
       mapping.one_parent_per_instruction &&
       mapping.cross_parent_packing_forbidden &&
       mapping.support_active_parent_must_issue &&
       mapping.result_aware_parent_elision_forbidden &&
       mapping.cross_parent_common_subexpression_elimination_forbidden &&
       mapping.additional_exactness_passes_charged_free &&
       kStrictFiveSecondCellThreshold ==
           static_cast<std::uint64_t>(mapping.maximum_warp_instructions_per_sm_cycle) *
               mapping.sm_count * mapping.clock_hz *
               mapping.projection_budget_seconds;
  return ok;
}

}  // namespace q3x::test::sm87_aot_real_p40_active_cell
