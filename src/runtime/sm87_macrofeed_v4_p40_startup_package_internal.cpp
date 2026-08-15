#include "sm87_macrofeed_v4_p40_startup_package_internal.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <tuple>
#include <utility>
#include <variant>

#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_STARTUP_PACKAGE_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_MACROFEED_V4_BF16_AB_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_MACROFEED_V4_FP8_ADMISSION)
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_STARTUP_PACKAGE_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_MACROFEED_V4_BF16_AB_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_MACROFEED_V4_FP8_ADMISSION)
namespace q3x::runtime::target_aot_complete_execution_detail {
namespace {

[[nodiscard]] std::uint64_t request_boundary_resident_root_identity(
    const ModelWeights* const model_weights,
    const ResidentWeights* const resident, const std::uint64_t owner_identity,
    const std::uint64_t allocation_identity,
    const std::uint64_t device_identity, const std::int32_t device_ordinal,
    const std::uintptr_t arena_begin, const std::uint64_t arena_bytes,
    const bool host_test_authority,
    const std::uint64_t issuer_nonce) noexcept {
  if (model_weights == nullptr || resident == nullptr || owner_identity == 0U ||
      allocation_identity == 0U || device_identity == 0U ||
      device_ordinal < 0 || arena_begin == 0U || arena_bytes == 0U ||
      arena_bytes >
          std::numeric_limits<std::uintptr_t>::max() - arena_begin ||
      (host_test_authority ? issuer_nonce == 0U
                           : issuer_nonce != 0U ||
                                 arena_bytes !=
                                     kPinnedQwen36_27BArenaBytes)) {
    return 0U;
  }
  auto mix = [](std::uint64_t value, const std::uint64_t input) noexcept {
    value ^= input + 0x9e37'79b9'7f4a'7c15ULL + (value << 6U) +
             (value >> 2U);
    return value;
  };
  std::uint64_t identity = host_test_authority
                               ? 0x5133'5854'4553'5452ULL
                               : 0x5133'5852'4553'524fULL;
  identity = mix(identity, reinterpret_cast<std::uintptr_t>(model_weights));
  identity = mix(identity, owner_identity);
  identity = mix(identity, allocation_identity);
  identity = mix(identity, device_identity);
  identity = mix(identity, static_cast<std::uint64_t>(device_ordinal + 1));
  identity = mix(identity, reinterpret_cast<std::uintptr_t>(resident));
  identity = mix(identity, arena_begin);
  identity = mix(identity, arena_bytes);
  identity = mix(identity, issuer_nonce);
  return identity;
}

}  // namespace

std::optional<Sm87TargetAotCompleteProjectionExecutionAccess>
Sm87TargetAotCompleteProjectionExecutionAccess::
    bind_request_boundary_startup(const ModelWeights& model_weights) noexcept {
  auto projection = bind(model_weights);
  if (!projection) {
    return std::nullopt;
  }
  const ResidentWeights* const resident = projection->owner_->prepared_resident_;
  if (resident == nullptr || !*resident || resident->arena_data() == nullptr ||
      resident->size_bytes() != kPinnedQwen36_27BArenaBytes) {
    return std::nullopt;
  }
  const std::uintptr_t resident_arena_begin =
      reinterpret_cast<std::uintptr_t>(resident->arena_data());
  const std::uint64_t resident_root_identity =
      request_boundary_resident_root_identity(
          &model_weights, resident, projection->owner_identity_,
          projection->allocation_identity_, projection->device_identity_,
          projection->device_ordinal_, resident_arena_begin,
          resident->size_bytes(), false, 0U);
  if (resident_root_identity == 0U) {
    return std::nullopt;
  }
  projection->resident_root_ = resident;
  projection->resident_root_identity_ = resident_root_identity;
  projection->resident_arena_begin_ = resident_arena_begin;
  projection->resident_arena_bytes_ = resident->size_bytes();
  projection->host_test_resident_authority_ = false;
  projection->host_test_issuer_nonce_ = 0U;
  return projection;
}

bool Sm87TargetAotCompleteProjectionExecutionAccess::resident_root_matches()
    const noexcept {
  if (model_weights_ == nullptr || owner_ == nullptr ||
      owner_->prepared_model_weights_ != model_weights_ ||
      owner_->prepared_resident_ != resident_root_ ||
      resident_root_identity_ == 0U || resident_arena_begin_ == 0U ||
      resident_arena_bytes_ == 0U) {
    return false;
  }
  if (host_test_resident_authority_) {
    return host_test_issuer_nonce_ == kHostTestResidentIssuerNonce &&
           resident_root_identity_ == request_boundary_resident_root_identity(
                                          model_weights_, resident_root_,
                                          owner_identity_,
                                          allocation_identity_,
                                          device_identity_, device_ordinal_,
                                          resident_arena_begin_,
                                          resident_arena_bytes_, true,
                                          host_test_issuer_nonce_);
  }
  if (host_test_issuer_nonce_ != 0U || resident_root_ == nullptr ||
      !*resident_root_ || resident_root_->arena_data() == nullptr ||
      reinterpret_cast<std::uintptr_t>(resident_root_->arena_data()) !=
          resident_arena_begin_ ||
      resident_root_->size_bytes() != resident_arena_bytes_) {
    return false;
  }
  return resident_root_identity_ == request_boundary_resident_root_identity(
                                        model_weights_, resident_root_,
                                        owner_identity_, allocation_identity_,
                                        device_identity_, device_ordinal_,
                                        resident_arena_begin_,
                                        resident_arena_bytes_, false, 0U);
}

}  // namespace q3x::runtime::target_aot_complete_execution_detail
#endif

namespace q3x::runtime::sm87_macrofeed_v4_p40_startup_package_detail {
namespace {

using Role = kernels::Sm87TargetAotProjectionRole;
using Encoding = kernels::Sm87TargetAotProjectionEncoding;
using Error = Sm87MacroFeedV4P40StartupPackageError;
using Status = Sm87MacroFeedV4P40StartupPackageStatus;
using Package = Sm87MacroFeedV4P40StartupPackage;
using CreateResult = Sm87MacroFeedV4P40StartupPackageCreateResult;
namespace Bound = kernels::sm87_macrofeed_v4_bound_launch_detail;

inline constexpr std::uint64_t kGateUpSealIssuerNonce =
    0x5133'4d46'5634'474eULL;
inline constexpr std::uint64_t kDownSealIssuerNonce =
    0x5133'4d46'5634'444eULL;
inline constexpr std::uint64_t kBf16AbSealIssuerNonce =
    0x5133'4d46'5634'424eULL;
inline constexpr std::uint64_t kBf16AbCapabilityIssuerNonce =
    0x5133'4d46'5634'4243ULL;
inline constexpr std::uint64_t kGdnQkvZSealIssuerNonce =
    0x5133'4d46'5634'4653ULL;
inline constexpr std::uint64_t kFullAttentionSealIssuerNonce =
    0x5133'4d46'5634'4641ULL;
inline constexpr std::uint64_t kRequestBoundarySealIssuerNonce =
    0x5133'4d46'5634'5242ULL;
inline constexpr std::uint64_t kEmbeddingC8000ResourceIdentity =
    0x7634'656d'6263'3830ULL;
inline constexpr std::uint64_t kFinalNormM1ResourceIdentity =
    0x7634'666e'6d31'3531ULL;
inline constexpr std::uint64_t kGreedyArgmaxM1ResourceIdentity =
    0x7634'6172'6731'3234ULL;
inline constexpr std::uint64_t kLmHeadM1ResourceIdentity =
    0x7634'6c6d'6831'3234ULL;
inline constexpr Role kGdnQkvZRole = Role::kFp8GdnQkvZ;
inline constexpr kernels::Sm87MacroFeedV4Fp8InputLayout
    kGdnQkvZInputLayout = kernels::Sm87MacroFeedV4Fp8InputLayout::
        kHiddenContiguousH5120V1;
inline constexpr kernels::Sm87MacroFeedV4Fp8Identity
    kGdnQkvZTacticIdentity = kernels::Sm87MacroFeedV4Fp8Identity::
        kGdnQkvZM64N128K64OrdinaryGridV1;
inline constexpr Role kAttentionOutputRole = Role::kFp8AttentionOutput;
inline constexpr kernels::Sm87MacroFeedV4Fp8InputLayout
    kGdnOutputInputLayout = kernels::Sm87MacroFeedV4Fp8InputLayout::
        kGdnContiguousVScratchV1;
inline constexpr kernels::Sm87MacroFeedV4Fp8Identity
    kGdnOutputTacticIdentity = kernels::Sm87MacroFeedV4Fp8Identity::
        kGdnAttentionOutputM64N128K64OrdinaryGridV1;
inline constexpr kernels::Sm87MacroFeedV4Fp8Identity
    kFullQkvTacticIdentity = kernels::Sm87MacroFeedV4Fp8Identity::
        kFullQkvM64N128K64OrdinaryGridV1;
inline constexpr kernels::Sm87MacroFeedV4Fp8Identity
    kFullOutputTacticIdentity = kernels::Sm87MacroFeedV4Fp8Identity::
        kAttentionOutputM64N128K64OrdinaryGridV1;
inline constexpr Role kFullQkvRole = Role::kFp8FullQkv;
inline constexpr kernels::Sm87MacroFeedV4Fp8InputLayout
    kFullQkvInputLayout = kernels::Sm87MacroFeedV4Fp8InputLayout::
        kHiddenContiguousH5120V1;
inline constexpr kernels::Sm87MacroFeedV4Fp8InputLayout
    kFullOutputInputLayout = kernels::Sm87MacroFeedV4Fp8InputLayout::
        kFullAttentionInterleavedQScratchV1;

[[nodiscard]] constexpr std::size_t full_attention_model_layer(
    const std::size_t full_ordinal) noexcept {
  return 4U * full_ordinal + 3U;
}

[[nodiscard]] constexpr bool model_layer_is_full_attention(
    const std::size_t model_layer) noexcept {
  return model_layer < kSm87MacroFeedV4P40StartupPackageLayers &&
         (model_layer + 1U) % 4U == 0U;
}

[[nodiscard]] constexpr std::size_t gdn_model_layer(
    const std::size_t gdn_ordinal) noexcept {
  return gdn_ordinal + gdn_ordinal / 3U;
}

[[nodiscard]] constexpr bool model_layer_is_gdn(
    const std::size_t model_layer) noexcept {
  return model_layer < kSm87MacroFeedV4P40StartupPackageLayers &&
         (model_layer + 1U) % 4U != 0U;
}

struct GdnContinuationSource final {
  const std::uint16_t* conv_weight = nullptr;
  const std::uint16_t* a_log = nullptr;
  const std::uint16_t* dt_bias = nullptr;
  const std::uint16_t* norm_weight = nullptr;
};

[[nodiscard]] constexpr std::uint64_t mix(
    std::uint64_t hash, std::uint64_t value) noexcept;

[[nodiscard]] std::uint32_t fp32_bits(const float value) noexcept {
  std::uint32_t bits = 0U;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

struct RequestBoundaryT0Range final {
  std::uintptr_t begin = 0U;
  std::uintptr_t end = 0U;
};

[[nodiscard]] bool request_boundary_t0_range(
    const void* const pointer, const std::uint64_t bytes,
    RequestBoundaryT0Range* const range) noexcept {
  if (pointer == nullptr || bytes == 0U || range == nullptr) {
    return false;
  }
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (begin % 16U != 0U ||
      bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
    return false;
  }
  *range = {begin, begin + static_cast<std::uintptr_t>(bytes)};
  return range->end > range->begin;
}

[[nodiscard]] constexpr bool request_boundary_ranges_disjoint(
    const RequestBoundaryT0Range& left,
    const RequestBoundaryT0Range& right) noexcept {
  return left.end <= right.begin || right.end <= left.begin;
}

[[nodiscard]] bool exact_gdn_continuation_source(
    const ModelWeights& model_weights, const std::size_t model_layer,
    GdnContinuationSource* const source) noexcept {
  if (source == nullptr || !model_layer_is_gdn(model_layer)) {
    return false;
  }
  *source = {};
  const auto* const linear = std::get_if<LinearAttentionWeights>(
      &model_weights.layer(model_layer).attention);
  if (linear == nullptr || linear->conv1d.data == nullptr ||
      linear->conv1d.shape !=
          std::array<std::size_t, 3U>{{
              kernels::kSm87TargetAotGdnTotalConvChannels, 1U,
              kernels::kSm87TargetAotGdnConvWidth}} ||
      linear->a_log.data == nullptr ||
      linear->a_log.element_count !=
          kernels::kSm87TargetAotGdnScalarHeadElements ||
      linear->dt_bias.data == nullptr ||
      linear->dt_bias.element_count !=
          kernels::kSm87TargetAotGdnScalarHeadElements ||
      linear->norm.data == nullptr ||
      linear->norm.element_count !=
          kernels::kSm87TargetAotGdnNormWeightElements) {
    return false;
  }
  const std::array<const std::uint16_t*, 4U> pointers{{
      linear->conv1d.data, linear->a_log.data, linear->dt_bias.data,
      linear->norm.data}};
  for (const auto* const pointer : pointers) {
    if (reinterpret_cast<std::uintptr_t>(pointer) %
            kernels::kSm87MacroFeedV4GdnPointerAlignment !=
        0U) {
      return false;
    }
  }
  source->conv_weight = linear->conv1d.data;
  source->a_log = linear->a_log.data;
  source->dt_bias = linear->dt_bias.data;
  source->norm_weight = linear->norm.data;
  return true;
}

[[nodiscard]] std::uint64_t gdn_weight_identity(
    const std::uint64_t domain, const std::uint64_t plan_identity,
    const std::uint64_t device_identity, const std::size_t model_layer,
    const std::size_t role_index, const std::uint16_t* const pointer,
    const std::uint64_t bytes) noexcept {
  if (domain == 0U || plan_identity == 0U || device_identity == 0U ||
      !model_layer_is_gdn(model_layer) || role_index >= 4U ||
      pointer == nullptr || bytes == 0U ||
      reinterpret_cast<std::uintptr_t>(pointer) %
              kernels::kSm87MacroFeedV4GdnPointerAlignment !=
          0U) {
    return 0U;
  }
  std::uint64_t identity = domain;
  identity = mix(identity, plan_identity);
  identity = mix(identity, device_identity);
  identity = mix(identity, model_layer + 1U);
  identity = mix(identity, role_index + 1U);
  identity = mix(identity, reinterpret_cast<std::uintptr_t>(pointer));
  identity = mix(identity, bytes);
  return identity == 0U ? domain : identity;
}

[[nodiscard]] constexpr std::array<Role, 4U> layer_roles(
    const std::size_t layer_index) noexcept {
  return {{Role::kNvFp4GateUp, Role::kNvFp4Down,
           sm87_target_aot_complete_is_full_layer(layer_index)
               ? Role::kFp8FullQkv
               : Role::kFp8GdnQkvZ,
           Role::kFp8AttentionOutput}};
}

[[nodiscard]] Status failure(
    const Error error, const char* const context, const int cuda_error = 0,
    const std::size_t layer = kSm87MacroFeedV4P40StartupPackageLayers,
    const Role role = Role::kInvalid) noexcept {
  return {error, context, layer, role, cuda_error};
}

[[nodiscard]] constexpr std::uint64_t mix(
    std::uint64_t hash, const std::uint64_t value) noexcept {
  hash ^= value + 0x9e37'79b9'7f4a'7c15ULL + (hash << 6U) + (hash >> 2U);
  return hash;
}

[[nodiscard]] std::uint64_t mix_string(
    std::uint64_t hash, const std::string_view value) noexcept {
  hash = mix(hash, value.size());
  for (const char byte : value) {
    hash = mix(hash, static_cast<std::uint8_t>(byte));
  }
  return hash;
}

[[nodiscard]] std::uint64_t mix_digest(
    std::uint64_t hash,
    const kernels::Sm87TargetAotProjectionSha256Digest& digest) noexcept {
  for (const std::uint8_t byte : digest.bytes) {
    hash = mix(hash, byte);
  }
  return hash;
}

[[nodiscard]] constexpr bool t0_range(
    const Sm87MacroFeedV4Bf16AbT0TensorDescriptor& tensor,
    std::uintptr_t* const begin, std::uintptr_t* const end) noexcept {
  if (begin == nullptr || end == nullptr || tensor.weight == nullptr ||
      reinterpret_cast<std::uintptr_t>(tensor.weight) %
              kernels::kSm87MacroFeedV4Bf16AbPointerAlignment !=
          0U) {
    return false;
  }
  *begin = reinterpret_cast<std::uintptr_t>(tensor.weight);
  constexpr std::uintptr_t kBytes = static_cast<std::uintptr_t>(
      kernels::kSm87MacroFeedV4Bf16AbWeightBytes);
  if (*begin > std::numeric_limits<std::uintptr_t>::max() - kBytes) {
    return false;
  }
  *end = *begin + kBytes;
  return *end > *begin;
}

[[nodiscard, maybe_unused]] std::uint64_t bf16_ab_pair_identity(
    const std::size_t ordinal, const std::size_t model_layer,
    const std::uint16_t* const a_weights,
    const std::uint16_t* const b_weights) noexcept {
  if (ordinal >= kSm87MacroFeedV4P40StartupPackageBf16AbPairs ||
      model_layer != gdn_model_layer(ordinal) ||
      !model_layer_is_gdn(model_layer) || a_weights == nullptr ||
      b_weights == nullptr) {
    return 0U;
  }
  std::uint64_t identity = 0x5133'4d46'4142'5052ULL;
  identity = mix(identity, ordinal + 1U);
  identity = mix(identity, model_layer + 1U);
  identity = mix(identity, reinterpret_cast<std::uintptr_t>(a_weights));
  identity = mix(identity, reinterpret_cast<std::uintptr_t>(b_weights));
  identity = mix(identity,
                 kernels::kSm87MacroFeedV4Bf16AbWeightBytes);
  identity = mix(identity,
                 static_cast<std::uint64_t>(
                     Sm87MacroFeedV4Bf16AbWeightRole::kA));
  identity = mix(identity,
                 static_cast<std::uint64_t>(
                     Sm87MacroFeedV4Bf16AbWeightRole::kB));
  return identity == 0U ? 0x5133'4d46'4142'5052ULL : identity;
}

[[nodiscard, maybe_unused]] std::uint64_t layer_norm_tensor_identity(
    const std::uint64_t package_identity,
    const std::uint64_t deployment_plan_identity,
    const std::uint64_t device_identity,
    const std::int32_t device_ordinal, const std::size_t model_layer,
    const bool post_attention, const std::uint16_t* const weight,
    const std::size_t element_count,
    const std::uint32_t epsilon_fp32_bits) noexcept {
  if (package_identity == 0U || deployment_plan_identity == 0U ||
      device_identity == 0U || device_ordinal < 0 ||
      model_layer >= kSm87MacroFeedV4P40StartupPackageLayers ||
      weight == nullptr ||
      element_count != kernels::kSm87MacroFeedV4NormResidualHidden ||
      epsilon_fp32_bits !=
          kernels::kSm87MacroFeedV4NormResidualEpsilonFp32Bits) {
    return 0U;
  }
  std::uint64_t identity = post_attention
                               ? 0x5133'4d46'4e50'5354ULL
                               : 0x5133'4d46'4e49'4e50ULL;
  identity = mix(identity, package_identity);
  identity = mix(identity, deployment_plan_identity);
  identity = mix(identity, device_identity);
  identity = mix(identity,
                 static_cast<std::uint64_t>(device_ordinal + 1));
  identity = mix(identity, model_layer + 1U);
  identity = mix(identity, post_attention ? 2U : 1U);
  identity = mix(identity, reinterpret_cast<std::uintptr_t>(weight));
  identity = mix(identity, element_count);
  identity = mix(
      identity, kernels::kSm87MacroFeedV4NormResidualWeightBytes);
  identity = mix(identity, epsilon_fp32_bits);
  return identity == 0U
             ? (post_attention ? 0x5133'4d46'4e50'5354ULL
                               : 0x5133'4d46'4e49'4e50ULL)
             : identity;
}

[[nodiscard, maybe_unused]] std::uint64_t layer_norm_pair_identity(
    const std::size_t model_layer,
    const std::uint64_t input_layernorm_identity,
    const std::uint64_t post_attention_layernorm_identity,
    const std::uint32_t epsilon_fp32_bits) noexcept {
  if (model_layer >= kSm87MacroFeedV4P40StartupPackageLayers ||
      input_layernorm_identity == 0U ||
      post_attention_layernorm_identity == 0U ||
      input_layernorm_identity == post_attention_layernorm_identity ||
      epsilon_fp32_bits !=
          kernels::kSm87MacroFeedV4NormResidualEpsilonFp32Bits) {
    return 0U;
  }
  std::uint64_t identity = 0x5133'4d46'4e50'4149ULL;
  identity = mix(identity, model_layer + 1U);
  identity = mix(identity, input_layernorm_identity);
  identity = mix(identity, post_attention_layernorm_identity);
  identity = mix(identity, epsilon_fp32_bits);
  return identity == 0U ? 0x5133'4d46'4e50'4149ULL : identity;
}

struct FullAttentionQkNormSource final {
  const std::uint16_t* q_norm = nullptr;
  const std::uint16_t* k_norm = nullptr;
};

[[nodiscard, maybe_unused]] bool exact_full_attention_qk_norm_source(
    const ModelWeights& model_weights, const std::size_t model_layer,
    FullAttentionQkNormSource* const source) noexcept {
  if (source == nullptr || !model_layer_is_full_attention(model_layer)) {
    return false;
  }
  *source = {};
  const auto* const full = std::get_if<FullAttentionWeights>(
      &model_weights.layer(model_layer).attention);
  constexpr std::uintptr_t kNormBytes = static_cast<std::uintptr_t>(
      kernels::kSm87MacroFeedV4FullAttentionPreprocessNormWeightBytes);
  if (full == nullptr || full->q_norm.data == nullptr ||
      full->k_norm.data == nullptr ||
      full->q_norm.element_count !=
          kernels::kSm87MacroFeedV4FullAttentionPreprocessHeadDimension ||
      full->k_norm.element_count !=
          kernels::kSm87MacroFeedV4FullAttentionPreprocessHeadDimension ||
      reinterpret_cast<std::uintptr_t>(full->q_norm.data) %
              kernels::kSm87MacroFeedV4FullAttentionPreprocessPointerAlignment !=
          0U ||
      reinterpret_cast<std::uintptr_t>(full->k_norm.data) %
              kernels::kSm87MacroFeedV4FullAttentionPreprocessPointerAlignment !=
          0U ||
      full->q_norm.data == full->k_norm.data) {
    return false;
  }
  const std::uintptr_t q_begin =
      reinterpret_cast<std::uintptr_t>(full->q_norm.data);
  const std::uintptr_t k_begin =
      reinterpret_cast<std::uintptr_t>(full->k_norm.data);
  if (q_begin > std::numeric_limits<std::uintptr_t>::max() - kNormBytes ||
      k_begin > std::numeric_limits<std::uintptr_t>::max() - kNormBytes) {
    return false;
  }
  const std::uintptr_t q_end = q_begin + kNormBytes;
  const std::uintptr_t k_end = k_begin + kNormBytes;
  if (!(q_end <= k_begin || k_end <= q_begin)) {
    return false;
  }
  source->q_norm = full->q_norm.data;
  source->k_norm = full->k_norm.data;
  return true;
}

[[nodiscard, maybe_unused]] std::uint64_t
full_attention_qk_norm_tensor_identity(
    const std::uint64_t package_or_source_domain,
    const std::uint64_t deployment_plan_identity,
    const std::uint64_t device_identity, const std::int32_t device_ordinal,
    const std::size_t full_ordinal, const std::size_t model_layer,
    const bool k_norm, const std::uint16_t* const weight) noexcept {
  constexpr std::size_t kElements =
      kernels::kSm87MacroFeedV4FullAttentionPreprocessHeadDimension;
  constexpr std::uint64_t kBytes =
      kernels::kSm87MacroFeedV4FullAttentionPreprocessNormWeightBytes;
  if (package_or_source_domain == 0U || deployment_plan_identity == 0U ||
      device_identity == 0U || device_ordinal < 0 ||
      full_ordinal >= kSm87MacroFeedV4P40StartupPackageFullLayers ||
      model_layer != full_attention_model_layer(full_ordinal) ||
      !model_layer_is_full_attention(model_layer) || weight == nullptr ||
      reinterpret_cast<std::uintptr_t>(weight) %
              kernels::kSm87MacroFeedV4FullAttentionPreprocessPointerAlignment !=
          0U) {
    return 0U;
  }
  std::uint64_t identity = k_norm ? 0x5133'4d46'464b'4e4dULL
                                  : 0x5133'4d46'4651'4e4dULL;
  identity = mix(identity, package_or_source_domain);
  identity = mix(identity, deployment_plan_identity);
  identity = mix(identity, device_identity);
  identity = mix(identity,
                 static_cast<std::uint64_t>(device_ordinal + 1));
  identity = mix(identity, full_ordinal + 1U);
  identity = mix(identity, model_layer + 1U);
  identity = mix(identity, k_norm ? 2U : 1U);
  identity = mix(identity, reinterpret_cast<std::uintptr_t>(weight));
  identity = mix(identity, kElements);
  identity = mix(identity, kBytes);
  return identity == 0U
             ? (k_norm ? 0x5133'4d46'464b'4e4dULL
                       : 0x5133'4d46'4651'4e4dULL)
             : identity;
}

[[nodiscard, maybe_unused]] std::uint64_t
full_attention_qk_norm_pair_identity(
    const std::size_t full_ordinal, const std::size_t model_layer,
    const std::uint64_t q_norm_identity,
    const std::uint64_t k_norm_identity) noexcept {
  if (full_ordinal >= kSm87MacroFeedV4P40StartupPackageFullLayers ||
      model_layer != full_attention_model_layer(full_ordinal) ||
      q_norm_identity == 0U || k_norm_identity == 0U ||
      q_norm_identity == k_norm_identity) {
    return 0U;
  }
  std::uint64_t identity = 0x5133'4d46'4651'4b50ULL;
  identity = mix(identity, full_ordinal + 1U);
  identity = mix(identity, model_layer + 1U);
  identity = mix(identity, q_norm_identity);
  identity = mix(identity, k_norm_identity);
  return identity == 0U ? 0x5133'4d46'4651'4b50ULL : identity;
}

template <typename UploadReceipt>
[[nodiscard]] bool authenticated_upload_complete(
    const UploadReceipt& upload, const std::uint64_t owner_identity,
    const std::uint64_t allocation_identity,
    const std::int32_t device_ordinal, const std::uintptr_t payload_begin,
    const std::uintptr_t payload_end,
    const std::uint64_t payload_bytes) noexcept {
  return upload.receipt_identity != 0U && owner_identity != 0U &&
         allocation_identity != 0U && device_ordinal >= 0 &&
         upload.device_allocation_owner_identity == owner_identity &&
         upload.device_allocation_identity == allocation_identity &&
         upload.device_ordinal == device_ordinal &&
         upload.device_payload_begin == payload_begin &&
         upload.device_payload_end == payload_end &&
         upload.device_payload_bytes == payload_bytes &&
         upload.host_payload_digest_verified_before_copy &&
         upload.host_payload_immutable_until_completion &&
         upload.copy_enqueued_to_exact_payload_range &&
         upload.completion_event_recorded_after_copy &&
         upload.completion_event_observed && upload.upload_completed &&
         upload.verification_copy_enqueued_from_exact_payload_range &&
         upload.verification_event_recorded_after_copy &&
         upload.verification_event_observed &&
         upload.verification_completed &&
         upload.device_payload_matches_host_payload &&
         upload.allocation_retained_for_asset_lifetime;
}

[[nodiscard]] constexpr std::uint64_t expected_consumer_tactic_identity(
    const std::size_t layer_index, const Role role) noexcept {
  if (layer_index >= kSm87MacroFeedV4P40StartupPackageLayers) {
    return 0U;
  }
  if (role == Role::kNvFp4GateUp) {
    return static_cast<std::uint64_t>(
        kernels::kSm87MacroFeedV4NvFp4GateUpIdentity);
  }
  if (role == Role::kNvFp4Down) {
    return static_cast<std::uint64_t>(
        kernels::kSm87MacroFeedV4NvFp4DownIdentity);
  }
  const bool full_layer =
      sm87_target_aot_complete_is_full_layer(layer_index);
  if (role == kGdnQkvZRole) {
    return full_layer ? 0U
                      : static_cast<std::uint64_t>(kGdnQkvZTacticIdentity);
  }
  if (role == Role::kFp8FullQkv) {
    return full_layer ? static_cast<std::uint64_t>(kFullQkvTacticIdentity)
                      : 0U;
  }
  if (role == kAttentionOutputRole) {
    return static_cast<std::uint64_t>(
        full_layer ? kFullOutputTacticIdentity : kGdnOutputTacticIdentity);
  }
  return 0U;
}

[[nodiscard]] std::uint64_t gate_up_seal_identity(
    const Sm87MacroFeedV4GateUpStartupSeal& seal) noexcept {
  const auto& resources = seal.resources;
  if (seal.package_identity == 0U ||
      seal.deployment_plan_identity == 0U ||
      seal.binding_catalog_identity == 0U ||
      seal.binding_count != kSm87MacroFeedV4P40StartupPackageLayers ||
      !seal.canonical_c8000_plan || !seal.issued_by_v4_package ||
      seal.caller_receipt_accepted || seal.launcher_authority ||
      seal.production_dispatch_eligible ||
      !resources.static_resource_gate_passed ||
      !kernels::sm87_macrofeed_v4_nvfp4_gate_up_resource_gate(resources)) {
    return 0U;
  }
  std::uint64_t identity = 0x5133'4d46'5634'4753ULL;
  identity = mix(identity, seal.package_identity);
  identity = mix(identity, seal.deployment_plan_identity);
  identity = mix(identity, seal.binding_catalog_identity);
  identity = mix(identity, seal.binding_count);
  identity = mix(identity, static_cast<std::uint64_t>(resources.identity));
  identity = mix(identity,
                 static_cast<std::uint64_t>(resources.device_ordinal + 1));
  identity = mix(identity, static_cast<std::uint64_t>(resources.compute_major));
  identity = mix(identity, static_cast<std::uint64_t>(resources.compute_minor));
  identity = mix(identity, static_cast<std::uint64_t>(resources.sm_count));
  identity = mix(identity, static_cast<std::uint64_t>(resources.binary_version));
  identity = mix(identity,
                 static_cast<std::uint64_t>(resources.registers_per_thread));
  identity = mix(identity, resources.static_shared_bytes);
  identity = mix(identity, resources.dynamic_shared_bytes);
  identity = mix(identity, resources.local_bytes);
  identity = mix(
      identity, static_cast<std::uint64_t>(resources.maximum_threads_per_block));
  identity = mix(identity,
                 static_cast<std::uint64_t>(resources.active_blocks_per_sm));
  identity = mix(identity, resources.kernel_compiled);
  identity = mix(identity, resources.static_resource_gate_passed);
  identity = mix(identity, resources.numerical_contract_qualified);
  identity = mix(identity, resources.production_dispatch_eligible);
  identity = mix(identity, seal.canonical_c8000_plan);
  identity = mix(identity, seal.issued_by_v4_package);
  identity = mix(identity, seal.caller_receipt_accepted);
  identity = mix(identity, seal.launcher_authority);
  identity = mix(identity, seal.production_dispatch_eligible);
  return identity == 0U ? 0x5133'4d46'5634'4753ULL : identity;
}

[[nodiscard]] std::uint64_t down_seal_identity(
    const Sm87MacroFeedV4DownStartupSeal& seal) noexcept {
  const auto& resources = seal.resources;
  if (seal.package_identity == 0U ||
      seal.deployment_plan_identity == 0U ||
      seal.binding_catalog_identity == 0U ||
      seal.binding_count != kSm87MacroFeedV4P40StartupPackageLayers ||
      !seal.canonical_c8000_plan || !seal.issued_by_v4_package ||
      seal.caller_receipt_accepted || seal.launcher_authority ||
      seal.production_dispatch_eligible ||
      !resources.static_resource_gate_passed ||
      !kernels::sm87_macrofeed_v4_nvfp4_down_resource_gate(resources)) {
    return 0U;
  }
  std::uint64_t identity = 0x5133'4d46'5634'4453ULL;
  identity = mix(identity, seal.package_identity);
  identity = mix(identity, seal.deployment_plan_identity);
  identity = mix(identity, seal.binding_catalog_identity);
  identity = mix(identity, seal.binding_count);
  identity = mix(identity, static_cast<std::uint64_t>(resources.identity));
  identity = mix(identity,
                 static_cast<std::uint64_t>(resources.device_ordinal + 1));
  identity = mix(identity, static_cast<std::uint64_t>(resources.compute_major));
  identity = mix(identity, static_cast<std::uint64_t>(resources.compute_minor));
  identity = mix(identity, static_cast<std::uint64_t>(resources.sm_count));
  identity = mix(identity, static_cast<std::uint64_t>(resources.binary_version));
  identity = mix(identity,
                 static_cast<std::uint64_t>(resources.registers_per_thread));
  identity = mix(identity, resources.static_shared_bytes);
  identity = mix(identity, resources.dynamic_shared_bytes);
  identity = mix(identity, resources.local_bytes);
  identity = mix(
      identity, static_cast<std::uint64_t>(resources.maximum_threads_per_block));
  identity = mix(identity,
                 static_cast<std::uint64_t>(resources.active_blocks_per_sm));
  identity = mix(identity, resources.shared_bytes_per_sm);
  identity = mix(identity, resources.optin_shared_bytes_per_block);
  identity = mix(identity, resources.kernel_compiled);
  identity = mix(identity, resources.static_resource_gate_passed);
  identity = mix(identity, resources.numerical_contract_qualified);
  identity = mix(identity, resources.production_dispatch_eligible);
  identity = mix(identity, seal.canonical_c8000_plan);
  identity = mix(identity, seal.issued_by_v4_package);
  identity = mix(identity, seal.caller_receipt_accepted);
  identity = mix(identity, seal.launcher_authority);
  identity = mix(identity, seal.production_dispatch_eligible);
  return identity == 0U ? 0x5133'4d46'5634'4453ULL : identity;
}

[[nodiscard]] constexpr bool bf16_ab_resource_equal(
    const kernels::Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot& left,
    const kernels::Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot& right)
    noexcept {
  return left.identity == right.identity &&
         left.device_ordinal == right.device_ordinal &&
         left.compute_major == right.compute_major &&
         left.compute_minor == right.compute_minor &&
         left.sm_count == right.sm_count &&
         left.binary_version == right.binary_version &&
         left.registers_per_thread == right.registers_per_thread &&
         left.static_shared_bytes == right.static_shared_bytes &&
         left.dynamic_shared_bytes == right.dynamic_shared_bytes &&
         left.local_bytes == right.local_bytes &&
         left.maximum_threads_per_block == right.maximum_threads_per_block &&
         left.active_blocks_per_sm == right.active_blocks_per_sm &&
         left.threads_per_block == right.threads_per_block &&
         left.physical_grid_ctas == right.physical_grid_ctas &&
         left.kernel_compiled == right.kernel_compiled &&
         left.exact_geometry == right.exact_geometry &&
         left.static_resource_gate_passed ==
             right.static_resource_gate_passed &&
         left.numerical_contract_qualified ==
             right.numerical_contract_qualified &&
         left.production_dispatch_eligible ==
             right.production_dispatch_eligible &&
         left.startup_package_unbound == right.startup_package_unbound &&
         left.execution_capability == right.execution_capability &&
         left.caller_snapshot_grants_production_authority ==
             right.caller_snapshot_grants_production_authority;
}

[[nodiscard]] std::uint64_t mix_bf16_ab_resource(
    std::uint64_t identity,
    const kernels::Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot&
        resources) noexcept {
  identity = mix(identity, static_cast<std::uint64_t>(resources.identity));
  identity = mix(identity,
                 static_cast<std::uint64_t>(resources.device_ordinal + 1));
  identity = mix(identity, static_cast<std::uint64_t>(resources.compute_major));
  identity = mix(identity, static_cast<std::uint64_t>(resources.compute_minor));
  identity = mix(identity, static_cast<std::uint64_t>(resources.sm_count));
  identity = mix(identity, static_cast<std::uint64_t>(resources.binary_version));
  identity = mix(identity,
                 static_cast<std::uint64_t>(resources.registers_per_thread));
  identity = mix(identity, resources.static_shared_bytes);
  identity = mix(identity, resources.dynamic_shared_bytes);
  identity = mix(identity, resources.local_bytes);
  identity = mix(
      identity, static_cast<std::uint64_t>(resources.maximum_threads_per_block));
  identity = mix(identity,
                 static_cast<std::uint64_t>(resources.active_blocks_per_sm));
  identity = mix(identity,
                 static_cast<std::uint64_t>(resources.threads_per_block));
  identity = mix(identity,
                 static_cast<std::uint64_t>(resources.physical_grid_ctas));
  identity = mix(identity, resources.kernel_compiled);
  identity = mix(identity, resources.exact_geometry);
  identity = mix(identity, resources.static_resource_gate_passed);
  identity = mix(identity, resources.numerical_contract_qualified);
  identity = mix(identity, resources.production_dispatch_eligible);
  identity = mix(identity, resources.startup_package_unbound);
  identity = mix(identity, resources.execution_capability);
  identity = mix(identity,
                 resources.caller_snapshot_grants_production_authority);
  return identity;
}

[[nodiscard]] constexpr bool fp8_resource_equal(
    const kernels::Sm87MacroFeedV4Fp8CudaResources& left,
    const kernels::Sm87MacroFeedV4Fp8CudaResources& right) noexcept {
  return left.identity == right.identity && left.role == right.role &&
         left.input_layout == right.input_layout &&
         left.device_ordinal == right.device_ordinal &&
         left.compute_major == right.compute_major &&
         left.compute_minor == right.compute_minor &&
         left.sm_count == right.sm_count &&
         left.binary_version == right.binary_version &&
         left.registers_per_thread == right.registers_per_thread &&
         left.static_shared_bytes == right.static_shared_bytes &&
         left.dynamic_shared_bytes == right.dynamic_shared_bytes &&
         left.local_bytes == right.local_bytes &&
         left.maximum_threads_per_block == right.maximum_threads_per_block &&
         left.active_blocks_per_sm == right.active_blocks_per_sm &&
         left.shared_bytes_per_sm == right.shared_bytes_per_sm &&
         left.optin_shared_bytes_per_block ==
             right.optin_shared_bytes_per_block &&
         left.kernel_compiled == right.kernel_compiled &&
         left.static_resource_gate_passed ==
             right.static_resource_gate_passed &&
         left.numerical_contract_qualified ==
             right.numerical_contract_qualified &&
         left.production_dispatch_eligible ==
             right.production_dispatch_eligible;
}

[[nodiscard]] std::uint64_t mix_fp8_resource(
    std::uint64_t identity,
    const kernels::Sm87MacroFeedV4Fp8CudaResources& resources) noexcept {
  identity = mix(identity, static_cast<std::uint64_t>(resources.identity));
  identity = mix(identity, static_cast<std::uint64_t>(resources.role));
  identity = mix(identity,
                 static_cast<std::uint64_t>(resources.input_layout));
  identity = mix(identity,
                 static_cast<std::uint64_t>(resources.device_ordinal + 1));
  identity = mix(identity, static_cast<std::uint64_t>(resources.compute_major));
  identity = mix(identity, static_cast<std::uint64_t>(resources.compute_minor));
  identity = mix(identity, static_cast<std::uint64_t>(resources.sm_count));
  identity = mix(identity, static_cast<std::uint64_t>(resources.binary_version));
  identity = mix(identity,
                 static_cast<std::uint64_t>(resources.registers_per_thread));
  identity = mix(identity, resources.static_shared_bytes);
  identity = mix(identity, resources.dynamic_shared_bytes);
  identity = mix(identity, resources.local_bytes);
  identity = mix(
      identity, static_cast<std::uint64_t>(resources.maximum_threads_per_block));
  identity = mix(identity,
                 static_cast<std::uint64_t>(resources.active_blocks_per_sm));
  identity = mix(identity, resources.shared_bytes_per_sm);
  identity = mix(identity, resources.optin_shared_bytes_per_block);
  identity = mix(identity, resources.kernel_compiled);
  identity = mix(identity, resources.static_resource_gate_passed);
  identity = mix(identity, resources.numerical_contract_qualified);
  identity = mix(identity, resources.production_dispatch_eligible);
  return identity;
}

[[nodiscard]] constexpr bool full_preprocess_resource_equal(
    const kernels::Sm87MacroFeedV4FullAttentionPreprocessAdmissionResourceSnapshot&
        left,
    const kernels::Sm87MacroFeedV4FullAttentionPreprocessAdmissionResourceSnapshot&
        right) noexcept {
  return left.identity == right.identity &&
         left.device_ordinal == right.device_ordinal &&
         left.compute_major == right.compute_major &&
         left.compute_minor == right.compute_minor &&
         left.sm_count == right.sm_count &&
         left.binary_version == right.binary_version &&
         left.kernel.registers_per_thread ==
             right.kernel.registers_per_thread &&
         left.kernel.static_shared_bytes == right.kernel.static_shared_bytes &&
         left.kernel.local_bytes == right.kernel.local_bytes &&
         left.kernel.maximum_threads_per_block ==
             right.kernel.maximum_threads_per_block &&
         left.kernel.active_blocks_per_sm ==
             right.kernel.active_blocks_per_sm &&
         left.kernel.threads_per_block == right.kernel.threads_per_block &&
         left.kernel.grid_x == right.kernel.grid_x &&
         left.kernel.grid_y == right.kernel.grid_y &&
         left.kernel.physical_grid_ctas ==
             right.kernel.physical_grid_ctas &&
         left.kernel_compiled == right.kernel_compiled &&
         left.exact_geometry == right.exact_geometry &&
         left.static_resource_gate_passed ==
             right.static_resource_gate_passed &&
         left.numerical_contract_qualified ==
             right.numerical_contract_qualified &&
         left.production_dispatch_eligible ==
             right.production_dispatch_eligible &&
         left.startup_package_unbound == right.startup_package_unbound &&
         left.execution_capability == right.execution_capability &&
         left.caller_snapshot_grants_production_authority ==
             right.caller_snapshot_grants_production_authority;
}

[[nodiscard, maybe_unused]] std::uint64_t mix_full_preprocess_resource(
    std::uint64_t identity,
    const kernels::Sm87MacroFeedV4FullAttentionPreprocessAdmissionResourceSnapshot&
        resources) noexcept {
  identity = mix(identity, static_cast<std::uint64_t>(resources.identity));
  identity = mix(identity,
                 static_cast<std::uint64_t>(resources.device_ordinal + 1));
  identity = mix(identity, static_cast<std::uint64_t>(resources.compute_major));
  identity = mix(identity, static_cast<std::uint64_t>(resources.compute_minor));
  identity = mix(identity, static_cast<std::uint64_t>(resources.sm_count));
  identity = mix(identity, static_cast<std::uint64_t>(resources.binary_version));
  identity = mix(
      identity,
      static_cast<std::uint64_t>(resources.kernel.registers_per_thread));
  identity = mix(identity, resources.kernel.static_shared_bytes);
  identity = mix(identity, resources.kernel.local_bytes);
  identity = mix(identity, static_cast<std::uint64_t>(
                               resources.kernel.maximum_threads_per_block));
  identity = mix(identity, static_cast<std::uint64_t>(
                               resources.kernel.active_blocks_per_sm));
  identity = mix(identity, static_cast<std::uint64_t>(
                               resources.kernel.threads_per_block));
  identity = mix(identity,
                 static_cast<std::uint64_t>(resources.kernel.grid_x));
  identity = mix(identity,
                 static_cast<std::uint64_t>(resources.kernel.grid_y));
  identity = mix(identity, static_cast<std::uint64_t>(
                               resources.kernel.physical_grid_ctas));
  identity = mix(identity, resources.kernel_compiled);
  identity = mix(identity, resources.exact_geometry);
  identity = mix(identity, resources.static_resource_gate_passed);
  identity = mix(identity, resources.numerical_contract_qualified);
  identity = mix(identity, resources.production_dispatch_eligible);
  identity = mix(identity, resources.startup_package_unbound);
  identity = mix(identity, resources.execution_capability);
  identity = mix(
      identity, resources.caller_snapshot_grants_production_authority);
  return identity;
}

[[nodiscard]] constexpr bool full_attention_resource_equal(
    const kernels::Sm87MacroFeedV4AttentionC8000AdmissionResourceSnapshot&
        left,
    const kernels::Sm87MacroFeedV4AttentionC8000AdmissionResourceSnapshot&
        right) noexcept {
  return left.identity == right.identity &&
         left.device_ordinal == right.device_ordinal &&
         left.compute_major == right.compute_major &&
         left.compute_minor == right.compute_minor &&
         left.sm_count == right.sm_count &&
         left.binary_version == right.binary_version &&
         left.kernel.registers_per_thread ==
             right.kernel.registers_per_thread &&
         left.kernel.static_shared_bytes == right.kernel.static_shared_bytes &&
         left.kernel.dynamic_shared_bytes ==
             right.kernel.dynamic_shared_bytes &&
         left.kernel.local_bytes == right.kernel.local_bytes &&
         left.kernel.maximum_threads_per_block ==
             right.kernel.maximum_threads_per_block &&
         left.kernel.active_blocks_per_sm ==
             right.kernel.active_blocks_per_sm &&
         left.kernel.threads_per_block == right.kernel.threads_per_block &&
         left.kernel.grid_x == right.kernel.grid_x &&
         left.kernel.grid_y == right.kernel.grid_y &&
         left.kernel.grid_z == right.kernel.grid_z &&
         left.kernel.physical_grid_ctas ==
             right.kernel.physical_grid_ctas &&
         left.kernel_compiled == right.kernel_compiled &&
         left.exact_geometry == right.exact_geometry &&
         left.static_resource_gate_passed ==
             right.static_resource_gate_passed &&
         left.numerical_contract_qualified ==
             right.numerical_contract_qualified &&
         left.production_dispatch_eligible ==
             right.production_dispatch_eligible &&
         left.startup_package_unbound == right.startup_package_unbound &&
         left.execution_capability == right.execution_capability &&
         left.caller_snapshot_grants_production_authority ==
             right.caller_snapshot_grants_production_authority;
}

[[nodiscard, maybe_unused]] std::uint64_t mix_full_attention_resource(
    std::uint64_t identity,
    const kernels::Sm87MacroFeedV4AttentionC8000AdmissionResourceSnapshot&
        resources) noexcept {
  identity = mix(identity, static_cast<std::uint64_t>(resources.identity));
  identity = mix(identity,
                 static_cast<std::uint64_t>(resources.device_ordinal + 1));
  identity = mix(identity, static_cast<std::uint64_t>(resources.compute_major));
  identity = mix(identity, static_cast<std::uint64_t>(resources.compute_minor));
  identity = mix(identity, static_cast<std::uint64_t>(resources.sm_count));
  identity = mix(identity, static_cast<std::uint64_t>(resources.binary_version));
  identity = mix(
      identity,
      static_cast<std::uint64_t>(resources.kernel.registers_per_thread));
  identity = mix(identity, resources.kernel.static_shared_bytes);
  identity = mix(identity, resources.kernel.dynamic_shared_bytes);
  identity = mix(identity, resources.kernel.local_bytes);
  identity = mix(identity, static_cast<std::uint64_t>(
                               resources.kernel.maximum_threads_per_block));
  identity = mix(identity, static_cast<std::uint64_t>(
                               resources.kernel.active_blocks_per_sm));
  identity = mix(identity, static_cast<std::uint64_t>(
                               resources.kernel.threads_per_block));
  identity = mix(identity,
                 static_cast<std::uint64_t>(resources.kernel.grid_x));
  identity = mix(identity,
                 static_cast<std::uint64_t>(resources.kernel.grid_y));
  identity = mix(identity,
                 static_cast<std::uint64_t>(resources.kernel.grid_z));
  identity = mix(identity, static_cast<std::uint64_t>(
                               resources.kernel.physical_grid_ctas));
  identity = mix(identity, resources.kernel_compiled);
  identity = mix(identity, resources.exact_geometry);
  identity = mix(identity, resources.static_resource_gate_passed);
  identity = mix(identity, resources.numerical_contract_qualified);
  identity = mix(identity, resources.production_dispatch_eligible);
  identity = mix(identity, resources.startup_package_unbound);
  identity = mix(identity, resources.execution_capability);
  identity = mix(
      identity, resources.caller_snapshot_grants_production_authority);
  return identity;
}

[[nodiscard]] constexpr bool request_boundary_fixed_resource_gate(
    const Bound::Sm87MacroFeedV4FixedKernelResource& resource,
    const std::size_t static_shared_bytes, const std::int32_t threads,
    const std::int32_t grid_ctas, const std::int32_t minimum_active_blocks,
    const std::int32_t exact_registers = 0,
    const std::int32_t exact_maximum_threads = 0,
    const std::int32_t exact_active_blocks = 0) noexcept {
  return resource.registers_per_thread > 0 &&
         (exact_registers == 0 ||
          resource.registers_per_thread == exact_registers) &&
         resource.static_shared_bytes == static_shared_bytes &&
         resource.local_bytes == 0U &&
         resource.maximum_threads_per_block >= threads &&
         (exact_maximum_threads == 0 ||
          resource.maximum_threads_per_block == exact_maximum_threads) &&
         resource.active_blocks_per_sm >= minimum_active_blocks &&
         (exact_active_blocks == 0 ||
          resource.active_blocks_per_sm == exact_active_blocks) &&
         resource.binary_version == 87 && resource.threads == threads &&
         resource.grid_ctas == grid_ctas;
}

[[nodiscard]] constexpr bool request_boundary_common_resource_gate(
    const std::uint64_t identity, const std::uint64_t expected_identity,
    const std::int32_t device_ordinal, const std::int32_t compute_major,
    const std::int32_t compute_minor, const std::int32_t sm_count,
    const bool exact_geometry,
    const bool static_resource_gate_passed) noexcept {
  return identity == expected_identity && device_ordinal >= 0 &&
         compute_major == 8 && compute_minor == 7 && sm_count == 16 &&
         exact_geometry && static_resource_gate_passed;
}

[[nodiscard]] constexpr bool request_boundary_embedding_resource_gate(
    const Bound::Sm87MacroFeedV4EmbeddingC8000ResourceSnapshot& resources)
    noexcept {
  return request_boundary_common_resource_gate(
             resources.identity, kEmbeddingC8000ResourceIdentity,
             resources.device_ordinal, resources.compute_major,
             resources.compute_minor, resources.sm_count,
             resources.exact_geometry, resources.static_resource_gate_passed) &&
         request_boundary_fixed_resource_gate(
             resources.gather, 0U, 256, 8'000, 1);
}

[[nodiscard]] constexpr bool request_boundary_final_norm_resource_gate(
    const Bound::Sm87MacroFeedV4FinalNormM1ResourceSnapshot& resources)
    noexcept {
  return request_boundary_common_resource_gate(
             resources.identity, kFinalNormM1ResourceIdentity,
             resources.device_ordinal, resources.compute_major,
             resources.compute_minor, resources.sm_count,
             resources.exact_geometry, resources.static_resource_gate_passed) &&
         request_boundary_fixed_resource_gate(
             resources.centered_rms_norm, 1'024U, 256, 1, 1);
}

[[nodiscard]] constexpr bool request_boundary_lm_head_resource_gate(
    const Bound::Sm87MacroFeedV4LmHeadM1ResourceSnapshot& resources)
    noexcept {
  return request_boundary_common_resource_gate(
             resources.identity, kLmHeadM1ResourceIdentity,
             resources.device_ordinal, resources.compute_major,
             resources.compute_minor, resources.sm_count,
             resources.exact_geometry, resources.static_resource_gate_passed) &&
         request_boundary_fixed_resource_gate(
             resources.activation_staged, 11'328U, 256, 64, 4, 64, 256, 4);
}

[[nodiscard]] constexpr bool request_boundary_greedy_resource_gate(
    const Bound::Sm87MacroFeedV4GreedyArgmaxM1ResourceSnapshot& resources)
    noexcept {
  return request_boundary_common_resource_gate(
             resources.identity, kGreedyArgmaxM1ResourceIdentity,
             resources.device_ordinal, resources.compute_major,
             resources.compute_minor, resources.sm_count,
             resources.exact_geometry, resources.static_resource_gate_passed) &&
         request_boundary_fixed_resource_gate(
             resources.partial, 3'072U, 256, 32, 1) &&
         request_boundary_fixed_resource_gate(
             resources.finalize, 0U, 32, 1, 1);
}

[[nodiscard]] std::uint64_t mix_request_boundary_fixed_resource(
    std::uint64_t identity,
    const Bound::Sm87MacroFeedV4FixedKernelResource& resource) noexcept {
  identity = mix(
      identity, static_cast<std::uint64_t>(resource.registers_per_thread));
  identity = mix(identity, resource.static_shared_bytes);
  identity = mix(identity, resource.local_bytes);
  identity = mix(
      identity,
      static_cast<std::uint64_t>(resource.maximum_threads_per_block));
  identity = mix(
      identity, static_cast<std::uint64_t>(resource.active_blocks_per_sm));
  identity = mix(identity,
                 static_cast<std::uint64_t>(resource.binary_version));
  identity = mix(identity, static_cast<std::uint64_t>(resource.threads));
  identity = mix(identity, static_cast<std::uint64_t>(resource.grid_ctas));
  return identity;
}

template <typename Snapshot>
[[nodiscard]] std::uint64_t mix_request_boundary_snapshot_common(
    std::uint64_t identity, const Snapshot& resources) noexcept {
  identity = mix(identity, resources.identity);
  identity = mix(identity,
                 static_cast<std::uint64_t>(resources.device_ordinal + 1));
  identity = mix(identity,
                 static_cast<std::uint64_t>(resources.compute_major));
  identity = mix(identity,
                 static_cast<std::uint64_t>(resources.compute_minor));
  identity = mix(identity, static_cast<std::uint64_t>(resources.sm_count));
  identity = mix(identity, resources.exact_geometry);
  identity = mix(identity, resources.static_resource_gate_passed);
  return identity;
}

[[nodiscard]] std::uint64_t mix_request_boundary_embedding_resource(
    std::uint64_t identity,
    const Bound::Sm87MacroFeedV4EmbeddingC8000ResourceSnapshot& resources)
    noexcept {
  identity = mix_request_boundary_snapshot_common(identity, resources);
  return mix_request_boundary_fixed_resource(identity, resources.gather);
}

[[nodiscard]] std::uint64_t mix_request_boundary_final_norm_resource(
    std::uint64_t identity,
    const Bound::Sm87MacroFeedV4FinalNormM1ResourceSnapshot& resources)
    noexcept {
  identity = mix_request_boundary_snapshot_common(identity, resources);
  return mix_request_boundary_fixed_resource(identity,
                                             resources.centered_rms_norm);
}

[[nodiscard]] std::uint64_t mix_request_boundary_lm_head_resource(
    std::uint64_t identity,
    const Bound::Sm87MacroFeedV4LmHeadM1ResourceSnapshot& resources)
    noexcept {
  identity = mix_request_boundary_snapshot_common(identity, resources);
  return mix_request_boundary_fixed_resource(identity,
                                             resources.activation_staged);
}

[[nodiscard]] std::uint64_t mix_request_boundary_greedy_resource(
    std::uint64_t identity,
    const Bound::Sm87MacroFeedV4GreedyArgmaxM1ResourceSnapshot& resources)
    noexcept {
  identity = mix_request_boundary_snapshot_common(identity, resources);
  identity = mix_request_boundary_fixed_resource(identity, resources.partial);
  return mix_request_boundary_fixed_resource(identity, resources.finalize);
}

#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_STARTUP_PACKAGE_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_MACROFEED_V4_BF16_AB_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_MACROFEED_V4_FP8_ADMISSION)
[[nodiscard]] bool live_current_device_allocation_range(
    const void* const pointer, const std::uint64_t bytes,
    const std::int32_t expected_device, int* const cuda_error,
    const std::uintptr_t expected_allocation_begin = 0U,
    const std::uintptr_t expected_allocation_end = 0U,
    const std::uint64_t expected_allocation_bytes = 0U) noexcept {
  if (cuda_error != nullptr) {
    *cuda_error = 0;
  }
  const bool exact_allocation_required =
      expected_allocation_begin != 0U || expected_allocation_end != 0U ||
      expected_allocation_bytes != 0U;
  if (pointer == nullptr || bytes == 0U || expected_device < 0 ||
      (exact_allocation_required &&
       (expected_allocation_begin == 0U ||
        expected_allocation_end <= expected_allocation_begin ||
        expected_allocation_bytes == 0U ||
        expected_allocation_end - expected_allocation_begin !=
            expected_allocation_bytes)) ||
      reinterpret_cast<std::uintptr_t>(pointer) % 16U != 0U) {
    if (cuda_error != nullptr) {
      *cuda_error = static_cast<int>(cudaErrorInvalidValue);
    }
    return false;
  }

  int current_device = -1;
  cudaError_t runtime_status = cudaGetDevice(&current_device);
  if (runtime_status != cudaSuccess || current_device != expected_device) {
    if (cuda_error != nullptr) {
      *cuda_error = runtime_status == cudaSuccess
                        ? static_cast<int>(cudaErrorInvalidDevice)
                        : static_cast<int>(runtime_status);
    }
    return false;
  }

  const std::uintptr_t range_begin =
      reinterpret_cast<std::uintptr_t>(pointer);
  if (bytes > std::numeric_limits<std::uintptr_t>::max() - range_begin) {
    if (cuda_error != nullptr) {
      *cuda_error = static_cast<int>(cudaErrorInvalidValue);
    }
    return false;
  }
  const kernels::Sm87MacroFeedV4Bf16AbByteRange range{
      range_begin, range_begin + static_cast<std::uintptr_t>(bytes), true};

  cudaPointerAttributes begin_attributes{};
  cudaPointerAttributes end_attributes{};
  runtime_status = cudaPointerGetAttributes(&begin_attributes, pointer);
  if (runtime_status != cudaSuccess) {
    if (cuda_error != nullptr) {
      *cuda_error = static_cast<int>(runtime_status);
    }
    return false;
  }
  const void* const last_byte =
      reinterpret_cast<const void*>(range.end - 1U);
  runtime_status = cudaPointerGetAttributes(&end_attributes, last_byte);
  if (runtime_status != cudaSuccess ||
      begin_attributes.type != cudaMemoryTypeDevice ||
      end_attributes.type != cudaMemoryTypeDevice ||
      begin_attributes.device != expected_device ||
      end_attributes.device != expected_device) {
    if (cuda_error != nullptr) {
      *cuda_error = runtime_status == cudaSuccess
                        ? static_cast<int>(cudaErrorInvalidDevicePointer)
                        : static_cast<int>(runtime_status);
    }
    return false;
  }

  CUdeviceptr allocation_begin = 0U;
  std::size_t allocation_bytes = 0U;
  const CUresult driver_status = cuMemGetAddressRange(
      &allocation_begin, &allocation_bytes,
      static_cast<CUdeviceptr>(range.begin));
  const std::uintptr_t allocation_start =
      static_cast<std::uintptr_t>(allocation_begin);
  const bool allocation_end_overflows =
      allocation_bytes >
      std::numeric_limits<std::uintptr_t>::max() - allocation_start;
  const std::uintptr_t allocation_end =
      allocation_end_overflows ? 0U : allocation_start + allocation_bytes;
  if (driver_status != CUDA_SUCCESS || allocation_begin == 0U ||
      allocation_bytes == 0U || allocation_end_overflows ||
      range.begin < allocation_start || range.end > allocation_end ||
      (exact_allocation_required &&
       (allocation_start != expected_allocation_begin ||
        allocation_end != expected_allocation_end ||
        allocation_bytes != expected_allocation_bytes))) {
    if (cuda_error != nullptr) {
      *cuda_error = driver_status == CUDA_SUCCESS
                        ? static_cast<int>(cudaErrorInvalidDevicePointer)
                        : static_cast<int>(driver_status);
    }
    return false;
  }

  CUdeviceptr end_allocation_begin = 0U;
  std::size_t end_allocation_bytes = 0U;
  const CUresult end_driver_status = cuMemGetAddressRange(
      &end_allocation_begin, &end_allocation_bytes,
      static_cast<CUdeviceptr>(range.end - 1U));
  if (end_driver_status != CUDA_SUCCESS ||
      end_allocation_begin != allocation_begin ||
      end_allocation_bytes != allocation_bytes) {
    if (cuda_error != nullptr) {
      *cuda_error = end_driver_status == CUDA_SUCCESS
                        ? static_cast<int>(cudaErrorInvalidDevicePointer)
                        : static_cast<int>(end_driver_status);
    }
    return false;
  }
  return true;
}

[[nodiscard]] bool resident_tensor_view_matches(
    const ResidentWeights& resident, const std::string_view name,
    const void* const expected_pointer,
    const io::safetensors::DType expected_dtype,
    const std::uint64_t expected_bytes, const std::uint64_t first_dimension,
    const std::uint64_t second_dimension,
    const std::size_t expected_rank) noexcept {
  const DeviceTensorView* const view = resident.find(name);
  if (view == nullptr || view->device_data != expected_pointer ||
      view->dtype != expected_dtype || view->byte_size != expected_bytes ||
      view->shape.size() != expected_rank) {
    return false;
  }
  if (expected_rank >= 1U && view->shape[0U] != first_dimension) {
    return false;
  }
  return expected_rank < 2U || view->shape[1U] == second_dimension;
}
#endif

[[nodiscard]] std::uint64_t bf16_ab_seal_identity(
    const Sm87MacroFeedV4Bf16AbStartupSeal& seal) noexcept {
  if (seal.package_identity == 0U ||
      seal.deployment_plan_identity == 0U ||
      seal.binding_catalog_identity == 0U ||
      seal.tensor_count !=
          kSm87MacroFeedV4P40StartupPackageBf16AbTensors ||
      seal.pair_count != kSm87MacroFeedV4P40StartupPackageBf16AbPairs ||
      !seal.canonical_natural_layer_order ||
      !seal.canonical_a_then_b_role_order ||
      !seal.complete_live_device_ranges || !seal.issued_by_v4_package ||
      seal.caller_resource_snapshot_accepted || seal.raw_pointer_exposed ||
      seal.launcher_authority || seal.production_dispatch_eligible ||
      !kernels::sm87_macrofeed_v4_bf16_ab_admission_resource_gate(
          seal.resources)) {
    return 0U;
  }
  std::uint64_t identity = 0x5133'4d46'5634'4253ULL;
  identity = mix(identity, seal.package_identity);
  identity = mix(identity, seal.deployment_plan_identity);
  identity = mix(identity, seal.binding_catalog_identity);
  identity = mix(identity, seal.tensor_count);
  identity = mix(identity, seal.pair_count);
  identity = mix_bf16_ab_resource(identity, seal.resources);
  identity = mix(identity, seal.canonical_natural_layer_order);
  identity = mix(identity, seal.canonical_a_then_b_role_order);
  identity = mix(identity, seal.complete_live_device_ranges);
  identity = mix(identity, seal.issued_by_v4_package);
  identity = mix(identity, seal.caller_resource_snapshot_accepted);
  identity = mix(identity, seal.raw_pointer_exposed);
  identity = mix(identity, seal.launcher_authority);
  identity = mix(identity, seal.production_dispatch_eligible);
  return identity == 0U ? 0x5133'4d46'5634'4253ULL : identity;
}

[[nodiscard]] std::uint64_t gdn_qkvz_seal_identity(
    const Sm87MacroFeedV4GdnQkvZStartupSeal& seal) noexcept {
  if (seal.package_identity == 0U ||
      seal.deployment_plan_identity == 0U ||
      seal.binding_catalog_identity == 0U ||
      seal.binding_count != kSm87MacroFeedV4P40StartupPackageGdnLayers ||
      seal.role != kGdnQkvZRole ||
      seal.input_layout != kGdnQkvZInputLayout ||
      seal.tactic_identity != kGdnQkvZTacticIdentity ||
      seal.resources.role != kGdnQkvZRole ||
      seal.resources.input_layout != kGdnQkvZInputLayout ||
      seal.resources.identity != kGdnQkvZTacticIdentity ||
      !seal.resources.static_resource_gate_passed ||
      !kernels::sm87_macrofeed_v4_fp8_resource_gate(seal.resources) ||
      seal.output_role != kAttentionOutputRole ||
      seal.output_input_layout != kGdnOutputInputLayout ||
      seal.output_tactic_identity != kGdnOutputTacticIdentity ||
      seal.output_resources.role != kAttentionOutputRole ||
      seal.output_resources.input_layout != kGdnOutputInputLayout ||
      seal.output_resources.identity != kGdnOutputTacticIdentity ||
      !seal.output_resources.static_resource_gate_passed ||
      !kernels::sm87_macrofeed_v4_fp8_resource_gate(
          seal.output_resources) ||
      !seal.canonical_natural_gdn_layer_order ||
      !seal.role_layout_and_tactic_fixed ||
      !seal.output_role_layout_and_tactic_fixed ||
      !seal.continuation_weights_execution_seal_required ||
      !seal.typed_asset_values_private ||
      seal.caller_resource_snapshot_accepted || seal.raw_pointer_exposed ||
      seal.launcher_authority || seal.production_dispatch_eligible) {
    return 0U;
  }
  std::uint64_t identity = 0x5133'4d46'5634'4653ULL;
  identity = mix(identity, seal.package_identity);
  identity = mix(identity, seal.deployment_plan_identity);
  identity = mix(identity, seal.binding_catalog_identity);
  identity = mix(identity, seal.binding_count);
  identity = mix(identity, static_cast<std::uint64_t>(seal.role));
  identity = mix(identity, static_cast<std::uint64_t>(seal.input_layout));
  identity = mix(identity, static_cast<std::uint64_t>(seal.tactic_identity));
  identity = mix_fp8_resource(identity, seal.resources);
  identity = mix(identity, static_cast<std::uint64_t>(seal.output_role));
  identity = mix(identity,
                 static_cast<std::uint64_t>(seal.output_input_layout));
  identity = mix(identity,
                 static_cast<std::uint64_t>(seal.output_tactic_identity));
  identity = mix_fp8_resource(identity, seal.output_resources);
  identity = mix(identity, seal.canonical_natural_gdn_layer_order);
  identity = mix(identity, seal.role_layout_and_tactic_fixed);
  identity = mix(identity, seal.output_role_layout_and_tactic_fixed);
  identity = mix(identity,
                 seal.continuation_weights_execution_seal_required);
  identity = mix(identity, seal.typed_asset_values_private);
  identity = mix(identity, seal.caller_resource_snapshot_accepted);
  identity = mix(identity, seal.raw_pointer_exposed);
  identity = mix(identity, seal.launcher_authority);
  identity = mix(identity, seal.production_dispatch_eligible);
  return identity == 0U ? 0x5133'4d46'5634'4653ULL : identity;
}

[[nodiscard]] std::uint64_t full_attention_source_seal_identity(
    const Sm87MacroFeedV4FullAttentionStartupSeal& seal) noexcept {
  if (seal.package_identity == 0U ||
      seal.deployment_plan_identity == 0U ||
      seal.source_catalog_identity == 0U ||
      seal.binding_count != kSm87MacroFeedV4P40StartupPackageFullLayers ||
      seal.qkv_role != kFullQkvRole ||
      seal.qkv_input_layout != kFullQkvInputLayout ||
      seal.qkv_tactic_identity != kFullQkvTacticIdentity ||
      seal.output_role != kAttentionOutputRole ||
      seal.output_input_layout != kFullOutputInputLayout ||
      seal.output_tactic_identity != kFullOutputTacticIdentity ||
      seal.q_norm_elements !=
          kernels::kSm87MacroFeedV4FullAttentionPreprocessHeadDimension ||
      seal.k_norm_elements !=
          kernels::kSm87MacroFeedV4FullAttentionPreprocessHeadDimension ||
      seal.norm_weight_bytes !=
          kernels::kSm87MacroFeedV4FullAttentionPreprocessNormWeightBytes ||
      !seal.canonical_natural_full_layer_order ||
      !seal.role_layout_and_tactics_fixed ||
      !seal.qk_norm_exact_shapes || !seal.qk_norm_live_device_ranges ||
      !seal.typed_asset_values_private ||
      !seal.observed_resource_execution_seal_deferred ||
      seal.caller_resource_snapshot_accepted || seal.raw_pointer_exposed ||
      seal.launcher_authority || seal.production_dispatch_eligible) {
    return 0U;
  }
  std::uint64_t identity = 0x5133'4d46'4655'4c53ULL;
  identity = mix(identity, seal.package_identity);
  identity = mix(identity, seal.deployment_plan_identity);
  identity = mix(identity, seal.source_catalog_identity);
  identity = mix(identity, seal.binding_count);
  identity = mix(identity, static_cast<std::uint64_t>(seal.qkv_role));
  identity = mix(identity, static_cast<std::uint64_t>(seal.qkv_input_layout));
  identity = mix(identity, static_cast<std::uint64_t>(seal.qkv_tactic_identity));
  identity = mix(identity, static_cast<std::uint64_t>(seal.output_role));
  identity = mix(identity,
                 static_cast<std::uint64_t>(seal.output_input_layout));
  identity = mix(identity,
                 static_cast<std::uint64_t>(seal.output_tactic_identity));
  identity = mix(identity, seal.q_norm_elements);
  identity = mix(identity, seal.k_norm_elements);
  identity = mix(identity, seal.norm_weight_bytes);
  identity = mix(identity, seal.canonical_natural_full_layer_order);
  identity = mix(identity, seal.role_layout_and_tactics_fixed);
  identity = mix(identity, seal.qk_norm_exact_shapes);
  identity = mix(identity, seal.qk_norm_live_device_ranges);
  identity = mix(identity, seal.typed_asset_values_private);
  identity = mix(identity, seal.observed_resource_execution_seal_deferred);
  identity = mix(identity, seal.caller_resource_snapshot_accepted);
  identity = mix(identity, seal.raw_pointer_exposed);
  identity = mix(identity, seal.launcher_authority);
  identity = mix(identity, seal.production_dispatch_eligible);
  return identity == 0U ? 0x5133'4d46'4655'4c53ULL : identity;
}

[[nodiscard]] std::uint64_t request_boundary_source_seal_identity(
    const Sm87MacroFeedV4RequestBoundaryStartupSeal& seal) noexcept {
  const bool exact_resident_authority =
      seal.normal_resident_authority != seal.host_test_resident_authority;
  if (seal.package_identity == 0U ||
      seal.deployment_plan_identity == 0U ||
      seal.source_catalog_identity == 0U ||
      seal.resident_root_identity == 0U || seal.resident_arena_bytes == 0U ||
      seal.binding_count !=
          kSm87MacroFeedV4P40StartupPackageRequestBoundaryBindings ||
      seal.device_ordinal < 0 ||
      seal.final_norm_epsilon_fp32_bits !=
          Bound::kSm87MacroFeedV4FinalNormEpsilonFp32Bits ||
      !kernels::sm87_target_aot_projection_scale_bits_valid(
          seal.weight_scale_2_fp32_bits) ||
      !kernels::sm87_target_aot_projection_scale_bits_valid(
          seal.input_scale_fp32_bits) ||
      !seal.embedding_exact_bf16_shape ||
      !seal.final_norm_exact_bf16_shape_and_epsilon ||
      !seal.lm_head_exact_canonical_nvfp4_shape ||
      !seal.device_scale_raw_bits_match_host ||
      !seal.input_scale_provenance_retained || seal.input_scale_consumed ||
      !seal.greedy_spec_exact || !seal.complete_live_device_ranges ||
      !seal.observed_resource_execution_seal_deferred ||
      !seal.final_representation_ready_diagnostic_only ||
      !seal.pure_prefill_state_committed_endpoint_unchanged ||
      !exact_resident_authority ||
      (seal.normal_resident_authority &&
       seal.resident_arena_bytes != kPinnedQwen36_27BArenaBytes) ||
      !seal.issued_by_v4_package || seal.caller_resource_snapshot_accepted ||
      seal.raw_pointer_exposed || seal.launcher_authority ||
      seal.production_dispatch_eligible) {
    return 0U;
  }
  std::uint64_t identity = 0x5133'4d46'5242'5353ULL;
  identity = mix(identity, seal.package_identity);
  identity = mix(identity, seal.deployment_plan_identity);
  identity = mix(identity, seal.source_catalog_identity);
  identity = mix(identity, seal.resident_root_identity);
  identity = mix(identity, seal.resident_arena_bytes);
  identity = mix(identity, seal.binding_count);
  identity = mix(identity,
                 static_cast<std::uint64_t>(seal.device_ordinal + 1));
  identity = mix(identity, seal.final_norm_epsilon_fp32_bits);
  identity = mix(identity, seal.weight_scale_2_fp32_bits);
  identity = mix(identity, seal.input_scale_fp32_bits);
  identity = mix(identity, seal.embedding_exact_bf16_shape);
  identity = mix(identity, seal.final_norm_exact_bf16_shape_and_epsilon);
  identity = mix(identity, seal.lm_head_exact_canonical_nvfp4_shape);
  identity = mix(identity, seal.device_scale_raw_bits_match_host);
  identity = mix(identity, seal.input_scale_provenance_retained);
  identity = mix(identity, seal.input_scale_consumed);
  identity = mix(identity, seal.greedy_spec_exact);
  identity = mix(identity, seal.complete_live_device_ranges);
  identity = mix(identity, seal.observed_resource_execution_seal_deferred);
  identity = mix(identity,
                 seal.final_representation_ready_diagnostic_only);
  identity = mix(identity,
                 seal.pure_prefill_state_committed_endpoint_unchanged);
  identity = mix(identity, seal.normal_resident_authority);
  identity = mix(identity, seal.host_test_resident_authority);
  identity = mix(identity, seal.issued_by_v4_package);
  identity = mix(identity, seal.caller_resource_snapshot_accepted);
  identity = mix(identity, seal.raw_pointer_exposed);
  identity = mix(identity, seal.launcher_authority);
  identity = mix(identity, seal.production_dispatch_eligible);
  return identity == 0U ? 0x5133'4d46'5242'5353ULL : identity;
}

}  // namespace

Sm87MacroFeedV4Bf16AbT0InventoryAudit
inspect_sm87_macrofeed_v4_bf16_ab_t0_inventory(
    const Sm87MacroFeedV4Bf16AbT0TensorDescriptor* const tensors,
    const std::size_t tensor_count) noexcept {
  Sm87MacroFeedV4Bf16AbT0InventoryAudit audit;
  audit.live_cuda_device_ranges_validated = false;
  audit.execution_capability = false;
  if (tensors == nullptr ||
      tensor_count != kSm87MacroFeedV4P40StartupPackageBf16AbTensors) {
    audit.failure_index = 0U;
    return audit;
  }

  std::array<std::uintptr_t,
             kSm87MacroFeedV4P40StartupPackageBf16AbTensors>
      begins{};
  std::array<std::uintptr_t,
             kSm87MacroFeedV4P40StartupPackageBf16AbTensors>
      ends{};
  std::uint64_t identity = 0x5133'4d46'4142'4341ULL;
  for (std::size_t index = 0U; index < tensor_count; ++index) {
    const auto& tensor = tensors[index];
    const std::size_t ordinal = index / 2U;
    const auto role = index % 2U == 0U
                          ? Sm87MacroFeedV4Bf16AbWeightRole::kA
                          : Sm87MacroFeedV4Bf16AbWeightRole::kB;
    if (tensor.gdn_ordinal != ordinal ||
        tensor.model_layer != gdn_model_layer(ordinal) ||
        !model_layer_is_gdn(tensor.model_layer)) {
      audit.failure_index = index;
      return audit;
    }
    if (tensor.role != role) {
      audit.canonical_natural_layer_order = true;
      audit.failure_index = index;
      return audit;
    }
    if (tensor.weight_kind != LinearWeightKind::kBf16 ||
        tensor.output_size !=
            kernels::kSm87MacroFeedV4Bf16AbRowsPerProjection ||
        tensor.input_size !=
            kernels::kSm87MacroFeedV4Bf16AbInputFeatures) {
      audit.canonical_natural_layer_order = true;
      audit.canonical_a_then_b_role_order = true;
      audit.failure_index = index;
      return audit;
    }
    if (!t0_range(tensor, &begins[index], &ends[index])) {
      audit.canonical_natural_layer_order = true;
      audit.canonical_a_then_b_role_order = true;
      audit.exact_bf16_shapes = true;
      audit.failure_index = index;
      return audit;
    }
    for (std::size_t earlier = 0U; earlier < index; ++earlier) {
      if (!(ends[earlier] <= begins[index] ||
            ends[index] <= begins[earlier])) {
        audit.canonical_natural_layer_order = true;
        audit.canonical_a_then_b_role_order = true;
        audit.exact_bf16_shapes = true;
        audit.failure_index = index;
        return audit;
      }
    }
    identity = mix(identity, index + 1U);
    identity = mix(identity, ordinal + 1U);
    identity = mix(identity, tensor.model_layer + 1U);
    identity = mix(identity, static_cast<std::uint64_t>(tensor.role));
    identity = mix(identity, static_cast<std::uint64_t>(tensor.weight_kind));
    identity = mix(identity, begins[index]);
    identity = mix(identity, ends[index]);
    identity = mix(identity, tensor.output_size);
    identity = mix(identity, tensor.input_size);
  }
  audit.catalog_identity =
      identity == 0U ? 0x5133'4d46'4142'4341ULL : identity;
  audit.tensors = tensor_count;
  audit.pairs = tensor_count / 2U;
  audit.failure_index = tensor_count;
  audit.canonical_natural_layer_order = true;
  audit.canonical_a_then_b_role_order = true;
  audit.exact_bf16_shapes = true;
  audit.nonnull_16b_aligned_disjoint_ranges = true;
  return audit;
}

Sm87MacroFeedV4RequestBoundaryT0SourceAudit
inspect_sm87_macrofeed_v4_request_boundary_t0_source(
    const Sm87MacroFeedV4RequestBoundaryT0SourceDescriptor& source) noexcept {
  Sm87MacroFeedV4RequestBoundaryT0SourceAudit audit;
  audit.live_cuda_device_ranges_validated = false;
  audit.execution_capability = false;

  std::array<RequestBoundaryT0Range, 6U> ranges{};
  if (source.embedding.output_size != Bound::kSm87MacroFeedV4EmbeddingVocabulary ||
      source.embedding.input_size != Bound::kSm87MacroFeedV4Hidden ||
      !request_boundary_t0_range(
          source.embedding.weight,
          Bound::kSm87MacroFeedV4EmbeddingTableBytes, &ranges[0U])) {
    audit.failure_index = 0U;
    return audit;
  }
  audit.embedding_exact_bf16_shape = true;
  std::uint64_t embedding_identity = 0x5133'4d46'5242'454dULL;
  embedding_identity = mix(embedding_identity, ranges[0U].begin);
  embedding_identity = mix(embedding_identity, ranges[0U].end);
  embedding_identity = mix(embedding_identity, source.embedding.output_size);
  embedding_identity = mix(embedding_identity, source.embedding.input_size);
  embedding_identity =
      mix(embedding_identity, Bound::kSm87MacroFeedV4EmbeddingTableBytes);
  audit.embedding_identity = embedding_identity;

  if (source.final_norm.element_count != Bound::kSm87MacroFeedV4Hidden ||
      source.final_norm_epsilon_fp32_bits !=
          Bound::kSm87MacroFeedV4FinalNormEpsilonFp32Bits ||
      !request_boundary_t0_range(
          source.final_norm.data, Bound::kSm87MacroFeedV4FinalNormBytes,
          &ranges[1U])) {
    audit.failure_index = 1U;
    return audit;
  }
  audit.final_norm_exact_bf16_shape_and_epsilon = true;
  std::uint64_t final_norm_identity = 0x5133'4d46'5242'464eULL;
  final_norm_identity = mix(final_norm_identity, ranges[1U].begin);
  final_norm_identity = mix(final_norm_identity, ranges[1U].end);
  final_norm_identity = mix(final_norm_identity, source.final_norm.element_count);
  final_norm_identity =
      mix(final_norm_identity, source.final_norm_epsilon_fp32_bits);
  audit.final_norm_identity = final_norm_identity;

  const auto* const lm_head = std::get_if<NvFp4LinearWeight>(&source.lm_head);
  if (lm_head == nullptr ||
      lm_head->output_size != Bound::kSm87MacroFeedV4LmHeadRows ||
      lm_head->input_size != Bound::kSm87MacroFeedV4LmHeadColumns ||
      !request_boundary_t0_range(
          lm_head->packed_weight,
          Bound::kSm87MacroFeedV4LmHeadPackedWeightBytes, &ranges[2U]) ||
      !request_boundary_t0_range(
          lm_head->block_scale,
          Bound::kSm87MacroFeedV4LmHeadBlockScaleBytes, &ranges[3U]) ||
      !request_boundary_t0_range(lm_head->weight_scale_2_device,
                                 sizeof(float), &ranges[4U]) ||
      !request_boundary_t0_range(lm_head->input_scale_device,
                                 sizeof(float), &ranges[5U])) {
    audit.failure_index = 2U;
    return audit;
  }
  audit.lm_head_exact_canonical_nvfp4_shape = true;

  audit.weight_scale_2_fp32_bits = fp32_bits(lm_head->weight_scale_2);
  if (!kernels::sm87_target_aot_projection_scale_bits_valid(
          audit.weight_scale_2_fp32_bits)) {
    audit.weight_scale_2_fp32_bits = 0U;
    audit.failure_index = 3U;
    return audit;
  }
  audit.input_scale_fp32_bits = fp32_bits(lm_head->input_scale);
  if (!kernels::sm87_target_aot_projection_scale_bits_valid(
          audit.input_scale_fp32_bits)) {
    audit.input_scale_fp32_bits = 0U;
    audit.failure_index = 4U;
    return audit;
  }
  audit.lm_head_scale_raw_bits_valid = true;
  audit.input_scale_provenance_retained = true;
  audit.input_scale_consumed = false;

  if (source.greedy_vocabulary != Bound::kSm87MacroFeedV4Vocabulary ||
      source.greedy_workspace_results !=
          Bound::kSm87MacroFeedV4GreedyWorkspaceResults ||
      !source.greedy_strict_left_to_right_fp32_order ||
      !source.greedy_smallest_index_tie_break ||
      !source.greedy_nonfinite_reported_and_ignored) {
    audit.failure_index = 5U;
    return audit;
  }
  audit.greedy_spec_exact = true;

  for (std::size_t right = 0U; right < ranges.size(); ++right) {
    for (std::size_t left = 0U; left < right; ++left) {
      if (!request_boundary_ranges_disjoint(ranges[left], ranges[right])) {
        audit.failure_index = 5U;
        return audit;
      }
    }
  }
  audit.nonnull_16b_aligned_disjoint_ranges = true;

  std::uint64_t lm_head_identity = 0x5133'4d46'5242'4c4dULL;
  for (std::size_t index = 2U; index < ranges.size(); ++index) {
    lm_head_identity = mix(lm_head_identity, ranges[index].begin);
    lm_head_identity = mix(lm_head_identity, ranges[index].end);
  }
  lm_head_identity = mix(lm_head_identity, lm_head->output_size);
  lm_head_identity = mix(lm_head_identity, lm_head->input_size);
  lm_head_identity = mix(
      lm_head_identity, Bound::kSm87MacroFeedV4LmHeadPackedWeightBytes);
  lm_head_identity = mix(
      lm_head_identity, Bound::kSm87MacroFeedV4LmHeadBlockScaleBytes);
  lm_head_identity = mix(lm_head_identity, audit.weight_scale_2_fp32_bits);
  lm_head_identity = mix(lm_head_identity, audit.input_scale_fp32_bits);
  lm_head_identity = mix(lm_head_identity,
                         audit.input_scale_provenance_retained);
  lm_head_identity = mix(lm_head_identity, audit.input_scale_consumed);
  audit.lm_head_identity = lm_head_identity;

  std::uint64_t greedy_identity = 0x5133'4d46'5242'4752ULL;
  greedy_identity = mix(greedy_identity, source.greedy_vocabulary);
  greedy_identity = mix(greedy_identity, source.greedy_workspace_results);
  greedy_identity = mix(
      greedy_identity, source.greedy_strict_left_to_right_fp32_order);
  greedy_identity =
      mix(greedy_identity, source.greedy_smallest_index_tie_break);
  greedy_identity =
      mix(greedy_identity, source.greedy_nonfinite_reported_and_ignored);
  audit.greedy_identity = greedy_identity;

  std::uint64_t catalog_identity = 0x5133'4d46'5242'5343ULL;
  catalog_identity = mix(catalog_identity, audit.embedding_identity);
  catalog_identity = mix(catalog_identity, audit.final_norm_identity);
  catalog_identity = mix(catalog_identity, audit.lm_head_identity);
  catalog_identity = mix(catalog_identity, audit.greedy_identity);
  catalog_identity = mix(catalog_identity, ranges.size());
  audit.catalog_identity = catalog_identity;
  audit.failure_index = ranges.size();
  return audit;
}

bool Sm87MacroFeedV4GateUpStartupSeal::valid() const noexcept {
  return issuer_nonce_ == kGateUpSealIssuerNonce && seal_identity != 0U &&
         seal_identity == gate_up_seal_identity(*this);
}

bool Sm87MacroFeedV4DownStartupSeal::valid() const noexcept {
  return issuer_nonce_ == kDownSealIssuerNonce && seal_identity != 0U &&
         seal_identity == down_seal_identity(*this);
}

bool Sm87MacroFeedV4Bf16AbStartupSeal::valid() const noexcept {
  return issuer_nonce_ == kBf16AbSealIssuerNonce && seal_identity != 0U &&
         seal_identity == bf16_ab_seal_identity(*this);
}

bool Sm87MacroFeedV4GdnQkvZStartupSeal::valid() const noexcept {
  return issuer_nonce_ == kGdnQkvZSealIssuerNonce &&
         seal_identity != 0U && seal_identity == gdn_qkvz_seal_identity(*this);
}

bool Sm87MacroFeedV4FullAttentionStartupSeal::valid() const noexcept {
  return issuer_nonce_ == kFullAttentionSealIssuerNonce &&
         seal_identity != 0U &&
         seal_identity == full_attention_source_seal_identity(*this);
}

bool Sm87MacroFeedV4RequestBoundaryStartupSeal::valid() const noexcept {
  return issuer_nonce_ == kRequestBoundarySealIssuerNonce &&
         seal_identity != 0U &&
         seal_identity == request_boundary_source_seal_identity(*this);
}

Sm87MacroFeedV4P40StartupPackage::Bf16AbStartupCapability::
    Bf16AbStartupCapability(
        const ModelWeights* const model_weights,
        std::array<Bf16AbPair,
                   kSm87MacroFeedV4P40StartupPackageBf16AbPairs>
            pairs,
        kernels::Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot resources,
        const std::uint64_t catalog_identity,
        const std::uint64_t package_identity,
        const std::uint64_t deployment_plan_identity,
        const std::uint64_t projection_owner_identity,
        const std::uint64_t projection_allocation_identity,
        const std::uint64_t projection_catalog_identity,
        const std::uint64_t projection_device_identity,
        const std::int32_t device_ordinal,
        const std::uint64_t issuer_nonce) noexcept
    : model_weights_(model_weights),
      pairs_(std::move(pairs)),
      resources_(resources),
      catalog_identity_(catalog_identity),
      package_identity_(package_identity),
      deployment_plan_identity_(deployment_plan_identity),
      projection_owner_identity_(projection_owner_identity),
      projection_allocation_identity_(projection_allocation_identity),
      projection_catalog_identity_(projection_catalog_identity),
      projection_device_identity_(projection_device_identity),
      device_ordinal_(device_ordinal),
      issuer_nonce_(issuer_nonce) {}

Sm87MacroFeedV4P40StartupPackage::Bf16AbStartupCapability::
    Bf16AbStartupCapability(Bf16AbStartupCapability&& other) noexcept
    : model_weights_(std::exchange(other.model_weights_, nullptr)),
      pairs_(std::move(other.pairs_)),
      resources_(std::exchange(
          other.resources_,
          kernels::Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot{})),
      catalog_identity_(std::exchange(other.catalog_identity_, 0U)),
      package_identity_(std::exchange(other.package_identity_, 0U)),
      deployment_plan_identity_(
          std::exchange(other.deployment_plan_identity_, 0U)),
      projection_owner_identity_(
          std::exchange(other.projection_owner_identity_, 0U)),
      projection_allocation_identity_(
          std::exchange(other.projection_allocation_identity_, 0U)),
      projection_catalog_identity_(
          std::exchange(other.projection_catalog_identity_, 0U)),
      projection_device_identity_(
          std::exchange(other.projection_device_identity_, 0U)),
      device_ordinal_(std::exchange(other.device_ordinal_, -1)),
      issuer_nonce_(std::exchange(other.issuer_nonce_, 0U)) {
  other.pairs_ = {};
}

Sm87MacroFeedV4P40StartupPackage::Bf16AbStartupCapability::
    ~Bf16AbStartupCapability() noexcept {
  model_weights_ = nullptr;
  pairs_ = {};
  resources_ = {};
  catalog_identity_ = 0U;
  package_identity_ = 0U;
  deployment_plan_identity_ = 0U;
  projection_owner_identity_ = 0U;
  projection_allocation_identity_ = 0U;
  projection_catalog_identity_ = 0U;
  projection_device_identity_ = 0U;
  device_ordinal_ = -1;
  issuer_nonce_ = 0U;
}

Sm87MacroFeedV4P40StartupPackage::Sm87MacroFeedV4P40StartupPackage(
    ProjectionAccess access,
    std::array<AssetCapability, kSm87MacroFeedV4P40StartupPackageArtifacts>
        capabilities,
    Bf16AbStartupCapability bf16_ab_capability,
    RequestBoundarySourceCatalog request_boundary_sources,
    Sm87MacroFeedV4PanelWavefrontPlan plan, StartupSeals seals,
    Sm87MacroFeedV4P40StartupPackageAudit audit) noexcept
    : projection_access_(std::move(access)),
      capabilities_(std::move(capabilities)),
      bf16_ab_capability_(std::move(bf16_ab_capability)),
      request_boundary_sources_(std::move(request_boundary_sources)),
      plan_(std::move(plan)),
      seals_(std::move(seals)),
      audit_(audit) {}

Sm87MacroFeedV4ProjectionStartupBinding::
    Sm87MacroFeedV4ProjectionStartupBinding(
        ProjectionAccess access, ProjectionAsset asset,
        Snapshot snapshot) noexcept
    : projection_access_(std::move(access)),
      asset_(std::move(asset)),
      snapshot_(std::move(snapshot)) {}

Sm87MacroFeedV4P40StartupPackageCreateResult::operator bool()
    const noexcept {
  return package != nullptr && static_cast<bool>(status) && audit.valid() &&
         package->valid() &&
         package->audit().package_identity == audit.package_identity;
}

Sm87MacroFeedV4P40StartupPackageCreateResult
Sm87MacroFeedV4P40StartupPackage::create(
    const ModelWeights& model_weights) noexcept {
#if !defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_STARTUP_PACKAGE_ADMISSION) || \
    !defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION) || \
    !defined(Q3X_ENABLE_SM87_MACROFEED_V4_BF16_AB_ADMISSION) || \
    !defined(Q3X_ENABLE_SM87_MACROFEED_V4_FP8_ADMISSION)
  (void)model_weights;
  CreateResult result;
  result.status = failure(Error::kAdmissionDisabled, "admission_disabled");
  return result;
#else
  auto access = ProjectionAccess::bind_request_boundary_startup(model_weights);
  if (!access) {
    CreateResult result;
    result.status =
        failure(Error::kProjectionAccessBind, "projection_access_bind");
    return result;
  }
  return create_from_private_access(model_weights, std::move(*access));
#endif
}

Sm87MacroFeedV4P40StartupPackageCreateResult
Sm87MacroFeedV4P40StartupPackage::create_from_host_test_authority(
    const ModelWeights& model_weights, ProjectionAccess access) noexcept {
#if !defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_STARTUP_PACKAGE_ADMISSION) || \
    !defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION) || \
    !defined(Q3X_ENABLE_SM87_MACROFEED_V4_BF16_AB_ADMISSION) || \
    !defined(Q3X_ENABLE_SM87_MACROFEED_V4_FP8_ADMISSION)
  (void)model_weights;
  (void)access;
  CreateResult result;
  result.status = failure(Error::kAdmissionDisabled, "admission_disabled");
  return result;
#else
  if (!access.attached() || !access.host_test_resident_authority_ ||
      access.host_test_issuer_nonce_ !=
          ProjectionAccess::kHostTestResidentIssuerNonce) {
    CreateResult result;
    result.status = failure(Error::kProjectionAccessBind,
                            "host_test_projection_access_bind");
    return result;
  }
  return create_from_private_access(model_weights, std::move(access));
#endif
}

Sm87MacroFeedV4P40StartupPackageCreateResult
Sm87MacroFeedV4P40StartupPackage::create_from_private_access(
    const ModelWeights& model_weights, ProjectionAccess access) noexcept {
#if !defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_STARTUP_PACKAGE_ADMISSION) || \
    !defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION) || \
    !defined(Q3X_ENABLE_SM87_MACROFEED_V4_BF16_AB_ADMISSION) || \
    !defined(Q3X_ENABLE_SM87_MACROFEED_V4_FP8_ADMISSION)
  (void)model_weights;
  (void)access;
  CreateResult result;
  result.status = failure(Error::kAdmissionDisabled, "admission_disabled");
  return result;
#else
  auto plan = make_sm87_macrofeed_v4_p40_panel_wavefront_plan();
  if (!sm87_macrofeed_v4_p40_panel_wavefront_plan_valid(plan) ||
      compute_deployment_plan_identity(plan) == 0U) {
    CreateResult result;
    result.status = failure(Error::kCanonicalPlan, "canonical_v4_plan");
    return result;
  }

  kernels::Sm87MacroFeedV4NvFp4GateUpCudaResources gate_up;
  int status =
      kernels::query_sm87_macrofeed_v4_nvfp4_gate_up_cuda_resources(&gate_up);
  if (status != 0 || !gate_up.static_resource_gate_passed ||
      !kernels::sm87_macrofeed_v4_nvfp4_gate_up_resource_gate(gate_up)) {
    CreateResult result;
    result.status = failure(Error::kGateUpStartupSeal,
                            "gate_up_startup_resource_seal", status);
    return result;
  }

  kernels::Sm87MacroFeedV4NvFp4DownCudaResources down;
  status = kernels::query_sm87_macrofeed_v4_nvfp4_down_cuda_resources(&down);
  if (status != 0 || !down.static_resource_gate_passed ||
      !kernels::sm87_macrofeed_v4_nvfp4_down_resource_gate(down)) {
    CreateResult result;
    result.status = failure(Error::kDownStartupSeal,
                            "down_startup_resource_seal", status);
    return result;
  }

  kernels::Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot bf16_ab;
  status = kernels::
      query_sm87_macrofeed_v4_bf16_ab_admission_resource_snapshot_cuda(
          &bf16_ab);
  if (status != 0 ||
      !kernels::sm87_macrofeed_v4_bf16_ab_admission_resource_gate(
          bf16_ab)) {
    CreateResult result;
    result.status = failure(Error::kBf16AbResourceSeal,
                            "bf16_ab_startup_resource_seal", status);
    return result;
  }
  kernels::Sm87MacroFeedV4Fp8CudaResources gdn_qkvz;
  status = kernels::query_sm87_macrofeed_v4_fp8_cuda_resources(
      kGdnQkvZRole, kGdnQkvZInputLayout, &gdn_qkvz);
  if (status == 0) {
    gdn_qkvz.static_resource_gate_passed =
        kernels::sm87_macrofeed_v4_fp8_resource_gate(gdn_qkvz);
  }
  if (status != 0 || gdn_qkvz.role != kGdnQkvZRole ||
      gdn_qkvz.input_layout != kGdnQkvZInputLayout ||
      gdn_qkvz.identity != kGdnQkvZTacticIdentity ||
      !gdn_qkvz.static_resource_gate_passed ||
      !kernels::sm87_macrofeed_v4_fp8_resource_gate(gdn_qkvz)) {
    CreateResult result;
    result.status = failure(Error::kGdnQkvZResourceSeal,
                            "gdn_qkvz_startup_resource_seal", status);
    return result;
  }
  kernels::Sm87MacroFeedV4Fp8CudaResources gdn_output;
  status = kernels::query_sm87_macrofeed_v4_fp8_cuda_resources(
      kAttentionOutputRole, kGdnOutputInputLayout, &gdn_output);
  if (status == 0) {
    gdn_output.static_resource_gate_passed =
        kernels::sm87_macrofeed_v4_fp8_resource_gate(gdn_output);
  }
  if (status != 0 || gdn_output.role != kAttentionOutputRole ||
      gdn_output.input_layout != kGdnOutputInputLayout ||
      gdn_output.identity != kGdnOutputTacticIdentity ||
      !gdn_output.static_resource_gate_passed ||
      !kernels::sm87_macrofeed_v4_fp8_resource_gate(gdn_output)) {
    CreateResult result;
    result.status = failure(Error::kGdnQkvZResourceSeal,
                            "gdn_output_startup_resource_seal", status,
                            kSm87MacroFeedV4P40StartupPackageLayers,
                            kAttentionOutputRole);
    return result;
  }
  return build_from_private_authority(std::move(access), std::move(plan),
                                      gate_up, down, bf16_ab, gdn_qkvz,
                                      gdn_output,
                                      model_weights);
#endif
}

#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_STARTUP_PACKAGE_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_MACROFEED_V4_BF16_AB_ADMISSION) && \
    defined(Q3X_ENABLE_SM87_MACROFEED_V4_FP8_ADMISSION)

std::uint64_t Sm87MacroFeedV4P40StartupPackage::
    compute_deployment_plan_identity(
        const Sm87MacroFeedV4PanelWavefrontPlan& plan) noexcept {
  if (!sm87_macrofeed_v4_p40_panel_wavefront_plan_valid(plan)) {
    return 0U;
  }
  std::uint64_t identity = 0x5133'4d46'5634'504cULL;
  for (const std::uint8_t byte : plan.magic) {
    identity = mix(identity, byte);
  }
  identity = mix(identity, plan.abi_major);
  identity = mix(identity, plan.abi_minor);
  identity = mix_string(identity, plan.candidate_id);
  identity = mix_string(identity, plan.deployment_plan_id);
  identity = mix_string(identity, plan.api.route_id);
  identity = mix_string(identity, plan.api.endpoint);
  identity = mix_string(identity, plan.api.served_model);
  identity = mix(identity, plan.api.prompt_tokens);
  identity = mix(identity, plan.api.maximum_output_tokens);
  identity = mix(identity, plan.api.batch_size);
  identity = mix(identity, plan.api.openai_compatible);
  identity = mix(identity, plan.api.exact_token_ids);
  identity = mix(identity, plan.api.cold_request);
  identity = mix(identity, plan.api.prefix_cache_disabled);
  identity = mix(identity, plan.api.kv_reuse_disabled);
  identity = mix(identity, plan.api.streaming_first_committed_token);
  identity = mix(identity, plan.api.full_prompt_consumption_required);
  identity = mix(identity, static_cast<std::uint64_t>(plan.traversal));
  identity = mix(identity, plan.prompt_tokens);
  identity = mix(identity, plan.panel_tokens);
  identity = mix(identity, plan.panel_count);
  identity = mix(identity, plan.layer_count);
  for (const auto& buffer : plan.workspace.buffers) {
    identity = mix(identity, static_cast<std::uint64_t>(buffer.role));
    identity = mix(identity, buffer.storage_identity);
    identity = mix(identity, buffer.offset);
    identity = mix(identity, buffer.bytes);
    identity = mix(identity, buffer.token_capacity);
    identity = mix(identity, buffer.row_width);
    identity = mix(identity, buffer.panel_local);
    identity = mix(identity, buffer.reuse_waits_for_completion);
  }
  identity = mix(identity, plan.workspace.transient_arena_bytes);
  identity = mix(identity, plan.workspace.maximum_temporary_tokens);
  identity = mix(identity, plan.workspace.ping_pong_hidden);
  identity = mix(identity, plan.workspace.scratch_reused_by_phase);
  identity = mix(identity, plan.workspace.full_p40_temporary_plane_allowed);
  identity = mix(
      identity, plan.workspace.persistent_kv_is_outside_transient_arena);
  identity = mix(
      identity,
      plan.workspace.persistent_conv_gdn_state_is_outside_transient_arena);
  identity = mix(
      identity,
      plan.phase_aliasing.attention_q_preprocess_overwrites_raw_q_slots);
  identity = mix(
      identity,
      plan.phase_aliasing.attention_online_core_reuses_processed_q);
  identity = mix(
      identity,
      plan.phase_aliasing.attention_interleaved_q_gate_layout_retained);
  identity = mix(
      identity,
      plan.phase_aliasing.attention_result_overwrites_q_slots_in_place);
  identity = mix(
      identity, plan.phase_aliasing.attention_gate_slots_remain_in_place);
  identity = mix(
      identity,
      plan.phase_aliasing
          .attention_output_projection_gathers_interleaved_q_slots);
  identity = mix(identity,
                 plan.phase_aliasing.gdn_recurrent_reuses_consumed_qkv);
  identity = mix(identity,
                 plan.phase_aliasing.gate_up_activation_owns_panel_scratch);
  identity = mix(identity,
                 plan.phase_aliasing.every_phase_fits_one_panel_scratch);
  identity = mix(identity,
                 plan.state_ownership.recurrent_epoch_bank_count);
  identity = mix(identity, plan.state_ownership.recurrent_epoch_bytes);
  identity = mix(identity, plan.state_ownership.recurrent_storage_bytes);
  identity = mix(
      identity, plan.state_ownership.active_recurrent_storage_identity);
  identity = mix(
      identity, plan.state_ownership.candidate_recurrent_storage_identity);
  identity = mix(
      identity, plan.state_ownership.private_kv_valid_end_storage_identity);
  identity = mix(identity,
                 plan.state_ownership.panel_commit_event_identity);
  identity = mix(identity,
                 plan.state_ownership.final_publish_event_identity);
  identity = mix(identity, plan.state_ownership.private_kv_valid_end);
  identity = mix(
      identity,
      plan.state_ownership
          .conv_history_copies_active_to_candidate_per_layer);
  identity = mix(
      identity,
      plan.state_ownership.gdn_state_writes_active_to_candidate_per_layer);
  identity = mix(
      identity,
      plan.state_ownership.candidate_epoch_fully_assigned_before_swap);
  identity = mix(
      identity,
      plan.state_ownership.whole_recurrent_epoch_copy_before_panel_allowed);
  identity = mix(
      identity, plan.state_ownership.active_candidate_swap_after_layer_63);
  identity = mix(
      identity, plan.state_ownership.panel_failure_discards_candidate_epoch);
  identity = mix(
      identity,
      plan.state_ownership.canonical_recurrent_publish_after_final_panel);
  identity = mix(
      identity,
      plan.state_ownership.sequence_length_is_final_visibility_fence);
  identity = mix(
      identity,
      plan.state_ownership.no_fallible_work_after_sequence_publication);
  for (const auto& panel : plan.panels) {
    identity = mix(identity, panel.panel_index + 1U);
    identity = mix(identity, panel.token_begin);
    identity = mix(identity, panel.token_count);
    identity = mix(identity, panel.sequence_begin);
    identity = mix(identity, panel.sequence_end);
    identity = mix(
        identity,
        static_cast<std::uint64_t>(panel.initial_workspace));
    identity = mix(identity,
                   static_cast<std::uint64_t>(panel.final_workspace));
    identity = mix(identity, panel.embedding_publishes_initial_workspace);
    identity = mix(identity, panel.workspace_reuse_waits_for_panel_commit);
    identity = mix(identity, panel.state_transaction.panel_index + 1U);
    identity = mix(identity, panel.state_transaction.token_begin);
    identity = mix(identity, panel.state_transaction.token_end);
    identity = mix(identity,
                   panel.state_transaction.incoming_state_epoch + 1U);
    identity = mix(identity,
                   panel.state_transaction.outgoing_state_epoch + 1U);
    identity = mix(
        identity,
        panel.state_transaction.commit_dependency_sequence_ordinal + 1U);
    identity = mix(identity, panel.state_transaction.kv_layer_count);
    identity = mix(identity, panel.state_transaction.conv_layer_count);
    identity = mix(identity, panel.state_transaction.gdn_layer_count);
    identity = mix(
        identity,
        panel.state_transaction.kv_uses_disjoint_final_token_slice);
    identity = mix(
        identity,
        panel.state_transaction.conv_and_gdn_use_private_next_epoch);
    identity = mix(identity,
                   panel.state_transaction.atomic_kv_conv_gdn_commit);
    identity = mix(identity, panel.state_transaction.commit_after_layer_63);
    identity = mix(identity,
                   panel.state_transaction.next_panel_waits_for_commit);
    identity = mix(
        identity,
        panel.state_transaction.rollback_discards_uncommitted_panel_state);
    identity = mix(
        identity,
        panel.state_transaction.state_private_to_prefill_until_request_commit);
    identity = mix(identity, panel.state_transaction.state_visible_to_decode);
    for (const auto& layer : panel.layers) {
      identity = mix(identity, layer.panel_index + 1U);
      identity = mix(identity, layer.sequence_ordinal + 1U);
      identity = mix(identity, layer.layer_index + 1U);
      identity = mix(identity, layer.token_begin);
      identity = mix(identity, layer.token_count);
      identity = mix(identity,
                     static_cast<std::uint64_t>(layer.layer_kind));
      identity = mix(identity,
                     static_cast<std::uint64_t>(layer.input_workspace));
      identity = mix(identity,
                     static_cast<std::uint64_t>(layer.output_workspace));
      identity = mix(identity,
                     static_cast<std::uint64_t>(layer.state_write_mode));
      identity = mix(
          identity, layer.input_consumed_before_output_publication);
      identity = mix(identity, layer.output_reuse_waits_for_completion);
      identity = mix(identity, layer.stages_kv);
      identity = mix(identity, layer.stages_conv_state);
      identity = mix(identity, layer.stages_gdn_state);
      identity = mix(identity, layer.publishes_state_to_decode);
    }
  }
  identity = mix(identity, plan.panel_loop_is_outermost);
  identity = mix(identity, plan.layer_loop_is_natural_order_innermost);
  identity = mix(identity, plan.final_request_commit_after_all_panels);
  identity = mix(identity, plan.partial_panel_commit_visible_to_decode);
  identity = mix(identity, plan.route.sm87_only);
  identity = mix(identity, plan.route.real_checkpoint_required);
  identity = mix(identity, plan.route.authenticated_aot_deployment_plan);
  identity = mix(identity, plan.route.startup_bound_tactics);
  identity = mix(identity, plan.route.request_time_jit_allowed);
  identity = mix(identity, plan.route.request_time_repack_allowed);
  identity = mix(identity, plan.route.request_time_autotune_allowed);
  identity = mix(identity, plan.route.fallback_allowed);
  identity = mix(identity, plan.route.cublaslt_allowed);
  identity = mix(identity, plan.route.mtp_allowed);
  identity = mix(identity, plan.route.approximate_numerics_allowed);
  identity = mix(identity, plan.route.default_off);
  identity = mix(identity, plan.route.test_only_contract);
  identity = mix(identity, plan.route.selector_bound);
  identity = mix(identity, plan.route.launcher_present);
  identity = mix(identity, plan.route.production_dispatch_eligible);
  identity = mix(identity, plan.route.numerical_qualification_complete);
  return identity == 0U ? 0x5133'4d46'5634'504cULL : identity;
}

bool Sm87MacroFeedV4P40StartupPackage::build_bf16_ab_pairs(
    const ModelWeights& model_weights, const std::int32_t device_ordinal,
    std::array<Bf16AbPair,
               kSm87MacroFeedV4P40StartupPackageBf16AbPairs>* const pairs,
    Sm87MacroFeedV4Bf16AbT0InventoryAudit* const inventory,
    int* const cuda_error, std::size_t* const failure_layer) noexcept {
  if (pairs == nullptr || inventory == nullptr || cuda_error == nullptr ||
      failure_layer == nullptr || device_ordinal < 0) {
    return false;
  }
  *pairs = {};
  *inventory = {};
  *cuda_error = 0;
  *failure_layer = kSm87MacroFeedV4P40StartupPackageLayers;

  std::array<Sm87MacroFeedV4Bf16AbT0TensorDescriptor,
             kSm87MacroFeedV4P40StartupPackageBf16AbTensors>
      tensors{};
  std::size_t ordinal = 0U;
  for (std::size_t model_layer = 0U;
       model_layer < kSm87MacroFeedV4P40StartupPackageLayers;
       ++model_layer) {
    const auto& attention = model_weights.layer(model_layer).attention;
    if (!model_layer_is_gdn(model_layer)) {
      if (!std::holds_alternative<FullAttentionWeights>(attention)) {
        *failure_layer = model_layer;
        return false;
      }
      continue;
    }
    if (ordinal >= pairs->size() ||
        gdn_model_layer(ordinal) != model_layer) {
      *failure_layer = model_layer;
      return false;
    }
    const auto* const linear =
        std::get_if<LinearAttentionWeights>(&attention);
    if (linear == nullptr) {
      *failure_layer = model_layer;
      return false;
    }
    const auto* const a =
        std::get_if<Bf16LinearWeight>(&linear->in_proj_a);
    const auto* const b =
        std::get_if<Bf16LinearWeight>(&linear->in_proj_b);
    if (a == nullptr || b == nullptr) {
      inventory->failure_index =
          2U * ordinal + static_cast<std::size_t>(a != nullptr);
      *failure_layer = model_layer;
      return false;
    }

    tensors[2U * ordinal] = {
        ordinal,
        model_layer,
        Sm87MacroFeedV4Bf16AbWeightRole::kA,
        LinearWeightKind::kBf16,
        a->weight,
        a->output_size,
        a->input_size,
    };
    tensors[2U * ordinal + 1U] = {
        ordinal,
        model_layer,
        Sm87MacroFeedV4Bf16AbWeightRole::kB,
        LinearWeightKind::kBf16,
        b->weight,
        b->output_size,
        b->input_size,
    };
    (*pairs)[ordinal] = {
        static_cast<std::uint32_t>(model_layer),
        a->weight,
        b->weight,
        bf16_ab_pair_identity(ordinal, model_layer, a->weight, b->weight),
    };
    ++ordinal;
  }
  if (ordinal != pairs->size()) {
    if (ordinal < pairs->size()) {
      inventory->failure_index = 2U * ordinal;
      *failure_layer = gdn_model_layer(ordinal);
    }
    return false;
  }

  auto structural = inspect_sm87_macrofeed_v4_bf16_ab_t0_inventory(
      tensors.data(), tensors.size());
  if (!structural.valid_t0()) {
    *inventory = structural;
    if (structural.failure_index < tensors.size()) {
      *failure_layer = tensors[structural.failure_index].model_layer;
    }
    return false;
  }
  for (std::size_t index = 0U; index < tensors.size(); ++index) {
    if (!live_current_device_allocation_range(
            tensors[index].weight,
            kernels::kSm87MacroFeedV4Bf16AbWeightBytes, device_ordinal,
            cuda_error)) {
      structural.failure_index = index;
      *inventory = structural;
      *failure_layer = tensors[index].model_layer;
      return false;
    }
  }
  for (std::size_t pair_index = 0U; pair_index < pairs->size();
       ++pair_index) {
    const auto& pair = (*pairs)[pair_index];
    if (pair.pair_identity == 0U || pair.a_weights == nullptr ||
        pair.b_weights == nullptr) {
      inventory->failure_index = 2U * pair_index;
      *failure_layer = pair.model_layer;
      return false;
    }
  }
  structural.live_cuda_device_ranges_validated = true;
  *inventory = structural;
  return true;
}

bool Sm87MacroFeedV4P40StartupPackage::build_request_boundary_source_catalog(
    const ModelWeights& model_weights, const ProjectionAccess& access,
    const std::uint64_t deployment_plan_identity,
    RequestBoundarySourceCatalog* const catalog, int* const cuda_error,
    std::size_t* const failure_index,
    const RequestBoundarySourceCatalog* const startup_authenticated_catalog)
    noexcept {
  if (catalog != nullptr) {
    *catalog = {};
  }
  if (cuda_error != nullptr) {
    *cuda_error = 0;
  }
  if (failure_index != nullptr) {
    *failure_index = 6U;
  }
  if (catalog == nullptr || cuda_error == nullptr || failure_index == nullptr ||
      deployment_plan_identity == 0U || !access.attached() ||
      access.model_weights_ != &model_weights ||
      !access.resident_root_matches() || access.owner_identity() == 0U ||
      access.allocation_identity() == 0U || access.catalog_identity() == 0U ||
      access.device_identity() == 0U || access.device_ordinal() < 0 ||
      access.resident_root_ == nullptr || access.resident_root_identity_ == 0U ||
      access.resident_arena_begin_ == 0U ||
      access.resident_arena_bytes_ == 0U ||
      access.resident_arena_bytes_ >
          std::numeric_limits<std::uintptr_t>::max() -
              access.resident_arena_begin_) {
    if (failure_index != nullptr) {
      *failure_index = 0U;
    }
    return false;
  }

  const bool host_authority = access.host_test_resident_authority_;
  const bool normal_authority = !host_authority;
  if ((host_authority &&
       access.host_test_issuer_nonce_ !=
           ProjectionAccess::kHostTestResidentIssuerNonce) ||
      (normal_authority &&
       (access.host_test_issuer_nonce_ != 0U ||
        access.resident_arena_bytes_ != kPinnedQwen36_27BArenaBytes))) {
    *failure_index = 0U;
    return false;
  }

  const Sm87MacroFeedV4RequestBoundaryT0SourceDescriptor source{
      model_weights.embed_tokens(),
      model_weights.final_norm(),
      model_weights.lm_head(),
      Bound::kSm87MacroFeedV4FinalNormEpsilonFp32Bits,
      Bound::kSm87MacroFeedV4Vocabulary,
      Bound::kSm87MacroFeedV4GreedyWorkspaceResults,
      true,
      true,
      true,
  };
  const auto structural =
      inspect_sm87_macrofeed_v4_request_boundary_t0_source(source);
  if (!structural.valid_t0()) {
    *failure_index = structural.failure_index;
    return false;
  }
  const auto* const lm_head =
      std::get_if<NvFp4LinearWeight>(&model_weights.lm_head());
  if (lm_head == nullptr) {
    *failure_index = 2U;
    return false;
  }

  // A normal seal is tied to the exact named tensors in the authenticated
  // ResidentWeights root, not merely to arbitrary ranges that happen to sit
  // inside the same CUDA allocation.  This rejects role substitution while
  // leaving the explicitly separate T0 host authority incapable of claiming
  // a production resident binding.
  if (normal_authority) {
    using DType = io::safetensors::DType;
    const ResidentWeights& resident = *access.resident_root_;
    const std::array<bool, 6U> canonical_views{{
        resident_tensor_view_matches(
            resident, "model.language_model.embed_tokens.weight",
            model_weights.embed_tokens().weight, DType::kBf16,
            Bound::kSm87MacroFeedV4EmbeddingTableBytes,
            Bound::kSm87MacroFeedV4EmbeddingVocabulary,
            Bound::kSm87MacroFeedV4Hidden, 2U),
        resident_tensor_view_matches(
            resident, "model.language_model.norm.weight",
            model_weights.final_norm().data, DType::kBf16,
            Bound::kSm87MacroFeedV4FinalNormBytes,
            Bound::kSm87MacroFeedV4Hidden, 0U, 1U),
        resident_tensor_view_matches(
            resident, "lm_head.weight", lm_head->packed_weight, DType::kU8,
            Bound::kSm87MacroFeedV4LmHeadPackedWeightBytes,
            Bound::kSm87MacroFeedV4LmHeadRows,
            Bound::kSm87MacroFeedV4LmHeadColumns / 2U, 2U),
        resident_tensor_view_matches(
            resident, "lm_head.weight_scale", lm_head->block_scale,
            DType::kF8E4M3,
            Bound::kSm87MacroFeedV4LmHeadBlockScaleBytes,
            Bound::kSm87MacroFeedV4LmHeadRows,
            Bound::kSm87MacroFeedV4LmHeadColumns / 16U, 2U),
        resident_tensor_view_matches(
            resident, "lm_head.weight_scale_2",
            lm_head->weight_scale_2_device, DType::kF32, sizeof(float), 0U,
            0U, 0U),
        resident_tensor_view_matches(
            resident, "lm_head.input_scale", lm_head->input_scale_device,
            DType::kF32, sizeof(float), 0U, 0U, 0U),
    }};
    for (std::size_t index = 0U; index < canonical_views.size(); ++index) {
      if (!canonical_views[index]) {
        *failure_index = index;
        return false;
      }
    }
  }

  const std::array<const void*, 6U> pointers{{
      model_weights.embed_tokens().weight,
      model_weights.final_norm().data,
      lm_head->packed_weight,
      lm_head->block_scale,
      lm_head->weight_scale_2_device,
      lm_head->input_scale_device,
  }};
  const std::array<std::uint64_t, 6U> bytes{{
      Bound::kSm87MacroFeedV4EmbeddingTableBytes,
      Bound::kSm87MacroFeedV4FinalNormBytes,
      Bound::kSm87MacroFeedV4LmHeadPackedWeightBytes,
      Bound::kSm87MacroFeedV4LmHeadBlockScaleBytes,
      sizeof(float),
      sizeof(float),
  }};
  const std::uintptr_t resident_arena_end =
      access.resident_arena_begin_ +
      static_cast<std::uintptr_t>(access.resident_arena_bytes_);
  for (std::size_t index = 0U; index < pointers.size(); ++index) {
    int range_error = 0;
    if (!live_current_device_allocation_range(
            pointers[index], bytes[index], access.device_ordinal(),
            &range_error, access.resident_arena_begin_, resident_arena_end,
            access.resident_arena_bytes_)) {
      *cuda_error = range_error;
      *failure_index = index;
      return false;
    }
  }
  // Exactly one D2H read of each scalar is allowed: the initial Startup
  // source seal.  Later package/catalog revalidation proves the immutable
  // ModelWeights/Resident attachment and reuses that sealed raw-bit receipt;
  // it does not introduce request-time scalar traffic.
  if (startup_authenticated_catalog == nullptr) {
    std::uint32_t weight_scale_2_device_bits = 0U;
    cudaError_t copy_status = cudaMemcpy(
        &weight_scale_2_device_bits, lm_head->weight_scale_2_device,
        sizeof(weight_scale_2_device_bits), cudaMemcpyDeviceToHost);
    if (copy_status != cudaSuccess ||
        weight_scale_2_device_bits != structural.weight_scale_2_fp32_bits) {
      *cuda_error = copy_status == cudaSuccess
                        ? static_cast<int>(cudaErrorInvalidValue)
                        : static_cast<int>(copy_status);
      *failure_index = 4U;
      return false;
    }
    std::uint32_t input_scale_device_bits = 0U;
    copy_status = cudaMemcpy(
        &input_scale_device_bits, lm_head->input_scale_device,
        sizeof(input_scale_device_bits), cudaMemcpyDeviceToHost);
    if (copy_status != cudaSuccess ||
        input_scale_device_bits != structural.input_scale_fp32_bits) {
      *cuda_error = copy_status == cudaSuccess
                        ? static_cast<int>(cudaErrorInvalidValue)
                        : static_cast<int>(copy_status);
      *failure_index = 5U;
      return false;
    }
  } else {
    if (!startup_authenticated_catalog->device_scale_raw_bits_match_host ||
        startup_authenticated_catalog->model_weights != &model_weights ||
        startup_authenticated_catalog->resident_root !=
            access.resident_root_ ||
        startup_authenticated_catalog->lm_head.weight_scale_2_device !=
            lm_head->weight_scale_2_device ||
        startup_authenticated_catalog->lm_head.input_scale_device !=
            lm_head->input_scale_device ||
        startup_authenticated_catalog->lm_head.weight_scale_2_fp32_bits !=
            structural.weight_scale_2_fp32_bits ||
        startup_authenticated_catalog->lm_head.input_scale_fp32_bits !=
            structural.input_scale_fp32_bits) {
      *failure_index = 4U;
      return false;
    }
  }

  RequestBoundarySourceCatalog sealed;
  sealed.model_weights = &model_weights;
  sealed.resident_root = access.resident_root_;
  sealed.embedding = {
      model_weights.embed_tokens().weight,
      model_weights.embed_tokens().output_size,
      model_weights.embed_tokens().input_size,
      Bound::kSm87MacroFeedV4EmbeddingTableBytes,
      structural.embedding_identity,
  };
  sealed.final_norm = {
      model_weights.final_norm().data,
      model_weights.final_norm().element_count,
      Bound::kSm87MacroFeedV4FinalNormBytes,
      Bound::kSm87MacroFeedV4FinalNormEpsilonFp32Bits,
      structural.final_norm_identity,
  };
  sealed.lm_head = {
      lm_head->packed_weight,
      lm_head->block_scale,
      lm_head->weight_scale_2_device,
      lm_head->input_scale_device,
      lm_head->output_size,
      lm_head->input_size,
      Bound::kSm87MacroFeedV4LmHeadPackedWeightBytes,
      Bound::kSm87MacroFeedV4LmHeadBlockScaleBytes,
      structural.weight_scale_2_fp32_bits,
      structural.input_scale_fp32_bits,
      structural.lm_head_identity,
      true,
      true,
      false,
  };
  sealed.greedy = {
      Bound::kSm87MacroFeedV4Vocabulary,
      Bound::kSm87MacroFeedV4GreedyWorkspaceResults,
      structural.greedy_identity,
      true,
      true,
      true,
  };
  sealed.projection_owner_identity = access.owner_identity();
  sealed.projection_allocation_identity = access.allocation_identity();
  sealed.projection_catalog_identity = access.catalog_identity();
  sealed.projection_device_identity = access.device_identity();
  sealed.resident_root_identity = access.resident_root_identity_;
  sealed.resident_arena_begin = access.resident_arena_begin_;
  sealed.resident_arena_end = resident_arena_end;
  sealed.resident_arena_bytes = access.resident_arena_bytes_;
  sealed.device_ordinal = access.device_ordinal();
  sealed.exact_shapes_and_specs = true;
  sealed.complete_live_device_ranges = true;
  sealed.device_scale_raw_bits_match_host = true;
  sealed.input_scale_provenance_retained = true;
  sealed.input_scale_consumed = false;
  sealed.normal_resident_authority = normal_authority;
  sealed.host_test_resident_authority = host_authority;

  std::uint64_t identity = 0x5133'4d46'5242'4341ULL;
  identity = mix(identity,
                 reinterpret_cast<std::uintptr_t>(sealed.model_weights));
  identity = mix(identity,
                 reinterpret_cast<std::uintptr_t>(sealed.resident_root));
  identity = mix(identity, deployment_plan_identity);
  identity = mix(identity, sealed.projection_owner_identity);
  identity = mix(identity, sealed.projection_allocation_identity);
  identity = mix(identity, sealed.projection_catalog_identity);
  identity = mix(identity, sealed.projection_device_identity);
  identity = mix(identity,
                 static_cast<std::uint64_t>(sealed.device_ordinal + 1));
  identity = mix(identity, sealed.resident_root_identity);
  identity = mix(identity, sealed.resident_arena_begin);
  identity = mix(identity, sealed.resident_arena_end);
  identity = mix(identity, sealed.resident_arena_bytes);
  identity = mix(identity, sealed.embedding.source_identity);
  identity = mix(identity, sealed.final_norm.source_identity);
  identity = mix(identity, sealed.lm_head.source_identity);
  identity = mix(identity, sealed.greedy.spec_identity);
  identity = mix(identity, sealed.final_norm.epsilon_fp32_bits);
  identity = mix(identity, sealed.lm_head.weight_scale_2_fp32_bits);
  identity = mix(identity, sealed.lm_head.input_scale_fp32_bits);
  identity = mix(identity, sealed.exact_shapes_and_specs);
  identity = mix(identity, sealed.complete_live_device_ranges);
  identity = mix(identity, sealed.device_scale_raw_bits_match_host);
  identity = mix(identity, sealed.input_scale_provenance_retained);
  identity = mix(identity, sealed.input_scale_consumed);
  identity = mix(identity, sealed.normal_resident_authority);
  identity = mix(identity, sealed.host_test_resident_authority);
  if (identity == 0U) {
    *failure_index = 0U;
    return false;
  }
  sealed.catalog_identity = identity;
  *catalog = sealed;
  *failure_index = pointers.size();
  return true;
}

bool Sm87MacroFeedV4P40StartupPackage::Bf16AbStartupCapability::valid(
    const ProjectionAccess& access) const noexcept {
  if (issuer_nonce_ != kBf16AbCapabilityIssuerNonce ||
      model_weights_ == nullptr || catalog_identity_ == 0U ||
      package_identity_ == 0U || deployment_plan_identity_ == 0U ||
      projection_owner_identity_ == 0U ||
      projection_allocation_identity_ == 0U ||
      projection_catalog_identity_ == 0U ||
      projection_device_identity_ == 0U || !access.attached() ||
      access.owner_identity() != projection_owner_identity_ ||
      access.allocation_identity() != projection_allocation_identity_ ||
      access.catalog_identity() != projection_catalog_identity_ ||
      access.device_identity() != projection_device_identity_ ||
      access.device_ordinal() != device_ordinal_ ||
      device_ordinal_ < 0 || resources_.device_ordinal != device_ordinal_ ||
      !kernels::sm87_macrofeed_v4_bf16_ab_admission_resource_gate(
          resources_)) {
    return false;
  }

  kernels::Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot fresh_resources;
  const int resource_status = kernels::
      query_sm87_macrofeed_v4_bf16_ab_admission_resource_snapshot_cuda(
          &fresh_resources);
  if (resource_status != 0 ||
      !bf16_ab_resource_equal(resources_, fresh_resources)) {
    return false;
  }

  std::array<Bf16AbPair,
             kSm87MacroFeedV4P40StartupPackageBf16AbPairs>
      fresh_pairs{};
  Sm87MacroFeedV4Bf16AbT0InventoryAudit fresh_inventory;
  int cuda_error = 0;
  std::size_t failure_layer = kSm87MacroFeedV4P40StartupPackageLayers;
  if (!Sm87MacroFeedV4P40StartupPackage::build_bf16_ab_pairs(
          *model_weights_, device_ordinal_, &fresh_pairs, &fresh_inventory,
          &cuda_error, &failure_layer) ||
      cuda_error != 0 || fresh_inventory.catalog_identity == 0U ||
      fresh_inventory.tensors !=
          kSm87MacroFeedV4P40StartupPackageBf16AbTensors ||
      fresh_inventory.pairs !=
          kSm87MacroFeedV4P40StartupPackageBf16AbPairs ||
      !fresh_inventory.canonical_natural_layer_order ||
      !fresh_inventory.canonical_a_then_b_role_order ||
      !fresh_inventory.exact_bf16_shapes ||
      !fresh_inventory.nonnull_16b_aligned_disjoint_ranges ||
      !fresh_inventory.live_cuda_device_ranges_validated ||
      fresh_inventory.execution_capability ||
      fresh_inventory.catalog_identity != catalog_identity_) {
    return false;
  }
  for (std::size_t ordinal = 0U; ordinal < pairs_.size(); ++ordinal) {
    const auto& retained = pairs_[ordinal];
    const auto& fresh = fresh_pairs[ordinal];
    if (retained.model_layer != gdn_model_layer(ordinal) ||
        fresh.model_layer != retained.model_layer ||
        fresh.a_weights != retained.a_weights ||
        fresh.b_weights != retained.b_weights ||
        fresh.pair_identity == 0U ||
        fresh.pair_identity != retained.pair_identity) {
      return false;
    }
  }
  return true;
}

std::uint64_t Sm87MacroFeedV4P40StartupPackage::
    compute_gdn_qkvz_asset_value_identity(
        const kernels::Sm87TargetAotFp8CudaAssetView& asset) noexcept {
  if (!kernels::sm87_target_aot_fp8_cuda_asset_valid(asset) ||
      asset.artifact_identity == 0U ||
      asset.source_inventory_identity == 0U ||
      asset.host_manifest_seal.value == 0U ||
      kernels::sm87_target_aot_projection_digest_is_zero(
          asset.host_payload_digest) ||
      asset.payload.begin == 0U || asset.payload.end <= asset.payload.begin ||
      asset.payload.bytes == 0U ||
      asset.payload.end - asset.payload.begin != asset.payload.bytes ||
      asset.tensor_scale_count == 0U ||
      asset.tensor_scale_count > asset.tensor_scale_bits.size() ||
      !asset.no_request_time_repacking ||
      !asset.no_request_time_scale_conversion || !asset.valid) {
    return 0U;
  }
  const auto& upload = asset.device_upload_receipt;
  std::uint64_t identity = 0x5133'4d46'4650'3841ULL;
  identity = mix(identity, asset.artifact_identity);
  identity = mix(identity, asset.source_inventory_identity);
  identity = mix(identity, static_cast<std::uint64_t>(asset.transform_identity));
  identity = mix_digest(identity, asset.host_payload_digest);
  identity = mix(identity, asset.host_manifest_seal.value);
  identity = mix(identity, asset.payload.begin);
  identity = mix(identity, asset.payload.end);
  identity = mix(identity, asset.payload.bytes);
  identity = mix(identity, upload.receipt_identity);
  identity = mix(identity, upload.device_allocation_owner_identity);
  identity = mix(identity, upload.device_allocation_identity);
  identity = mix(identity,
                 static_cast<std::uint64_t>(upload.device_ordinal + 1));
  identity = mix(identity, upload.device_allocation_begin);
  identity = mix(identity, upload.device_allocation_end);
  identity = mix(identity, upload.device_allocation_bytes);
  identity = mix(identity, upload.device_payload_begin);
  identity = mix(identity, upload.device_payload_end);
  identity = mix(identity, upload.device_payload_bytes);
  identity = mix(identity, asset.tensor_scale_count);
  for (std::size_t index = 0U; index < asset.tensor_scale_bits.size();
       ++index) {
    identity = mix(identity, asset.tensor_scale_bits[index]);
    identity = mix(identity,
                   asset.compensated_tensor_scale_bf16_bits[index]);
  }
  identity = mix(identity, asset.no_request_time_repacking);
  identity = mix(identity, asset.no_request_time_scale_conversion);
  identity = mix(identity, asset.valid);
  return identity == 0U ? 0x5133'4d46'4650'3841ULL : identity;
}

std::uint64_t Sm87MacroFeedV4P40StartupPackage::
    compute_nvfp4_asset_value_identity(
        const kernels::Sm87TargetAotNvFp4CudaAssetView& asset) noexcept {
  if (!kernels::sm87_target_aot_nvfp4_cuda_asset_valid(asset) ||
      asset.artifact_identity == 0U ||
      asset.source_inventory_identity == 0U ||
      asset.host_manifest_seal.value == 0U ||
      kernels::sm87_target_aot_projection_digest_is_zero(
          asset.host_payload_digest) ||
      asset.payload.begin == 0U || asset.payload.end <= asset.payload.begin ||
      asset.payload.bytes == 0U ||
      asset.payload.end - asset.payload.begin != asset.payload.bytes ||
      asset.tensor_scale_count == 0U ||
      asset.tensor_scale_count > asset.tensor_scale_bits.size() ||
      !asset.no_request_time_repacking || !asset.valid) {
    return 0U;
  }
  const auto& upload = asset.device_upload_receipt;
  std::uint64_t identity = 0x5133'4d46'4e56'3441ULL;
  identity = mix(identity, asset.artifact_identity);
  identity = mix(identity, asset.source_inventory_identity);
  identity = mix(identity, static_cast<std::uint64_t>(asset.transform_identity));
  identity = mix_digest(identity, asset.host_payload_digest);
  identity = mix(identity, asset.host_manifest_seal.value);
  identity = mix(identity, asset.payload.begin);
  identity = mix(identity, asset.payload.end);
  identity = mix(identity, asset.payload.bytes);
  identity = mix(identity, upload.receipt_identity);
  identity = mix(identity, upload.device_allocation_owner_identity);
  identity = mix(identity, upload.device_allocation_identity);
  identity = mix(identity,
                 static_cast<std::uint64_t>(upload.device_ordinal + 1));
  identity = mix(identity, upload.device_allocation_begin);
  identity = mix(identity, upload.device_allocation_end);
  identity = mix(identity, upload.device_allocation_bytes);
  identity = mix(identity, upload.device_payload_begin);
  identity = mix(identity, upload.device_payload_end);
  identity = mix(identity, upload.device_payload_bytes);
  identity = mix(identity, asset.tensor_scale_count);
  for (const std::uint32_t bits : asset.tensor_scale_bits) {
    identity = mix(identity, bits);
  }
  identity = mix(identity, asset.no_request_time_repacking);
  identity = mix(identity, asset.valid);
  return identity == 0U ? 0x5133'4d46'4e56'3441ULL : identity;
}

std::uint64_t Sm87MacroFeedV4P40StartupPackage::
    compute_gdn_qkvz_binding_catalog_identity(
        const ProjectionAccess& access,
        const std::array<AssetCapability,
                         kSm87MacroFeedV4P40StartupPackageArtifacts>&
            capabilities,
        const std::uint64_t plan_identity,
        const kernels::Sm87MacroFeedV4Fp8CudaResources& resources,
        const kernels::Sm87MacroFeedV4Fp8CudaResources& output_resources,
        const ModelWeights& model_weights) noexcept {
  const std::uint64_t projection_catalog_identity =
      access.catalog_identity();
  if (!access.attached() || access.owner_identity() == 0U ||
      access.allocation_identity() == 0U || access.device_identity() == 0U ||
      access.device_ordinal() < 0 || projection_catalog_identity == 0U ||
      plan_identity == 0U || resources.role != kGdnQkvZRole ||
      resources.input_layout != kGdnQkvZInputLayout ||
      resources.identity != kGdnQkvZTacticIdentity ||
      resources.device_ordinal != access.device_ordinal() ||
      !resources.static_resource_gate_passed ||
      !kernels::sm87_macrofeed_v4_fp8_resource_gate(resources) ||
      output_resources.role != kAttentionOutputRole ||
      output_resources.input_layout != kGdnOutputInputLayout ||
      output_resources.identity != kGdnOutputTacticIdentity ||
      output_resources.device_ordinal != access.device_ordinal() ||
      !output_resources.static_resource_gate_passed ||
      !kernels::sm87_macrofeed_v4_fp8_resource_gate(output_resources)) {
    return 0U;
  }

  std::uint64_t identity = 0x5133'4d46'4743'4154ULL;
  identity = mix(identity, access.owner_identity());
  identity = mix(identity, access.allocation_identity());
  identity = mix(identity, projection_catalog_identity);
  identity = mix(identity, access.device_identity());
  identity = mix(identity,
                 static_cast<std::uint64_t>(access.device_ordinal() + 1));
  identity = mix(identity, plan_identity);
  identity = mix(identity, kSm87MacroFeedV4P40StartupPackageGdnLayers);
  identity = mix(identity, static_cast<std::uint64_t>(kGdnQkvZRole));
  identity = mix(identity,
                 static_cast<std::uint64_t>(kGdnQkvZInputLayout));
  identity = mix(identity,
                 static_cast<std::uint64_t>(kGdnQkvZTacticIdentity));
  identity = mix_fp8_resource(identity, resources);
  identity = mix(identity, static_cast<std::uint64_t>(kAttentionOutputRole));
  identity = mix(identity,
                 static_cast<std::uint64_t>(kGdnOutputInputLayout));
  identity = mix(identity,
                 static_cast<std::uint64_t>(kGdnOutputTacticIdentity));
  identity = mix_fp8_resource(identity, output_resources);

  for (std::size_t ordinal = 0U;
       ordinal < kSm87MacroFeedV4P40StartupPackageGdnLayers; ++ordinal) {
    const std::size_t model_layer = gdn_model_layer(ordinal);
    const std::size_t qkvz_index =
        sm87_target_aot_complete_descriptor_ordinal(model_layer,
                                                    kGdnQkvZRole);
    const std::size_t output_index =
        sm87_target_aot_complete_descriptor_ordinal(model_layer,
                                                    kAttentionOutputRole);
    if (!model_layer_is_gdn(model_layer) ||
        qkvz_index >= capabilities.size() ||
        output_index >= capabilities.size()) {
      return 0U;
    }
    const std::array<std::pair<const AssetCapability*, Role>, 2U> projections{{
        {&capabilities[qkvz_index], kGdnQkvZRole},
        {&capabilities[output_index], kAttentionOutputRole},
    }};
    identity = mix(identity, ordinal + 1U);
    identity = mix(identity, model_layer + 1U);
    for (const auto& projection : projections) {
      const auto& capability = *projection.first;
      const Role role = projection.second;
      const auto layout =
          kernels::sm87_target_aot_projection_packed_layout(role);
      if (!capability.asset || !layout.valid() ||
          capability.layer_index != model_layer || capability.role != role ||
          capability.encoding != layout.encoding ||
          capability.artifact_identity == 0U ||
          capability.source_inventory_identity == 0U ||
          capability.manifest_seal == 0U ||
          capability.upload_receipt_identity == 0U ||
          capability.payload_bytes != layout.payload_bytes ||
          capability.payload_begin == 0U ||
          capability.payload_end <= capability.payload_begin ||
          capability.payload_end - capability.payload_begin !=
              capability.payload_bytes ||
          capability.source_count != layout.partition_count) {
        return 0U;
      }
      const auto* const asset = capability.asset->borrow_fp8_cuda_asset();
      const std::uint64_t asset_value_identity =
          asset == nullptr ? 0U
                           : compute_gdn_qkvz_asset_value_identity(*asset);
      if (asset_value_identity == 0U ||
          capability.asset->borrow_nvfp4_cuda_asset() != nullptr ||
          asset->payload.role != role ||
          asset->artifact_identity != capability.artifact_identity ||
          asset->source_inventory_identity !=
              capability.source_inventory_identity ||
          asset->host_manifest_seal.value != capability.manifest_seal ||
          asset->device_upload_receipt.receipt_identity !=
              capability.upload_receipt_identity ||
          asset->host_payload_digest != capability.payload_digest ||
          asset->payload.begin != capability.payload_begin ||
          asset->payload.end != capability.payload_end ||
          asset->payload.bytes != capability.payload_bytes ||
          asset->tensor_scale_count != capability.source_count) {
        return 0U;
      }
      for (std::size_t source_index = 0U;
           source_index < capability.source_count; ++source_index) {
        if (asset->tensor_scale_bits[source_index] !=
            capability.tensor_scale_bits[source_index]) {
          return 0U;
        }
      }
      identity = mix(identity, static_cast<std::uint64_t>(role));
      identity = mix(identity, capability.artifact_identity);
      identity = mix(identity, capability.source_inventory_identity);
      identity = mix(identity, capability.manifest_seal);
      identity = mix(identity, capability.upload_receipt_identity);
      identity = mix_digest(identity, capability.payload_digest);
      identity = mix(identity, capability.payload_begin);
      identity = mix(identity, capability.payload_end);
      identity = mix(identity, capability.payload_bytes);
      identity = mix(identity, capability.source_count);
      identity = mix(identity, asset_value_identity);
    }

    GdnContinuationSource continuation;
    if (!exact_gdn_continuation_source(model_weights, model_layer,
                                       &continuation)) {
      // Startup retains the ModelWeights lifetime root but deliberately does
      // not promote host/fake continuation pointers into execution facts.
      // Exact shapes and live CUDA ranges are mandatory only when the private
      // execution catalog is sealed.
      identity = mix(identity, 0x5133'4d46'4744'4e50ULL);
      identity = mix(identity, model_layer + 1U);
      continue;
    }
    const std::array<const std::uint16_t*, 4U> weights{{
        continuation.conv_weight, continuation.a_log,
        continuation.dt_bias, continuation.norm_weight}};
    const std::array<std::uint64_t, 4U> weight_bytes{{
        kernels::kSm87MacroFeedV4GdnConvWeightBytes,
        kernels::kSm87MacroFeedV4GdnHeadVectorBytes,
        kernels::kSm87MacroFeedV4GdnHeadVectorBytes,
        kernels::kSm87MacroFeedV4GdnNormWeightBytes}};
    for (std::size_t role_index = 0U; role_index < weights.size();
         ++role_index) {
      const std::uint64_t weight_identity = gdn_weight_identity(
          0x5133'4d46'4744'4e57ULL, plan_identity,
          access.device_identity(), model_layer, role_index,
          weights[role_index], weight_bytes[role_index]);
      if (weight_identity == 0U) {
        return 0U;
      }
      identity = mix(identity, weight_identity);
    }
  }
  return identity == 0U ? 0x5133'4d46'4743'4154ULL : identity;
}

std::uint64_t Sm87MacroFeedV4P40StartupPackage::
    compute_mlp_pair_binding_catalog_identity(
        const ProjectionAccess& access,
        const std::array<AssetCapability,
                         kSm87MacroFeedV4P40StartupPackageArtifacts>&
            capabilities,
        const std::uint64_t plan_identity) noexcept {
  if (!access.attached() || access.owner_identity() == 0U ||
      access.allocation_identity() == 0U || access.catalog_identity() == 0U ||
      access.device_identity() == 0U || access.device_ordinal() < 0 ||
      plan_identity == 0U) {
    return 0U;
  }
  std::uint64_t identity = 0x5133'4d46'4d4c'5043ULL;
  identity = mix(identity, access.owner_identity());
  identity = mix(identity, access.allocation_identity());
  identity = mix(identity, access.catalog_identity());
  identity = mix(identity, access.device_identity());
  identity = mix(identity,
                 static_cast<std::uint64_t>(access.device_ordinal() + 1));
  identity = mix(identity, plan_identity);
  identity = mix(identity, kSm87MacroFeedV4P40StartupPackageLayers);
  for (std::size_t model_layer = 0U;
       model_layer < kSm87MacroFeedV4P40StartupPackageLayers;
       ++model_layer) {
    identity = mix(identity, model_layer + 1U);
    for (const Role role : {Role::kNvFp4GateUp, Role::kNvFp4Down}) {
      const std::size_t index =
          sm87_target_aot_complete_descriptor_ordinal(model_layer, role);
      if (index >= capabilities.size()) {
        return 0U;
      }
      const auto& capability = capabilities[index];
      const auto layout =
          kernels::sm87_target_aot_projection_packed_layout(role);
      if (!capability.asset || !layout.valid() ||
          capability.layer_index != model_layer || capability.role != role ||
          capability.encoding != layout.encoding ||
          capability.payload_bytes != layout.payload_bytes ||
          capability.payload_begin == 0U ||
          capability.payload_end <= capability.payload_begin ||
          capability.source_count != layout.partition_count) {
        return 0U;
      }
      const auto* const asset = capability.asset->borrow_nvfp4_cuda_asset();
      const std::uint64_t asset_identity =
          asset == nullptr ? 0U : compute_nvfp4_asset_value_identity(*asset);
      if (asset_identity == 0U ||
          capability.asset->borrow_fp8_cuda_asset() != nullptr ||
          asset->payload.role != role ||
          asset->artifact_identity != capability.artifact_identity ||
          asset->source_inventory_identity !=
              capability.source_inventory_identity) {
        return 0U;
      }
      identity = mix(identity, static_cast<std::uint64_t>(role));
      identity = mix(identity, asset_identity);
    }
  }
  return identity == 0U ? 0x5133'4d46'4d4c'5043ULL : identity;
}

std::uint64_t Sm87MacroFeedV4P40StartupPackage::
    compute_full_attention_source_catalog_identity(
        const ProjectionAccess& access,
        const std::array<AssetCapability,
                         kSm87MacroFeedV4P40StartupPackageArtifacts>&
            capabilities,
        const std::uint64_t plan_identity,
        const ModelWeights& model_weights,
        std::size_t* const failure_ordinal,
        bool* const failure_k_norm,
        int* const cuda_error) noexcept {
  if (failure_ordinal != nullptr) {
    *failure_ordinal = kSm87MacroFeedV4P40StartupPackageFullLayers;
  }
  if (failure_k_norm != nullptr) {
    *failure_k_norm = false;
  }
  if (cuda_error != nullptr) {
    *cuda_error = 0;
  }
  if (!access.attached() || access.owner_identity() == 0U ||
      access.allocation_identity() == 0U || access.catalog_identity() == 0U ||
      access.device_identity() == 0U || access.device_ordinal() < 0 ||
      plan_identity == 0U) {
    if (failure_ordinal != nullptr) {
      *failure_ordinal = 0U;
    }
    return 0U;
  }

  constexpr std::uint64_t kNormBytes =
      kernels::kSm87MacroFeedV4FullAttentionPreprocessNormWeightBytes;
  std::array<std::uintptr_t,
             2U * kSm87MacroFeedV4P40StartupPackageFullLayers>
      norm_begins{};
  std::array<std::uintptr_t,
             2U * kSm87MacroFeedV4P40StartupPackageFullLayers>
      norm_ends{};
  std::uint64_t identity = 0x5133'4d46'4653'4341ULL;
  identity = mix(identity, access.owner_identity());
  identity = mix(identity, access.allocation_identity());
  identity = mix(identity, access.catalog_identity());
  identity = mix(identity, access.device_identity());
  identity = mix(identity,
                 static_cast<std::uint64_t>(access.device_ordinal() + 1));
  identity = mix(identity, plan_identity);
  identity = mix(identity, kSm87MacroFeedV4P40StartupPackageFullLayers);
  identity = mix(identity, static_cast<std::uint64_t>(kFullQkvRole));
  identity = mix(identity, static_cast<std::uint64_t>(kFullQkvInputLayout));
  identity = mix(identity, static_cast<std::uint64_t>(kFullQkvTacticIdentity));
  identity = mix(identity, static_cast<std::uint64_t>(kAttentionOutputRole));
  identity = mix(identity, static_cast<std::uint64_t>(kFullOutputInputLayout));
  identity = mix(identity,
                 static_cast<std::uint64_t>(kFullOutputTacticIdentity));

  for (std::size_t ordinal = 0U;
       ordinal < kSm87MacroFeedV4P40StartupPackageFullLayers; ++ordinal) {
    const auto fail = [&](const bool k_norm, const int error = 0) noexcept {
      if (failure_ordinal != nullptr) {
        *failure_ordinal = ordinal;
      }
      if (failure_k_norm != nullptr) {
        *failure_k_norm = k_norm;
      }
      if (cuda_error != nullptr) {
        *cuda_error = error;
      }
      return std::uint64_t{0U};
    };
    const std::size_t model_layer = full_attention_model_layer(ordinal);
    if (!model_layer_is_full_attention(model_layer)) {
      return fail(false);
    }
    identity = mix(identity, ordinal + 1U);
    identity = mix(identity, model_layer + 1U);
    for (const auto& [role, layout, tactic] :
         {std::tuple<Role, kernels::Sm87MacroFeedV4Fp8InputLayout,
                     kernels::Sm87MacroFeedV4Fp8Identity>{
              kFullQkvRole, kFullQkvInputLayout, kFullQkvTacticIdentity},
          std::tuple<Role, kernels::Sm87MacroFeedV4Fp8InputLayout,
                     kernels::Sm87MacroFeedV4Fp8Identity>{
              kAttentionOutputRole, kFullOutputInputLayout,
              kFullOutputTacticIdentity}}) {
      const std::size_t index =
          sm87_target_aot_complete_descriptor_ordinal(model_layer, role);
      if (index >= capabilities.size()) {
        return fail(false);
      }
      const auto& capability = capabilities[index];
      const auto packed_layout =
          kernels::sm87_target_aot_projection_packed_layout(role);
      const auto* const asset = capability.asset
                                    ? capability.asset->borrow_fp8_cuda_asset()
                                    : nullptr;
      const std::uint64_t asset_identity =
          asset == nullptr ? 0U
                           : compute_gdn_qkvz_asset_value_identity(*asset);
      if (!packed_layout.valid() || asset_identity == 0U ||
          capability.asset->borrow_nvfp4_cuda_asset() != nullptr ||
          capability.layer_index != model_layer || capability.role != role ||
          capability.encoding != packed_layout.encoding ||
          capability.payload_bytes != packed_layout.payload_bytes ||
          capability.source_count != packed_layout.partition_count ||
          asset->payload.role != role ||
          asset->artifact_identity != capability.artifact_identity ||
          asset->source_inventory_identity !=
              capability.source_inventory_identity ||
          asset->host_manifest_seal.value != capability.manifest_seal ||
          asset->device_upload_receipt.receipt_identity !=
              capability.upload_receipt_identity ||
          asset->host_payload_digest != capability.payload_digest ||
          asset->payload.begin != capability.payload_begin ||
          asset->payload.end != capability.payload_end ||
          asset->payload.bytes != capability.payload_bytes ||
          asset->tensor_scale_count != capability.source_count ||
          expected_consumer_tactic_identity(model_layer, role) !=
              static_cast<std::uint64_t>(tactic) ||
          !authenticated_upload_complete(
              asset->device_upload_receipt, access.owner_identity(),
              access.allocation_identity(), access.device_ordinal(),
              asset->payload.begin, asset->payload.end,
              asset->payload.bytes)) {
        return fail(false);
      }
      for (std::size_t source = 0U; source < capability.source_count;
           ++source) {
        if (asset->tensor_scale_bits[source] !=
            capability.tensor_scale_bits[source]) {
          return fail(false);
        }
      }
      identity = mix(identity, static_cast<std::uint64_t>(role));
      identity = mix(identity, static_cast<std::uint64_t>(layout));
      identity = mix(identity, static_cast<std::uint64_t>(tactic));
      identity = mix(identity, asset_identity);
    }

    FullAttentionQkNormSource norms;
    if (!exact_full_attention_qk_norm_source(model_weights, model_layer,
                                             &norms)) {
      return fail(false);
    }
    const std::array<const std::uint16_t*, 2U> weights{{norms.q_norm,
                                                        norms.k_norm}};
    std::array<std::uint64_t, 2U> norm_identities{};
    for (std::size_t norm_index = 0U; norm_index < weights.size();
         ++norm_index) {
      int range_error = 0;
      if (!live_current_device_allocation_range(
              weights[norm_index], kNormBytes, access.device_ordinal(),
              &range_error)) {
        return fail(norm_index == 1U, range_error);
      }
      const std::size_t flat = 2U * ordinal + norm_index;
      const std::uintptr_t begin =
          reinterpret_cast<std::uintptr_t>(weights[norm_index]);
      if (begin > std::numeric_limits<std::uintptr_t>::max() - kNormBytes) {
        return fail(norm_index == 1U,
                    static_cast<int>(cudaErrorInvalidValue));
      }
      norm_begins[flat] = begin;
      norm_ends[flat] = begin + kNormBytes;
      for (std::size_t earlier = 0U; earlier < flat; ++earlier) {
        if (!(norm_ends[earlier] <= norm_begins[flat] ||
              norm_ends[flat] <= norm_begins[earlier])) {
          return fail(norm_index == 1U,
                      static_cast<int>(cudaErrorInvalidValue));
        }
      }
      norm_identities[norm_index] =
          full_attention_qk_norm_tensor_identity(
              0x5133'4d46'4653'4e44ULL, plan_identity,
              access.device_identity(), access.device_ordinal(), ordinal,
              model_layer, norm_index == 1U, weights[norm_index]);
      if (norm_identities[norm_index] == 0U) {
        return fail(norm_index == 1U);
      }
    }
    const std::uint64_t pair_identity =
        full_attention_qk_norm_pair_identity(
            ordinal, model_layer, norm_identities[0], norm_identities[1]);
    if (pair_identity == 0U) {
      return fail(false);
    }
    identity = mix(identity, norm_identities[0]);
    identity = mix(identity, norm_identities[1]);
    identity = mix(identity, pair_identity);
  }
  return identity == 0U ? 0x5133'4d46'4653'4341ULL : identity;
}

std::uint64_t Sm87MacroFeedV4P40StartupPackage::compute_package_identity(
    const ProjectionAccess& access,
    const std::array<AssetCapability,
                     kSm87MacroFeedV4P40StartupPackageArtifacts>&
        capabilities,
    const Sm87MacroFeedV4PanelWavefrontPlan& plan,
    const kernels::Sm87MacroFeedV4NvFp4GateUpCudaResources& gate_up,
    const kernels::Sm87MacroFeedV4NvFp4DownCudaResources& down,
    const std::uint64_t bf16_ab_binding_catalog_identity,
    const kernels::Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot&
        bf16_ab,
    const std::uint64_t mlp_pair_binding_catalog_identity,
    const std::uint64_t gdn_qkvz_binding_catalog_identity,
    const std::uint64_t full_attention_source_catalog_identity,
    const std::uint64_t request_boundary_source_catalog_identity,
    const kernels::Sm87MacroFeedV4Fp8CudaResources& gdn_qkvz,
    const kernels::Sm87MacroFeedV4Fp8CudaResources& gdn_output,
    const std::size_t sources) noexcept {
  const std::uint64_t plan_identity =
      compute_deployment_plan_identity(plan);
  const std::uint64_t catalog_identity = access.catalog_identity();
  if (!access.attached() || access.owner_identity() == 0U ||
      access.allocation_identity() == 0U || access.device_identity() == 0U ||
      access.device_ordinal() < 0 || catalog_identity == 0U ||
      plan_identity == 0U || bf16_ab_binding_catalog_identity == 0U ||
      mlp_pair_binding_catalog_identity == 0U ||
      gdn_qkvz_binding_catalog_identity == 0U ||
      full_attention_source_catalog_identity == 0U ||
      request_boundary_source_catalog_identity == 0U ||
      sources != kSm87MacroFeedV4P40StartupPackageSources ||
      !gate_up.static_resource_gate_passed ||
      !down.static_resource_gate_passed ||
      !kernels::sm87_macrofeed_v4_nvfp4_gate_up_resource_gate(gate_up) ||
      !kernels::sm87_macrofeed_v4_nvfp4_down_resource_gate(down) ||
      !kernels::sm87_macrofeed_v4_bf16_ab_admission_resource_gate(
          bf16_ab) ||
      !gdn_qkvz.static_resource_gate_passed ||
      !kernels::sm87_macrofeed_v4_fp8_resource_gate(gdn_qkvz) ||
      gdn_qkvz.role != kGdnQkvZRole ||
      gdn_qkvz.input_layout != kGdnQkvZInputLayout ||
      gdn_qkvz.identity != kGdnQkvZTacticIdentity ||
      !gdn_output.static_resource_gate_passed ||
      !kernels::sm87_macrofeed_v4_fp8_resource_gate(gdn_output) ||
      gdn_output.role != kAttentionOutputRole ||
      gdn_output.input_layout != kGdnOutputInputLayout ||
      gdn_output.identity != kGdnOutputTacticIdentity ||
      gate_up.device_ordinal != access.device_ordinal() ||
      down.device_ordinal != access.device_ordinal() ||
      bf16_ab.device_ordinal != access.device_ordinal() ||
      gdn_qkvz.device_ordinal != access.device_ordinal() ||
      gdn_output.device_ordinal != access.device_ordinal()) {
    return 0U;
  }

  std::uint64_t identity = 0x5133'4d46'5634'504bULL;
  for (const std::uint8_t byte : kSm87MacroFeedV4P40StartupPackageMagic) {
    identity = mix(identity, byte);
  }
  identity = mix(identity, kSm87MacroFeedV4P40StartupPackageAbiMajor);
  identity = mix(identity, kSm87MacroFeedV4P40StartupPackageAbiMinor);
  identity = mix(identity, plan_identity);
  identity = mix(identity, access.owner_identity());
  identity = mix(identity, access.allocation_identity());
  identity = mix(identity, catalog_identity);
  identity = mix(identity, access.device_identity());
  identity = mix(identity,
                 static_cast<std::uint64_t>(access.device_ordinal() + 1));
  identity = mix(identity, capabilities.size());
  identity = mix(identity, sources);
  identity = mix(identity, bf16_ab_binding_catalog_identity);
  identity = mix(identity, mlp_pair_binding_catalog_identity);
  identity = mix(identity, gdn_qkvz_binding_catalog_identity);
  identity = mix(identity, full_attention_source_catalog_identity);
  identity = mix(identity, request_boundary_source_catalog_identity);
  for (std::size_t index = 0U; index < capabilities.size(); ++index) {
    const auto& capability = capabilities[index];
    if (!capability.asset || capability.artifact_identity == 0U ||
        capability.source_inventory_identity == 0U ||
        capability.manifest_seal == 0U ||
        capability.upload_receipt_identity == 0U ||
        kernels::sm87_target_aot_projection_digest_is_zero(
            capability.payload_digest) ||
        capability.payload_begin == 0U || capability.payload_bytes == 0U ||
        capability.payload_end <= capability.payload_begin ||
        capability.source_count == 0U) {
      return 0U;
    }
    identity = mix(identity, index + 1U);
    identity = mix(identity, capability.layer_index + 1U);
    identity = mix(identity, static_cast<std::uint64_t>(capability.role));
    identity = mix(identity,
                   static_cast<std::uint64_t>(capability.encoding));
    identity = mix(identity, capability.artifact_identity);
    identity = mix(identity, capability.source_inventory_identity);
    identity = mix(identity, capability.manifest_seal);
    identity = mix(identity, capability.upload_receipt_identity);
    identity = mix_digest(identity, capability.payload_digest);
    identity = mix(identity, capability.payload_begin);
    identity = mix(identity, capability.payload_end);
    identity = mix(identity, capability.payload_bytes);
    identity = mix(identity, capability.source_count);
    for (const std::uint32_t bits : capability.tensor_scale_bits) {
      identity = mix(identity, bits);
    }
  }
  identity = mix(identity, static_cast<std::uint64_t>(gate_up.identity));
  identity = mix(identity, static_cast<std::uint64_t>(gate_up.device_ordinal + 1));
  identity = mix(identity, static_cast<std::uint64_t>(gate_up.compute_major));
  identity = mix(identity, static_cast<std::uint64_t>(gate_up.compute_minor));
  identity = mix(identity, static_cast<std::uint64_t>(gate_up.sm_count));
  identity = mix(identity, static_cast<std::uint64_t>(gate_up.binary_version));
  identity = mix(identity,
                 static_cast<std::uint64_t>(gate_up.registers_per_thread));
  identity = mix(identity, gate_up.static_shared_bytes);
  identity = mix(identity, gate_up.dynamic_shared_bytes);
  identity = mix(identity, gate_up.local_bytes);
  identity = mix(
      identity,
      static_cast<std::uint64_t>(gate_up.maximum_threads_per_block));
  identity = mix(identity,
                 static_cast<std::uint64_t>(gate_up.active_blocks_per_sm));
  identity = mix(identity, gate_up.kernel_compiled);
  identity = mix(identity, gate_up.static_resource_gate_passed);
  identity = mix(identity, gate_up.numerical_contract_qualified);
  identity = mix(identity, gate_up.production_dispatch_eligible);
  identity = mix(identity, static_cast<std::uint64_t>(down.identity));
  identity = mix(identity, static_cast<std::uint64_t>(down.device_ordinal + 1));
  identity = mix(identity, static_cast<std::uint64_t>(down.compute_major));
  identity = mix(identity, static_cast<std::uint64_t>(down.compute_minor));
  identity = mix(identity, static_cast<std::uint64_t>(down.sm_count));
  identity = mix(identity, static_cast<std::uint64_t>(down.binary_version));
  identity = mix(identity,
                 static_cast<std::uint64_t>(down.registers_per_thread));
  identity = mix(identity, down.static_shared_bytes);
  identity = mix(identity, down.dynamic_shared_bytes);
  identity = mix(identity, down.local_bytes);
  identity = mix(
      identity,
      static_cast<std::uint64_t>(down.maximum_threads_per_block));
  identity = mix(identity,
                 static_cast<std::uint64_t>(down.active_blocks_per_sm));
  identity = mix(identity, down.shared_bytes_per_sm);
  identity = mix(identity, down.optin_shared_bytes_per_block);
  identity = mix(identity, down.kernel_compiled);
  identity = mix(identity, down.static_resource_gate_passed);
  identity = mix(identity, down.numerical_contract_qualified);
  identity = mix(identity, down.production_dispatch_eligible);
  identity = mix_bf16_ab_resource(identity, bf16_ab);
  identity = mix_fp8_resource(identity, gdn_qkvz);
  identity = mix_fp8_resource(identity, gdn_output);
  return identity == 0U ? 0x5133'4d46'5634'504bULL : identity;
}

Sm87MacroFeedV4P40StartupPackage::StartupSeals
Sm87MacroFeedV4P40StartupPackage::mint_startup_seals(
    const std::uint64_t package_identity,
    const std::uint64_t plan_identity,
    const kernels::Sm87MacroFeedV4NvFp4GateUpCudaResources& gate_up,
    const kernels::Sm87MacroFeedV4NvFp4DownCudaResources& down,
    const std::uint64_t bf16_ab_binding_catalog_identity,
    const kernels::Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot&
        bf16_ab,
    const std::uint64_t mlp_pair_binding_catalog_identity,
    const std::uint64_t gdn_qkvz_binding_catalog_identity,
    const std::uint64_t full_attention_source_catalog_identity,
    const RequestBoundarySourceCatalog& request_boundary_sources,
    const kernels::Sm87MacroFeedV4Fp8CudaResources& gdn_qkvz,
    const kernels::Sm87MacroFeedV4Fp8CudaResources& gdn_output) noexcept {
  StartupSeals seals;
  seals.gate_up.package_identity = package_identity;
  seals.gate_up.deployment_plan_identity = plan_identity;
  seals.gate_up.binding_catalog_identity = mlp_pair_binding_catalog_identity;
  seals.gate_up.binding_count = kSm87MacroFeedV4P40StartupPackageLayers;
  seals.gate_up.resources = gate_up;
  seals.gate_up.canonical_c8000_plan = true;
  seals.gate_up.issued_by_v4_package = true;
  seals.gate_up.caller_receipt_accepted = false;
  seals.gate_up.launcher_authority = false;
  seals.gate_up.production_dispatch_eligible = false;
  seals.gate_up.issuer_nonce_ = kGateUpSealIssuerNonce;
  seals.gate_up.seal_identity = gate_up_seal_identity(seals.gate_up);

  seals.down.package_identity = package_identity;
  seals.down.deployment_plan_identity = plan_identity;
  seals.down.binding_catalog_identity = mlp_pair_binding_catalog_identity;
  seals.down.binding_count = kSm87MacroFeedV4P40StartupPackageLayers;
  seals.down.resources = down;
  seals.down.canonical_c8000_plan = true;
  seals.down.issued_by_v4_package = true;
  seals.down.caller_receipt_accepted = false;
  seals.down.launcher_authority = false;
  seals.down.production_dispatch_eligible = false;
  seals.down.issuer_nonce_ = kDownSealIssuerNonce;
  seals.down.seal_identity = down_seal_identity(seals.down);

  seals.bf16_ab.package_identity = package_identity;
  seals.bf16_ab.deployment_plan_identity = plan_identity;
  seals.bf16_ab.binding_catalog_identity =
      bf16_ab_binding_catalog_identity;
  seals.bf16_ab.tensor_count =
      kSm87MacroFeedV4P40StartupPackageBf16AbTensors;
  seals.bf16_ab.pair_count =
      kSm87MacroFeedV4P40StartupPackageBf16AbPairs;
  seals.bf16_ab.resources = bf16_ab;
  seals.bf16_ab.canonical_natural_layer_order = true;
  seals.bf16_ab.canonical_a_then_b_role_order = true;
  seals.bf16_ab.complete_live_device_ranges = true;
  seals.bf16_ab.issued_by_v4_package = true;
  seals.bf16_ab.caller_resource_snapshot_accepted = false;
  seals.bf16_ab.raw_pointer_exposed = false;
  seals.bf16_ab.launcher_authority = false;
  seals.bf16_ab.production_dispatch_eligible = false;
  seals.bf16_ab.issuer_nonce_ = kBf16AbSealIssuerNonce;
  seals.bf16_ab.seal_identity = bf16_ab_seal_identity(seals.bf16_ab);

  seals.gdn_qkvz.package_identity = package_identity;
  seals.gdn_qkvz.deployment_plan_identity = plan_identity;
  seals.gdn_qkvz.binding_catalog_identity =
      gdn_qkvz_binding_catalog_identity;
  seals.gdn_qkvz.binding_count =
      kSm87MacroFeedV4P40StartupPackageGdnLayers;
  seals.gdn_qkvz.resources = gdn_qkvz;
  seals.gdn_qkvz.role = kGdnQkvZRole;
  seals.gdn_qkvz.input_layout = kGdnQkvZInputLayout;
  seals.gdn_qkvz.tactic_identity = kGdnQkvZTacticIdentity;
  seals.gdn_qkvz.output_resources = gdn_output;
  seals.gdn_qkvz.output_role = kAttentionOutputRole;
  seals.gdn_qkvz.output_input_layout = kGdnOutputInputLayout;
  seals.gdn_qkvz.output_tactic_identity = kGdnOutputTacticIdentity;
  seals.gdn_qkvz.canonical_natural_gdn_layer_order = true;
  seals.gdn_qkvz.role_layout_and_tactic_fixed = true;
  seals.gdn_qkvz.output_role_layout_and_tactic_fixed = true;
  seals.gdn_qkvz.continuation_weights_execution_seal_required = true;
  seals.gdn_qkvz.typed_asset_values_private = true;
  seals.gdn_qkvz.caller_resource_snapshot_accepted = false;
  seals.gdn_qkvz.raw_pointer_exposed = false;
  seals.gdn_qkvz.launcher_authority = false;
  seals.gdn_qkvz.production_dispatch_eligible = false;
  seals.gdn_qkvz.issuer_nonce_ = kGdnQkvZSealIssuerNonce;
  seals.gdn_qkvz.seal_identity =
      gdn_qkvz_seal_identity(seals.gdn_qkvz);

  seals.full_attention.package_identity = package_identity;
  seals.full_attention.deployment_plan_identity = plan_identity;
  seals.full_attention.source_catalog_identity =
      full_attention_source_catalog_identity;
  seals.full_attention.binding_count =
      kSm87MacroFeedV4P40StartupPackageFullLayers;
  seals.full_attention.qkv_role = kFullQkvRole;
  seals.full_attention.qkv_input_layout = kFullQkvInputLayout;
  seals.full_attention.qkv_tactic_identity = kFullQkvTacticIdentity;
  seals.full_attention.output_role = kAttentionOutputRole;
  seals.full_attention.output_input_layout = kFullOutputInputLayout;
  seals.full_attention.output_tactic_identity = kFullOutputTacticIdentity;
  seals.full_attention.q_norm_elements =
      kernels::kSm87MacroFeedV4FullAttentionPreprocessHeadDimension;
  seals.full_attention.k_norm_elements =
      kernels::kSm87MacroFeedV4FullAttentionPreprocessHeadDimension;
  seals.full_attention.norm_weight_bytes =
      kernels::kSm87MacroFeedV4FullAttentionPreprocessNormWeightBytes;
  seals.full_attention.canonical_natural_full_layer_order = true;
  seals.full_attention.role_layout_and_tactics_fixed = true;
  seals.full_attention.qk_norm_exact_shapes = true;
  seals.full_attention.qk_norm_live_device_ranges = true;
  seals.full_attention.typed_asset_values_private = true;
  seals.full_attention.observed_resource_execution_seal_deferred = true;
  seals.full_attention.caller_resource_snapshot_accepted = false;
  seals.full_attention.raw_pointer_exposed = false;
  seals.full_attention.launcher_authority = false;
  seals.full_attention.production_dispatch_eligible = false;
  seals.full_attention.issuer_nonce_ = kFullAttentionSealIssuerNonce;
  seals.full_attention.seal_identity =
      full_attention_source_seal_identity(seals.full_attention);

  seals.request_boundary.package_identity = package_identity;
  seals.request_boundary.deployment_plan_identity = plan_identity;
  seals.request_boundary.source_catalog_identity =
      request_boundary_sources.catalog_identity;
  seals.request_boundary.resident_root_identity =
      request_boundary_sources.resident_root_identity;
  seals.request_boundary.resident_arena_bytes =
      request_boundary_sources.resident_arena_bytes;
  seals.request_boundary.binding_count =
      kSm87MacroFeedV4P40StartupPackageRequestBoundaryBindings;
  seals.request_boundary.device_ordinal = request_boundary_sources.device_ordinal;
  seals.request_boundary.final_norm_epsilon_fp32_bits =
      request_boundary_sources.final_norm.epsilon_fp32_bits;
  seals.request_boundary.weight_scale_2_fp32_bits =
      request_boundary_sources.lm_head.weight_scale_2_fp32_bits;
  seals.request_boundary.input_scale_fp32_bits =
      request_boundary_sources.lm_head.input_scale_fp32_bits;
  seals.request_boundary.embedding_exact_bf16_shape = true;
  seals.request_boundary.final_norm_exact_bf16_shape_and_epsilon = true;
  seals.request_boundary.lm_head_exact_canonical_nvfp4_shape = true;
  seals.request_boundary.device_scale_raw_bits_match_host =
      request_boundary_sources.device_scale_raw_bits_match_host;
  seals.request_boundary.input_scale_provenance_retained = true;
  seals.request_boundary.input_scale_consumed = false;
  seals.request_boundary.greedy_spec_exact = true;
  seals.request_boundary.complete_live_device_ranges = true;
  seals.request_boundary.observed_resource_execution_seal_deferred = true;
  seals.request_boundary.final_representation_ready_diagnostic_only = true;
  seals.request_boundary.pure_prefill_state_committed_endpoint_unchanged =
      true;
  seals.request_boundary.normal_resident_authority =
      request_boundary_sources.normal_resident_authority;
  seals.request_boundary.host_test_resident_authority =
      request_boundary_sources.host_test_resident_authority;
  seals.request_boundary.issued_by_v4_package = true;
  seals.request_boundary.caller_resource_snapshot_accepted = false;
  seals.request_boundary.raw_pointer_exposed = false;
  seals.request_boundary.launcher_authority = false;
  seals.request_boundary.production_dispatch_eligible = false;
  seals.request_boundary.issuer_nonce_ = kRequestBoundarySealIssuerNonce;
  seals.request_boundary.seal_identity =
      request_boundary_source_seal_identity(seals.request_boundary);
  return seals;
}

bool Sm87MacroFeedV4P40StartupPackage::startup_seals_valid(
    const StartupSeals& seals, const std::uint64_t package_identity,
    const std::uint64_t plan_identity,
    const std::int32_t device_ordinal) noexcept {
  return package_identity != 0U && plan_identity != 0U &&
         device_ordinal >= 0 && seals.gate_up.valid() &&
         seals.down.valid() && seals.bf16_ab.valid() &&
         seals.gdn_qkvz.valid() && seals.full_attention.valid() &&
         seals.request_boundary.valid() &&
         seals.gate_up.package_identity == package_identity &&
         seals.down.package_identity == package_identity &&
         seals.bf16_ab.package_identity == package_identity &&
         seals.gdn_qkvz.package_identity == package_identity &&
         seals.full_attention.package_identity == package_identity &&
         seals.request_boundary.package_identity == package_identity &&
         seals.gate_up.deployment_plan_identity == plan_identity &&
         seals.down.deployment_plan_identity == plan_identity &&
         seals.bf16_ab.deployment_plan_identity == plan_identity &&
         seals.gdn_qkvz.deployment_plan_identity == plan_identity &&
         seals.full_attention.deployment_plan_identity == plan_identity &&
         seals.request_boundary.deployment_plan_identity == plan_identity &&
         seals.gate_up.binding_catalog_identity != 0U &&
         seals.gate_up.binding_catalog_identity ==
             seals.down.binding_catalog_identity &&
         seals.gate_up.binding_count ==
             kSm87MacroFeedV4P40StartupPackageLayers &&
         seals.down.binding_count ==
             kSm87MacroFeedV4P40StartupPackageLayers &&
         seals.bf16_ab.binding_catalog_identity != 0U &&
         seals.gdn_qkvz.binding_catalog_identity != 0U &&
         seals.full_attention.source_catalog_identity != 0U &&
         seals.full_attention.binding_count ==
             kSm87MacroFeedV4P40StartupPackageFullLayers &&
         seals.request_boundary.source_catalog_identity != 0U &&
         seals.request_boundary.binding_count ==
             kSm87MacroFeedV4P40StartupPackageRequestBoundaryBindings &&
         seals.gate_up.resources.device_ordinal == device_ordinal &&
         seals.down.resources.device_ordinal == device_ordinal &&
         seals.bf16_ab.resources.device_ordinal == device_ordinal &&
         seals.gdn_qkvz.resources.device_ordinal == device_ordinal &&
         seals.gdn_qkvz.output_resources.device_ordinal == device_ordinal &&
         seals.request_boundary.device_ordinal == device_ordinal;
}

Sm87MacroFeedV4P40StartupPackageCreateResult
Sm87MacroFeedV4P40StartupPackage::build_from_private_authority(
    ProjectionAccess access, Sm87MacroFeedV4PanelWavefrontPlan plan,
    kernels::Sm87MacroFeedV4NvFp4GateUpCudaResources gate_up,
    kernels::Sm87MacroFeedV4NvFp4DownCudaResources down,
    kernels::Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot bf16_ab,
    kernels::Sm87MacroFeedV4Fp8CudaResources gdn_qkvz,
    kernels::Sm87MacroFeedV4Fp8CudaResources gdn_output,
    const ModelWeights& model_weights) noexcept {
  CreateResult result;
  if (!access.attached() ||
      access.artifact_count() != kSm87MacroFeedV4P40StartupPackageArtifacts) {
    result.status =
        failure(Error::kProjectionAttachment, "projection_attachment");
    return result;
  }
  const std::uint64_t owner_identity = access.owner_identity();
  const std::uint64_t allocation_identity = access.allocation_identity();
  const std::uint64_t catalog_identity = access.catalog_identity();
  const std::uint64_t device_identity = access.device_identity();
  const std::int32_t device_ordinal = access.device_ordinal();
  const std::uint64_t plan_identity =
      compute_deployment_plan_identity(plan);
  if (owner_identity == 0U || allocation_identity == 0U ||
      catalog_identity == 0U || device_identity == 0U ||
      device_ordinal < 0 || plan_identity == 0U) {
    result.status = failure(Error::kProjectionCatalog, "projection_catalog");
    return result;
  }
  if (gate_up.device_ordinal != device_ordinal ||
      down.device_ordinal != device_ordinal ||
      bf16_ab.device_ordinal != device_ordinal ||
      gdn_qkvz.device_ordinal != device_ordinal ||
      gdn_output.device_ordinal != device_ordinal) {
    result.status = failure(Error::kDeviceMismatch, "startup_seal_device");
    return result;
  }
  if (!kernels::sm87_macrofeed_v4_bf16_ab_admission_resource_gate(
          bf16_ab)) {
    result.status =
        failure(Error::kBf16AbResourceSeal, "bf16_ab_resource_gate");
    return result;
  }
  if (gdn_qkvz.role != kGdnQkvZRole ||
      gdn_qkvz.input_layout != kGdnQkvZInputLayout ||
      gdn_qkvz.identity != kGdnQkvZTacticIdentity ||
      !kernels::sm87_macrofeed_v4_fp8_resource_gate(gdn_qkvz)) {
    result.status = failure(Error::kGdnQkvZResourceSeal,
                            "gdn_qkvz_resource_gate");
    return result;
  }
  if (gdn_output.role != kAttentionOutputRole ||
      gdn_output.input_layout != kGdnOutputInputLayout ||
      gdn_output.identity != kGdnOutputTacticIdentity ||
      !kernels::sm87_macrofeed_v4_fp8_resource_gate(gdn_output)) {
    result.status = failure(Error::kGdnQkvZResourceSeal,
                            "gdn_output_resource_gate", 0,
                            kSm87MacroFeedV4P40StartupPackageLayers,
                            kAttentionOutputRole);
    return result;
  }

  std::array<Bf16AbPair,
             kSm87MacroFeedV4P40StartupPackageBf16AbPairs>
      bf16_ab_pairs{};
  Sm87MacroFeedV4Bf16AbT0InventoryAudit bf16_ab_inventory;
  int bf16_ab_cuda_error = 0;
  std::size_t bf16_ab_failure_layer =
      kSm87MacroFeedV4P40StartupPackageLayers;
  if (!build_bf16_ab_pairs(model_weights, device_ordinal, &bf16_ab_pairs,
                           &bf16_ab_inventory, &bf16_ab_cuda_error,
                           &bf16_ab_failure_layer)) {
    result.status = failure(
        bf16_ab_cuda_error == 0 ? Error::kBf16AbModelInventory
                               : Error::kBf16AbDeviceRange,
        bf16_ab_cuda_error == 0 ? "bf16_ab_model_inventory"
                                : "bf16_ab_live_device_range",
        bf16_ab_cuda_error, bf16_ab_failure_layer);
    return result;
  }
  if (bf16_ab_inventory.catalog_identity == 0U ||
      bf16_ab_inventory.tensors !=
          kSm87MacroFeedV4P40StartupPackageBf16AbTensors ||
      bf16_ab_inventory.pairs !=
          kSm87MacroFeedV4P40StartupPackageBf16AbPairs ||
      !bf16_ab_inventory.canonical_natural_layer_order ||
      !bf16_ab_inventory.canonical_a_then_b_role_order ||
      !bf16_ab_inventory.exact_bf16_shapes ||
      !bf16_ab_inventory.nonnull_16b_aligned_disjoint_ranges ||
      !bf16_ab_inventory.live_cuda_device_ranges_validated ||
      bf16_ab_inventory.execution_capability) {
    result.status =
        failure(Error::kBf16AbModelInventory, "bf16_ab_inventory_audit");
    return result;
  }

  std::array<AssetCapability, kSm87MacroFeedV4P40StartupPackageArtifacts>
      capabilities{};
  std::array<std::uint64_t, kSm87MacroFeedV4P40StartupPackageArtifacts>
      artifact_identities{};
  std::array<std::uint64_t, kSm87MacroFeedV4P40StartupPackageArtifacts>
      inventory_identities{};
  std::size_t artifacts = 0U;
  std::size_t sources = 0U;
  std::size_t gate_up_assets = 0U;
  std::size_t down_assets = 0U;
  std::size_t gdn_assets = 0U;
  std::size_t full_assets = 0U;
  std::size_t output_assets = 0U;

  for (std::size_t layer_index = 0U;
       layer_index < kSm87MacroFeedV4P40StartupPackageLayers;
       ++layer_index) {
    for (const Role role : layer_roles(layer_index)) {
      const std::size_t index =
          sm87_target_aot_complete_descriptor_ordinal(layer_index, role);
      if (index != artifacts || index >= capabilities.size()) {
        result.status = failure(Error::kProjectionInventory,
                                "projection_descriptor_order", 0,
                                layer_index, role);
        return result;
      }
      auto asset = access.resolve(layer_index, role);
      const auto layout =
          kernels::sm87_target_aot_projection_packed_layout(role);
      if (!asset || !layout.valid() ||
          asset->layer_index() != layer_index || asset->role() != role ||
          asset->artifact_identity() == 0U ||
          asset->source_inventory_identity() == 0U ||
          asset->encoding() != layout.encoding ||
          asset->payload_bytes() != layout.payload_bytes ||
          std::find(artifact_identities.begin(),
                    artifact_identities.begin() + artifacts,
                    asset->artifact_identity()) !=
              artifact_identities.begin() + artifacts ||
          std::find(inventory_identities.begin(),
                    inventory_identities.begin() + artifacts,
                    asset->source_inventory_identity()) !=
              inventory_identities.begin() + artifacts) {
        result.status = failure(Error::kProjectionInventory,
                                "projection_asset_identity", 0,
                                layer_index, role);
        return result;
      }

      AssetCapability capability;
      capability.layer_index = layer_index;
      capability.role = role;
      capability.encoding = asset->encoding();
      capability.artifact_identity = asset->artifact_identity();
      capability.source_inventory_identity =
          asset->source_inventory_identity();
      capability.payload_bytes = asset->payload_bytes();

      bool typed_borrow_valid = false;
      if (sm87_target_aot_complete_role_is_nvfp4(role)) {
        const auto* const view = asset->borrow_nvfp4_cuda_asset();
        if (view != nullptr && asset->borrow_fp8_cuda_asset() == nullptr &&
            kernels::sm87_target_aot_nvfp4_cuda_asset_valid(*view) &&
            view->artifact_identity == capability.artifact_identity &&
            view->source_inventory_identity ==
                capability.source_inventory_identity &&
            view->tensor_scale_count == layout.partition_count &&
            view->tensor_scale_count <= capability.tensor_scale_bits.size() &&
            authenticated_upload_complete(
                view->device_upload_receipt, owner_identity,
                allocation_identity, device_ordinal, view->payload.begin,
                view->payload.end, view->payload.bytes)) {
          capability.manifest_seal = view->host_manifest_seal.value;
          capability.upload_receipt_identity =
              view->device_upload_receipt.receipt_identity;
          capability.payload_digest = view->host_payload_digest;
          capability.payload_begin = view->payload.begin;
          capability.payload_end = view->payload.end;
          capability.source_count = view->tensor_scale_count;
          for (std::size_t source = 0U;
               source < capability.source_count; ++source) {
            capability.tensor_scale_bits[source] =
                view->tensor_scale_bits[source];
          }
          typed_borrow_valid = true;
        }
      } else if (sm87_target_aot_complete_role_is_fp8(role)) {
        const auto* const view = asset->borrow_fp8_cuda_asset();
        if (view != nullptr && asset->borrow_nvfp4_cuda_asset() == nullptr &&
            kernels::sm87_target_aot_fp8_cuda_asset_valid(*view) &&
            view->artifact_identity == capability.artifact_identity &&
            view->source_inventory_identity ==
                capability.source_inventory_identity &&
            view->tensor_scale_count == layout.partition_count &&
            view->tensor_scale_count <= capability.tensor_scale_bits.size() &&
            authenticated_upload_complete(
                view->device_upload_receipt, owner_identity,
                allocation_identity, device_ordinal, view->payload.begin,
                view->payload.end, view->payload.bytes)) {
          capability.manifest_seal = view->host_manifest_seal.value;
          capability.upload_receipt_identity =
              view->device_upload_receipt.receipt_identity;
          capability.payload_digest = view->host_payload_digest;
          capability.payload_begin = view->payload.begin;
          capability.payload_end = view->payload.end;
          capability.source_count = view->tensor_scale_count;
          for (std::size_t source = 0U;
               source < capability.source_count; ++source) {
            capability.tensor_scale_bits[source] =
                view->tensor_scale_bits[source];
          }
          typed_borrow_valid = true;
        }
      }
      if (!typed_borrow_valid || capability.manifest_seal == 0U ||
          capability.upload_receipt_identity == 0U ||
          kernels::sm87_target_aot_projection_digest_is_zero(
              capability.payload_digest) ||
          capability.payload_begin == 0U ||
          capability.payload_end <= capability.payload_begin ||
          capability.payload_end - capability.payload_begin !=
              capability.payload_bytes ||
          capability.source_count != layout.partition_count) {
        result.status = failure(Error::kProjectionAssetBorrow,
                                "projection_asset_borrow", 0,
                                layer_index, role);
        return result;
      }

      capability.asset = std::move(*asset);
      capabilities[index] = std::move(capability);
      artifact_identities[artifacts] =
          capabilities[index].artifact_identity;
      inventory_identities[artifacts] =
          capabilities[index].source_inventory_identity;
      sources += capabilities[index].source_count;
      ++artifacts;
      if (role == Role::kNvFp4GateUp) {
        ++gate_up_assets;
      } else if (role == Role::kNvFp4Down) {
        ++down_assets;
      } else if (role == Role::kFp8GdnQkvZ) {
        ++gdn_assets;
      } else if (role == Role::kFp8FullQkv) {
        ++full_assets;
      } else if (role == Role::kFp8AttentionOutput) {
        ++output_assets;
      }
    }
  }
  if (artifacts != kSm87MacroFeedV4P40StartupPackageArtifacts ||
      sources != kSm87MacroFeedV4P40StartupPackageSources ||
      gate_up_assets != kSm87MacroFeedV4P40StartupPackageLayers ||
      down_assets != kSm87MacroFeedV4P40StartupPackageLayers ||
      gdn_assets != kSm87MacroFeedV4P40StartupPackageGdnLayers ||
      full_assets != kSm87MacroFeedV4P40StartupPackageFullLayers ||
      output_assets != kSm87MacroFeedV4P40StartupPackageLayers ||
      access.catalog_identity() != catalog_identity) {
    result.status =
        failure(Error::kProjectionInventory, "projection_inventory");
    return result;
  }

  const std::uint64_t mlp_pair_binding_catalog_identity =
      compute_mlp_pair_binding_catalog_identity(access, capabilities,
                                                plan_identity);
  if (mlp_pair_binding_catalog_identity == 0U) {
    result.status = failure(Error::kBindingConstruction,
                            "mlp_pair_catalog_seal");
    return result;
  }

  const std::uint64_t gdn_qkvz_binding_catalog_identity =
      compute_gdn_qkvz_binding_catalog_identity(
          access, capabilities, plan_identity, gdn_qkvz, gdn_output,
          model_weights);
  if (gdn_qkvz_binding_catalog_identity == 0U) {
    result.status =
        failure(Error::kGdnQkvZCatalogSeal, "gdn_qkvz_catalog_seal");
    return result;
  }

  std::size_t full_failure_ordinal =
      kSm87MacroFeedV4P40StartupPackageFullLayers;
  bool full_failure_k_norm = false;
  int full_cuda_error = 0;
  const std::uint64_t full_attention_source_catalog_identity =
      compute_full_attention_source_catalog_identity(
          access, capabilities, plan_identity, model_weights,
          &full_failure_ordinal, &full_failure_k_norm, &full_cuda_error);
  if (full_attention_source_catalog_identity == 0U) {
    const std::size_t failure_layer =
        full_failure_ordinal < kSm87MacroFeedV4P40StartupPackageFullLayers
            ? full_attention_model_layer(full_failure_ordinal)
            : kSm87MacroFeedV4P40StartupPackageLayers;
    result.status = failure(
        Error::kFullAttentionSourceCatalogSeal,
        full_cuda_error == 0 ? "full_attention_source_catalog"
                             : (full_failure_k_norm
                                    ? "full_attention_k_norm_live_range"
                                    : "full_attention_q_norm_live_range"),
        full_cuda_error, failure_layer,
        Role::kInvalid);
    return result;
  }

  RequestBoundarySourceCatalog request_boundary_sources;
  int request_boundary_cuda_error = 0;
  std::size_t request_boundary_failure_index = 6U;
  if (!build_request_boundary_source_catalog(
          model_weights, access, plan_identity, &request_boundary_sources,
          &request_boundary_cuda_error, &request_boundary_failure_index,
          nullptr)) {
    result.status = failure(
        Error::kRequestBoundarySourceCatalogSeal,
        request_boundary_cuda_error == 0
            ? "request_boundary_source_catalog"
            : "request_boundary_resident_live_range",
        request_boundary_cuda_error, request_boundary_failure_index);
    return result;
  }

  const std::uint64_t package_identity = compute_package_identity(
      access, capabilities, plan, gate_up, down,
      bf16_ab_inventory.catalog_identity, bf16_ab,
      mlp_pair_binding_catalog_identity,
      gdn_qkvz_binding_catalog_identity,
      full_attention_source_catalog_identity,
      request_boundary_sources.catalog_identity, gdn_qkvz, gdn_output,
      sources);
  StartupSeals seals = mint_startup_seals(
      package_identity, plan_identity, gate_up, down,
      bf16_ab_inventory.catalog_identity, bf16_ab,
      mlp_pair_binding_catalog_identity,
      gdn_qkvz_binding_catalog_identity,
      full_attention_source_catalog_identity, request_boundary_sources,
      gdn_qkvz, gdn_output);
  if (package_identity == 0U ||
      !startup_seals_valid(seals, package_identity, plan_identity,
                           device_ordinal)) {
    result.status = failure(Error::kPackageIdentity, "startup_seals");
    return result;
  }

  Sm87MacroFeedV4P40StartupPackageAudit audit;
  audit.magic = kSm87MacroFeedV4P40StartupPackageMagic;
  audit.abi_major = kSm87MacroFeedV4P40StartupPackageAbiMajor;
  audit.abi_minor = kSm87MacroFeedV4P40StartupPackageAbiMinor;
  audit.candidate_id = kSm87MacroFeedV4CandidateId;
  audit.deployment_plan_id = kSm87MacroFeedV4P40DeploymentPlanId;
  audit.deployment_plan_identity = plan_identity;
  audit.package_identity = package_identity;
  audit.owner_identity = owner_identity;
  audit.allocation_identity = allocation_identity;
  audit.catalog_identity = catalog_identity;
  audit.device_identity = device_identity;
  audit.device_ordinal = device_ordinal;
  audit.layers = kSm87MacroFeedV4P40StartupPackageLayers;
  audit.artifacts = artifacts;
  audit.sources = sources;
  audit.gate_up_assets = gate_up_assets;
  audit.down_assets = down_assets;
  audit.gdn_projection_assets = gdn_assets;
  audit.full_projection_assets = full_assets;
  audit.attention_output_assets = output_assets;
  audit.bf16_ab_tensors = bf16_ab_inventory.tensors;
  audit.bf16_ab_pairs = bf16_ab_inventory.pairs;
  audit.bf16_ab_binding_catalog_identity =
      bf16_ab_inventory.catalog_identity;
  audit.bf16_ab_resource_seal_identity = seals.bf16_ab.seal_identity;
  audit.gdn_qkvz_bindings = kSm87MacroFeedV4P40StartupPackageGdnLayers;
  audit.gdn_qkvz_binding_catalog_identity =
      gdn_qkvz_binding_catalog_identity;
  audit.gdn_qkvz_resource_seal_identity = seals.gdn_qkvz.seal_identity;
  audit.full_attention_source_bindings =
      kSm87MacroFeedV4P40StartupPackageFullLayers;
  audit.full_attention_source_catalog_identity =
      full_attention_source_catalog_identity;
  audit.full_attention_source_seal_identity =
      seals.full_attention.seal_identity;
  audit.request_boundary_source_bindings =
      kSm87MacroFeedV4P40StartupPackageRequestBoundaryBindings;
  audit.request_boundary_source_catalog_identity =
      request_boundary_sources.catalog_identity;
  audit.request_boundary_source_seal_identity =
      seals.request_boundary.seal_identity;
  audit.request_boundary_resident_root_identity =
      request_boundary_sources.resident_root_identity;
  audit.request_boundary_resident_arena_bytes =
      request_boundary_sources.resident_arena_bytes;
  audit.canonical_plan_generated_internally = true;
  audit.caller_plan_accepted = false;
  audit.complete_projection_access_retained = true;
  audit.catalog_revalidated = true;
  audit.typed_capabilities_retained = true;
  audit.authenticated_source_manifests_retained = true;
  audit.authenticated_upload_readback_retained = true;
  audit.projection_bindings_complete = true;
  audit.nvfp4_startup_seals_complete = true;
  audit.bf16_ab_nonowning_model_weights_dependency_bound = true;
  audit.bf16_ab_projection_owner_identity_retained = true;
  audit.bf16_ab_natural_layer_order_complete = true;
  audit.bf16_ab_a_then_b_roles_complete = true;
  audit.bf16_ab_live_device_ranges_complete = true;
  audit.bf16_ab_resource_seal_complete = true;
  audit.bf16_ab_private_capability_retained = true;
  audit.bf16_ab_raw_pointer_publicly_exposed = false;
  audit.gdn_qkvz_natural_layer_order_complete = true;
  audit.gdn_qkvz_role_layout_tactic_fixed = true;
  audit.gdn_qkvz_asset_value_snapshots_private = true;
  audit.gdn_qkvz_resource_seal_complete = true;
  audit.gdn_qkvz_raw_pointer_publicly_exposed = false;
  audit.full_attention_natural_layer_order_complete = true;
  audit.full_attention_role_layout_tactics_fixed = true;
  audit.full_attention_qk_norm_shapes_exact = true;
  audit.full_attention_qk_norm_live_device_ranges_complete = true;
  audit.full_attention_typed_asset_values_private = true;
  audit.full_attention_observed_resource_execution_catalog_sealed = false;
  audit.request_boundary_exact_shapes_and_specs = true;
  audit.request_boundary_live_device_ranges_complete = true;
  audit.request_boundary_device_scale_raw_bits_match_host = true;
  audit.request_boundary_input_scale_provenance_retained = true;
  audit.request_boundary_input_scale_consumed = false;
  audit.request_boundary_observed_resource_execution_catalog_sealed = false;
  audit.request_boundary_final_representation_ready_diagnostic_only = true;
  audit.pure_prefill_state_committed_endpoint_unchanged = true;
  audit.request_boundary_normal_resident_authority =
      request_boundary_sources.normal_resident_authority;
  audit.request_boundary_host_test_resident_authority =
      request_boundary_sources.host_test_resident_authority;
  audit.request_boundary_raw_pointer_publicly_exposed = false;
  audit.caller_raw_receipts_accepted = false;
  audit.v3_execution_identity_reused = false;
  audit.request_time_repack_jit_autotune_or_fallback_permitted = false;
  audit.fp8_executor_bound = false;
  audit.gdn_executor_bound = false;
  audit.attention_executor_bound = false;
  audit.request_state_bound = false;
  audit.finalizer_bound = false;
  audit.physical_receipt_bound = false;
  audit.host_only = true;
  audit.default_off = true;
  audit.test_only = true;
  audit.selector_bound = false;
  audit.launcher_present = false;
  audit.execution_ready = false;
  audit.numerical_qualification_complete = false;
  audit.production_dispatch_eligible = false;
  if (!audit.valid()) {
    result.status = failure(Error::kPackageIdentity, "package_audit");
    return result;
  }

  Bf16AbStartupCapability bf16_ab_capability(
      &model_weights, std::move(bf16_ab_pairs), bf16_ab,
      bf16_ab_inventory.catalog_identity, package_identity, plan_identity,
      owner_identity, allocation_identity, catalog_identity, device_identity,
      device_ordinal, kBf16AbCapabilityIssuerNonce);
  if (!bf16_ab_capability.valid(access)) {
    result.status =
        failure(Error::kBf16AbDeviceRange, "bf16_ab_capability_revalidation");
    return result;
  }

  auto package = std::unique_ptr<Package>(new (std::nothrow) Package(
      std::move(access), std::move(capabilities),
      std::move(bf16_ab_capability), std::move(request_boundary_sources),
      std::move(plan), std::move(seals), audit));
  if (!package) {
    result.status = failure(Error::kAllocationFailure, "package_allocation");
    return result;
  }
  if (!package->populate_projection_bindings() || !package->valid()) {
    result.status =
        failure(Error::kBindingConstruction, "binding_revalidation");
    return result;
  }
  result.audit = package->audit();
  result.package = std::move(package);
  result.status = {};
  return result;
}

std::uint64_t Sm87MacroFeedV4ProjectionStartupBinding::
    compute_binding_identity(const Snapshot& snapshot) noexcept {
  const auto layout =
      kernels::sm87_target_aot_projection_packed_layout(snapshot.role);
  const std::uint64_t expected_tactic =
      expected_consumer_tactic_identity(snapshot.layer_index, snapshot.role);
  if (snapshot.package_identity == 0U ||
      snapshot.deployment_plan_identity == 0U ||
      snapshot.owner_identity == 0U ||
      snapshot.allocation_identity == 0U ||
      snapshot.catalog_identity == 0U || snapshot.device_identity == 0U ||
      snapshot.artifact_identity == 0U ||
      snapshot.source_inventory_identity == 0U ||
      snapshot.manifest_seal == 0U ||
      snapshot.upload_receipt_identity == 0U ||
      kernels::sm87_target_aot_projection_digest_is_zero(
          snapshot.payload_digest) ||
      snapshot.device_ordinal < 0 ||
      snapshot.layer_index >= kSm87MacroFeedV4P40StartupPackageLayers ||
      !layout.valid() || snapshot.encoding != layout.encoding ||
      snapshot.payload_begin == 0U ||
      snapshot.payload_bytes != layout.payload_bytes ||
      snapshot.payload_begin >
          std::numeric_limits<std::uintptr_t>::max() -
              snapshot.payload_bytes ||
      snapshot.payload_end != snapshot.payload_begin + snapshot.payload_bytes ||
      snapshot.source_count != layout.partition_count ||
      snapshot.source_count > snapshot.tensor_scale_bits.size() ||
      snapshot.consumer_tactic_identity != expected_tactic ||
      !snapshot.issued_from_live_complete_asset ||
      !snapshot.canonical_payload_layout_retained ||
      snapshot.caller_raw_receipt_accepted ||
      snapshot.v3_execution_identity_reused || !snapshot.t0_only ||
      snapshot.launcher_authority ||
      snapshot.production_dispatch_eligible) {
    return 0U;
  }
  for (std::size_t index = 0U;
       index < snapshot.tensor_scale_bits.size(); ++index) {
    if (index < snapshot.source_count) {
      if (!kernels::sm87_target_aot_projection_scale_bits_valid(
              snapshot.tensor_scale_bits[index])) {
        return 0U;
      }
    } else if (snapshot.tensor_scale_bits[index] != 0U) {
      return 0U;
    }
  }

  std::uint64_t identity = 0x5133'4d46'5634'424eULL;
  identity = mix(identity, snapshot.package_identity);
  identity = mix(identity, snapshot.deployment_plan_identity);
  identity = mix(identity, snapshot.owner_identity);
  identity = mix(identity, snapshot.allocation_identity);
  identity = mix(identity, snapshot.catalog_identity);
  identity = mix(identity, snapshot.device_identity);
  identity = mix(identity, snapshot.consumer_tactic_identity);
  identity = mix(identity, snapshot.artifact_identity);
  identity = mix(identity, snapshot.source_inventory_identity);
  identity = mix(identity, snapshot.manifest_seal);
  identity = mix(identity, snapshot.upload_receipt_identity);
  identity = mix_digest(identity, snapshot.payload_digest);
  identity = mix(identity, snapshot.payload_begin);
  identity = mix(identity, snapshot.payload_end);
  identity = mix(identity, snapshot.payload_bytes);
  identity = mix(identity,
                 static_cast<std::uint64_t>(snapshot.device_ordinal + 1));
  identity = mix(identity, snapshot.layer_index + 1U);
  identity = mix(identity, static_cast<std::uint64_t>(snapshot.role));
  identity = mix(identity, static_cast<std::uint64_t>(snapshot.encoding));
  identity = mix(identity, snapshot.source_count);
  for (const std::uint32_t bits : snapshot.tensor_scale_bits) {
    identity = mix(identity, bits);
  }
  identity = mix(identity, snapshot.issued_from_live_complete_asset);
  identity = mix(identity, snapshot.canonical_payload_layout_retained);
  identity = mix(identity, snapshot.caller_raw_receipt_accepted);
  identity = mix(identity, snapshot.v3_execution_identity_reused);
  identity = mix(identity, snapshot.t0_only);
  identity = mix(identity, snapshot.launcher_authority);
  identity = mix(identity, snapshot.production_dispatch_eligible);
  return identity == 0U ? 0x5133'4d46'5634'424eULL : identity;
}

std::uint32_t Sm87MacroFeedV4ProjectionStartupBinding::tensor_scale_bits(
    const std::size_t source_index) const noexcept {
  return source_index < snapshot_.source_count
             ? snapshot_.tensor_scale_bits[source_index]
             : 0U;
}

bool Sm87MacroFeedV4ProjectionStartupBinding::valid_with_catalog(
    const std::uint64_t catalog_identity) const noexcept {
  return catalog_identity != 0U && projection_access_.attached() &&
         projection_access_.catalog_identity() == catalog_identity &&
         valid_with_prevalidated_catalog(catalog_identity);
}

bool Sm87MacroFeedV4ProjectionStartupBinding::
    valid_with_prevalidated_catalog(
        const std::uint64_t catalog_identity) const noexcept {
  if (snapshot_.binding_identity == 0U ||
      snapshot_.binding_identity != compute_binding_identity(snapshot_) ||
      !projection_access_.attached() ||
      projection_access_.owner_identity() != snapshot_.owner_identity ||
      projection_access_.allocation_identity() !=
          snapshot_.allocation_identity ||
      catalog_identity != snapshot_.catalog_identity ||
      projection_access_.device_identity() != snapshot_.device_identity ||
      projection_access_.device_ordinal() != snapshot_.device_ordinal ||
      asset_.layer_index() != snapshot_.layer_index ||
      asset_.role() != snapshot_.role ||
      asset_.encoding() != snapshot_.encoding ||
      asset_.artifact_identity() != snapshot_.artifact_identity ||
      asset_.source_inventory_identity() !=
          snapshot_.source_inventory_identity ||
      asset_.payload_bytes() != snapshot_.payload_bytes) {
    return false;
  }
  auto fresh = projection_access_.resolve(snapshot_.layer_index,
                                          snapshot_.role);
  if (!fresh || fresh->artifact_identity() != snapshot_.artifact_identity ||
      fresh->source_inventory_identity() !=
          snapshot_.source_inventory_identity ||
      fresh->encoding() != snapshot_.encoding ||
      fresh->payload_bytes() != snapshot_.payload_bytes) {
    return false;
  }

  const auto common_view_matches = [&](const auto& view) noexcept {
    if (view.artifact_identity != snapshot_.artifact_identity ||
        view.source_inventory_identity !=
            snapshot_.source_inventory_identity ||
        view.host_manifest_seal.value != snapshot_.manifest_seal ||
        view.device_upload_receipt.receipt_identity !=
            snapshot_.upload_receipt_identity ||
        view.host_payload_digest != snapshot_.payload_digest ||
        view.payload.begin != snapshot_.payload_begin ||
        view.payload.end != snapshot_.payload_end ||
        view.payload.bytes != snapshot_.payload_bytes ||
        view.tensor_scale_count != snapshot_.source_count ||
        !authenticated_upload_complete(
            view.device_upload_receipt, snapshot_.owner_identity,
            snapshot_.allocation_identity, snapshot_.device_ordinal,
            snapshot_.payload_begin, snapshot_.payload_end,
            snapshot_.payload_bytes)) {
      return false;
    }
    for (std::size_t index = 0U; index < snapshot_.source_count; ++index) {
      if (view.tensor_scale_bits[index] !=
          snapshot_.tensor_scale_bits[index]) {
        return false;
      }
    }
    return true;
  };

  if (sm87_target_aot_complete_role_is_nvfp4(snapshot_.role)) {
    const auto* const view = asset_.borrow_nvfp4_cuda_asset();
    const auto* const fresh_view = fresh->borrow_nvfp4_cuda_asset();
    return view != nullptr && fresh_view != nullptr &&
           asset_.borrow_fp8_cuda_asset() == nullptr &&
           kernels::sm87_target_aot_nvfp4_cuda_asset_valid(*view) &&
           kernels::sm87_target_aot_nvfp4_cuda_asset_valid(*fresh_view) &&
           common_view_matches(*view) && common_view_matches(*fresh_view);
  }
  if (sm87_target_aot_complete_role_is_fp8(snapshot_.role)) {
    const auto* const view = asset_.borrow_fp8_cuda_asset();
    const auto* const fresh_view = fresh->borrow_fp8_cuda_asset();
    return view != nullptr && fresh_view != nullptr &&
           asset_.borrow_nvfp4_cuda_asset() == nullptr &&
           kernels::sm87_target_aot_fp8_cuda_asset_valid(*view) &&
           kernels::sm87_target_aot_fp8_cuda_asset_valid(*fresh_view) &&
           common_view_matches(*view) && common_view_matches(*fresh_view);
  }
  return false;
}

bool Sm87MacroFeedV4ProjectionStartupBinding::valid() const noexcept {
  if (!projection_access_.attached()) {
    return false;
  }
  const std::uint64_t catalog_identity =
      projection_access_.catalog_identity();
  return catalog_identity != 0U &&
         valid_with_prevalidated_catalog(catalog_identity);
}

bool Sm87MacroFeedV4ProjectionStartupBinding::valid_for(
    const std::size_t layer_index, const Role role,
    const std::uint64_t package_identity) const noexcept {
  if (!projection_access_.attached()) {
    return false;
  }
  const std::uint64_t catalog_identity =
      projection_access_.catalog_identity();
  return catalog_identity != 0U && valid_for_prevalidated_catalog(
                                       layer_index, role, package_identity,
                                       catalog_identity);
}

bool Sm87MacroFeedV4ProjectionStartupBinding::
    valid_for_prevalidated_catalog(
        const std::size_t layer_index, const Role role,
        const std::uint64_t package_identity,
        const std::uint64_t catalog_identity) const noexcept {
  return layer_index == snapshot_.layer_index && role == snapshot_.role &&
         package_identity != 0U &&
         package_identity == snapshot_.package_identity &&
         valid_with_prevalidated_catalog(catalog_identity);
}

const Sm87MacroFeedV4P40StartupPackage::AssetCapability*
Sm87MacroFeedV4P40StartupPackage::capability(
    const std::size_t layer_index, const Role role) const noexcept {
  const std::size_t index =
      sm87_target_aot_complete_descriptor_ordinal(layer_index, role);
  if (index >= capabilities_.size()) {
    return nullptr;
  }
  const auto& capability = capabilities_[index];
  return capability.asset && capability.layer_index == layer_index &&
                 capability.role == role &&
                 capability.artifact_identity != 0U &&
                 capability.source_inventory_identity != 0U &&
                 capability.manifest_seal != 0U &&
                 capability.upload_receipt_identity != 0U &&
                 capability.payload_begin != 0U &&
                 capability.payload_end > capability.payload_begin &&
                 capability.payload_bytes != 0U &&
                 capability.source_count != 0U
             ? &capability
             : nullptr;
}

std::optional<Sm87MacroFeedV4ProjectionStartupBinding>
Sm87MacroFeedV4P40StartupPackage::make_projection_binding(
    const std::size_t layer_index, const Role role) const noexcept {
  if (!audit_.valid() || !projection_access_.attached()) {
    return std::nullopt;
  }
  const AssetCapability* const retained = capability(layer_index, role);
  auto asset = projection_access_.resolve(layer_index, role);
  if (retained == nullptr || !asset || !retained->asset ||
      asset->artifact_identity() != retained->artifact_identity ||
      asset->source_inventory_identity() !=
          retained->source_inventory_identity ||
      asset->encoding() != retained->encoding ||
      asset->payload_bytes() != retained->payload_bytes) {
    return std::nullopt;
  }

  Sm87MacroFeedV4ProjectionStartupBinding::Snapshot snapshot;
  snapshot.package_identity = audit_.package_identity;
  snapshot.deployment_plan_identity = audit_.deployment_plan_identity;
  snapshot.owner_identity = audit_.owner_identity;
  snapshot.allocation_identity = audit_.allocation_identity;
  snapshot.catalog_identity = audit_.catalog_identity;
  snapshot.device_identity = audit_.device_identity;
  snapshot.consumer_tactic_identity =
      expected_consumer_tactic_identity(layer_index, role);
  snapshot.artifact_identity = retained->artifact_identity;
  snapshot.source_inventory_identity = retained->source_inventory_identity;
  snapshot.manifest_seal = retained->manifest_seal;
  snapshot.upload_receipt_identity = retained->upload_receipt_identity;
  snapshot.payload_digest = retained->payload_digest;
  snapshot.payload_begin = retained->payload_begin;
  snapshot.payload_end = retained->payload_end;
  snapshot.payload_bytes = retained->payload_bytes;
  snapshot.device_ordinal = audit_.device_ordinal;
  snapshot.layer_index = layer_index;
  snapshot.role = role;
  snapshot.encoding = retained->encoding;
  snapshot.tensor_scale_bits = retained->tensor_scale_bits;
  snapshot.source_count = retained->source_count;
  snapshot.issued_from_live_complete_asset = true;
  snapshot.canonical_payload_layout_retained = true;
  snapshot.caller_raw_receipt_accepted = false;
  snapshot.v3_execution_identity_reused = false;
  snapshot.t0_only = true;
  snapshot.launcher_authority = false;
  snapshot.production_dispatch_eligible = false;
  snapshot.binding_identity =
      Sm87MacroFeedV4ProjectionStartupBinding::compute_binding_identity(
          snapshot);
  if (snapshot.binding_identity == 0U) {
    return std::nullopt;
  }
  Sm87MacroFeedV4ProjectionStartupBinding binding(
      projection_access_, std::move(*asset), std::move(snapshot));
  if (!binding.valid_with_prevalidated_catalog(audit_.catalog_identity)) {
    return std::nullopt;
  }
  return std::optional<Sm87MacroFeedV4ProjectionStartupBinding>(
      std::move(binding));
}

bool Sm87MacroFeedV4P40StartupPackage::base_valid() const noexcept {
  const std::uint64_t plan_identity =
      compute_deployment_plan_identity(plan_);
  if (!audit_.valid() || !projection_access_.attached() ||
      projection_access_.owner_identity() != audit_.owner_identity ||
      projection_access_.allocation_identity() !=
          audit_.allocation_identity ||
      projection_access_.catalog_identity() != audit_.catalog_identity ||
      projection_access_.device_identity() != audit_.device_identity ||
      projection_access_.device_ordinal() != audit_.device_ordinal ||
      projection_access_.artifact_count() != audit_.artifacts ||
      plan_identity == 0U ||
      plan_identity != audit_.deployment_plan_identity ||
      !startup_seals_valid(seals_, audit_.package_identity, plan_identity,
                           audit_.device_ordinal) ||
      !bf16_ab_capability_.valid(projection_access_) ||
      bf16_ab_capability_.model_weights_ == nullptr ||
      bf16_ab_capability_.catalog_identity_ !=
          audit_.bf16_ab_binding_catalog_identity ||
      bf16_ab_capability_.package_identity_ != audit_.package_identity ||
      bf16_ab_capability_.deployment_plan_identity_ != plan_identity ||
      bf16_ab_capability_.projection_owner_identity_ !=
          audit_.owner_identity ||
      bf16_ab_capability_.projection_allocation_identity_ !=
          audit_.allocation_identity ||
      bf16_ab_capability_.projection_catalog_identity_ !=
          audit_.catalog_identity ||
      bf16_ab_capability_.projection_device_identity_ !=
          audit_.device_identity ||
      bf16_ab_capability_.device_ordinal_ != audit_.device_ordinal ||
      seals_.bf16_ab.binding_catalog_identity !=
          audit_.bf16_ab_binding_catalog_identity ||
      seals_.bf16_ab.seal_identity !=
          audit_.bf16_ab_resource_seal_identity ||
      !bf16_ab_resource_equal(seals_.bf16_ab.resources,
                              bf16_ab_capability_.resources_) ||
      seals_.gdn_qkvz.binding_count !=
          kSm87MacroFeedV4P40StartupPackageGdnLayers ||
      seals_.gdn_qkvz.binding_catalog_identity !=
          audit_.gdn_qkvz_binding_catalog_identity ||
      seals_.gdn_qkvz.seal_identity !=
          audit_.gdn_qkvz_resource_seal_identity ||
      seals_.gdn_qkvz.role != kGdnQkvZRole ||
      seals_.gdn_qkvz.input_layout != kGdnQkvZInputLayout ||
      seals_.gdn_qkvz.tactic_identity != kGdnQkvZTacticIdentity ||
      seals_.gdn_qkvz.resources.device_ordinal != audit_.device_ordinal ||
      seals_.gdn_qkvz.output_role != kAttentionOutputRole ||
      seals_.gdn_qkvz.output_input_layout != kGdnOutputInputLayout ||
      seals_.gdn_qkvz.output_tactic_identity != kGdnOutputTacticIdentity ||
      seals_.gdn_qkvz.output_resources.device_ordinal !=
          audit_.device_ordinal ||
      !kernels::sm87_macrofeed_v4_fp8_resource_gate(
          seals_.gdn_qkvz.resources) ||
      !kernels::sm87_macrofeed_v4_fp8_resource_gate(
          seals_.gdn_qkvz.output_resources) ||
      seals_.full_attention.binding_count !=
          kSm87MacroFeedV4P40StartupPackageFullLayers ||
      seals_.full_attention.source_catalog_identity !=
          audit_.full_attention_source_catalog_identity ||
      seals_.full_attention.seal_identity !=
          audit_.full_attention_source_seal_identity ||
      seals_.full_attention.qkv_role != kFullQkvRole ||
      seals_.full_attention.qkv_input_layout != kFullQkvInputLayout ||
      seals_.full_attention.qkv_tactic_identity != kFullQkvTacticIdentity ||
      seals_.full_attention.output_role != kAttentionOutputRole ||
      seals_.full_attention.output_input_layout != kFullOutputInputLayout ||
      seals_.full_attention.output_tactic_identity !=
          kFullOutputTacticIdentity ||
      seals_.request_boundary.source_catalog_identity !=
          audit_.request_boundary_source_catalog_identity ||
      seals_.request_boundary.seal_identity !=
          audit_.request_boundary_source_seal_identity ||
      seals_.request_boundary.resident_root_identity !=
          audit_.request_boundary_resident_root_identity ||
      seals_.request_boundary.resident_arena_bytes !=
          audit_.request_boundary_resident_arena_bytes ||
      request_boundary_sources_.model_weights !=
          bf16_ab_capability_.model_weights_ ||
      request_boundary_sources_.resident_root == nullptr ||
      request_boundary_sources_.catalog_identity !=
          audit_.request_boundary_source_catalog_identity ||
      request_boundary_sources_.resident_root_identity !=
          audit_.request_boundary_resident_root_identity ||
      request_boundary_sources_.resident_arena_bytes !=
          audit_.request_boundary_resident_arena_bytes ||
      request_boundary_sources_.device_ordinal != audit_.device_ordinal ||
      request_boundary_sources_.projection_owner_identity !=
          audit_.owner_identity ||
      request_boundary_sources_.projection_allocation_identity !=
          audit_.allocation_identity ||
      request_boundary_sources_.projection_catalog_identity !=
          audit_.catalog_identity ||
      request_boundary_sources_.projection_device_identity !=
          audit_.device_identity ||
      !request_boundary_sources_.exact_shapes_and_specs ||
      !request_boundary_sources_.complete_live_device_ranges ||
      !request_boundary_sources_.device_scale_raw_bits_match_host ||
      !request_boundary_sources_.input_scale_provenance_retained ||
      request_boundary_sources_.input_scale_consumed ||
      (request_boundary_sources_.normal_resident_authority ==
       request_boundary_sources_.host_test_resident_authority) ||
      seals_.gate_up.binding_catalog_identity == 0U ||
      seals_.gate_up.binding_catalog_identity !=
          seals_.down.binding_catalog_identity ||
      compute_mlp_pair_binding_catalog_identity(
          projection_access_, capabilities_, plan_identity) !=
          seals_.gate_up.binding_catalog_identity ||
      compute_gdn_qkvz_binding_catalog_identity(
          projection_access_, capabilities_, plan_identity,
          seals_.gdn_qkvz.resources,
          seals_.gdn_qkvz.output_resources,
          *bf16_ab_capability_.model_weights_) !=
          audit_.gdn_qkvz_binding_catalog_identity) {
    return false;
  }

  std::size_t full_failure_ordinal =
      kSm87MacroFeedV4P40StartupPackageFullLayers;
  bool full_failure_k_norm = false;
  int full_cuda_error = 0;
  if (compute_full_attention_source_catalog_identity(
          projection_access_, capabilities_, plan_identity,
          *bf16_ab_capability_.model_weights_, &full_failure_ordinal,
          &full_failure_k_norm, &full_cuda_error) !=
          audit_.full_attention_source_catalog_identity ||
      full_failure_ordinal !=
          kSm87MacroFeedV4P40StartupPackageFullLayers ||
      full_failure_k_norm || full_cuda_error != 0) {
    return false;
  }

  RequestBoundarySourceCatalog fresh_request_boundary_sources;
  int request_boundary_cuda_error = 0;
  std::size_t request_boundary_failure_index = 6U;
  if (!build_request_boundary_source_catalog(
          *bf16_ab_capability_.model_weights_, projection_access_,
          plan_identity, &fresh_request_boundary_sources,
          &request_boundary_cuda_error, &request_boundary_failure_index,
          &request_boundary_sources_) ||
      request_boundary_cuda_error != 0 || request_boundary_failure_index != 6U ||
      fresh_request_boundary_sources.catalog_identity !=
          request_boundary_sources_.catalog_identity ||
      fresh_request_boundary_sources.model_weights !=
          request_boundary_sources_.model_weights ||
      fresh_request_boundary_sources.resident_root !=
          request_boundary_sources_.resident_root ||
      fresh_request_boundary_sources.resident_root_identity !=
          request_boundary_sources_.resident_root_identity ||
      fresh_request_boundary_sources.resident_arena_begin !=
          request_boundary_sources_.resident_arena_begin ||
      fresh_request_boundary_sources.resident_arena_end !=
          request_boundary_sources_.resident_arena_end ||
      fresh_request_boundary_sources.resident_arena_bytes !=
          request_boundary_sources_.resident_arena_bytes ||
      fresh_request_boundary_sources.embedding.table !=
          request_boundary_sources_.embedding.table ||
      fresh_request_boundary_sources.embedding.source_identity !=
          request_boundary_sources_.embedding.source_identity ||
      fresh_request_boundary_sources.final_norm.centered_weight !=
          request_boundary_sources_.final_norm.centered_weight ||
      fresh_request_boundary_sources.final_norm.source_identity !=
          request_boundary_sources_.final_norm.source_identity ||
      fresh_request_boundary_sources.lm_head.canonical_packed_weight !=
          request_boundary_sources_.lm_head.canonical_packed_weight ||
      fresh_request_boundary_sources.lm_head.canonical_block_scale !=
          request_boundary_sources_.lm_head.canonical_block_scale ||
      fresh_request_boundary_sources.lm_head.weight_scale_2_device !=
          request_boundary_sources_.lm_head.weight_scale_2_device ||
      fresh_request_boundary_sources.lm_head.input_scale_device !=
          request_boundary_sources_.lm_head.input_scale_device ||
      fresh_request_boundary_sources.lm_head.source_identity !=
          request_boundary_sources_.lm_head.source_identity ||
      fresh_request_boundary_sources.greedy.spec_identity !=
          request_boundary_sources_.greedy.spec_identity ||
      fresh_request_boundary_sources.device_scale_raw_bits_match_host !=
          request_boundary_sources_.device_scale_raw_bits_match_host ||
      fresh_request_boundary_sources.normal_resident_authority !=
          request_boundary_sources_.normal_resident_authority ||
      fresh_request_boundary_sources.host_test_resident_authority !=
          request_boundary_sources_.host_test_resident_authority) {
    return false;
  }

  std::size_t artifacts = 0U;
  std::size_t sources = 0U;
  for (std::size_t layer_index = 0U;
       layer_index < kSm87MacroFeedV4P40StartupPackageLayers;
       ++layer_index) {
    for (const Role role : layer_roles(layer_index)) {
      const std::size_t index =
          sm87_target_aot_complete_descriptor_ordinal(layer_index, role);
      const AssetCapability* const retained =
          capability(layer_index, role);
      if (index != artifacts || retained == nullptr ||
          index >= capabilities_.size()) {
        return false;
      }
      auto fresh = projection_access_.resolve(layer_index, role);
      if (!fresh || fresh->artifact_identity() != retained->artifact_identity ||
          fresh->source_inventory_identity() !=
              retained->source_inventory_identity ||
          fresh->encoding() != retained->encoding ||
          fresh->payload_bytes() != retained->payload_bytes) {
        return false;
      }
      const auto live_view_matches = [&](const auto& view) noexcept {
        if (view.artifact_identity != retained->artifact_identity ||
            view.source_inventory_identity !=
                retained->source_inventory_identity ||
            view.host_manifest_seal.value != retained->manifest_seal ||
            view.device_upload_receipt.receipt_identity !=
                retained->upload_receipt_identity ||
            view.host_payload_digest != retained->payload_digest ||
            view.payload.begin != retained->payload_begin ||
            view.payload.end != retained->payload_end ||
            view.payload.bytes != retained->payload_bytes ||
            view.tensor_scale_count != retained->source_count ||
            !authenticated_upload_complete(
                view.device_upload_receipt, audit_.owner_identity,
                audit_.allocation_identity, audit_.device_ordinal,
                retained->payload_begin, retained->payload_end,
                retained->payload_bytes)) {
          return false;
        }
        for (std::size_t source = 0U; source < retained->source_count;
             ++source) {
          if (view.tensor_scale_bits[source] !=
              retained->tensor_scale_bits[source]) {
            return false;
          }
        }
        return true;
      };
      bool view_valid = false;
      if (sm87_target_aot_complete_role_is_nvfp4(role)) {
        const auto* const view = fresh->borrow_nvfp4_cuda_asset();
        view_valid =
            view != nullptr && fresh->borrow_fp8_cuda_asset() == nullptr &&
            kernels::sm87_target_aot_nvfp4_cuda_asset_valid(*view) &&
            live_view_matches(*view);
      } else {
        const auto* const view = fresh->borrow_fp8_cuda_asset();
        view_valid =
            view != nullptr && fresh->borrow_nvfp4_cuda_asset() == nullptr &&
            kernels::sm87_target_aot_fp8_cuda_asset_valid(*view) &&
            live_view_matches(*view);
      }
      if (!view_valid) {
        return false;
      }
      sources += retained->source_count;
      ++artifacts;
    }
  }
  return artifacts == audit_.artifacts && sources == audit_.sources &&
         audit_.package_identity == compute_package_identity(
                                        projection_access_, capabilities_,
                                        plan_, seals_.gate_up.resources,
                                        seals_.down.resources,
                                        audit_.bf16_ab_binding_catalog_identity,
                                        seals_.bf16_ab.resources,
                                        seals_.gate_up.binding_catalog_identity,
                                        audit_.gdn_qkvz_binding_catalog_identity,
                                        audit_.full_attention_source_catalog_identity,
                                        audit_.request_boundary_source_catalog_identity,
                                        seals_.gdn_qkvz.resources,
                                        seals_.gdn_qkvz.output_resources,
                                        sources);
}

std::uint64_t Sm87MacroFeedV4P40StartupPackage::
    compute_gdn_output_execution_binding_identity(
        const GdnOutputExecutionBinding& binding,
        const GdnLayerExecutionBinding& owner) noexcept {
  const auto& asset = binding.asset;
  const auto& upload = asset.device_upload_receipt;
  const std::uint64_t asset_identity =
      compute_gdn_qkvz_asset_value_identity(asset);
  if (owner.gdn_ordinal >= kSm87MacroFeedV4P40StartupPackageGdnLayers ||
      owner.model_layer != gdn_model_layer(owner.gdn_ordinal) ||
      binding.role != kAttentionOutputRole ||
      binding.input_layout != kGdnOutputInputLayout ||
      binding.tactic_identity != kGdnOutputTacticIdentity ||
      binding.resources.role != kAttentionOutputRole ||
      binding.resources.input_layout != kGdnOutputInputLayout ||
      binding.resources.identity != kGdnOutputTacticIdentity ||
      !binding.resources.static_resource_gate_passed ||
      !kernels::sm87_macrofeed_v4_fp8_resource_gate(binding.resources) ||
      binding.projection_binding_identity == 0U ||
      binding.asset_value_identity == 0U ||
      binding.asset_value_identity != asset_identity ||
      asset.payload.role != kAttentionOutputRole ||
      upload.role != kAttentionOutputRole ||
      upload.device_allocation_owner_identity != owner.owner_identity ||
      upload.device_allocation_identity != owner.allocation_identity ||
      upload.device_ordinal != binding.resources.device_ordinal ||
      !binding.live_cuda_payload_range_validated) {
    return 0U;
  }
  std::uint64_t identity = 0x5133'4d46'4744'4e4fULL;
  identity = mix(identity, owner.gdn_ordinal + 1U);
  identity = mix(identity, owner.model_layer + 1U);
  identity = mix(identity, static_cast<std::uint64_t>(binding.role));
  identity = mix(identity, static_cast<std::uint64_t>(binding.input_layout));
  identity = mix(identity, static_cast<std::uint64_t>(binding.tactic_identity));
  identity = mix(identity, binding.projection_binding_identity);
  identity = mix(identity, binding.asset_value_identity);
  identity = mix_fp8_resource(identity, binding.resources);
  return identity == 0U ? 0x5133'4d46'4744'4e4fULL : identity;
}

std::uint64_t Sm87MacroFeedV4P40StartupPackage::
    compute_gdn_qkvz_execution_binding_identity(
        const GdnLayerExecutionBinding& binding) noexcept {
  const auto& asset = binding.asset;
  const auto& upload = asset.device_upload_receipt;
  const std::uint64_t asset_value_identity =
      compute_gdn_qkvz_asset_value_identity(asset);
  std::uint64_t expected_continuation_identity =
      0x5133'4d46'4744'4e43ULL;
  for (const std::uint64_t item : {
           binding.continuation.conv_weight_identity,
           binding.continuation.a_log_identity,
           binding.continuation.dt_bias_identity,
           binding.continuation.norm_weight_identity}) {
    expected_continuation_identity = mix(expected_continuation_identity, item);
  }
  if (binding.gdn_ordinal >=
          kSm87MacroFeedV4P40StartupPackageGdnLayers ||
      binding.model_layer != gdn_model_layer(binding.gdn_ordinal) ||
      !model_layer_is_gdn(binding.model_layer) ||
      binding.role != kGdnQkvZRole ||
      binding.input_layout != kGdnQkvZInputLayout ||
      binding.tactic_identity != kGdnQkvZTacticIdentity ||
      binding.resources.role != kGdnQkvZRole ||
      binding.resources.input_layout != kGdnQkvZInputLayout ||
      binding.resources.identity != kGdnQkvZTacticIdentity ||
      !binding.resources.static_resource_gate_passed ||
      !kernels::sm87_macrofeed_v4_fp8_resource_gate(binding.resources) ||
      binding.package_identity == 0U ||
      binding.deployment_plan_identity == 0U ||
      binding.owner_identity == 0U || binding.allocation_identity == 0U ||
      binding.projection_catalog_identity == 0U ||
      binding.device_identity == 0U || binding.resource_seal_identity == 0U ||
      binding.projection_binding_identity == 0U ||
      binding.asset_value_identity == 0U ||
      binding.asset_value_identity != asset_value_identity ||
      asset.payload.role != kGdnQkvZRole ||
      upload.artifact_identity != asset.artifact_identity ||
      upload.source_inventory_identity != asset.source_inventory_identity ||
      upload.role != kGdnQkvZRole ||
      upload.device_allocation_owner_identity != binding.owner_identity ||
      upload.device_allocation_identity != binding.allocation_identity ||
      upload.device_ordinal != binding.resources.device_ordinal ||
      binding.gdn_output.binding_identity == 0U ||
      binding.gdn_output.binding_identity !=
          compute_gdn_output_execution_binding_identity(binding.gdn_output,
                                                        binding) ||
      binding.continuation.conv_weight == nullptr ||
      binding.continuation.a_log == nullptr ||
      binding.continuation.dt_bias == nullptr ||
      binding.continuation.norm_weight == nullptr ||
      binding.continuation.conv_weight_identity == 0U ||
      binding.continuation.a_log_identity == 0U ||
      binding.continuation.dt_bias_identity == 0U ||
      binding.continuation.norm_weight_identity == 0U ||
      binding.continuation.aggregate_identity !=
          expected_continuation_identity ||
      !binding.continuation.exact_shapes ||
      !binding.continuation.live_cuda_weight_ranges_validated ||
      !binding.live_cuda_payload_range_validated || binding.request_selectable ||
      binding.launcher_authority || binding.production_dispatch_eligible) {
    return 0U;
  }

  std::uint64_t identity = 0x5133'4d46'4742'4e44ULL;
  identity = mix(identity, binding.gdn_ordinal + 1U);
  identity = mix(identity, binding.model_layer + 1U);
  identity = mix(identity, static_cast<std::uint64_t>(binding.role));
  identity = mix(identity,
                 static_cast<std::uint64_t>(binding.input_layout));
  identity = mix(identity,
                 static_cast<std::uint64_t>(binding.tactic_identity));
  identity = mix(identity, binding.package_identity);
  identity = mix(identity, binding.deployment_plan_identity);
  identity = mix(identity, binding.owner_identity);
  identity = mix(identity, binding.allocation_identity);
  identity = mix(identity, binding.projection_catalog_identity);
  identity = mix(identity, binding.device_identity);
  identity = mix(identity, binding.resource_seal_identity);
  identity = mix(identity, binding.projection_binding_identity);
  identity = mix(identity, binding.asset_value_identity);
  identity = mix_fp8_resource(identity, binding.resources);
  identity = mix(identity, binding.gdn_output.binding_identity);
  identity = mix(identity, binding.continuation.aggregate_identity);
  identity = mix(identity, binding.live_cuda_payload_range_validated);
  identity = mix(identity, binding.request_selectable);
  identity = mix(identity, binding.launcher_authority);
  identity = mix(identity, binding.production_dispatch_eligible);
  return identity == 0U ? 0x5133'4d46'4742'4e44ULL : identity;
}

std::uint64_t Sm87MacroFeedV4P40StartupPackage::
    compute_full_attention_resource_bundle_identity(
        const FullAttentionExecutionResourceObservations& observations,
        const std::uint64_t package_identity,
        const std::uint64_t deployment_plan_identity,
        const std::uint64_t device_identity,
        const std::int32_t device_ordinal) noexcept {
  if (package_identity == 0U || deployment_plan_identity == 0U ||
      device_identity == 0U || device_ordinal < 0 ||
      !observations.source_private_queries_completed ||
      observations.caller_resource_snapshot_accepted ||
      observations.qkv.role != kFullQkvRole ||
      observations.qkv.input_layout != kFullQkvInputLayout ||
      observations.qkv.identity != kFullQkvTacticIdentity ||
      observations.output.role != kAttentionOutputRole ||
      observations.output.input_layout != kFullOutputInputLayout ||
      observations.output.identity != kFullOutputTacticIdentity ||
      observations.qkv.device_ordinal != device_ordinal ||
      observations.output.device_ordinal != device_ordinal ||
      observations.preprocess.device_ordinal != device_ordinal ||
      observations.attention.device_ordinal != device_ordinal ||
      !observations.qkv.static_resource_gate_passed ||
      !observations.output.static_resource_gate_passed ||
      !kernels::sm87_macrofeed_v4_fp8_resource_gate(observations.qkv) ||
      !kernels::sm87_macrofeed_v4_fp8_resource_gate(observations.output) ||
      !kernels::
          sm87_macrofeed_v4_full_attention_preprocess_admission_resource_gate(
              observations.preprocess) ||
      !kernels::sm87_macrofeed_v4_attention_c8000_admission_resource_gate(
          observations.attention)) {
    return 0U;
  }
  std::uint64_t identity = 0x5133'4d46'4652'5342ULL;
  identity = mix(identity, package_identity);
  identity = mix(identity, deployment_plan_identity);
  identity = mix(identity, device_identity);
  identity = mix(identity, static_cast<std::uint64_t>(device_ordinal + 1));
  identity = mix_fp8_resource(identity, observations.qkv);
  identity = mix_fp8_resource(identity, observations.output);
  identity = mix_full_preprocess_resource(identity, observations.preprocess);
  identity = mix_full_attention_resource(identity, observations.attention);
  identity = mix(identity, observations.source_private_queries_completed);
  identity = mix(identity, observations.caller_resource_snapshot_accepted);
  return identity == 0U ? 0x5133'4d46'4652'5342ULL : identity;
}

std::uint64_t Sm87MacroFeedV4P40StartupPackage::
    compute_request_boundary_resource_bundle_identity(
        const RequestBoundaryExecutionResourceObservations& observations,
        const std::uint64_t package_identity,
        const std::uint64_t deployment_plan_identity,
        const std::uint64_t device_identity,
        const std::int32_t device_ordinal) noexcept {
  if (package_identity == 0U || deployment_plan_identity == 0U ||
      device_identity == 0U || device_ordinal < 0 ||
      !observations.source_private_queries_completed ||
      observations.caller_resource_snapshot_accepted ||
      observations.embedding.device_ordinal != device_ordinal ||
      observations.final_norm.device_ordinal != device_ordinal ||
      observations.lm_head.device_ordinal != device_ordinal ||
      observations.greedy.device_ordinal != device_ordinal ||
      !request_boundary_embedding_resource_gate(observations.embedding) ||
      !request_boundary_final_norm_resource_gate(observations.final_norm) ||
      !request_boundary_lm_head_resource_gate(observations.lm_head) ||
      !request_boundary_greedy_resource_gate(observations.greedy)) {
    return 0U;
  }
  std::uint64_t identity = 0x5133'4d46'5242'5253ULL;
  identity = mix(identity, package_identity);
  identity = mix(identity, deployment_plan_identity);
  identity = mix(identity, device_identity);
  identity = mix(identity, static_cast<std::uint64_t>(device_ordinal + 1));
  identity =
      mix_request_boundary_embedding_resource(identity, observations.embedding);
  identity = mix_request_boundary_final_norm_resource(identity,
                                                      observations.final_norm);
  identity =
      mix_request_boundary_lm_head_resource(identity, observations.lm_head);
  identity =
      mix_request_boundary_greedy_resource(identity, observations.greedy);
  identity = mix(identity, observations.source_private_queries_completed);
  identity = mix(identity, observations.caller_resource_snapshot_accepted);
  return identity == 0U ? 0x5133'4d46'5242'5253ULL : identity;
}

std::uint64_t Sm87MacroFeedV4P40StartupPackage::
    compute_request_boundary_execution_binding_identity(
        const RequestBoundaryExecutionBinding& binding) noexcept {
  const bool exact_resident_authority =
      binding.normal_resident_authority &&
      !binding.host_test_resident_authority;
  if (binding.package_identity == 0U ||
      binding.deployment_plan_identity == 0U ||
      binding.projection_owner_identity == 0U ||
      binding.projection_allocation_identity == 0U ||
      binding.projection_catalog_identity == 0U ||
      binding.device_identity == 0U || binding.device_ordinal < 0 ||
      binding.embedding_resources.device_ordinal != binding.device_ordinal ||
      binding.final_norm_resources.device_ordinal != binding.device_ordinal ||
      binding.lm_head_resources.device_ordinal != binding.device_ordinal ||
      binding.greedy_resources.device_ordinal != binding.device_ordinal ||
      binding.resident_root == nullptr ||
      binding.resident_root_identity == 0U ||
      binding.resident_arena_begin == 0U ||
      binding.resident_arena_end <= binding.resident_arena_begin ||
      binding.resident_arena_bytes != kPinnedQwen36_27BArenaBytes ||
      binding.resident_arena_end - binding.resident_arena_begin !=
          binding.resident_arena_bytes ||
      binding.source_catalog_identity == 0U ||
      binding.resource_bundle_identity == 0U ||
      binding.embedding.table == nullptr ||
      binding.embedding.vocabulary != Bound::kSm87MacroFeedV4EmbeddingVocabulary ||
      binding.embedding.hidden != Bound::kSm87MacroFeedV4Hidden ||
      binding.embedding.bytes != Bound::kSm87MacroFeedV4EmbeddingTableBytes ||
      binding.embedding.source_identity == 0U ||
      binding.final_norm.centered_weight == nullptr ||
      binding.final_norm.elements != Bound::kSm87MacroFeedV4Hidden ||
      binding.final_norm.bytes != Bound::kSm87MacroFeedV4FinalNormBytes ||
      binding.final_norm.epsilon_fp32_bits !=
          Bound::kSm87MacroFeedV4FinalNormEpsilonFp32Bits ||
      binding.final_norm.source_identity == 0U ||
      binding.lm_head.canonical_packed_weight == nullptr ||
      binding.lm_head.canonical_block_scale == nullptr ||
      binding.lm_head.weight_scale_2_device == nullptr ||
      binding.lm_head.input_scale_device == nullptr ||
      binding.lm_head.rows != Bound::kSm87MacroFeedV4LmHeadRows ||
      binding.lm_head.columns != Bound::kSm87MacroFeedV4LmHeadColumns ||
      binding.lm_head.packed_weight_bytes !=
          Bound::kSm87MacroFeedV4LmHeadPackedWeightBytes ||
      binding.lm_head.block_scale_bytes !=
          Bound::kSm87MacroFeedV4LmHeadBlockScaleBytes ||
      !kernels::sm87_target_aot_projection_scale_bits_valid(
          binding.lm_head.weight_scale_2_fp32_bits) ||
      !kernels::sm87_target_aot_projection_scale_bits_valid(
          binding.lm_head.input_scale_fp32_bits) ||
      binding.lm_head.source_identity == 0U ||
      !binding.lm_head.canonical_resident_nvfp4 ||
      !binding.lm_head.input_scale_provenance_retained ||
      binding.lm_head.input_scale_consumed ||
      binding.greedy.vocabulary != Bound::kSm87MacroFeedV4Vocabulary ||
      binding.greedy.workspace_results !=
          Bound::kSm87MacroFeedV4GreedyWorkspaceResults ||
      binding.greedy.spec_identity == 0U ||
      !binding.greedy.strict_left_to_right_fp32_order ||
      !binding.greedy.smallest_index_tie_break ||
      !binding.greedy.nonfinite_reported_and_ignored ||
      !request_boundary_embedding_resource_gate(binding.embedding_resources) ||
      !request_boundary_final_norm_resource_gate(binding.final_norm_resources) ||
      !request_boundary_lm_head_resource_gate(binding.lm_head_resources) ||
      !request_boundary_greedy_resource_gate(binding.greedy_resources) ||
      !binding.source_private_resource_queries ||
      !binding.device_scale_raw_bits_match_host ||
      !binding.input_scale_provenance_retained ||
      binding.input_scale_consumed ||
      !binding.final_representation_ready_diagnostic_only ||
      !binding.pure_prefill_state_committed_endpoint_unchanged ||
      !exact_resident_authority || binding.request_selectable ||
      binding.launcher_authority || binding.production_dispatch_eligible) {
    return 0U;
  }
  std::uint64_t identity = 0x5133'4d46'5242'4244ULL;
  identity = mix(identity, binding.package_identity);
  identity = mix(identity, binding.deployment_plan_identity);
  identity = mix(identity, binding.projection_owner_identity);
  identity = mix(identity, binding.projection_allocation_identity);
  identity = mix(identity, binding.projection_catalog_identity);
  identity = mix(identity, binding.device_identity);
  identity = mix(identity,
                 static_cast<std::uint64_t>(binding.device_ordinal + 1));
  identity = mix(
      identity, reinterpret_cast<std::uintptr_t>(binding.resident_root));
  identity = mix(identity, binding.resident_root_identity);
  identity = mix(identity, binding.resident_arena_begin);
  identity = mix(identity, binding.resident_arena_end);
  identity = mix(identity, binding.resident_arena_bytes);
  identity = mix(identity, binding.source_catalog_identity);
  identity = mix(identity, binding.resource_bundle_identity);
  identity = mix(identity, binding.embedding.source_identity);
  identity = mix(identity, binding.final_norm.source_identity);
  identity = mix(identity, binding.lm_head.source_identity);
  identity = mix(identity, binding.greedy.spec_identity);
  identity = mix(identity, binding.lm_head.weight_scale_2_fp32_bits);
  identity = mix(identity, binding.lm_head.input_scale_fp32_bits);
  identity =
      mix_request_boundary_embedding_resource(identity,
                                              binding.embedding_resources);
  identity = mix_request_boundary_final_norm_resource(
      identity, binding.final_norm_resources);
  identity =
      mix_request_boundary_lm_head_resource(identity,
                                           binding.lm_head_resources);
  identity = mix_request_boundary_greedy_resource(identity,
                                                  binding.greedy_resources);
  identity = mix(identity, binding.source_private_resource_queries);
  identity = mix(identity, binding.device_scale_raw_bits_match_host);
  identity = mix(identity, binding.input_scale_provenance_retained);
  identity = mix(identity, binding.input_scale_consumed);
  identity = mix(identity,
                 binding.final_representation_ready_diagnostic_only);
  identity = mix(identity,
                 binding.pure_prefill_state_committed_endpoint_unchanged);
  identity = mix(identity, binding.normal_resident_authority);
  identity = mix(identity, binding.host_test_resident_authority);
  identity = mix(identity, binding.request_selectable);
  identity = mix(identity, binding.launcher_authority);
  identity = mix(identity, binding.production_dispatch_eligible);
  return identity == 0U ? 0x5133'4d46'5242'4244ULL : identity;
}

std::uint64_t Sm87MacroFeedV4P40StartupPackage::
    compute_full_attention_qkv_execution_binding_identity(
        const FullAttentionQkvExecutionBinding& binding,
        const FullAttentionLayerExecutionBinding& owner) noexcept {
  const auto& upload = binding.asset.device_upload_receipt;
  const std::uint64_t asset_identity =
      compute_gdn_qkvz_asset_value_identity(binding.asset);
  if (owner.full_ordinal >= kSm87MacroFeedV4P40StartupPackageFullLayers ||
      owner.model_layer != full_attention_model_layer(owner.full_ordinal) ||
      binding.role != kFullQkvRole ||
      binding.input_layout != kFullQkvInputLayout ||
      binding.tactic_identity != kFullQkvTacticIdentity ||
      binding.resources.role != kFullQkvRole ||
      binding.resources.input_layout != kFullQkvInputLayout ||
      binding.resources.identity != kFullQkvTacticIdentity ||
      !fp8_resource_equal(binding.resources, owner.qkv.resources) ||
      !kernels::sm87_macrofeed_v4_fp8_resource_gate(binding.resources) ||
      binding.projection_binding_identity == 0U || asset_identity == 0U ||
      binding.asset_value_identity != asset_identity ||
      binding.asset.payload.role != kFullQkvRole ||
      upload.role != kFullQkvRole ||
      upload.device_allocation_owner_identity != owner.owner_identity ||
      upload.device_allocation_identity != owner.allocation_identity ||
      upload.device_ordinal != binding.resources.device_ordinal ||
      !binding.live_cuda_payload_range_validated) {
    return 0U;
  }
  std::uint64_t identity = 0x5133'4d46'4651'4b42ULL;
  identity = mix(identity, owner.full_ordinal + 1U);
  identity = mix(identity, owner.model_layer + 1U);
  identity = mix(identity, static_cast<std::uint64_t>(binding.role));
  identity = mix(identity, static_cast<std::uint64_t>(binding.input_layout));
  identity = mix(identity, static_cast<std::uint64_t>(binding.tactic_identity));
  identity = mix(identity, binding.projection_binding_identity);
  identity = mix(identity, binding.asset_value_identity);
  identity = mix_fp8_resource(identity, binding.resources);
  identity = mix(identity, binding.live_cuda_payload_range_validated);
  return identity == 0U ? 0x5133'4d46'4651'4b42ULL : identity;
}

std::uint64_t Sm87MacroFeedV4P40StartupPackage::
    compute_full_attention_output_execution_binding_identity(
        const FullAttentionOutputExecutionBinding& binding,
        const FullAttentionLayerExecutionBinding& owner) noexcept {
  const auto& upload = binding.asset.device_upload_receipt;
  const std::uint64_t asset_identity =
      compute_gdn_qkvz_asset_value_identity(binding.asset);
  if (owner.full_ordinal >= kSm87MacroFeedV4P40StartupPackageFullLayers ||
      owner.model_layer != full_attention_model_layer(owner.full_ordinal) ||
      binding.role != kAttentionOutputRole ||
      binding.input_layout != kFullOutputInputLayout ||
      binding.tactic_identity != kFullOutputTacticIdentity ||
      binding.resources.role != kAttentionOutputRole ||
      binding.resources.input_layout != kFullOutputInputLayout ||
      binding.resources.identity != kFullOutputTacticIdentity ||
      !fp8_resource_equal(binding.resources, owner.output.resources) ||
      !kernels::sm87_macrofeed_v4_fp8_resource_gate(binding.resources) ||
      binding.projection_binding_identity == 0U || asset_identity == 0U ||
      binding.asset_value_identity != asset_identity ||
      binding.asset.payload.role != kAttentionOutputRole ||
      upload.role != kAttentionOutputRole ||
      upload.device_allocation_owner_identity != owner.owner_identity ||
      upload.device_allocation_identity != owner.allocation_identity ||
      upload.device_ordinal != binding.resources.device_ordinal ||
      !binding.live_cuda_payload_range_validated) {
    return 0U;
  }
  std::uint64_t identity = 0x5133'4d46'464f'5542ULL;
  identity = mix(identity, owner.full_ordinal + 1U);
  identity = mix(identity, owner.model_layer + 1U);
  identity = mix(identity, static_cast<std::uint64_t>(binding.role));
  identity = mix(identity, static_cast<std::uint64_t>(binding.input_layout));
  identity = mix(identity, static_cast<std::uint64_t>(binding.tactic_identity));
  identity = mix(identity, binding.projection_binding_identity);
  identity = mix(identity, binding.asset_value_identity);
  identity = mix_fp8_resource(identity, binding.resources);
  identity = mix(identity, binding.live_cuda_payload_range_validated);
  return identity == 0U ? 0x5133'4d46'464f'5542ULL : identity;
}

std::uint64_t Sm87MacroFeedV4P40StartupPackage::
    compute_full_attention_execution_binding_identity(
        const FullAttentionLayerExecutionBinding& binding) noexcept {
  const std::uint64_t qkv_identity =
      compute_full_attention_qkv_execution_binding_identity(binding.qkv,
                                                             binding);
  const std::uint64_t output_identity =
      compute_full_attention_output_execution_binding_identity(binding.output,
                                                                binding);
  const std::uint64_t q_identity = full_attention_qk_norm_tensor_identity(
      binding.package_identity, binding.deployment_plan_identity,
      binding.device_identity, binding.qkv.resources.device_ordinal,
      binding.full_ordinal, binding.model_layer, false,
      binding.qk_norm.q_norm);
  const std::uint64_t k_identity = full_attention_qk_norm_tensor_identity(
      binding.package_identity, binding.deployment_plan_identity,
      binding.device_identity, binding.qkv.resources.device_ordinal,
      binding.full_ordinal, binding.model_layer, true,
      binding.qk_norm.k_norm);
  const std::uint64_t pair_identity =
      full_attention_qk_norm_pair_identity(
          binding.full_ordinal, binding.model_layer, q_identity, k_identity);
  if (binding.full_ordinal >=
          kSm87MacroFeedV4P40StartupPackageFullLayers ||
      binding.model_layer != full_attention_model_layer(binding.full_ordinal) ||
      !model_layer_is_full_attention(binding.model_layer) ||
      binding.package_identity == 0U ||
      binding.deployment_plan_identity == 0U ||
      binding.owner_identity == 0U || binding.allocation_identity == 0U ||
      binding.projection_catalog_identity == 0U ||
      binding.device_identity == 0U || binding.source_catalog_identity == 0U ||
      binding.resource_bundle_identity == 0U || qkv_identity == 0U ||
      output_identity == 0U || binding.qkv.binding_identity != qkv_identity ||
      binding.output.binding_identity != output_identity ||
      q_identity == 0U || k_identity == 0U || pair_identity == 0U ||
      binding.qk_norm.q_norm_identity != q_identity ||
      binding.qk_norm.k_norm_identity != k_identity ||
      binding.qk_norm.pair_identity != pair_identity ||
      binding.qk_norm.q_norm_elements !=
          kernels::kSm87MacroFeedV4FullAttentionPreprocessHeadDimension ||
      binding.qk_norm.k_norm_elements !=
          kernels::kSm87MacroFeedV4FullAttentionPreprocessHeadDimension ||
      !binding.qk_norm.exact_shapes ||
      !binding.qk_norm.live_cuda_weight_ranges_validated ||
      !kernels::
          sm87_macrofeed_v4_full_attention_preprocess_admission_resource_gate(
              binding.preprocess_resources) ||
      !kernels::sm87_macrofeed_v4_attention_c8000_admission_resource_gate(
          binding.attention_resources) ||
      binding.preprocess_resources.device_ordinal !=
          binding.qkv.resources.device_ordinal ||
      binding.attention_resources.device_ordinal !=
          binding.qkv.resources.device_ordinal ||
      !binding.live_cuda_ranges_validated ||
      !binding.source_private_resource_queries || binding.request_selectable ||
      binding.launcher_authority || binding.production_dispatch_eligible) {
    return 0U;
  }
  std::uint64_t identity = 0x5133'4d46'464c'4244ULL;
  identity = mix(identity, binding.full_ordinal + 1U);
  identity = mix(identity, binding.model_layer + 1U);
  identity = mix(identity, binding.package_identity);
  identity = mix(identity, binding.deployment_plan_identity);
  identity = mix(identity, binding.owner_identity);
  identity = mix(identity, binding.allocation_identity);
  identity = mix(identity, binding.projection_catalog_identity);
  identity = mix(identity, binding.device_identity);
  identity = mix(identity, binding.source_catalog_identity);
  identity = mix(identity, binding.resource_bundle_identity);
  identity = mix(identity, qkv_identity);
  identity = mix(identity, output_identity);
  identity = mix(identity, pair_identity);
  identity = mix_full_preprocess_resource(identity,
                                           binding.preprocess_resources);
  identity = mix_full_attention_resource(identity, binding.attention_resources);
  identity = mix(identity, binding.live_cuda_ranges_validated);
  identity = mix(identity, binding.source_private_resource_queries);
  identity = mix(identity, binding.request_selectable);
  identity = mix(identity, binding.launcher_authority);
  identity = mix(identity, binding.production_dispatch_eligible);
  return identity == 0U ? 0x5133'4d46'464c'4244ULL : identity;
}

bool Sm87MacroFeedV4P40StartupPackage::
    seal_bf16_ab_execution_catalog_for_execution_package(
        Bf16AbExecutionBindingCatalog* const catalog) const noexcept {
  if (catalog == nullptr) {
    return false;
  }
  catalog->fill({});
  // valid() performs the complete target-AOT, BF16 inventory, live-allocation
  // and resource revalidation once at execution-package construction.  The
  // resulting immutable catalog is retained under the same Engine lifetime
  // root; request execution must not return through this seam.
  if (!valid()) {
    return false;
  }
  for (std::size_t ordinal = 0U;
       ordinal < bf16_ab_capability_.pairs_.size(); ++ordinal) {
    const auto& pair = bf16_ab_capability_.pairs_[ordinal];
    if (pair.model_layer != gdn_model_layer(ordinal) ||
        !model_layer_is_gdn(pair.model_layer) || pair.a_weights == nullptr ||
        pair.b_weights == nullptr || pair.pair_identity == 0U) {
      catalog->fill({});
      return false;
    }
  }
  *catalog = bf16_ab_capability_.pairs_;
  return true;
}

bool Sm87MacroFeedV4P40StartupPackage::
    seal_gdn_layer_execution_catalog_for_execution_package(
        GdnLayerExecutionBindingCatalog* const catalog,
        std::uint64_t* const catalog_identity,
        std::size_t* const failure_ordinal,
        int* const cuda_error) const noexcept {
  if (catalog != nullptr) {
    catalog->fill({});
  }
  if (catalog_identity != nullptr) {
    *catalog_identity = 0U;
  }
  if (failure_ordinal != nullptr) {
    *failure_ordinal = kSm87MacroFeedV4P40StartupPackageGdnLayers;
  }
  if (cuda_error != nullptr) {
    *cuda_error = 0;
  }
  if (catalog == nullptr || catalog_identity == nullptr ||
      failure_ordinal == nullptr || cuda_error == nullptr || !valid()) {
    return false;
  }

  kernels::Sm87MacroFeedV4Fp8CudaResources observed_resources;
  const int query_status =
      kernels::query_sm87_macrofeed_v4_fp8_cuda_resources(
          kGdnQkvZRole, kGdnQkvZInputLayout, &observed_resources);
  if (query_status == 0) {
    observed_resources.static_resource_gate_passed =
        kernels::sm87_macrofeed_v4_fp8_resource_gate(observed_resources);
  }
  if (query_status != 0 ||
      !fp8_resource_equal(observed_resources, seals_.gdn_qkvz.resources) ||
      !observed_resources.static_resource_gate_passed ||
      !kernels::sm87_macrofeed_v4_fp8_resource_gate(observed_resources)) {
    *failure_ordinal = 0U;
    *cuda_error = query_status != 0
                      ? query_status
                      : static_cast<int>(cudaErrorInvalidValue);
    return false;
  }

  kernels::Sm87MacroFeedV4Fp8CudaResources observed_output_resources;
  const int output_query_status =
      kernels::query_sm87_macrofeed_v4_fp8_cuda_resources(
          kAttentionOutputRole, kGdnOutputInputLayout,
          &observed_output_resources);
  if (output_query_status == 0) {
    observed_output_resources.static_resource_gate_passed =
        kernels::sm87_macrofeed_v4_fp8_resource_gate(
            observed_output_resources);
  }
  if (output_query_status != 0 ||
      !fp8_resource_equal(observed_output_resources,
                          seals_.gdn_qkvz.output_resources) ||
      !observed_output_resources.static_resource_gate_passed ||
      !kernels::sm87_macrofeed_v4_fp8_resource_gate(
          observed_output_resources)) {
    *failure_ordinal = 0U;
    *cuda_error = output_query_status != 0
                      ? output_query_status
                      : static_cast<int>(cudaErrorInvalidValue);
    return false;
  }

  GdnLayerExecutionBindingCatalog sealed{};
  std::uint64_t identity = 0x5133'4d46'4745'5843ULL;
  identity = mix(identity, audit_.gdn_qkvz_binding_catalog_identity);
  identity = mix(identity, seals_.gdn_qkvz.seal_identity);
  identity = mix(identity, sealed.size());
  for (std::size_t ordinal = 0U; ordinal < sealed.size(); ++ordinal) {
    const std::size_t model_layer = gdn_model_layer(ordinal);
    const std::size_t descriptor =
        sm87_target_aot_complete_descriptor_ordinal(model_layer,
                                                    kGdnQkvZRole);
    const std::size_t output_descriptor =
        sm87_target_aot_complete_descriptor_ordinal(model_layer,
                                                    kAttentionOutputRole);
    if (!model_layer_is_gdn(model_layer) ||
        descriptor >= projection_bindings_.size() ||
        output_descriptor >= projection_bindings_.size() ||
        !projection_bindings_[descriptor] ||
        !projection_bindings_[output_descriptor] ||
        !projection_bindings_[descriptor]->valid_for_prevalidated_catalog(
            model_layer, kGdnQkvZRole, audit_.package_identity,
            audit_.catalog_identity) ||
        projection_bindings_[descriptor]->deployment_plan_identity() !=
            audit_.deployment_plan_identity ||
        projection_bindings_[descriptor]->consumer_tactic_identity() !=
            static_cast<std::uint64_t>(kGdnQkvZTacticIdentity) ||
        !projection_bindings_[output_descriptor]
             ->valid_for_prevalidated_catalog(
                 model_layer, kAttentionOutputRole, audit_.package_identity,
                 audit_.catalog_identity) ||
        projection_bindings_[output_descriptor]
                ->deployment_plan_identity() !=
            audit_.deployment_plan_identity ||
        projection_bindings_[output_descriptor]
                ->consumer_tactic_identity() !=
            static_cast<std::uint64_t>(kGdnOutputTacticIdentity)) {
      catalog->fill({});
      *failure_ordinal = ordinal;
      return false;
    }

    const auto& projection_binding = *projection_bindings_[descriptor];
    const auto& output_projection_binding =
        *projection_bindings_[output_descriptor];
    const auto* const asset = projection_binding.asset_.borrow_fp8_cuda_asset();
    const auto* const output_asset =
        output_projection_binding.asset_.borrow_fp8_cuda_asset();
    if (asset == nullptr ||
        projection_binding.asset_.borrow_nvfp4_cuda_asset() != nullptr ||
        !kernels::sm87_target_aot_fp8_cuda_asset_valid(*asset) ||
        asset->payload.role != kGdnQkvZRole ||
        asset->payload.begin !=
            asset->device_upload_receipt.device_payload_begin ||
        asset->payload.end !=
            asset->device_upload_receipt.device_payload_end ||
        asset->payload.bytes !=
            asset->device_upload_receipt.device_payload_bytes ||
        asset->device_upload_receipt.device_allocation_owner_identity !=
            audit_.owner_identity ||
        asset->device_upload_receipt.device_allocation_identity !=
            audit_.allocation_identity ||
        asset->device_upload_receipt.device_ordinal != audit_.device_ordinal ||
        compute_gdn_qkvz_asset_value_identity(*asset) == 0U ||
        !live_current_device_allocation_range(
            reinterpret_cast<const std::uint16_t*>(asset->payload.begin),
            asset->payload.bytes, audit_.device_ordinal, cuda_error,
            asset->device_upload_receipt.device_allocation_begin,
            asset->device_upload_receipt.device_allocation_end,
            asset->device_upload_receipt.device_allocation_bytes)) {
      catalog->fill({});
      *failure_ordinal = ordinal;
      return false;
    }
    if (output_asset == nullptr ||
        output_projection_binding.asset_.borrow_nvfp4_cuda_asset() !=
            nullptr ||
        !kernels::sm87_target_aot_fp8_cuda_asset_valid(*output_asset) ||
        output_asset->payload.role != kAttentionOutputRole ||
        output_asset->payload.begin !=
            output_asset->device_upload_receipt.device_payload_begin ||
        output_asset->payload.end !=
            output_asset->device_upload_receipt.device_payload_end ||
        output_asset->payload.bytes !=
            output_asset->device_upload_receipt.device_payload_bytes ||
        output_asset->device_upload_receipt.device_allocation_owner_identity !=
            audit_.owner_identity ||
        output_asset->device_upload_receipt.device_allocation_identity !=
            audit_.allocation_identity ||
        output_asset->device_upload_receipt.device_ordinal !=
            audit_.device_ordinal ||
        compute_gdn_qkvz_asset_value_identity(*output_asset) == 0U ||
        !live_current_device_allocation_range(
            reinterpret_cast<const std::uint16_t*>(
                output_asset->payload.begin),
            output_asset->payload.bytes, audit_.device_ordinal, cuda_error,
            output_asset->device_upload_receipt.device_allocation_begin,
            output_asset->device_upload_receipt.device_allocation_end,
            output_asset->device_upload_receipt.device_allocation_bytes)) {
      catalog->fill({});
      *failure_ordinal = ordinal;
      return false;
    }

    GdnContinuationSource continuation_source;
    if (bf16_ab_capability_.model_weights_ == nullptr ||
        !exact_gdn_continuation_source(*bf16_ab_capability_.model_weights_,
                                       model_layer,
                                       &continuation_source)) {
      catalog->fill({});
      *failure_ordinal = ordinal;
      *cuda_error = static_cast<int>(cudaErrorInvalidValue);
      return false;
    }
    const std::array<const std::uint16_t*, 4U> continuation_weights{{
        continuation_source.conv_weight, continuation_source.a_log,
        continuation_source.dt_bias, continuation_source.norm_weight}};
    const std::array<std::uint64_t, 4U> continuation_bytes{{
        kernels::kSm87MacroFeedV4GdnConvWeightBytes,
        kernels::kSm87MacroFeedV4GdnHeadVectorBytes,
        kernels::kSm87MacroFeedV4GdnHeadVectorBytes,
        kernels::kSm87MacroFeedV4GdnNormWeightBytes}};
    for (std::size_t weight_index = 0U;
         weight_index < continuation_weights.size(); ++weight_index) {
      if (!live_current_device_allocation_range(
              continuation_weights[weight_index],
              continuation_bytes[weight_index], audit_.device_ordinal,
              cuda_error)) {
        catalog->fill({});
        *failure_ordinal = ordinal;
        return false;
      }
    }

    auto& binding = sealed[ordinal];
    binding.gdn_ordinal = static_cast<std::uint32_t>(ordinal);
    binding.model_layer = static_cast<std::uint32_t>(model_layer);
    binding.role = kGdnQkvZRole;
    binding.input_layout = kGdnQkvZInputLayout;
    binding.tactic_identity = kGdnQkvZTacticIdentity;
    binding.asset = *asset;
    binding.resources = observed_resources;
    binding.package_identity = audit_.package_identity;
    binding.deployment_plan_identity = audit_.deployment_plan_identity;
    binding.owner_identity = audit_.owner_identity;
    binding.allocation_identity = audit_.allocation_identity;
    binding.projection_catalog_identity = audit_.catalog_identity;
    binding.device_identity = audit_.device_identity;
    binding.resource_seal_identity = seals_.gdn_qkvz.seal_identity;
    binding.projection_binding_identity =
        projection_binding.binding_identity();
    binding.asset_value_identity =
        compute_gdn_qkvz_asset_value_identity(binding.asset);
    binding.gdn_output.role = kAttentionOutputRole;
    binding.gdn_output.input_layout = kGdnOutputInputLayout;
    binding.gdn_output.tactic_identity = kGdnOutputTacticIdentity;
    binding.gdn_output.asset = *output_asset;
    binding.gdn_output.resources = observed_output_resources;
    binding.gdn_output.projection_binding_identity =
        output_projection_binding.binding_identity();
    binding.gdn_output.asset_value_identity =
        compute_gdn_qkvz_asset_value_identity(binding.gdn_output.asset);
    binding.gdn_output.live_cuda_payload_range_validated = true;

    binding.continuation.conv_weight = continuation_source.conv_weight;
    binding.continuation.a_log = continuation_source.a_log;
    binding.continuation.dt_bias = continuation_source.dt_bias;
    binding.continuation.norm_weight = continuation_source.norm_weight;
    std::array<std::uint64_t*, 4U> continuation_identities{{
        &binding.continuation.conv_weight_identity,
        &binding.continuation.a_log_identity,
        &binding.continuation.dt_bias_identity,
        &binding.continuation.norm_weight_identity}};
    std::uint64_t continuation_identity = 0x5133'4d46'4744'4e43ULL;
    for (std::size_t weight_index = 0U;
         weight_index < continuation_weights.size(); ++weight_index) {
      *continuation_identities[weight_index] = gdn_weight_identity(
          0x5133'4d46'4744'4e57ULL,
          binding.deployment_plan_identity, binding.device_identity,
          model_layer, weight_index, continuation_weights[weight_index],
          continuation_bytes[weight_index]);
      continuation_identity =
          mix(continuation_identity, *continuation_identities[weight_index]);
    }
    binding.continuation.aggregate_identity = continuation_identity;
    binding.continuation.exact_shapes = true;
    binding.continuation.live_cuda_weight_ranges_validated = true;
    binding.live_cuda_payload_range_validated = true;
    binding.request_selectable = false;
    binding.launcher_authority = false;
    binding.production_dispatch_eligible = false;
    binding.gdn_output.binding_identity =
        compute_gdn_output_execution_binding_identity(binding.gdn_output,
                                                      binding);
    binding.binding_identity =
        compute_gdn_qkvz_execution_binding_identity(binding);
    if (binding.binding_identity == 0U) {
      catalog->fill({});
      *failure_ordinal = ordinal;
      *cuda_error = static_cast<int>(cudaErrorInvalidValue);
      return false;
    }
    identity = mix(identity, ordinal + 1U);
    identity = mix(identity, binding.binding_identity);
  }

  if (identity == 0U ||
      compute_gdn_qkvz_binding_catalog_identity(
          projection_access_, capabilities_, audit_.deployment_plan_identity,
          observed_resources, observed_output_resources,
          *bf16_ab_capability_.model_weights_) !=
          audit_.gdn_qkvz_binding_catalog_identity) {
    catalog->fill({});
    *failure_ordinal = 0U;
    *cuda_error = static_cast<int>(cudaErrorInvalidValue);
    return false;
  }
  *catalog = sealed;
  *catalog_identity = identity;
  *failure_ordinal = sealed.size();
  *cuda_error = 0;
  return true;
}

bool Sm87MacroFeedV4P40StartupPackage::
    seal_gdn_qkvz_execution_catalog_for_execution_package(
        GdnQkvZExecutionBindingCatalog* const catalog,
        std::uint64_t* const catalog_identity,
        std::size_t* const failure_ordinal,
        int* const cuda_error) const noexcept {
  return seal_gdn_layer_execution_catalog_for_execution_package(
      catalog, catalog_identity, failure_ordinal, cuda_error);
}

bool Sm87MacroFeedV4P40StartupPackage::
    seal_full_attention_execution_catalog_for_execution_package(
        const FullAttentionExecutionResourceObservations& observations,
        FullAttentionLayerExecutionBindingCatalog* const catalog,
        std::uint64_t* const catalog_identity,
        std::size_t* const failure_ordinal,
        int* const cuda_error) const noexcept {
  if (catalog != nullptr) {
    catalog->fill({});
  }
  if (catalog_identity != nullptr) {
    *catalog_identity = 0U;
  }
  if (failure_ordinal != nullptr) {
    *failure_ordinal = kSm87MacroFeedV4P40StartupPackageFullLayers;
  }
  if (cuda_error != nullptr) {
    *cuda_error = 0;
  }
  if (catalog == nullptr || catalog_identity == nullptr ||
      failure_ordinal == nullptr || cuda_error == nullptr || !valid()) {
    return false;
  }
  const auto fail = [&](const std::size_t ordinal,
                        const int error = 0) noexcept {
    catalog->fill({});
    *catalog_identity = 0U;
    *failure_ordinal = ordinal;
    *cuda_error = error;
    return false;
  };
  const std::uint64_t resource_bundle_identity =
      compute_full_attention_resource_bundle_identity(
          observations, audit_.package_identity,
          audit_.deployment_plan_identity, audit_.device_identity,
          audit_.device_ordinal);
  if (resource_bundle_identity == 0U) {
    return fail(0U, static_cast<int>(cudaErrorInvalidValue));
  }

  FullAttentionLayerExecutionBindingCatalog sealed{};
  std::uint64_t identity = 0x5133'4d46'4645'5843ULL;
  identity = mix(identity, audit_.full_attention_source_catalog_identity);
  identity = mix(identity, seals_.full_attention.seal_identity);
  identity = mix(identity, resource_bundle_identity);
  identity = mix(identity, sealed.size());
  for (std::size_t ordinal = 0U; ordinal < sealed.size(); ++ordinal) {
    const std::size_t model_layer = full_attention_model_layer(ordinal);
    const std::size_t qkv_descriptor =
        sm87_target_aot_complete_descriptor_ordinal(model_layer,
                                                    kFullQkvRole);
    const std::size_t output_descriptor =
        sm87_target_aot_complete_descriptor_ordinal(model_layer,
                                                    kAttentionOutputRole);
    if (!model_layer_is_full_attention(model_layer) ||
        qkv_descriptor >= projection_bindings_.size() ||
        output_descriptor >= projection_bindings_.size() ||
        !projection_bindings_[qkv_descriptor] ||
        !projection_bindings_[output_descriptor] ||
        !projection_bindings_[qkv_descriptor]
             ->valid_for_prevalidated_catalog(
                 model_layer, kFullQkvRole, audit_.package_identity,
                 audit_.catalog_identity) ||
        projection_bindings_[qkv_descriptor]
                ->deployment_plan_identity() !=
            audit_.deployment_plan_identity ||
        projection_bindings_[qkv_descriptor]
                ->consumer_tactic_identity() !=
            static_cast<std::uint64_t>(kFullQkvTacticIdentity) ||
        !projection_bindings_[output_descriptor]
             ->valid_for_prevalidated_catalog(
                 model_layer, kAttentionOutputRole, audit_.package_identity,
                 audit_.catalog_identity) ||
        projection_bindings_[output_descriptor]
                ->deployment_plan_identity() !=
            audit_.deployment_plan_identity ||
        projection_bindings_[output_descriptor]
                ->consumer_tactic_identity() !=
            static_cast<std::uint64_t>(kFullOutputTacticIdentity)) {
      return fail(ordinal);
    }

    const auto& qkv_projection = *projection_bindings_[qkv_descriptor];
    const auto& output_projection =
        *projection_bindings_[output_descriptor];
    const auto* const qkv_asset =
        qkv_projection.asset_.borrow_fp8_cuda_asset();
    const auto* const output_asset =
        output_projection.asset_.borrow_fp8_cuda_asset();
    const auto live_asset = [&](const kernels::Sm87TargetAotFp8CudaAssetView*
                                    asset,
                                const Role role) noexcept {
      if (asset == nullptr ||
          !kernels::sm87_target_aot_fp8_cuda_asset_valid(*asset) ||
          asset->payload.role != role ||
          asset->payload.begin !=
              asset->device_upload_receipt.device_payload_begin ||
          asset->payload.end !=
              asset->device_upload_receipt.device_payload_end ||
          asset->payload.bytes !=
              asset->device_upload_receipt.device_payload_bytes ||
          asset->device_upload_receipt.device_allocation_owner_identity !=
              audit_.owner_identity ||
          asset->device_upload_receipt.device_allocation_identity !=
              audit_.allocation_identity ||
          asset->device_upload_receipt.device_ordinal !=
              audit_.device_ordinal ||
          compute_gdn_qkvz_asset_value_identity(*asset) == 0U) {
        *cuda_error = static_cast<int>(cudaErrorInvalidValue);
        return false;
      }
      return live_current_device_allocation_range(
          reinterpret_cast<const std::uint16_t*>(asset->payload.begin),
          asset->payload.bytes, audit_.device_ordinal, cuda_error,
          asset->device_upload_receipt.device_allocation_begin,
          asset->device_upload_receipt.device_allocation_end,
          asset->device_upload_receipt.device_allocation_bytes);
    };
    if (qkv_projection.asset_.borrow_nvfp4_cuda_asset() != nullptr ||
        output_projection.asset_.borrow_nvfp4_cuda_asset() != nullptr ||
        !live_asset(qkv_asset, kFullQkvRole) ||
        !live_asset(output_asset, kAttentionOutputRole)) {
      return fail(ordinal, *cuda_error);
    }

    FullAttentionQkNormSource norm_source;
    if (bf16_ab_capability_.model_weights_ == nullptr ||
        !exact_full_attention_qk_norm_source(
            *bf16_ab_capability_.model_weights_, model_layer,
            &norm_source)) {
      return fail(ordinal, static_cast<int>(cudaErrorInvalidValue));
    }
    constexpr std::uint64_t kNormBytes =
        kernels::kSm87MacroFeedV4FullAttentionPreprocessNormWeightBytes;
    if (!live_current_device_allocation_range(
            norm_source.q_norm, kNormBytes, audit_.device_ordinal,
            cuda_error) ||
        !live_current_device_allocation_range(
            norm_source.k_norm, kNormBytes, audit_.device_ordinal,
            cuda_error)) {
      return fail(ordinal, *cuda_error);
    }

    auto& binding = sealed[ordinal];
    binding.full_ordinal = static_cast<std::uint32_t>(ordinal);
    binding.model_layer = static_cast<std::uint32_t>(model_layer);
    binding.package_identity = audit_.package_identity;
    binding.deployment_plan_identity = audit_.deployment_plan_identity;
    binding.owner_identity = audit_.owner_identity;
    binding.allocation_identity = audit_.allocation_identity;
    binding.projection_catalog_identity = audit_.catalog_identity;
    binding.device_identity = audit_.device_identity;
    binding.source_catalog_identity =
        audit_.full_attention_source_catalog_identity;
    binding.resource_bundle_identity = resource_bundle_identity;

    binding.qkv.role = kFullQkvRole;
    binding.qkv.input_layout = kFullQkvInputLayout;
    binding.qkv.tactic_identity = kFullQkvTacticIdentity;
    binding.qkv.asset = *qkv_asset;
    binding.qkv.resources = observations.qkv;
    binding.qkv.projection_binding_identity =
        qkv_projection.binding_identity();
    binding.qkv.asset_value_identity =
        compute_gdn_qkvz_asset_value_identity(binding.qkv.asset);
    binding.qkv.live_cuda_payload_range_validated = true;

    binding.output.role = kAttentionOutputRole;
    binding.output.input_layout = kFullOutputInputLayout;
    binding.output.tactic_identity = kFullOutputTacticIdentity;
    binding.output.asset = *output_asset;
    binding.output.resources = observations.output;
    binding.output.projection_binding_identity =
        output_projection.binding_identity();
    binding.output.asset_value_identity =
        compute_gdn_qkvz_asset_value_identity(binding.output.asset);
    binding.output.live_cuda_payload_range_validated = true;

    binding.qk_norm.q_norm = norm_source.q_norm;
    binding.qk_norm.k_norm = norm_source.k_norm;
    binding.qk_norm.q_norm_elements =
        kernels::kSm87MacroFeedV4FullAttentionPreprocessHeadDimension;
    binding.qk_norm.k_norm_elements =
        kernels::kSm87MacroFeedV4FullAttentionPreprocessHeadDimension;
    binding.qk_norm.q_norm_identity =
        full_attention_qk_norm_tensor_identity(
            binding.package_identity, binding.deployment_plan_identity,
            binding.device_identity, audit_.device_ordinal, ordinal,
            model_layer, false, norm_source.q_norm);
    binding.qk_norm.k_norm_identity =
        full_attention_qk_norm_tensor_identity(
            binding.package_identity, binding.deployment_plan_identity,
            binding.device_identity, audit_.device_ordinal, ordinal,
            model_layer, true, norm_source.k_norm);
    binding.qk_norm.pair_identity =
        full_attention_qk_norm_pair_identity(
            ordinal, model_layer, binding.qk_norm.q_norm_identity,
            binding.qk_norm.k_norm_identity);
    binding.qk_norm.exact_shapes = true;
    binding.qk_norm.live_cuda_weight_ranges_validated = true;
    binding.preprocess_resources = observations.preprocess;
    binding.attention_resources = observations.attention;
    if (!fp8_resource_equal(binding.qkv.resources, observations.qkv) ||
        !fp8_resource_equal(binding.output.resources,
                            observations.output) ||
        !full_preprocess_resource_equal(binding.preprocess_resources,
                                        observations.preprocess) ||
        !full_attention_resource_equal(binding.attention_resources,
                                       observations.attention)) {
      return fail(ordinal, static_cast<int>(cudaErrorInvalidValue));
    }
    binding.live_cuda_ranges_validated = true;
    binding.source_private_resource_queries = true;
    binding.request_selectable = false;
    binding.launcher_authority = false;
    binding.production_dispatch_eligible = false;
    binding.qkv.binding_identity =
        compute_full_attention_qkv_execution_binding_identity(binding.qkv,
                                                               binding);
    binding.output.binding_identity =
        compute_full_attention_output_execution_binding_identity(
            binding.output, binding);
    binding.binding_identity =
        compute_full_attention_execution_binding_identity(binding);
    if (binding.binding_identity == 0U) {
      return fail(ordinal, static_cast<int>(cudaErrorInvalidValue));
    }
    identity = mix(identity, ordinal + 1U);
    identity = mix(identity, binding.binding_identity);
  }

  std::size_t source_failure =
      kSm87MacroFeedV4P40StartupPackageFullLayers;
  bool source_failure_k = false;
  int source_cuda_error = 0;
  if (identity == 0U ||
      compute_full_attention_source_catalog_identity(
          projection_access_, capabilities_, audit_.deployment_plan_identity,
          *bf16_ab_capability_.model_weights_, &source_failure,
          &source_failure_k, &source_cuda_error) !=
          audit_.full_attention_source_catalog_identity) {
    return fail(source_failure < sealed.size() ? source_failure : 0U,
                source_cuda_error == 0
                    ? static_cast<int>(cudaErrorInvalidValue)
                    : source_cuda_error);
  }
  *catalog = sealed;
  *catalog_identity = identity;
  *failure_ordinal = sealed.size();
  *cuda_error = 0;
  return true;
}

bool Sm87MacroFeedV4P40StartupPackage::
    seal_request_boundary_execution_catalog_for_execution_package(
        const RequestBoundaryExecutionResourceObservations& observations,
        RequestBoundaryExecutionBindingCatalog* const catalog,
        std::uint64_t* const catalog_identity,
        std::size_t* const failure_index,
        int* const cuda_error) const noexcept {
  if (catalog != nullptr) {
    catalog->fill({});
  }
  if (catalog_identity != nullptr) {
    *catalog_identity = 0U;
  }
  if (failure_index != nullptr) {
    *failure_index =
        kSm87MacroFeedV4P40StartupPackageRequestBoundaryBindings;
  }
  if (cuda_error != nullptr) {
    *cuda_error = 0;
  }
  if (catalog == nullptr || catalog_identity == nullptr ||
      failure_index == nullptr || cuda_error == nullptr || !valid()) {
    return false;
  }
  const auto fail = [&](const std::size_t index,
                        const int error = 0) noexcept {
    catalog->fill({});
    *catalog_identity = 0U;
    *failure_index = index;
    *cuda_error = error;
    return false;
  };

  // A host fixture can exercise the Startup/T0 inventory but can never mint
  // the normal Engine-lifetime execution catalog.
  if (!request_boundary_sources_.normal_resident_authority ||
      request_boundary_sources_.host_test_resident_authority ||
      projection_access_.host_test_resident_authority_ ||
      projection_access_.host_test_issuer_nonce_ != 0U ||
      request_boundary_sources_.resident_root !=
          projection_access_.resident_root_ ||
      request_boundary_sources_.resident_root_identity !=
          projection_access_.resident_root_identity_ ||
      request_boundary_sources_.resident_arena_begin !=
          projection_access_.resident_arena_begin_ ||
      request_boundary_sources_.resident_arena_bytes !=
          kPinnedQwen36_27BArenaBytes ||
      request_boundary_sources_.resident_arena_bytes !=
          projection_access_.resident_arena_bytes_) {
    return fail(0U, static_cast<int>(cudaErrorInvalidValue));
  }

  if (!request_boundary_embedding_resource_gate(observations.embedding)) {
    return fail(0U, static_cast<int>(cudaErrorInvalidValue));
  }
  if (!request_boundary_final_norm_resource_gate(observations.final_norm)) {
    return fail(1U, static_cast<int>(cudaErrorInvalidValue));
  }
  if (!request_boundary_lm_head_resource_gate(observations.lm_head)) {
    return fail(2U, static_cast<int>(cudaErrorInvalidValue));
  }
  if (!request_boundary_greedy_resource_gate(observations.greedy) ||
      !observations.source_private_queries_completed ||
      observations.caller_resource_snapshot_accepted) {
    return fail(3U, static_cast<int>(cudaErrorInvalidValue));
  }
  const std::uint64_t resource_bundle_identity =
      compute_request_boundary_resource_bundle_identity(
          observations, audit_.package_identity,
          audit_.deployment_plan_identity, audit_.device_identity,
          audit_.device_ordinal);
  if (resource_bundle_identity == 0U) {
    return fail(0U, static_cast<int>(cudaErrorInvalidValue));
  }

  RequestBoundarySourceCatalog fresh_sources;
  int source_cuda_error = 0;
  std::size_t source_failure = 6U;
  if (request_boundary_sources_.model_weights == nullptr ||
      !build_request_boundary_source_catalog(
          *request_boundary_sources_.model_weights, projection_access_,
          audit_.deployment_plan_identity, &fresh_sources,
          &source_cuda_error, &source_failure,
          &request_boundary_sources_) ||
      fresh_sources.catalog_identity !=
          request_boundary_sources_.catalog_identity ||
      fresh_sources.resident_root_identity !=
          request_boundary_sources_.resident_root_identity ||
      !fresh_sources.normal_resident_authority ||
      fresh_sources.host_test_resident_authority) {
    return fail(source_failure < 6U ? source_failure : 0U,
                source_cuda_error == 0
                    ? static_cast<int>(cudaErrorInvalidValue)
                    : source_cuda_error);
  }

  RequestBoundaryExecutionBinding binding;
  binding.embedding = fresh_sources.embedding;
  binding.final_norm = fresh_sources.final_norm;
  binding.lm_head = fresh_sources.lm_head;
  binding.greedy = fresh_sources.greedy;
  binding.embedding_resources = observations.embedding;
  binding.final_norm_resources = observations.final_norm;
  binding.lm_head_resources = observations.lm_head;
  binding.greedy_resources = observations.greedy;
  binding.package_identity = audit_.package_identity;
  binding.deployment_plan_identity = audit_.deployment_plan_identity;
  binding.projection_owner_identity = audit_.owner_identity;
  binding.projection_allocation_identity = audit_.allocation_identity;
  binding.projection_catalog_identity = audit_.catalog_identity;
  binding.device_identity = audit_.device_identity;
  binding.device_ordinal = audit_.device_ordinal;
  binding.resident_root = fresh_sources.resident_root;
  binding.resident_root_identity = fresh_sources.resident_root_identity;
  binding.resident_arena_begin = fresh_sources.resident_arena_begin;
  binding.resident_arena_end = fresh_sources.resident_arena_end;
  binding.resident_arena_bytes = fresh_sources.resident_arena_bytes;
  binding.source_catalog_identity = fresh_sources.catalog_identity;
  binding.resource_bundle_identity = resource_bundle_identity;
  binding.source_private_resource_queries = true;
  binding.device_scale_raw_bits_match_host = true;
  binding.input_scale_provenance_retained = true;
  binding.input_scale_consumed = false;
  binding.final_representation_ready_diagnostic_only = true;
  binding.pure_prefill_state_committed_endpoint_unchanged = true;
  binding.normal_resident_authority = true;
  binding.host_test_resident_authority = false;
  binding.request_selectable = false;
  binding.launcher_authority = false;
  binding.production_dispatch_eligible = false;
  binding.binding_identity =
      compute_request_boundary_execution_binding_identity(binding);
  if (binding.binding_identity == 0U) {
    return fail(0U, static_cast<int>(cudaErrorInvalidValue));
  }

  RequestBoundaryExecutionBindingCatalog sealed{};
  sealed[0U] = binding;
  std::uint64_t identity = 0x5133'4d46'5242'5843ULL;
  identity = mix(identity, audit_.package_identity);
  identity = mix(identity, audit_.deployment_plan_identity);
  identity = mix(identity, audit_.request_boundary_source_catalog_identity);
  identity = mix(identity, audit_.request_boundary_source_seal_identity);
  identity = mix(identity, audit_.request_boundary_resident_root_identity);
  identity = mix(identity, audit_.request_boundary_resident_arena_bytes);
  identity = mix(identity, resource_bundle_identity);
  identity = mix(identity, sealed.size());
  identity = mix(identity, binding.binding_identity);
  if (identity == 0U) {
    return fail(0U, static_cast<int>(cudaErrorInvalidValue));
  }
  *catalog = sealed;
  *catalog_identity = identity;
  *failure_index = sealed.size();
  *cuda_error = 0;
  return true;
}

std::uint64_t Sm87MacroFeedV4P40StartupPackage::
    compute_mlp_pair_execution_binding_identity(
        const MlpPairExecutionBinding& binding) noexcept {
  const std::uint64_t gate_identity =
      compute_nvfp4_asset_value_identity(binding.gate_up.asset);
  const std::uint64_t down_identity =
      compute_nvfp4_asset_value_identity(binding.down.asset);
  const auto& gate_upload = binding.gate_up.asset.device_upload_receipt;
  const auto& down_upload = binding.down.asset.device_upload_receipt;
  if (binding.model_layer >= kSm87MacroFeedV4P40StartupPackageLayers ||
      binding.package_identity == 0U ||
      binding.deployment_plan_identity == 0U ||
      binding.owner_identity == 0U || binding.allocation_identity == 0U ||
      binding.projection_catalog_identity == 0U ||
      binding.device_identity == 0U ||
      binding.gate_up.asset.payload.role != Role::kNvFp4GateUp ||
      binding.down.asset.payload.role != Role::kNvFp4Down ||
      binding.gate_up.projection_binding_identity == 0U ||
      binding.down.projection_binding_identity == 0U ||
      binding.gate_up.asset_value_identity == 0U ||
      binding.down.asset_value_identity == 0U ||
      binding.gate_up.asset_value_identity != gate_identity ||
      binding.down.asset_value_identity != down_identity ||
      !kernels::sm87_macrofeed_v3_nvfp4_gate_up_payload_receipt_valid(
          binding.gate_up.payload_receipt) ||
      !kernels::sm87_macrofeed_v3_nvfp4_down_payload_receipt_valid(
          binding.down.payload_receipt) ||
      binding.gate_up.payload_receipt.payload_identity !=
          binding.gate_up.asset.artifact_identity ||
      binding.gate_up.payload_receipt.device_ordinal !=
          gate_upload.device_ordinal ||
      binding.gate_up.payload_receipt.payload_begin !=
          binding.gate_up.asset.payload.begin ||
      binding.gate_up.payload_receipt.payload_end !=
          binding.gate_up.asset.payload.end ||
      binding.gate_up.payload_receipt.payload_bytes !=
          binding.gate_up.asset.payload.bytes ||
      binding.down.payload_receipt.payload_identity !=
          binding.down.asset.artifact_identity ||
      binding.down.payload_receipt.device_ordinal !=
          down_upload.device_ordinal ||
      binding.down.payload_receipt.payload_begin !=
          binding.down.asset.payload.begin ||
      binding.down.payload_receipt.payload_end !=
          binding.down.asset.payload.end ||
      binding.down.payload_receipt.payload_bytes !=
          binding.down.asset.payload.bytes ||
      binding.gate_up.tactic_identity != static_cast<std::uint64_t>(
          kernels::kSm87MacroFeedV4NvFp4GateUpIdentity) ||
      binding.down.tactic_identity != static_cast<std::uint64_t>(
          kernels::kSm87MacroFeedV4NvFp4DownIdentity) ||
      gate_upload.device_allocation_owner_identity != binding.owner_identity ||
      down_upload.device_allocation_owner_identity != binding.owner_identity ||
      gate_upload.device_allocation_identity != binding.allocation_identity ||
      down_upload.device_allocation_identity != binding.allocation_identity ||
      !binding.live_cuda_payload_ranges_validated ||
      binding.request_selectable || binding.launcher_authority ||
      binding.production_dispatch_eligible) {
    return 0U;
  }
  std::uint64_t identity = 0x5133'4d46'4d4c'5042ULL;
  identity = mix(identity, binding.model_layer + 1U);
  identity = mix(identity, binding.package_identity);
  identity = mix(identity, binding.deployment_plan_identity);
  identity = mix(identity, binding.owner_identity);
  identity = mix(identity, binding.allocation_identity);
  identity = mix(identity, binding.projection_catalog_identity);
  identity = mix(identity, binding.device_identity);
  identity = mix(identity, binding.gate_up.projection_binding_identity);
  identity = mix(identity, binding.gate_up.asset_value_identity);
  identity = mix(identity, binding.gate_up.tactic_identity);
  identity = mix(identity, binding.gate_up.payload_receipt.receipt_identity);
  identity = mix(identity, binding.down.projection_binding_identity);
  identity = mix(identity, binding.down.asset_value_identity);
  identity = mix(identity, binding.down.tactic_identity);
  identity = mix(identity, binding.down.payload_receipt.receipt_identity);
  identity = mix(identity, binding.live_cuda_payload_ranges_validated);
  identity = mix(identity, binding.request_selectable);
  identity = mix(identity, binding.launcher_authority);
  identity = mix(identity, binding.production_dispatch_eligible);
  return identity == 0U ? 0x5133'4d46'4d4c'5042ULL : identity;
}

bool Sm87MacroFeedV4P40StartupPackage::
    seal_mlp_pair_execution_catalog_for_execution_package(
        MlpPairExecutionBindingCatalog* const catalog,
        std::uint64_t* const catalog_identity,
        std::size_t* const failure_layer,
        int* const cuda_error) const noexcept {
  if (catalog != nullptr) {
    catalog->fill({});
  }
  if (catalog_identity != nullptr) {
    *catalog_identity = 0U;
  }
  if (failure_layer != nullptr) {
    *failure_layer = kSm87MacroFeedV4P40StartupPackageLayers;
  }
  if (cuda_error != nullptr) {
    *cuda_error = 0;
  }
  if (catalog == nullptr || catalog_identity == nullptr ||
      failure_layer == nullptr || cuda_error == nullptr || !valid()) {
    return false;
  }

  MlpPairExecutionBindingCatalog sealed{};
  std::uint64_t identity = 0x5133'4d46'4d4c'5045ULL;
  identity = mix(identity, seals_.gate_up.binding_catalog_identity);
  identity = mix(identity, seals_.gate_up.seal_identity);
  identity = mix(identity, seals_.down.seal_identity);
  identity = mix(identity, sealed.size());
  for (std::size_t model_layer = 0U; model_layer < sealed.size();
       ++model_layer) {
    const std::size_t gate_descriptor =
        sm87_target_aot_complete_descriptor_ordinal(
            model_layer, Role::kNvFp4GateUp);
    const std::size_t down_descriptor =
        sm87_target_aot_complete_descriptor_ordinal(
            model_layer, Role::kNvFp4Down);
    if (gate_descriptor >= projection_bindings_.size() ||
        down_descriptor >= projection_bindings_.size() ||
        !projection_bindings_[gate_descriptor] ||
        !projection_bindings_[down_descriptor]) {
      *failure_layer = model_layer;
      return false;
    }
    const auto& gate_projection = *projection_bindings_[gate_descriptor];
    const auto& down_projection = *projection_bindings_[down_descriptor];
    const auto* const gate_asset =
        gate_projection.asset_.borrow_nvfp4_cuda_asset();
    const auto* const down_asset =
        down_projection.asset_.borrow_nvfp4_cuda_asset();
    const auto* const gate_source_descriptor =
        gate_projection.asset_.descriptor_;
    const auto* const down_source_descriptor =
        down_projection.asset_.descriptor_;
    const auto gate_layout = kernels::sm87_target_aot_projection_packed_layout(
        Role::kNvFp4GateUp);
    const auto down_layout = kernels::sm87_target_aot_projection_packed_layout(
        Role::kNvFp4Down);
    if (!gate_projection.valid_for_prevalidated_catalog(
            model_layer, Role::kNvFp4GateUp, audit_.package_identity,
            audit_.catalog_identity) ||
        !down_projection.valid_for_prevalidated_catalog(
            model_layer, Role::kNvFp4Down, audit_.package_identity,
            audit_.catalog_identity) ||
        gate_asset == nullptr || down_asset == nullptr ||
        gate_projection.asset_.borrow_fp8_cuda_asset() != nullptr ||
        down_projection.asset_.borrow_fp8_cuda_asset() != nullptr ||
        gate_source_descriptor == nullptr ||
        down_source_descriptor == nullptr || !gate_layout.valid() ||
        !down_layout.valid() ||
        gate_source_descriptor->layer_index != model_layer ||
        down_source_descriptor->layer_index != model_layer ||
        gate_source_descriptor->role != Role::kNvFp4GateUp ||
        down_source_descriptor->role != Role::kNvFp4Down ||
        gate_source_descriptor->manifest.artifact_identity !=
            gate_asset->artifact_identity ||
        down_source_descriptor->manifest.artifact_identity !=
            down_asset->artifact_identity ||
        gate_source_descriptor->manifest.seal.value !=
            gate_asset->host_manifest_seal.value ||
        down_source_descriptor->manifest.seal.value !=
            down_asset->host_manifest_seal.value ||
        gate_source_descriptor->source_inventory.identity !=
            gate_asset->source_inventory_identity ||
        down_source_descriptor->source_inventory.identity !=
            down_asset->source_inventory_identity ||
        !gate_source_descriptor->source_inventory.valid(gate_layout) ||
        !down_source_descriptor->source_inventory.valid(down_layout) ||
        !kernels::sm87_target_aot_projection_validate_packed_manifest(
            gate_source_descriptor->manifest,
            gate_source_descriptor->source_inventory) ||
        !kernels::sm87_target_aot_projection_validate_packed_manifest(
            down_source_descriptor->manifest,
            down_source_descriptor->source_inventory) ||
        gate_source_descriptor->source_inventory.source_count != 2U ||
        down_source_descriptor->source_inventory.source_count != 1U ||
        gate_source_descriptor->source_inventory.sources[0U].logical_role !=
            kernels::Sm87TargetAotLogicalRole::kNvFp4Gate ||
        gate_source_descriptor->source_inventory.sources[1U].logical_role !=
            kernels::Sm87TargetAotLogicalRole::kNvFp4Up ||
        down_source_descriptor->source_inventory.sources[0U].logical_role !=
            kernels::Sm87TargetAotLogicalRole::kNvFp4Down ||
        gate_source_descriptor->source_inventory.sources[0U]
                .tensor_scale_bits != gate_asset->tensor_scale_bits[0U] ||
        gate_source_descriptor->source_inventory.sources[1U]
                .tensor_scale_bits != gate_asset->tensor_scale_bits[1U] ||
        down_source_descriptor->source_inventory.sources[0U]
                .tensor_scale_bits != down_asset->tensor_scale_bits[0U] ||
        !kernels::sm87_target_aot_nvfp4_cuda_asset_valid(*gate_asset) ||
        !kernels::sm87_target_aot_nvfp4_cuda_asset_valid(*down_asset) ||
        !authenticated_upload_complete(
            gate_asset->device_upload_receipt, audit_.owner_identity,
            audit_.allocation_identity, audit_.device_ordinal,
            gate_asset->payload.begin, gate_asset->payload.end,
            gate_asset->payload.bytes) ||
        !authenticated_upload_complete(
            down_asset->device_upload_receipt, audit_.owner_identity,
            audit_.allocation_identity, audit_.device_ordinal,
            down_asset->payload.begin, down_asset->payload.end,
            down_asset->payload.bytes) ||
        compute_nvfp4_asset_value_identity(*gate_asset) == 0U ||
        compute_nvfp4_asset_value_identity(*down_asset) == 0U ||
        !live_current_device_allocation_range(
            reinterpret_cast<const std::uint16_t*>(gate_asset->payload.begin),
            gate_asset->payload.bytes, audit_.device_ordinal, cuda_error,
            gate_asset->device_upload_receipt.device_allocation_begin,
            gate_asset->device_upload_receipt.device_allocation_end,
            gate_asset->device_upload_receipt.device_allocation_bytes) ||
        !live_current_device_allocation_range(
            reinterpret_cast<const std::uint16_t*>(down_asset->payload.begin),
            down_asset->payload.bytes, audit_.device_ordinal, cuda_error,
            down_asset->device_upload_receipt.device_allocation_begin,
            down_asset->device_upload_receipt.device_allocation_end,
            down_asset->device_upload_receipt.device_allocation_bytes)) {
      catalog->fill({});
      *failure_layer = model_layer;
      return false;
    }

    kernels::Sm87MacroFeedV3NvFp4GateUpPayloadReceipt gate_receipt{};
    gate_receipt.plan_identity =
        kernels::kSm87MacroFeedV3NvFp4GateUpIdentity;
    gate_receipt.payload_identity = gate_asset->artifact_identity;
    gate_receipt.gate_source_identity =
        gate_source_descriptor->source_inventory.sources[0U].tensor_identity;
    gate_receipt.up_source_identity =
        gate_source_descriptor->source_inventory.sources[1U].tensor_identity;
    gate_receipt.device_ordinal = audit_.device_ordinal;
    gate_receipt.payload_begin = gate_asset->payload.begin;
    gate_receipt.payload_end = gate_asset->payload.end;
    gate_receipt.payload_bytes = gate_asset->payload.bytes;
    gate_receipt.gate_partition_bytes =
        gate_layout.partitions[0U].payload_bytes;
    gate_receipt.up_partition_bytes = gate_layout.partitions[1U].payload_bytes;
    gate_receipt.canonical_consumer_n64_k16_lane_component_v1 =
        gate_asset->transform_identity ==
        kernels::Sm87TargetAotProjectionPackedTransformIdentity::
            kCanonicalNkToConsumerN64K16LaneComponentV1;
    gate_receipt.canonical_gate_then_up_partition_order = true;
    gate_receipt.independent_tensor_scales =
        gate_layout.partitions[0U].independent_tensor_scale &&
        gate_layout.partitions[1U].independent_tensor_scale;
    gate_receipt.host_bytes_authenticated_before_copy =
        gate_asset->device_upload_receipt
            .host_payload_digest_verified_before_copy &&
        gate_asset->device_upload_receipt
            .host_payload_immutable_until_completion;
    gate_receipt.device_readback_authenticated =
        gate_asset->device_upload_receipt.verification_event_observed &&
        gate_asset->device_upload_receipt.verification_completed &&
        gate_asset->device_upload_receipt.device_payload_matches_host_payload;
    gate_receipt.allocation_retained_for_launch =
        gate_asset->device_upload_receipt
            .allocation_retained_for_asset_lifetime;
    gate_receipt.receipt_identity = kernels::
        sm87_macrofeed_v3_nvfp4_gate_up_compute_payload_receipt_identity(
            gate_receipt);

    kernels::Sm87MacroFeedV3NvFp4DownPayloadReceipt down_receipt{};
    down_receipt.plan_identity = kernels::kSm87MacroFeedV3NvFp4DownIdentity;
    down_receipt.payload_identity = down_asset->artifact_identity;
    down_receipt.device_ordinal = audit_.device_ordinal;
    down_receipt.payload_begin = down_asset->payload.begin;
    down_receipt.payload_end = down_asset->payload.end;
    down_receipt.payload_bytes = down_asset->payload.bytes;
    down_receipt.canonical_consumer_n64_k16_lane_component_v1 =
        down_asset->transform_identity ==
        kernels::Sm87TargetAotProjectionPackedTransformIdentity::
            kCanonicalNkToConsumerN64K16LaneComponentV1;
    down_receipt.host_bytes_authenticated_before_copy =
        down_asset->device_upload_receipt
            .host_payload_digest_verified_before_copy &&
        down_asset->device_upload_receipt
            .host_payload_immutable_until_completion;
    down_receipt.device_readback_authenticated =
        down_asset->device_upload_receipt.verification_event_observed &&
        down_asset->device_upload_receipt.verification_completed &&
        down_asset->device_upload_receipt.device_payload_matches_host_payload;
    down_receipt.allocation_retained_for_launch =
        down_asset->device_upload_receipt
            .allocation_retained_for_asset_lifetime;
    down_receipt.receipt_identity = kernels::
        sm87_macrofeed_v3_nvfp4_down_compute_payload_receipt_identity(
            down_receipt);
    if (!kernels::sm87_macrofeed_v3_nvfp4_gate_up_payload_receipt_valid(
            gate_receipt) ||
        !kernels::sm87_macrofeed_v3_nvfp4_down_payload_receipt_valid(
            down_receipt)) {
      catalog->fill({});
      *failure_layer = model_layer;
      *cuda_error = static_cast<int>(cudaErrorInvalidValue);
      return false;
    }

    auto& binding = sealed[model_layer];
    binding.model_layer = static_cast<std::uint32_t>(model_layer);
    binding.gate_up.asset = *gate_asset;
    binding.gate_up.payload_receipt = gate_receipt;
    binding.gate_up.projection_binding_identity =
        gate_projection.binding_identity();
    binding.gate_up.asset_value_identity =
        compute_nvfp4_asset_value_identity(*gate_asset);
    binding.gate_up.tactic_identity = static_cast<std::uint64_t>(
        kernels::kSm87MacroFeedV4NvFp4GateUpIdentity);
    binding.down.asset = *down_asset;
    binding.down.payload_receipt = down_receipt;
    binding.down.projection_binding_identity =
        down_projection.binding_identity();
    binding.down.asset_value_identity =
        compute_nvfp4_asset_value_identity(*down_asset);
    binding.down.tactic_identity = static_cast<std::uint64_t>(
        kernels::kSm87MacroFeedV4NvFp4DownIdentity);
    binding.package_identity = audit_.package_identity;
    binding.deployment_plan_identity = audit_.deployment_plan_identity;
    binding.owner_identity = audit_.owner_identity;
    binding.allocation_identity = audit_.allocation_identity;
    binding.projection_catalog_identity = audit_.catalog_identity;
    binding.device_identity = audit_.device_identity;
    binding.live_cuda_payload_ranges_validated = true;
    binding.request_selectable = false;
    binding.launcher_authority = false;
    binding.production_dispatch_eligible = false;
    binding.binding_identity =
        compute_mlp_pair_execution_binding_identity(binding);
    if (binding.binding_identity == 0U) {
      catalog->fill({});
      *failure_layer = model_layer;
      *cuda_error = static_cast<int>(cudaErrorInvalidValue);
      return false;
    }
    identity = mix(identity, model_layer + 1U);
    identity = mix(identity, binding.binding_identity);
  }
  if (compute_mlp_pair_binding_catalog_identity(
          projection_access_, capabilities_, audit_.deployment_plan_identity) !=
      seals_.gate_up.binding_catalog_identity) {
    *failure_layer = 0U;
    *cuda_error = static_cast<int>(cudaErrorInvalidValue);
    return false;
  }
  *catalog = sealed;
  *catalog_identity = identity;
  *failure_layer = sealed.size();
  return true;
}

bool Sm87MacroFeedV4P40StartupPackage::
    seal_layer_norm_execution_catalog_for_execution_package(
        LayerNormExecutionBindingCatalog* const catalog,
        std::uint64_t* const catalog_identity,
        std::size_t* const failure_layer,
        bool* const failure_post_attention,
        int* const cuda_error) const noexcept {
  if (catalog != nullptr) {
    catalog->fill({});
  }
  if (catalog_identity != nullptr) {
    *catalog_identity = 0U;
  }
  if (failure_layer != nullptr) {
    *failure_layer = kSm87MacroFeedV4P40StartupPackageLayers;
  }
  if (failure_post_attention != nullptr) {
    *failure_post_attention = false;
  }
  if (cuda_error != nullptr) {
    *cuda_error = 0;
  }
  if (catalog == nullptr || catalog_identity == nullptr ||
      failure_layer == nullptr || failure_post_attention == nullptr ||
      cuda_error == nullptr) {
    return false;
  }

  // This is the sole complete package revalidation.  Everything below is a
  // construction-time seal over the already-retained ModelWeights root.
  if (!valid()) {
    return false;
  }
  const ModelWeights* const model_weights =
      bf16_ab_capability_.model_weights_;
  if (model_weights == nullptr) {
    return false;
  }

  LayerNormExecutionBindingCatalog sealed{};
  std::array<std::uintptr_t,
             2U * kSm87MacroFeedV4P40StartupPackageLayers>
      range_begins{};
  std::array<std::uintptr_t,
             2U * kSm87MacroFeedV4P40StartupPackageLayers>
      range_ends{};
  std::size_t sealed_ranges = 0U;

  constexpr std::size_t kElements =
      kernels::kSm87MacroFeedV4NormResidualHidden;
  constexpr std::uint64_t kBytes =
      kernels::kSm87MacroFeedV4NormResidualWeightBytes;
  constexpr std::uintptr_t kAlignment = 256U;
  constexpr std::uint32_t kEpsilonFp32Bits =
      kernels::kSm87MacroFeedV4NormResidualEpsilonFp32Bits;
  static_assert(kElements == 5'120U);
  static_assert(kBytes == 10'240U);
  static_assert(kEpsilonFp32Bits == 0x3586'37bdU);

  for (std::size_t model_layer = 0U;
       model_layer < kSm87MacroFeedV4P40StartupPackageLayers;
       ++model_layer) {
    const DecoderLayerWeights& layer = model_weights->layer(model_layer);
    const std::array<const Bf16VectorWeight*, 2U> weights{{
        &layer.input_layernorm,
        &layer.post_attention_layernorm,
    }};
    auto& binding = sealed[model_layer];
    binding.model_layer = static_cast<std::uint32_t>(model_layer);
    binding.epsilon_fp32_bits = kEpsilonFp32Bits;

    for (std::size_t role_index = 0U; role_index < weights.size();
         ++role_index) {
      const bool post_attention = role_index == 1U;
      const Bf16VectorWeight& weight = *weights[role_index];
      const std::uintptr_t begin =
          reinterpret_cast<std::uintptr_t>(weight.data);
      if (weight.data == nullptr || weight.element_count != kElements ||
          begin % kAlignment != 0U ||
          begin > std::numeric_limits<std::uintptr_t>::max() - kBytes) {
        *failure_layer = model_layer;
        *failure_post_attention = post_attention;
        *cuda_error = static_cast<int>(cudaErrorInvalidValue);
        return false;
      }
      const std::uintptr_t end = begin + kBytes;
      int range_cuda_error = 0;
      if (!live_current_device_allocation_range(
              weight.data, kBytes, audit_.device_ordinal,
              &range_cuda_error)) {
        *failure_layer = model_layer;
        *failure_post_attention = post_attention;
        *cuda_error = range_cuda_error;
        return false;
      }
      for (std::size_t earlier = 0U; earlier < sealed_ranges; ++earlier) {
        if (!(range_ends[earlier] <= begin ||
              end <= range_begins[earlier])) {
          *failure_layer = model_layer;
          *failure_post_attention = post_attention;
          *cuda_error = static_cast<int>(cudaErrorInvalidValue);
          return false;
        }
      }
      range_begins[sealed_ranges] = begin;
      range_ends[sealed_ranges] = end;
      ++sealed_ranges;

      const std::uint64_t tensor_identity = layer_norm_tensor_identity(
          audit_.package_identity, audit_.deployment_plan_identity,
          audit_.device_identity, audit_.device_ordinal, model_layer,
          post_attention, weight.data, weight.element_count,
          kEpsilonFp32Bits);
      if (tensor_identity == 0U) {
        *failure_layer = model_layer;
        *failure_post_attention = post_attention;
        *cuda_error = static_cast<int>(cudaErrorInvalidValue);
        return false;
      }
      if (post_attention) {
        binding.post_attention_layernorm = weight.data;
        binding.post_attention_layernorm_identity = tensor_identity;
      } else {
        binding.input_layernorm = weight.data;
        binding.input_layernorm_identity = tensor_identity;
      }
    }

    binding.pair_identity = layer_norm_pair_identity(
        model_layer, binding.input_layernorm_identity,
        binding.post_attention_layernorm_identity,
        binding.epsilon_fp32_bits);
    if (binding.pair_identity == 0U) {
      *failure_layer = model_layer;
      *failure_post_attention = true;
      *cuda_error = static_cast<int>(cudaErrorInvalidValue);
      return false;
    }
  }

  if (sealed_ranges != 2U * sealed.size()) {
    return false;
  }
  std::uint64_t identity = 0x5133'4d46'4e43'4154ULL;
  identity = mix(identity, audit_.package_identity);
  identity = mix(identity, audit_.deployment_plan_identity);
  identity = mix(identity, audit_.device_identity);
  identity = mix(identity,
                 static_cast<std::uint64_t>(audit_.device_ordinal + 1));
  identity = mix(identity, sealed.size());
  identity = mix(identity, sealed_ranges);
  identity = mix(identity, kElements);
  identity = mix(identity, kBytes);
  identity = mix(identity, kAlignment);
  identity = mix(identity, kEpsilonFp32Bits);
  for (std::size_t model_layer = 0U; model_layer < sealed.size();
       ++model_layer) {
    const auto& binding = sealed[model_layer];
    identity = mix(identity, model_layer + 1U);
    identity = mix(identity, binding.model_layer + 1U);
    identity = mix(identity, binding.input_layernorm_identity);
    identity = mix(identity, binding.post_attention_layernorm_identity);
    identity = mix(identity, binding.pair_identity);
    identity = mix(identity, binding.epsilon_fp32_bits);
  }
  if (identity == 0U) {
    return false;
  }

  *catalog = sealed;
  *catalog_identity = identity;
  return true;
}

bool Sm87MacroFeedV4P40StartupPackage::populate_projection_bindings()
    noexcept {
  if (!base_valid()) {
    return false;
  }
  std::size_t bindings = 0U;
  for (std::size_t layer_index = 0U;
       layer_index < kSm87MacroFeedV4P40StartupPackageLayers;
       ++layer_index) {
    for (const Role role : layer_roles(layer_index)) {
      const std::size_t index =
          sm87_target_aot_complete_descriptor_ordinal(layer_index, role);
      if (index != bindings || index >= projection_bindings_.size() ||
          projection_bindings_[index].has_value()) {
        return false;
      }
      auto binding = make_projection_binding(layer_index, role);
      if (!binding) {
        return false;
      }
      projection_bindings_[index].emplace(std::move(*binding));
      ++bindings;
    }
  }
  return bindings == projection_bindings_.size();
}

bool Sm87MacroFeedV4P40StartupPackage::valid() const noexcept {
  if (!base_valid()) {
    return false;
  }
  std::size_t bindings = 0U;
  for (std::size_t layer_index = 0U;
       layer_index < kSm87MacroFeedV4P40StartupPackageLayers;
       ++layer_index) {
    for (const Role role : layer_roles(layer_index)) {
      const std::size_t index =
          sm87_target_aot_complete_descriptor_ordinal(layer_index, role);
      if (index != bindings || index >= projection_bindings_.size() ||
          !projection_bindings_[index] ||
          !projection_bindings_[index]->valid_for_prevalidated_catalog(
              layer_index, role, audit_.package_identity,
              audit_.catalog_identity) ||
          projection_bindings_[index]->deployment_plan_identity() !=
              audit_.deployment_plan_identity) {
        return false;
      }
      ++bindings;
    }
  }
  return bindings == projection_bindings_.size();
}

const Sm87MacroFeedV4ProjectionStartupBinding*
Sm87MacroFeedV4P40StartupPackage::borrow_projection_startup_binding(
    const std::size_t layer_index, const Role role) const noexcept {
  const std::size_t index =
      sm87_target_aot_complete_descriptor_ordinal(layer_index, role);
  if (index >= projection_bindings_.size() ||
      !projection_bindings_[index] || !base_valid()) {
    return nullptr;
  }
  const auto& binding = *projection_bindings_[index];
  return binding.valid_for_prevalidated_catalog(
             layer_index, role, audit_.package_identity,
             audit_.catalog_identity) &&
                 binding.deployment_plan_identity() ==
                     audit_.deployment_plan_identity
             ? &binding
             : nullptr;
}

bool Sm87MacroFeedV4P40StartupPackage::borrow_projection_startup_catalog(
    ProjectionStartupBindingCatalog* const catalog) const noexcept {
  if (catalog == nullptr) {
    return false;
  }
  catalog->fill(nullptr);
  if (!base_valid()) {
    return false;
  }

  std::size_t bindings = 0U;
  for (std::size_t layer_index = 0U;
       layer_index < kSm87MacroFeedV4P40StartupPackageLayers;
       ++layer_index) {
    for (const Role role : layer_roles(layer_index)) {
      const std::size_t index =
          sm87_target_aot_complete_descriptor_ordinal(layer_index, role);
      if (index != bindings || index >= projection_bindings_.size() ||
          !projection_bindings_[index] ||
          !projection_bindings_[index]->valid_for_prevalidated_catalog(
              layer_index, role, audit_.package_identity,
              audit_.catalog_identity) ||
          projection_bindings_[index]->deployment_plan_identity() !=
              audit_.deployment_plan_identity) {
        catalog->fill(nullptr);
        return false;
      }
      (*catalog)[index] = &*projection_bindings_[index];
      ++bindings;
    }
  }
  if (bindings != catalog->size()) {
    catalog->fill(nullptr);
    return false;
  }
  return true;
}

#else

std::uint32_t Sm87MacroFeedV4ProjectionStartupBinding::tensor_scale_bits(
    const std::size_t source_index) const noexcept {
  (void)source_index;
  return 0U;
}

bool Sm87MacroFeedV4ProjectionStartupBinding::valid() const noexcept {
  return false;
}

bool Sm87MacroFeedV4P40StartupPackage::
    seal_gdn_layer_execution_catalog_for_execution_package(
        GdnLayerExecutionBindingCatalog* const catalog,
        std::uint64_t* const catalog_identity,
        std::size_t* const failure_ordinal,
        int* const cuda_error) const noexcept {
  return seal_gdn_qkvz_execution_catalog_for_execution_package(
      catalog, catalog_identity, failure_ordinal, cuda_error);
}

bool Sm87MacroFeedV4P40StartupPackage::
    seal_mlp_pair_execution_catalog_for_execution_package(
        MlpPairExecutionBindingCatalog* const catalog,
        std::uint64_t* const catalog_identity,
        std::size_t* const failure_layer,
        int* const cuda_error) const noexcept {
  if (catalog != nullptr) {
    catalog->fill({});
  }
  if (catalog_identity != nullptr) {
    *catalog_identity = 0U;
  }
  if (failure_layer != nullptr) {
    *failure_layer = kSm87MacroFeedV4P40StartupPackageLayers;
  }
  if (cuda_error != nullptr) {
    *cuda_error = 0;
  }
  return false;
}

bool Sm87MacroFeedV4P40StartupPackage::
    seal_full_attention_execution_catalog_for_execution_package(
        const FullAttentionExecutionResourceObservations& observations,
        FullAttentionLayerExecutionBindingCatalog* const catalog,
        std::uint64_t* const catalog_identity,
        std::size_t* const failure_ordinal,
        int* const cuda_error) const noexcept {
  (void)observations;
  if (catalog != nullptr) {
    catalog->fill({});
  }
  if (catalog_identity != nullptr) {
    *catalog_identity = 0U;
  }
  if (failure_ordinal != nullptr) {
    *failure_ordinal = kSm87MacroFeedV4P40StartupPackageFullLayers;
  }
  if (cuda_error != nullptr) {
    *cuda_error = 0;
  }
  return false;
}

bool Sm87MacroFeedV4P40StartupPackage::
    seal_request_boundary_execution_catalog_for_execution_package(
        const RequestBoundaryExecutionResourceObservations& observations,
        RequestBoundaryExecutionBindingCatalog* const catalog,
        std::uint64_t* const catalog_identity,
        std::size_t* const failure_index,
        int* const cuda_error) const noexcept {
  (void)observations;
  if (catalog != nullptr) {
    catalog->fill({});
  }
  if (catalog_identity != nullptr) {
    *catalog_identity = 0U;
  }
  if (failure_index != nullptr) {
    *failure_index =
        kSm87MacroFeedV4P40StartupPackageRequestBoundaryBindings;
  }
  if (cuda_error != nullptr) {
    *cuda_error = 0;
  }
  return false;
}

bool Sm87MacroFeedV4ProjectionStartupBinding::valid_with_catalog(
    const std::uint64_t catalog_identity) const noexcept {
  (void)catalog_identity;
  return false;
}

bool Sm87MacroFeedV4ProjectionStartupBinding::
    valid_with_prevalidated_catalog(
        const std::uint64_t catalog_identity) const noexcept {
  (void)catalog_identity;
  return false;
}

bool Sm87MacroFeedV4ProjectionStartupBinding::
    valid_for_prevalidated_catalog(
        const std::size_t layer_index, const Role role,
        const std::uint64_t package_identity,
        const std::uint64_t catalog_identity) const noexcept {
  (void)layer_index;
  (void)role;
  (void)package_identity;
  (void)catalog_identity;
  return false;
}

bool Sm87MacroFeedV4ProjectionStartupBinding::valid_for(
    const std::size_t layer_index, const Role role,
    const std::uint64_t package_identity) const noexcept {
  (void)layer_index;
  (void)role;
  (void)package_identity;
  return false;
}

bool Sm87MacroFeedV4P40StartupPackage::valid() const noexcept {
  return false;
}

bool Sm87MacroFeedV4P40StartupPackage::base_valid() const noexcept {
  return false;
}

bool Sm87MacroFeedV4P40StartupPackage::
    seal_bf16_ab_execution_catalog_for_execution_package(
        Bf16AbExecutionBindingCatalog* const catalog) const noexcept {
  if (catalog != nullptr) {
    catalog->fill({});
  }
  return false;
}

bool Sm87MacroFeedV4P40StartupPackage::
    seal_layer_norm_execution_catalog_for_execution_package(
        LayerNormExecutionBindingCatalog* const catalog,
        std::uint64_t* const catalog_identity,
        std::size_t* const failure_layer,
        bool* const failure_post_attention,
        int* const cuda_error) const noexcept {
  if (catalog != nullptr) {
    catalog->fill({});
  }
  if (catalog_identity != nullptr) {
    *catalog_identity = 0U;
  }
  if (failure_layer != nullptr) {
    *failure_layer = kSm87MacroFeedV4P40StartupPackageLayers;
  }
  if (failure_post_attention != nullptr) {
    *failure_post_attention = false;
  }
  if (cuda_error != nullptr) {
    *cuda_error = 0;
  }
  return false;
}

bool Sm87MacroFeedV4P40StartupPackage::
    seal_gdn_qkvz_execution_catalog_for_execution_package(
        GdnQkvZExecutionBindingCatalog* const catalog,
        std::uint64_t* const catalog_identity,
        std::size_t* const failure_ordinal,
        int* const cuda_error) const noexcept {
  if (catalog != nullptr) {
    catalog->fill({});
  }
  if (catalog_identity != nullptr) {
    *catalog_identity = 0U;
  }
  if (failure_ordinal != nullptr) {
    *failure_ordinal = kSm87MacroFeedV4P40StartupPackageGdnLayers;
  }
  if (cuda_error != nullptr) {
    *cuda_error = 0;
  }
  return false;
}

bool Sm87MacroFeedV4P40StartupPackage::populate_projection_bindings()
    noexcept {
  return false;
}

const Sm87MacroFeedV4ProjectionStartupBinding*
Sm87MacroFeedV4P40StartupPackage::borrow_projection_startup_binding(
    const std::size_t layer_index, const Role role) const noexcept {
  (void)layer_index;
  (void)role;
  return nullptr;
}

bool Sm87MacroFeedV4P40StartupPackage::borrow_projection_startup_catalog(
    ProjectionStartupBindingCatalog* const catalog) const noexcept {
  if (catalog != nullptr) {
    catalog->fill(nullptr);
  }
  return false;
}

#endif

}  // namespace q3x::runtime::sm87_macrofeed_v4_p40_startup_package_detail
