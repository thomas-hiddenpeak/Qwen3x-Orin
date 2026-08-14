#include "sm87_macrofeed_v3_receipt_internal.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace q3x::runtime {
namespace {

constexpr std::uint64_t kFnvOffset = 14'695'981'039'346'656'037ULL;
constexpr std::uint64_t kFnvPrime = 1'099'511'628'211ULL;

void hash_byte(std::uint64_t* const hash, const std::uint8_t value) noexcept {
  *hash ^= value;
  *hash *= kFnvPrime;
}

void hash_u64(std::uint64_t* const hash, const std::uint64_t value) noexcept {
  for (std::size_t byte = 0U; byte < sizeof(value); ++byte) {
    hash_byte(hash, static_cast<std::uint8_t>(value >> (byte * 8U)));
  }
}

void hash_bool(std::uint64_t* const hash, const bool value) noexcept {
  hash_byte(hash, value ? 1U : 0U);
}

void hash_physical_identity(
    std::uint64_t* const hash,
    const Sm87MacroFeedV3PhysicalIdentity& identity) noexcept {
  hash_u64(hash, identity.package_identity);
  hash_u64(hash, identity.owner_identity);
  hash_u64(hash, identity.allocation_identity);
  hash_u64(hash, identity.catalog_identity);
  hash_u64(hash, identity.device_identity);
  hash_u64(hash, static_cast<std::uint32_t>(identity.device_ordinal));
}

[[nodiscard]] bool physical_identity_valid(
    const Sm87MacroFeedV3PhysicalIdentity& identity) noexcept {
  return identity.package_identity != 0U && identity.owner_identity != 0U &&
         identity.allocation_identity != 0U &&
         identity.catalog_identity != 0U && identity.device_identity != 0U &&
         identity.device_ordinal >= 0;
}

[[nodiscard]] bool physical_identity_equal(
    const Sm87MacroFeedV3PhysicalIdentity& left,
    const Sm87MacroFeedV3PhysicalIdentity& right) noexcept {
  return left.package_identity == right.package_identity &&
         left.owner_identity == right.owner_identity &&
         left.allocation_identity == right.allocation_identity &&
         left.catalog_identity == right.catalog_identity &&
         left.device_identity == right.device_identity &&
         left.device_ordinal == right.device_ordinal;
}

[[nodiscard]] bool artifact_identities_valid(
    const Sm87MacroFeedV3ArtifactIdentities& artifacts) noexcept {
  const std::array<std::uint64_t, kSm87MacroFeedV3ReceiptArtifactsPerLayer>
      values{{artifacts.fp8_input, artifacts.fp8_output,
              artifacts.nvfp4_gate_up, artifacts.nvfp4_down}};
  for (std::size_t first = 0U; first < values.size(); ++first) {
    if (values[first] == 0U) {
      return false;
    }
    for (std::size_t second = first + 1U; second < values.size(); ++second) {
      if (values[first] == values[second]) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] bool forbidden_route_absent(
    const Sm87MacroFeedV3LayerReceipt& receipt) noexcept {
  return !receipt.used_fallback && !receipt.used_mtp &&
         !receipt.used_cublaslt && !receipt.used_request_jit &&
         !receipt.used_request_repack && !receipt.used_request_autotune;
}

[[nodiscard]] bool layer_fields_valid(
    const Sm87MacroFeedV3LayerReceipt& receipt) noexcept {
  if (receipt.request_identity == 0U ||
      receipt.layer_type !=
          sm87_macrofeed_v3_expected_layer_type(receipt.layer_index) ||
      !artifact_identities_valid(receipt.artifacts) ||
      !physical_identity_valid(receipt.physical_identity) ||
      receipt.physical_counts.fp8_input != 1U ||
      receipt.physical_counts.fp8_output != 1U ||
      receipt.physical_counts.nvfp4_gate_up != 1U ||
      receipt.physical_counts.nvfp4_down != 1U ||
      !receipt.fill_completed || !receipt.drain_completed ||
      !receipt.completion_observed || !forbidden_route_absent(receipt)) {
    return false;
  }

  if (receipt.layer_type == Sm87MacroFeedV3LayerType::kGdn) {
    return receipt.physical_counts.bf16_ab == 1U &&
           receipt.physical_counts.gdn == 9U &&
           receipt.physical_counts.attention_calls == 0U &&
           receipt.physical_counts.attention_preprocess == 0U &&
           !receipt.attention_preprocess_count_observed;
  }
  if (receipt.layer_type == Sm87MacroFeedV3LayerType::kFullAttention) {
    return receipt.physical_counts.bf16_ab == 0U &&
           receipt.physical_counts.gdn == 0U &&
           receipt.physical_counts.attention_calls == 1U &&
           receipt.physical_counts.attention_preprocess ==
               kSm87MacroFeedV3ReceiptAttentionPreprocessPerLayer &&
           receipt.attention_preprocess_count_observed;
  }
  return false;
}

void add_layer_to_aggregate(
    const Sm87MacroFeedV3LayerReceipt& layer,
    Sm87MacroFeedV3PhysicalAggregate* const aggregate) noexcept {
  ++aggregate->completed_layers;
  aggregate->completed_gdn_layers +=
      layer.layer_type == Sm87MacroFeedV3LayerType::kGdn ? 1U : 0U;
  aggregate->completed_attention_layers +=
      layer.layer_type == Sm87MacroFeedV3LayerType::kFullAttention ? 1U : 0U;
  aggregate->fill_completed_layers += layer.fill_completed ? 1U : 0U;
  aggregate->drain_completed_layers += layer.drain_completed ? 1U : 0U;
  aggregate->completion_observed_layers +=
      layer.completion_observed ? 1U : 0U;
  aggregate->attention_preprocess_count_observed_layers +=
      layer.attention_preprocess_count_observed ? 1U : 0U;

  aggregate->fp8_input += layer.physical_counts.fp8_input;
  aggregate->fp8_output += layer.physical_counts.fp8_output;
  aggregate->fp8 += layer.physical_counts.fp8_input +
                    layer.physical_counts.fp8_output;
  aggregate->nvfp4_gate_up += layer.physical_counts.nvfp4_gate_up;
  aggregate->nvfp4_down += layer.physical_counts.nvfp4_down;
  aggregate->bf16_ab += layer.physical_counts.bf16_ab;
  aggregate->gdn += layer.physical_counts.gdn;
  aggregate->attention_calls += layer.physical_counts.attention_calls;
  aggregate->attention_preprocess +=
      layer.physical_counts.attention_preprocess;

  aggregate->used_fallback |= layer.used_fallback;
  aggregate->used_mtp |= layer.used_mtp;
  aggregate->used_cublaslt |= layer.used_cublaslt;
  aggregate->used_request_jit |= layer.used_request_jit;
  aggregate->used_request_repack |= layer.used_request_repack;
  aggregate->used_request_autotune |= layer.used_request_autotune;
}

[[nodiscard]] bool aggregate_complete(
    const Sm87MacroFeedV3PhysicalAggregate& aggregate) noexcept {
  return aggregate.completed_layers == kSm87MacroFeedV3ReceiptLayerCount &&
         aggregate.completed_gdn_layers ==
             kSm87MacroFeedV3ReceiptGdnLayerCount &&
         aggregate.completed_attention_layers ==
             kSm87MacroFeedV3ReceiptAttentionLayerCount &&
         aggregate.fill_completed_layers ==
             kSm87MacroFeedV3ReceiptLayerCount &&
         aggregate.drain_completed_layers ==
             kSm87MacroFeedV3ReceiptLayerCount &&
         aggregate.completion_observed_layers ==
             kSm87MacroFeedV3ReceiptLayerCount &&
         aggregate.attention_preprocess_count_observed_layers ==
             kSm87MacroFeedV3ReceiptAttentionLayerCount &&
         aggregate.fp8_input == kSm87MacroFeedV3ReceiptLayerCount &&
         aggregate.fp8_output == kSm87MacroFeedV3ReceiptLayerCount &&
         aggregate.fp8 == kSm87MacroFeedV3ReceiptFp8Launches &&
         aggregate.nvfp4_gate_up ==
             kSm87MacroFeedV3ReceiptGateUpLaunches &&
         aggregate.nvfp4_down == kSm87MacroFeedV3ReceiptDownLaunches &&
         aggregate.bf16_ab == kSm87MacroFeedV3ReceiptBf16AbLaunches &&
         aggregate.gdn == kSm87MacroFeedV3ReceiptGdnLaunches &&
         aggregate.attention_calls ==
             kSm87MacroFeedV3ReceiptAttentionCalls &&
         aggregate.attention_preprocess ==
             kSm87MacroFeedV3ReceiptAttentionPreprocessPerRequest &&
         !aggregate.used_fallback && !aggregate.used_mtp &&
         !aggregate.used_cublaslt && !aggregate.used_request_jit &&
         !aggregate.used_request_repack && !aggregate.used_request_autotune;
}

[[nodiscard]] bool aggregate_equal(
    const Sm87MacroFeedV3PhysicalAggregate& left,
    const Sm87MacroFeedV3PhysicalAggregate& right) noexcept {
  return left.completed_layers == right.completed_layers &&
         left.completed_gdn_layers == right.completed_gdn_layers &&
         left.completed_attention_layers == right.completed_attention_layers &&
         left.fill_completed_layers == right.fill_completed_layers &&
         left.drain_completed_layers == right.drain_completed_layers &&
         left.completion_observed_layers ==
             right.completion_observed_layers &&
         left.attention_preprocess_count_observed_layers ==
             right.attention_preprocess_count_observed_layers &&
         left.fp8_input == right.fp8_input &&
         left.fp8_output == right.fp8_output && left.fp8 == right.fp8 &&
         left.nvfp4_gate_up == right.nvfp4_gate_up &&
         left.nvfp4_down == right.nvfp4_down &&
         left.bf16_ab == right.bf16_ab && left.gdn == right.gdn &&
         left.attention_calls == right.attention_calls &&
         left.attention_preprocess == right.attention_preprocess &&
         left.used_fallback == right.used_fallback &&
         left.used_mtp == right.used_mtp &&
         left.used_cublaslt == right.used_cublaslt &&
         left.used_request_jit == right.used_request_jit &&
         left.used_request_repack == right.used_request_repack &&
         left.used_request_autotune == right.used_request_autotune;
}

void hash_aggregate(std::uint64_t* const hash,
                    const Sm87MacroFeedV3PhysicalAggregate& aggregate) noexcept {
  hash_u64(hash, aggregate.completed_layers);
  hash_u64(hash, aggregate.completed_gdn_layers);
  hash_u64(hash, aggregate.completed_attention_layers);
  hash_u64(hash, aggregate.fill_completed_layers);
  hash_u64(hash, aggregate.drain_completed_layers);
  hash_u64(hash, aggregate.completion_observed_layers);
  hash_u64(hash, aggregate.attention_preprocess_count_observed_layers);
  hash_u64(hash, aggregate.fp8_input);
  hash_u64(hash, aggregate.fp8_output);
  hash_u64(hash, aggregate.fp8);
  hash_u64(hash, aggregate.nvfp4_gate_up);
  hash_u64(hash, aggregate.nvfp4_down);
  hash_u64(hash, aggregate.bf16_ab);
  hash_u64(hash, aggregate.gdn);
  hash_u64(hash, aggregate.attention_calls);
  hash_u64(hash, aggregate.attention_preprocess);
  hash_bool(hash, aggregate.used_fallback);
  hash_bool(hash, aggregate.used_mtp);
  hash_bool(hash, aggregate.used_cublaslt);
  hash_bool(hash, aggregate.used_request_jit);
  hash_bool(hash, aggregate.used_request_repack);
  hash_bool(hash, aggregate.used_request_autotune);
}

[[nodiscard]] std::uint64_t transaction_identity(
    const Sm87MacroFeedV3TransactionReceipt& transaction) noexcept {
  std::uint64_t hash = kFnvOffset;
  constexpr std::array<std::uint8_t, 18U> kDomain{{
      'q', '3', 'x', '.', 'm', 'a', 'c', 'r', 'o', 'f', 'e', 'e', 'd', '.',
      't', 'x', '.', '1'}};
  for (const std::uint8_t byte : kDomain) {
    hash_byte(&hash, byte);
  }
  for (const std::uint8_t byte : transaction.magic) {
    hash_byte(&hash, byte);
  }
  hash_u64(&hash, transaction.abi_major);
  hash_u64(&hash, transaction.abi_minor);
  hash_u64(&hash, transaction.request_identity);
  hash_physical_identity(&hash, transaction.physical_identity);
  hash_u64(&hash, transaction.layer_receipt_count);
  for (const auto& layer : transaction.layers) {
    hash_u64(&hash, layer.receipt_identity);
  }
  hash_aggregate(&hash, transaction.aggregate);
  hash_bool(&hash, transaction.request_completion_observed);
  hash_bool(&hash, transaction.complete);
  return hash;
}

[[nodiscard]] bool magic_equal(
    const std::array<std::uint8_t, 8U>& left,
    const std::array<std::uint8_t, 8U>& right) noexcept {
  for (std::size_t index = 0U; index < left.size(); ++index) {
    if (left[index] != right[index]) {
      return false;
    }
  }
  return true;
}

}  // namespace

std::uint64_t macrofeed_v3_receipt_detail::compute_layer_receipt_identity(
    const Sm87MacroFeedV3LayerReceipt& receipt) noexcept {
  std::uint64_t hash = kFnvOffset;
  constexpr std::array<std::uint8_t, 21U> kDomain{{
      'q', '3', 'x', '.', 'm', 'a', 'c', 'r', 'o', 'f', 'e', 'e', 'd', '.',
      'l', 'a', 'y', 'e', 'r', '.', '1'}};
  for (const std::uint8_t byte : kDomain) {
    hash_byte(&hash, byte);
  }
  hash_u64(&hash, receipt.request_identity);
  hash_u64(&hash, receipt.layer_index);
  hash_u64(&hash, static_cast<std::uint8_t>(receipt.layer_type));
  hash_u64(&hash, receipt.artifacts.fp8_input);
  hash_u64(&hash, receipt.artifacts.fp8_output);
  hash_u64(&hash, receipt.artifacts.nvfp4_gate_up);
  hash_u64(&hash, receipt.artifacts.nvfp4_down);
  hash_physical_identity(&hash, receipt.physical_identity);
  hash_u64(&hash, receipt.physical_counts.fp8_input);
  hash_u64(&hash, receipt.physical_counts.fp8_output);
  hash_u64(&hash, receipt.physical_counts.nvfp4_gate_up);
  hash_u64(&hash, receipt.physical_counts.nvfp4_down);
  hash_u64(&hash, receipt.physical_counts.bf16_ab);
  hash_u64(&hash, receipt.physical_counts.gdn);
  hash_u64(&hash, receipt.physical_counts.attention_calls);
  hash_u64(&hash, receipt.physical_counts.attention_preprocess);
  hash_bool(&hash, receipt.fill_completed);
  hash_bool(&hash, receipt.drain_completed);
  hash_bool(&hash, receipt.completion_observed);
  hash_bool(&hash, receipt.attention_preprocess_count_observed);
  hash_bool(&hash, receipt.used_fallback);
  hash_bool(&hash, receipt.used_mtp);
  hash_bool(&hash, receipt.used_cublaslt);
  hash_bool(&hash, receipt.used_request_jit);
  hash_bool(&hash, receipt.used_request_repack);
  hash_bool(&hash, receipt.used_request_autotune);
  return hash;
}

Sm87MacroFeedV3LayerReceipt macrofeed_v3_receipt_detail::seal_layer_receipt(
    const Sm87MacroFeedV3LayerReceipt& observed) noexcept {
  Sm87MacroFeedV3LayerReceipt result = observed;
  result.receipt_identity = 0U;
  if (!layer_fields_valid(result)) {
    return {};
  }
  result.receipt_identity = compute_layer_receipt_identity(result);
  return result.receipt_identity != 0U ? result
                                       : Sm87MacroFeedV3LayerReceipt{};
}

bool sm87_macrofeed_v3_layer_receipt_valid(
    const Sm87MacroFeedV3LayerReceipt& receipt) noexcept {
  return receipt.receipt_identity != 0U && layer_fields_valid(receipt) &&
         receipt.receipt_identity ==
             macrofeed_v3_receipt_detail::compute_layer_receipt_identity(
                 receipt);
}

bool macrofeed_v3_receipt_detail::build_transaction_receipt(
    const std::uint64_t request_identity,
    const Sm87MacroFeedV3LayerReceipt* const layer_receipts,
    const std::size_t layer_receipt_count,
    const bool request_completion_observed,
    Sm87MacroFeedV3TransactionReceipt* const transaction) noexcept {
  if (transaction == nullptr) {
    return false;
  }
  *transaction = {};
  if (request_identity == 0U || layer_receipts == nullptr ||
      layer_receipt_count != kSm87MacroFeedV3ReceiptLayerCount) {
    return false;
  }
  Sm87MacroFeedV3TransactionReceipt candidate{};

  std::array<bool, kSm87MacroFeedV3ReceiptLayerCount> seen_layers{};
  std::array<std::uint64_t,
             kSm87MacroFeedV3ReceiptLayerCount *
                 kSm87MacroFeedV3ReceiptArtifactsPerLayer>
      seen_artifacts{};
  std::size_t seen_artifact_count = 0U;
  Sm87MacroFeedV3PhysicalIdentity common_identity{};
  bool common_identity_set = false;

  for (std::size_t ordinal = 0U; ordinal < layer_receipt_count; ++ordinal) {
    const Sm87MacroFeedV3LayerReceipt& layer = layer_receipts[ordinal];
    if (!sm87_macrofeed_v3_layer_receipt_valid(layer) ||
        layer.request_identity != request_identity ||
        layer.layer_index >= seen_layers.size() ||
        seen_layers[layer.layer_index]) {
      return false;
    }
    if (!common_identity_set) {
      common_identity = layer.physical_identity;
      common_identity_set = true;
    } else if (!physical_identity_equal(common_identity,
                                         layer.physical_identity)) {
      return false;
    }

    const std::array<std::uint64_t,
                     kSm87MacroFeedV3ReceiptArtifactsPerLayer>
        artifacts{{layer.artifacts.fp8_input, layer.artifacts.fp8_output,
                   layer.artifacts.nvfp4_gate_up,
                   layer.artifacts.nvfp4_down}};
    for (const std::uint64_t artifact : artifacts) {
      for (std::size_t index = 0U; index < seen_artifact_count; ++index) {
        if (seen_artifacts[index] == artifact) {
          return false;
        }
      }
      seen_artifacts[seen_artifact_count++] = artifact;
    }

    seen_layers[layer.layer_index] = true;
    candidate.layers[layer.layer_index] = layer;
    add_layer_to_aggregate(layer, &candidate.aggregate);
  }

  for (const bool seen : seen_layers) {
    if (!seen) {
      return false;
    }
  }
  if (!common_identity_set || !aggregate_complete(candidate.aggregate)) {
    return false;
  }

  candidate.magic = kSm87MacroFeedV3ReceiptMagic;
  candidate.abi_major = kSm87MacroFeedV3ReceiptAbiMajor;
  candidate.abi_minor = kSm87MacroFeedV3ReceiptAbiMinor;
  candidate.request_identity = request_identity;
  candidate.physical_identity = common_identity;
  candidate.layer_receipt_count = layer_receipt_count;
  candidate.request_completion_observed = request_completion_observed;
  candidate.complete = request_completion_observed;
  candidate.receipt_identity = transaction_identity(candidate);
  if (candidate.receipt_identity == 0U) {
    return false;
  }
  *transaction = candidate;
  return true;
}

bool sm87_macrofeed_v3_transaction_receipt_valid(
    const Sm87MacroFeedV3TransactionReceipt& transaction) noexcept {
  if (transaction.receipt_identity == 0U ||
      !magic_equal(transaction.magic, kSm87MacroFeedV3ReceiptMagic) ||
      transaction.abi_major != kSm87MacroFeedV3ReceiptAbiMajor ||
      transaction.abi_minor != kSm87MacroFeedV3ReceiptAbiMinor ||
      transaction.request_identity == 0U ||
      !physical_identity_valid(transaction.physical_identity) ||
      transaction.layer_receipt_count !=
          kSm87MacroFeedV3ReceiptLayerCount ||
      transaction.complete != transaction.request_completion_observed ||
      !aggregate_complete(transaction.aggregate)) {
    return false;
  }

  Sm87MacroFeedV3TransactionReceipt expected{};
  if (!macrofeed_v3_receipt_detail::build_transaction_receipt(
          transaction.request_identity, transaction.layers.data(),
          transaction.layer_receipt_count,
          transaction.request_completion_observed, &expected)) {
    return false;
  }
  if (transaction.receipt_identity != expected.receipt_identity ||
      !physical_identity_equal(transaction.physical_identity,
                               expected.physical_identity) ||
      !aggregate_equal(transaction.aggregate, expected.aggregate)) {
    return false;
  }
  for (std::size_t layer = 0U; layer < transaction.layers.size(); ++layer) {
    if (transaction.layers[layer].layer_index != layer ||
        transaction.layers[layer].receipt_identity !=
            expected.layers[layer].receipt_identity) {
      return false;
    }
  }
  return true;
}

bool sm87_macrofeed_v3_transaction_complete(
    const Sm87MacroFeedV3TransactionReceipt& transaction) noexcept {
  return sm87_macrofeed_v3_transaction_receipt_valid(transaction) &&
         transaction.request_completion_observed && transaction.complete;
}

}  // namespace q3x::runtime
