#include "../src/runtime/sm87_macrofeed_v3_receipt_internal.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>

namespace {

namespace runtime = q3x::runtime;
namespace issuer = q3x::runtime::macrofeed_v3_receipt_detail;

constexpr std::uint64_t kRequestIdentity = 0x5133'5251'5545'5354ULL;

struct Test final {
  bool ok = true;

  void expect(const bool condition, const char* const message) {
    if (!condition) {
      ok = false;
      std::cerr << "FAIL: " << message << '\n';
    }
  }
};

[[nodiscard]] runtime::Sm87MacroFeedV3LayerReceipt make_layer(
    const std::size_t layer) {
  runtime::Sm87MacroFeedV3LayerReceipt observed;
  observed.request_identity = kRequestIdentity;
  observed.layer_index = layer;
  observed.layer_type = runtime::sm87_macrofeed_v3_expected_layer_type(layer);
  const std::uint64_t first_artifact = 1'000U + layer * 4U;
  observed.artifacts.fp8_input = first_artifact;
  observed.artifacts.fp8_output = first_artifact + 1U;
  observed.artifacts.nvfp4_gate_up = first_artifact + 2U;
  observed.artifacts.nvfp4_down = first_artifact + 3U;
  observed.physical_identity.package_identity = 101U;
  observed.physical_identity.owner_identity = 102U;
  observed.physical_identity.allocation_identity = 103U;
  observed.physical_identity.catalog_identity = 104U;
  observed.physical_identity.device_identity = 105U;
  observed.physical_identity.device_ordinal = 0;
  observed.physical_counts.fp8_input = 1U;
  observed.physical_counts.fp8_output = 1U;
  observed.physical_counts.nvfp4_gate_up = 1U;
  observed.physical_counts.nvfp4_down = 1U;
  if (observed.layer_type == runtime::Sm87MacroFeedV3LayerType::kGdn) {
    observed.physical_counts.bf16_ab = 1U;
    observed.physical_counts.gdn = 9U;
  } else {
    observed.physical_counts.attention_calls = 1U;
    observed.physical_counts.attention_preprocess =
        runtime::kSm87MacroFeedV3ReceiptAttentionPreprocessPerLayer;
    observed.attention_preprocess_count_observed = true;
  }
  observed.fill_completed = true;
  observed.drain_completed = true;
  observed.completion_observed = true;
  // T0 issuer simulation only.  This does not claim that CUDA ran.
  return issuer::seal_layer_receipt(observed);
}

[[nodiscard]] std::array<runtime::Sm87MacroFeedV3LayerReceipt,
                         runtime::kSm87MacroFeedV3ReceiptLayerCount>
make_layers() {
  std::array<runtime::Sm87MacroFeedV3LayerReceipt,
             runtime::kSm87MacroFeedV3ReceiptLayerCount>
      layers{};
  for (std::size_t layer = 0U; layer < layers.size(); ++layer) {
    layers[layer] = make_layer(layer);
  }
  return layers;
}

[[nodiscard]] std::uint64_t expected_preprocess(
    const std::array<runtime::Sm87MacroFeedV3LayerReceipt,
                     runtime::kSm87MacroFeedV3ReceiptLayerCount>& layers) {
  std::uint64_t total = 0U;
  for (const auto& layer : layers) {
    total += layer.physical_counts.attention_preprocess;
  }
  return total;
}

}  // namespace

int main() {
  Test test;
  auto layers = make_layers();

  std::size_t gdn_layers = 0U;
  std::size_t attention_layers = 0U;
  for (std::size_t layer = 0U; layer < layers.size(); ++layer) {
    test.expect(runtime::sm87_macrofeed_v3_layer_receipt_valid(layers[layer]),
                "sealed layer receipt validates");
    test.expect(layers[layer].layer_index == layer,
                "layer ordinal remains exact");
    if (layers[layer].layer_type ==
        runtime::Sm87MacroFeedV3LayerType::kGdn) {
      ++gdn_layers;
    } else if (layers[layer].layer_type ==
               runtime::Sm87MacroFeedV3LayerType::kFullAttention) {
      ++attention_layers;
    }
  }
  test.expect(gdn_layers == runtime::kSm87MacroFeedV3ReceiptGdnLayerCount &&
                  attention_layers ==
                      runtime::kSm87MacroFeedV3ReceiptAttentionLayerCount,
              "fixed 48 GDN / 16 Attention topology");

  std::array<runtime::Sm87MacroFeedV3LayerReceipt,
             runtime::kSm87MacroFeedV3ReceiptLayerCount>
      reversed{};
  for (std::size_t index = 0U; index < layers.size(); ++index) {
    reversed[index] = layers[layers.size() - 1U - index];
  }

  runtime::Sm87MacroFeedV3TransactionReceipt incomplete{};
  test.expect(issuer::build_transaction_receipt(
                  kRequestIdentity, reversed.data(), reversed.size(), false,
                  &incomplete),
              "unordered exact layer set builds canonical transaction");
  test.expect(runtime::sm87_macrofeed_v3_transaction_receipt_valid(incomplete),
              "pre-completion transaction remains structurally coherent");
  test.expect(!incomplete.complete &&
                  !runtime::sm87_macrofeed_v3_transaction_complete(incomplete),
              "request completion observation gates complete");
  for (std::size_t layer = 0U; layer < incomplete.layers.size(); ++layer) {
    test.expect(incomplete.layers[layer].layer_index == layer,
                "transaction stores layers in canonical order");
  }

  runtime::Sm87MacroFeedV3TransactionReceipt complete{};
  test.expect(issuer::build_transaction_receipt(
                  kRequestIdentity, layers.data(), layers.size(), true,
                  &complete),
              "complete 64-layer physical transaction builds");
  test.expect(runtime::sm87_macrofeed_v3_transaction_receipt_valid(complete) &&
                  runtime::sm87_macrofeed_v3_transaction_complete(complete),
              "completion-observed transaction validates complete");
  test.expect(
      complete.aggregate.completed_layers == 64U &&
          complete.aggregate.completed_gdn_layers == 48U &&
          complete.aggregate.completed_attention_layers == 16U &&
          complete.aggregate.fp8_input == 64U &&
          complete.aggregate.fp8_output == 64U &&
          complete.aggregate.fp8 == 128U &&
          complete.aggregate.nvfp4_gate_up == 64U &&
          complete.aggregate.nvfp4_down == 64U &&
          complete.aggregate.bf16_ab == 48U &&
          complete.aggregate.gdn == 432U &&
          complete.aggregate.attention_calls == 16U &&
          complete.aggregate.attention_preprocess_count_observed_layers ==
              16U,
      "physical request totals are exact");
  test.expect(
      complete.aggregate.attention_preprocess ==
              runtime::kSm87MacroFeedV3ReceiptAttentionPreprocessPerRequest &&
          complete.aggregate.attention_preprocess ==
              expected_preprocess(layers),
      "fixed five-panel Attention preprocess aggregates to 80");
  test.expect(!complete.aggregate.used_fallback &&
                  !complete.aggregate.used_mtp &&
                  !complete.aggregate.used_cublaslt &&
                  !complete.aggregate.used_request_jit &&
                  !complete.aggregate.used_request_repack &&
                  !complete.aggregate.used_request_autotune,
              "all forbidden routes remain absent");

  auto tampered_layer = complete;
  ++tampered_layer.layers[0U].artifacts.fp8_input;
  test.expect(
      !runtime::sm87_macrofeed_v3_transaction_receipt_valid(tampered_layer),
      "post-seal layer artifact tamper is rejected");

  auto tampered_aggregate = complete;
  --tampered_aggregate.aggregate.gdn;
  test.expect(
      !runtime::sm87_macrofeed_v3_transaction_receipt_valid(tampered_aggregate),
      "post-seal aggregate-count tamper is rejected");

  auto tampered_magic = complete;
  tampered_magic.magic[0U] ^= 1U;
  test.expect(
      !runtime::sm87_macrofeed_v3_transaction_receipt_valid(tampered_magic),
      "transaction identity/magic tamper is rejected");

  auto premature_complete = incomplete;
  premature_complete.complete = true;
  test.expect(!runtime::sm87_macrofeed_v3_transaction_receipt_valid(
                  premature_complete),
              "complete cannot precede request completion observation");

  runtime::Sm87MacroFeedV3TransactionReceipt rejected{};
  test.expect(!issuer::build_transaction_receipt(
                  kRequestIdentity, layers.data(), layers.size() - 1U, true,
                  &rejected) &&
                  rejected.receipt_identity == 0U,
              "missing layer is rejected with a zero transaction");

  auto duplicate_layer = layers;
  duplicate_layer.back() = duplicate_layer[duplicate_layer.size() - 2U];
  test.expect(!issuer::build_transaction_receipt(
                  kRequestIdentity, duplicate_layer.data(),
                  duplicate_layer.size(), true, &rejected),
              "duplicate layer and corresponding omission are rejected");

  auto wrong_count = layers;
  wrong_count[0U].physical_counts.gdn = 8U;
  wrong_count[0U].receipt_identity =
      issuer::compute_layer_receipt_identity(wrong_count[0U]);
  test.expect(!runtime::sm87_macrofeed_v3_layer_receipt_valid(wrong_count[0U]) &&
                  !issuer::build_transaction_receipt(
                      kRequestIdentity, wrong_count.data(), wrong_count.size(),
                      true, &rejected),
              "wrong GDN physical count is rejected even when rehashed");

  constexpr std::array<std::uint32_t, 4U> kInvalidPreprocessCounts{{
      0U, std::numeric_limits<std::uint32_t>::max(), 4U, 6U}};
  for (const std::uint32_t invalid_count : kInvalidPreprocessCounts) {
    auto wrong_preprocess = layers;
    wrong_preprocess[3U].physical_counts.attention_preprocess = invalid_count;
    wrong_preprocess[3U].receipt_identity =
        issuer::compute_layer_receipt_identity(wrong_preprocess[3U]);
    runtime::Sm87MacroFeedV3TransactionReceipt rejected_preprocess = complete;
    test.expect(
        !runtime::sm87_macrofeed_v3_layer_receipt_valid(
            wrong_preprocess[3U]) &&
            issuer::seal_layer_receipt(wrong_preprocess[3U])
                    .receipt_identity == 0U &&
            !issuer::build_transaction_receipt(
                kRequestIdentity, wrong_preprocess.data(),
                wrong_preprocess.size(), true, &rejected_preprocess) &&
            rejected_preprocess.receipt_identity == 0U,
        "Attention preprocess count other than fixed five is rejected");
  }

  auto forbidden = layers;
  forbidden[3U].used_cublaslt = true;
  forbidden[3U].receipt_identity =
      issuer::compute_layer_receipt_identity(forbidden[3U]);
  test.expect(!runtime::sm87_macrofeed_v3_layer_receipt_valid(forbidden[3U]) &&
                  !issuer::build_transaction_receipt(
                      kRequestIdentity, forbidden.data(), forbidden.size(),
                      true, &rejected),
              "forbidden route is rejected even when rehashed");

  auto duplicate_artifact = layers;
  duplicate_artifact[1U].artifacts.fp8_input =
      duplicate_artifact[0U].artifacts.fp8_input;
  duplicate_artifact[1U] =
      issuer::seal_layer_receipt(duplicate_artifact[1U]);
  test.expect(runtime::sm87_macrofeed_v3_layer_receipt_valid(
                  duplicate_artifact[1U]) &&
                  !issuer::build_transaction_receipt(
                      kRequestIdentity, duplicate_artifact.data(),
                      duplicate_artifact.size(), true, &rejected),
              "cross-layer duplicate artifact identity is rejected");

  auto wrong_owner = layers;
  ++wrong_owner.back().physical_identity.device_identity;
  wrong_owner.back() =
      issuer::seal_layer_receipt(wrong_owner.back());
  test.expect(runtime::sm87_macrofeed_v3_layer_receipt_valid(
                  wrong_owner.back()) &&
                  !issuer::build_transaction_receipt(
                      kRequestIdentity, wrong_owner.data(), wrong_owner.size(),
                      true, &rejected),
              "mixed package/owner/allocation/catalog/device tuple rejected");

  auto not_drained = layers[0U];
  not_drained.drain_completed = false;
  test.expect(issuer::seal_layer_receipt(not_drained).receipt_identity == 0U,
              "layer cannot seal before fill/drain completion");

  if (test.ok) {
    std::cout << "sm87_macrofeed_v3_receipt_test: PASS\n";
  }
  return test.ok ? 0 : 1;
}
