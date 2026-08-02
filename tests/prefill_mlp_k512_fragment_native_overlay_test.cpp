#include "q3x/runtime/prefill_mlp_k512_fragment_native_overlay.h"

#include "q3x/kernels/sm87_a4w4_down_k512_fragment_native.h"
#include "q3x/kernels/sm87_a4w4_gateup_k512_fragment_native.h"
#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"
#include "q3x/runtime/prefill_quantized_contract.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace kernels = q3x::kernels;
namespace runtime = q3x::runtime;

class Test final {
 public:
  void expect(const bool condition, const std::string_view message) {
    if (!condition) {
      std::cerr << "FAILED: " << message << '\n';
      ++failures_;
    }
  }
  [[nodiscard]] int result() const noexcept { return failures_ == 0 ? 0 : 1; }

 private:
  int failures_ = 0;
};

[[nodiscard]] std::uint8_t pattern(const std::size_t index,
                                   const std::uint32_t salt) noexcept {
  std::uint32_t value = static_cast<std::uint32_t>(index) ^ salt;
  value ^= value >> 16U;
  value *= 0x7feb352dU;
  value ^= value >> 15U;
  value *= 0x846ca68bU;
  value ^= value >> 16U;
  return static_cast<std::uint8_t>(value);
}

void fill_pattern(std::vector<std::uint8_t>& values,
                  const std::uint32_t salt) {
  for (std::size_t index = 0U; index < values.size(); ++index) {
    values[index] = pattern(index, salt);
  }
}

[[nodiscard]] runtime::PrefillMLPK512OverlayReceipt make_v1_receipt() {
  runtime::PrefillMLPK512OverlayReceipt receipt;
  receipt.production_residency_eligible = true;
  receipt.physical_layout = std::string(runtime::kPrefillMLPK512OverlayLayout);
  receipt.source_checkpoint_id = "synthetic-qwen36-27b";
  receipt.source_config_sha256 = std::string(64U, 'a');
  receipt.source_index_sha256 = std::string(64U, 'b');
  receipt.manifest_sha256 = std::string(64U, '4');
  receipt.policy_sha256 = std::string(64U, '5');
  receipt.policy_bytes = 1'234U;
  receipt.required_base.physical_layout =
      std::string(runtime::kPrefillA4K128PhysicalLayout);
  receipt.required_base.manifest_sha256 = std::string(64U, '1');
  receipt.required_base.policy_sha256 = std::string(64U, '2');
  receipt.required_base.payload_sha256 = std::string(64U, '3');
  receipt.payload_sha256 = std::string(64U, '6');
  receipt.payload_bytes = runtime::kPrefillMLPK512OverlayPayloadBytes;
  receipt.projection_count = runtime::kPrefillMLPK512OverlayProjectionCount;
  return receipt;
}

[[nodiscard]] runtime::PrefillMLPK512OverlayReceipt make_k256_v1_receipt() {
  runtime::PrefillMLPK512OverlayReceipt receipt = make_v1_receipt();
  receipt.required_base.physical_layout =
      std::string(runtime::kPrefillA4K256PhysicalLayout);
  receipt.required_base.manifest_sha256 = std::string(64U, '9');
  receipt.required_base.policy_sha256 = std::string(64U, 'a');
  receipt.required_base.payload_sha256 = std::string(64U, 'b');
  receipt.manifest_sha256 = std::string(64U, 'c');
  receipt.policy_sha256 = std::string(64U, 'd');
  receipt.payload_sha256 = std::string(64U, 'e');
  return receipt;
}

[[nodiscard]] runtime::PrefillMLPK512FragmentNativeReceipt make_v2_receipt(
    const runtime::PrefillMLPK512FragmentNativeManifest& manifest) {
  runtime::PrefillMLPK512FragmentNativeReceipt receipt;
  receipt.production_residency_eligible = true;
  receipt.physical_layout = manifest.physical_layout;
  receipt.source_checkpoint_id = manifest.source_checkpoint_id;
  receipt.source_config_sha256 = manifest.source_config_sha256;
  receipt.source_index_sha256 = manifest.source_index_sha256;
  receipt.required_base = manifest.required_base;
  receipt.source_v1 = manifest.source_v1;
  receipt.manifest_sha256 = manifest.manifest_sha256;
  receipt.payload_sha256 = std::string(64U, '8');
  receipt.payload_bytes = manifest.payload_bytes;
  receipt.layer_count = manifest.layer_count;
  return receipt;
}

[[nodiscard]] runtime::PrefillMLPK512PairedGateUpCanonicalDownReceipt
make_hybrid_receipt(
    const runtime::PrefillMLPK512PairedGateUpCanonicalDownManifest& manifest) {
  runtime::PrefillMLPK512PairedGateUpCanonicalDownReceipt receipt;
  receipt.production_residency_eligible = true;
  receipt.physical_layout = manifest.physical_layout;
  receipt.source_checkpoint_id = manifest.source_checkpoint_id;
  receipt.source_config_sha256 = manifest.source_config_sha256;
  receipt.source_index_sha256 = manifest.source_index_sha256;
  receipt.required_base = manifest.required_base;
  receipt.source_v1 = manifest.source_v1;
  receipt.manifest_sha256 = manifest.manifest_sha256;
  receipt.payload_sha256 = std::string(64U, 'f');
  receipt.payload_bytes = manifest.payload_bytes;
  receipt.layer_count = manifest.layer_count;
  return receipt;
}

[[nodiscard]] std::string read_file(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

[[nodiscard]] bool all_once(const std::vector<std::uint8_t>& visits) {
  return std::all_of(visits.begin(), visits.end(),
                     [](const std::uint8_t value) { return value == 1U; });
}

void test_inventory(
    Test& test,
    const runtime::PrefillMLPK512FragmentNativeManifest& manifest) {
  test.expect(
      manifest.payload_bytes == 8'623'226'880ULL &&
          manifest.layer_count == 64U &&
          runtime::kPrefillMLPK512FragmentNativeLayerBytes ==
              134'737'920ULL &&
          runtime::kPrefillMLPK512FragmentNativeGateUpRecordBytes ==
              89'825'280ULL,
      "v2 is one equal-byte 64-layer composite payload");
  std::uint64_t cursor = 0U;
  for (std::size_t layer = 0U; layer < 64U; ++layer) {
    const auto view =
        runtime::prefill_mlp_k512_fragment_native_layer_view(layer);
    test.expect(view.valid && view.layer_index == layer &&
                    view.layer_offset == cursor &&
                    view.gateup_code_offset == cursor &&
                    view.gateup_scale_offset ==
                        cursor + 89'128'960ULL &&
                    view.down_code_offset == cursor + 89'825'280ULL &&
                    view.down_scale_offset == cursor + 134'389'760ULL,
                "layer view has exact GateUp/Down transaction offsets");
    cursor += runtime::kPrefillMLPK512FragmentNativeLayerBytes;
  }
  test.expect(cursor == runtime::kPrefillMLPK512FragmentNativePayloadBytes &&
                  !runtime::prefill_mlp_k512_fragment_native_layer_view(64U)
                       .valid &&
                  runtime::validate_prefill_mlp_k512_fragment_native_manifest(
                      manifest)
                      .ok(),
              "64 views span the payload and out-of-range fails closed");
}

void test_gateup_bijection(Test& test) {
  constexpr std::size_t kN = 64U;
  constexpr std::size_t kK = 512U;
  constexpr std::size_t kCodeBytes = kN * kK / 2U;
  constexpr std::size_t kScaleBytes = kN * (kK / 512U) * 2U;
  std::vector<std::uint8_t> gate(kCodeBytes);
  std::vector<std::uint8_t> up(kCodeBytes);
  std::vector<std::uint8_t> gate_scales(kScaleBytes);
  std::vector<std::uint8_t> up_scales(kScaleBytes);
  fill_pattern(gate, 0x13579bdfU);
  fill_pattern(up, 0x2468ace0U);
  fill_pattern(gate_scales, 0x11223344U);
  fill_pattern(up_scales, 0x55667788U);
  std::vector<std::uint8_t> paired(2U * kCodeBytes, 0U);
  std::vector<std::uint8_t> paired_scales(2U * kScaleBytes, 0U);
  const auto forward =
      runtime::permute_prefill_mlp_k512_gateup_fragment_native(
          gate.data(), gate.size(), gate_scales.data(), gate_scales.size(),
          up.data(), up.size(), up_scales.data(), up_scales.size(), kN, kK,
          paired.data(), paired.size(), paired_scales.data(),
          paired_scales.size());
  std::vector<std::uint8_t> recovered_gate(kCodeBytes, 0U);
  std::vector<std::uint8_t> recovered_up(kCodeBytes, 0U);
  std::vector<std::uint8_t> recovered_gate_scales(kScaleBytes, 0U);
  std::vector<std::uint8_t> recovered_up_scales(kScaleBytes, 0U);
  const auto reverse =
      runtime::unpermute_prefill_mlp_k512_gateup_fragment_native(
          paired.data(), paired.size(), paired_scales.data(),
          paired_scales.size(), kN, kK, recovered_gate.data(),
          recovered_gate.size(), recovered_gate_scales.data(),
          recovered_gate_scales.size(), recovered_up.data(),
          recovered_up.size(), recovered_up_scales.data(),
          recovered_up_scales.size());
  test.expect(forward.ok() && reverse.ok() && recovered_gate == gate &&
                  recovered_up == up &&
                  recovered_gate_scales == gate_scales &&
                  recovered_up_scales == up_scales,
              "Gate+Up minimum shape permutes and reverses bit-exactly");

  std::vector<std::uint8_t> gate_visits(kCodeBytes, 0U);
  std::vector<std::uint8_t> up_visits(kCodeBytes, 0U);
  std::vector<std::uint8_t> paired_visits(2U * kCodeBytes, 0U);
  for (std::size_t phase = 0U; phase < 8U; ++phase) {
    for (std::size_t fragment = 0U; fragment < 8U; ++fragment) {
      const std::size_t fragment_n = fragment * 8U;
      for (std::size_t lane = 0U; lane < 32U; ++lane) {
        const std::size_t row = fragment_n + lane / 4U;
        const std::size_t first = kernels::sm87_a4w4_consumer_packed_offset(
            row, phase, 4U * (lane % 4U), 8U);
        const std::size_t second = kernels::sm87_a4w4_consumer_packed_offset(
            row, phase, 16U + 4U * (lane % 4U), 8U);
        const std::size_t destination =
            kernels::sm87_a4w4_gateup_k512_fragment_native_code_slot_offset(
                fragment_n, 0U, phase, lane, 1U);
        for (std::size_t byte = 0U; byte < 4U; ++byte) {
          ++gate_visits[first + byte];
          ++gate_visits[second + byte];
          ++up_visits[first + byte];
          ++up_visits[second + byte];
          ++paired_visits[destination + byte];
          ++paired_visits[destination + 4U + byte];
          ++paired_visits[destination + 8U + byte];
          ++paired_visits[destination + 12U + byte];
        }
      }
    }
  }
  test.expect(all_once(gate_visits) && all_once(up_visits) &&
                  all_once(paired_visits),
              "Gate+Up exhaustive byte map is a true bijection");
}

void test_down_bijection(Test& test) {
  constexpr std::size_t kN = 128U;
  constexpr std::size_t kK = 512U;
  constexpr std::size_t kCodeBytes = kN * kK / 2U;
  constexpr std::size_t kScaleBytes = kN * (kK / 512U) * 2U;
  std::vector<std::uint8_t> canonical(kCodeBytes);
  std::vector<std::uint8_t> scales(kScaleBytes);
  fill_pattern(canonical, 0xa5a5f00dU);
  fill_pattern(scales, 0x0badcafeU);
  std::vector<std::uint8_t> fragment(kCodeBytes, 0U);
  std::vector<std::uint8_t> fragment_scales(kScaleBytes, 0U);
  const auto forward = runtime::permute_prefill_mlp_k512_down_fragment_native(
      canonical.data(), canonical.size(), scales.data(), scales.size(), kN,
      kK, fragment.data(), fragment.size(), fragment_scales.data(),
      fragment_scales.size());
  std::vector<std::uint8_t> recovered(kCodeBytes, 0U);
  std::vector<std::uint8_t> recovered_scales(kScaleBytes, 0U);
  const auto reverse =
      runtime::unpermute_prefill_mlp_k512_down_fragment_native(
          fragment.data(), fragment.size(), fragment_scales.data(),
          fragment_scales.size(), kN, kK, recovered.data(), recovered.size(),
          recovered_scales.data(), recovered_scales.size());
  test.expect(forward.ok() && reverse.ok() && recovered == canonical &&
                  recovered_scales == scales && fragment_scales == scales,
              "Down minimum shape permutes/reverses codes and preserves scales");

  std::vector<std::uint8_t> source_visits(kCodeBytes, 0U);
  std::vector<std::uint8_t> destination_visits(kCodeBytes, 0U);
  for (std::size_t phase = 0U; phase < 8U; ++phase) {
    for (std::size_t warp = 0U; warp < 8U; ++warp) {
      for (std::size_t lane = 0U; lane < 32U; ++lane) {
        const std::size_t destination =
            kernels::sm87_a4w4_down_k512_fragment_b_vector_offset(
                0U, 0U, phase, warp, lane, 1U);
        for (std::size_t word = 0U; word < 4U; ++word) {
          const auto coordinate =
              kernels::sm87_a4w4_down_k512_fragment_b_word_coordinate(
                  warp, lane, word);
          const std::size_t source =
              kernels::sm87_a4w4_consumer_packed_offset(
                  coordinate.n, phase, coordinate.byte_in_k64, 8U);
          for (std::size_t byte = 0U; byte < 4U; ++byte) {
            ++source_visits[source + byte];
            ++destination_visits[destination + word * 4U + byte];
          }
        }
      }
    }
  }
  test.expect(all_once(source_visits) && all_once(destination_visits),
              "Down exhaustive byte map is a true bijection");
}

void test_receipt(Test& test,
                  const runtime::PrefillMLPK512FragmentNativeManifest& m) {
  const fs::path path =
      fs::temp_directory_path() /
      ("q3x-mlp-k512-fragment-receipt-" + std::to_string(::getpid()) +
       ".json");
  std::error_code ignored;
  fs::remove(path, ignored);
  const auto receipt = make_v2_receipt(m);
  const auto first =
      runtime::write_prefill_mlp_k512_fragment_native_receipt_no_replace(
          receipt, path);
  struct stat status {};
  test.expect(first.ok() && ::lstat(path.c_str(), &status) == 0 &&
                  (status.st_mode & (S_IWUSR | S_IWGRP | S_IWOTH)) == 0,
              "receipt is atomically published read-only");
  const auto second =
      runtime::write_prefill_mlp_k512_fragment_native_receipt_no_replace(
          receipt, path);
  test.expect(second.code ==
                  runtime::PrefillMLPK512OverlayErrorCode::
                      kPublicationConflict,
              "receipt publication never overwrites an existing target");

  const std::string document = read_file(path);
  runtime::PrefillMLPK512OverlayDiagnostic diagnostic;
  const auto parsed =
      runtime::parse_prefill_mlp_k512_fragment_native_receipt(document,
                                                              diagnostic);
  test.expect(parsed.has_value() && diagnostic.ok() &&
                  parsed->receipt_sha256.size() == 64U &&
                  parsed->source_v1.receipt_sha256 == std::string(64U, '7'),
              "strict receipt retains exact v1 receipt and computes own SHA");

  std::string tampered = document;
  const std::string policy_digest(64U, '5');
  const std::size_t position = tampered.find(policy_digest);
  if (position != std::string::npos) {
    tampered[position] = 'c';
  }
  test.expect(position != std::string::npos &&
                  !runtime::parse_prefill_mlp_k512_fragment_native_receipt(
                      tampered, diagnostic),
              "receipt rejects source policy binding tampering");
  fs::remove(path, ignored);
}

void test_hybrid_publication_contract(
    Test& test,
    const runtime::PrefillMLPK512PairedGateUpCanonicalDownManifest& manifest) {
  test.expect(
      manifest.physical_layout ==
              runtime::kPrefillMLPK512PairedGateUpCanonicalDownLayout &&
          manifest.required_base.physical_layout ==
              runtime::kPrefillA4K256PhysicalLayout &&
          manifest.payload_bytes == 8'623'226'880ULL &&
          manifest.layer_count == 64U &&
          runtime::validate_prefill_mlp_k512_paired_gateup_canonical_down_manifest(
              manifest)
              .ok(),
      "hybrid manifest is independently identified and K256-bound");
  test.expect(
      runtime::kPrefillMLPK512FragmentNativeGateUpRecordBytes * 64U ==
              5'748'817'920ULL &&
          runtime::kPrefillMLPK512FragmentNativeDownCodeBytes * 64U ==
              2'852'126'720ULL &&
          runtime::kPrefillMLPK512FragmentNativeDownScaleBytes * 64U ==
              22'282'240ULL,
      "hybrid equal-byte inventory partitions exact reusable and canonical ranges");

  const fs::path path =
      fs::temp_directory_path() /
      ("q3x-mlp-k512-hybrid-receipt-" + std::to_string(::getpid()) +
       ".json");
  std::error_code ignored;
  fs::remove(path, ignored);
  const auto receipt = make_hybrid_receipt(manifest);
  const auto first =
      runtime::write_prefill_mlp_k512_paired_gateup_canonical_down_receipt_no_replace(
          receipt, path);
  struct stat status {};
  test.expect(first.ok() && ::lstat(path.c_str(), &status) == 0 &&
                  (status.st_mode & (S_IWUSR | S_IWGRP | S_IWOTH)) == 0,
              "hybrid receipt publishes atomically and read-only");
  const auto second =
      runtime::write_prefill_mlp_k512_paired_gateup_canonical_down_receipt_no_replace(
          receipt, path);
  test.expect(second.code ==
                  runtime::PrefillMLPK512OverlayErrorCode::
                      kPublicationConflict,
              "hybrid receipt never replaces an existing target");
  const std::string document = read_file(path);
  runtime::PrefillMLPK512OverlayDiagnostic diagnostic;
  const auto parsed =
      runtime::parse_prefill_mlp_k512_paired_gateup_canonical_down_receipt(
          document, diagnostic);
  test.expect(parsed.has_value() && diagnostic.ok() &&
                  parsed->physical_layout ==
                      runtime::kPrefillMLPK512PairedGateUpCanonicalDownLayout &&
                  parsed->required_base.physical_layout ==
                      runtime::kPrefillA4K256PhysicalLayout,
              "strict hybrid receipt retains layout and K256 trust chain");
  std::string wrong_down = document;
  const std::string canonical_down(
      runtime::kPrefillMLPK512CanonicalDownComponentLayout);
  const std::size_t down_position = wrong_down.find(canonical_down);
  if (down_position != std::string::npos) {
    wrong_down[down_position] = 'x';
  }
  test.expect(
      down_position != std::string::npos &&
          !runtime::parse_prefill_mlp_k512_paired_gateup_canonical_down_receipt(
              wrong_down, diagnostic),
      "hybrid receipt rejects a Down-layout discriminator change");
  fs::remove(path, ignored);

  runtime::PrefillMLPK512PairedGateUpCanonicalDownCompositionOptions invalid;
  const auto incomplete =
      runtime::compose_authenticated_prefill_mlp_k512_paired_gateup_canonical_down(
          invalid);
  test.expect(
      incomplete.diagnostic.code ==
          runtime::PrefillMLPK512OverlayErrorCode::kInvalidOption,
      "hybrid composer rejects incomplete source trust roots before I/O");

  const fs::path occupied =
      fs::temp_directory_path() /
      ("q3x-mlp-k512-hybrid-occupied-" + std::to_string(::getpid()));
  fs::remove(occupied, ignored);
  {
    std::ofstream output(occupied, std::ios::binary);
    output << 'x';
  }
  runtime::PrefillMLPK512PairedGateUpCanonicalDownCompositionOptions conflict;
  conflict.source_v1_payload_path = occupied.string() + ".source";
  conflict.source_v1_receipt_path = occupied.string() + ".receipt";
  conflict.source_v1_policy_path = occupied.string() + ".policy";
  conflict.expected_source_v1_receipt_sha256.assign(64U, 'a');
  conflict.output_path = occupied;
  const auto conflict_result =
      runtime::compose_authenticated_prefill_mlp_k512_paired_gateup_canonical_down(
          conflict);
  test.expect(
      conflict_result.diagnostic.code ==
          runtime::PrefillMLPK512OverlayErrorCode::kPublicationConflict,
      "hybrid composer fails closed before replacing an occupied payload");
  fs::remove(occupied, ignored);
}

}  // namespace

int main() {
  Test test;
  const auto manifest = runtime::build_prefill_mlp_k512_fragment_native_manifest(
      make_v1_receipt(), std::string(64U, '7'));
  test.expect(static_cast<bool>(manifest),
              "valid authenticated v1 binding builds v2 manifest");
  if (!manifest) {
    std::cerr << runtime::to_string(manifest.diagnostic.code) << ' '
              << manifest.diagnostic.context << ' '
              << manifest.diagnostic.message << '\n';
    return 1;
  }
  test_inventory(test, *manifest.value);
  test_gateup_bijection(test);
  test_down_bijection(test);
  test_receipt(test, *manifest.value);
  const auto hybrid =
      runtime::build_prefill_mlp_k512_paired_gateup_canonical_down_manifest(
          make_k256_v1_receipt(), std::string(64U, '7'));
  test.expect(static_cast<bool>(hybrid),
              "valid K256-bound v1 builds an independent hybrid manifest");
  if (hybrid) {
    test_hybrid_publication_contract(test, *hybrid.value);
  }
  const auto wrong_base =
      runtime::build_prefill_mlp_k512_paired_gateup_canonical_down_manifest(
          make_v1_receipt(), std::string(64U, '7'));
  test.expect(!wrong_base,
              "hybrid manifest rejects the historical K128-bound v1 root");
  if (test.result() == 0) {
    std::cout << "MLP K512 fragment-native overlay host contract passed\n";
  }
  return test.result();
}
