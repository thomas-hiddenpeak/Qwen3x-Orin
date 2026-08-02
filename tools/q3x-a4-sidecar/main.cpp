#include "q3x/runtime/prefill_a4_sidecar_converter.h"
#include "q3x/runtime/prefill_attention_o_k512_overlay.h"
#include "q3x/runtime/prefill_mlp_k512_fragment_native_overlay.h"
#include "q3x/runtime/prefill_mlp_k512_overlay.h"

#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>

namespace {

namespace runtime = q3x::runtime;

void print_usage(std::ostream& output) {
  output
      << "qwen3x-a4-sidecar: authenticated offline A4 sidecar converter\n\n"
      << "Usage:\n"
      << "  qwen3x-a4-sidecar policy-template MODEL_DIR OUTPUT"
         " --weight-clip R --activation-clip R"
         " [--sidecar-kind a4-k64|a4-k128|a4-k256]\n"
      << "  qwen3x-a4-sidecar convert MODEL_DIR POLICY.json OUTPUT"
         " [--sidecar-kind a4-k64|a4-k128|a4-k256]"
         " [--row-chunk N] [--no-preallocate]\n"
      << "  qwen3x-a4-sidecar attention-o-k512-policy-template"
         " MODEL_DIR BASE_A4_RECEIPT OUTPUT"
         " --weight-clip R --activation-clip R\n"
      << "  qwen3x-a4-sidecar attention-o-k512-convert"
         " MODEL_DIR POLICY.json OUTPUT"
         " [--row-chunk N] [--no-preallocate]\n"
      << "  qwen3x-a4-sidecar mlp-k512-policy-template"
         " MODEL_DIR BASE_A4_RECEIPT OUTPUT_POLICY"
         " WEIGHT_CLIP ACTIVATION_CLIP\n"
      << "  qwen3x-a4-sidecar mlp-k512-convert"
         " MODEL_DIR POLICY.json OUTPUT_PAYLOAD [ROW_CHUNK]\n"
      << "  qwen3x-a4-sidecar mlp-k512-fragment-native-convert"
         " SOURCE_PAYLOAD SOURCE_RECEIPT SOURCE_POLICY"
         " EXPECTED_RECEIPT_SHA OUTPUT [ROWS]\n"
      << "  qwen3x-a4-sidecar help\n\n"
      << "The convert command accepts only the pinned Qwen3.6-27B-NVFP4 "
         "checkpoint and a versioned production_calibrated policy covering "
         "all 400 projections. It authenticates all source shards, streams "
         "bounded row chunks, and atomically publishes OUTPUT plus "
         "OUTPUT.receipt.json. Existing targets are never replaced.\n\n"
      << "policy-template builds the selected 400-projection manifest and "
         "atomically writes a no-replace candidate policy. Both clip ratios "
         "are mandatory; there are no defaults. Template status is not an "
         "accuracy, performance, or capability decision.\n\n"
      << "A4-K64 is the unchanged v1 default. A4-K128 and A4-K256 select "
         "separate v2/v3 packed-K64 shared-scale layouts.\n\n"
      << "--row-chunk must be 64, 128, 192, or 256; N64 is the physical "
         "consumer block, not a fixed kernel CTA tile.\n\n"
      << "Experimental nearest-even is exposed only through the bounded host "
         "quantization API for correctness/smoke tests and cannot create a "
         "production receipt.\n\n"
      << "The Attention-O K512 commands create a separate 64-projection "
         "overlay. They read the original authenticated FP8 checkpoint "
         "weights, never the base sidecar. The policy binds the exact base "
         "A4 receipt and requires independent explicit weight and "
         "activation clip ratios.\n\n"
      << "The MLP K512 commands create a separate 192-projection Gate/Up/Down "
         "overlay from the original authenticated checkpoint weights. Its "
         "policy also binds the exact base A4-K128/K256 receipt. The MLP command "
         "arguments are positional; both clip ratios are mandatory and "
         "ROW_CHUNK, when present, must be a positive supported row chunk.\n\n"
      << "The fragment-native command performs a lossless offline v1-to-v2 "
         "permutation. EXPECTED_RECEIPT_SHA is mandatory and must be the "
         "explicit lowercase SHA-256 of SOURCE_RECEIPT; the command never "
         "trusts a discovered receipt implicitly. ROWS may be 512 or 1024.\n";
}

[[nodiscard]] bool parse_ratio(const char* text, double& output) {
  if (text == nullptr || *text == '\0') {
    return false;
  }
  errno = 0;
  char* end = nullptr;
  const double value = std::strtod(text, &end);
  if (errno != 0 || end == text || *end != '\0' || !std::isfinite(value) ||
      value < runtime::kPrefillA4MinimumClipRatio || value > 1.0) {
    return false;
  }
  output = value;
  return true;
}

[[nodiscard]] bool parse_size(const char* text, std::size_t& output) {
  if (text == nullptr || *text == '\0' || *text == '-') {
    return false;
  }
  errno = 0;
  char* end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || value == 0U ||
      value > std::numeric_limits<std::size_t>::max()) {
    return false;
  }
  output = static_cast<std::size_t>(value);
  return true;
}

[[nodiscard]] bool parse_lower_sha256(const char* text,
                                      std::string& output) {
  if (text == nullptr || std::char_traits<char>::length(text) != 64U) {
    return false;
  }
  for (std::size_t index = 0U; index < 64U; ++index) {
    const char value = text[index];
    if (!((value >= '0' && value <= '9') ||
          (value >= 'a' && value <= 'f'))) {
      return false;
    }
  }
  output = text;
  return true;
}

[[nodiscard]] bool parse_sidecar_kind(
    const char* text, runtime::PrefillSidecarKind& output) {
  if (text == nullptr) {
    return false;
  }
  const std::string_view value(text);
  if (value == "a4-k64" || value == "a4_k64") {
    output = runtime::PrefillSidecarKind::kA4K64;
    return true;
  }
  if (value == "a4-k128" || value == "a4_k128") {
    output = runtime::PrefillSidecarKind::kA4K128;
    return true;
  }
  if (value == "a4-k256" || value == "a4_k256") {
    output = runtime::PrefillSidecarKind::kA4K256;
    return true;
  }
  return false;
}

void print_diagnostic(const runtime::PrefillA4ConverterDiagnostic& diagnostic) {
  std::cerr << "error.code=" << runtime::to_string(diagnostic.code)
            << "\nerror.context=" << diagnostic.context
            << "\nerror.message=" << diagnostic.message;
  if (!diagnostic.expected.empty()) {
    std::cerr << "\nerror.expected=" << diagnostic.expected;
  }
  if (!diagnostic.actual.empty()) {
    std::cerr << "\nerror.actual=" << diagnostic.actual;
  }
  if (diagnostic.system_error != 0) {
    std::cerr << "\nerror.errno=" << diagnostic.system_error;
  }
  std::cerr << '\n';
}

void print_diagnostic(
    const runtime::PrefillAttentionOK512OverlayDiagnostic& diagnostic) {
  std::cerr << "error.code=" << runtime::to_string(diagnostic.code)
            << "\nerror.context=" << diagnostic.context
            << "\nerror.message=" << diagnostic.message;
  if (!diagnostic.expected.empty()) {
    std::cerr << "\nerror.expected=" << diagnostic.expected;
  }
  if (!diagnostic.actual.empty()) {
    std::cerr << "\nerror.actual=" << diagnostic.actual;
  }
  if (diagnostic.system_error != 0) {
    std::cerr << "\nerror.errno=" << diagnostic.system_error;
  }
  std::cerr << '\n';
}

void print_diagnostic(
    const runtime::PrefillMLPK512OverlayDiagnostic& diagnostic) {
  std::cerr << "error.code=" << runtime::to_string(diagnostic.code)
            << "\nerror.context=" << diagnostic.context
            << "\nerror.message=" << diagnostic.message;
  if (!diagnostic.expected.empty()) {
    std::cerr << "\nerror.expected=" << diagnostic.expected;
  }
  if (!diagnostic.actual.empty()) {
    std::cerr << "\nerror.actual=" << diagnostic.actual;
  }
  if (diagnostic.system_error != 0) {
    std::cerr << "\nerror.errno=" << diagnostic.system_error;
  }
  std::cerr << '\n';
}

int run_mlp_k512_convert(const int argc, char** argv) {
  if (argc != 5 && argc != 6) {
    print_usage(std::cerr);
    return 2;
  }
  runtime::PrefillMLPK512OverlayConversionOptions options;
  options.model_directory = argv[2];
  options.calibration_policy_path = argv[3];
  options.output_path = argv[4];
  if (argc == 6 && !parse_size(argv[5], options.row_chunk_size)) {
    std::cerr << "invalid ROW_CHUNK value\n";
    return 2;
  }
  const auto result =
      runtime::convert_pinned_qwen36_27b_prefill_mlp_k512_overlay(options);
  if (!result) {
    print_diagnostic(result.diagnostic);
    return 1;
  }
  const auto& receipt = *result.receipt;
  std::cout << "status=published"
            << "\noverlay=mlp_k512"
            << "\nproduction_residency_eligible="
            << (receipt.production_residency_eligible ? "true" : "false")
            << "\nphysical_layout=" << receipt.physical_layout
            << "\nsource_checkpoint_id=" << receipt.source_checkpoint_id
            << "\nsource_config_sha256=" << receipt.source_config_sha256
            << "\nsource_index_sha256=" << receipt.source_index_sha256
            << "\nbase_physical_layout="
            << receipt.required_base.physical_layout
            << "\nbase_manifest_sha256="
            << receipt.required_base.manifest_sha256
            << "\nbase_policy_sha256=" << receipt.required_base.policy_sha256
            << "\nbase_payload_sha256=" << receipt.required_base.payload_sha256
            << "\nmanifest_sha256=" << receipt.manifest_sha256
            << "\npolicy_sha256=" << receipt.policy_sha256
            << "\npayload_sha256=" << receipt.payload_sha256
            << "\npayload_bytes=" << receipt.payload_bytes
            << "\nprojections=" << result.stats.projections_converted
            << "\nrows=" << result.stats.rows_converted
            << "\nsource_bytes_read=" << result.stats.source_bytes_read
            << "\noutput_bytes_written=" << result.stats.output_bytes_written
            << "\npeak_working_bytes=" << result.stats.peak_working_bytes
            << '\n';
  return 0;
}

int run_mlp_k512_fragment_native_convert(const int argc, char** argv) {
  if (argc != 7 && argc != 8) {
    print_usage(std::cerr);
    return 2;
  }
  runtime::PrefillMLPK512FragmentNativeConversionOptions options;
  options.source_v1_payload_path = argv[2];
  options.source_v1_receipt_path = argv[3];
  options.source_v1_policy_path = argv[4];
  if (!parse_lower_sha256(argv[5],
                          options.expected_source_v1_receipt_sha256)) {
    std::cerr << "EXPECTED_RECEIPT_SHA must be one explicit lowercase "
                 "SHA-256\n";
    return 2;
  }
  options.output_path = argv[6];
  if (argc == 8 &&
      (!parse_size(argv[7], options.outer_chunk_rows) ||
       (options.outer_chunk_rows != 512U &&
        options.outer_chunk_rows != 1'024U))) {
    std::cerr << "ROWS must be 512 or 1024\n";
    return 2;
  }
  const auto result =
      runtime::convert_authenticated_prefill_mlp_k512_to_fragment_native(
          options);
  if (!result) {
    print_diagnostic(result.diagnostic);
    return 1;
  }
  const auto& receipt = *result.receipt;
  std::cout << "status=published"
            << "\noverlay=mlp_k512_fragment_native"
            << "\nproduction_residency_eligible="
            << (receipt.production_residency_eligible ? "true" : "false")
            << "\nphysical_layout=" << receipt.physical_layout
            << "\nsource_checkpoint_id=" << receipt.source_checkpoint_id
            << "\nsource_config_sha256=" << receipt.source_config_sha256
            << "\nsource_index_sha256=" << receipt.source_index_sha256
            << "\nsource_v1_physical_layout="
            << receipt.source_v1.physical_layout
            << "\nsource_v1_receipt_sha256="
            << receipt.source_v1.receipt_sha256
            << "\nsource_v1_manifest_sha256="
            << receipt.source_v1.manifest_sha256
            << "\nsource_v1_policy_sha256="
            << receipt.source_v1.policy_sha256
            << "\nsource_v1_payload_sha256="
            << receipt.source_v1.payload_sha256
            << "\nmanifest_sha256=" << receipt.manifest_sha256
            << "\npayload_sha256=" << receipt.payload_sha256
            << "\npayload_bytes=" << receipt.payload_bytes
            << "\nlayers=" << receipt.layer_count
            << "\nsource_bytes_read=" << result.stats.source_bytes_read
            << "\noutput_bytes_written=" << result.stats.output_bytes_written
            << "\npeak_working_bytes=" << result.stats.peak_working_bytes
            << "\nlayers_permuted=" << result.stats.layers_permuted << '\n';
  return 0;
}

int run_mlp_k512_policy_template(const int argc, char** argv) {
  if (argc != 7) {
    print_usage(std::cerr);
    return 2;
  }
  runtime::PrefillMLPK512OverlayPolicyTemplateOptions options;
  options.model_directory = argv[2];
  options.base_k128_receipt_path = argv[3];
  options.output_path = argv[4];
  if (!parse_ratio(argv[5], options.weight_clip_ratio)) {
    std::cerr << "invalid WEIGHT_CLIP value\n";
    return 2;
  }
  if (!parse_ratio(argv[6], options.activation_clip_ratio)) {
    std::cerr << "invalid ACTIVATION_CLIP value\n";
    return 2;
  }
  const auto result =
      runtime::write_qwen36_27b_prefill_mlp_k512_overlay_policy_template(
          options);
  if (!result) {
    print_diagnostic(result.diagnostic);
    return 1;
  }
  const auto& policy = *result.value;
  std::cout << "status=candidate_template"
            << "\noverlay=mlp_k512"
            << "\nproduction_residency_eligible=not_evaluated"
            << "\nphysical_layout=" << policy.physical_layout
            << "\nsource_checkpoint_id=" << policy.source_checkpoint_id
            << "\nsource_config_sha256=" << policy.source_config_sha256
            << "\nsource_index_sha256=" << policy.source_index_sha256
            << "\nbase_physical_layout="
            << policy.required_base.physical_layout
            << "\nbase_manifest_sha256="
            << policy.required_base.manifest_sha256
            << "\nbase_policy_sha256=" << policy.required_base.policy_sha256
            << "\nbase_payload_sha256=" << policy.required_base.payload_sha256
            << "\nmanifest_sha256=" << policy.manifest_sha256
            << "\npolicy_sha256=" << policy.policy_sha256
            << "\npolicy_bytes=" << policy.policy_bytes
            << "\nprojections=" << policy.projections.size()
            << "\nweight_clip_ratio="
            << policy.projections.front().weight_clip_ratio
            << "\nactivation_clip_ratio="
            << policy.projections.front().activation_clip_ratio << '\n';
  return 0;
}

int run_attention_o_k512_convert(const int argc, char** argv) {
  if (argc < 5) {
    print_usage(std::cerr);
    return 2;
  }
  runtime::PrefillAttentionOK512OverlayConversionOptions options;
  options.model_directory = argv[2];
  options.calibration_policy_path = argv[3];
  options.output_path = argv[4];
  for (int index = 5; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--row-chunk") {
      if (++index >= argc || !parse_size(argv[index], options.row_chunk_size)) {
        std::cerr << "invalid --row-chunk value\n";
        return 2;
      }
    } else if (argument == "--no-preallocate") {
      options.preallocate_output = false;
    } else {
      std::cerr << "unknown argument: " << argument << '\n';
      return 2;
    }
  }
  const auto result =
      runtime::convert_pinned_qwen36_27b_prefill_attention_o_k512_overlay(
          options);
  if (!result) {
    print_diagnostic(result.diagnostic);
    return 1;
  }
  const auto& receipt = *result.receipt;
  std::cout << "status=published"
            << "\noverlay=attention_o_k512"
            << "\nproduction_residency_eligible="
            << (receipt.production_residency_eligible ? "true" : "false")
            << "\nphysical_layout=" << receipt.physical_layout
            << "\nmanifest_sha256=" << receipt.manifest_sha256
            << "\npolicy_sha256=" << receipt.policy_sha256
            << "\npayload_sha256=" << receipt.payload_sha256
            << "\npayload_bytes=" << receipt.payload_bytes
            << "\nprojections=" << result.stats.projections_converted
            << "\nrows=" << result.stats.rows_converted
            << "\nsource_bytes_read=" << result.stats.source_bytes_read
            << "\noutput_bytes_written=" << result.stats.output_bytes_written
            << "\npeak_working_bytes=" << result.stats.peak_working_bytes
            << '\n';
  return 0;
}

int run_attention_o_k512_policy_template(const int argc, char** argv) {
  if (argc < 6) {
    print_usage(std::cerr);
    return 2;
  }
  runtime::PrefillAttentionOK512OverlayPolicyTemplateOptions options;
  options.model_directory = argv[2];
  options.base_k128_receipt_path = argv[3];
  options.output_path = argv[4];
  bool have_weight_clip = false;
  bool have_activation_clip = false;
  for (int index = 5; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--weight-clip") {
      if (have_weight_clip || ++index >= argc ||
          !parse_ratio(argv[index], options.weight_clip_ratio)) {
        std::cerr << "invalid or duplicate --weight-clip\n";
        return 2;
      }
      have_weight_clip = true;
    } else if (argument == "--activation-clip") {
      if (have_activation_clip || ++index >= argc ||
          !parse_ratio(argv[index], options.activation_clip_ratio)) {
        std::cerr << "invalid or duplicate --activation-clip\n";
        return 2;
      }
      have_activation_clip = true;
    } else {
      std::cerr << "unknown argument: " << argument << '\n';
      return 2;
    }
  }
  if (!have_weight_clip || !have_activation_clip) {
    std::cerr << "both explicit clip ratios are required\n";
    return 2;
  }
  const auto result =
      runtime::write_qwen36_27b_prefill_attention_o_k512_overlay_policy_template(
          options);
  if (!result) {
    print_diagnostic(result.diagnostic);
    return 1;
  }
  const auto& policy = *result.value;
  std::cout << "status=candidate_template"
            << "\noverlay=attention_o_k512"
            << "\nproduction_residency_eligible=not_evaluated"
            << "\nphysical_layout=" << policy.physical_layout
            << "\nmanifest_sha256=" << policy.manifest_sha256
            << "\npolicy_sha256=" << policy.policy_sha256
            << "\npolicy_bytes=" << policy.policy_bytes
            << "\nprojections=" << policy.projections.size()
            << "\nweight_clip_ratio="
            << policy.projections.front().weight_clip_ratio
            << "\nactivation_clip_ratio="
            << policy.projections.front().activation_clip_ratio << '\n';
  return 0;
}

int run_convert(const int argc, char** argv) {
  if (argc < 5) {
    print_usage(std::cerr);
    return 2;
  }
  runtime::PrefillA4SidecarConversionOptions options;
  options.model_directory = argv[2];
  options.calibration_policy_path = argv[3];
  options.output_path = argv[4];
  for (int index = 5; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--row-chunk") {
      if (++index >= argc || !parse_size(argv[index], options.row_chunk_size)) {
        std::cerr << "invalid --row-chunk value\n";
        return 2;
      }
    } else if (argument == "--no-preallocate") {
      options.preallocate_output = false;
    } else if (argument == "--sidecar-kind") {
      if (++index >= argc ||
          !parse_sidecar_kind(argv[index], options.sidecar_kind)) {
        std::cerr << "invalid --sidecar-kind value\n";
        return 2;
      }
    } else {
      std::cerr << "unknown argument: " << argument << '\n';
      return 2;
    }
  }
  const runtime::PrefillA4SidecarConversionResult result =
      runtime::convert_pinned_qwen36_27b_prefill_a4_sidecar(options);
  if (!result) {
    print_diagnostic(result.diagnostic);
    return 1;
  }
  const runtime::PrefillA4PublicationReceipt& receipt = *result.receipt;
  std::cout << "status=published"
            << "\nmode=" << runtime::to_string(receipt.mode)
            << "\nproduction_residency_eligible="
            << (receipt.production_residency_eligible ? "true" : "false")
            << "\nsidecar_kind=" << runtime::to_string(receipt.sidecar_kind)
            << "\npacked_k_group_size=" << receipt.packed_k_group_size
            << "\nscale_group_size=" << receipt.scale_group_size
            << "\nphysical_layout=" << receipt.physical_layout
            << "\nmanifest_sha256=" << receipt.manifest_sha256
            << "\npolicy_sha256=" << receipt.policy_sha256
            << "\npayload_sha256=" << receipt.payload_sha256
            << "\npayload_bytes=" << receipt.payload_bytes
            << "\nprojections=" << result.stats.projections_converted
            << "\nrows=" << result.stats.rows_converted
            << "\nsource_bytes_read=" << result.stats.source_bytes_read
            << "\noutput_bytes_written=" << result.stats.output_bytes_written
            << "\npeak_working_bytes=" << result.stats.peak_working_bytes
            << '\n';
  return 0;
}

int run_policy_template(const int argc, char** argv) {
  if (argc < 4) {
    print_usage(std::cerr);
    return 2;
  }
  runtime::PrefillA4PolicyTemplateWriteOptions options;
  options.output_path = argv[3];
  bool have_weight_clip = false;
  bool have_activation_clip = false;
  for (int index = 4; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--weight-clip") {
      if (have_weight_clip || ++index >= argc ||
          !parse_ratio(argv[index], options.weight_clip_ratio)) {
        std::cerr << "invalid or duplicate --weight-clip; explicit ratio in "
                     "[1/256,1] is required\n";
        return 2;
      }
      have_weight_clip = true;
    } else if (argument == "--activation-clip") {
      if (have_activation_clip || ++index >= argc ||
          !parse_ratio(argv[index], options.activation_clip_ratio)) {
        std::cerr << "invalid or duplicate --activation-clip; explicit ratio "
                     "in [1/256,1] is required\n";
        return 2;
      }
      have_activation_clip = true;
    } else if (argument == "--sidecar-kind") {
      if (++index >= argc ||
          !parse_sidecar_kind(argv[index], options.sidecar_kind)) {
        std::cerr << "invalid --sidecar-kind value\n";
        return 2;
      }
    } else {
      std::cerr << "unknown argument: " << argument << '\n';
      return 2;
    }
  }
  if (!have_weight_clip || !have_activation_clip) {
    std::cerr << "both --weight-clip and --activation-clip are required; "
                 "neither has a default\n";
    return 2;
  }
  const runtime::PrefillA4PolicyTemplateWriteResult result =
      runtime::write_pinned_qwen36_27b_prefill_a4_calibration_policy_template(
          argv[2], options);
  if (!result) {
    print_diagnostic(result.diagnostic);
    return 1;
  }
  const runtime::PrefillA4CalibrationPolicy& policy = *result.policy;
  std::cout
      << "status=candidate_template"
      << "\nmode=" << runtime::to_string(policy.mode)
      << "\nproduction_residency_eligible=not_evaluated"
      << "\nproduction_residency_eligible_scope="
         "authenticated_abi_only_not_accuracy_performance_or_capability"
      << "\nsidecar_kind=" << runtime::to_string(policy.sidecar_kind)
      << "\nphysical_layout=" << policy.physical_layout
      << "\nmanifest_sha256=" << policy.manifest_sha256
      << "\npolicy_sha256=" << policy.policy_sha256
      << "\npolicy_bytes=" << policy.policy_bytes
      << "\nprojections=" << policy.projections.size() << '\n';
  return 0;
}

}  // namespace

int main(const int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "help") {
    print_usage(std::cout);
    return 0;
  }
  if (argc >= 2 && std::string_view(argv[1]) == "convert") {
    return run_convert(argc, argv);
  }
  if (argc >= 2 && std::string_view(argv[1]) == "policy-template") {
    return run_policy_template(argc, argv);
  }
  if (argc >= 2 &&
      std::string_view(argv[1]) == "attention-o-k512-policy-template") {
    return run_attention_o_k512_policy_template(argc, argv);
  }
  if (argc >= 2 &&
      std::string_view(argv[1]) == "attention-o-k512-convert") {
    return run_attention_o_k512_convert(argc, argv);
  }
  if (argc >= 2 &&
      std::string_view(argv[1]) == "mlp-k512-policy-template") {
    return run_mlp_k512_policy_template(argc, argv);
  }
  if (argc >= 2 && std::string_view(argv[1]) == "mlp-k512-convert") {
    return run_mlp_k512_convert(argc, argv);
  }
  if (argc >= 2 &&
      std::string_view(argv[1]) ==
          "mlp-k512-fragment-native-convert") {
    return run_mlp_k512_fragment_native_convert(argc, argv);
  }
  print_usage(std::cerr);
  return 2;
}
