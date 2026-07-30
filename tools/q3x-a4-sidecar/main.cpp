#include "q3x/runtime/prefill_a4_sidecar_converter.h"

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
      << "qwen3x-a4-sidecar: authenticated offline A4-K64 converter\n\n"
      << "Usage:\n"
      << "  qwen3x-a4-sidecar policy-template MODEL_DIR OUTPUT"
         " --weight-clip R --activation-clip R\n"
      << "  qwen3x-a4-sidecar convert MODEL_DIR POLICY.json OUTPUT"
         " [--row-chunk N] [--no-preallocate]\n"
      << "  qwen3x-a4-sidecar help\n\n"
      << "The convert command accepts only the pinned Qwen3.6-27B-NVFP4 "
         "checkpoint and a versioned production_calibrated policy covering "
         "all 400 projections. It authenticates all source shards, streams "
         "bounded row chunks, and atomically publishes OUTPUT plus "
         "OUTPUT.receipt.json. Existing targets are never replaced.\n\n"
      << "policy-template builds the fixed A4-K64 400-projection manifest and "
         "atomically writes a no-replace candidate policy. Both clip ratios "
         "are mandatory; there are no defaults. Template status is not an "
         "accuracy, performance, or capability decision.\n\n"
      << "--row-chunk must be 64, 128, 192, or 256; N64 is the physical "
         "consumer block, not a fixed kernel CTA tile.\n\n"
      << "Experimental nearest-even is exposed only through the bounded host "
         "quantization API for correctness/smoke tests and cannot create a "
         "production receipt.\n";
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
    } else {
      std::cerr << "unknown argument: " << argument << '\n';
      return 2;
    }
  }
  const runtime::PrefillA4SidecarConversionResult result =
      runtime::convert_pinned_qwen36_27b_prefill_a4_k64_sidecar(options);
  if (!result) {
    print_diagnostic(result.diagnostic);
    return 1;
  }
  const runtime::PrefillA4PublicationReceipt& receipt = *result.receipt;
  std::cout << "status=published"
            << "\nmode=" << runtime::to_string(receipt.mode)
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
  print_usage(std::cerr);
  return 2;
}
