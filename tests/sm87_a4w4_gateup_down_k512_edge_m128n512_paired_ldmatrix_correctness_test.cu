#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix.h"

// Reuse the audited canonical payload, guard, comparison, and Down-v1 test
// utilities.  The imported main is not executed; this test adds the paired-B
// offline publication and invokes the new combined admission.
#define main q3x_imported_m128n512_direct_b_main
#include "sm87_a4w4_gateup_down_k512_edge_m128n512_ldmatrix_correctness_test.cu"
#undef main

namespace {

struct PairedPayload final {
  std::vector<std::uint8_t> codes;
  std::vector<std::uint16_t> scales;
};

[[nodiscard]] PairedPayload make_paired_payload(
    const GateUpPayload& canonical, const std::size_t n,
    const std::size_t k) {
  const std::size_t k64_groups = k / 64U;
  const std::size_t k512_groups = k / 512U;
  PairedPayload result{
      std::vector<std::uint8_t>(
          kernels::sm87_a4w4_gateup_k512_fragment_native_code_capacity_bytes(
              n, k)),
      std::vector<std::uint16_t>(
          kernels::sm87_a4w4_gateup_k512_fragment_native_scale_capacity_elements(
              n, k))};
  for (std::size_t row = 0U; row < n; ++row) {
    for (std::size_t group = 0U; group < k512_groups; ++group) {
      const std::size_t canonical_scale =
          kernels::sm87_a4w4_gateup_down_edge_scale_offset(
              row, group, k512_groups);
      const std::size_t paired_scale =
          kernels::sm87_a4w4_gateup_k512_fragment_native_scale_pair_offset(
              row, group, k512_groups);
      result.scales[paired_scale] = canonical.gate_scales[canonical_scale];
      result.scales[paired_scale + 1U] =
          canonical.up_scales[canonical_scale];
    }
  }
  for (std::size_t block = 0U; block < n / 64U; ++block) {
    for (std::size_t group = 0U; group < k512_groups; ++group) {
      for (std::size_t phase = 0U; phase < 8U; ++phase) {
        for (std::size_t fragment = 0U; fragment < 8U; ++fragment) {
          const std::size_t fragment_n = block * 64U + fragment * 8U;
          for (std::size_t lane = 0U; lane < 32U; ++lane) {
            const std::size_t row = fragment_n + lane / 4U;
            const std::size_t canonical0 =
                kernels::sm87_a4w4_gateup_down_edge_packed_offset(
                    row, group * 8U + phase, 4U * (lane & 3U),
                    k64_groups);
            const std::size_t canonical1 =
                kernels::sm87_a4w4_gateup_down_edge_packed_offset(
                    row, group * 8U + phase,
                    16U + 4U * (lane & 3U), k64_groups);
            const std::size_t native =
                kernels::sm87_a4w4_gateup_k512_fragment_native_code_slot_offset(
                    fragment_n, group, phase, lane, k512_groups);
            std::memcpy(result.codes.data() + native,
                        canonical.gate.data() + canonical0, 4U);
            std::memcpy(result.codes.data() + native + 4U,
                        canonical.gate.data() + canonical1, 4U);
            std::memcpy(result.codes.data() + native + 8U,
                        canonical.up.data() + canonical0, 4U);
            std::memcpy(result.codes.data() + native + 12U,
                        canonical.up.data() + canonical1, 4U);
          }
        }
      }
    }
  }
  return result;
}

[[nodiscard]] bool capture_paired_graph(
    GuardedDevice<std::uint8_t>& a,
    GuardedDevice<std::uint16_t>& a_scales,
    GuardedDevice<std::uint8_t>& paired_codes,
    GuardedDevice<std::uint16_t>& paired_scales,
    const std::size_t logical_m, const std::size_t launch_m,
    const std::size_t n, const std::size_t k,
    GuardedDevice<std::uint8_t>& scratch,
    GuardedDevice<std::uint8_t>& output,
    GuardedDevice<std::uint16_t>& output_scales,
    const unsigned int maximum_ctas) {
  cudaStream_t stream{};
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  bool ok = cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                    "create paired graph stream") &&
            cuda_ok(cudaStreamBeginCapture(
                        stream, cudaStreamCaptureModeThreadLocal),
                    "begin paired graph capture");
  if (ok) {
    ok = launch_ok(
             kernels::launch_sm87_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_test_cuda(
                 a.payload(), a.payload_count(), a_scales.payload(),
                 a_scales.payload_count(), paired_codes.payload(),
                 paired_codes.payload_count(), paired_scales.payload(),
                 paired_scales.payload_count(), logical_m, launch_m, n, k,
                 kClipRatio, scratch.payload(), scratch.payload_count(),
                 output.payload(), output.payload_count(),
                 output_scales.payload(), output_scales.payload_count(),
                 maximum_ctas, stream),
             "capture paired launch") &&
         cuda_ok(cudaStreamEndCapture(stream, &graph),
                 "end paired graph capture") &&
         cuda_ok(cudaGraphInstantiate(&executable, graph, nullptr, nullptr,
                                      0U),
                 "instantiate paired graph") &&
         cuda_ok(cudaGraphLaunch(executable, stream),
                 "paired graph replay one") &&
         cuda_ok(cudaGraphLaunch(executable, stream),
                 "paired graph replay two") &&
         cuda_ok(cudaStreamSynchronize(stream),
                 "synchronize paired graph");
  }
  if (executable != nullptr) {
    (void)cudaGraphExecDestroy(executable);
  }
  if (graph != nullptr) {
    (void)cudaGraphDestroy(graph);
  }
  if (stream != nullptr) {
    (void)cudaStreamDestroy(stream);
  }
  return ok;
}

[[nodiscard]] bool run_paired_case(
    const std::size_t logical_m, const std::size_t launch_m,
    const std::size_t n, const std::size_t k,
    const unsigned int maximum_ctas, const bool graph,
    const bool downstream) {
  const std::string shape = "M" + std::to_string(logical_m) + "/P" +
                            std::to_string(launch_m) + " N" +
                            std::to_string(n) + " K" + std::to_string(k);
  const GateUpPayload canonical = make_gateup_payload(launch_m, n, k);
  const PairedPayload paired = make_paired_payload(canonical, n, k);
  const auto plan =
      kernels::sm87_a4w4_gateup_down_edge_m128n512_paired_ldmatrix_test_plan(
          logical_m, launch_m, n, k, maximum_ctas);
  const std::size_t output_bytes =
      kernels::sm87_a4w4_gateup_down_edge_packed_capacity_bytes(
          launch_m, n);
  const std::size_t output_scale_count =
      kernels::sm87_a4w4_gateup_down_edge_scale_capacity_elements(
          launch_m, n);

  GuardedDevice<std::uint8_t> a;
  GuardedDevice<std::uint16_t> a_scales;
  GuardedDevice<std::uint8_t> gate;
  GuardedDevice<std::uint16_t> gate_scales;
  GuardedDevice<std::uint8_t> up;
  GuardedDevice<std::uint16_t> up_scales;
  GuardedDevice<std::uint8_t> paired_codes;
  GuardedDevice<std::uint16_t> paired_scales;
  GuardedDevice<std::uint8_t> scratch;
  GuardedDevice<std::uint8_t> baseline_output;
  GuardedDevice<std::uint16_t> baseline_scales;
  GuardedDevice<std::uint8_t> candidate_output;
  GuardedDevice<std::uint16_t> candidate_scales;
  if (!a.initialize(canonical.a, kByteGuard, kByteSentinel, "A") ||
      !a_scales.initialize(canonical.a_scales, kWordGuard,
                           kWordSentinel, "A scales") ||
      !gate.initialize(canonical.gate, kByteGuard, kByteSentinel,
                       "canonical Gate") ||
      !gate_scales.initialize(canonical.gate_scales, kWordGuard,
                              kWordSentinel, "canonical Gate scales") ||
      !up.initialize(canonical.up, kByteGuard, kByteSentinel,
                     "canonical Up") ||
      !up_scales.initialize(canonical.up_scales, kWordGuard,
                            kWordSentinel, "canonical Up scales") ||
      !paired_codes.initialize(paired.codes, kByteGuard, kByteSentinel,
                               "paired codes") ||
      !paired_scales.initialize(paired.scales, kWordGuard, kWordSentinel,
                                "paired scales") ||
      !scratch.initialize(
          std::vector<std::uint8_t>(plan.required_scratch_bytes, 0x3cU),
          kByteGuard, kByteSentinel, "scratch") ||
      !baseline_output.initialize(
          std::vector<std::uint8_t>(output_bytes, kByteSentinel),
          kByteGuard, kByteSentinel, "baseline output") ||
      !baseline_scales.initialize(
          std::vector<std::uint16_t>(output_scale_count, kWordSentinel),
          kWordGuard, kWordSentinel, "baseline output scales") ||
      !candidate_output.initialize(
          std::vector<std::uint8_t>(output_bytes, kByteSentinel),
          kByteGuard, kByteSentinel, "candidate output") ||
      !candidate_scales.initialize(
          std::vector<std::uint16_t>(output_scale_count, kWordSentinel),
          kWordGuard, kWordSentinel, "candidate output scales")) {
    std::cerr << shape << " allocation failed\n";
    return false;
  }

  const int short_scratch =
      kernels::launch_sm87_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_test_cuda(
          a.payload(), a.payload_count(), a_scales.payload(),
          a_scales.payload_count(), paired_codes.payload(),
          paired_codes.payload_count(), paired_scales.payload(),
          paired_scales.payload_count(), logical_m, launch_m, n, k,
          kClipRatio, scratch.payload(), scratch.payload_count() - 1U,
          candidate_output.payload(), candidate_output.payload_count(),
          candidate_scales.payload(), candidate_scales.payload_count(),
          maximum_ctas);
  const int aliased_scratch =
      kernels::launch_sm87_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_test_cuda(
          a.payload(), a.payload_count(), a_scales.payload(),
          a_scales.payload_count(), paired_codes.payload(),
          paired_codes.payload_count(), paired_scales.payload(),
          paired_scales.payload_count(), logical_m, launch_m, n, k,
          kClipRatio, candidate_output.payload(), plan.required_scratch_bytes,
          candidate_output.payload(), candidate_output.payload_count(),
          candidate_scales.payload(), candidate_scales.payload_count(),
          maximum_ctas);
  if (short_scratch != static_cast<int>(cudaErrorInvalidValue) ||
      aliased_scratch != static_cast<int>(cudaErrorInvalidValue)) {
    std::cerr << shape << " scratch capacity/alias rejection failed\n";
    return false;
  }

  if (!launch_ok(
          kernels::launch_sm87_a4w4_gateup_down_k512_edge_test_cuda(
              a.payload(), a.payload_count(), a_scales.payload(),
              a_scales.payload_count(), gate.payload(), gate.payload_count(),
              gate_scales.payload(), gate_scales.payload_count(), up.payload(),
              up.payload_count(), up_scales.payload(), up_scales.payload_count(),
              logical_m, launch_m, n, k, kClipRatio,
              baseline_output.payload(), baseline_output.payload_count(),
              baseline_scales.payload(), baseline_scales.payload_count(),
              maximum_ctas),
          "launch incumbent " + shape) ||
      !launch_ok(
          kernels::launch_sm87_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_test_cuda(
              a.payload(), a.payload_count(), a_scales.payload(),
              a_scales.payload_count(), paired_codes.payload(),
              paired_codes.payload_count(), paired_scales.payload(),
              paired_scales.payload_count(), logical_m, launch_m, n, k,
              kClipRatio, scratch.payload(), scratch.payload_count(),
              candidate_output.payload(), candidate_output.payload_count(),
              candidate_scales.payload(), candidate_scales.payload_count(),
              maximum_ctas),
          "launch paired " + shape) ||
      !cuda_ok(cudaDeviceSynchronize(), "synchronize " + shape)) {
    return false;
  }

  std::vector<std::uint8_t> baseline_output_host;
  std::vector<std::uint8_t> candidate_output_host;
  std::vector<std::uint16_t> baseline_scales_host;
  std::vector<std::uint16_t> candidate_scales_host;
  std::vector<std::uint8_t> scratch_host;
  if (!baseline_output.copy(baseline_output_host, "baseline output") ||
      !candidate_output.copy(candidate_output_host, "candidate output") ||
      !baseline_scales.copy(baseline_scales_host, "baseline scales") ||
      !candidate_scales.copy(candidate_scales_host, "candidate scales") ||
      !scratch.copy(scratch_host, "scratch") ||
      !baseline_output.guards_intact(baseline_output_host,
                                     "baseline output") ||
      !candidate_output.guards_intact(candidate_output_host,
                                      "candidate output") ||
      !baseline_scales.guards_intact(baseline_scales_host,
                                     "baseline scales") ||
      !candidate_scales.guards_intact(candidate_scales_host,
                                      "candidate scales") ||
      !scratch.guards_intact(scratch_host, "scratch") ||
      !compare(baseline_output_host, candidate_output_host,
               shape + " packed") ||
      !compare(baseline_scales_host, candidate_scales_host,
               shape + " scales")) {
    return false;
  }

  if (graph &&
      (!capture_paired_graph(a, a_scales, paired_codes, paired_scales,
                             logical_m, launch_m, n, k, scratch,
                             candidate_output, candidate_scales,
                             maximum_ctas) ||
       !candidate_output.copy(candidate_output_host,
                              "graph candidate output") ||
       !candidate_scales.copy(candidate_scales_host,
                              "graph candidate scales") ||
       !scratch.copy(scratch_host, "graph scratch") ||
       !candidate_output.guards_intact(candidate_output_host,
                                       "graph candidate output") ||
       !candidate_scales.guards_intact(candidate_scales_host,
                                       "graph candidate scales") ||
       !scratch.guards_intact(scratch_host, "graph scratch") ||
       !compare(baseline_output_host, candidate_output_host,
                shape + " graph packed") ||
       !compare(baseline_scales_host, candidate_scales_host,
                shape + " graph scales"))) {
    return false;
  }

  if (downstream) {
    constexpr std::size_t down_n = 128U;
    constexpr std::size_t down_stride = down_n + 8U;
    const DownPayload down_host = make_down_payload(down_n, n);
    GuardedDevice<std::uint8_t> down_weight;
    GuardedDevice<std::uint16_t> down_scales;
    GuardedDevice<std::uint16_t> baseline_down;
    GuardedDevice<std::uint16_t> candidate_down;
    const std::size_t down_elements = launch_m * down_stride;
    if (!down_weight.initialize(down_host.weight, kByteGuard,
                                kByteSentinel, "Down weight") ||
        !down_scales.initialize(down_host.scales, kWordGuard,
                                kWordSentinel, "Down scales") ||
        !baseline_down.initialize(
            std::vector<std::uint16_t>(down_elements, kBf16Sentinel),
            kWordGuard, kBf16Sentinel, "baseline Down") ||
        !candidate_down.initialize(
            std::vector<std::uint16_t>(down_elements, kBf16Sentinel),
            kWordGuard, kBf16Sentinel, "candidate Down") ||
        !launch_ok(
            kernels::launch_sm87_a4w4_down_k512_macrocell_test_bf16_cuda(
                baseline_output.payload(), baseline_output.payload_count(),
                baseline_scales.payload(), baseline_scales.payload_count(),
                down_weight.payload(), down_weight.payload_count(),
                down_scales.payload(), down_scales.payload_count(), launch_m,
                down_n, n, baseline_down.payload(), down_stride,
                down_elements, 2U),
            "launch baseline Down") ||
        !launch_ok(
            kernels::launch_sm87_a4w4_down_k512_macrocell_test_bf16_cuda(
                candidate_output.payload(), candidate_output.payload_count(),
                candidate_scales.payload(), candidate_scales.payload_count(),
                down_weight.payload(), down_weight.payload_count(),
                down_scales.payload(), down_scales.payload_count(), launch_m,
                down_n, n, candidate_down.payload(), down_stride,
                down_elements, 2U),
            "launch candidate Down") ||
        !cuda_ok(cudaDeviceSynchronize(), "synchronize Down")) {
      return false;
    }
    std::vector<std::uint16_t> baseline_down_host;
    std::vector<std::uint16_t> candidate_down_host;
    if (!baseline_down.copy(baseline_down_host, "baseline Down") ||
        !candidate_down.copy(candidate_down_host, "candidate Down") ||
        !baseline_down.guards_intact(baseline_down_host,
                                     "baseline Down") ||
        !candidate_down.guards_intact(candidate_down_host,
                                      "candidate Down") ||
        !compare(baseline_down_host, candidate_down_host,
                 shape + " Down BF16") ||
        !down_weight.unchanged("Down weight") ||
        !down_scales.unchanged("Down scales")) {
      return false;
    }
  }

  if (!a.unchanged("A") || !a_scales.unchanged("A scales") ||
      !gate.unchanged("canonical Gate") ||
      !gate_scales.unchanged("canonical Gate scales") ||
      !up.unchanged("canonical Up") ||
      !up_scales.unchanged("canonical Up scales") ||
      !paired_codes.unchanged("paired codes") ||
      !paired_scales.unchanged("paired scales")) {
    return false;
  }
  std::cout << "PASS: paired M128N512 LDSM bit-exact " << shape
            << (graph ? " graphx2" : "")
            << (downstream ? " Down-v1-BF16" : "") << '\n';
  return true;
}

}  // namespace

int main() {
  const int target = target_status();
  if (target != 0) {
    return target;
  }
  kernels::Sm87A4W4GateUpDownEdgeM128N512PairedLdmatrixResources resources{};
  if (!launch_ok(
          kernels::query_sm87_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_resources_cuda(
              &resources),
          "query paired M128N512 resources") ||
      resources.registers_per_thread > 128 ||
      resources.local_bytes != 0U ||
      resources.dynamic_shared_bytes != 132'096U ||
      resources.active_blocks_per_sm != 1) {
    std::cerr << "paired M128N512 resource gate failed\n";
    return 1;
  }
  const bool ok =
      run_paired_case(128U, 128U, 512U, 512U, 1U, false, false) &&
      run_paired_case(2'049U, 2'176U, 1'024U, 5'120U, 16U, true, true);
  if (!ok) {
    return 1;
  }
  std::cout << "paired M128N512 admission passed: regs="
            << resources.registers_per_thread
            << " dynamic_shared=" << resources.dynamic_shared_bytes
            << " active_blocks_per_sm=" << resources.active_blocks_per_sm
            << '\n';
  return 0;
}
