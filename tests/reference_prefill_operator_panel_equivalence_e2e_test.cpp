#include "q3x/core/sha256.h"
#include "q3x/kernels/sm87_fp8_marlin_w8a16.h"
#include "q3x/kernels/sm87_nvfp4_marlin.h"
#include "q3x/runtime/decode_ops.h"
#include "q3x/runtime/reference_engine.h"

#include "reference_engine_prefill_authority.h"
#include "reference_runner_full_attention_oracle_internal.h"
#include "reference_runner_gdn_chunk64_native_admission.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace core = q3x::core;
namespace runtime = q3x::runtime;
namespace engine_detail = q3x::runtime::reference_engine_detail;
namespace runner_detail = q3x::runtime::reference_runner_detail;

constexpr std::array<std::size_t, 5U> kPromptTokenCounts{
    {513U, 1'025U, 7'712U, 8'192U, 8'193U}};
constexpr std::size_t kPromptTemplateTokens = 12U;
constexpr std::size_t kHashCopyChunkBytes = 4U * 1024U * 1024U;
constexpr std::size_t kKvHeadCount = 4U;
constexpr std::size_t kKvHeadDimension = 256U;
constexpr std::size_t kQueryHeadCount = 24U;
constexpr std::size_t kQueryElementsPerToken =
    kQueryHeadCount * kKvHeadDimension;
constexpr std::size_t kKvElementsPerToken =
    kKvHeadCount * kKvHeadDimension;
constexpr std::size_t kLayer3AttentionOracleTokens = 513U;
constexpr std::size_t kLayer3AttentionOracleQueryElements =
    kLayer3AttentionOracleTokens * kQueryElementsPerToken;
constexpr std::size_t kLayer3AttentionOracleKvElements =
    kLayer3AttentionOracleTokens * kKvElementsPerToken;
constexpr std::size_t kBf16Bytes = sizeof(std::uint16_t);

class DeviceBf16Buffer final {
 public:
  DeviceBf16Buffer() = default;
  ~DeviceBf16Buffer() {
    if (data_ != nullptr) {
      (void)cudaFree(data_);
    }
  }
  DeviceBf16Buffer(const DeviceBf16Buffer&) = delete;
  DeviceBf16Buffer& operator=(const DeviceBf16Buffer&) = delete;

  [[nodiscard]] cudaError_t allocate(const std::size_t elements) noexcept {
    if (data_ != nullptr || elements == 0U) {
      return cudaErrorInvalidValue;
    }
    const cudaError_t status =
        cudaMalloc(reinterpret_cast<void**>(&data_),
                   elements * sizeof(std::uint16_t));
    if (status != cudaSuccess) {
      return status;
    }
    elements_ = elements;
    return cudaSuccess;
  }

  [[nodiscard]] std::uint16_t* data() noexcept { return data_; }
  [[nodiscard]] const std::uint16_t* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t elements() const noexcept { return elements_; }

 private:
  std::uint16_t* data_ = nullptr;
  std::size_t elements_ = 0U;
};

struct Layer3AttentionP513OracleCollector {
  DeviceBf16Buffer canonical_output_device;
  DeviceBf16Buffer group_q64_output_device;
  DeviceBf16Buffer flashinfer_output_device;
  DeviceBf16Buffer query_snapshot_device;
  DeviceBf16Buffer gate_snapshot_device;
  DeviceBf16Buffer key_snapshot_device;
  DeviceBf16Buffer value_snapshot_device;

  std::vector<std::uint16_t> canonical_output;
  std::vector<std::uint16_t> group_q64_output;
  std::vector<std::uint16_t> flashinfer_output;
  std::vector<std::uint16_t> query_before;
  std::vector<std::uint16_t> query_after;
  std::vector<std::uint16_t> gate_before;
  std::vector<std::uint16_t> gate_after;
  std::vector<std::uint16_t> key_before;
  std::vector<std::uint16_t> key_after;
  std::vector<std::uint16_t> value_before;
  std::vector<std::uint16_t> value_after;

  std::size_t hook_calls = 0U;
  bool prepared = false;
  bool completed = false;
  const char* failed_operation = "none";
  int cuda_error = static_cast<int>(cudaSuccess);

  [[nodiscard]] bool prepare() noexcept {
    try {
      canonical_output.resize(kLayer3AttentionOracleQueryElements);
      group_q64_output.resize(kLayer3AttentionOracleQueryElements);
      flashinfer_output.resize(kLayer3AttentionOracleQueryElements);
      query_before.resize(kLayer3AttentionOracleQueryElements);
      query_after.resize(kLayer3AttentionOracleQueryElements);
      gate_before.resize(kLayer3AttentionOracleQueryElements);
      gate_after.resize(kLayer3AttentionOracleQueryElements);
      key_before.resize(kLayer3AttentionOracleKvElements);
      key_after.resize(kLayer3AttentionOracleKvElements);
      value_before.resize(kLayer3AttentionOracleKvElements);
      value_after.resize(kLayer3AttentionOracleKvElements);
    } catch (const std::bad_alloc&) {
      failed_operation = "host_allocation";
      return false;
    } catch (const std::length_error&) {
      failed_operation = "host_vector_length";
      return false;
    }

    const auto allocate_device = [this](DeviceBf16Buffer& buffer,
                                        const std::size_t elements) noexcept {
      const cudaError_t status = buffer.allocate(elements);
      if (status == cudaSuccess) {
        return true;
      }
      failed_operation = "device_allocation";
      cuda_error = static_cast<int>(status);
      return false;
    };
    prepared = allocate_device(canonical_output_device,
                               kLayer3AttentionOracleQueryElements) &&
               allocate_device(group_q64_output_device,
                               kLayer3AttentionOracleQueryElements) &&
               allocate_device(flashinfer_output_device,
                               kLayer3AttentionOracleQueryElements) &&
               allocate_device(query_snapshot_device,
                               kLayer3AttentionOracleQueryElements) &&
               allocate_device(gate_snapshot_device,
                               kLayer3AttentionOracleQueryElements) &&
               allocate_device(key_snapshot_device,
                               kLayer3AttentionOracleKvElements) &&
               allocate_device(value_snapshot_device,
                               kLayer3AttentionOracleKvElements);
    if (!prepared) {
      failed_operation = "device_allocation";
    }
    return prepared;
  }
};

[[nodiscard]] bool record_oracle_cuda(
    Layer3AttentionP513OracleCollector& collector,
    const cudaError_t status, const char* const operation) noexcept {
  if (status == cudaSuccess) {
    return true;
  }
  collector.failed_operation = operation;
  collector.cuda_error = static_cast<int>(status);
  return false;
}

void collect_layer3_attention_p513_oracle(
    const runner_detail::PrefillLayer3AttentionP513OracleView& view,
    void* const context) noexcept {
  auto* const collector =
      static_cast<Layer3AttentionP513OracleCollector*>(context);
  if (collector == nullptr) {
    return;
  }
  ++collector->hook_calls;
  if (!collector->prepared || collector->hook_calls != 1U ||
      view.layer != 3U || view.first_position != 0U ||
      view.token_count != kLayer3AttentionOracleTokens ||
      view.processed_query == nullptr || view.key_cache == nullptr ||
      view.value_cache == nullptr || view.packed_gate == nullptr) {
    collector->failed_operation = "hook_contract";
    collector->cuda_error = static_cast<int>(cudaErrorInvalidValue);
    return;
  }

  const auto stream = static_cast<cudaStream_t>(view.cuda_stream);
  constexpr std::size_t kQueryBytes =
      kLayer3AttentionOracleQueryElements * sizeof(std::uint16_t);
  constexpr std::size_t kKvBytes =
      kLayer3AttentionOracleKvElements * sizeof(std::uint16_t);
  if (!record_oracle_cuda(
          *collector,
          cudaMemcpyAsync(collector->query_snapshot_device.data(),
                          view.processed_query, kQueryBytes,
                          cudaMemcpyDeviceToDevice, stream),
          "snapshot_query") ||
      !record_oracle_cuda(
          *collector,
          cudaMemcpyAsync(collector->gate_snapshot_device.data(),
                          view.packed_gate, kQueryBytes,
                          cudaMemcpyDeviceToDevice, stream),
          "snapshot_gate") ||
      !record_oracle_cuda(
          *collector,
          cudaMemcpyAsync(collector->key_snapshot_device.data(), view.key_cache,
                          kKvBytes, cudaMemcpyDeviceToDevice, stream),
          "snapshot_key") ||
      !record_oracle_cuda(
          *collector,
          cudaMemcpyAsync(collector->value_snapshot_device.data(),
                          view.value_cache, kKvBytes,
                          cudaMemcpyDeviceToDevice, stream),
          "snapshot_value")) {
    return;
  }

  const runtime::LayerMajorPrefillArithmeticSpanLedger ledger =
      runtime::make_layer_major_prefill_arithmetic_span_ledger(
          kLayer3AttentionOracleTokens);
  if (!runtime::is_valid_layer_major_prefill_arithmetic_span_ledger(ledger) ||
      ledger.span_count != 2U || ledger.spans[0U].token_offset != 0U ||
      ledger.spans[0U].token_count != 257U ||
      ledger.spans[1U].token_offset != 257U ||
      ledger.spans[1U].token_count != 256U) {
    collector->failed_operation = "canonical_span_ledger";
    collector->cuda_error = static_cast<int>(cudaErrorInvalidValue);
    return;
  }
  for (std::size_t span_index = 0U; span_index < ledger.span_count;
       ++span_index) {
    const runtime::LayerMajorPrefillArithmeticSpan& span =
        ledger.spans[span_index];
    const std::size_t query_offset =
        static_cast<std::size_t>(span.token_offset) * kQueryElementsPerToken;
    const int launch_status = runtime::
        launch_bulk_causal_gqa_sigmoid_gate_24_4_256_fixed_cuda(
            view.processed_query + query_offset, view.key_cache,
            view.value_cache, view.packed_gate + query_offset,
            view.first_position + span.token_offset, span.token_count,
            collector->canonical_output_device.data() + query_offset,
            view.cuda_stream);
    if (!record_oracle_cuda(*collector,
                            static_cast<cudaError_t>(launch_status),
                            span_index == 0U ? "canonical_span_257"
                                             : "canonical_span_256")) {
      return;
    }
  }

  const int group_status = runtime::
      launch_bulk_causal_gqa_sigmoid_gate_24_4_256_group_q64_panel_fixed_cuda(
          view.processed_query, view.key_cache, view.value_cache,
          view.packed_gate, view.first_position, view.token_count,
          collector->group_q64_output_device.data(), view.cuda_stream);
  if (!record_oracle_cuda(*collector,
                          static_cast<cudaError_t>(group_status),
                          "group_q64_panel")) {
    return;
  }
  const int flashinfer_status = runtime::
      launch_bulk_causal_gqa_sigmoid_gate_24_4_256_flashinfer_exact_panel_fixed_cuda(
          view.processed_query, view.key_cache, view.value_cache,
          view.packed_gate, view.first_position, view.token_count,
          collector->flashinfer_output_device.data(), view.cuda_stream);
  if (!record_oracle_cuda(*collector,
                          static_cast<cudaError_t>(flashinfer_status),
                          "flashinfer_panel") ||
      !record_oracle_cuda(*collector, cudaStreamSynchronize(stream),
                          "oracle_synchronize")) {
    return;
  }

  const auto copy_to_host = [&collector](
                                std::vector<std::uint16_t>& destination,
                                const std::uint16_t* const source,
                                const std::size_t bytes,
                                const char* const operation) noexcept {
    return record_oracle_cuda(
        *collector,
        cudaMemcpy(destination.data(), source, bytes, cudaMemcpyDeviceToHost),
        operation);
  };
  if (!copy_to_host(collector->canonical_output,
                    collector->canonical_output_device.data(), kQueryBytes,
                    "copy_canonical_output") ||
      !copy_to_host(collector->group_q64_output,
                    collector->group_q64_output_device.data(), kQueryBytes,
                    "copy_group_q64_output") ||
      !copy_to_host(collector->flashinfer_output,
                    collector->flashinfer_output_device.data(), kQueryBytes,
                    "copy_flashinfer_output") ||
      !copy_to_host(collector->query_before,
                    collector->query_snapshot_device.data(), kQueryBytes,
                    "copy_query_before") ||
      !copy_to_host(collector->query_after, view.processed_query, kQueryBytes,
                    "copy_query_after") ||
      !copy_to_host(collector->gate_before,
                    collector->gate_snapshot_device.data(), kQueryBytes,
                    "copy_gate_before") ||
      !copy_to_host(collector->gate_after, view.packed_gate, kQueryBytes,
                    "copy_gate_after") ||
      !copy_to_host(collector->key_before,
                    collector->key_snapshot_device.data(), kKvBytes,
                    "copy_key_before") ||
      !copy_to_host(collector->key_after, view.key_cache, kKvBytes,
                    "copy_key_after") ||
      !copy_to_host(collector->value_before,
                    collector->value_snapshot_device.data(), kKvBytes,
                    "copy_value_before") ||
      !copy_to_host(collector->value_after, view.value_cache, kKvBytes,
                    "copy_value_after")) {
    return;
  }
  collector->completed = true;
}

class ScopedLayer3AttentionP513Oracle final {
 public:
  explicit ScopedLayer3AttentionP513Oracle(
      Layer3AttentionP513OracleCollector& collector) noexcept
      : previous_(runner_detail::
                      exchange_prefill_layer3_attention_p513_oracle_hook(
                          {collect_layer3_attention_p513_oracle,
                           &collector})) {}

  ~ScopedLayer3AttentionP513Oracle() {
    (void)runner_detail::exchange_prefill_layer3_attention_p513_oracle_hook(
        previous_);
  }
  ScopedLayer3AttentionP513Oracle(
      const ScopedLayer3AttentionP513Oracle&) = delete;
  ScopedLayer3AttentionP513Oracle& operator=(
      const ScopedLayer3AttentionP513Oracle&) = delete;

 private:
  runner_detail::PrefillLayer3AttentionP513OracleHook previous_{};
};

[[nodiscard]] std::string bf16_hex(const std::uint16_t bits) {
  constexpr char kHex[] = "0123456789abcdef";
  std::string result = "0x0000";
  result[2U] = kHex[(bits >> 12U) & 0xFU];
  result[3U] = kHex[(bits >> 8U) & 0xFU];
  result[4U] = kHex[(bits >> 4U) & 0xFU];
  result[5U] = kHex[bits & 0xFU];
  return result;
}

struct FirstBf16Mismatch {
  bool found = false;
  std::size_t index = 0U;
};

[[nodiscard]] FirstBf16Mismatch first_bf16_mismatch(
    const std::vector<std::uint16_t>& left,
    const std::vector<std::uint16_t>& right) noexcept {
  const std::size_t count = std::min(left.size(), right.size());
  for (std::size_t index = 0U; index < count; ++index) {
    if (left[index] != right[index]) {
      return {true, index};
    }
  }
  return {};
}

[[nodiscard]] float bf16_to_float(const std::uint16_t bits) noexcept {
  const std::uint32_t float_bits = static_cast<std::uint32_t>(bits) << 16U;
  float value = 0.0F;
  static_assert(sizeof(value) == sizeof(float_bits));
  std::memcpy(&value, &float_bits, sizeof(value));
  return value;
}

struct Bf16PairNumericMetrics {
  std::size_t elements = 0U;
  std::size_t mismatch_count = 0U;
  long double max_abs = 0.0L;
  long double mean_abs = 0.0L;
  long double nrmse_reference_rms = 0.0L;
  long double cosine = 0.0L;
  bool finite = false;
};

[[nodiscard]] Bf16PairNumericMetrics compute_bf16_pair_numeric_metrics(
    const std::vector<std::uint16_t>& reference,
    const std::vector<std::uint16_t>& candidate,
    const std::size_t token_begin, const std::size_t token_count) noexcept {
  Bf16PairNumericMetrics metrics;
  if (token_count == 0U ||
      token_begin > kLayer3AttentionOracleTokens ||
      token_count > kLayer3AttentionOracleTokens - token_begin) {
    return metrics;
  }
  const std::size_t begin = token_begin * kQueryElementsPerToken;
  const std::size_t count = token_count * kQueryElementsPerToken;
  if (reference.size() < begin + count || candidate.size() < begin + count) {
    return metrics;
  }

  long double absolute_sum = 0.0L;
  long double squared_error_sum = 0.0L;
  long double reference_squared_sum = 0.0L;
  long double candidate_squared_sum = 0.0L;
  long double dot_sum = 0.0L;
  for (std::size_t index = begin; index < begin + count; ++index) {
    const long double reference_value =
        static_cast<long double>(bf16_to_float(reference[index]));
    const long double candidate_value =
        static_cast<long double>(bf16_to_float(candidate[index]));
    if (!std::isfinite(reference_value) ||
        !std::isfinite(candidate_value)) {
      return metrics;
    }
    const long double difference = candidate_value - reference_value;
    const long double absolute = std::fabs(difference);
    metrics.max_abs = std::max(metrics.max_abs, absolute);
    absolute_sum += absolute;
    squared_error_sum += difference * difference;
    reference_squared_sum += reference_value * reference_value;
    candidate_squared_sum += candidate_value * candidate_value;
    dot_sum += reference_value * candidate_value;
    metrics.mismatch_count += reference[index] != candidate[index] ? 1U : 0U;
  }

  metrics.elements = count;
  metrics.mean_abs = absolute_sum / static_cast<long double>(count);
  const long double rmse =
      std::sqrt(squared_error_sum / static_cast<long double>(count));
  const long double reference_rms =
      std::sqrt(reference_squared_sum / static_cast<long double>(count));
  if (reference_rms == 0.0L) {
    if (rmse != 0.0L) {
      return metrics;
    }
    metrics.nrmse_reference_rms = 0.0L;
  } else {
    metrics.nrmse_reference_rms = rmse / reference_rms;
  }
  const long double cosine_denominator =
      std::sqrt(reference_squared_sum * candidate_squared_sum);
  if (cosine_denominator == 0.0L) {
    if (reference_squared_sum != 0.0L || candidate_squared_sum != 0.0L) {
      return metrics;
    }
    metrics.cosine = 1.0L;
  } else {
    metrics.cosine = dot_sum / cosine_denominator;
  }
  metrics.finite = std::isfinite(metrics.max_abs) &&
                   std::isfinite(metrics.mean_abs) &&
                   std::isfinite(metrics.nrmse_reference_rms) &&
                   std::isfinite(metrics.cosine);
  return metrics;
}

[[nodiscard]] bool hash_host_bf16_output(
    const std::vector<std::uint16_t>& output,
    core::Sha256Digest& digest) noexcept {
  if (output.size() != kLayer3AttentionOracleQueryElements) {
    return false;
  }
  core::Sha256 hash;
  if (!hash.update(output.data(), output.size() * sizeof(output.front()))) {
    return false;
  }
  digest = hash.finalize();
  return true;
}

[[nodiscard]] bool print_bf16_pair_numeric_metrics(
    const std::string_view pair,
    const std::vector<std::uint16_t>& reference,
    const std::vector<std::uint16_t>& candidate) {
  struct Segment {
    const char* name;
    std::size_t token_begin;
    std::size_t token_count;
  };
  constexpr std::array<Segment, 3U> kSegments{{
      {"tokens_0_512", 0U, 513U},
      {"tokens_0_256", 0U, 257U},
      {"tokens_257_512", 257U, 256U},
  }};
  for (const Segment& segment : kSegments) {
    const Bf16PairNumericMetrics metrics =
        compute_bf16_pair_numeric_metrics(reference, candidate,
                                          segment.token_begin,
                                          segment.token_count);
    if (!metrics.finite) {
      return false;
    }
    const std::ios_base::fmtflags previous_flags = std::cout.flags();
    const std::streamsize previous_precision = std::cout.precision();
    std::cout << "P513_ATTENTION_NUMERIC pair=" << pair
              << " segment=" << segment.name
              << " token_begin=" << segment.token_begin
              << " token_end_inclusive="
              << (segment.token_begin + segment.token_count - 1U)
              << " elements=" << metrics.elements
              << " bf16_mismatch_count=" << metrics.mismatch_count
              << std::scientific
              << std::setprecision(std::numeric_limits<long double>::max_digits10)
              << " max_abs=" << metrics.max_abs
              << " mean_abs=" << metrics.mean_abs
              << " nrmse_reference_rms=" << metrics.nrmse_reference_rms
              << " cosine=" << metrics.cosine << '\n';
    std::cout.flags(previous_flags);
    std::cout.precision(previous_precision);
  }
  return true;
}

void print_layer3_attention_first_mismatch(
    const std::string_view pair, const FirstBf16Mismatch mismatch,
    const Layer3AttentionP513OracleCollector& collector) {
  std::cout << "P513_ATTENTION_FIRST pair=" << pair;
  if (!mismatch.found) {
    std::cout << " mismatch=false\n";
    return;
  }
  const std::size_t token = mismatch.index / kQueryElementsPerToken;
  const std::size_t in_token = mismatch.index % kQueryElementsPerToken;
  const std::size_t head = in_token / kKvHeadDimension;
  const std::size_t dimension = in_token % kKvHeadDimension;
  std::cout << " mismatch=true token=" << token << " head=" << head
            << " dim=" << dimension
            << " canonical="
            << bf16_hex(collector.canonical_output[mismatch.index])
            << " group_q64="
            << bf16_hex(collector.group_q64_output[mismatch.index])
            << " flashinfer="
            << bf16_hex(collector.flashinfer_output[mismatch.index]) << '\n';
}

enum class Layer3AttentionOracleReportStatus : std::uint8_t {
  kInfrastructureFailure = 0,
  kBitwisePass,
  kBitwiseMismatch,
};

[[nodiscard]] Layer3AttentionOracleReportStatus
report_layer3_attention_p513_oracle(
    const Layer3AttentionP513OracleCollector& collector) {
  if (!collector.prepared || !collector.completed ||
      collector.hook_calls != 1U) {
    std::cerr << "P513_ATTENTION_ORACLE_ERROR prepared="
              << (collector.prepared ? "true" : "false")
              << " completed=" << (collector.completed ? "true" : "false")
              << " hook_calls=" << collector.hook_calls
              << " operation=" << collector.failed_operation
              << " cuda_error=" << collector.cuda_error << '\n';
    return Layer3AttentionOracleReportStatus::kInfrastructureFailure;
  }

  const bool query_immutable = collector.query_before == collector.query_after;
  const bool gate_immutable = collector.gate_before == collector.gate_after;
  const bool key_immutable = collector.key_before == collector.key_after;
  const bool value_immutable = collector.value_before == collector.value_after;
  const bool inputs_immutable = query_immutable && gate_immutable &&
                                key_immutable && value_immutable;

  if (!inputs_immutable) {
    std::cout << "P513_ATTENTION_ORACLE_SUMMARY layer=3 tokens=513"
                 " canonical_spans=257,256 capture_status=INVALID"
              << " query_immutable=" << (query_immutable ? "true" : "false")
              << " key_immutable=" << (key_immutable ? "true" : "false")
              << " value_immutable=" << (value_immutable ? "true" : "false")
              << " gate_immutable=" << (gate_immutable ? "true" : "false")
              << " flashinfer_bitwise_gate=NOT_RUN"
                 " accuracy_qualification=NOT_RUN\n";
    return Layer3AttentionOracleReportStatus::kInfrastructureFailure;
  }

  const FirstBf16Mismatch exact_group = first_bf16_mismatch(
      collector.canonical_output, collector.group_q64_output);
  const FirstBf16Mismatch exact_flashinfer = first_bf16_mismatch(
      collector.canonical_output, collector.flashinfer_output);
  const FirstBf16Mismatch group_flashinfer = first_bf16_mismatch(
      collector.group_q64_output, collector.flashinfer_output);
  print_layer3_attention_first_mismatch("canonical_vs_group_q64", exact_group,
                                        collector);
  print_layer3_attention_first_mismatch("canonical_vs_flashinfer",
                                        exact_flashinfer, collector);
  print_layer3_attention_first_mismatch("group_q64_vs_flashinfer",
                                        group_flashinfer, collector);

  core::Sha256Digest canonical_output_sha;
  core::Sha256Digest group_q64_output_sha;
  core::Sha256Digest flashinfer_output_sha;
  if (!hash_host_bf16_output(collector.canonical_output,
                             canonical_output_sha) ||
      !hash_host_bf16_output(collector.group_q64_output,
                             group_q64_output_sha) ||
      !hash_host_bf16_output(collector.flashinfer_output,
                             flashinfer_output_sha)) {
    std::cerr << "P513_ATTENTION_ORACLE_ERROR operation=output_sha256\n";
    return Layer3AttentionOracleReportStatus::kInfrastructureFailure;
  }
  std::cout << "P513_ATTENTION_OUTPUT_SHA256 canonical="
            << canonical_output_sha.hex()
            << " group_q64=" << group_q64_output_sha.hex()
            << " flashinfer=" << flashinfer_output_sha.hex() << '\n';
  std::cout << "P513_ATTENTION_NUMERIC_CONTRACT"
               " nrmse=rmse_over_reference_rms"
               " cosine=dot_over_l2_product"
               " accumulation=long_double"
               " bf16_decode=upper16_ieee754\n";
  if (!print_bf16_pair_numeric_metrics(
          "canonical_vs_group_q64", collector.canonical_output,
          collector.group_q64_output) ||
      !print_bf16_pair_numeric_metrics(
          "canonical_vs_flashinfer", collector.canonical_output,
          collector.flashinfer_output) ||
      !print_bf16_pair_numeric_metrics(
          "group_q64_vs_flashinfer", collector.group_q64_output,
          collector.flashinfer_output)) {
    std::cerr << "P513_ATTENTION_ORACLE_ERROR operation=numeric_metrics\n";
    return Layer3AttentionOracleReportStatus::kInfrastructureFailure;
  }

  std::size_t exact_group_total = 0U;
  std::size_t exact_flashinfer_total = 0U;
  std::size_t group_flashinfer_total = 0U;
  for (std::size_t token = 0U; token < kLayer3AttentionOracleTokens;
       ++token) {
    const std::size_t begin = token * kQueryElementsPerToken;
    const std::size_t end = begin + kQueryElementsPerToken;
    std::size_t exact_group_row = 0U;
    std::size_t exact_flashinfer_row = 0U;
    std::size_t group_flashinfer_row = 0U;
    for (std::size_t index = begin; index < end; ++index) {
      exact_group_row += collector.canonical_output[index] !=
                                 collector.group_q64_output[index]
                             ? 1U
                             : 0U;
      exact_flashinfer_row += collector.canonical_output[index] !=
                                      collector.flashinfer_output[index]
                                  ? 1U
                                  : 0U;
      group_flashinfer_row += collector.group_q64_output[index] !=
                                      collector.flashinfer_output[index]
                                  ? 1U
                                  : 0U;
    }
    exact_group_total += exact_group_row;
    exact_flashinfer_total += exact_flashinfer_row;
    group_flashinfer_total += group_flashinfer_row;
    std::cout << "P513_ATTENTION_ROW token=" << token
              << " canonical_vs_group_q64=" << exact_group_row
              << " canonical_vs_flashinfer=" << exact_flashinfer_row
              << " group_q64_vs_flashinfer=" << group_flashinfer_row << '\n';
  }

  std::cout << "P513_ATTENTION_ORACLE_SUMMARY layer=3 tokens=513"
               " canonical_spans=257,256"
               " capture_status=VALID"
            << " query_immutable=" << (query_immutable ? "true" : "false")
            << " key_immutable=" << (key_immutable ? "true" : "false")
            << " value_immutable=" << (value_immutable ? "true" : "false")
            << " gate_immutable=" << (gate_immutable ? "true" : "false")
            << " canonical_vs_group_q64=" << exact_group_total
            << " canonical_vs_flashinfer=" << exact_flashinfer_total
            << " group_q64_vs_flashinfer=" << group_flashinfer_total
            << " flashinfer_bitwise_gate="
            << (exact_flashinfer_total == 0U ? "PASS" : "FAIL")
            << " accuracy_qualification=NOT_RUN"
            << '\n';
  return exact_flashinfer_total == 0U
             ? Layer3AttentionOracleReportStatus::kBitwisePass
             : Layer3AttentionOracleReportStatus::kBitwiseMismatch;
}

enum class SnapshotError : std::uint8_t {
  kNone = 0,
  kDuplicateHook,
  kInvalidState,
  kUnexpectedSequenceLength,
  kInvalidPersistentRegion,
  kInvalidKvRegion,
  kInvalidPromptResidual,
  kInvalidFinalHidden,
  kHashFailure,
};

[[nodiscard]] const char* to_string(const SnapshotError error) noexcept {
  switch (error) {
    case SnapshotError::kNone:
      return "none";
    case SnapshotError::kDuplicateHook:
      return "duplicate_hook";
    case SnapshotError::kInvalidState:
      return "invalid_state";
    case SnapshotError::kUnexpectedSequenceLength:
      return "unexpected_sequence_length";
    case SnapshotError::kInvalidPersistentRegion:
      return "invalid_persistent_region";
    case SnapshotError::kInvalidKvRegion:
      return "invalid_kv_region";
    case SnapshotError::kInvalidPromptResidual:
      return "invalid_prompt_residual";
    case SnapshotError::kInvalidFinalHidden:
      return "invalid_final_hidden";
    case SnapshotError::kHashFailure:
      return "hash_failure";
  }
  return "unknown";
}

struct StateSnapshot {
  explicit StateSnapshot(const std::size_t prompt_tokens)
      : expected_prompt_tokens(prompt_tokens),
        copy_scratch(kHashCopyChunkBytes) {}

  std::size_t expected_prompt_tokens = 0U;
  std::size_t hook_calls = 0U;
  std::uint32_t sequence_length = 0U;
  SnapshotError error = SnapshotError::kNone;
  int cuda_error = static_cast<int>(cudaSuccess);
  core::Sha256Digest conv_state;
  core::Sha256Digest gdn_state;
  std::array<core::Sha256Digest, runtime::kRequestFullLayerCount> key_cache;
  std::array<core::Sha256Digest, runtime::kRequestFullLayerCount> value_cache;
  core::Sha256Digest prompt_residual;
  core::Sha256Digest final_hidden;
  core::Sha256Digest aggregate;
  std::vector<std::uint8_t> copy_scratch;
};

[[nodiscard]] bool region_fits(const runtime::RequestRegion& region,
                               const std::uint64_t arena_bytes) noexcept {
  return region.byte_size != 0U && region.element_size_bytes != 0U &&
         region.arena_offset <= arena_bytes &&
         region.byte_size <= arena_bytes - region.arena_offset;
}

[[nodiscard]] bool hash_device_bytes(
    const std::uint8_t* const source, const std::size_t bytes,
    std::vector<std::uint8_t>& scratch, core::Sha256Digest& digest,
    int& cuda_error) noexcept {
  if (source == nullptr || bytes == 0U || scratch.empty()) {
    return false;
  }
  core::Sha256 hash;
  std::size_t offset = 0U;
  while (offset < bytes) {
    const std::size_t chunk =
        std::min(scratch.size(), bytes - offset);
    const cudaError_t status = cudaMemcpy(
        scratch.data(), source + offset, chunk, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
      cuda_error = static_cast<int>(status);
      return false;
    }
    if (!hash.update(scratch.data(), chunk)) {
      return false;
    }
    offset += chunk;
  }
  digest = hash.finalize();
  return true;
}

[[nodiscard]] bool hash_arena_region(
    const runtime::RequestState& state,
    const runtime::RequestRegion& region, const std::size_t bytes,
    StateSnapshot& snapshot, core::Sha256Digest& digest) noexcept {
  if (!region_fits(region, state.arena_bytes()) || bytes == 0U ||
      bytes > region.byte_size ||
      region.arena_offset >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
      bytes > std::numeric_limits<std::size_t>::max() -
                  static_cast<std::size_t>(region.arena_offset)) {
    return false;
  }
  const auto* const arena =
      static_cast<const std::uint8_t*>(state.arena_data());
  return hash_device_bytes(
      arena + static_cast<std::size_t>(region.arena_offset), bytes,
      snapshot.copy_scratch, digest, snapshot.cuda_error);
}

void collect_generate_return_snapshot(
    const runtime::RequestState& state, void* const context) noexcept {
  auto* const snapshot = static_cast<StateSnapshot*>(context);
  if (snapshot == nullptr) {
    return;
  }
  ++snapshot->hook_calls;
  if (snapshot->hook_calls != 1U) {
    snapshot->error = SnapshotError::kDuplicateHook;
    return;
  }
  if (!state || state.memory_profile() !=
                    runtime::RequestMemoryProfile::kLayerMajorC8192 ||
      state.arena_data() == nullptr || state.layer_major_plan() == nullptr ||
      snapshot->copy_scratch.empty()) {
    snapshot->error = SnapshotError::kInvalidState;
    return;
  }
  const cudaError_t synchronize_status = cudaDeviceSynchronize();
  if (synchronize_status != cudaSuccess) {
    snapshot->cuda_error = static_cast<int>(synchronize_status);
    snapshot->error = SnapshotError::kHashFailure;
    return;
  }

  snapshot->sequence_length = state.sequence_length();
  if (snapshot->sequence_length != snapshot->expected_prompt_tokens ||
      snapshot->expected_prompt_tokens > state.max_sequence_length()) {
    snapshot->error = SnapshotError::kUnexpectedSequenceLength;
    return;
  }

  const runtime::RequestMemoryPlan& common = state.plan();
  if (common.conv_state.byte_size != runtime::kRequestConvStateBytes ||
      common.gdn_state.byte_size != runtime::kRequestGdnStateBytes ||
      !hash_arena_region(state, common.conv_state,
                         static_cast<std::size_t>(common.conv_state.byte_size),
                         *snapshot, snapshot->conv_state) ||
      !hash_arena_region(state, common.gdn_state,
                         static_cast<std::size_t>(common.gdn_state.byte_size),
                         *snapshot, snapshot->gdn_state)) {
    snapshot->error = snapshot->cuda_error == static_cast<int>(cudaSuccess)
                          ? SnapshotError::kInvalidPersistentRegion
                          : SnapshotError::kHashFailure;
    return;
  }

  constexpr std::size_t kKvBytesPerLayerPosition =
      kKvHeadCount * kKvHeadDimension * kBf16Bytes;
  if (snapshot->expected_prompt_tokens >
      std::numeric_limits<std::size_t>::max() /
          kKvBytesPerLayerPosition) {
    snapshot->error = SnapshotError::kInvalidKvRegion;
    return;
  }
  const std::size_t used_kv_bytes =
      snapshot->expected_prompt_tokens * kKvBytesPerLayerPosition;
  for (std::size_t slot = 0U; slot < runtime::kRequestFullLayerCount;
       ++slot) {
    const runtime::RequestRegion& key = common.key_cache[slot];
    const runtime::RequestRegion& value = common.value_cache[slot];
    const bool valid_shape =
        key.element_size_bytes == kBf16Bytes &&
        value.element_size_bytes == kBf16Bytes &&
        key.byte_size == value.byte_size && used_kv_bytes <= key.byte_size;
    if (!valid_shape ||
        !hash_arena_region(state, key, used_kv_bytes, *snapshot,
                           snapshot->key_cache[slot]) ||
        !hash_arena_region(state, value, used_kv_bytes, *snapshot,
                           snapshot->value_cache[slot])) {
      snapshot->error = snapshot->cuda_error == static_cast<int>(cudaSuccess)
                            ? SnapshotError::kInvalidKvRegion
                            : SnapshotError::kHashFailure;
      return;
    }
  }

  const runtime::LayerMajorRequestMemoryPlan& layer_major =
      *state.layer_major_plan();
  const runtime::RequestMatrixRegion& prompt =
      layer_major.prompt_residual_bf16;
  if (prompt.storage.element_size_bytes != kBf16Bytes ||
      prompt.columns != runtime::kReferenceHiddenSize ||
      prompt.row_stride_elements != runtime::kReferenceHiddenSize ||
      prompt.row_capacity < snapshot->expected_prompt_tokens ||
      snapshot->expected_prompt_tokens >
          std::numeric_limits<std::size_t>::max() /
              (runtime::kReferenceHiddenSize * kBf16Bytes)) {
    snapshot->error = SnapshotError::kInvalidPromptResidual;
    return;
  }
  const std::size_t prompt_bytes =
      snapshot->expected_prompt_tokens * runtime::kReferenceHiddenSize *
      kBf16Bytes;
  if (!hash_arena_region(state, prompt.storage, prompt_bytes, *snapshot,
                         snapshot->prompt_residual)) {
    snapshot->error = snapshot->cuda_error == static_cast<int>(cudaSuccess)
                          ? SnapshotError::kInvalidPromptResidual
                          : SnapshotError::kHashFailure;
    return;
  }

  const runtime::RequestMatrixRegion& final_hidden =
      layer_major.final_hidden_bf16;
  if (final_hidden.storage.element_size_bytes != kBf16Bytes ||
      final_hidden.row_capacity != 1U ||
      final_hidden.columns != runtime::kReferenceHiddenSize ||
      final_hidden.row_stride_elements != runtime::kReferenceHiddenSize ||
      !hash_arena_region(
          state, final_hidden.storage,
          static_cast<std::size_t>(final_hidden.storage.byte_size),
          *snapshot, snapshot->final_hidden)) {
    snapshot->error = snapshot->cuda_error == static_cast<int>(cudaSuccess)
                          ? SnapshotError::kInvalidFinalHidden
                          : SnapshotError::kHashFailure;
    return;
  }

  core::Sha256 aggregate;
  const auto add_digest = [&aggregate](
                              const core::Sha256Digest& digest) noexcept {
    return aggregate.update(digest.bytes.data(), digest.bytes.size());
  };
  bool complete = add_digest(snapshot->conv_state) &&
                  add_digest(snapshot->gdn_state);
  for (std::size_t slot = 0U; slot < runtime::kRequestFullLayerCount;
       ++slot) {
    complete = complete && add_digest(snapshot->key_cache[slot]) &&
               add_digest(snapshot->value_cache[slot]);
  }
  complete = complete && add_digest(snapshot->prompt_residual) &&
             add_digest(snapshot->final_hidden);
  if (!complete) {
    snapshot->error = SnapshotError::kHashFailure;
    return;
  }
  snapshot->aggregate = aggregate.finalize();
}

class ScopedGenerateReturnSnapshot final {
 public:
  explicit ScopedGenerateReturnSnapshot(StateSnapshot& snapshot) noexcept
      : previous_(
            runner_detail::exchange_reference_engine_generate_return_snapshot_hook(
                {collect_generate_return_snapshot, &snapshot})) {}

  ~ScopedGenerateReturnSnapshot() {
    (void)runner_detail::
        exchange_reference_engine_generate_return_snapshot_hook(previous_);
  }

  ScopedGenerateReturnSnapshot(const ScopedGenerateReturnSnapshot&) = delete;
  ScopedGenerateReturnSnapshot& operator=(
      const ScopedGenerateReturnSnapshot&) = delete;

 private:
  runner_detail::ReferenceEngineGenerateReturnSnapshotHook previous_{};
};

class ScopedCompatibilityOracle final {
 public:
  explicit ScopedCompatibilityOracle(const bool enabled) noexcept
      : previous_(engine_detail::
                      exchange_reference_engine_prefill_compatibility_oracle_for_test(
                          enabled)) {}

  ~ScopedCompatibilityOracle() {
    (void)engine_detail::
        exchange_reference_engine_prefill_compatibility_oracle_for_test(
            previous_);
  }

  ScopedCompatibilityOracle(const ScopedCompatibilityOracle&) = delete;
  ScopedCompatibilityOracle& operator=(const ScopedCompatibilityOracle&) =
      delete;

 private:
  bool previous_ = false;
};

[[nodiscard]] std::string model_directory_from(
    const int argc, char** const argv) {
  if (argc >= 2 && argv[1] != nullptr && argv[1][0] != '\0' &&
      std::string_view(argv[1]) != "-") {
    return argv[1];
  }
  const char* const environment = std::getenv("Q3X_E2E_MODEL_DIR");
  return environment == nullptr ? std::string{} : std::string(environment);
}

[[nodiscard]] std::string repeated_hello_prompt(
    const std::size_t prompt_tokens) {
  if (prompt_tokens < kPromptTemplateTokens) {
    return {};
  }
  const std::size_t words = prompt_tokens - kPromptTemplateTokens;
  std::string prompt;
  prompt.reserve(words * 6U);
  for (std::size_t index = 0U; index < words; ++index) {
    if (index != 0U) {
      prompt.push_back(' ');
    }
    prompt.append("hello");
  }
  return prompt;
}

void print_diagnostic(
    const runtime::ReferenceEngineDiagnostic& diagnostic) {
  std::cerr << "code=" << runtime::to_string(diagnostic.code)
            << " stage=" << diagnostic.stage
            << " message=" << diagnostic.message
            << " context=" << diagnostic.context
            << " cuda_error=" << diagnostic.cuda_error
            << " layer=" << diagnostic.layer
            << " operation=" << diagnostic.operation << '\n';
}

[[nodiscard]] bool same_route(
    const runtime::PrefillRouteEvidence& left,
    const runtime::PrefillRouteEvidence& right) noexcept {
  if (left.completed_layer_passes != right.completed_layer_passes ||
      left.expected_layer_passes != right.expected_layer_passes ||
      left.request_active != right.request_active ||
      left.complete != right.complete || left.valid != right.valid ||
      left.error != right.error ||
      left.forbidden_boundary_hits != right.forbidden_boundary_hits) {
    return false;
  }
  for (std::size_t role = 0U; role < runtime::kPrefillOperatorRoleCount;
       ++role) {
    const runtime::PrefillOperatorRouteCounts& a = left.operators[role];
    const runtime::PrefillOperatorRouteCounts& b = right.operators[role];
    if (a.production_hits != b.production_hits ||
        a.exact_fallback_hits != b.exact_fallback_hits ||
        a.forbidden_hits != b.forbidden_hits) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool valid_completed_native_route(
    const runtime::PrefillRouteEvidence& evidence,
    const std::uint64_t logical_panels) noexcept {
  if (!evidence.complete || !evidence.valid || evidence.request_active ||
      evidence.error != runtime::PrefillRouteEvidenceError::kNone ||
      evidence.completed_layer_passes != logical_panels ||
      evidence.expected_layer_passes != logical_panels) {
    return false;
  }
  for (const std::uint64_t count : evidence.forbidden_boundary_hits) {
    if (count != 0U) {
      return false;
    }
  }
  for (std::size_t role = 0U; role < runtime::kPrefillOperatorRoleCount;
       ++role) {
    const runtime::PrefillOperatorRouteCounts& counts =
        evidence.operators[role];
    if (counts.production_hits !=
            runtime::kExpectedPrefillLogicalOperatorsPerTile[role] *
                logical_panels ||
        counts.exact_fallback_hits != 0U || counts.forbidden_hits != 0U) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool valid_snapshot(const StateSnapshot& snapshot) noexcept {
  return snapshot.hook_calls == 1U &&
         snapshot.sequence_length == snapshot.expected_prompt_tokens &&
         snapshot.error == SnapshotError::kNone &&
         snapshot.cuda_error == static_cast<int>(cudaSuccess);
}

[[nodiscard]] bool same_snapshots(const StateSnapshot& oracle,
                                  const StateSnapshot& panel) {
  bool exact = oracle.conv_state == panel.conv_state &&
               oracle.gdn_state == panel.gdn_state &&
               oracle.prompt_residual == panel.prompt_residual &&
               oracle.final_hidden == panel.final_hidden &&
               oracle.aggregate == panel.aggregate;
  for (std::size_t slot = 0U; slot < runtime::kRequestFullLayerCount;
       ++slot) {
    exact = exact && oracle.key_cache[slot] == panel.key_cache[slot] &&
            oracle.value_cache[slot] == panel.value_cache[slot];
  }
  if (!exact) {
    const auto report = [](const std::string_view region,
                           const core::Sha256Digest& expected,
                           const core::Sha256Digest& actual) {
      if (!(expected == actual)) {
        std::cerr << "mismatch region=" << region
                  << " oracle_sha256=" << expected.hex()
                  << " panel_sha256=" << actual.hex() << '\n';
      }
    };
    report("conv_state", oracle.conv_state, panel.conv_state);
    report("gdn_state", oracle.gdn_state, panel.gdn_state);
    report("prompt_residual", oracle.prompt_residual,
           panel.prompt_residual);
    report("final_hidden", oracle.final_hidden, panel.final_hidden);
    for (std::size_t slot = 0U; slot < runtime::kRequestFullLayerCount;
         ++slot) {
      const std::size_t layer = 3U + 4U * slot;
      report("key_cache_layer_" + std::to_string(layer),
             oracle.key_cache[slot], panel.key_cache[slot]);
      report("value_cache_layer_" + std::to_string(layer),
             oracle.value_cache[slot], panel.value_cache[slot]);
    }
  }
  return exact;
}

[[nodiscard]] bool run_generation(
    runtime::ReferenceEngine& engine, const std::string& prompt,
    const std::size_t prompt_tokens, const bool compatibility_oracle,
    StateSnapshot& snapshot, runtime::ReferenceGenerateResult& result) {
  runtime::ReferenceGenerateOptions options;
  options.max_new_tokens = 1U;
  options.prefill_chunk_size = runtime::kMaximumRequestPrefillChunkSize;
  options.logits_mode = runtime::ReferenceLogitsMode::kPredictedTokenOnly;
  options.prefill_execution_mode =
      runtime::ReferencePrefillExecutionMode::kWholeRequestLayerMajor;
  {
    const ScopedGenerateReturnSnapshot snapshot_hook(snapshot);
    const ScopedCompatibilityOracle oracle_route(compatibility_oracle);
    result = engine.generate(prompt, options);
  }
  if (!result) {
    std::cerr << (compatibility_oracle ? "compatibility oracle" : "panel")
              << " generation failed for P" << prompt_tokens << ": ";
    print_diagnostic(result.diagnostic);
    return false;
  }
  if (!valid_snapshot(snapshot)) {
    std::cerr << (compatibility_oracle ? "compatibility oracle" : "panel")
              << " snapshot failed for P" << prompt_tokens
              << " calls=" << snapshot.hook_calls
              << " sequence_length=" << snapshot.sequence_length
              << " snapshot_error=" << to_string(snapshot.error)
              << " cuda_error=" << snapshot.cuda_error << '\n';
    return false;
  }
  return true;
}

[[nodiscard]] bool run_case(runtime::ReferenceEngine& engine,
                            const std::size_t prompt_tokens,
                            const runtime::LayerMajorPrefillFullAttentionTactic
                                attention_tactic,
                            const runtime::LayerMajorPrefillProjectionTactic
                                projection_tactic) {
  const bool native_group_q64_panel =
      attention_tactic == runtime::LayerMajorPrefillFullAttentionTactic::
                              kNativeGroupQ64Panel;
  const bool native_group_q128_v4_panel =
      attention_tactic == runtime::LayerMajorPrefillFullAttentionTactic::
                              kNativeGroupQ128V4Panel;
  const bool native_flashinfer_exact_panel =
      attention_tactic == runtime::LayerMajorPrefillFullAttentionTactic::
                              kNativeFlashInferExactPanel;
  const bool native_group_panel =
      native_group_q64_panel || native_group_q128_v4_panel;
  const bool native_panel_attention =
      native_group_panel || native_flashinfer_exact_panel;
  const bool segmented_marlin_operator_panel =
      projection_tactic == runtime::LayerMajorPrefillProjectionTactic::
                               kSegmentedMarlinOperatorPanel;
  const bool native_large_m_operator_panel =
      projection_tactic == runtime::LayerMajorPrefillProjectionTactic::
                               kNativeQuantizedLargeMOperatorPanel;
  const std::size_t expected_logical_panels =
      (prompt_tokens + runtime::kLayerMajorPrefillOperatorPanelTokens - 1U) /
      runtime::kLayerMajorPrefillOperatorPanelTokens;
  runtime::PrefillExecutionPlanOptions topology_options;
  topology_options.prompt_token_count = prompt_tokens;
  const runtime::PrefillExecutionPlanResult topology =
      runtime::build_unbound_layer_major_prefill_execution_plan(
          topology_options);
  bool topology_exact =
      topology && topology.value->panel_count == expected_logical_panels;
  if (topology_exact) {
    std::uint32_t expected_first_position = 0U;
    for (std::size_t panel_index = 0U;
         panel_index < topology.value->panel_count; ++panel_index) {
      const runtime::PrefillOperatorPanel& panel =
          topology.value->panels[panel_index];
      topology_exact =
          topology_exact && panel.first_position == expected_first_position &&
          panel.token_count >= runtime::kPrefillPhysicalSegmentM32Tokens &&
          panel.end_position == panel.first_position + panel.token_count;
      expected_first_position = panel.end_position;
    }
    topology_exact =
        topology_exact && expected_first_position == prompt_tokens;
  }
  if (topology_exact && prompt_tokens == 8'192U) {
    topology_exact = topology_exact && topology.value->panel_count == 1U &&
                     topology.value->panels[0].token_count == 8'192U;
  } else if (topology_exact && prompt_tokens == 8'193U) {
    topology_exact = topology_exact && topology.value->panel_count == 2U &&
                     topology.value->panels[0].token_count == 4'097U &&
                     topology.value->panels[1].token_count == 4'096U;
  }
  if (!topology_exact) {
    std::cerr << "operator-panel topology gate failed for P"
              << prompt_tokens << '\n';
    return false;
  }

  const std::string prompt = repeated_hello_prompt(prompt_tokens);
  StateSnapshot oracle_snapshot(prompt_tokens);
  StateSnapshot panel_snapshot(prompt_tokens);
  runtime::ReferenceGenerateResult oracle_result;
  runtime::ReferenceGenerateResult panel_result;
  if (!run_generation(engine, prompt, prompt_tokens, true, oracle_snapshot,
                      oracle_result) ||
      !run_generation(engine, prompt, prompt_tokens, false, panel_snapshot,
                      panel_result)) {
    return false;
  }

  const runtime::ReferenceGeneration& oracle = *oracle_result.value;
  const runtime::ReferenceGeneration& panel = *panel_result.value;
  const bool architecture_candidate =
      native_panel_attention || segmented_marlin_operator_panel ||
      native_large_m_operator_panel;
  std::size_t expected_segmented_projection_physical_launches = 0U;
  if (segmented_marlin_operator_panel) {
    for (std::size_t panel_index = 0U;
         panel_index < topology.value->panel_count; ++panel_index) {
      const std::size_t panel_tokens =
          topology.value->panels[panel_index].token_count;
      const std::size_t nvfp4_launches =
          q3x::kernels::sm87_nvfp4_marlin_execution_plan(panel_tokens)
              .launch_count;
      const std::size_t fp8_large_n_launches =
          q3x::kernels::sm87_fp8_marlin_execution_plan(panel_tokens, 5'120U)
              .launch_count;
      const std::size_t fp8_small_n_launches =
          q3x::kernels::sm87_fp8_marlin_execution_plan(panel_tokens, 1'024U)
              .launch_count;
      expected_segmented_projection_physical_launches +=
          128U * nvfp4_launches + 176U * fp8_large_n_launches +
          32U * fp8_small_n_launches;
    }
  }
  std::size_t expected_native_large_m_projection_physical_launches = 0U;
  std::size_t expected_native_large_m_projection_bulk_hits = 0U;
  std::size_t expected_native_large_m_projection_oracle_partial_hits = 0U;
  if (native_large_m_operator_panel) {
    for (std::size_t panel_index = 0U;
         panel_index < topology.value->panel_count; ++panel_index) {
      const std::size_t panel_tokens =
          topology.value->panels[panel_index].token_count;
      if (panel_tokens ==
          runtime::kLayerMajorPrefillOperatorPanelTokens) {
        expected_native_large_m_projection_bulk_hits += 336U;
      } else {
        expected_native_large_m_projection_oracle_partial_hits += 336U;
      }
      std::size_t nvfp4_launches = 1U;
      std::size_t fp8_large_n_launches = 1U;
      std::size_t fp8_small_n_launches = 1U;
      if (panel_tokens !=
          runtime::kLayerMajorPrefillOperatorPanelTokens) {
        nvfp4_launches = 0U;
        fp8_large_n_launches = 0U;
        fp8_small_n_launches = 0U;
        const runtime::LayerMajorPrefillArithmeticSpanLedger ledger =
            runtime::make_layer_major_prefill_arithmetic_span_ledger(
                panel_tokens);
        for (std::size_t span_index = 0U;
             span_index < ledger.span_count; ++span_index) {
          const std::size_t span_tokens = ledger.spans[span_index].token_count;
          nvfp4_launches +=
              q3x::kernels::sm87_nvfp4_marlin_execution_plan(span_tokens)
                  .launch_count;
          fp8_large_n_launches +=
              q3x::kernels::sm87_fp8_marlin_execution_plan(span_tokens,
                                                            5'120U)
                  .launch_count;
          fp8_small_n_launches +=
              q3x::kernels::sm87_fp8_marlin_execution_plan(span_tokens,
                                                            1'024U)
                  .launch_count;
        }
      }
      // Gate+Up is one merged Marlin projection. Across 64 layers there are
      // 208 FP8 projections plus 128 NVFP4 Gate+Up/Down projections.
      expected_native_large_m_projection_physical_launches +=
          128U * nvfp4_launches + 176U * fp8_large_n_launches +
          32U * fp8_small_n_launches;
    }
  }
  const std::string_view expected_deployment_plan_id =
      native_large_m_operator_panel
          ? (native_flashinfer_exact_panel
                 ? runtime::
                       kLayerMajorNativeQuantizedLargeMProjectionFlashInferExactDeploymentPlanId
             : native_group_q128_v4_panel
                 ? runtime::
                       kLayerMajorNativeQuantizedLargeMProjectionGroupQ128V4DeploymentPlanId
             : native_group_q64_panel
                 ? runtime::
                       kLayerMajorNativeQuantizedLargeMProjectionGroupQ64DeploymentPlanId
                 : runtime::
                       kLayerMajorNativeQuantizedLargeMProjectionDeploymentPlanId)
      : segmented_marlin_operator_panel
          ? (native_flashinfer_exact_panel
                 ? runtime::
                       kLayerMajorSegmentedMarlinProjectionFlashInferExactDeploymentPlanId
             : native_group_q128_v4_panel
                 ? runtime::
                       kLayerMajorSegmentedMarlinProjectionGroupQ128V4DeploymentPlanId
             : native_group_q64_panel
                 ? runtime::
                       kLayerMajorSegmentedMarlinProjectionGroupQ64DeploymentPlanId
                 : runtime::
                       kLayerMajorSegmentedMarlinProjectionDeploymentPlanId)
          : (native_flashinfer_exact_panel
                 ? runtime::kLayerMajorNativeFlashInferExactPanelDeploymentPlanId
             : native_group_q128_v4_panel
                 ? runtime::kLayerMajorNativeGroupQ128V4PanelDeploymentPlanId
             : native_group_q64_panel
                 ? runtime::kLayerMajorNativeGroupQ64PanelDeploymentPlanId
                 : runtime::kLayerMajorOperatorPanelDeploymentPlanId);
  const bool output_exact =
      oracle.prompt_token_ids.size() == prompt_tokens &&
      panel.prompt_token_ids == oracle.prompt_token_ids &&
      panel.generated_token_ids == oracle.generated_token_ids &&
      panel.generated_text == oracle.generated_text &&
      panel.stop_reason == oracle.stop_reason;
  const bool route_exact = same_route(oracle.prefill_route_evidence,
                                      panel.prefill_route_evidence);
  const bool route_contract =
      oracle.prefill_execution_mode ==
          runtime::ReferencePrefillExecutionMode::kWholeRequestLayerMajor &&
      panel.prefill_execution_mode == oracle.prefill_execution_mode &&
      oracle.prefill_deployment_plan_id == expected_deployment_plan_id &&
      panel.prefill_deployment_plan_id ==
          oracle.prefill_deployment_plan_id &&
      oracle.prefill_logical_panel_count == expected_logical_panels &&
      panel.prefill_logical_panel_count ==
          oracle.prefill_logical_panel_count &&
      valid_completed_native_route(oracle.prefill_route_evidence,
                                   expected_logical_panels) &&
      valid_completed_native_route(panel.prefill_route_evidence,
                                   expected_logical_panels) &&
      panel.prefill_operator_panel_executor_hits ==
          runtime::kReferenceDecoderLayerCount * expected_logical_panels &&
      panel.prefill_native_group_q64_panel_hits ==
          (native_group_q64_panel
               ? runtime::kRequestFullLayerCount * expected_logical_panels
               : 0U) &&
      panel.prefill_native_group_q128_v4_panel_hits ==
          (native_group_q128_v4_panel
               ? runtime::kRequestFullLayerCount * expected_logical_panels
               : 0U) &&
      panel.prefill_native_flashinfer_exact_panel_hits ==
          (native_flashinfer_exact_panel
               ? runtime::kRequestFullLayerCount * expected_logical_panels
               : 0U) &&
      (!native_panel_attention || panel.prefill_generic_qt2_hits == 0U) &&
      panel.prefill_segmented_panel_projection_hits ==
          (segmented_marlin_operator_panel
               ? 336U * expected_logical_panels
               : 0U) &&
      panel.prefill_segmented_panel_projection_physical_launches ==
          expected_segmented_projection_physical_launches &&
      panel.prefill_native_large_m_projection_hits ==
          (native_large_m_operator_panel
               ? 336U * expected_logical_panels
               : 0U) &&
      panel.prefill_native_large_m_projection_bulk_hits ==
          expected_native_large_m_projection_bulk_hits &&
      panel.prefill_native_large_m_projection_oracle_partial_hits ==
          expected_native_large_m_projection_oracle_partial_hits &&
      panel.prefill_native_large_m_projection_hits ==
          panel.prefill_native_large_m_projection_bulk_hits +
              panel.prefill_native_large_m_projection_oracle_partial_hits &&
      panel.prefill_native_large_m_projection_physical_launches ==
          expected_native_large_m_projection_physical_launches &&
      oracle.prefill_operator_panel_executor_hits == 0U &&
      oracle.prefill_native_group_q64_panel_hits == 0U &&
      oracle.prefill_native_group_q128_v4_panel_hits == 0U &&
      oracle.prefill_native_flashinfer_exact_panel_hits == 0U &&
      oracle.prefill_segmented_panel_projection_hits == 0U &&
      oracle.prefill_segmented_panel_projection_physical_launches == 0U &&
      oracle.prefill_native_large_m_projection_hits == 0U &&
      oracle.prefill_native_large_m_projection_bulk_hits == 0U &&
      oracle.prefill_native_large_m_projection_oracle_partial_hits == 0U &&
      oracle.prefill_native_large_m_projection_physical_launches == 0U;
  const bool state_exact =
      same_snapshots(oracle_snapshot, panel_snapshot);
  const bool direction_passed =
      output_exact && route_exact && route_contract;
  const bool exact_equivalence_passed = direction_passed && state_exact;
  // A projection-only candidate changes wrapper segmentation but not the
  // admitted arithmetic, recurrent state, or Attention tactic, so it must
  // retain the complete real-model state oracle. Native grouped and
  // FlashInfer panel Attention remain explicitly accuracy-unqualified and are
  // screened only for output and sealed-route integrity here.
  const bool candidate_passed =
      direction_passed && (native_panel_attention || state_exact);
  const bool passed = architecture_candidate ? candidate_passed
                                              : exact_equivalence_passed;
  std::cout << (architecture_candidate
                    ? "PREFILL_OPERATOR_PANEL_DIRECTION_SCREEN"
                    : "PREFILL_OPERATOR_PANEL_EQUIVALENCE")
            << " prompt_tokens=" << prompt_tokens
            << " logical_panels=" << expected_logical_panels
            << " oracle_state_sha256=" << oracle_snapshot.aggregate.hex()
            << " panel_state_sha256=" << panel_snapshot.aggregate.hex()
            << " generated_token="
            << (panel.generated_token_ids.empty()
                    ? runtime::kReferenceVocabularySize
                    : panel.generated_token_ids.front())
            << " generated_text=" << panel.generated_text
            << " output_exact=" << (output_exact ? "true" : "false")
            << " route_exact=" << (route_exact ? "true" : "false")
            << " route_contract=" << (route_contract ? "true" : "false")
            << " state_exact=" << (state_exact ? "true" : "false")
            << " attention_tactic="
            << runtime::to_string(attention_tactic)
            << " projection_tactic="
            << runtime::to_string(projection_tactic)
            << " operator_panel_executor_hits="
            << panel.prefill_operator_panel_executor_hits
            << " native_group_q64_panel_hits="
            << panel.prefill_native_group_q64_panel_hits
            << " native_group_q128_v4_panel_hits="
            << panel.prefill_native_group_q128_v4_panel_hits
            << " native_flashinfer_exact_panel_hits="
            << panel.prefill_native_flashinfer_exact_panel_hits
            << " generic_qt2_hits=" << panel.prefill_generic_qt2_hits
            << " segmented_panel_projection_hits="
            << panel.prefill_segmented_panel_projection_hits
            << " segmented_panel_projection_physical_launches="
            << panel.prefill_segmented_panel_projection_physical_launches
            << " native_large_m_projection_hits="
            << panel.prefill_native_large_m_projection_hits
            << " native_large_m_projection_bulk_hits="
            << panel.prefill_native_large_m_projection_bulk_hits
            << " native_large_m_projection_oracle_partial_hits="
            << panel.prefill_native_large_m_projection_oracle_partial_hits
            << " native_large_m_projection_physical_launches="
            << panel.prefill_native_large_m_projection_physical_launches
            << (architecture_candidate
                    ? (candidate_passed
                           ? " direction_gate=PASS accuracy_gate=NOT_RUN"
                           : " direction_gate=FAIL accuracy_gate=NOT_RUN")
                    : (exact_equivalence_passed
                           ? " equivalence_gate=PASS"
                           : " equivalence_gate=FAIL"))
            << (architecture_candidate
                    ? " qualification=ACCURACY_UNQUALIFIED"
                      " authority=REAL_MODEL_DIRECTION_SCREEN_ONLY\n"
                    : " qualification=BITWISE_EXACT"
                      " authority=REAL_MODEL_CORRECTNESS_ONLY\n");
  return passed;
}

}  // namespace

int main(const int argc, char** const argv) {
  if (argc > 2) {
    std::cerr << "usage: q3x_reference_prefill_operator_panel_"
                 "equivalence_e2e_test [MODEL_DIR|-]\n";
    return 2;
  }
  const std::string model_directory = model_directory_from(argc, argv);
  if (model_directory.empty()) {
    std::cout << "SKIP: set Q3X_E2E_MODEL_DIR to the pinned model directory\n";
    return 77;
  }
  const char* const enabled =
      std::getenv("Q3X_RUN_PREFILL_OPERATOR_PANEL_EQUIVALENCE");
  if (enabled == nullptr || std::string_view(enabled) != "1") {
    std::cout << "SKIP: set Q3X_RUN_PREFILL_OPERATOR_PANEL_EQUIVALENCE=1 "
                 "after a clean-host GPU preflight\n";
    return 77;
  }
  const char* const layer3_attention_oracle_value =
      std::getenv("Q3X_RUN_PREFILL_ATTENTION_LAYER3_P513_ORACLE");
  const bool layer3_attention_oracle =
      layer3_attention_oracle_value != nullptr &&
      std::string_view(layer3_attention_oracle_value) == "1";
  const char* const attention_tactic =
      std::getenv("Q3X_TEST_PREFILL_ATTENTION_TACTIC");
  const bool native_group_q64_panel =
      attention_tactic != nullptr &&
      std::string_view(attention_tactic) == "native-group-q64-panel";
  const bool native_group_q128_v4_panel =
      attention_tactic != nullptr &&
      std::string_view(attention_tactic) == "native-group-q128-v4-panel";
  const bool native_flashinfer_exact_panel =
      attention_tactic != nullptr &&
      std::string_view(attention_tactic) ==
          "native-flashinfer-exact-panel";
  if (attention_tactic != nullptr && !native_group_q64_panel &&
      !native_group_q128_v4_panel &&
      !native_flashinfer_exact_panel &&
      std::string_view(attention_tactic) != "exact-segmented") {
    std::cerr << "Q3X_TEST_PREFILL_ATTENTION_TACTIC must be "
                 "exact-segmented, native-group-q64-panel, or "
                 "native-group-q128-v4-panel, or "
                 "native-flashinfer-exact-panel\n";
    return 2;
  }
  const char* const projection_tactic =
      std::getenv("Q3X_TEST_PREFILL_PROJECTION_TACTIC");
  const bool segmented_marlin_operator_panel =
      projection_tactic != nullptr &&
      std::string_view(projection_tactic) ==
          "segmented-marlin-operator-panel";
  const bool native_large_m_operator_panel =
      projection_tactic != nullptr &&
      std::string_view(projection_tactic) ==
          "native-quantized-large-m-operator-panel";
  if (projection_tactic != nullptr && !segmented_marlin_operator_panel &&
      !native_large_m_operator_panel &&
      std::string_view(projection_tactic) != "exact-segmented") {
    std::cerr << "Q3X_TEST_PREFILL_PROJECTION_TACTIC must be "
                 "exact-segmented, segmented-marlin-operator-panel, or "
                 "native-quantized-large-m-operator-panel\n";
    return 2;
  }
  std::vector<std::size_t> prompt_token_counts(kPromptTokenCounts.begin(),
                                                kPromptTokenCounts.end());
  const char* const selected_prompt_tokens =
      std::getenv("Q3X_TEST_PREFILL_PROMPT_TOKENS");
  if (selected_prompt_tokens != nullptr) {
    const auto selected = std::find_if(
        kPromptTokenCounts.begin(), kPromptTokenCounts.end(),
        [selected_prompt_tokens](const std::size_t value) {
          return std::to_string(value) == selected_prompt_tokens;
        });
    if (selected == kPromptTokenCounts.end()) {
      std::cerr << "Q3X_TEST_PREFILL_PROMPT_TOKENS must be one of "
                   "513, 1025, 7712, 8192, or 8193\n";
      return 2;
    }
    prompt_token_counts.assign(1U, *selected);
  }
  if (layer3_attention_oracle) {
    if ((attention_tactic != nullptr &&
         std::string_view(attention_tactic) != "exact-segmented") ||
        (projection_tactic != nullptr &&
         std::string_view(projection_tactic) != "exact-segmented")) {
      std::cerr << "the layer3 P513 Attention oracle requires exact-segmented "
                   "engine tactics\n";
      return 2;
    }
    prompt_token_counts.assign(1U, kLayer3AttentionOracleTokens);
  }

  try {
    runtime::ReferenceEngineOptions options;
    options.request_options.prefill_chunk_size =
        runtime::kMaximumRequestPrefillChunkSize;
    options.request_options.max_sequence_length =
        prompt_token_counts.back() + 1U;
    options.projection_backend = runtime::ProjectionBackend::kSm87WeightOnly;
    options.prefill_execution_mode =
        runtime::ReferencePrefillExecutionMode::kWholeRequestLayerMajor;
    if (native_flashinfer_exact_panel) {
      options.prefill_full_attention_tactic = runtime::
          LayerMajorPrefillFullAttentionTactic::
              kNativeFlashInferExactPanel;
    } else if (native_group_q128_v4_panel) {
      options.prefill_full_attention_tactic = runtime::
          LayerMajorPrefillFullAttentionTactic::kNativeGroupQ128V4Panel;
    } else if (native_group_q64_panel) {
      options.prefill_full_attention_tactic = runtime::
          LayerMajorPrefillFullAttentionTactic::kNativeGroupQ64Panel;
    } else {
      options.prefill_full_attention_tactic = runtime::
          LayerMajorPrefillFullAttentionTactic::kExactSegmentedC512;
    }
    if (native_large_m_operator_panel) {
      options.prefill_projection_tactic = runtime::
          LayerMajorPrefillProjectionTactic::
              kNativeQuantizedLargeMOperatorPanel;
    } else if (segmented_marlin_operator_panel) {
      options.prefill_projection_tactic = runtime::
          LayerMajorPrefillProjectionTactic::kSegmentedMarlinOperatorPanel;
    } else {
      options.prefill_projection_tactic = runtime::
          LayerMajorPrefillProjectionTactic::kExactSegmentedC512;
    }
    runtime::ReferenceEngineCreateResult created =
        runtime::create_reference_engine(
            std::filesystem::path(model_directory), options);
    if (!created) {
      std::cerr << "engine creation failed: ";
      print_diagnostic(created.diagnostic);
      return 1;
    }

    if (layer3_attention_oracle) {
      Layer3AttentionP513OracleCollector collector;
      if (!collector.prepare()) {
        std::cerr << "failed to prepare layer3 P513 Attention oracle: "
                  << collector.failed_operation
                  << " cuda_error=" << collector.cuda_error << '\n';
        return 1;
      }
      bool generation_passed = false;
      {
        const ScopedLayer3AttentionP513Oracle oracle_hook(collector);
        generation_passed = run_case(
            *created.value, kLayer3AttentionOracleTokens,
            runtime::LayerMajorPrefillFullAttentionTactic::
                kExactSegmentedC512,
            runtime::LayerMajorPrefillProjectionTactic::kExactSegmentedC512);
      }
      const Layer3AttentionOracleReportStatus attention_status =
          report_layer3_attention_p513_oracle(collector);
      if (!generation_passed ||
          attention_status ==
              Layer3AttentionOracleReportStatus::kInfrastructureFailure) {
        return 1;
      }
      return attention_status ==
                     Layer3AttentionOracleReportStatus::kBitwisePass
                 ? 0
                 : 3;
    }

    for (const std::size_t prompt_tokens : prompt_token_counts) {
      if (!run_case(*created.value, prompt_tokens,
                    options.prefill_full_attention_tactic,
                    options.prefill_projection_tactic)) {
        return 1;
      }
    }
    return 0;
  } catch (const std::bad_alloc&) {
    std::cerr << "host allocation failed\n";
    return 1;
  }
}
