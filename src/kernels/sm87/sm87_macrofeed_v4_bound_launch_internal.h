#pragma once

#include "q3x/kernels/sm87_macrofeed_v4_bf16_ab.h"
#include "q3x/kernels/sm87_macrofeed_v4_attention_c8000.h"
#include "q3x/kernels/sm87_macrofeed_v4_fp8.h"
#include "q3x/kernels/sm87_macrofeed_v4_full_attention_preprocess.h"
#include "q3x/kernels/sm87_macrofeed_v4_gdn_c8000.h"
#include "q3x/kernels/sm87_macrofeed_v4_norm_residual.h"
#include "q3x/kernels/sm87_macrofeed_v4_nvfp4_down.h"
#include "q3x/kernels/sm87_macrofeed_v4_nvfp4_gate_up.h"
#include "q3x/runtime/decode_ops.h"

#include <cstddef>
#include <cstdint>

namespace q3x::runtime::sm87_macrofeed_v4_execution_events_detail {
class Sm87MacroFeedV4ExecutionEventsOwner;
}

namespace q3x::kernels::sm87_macrofeed_v4_bound_launch_detail {

// Fixed GDN-QKVZ C8000 binding.  There is intentionally no caller-selected
// role, layout, token count, row stride, K/V output, or CUDA stream in this
// schema.  The bound body always consumes contiguous H5120 and publishes the
// 16,384 projected values into the canonical 17,408-wide phase scratch.
struct Sm87MacroFeedV4GdnQkvzC8000Arguments final {
  const std::uint16_t* hidden_input = nullptr;
  Sm87TargetAotFp8CudaAssetView asset{};
  std::uint16_t* phase_scratch = nullptr;
};

// Fixed exact GDN continuation.  Geometry and stream are owner-bound; the
// numerical constants and recurrent epochs are construction-sealed values.
struct Sm87MacroFeedV4GdnContinuationC8000Arguments final {
  std::uint16_t* phase_scratch = nullptr;
  const std::uint16_t* conv_weight = nullptr;
  const std::uint16_t* a_log = nullptr;
  const std::uint16_t* dt_bias = nullptr;
  const std::uint16_t* norm_weight = nullptr;
  const std::uint16_t* active_conv_history = nullptr;
  std::uint16_t* candidate_conv_history = nullptr;
  const std::uint16_t* active_recurrent_state = nullptr;
  std::uint16_t* candidate_recurrent_state = nullptr;
  const std::uint32_t* cancellation_signal = nullptr;
  std::uint32_t l2_epsilon_fp32_bits = 0U;
  std::uint32_t norm_epsilon_fp32_bits = 0U;
};

// Physical acceptance ledger for the multi-operation continuation body.
// Unlike a success receipt, these counters survive a later launch failure so
// the EventsOwner can poison and drain the exact amount of accepted work.
struct Sm87MacroFeedV4GdnContinuationSubmitLedger final {
  std::size_t accepted_kernel_launches = 0U;
  std::size_t asynchronous_d2d_copies = 0U;
  std::size_t conv_history_copy_bytes = 0U;
};

// Fixed GDN Attention-O binding.  The input is always the contiguous V slice
// at physical offset 4096 of the canonical phase scratch.  No caller-selected
// role, layout, tactic, row stride, token count, or stream is present.
struct Sm87MacroFeedV4GdnOC8000Arguments final {
  const std::uint16_t* phase_scratch = nullptr;
  Sm87TargetAotFp8CudaAssetView asset{};
  std::uint16_t* branch_output = nullptr;
};

// Fixed Full-QKV binding.  K/V are the authenticated active-panel slices;
// the public kernel writes exactly 8,000 local rows and therefore must not be
// given either full-allocation origin by this seam.
struct Sm87MacroFeedV4FullQkvC8000Arguments final {
  const std::uint16_t* hidden_input = nullptr;
  Sm87TargetAotFp8CudaAssetView asset{};
  std::uint16_t* q_gate_scratch = nullptr;
  std::uint16_t* key_panel_output = nullptr;
  std::uint16_t* value_panel_output = nullptr;
};

// Fixed exact in-place Q/K preprocessing binding.  Unlike Full-QKV, the key
// pointer is the complete private allocation origin because first_position is
// applied inside the kernel.  V is deliberately unaddressable here.
struct Sm87MacroFeedV4FullAttentionPreprocessC8000Arguments final {
  std::uint16_t* q_gate_scratch = nullptr;
  std::uint16_t* key_cache_origin = nullptr;
  const std::uint16_t* q_norm_weight = nullptr;
  const std::uint16_t* k_norm_weight = nullptr;
  const float* cosines = nullptr;
  const float* sines = nullptr;
  std::size_t first_position = 0U;
};

// Fixed exact causal Attention binding.  Both K/V pointers are complete
// private allocation origins so the core can consume [0, first+8000).
struct Sm87MacroFeedV4AttentionCoreC8000Arguments final {
  std::uint16_t* q_gate_scratch = nullptr;
  const std::uint16_t* key_cache_origin = nullptr;
  const std::uint16_t* value_cache_origin = nullptr;
  std::size_t first_position = 0U;
};

// Fixed Full-Attention O binding.  Tactic 4603 gathers only the interleaved Q
// slots; the Gate slots and row gap are not caller-selectable inputs.
struct Sm87MacroFeedV4FullAttentionOC8000Arguments final {
  const std::uint16_t* q_gate_scratch = nullptr;
  Sm87TargetAotFp8CudaAssetView asset{};
  std::uint16_t* branch_output = nullptr;
};

struct Sm87MacroFeedV4ResidualPostNormC8000Arguments final {
  std::uint16_t* left_residual_then_normalized = nullptr;
  std::uint16_t* right_branch_then_residual = nullptr;
  const std::uint16_t* centered_weight = nullptr;
};

struct Sm87MacroFeedV4GateUpC8000Arguments final {
  const std::uint16_t* normalized_input = nullptr;
  const std::uint8_t* payload = nullptr;
  std::size_t payload_bytes = 0U;
  float gate_tensor_scale = 0.0F;
  float up_tensor_scale = 0.0F;
  std::uint16_t* intermediate_output = nullptr;
  Sm87MacroFeedV3NvFp4GateUpPayloadReceipt
      canonical_v3_payload_receipt{};
};

struct Sm87MacroFeedV4DownC8000Arguments final {
  const std::uint16_t* intermediate_input = nullptr;
  const std::uint8_t* payload = nullptr;
  std::size_t payload_bytes = 0U;
  float tensor_scale = 0.0F;
  std::uint16_t* residual_output = nullptr;
  Sm87MacroFeedV3NvFp4DownPayloadReceipt payload_receipt{};
};

// These four boundary bodies are the fixed outer edges of the C8000 request:
// prompt ingestion, final hidden normalization, vocabulary projection, and
// greedy selection.  Shape and numerical facts are deliberately absent from
// the caller-fillable argument records.  The constants below are the complete
// ABI and byte-range contract used by construction and by the locked hot
// seams.
inline constexpr std::size_t kSm87MacroFeedV4EmbeddingTokenCount = 8'000U;
inline constexpr std::size_t kSm87MacroFeedV4EmbeddingVocabulary = 248'320U;
inline constexpr std::size_t kSm87MacroFeedV4Hidden = 5'120U;
inline constexpr std::size_t kSm87MacroFeedV4EmbeddingTableBytes =
    kSm87MacroFeedV4EmbeddingVocabulary * kSm87MacroFeedV4Hidden *
    sizeof(std::uint16_t);
inline constexpr std::size_t kSm87MacroFeedV4EmbeddingTokenIdBytes =
    kSm87MacroFeedV4EmbeddingTokenCount * sizeof(std::uint32_t);
inline constexpr std::size_t kSm87MacroFeedV4EmbeddingOutputBytes =
    kSm87MacroFeedV4EmbeddingTokenCount * kSm87MacroFeedV4Hidden *
    sizeof(std::uint16_t);

inline constexpr std::uint32_t kSm87MacroFeedV4FinalNormEpsilonFp32Bits =
    0x3586'37bdU;
inline constexpr std::size_t kSm87MacroFeedV4FinalNormBytes =
    kSm87MacroFeedV4Hidden * sizeof(std::uint16_t);

inline constexpr std::size_t kSm87MacroFeedV4Vocabulary = 248'320U;
inline constexpr std::size_t kSm87MacroFeedV4VocabularyBf16Bytes =
    kSm87MacroFeedV4Vocabulary * sizeof(std::uint16_t);
inline constexpr std::size_t kSm87MacroFeedV4GreedyWorkspaceResults =
    q3x::runtime::kBf16GreedyArgmaxWorkspaceResults;
inline constexpr std::size_t kSm87MacroFeedV4GreedyWorkspaceBytes =
    kSm87MacroFeedV4GreedyWorkspaceResults *
    sizeof(q3x::runtime::Bf16GreedyArgmaxResult);

inline constexpr std::size_t kSm87MacroFeedV4LmHeadRows = 248'320U;
inline constexpr std::size_t kSm87MacroFeedV4LmHeadColumns = 5'120U;
inline constexpr std::size_t kSm87MacroFeedV4LmHeadPackedWeightBytes =
    kSm87MacroFeedV4LmHeadRows * kSm87MacroFeedV4LmHeadColumns / 2U;
inline constexpr std::size_t kSm87MacroFeedV4LmHeadBlockScaleBytes =
    kSm87MacroFeedV4LmHeadRows * kSm87MacroFeedV4LmHeadColumns / 16U;
inline constexpr std::size_t kSm87MacroFeedV4LmHeadActivationBytes =
    kSm87MacroFeedV4LmHeadColumns * sizeof(std::uint16_t);
inline constexpr std::size_t kSm87MacroFeedV4LmHeadOutputBytes =
    kSm87MacroFeedV4LmHeadRows * sizeof(std::uint16_t);

static_assert(kSm87MacroFeedV4EmbeddingTableBytes == 2'542'796'800U);
static_assert(kSm87MacroFeedV4EmbeddingTokenIdBytes == 32'000U);
static_assert(kSm87MacroFeedV4EmbeddingOutputBytes == 81'920'000U);
static_assert(kSm87MacroFeedV4FinalNormBytes == 10'240U);
static_assert(kSm87MacroFeedV4GreedyWorkspaceResults == 33U);
static_assert(kSm87MacroFeedV4GreedyWorkspaceBytes == 264U);
static_assert(kSm87MacroFeedV4LmHeadPackedWeightBytes == 635'699'200U);
static_assert(kSm87MacroFeedV4LmHeadBlockScaleBytes == 79'462'400U);
static_assert(kSm87MacroFeedV4LmHeadActivationBytes == 10'240U);
static_assert(kSm87MacroFeedV4LmHeadOutputBytes == 496'640U);

struct Sm87MacroFeedV4EmbeddingC8000Arguments final {
  const std::uint16_t* embedding_table = nullptr;
  const std::uint32_t* token_ids = nullptr;
  std::uint16_t* hidden_output = nullptr;
};

struct Sm87MacroFeedV4FinalNormM1Arguments final {
  const std::uint16_t* hidden_input = nullptr;
  const std::uint16_t* centered_weight = nullptr;
  std::uint16_t* normalized_output = nullptr;
};

struct Sm87MacroFeedV4GreedyArgmaxM1Arguments final {
  const std::uint16_t* logits = nullptr;
  // Exactly 33 externally owned results: [0,32) partials and [32] final.
  q3x::runtime::Bf16GreedyArgmaxResult* result_workspace = nullptr;
};

struct Sm87MacroFeedV4LmHeadM1Arguments final {
  const std::uint8_t* packed_weights = nullptr;
  const std::uint8_t* block_scales = nullptr;
  // The exact leaf applies this weight-side scale once.  No input scale is
  // present in this ABI and the seam must never synthesize or multiply one.
  float weight_scale_2 = 0.0F;
  const std::uint16_t* activation = nullptr;
  std::uint16_t* logits_output = nullptr;
};

struct Sm87MacroFeedV4FixedKernelResource final {
  std::int32_t registers_per_thread = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  std::int32_t maximum_threads_per_block = 0;
  std::int32_t active_blocks_per_sm = 0;
  std::int32_t binary_version = 0;
  std::int32_t threads = 0;
  std::int32_t grid_ctas = 0;
};

struct Sm87MacroFeedV4EmbeddingC8000ResourceSnapshot final {
  std::uint64_t identity = 0U;
  std::int32_t device_ordinal = -1;
  std::int32_t compute_major = 0;
  std::int32_t compute_minor = 0;
  std::int32_t sm_count = 0;
  Sm87MacroFeedV4FixedKernelResource gather{};
  bool exact_geometry = false;
  bool static_resource_gate_passed = false;
};

struct Sm87MacroFeedV4FinalNormM1ResourceSnapshot final {
  std::uint64_t identity = 0U;
  std::int32_t device_ordinal = -1;
  std::int32_t compute_major = 0;
  std::int32_t compute_minor = 0;
  std::int32_t sm_count = 0;
  Sm87MacroFeedV4FixedKernelResource centered_rms_norm{};
  bool exact_geometry = false;
  bool static_resource_gate_passed = false;
};

struct Sm87MacroFeedV4GreedyArgmaxM1ResourceSnapshot final {
  std::uint64_t identity = 0U;
  std::int32_t device_ordinal = -1;
  std::int32_t compute_major = 0;
  std::int32_t compute_minor = 0;
  std::int32_t sm_count = 0;
  Sm87MacroFeedV4FixedKernelResource partial{};
  Sm87MacroFeedV4FixedKernelResource finalize{};
  bool exact_geometry = false;
  bool static_resource_gate_passed = false;
};

struct Sm87MacroFeedV4LmHeadM1ResourceSnapshot final {
  std::uint64_t identity = 0U;
  std::int32_t device_ordinal = -1;
  std::int32_t compute_major = 0;
  std::int32_t compute_minor = 0;
  std::int32_t sm_count = 0;
  Sm87MacroFeedV4FixedKernelResource activation_staged{};
  bool exact_geometry = false;
  bool static_resource_gate_passed = false;
};

// Accepted-prefix evidence is intentionally physical rather than a success
// receipt.  A later launch failure leaves the number of prior accepted CUDA
// operations intact for poison/drain accounting.
struct Sm87MacroFeedV4FixedSubmitLedger final {
  std::size_t accepted_kernel_launches = 0U;
};

// The CUDA stream exists only inside an EventsOwner-locked submission.  The
// execution package cannot construct, copy, move, inspect, or retain this
// capability, so a raw handle never crosses the owner boundary.
class Sm87MacroFeedV4LockedSubmitToken final {
 public:
  Sm87MacroFeedV4LockedSubmitToken() = delete;
  Sm87MacroFeedV4LockedSubmitToken(
      const Sm87MacroFeedV4LockedSubmitToken&) = delete;
  Sm87MacroFeedV4LockedSubmitToken& operator=(
      const Sm87MacroFeedV4LockedSubmitToken&) = delete;
  Sm87MacroFeedV4LockedSubmitToken(
      Sm87MacroFeedV4LockedSubmitToken&&) = delete;
  Sm87MacroFeedV4LockedSubmitToken& operator=(
      Sm87MacroFeedV4LockedSubmitToken&&) = delete;

 private:
  explicit Sm87MacroFeedV4LockedSubmitToken(void* cuda_stream) noexcept
      : cuda_stream_(cuda_stream) {}

  void* cuda_stream_ = nullptr;

  friend class q3x::runtime::
      sm87_macrofeed_v4_execution_events_detail::
          Sm87MacroFeedV4ExecutionEventsOwner;
  friend int enqueue_input_norm_prevalidated(
      const Sm87MacroFeedV4LockedSubmitToken&,
      const Sm87MacroFeedV4InputNormArguments&,
      const Sm87MacroFeedV4NormResidualAdmissionResourceSnapshot&,
      std::size_t*) noexcept;
  friend int enqueue_bf16_ab_prevalidated(
      const Sm87MacroFeedV4LockedSubmitToken&,
      const Sm87MacroFeedV4Bf16AbArguments&,
      const Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot&,
      std::size_t*) noexcept;
  friend int enqueue_gdn_qkvz_c8000_prevalidated(
      const Sm87MacroFeedV4LockedSubmitToken&,
      const Sm87MacroFeedV4GdnQkvzC8000Arguments&,
      const Sm87MacroFeedV4Fp8CudaResources&,
      std::size_t*) noexcept;
  friend int enqueue_gdn_continuation_c8000_prevalidated(
      const Sm87MacroFeedV4LockedSubmitToken&,
      const Sm87MacroFeedV4GdnContinuationC8000Arguments&,
      const Sm87MacroFeedV4GdnC8000AdmissionResourceSnapshot&,
      Sm87MacroFeedV4GdnContinuationSubmitLedger*) noexcept;
  friend int enqueue_gdn_o_c8000_prevalidated(
      const Sm87MacroFeedV4LockedSubmitToken&,
      const Sm87MacroFeedV4GdnOC8000Arguments&,
      const Sm87MacroFeedV4Fp8CudaResources&,
      std::size_t*) noexcept;
  friend int enqueue_full_qkv_c8000_prevalidated(
      const Sm87MacroFeedV4LockedSubmitToken&,
      const Sm87MacroFeedV4FullQkvC8000Arguments&,
      const Sm87MacroFeedV4Fp8CudaResources&,
      std::size_t*) noexcept;
  friend int enqueue_full_attention_preprocess_c8000_prevalidated(
      const Sm87MacroFeedV4LockedSubmitToken&,
      const Sm87MacroFeedV4FullAttentionPreprocessC8000Arguments&,
      const Sm87MacroFeedV4FullAttentionPreprocessAdmissionResourceSnapshot&,
      std::size_t*) noexcept;
  friend int enqueue_attention_c8000_prevalidated(
      const Sm87MacroFeedV4LockedSubmitToken&,
      const Sm87MacroFeedV4AttentionCoreC8000Arguments&,
      const Sm87MacroFeedV4AttentionC8000AdmissionResourceSnapshot&,
      std::size_t*) noexcept;
  friend int enqueue_full_attention_o_c8000_prevalidated(
      const Sm87MacroFeedV4LockedSubmitToken&,
      const Sm87MacroFeedV4FullAttentionOC8000Arguments&,
      const Sm87MacroFeedV4Fp8CudaResources&,
      std::size_t*) noexcept;
  friend int enqueue_residual_post_norm_prevalidated(
      const Sm87MacroFeedV4LockedSubmitToken&,
      const Sm87MacroFeedV4ResidualPostNormC8000Arguments&,
      const Sm87MacroFeedV4NormResidualAdmissionResourceSnapshot&,
      std::size_t*) noexcept;
  friend int enqueue_gate_up_c8000_prevalidated(
      const Sm87MacroFeedV4LockedSubmitToken&,
      const Sm87MacroFeedV4GateUpC8000Arguments&,
      const Sm87MacroFeedV4NvFp4GateUpCudaResources&,
      std::size_t*) noexcept;
  friend int enqueue_down_c8000_prevalidated(
      const Sm87MacroFeedV4LockedSubmitToken&,
      const Sm87MacroFeedV4DownC8000Arguments&,
      const Sm87MacroFeedV4NvFp4DownCudaResources&,
      std::size_t*) noexcept;
  friend int enqueue_embedding_c8000_prevalidated(
      const Sm87MacroFeedV4LockedSubmitToken&,
      const Sm87MacroFeedV4EmbeddingC8000Arguments&,
      const Sm87MacroFeedV4EmbeddingC8000ResourceSnapshot&,
      Sm87MacroFeedV4FixedSubmitLedger*) noexcept;
  friend int enqueue_final_norm_m1_prevalidated(
      const Sm87MacroFeedV4LockedSubmitToken&,
      const Sm87MacroFeedV4FinalNormM1Arguments&,
      const Sm87MacroFeedV4FinalNormM1ResourceSnapshot&,
      Sm87MacroFeedV4FixedSubmitLedger*) noexcept;
  friend int enqueue_greedy_argmax_m1_prevalidated(
      const Sm87MacroFeedV4LockedSubmitToken&,
      const Sm87MacroFeedV4GreedyArgmaxM1Arguments&,
      const Sm87MacroFeedV4GreedyArgmaxM1ResourceSnapshot&,
      Sm87MacroFeedV4FixedSubmitLedger*) noexcept;
  friend int enqueue_lm_head_m1_prevalidated(
      const Sm87MacroFeedV4LockedSubmitToken&,
      const Sm87MacroFeedV4LmHeadM1Arguments&,
      const Sm87MacroFeedV4LmHeadM1ResourceSnapshot&,
      Sm87MacroFeedV4FixedSubmitLedger*) noexcept;
};

// Construction-prevalidated V4 launch seams.  These functions intentionally
// perform no device, function, occupancy, stream-device, allocation-range,
// JIT, repack, autotune, or tactic query.  They are source-private and may be
// called only by the event owner while it holds its stream submission lock.
// The execution package must have sealed every pointer/resource fact and own
// the complete CUDA allocation lifetime.
[[nodiscard]] int enqueue_input_norm_prevalidated(
    const Sm87MacroFeedV4LockedSubmitToken& token,
    const Sm87MacroFeedV4InputNormArguments& arguments,
    const Sm87MacroFeedV4NormResidualAdmissionResourceSnapshot& resources,
    std::size_t* submitted_launches) noexcept;

[[nodiscard]] int enqueue_bf16_ab_prevalidated(
    const Sm87MacroFeedV4LockedSubmitToken& token,
    const Sm87MacroFeedV4Bf16AbArguments& arguments,
    const Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot& resources,
    std::size_t* submitted_launches) noexcept;

[[nodiscard]] int enqueue_gdn_qkvz_c8000_prevalidated(
    const Sm87MacroFeedV4LockedSubmitToken& token,
    const Sm87MacroFeedV4GdnQkvzC8000Arguments& arguments,
    const Sm87MacroFeedV4Fp8CudaResources& resources,
    std::size_t* submitted_launches) noexcept;

[[nodiscard]] int enqueue_gdn_continuation_c8000_prevalidated(
    const Sm87MacroFeedV4LockedSubmitToken& token,
    const Sm87MacroFeedV4GdnContinuationC8000Arguments& arguments,
    const Sm87MacroFeedV4GdnC8000AdmissionResourceSnapshot& resources,
    Sm87MacroFeedV4GdnContinuationSubmitLedger* submit_ledger) noexcept;

[[nodiscard]] int enqueue_gdn_o_c8000_prevalidated(
    const Sm87MacroFeedV4LockedSubmitToken& token,
    const Sm87MacroFeedV4GdnOC8000Arguments& arguments,
    const Sm87MacroFeedV4Fp8CudaResources& resources,
    std::size_t* submitted_launches) noexcept;

[[nodiscard]] int enqueue_full_qkv_c8000_prevalidated(
    const Sm87MacroFeedV4LockedSubmitToken& token,
    const Sm87MacroFeedV4FullQkvC8000Arguments& arguments,
    const Sm87MacroFeedV4Fp8CudaResources& resources,
    std::size_t* submitted_launches) noexcept;

[[nodiscard]] int enqueue_full_attention_preprocess_c8000_prevalidated(
    const Sm87MacroFeedV4LockedSubmitToken& token,
    const Sm87MacroFeedV4FullAttentionPreprocessC8000Arguments& arguments,
    const Sm87MacroFeedV4FullAttentionPreprocessAdmissionResourceSnapshot&
        resources,
    std::size_t* submitted_launches) noexcept;

[[nodiscard]] int enqueue_attention_c8000_prevalidated(
    const Sm87MacroFeedV4LockedSubmitToken& token,
    const Sm87MacroFeedV4AttentionCoreC8000Arguments& arguments,
    const Sm87MacroFeedV4AttentionC8000AdmissionResourceSnapshot& resources,
    std::size_t* submitted_launches) noexcept;

[[nodiscard]] int enqueue_full_attention_o_c8000_prevalidated(
    const Sm87MacroFeedV4LockedSubmitToken& token,
    const Sm87MacroFeedV4FullAttentionOC8000Arguments& arguments,
    const Sm87MacroFeedV4Fp8CudaResources& resources,
    std::size_t* submitted_launches) noexcept;

[[nodiscard]] int enqueue_residual_post_norm_prevalidated(
    const Sm87MacroFeedV4LockedSubmitToken& token,
    const Sm87MacroFeedV4ResidualPostNormC8000Arguments& arguments,
    const Sm87MacroFeedV4NormResidualAdmissionResourceSnapshot& resources,
    std::size_t* submitted_launches) noexcept;

[[nodiscard]] int enqueue_gate_up_c8000_prevalidated(
    const Sm87MacroFeedV4LockedSubmitToken& token,
    const Sm87MacroFeedV4GateUpC8000Arguments& arguments,
    const Sm87MacroFeedV4NvFp4GateUpCudaResources& resources,
    std::size_t* submitted_launches) noexcept;

[[nodiscard]] int enqueue_down_c8000_prevalidated(
    const Sm87MacroFeedV4LockedSubmitToken& token,
    const Sm87MacroFeedV4DownC8000Arguments& arguments,
    const Sm87MacroFeedV4NvFp4DownCudaResources& resources,
    std::size_t* submitted_launches) noexcept;

// Construction-only observation contracts.  They may query the current
// device, function attributes, and occupancy, but they allocate nothing and
// confer no request-time authority.  Only an immutable retained snapshot plus
// a LockedSubmitToken can reach the corresponding hot seam.
[[nodiscard]] int query_embedding_c8000_resources_cuda(
    Sm87MacroFeedV4EmbeddingC8000ResourceSnapshot* resources) noexcept;

[[nodiscard]] int query_final_norm_m1_resources_cuda(
    Sm87MacroFeedV4FinalNormM1ResourceSnapshot* resources) noexcept;

[[nodiscard]] int query_greedy_argmax_m1_resources_cuda(
    Sm87MacroFeedV4GreedyArgmaxM1ResourceSnapshot* resources) noexcept;

[[nodiscard]] int query_lm_head_m1_resources_cuda(
    Sm87MacroFeedV4LmHeadM1ResourceSnapshot* resources) noexcept;

[[nodiscard]] int enqueue_embedding_c8000_prevalidated(
    const Sm87MacroFeedV4LockedSubmitToken& token,
    const Sm87MacroFeedV4EmbeddingC8000Arguments& arguments,
    const Sm87MacroFeedV4EmbeddingC8000ResourceSnapshot& resources,
    Sm87MacroFeedV4FixedSubmitLedger* submit_ledger) noexcept;

[[nodiscard]] int enqueue_final_norm_m1_prevalidated(
    const Sm87MacroFeedV4LockedSubmitToken& token,
    const Sm87MacroFeedV4FinalNormM1Arguments& arguments,
    const Sm87MacroFeedV4FinalNormM1ResourceSnapshot& resources,
    Sm87MacroFeedV4FixedSubmitLedger* submit_ledger) noexcept;

[[nodiscard]] int enqueue_greedy_argmax_m1_prevalidated(
    const Sm87MacroFeedV4LockedSubmitToken& token,
    const Sm87MacroFeedV4GreedyArgmaxM1Arguments& arguments,
    const Sm87MacroFeedV4GreedyArgmaxM1ResourceSnapshot& resources,
    Sm87MacroFeedV4FixedSubmitLedger* submit_ledger) noexcept;

[[nodiscard]] int enqueue_lm_head_m1_prevalidated(
    const Sm87MacroFeedV4LockedSubmitToken& token,
    const Sm87MacroFeedV4LmHeadM1Arguments& arguments,
    const Sm87MacroFeedV4LmHeadM1ResourceSnapshot& resources,
    Sm87MacroFeedV4FixedSubmitLedger* submit_ledger) noexcept;

}  // namespace q3x::kernels::sm87_macrofeed_v4_bound_launch_detail
