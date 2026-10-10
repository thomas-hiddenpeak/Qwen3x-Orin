#include "mtp_device_internal.h"
#include "mtp_prefill_internal.h"
#if defined(Q3X_ENABLE_FUSED_DECODE)
#include "decode_fused_gqa_internal.h"
#endif
#include "model/mtp_weights_internal.h"
#include "q3x/core/sha256.h"
#include "q3x/runtime/decode_ops.h"
#include "q3x/runtime/layout_ops.h"
#include "q3x/runtime/resident_weights.h"

#include <cuda_runtime_api.h>
#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace q3x::runtime::mtp_detail {

namespace {
constexpr std::uint64_t kReserve = 8ULL << 30;
constexpr std::uint64_t kMtpBegin = 59416;
constexpr std::uint64_t kWeightBytes = 849398784;
constexpr std::size_t H = 5120, V = 248320;
void check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
void cuda_check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void* allocate(std::size_t bytes) {
  std::size_t free = 0, total = 0;
  cuda_check(cudaMemGetInfo(&free, &total));
  if (free < bytes || free - bytes < kReserve)
    throw std::runtime_error("MTP retained-free reserve: free=" + std::to_string(free) +
        " requested=" + std::to_string(bytes) + " reserve=" + std::to_string(kReserve));
  void* result = nullptr;
  cuda_check(cudaMalloc(&result, bytes));
  return result;
}
struct Fd {
  int value = -1;
  ~Fd() { if (value >= 0) ::close(value); }
};
int open_root(const std::filesystem::path& path) {
  check(path.is_absolute(), "MTP model path must be absolute");
  Fd directory{::open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC)};
  check(directory.value >= 0, "MTP open root");
  for (const auto& part : path) {
    if (part == "/" || part == ".") continue;
    check(part != "..", "MTP unsafe root component");
    int next = ::openat(directory.value, part.c_str(),
                        O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    check(next >= 0, "MTP unsafe model directory");
    ::close(directory.value);
    directory.value = next;
  }
  const int result = directory.value;
  directory.value = -1;
  return result;
}
}

Weights::Weights(const std::filesystem::path& directory) {
  const auto manifest = model::weights::build_qwen36_27b_text_manifest(directory);
  check(bool(manifest), "MTP checkpoint manifest rejected");
  const auto plan = model::mtp_detail::plan_weights(*manifest.value);
  check(plan.value.has_value(), "MTP tensor contract rejected");
  const auto& identity = pinned_qwen36_27b_shards().at(2);
  // Compile-time source offsets plus full-file authentication close the gap
  // between the earlier manifest read and the file descriptor copied below.
  for (std::size_t i = 0; i < offsets_.size(); ++i) {
    const auto& entry = plan.value->tensors[i];
    check(entry.locator->shard == identity.filename &&
          entry.locator->file_begin == kMtpBegin + entry.arena_offset,
          "MTP pinned source range mismatch");
    offsets_[i] = entry.arena_offset;
  }
  Fd root{open_root(directory)};
  Fd file{::openat(root.value, identity.filename.c_str(),
                   O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK)};
  check(file.value >= 0, "MTP open shard");
  struct stat before{}, after{};
  check(::fstat(file.value, &before) == 0 && S_ISREG(before.st_mode) &&
        before.st_size >= 0 && std::uint64_t(before.st_size) == identity.file_size,
        "MTP shard type/size mismatch");
  try {
    arena_ = allocate(kWeightBytes);
    std::vector<char> chunk(32U << 20);
    core::Sha256 hash;
    std::uint64_t offset = 0;
    while (offset < identity.file_size) {
      const std::size_t wanted = std::min<std::uint64_t>(chunk.size(), identity.file_size - offset);
      ssize_t received;
      do { received = ::read(file.value, chunk.data(), wanted); } while (received < 0 && errno == EINTR);
      check(received > 0, "MTP shard read failed");
      const std::uint64_t size = static_cast<std::uint64_t>(received);
      check(hash.update(chunk.data(), size), "MTP hash update");
      const auto begin = std::max(offset, kMtpBegin);
      const auto end = std::min(offset + size, kMtpBegin + kWeightBytes);
      if (begin < end) {
        cuda_check(cudaMemcpy(static_cast<char*>(arena_) + begin - kMtpBegin,
                              chunk.data() + begin - offset, end - begin,
                              cudaMemcpyHostToDevice));
      }
      offset += size;
    }
    char extra;
    ssize_t eof;
    do { eof = ::read(file.value, &extra, 1); } while (eof < 0 && errno == EINTR);
    check(eof == 0 && ::fstat(file.value, &after) == 0 &&
          before.st_size == after.st_size && before.st_ino == after.st_ino &&
          before.st_dev == after.st_dev &&
          before.st_mtim.tv_sec == after.st_mtim.tv_sec &&
          before.st_mtim.tv_nsec == after.st_mtim.tv_nsec &&
          before.st_ctim.tv_sec == after.st_ctim.tv_sec &&
          before.st_ctim.tv_nsec == after.st_ctim.tv_nsec,
          "MTP shard changed during authentication");
    check(hash.finalize().hex() == identity.sha256, "MTP full shard SHA256 mismatch");
  } catch (...) {
    if (arena_) cudaFree(arena_);
    arena_ = nullptr;
    throw;
  }
}
Weights::~Weights() { if (arena_) cudaFree(arena_); }
const std::uint16_t* Weights::tensor(std::size_t index) const noexcept {
  if (index >= offsets_.size() || arena_ == nullptr) return nullptr;
  return reinterpret_cast<const std::uint16_t*>(static_cast<const char*>(arena_) + offsets_[index]);
}

struct Draft::Impl {
  const Weights& weights;
  const ModelWeights& base;
  std::uint32_t capacity, position = 0;
  const float* cosines;
  const float* sines;
  void* arena = nullptr;
  cudaStream_t stream = nullptr;
  bool poisoned = false;
  bool ordered_attention = false;
  std::uint64_t ordered_attention_steps = 0;
  int error = 0;
  std::uint16_t *concat, *residual, *normalized, *branch, *final_hidden;
  std::uint16_t *q_gate, *query, *gate, *attention, *mlp_gate, *mlp_up;
  std::uint16_t *key, *value, *logits;
  std::uint16_t *prefill_concat, *prefill_fc, *prefill_norm;
  float* scratch;
  std::size_t scratch_count;
  Bf16GreedyArgmaxResult* argmax;
  Impl(const Weights& w, const ModelWeights& b, std::uint32_t cap,
       const float* cos, const float* sin, bool use_ordered_attention)
      : weights(w), base(b), capacity(cap), cosines(cos), sines(sin) {
    check(cap >= 1 && cap <= 44095 && cos && sin, "MTP draft capacity/RoPE");
    check(b.embed_tokens().input_size == H && b.embed_tokens().output_size == V &&
          b.embed_tokens().weight != nullptr && linear_input_size(b.lm_head()) == H &&
          linear_output_size(b.lm_head()) == V, "MTP shared embedding/lm-head");
    scratch_count = std::max<std::size_t>(V, 24ULL * cap);
#if defined(Q3X_ENABLE_FUSED_DECODE)
    ordered_attention = use_ordered_attention &&
        scratch_count * sizeof(float) >= fused_decode::kWorkspaceBytes;
    if (ordered_attention) cuda_check(static_cast<cudaError_t>(fused_decode::prepare()));
#else
    (void)use_ordered_attention;
#endif
    const std::size_t bytes = 400000 + 4096ULL * cap + 2 * V + 4 * scratch_count + 8 * kDraftPrefillBatch * H;
    arena = allocate(bytes);
    try {
      cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      std::size_t offset = 0;
      const auto take = [&](std::size_t size) {
        offset = (offset + 255) & ~std::size_t(255);
        auto* ptr = static_cast<char*>(arena) + offset;
        offset += size;
        check(offset <= bytes, "MTP draft workspace bound");
        return ptr;
      };
      concat = reinterpret_cast<std::uint16_t*>(take(4 * H));
      residual = reinterpret_cast<std::uint16_t*>(take(2 * H));
      normalized = reinterpret_cast<std::uint16_t*>(take(2 * H));
      branch = reinterpret_cast<std::uint16_t*>(take(2 * H));
      final_hidden = reinterpret_cast<std::uint16_t*>(take(2 * H));
      q_gate = reinterpret_cast<std::uint16_t*>(take(2 * 12288));
      query = reinterpret_cast<std::uint16_t*>(take(2 * 6144));
      gate = reinterpret_cast<std::uint16_t*>(take(2 * 6144));
      attention = reinterpret_cast<std::uint16_t*>(take(2 * 6144));
      mlp_gate = reinterpret_cast<std::uint16_t*>(take(2 * 17408));
      mlp_up = reinterpret_cast<std::uint16_t*>(take(2 * 17408));
      key = reinterpret_cast<std::uint16_t*>(take(2048ULL * cap));
      value = reinterpret_cast<std::uint16_t*>(take(2048ULL * cap));
      logits = reinterpret_cast<std::uint16_t*>(take(2 * V));
      scratch = reinterpret_cast<float*>(take(4 * scratch_count));
      argmax = reinterpret_cast<Bf16GreedyArgmaxResult*>(take(33 * sizeof(Bf16GreedyArgmaxResult)));
      prefill_concat = reinterpret_cast<std::uint16_t*>(take(kDraftPrefillBatch * 4 * H));
      prefill_fc = reinterpret_cast<std::uint16_t*>(take(kDraftPrefillBatch * 2 * H));
      prefill_norm = reinterpret_cast<std::uint16_t*>(take(kDraftPrefillBatch * 2 * H));
    } catch (...) {
      if (stream) cudaStreamDestroy(stream);
      cudaFree(arena);
      throw;
    }
  }
  ~Impl() {
    if (stream) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); }
    if (arena) cudaFree(arena);
  }
  bool call(int status) noexcept {
    if (status == cudaSuccess) return true;
    error = status;
    poisoned = true;
    cudaStreamSynchronize(stream);
    return false;
  }
  bool project(std::size_t index, const std::uint16_t* input, std::uint16_t* output) noexcept {
    const auto& spec = model::mtp_detail::kTensorSpecs[index];
    return call(launch_mtp_prefill_projection(weights.tensor(index), input,
        1, spec.rows, spec.columns, output, stream));
  }
  int attention_step() noexcept {
#if defined(Q3X_ENABLE_FUSED_DECODE)
    if (ordered_attention && position + 1 >= fused_decode::kMinimumSequence) {
      const int status = fused_decode::launch(query, key, value, position + 1,
          scratch, scratch_count * sizeof(float), attention, stream);
      if (status == cudaSuccess) ++ordered_attention_steps;
      return status;
    }
#endif
    return launch_gqa_attention_reference_cuda(query, key, value, 24, 4,
        position + 1, 256, 0.0625F, scratch, scratch_count, attention, stream);
  }
  bool norm(const std::uint16_t* input, std::size_t index, std::uint16_t* output) noexcept {
    return call(launch_centered_rms_norm_reference_cuda(input, weights.tensor(index),
                                                       H, 1.0e-6F, output, stream));
  }
};
Draft::Draft(const Weights& weights, const ModelWeights& model,
             std::uint32_t capacity, const float* cosines, const float* sines,
             bool use_ordered_attention)
    : impl_(std::make_unique<Impl>(weights, model, capacity, cosines, sines,
                                  use_ordered_attention)) {}
Draft::~Draft() = default;
std::uint32_t Draft::position() const noexcept { return impl_->position; }
const std::uint16_t* Draft::hidden() const noexcept { return impl_->final_hidden; }
const std::uint16_t* Draft::keys() const noexcept { return impl_->key; }
const std::uint16_t* Draft::values() const noexcept { return impl_->value; }
int Draft::error() const noexcept { return impl_->error; }
std::uint64_t Draft::ordered_attention_steps() const noexcept {
  return impl_->ordered_attention_steps;
}
bool Draft::rewind(std::uint32_t position) noexcept {
  auto& p = *impl_;
  if (p.poisoned || position > p.position) return false;
  p.position = position;
  return true;
}
bool Draft::reset() noexcept {
  auto& p = *impl_;
  p.ordered_attention_steps = 0;
  if (!p.call(cudaStreamSynchronize(p.stream))) return false;
  if (!p.call(cudaMemsetAsync(p.key, 0, 2048ULL * p.capacity, p.stream)) ||
      !p.call(cudaMemsetAsync(p.value, 0, 2048ULL * p.capacity, p.stream)) ||
      !p.call(cudaStreamSynchronize(p.stream))) return false;
  p.position = 0;
  p.poisoned = false;
  p.error = 0;
  return true;
}
bool Draft::initialize_kv(const std::uint32_t* tokens,
                         const std::uint16_t* hidden, std::uint32_t rows,
                         bool (*cancel)(void*) noexcept, void* context) noexcept {
  auto& p = *impl_;
  if (p.poisoned) return false;
  if (p.position != 0 || rows > p.capacity || (rows && (!tokens || !hidden)))
    return p.call(cudaErrorInvalidValue);
  for (std::uint32_t i = 0; i < rows; ++i)
    if (tokens[i] >= V) return p.call(cudaErrorInvalidValue);
  const auto cancelled = [&]() noexcept {
    if (!cancel || !cancel(context)) return false;
    (void)poison();
    return true;
  };
  if (cancelled()) return false;
  for (std::uint32_t first = 0; first < rows; first += kDraftPrefillBatch) {
    const auto count = std::min<std::uint32_t>(kDraftPrefillBatch, rows - first);
    for (std::uint32_t row = 0; row < count; ++row) {
      auto* concat = p.prefill_concat + row * 2 * H;
      if (!p.call(launch_embedding_gather_reference_cuda(p.base.embed_tokens().weight,
              V, H, tokens[first + row], concat, p.stream)) ||
          !p.norm(concat, 13, concat) ||
          !p.norm(hidden + (first + row) * H, 14, concat + H)) return false;
    }
    auto* key = p.key + 1024ULL * first;
    auto* value = p.value + 1024ULL * first;
    if (!p.call(launch_mtp_prefill_projection(p.weights.tensor(0), p.prefill_concat,
              count, H, 2 * H, p.prefill_fc, p.stream)) ||
        !p.call(launch_headwise_centered_rms_norm_reference_cuda(p.prefill_fc,
              p.weights.tensor(1), count, H, 1.e-6F, p.prefill_norm, p.stream)) ||
        !p.call(launch_mtp_prefill_projection(p.weights.tensor(7), p.prefill_norm,
              count, 1024, H, key, p.stream)) ||
        !p.call(launch_mtp_prefill_projection(p.weights.tensor(11), p.prefill_norm,
              count, 1024, H, value, p.stream)) ||
        !p.call(launch_headwise_centered_rms_norm_reference_cuda(key,
              p.weights.tensor(6), count * 4, 256, 1.e-6F, key, p.stream))) return false;
    for (std::uint32_t row = 0; row < count; ++row)
      if (!p.call(launch_partial_neox_rope_256_64_reference_cuda(key + row * 1024,
              p.cosines + 32ULL * (first + row), p.sines + 32ULL * (first + row),
              4, key + row * 1024, p.stream))) return false;
    if (!p.call(cudaStreamSynchronize(p.stream))) return false;
    p.position = first + count;
    if (cancelled()) return false;
  }
  return true;
}
// Only K/V remain live when rebuilding a selected target-conditioned prefix.
// Proposal execution below remains the independent full-layer implementation.
bool Draft::append_kv(std::uint32_t token, const std::uint16_t* hidden) noexcept {
  auto& p = *impl_;
  if (p.poisoned) return false;
  if (token >= V || !hidden || p.position >= p.capacity)
    return p.call(cudaErrorInvalidValue);
  auto* key = p.key + 1024ULL * p.position;
  auto* value = p.value + 1024ULL * p.position;
  if (!p.call(launch_embedding_gather_reference_cuda(p.base.embed_tokens().weight,
              V, H, token, p.concat, p.stream)) ||
      !p.norm(p.concat, 13, p.concat) || !p.norm(hidden, 14, p.concat + H) ||
      !p.project(0, p.concat, p.residual) || !p.norm(p.residual, 1, p.normalized) ||
      !p.project(7, p.normalized, key) || !p.project(11, p.normalized, value) ||
      !p.call(launch_headwise_centered_rms_norm_reference_cuda(key, p.weights.tensor(6),
              4, 256, 1.0e-6F, key, p.stream)) ||
      !p.call(launch_partial_neox_rope_256_64_reference_cuda(key,
              p.cosines + 32ULL * p.position, p.sines + 32ULL * p.position,
              4, key, p.stream)) || !p.call(cudaStreamSynchronize(p.stream))) return false;
  ++p.position;
  return true;
}
bool Draft::step(std::uint32_t token, const std::uint16_t* hidden,
                 bool with_logits, std::uint32_t& prediction, float* confidence) noexcept {
  auto& p = *impl_;
  if (p.poisoned) return false;
  if (token >= V || hidden == nullptr || p.position >= p.capacity || (confidence && !with_logits)) {
    return p.call(cudaErrorInvalidValue);
  }
  auto* key = p.key + 1024ULL * p.position;
  auto* value = p.value + 1024ULL * p.position;
  const auto call = [&](int status) { return p.call(status); };
  if (!call(launch_embedding_gather_reference_cuda(p.base.embed_tokens().weight,
             V, H, token, p.concat, p.stream)) ||
      !p.norm(p.concat, 13, p.concat) || !p.norm(hidden, 14, p.concat + H) ||
      !p.project(0, p.concat, p.residual) || !p.norm(p.residual, 1, p.normalized) ||
      !p.project(10, p.normalized, p.q_gate) ||
      !p.project(7, p.normalized, key) || !p.project(11, p.normalized, value) ||
      !call(launch_split_interleaved_q_gate_reference_cuda(p.q_gate, 24, 256,
             p.query, p.gate, p.stream)) ||
      !call(launch_headwise_centered_rms_norm_reference_cuda(p.query, p.weights.tensor(9),
             24, 256, 1.0e-6F, p.query, p.stream)) ||
      !call(launch_headwise_centered_rms_norm_reference_cuda(key, p.weights.tensor(6),
             4, 256, 1.0e-6F, key, p.stream)) ||
      !call(launch_partial_neox_rope_256_64_reference_cuda(p.query,
             p.cosines + 32ULL * p.position, p.sines + 32ULL * p.position, 24, p.query, p.stream)) ||
      !call(launch_partial_neox_rope_256_64_reference_cuda(key,
             p.cosines + 32ULL * p.position, p.sines + 32ULL * p.position, 4, key, p.stream)) ||
      !call(p.attention_step()) ||
      !call(launch_sigmoid_gate_reference_cuda(p.attention, p.gate, 6144, p.attention, p.stream)) ||
      !p.project(8, p.attention, p.branch) ||
      !call(launch_residual_add_reference_cuda(p.residual, p.branch, H, p.residual, p.stream)) ||
      !p.norm(p.residual, 5, p.normalized) ||
      !p.project(3, p.normalized, p.mlp_gate) || !p.project(4, p.normalized, p.mlp_up) ||
      !call(launch_silu_mul_reference_cuda(p.mlp_gate, p.mlp_up, 17408, p.mlp_gate, p.stream)) ||
      !p.project(2, p.mlp_gate, p.branch) ||
      !call(launch_residual_add_reference_cuda(p.residual, p.branch, H, p.residual, p.stream)) ||
      !p.norm(p.residual, 12, p.final_hidden)) return false;
  if (with_logits) {
    if (!call(launch_projection_to_bf16_cuda(ProjectionBackend::kSm87WeightOnly,
               p.base.lm_head(), p.final_hidden, p.scratch, p.scratch_count, p.logits, p.stream)) ||
        !call(launch_bf16_greedy_argmax_cuda(p.logits, V, p.argmax, p.stream))) return false;
    Bf16GreedyArgmaxResult host{};
    if (confidence &&
        (!call(launch_mtp_draft_confidence(p.logits, V, p.scratch, p.stream)) ||
         !call(cudaMemcpyAsync(confidence, p.scratch, sizeof(float), cudaMemcpyDeviceToHost, p.stream))))
      return false;
    if (!call(cudaMemcpyAsync(&host, p.argmax, sizeof(host), cudaMemcpyDeviceToHost, p.stream)) ||
        !call(cudaStreamSynchronize(p.stream))) return false;
    if (host.has_nonfinite || host.index >= V ||
        (confidence && (!std::isfinite(*confidence) || *confidence <= 0 || *confidence > 1))) return p.call(cudaErrorInvalidValue);
    prediction = host.index;
  } else {
    if (!call(cudaStreamSynchronize(p.stream))) return false;
    prediction = 0;
  }
  ++p.position;
  return true;
}

bool Draft::poison() noexcept {
  auto& p = *impl_;
  const auto status = cudaStreamSynchronize(p.stream);
  p.poisoned = true;
  if (status != cudaSuccess) p.error = status;
  return status == cudaSuccess;
}

int prepare_mtp_verify_projection_device() noexcept;
TargetTransaction::TargetTransaction(ReferenceRunner& target, RequestState& state, Draft& draft,
                                     bool multirow)
    : multirow_(multirow), target_(target), state_(state), draft_(draft) {
  const bool whole_core = state_.memory_profile() == RequestMemoryProfile::kLayerMajorP40WholeCore;
  check(target_.state_ == &state_ &&
        target_.projection_backend_ == ProjectionBackend::kSm87WeightOnly &&
        linear_weight_kind(target_.weights_->lm_head()) != LinearWeightKind::kBf16 &&
        (state_.memory_profile() == RequestMemoryProfile::kLegacyC512 ||
         (whole_core && target_.layer_major_request_views_.has_value())),
        "MTP scalar transaction requires its exact supported state owner");
  const auto& plan = state_.plan();
  check(!multirow_ || plan.prefill_chunk_size >= 4, "MTP multi-row workspace capacity");
  if (multirow_) cuda_check(static_cast<cudaError_t>(prepare_mtp_verify_projection_device()));
  recurrent_offset_ = plan.conv_state.arena_offset;
  recurrent_bytes_ = plan.conv_state.byte_size + plan.gdn_state.byte_size;
  check(recurrent_bytes_ == 78446592 &&
        plan.gdn_state.arena_offset == recurrent_offset_ + plan.conv_state.byte_size,
        "MTP recurrent snapshot layout");
  slot_bytes_ = recurrent_bytes_ + 2 * H + 2 * V;
  if (whole_core) {
    // Prefill's family workspace is dead after commit; all Decode users bind
    // the disjoint C512 bundle. Borrow it without taking allocation ownership.
    const auto views = state_.layer_major_p40_whole_core_views();
    check(bool(views), "MTP whole-core scratch view");
    const auto& workspace = views.value->linear.prompt_wide_workspace;
    const std::uint64_t snapshot_bytes = 5 * slot_bytes_;
    const std::uint64_t hidden_bytes = 2ULL * H * state_.max_sequence_length();
    check(workspace.device_data &&
          reinterpret_cast<std::uintptr_t>(workspace.device_data) % 256 == 0 &&
          snapshot_bytes % 256 == 0 && workspace.byte_size >= snapshot_bytes + hidden_bytes,
          "MTP post-Prefill workspace capacity/alignment");
    snapshots_ = workspace.device_data;
    prompt_hidden_ = reinterpret_cast<std::uint16_t*>(
        static_cast<char*>(snapshots_) + snapshot_bytes);
  } else {
    snapshots_ = allocate(5 * slot_bytes_);
    owns_snapshots_ = true;
  }
}
TargetTransaction::~TargetTransaction() {
  if (active_) (void)abort();
  if (owns_snapshots_ && snapshots_) cudaFree(snapshots_);
}
bool TargetTransaction::initialize_whole_core_prefill(
    const std::uint32_t* prompt, std::uint32_t count,
    bool (*cancel)(void*) noexcept, void* cancel_context,
    bool committed_service_handoff) noexcept {
  const auto& route = target_.prefill_route_evidence_;
  // Service control calls here only after the runner commits the entire prompt;
  // request-level evidence is finalized after generation. Never allow this
  // handoff while any whole-request stage is still active.
  bool route_ready = route.complete;
#if defined(Q3X_ENABLE_MTP_SERVICE_ADMISSION)
  if (committed_service_handoff && route.request_active && !route.complete &&
      route.error == PrefillRouteEvidenceError::kNone && route.completed_layer_passes > 0) {
    route_ready = true;
    for (const auto& op : route.operators) route_ready &= op.forbidden_hits == 0;
    for (auto hits : route.forbidden_boundary_hits) route_ready &= hits == 0;
  }
#else
  (void)committed_service_handoff;
#endif
  if (active_ || target_.poisoned_ || target_.whole_request_prefill_active() ||
      !prompt_hidden_ || !prompt || count == 0 ||
      count != state_.current_position() || count > state_.max_sequence_length() ||
      !route_ready) return false;
  for (std::uint32_t i = 0; i < count; ++i) if (prompt[i] >= V) return false;
  const auto& residual = target_.layer_major_request_views_->prompt_residual_bf16;
  if (residual.columns != H || residual.row_stride_elements != H ||
      residual.row_capacity < count || !residual.storage.device_data) return false;
  target_.request_reuse_boundary_ = 2;
  target_.committed_request_positions_ = 0;
  seed_cache_valid_ = false;
  entry_snapshots_ = direct_gdn_rows_ = seed_kv_reuses_ = 0;
  const auto fail = [this]() noexcept { (void)abort(); return false; };
  auto stream = static_cast<cudaStream_t>(target_.stream_);
  // Same independent one-row reduction as the ordinary final-row handoff;
  // only the grid is widened. Residual and persistent target state stay intact.
  if (launch_headwise_centered_rms_norm_reference_cuda(
          static_cast<const std::uint16_t*>(residual.storage.device_data),
          target_.weights_->final_norm().data, count, H, 1.e-6F,
          prompt_hidden_, stream) != cudaSuccess ||
      cudaMemcpyAsync(target_.views_.hidden[1], prompt_hidden_ + (count - 1ULL) * H,
                      2 * H, cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess || !draft_.reset()) return fail();
  if (!draft_.initialize_kv(prompt + 1, prompt_hidden_, count - 1,
                            cancel, cancel_context)) return fail();
  target_.trace_valid_ = false;
  target_.retained_prefill_hidden_valid_ = false;
  return draft_.position() + 1 == count;
}
const std::uint16_t* TargetTransaction::hidden(const ReferenceRunner& runner) noexcept {
  return runner.views_.hidden[1];
}
const std::uint16_t* TargetTransaction::logits(const ReferenceRunner& runner) noexcept {
  return reinterpret_cast<const std::uint16_t*>(runner.views_.fp32_scratch);
}
const float* TargetTransaction::cosines(const ReferenceRunner& runner) noexcept {
  return runner.views_.rope_cos;
}
const float* TargetTransaction::sines(const ReferenceRunner& runner) noexcept {
  return runner.views_.rope_sin;
}
bool TargetTransaction::snapshot(std::uint32_t slot) noexcept {
  auto stream = static_cast<cudaStream_t>(target_.stream_);
  auto* dst = static_cast<char*>(snapshots_) + slot_bytes_ * slot;
  if (slot == 0) {
    // Entry recurrent state and logits have no consumer: abort always poisons.
    if (cudaMemcpyAsync(dst + recurrent_bytes_, hidden(target_), 2 * H,
                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) return false;
    ++entry_snapshots_;
    return true;
  }
  return cudaMemcpyAsync(dst, static_cast<char*>(state_.arena_data()) + recurrent_offset_,
                         recurrent_bytes_, cudaMemcpyDeviceToDevice, stream) == cudaSuccess &&
         cudaMemcpyAsync(dst + recurrent_bytes_, hidden(target_), 2 * H,
                         cudaMemcpyDeviceToDevice, stream) == cudaSuccess &&
         cudaMemcpyAsync(dst + recurrent_bytes_ + 2 * H, logits(target_), 2 * V,
                         cudaMemcpyDeviceToDevice, stream) == cudaSuccess &&
         cudaStreamSynchronize(stream) == cudaSuccess;
}
bool TargetTransaction::begin(std::uint32_t rows) noexcept {
  if (active_ || target_.poisoned_ || target_.whole_request_prefill_active() ||
      rows < 1 || rows > 4 || state_.current_position() == 0 ||
      draft_.position() + 1 != state_.current_position() ||
      rows > state_.max_sequence_length() - state_.current_position()) return false;
  active_ = true;
  seed_cache_valid_ = false;
  verified_ = false;
  committed_ = 0;
  entry_position_ = state_.current_position();
  rows_ = rows;
  target_.request_reuse_boundary_ = 2;  // kUncertain: never grant prefix reset authority.
  target_.committed_request_positions_ = 0;
  return snapshot(0);
}
bool TargetTransaction::propose(std::uint32_t seed, std::uint32_t count,
                                std::uint32_t* draft) noexcept {
  if (!active_ || verified_ || count + 1 != rows_ || draft == nullptr ||
      draft_.position() + 1 != entry_position_) return false;
  auto* input_hidden = reinterpret_cast<const std::uint16_t*>(
      static_cast<const char*>(snapshots_) + recurrent_bytes_);
  for (std::uint32_t i = 0; i < count; ++i) {
    if (!draft_.step(i == 0 ? seed : draft[i - 1],
                     i == 0 ? input_hidden : draft_.hidden(), true, draft[i])) return false;
    if (i == 0) { cached_seed_ = seed; seed_cache_valid_ = true; }
  }
  return true;
}
bool TargetTransaction::propose_bounded(std::uint32_t seed, std::uint32_t maximum,
    std::uint32_t* draft, std::uint32_t& actual) noexcept {
  actual = 0;
  if (!active_ || verified_ || maximum == 0 || maximum + 1 != rows_ ||
      draft == nullptr || draft_.position() + 1 != entry_position_) return false;
  const auto* input_hidden = reinterpret_cast<const std::uint16_t*>(
      static_cast<const char*>(snapshots_) + recurrent_bytes_);
  for (std::uint32_t i = 0; i < maximum; ++i) {
    float confidence = 1;
    if (!draft_.step(i == 0 ? seed : draft[i - 1],
        i == 0 ? input_hidden : draft_.hidden(), true, draft[i],
        i + 1 < maximum ? &confidence : nullptr)) return false;
    if (i == 0) { cached_seed_ = seed; seed_cache_valid_ = true; }
    ++actual;
    if (confidence < 0.4F) break;
  }
  rows_ = actual + 1;
  return true;
}
bool TargetTransaction::verify(std::uint32_t seed, const std::uint32_t* draft,
                               std::uint32_t count, std::uint32_t* predictions) noexcept {
  if (!active_ || verified_ || count + 1 != rows_ || !draft || !predictions) return false;
  ReferenceStepOptions options;
  options.compute_logits = true;
  options.logits_mode = ReferenceLogitsMode::kPredictedTokenOnly;
  for (std::uint32_t i = 0; i <= count; ++i)
    inputs_[i] = i == 0 ? seed : draft[i - 1];
  if (multirow_) {
    if (!verify_multirow(predictions)) return false;
  } else {
    for (std::uint32_t i = 0; i <= count; ++i) {
      const auto step = target_.step(inputs_[i], options);
      if (!step || !step.value->prediction || !snapshot(i + 1)) return false;
      predictions[i] = predictions_[i] = step.value->prediction->predicted_token_id;
    }
  }
  // The first proposal already used the immutable target entry hidden.
  // All later recursive-draft rows remain invalid until target reconciliation.
  seed_cache_valid_ = seed_cache_valid_ && count != 0 && cached_seed_ == seed;
  if (!draft_.rewind(entry_position_ - (seed_cache_valid_ ? 0 : 1))) return false;
  verified_ = true;
  return true;
}
bool TargetTransaction::commit_prefix(std::uint32_t rows, std::uint32_t pending) noexcept {
  if (!active_ || !verified_ || rows != committed_ + 1 || rows > rows_ ||
      predictions_[rows - 1] != pending ||
      (rows > 1 && inputs_[rows - 1] != predictions_[rows - 2])) return false;
  auto* src = static_cast<const char*>(snapshots_) + rows * slot_bytes_;
  auto stream = static_cast<cudaStream_t>(target_.stream_);
  if (cudaMemcpyAsync(static_cast<char*>(state_.arena_data()) + recurrent_offset_, src,
                       recurrent_bytes_, cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
      cudaMemcpyAsync(target_.views_.hidden[1], src + recurrent_bytes_, 2 * H,
                       cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
      cudaMemcpyAsync(target_.views_.fp32_scratch, src + recurrent_bytes_ + 2 * H,
                       2 * V, cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) return false;
  const auto* previous_hidden = reinterpret_cast<const std::uint16_t*>(
      static_cast<const char*>(snapshots_) + (rows - 1) * slot_bytes_ + recurrent_bytes_);
  if (rows == 1 && seed_cache_valid_) {
    if (draft_.position() != entry_position_) return false;
    ++seed_kv_reuses_;
    seed_cache_valid_ = false;
  } else if (!draft_.append_kv(inputs_[rows - 1], previous_hidden)) return false;
  if (!state_.set_sequence_length(entry_position_ + rows)) return false;
  target_.trace_valid_ = false;
  target_.retained_prefill_hidden_valid_ = false;
  committed_ = rows;
  return true;
}
bool TargetTransaction::finish() noexcept {
  if (!active_ || !verified_ || committed_ == 0 ||
      state_.current_position() != entry_position_ + committed_ ||
      draft_.position() + 1 != state_.current_position()) return false;
  active_ = false;
  seed_cache_valid_ = false;
  verified_ = false;
  return true;
}
bool TargetTransaction::abort() noexcept {
  const auto status = cudaStreamSynchronize(static_cast<cudaStream_t>(target_.stream_));
  const bool draft_ok = draft_.poison();
  target_.poisoned_ = true;
  target_.trace_valid_ = false;
  target_.retained_prefill_hidden_valid_ = false;
  target_.request_reuse_boundary_ = 2;
  target_.committed_request_positions_ = 0;
  active_ = false;
  seed_cache_valid_ = false;
  verified_ = false;
  return status == cudaSuccess && draft_ok;
}

}  // namespace q3x::runtime::mtp_detail
