// Reuse the whole-core state capture, comparing each Decode Attention call on
// the actual whole-core-produced Q/K/V. This does not compare Prefill backends.
#define main whole_core_capture_main
#include "reference_whole_core_state_capture_test.cpp"
#undef main
#include "decode_fused_gqa_internal.h"
#include "q3x/runtime/decode_ops.h"

namespace {
struct ExactDecodeCapture {
  bool scalar = false;
  std::size_t calls = 0U;
};

int compare_attention(const rt::fused_decode::Observation& view,
                      void* context) noexcept {
  auto& capture = *static_cast<ExactDecodeCapture*>(context);
  constexpr std::size_t count = 24U * 256U;
  std::array<std::uint16_t, count> candidate{}, scalar{};
  auto stream = static_cast<cudaStream_t>(view.stream);
  auto status = cudaStreamSynchronize(stream);
  if (status != cudaSuccess) return static_cast<int>(status);
  status = cudaMemcpy(candidate.data(), view.output, sizeof(candidate),
                      cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) return static_cast<int>(status);
  const int launched = rt::launch_gqa_attention_reference_cuda(
      view.query, view.key, view.value, 24U, 4U, view.sequence, 256U,
      1.0F / 16.0F, view.workspace, view.workspace_elements,
      view.output, view.stream);
  if (launched != 0) return launched;
  status = cudaStreamSynchronize(stream);
  if (status != cudaSuccess) return static_cast<int>(status);
  status = cudaMemcpy(scalar.data(), view.output, sizeof(scalar),
                      cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) return static_cast<int>(status);
  if (candidate != scalar) return static_cast<int>(cudaErrorInvalidValue);
  ++capture.calls;
  if (!capture.scalar) {
    status = cudaMemcpyAsync(view.output, candidate.data(), sizeof(candidate),
                             cudaMemcpyHostToDevice, stream);
    if (status != cudaSuccess) return static_cast<int>(status);
    status = cudaStreamSynchronize(stream);
  }
  return static_cast<int>(status);
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 5 || (std::string_view(argv[4]) != "scalar" &&
                    std::string_view(argv[4]) != "fused")) {
    std::cerr << "usage: MODEL REQUEST OUTPUT scalar|fused\n";
    return 2;
  }
  ExactDecodeCapture capture;
  capture.scalar = std::string_view(argv[4]) == "scalar";
  rt::fused_decode::set_observer(compare_attention, &capture);
  char route_flag[] = "--route", route[] = "wholecore";
  char* args[] = {argv[0], argv[1], argv[2], argv[3], route_flag, route};
  const int status = whole_core_capture_main(6, args);
  rt::fused_decode::set_observer(nullptr, nullptr);
  std::cerr << "exact_attention_calls=" << capture.calls << '\n';
  return status != 0 ? status : (capture.calls == 15U * 16U ? 0 : 1);
}
