#include "runtime/mtp_gdn_state_internal.h"
#include "q3x/runtime/gdn_decode.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
struct Buffer {
  std::uint16_t* allocation = nullptr;
  std::size_t size;
  explicit Buffer(std::size_t n) : size(n) {
    check(cudaMalloc(&allocation, (size + 64) * 2));
    check(cudaMemset(allocation, 0xa5, (size + 64) * 2));
  }
  ~Buffer() { if (allocation) cudaFree(allocation); }
  std::uint16_t* data() const { return allocation + 32; }
  void fill(unsigned seed, float magnitude = 0.25F) {
    std::vector<std::uint16_t> values(size);
    for (std::size_t i = 0; i < size; ++i) {
      const int v = static_cast<int>((i * 73 + seed * 31) % 257) - 128;
      values[i] = __bfloat16_as_ushort(__float2bfloat16(v * magnitude / 128));
    }
    check(cudaMemcpy(data(), values.data(), size * 2, cudaMemcpyHostToDevice));
  }
  std::vector<std::uint16_t> read() const {
    std::vector<std::uint16_t> values(size + 64);
    check(cudaMemcpy(values.data(), allocation, values.size() * 2, cudaMemcpyDeviceToHost));
    for (unsigned i = 0; i < 32; ++i)
      require(values[i] == 0xa5a5 && values[values.size()-1-i] == 0xa5a5, "guard modified");
    return {values.begin()+32, values.end()-32};
  }
};
}  // namespace

int main() {
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  const std::uint16_t** handle = nullptr;
  try {
    Buffer qkv(10240), a(48), b(48), log(48), bias(48), norm(128), gate(6144);
    Buffer state0(48*128*128), state1(48*128*128), selected(48*128*128);
    Buffer reference(48*128*128), output(6144), expected(6144);
    check(cudaStreamCreate(&stream));
    check(cudaMalloc(&handle, sizeof(*handle)));
    auto candidate = [&](const std::uint16_t* const* address) {
      return static_cast<cudaError_t>(q3x::runtime::mtp_detail::launch_mtp_gdn_selected_state(
          qkv.data(), a.data(), b.data(), log.data(), bias.data(), address,
          selected.data(), norm.data(), gate.data(), output.data(), stream));
    };
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    require(candidate(nullptr) == cudaErrorInvalidValue, "null handle accepted");
    require(candidate(reinterpret_cast<const std::uint16_t* const*>(
                reinterpret_cast<char*>(handle)+1)) == cudaErrorInvalidValue,
            "misaligned handle accepted");
    require(candidate(reinterpret_cast<const std::uint16_t* const*>(selected.data())) ==
                cudaErrorInvalidValue, "overlapping handle accepted");
    check(cudaStreamEndCapture(stream, &graph));
    std::size_t nodes = 0;
    check(cudaGraphGetNodes(graph, nullptr, &nodes));
    require(nodes == 0, "invalid call enqueued work");
    check(cudaGraphDestroy(graph)); graph = nullptr;

    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    check(candidate(handle));
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphInstantiateWithFlags(&executable, graph, 0));
    unsigned cases = 0, replays = 0, aliases = 0;
    for (unsigned iteration = 0; iteration < 12; ++iteration) {
      qkv.fill(iteration); a.fill(iteration+1, 4.F); b.fill(iteration+2, 4.F);
      log.fill(iteration+3); bias.fill(iteration+4); norm.fill(iteration+5, 1.F);
      gate.fill(iteration+6, 2.F);
      state0.fill(iteration+7); state1.fill(iteration+8); selected.fill(iteration+9);
      // The third source exactly aliases the captured output. No graph node
      // update is performed: only the device pointer slot changes.
      const std::uint16_t* source = iteration%3 == 0 ? state0.data() :
          iteration%3 == 1 ? state1.data() : selected.data();
      check(cudaMemcpyAsync(handle, &source, sizeof(source), cudaMemcpyHostToDevice, stream));
      check(static_cast<cudaError_t>(q3x::runtime::launch_gated_delta_net_update_plain_rms_norm_silu_gate_cuda(
          qkv.data(), a.data(), b.data(), log.data(), bias.data(), source,
          reference.data(), 1.e-6F, norm.data(), gate.data(), 48, 128,
          1.e-6F, expected.data(), {}, stream)));
      if (iteration < 3) check(candidate(handle));
      else { check(cudaGraphLaunch(executable, stream)); ++replays; }
      check(cudaStreamSynchronize(stream));
      require(reference.read() == selected.read(), "complete GDN state differs");
      require(expected.read() == output.read(), "complete GDN output differs");
      for (const Buffer* value : {&qkv, &a, &b, &log, &bias, &norm, &gate, &state0, &state1})
        (void)value->read();
      ++cases; aliases += source == selected.data();
    }
    check(cudaGraphExecDestroy(executable)); executable = nullptr;
    check(cudaGraphDestroy(graph)); graph = nullptr;
    check(cudaFree(handle)); handle = nullptr;
    check(cudaStreamDestroy(stream)); stream = nullptr;
    std::printf("complete_state_output_cases=%u graph_replays=%u exact_alias_cases=%u invalid_zero_enqueue=3 guards=pass\n", cases, replays, aliases);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    if (stream) cudaStreamSynchronize(stream);
    if (executable) cudaGraphExecDestroy(executable);
    if (graph) cudaGraphDestroy(graph);
    if (handle) cudaFree(handle);
    if (stream) cudaStreamDestroy(stream);
    return 1;
  }
}
