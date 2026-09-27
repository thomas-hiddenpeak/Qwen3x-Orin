#include "q3x/kernels/gdn_prefill_prompt_wide_chunk_graph_abi.h"
#include "q3x/kernels/sm87_bf16_ab_prefill.h"
#include "../src/kernels/sm87/gdn_prefill_chunk64_native_sm87.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void check(int status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(static_cast<cudaError_t>(status)));
}
struct Buffer {
  std::uint16_t* p = nullptr;
  std::size_t n;
  explicit Buffer(std::size_t elements, std::uint16_t value = 0) : n(elements) {
    check(cudaMalloc(&p, (n + 128) * 2));
    std::vector<std::uint16_t> data(n + 128, value);
    std::fill(data.begin() + n, data.end(), 0x7fc1);
    check(cudaMemcpy(p, data.data(), data.size() * 2, cudaMemcpyHostToDevice));
  }
  ~Buffer() { (void)cudaFree(p); }
  Buffer(const Buffer&) = delete;
  std::vector<std::uint16_t> read() const {
    std::vector<std::uint16_t> data(n + 128);
    check(cudaMemcpy(data.data(), p, data.size() * 2, cudaMemcpyDeviceToHost));
    for (std::size_t i = n; i < data.size(); ++i)
      if (data[i] != 0x7fc1) throw std::runtime_error("output guard overwritten");
    data.resize(n);
    return data;
  }
  void patterned(unsigned seed) {
    std::vector<std::uint16_t> data(n);
    for (auto& x : data) {
      seed = seed * 1664525U + 1013904223U;
      float value = static_cast<float>(static_cast<int>((seed >> 16) % 257U) - 128) / 512.0F;
      std::uint32_t bits; std::memcpy(&bits, &value, sizeof bits);
      x = static_cast<std::uint16_t>(bits >> 16);
    }
    check(cudaMemcpy(p, data.data(), n * 2, cudaMemcpyHostToDevice));
  }
};
void equal(const Buffer& a, const Buffer& b, const char* role) {
  if (a.read() != b.read()) throw std::runtime_error(role);
}
void ab(std::size_t tokens) {
  // Compare actual-length masked tiles against the identical arithmetic on
  // a complete M64 grid. Extra rows are independent, not prompt tokens.
  const auto padded = (tokens + 63) / 64 * 64;
  Buffer wa(48 * 5120), wb(48 * 5120), input(padded * 5120);
  wa.patterned(1); wb.patterned(2); input.patterned(3);
  Buffer a(tokens * 48), b(tokens * 48), full_a(padded * 48), full_b(padded * 48);
  check(q3x::kernels::launch_sm87_bf16_ab_prompt_wide_p40_cuda(wa.p, wb.p, input.p, tokens, a.p, b.p));
  check(q3x::kernels::launch_sm87_bf16_ab_prompt_wide_p40_cuda(wa.p, wb.p, input.p, padded, full_a.p, full_b.p));
  check(cudaDeviceSynchronize());
  auto aa = full_a.read(), bb = full_b.read(); aa.resize(tokens * 48); bb.resize(tokens * 48);
  if (aa != a.read() || bb != b.read()) throw std::runtime_error("A/B tail changed valid-row bits");
}
void gdn(std::size_t tokens) {
  namespace rt = q3x::runtime;
  namespace graph = rt::gdn_prefill_prompt_wide_chunk_graph_detail;
  const auto plan = q3x::kernels::make_gdn_prompt_wide_chunk_graph_workspace_plan(tokens);
  if (!plan.ok()) throw std::runtime_error("GDN geometry rejected");
  const auto padded = plan.padded_token_count;
  Buffer ws(plan.layout.total_bytes / 2), raw(padded * rt::kGdnQkvChannels);
  Buffer weight(rt::kGdnQkvChannels * rt::kGdnConvKernelWidth);
  Buffer hist(rt::kGdnQkvChannels * rt::kGdnConvHistoryWidth), hist_ref(hist.n);
  Buffer conv(tokens * rt::kGdnQkvChannels), conv_ref(raw.n), state(rt::kGdnStateElements), state_ref(state.n);
  Buffer a(padded * rt::kGdnValueHeadCount), b(a.n), alog(rt::kGdnValueHeadCount);
  Buffer bias(alog.n), norm(rt::kGdnHeadDimension, 0x3f80), z(padded * rt::kGdnVElements);
  Buffer output(tokens * rt::kGdnVElements), output_ref(z.n);
  raw.patterned(4); weight.patterned(5); hist.patterned(6); hist_ref.patterned(6);
  state.patterned(7); state_ref.patterned(7); a.patterned(8); b.patterned(9); z.patterned(10);
  check(graph::launch(ws.p, plan.layout.total_bytes, raw.p, tokens, weight.p, hist.p, conv.p,
      a.p, b.p, alog.p, bias.p, state.p, state.p, 1e-6F, norm.p, z.p, 1e-6F, output.p));
  // The tail oracle keeps the same single-call FP32 state lifetime. Real
  // extra rows in this test have beta=0 and delta-gate=0, so they are neutral
  // state transitions. Splitting at C512 would add a BF16 state publication
  // and is deliberately NOT used as an exact oracle for multi-chunk calls.
  std::vector<std::uint16_t> neutral((padded - tokens) * rt::kGdnValueHeadCount, 0xff80);
  if (!neutral.empty()) {
    check(cudaMemcpy(a.p + tokens * rt::kGdnValueHeadCount, neutral.data(), neutral.size() * 2, cudaMemcpyHostToDevice));
    check(cudaMemcpy(b.p + tokens * rt::kGdnValueHeadCount, neutral.data(), neutral.size() * 2, cudaMemcpyHostToDevice));
  }
  if (tokens <= 512) {
    const auto ref_bytes = rt::gdn_prefill_chunk64_native_detail::workspace_bytes();
    Buffer ref_ws(ref_bytes / 2);
    check(rt::gdn_prefill_chunk64_native_detail::launch_fused_conv_compact_qk_preprocess(
        ref_ws.p, ref_bytes, raw.p, tokens, weight.p, hist_ref.p, conv_ref.p, 1e-6F));
    check(rt::gdn_prefill_chunk64_native_detail::launch_qk_preprocessed(
        ref_ws.p, ref_bytes, conv_ref.p, tokens, a.p, b.p, alog.p, bias.p, state_ref.p, state_ref.p,
        1e-6F, norm.p, z.p, 1e-6F, output_ref.p));
    check(cudaDeviceSynchronize());
    (void)ref_ws.read();
    equal(hist, hist_ref, "convolution history differs");
  } else {
    check(graph::launch(ws.p, plan.layout.total_bytes, raw.p, padded, weight.p, hist_ref.p, conv_ref.p,
        a.p, b.p, alog.p, bias.p, state_ref.p, state_ref.p, 1e-6F, norm.p, z.p, 1e-6F, output_ref.p));
    auto raw_host = raw.read(), history = hist.read();
    for (std::size_t ch = 0; ch < rt::kGdnQkvChannels; ++ch)
      for (std::size_t t = 0; t < rt::kGdnConvHistoryWidth; ++t)
        if (history[ch * 3 + t] != raw_host[(tokens - 3 + t) * rt::kGdnQkvChannels + ch])
          throw std::runtime_error("tail advanced logical convolution history");
  }
  check(cudaDeviceSynchronize());
  auto conv_expected = conv_ref.read(), output_expected = output_ref.read();
  conv_expected.resize(conv.n); output_expected.resize(output.n);
  if (conv.read() != conv_expected) throw std::runtime_error("convolution differs");
  equal(state, state_ref, "GDN state differs");
  if (output.read() != output_expected) throw std::runtime_error("GDN output differs");
  (void)ws.read();
}
}
int main() {
  try {
    for (std::size_t m : {1U, 2U, 3U, 19U, 43U, 44U, 63U, 64U, 65U, 511U, 512U, 513U, 7999U, 8000U, 8001U, 8193U}) {
      ab(m); gdn(m);
      std::cout << "P" << m << " A/B + GDN output/state/conv bitwise pass\n" << std::flush;
    }
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
  return 0;
}
