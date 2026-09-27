#include "decode_fused_gqa_internal.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace fd = q3x::runtime::fused_decode;
void check(int status) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(static_cast<cudaError_t>(status)));
    std::exit(1);
  }
}
int main() {
  check(fd::prepare());
  cudaStream_t stream{};
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  constexpr std::size_t qn = 24 * 256;
  constexpr std::size_t kn = fd::kMaximumSequence * 4 * 256;
  std::uint16_t *q{}, *k{}, *v{}, *o{};
  void* workspace{};
  check(cudaMalloc(&q, qn * 2)); check(cudaMalloc(&k, kn * 2));
  check(cudaMalloc(&v, kn * 2)); check(cudaMalloc(&o, (qn + 128) * 2));
  check(cudaMalloc(&workspace, fd::kWorkspaceBytes + 256));
  check(cudaMemsetAsync(q, 0, qn * 2, stream));
  check(cudaMemsetAsync(k, 0, kn * 2, stream));
  // Zero scores have uniform attention; head-specific values detect GQA
  // mapping and position-dependent values detect partition/tail coverage.
  std::vector<__nv_bfloat16> values(kn);
  for (std::size_t p = 0; p < fd::kMaximumSequence; ++p)
    for (std::size_t h = 0; h < 4; ++h)
      for (std::size_t d = 0; d < 256; ++d)
        values[p * 1024 + h * 256 + d] = __float2bfloat16(
            static_cast<float>(h) + static_cast<float>(p % 13) / 16.0f);
  check(cudaMemcpyAsync(v, values.data(), kn * 2, cudaMemcpyHostToDevice, stream));
  std::vector<std::uint16_t> result(qn + 128), repeat(qn + 128);
  for (std::size_t s : {512U, 513U, 2047U, 8192U, 40000U, 44095U}) {
    std::fprintf(stderr,"checking sequence=%zu\n",s);
    check(cudaMemsetAsync(o, 0xa5, (qn + 128) * 2, stream));
    check(cudaMemsetAsync(workspace, 0xa5, fd::kWorkspaceBytes + 256, stream));
    check(fd::launch(q,k,v,s,workspace,fd::kWorkspaceBytes,o,stream));
    check(cudaMemcpyAsync(result.data(),o,result.size()*2,cudaMemcpyDeviceToHost,stream));
    check(cudaStreamSynchronize(stream));
    double average = 0;
    for (std::size_t p = 0; p < s; ++p) average += static_cast<double>(p%13)/16;
    average /= s;
    for (std::size_t i = 0; i < qn; ++i) {
      const float actual = __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&result[i]));
      const double expected = static_cast<double>(i/256/6) + average;
      // Final BF16 publication; this is numerical admission only.
      if (!std::isfinite(actual) || std::abs(actual-expected) > 0.03125) return 2;
    }
    for (std::size_t i=qn; i<result.size(); ++i) if (result[i]!=0xa5a5) return 3;
    std::vector<unsigned char> guard(256);
    check(cudaMemcpy(guard.data(),static_cast<unsigned char*>(workspace)+fd::kWorkspaceBytes,
                     256,cudaMemcpyDeviceToHost));
    for (auto x : guard) if (x != 0xa5) return 4;
    cudaGraph_t graph{}; cudaGraphExec_t exec{};
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    check(fd::launch(q,k,v,s,workspace,fd::kWorkspaceBytes,o,stream));
    check(cudaStreamEndCapture(stream,&graph));
    check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    for (int n=0;n<2;++n) {
      check(cudaGraphLaunch(exec,stream));
      check(cudaMemcpyAsync(repeat.data(),o,repeat.size()*2,cudaMemcpyDeviceToHost,stream));
      check(cudaStreamSynchronize(stream));
      if (result!=repeat) return 5;
    }
    check(cudaGraphExecDestroy(exec)); check(cudaGraphDestroy(graph));
  }
  // Cancellation across partitions exposes premature BF16 publication.
  // The first 256 values average 1 + 2^-8 (a BF16 midpoint); the next
  // 256 average -1. The correct final BF16 mean is exactly 2^-9, while
  // rounding each partition first destroys the residual and returns zero.
  constexpr std::size_t cancellation_sequence = 512;
  for (std::size_t p=0; p<cancellation_sequence; ++p)
    for (std::size_t i=0; i<1024; ++i)
      values[p*1024+i] = __float2bfloat16(
          p < 128 ? 1.0f : (p < 256 ? 1.0078125f : -1.0f));
  check(cudaMemcpyAsync(v,values.data(),cancellation_sequence*1024*2,
                        cudaMemcpyHostToDevice,stream));
  check(fd::launch(q,k,v,cancellation_sequence,workspace,fd::kWorkspaceBytes,o,stream));
  check(cudaMemcpyAsync(result.data(),o,qn*2,cudaMemcpyDeviceToHost,stream));
  check(cudaStreamSynchronize(stream));
  const auto expected_cancellation = __float2bfloat16(0.001953125f);
  for (std::size_t i=0; i<qn; ++i)
    if (result[i] != *reinterpret_cast<const std::uint16_t*>(&expected_cancellation)) {
      std::fprintf(stderr,"partition cancellation residual lost at %zu\n",i);
      return 10;
    }
  // Independent FP64 softmax/PV oracle on deterministic nonuniform BF16
  // operands. No scalar CUDA or FlashInfer routine supplies expected values.
  constexpr std::size_t rs = 513;
  std::vector<__nv_bfloat16> hq(qn), hk(rs*1024), hv(rs*1024);
  for (std::size_t i=0;i<hq.size();++i)
    hq[i]=__float2bfloat16(static_cast<float>(static_cast<int>((i*17)%61)-30)/32);
  for (std::size_t i=0;i<hk.size();++i) {
    hk[i]=__float2bfloat16(static_cast<float>(static_cast<int>((i*13)%67)-33)/32);
    hv[i]=__float2bfloat16(static_cast<float>(static_cast<int>((i*23)%71)-35)/32);
  }
  check(cudaMemcpyAsync(q,hq.data(),qn*2,cudaMemcpyHostToDevice,stream));
  check(cudaMemcpyAsync(k,hk.data(),hk.size()*2,cudaMemcpyHostToDevice,stream));
  check(cudaMemcpyAsync(v,hv.data(),hv.size()*2,cudaMemcpyHostToDevice,stream));
  check(fd::launch(q,k,v,rs,workspace,fd::kWorkspaceBytes,o,stream));
  check(cudaMemcpyAsync(result.data(),o,qn*2,cudaMemcpyDeviceToHost,stream));
  check(cudaStreamSynchronize(stream));
  double error=0, norm=0;
  for (std::size_t h=0;h<24;++h) {
    std::vector<double> weights(rs);
    double max_score=-1e300;
    for (std::size_t pos=0;pos<rs;++pos) {
      double dot=0;
      for (std::size_t d=0;d<256;++d)
        dot+=static_cast<double>(__bfloat162float(hq[h*256+d]))*
             __bfloat162float(hk[pos*1024+(h/6)*256+d]);
      weights[pos]=dot/16; max_score=std::max(max_score,weights[pos]);
    }
    double den=0;
    for (auto& w:weights) {w=std::exp(w-max_score);den+=w;}
    for (std::size_t d=0;d<256;++d) {
      double expected=0;
      for (std::size_t pos=0;pos<rs;++pos)
        expected+=weights[pos]*__bfloat162float(hv[pos*1024+(h/6)*256+d]);
      expected/=den;
      const auto actual=__bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&result[h*256+d]));
      if (!std::isfinite(actual)) return 7;
      error+=(actual-expected)*(actual-expected); norm+=expected*expected;
    }
  }
  const double relative=std::sqrt(error/norm);
  std::printf("independent_fp64_relative_l2=%.9g\n",relative);
  // One BF16 unit roundoff for this bounded admission; not a model threshold.
  if (relative > 1.0/128.0) return 8;
  std::fill(hv.begin(),hv.end(),__float2bfloat16(NAN));
  check(cudaMemcpyAsync(v,hv.data(),hv.size()*2,cudaMemcpyHostToDevice,stream));
  check(fd::launch(q,k,v,rs,workspace,fd::kWorkspaceBytes,o,stream));
  check(cudaMemcpyAsync(result.data(),o,qn*2,cudaMemcpyDeviceToHost,stream));
  check(cudaStreamSynchronize(stream));
  for (std::size_t i=0;i<qn;++i)
    if (std::isfinite(__bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&result[i])))) return 9;
  if (fd::launch(q,k,v,511,workspace,fd::kWorkspaceBytes,o,stream)!=cudaErrorInvalidValue ||
      fd::launch(q,k,v,44096,workspace,fd::kWorkspaceBytes,o,stream)!=cudaErrorInvalidValue ||
      fd::launch(q,k,v,512,workspace,fd::kWorkspaceBytes-1,o,stream)!=cudaErrorInvalidValue ||
      fd::launch(q,k,v,512,workspace,fd::kWorkspaceBytes,q,stream)!=cudaErrorInvalidValue ||
      fd::launch(q,k,k,512,workspace,fd::kWorkspaceBytes,o,stream)!=cudaErrorInvalidValue)
    return 6;
  check(cudaFree(workspace)); check(cudaFree(o)); check(cudaFree(v));
  check(cudaFree(k)); check(cudaFree(q)); check(cudaStreamDestroy(stream));
  std::puts("PASS fused Decode bounds, GQA mapping, tails, guards, Graph repeatability");
}
