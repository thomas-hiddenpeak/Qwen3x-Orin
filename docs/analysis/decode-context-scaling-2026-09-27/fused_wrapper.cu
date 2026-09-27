// Research-only integration harness. FlashInfer headers retain Apache-2.0 licenses.
// Does not implement the public probabilities-scratch postcondition.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <array>
using U=std::uint16_t;
#define ARGS const U* q,const U* k,const U* v,std::size_t qh,std::size_t kh,std::size_t s,std::size_t d,float scale,float* scratch,std::size_t cap,U* out,void* opaque
#define PASS q,k,v,qh,kh,s,d,scale,scratch,cap,out,opaque
extern "C" int original(ARGS) asm("__real__ZN3q3x7runtime35launch_gqa_attention_reference_cudaEPKtS2_S2_mmmmfPfmPtPv");
extern "C" int wrapped(ARGS) asm("__wrap__ZN3q3x7runtime35launch_gqa_attention_reference_cudaEPKtS2_S2_mmmmfPfmPtPv");
static unsigned long hits=0,shadow_hits=0;
__attribute__((destructor)) static void receipt(){std::fprintf(stderr,"RESEARCH_FUSED_GQA hits=%lu shadow_hits=%lu numerical_qualified=false production_eligible=false\n",hits,shadow_hits);}
static int fused(ARGS){
  using P=flashinfer::SinglePrefillParams<__nv_bfloat16,__nv_bfloat16,__nv_bfloat16>;
  auto st=static_cast<cudaStream_t>(opaque);
  P p((__nv_bfloat16*)q,(__nv_bfloat16*)k,(__nv_bfloat16*)v,nullptr,(__nv_bfloat16*)out,nullptr,nullptr,24,4,1,(std::uint32_t)s,24*256,256,4*256,256,256,-1,0.0f,scale,1.0f,10000.0f);
  try{return (int)flashinfer::SinglePrefillWithKVCacheDispatched<256,256,flashinfer::PosEncodingMode::kNone,false,flashinfer::MaskMode::kNone,flashinfer::DefaultAttention<false,false,false,false>>(p,(__nv_bfloat16*)scratch,st);}
  catch(const std::exception& e){std::fprintf(stderr,"RESEARCH_FUSED_ERROR %s\n",e.what());return (int)cudaErrorUnknown;}
}
static float decode(U x){std::uint32_t b=(std::uint32_t)x<<16;float f;std::memcpy(&f,&b,4);return f;}
extern "C" int wrapped(ARGS){
  if(qh!=24||kh!=4||d!=256||s<512||s>44095||scale!=0.0625f)return original(PASS);
  if(!q||!k||!v||!out||!scratch||cap<24*s)return (int)cudaErrorInvalidValue;
  auto st=static_cast<cudaStream_t>(opaque);cudaStreamCaptureStatus cs;
  if(cudaStreamIsCapturing(st,&cs)!=cudaSuccess)return (int)cudaErrorUnknown;
  if(cs!=cudaStreamCaptureStatusNone)return original(PASS);
  if(((s+255)/256)*(24*256*2+24*4)>cap*sizeof(float))return (int)cudaErrorInvalidValue;
  bool shadow=std::getenv("DECODE_RESEARCH_SHADOW")!=nullptr;
  if(shadow){
    int r=original(PASS);if(r)return r;
    if(s!=40000&&s!=40001)return 0;
    std::array<U,6144> a,b,c;
    if(cudaMemcpyAsync(a.data(),out,sizeof(a),cudaMemcpyDeviceToHost,st)!=cudaSuccess||cudaStreamSynchronize(st)!=cudaSuccess)return (int)cudaErrorUnknown;
    r=fused(PASS);if(r)return r;
    if(cudaMemcpyAsync(b.data(),out,sizeof(b),cudaMemcpyDeviceToHost,st)!=cudaSuccess||cudaStreamSynchronize(st)!=cudaSuccess)return (int)cudaErrorUnknown;
    r=fused(PASS);if(r)return r;
    if(cudaMemcpyAsync(c.data(),out,sizeof(c),cudaMemcpyDeviceToHost,st)!=cudaSuccess||cudaStreamSynchronize(st)!=cudaSuccess)return (int)cudaErrorUnknown;
    double err=0,norm=0,maxabs=0;int different=0,replay=0,nonfinite=0;
    for(int i=0;i<6144;i++){double av=decode(a[i]),bv=decode(b[i]),diff=av-bv;err+=diff*diff;norm+=av*av;maxabs=std::fmax(maxabs,std::fabs(diff));different+=(a[i]!=b[i]);replay+=(b[i]!=c[i]);nonfinite+=!std::isfinite(bv);}
    std::fprintf(stderr,"RESEARCH_SHADOW seq=%zu call=%lu nrmse=%.9g maxabs=%.9g bf16_diff=%d replay_diff=%d nonfinite=%d\n",s,shadow_hits++,std::sqrt(err/std::fmax(norm,1e-30)),maxabs,different,replay,nonfinite);
    if(nonfinite||replay||std::sqrt(err/std::fmax(norm,1e-30))>0.02)return (int)cudaErrorInvalidValue;
    if(cudaMemcpyAsync(out,a.data(),sizeof(a),cudaMemcpyHostToDevice,st)!=cudaSuccess||cudaStreamSynchronize(st)!=cudaSuccess)return (int)cudaErrorUnknown;
    return 0;
  }
  hits++;
  return fused(PASS);
}
