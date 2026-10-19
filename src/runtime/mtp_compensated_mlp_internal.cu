#include "mtp_compensated_mlp_internal.h"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <limits>
namespace q3x::runtime::mtp_detail {
namespace compensated {
__device__ float load_bf16(std::uint16_t x) { return __uint_as_float(unsigned(x) << 16); }
__device__ std::uint16_t publish(float x) {
  const unsigned b=__float_as_uint(x);
  return (b&0x7fffffffU)>0x7f800000U ? (b>>16)|0x40U : (b+0x7fffU+((b>>16)&1))>>16;
}
__global__ void quantize(const std::uint16_t* input, unsigned k,
    std::int8_t* codes, float* scales) {
  __shared__ float maxima[128];
  const unsigned t=threadIdx.x, i=blockIdx.x*128+t;
  const float x=load_bf16(input[i]);
  const int bad=__syncthreads_or(!isfinite(x));
  maxima[t]=fabsf(x); __syncthreads();
  for(unsigned d=64;d;d>>=1){if(t<d)maxima[t]=fmaxf(maxima[t],maxima[t+d]);__syncthreads();}
  float s0=__fdiv_rn(maxima[0],127.f); if(s0==0)s0=1.f;
  if(bad)s0=CUDART_NAN_F;
  const int q0=bad?0:max(-127,min(127,__float2int_rn(__fdiv_rn(x,s0))));
  const float r=fmaf(-float(q0),s0,x);
  __syncthreads(); // All readers finish before maxima is repurposed.
  maxima[t]=fabsf(r); __syncthreads();
  for(unsigned d=64;d;d>>=1){if(t<d)maxima[t]=fmaxf(maxima[t],maxima[t+d]);__syncthreads();}
  float s1=__fdiv_rn(maxima[0],127.f);if(s1==0)s1=1.f;
  if(bad)s1=CUDART_NAN_F;
  const int q1=bad?0:max(-127,min(127,__float2int_rn(__fdiv_rn(r,s1))));
  codes[i]=static_cast<std::int8_t>(q0);codes[k+i]=static_cast<std::int8_t>(q1);
  if(t==0){scales[blockIdx.x]=s0;scales[k/128+blockIdx.x]=s1;}
}
__device__ unsigned offset(unsigned row,unsigned col){return row*128+(col^((row&7)*16));}
__device__ void copy16(void* dst,const void* src,unsigned bytes=16){
  const unsigned a=static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(a),"l"(src),"r"(bytes):"memory");
}
__device__ void stage(std::uint8_t* w,std::int8_t* x,
    const std::uint8_t* weights,const std::int8_t* codes,unsigned k,unsigned tile){
  for(unsigned i=threadIdx.x;i<64*8;i+=128){unsigned row=i/8,col=(i%8)*16;
    copy16(w+offset(row,col),weights+(blockIdx.x*64+row)*k+tile*128+col);}
  if(threadIdx.x<8*8){unsigned row=threadIdx.x/8,col=(threadIdx.x%8)*16;
    copy16(x+offset(row,col),row<2?codes+row*k+tile*128+col:codes,row<2?16:0);}
  asm volatile("cp.async.commit_group;":::"memory");
}
__global__ void project(const std::uint8_t* weights,const std::uint16_t* weight_scales,
    const std::int8_t* codes,const float* scales,unsigned k,std::uint16_t* output){
  __shared__ __align__(16) std::uint8_t w[2][64*128];
  __shared__ __align__(16) std::int8_t x[2][8*128];
  const unsigned lane=threadIdx.x%32,wr=(threadIdx.x/32)*16+lane/4;
  const unsigned xr=lane/4,part=(lane%4)*4,groups=k/128;
  float y0=0,y1=0;
  stage(w[0],x[0],weights,codes,k,0);stage(w[1],x[1],weights,codes,k,1);
  asm volatile("cp.async.wait_group 1;":::"memory");__syncthreads();
  for(unsigned g=0;g<groups;++g){
    const unsigned b=g&1; int c0=0,c1=0,c2=0,c3=0;
    #pragma unroll
    for(unsigned step=0;step<4;++step){
      const unsigned q=step*32+part;
      const unsigned a0=*reinterpret_cast<const unsigned*>(w[b]+offset(wr,q))^0x80808080U;
      const unsigned a1=*reinterpret_cast<const unsigned*>(w[b]+offset(wr+8,q))^0x80808080U;
      const unsigned a2=*reinterpret_cast<const unsigned*>(w[b]+offset(wr,q+16))^0x80808080U;
      const unsigned a3=*reinterpret_cast<const unsigned*>(w[b]+offset(wr+8,q+16))^0x80808080U;
      const unsigned b0=*reinterpret_cast<const unsigned*>(x[b]+offset(xr,q));
      const unsigned b1=*reinterpret_cast<const unsigned*>(x[b]+offset(xr,q+16));
      asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        :"+r"(c0),"+r"(c1),"+r"(c2),"+r"(c3)
        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
    if(lane%4==0){
      const float ws0=load_bf16(weight_scales[(blockIdx.x*64+wr)*groups+g]);
      const float ws1=load_bf16(weight_scales[(blockIdx.x*64+wr+8)*groups+g]);
      y0=fmaf(float(c0),__fmul_rn(scales[g],ws0),y0);
      y0=fmaf(float(c1),__fmul_rn(scales[groups+g],ws0),y0);
      y1=fmaf(float(c2),__fmul_rn(scales[g],ws1),y1);
      y1=fmaf(float(c3),__fmul_rn(scales[groups+g],ws1),y1);
    }
    __syncthreads();
    if(g+2<groups){stage(w[b],x[b],weights,codes,k,g+2);asm volatile("cp.async.wait_group 1;":::"memory");}
    else {asm volatile("cp.async.wait_group 0;":::"memory");}
    __syncthreads();
  }
  if(lane%4==0){output[blockIdx.x*64+wr]=publish(y0);output[blockIdx.x*64+wr+8]=publish(y1);}
}
struct Span{std::uintptr_t b,e;};
bool span(const void* p,std::size_t bytes,Span& s){
  const auto b=reinterpret_cast<std::uintptr_t>(p);
  if(!b||(b&15)||bytes>std::numeric_limits<std::uintptr_t>::max()-b)return false;
  s={b,b+bytes};return true;
}
bool separate(Span a,Span b){return a.e<=b.b||b.e<=a.b;}
}
int launch_mtp_compensated_quantize(const std::uint16_t* input,unsigned k,
    void* scratch,std::size_t bytes,void* stream) noexcept{
  using namespace compensated;Span a,b;
  if((k!=5120&&k!=17408)||bytes<mtp_compensated_bytes(k)||
     !span(input,2ULL*k,a)||!span(scratch,mtp_compensated_bytes(k),b)||!separate(a,b))return cudaErrorInvalidValue;
  auto* codes=static_cast<std::int8_t*>(scratch);auto* scales=reinterpret_cast<float*>(codes+2*k);
  quantize<<<k/128,128,0,static_cast<cudaStream_t>(stream)>>>(input,k,codes,scales);
  return cudaGetLastError();
}
int launch_mtp_compensated_project(const std::uint8_t* weights,const std::uint16_t* scales,
    unsigned n,unsigned k,const void* scratch,std::size_t bytes,std::uint16_t* output,void* stream) noexcept{
  using namespace compensated;Span s[4];
  if(!((n==17408&&k==5120)||(n==5120&&k==17408))||bytes<mtp_compensated_bytes(k)||
     !span(weights,std::size_t(n)*k,s[0])||!span(scales,2ULL*n*(k/128),s[1])||
     !span(scratch,mtp_compensated_bytes(k),s[2])||!span(output,2ULL*n,s[3]))return cudaErrorInvalidValue;
  for(unsigned i=0;i<4;++i)for(unsigned j=0;j<i;++j)if(!separate(s[i],s[j]))return cudaErrorInvalidValue;
  const auto* codes=static_cast<const std::int8_t*>(scratch);
  project<<<n/64,128,0,static_cast<cudaStream_t>(stream)>>>(weights,scales,codes,reinterpret_cast<const float*>(codes+2*k),k,output);
  return cudaGetLastError();
}
}
