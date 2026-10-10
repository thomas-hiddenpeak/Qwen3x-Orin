#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace q3x::runtime::mtp_detail {
namespace {
__device__ __forceinline__ float bf16(std::uint16_t x) {
  return __uint_as_float(static_cast<unsigned>(x) << 16);
}
__device__ __forceinline__ std::uint16_t rounded(float x) {
  unsigned b = __float_as_uint(x);
  if ((b & 0x7fffffffU) > 0x7f800000U) return (b >> 16) | 0x40U;
  return (b + 0x7fffU + ((b >> 16) & 1U)) >> 16;
}
__device__ __forceinline__ float fp8(unsigned bits) {
  const unsigned sign = (bits & 128U) << 24, mag = bits & 127U;
  const unsigned e = mag >> 3, m = mag & 7U;
  if (mag == 127) return __uint_as_float(sign | 0x7fc00000U);
  if (!e) {
    if (!m) return __uint_as_float(sign);
    const unsigned leading = m >= 4 ? 2 : (m >= 2 ? 1 : 0);
    return __uint_as_float(sign | ((118U + leading) << 23) |
                           ((m - (1U << leading)) << (23 - leading)));
  }
  return __uint_as_float(sign | ((120U + e) << 23) | (m << 20));
}
__device__ __forceinline__ float warp_sum(float x) {
  x += __shfl_down_sync(0xffffffffU, x, 16);
  x += __shfl_down_sync(0xffffffffU, x, 8);
  x += __shfl_down_sync(0xffffffffU, x, 4);
  x += __shfl_down_sync(0xffffffffU, x, 2);
  x += __shfl_down_sync(0xffffffffU, x, 1);
  return x;
}

// Preserve four independent scalar warp trees, but move their live roots
// into four eight-lane groups as the number of live leaves shrinks.
__device__ __forceinline__ float warp_sum_four(float s0, float s1,
                                               float s2, float s3) {
  const unsigned lane = threadIdx.x % 32;
  const float p0 = __shfl_xor_sync(0xffffffffU, s0, 16);
  const float p1 = __shfl_xor_sync(0xffffffffU, s1, 16);
  const float p2 = __shfl_xor_sync(0xffffffffU, s2, 16);
  const float p3 = __shfl_xor_sync(0xffffffffU, s3, 16);
  const float t0 = __fadd_rn(lane < 16 ? s0 : p2, lane < 16 ? p0 : s2);
  const float t1 = __fadd_rn(lane < 16 ? s1 : p3, lane < 16 ? p1 : s3);
  const float q0 = __shfl_xor_sync(0xffffffffU, t0, 8);
  const float q1 = __shfl_xor_sync(0xffffffffU, t1, 8);
  float sum = __fadd_rn((lane & 8) ? q1 : t0, (lane & 8) ? t1 : q0);
  sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffU, sum, 4, 8));
  sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffU, sum, 2, 8));
  sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffU, sum, 1, 8));
  return sum;
}

// Cache warming has no register result or correctness dependency. The demand
// loads below remain authoritative if the hardware ignores this hint.
__device__ __forceinline__ void prefetch_fp8_operand(const void* p) {
  asm volatile("prefetch.global.L1 [%0];" :: "l"(p) : "memory");
}

// Four output channels share vector activation loads. Each weight and scale
// serves all speculative rows; the per-row four-chain arithmetic is unchanged.
template<unsigned M, bool Sidecar>
__global__ __launch_bounds__(256, (M==2 ? 4 : (M==3 ? 3 : 2)))
void fp8_rows(const std::uint8_t* weights, float scale,
    const std::uint16_t* x, unsigned n, unsigned k, std::uint16_t* y) {
  __shared__ float table[256], partial[M][4][8];
  const unsigned tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  const unsigned slot = tid ^ (tid >> 5);
  table[slot] = fp8(tid);
  __syncthreads();
  const unsigned row = blockIdx.x * 4;
  float a[M][4][4] = {};
  for (unsigned base = tid * 4; base < k; base += 1024) {
    // Warm exactly the next existing K block; the tail issues no hint.
    if (base + 1024 < k) {
      const unsigned next = base + 1024;
      if constexpr (Sidecar) {
        prefetch_fp8_operand(reinterpret_cast<const uint4*>(weights) +
                             blockIdx.x * (k/4) + next/4);
      } else {
#pragma unroll
        for (unsigned r=0;r<4;++r)
          prefetch_fp8_operand(weights + (row+r)*k + next);
      }
#pragma unroll
      for (unsigned m=0;m<M;++m)
        prefetch_fp8_operand(x + m*k + next);
    }
    uint4 packed;
    if constexpr (Sidecar) {
      packed = __ldcs(reinterpret_cast<const uint4*>(weights) +
                      blockIdx.x * (k/4) + base/4);
    } else {
      packed.x = __ldcs(reinterpret_cast<const unsigned*>(weights + row*k + base));
      packed.y = __ldcs(reinterpret_cast<const unsigned*>(weights + (row+1)*k + base));
      packed.z = __ldcs(reinterpret_cast<const unsigned*>(weights + (row+2)*k + base));
      packed.w = __ldcs(reinterpret_cast<const unsigned*>(weights + (row+3)*k + base));
    }
    if constexpr (!Sidecar) {
      packed.x ^= (packed.x >> 5) & 0x07070707U;
      packed.y ^= (packed.y >> 5) & 0x07070707U;
      packed.z ^= (packed.z >> 5) & 0x07070707U;
      packed.w ^= (packed.w >> 5) & 0x07070707U;
    }
    const unsigned words[4] = {packed.x,packed.y,packed.z,packed.w};
    unsigned long long acts[M];
#pragma unroll
    for (unsigned m=0;m<M;++m)
      acts[m] = *reinterpret_cast<const unsigned long long*>(x + m*k + base);
#pragma unroll
    for (unsigned c=0;c<4;++c) {
      float av[M];
#pragma unroll
      for (unsigned m=0;m<M;++m) av[m] = bf16(acts[m] >> (c*16));
#pragma unroll
      for (unsigned r=0;r<4;++r) {
        const float w = table[(words[r] >> (c*8)) & 255];
#pragma unroll
        for (unsigned m=0;m<M;++m) a[m][r][c] = fmaf(w,av[m],a[m][r][c]);
      }
    }
  }
#pragma unroll
  for (unsigned m=0;m<M;++m) {
    float merged[4];
#pragma unroll
    for (unsigned r=0;r<4;++r)
      merged[r] = (a[m][r][0]+a[m][r][1])+(a[m][r][2]+a[m][r][3]);
    const float sum = warp_sum_four(merged[0], merged[1], merged[2], merged[3]);
    if (!(lane & 7)) partial[m][lane/8][warp] = sum;
  }
  __syncthreads();
  if (warp < M) {
    const unsigned r = lane/8;
    float sum = partial[warp][r][lane%8];
    // The original 32-lane tree adds zeros at offsets 16 and 8 first.
    sum = __fadd_rn(sum, 0.f);
    sum = __fadd_rn(sum, 0.f);
    sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffU, sum, 4, 8));
    sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffU, sum, 2, 8));
    sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffU, sum, 1, 8));
    if (!(lane & 7)) y[warp*n+row+r] = rounded(sum * scale);
  }
}

// Reconstruct exactly the existing [row-quad][K512] six-bit scale feed.
__device__ __forceinline__ unsigned scale6_codes(const std::uint8_t* scales,
    unsigned scale_base, unsigned row, unsigned tile, unsigned k, unsigned lane) {
  const auto* words = reinterpret_cast<const unsigned*>(scales) +
                     ((row/4)*(k/512)+tile)*24;
  const unsigned resident = lane<24 ? __ldcs(words+lane) : 0;
  const unsigned bit = ((lane/2)*8+(lane&1)*4)*6, shift=bit&31;
  const unsigned lo = __shfl_sync(0xffffffffU,resident,bit/32);
  const unsigned hi = __shfl_sync(0xffffffffU,resident,bit/32+1);
  unsigned deltas = lo >> shift;
  if (shift>8) deltas |= hi << (32-shift);
  deltas &= 0xffffffU;
  return ((deltas&63) | (((deltas>>6)&63)<<8) |
          (((deltas>>12)&63)<<16) | (((deltas>>18)&63)<<24)) + scale_base*0x01010101U;
}

__device__ __constant__ float nv_values[16]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,
                                            -0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
// Products of finite E2M1 and E4M3 operands fit exactly in BF16.
// The negative-zero addend preserves the sign of a zero multiplication.
__device__ __forceinline__ unsigned scaled_weight_pair(unsigned pair,
                                                       unsigned scale) {
  unsigned result;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(result)
      : "r"(pair), "r"(scale), "r"(0x80008000U));
  return result;
}
template<unsigned M, unsigned Layout>
__global__ void nvfp4_rows(const std::uint8_t* weights, const std::uint8_t* scales,
    unsigned scale_base, float scale, const std::uint16_t* x,
    unsigned n, unsigned k, std::uint16_t* y) {
  __shared__ unsigned table[256], values[256];
  // Share complete immutable inputs across the four original warp consumers.
  // Cap at three rows so M4 retains four-CTA shared-memory capacity.
  constexpr unsigned staged_rows = Layout==1 ? (M<3 ? M : 3) : 0;
  __shared__ __align__(16) std::uint16_t activation[staged_rows ? staged_rows*5120 : 1];
  if constexpr (Layout==1) {
    for(unsigned i=threadIdx.x;i<staged_rows*5120/8;i+=blockDim.x)
      reinterpret_cast<ulonglong2*>(activation)[i]=
          reinterpret_cast<const ulonglong2*>(x)[i];
  }
  for (unsigned i=threadIdx.x;i<256;i+=blockDim.x) {
    const unsigned scale = __float_as_uint(fp8(i)) >> 16;
    table[i] = scale | (scale << 16);
    values[i] = (__float_as_uint(nv_values[i & 15]) >> 16) |
                (__float_as_uint(nv_values[i >> 4]) & 0xffff0000U);
  }
  __syncthreads();
  const unsigned lane=threadIdx.x%32, row=(blockIdx.x*4+threadIdx.x/32)*4;
  float a[M][4][4]={};
  for(unsigned tile=0;tile<k/512;++tile) {
    unsigned phase_codes[2];
    if constexpr(Layout==2) {
      const unsigned local=scale6_codes(scales,scale_base,row,tile,k,lane);
      const unsigned partner=__shfl_xor_sync(0xffffffffU,local,1);
      phase_codes[0]=(lane&1)?partner:local;
      phase_codes[1]=(lane&1)?local:partner;
    }
#pragma unroll
    for(unsigned phase=0;phase<2;++phase) {
      const unsigned base=tile*512+phase*256+lane*8;
      uint4 packed; unsigned codes=0;
      if constexpr(Layout==1) {
        const auto* record=weights+(row/4)*11520+tile*1152+phase*576;
        packed=__ldcs(reinterpret_cast<const uint4*>(record)+lane);
        codes=__ldcs(reinterpret_cast<const unsigned*>(record+512)+lane/2);
      } else if constexpr(Layout==2) {
        packed=__ldcs(reinterpret_cast<const uint4*>(weights)+
                       (row/4)*(k/512)*64+tile*64+phase*32+lane);
        codes=phase_codes[phase];
      } else {
        packed.x=__ldcs(reinterpret_cast<const unsigned*>(weights+row*(k/2)+base/2));
        packed.y=__ldcs(reinterpret_cast<const unsigned*>(weights+(row+1)*(k/2)+base/2));
        packed.z=__ldcs(reinterpret_cast<const unsigned*>(weights+(row+2)*(k/2)+base/2));
        packed.w=__ldcs(reinterpret_cast<const unsigned*>(weights+(row+3)*(k/2)+base/2));
#pragma unroll
        for(unsigned r=0;r<4;++r) codes |= unsigned(scales[(row+r)*(k/16)+base/16])<<(r*8);
      }
      const unsigned words[4]={packed.x,packed.y,packed.z,packed.w};
      unsigned bs[4];
#pragma unroll
      for(unsigned r=0;r<4;++r) bs[r]=table[(codes>>(r*8))&255];
      ulonglong2 acts[M];
#pragma unroll
      for(unsigned m=0;m<M;++m) {
        if constexpr(Layout==1) {
          acts[m]=m<staged_rows ?
              *reinterpret_cast<const ulonglong2*>(activation+m*5120+base) :
              *reinterpret_cast<const ulonglong2*>(x+m*k+base);
        } else acts[m]=*reinterpret_cast<const ulonglong2*>(x+m*k+base);
      }
#pragma unroll
      for(unsigned half=0;half<2;++half)
#pragma unroll
        for(unsigned pair=0;pair<2;++pair) {
          float av0[M], av1[M];
#pragma unroll
          for(unsigned m=0;m<M;++m) {
            const auto raw = half ? acts[m].y : acts[m].x;
            av0[m]=bf16(raw>>(pair*32));
            av1[m]=bf16(raw>>(pair*32+16));
          }
#pragma unroll
          for(unsigned r=0;r<4;++r) {
            const unsigned weights2=scaled_weight_pair(
                values[(words[r]>>((half*2+pair)*8))&255],bs[r]);
            const float w0=bf16(weights2),w1=bf16(weights2>>16);
#pragma unroll
            for(unsigned m=0;m<M;++m) {
              a[m][r][pair*2]=fmaf(w0,av0[m],a[m][r][pair*2]);
              a[m][r][pair*2+1]=fmaf(w1,av1[m],a[m][r][pair*2+1]);
            }
          }
        }
    }
  }
#pragma unroll
  for(unsigned m=0;m<M;++m) {
    float merged[4];
#pragma unroll
    for(unsigned r=0;r<4;++r)
      merged[r]=(a[m][r][0]+a[m][r][1])+(a[m][r][2]+a[m][r][3]);
    const float sum=warp_sum_four(merged[0],merged[1],merged[2],merged[3])*scale;
    if(!(lane&7)) y[m*n+row+lane/8]=rounded(sum);
  }
}
template<unsigned M, unsigned Layout, bool Persistent = false>
__global__ void nvfp4_head_rows(const std::uint8_t* weights, const std::uint8_t* scales,
    unsigned scale_base, float scale, const std::uint16_t* x,
    unsigned n, unsigned k, std::uint16_t* y) {
  __shared__ unsigned table[256], values[256];
  __shared__ __align__(16) std::uint16_t activation[Persistent ? M*5120 : 1];
  if constexpr(Persistent) {
    for(unsigned i=threadIdx.x;i<M*5120/8;i+=blockDim.x)
      reinterpret_cast<ulonglong2*>(activation)[i]=reinterpret_cast<const ulonglong2*>(x)[i];
    x=activation;
  }
  for (unsigned i=threadIdx.x;i<256;i+=blockDim.x) {
    const unsigned scale = __float_as_uint(fp8(i)) >> 16;
    table[i] = scale | (scale << 16);
    values[i] = (__float_as_uint(nv_values[i & 15]) >> 16) |
                (__float_as_uint(nv_values[i >> 4]) & 0xffff0000U);
  }
  __syncthreads();
  const unsigned lane=threadIdx.x%32;
  const unsigned first=(blockIdx.x*4+threadIdx.x/32)*4;
  for(unsigned row=first;row<n;row+=Persistent ? gridDim.x*16 : n) {
  float a[M][4][4]={};
  for(unsigned tile=0;tile<k/512;++tile) {
    unsigned phase_codes[2];
    if constexpr(Layout==2) {
      const unsigned local=scale6_codes(scales,scale_base,row,tile,k,lane);
      const unsigned partner=__shfl_xor_sync(0xffffffffU,local,1);
      phase_codes[0]=(lane&1)?partner:local;
      phase_codes[1]=(lane&1)?local:partner;
    }
#pragma unroll
    for(unsigned phase=0;phase<2;++phase) {
      const unsigned base=tile*512+phase*256+lane*8;
      uint4 packed; unsigned codes=0;
      if constexpr(Layout==1) {
        const auto* record=weights+(row/4)*11520+tile*1152+phase*576;
        packed=__ldcs(reinterpret_cast<const uint4*>(record)+lane);
        codes=__ldcs(reinterpret_cast<const unsigned*>(record+512)+lane/2);
      } else if constexpr(Layout==2) {
        packed=__ldcs(reinterpret_cast<const uint4*>(weights)+
                       (row/4)*(k/512)*64+tile*64+phase*32+lane);
        codes=phase_codes[phase];
      } else {
        packed.x=__ldcs(reinterpret_cast<const unsigned*>(weights+row*(k/2)+base/2));
        packed.y=__ldcs(reinterpret_cast<const unsigned*>(weights+(row+1)*(k/2)+base/2));
        packed.z=__ldcs(reinterpret_cast<const unsigned*>(weights+(row+2)*(k/2)+base/2));
        packed.w=__ldcs(reinterpret_cast<const unsigned*>(weights+(row+3)*(k/2)+base/2));
#pragma unroll
        for(unsigned r=0;r<4;++r) codes |= unsigned(scales[(row+r)*(k/16)+base/16])<<(r*8);
      }
      const unsigned words[4]={packed.x,packed.y,packed.z,packed.w};
      unsigned bs[4];
#pragma unroll
      for(unsigned r=0;r<4;++r) bs[r]=table[(codes>>(r*8))&255];
      ulonglong2 acts[M];
#pragma unroll
      for(unsigned m=0;m<M;++m) acts[m]=*reinterpret_cast<const ulonglong2*>(x+m*k+base);
#pragma unroll
      for(unsigned half=0;half<2;++half)
#pragma unroll
        for(unsigned pair=0;pair<2;++pair) {
          float av0[M], av1[M];
#pragma unroll
          for(unsigned m=0;m<M;++m) {
            const auto raw = half ? acts[m].y : acts[m].x;
            av0[m]=bf16(raw>>(pair*32));
            av1[m]=bf16(raw>>(pair*32+16));
          }
#pragma unroll
          for(unsigned r=0;r<4;++r) {
            const unsigned weights2=scaled_weight_pair(
                values[(words[r]>>((half*2+pair)*8))&255],bs[r]);
            const float w0=bf16(weights2),w1=bf16(weights2>>16);
#pragma unroll
            for(unsigned m=0;m<M;++m) {
              a[m][r][pair*2]=fmaf(w0,av0[m],a[m][r][pair*2]);
              a[m][r][pair*2+1]=fmaf(w1,av1[m],a[m][r][pair*2+1]);
            }
          }
        }
    }
  }
#pragma unroll
  for(unsigned m=0;m<M;++m) {
    float merged[4];
#pragma unroll
    for(unsigned r=0;r<4;++r)
      merged[r]=(a[m][r][0]+a[m][r][1])+(a[m][r][2]+a[m][r][3]);
    const float sum=warp_sum_four(merged[0],merged[1],merged[2],merged[3])*scale;
    if(!(lane&7)) y[m*n+row+lane/8]=rounded(sum);
  }
  }
}
template<unsigned M>
void launch(const std::uint8_t* weights,const std::uint8_t* scales,float scale,
    const std::uint16_t* x,unsigned n,unsigned k,std::uint16_t* y,
    const std::uint8_t* sidecar,const std::uint8_t* scale6,unsigned scale_base,
    cudaStream_t stream) {
  if(scales) {
    if(n==248320) nvfp4_head_rows<M,0,true><<<64,128,0,stream>>>(weights,scales,0,scale,x,n,k,y);
    else if(sidecar && k==5120) nvfp4_rows<M,1><<<n/16,128,0,stream>>>(sidecar,scales,0,scale,x,n,k,y);
    else if(sidecar && scale6) nvfp4_rows<M,2><<<n/16,128,0,stream>>>(sidecar,scale6,scale_base,scale,x,n,k,y);
    else nvfp4_rows<M,0><<<n/16,128,0,stream>>>(weights,scales,0,scale,x,n,k,y);
  } else {
    if(sidecar) fp8_rows<M,true><<<n/4,256,0,stream>>>(sidecar,scale,x,n,k,y);
    else fp8_rows<M,false><<<n/4,256,0,stream>>>(weights,scale,x,n,k,y);
  }
}
} // namespace
int prepare_mtp_verify_projection_device() noexcept {
  int device = 0;
  cudaDeviceProp properties{};
  auto status = cudaGetDevice(&device);
  if(status != cudaSuccess) return status;
  status = cudaGetDeviceProperties(&properties,device);
  if(status != cudaSuccess) return status;
  if(properties.major != 8 || properties.minor != 7 ||
      properties.multiProcessorCount != 16) return cudaErrorInvalidDevice;
  (void)cudaGetLastError();
  status = cudaFuncSetAttribute(nvfp4_rows<2,1>,
      cudaFuncAttributePreferredSharedMemoryCarveout,cudaSharedmemCarveoutMaxShared);
  if(status != cudaSuccess) return status;
  status = cudaFuncSetAttribute(nvfp4_rows<3,1>,
      cudaFuncAttributePreferredSharedMemoryCarveout,cudaSharedmemCarveoutMaxShared);
  if(status != cudaSuccess) return status;
  return cudaFuncSetAttribute(nvfp4_rows<4,1>,
      cudaFuncAttributePreferredSharedMemoryCarveout,cudaSharedmemCarveoutMaxShared);
}
int launch_mtp_verify_projection(const std::uint8_t* weights,
    const std::uint8_t* scales,float scale,const std::uint16_t* x,
    unsigned count,unsigned n,unsigned k,std::uint16_t* y,void* stream,
    const std::uint8_t* sidecar,const std::uint8_t* scale6,unsigned scale_base) noexcept {
  if(!weights || !x || !y || count<2 || count>4 || !std::isfinite(scale) || scale<0)
    return cudaErrorInvalidValue;
  const bool shape=scales ? (((n==17408 || n==248320) && k==5120)||(n==5120 && k==17408)) :
    ((k==5120 && (n==1024 || n==6144 || n==10240 || n==12288)) || (k==6144 && n==5120));
  if(!shape || (n==248320 && (sidecar || scale6)) || (sidecar && !scales && !(n==5120 && k==6144)) ||
     (sidecar && scales && k==17408 && (!scale6 || scale_base>192))) return cudaErrorInvalidValue;
  (void)cudaGetLastError();
  auto s=static_cast<cudaStream_t>(stream);
  if(count==2) launch<2>(weights,scales,scale,x,n,k,y,sidecar,scale6,scale_base,s);
  else if(count==3) launch<3>(weights,scales,scale,x,n,k,y,sidecar,scale6,scale_base,s);
  else launch<4>(weights,scales,scale,x,n,k,y,sidecar,scale6,scale_base,s);
  return cudaGetLastError();
}
} // namespace q3x::runtime::mtp_detail
