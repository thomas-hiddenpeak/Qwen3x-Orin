// Frozen scalar arithmetic from reference/decode_ops.cu at 6bed3ef.
#include "mtp_verify_graph_internal.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <stdexcept>
#include <algorithm>
namespace mtp = q3x::runtime::mtp_detail;
__device__ float decode_bf16_device(std::uint16_t x) { return __uint_as_float(unsigned(x)<<16); }
__device__ std::uint16_t encode_bf16_device(float x) {
 unsigned b=__float_as_uint(x);
 if((b&0x7fffffffU)>0x7f800000U)return (b>>16)|0x0040U;
 return (b+0x7fffU+((b>>16)&1U))>>16;
}
__global__ void reference_preprocess(
    const std::uint16_t* const interleaved_q_gate,
    std::uint16_t* const key,
    const std::uint16_t* const q_weight,
    const std::uint16_t* const k_weight,
    const float epsilon,
    std::uint16_t* const query_output,
    std::uint16_t* const gate_output,
    const float* const cosines,
    const float* const sines,
    const std::size_t first_position) {
  constexpr unsigned kFullPreprocessQueryHeads = 24, kFullPreprocessKvHeads = 4;
  constexpr unsigned kFullPreprocessHeadDimension = 256, kThreads = 256;
  constexpr unsigned kQwenRotaryDimension = 64;
  constexpr std::size_t kCombinedHeads =
      kFullPreprocessQueryHeads + kFullPreprocessKvHeads;
  __shared__ float partial[kThreads];

  const std::size_t combined_head = static_cast<std::size_t>(blockIdx.x);
  const std::size_t token = combined_head / kCombinedHeads;
  const std::size_t token_head = combined_head - token * kCombinedHeads;
  const bool is_query = token_head < kFullPreprocessQueryHeads;
  const std::size_t head =
      is_query ? token_head : token_head - kFullPreprocessQueryHeads;
  const std::size_t dimension = threadIdx.x;

  std::size_t packed_offset = 0U;
  const std::uint16_t* weight = nullptr;
  float value = 0.0F;
  if (is_query) {
    packed_offset =
        (token * kFullPreprocessQueryHeads + head) *
            kFullPreprocessHeadDimension +
        dimension;
    const std::size_t interleaved_offset =
        (token * kFullPreprocessQueryHeads + head) *
            (2U * kFullPreprocessHeadDimension) +
        dimension;
    value = decode_bf16_device(interleaved_q_gate[interleaved_offset]);
    gate_output[packed_offset] =
        interleaved_q_gate[interleaved_offset +
                           kFullPreprocessHeadDimension];
    weight = q_weight;
  } else {
    packed_offset =
        (token * kFullPreprocessKvHeads + head) *
            kFullPreprocessHeadDimension +
        dimension;
    value = decode_bf16_device(key[packed_offset]);
    weight = k_weight;
  }

  float sum = 0.0F;
  sum = fmaf(value, value, sum);
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned int stride = kThreads / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inverse_rms =
      rsqrtf(partial[0] /
                 static_cast<float>(kFullPreprocessHeadDimension) +
             epsilon);
  const float gamma = decode_bf16_device(weight[dimension]) + 1.0F;
  const std::uint16_t normalized =
      encode_bf16_device(value * inverse_rms * gamma);
  std::uint16_t* const normalized_output = is_query ? query_output : key;
  normalized_output[packed_offset] = normalized;

  constexpr std::size_t kHalfRotary = kQwenRotaryDimension / 2U;
  __syncthreads();
  if (dimension < kHalfRotary) {
    std::uint16_t* const head_output =
        normalized_output + packed_offset - dimension;
    const std::size_t table_offset =
        (first_position + token) * kHalfRotary + dimension;
    const float cosine = cosines[table_offset];
    const float sine = sines[table_offset];
    const float first = decode_bf16_device(head_output[dimension]);
    const float second =
        decode_bf16_device(head_output[dimension + kHalfRotary]);
    // Spell out which product is rounded before the FFMA. This is the exact
    // instruction order emitted for qk_partial_neox_rope_tile above.
    const float rotated_first = fmaf(first, cosine, -(second * sine));
    const float rotated_second = fmaf(second, cosine, first * sine);
    head_output[dimension] = encode_bf16_device(rotated_first);
    head_output[dimension + kHalfRotary] =
        encode_bf16_device(rotated_second);
  }
}


void ck(int s){if(s)throw std::runtime_error(cudaGetErrorString(cudaError_t(s)));}
void req(bool b){if(!b)throw std::runtime_error("embedding/preprocess/cache/guard mismatch");}
int main(int argc,char**){
 unsigned cases=0;bool small=argc>1;
#if defined(Q3X_TEST_TREE_INPUTS)
 constexpr bool tree=true;
#else
 constexpr bool tree=false;
#endif
 for(unsigned rows:{2U,3U,4U}) {
  if(tree && rows==2)continue;
  const unsigned capacity=44095;
  std::vector<std::uint16_t> q(rows*12288),k(rows*1024),v(k.size()),weights(512),emb(4*5120),actual(rows*12288+16),expected(actual.size()),tokens(rows*5120);
  std::vector<float> cosines(capacity*32),sines(cosines.size());
  for(size_t i=0;i<q.size();++i)q[i]=(i&1?0x8000:0)|(0x3c00+(i*11)%768);
  for(size_t i=0;i<k.size();++i){k[i]=(i&1?0x8000:0)|(0x3b00+(i*7)%768);v[i]=0x3e00+(i*19)%256;}
  for(size_t i=0;i<weights.size();++i)weights[i]=0x3b00+i%256;
  for(size_t i=0;i<emb.size();++i)emb[i]=unsigned(i);
  for(size_t i=0;i<cosines.size();++i){cosines[i]=float(int(i%257)-128)/128.f;sines[i]=float(int(i%127)-63)/64.f;}
  std::uint16_t *dq,*dk,*dv,*dw,*de,*dt,*da,*db,*ka,*kb,*va,*vb;float *dc,*ds;mtp::MtpVerifyDeviceInputs* input;
  size_t cache_bytes=capacity*1024ULL*2;
  ck(cudaMalloc(&dq,q.size()*2));ck(cudaMalloc(&dk,k.size()*2));ck(cudaMalloc(&dv,v.size()*2));ck(cudaMalloc(&dw,weights.size()*2));ck(cudaMalloc(&de,emb.size()*2));ck(cudaMalloc(&dt,tokens.size()*2));ck(cudaMalloc(&da,actual.size()*2));ck(cudaMalloc(&db,expected.size()*2));ck(cudaMalloc(&ka,cache_bytes));ck(cudaMalloc(&kb,cache_bytes));ck(cudaMalloc(&va,cache_bytes));ck(cudaMalloc(&vb,cache_bytes));ck(cudaMalloc(&dc,cosines.size()*4));ck(cudaMalloc(&ds,sines.size()*4));ck(cudaMalloc(&input,sizeof(*input)));
  ck(cudaMemcpy(dq,q.data(),q.size()*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(dv,v.data(),v.size()*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(dw,weights.data(),weights.size()*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(de,emb.data(),emb.size()*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(dc,cosines.data(),cosines.size()*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(ds,sines.data(),sines.size()*4,cudaMemcpyHostToDevice));
  cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t exec;
  ck(cudaStreamCreate(&stream));ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
  ck(mtp::launch_mtp_graph_embedding(de,input,rows,dt,stream));
  ck((tree ? mtp::launch_mtp_tree_graph_preprocess : mtp::launch_mtp_graph_preprocess)(dq,dk,dv,dw,dw+256,dc,ds,input,rows,db+8,kb,vb,stream));
  ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
  for(unsigned position:{0U,64U,509U,511U,512U,8192U,44091U}) {
   if(small && (rows!=3 || position!=511))continue;
   mtp::MtpVerifyDeviceInputs host{};host.position=position;
   for(unsigned row=0;row<rows;++row)host.token_ids[row]=(row+position)%4;
   ck(cudaMemcpy(input,&host,sizeof(host),cudaMemcpyHostToDevice));ck(cudaMemcpy(dk,k.data(),k.size()*2,cudaMemcpyHostToDevice));
   ck(cudaMemset(ka,0xa5,cache_bytes));ck(cudaMemset(kb,0xa5,cache_bytes));ck(cudaMemset(va,0xa5,cache_bytes));ck(cudaMemset(vb,0xa5,cache_bytes));ck(cudaMemset(da,0xa5,actual.size()*2));ck(cudaMemset(db,0xa5,expected.size()*2));
   for(unsigned row=0;row<rows;++row){
    ck(cudaMemcpy(ka+(position+row)*1024ULL,k.data()+row*1024,2048,cudaMemcpyHostToDevice));ck(cudaMemcpy(va+(position+row)*1024ULL,v.data()+row*1024,2048,cudaMemcpyHostToDevice));
    reference_preprocess<<<28,256>>>(dq+row*12288,ka+(position+row)*1024ULL,dw,dw+256,1.e-6F,da+8+row*12288,da+8+row*12288+6144,dc,ds,position+(tree && row==rows-1 ? 1 : row));ck(cudaGetLastError());
   }
   ck(cudaDeviceSynchronize());ck(cudaGraphLaunch(exec,stream));ck(cudaStreamSynchronize(stream));
   ck(cudaMemcpy(actual.data(),da,actual.size()*2,cudaMemcpyDeviceToHost));ck(cudaMemcpy(expected.data(),db,expected.size()*2,cudaMemcpyDeviceToHost));req(actual==expected);
   ck(cudaMemcpy(tokens.data(),dt,tokens.size()*2,cudaMemcpyDeviceToHost));
   for(unsigned row=0;row<rows;++row)req(std::equal(tokens.begin()+row*5120,tokens.begin()+(row+1)*5120,emb.begin()+host.token_ids[row]*5120));
   std::vector<std::uint16_t> first(capacity*1024ULL),second(first.size());
   for(auto pair:{std::make_pair(ka,kb),std::make_pair(va,vb)}){ck(cudaMemcpy(first.data(),pair.first,cache_bytes,cudaMemcpyDeviceToHost));ck(cudaMemcpy(second.data(),pair.second,cache_bytes,cudaMemcpyDeviceToHost));req(first==second);}
   printf("rows=%u position=%u embedding_query_gate_full_KV_guards=bitwise_equal\n",rows,position);fflush(stdout);++cases;
  }
  ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(graph));ck(cudaStreamDestroy(stream));
  for(auto ptr:{dq,dk,dv,dw,de,dt,da,db,ka,kb,va,vb})ck(cudaFree(ptr));ck(cudaFree(dc));ck(cudaFree(ds));ck(cudaFree(input));
 }
 printf("input_graph_cases=%u complete_cache_extent=44095\n",cases);return cases==(small?1U:tree?14U:21U)?0:1;
}
