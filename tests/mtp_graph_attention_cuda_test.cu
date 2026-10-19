#include "mtp_verify_graph_internal.h"
#include <cuda_runtime.h>
namespace q3x::runtime::mtp_detail {
int launch_mtp_verify_attention(const std::uint16_t*, const std::uint16_t*,
    const std::uint16_t*, unsigned, unsigned, void*, std::size_t,
    std::uint16_t*, void*) noexcept;
int launch_mtp_short_attention(const std::uint16_t*, const std::uint16_t*,
    const std::uint16_t*, unsigned, unsigned, std::uint16_t*, void*) noexcept;
}
#include <vector>
#include <cstdio>
#include <stdexcept>
#include <algorithm>
namespace next = q3x::runtime::mtp_detail;
// Static entry is the regression oracle; a separately frozen v101 oracle
// also qualified this interface in the work-package evidence.
void ck(int status) { if(status) throw std::runtime_error(cudaGetErrorString(cudaError_t(status))); }
void req(bool value) { if(!value) throw std::runtime_error("exact dynamic Graph mismatch"); }
int main(int argc, char**) {
 unsigned cases=0,replays=0; const bool small=argc>1;
 for(unsigned m: {2U,3U,4U}) for(unsigned cap: {68U,129U,512U-m,515U,8192U,40000U,44096U-m}) for(unsigned pattern:{0U,1U,2U}) {
  if(small && (m!=3 || cap!=515)) continue;
  bool short_path=cap<512; unsigned last=cap+m-1,stride=(last+3)&~3U;
  std::vector<std::uint16_t> q(m*12288),k(last*1024),v(k.size()),a(m*6144+16),b(a.size());
  for(size_t i=0;i<q.size();++i) q[i]=(i&1?0x8000:0)|(0x3c00+(i*11)%768);
  for(size_t i=0;i<k.size();++i){k[i]=(i&1?0x8000:0)|(0x3b00+(i*7)%768);v[i]=(i&1?0x8000:0)|(0x3c00+(i*19)%768);}
  if(pattern==1){for(size_t i=0;i<q.size();i+=3)q[i]=i&1?0x8000:0;for(size_t i=0;i<k.size();i+=5)k[i]=i&1?0x8000:0;}
  if(pattern==2){for(size_t i=0;i<q.size();i+=257)q[i]=0x7fc1;for(size_t i=0;i<k.size();i+=1021)k[i]=0x7f80;for(size_t i=0;i<v.size();i+=1031)v[i]=0x8000;}
  std::uint16_t *dq,*dk,*dv,*da,*db;float *pa,*pb;next::MtpVerifyDeviceInputs* parameters;
  size_t pwords=m*24ULL*stride+8;
  ck(cudaMalloc(&dq,q.size()*2));ck(cudaMalloc(&dk,k.size()*2));ck(cudaMalloc(&dv,v.size()*2));ck(cudaMalloc(&da,a.size()*2));ck(cudaMalloc(&db,b.size()*2));ck(cudaMalloc(&pa,pwords*4));ck(cudaMalloc(&pb,pwords*4));ck(cudaMalloc(&parameters,sizeof(*parameters)));
  ck(cudaMemcpy(dq,q.data(),q.size()*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(dk,k.data(),k.size()*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(dv,v.data(),v.size()*2,cudaMemcpyHostToDevice));
  cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t exec;
  ck(cudaStreamCreate(&stream));ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
  ck(next::launch_mtp_graph_attention(dq,dk,dv,parameters,cap,m,pb+4,(pwords-8)*4,db+8,stream));
  ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
  size_t count=0;ck(cudaGraphGetNodes(graph,nullptr,&count));std::vector<cudaGraphNode_t> nodes(count);ck(cudaGraphGetNodes(graph,nodes.data(),&count));
  cudaGraphNode_t qk=nullptr;cudaKernelNodeParams params{};unsigned matches=0;
  for(auto node:nodes){cudaGraphNodeType type;ck(cudaGraphNodeGetType(node,&type));if(type==cudaGraphNodeTypeKernel){cudaKernelNodeParams test{};ck(cudaGraphKernelNodeGetParams(node,&test));if(next::is_mtp_graph_qk_kernel(test.func)){qk=node;params=test;++matches;}}}
  req(matches==(short_path?0U:1U));
  unsigned iteration=0;
  for(unsigned first: {cap,short_path?65U:512U,short_path?65U:512U,cap-1}) {
    next::MtpVerifyDeviceInputs host{};host.position=first-1;ck(cudaMemcpy(parameters,&host,sizeof(host),cudaMemcpyHostToDevice));
    ck(cudaMemset(da,0xa5,a.size()*2));ck(cudaMemset(db,0xa5,b.size()*2));ck(cudaMemset(pa,0xa5,pwords*4));ck(cudaMemset(pb,0xa5,pwords*4));
    if(short_path) ck(next::launch_mtp_short_attention(dq,dk,dv,first,m,da+8,nullptr));
    else ck(next::launch_mtp_verify_attention(dq,dk,dv,first,m,pa+4,(pwords-8)*4,da+8,nullptr));
    ck(cudaDeviceSynchronize());
    // Replay at capacity, then a smaller length with the capacity grid, then
    // shrink and regrow just the QK grid, retaining all kernel args.
    if(qk && iteration>=2){params.gridDim.x=(first+m-1+127)/128;ck(cudaGraphExecKernelNodeSetParams(exec,qk,&params));}
    ck(cudaGraphLaunch(exec,stream));ck(cudaStreamSynchronize(stream));
    ck(cudaMemcpy(a.data(),da,a.size()*2,cudaMemcpyDeviceToHost));ck(cudaMemcpy(b.data(),db,b.size()*2,cudaMemcpyDeviceToHost));req(a==b);
    for(unsigned i=0;i<8;++i){req(b[i]==0xa5a5);req(b[m*6144+8+i]==0xa5a5);}
    if(!short_path){std::vector<unsigned> x(pwords),y(pwords);ck(cudaMemcpy(x.data(),pa,pwords*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(y.data(),pb,pwords*4,cudaMemcpyDeviceToHost));req(x==y);}
    ++iteration;++replays;
  }
  // Invalid host contracts enqueue no work, including null/aliased capsule.
  req(next::launch_mtp_graph_attention(dq,dk,dv,nullptr,cap,m,pb+4,(pwords-8)*4,db+8,stream)==cudaErrorInvalidValue);
  req(next::launch_mtp_graph_attention(dq,dk,dv,reinterpret_cast<next::MtpVerifyDeviceInputs*>(dq),cap,m,pb+4,(pwords-8)*4,db+8,stream)==cudaErrorInvalidValue);
  req(next::launch_mtp_graph_attention(dq,dk,dv,parameters,cap,5,pb+4,(pwords-8)*4,db+8,stream)==cudaErrorInvalidValue);
  ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(graph));ck(cudaStreamDestroy(stream));
  ck(cudaFree(dq));ck(cudaFree(dk));ck(cudaFree(dv));ck(cudaFree(da));ck(cudaFree(db));ck(cudaFree(pa));ck(cudaFree(pb));ck(cudaFree(parameters));
  printf("rows=%u capacity_first=%u pattern=%u changing_length_graph_output_probabilities_guards=bitwise_equal\n",m,cap,pattern);fflush(stdout);++cases;
 }
 printf("graph_cases=%u graph_replays=%u fixed_input_pointer=pass qk_grid_update=pass invalid_contracts=pass\n",cases,replays);
 return cases==(small?3U:63U)?0:1;
}
