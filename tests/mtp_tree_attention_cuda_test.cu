#include "mtp_verify_graph_internal.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>
namespace m = q3x::runtime::mtp_detail;
void ck(int status) {
  if (status) throw std::runtime_error(cudaGetErrorString(cudaError_t(status)));
}
void require(bool value) {
  if (!value) throw std::runtime_error("tree Attention differs from contiguous path");
}
struct Buffer {
  void* data = nullptr;
  explicit Buffer(size_t bytes) { ck(cudaMalloc(&data, bytes)); }
  ~Buffer() { cudaFree(data); }
  template<class T> T* as() { return static_cast<T*>(data); }
};
int main(int argc, char**) {
  unsigned cases = 0;
  const bool small = argc > 1;
  for (unsigned rows : {3U, 4U}) for (unsigned cap : {68U, 508U, 515U, 8192U, 40000U}) {
    if (small && (rows != 3 || cap != 515)) continue;
    const bool short_path = cap < 512;
    const size_t kvwords = (cap + rows - 1) * 1024ULL;
    const size_t outwords = rows * 6144ULL;
    const size_t pbytes = rows * 24ULL * ((cap + rows + 2) & ~3U) * 4;
    std::vector<std::uint16_t> q(rows * 12288), k(kvwords), v(kvwords);
    // Different values at sibling/self slots make accidental sibling reads visible.
    for (size_t i = 0; i < q.size(); ++i) q[i] = (i & 1 ? 0x8000 : 0) | (0x3c00 + (i * 11) % 768);
    for (size_t i = 0; i < k.size(); ++i) {
      k[i] = (i & 1 ? 0x8000 : 0) | (0x3b00 + (i * 7) % 768);
      v[i] = (i & 1 ? 0x8000 : 0) | (0x3c00 + (i * 19) % 768);
    }
    Buffer dq(q.size()*2), dk(kvwords*2), dv(kvwords*2), out((outwords+16)*2),
        linear(outwords*2), sibling(2*6144*2), p(pbytes), capsule(sizeof(m::MtpVerifyDeviceInputs));
    ck(cudaMemcpy(dq.data,q.data(),q.size()*2,cudaMemcpyHostToDevice));
    cudaStream_t stream; cudaGraph_t graph; cudaGraphExec_t exec;
    ck(cudaStreamCreate(&stream));
    ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    ck(m::launch_mtp_tree_graph_attention(dq.as<std::uint16_t>(),dk.as<std::uint16_t>(),dv.as<std::uint16_t>(),
        capsule.as<m::MtpVerifyDeviceInputs>(),cap,rows,p.data,pbytes,out.as<std::uint16_t>()+8,stream));
    ck(cudaStreamEndCapture(stream,&graph));
    ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    for (unsigned first : {cap, short_path ? 65U : 512U, cap-1}) {
      m::MtpVerifyDeviceInputs host{}; host.position=first-1;
      ck(cudaMemcpy(capsule.data,&host,sizeof(host),cudaMemcpyHostToDevice));
      ck(cudaMemcpy(dq.data,q.data(),q.size()*2,cudaMemcpyHostToDevice));
      ck(cudaMemcpy(dk.data,k.data(),kvwords*2,cudaMemcpyHostToDevice));
      ck(cudaMemcpy(dv.data,v.data(),kvwords*2,cudaMemcpyHostToDevice));
      ck(cudaMemset(out.data,0xa5,(outwords+16)*2));
      ck(cudaGraphLaunch(exec,stream)); ck(cudaStreamSynchronize(stream));
      std::vector<std::uint16_t> actual(outwords+16), expected(outwords);
      ck(cudaMemcpy(actual.data(),out.data,actual.size()*2,cudaMemcpyDeviceToHost));
      ck(m::launch_mtp_graph_attention(dq.as<std::uint16_t>(),dk.as<std::uint16_t>(),dv.as<std::uint16_t>(),
          capsule.as<m::MtpVerifyDeviceInputs>(),cap,rows,p.data,pbytes,linear.as<std::uint16_t>(),stream));
      ck(cudaStreamSynchronize(stream));
      ck(cudaMemcpy(expected.data(),linear.data,outwords*2,cudaMemcpyDeviceToHost));
      require(std::equal(expected.begin(),expected.begin()+(rows-1)*6144,actual.begin()+8));
      // Independent contiguous-history oracle: seed followed by alternate only.
      ck(cudaMemcpy(dq.as<std::uint16_t>()+12288,q.data()+(rows-1)*12288,12288*2,cudaMemcpyHostToDevice));
      ck(cudaMemcpy(dk.as<std::uint16_t>()+first*1024,k.data()+(first+rows-2)*1024,1024*2,cudaMemcpyHostToDevice));
      ck(cudaMemcpy(dv.as<std::uint16_t>()+first*1024,v.data()+(first+rows-2)*1024,1024*2,cudaMemcpyHostToDevice));
      ck(m::launch_mtp_graph_attention(dq.as<std::uint16_t>(),dk.as<std::uint16_t>(),dv.as<std::uint16_t>(),
          capsule.as<m::MtpVerifyDeviceInputs>(),cap,2,p.data,pbytes,sibling.as<std::uint16_t>(),stream));
      ck(cudaStreamSynchronize(stream));
      std::vector<std::uint16_t> branch(2*6144);
      ck(cudaMemcpy(branch.data(),sibling.data,branch.size()*2,cudaMemcpyDeviceToHost));
      require(std::equal(branch.begin()+6144,branch.end(),actual.begin()+8+(rows-1)*6144));
      for(unsigned i=0;i<8;++i) require(actual[i]==0xa5a5 && actual[outwords+8+i]==0xa5a5);
      ++cases;
    }
    ck(cudaGraphExecDestroy(exec)); ck(cudaGraphDestroy(graph)); ck(cudaStreamDestroy(stream));
  }
  printf("tree_attention_replays=%u complete_outputs_and_guards=bitwise_equal\n",cases);
  return cases == (small ? 3U : 30U) ? 0 : 1;
}
