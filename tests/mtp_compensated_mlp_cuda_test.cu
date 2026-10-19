#include "mtp_compensated_mlp_internal.cu"
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <stdexcept>
namespace m=q3x::runtime::mtp_detail;
void ck(int s){if(s)throw std::runtime_error(cudaGetErrorString(cudaError_t(s)));}
void req(bool b){if(!b)throw std::runtime_error("compensated MLP contract");}
float dec(unsigned short b){unsigned u=unsigned(b)<<16;float f;memcpy(&f,&u,4);return f;}
unsigned short bf(float f){unsigned b;memcpy(&b,&f,4);return (b&0x7fffffffU)>0x7f800000U?(b>>16)|0x40U:(b+0x7fff+((b>>16)&1))>>16;}
int main(){
 // Each MMA A byte and each live B column varies independently; include extrema.
 constexpr unsigned K=256,N=64,G=K/128;
 unsigned char* dw;unsigned short *ds,*dy; signed char* dq;float* da;
 ck(cudaMalloc(&dw,N*K));ck(cudaMalloc(&ds,N*G*2));ck(cudaMalloc(&dq,2*K));ck(cudaMalloc(&da,2*G*4));ck(cudaMalloc(&dy,(N+16)*2));
 std::vector<unsigned char>w(N*K);std::vector<unsigned short>s(N*G),out(N+16);std::vector<signed char>q(2*K);std::vector<float>a(2*G);
 for(unsigned kind=0;kind<6;++kind){
  for(unsigned i=0;i<w.size();++i)w[i]=kind>=4?(kind==4?255:1):1+(i*71+i/K*13)%255;
  for(unsigned i=0;i<s.size();++i)s[i]=bf(std::ldexp(1.f,int(i%5)-3));
  for(unsigned i=0;i<q.size();++i)q[i]=kind>=4?(kind==4?127:-127):int((i*37+i/K*17)%255)-127;
  if(kind==1)for(unsigned i=K;i<2*K;++i)q[i]=0;
  if(kind==2)for(unsigned i=0;i<K;++i)q[i]=0;
  if(kind==3)std::fill(q.begin(),q.end(),0);
  for(unsigned i=0;i<a.size();++i)a[i]=std::ldexp(1.f,-int(i)-4);
  ck(cudaMemcpy(dw,w.data(),w.size(),cudaMemcpyHostToDevice));ck(cudaMemcpy(ds,s.data(),s.size()*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(dq,q.data(),q.size(),cudaMemcpyHostToDevice));ck(cudaMemcpy(da,a.data(),a.size()*4,cudaMemcpyHostToDevice));ck(cudaMemset(dy,0xa5,(N+16)*2));
  m::compensated::project<<<1,128>>>(dw,ds,dq,da,K,dy+8);ck(cudaGetLastError());ck(cudaMemcpy(out.data(),dy,out.size()*2,cudaMemcpyDeviceToHost));
  for(unsigned row=0;row<N;++row){float y=0;for(unsigned g=0;g<G;++g){int d0=0,d1=0;for(unsigned j=0;j<128;++j){int v=int(w[row*K+g*128+j])-128;d0+=v*q[g*128+j];d1+=v*q[K+g*128+j];}float ws=dec(s[row*G+g]);y=std::fma(float(d0),a[g]*ws,y);y=std::fma(float(d1),a[G+g]*ws,y);}req(out[row+8]==bf(y));}
  for(unsigned i=0;i<8;++i)req(out[i]==0xa5a5&&out[N+8+i]==0xa5a5);
 }
 ck(cudaFree(dw));ck(cudaFree(ds));ck(cudaFree(dq));ck(cudaFree(da));ck(cudaFree(dy));
 puts("two-column integer MMA: complete 64 rows, independent columns, zero, signed extrema, guards exact");
 constexpr unsigned Q=5120;const auto bytes=m::mtp_compensated_bytes(Q);
 unsigned short* dx;void* scratch;ck(cudaMalloc(&dx,2*Q));ck(cudaMalloc(&scratch,bytes+32));auto* body=static_cast<char*>(scratch)+16;
 std::vector<unsigned short>input(Q);std::vector<unsigned char>packed(bytes+32);
 auto check=[&](unsigned kind){
  ck(cudaMemcpy(packed.data(),scratch,packed.size(),cudaMemcpyDeviceToHost));
  for(unsigned i=0;i<16;++i)req(packed[i]==0xa5&&packed[bytes+16+i]==0xa5);
  for(unsigned g=0;g<Q/128;++g){
   float got[2];for(unsigned p=0;p<2;++p)memcpy(&got[p],packed.data()+16+2*Q+4*(p*(Q/128)+g),4);
   if(kind>=3&&g==0){req(std::isnan(got[0])&&std::isnan(got[1]));continue;}
   float mx=0;for(unsigned i=0;i<128;++i)mx=fmaxf(mx,fabsf(dec(input[g*128+i])));float s0=mx/127.f;if(s0==0)s0=1;
   float r[128];int q0[128];mx=0;for(unsigned i=0;i<128;++i){float x=dec(input[g*128+i]);q0[i]=std::max(-127,std::min(127,int(std::nearbyint(x/s0))));r[i]=std::fma(-float(q0[i]),s0,x);mx=fmaxf(mx,fabsf(r[i]));}
   float s1=mx/127.f;if(s1==0)s1=1;req(got[0]==s0&&got[1]==s1);
   for(unsigned i=0;i<128;++i){int q1=std::max(-127,std::min(127,int(std::nearbyint(r[i]/s1))));req(int(static_cast<signed char>(packed[16+g*128+i]))==q0[i]);req(int(static_cast<signed char>(packed[16+Q+g*128+i]))==q1);}
  }
 };
 req(m::launch_mtp_compensated_quantize(nullptr,Q,body,bytes,nullptr)==cudaErrorInvalidValue);
 req(m::launch_mtp_compensated_quantize(dx,Q,dx,bytes,nullptr)==cudaErrorInvalidValue);
 req(m::launch_mtp_compensated_quantize(dx,Q,body,bytes-1,nullptr)==cudaErrorInvalidValue);
 cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t exec;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
 ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));ck(m::launch_mtp_compensated_quantize(dx,Q,body,bytes,stream));ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
 for(unsigned kind=0;kind<5;++kind){
  for(unsigned i=0;i<Q;++i)input[i]=kind==1?0:bf(std::ldexp(float(int(i%257)-128)/113.f,int(i%11)-5));
  if(kind==2)for(unsigned i=0;i<Q;++i)input[i]=bf(dec(input[i])*7.125f);
  if(kind==3)input[0]=0x7fc1;if(kind==4)input[0]=0x7f80;
  ck(cudaMemcpyAsync(dx,input.data(),Q*2,cudaMemcpyHostToDevice,stream));ck(cudaMemsetAsync(scratch,0xa5,bytes+32,stream));ck(cudaGraphLaunch(exec,stream));ck(cudaStreamSynchronize(stream));check(kind);
  auto graph_result=packed;ck(cudaMemset(scratch,0xa5,bytes+32));ck(m::launch_mtp_compensated_quantize(dx,Q,body,bytes,nullptr));check(kind);req(packed==graph_result);
 }
 ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(graph));ck(cudaStreamDestroy(stream));ck(cudaFree(dx));ck(cudaFree(scratch));
 puts("FP32 residual quantization, zero/nonfinite, guards, invalid arguments, five changing-input Graph replays exact");
}
