#include "mtp_draft_integer_head_internal.cu"
#include "q3x/kernels/sm87_weight_only_gemv.h"
#include <fstream>
#include <vector>
#include <string>
#include <cstdio>
#include <stdexcept>
#include <cstring>
namespace mtp=q3x::runtime::mtp_detail;
void ck(int s){if(s)throw std::runtime_error(cudaGetErrorString(cudaError_t(s)));}
void req(bool b){if(!b)throw std::runtime_error("head admission contract");}
std::vector<char> read(std::string p,size_t n){std::ifstream f(p,std::ios::binary);std::vector<char>b(n);f.read(b.data(),n);req(f&&f.peek()==EOF);return b;}
void write(std::string p,const void* b,size_t n){std::ofstream f(p,std::ios::binary);f.write((const char*)b,n);req(bool(f));}
__global__ void all_codes(unsigned* output){
 unsigned i=blockIdx.x*256+threadIdx.x;
 output[i]=mtp::integer_head::unpack_four(i);
 output[65536+i]=__dp4a(int(output[i]),int(0x7f7f7f7f),0);
 output[131072+i]=__dp4a(int(output[i]),int(0x81818181),0);
}
void synthetic(){
 unsigned* codes;ck(cudaMalloc(&codes,3*65536*4));all_codes<<<256,256>>>(codes);ck(cudaGetLastError());
 std::vector<unsigned> decoded(3*65536);ck(cudaMemcpy(decoded.data(),codes,3*65536*4,cudaMemcpyDeviceToHost));ck(cudaFree(codes));
 const int f4[16]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
 for(unsigned p=0;p<65536;++p)for(unsigned part=0;part<4;++part)
  req(int(static_cast<signed char>((decoded[p]>>(part*8))&255))==f4[(p>>(part*4))&15]);
 for(unsigned p=0;p<65536;++p){int sum=0;for(unsigned i=0;i<4;++i)sum+=f4[(p>>(i*4))&15]*127;req(int(decoded[65536+p])==sum&&int(decoded[131072+p])==-sum);}
 puts("65536 packed codes /262144 signed integer values exact");fflush(stdout);
 unsigned char *dw,*ds;unsigned short *dx,*dy;unsigned* workspace;
 ck(cudaMalloc(&dw,4*2560));ck(cudaMalloc(&ds,4*320));ck(cudaMalloc(&dx,10240));ck(cudaMalloc(&dy,16));ck(cudaMalloc(&workspace,6432));
 std::vector<unsigned char>weights(4*2560),scales(4*320,0x38);for(unsigned i=0;i<weights.size();++i)weights[i]=i&255;
 ck(cudaMemcpy(dw,weights.data(),weights.size(),cudaMemcpyHostToDevice));ck(cudaMemcpy(ds,scales.data(),scales.size(),cudaMemcpyHostToDevice));
 std::vector<unsigned short>input(5120);std::vector<unsigned char>bytes(6432);
 for(unsigned kind=0;kind<4;++kind){
  for(unsigned i=0;i<5120;++i){float x=kind==1?0.f:float(int(i%257)-128)/32;unsigned bits;memcpy(&bits,&x,4);input[i]=bits>>16;}
  if(kind==2)input[0]=0x7fc1;if(kind==3)input[0]=0x7f80;
  ck(cudaMemcpy(dx,input.data(),10240,cudaMemcpyHostToDevice));ck(cudaMemset(workspace,0xa5,6432));
  auto* q=workspace+4;auto* scale=reinterpret_cast<float*>(reinterpret_cast<char*>(q)+5120);
  mtp::integer_head::quantize<<<20,256>>>(reinterpret_cast<const __nv_bfloat16*>(dx),q,scale);ck(cudaGetLastError());
  mtp::integer_head::project<<<1,128>>>(dw,ds,1.f,q,scale,reinterpret_cast<__nv_bfloat16*>(dy));ck(cudaGetLastError());
  ck(cudaMemcpy(bytes.data(),workspace,6432,cudaMemcpyDeviceToHost));
  for(unsigned i=0;i<16;++i)req(bytes[i]==0xa5&&bytes[6416+i]==0xa5);
  std::vector<unsigned short>y(4);ck(cudaMemcpy(y.data(),dy,8,cudaMemcpyDeviceToHost));
  for(unsigned g=0;g<320;++g){
   float actual_scale;memcpy(&actual_scale,bytes.data()+16+5120+g*4,4);
   if(kind>=2&&g==0){req(std::isnan(actual_scale));continue;}
   float maximum=0;for(unsigned i=0;i<16;++i){unsigned bits=unsigned(input[g*16+i])<<16;float x;memcpy(&x,&bits,4);maximum=fmaxf(maximum,fabsf(x));}
   float expected=maximum/127.f;req(actual_scale==expected);
   for(unsigned i=0;i<16;++i){unsigned bits=unsigned(input[g*16+i])<<16;float x;memcpy(&x,&bits,4);int qx=maximum?int(std::nearbyint(x/expected)):0;req(int(static_cast<signed char>(bytes[16+g*16+i]))==qx);}
  }
  for(unsigned row=0;row<4;++row){
   if(kind>=2){req((y[row]&0x7f80)==0x7f80&&(y[row]&127));continue;}
   double expected=0;for(unsigned k=0;k<5120;++k){float sc;memcpy(&sc,bytes.data()+16+5120+(k/16)*4,4);int a=static_cast<signed char>(bytes[16+k]);unsigned code=(weights[row*2560+k/2]>>((k%2)*4))&15;expected+=double(a)*sc*f4[code]*.5;}
   float rounded=float(expected);unsigned bits;memcpy(&bits,&rounded,4);unsigned short bf=(bits+0x7fff+((bits>>16)&1))>>16;unsigned ybits=unsigned(y[row])<<16;float actual;memcpy(&actual,&ybits,4);req(fabs(double(actual)-expected)<=.004*fabs(expected)+.001);if(kind==1)req(y[row]==bf);
  }
 }
 ck(cudaFree(dw));ck(cudaFree(ds));ck(cudaFree(dx));ck(cudaFree(dy));ck(cudaFree(workspace));puts("integer dots, input quantization, zero/nonfinite propagation and scratch guards pass");
}
int main(int argc,char**argv){
 synthetic();if(argc==2&&std::string(argv[1])=="--small")return 0;
 req(argc==3);std::string root=argv[1],out=argv[2];
 auto w=read(root+"/weight.bin",635699200),s=read(root+"/scales.bin",79462400),a=read(root+"/input.bf16",9*10240),g=read(root+"/global.bin",4);float scale;memcpy(&scale,g.data(),4);
 unsigned char *dw,*ds;unsigned short *dx,*dy,*dz;void* workspace;ck(cudaMalloc(&workspace,6432));
 ck(cudaMalloc(&dw,w.size()));ck(cudaMalloc(&ds,s.size()));ck(cudaMalloc(&dx,10240));ck(cudaMalloc(&dy,496640+32));ck(cudaMalloc(&dz,496640+32));
 ck(cudaMemcpy(dw,w.data(),w.size(),cudaMemcpyHostToDevice));ck(cudaMemcpy(ds,s.data(),s.size(),cudaMemcpyHostToDevice));
 auto launch=[&](const unsigned short* x,unsigned short* y,void* stream=nullptr){return mtp::launch_mtp_integer_draft_head(dw,ds,scale,x,248320,5120,static_cast<char*>(workspace)+16,6400,y,stream);};
 req(launch(dx,dx)==cudaErrorInvalidValue);req(launch(nullptr,dy)==cudaErrorInvalidValue);req(launch(dx,dy+1)==cudaErrorInvalidValue);
 req(mtp::launch_mtp_integer_draft_head(dw,ds,scale,dx,248319,5120,workspace,6400,dy,nullptr)==cudaErrorInvalidValue);
 req(mtp::launch_mtp_integer_draft_head(dw,ds,-1,dx,248320,5120,workspace,6400,dy,nullptr)==cudaErrorInvalidValue);
 std::vector<unsigned short>b(248336),c(b.size());
 for(unsigned row=0;row<9;++row){
  ck(cudaMemcpy(dx,a.data()+row*10240,10240,cudaMemcpyHostToDevice));ck(cudaMemset(dy,0xa5,b.size()*2));ck(cudaMemset(dz,0xa5,c.size()*2));
  ck(cudaMemset(workspace,0xa5,6432));ck(launch(dx,dy+8));
  ck(q3x::kernels::launch_sm87_nvfp4_w4a16_gemv_bf16_cuda(dw,ds,scale,dx,248320,5120,dz+8,nullptr));ck(cudaDeviceSynchronize());
  ck(cudaMemcpy(b.data(),dy,b.size()*2,cudaMemcpyDeviceToHost));ck(cudaMemcpy(c.data(),dz,c.size()*2,cudaMemcpyDeviceToHost));
  for(unsigned i=0;i<8;++i)req(b[i]==0xa5a5&&c[i]==0xa5a5&&b[248328+i]==0xa5a5&&c[248328+i]==0xa5a5);
  std::vector<unsigned char> guard(6432);ck(cudaMemcpy(guard.data(),workspace,6432,cudaMemcpyDeviceToHost));for(unsigned j=0;j<16;++j)req(guard[j]==0xa5&&guard[6416+j]==0xa5);
  write(out+"/matrix-"+std::to_string(row)+".bf16",b.data()+8,496640);write(out+"/scalar-"+std::to_string(row)+".bf16",c.data()+8,496640);
  printf("row=%u complete_vocab=248320 guards=pass\n",row);fflush(stdout);
 }
 cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t exec;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
 ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));ck(launch(dx,dy+8,stream));ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
 for(int j=0;j<3;++j){ck(cudaMemcpyAsync(dx,a.data()+j*10240,10240,cudaMemcpyHostToDevice,stream));auto expected=read(out+"/matrix-"+std::to_string(j)+".bf16",496640);memcpy(b.data()+8,expected.data(),496640);ck(cudaMemsetAsync(dy,0xa5,b.size()*2,stream));ck(cudaGraphLaunch(exec,stream));ck(cudaStreamSynchronize(stream));ck(cudaMemcpy(c.data(),dy,c.size()*2,cudaMemcpyDeviceToHost));req(c==b);}
 ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(graph));ck(cudaStreamDestroy(stream));
 puts("graph complete logits and guards exact on three changing-input replays");
 ck(cudaFree(dw));ck(cudaFree(ds));ck(cudaFree(dx));ck(cudaFree(dy));ck(cudaFree(dz));ck(cudaFree(workspace));
}
