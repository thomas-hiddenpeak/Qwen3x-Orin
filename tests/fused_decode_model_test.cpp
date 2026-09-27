// Reuse the established ordinary full-state capture protocol; this executable
// adds a same-input Attention observer and never changes the server interface.
#define main ordinary_capture_main
#include "reference_ordinary_generation_capture_test.cpp"
#undef main
#include "decode_fused_gqa_internal.h"
namespace fd = q3x::runtime::fused_decode;
namespace {
struct AttentionCapture {
  bool restore_scalar = false;
  std::size_t first_sequence = 0;
  std::ofstream output;
  unsigned calls = 0;
};
int attention_observer(const fd::Observation& view, void* context) noexcept {
  auto& c = *static_cast<AttentionCapture*>(context);
  try {
    constexpr std::size_t count = 24*256;
    std::array<std::uint16_t,count> fused{}, repeat{}, scalar{};
    const auto stream = static_cast<cudaStream_t>(view.stream);
    const auto copy = [&](auto& target) {
      auto code = cudaMemcpyAsync(target.data(),view.output,count*2,cudaMemcpyDeviceToHost,stream);
      return code == cudaSuccess ? cudaStreamSynchronize(stream) : code;
    };
    int code = static_cast<int>(copy(fused));
    if (code) return code;
    code=fd::launch(view.query,view.key,view.value,view.sequence,view.workspace,
                    view.workspace_elements*sizeof(float),view.output,view.stream);
    if (code) return code;
    code=static_cast<int>(copy(repeat)); if(code) return code;
    code=rt::launch_gqa_attention_reference_cuda(view.query,view.key,view.value,
        24,4,view.sequence,256,0.0625f,view.workspace,view.workspace_elements,
        view.output,view.stream);
    if (code) return code;
    code=static_cast<int>(copy(scalar)); if(code) return code;
    double error=0,norm=0,maxabs=0; unsigned nonfinite=0;
    for (std::size_t i=0;i<count;++i) {
      const double a=bf16(fused[i]),b=bf16(scalar[i]);
      nonfinite+=!std::isfinite(a)||!std::isfinite(b);
      error+=(a-b)*(a-b); norm+=b*b; maxabs=std::max(maxabs,std::abs(a-b));
    }
    const double relative=std::sqrt(error/std::max(norm,1e-30));
    c.output << std::setprecision(12) << "{\"sequence\":" << view.sequence
      << ",\"layer\":" << view.layer << ",\"relative_l2_vs_scalar\":" << relative
      << ",\"maxabs\":" << maxabs << ",\"repeat_equal\":" << (fused==repeat?"true":"false")
      << ",\"nonfinite\":" << nonfinite;
    if (!c.first_sequence) c.first_sequence=view.sequence;
    if (view.sequence==c.first_sequence && (view.layer==3||view.layer==23||view.layer==43||view.layer==63)) {
      std::array<std::uint16_t,count> query{};
      std::vector<std::uint16_t> key(view.sequence*1024),value(key.size());
      code=static_cast<int>(cudaMemcpy(query.data(),view.query,count*2,cudaMemcpyDeviceToHost));
      if(code) return code;
      code=static_cast<int>(cudaMemcpy(key.data(),view.key,key.size()*2,cudaMemcpyDeviceToHost));
      if(code) return code;
      code=static_cast<int>(cudaMemcpy(value.data(),view.value,value.size()*2,cudaMemcpyDeviceToHost));
      if(code) return code;
      double fused_error=0,scalar_error=0,oracle_norm=0;
      for (std::size_t h : {0U,23U}) {
        std::vector<double> probabilities(view.sequence);
        double maximum=-1e300;
        for (std::size_t p=0;p<view.sequence;++p) {
          double dot=0;
          for (std::size_t d=0;d<256;++d)
            dot+=static_cast<double>(bf16(query[h*256+d]))*bf16(key[p*1024+(h/6)*256+d]);
          probabilities[p]=dot/16;maximum=std::max(maximum,probabilities[p]);
        }
        double denominator=0;
        for(auto& x:probabilities) {x=std::exp(x-maximum);denominator+=x;}
        for(std::size_t d=0;d<256;++d) {
          double expected=0;
          for(std::size_t p=0;p<view.sequence;++p)
            expected+=probabilities[p]*bf16(value[p*1024+(h/6)*256+d]);
          expected/=denominator;
          fused_error+=std::pow(bf16(fused[h*256+d])-expected,2);
          scalar_error+=std::pow(bf16(scalar[h*256+d])-expected,2);
          oracle_norm+=expected*expected;
        }
      }
      c.output << ",\"fp64_heads\":[0,23],\"fused_relative_l2_vs_fp64\":"
        << std::sqrt(fused_error/std::max(oracle_norm,1e-30))
        << ",\"scalar_relative_l2_vs_fp64\":" << std::sqrt(scalar_error/std::max(oracle_norm,1e-30));
    }
    c.output << "}\n";c.output.flush();++c.calls;
    if (nonfinite || fused!=repeat || relative>1.0/128.0) return cudaErrorInvalidValue;
    if (!c.restore_scalar) {
      code=static_cast<int>(cudaMemcpyAsync(view.output,fused.data(),count*2,cudaMemcpyHostToDevice,stream));
      if(code) return code;
      return static_cast<int>(cudaStreamSynchronize(stream));
    }
    return 0;
  } catch (...) {return cudaErrorUnknown;}
}
}
int main(int argc,char** argv) {
  if(argc!=6) {std::cerr << "usage: MODEL REQUEST OUTPUT PROMPT_TOKENS scalar|fused\n";return 2;}
  AttentionCapture capture;
  capture.restore_scalar=std::string_view(argv[5])=="scalar";
  if(!capture.restore_scalar && std::string_view(argv[5])!="fused") return 2;
  capture.output.open(std::string(argv[3])+".attention.jsonl");
  if(!capture.output) return 2;
  fd::set_observer(attention_observer,&capture);
  char flag[]="--prompt-tokens", variant[]="--variant", liveness[]="liveness";
  char* args[]={argv[0],argv[1],argv[2],argv[3],flag,argv[4],variant,liveness};
  const int status=ordinary_capture_main(8,args);
  fd::set_observer(nullptr,nullptr);
  std::cerr << "attention_shadow_calls=" << capture.calls << '\n';
  return status;
}
