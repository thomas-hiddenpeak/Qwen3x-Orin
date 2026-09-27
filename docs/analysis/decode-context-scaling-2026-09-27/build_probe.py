import pathlib,subprocess,shlex,json,hashlib
r=pathlib.Path('/home/rm01/Qwen3x-Orin');w=r/'.q3x-work/decode-context-investigation-20260927';b=r/'.q3x-work/build/orin-release'
compile=['/usr/local/cuda/bin/nvcc','-std=c++17','-O3','-DNDEBUG','-arch=sm_87','--expt-relaxed-constexpr','-Xcompiler=-fPIC','-I'+str(r/'third_party/flashinfer/include'),'-I'+str(r/'third_party/cutlass/include'),'-c',str(w/'fused_wrapper.cu'),'-o',str(w/'fused_wrapper.o')]
link=shlex.split((b/'CMakeFiles/qwen3x-eval-server.dir/link.txt').read_text())
for i,x in enumerate(link):
    if x.endswith(('.a','.o')) and not x.startswith('/'):link[i]=str(b/x)
idx=link.index('-o');link[idx+1]=str(w/'qwen3x-decode-research')
link[1:1]=[str(w/'fused_wrapper.o'),'-Wl,--wrap,_ZN3q3x7runtime35launch_gqa_attention_reference_cudaEPKtS2_S2_mmmmfPfmPtPv']
(w/'build-commands.json').write_text(json.dumps({'compile':compile,'link':link},indent=2))
subprocess.run(compile,check=True);subprocess.run(link,check=True)
files=[w/'fused_wrapper.cu',w/'fused_wrapper.o',w/'qwen3x-decode-research']+[pathlib.Path(x) for x in link if x.endswith(('.a','.o'))]
(w/'build-hashes.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
print('research binary built',flush=True)
