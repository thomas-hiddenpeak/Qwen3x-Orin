import pathlib,subprocess,os,time,json
w=pathlib.Path('.q3x-work/decode-qualification-20260927')
start=time.monotonic()
while not (w/'capability-direct-fused/cleanup.json').exists():
 if time.monotonic()-start>1200:raise RuntimeError('fused completion timeout')
 time.sleep(1)
assert (w/'capability-direct-fused/lifecycle.json').exists(), 'fused lifecycle failed'
kind='lifecycle-scalar-r2'
with (w/(kind+'-parent.log')).open('w') as f:
 subprocess.run(['python3','-B',str(w/'run_capability_direct_r2.py'),kind],env=dict(os.environ,DECODE_API_EXECUTABLE=str(pathlib.Path('.q3x-work/build/orin-release/qwen3x-eval-server').resolve()),CAPABILITY_LIFECYCLE_ONLY='1'),stdout=f,stderr=subprocess.STDOUT,check=True)
print('scalar lifecycle completed',flush=True)
cmd=['python3','-B',str(w/'run_owned.py'),'fused-p576-repeat','.q3x-work/build/fused-decode-admission/q3x_fused_decode_model_test','/home/rm01/models/dev/llm/nvidia/Qwen3.6-27B-NVFP4','.q3x-work/evidence/terminal-prefix-main-p40000-api-20260909-r2/request-body.json',str(w/'fused-p576-repeat.json'),'576','fused',str(w/'tokens-p576.txt')]
with (w/'fused-p576-repeat-parent.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
subprocess.run(['python3','-B',str(w/'compare.py'),str(w/'fused-p576.json'),str(w/'fused-p576-repeat.json'),str(w/'compare-fused-replay.json')],env=dict(os.environ,OPENBLAS_NUM_THREADS='1'),check=True)
print('fused full-state repeat completed',flush=True)
