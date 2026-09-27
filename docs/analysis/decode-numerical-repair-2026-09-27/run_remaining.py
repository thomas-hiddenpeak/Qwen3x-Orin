import pathlib,json,subprocess,os
w=pathlib.Path('.q3x-work/decode-numerical-repair-20260927');old=pathlib.Path('.q3x-work/decode-qualification-20260927')
def run(name,cmd,env=None):
 with (w/(name+'-parent.log')).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,env=env)
def compare(a,b,dst):
 run('compare-'+dst,['python3','-B',str(w/'compare.py'),str(a),str(b),str(w/(dst+'.json'))],dict(os.environ,OPENBLAS_NUM_THREADS='1'))
run('contract-regression',['python3','-B',str(w/'run_owned.py'),'contract-regression','.q3x-work/build/fused-decode-admission/q3x_fused_decode_test'])
run('protocol',['ctest','--test-dir','.q3x-work/build/fused-decode-admission','-R','^fused_decode_protocol$','--output-on-failure'])
for name,n,arm in [('scalar-p576',576,'scalar'),('fused-p8192',8192,'fused'),('fused-p40000',40000,'fused')]:
 run(name,['python3','-B',str(w/'run_owned.py'),name,'.q3x-work/build/fused-decode-admission/q3x_fused_decode_model_test','/home/rm01/models/dev/llm/nvidia/Qwen3.6-27B-NVFP4','.q3x-work/evidence/terminal-prefix-main-p40000-api-20260909-r2/request-body.json',str(w/(name+'.json')),str(n),arm,str(old/f'tokens-p{n}.txt')])
 compare(old/f'scalar-p{n}.json',w/(name+'.json'),'scalar-bridge' if arm=='scalar' else f'compare-p{n}')
 if arm=='scalar':assert json.loads((w/'scalar-bridge.json').read_text())['strict_production_equality']
 print('completed',name,flush=True)
run('api-v3',['python3','-B',str(w/'run_api.py'),'api-v3'],dict(os.environ,DECODE_API_EXECUTABLE=str(pathlib.Path('.q3x-work/build/fused-decode-admission/qwen3x-eval-server-fused-decode-admission').resolve())))
print('all numerical/API guardrails completed',flush=True)
