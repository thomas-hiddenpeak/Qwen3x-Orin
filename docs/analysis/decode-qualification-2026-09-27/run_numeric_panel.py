import pathlib,json,subprocess,os
w=pathlib.Path('.q3x-work/decode-qualification-20260927')
check=json.loads((w/'compare-scalar-replay.json').read_text());assert check['strict_production_equality']
exe='.q3x-work/build/fused-decode-admission/q3x_fused_decode_model_test'
model='/home/rm01/models/dev/llm/nvidia/Qwen3.6-27B-NVFP4'
req='.q3x-work/evidence/terminal-prefix-main-p40000-api-20260909-r2/request-body.json'
for n in [8192,40000]:
 for kind in ['scalar','fused']:
  name=f'{kind}-p{n}';cmd=['python3','-B',str(w/'run_owned.py'),name,exe,model,req,str(w/(name+'.json')),str(n),kind]
  if kind=='fused':cmd.append(str(w/f'tokens-p{n}.txt'))
  with (w/(name+'-parent.log')).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
  data=json.loads((w/(name+'.json')).read_text());assert data['status']=='pass'
  if kind=='scalar':(w/f'tokens-p{n}.txt').write_text(' '.join(map(str,data['generated_ids'])))
 subprocess.run(['python3',str(w/'compare.py'),str(w/f'scalar-p{n}.json'),str(w/f'fused-p{n}.json'),str(w/f'compare-p{n}.json')],env=dict(os.environ,OPENBLAS_NUM_THREADS='1'),check=True)
 print('completed numerical',n,flush=True)
