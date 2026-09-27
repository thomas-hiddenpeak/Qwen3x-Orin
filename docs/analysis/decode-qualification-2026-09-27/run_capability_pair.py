import pathlib,subprocess,os,time,json
w=pathlib.Path('.q3x-work/decode-qualification-20260927')
start=time.monotonic()
while not (w/'compare-p40000.json').exists():
 if time.monotonic()-start>1500:raise RuntimeError('numerical panel did not close')
 time.sleep(1)
for kind,exe in [('capability-scalar','.q3x-work/build/orin-release/qwen3x-eval-server'),('capability-fused','.q3x-work/build/fused-decode-admission/qwen3x-eval-server-fused-decode-admission')]:
 with (w/(kind+'-parent.log')).open('w') as f:
  subprocess.run(['python3','-B',str(w/'run_capability.py'),kind],env=dict(os.environ,DECODE_API_EXECUTABLE=str(pathlib.Path(exe).resolve())),stdout=f,stderr=subprocess.STDOUT,check=True)
 print('completed',kind,flush=True)
b=json.loads((w/'capability-scalar/capability-results.json').read_text());c=json.loads((w/'capability-fused/capability-results.json').read_text());assert len(b)==len(c)==20
pairs=[]
for x,y in zip(b,c):
 assert x['request_sha256']==y['request_sha256'] and x['valid'] and y['valid']
 pairs.append({'subject':x['subject'],'id':x['id'],'target':x['target'],'scalar':x['answer'],'fused':y['answer'],'lost':x['correct'] and not y['correct'],'gained':not x['correct'] and y['correct']})
r={'count':20,'scalar_correct':sum(x['correct'] for x in b),'fused_correct':sum(x['correct'] for x in c),'lost':sum(x['lost'] for x in pairs),'gained':sum(x['gained'] for x in pairs),'pairs':pairs,'scope':'four-subject first-five C-Eval 5-shot regression screen; no full-suite or long-context capability claim'}
(w/'capability-comparison.json').write_text(json.dumps(r,indent=2));print(r,flush=True)
