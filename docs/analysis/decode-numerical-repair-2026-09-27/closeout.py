import pathlib,json,time,subprocess,os
w=pathlib.Path('.q3x-work/decode-numerical-repair-20260927')
start=time.monotonic()
while not (w/'api-v3/cleanup.json').exists():
 if time.monotonic()-start>1600:raise RuntimeError('API completion timeout')
 time.sleep(1)
assert (w/'api-v3/lifecycle.json').exists()
with (w/'v2-regression-parent.log').open('w') as f:
 p=subprocess.run(['python3','-B',str(w/'run_owned.py'),'v2-regression',str(w/'v2-cancellation-regression')],stdout=f,stderr=subprocess.STDOUT)
assert p.returncode==10, p.returncode
print('v2 fails the added precision regression as expected',flush=True)
p=subprocess.run(['sudo','-n','python3','-B',str(w/'preflight.py'),'--output',str(w/'postflight.json'),'--samples','3'],capture_output=True,text=True)
print(p.stdout,flush=True)
verdict=json.loads(p.stdout)
assert not verdict.get('collection_errors') and all(x['code']=='unexpected_cpu_consumers' and set(x.get('pids',[]))<={872114} for x in verdict.get('reasons',[]))
for p in w.rglob('*.json'):
 if p.stat().st_uid==0:
  assert p.name.startswith('preflight') or p.name=='postflight.json'
  subprocess.run(['sudo','-n','chown',f'{os.getuid()}:{os.getgid()}',str(p)],check=True)
print('resource closure completed',flush=True)
