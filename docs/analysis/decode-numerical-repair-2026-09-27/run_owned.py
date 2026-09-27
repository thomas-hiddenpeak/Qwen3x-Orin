import pathlib,subprocess,os,json,time,signal,threading,re,sys,hashlib
root=pathlib.Path('/home/rm01/Qwen3x-Orin');work=root/'.q3x-work/decode-numerical-repair-20260927'
name=sys.argv[1];cmd=sys.argv[2:];out=work/name;out.mkdir()
def save(n,x):(out/n).write_text(json.dumps(x,indent=2))
p=subprocess.run(['sudo','-n','python3','-B',str(work/'preflight.py'),'--output',str(out/'preflight.json'),'--samples','3'],capture_output=True,text=True)
print(p.stdout,flush=True);pre=json.loads(p.stdout)
benign=not pre.get('collection_errors') and all(r.get('code')=='unexpected_cpu_consumers' and set(r.get('pids',[]))<={872114} for r in pre.get('reasons',[]))
if p.returncode and not benign:raise RuntimeError('preflight rejected')
save('preflight-context.json',{'ordinary_correctness_or_diagnostic':True,'active_control_cpu_only':bool(p.returncode)})
mem=pathlib.Path('/proc/meminfo').read_text();subprocess.run(['sync'],check=True)
drop=subprocess.run(['sudo','-n','tee','/proc/sys/vm/drop_caches'],input='3\n',capture_output=True,text=True)
save('cache.json',{'before':mem,'after':pathlib.Path('/proc/meminfo').read_text(),'rc':drop.returncode})
env=os.environ.copy()
for k in list(env):
 if k.startswith('Q3X_'):env.pop(k)
env.update(CUDA_MODULE_LOADING='LAZY',Q3X_RUN_SM87_DECODE_GRAPH_PRODUCTION='1',PYTHONDONTWRITEBYTECODE='1')
save('identity.json',{'command':cmd,'sha256':hashlib.sha256(pathlib.Path(cmd[0]).read_bytes()).hexdigest(),'git':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'dirty_diff_sha256':hashlib.sha256(subprocess.check_output(['git','diff'])).hexdigest(),'loading':'LAZY'})
f=open(out/'process.log','w');child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True);save('process.json',{'pid':child.pid})
mon=subprocess.Popen(['/usr/bin/tegrastats','--interval','1000'],stdout=subprocess.PIPE,text=True);temperatures=[]
def monitor():
 with open(out/'telemetry.log','w') as t:
  for line in mon.stdout:
   line=re.sub(r'\b\S*fan\S*\s+\S+','',line,flags=re.I);t.write(line);t.flush()
   values=[float(x) for x in re.findall(r'@([\d.]+)C',line)]
   if values:temperatures.append(max(values))
   if values and max(values)>90:os.killpg(child.pid,signal.SIGTERM);return
thread=threading.Thread(target=monitor,daemon=True);thread.start();start=time.monotonic();rc=None
try:rc=child.wait(timeout=1200)
finally:
 if child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=30)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
 mon.terminate();mon.wait(timeout=5);thread.join(timeout=5);f.close()
 save('result.json',{'rc':child.returncode,'seconds':time.monotonic()-start,'max_temperature_c':max(temperatures) if temperatures else None,'memory_after':pathlib.Path('/proc/meminfo').read_text()})
print('completed',name,child.returncode,flush=True);sys.exit(child.returncode)
