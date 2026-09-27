import importlib.util,sys,pathlib,os,json
root=pathlib.Path('/home/rm01/Qwen3x-Orin')
p=root/'docs/analysis/decode-context-scaling-2026-09-27/run_panel.py'
s=importlib.util.spec_from_file_location('panel',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
# Execute a frozen adapted copy: same preflight/cleanup machinery, separate
# artifact directories, explicit executable and identical loading protocol.
source=p.read_text()
source=source.replace("OUT=WORK/'decode-context-investigation-20260927'", "OUT=WORK/'decode-product-20260927'")
source=source.replace("e['CUDA_MODULE_LOADING']='EAGER'", "e['CUDA_MODULE_LOADING']='LAZY'")
source=source.replace("str(WORK/'build/orin-release/qwen3x-eval-server') if kind.startswith('native') else str(OUT/'qwen3x-decode-research')", "os.environ['DECODE_API_EXECUTABLE']")
source=source.replace("[1089,8192,40000]", "([1089] if kind.endswith('schema') else [] if kind.endswith('diagnostic') else [19,1089,40000] if kind.endswith('regression') else [40000] if kind.endswith('long') else [1089,8192,40000])")
source=source.replace('max_tokens=32,temperature=0',"max_tokens=(16 if kind.endswith(('regression','schema')) else (256 if kind.endswith('long') else 32)),temperature=0")
source=source.replace("if not done or not usage or usage['prompt_tokens']!=n:","if not done or not usage or usage['prompt_tokens']!=n or usage['completion_tokens']!=body['max_tokens'] or len(times)!=body['max_tokens']:")
source=source.replace("print(kind,'ready',time.monotonic()-started,flush=True)","print(kind,'ready',time.monotonic()-started,flush=True)\n        with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz') as r: save(out/'health.json',json.load(r))")
source=source.replace("'scope':'T3 diagnostic context curve; eager vLLM reference; no release selection'","'scope':'T3 convergence regression/admission; LAZY loading; no numerical or release qualification'")
source=source.replace("except subprocess.TimeoutExpired:os.killpg(srv.pid,signal.SIGTERM);srv.wait(timeout=20)", "except subprocess.TimeoutExpired:\n                os.killpg(srv.pid,signal.SIGTERM)\n                try:srv.wait(timeout=20)\n                except subprocess.TimeoutExpired:os.killpg(srv.pid,signal.SIGKILL);srv.wait(timeout=10)")
exec(compile(source,str(p),'exec'),{'__name__':'__main__'})
