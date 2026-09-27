import os,sys,json,time,subprocess,signal,pathlib,hashlib,urllib.request,threading,re
ROOT=pathlib.Path('/home/rm01/Qwen3x-Orin'); WORK=ROOT/'.q3x-work'; OUT=WORK/'decode-qualification-20260927'
MODEL=pathlib.Path('/home/rm01/models/dev/llm/nvidia/Qwen3.6-27B-NVFP4')
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def save(p,x): p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
def environment():
    e=os.environ.copy()
    for k in list(e):
        if k.startswith(('Q3X_','VLLM_')): e.pop(k)
    e.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',FLASHINFER_NO_DOWNLOAD='1',VLLM_NO_USAGE_STATS='1',VLLM_DO_NOT_TRACK='1',VLLM_LOG_STATS_INTERVAL='10',CUDA_HOME='/usr/local/cuda-13.3',LD_LIBRARY_PATH='/usr/local/cuda-13.3/lib64',MAX_JOBS='4')
    for k,v in {'TMPDIR':'tmp','XDG_CACHE_HOME':'cache','XDG_CONFIG_HOME':'config','XDG_DATA_HOME':'data','HF_HOME':'cache/hf','TORCH_HOME':'cache/torch','TORCH_EXTENSIONS_DIR':'cache/torch-extensions','TORCHINDUCTOR_CACHE_DIR':'cache/torchinductor','TRITON_CACHE_DIR':'cache/triton','FLASHINFER_WORKSPACE_BASE':'cache/flashinfer','VLLM_CACHE_ROOT':'cache/vllm','VLLM_CONFIG_ROOT':'cache/vllm/config','VLLM_RPC_BASE_PATH':'.','CUDA_CACHE_PATH':'cache/cuda','VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR':'cache/flashinfer-autotune','HUMMING_CACHE_DIR':'cache/humming','HUMMING_TMP_DIR':'tmp/humming'}.items():
        p=(WORK/v).resolve(); p.mkdir(parents=True,exist_ok=True); e[k]=str(p)
    e['CUDA_MODULE_LOADING']='LAZY'
    e['PATH']='/home/rm01/vllmEvn/.venv/bin:/usr/local/cuda-13.3/bin:'+e['PATH']
    return e

def run(kind):
    out=OUT/kind; out.mkdir(exist_ok=True)
    pre=subprocess.run(['sudo','-n','python3','-B',str(OUT/'preflight.py'),'--output',str(out/f'preflight-{time.time_ns()}.json'),'--samples','5'],capture_output=True,text=True)
    print(pre.stdout,flush=True)
    if pre.returncode:
        verdict=json.loads(pre.stdout)
        benign=not verdict.get('collection_errors') and all(r.get('code')=='unexpected_cpu_consumers' and set(r.get('pids',[])) <= {872114} for r in verdict.get('reasons',[]))
        if not benign: raise RuntimeError('preflight failed '+pre.stdout+pre.stderr)
        save(out/'preflight-qualification.json',{'scope':'ordinary diagnostic','observation':'active Codex control process CPU activity; no unexpected GPU holder; no release timing claim','raw_verdict':verdict})
    mem_before=pathlib.Path('/proc/meminfo').read_text(); subprocess.run(['sync'],check=True)
    dc=subprocess.run(['sudo','-n','tee','/proc/sys/vm/drop_caches'],input='3\n',capture_output=True,text=True)
    save(out/'cache-preparation.json',{'before':mem_before,'after':pathlib.Path('/proc/meminfo').read_text(),'rc':dc.returncode,'stdout':dc.stdout,'stderr':dc.stderr})
    port=18872 if kind=='vllm' else 18871
    if kind!='vllm':
        cmd=[os.environ['DECODE_API_EXECUTABLE'],str(MODEL),'--host','127.0.0.1','--port',str(port),'--model','qwen3.6-27b-nvfp4']
    else:
        cmd=['/home/rm01/vllmEvn/.venv/bin/vllm','serve',str(MODEL),'--host','127.0.0.1','--port',str(port),'--served-model-name','qwen3.6-27b-nvfp4','--max-model-len','44096','--max-num-seqs','1','--max-num-batched-tokens','44096','--gpu-memory-utilization','0.78','--kv-cache-dtype','bfloat16','--mamba-cache-dtype','bfloat16','--mamba-ssm-cache-dtype','bfloat16','--no-enable-prefix-caching','--no-enable-chunked-prefill','--attention-backend','FLASHINFER','--mamba-backend','TRITON','--generation-config','vllm','--default-chat-template-kwargs','{"enable_thinking":false}','--enforce-eager']
    save(out/'identity.json',{'command':cmd,'executable_sha256':sha(cmd[0]),'git':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'model_config_sha256':sha(MODEL/'config.json'),'environment_overrides':{k:v for k,v in environment().items() if k in ['CUDA_MODULE_LOADING','CUDA_HOME','FLASHINFER_WORKSPACE_BASE','VLLM_CACHE_ROOT','TRITON_CACHE_DIR']},'scope':'C-Eval 20-case API regression screen; no timing or release qualification'})
    if kind=='fused-profile':
        cmd.append('--nvtx-phase-ranges')
        cmd=['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--capture-range=nvtx','--nvtx-capture=q3x.decode.step','--capture-range-end=stop','--force-overwrite=true','--output',str(out/'capture')]+cmd
        save(out/'profiler-command.json',cmd)
    env=environment()
    if kind=='shadow':env['DECODE_RESEARCH_SHADOW']='1'
    log=open(out/'server.log','w'); srv=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT,start_new_session=True)
    save(out/'process.json',{'pid':srv.pid})
    mon=subprocess.Popen(['/usr/bin/tegrastats','--interval','1000'],stdout=subprocess.PIPE,text=True)
    temps=[]
    def monitor():
        with open(out/'telemetry.log','w') as f:
            for line in mon.stdout:
                line=re.sub(r'\b\S*fan\S*\s+\S+','',line,flags=re.I); f.write(line); f.flush()
                vals=[float(x) for x in re.findall(r'@([\d.]+)C',line)]
                if vals: temps.append(max(vals))
                if vals and max(vals)>90:
                    os.killpg(srv.pid,signal.SIGTERM); return
    t=threading.Thread(target=monitor,daemon=True);t.start()
    started=time.monotonic(); results=[]
    try:
        ready=False
        for _ in range(1200):
            if srv.poll() is not None: raise RuntimeError('server exited '+str(srv.returncode))
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/'+('health' if kind=='vllm' else 'healthz'),timeout=1) as r: ready=r.status==200
            except Exception: pass
            if ready:break
            time.sleep(1)
        if not ready:raise RuntimeError('startup timeout')
        print(kind,'ready',time.monotonic()-started,flush=True)
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz') as r: save(out/'health.json',json.load(r))
        from capability_requests import run_requests, lifecycle
        run_requests(out,port)
        lifecycle(out,port)
    finally:
        if srv.poll() is None:
            os.killpg(srv.pid,signal.SIGINT)
            try:srv.wait(timeout=40)
            except subprocess.TimeoutExpired:
                os.killpg(srv.pid,signal.SIGTERM)
                try:srv.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(srv.pid,signal.SIGKILL);srv.wait(timeout=10)
        mon.terminate();mon.wait(timeout=5);t.join(timeout=5);log.close()
        save(out/'cleanup.json',{'server_returncode':srv.returncode,'max_temp_c':max(temps) if temps else None,'memory':pathlib.Path('/proc/meminfo').read_text()})
        print(kind,'stopped',srv.returncode,flush=True)
if __name__=='__main__':run(sys.argv[1])
