import json,pathlib,hashlib,re,subprocess
ROOT=pathlib.Path('/home/rm01/Qwen3x-Orin');OUT=ROOT/'.q3x-work/decode-context-investigation-20260927'
def read(p):return json.loads((OUT/p).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def phase(k):
 log=(OUT/k/'server.log').read_text();w=[]
 for line in log.splitlines():
  try:x=json.loads(line)
  except ValueError:continue
  if x.get('record')=='target-prefill-witness-v20':w.append(x)
 results=read(k+'/results.json')
 for r in results:
  events=read(k+'/events-'+str(r['prompt_tokens'])+'.json')
  finishes=[ch['finish_reason'] for e in events for ch in e['event'].get('choices',[]) if ch.get('finish_reason')]
  assert finishes==['length']
  r['finish_reason']='length'
  matches=[x for x in w if x['request']['body_sha256']==r['request_sha256']]
  assert len(matches)==1 and r['done'] and r['usage']['completion_tokens']==32 and r['content_events']==32
  x=matches[0];assert x['prompt']['fully_consumed'] and x['prompt']['consumed_tokens']==r['prompt_tokens']
  r['decode_tokens_per_second']=1000/r['decode_ms_per_token']
  r['server_decode_ms_per_token']=x['timing']['decode']['milliseconds']/31
  r['pure_prefill_ms']=x['timing']['pure_prefill']['milliseconds']
  r['token_ids_u32le_sha256']=x['prompt']['token_ids_u32le_sha256']
  r['text_sha256']=hashlib.sha256(r['text'].encode()).hexdigest()
  r['max_client_server_tpot_difference_ms']=abs(r['decode_ms_per_token']-r['server_decode_ms_per_token'])
 receipts=re.findall(r'RESEARCH_FUSED_GQA hits=(\d+) shadow_hits=(\d+)',log)
 return {'identity':read(k+'/identity.json'),'results':results,'cleanup':read(k+'/cleanup.json'),'wrapper_receipts':receipts}
data={'record':'qwen36-27b-decode-context-scaling-investigation','date':'2026-09-27','decision':'research direction only; no production numerical-contract change, architecture selection, release or quality qualification','work_package':read('work-package.json'),'source_audit':read('source-audit.json'),'native_source_identity':read('native-source-identity.json'),'reference_source_identity':read('reference-source-identity.json'),'prototype_build_commands':read('build-commands.json'),'prototype_build_hashes':read('build-hashes.json'),'shadow_admission':read('shadow/summary.json'),'phases':{k:phase(k) for k in ['native','shadow','fused']}}
if (OUT/'native-eager/results.json').exists():data['phases']['native-eager']=phase('native-eager')
if (OUT/'fused-profile/results.json').exists():data['phases']['fused-profile']=phase('fused-profile')
if (OUT/'fused-profile/kernel-summary.json').exists():data['profile']=read('fused-profile/kernel-summary.json')
if 'fused-profile' in data['phases']:data['phases']['fused-profile']['timing_authority']='T4 only: capture termination/report handling distorts streamed TPOT; do not use as a performance repeat'
a=data['phases'].get('native-eager',data['phases']['native'])['results'];b=data['phases']['fused']['results'];assert len(a)==len(b)==3
data['primary_environment']='Native original/default module-loading environment versus fused CUDA_MODULE_LOADING=EAGER. Two matched EAGER native startups failed before requests; no fully environment-matched pair was obtained. Directional comparison only, not architecture selection or formal performance qualification.'
data['final_audit']=read('final-audit.json')
data['dynamic_link_audit']=read('dynamic-link-audit.json')
data['comparison']=[]
for x,y in zip(a,b):
 assert x['request_sha256']==y['request_sha256'] and x['token_ids_u32le_sha256']==y['token_ids_u32le_sha256']
 data['comparison'].append({'prompt_tokens':x['prompt_tokens'],'baseline_ms':x['decode_ms_per_token'],'fused_ms':y['decode_ms_per_token'],'baseline_tokens_s':x['decode_tokens_per_second'],'fused_tokens_s':y['decode_tokens_per_second'],'speedup':x['decode_ms_per_token']/y['decode_ms_per_token'],'text_equal':x['text']==y['text']})
data['context_excess_ms']={'baseline':a[-1]['decode_ms_per_token']-a[0]['decode_ms_per_token'],'fused':b[-1]['decode_ms_per_token']-b[0]['decode_ms_per_token']}
data['context_excess_ms']['reduction_fraction']=1-data['context_excess_ms']['fused']/data['context_excess_ms']['baseline']
if 'fused-profile' in data['phases']:data['profile_process_output_equal']=b[-1]['text']==data['phases']['fused-profile']['results'][0]['text']
data['failed_attempts']={'native_first_preflight':'active Codex CPU observation; no server started; qualified as ordinary diagnostic on fresh retry','vllm_first':'scheduler configuration rejected max_num_batched_tokens=40000 < max_model_len=44096; no CUDA/API timing','vllm_second':'operator-stopped after >9 minutes of maximum-size multimodal startup before ready; no reference timing','fused_first':'existing sealed startup Graph-memory increment gate rejected 360701952 B > 256 MiB; wrapper hits=0, no request or performance result','fused_second':'unchanged retry also failed Graph memory gate, 357687296 B; no request or timing','native_eager_attempts':'both baseline EAGER startups failed Graph memory gate at 336334848 and319684608 B before API readiness; no timing; EAGER is not a qualified startup repair','eager_startup':'CUDA_MODULE_LOADING=EAGER used for successful third prototype attempt; matched baseline attempts did not reach readiness; preserve loading-mode mismatch and all startup failures'}
data['limitations']=['module-loading environment differs between completed native and fused arms; attempted matched control blocked at startup; gain is directional, not isolated formal timing','one pass per length, sequential lengths, no mirrored process repetition or statistical qualification','baseline Legacy-C512, not separate whole-core P40000/O16 deployment','changed numerical class and public scratch postcondition; research runner shim only','logical native route witness is unaware of link shim: final-step unchanged/exact claims in that witness are not candidate numerical attestation; wrapper receipt and linked binary identify intervention','quality, full state/logits, long output, Graph, cancellation, full context matrix and final packaging unqualified','stock-vLLM attempt produced no usable rate; no fresh vLLM parity claim','T4 historical baseline profile and new bounded candidate profile are not a matched profiling pair','no hardware ceiling or target-relaxation claim']
files=[p for p in OUT.rglob('*') if p.is_file() and p.suffix in ('.json','.log','.py','.cu','.txt','.nsys-rep','.sqlite') and p.name!='metadata.json']
data['raw_artifacts']={str(p.relative_to(ROOT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(files)}
(OUT/'metadata.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'comparison':data['comparison'],'context_excess':data['context_excess_ms'],'profile_output_equal':data.get('profile_process_output_equal')},indent=2))
