import pathlib,json,hashlib
w=pathlib.Path('.q3x-work/decode-numerical-repair-20260927');old=pathlib.Path('.q3x-work/decode-qualification-20260927')
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
lengths=[]
for n in [576,8192,40000]:
 x=read(w/f'compare-p{n}.json');previous=read(old/f'compare-p{n}.json')
 def maxima(r):
  ls=[a for a in r['rows'] if 'kl' in a];st=[a for a in r['rows'] if 'kl' not in a]
  return {'maximum_logit_kl':max(a['kl'] for a in ls),'maximum_state_span_relative_l2':max(a['relative_l2'] for a in st),'argmax_changes':[a for a in ls if a['baseline_argmax']!=a['candidate_argmax']],'engineering_alarms':r['engineering_alarms'],'strict_equality':r['strict_production_equality'],'conv_gdn':[a for a in st if 'conv' in a['span'] or 'gdn' in a['span']]}
 rows=[json.loads(l) for l in (w/f'fused-p{n}.json.attention.jsonl').read_text().splitlines()]
 lengths.append({'prompt':n,'outputs':16,'v2':maxima(previous),'v3':maxima(x),'attention_comparisons':len(rows),'all_repeat_equal':all(a['repeat_equal'] for a in rows),'all_finite':all(a['nonfinite']==0 for a in rows),'max_attention_relative_l2':max(a['relative_l2_vs_scalar'] for a in rows),'fp64_cells':[a for a in rows if 'fp64_heads' in a]})
b=read(old/'capability-direct-scalar/capability-results.json');c=read(w/'api-v3/capability-results.json');assert len(b)==len(c)==20
pairs=[]
for a,z in zip(b,c):
 assert a['request_sha256']==z['request_sha256'] and a['valid'] and z['valid']
 pairs.append({'subject':a['subject'],'id':a['id'],'target':a['target'],'baseline_answer':a['answer'],'v3_answer':z['answer'],'lost':a['correct'] and not z['correct'],'gained':not a['correct'] and z['correct'],'same_usage':a['usage']==z['usage']})
life=read(w/'api-v3/lifecycle.json');witness=[]
for line in (w/'api-v3/server.log').read_text().splitlines():
 try:r=json.loads(line)
 except ValueError:continue
 if str(r.get('record','')).startswith('target-prefill-witness'):witness.append(r)
manifest=[]
for p in sorted(w.rglob('*')):
 if p.is_file() and p.name not in ['summary.json','summary-console.log']:
  manifest.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
r={'schema_version':1,'parent':'4139551','work_package':read(w/'protocol.json'),'decision':'precision defects repaired in isolated v3; not production numerical qualification','lengths':lengths,'scalar_bridge_exact':read(w/'scalar-byte-bridge.json')['all_byte_exact'],'api':read(w/'api-v3/results.json'),'capability':{'scope':'same frozen 20-case C-Eval-derived direct-answer screen; not full-suite/long-context qualification','baseline_correct':sum(a['correct'] for a in b),'v3_correct':sum(a['correct'] for a in c),'lost':sum(a['lost'] for a in pairs),'gained':sum(a['gained'] for a in pairs),'pairs':pairs},'lifecycle':{k:v for k,v in life.items() if k not in ['cancelled_events','recovery_events']},'witnesses':{'count':len(witness),'records':sorted(set(a['record'] for a in witness)),'last_reset':witness[-1]['request_state_reset']},'health':read(w/'api-v3/health.json'),'cleanup':read(w/'api-v3/cleanup.json'),'default_bridge':read(w/'default-bridge.json'),'postflight':read(w/'postflight.json'),'thermal_audit':read(w/'thermal-audit.json'),'regression':{'v3':read(w/'contract-regression/result.json'),'v2_expected_failure':read(w/'v2-regression-r2/result.json')},'raw_files':manifest}
(w/'summary.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'lengths':[{k:a[k] for k in ['prompt','max_attention_relative_l2','all_finite','all_repeat_equal']} for a in lengths],'capability':r['capability'],'api':r['api']},ensure_ascii=False,indent=2))
