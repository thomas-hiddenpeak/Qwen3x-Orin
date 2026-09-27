import pathlib,json,hashlib,subprocess
w=pathlib.Path('.q3x-work/decode-qualification-20260927')
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
lengths=[]
for n in [576,8192,40000]:
 compare=read(w/f'compare-p{n}.json');rows=compare['rows'];logits=[r for r in rows if 'kl' in r];state=[r for r in rows if 'kl' not in r]
 att=[]
 for arm in ['scalar','fused']:
  rs=[json.loads(l) for l in (w/f'{arm}-p{n}.json.attention.jsonl').read_text().splitlines()]
  att.append({'arm':arm,'comparisons':len(rs),'finite':all(r['nonfinite']==0 for r in rs),'same_input_repeat_equal':all(r['repeat_equal'] for r in rs),'max_relative_l2_vs_scalar':max(r['relative_l2_vs_scalar'] for r in rs),'independent_fp64':[r for r in rs if 'fp64_heads' in r]})
 lengths.append({'prompt':n,'outputs':16,'same_inputs':compare['same_forced_inputs'],'prefix_observations_equal':compare['prefix_observations_equal'],'full_logits_max_kl':max(r['kl'] for r in logits),'full_logits_max_centered_rmse':max(r['centered_rmse'] for r in logits),'raw_argmax_changes':[r for r in logits if r['baseline_argmax']!=r['candidate_argmax']],'maximum_state_span_relative_l2':max(r['relative_l2'] for r in state),'conv_gdn':[r for r in state if 'conv' in r['span'] or 'gdn' in r['span']],'engineering_alarms':compare['engineering_alarms'],'strict_equality':compare['strict_production_equality'],'attention':att})
cap=read(w/'capability-direct-comparison.json') if (w/'capability-direct-comparison.json').exists() else None
lifecycle=[]
for arm in ['scalar','fused']:
 p=w/('lifecycle-scalar-r2' if arm=='scalar' else 'capability-direct-fused');life=read(p/'lifecycle.json') if (p/'lifecycle.json').exists() else None
 witnesses=[]
 if (p/'server.log').exists():
  for line in (p/'server.log').read_text().splitlines():
   try:r=json.loads(line)
   except ValueError:continue
   if str(r.get('record','')).startswith('target-prefill-witness'):witnesses.append(r)
 lifecycle.append({'arm':arm,'artifact_directory':str(p),'checks':None if life is None else {k:v for k,v in life.items() if k not in ['cancelled_events','recovery_events']},'successful_witnesses':len(witnesses),'last_reset':witnesses[-1].get('request_state_reset') if witnesses else None,'cleanup':read(p/'cleanup.json') if (p/'cleanup.json').exists() else None})
files=[]
for p in sorted(w.rglob('*')):
 if p.is_file() and p.name not in ['summary.json','summary-console.log']:
  files.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
r={'schema_version':1,'parent_commit':'5038344','decision_unit':'architecture_candidate_numerical_qualification','architecture':'AC-DECODE-FUSED-GQA-PRODUCT-20260927','decision':'not_admitted_after_predeclared_numerical_alarms; no production-contract change','owner_speed_acceptance':'approximately 8.55 tok/s temporarily accepted, no further performance exploration','scalar_teacher_forcing_control_exact':read(w/'compare-scalar-replay.json')['strict_production_equality'],'fused_repeat_control_exact':read(w/'compare-fused-replay.json')['strict_production_equality'] if (w/'compare-fused-replay.json').exists() else None,'lengths':lengths,'capability':cap,'invalid_verbose_capability_calibration':read(w/'capability-scalar/capability-results.json'),'lifecycle':lifecycle,'default_artifact_bridge':read(w/'default-artifact-bridge.json'),'lifecycle_harness_erratum':read(w/'lifecycle-harness-erratum.json'),'postflight':read(w/'postflight.json'),'not_executed_after_numerical_stop':['whole-core Prefill composition and O256 capacity expansion','mirrored composed performance qualification','production install or numerical promotion','complete public benchmark suite'],'raw_files':files}
(w/'summary.json').write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'lengths':[{k:x[k] for k in ['prompt','full_logits_max_kl','maximum_state_span_relative_l2','strict_equality']} for x in lengths],'capability':cap,'invalid_verbose_capability_calibration':read(w/'capability-scalar/capability-results.json'),'lifecycle':lifecycle},ensure_ascii=False,indent=2))
