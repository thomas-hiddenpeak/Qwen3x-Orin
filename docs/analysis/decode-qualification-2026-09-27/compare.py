import json,sys,pathlib,numpy as np,hashlib
base=pathlib.Path(sys.argv[1]);cand=pathlib.Path(sys.argv[2]);out=pathlib.Path(sys.argv[3])
b=json.loads(base.read_text());c=json.loads(cand.read_text());rows=[];alarms=[]
assert b['status']==c['status']=='pass'
assert b['prompt_ids']==c['prompt_ids'] and b['generated_ids']==c['generated_ids']
assert b['terminal_prefix_observation']==c['terminal_prefix_observation']
def values(p):return (np.fromfile(p,dtype='<u2').astype(np.uint32)<<16).view(np.float32).astype(np.float64)
for p in sorted(base.parent.glob(base.name+'.step*.bf16')):
 suffix=p.name[len(base.name):];q=pathlib.Path(str(cand)+suffix)
 x,y=values(p),values(q);assert x.shape==y.shape
 finite=bool(np.isfinite(x).all() and np.isfinite(y).all());assert finite
 d=y-x;rel=float(np.linalg.norm(d)/max(np.linalg.norm(x),1e-30))
 row={'span':suffix,'elements':len(x),'finite':finite,'equal':bool(np.array_equal(x,y)),'relative_l2':rel,'maxabs':float(np.max(abs(d)))}
 if '.logits.' in suffix:
  def logprob(a):
   z=a-a.max();return z-np.log(np.exp(z).sum())
  lp,lq=logprob(x),logprob(y)
  row.update(kl=float(np.sum(np.exp(lp)*(lp-lq))),centered_rmse=float(np.std(d)),baseline_argmax=int(x.argmax()),candidate_argmax=int(y.argmax()),baseline_top2_margin=float(np.sort(x)[-1]-np.sort(x)[-2]))
  if row['kl']>0.001:alarms.append({'span':suffix,'KL':row['kl']})
 elif rel>0.01:alarms.append({'span':suffix,'relative_l2':rel})
 rows.append(row)
a={'scope':'teacher-forced numerical characterization, no automatic production qualification','baseline':str(base),'candidate':str(cand),'rows':rows,'engineering_alarms':alarms,'strict_production_equality':all(r['equal'] for r in rows),'same_forced_inputs':True,'prefix_observations_equal':True}
out.write_text(json.dumps(a,indent=2));print(json.dumps({'full_logits_max_KL':max(r.get('kl',0) for r in rows),'state_max_rel_l2':max(r['relative_l2'] for r in rows if 'kl' not in r),'alarms':alarms,'strict_production_equality':a['strict_production_equality']},indent=2))
