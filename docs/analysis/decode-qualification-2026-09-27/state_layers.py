import numpy as np,pathlib,json
w=pathlib.Path('.q3x-work/decode-qualification-20260927');rows=[]
for n in [576,8192,40000]:
 for step in [0,15]:
  for role in ['conv','gdn']:
   a=np.memmap(w/f'scalar-p{n}.json.step{step}.{role}.bf16',dtype='<u2',mode='r');b=np.memmap(w/f'fused-p{n}.json.step{step}.{role}.bf16',dtype='<u2',mode='r');assert len(a)==len(b) and len(a)%48==0
   per=len(a)//48
   for slot in range(48):
    start=slot*per;end=start+per
    x=(np.array(a[start:end],dtype=np.uint32)<<16).view(np.float32).astype(np.float64);y=(np.array(b[start:end],dtype=np.uint32)<<16).view(np.float32).astype(np.float64)
    rows.append({'prompt':n,'step':step,'role':role,'layer':slot+slot//3,'relative_l2':float(np.linalg.norm(y-x)/max(np.linalg.norm(x),1e-30)),'maxabs':float(np.max(abs(y-x)))})
(w/'state-layer-diagnostics.json').write_text(json.dumps({'scope':'post-screen localization only; no new threshold or numerical acceptance','rows':rows},indent=2))
for n in [576,8192,40000]:
 for role in ['conv','gdn']:print(n,role,max((r for r in rows if r['prompt']==n and r['role']==role),key=lambda r:r['relative_l2']))
