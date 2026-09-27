import pathlib,hashlib,json
w=pathlib.Path('.q3x-work/decode-numerical-repair-20260927');old=pathlib.Path('.q3x-work/decode-qualification-20260927');rows=[]
for p in sorted(old.glob('scalar-p576.json.step*.bf16')):
 q=w/p.name;a=hashlib.sha256(p.read_bytes()).hexdigest();b=hashlib.sha256(q.read_bytes()).hexdigest()
 rows.append({'suffix':p.name,'baseline_sha256':a,'current_sha256':b,'equal':a==b})
assert len(rows)==84,len(rows)
assert all(r['equal'] for r in rows)
(w/'scalar-byte-bridge.json').write_text(json.dumps({'all_byte_exact':True,'spans':rows},indent=2)+'\n')
print('scalar raw buffers byte-exact',len(rows))
