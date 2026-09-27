import pathlib,json,time,re,urllib.request,hashlib

def run_requests(out,port):
 root=out.parent
 cases=json.loads((root/'capability-direct-cases.json').read_text());results=[]
 for i,case in enumerate(cases):
  body=case['request'];payload=json.dumps(body,ensure_ascii=False).encode();start=time.monotonic();events=[];text='';usage=None;finish=None;done=False
  request=urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',data=payload,headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(request,timeout=240) as response:
   for raw in response:
    line=raw.decode().strip()
    if not line.startswith('data:'):continue
    data=line[5:].strip()
    if data=='[DONE]':done=True;break
    event=json.loads(data);events.append({'seconds':time.monotonic()-start,'event':event})
    if 'error' in event:raise RuntimeError(event)
    if event.get('usage'):usage=event['usage']
    for choice in event.get('choices',[]):
     text+=choice.get('delta',{}).get('content','')
     if choice.get('finish_reason'):finish=choice['finish_reason']
  (out/f'events-{i:02d}.json').write_text(json.dumps(events,ensure_ascii=False,indent=2))
  matches=re.findall(r'答案：([A-D])',text)
  valid=bool(done and usage and finish=='stop' and matches and usage['completion_tokens']<1024)
  result={'index':i,'subject':case['subject'],'id':case['id'],'target':case['target'],'answer':matches[-1] if matches else None,'valid':valid,'correct':bool(matches and matches[-1]==case['target']),'finish':finish,'done':done,'usage':usage,'text':text,'request_sha256':hashlib.sha256(payload).hexdigest()}
  results.append(result);(out/'capability-results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False),flush=True)
  # First sample calibrates only answer completeness, never selection by score.
  if not valid:raise RuntimeError('capability response contract failed; no score')
 summary={'requests':len(results),'parseable':sum(r['valid'] for r in results),'correct':sum(r['correct'] for r in results),'scope':'20-case four-subject 5-shot C-Eval regression screen, not full-suite certification'}
 (out/'capability-summary.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)

def lifecycle(out,port):
 import urllib.error
 cases=json.loads((out.parent/'capability-direct-cases.json').read_text());body=cases[0]['request'];url=f'http://127.0.0.1:{port}/v1/chat/completions'
 prior=json.loads((out/'capability-results.json').read_text())[0]
 malformed=dict(body,max_tokens=-1)
 try:
  urllib.request.urlopen(urllib.request.Request(url,data=json.dumps(malformed).encode(),headers={'Content-Type':'application/json'}),timeout=10)
  raise AssertionError('malformed request accepted')
 except urllib.error.HTTPError as e:
  assert e.code==400;error_body=e.read().decode()
 payload=json.dumps(body,ensure_ascii=False).encode();req=lambda:urllib.request.Request(url,data=payload,headers={'Content-Type':'application/json'})
 original=json.loads(pathlib.Path('.q3x-work/evidence/terminal-prefix-main-p40000-api-20260909-r2/request-body.json').read_text())
 original.update(prompt=original['prompt'][:1089],max_tokens=256,temperature=0,stream=True,stream_options={'include_usage':True})
 cancel_req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/completions',data=json.dumps(original).encode(),headers={'Content-Type':'application/json'})
 cancelled=[]
 with urllib.request.urlopen(cancel_req,timeout=240) as r:
  for raw in r:
   line=raw.decode().strip()
   if not line.startswith('data:') or line[5:].strip()=='[DONE]':continue
   event=json.loads(line[5:].strip())
   if 'error' in event: raise RuntimeError(event)
   if any(ch.get('text') for ch in event.get('choices',[])):cancelled.append(event)
   if len(cancelled)==3:break
 assert len(cancelled)==3
 time.sleep(0.5)
 text='';events=[];usage=None;finish=None;done=False
 with urllib.request.urlopen(req(),timeout=240) as r:
  for raw in r:
   line=raw.decode().strip()
   if not line.startswith('data:'):continue
   data=line[5:].strip()
   if data=='[DONE]':done=True;break
   event=json.loads(data);events.append(event)
   if event.get('usage'):usage=event['usage']
   for ch in event.get('choices',[]):
    text+=ch.get('delta',{}).get('content','')
    if ch.get('finish_reason'):finish=ch['finish_reason']
 assert done and finish=='stop' and text==prior['text'] and usage==prior['usage']
 with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz') as r:assert r.status==200
 (out/'lifecycle.json').write_text(json.dumps({'malformed_http':400,'malformed_body':error_body,'cancelled_after_content_events':len(cancelled),'cancelled_events':cancelled,'recovery_events':events,'recovery_text_equal':True,'recovery_usage_equal':True,'done':done,'finish':finish,'scope':'same-server short benchmark reuse and disconnect recovery; require matching conservative reset witness'},ensure_ascii=False,indent=2))
