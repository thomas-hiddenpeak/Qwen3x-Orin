import pathlib,json,hashlib,pyarrow.parquet as pq
w=pathlib.Path('.q3x-work/decode-qualification-20260927');subjects={'computer_network':'计算机网络','business_administration':'工商管理','law':'法学','clinical_medicine':'临床医学'};cases=[]
def fmt(r):return '问题：'+r['question']+'\n选项：\n'+'\n'.join(f'{c}. {r[c]}' for c in 'ABCD')
for subject,label in subjects.items():
 dev=pq.read_table(w/'ceval'/f'{subject}-dev.parquet').to_pylist()[:5];val=pq.read_table(w/'ceval'/f'{subject}-val.parquet').to_pylist()[:5]
 few='以下是一些示例问题：\n\n'+'\n\n'.join(fmt(r)+'\n解析：'+r.get('explanation','')+'\n答案：'+r['answer'] for r in dev)+'\n\n\n'
 for row in val:
  prompt=few+f'以下是中国关于{label}的单项选择题，请选出其中的正确答案。你的回答的最后一行应该是这样的格式："答案：[LETTER]"（不带引号），其中 [LETTER] 是 A、B、C、D 中的一个。\n\n'+fmt(row)+'\n'
  body={'model':'qwen3.6-27b-nvfp4','messages':[{'role':'user','content':prompt}],'max_tokens':1024,'temperature':0,'stream':True,'stream_options':{'include_usage':True}}
  cases.append({'subject':subject,'id':row['id'],'target':row['answer'],'request':body})
(w/'capability-cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2));print(len(cases));print(hashlib.sha256((w/'capability-cases.json').read_bytes()).hexdigest())
