import pathlib,sqlite3,json
p=pathlib.Path('/home/rm01/Qwen3x-Orin/.q3x-work/decode-context-investigation-20260927/fused-profile')
c=sqlite3.connect('file:'+str(p/'capture.sqlite')+'?mode=ro',uri=True)
rows=c.execute('SELECT s.value,COUNT(*),SUM(k.end-k.start)/1e6,MIN(k.start),MAX(k.end) FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName GROUP BY k.demangledName ORDER BY SUM(k.end-k.start) DESC').fetchall()
items=[{'kernel':r[0],'calls':r[1],'gpu_ms':r[2]} for r in rows]
attention=[r for r in items if 'SinglePrefillWithKVCacheKernel' in r['kernel'] or 'MergeStates' in r['kernel']]
assert sum(r['calls'] for r in attention)==32,attention
assert not any('attention_scores_warp_positions' in r['kernel'] or 'attention_values_exact_24_4_256' in r['kernel'] for r in items)
resources=c.execute('SELECT DISTINCT s.value,k.registersPerThread,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,k.dynamicSharedMemory,k.localMemoryPerThread FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE s.value LIKE "%SinglePrefillWithKVCacheKernel%" OR s.value LIKE "%MergeStates%"').fetchall()
nvtx=c.execute('SELECT n.start,n.end,s.value FROM NVTX_EVENTS n JOIN StringIds s ON s.id=n.textId').fetchall()
assert len(nvtx)==1 and nvtx[0][2]=='q3x.decode.step'
assert all(r[3]>=nvtx[0][0] and r[4]<=nvtx[0][1] for r in rows)
processes=c.execute('SELECT DISTINCT p.globalPid,p.pid,p.name FROM PROCESSES p JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON k.globalPid=p.globalPid').fetchall()
assert len(processes)==1 and 'qwen3x' in processes[0][2],processes
result={'nvtx_ranges':nvtx,'kernel_processes':processes,'scope':'one first Decode step after P40000, S40001; capture-range q3x.decode.step; diagnostic only, no matched baseline profile','kernels':items,'kernel_count':sum(r['calls'] for r in items),'all_kernel_busy_sum_ms':sum(r['gpu_ms'] for r in items),'attention_busy_ms':sum(r['gpu_ms'] for r in attention),'kernel_span_ms':(max(r[4] for r in rows)-min(r[3] for r in rows))/1e6,'streams':c.execute('SELECT DISTINCT streamId FROM CUPTI_ACTIVITY_KIND_KERNEL').fetchall(),'globalPids':c.execute('SELECT DISTINCT globalPid FROM CUPTI_ACTIVITY_KIND_KERNEL').fetchall(),'attention_resources_columns':['kernel','registers_per_thread','grid_x','grid_y','grid_z','block_x','block_y','block_z','dynamic_shared_bytes','local_bytes_per_thread'],'attention_resources':resources}
(p/'kernel-summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('kernels','attention_resources')},indent=2));print(json.dumps(items[:5],indent=2))
