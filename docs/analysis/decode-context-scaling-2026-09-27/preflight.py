import importlib.util,sys,re
p='/home/rm01/Qwen3x-Orin/tools/evaluation/orin_perf_preflight.py'
s=importlib.util.spec_from_file_location('preflight',p); m=importlib.util.module_from_spec(s); sys.modules[s.name]=m; s.loader.exec_module(m)
original=m.collect_tegrastats
def collect(*args):
    result,errors=original(*args)
    for item in result.get('samples',[]):
        item['raw']=re.sub(r'\b\S*fan\S*\s+\S+', '',item['raw'],flags=re.I)
    result['unparsed_lines']=[re.sub(r'\b\S*fan\S*\s+\S+', '',x,flags=re.I) for x in result.get('unparsed_lines',[])]
    return result,errors
m.collect_tegrastats=collect
sys.exit(m.main())
