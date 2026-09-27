import pathlib,time,subprocess,json,hashlib,os
w=pathlib.Path('.q3x-work/decode-qualification-20260927')
start=time.monotonic()
while not (w/'compare-fused-replay.json').exists():
 if time.monotonic()-start>900:raise RuntimeError('repeat timeout')
 time.sleep(1)
assert json.loads((w/'compare-fused-replay.json').read_text())['strict_production_equality']
p=pathlib.Path('.q3x-work/build/orin-release/qwen3x-eval-server')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before=sha(p)
with (w/'build-final.log').open('w') as f:
 for cmd in [['cmake','--build','--preset','orin-release','--target','qwen3x-eval-server','-j','4'],['cmake','--build','.q3x-work/build/fused-decode-admission','--target','q3x_fused_decode_test','q3x_fused_decode_protocol_test','-j','4']]:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
after=sha(p)
syms=subprocess.check_output(['nm','-C',str(p)],text=True)
forbidden=[x for x in syms.splitlines() if any(k in x for k in ['fused_decode::','cublasLt','q3x_launch_attention_scores_split','q3x_launch_attention_values_split'])]
(w/'default-artifact-bridge.json').write_text(json.dumps({'before_sha256':before,'after_sha256':after,'byte_identical':before==after,'forbidden_symbols':forbidden,'baseline_api_elf':json.loads((w/'capability-direct-scalar/identity.json').read_text())['executable_sha256'],'scope':'fresh Release/OFF rebuild; exact ELF bridge to this batch baseline API checks; no fused default promotion'},indent=2))
assert before==after and not forbidden
with (w/'fused-contract-parent.log').open('w') as f:
 subprocess.run(['python3','-B',str(w/'run_owned.py'),'fused-contract','.q3x-work/build/fused-decode-admission/q3x_fused_decode_test'],stdout=f,stderr=subprocess.STDOUT,check=True)
with (w/'host-protocol.log').open('w') as f:subprocess.run(['ctest','--test-dir','.q3x-work/build/fused-decode-admission','-R','^fused_decode_protocol$','--output-on-failure'],stdout=f,stderr=subprocess.STDOUT,check=True)
print('build, artifact bridge, fused contract and host protocol passed',flush=True)
