"""Run all eight resident-weight BF16 forward configurations in the NPU environment."""
import json
import os
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parents[2]
out=root/'benchmark_results/megamoe'
out.mkdir(parents=True, exist_ok=True)
env=os.environ.copy()
env.update(MOONEP_BACKEND='npu',ASCEND_MOONEP_HEAP='shmem',ASCEND_MOONEP_HEAP_BYTES='8589934592',OMP_NUM_THREADS='2',HCCL_NPU_SOCKET_PORT_RANGE='62000-62031')
env['LD_LIBRARY_PATH']=':'.join(p for p in env.get('LD_LIBRARY_PATH','').split(':') if p)
statuses=[]
for e,h,i,s in [(96,5120,2304,64),(96,5120,2304,256),(96,7168,3072,64),(96,7168,3072,256),(384,5120,2304,4096),(384,5120,2304,16384),(384,7168,3072,4096),(384,7168,3072,16384)]:
    stem=f'full_e{e}_h{h}_s{s}'
    target=out/(stem+'.csv')
    if os.environ.get('MOONEP_SKIP_EXISTING')=='1' and target.exists():
        print('SKIP',stem,flush=True)
        statuses.append(dict(case=stem,returncode=0,existing=True))
        continue
    if target.exists(): target.unlink()
    args=f'--tokens {s} --hidden {h} --intermediate {i} --experts {e} --peak-tflops 376 --out {target}'
    print('START',stem,flush=True)
    with (out/(stem+'.log')).open('w') as f:
        proc=subprocess.run(['timeout','--kill-after=20s','420s','bash','npu_patch/benchmarks/run_bench.sh','bench_moe_forward.py',args],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
    statuses.append(dict(case=stem,returncode=proc.returncode))
    (out/'full_run_status.json').write_text(json.dumps(statuses,indent=2)+'\n')
    print('END',stem,proc.returncode,flush=True)
