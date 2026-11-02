import os
from pathlib import Path
import subprocess
import json
root = Path(__file__).resolve().parents[2]
out = root / 'benchmark_results/megamoe'
out.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
env.update(MOONEP_BACKEND='npu', ASCEND_MOONEP_HEAP='shmem', ASCEND_MOONEP_HEAP_BYTES='8589934592', HCCL_NPU_SOCKET_PORT_RANGE='62000-62031', OMP_NUM_THREADS='2')
env['LD_LIBRARY_PATH'] = ':'.join(p for p in env.get('LD_LIBRARY_PATH','').split(':') if p)
cases = [(96,5120,2304,64),(96,5120,2304,256),(96,7168,3072,64),(96,7168,3072,256),(384,5120,2304,4096),(384,5120,2304,16384),(384,7168,3072,4096),(384,7168,3072,16384)]
statuses = []
for e,h,i,s in cases:
    stem=f'e{e}_h{h}_s{s}'
    csv_path = out / (stem+'.csv')
    if csv_path.exists():
        csv_path.unlink()
    args=f'--tokens {s} --hidden {h} --experts {e} --topk 6 --routing random-topk --plan-source golden --flows dispatch_fwd,combine_fwd --warmup 5 --iters 20 --no-cudagraph --no-allow-stub --out {out / (stem+".csv")}'
    cmd=['timeout','--signal=TERM','--kill-after=20s','300s','bash','npu_patch/benchmarks/run_bench.sh','bench_comm.py',args]
    print('START',stem,flush=True)
    with (out / (stem+'.log')).open('w') as log:
        result = subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    statuses.append(dict(experts=e,hidden=h,intermediate=i,tokens=s,returncode=result.returncode))
    (out/'run_status.json').write_text(json.dumps(statuses,indent=2))
    print('END',stem,result.returncode,flush=True)
