"""Build/test the exact production objects, then measure C planning and allocations."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import platform
import shlex
import statistics
import subprocess
import sys

import runtime as rt

ROOT=Path(__file__).resolve().parents[1]
SOURCES=['include/tiny_arena.h','src/internal.h','src/graph.c','src/planner.c','src/execute.c',
    'tools/workloads.h','tools/workloads.c','tools/benchmark.c','tools/runtime.py',
    'tools/run_benchmark.py','tools/verify_report.py','tools/visualize.py','tests/test_native.c',
    'tests/test_planner.py','tests/test_reporting.py','tests/test_mutations.py','tests/check_no_heap.py','Makefile']
OBJECTS=['build/graph.o','build/planner.o','build/execute.o','build/workloads.o']
NAMES=['chain','residual','fork_outputs','residual_stack']
MODES=['separate','reuse']


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def command(args,**kwargs):return subprocess.check_output(args,cwd=ROOT,text=True,**kwargs)


def build():
    before={p:digest(ROOT/p) for p in SOURCES}
    output=command(['make','all'])
    commands=[shlex.split(line) for line in output.splitlines() if line.startswith('cc ')]
    rt.require(len(commands)==7,'unexpected compiler/link command inventory')
    subprocess.run(['build/test_native'],cwd=ROOT,check=True)
    for mode in ([],['-O']):
        subprocess.run([sys.executable]+mode+['tests/test_planner.py','build/libtiny_arena.so'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'tests/check_no_heap.py']+OBJECTS[:3],cwd=ROOT,check=True)
    rt.require(before=={p:digest(ROOT/p) for p in SOURCES},'source changed during build/test')
    record=dict(sources=before,commands=commands,compiler=command(['cc','--version']).splitlines()[0],
        objects={p:digest(ROOT/p) for p in OBJECTS},binaries={p:digest(ROOT/p) for p in
        ['build/libtiny_arena.so','build/benchmark','build/test_native']})
    save(ROOT/'build/build.json',record)
    return record


def schedule(run,samples):
    out=[]
    for sample in range(samples):
        for order in range(4):
            case=(order+sample+run)%4
            for pos in range(2):out.append((sample,NAMES[case],MODES[(pos+run+sample+case)%2]))
    return out


def summaries(records):
    stats={}
    for name in NAMES:
        stats[name]={}
        for mode in MODES:
            values=[float(statistics.median(row['elapsed_ns']/r['repetitions'] for row in r['raw']
                    if row['graph']==name and row['mode']==mode)) for r in records]
            stats[name][mode]=dict(median_ns=float(statistics.median(values)),process_medians_ns=values,
                minimum_ns=min(values),maximum_ns=max(values))
    return stats


def benchmark(args):
    revision=command(['git','rev-parse','HEAD']).strip()
    sources={p:digest(ROOT/p) for p in SOURCES}
    for name,sha in sources.items():
        blob=subprocess.check_output(['git','show',revision+':'+name],cwd=ROOT)
        rt.require(hashlib.sha256(blob).hexdigest()==sha,'commit measurement sources first: '+name)
    record=build()
    args.output.mkdir(parents=True,exist_ok=False)
    lib=rt.load(ROOT/'build/libtiny_arena.so')
    correctness=rt.fixtures(lib)
    save(args.output/'correctness.json',correctness)
    records=[]
    for run in range(args.runs):
        raw=command(['build/benchmark',str(run),str(args.samples),str(args.repetitions)])
        data=json.loads(raw);save(args.output/('run-%02d.json'%run),data);records.append(data)
        print('Recorded C process',run+1,flush=True)
    rt.require(sources=={p:digest(ROOT/p) for p in SOURCES} and all(digest(ROOT/p)==sha
        for p,sha in {**record['objects'],**record['binaries']}.items()),'source or production build changed during measurement')
    cpu=None
    if sys.platform=='darwin':
        result=subprocess.run(['sysctl','-n','machdep.cpu.brand_string'],capture_output=True,text=True)
        cpu=result.stdout.strip() if result.returncode==0 else None
    else:cpu=platform.processor() or None
    save(args.output/'manifest.json',dict(schema=1,protocol='static-arena-inclusive-lifetimes-v1',
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),source_revision=revision,sources=sources,
        build=record,runs=args.runs,samples=args.samples,repetitions=args.repetitions,
        timer='C CLOCK_MONOTONIC; complete plan build/validation plus loop and volatile sink',
        python=platform.python_version(),platform=dict(system=platform.system(),release=platform.release(),
            machine=platform.machine(),cpu=cpu),
        abi=dict(size_t_bytes=rt.C.sizeof(rt.S),node_bytes=rt.C.sizeof(rt.Node),graph_bytes=rt.C.sizeof(rt.Graph),
            plan_bytes=rt.C.sizeof(rt.Plan),scratch_bytes=rt.C.sizeof(rt.Scratch),alignment=rt.ALIGN),
        summaries=summaries(records),files={p.name:digest(p) for p in args.output.glob('*.json')}))
    # Verification and rendering happen after persisted evidence has been checked.
    from verify_report import verify
    from visualize import render
    verify(args.output);render(args.output,args.output/'visualization')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results/local/arena')
    p.add_argument('--runs',type=int,default=6);p.add_argument('--samples',type=int,default=20)
    p.add_argument('--repetitions',type=int,default=1024)
    args=p.parse_args()
    rt.require(1<=args.runs<=24 and 1<=args.samples<=100 and 128<=args.repetitions<=4096,'invalid counts')
    benchmark(args)
