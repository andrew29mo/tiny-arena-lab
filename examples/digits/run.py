"""Build/test/measure a real model through the unchanged production arena engine."""
import argparse
import ctypes as C
import datetime as dt
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import common as cm
from common import rt,ROOT,HERE
BUILD=ROOT/'build/digits'
SOURCES=['include/tiny_arena.h','src/internal.h','src/graph.c','src/planner.c','src/execute.c',
         'tools/runtime.py','tools/visualize.py']+['examples/digits/'+n for n in
         ['model.json','fixture.json','common.py','fixture.h','fixture.c','benchmark.c','run.py','verify.py','test_digits.py','prepare.py','requirements.txt']]


def commands(sanitize=False):
    flags=cm.FLAGS if not sanitize else ['-std=c11','-O1','-g','-Wall','-Wextra','-Wpedantic','-Werror',
        '-ffp-contract=off','-fsanitize=address,undefined','-fno-omit-frame-pointer']
    includes=['-Iinclude','-Iexamples/digits','-Ibuild/digits']
    if sanitize:return [['cc']+includes+flags+['examples/digits/benchmark.c','examples/digits/fixture.c',
        'src/graph.c','src/planner.c','src/execute.c','-lm','-o','build/digits/sanitize']]
    result=[['cc']+includes+flags+['-fPIC','-c',source,'-o','build/digits/'+name+'.o'] for name,source in
        [('graph','src/graph.c'),('planner','src/planner.c'),('execute','src/execute.c'),('fixture','examples/digits/fixture.c')]]
    objs=['build/digits/'+n+'.o' for n in ('graph','planner','execute','fixture')]
    return result+[['cc','-shared']+objs+['-lm','-o','build/digits/engine.so'],
        ['cc']+includes+flags+['examples/digits/benchmark.c']+objs+['-lm','-o','build/digits/benchmark']]


def build(sanitize=False):
    BUILD.mkdir(parents=True,exist_ok=True);(BUILD/'digits_data.h').write_text(cm.header())
    for cmd in commands(sanitize):subprocess.run(cmd,cwd=ROOT,check=True)
    if sanitize:
        subprocess.run([str(BUILD/'sanitize'),'0','1'],stdout=subprocess.DEVNULL,check=True,
            env={**os.environ,'ASAN_OPTIONS':'halt_on_error=1','UBSAN_OPTIONS':'halt_on_error=1'})
        print('PASS: ASan/UBSan real-model capacity rejection, both layouts, every held-out example and tail')
        return
    subprocess.run([sys.executable,str(ROOT/'tests/check_no_heap.py')]+[
        'build/digits/'+n+'.o' for n in ('graph','planner','execute')],cwd=ROOT,check=True)
    cm.save(BUILD/'build.json',dict(commands=commands(),compiler=subprocess.check_output(['cc','--version'],text=True).splitlines()[0],
        generated_header_sha256=cm.digest(BUILD/'digits_data.h'),objects={str(p.relative_to(ROOT)):cm.digest(p)
        for p in sorted(BUILD.glob('*.o'))},binaries={str(p.relative_to(ROOT)):cm.digest(p) for p in
        [BUILD/'engine.so',BUILD/'benchmark']}))


def correctness(lib=None):
    model,f=cm.load_artifacts();lib=lib or cm.library();records={}
    rt.require(list(lib.digits_inputs()[:450*64])==[v/16 for row in f['pixels'] for v in row] and
        list(lib.digits_labels()[:450])==f['labels'],'actual C fixture inputs/labels differ')
    for batch,budget in cm.BATCHES.items():
        g,nodes,out=cm.graph(lib,batch);plans=[];arenas=[];predictions=[[],[]];logits=[[],[]];worst=[0.,0.]
        for mode in range(2):
            p,s=rt.Plan(),rt.Scratch()
            rt.require(lib.ta_plan_build(C.byref(g),mode,C.byref(p),C.byref(s))==0,'model plan failed')
            plans.append(p);arenas.append(rt.Arena(budget if mode else p.arena_bytes))
        limited=rt.Arena(budget);dummy=(rt.F*(batch*64))();inp=(rt.P*1)(dummy);counts=(rt.S*1)(batch*64)
        before=C.string_at(limited.address,budget)
        seen=[]
        @rt.OBS
        def rejected(i,data,count,ctx):seen.append(i)
        rt.require(lib.ta_run(C.byref(g),C.byref(plans[0]),limited.address,budget,inp,counts,1,rejected,None)==4
            and not seen and C.string_at(limited.address,budget)==before,'baseline budget rejection wrote/executed')
        limited.guards()
        graph_snapshot=None;all_intermediates=[[],[]];all_offsets=[[],[]]
        for start in range(0,450,batch):
            rows=min(batch,450-start);flat=sum(f['pixels'][start:start+rows],[])
            values=[v/16 for v in flat]+[0.]*(batch*64-len(flat))
            data=(rt.F*(batch*64))(*values);inputs=(rt.P*1)(data)
            snap=rt.snapshot(g,inputs);gold=rt.oracle(snap)
            rt.require(snap['nodes'][1]['weights']==sum(model['weights'][0],[]) and
                snap['nodes'][1]['bias']==model['biases'][0] and
                snap['nodes'][3]['weights']==sum(model['weights'][1],[]) and
                snap['nodes'][3]['bias']==model['biases'][1],'actual C fixture parameters differ')
            if graph_snapshot is None:graph_snapshot=snap
            traces=[]
            for mode,(p,arena) in enumerate(zip(plans,arenas)):
                info=rt.layout(p);rt.validate_layout(snap,info,('separate','reuse')[mode])
                observed=[];offsets=[]
                @rt.OBS
                def observer(i,data,count,ctx):
                    observed.append((int(i),C.string_at(data,count*4)))
                    offsets.append(C.cast(data,C.c_void_p).value-arena.address)
                rt.require(lib.ta_run(C.byref(g),C.byref(p),arena.address,arena.size,inputs,counts,1,observer,None)==0,
                    'real production model execution failed')
                rt.require([i for i,_ in observed]==list(range(4)) and offsets==info['offset'],'trace/address mismatch')
                all_intermediates[mode].append([list((rt.F*(len(raw)//4)).from_buffer_copy(raw)) for _,raw in observed])
                all_offsets[mode].append(offsets)
                for (id,raw),reference in zip(observed,gold):
                    actual=list((rt.F*len(reference)).from_buffer_copy(raw));rt.close(actual,reference)
                    worst[mode]=max(worst[mode],max(abs(a-b) for a,b in zip(actual,reference)))
                result=list((rt.F*(batch*10)).from_address(arena.address+p.offset[3]))[:rows*10]
                for i in range(rows):
                    v=result[i*10:(i+1)*10];rt.close(v,f['reference_logits'][start+i])
                    logits[mode].append(v);predictions[mode].append(max(range(10),key=lambda j:v[j]))
                arena.guards();traces.append(observed)
            rt.require(traces[0]==traces[1],'layouts differ bitwise at intermediate tensor')
        rt.require(predictions[0]==predictions[1]==f['reference_predictions'],'reference class mismatch')
        correct=sum(a==b for a,b in zip(predictions[1],f['labels']))
        rt.require(correct/450>=.94,'fixed accuracy floor')
        records[str(batch)]=dict(budget_bytes=budget,plans=[rt.layout(p) for p in plans],
            graph=graph_snapshot,predictions=predictions,logits=logits,correct=correct,
            intermediates=all_intermediates,observed_offsets=all_offsets,
            processed_rows=math.ceil(450/batch)*batch,coverage=f['test_ids'],
            max_intermediate_abs_error=worst,all_intermediates_bitwise_equal=True,
            baseline_budget_status=4,reuse_budget_status=0,guards_ok=True)
    return records


def summaries(records):
    out={}
    for batch in cm.BATCHES:
        for mode in range(2):
            for kind in range(2):
                values=[statistics.median(t['elapsed_ns']/(1024 if kind==0 else 4*450)
                    for t in r['raw'] if (t['batch'],t['mode'],t['kind'])==(batch,mode,kind)) for r in records]
                out['%d/%d/%d'%(batch,mode,kind)]=dict(median_ns=statistics.median(values),
                    process_medians_ns=values,minimum_ns=min(values),maximum_ns=max(values))
    return out


def measure(output,runs,samples):
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    sources={p:cm.digest(ROOT/p) for p in SOURCES}
    for name,sha in sources.items():
        rt.require(__import__('hashlib').sha256(subprocess.check_output(['git','show',revision+':'+name],cwd=ROOT)).hexdigest()==sha,
                   'commit measured sources first: '+name)
    build();checks=correctness();record=json.loads((BUILD/'build.json').read_text())
    output.mkdir(parents=True,exist_ok=False);cm.save(output/'correctness.json',checks);records=[]
    for run in range(runs):
        data=json.loads(subprocess.check_output([str(BUILD/'benchmark'),str(run),str(samples)],text=True))
        cm.save(output/('run-%02d.json'%run),data);records.append(data);print('Recorded real-model process',run+1,flush=True)
    rt.require(sources=={p:cm.digest(ROOT/p) for p in SOURCES} and all(cm.digest(ROOT/p)==h for p,h in
        {**record['objects'],**record['binaries']}.items()),'sources/build changed during run')
    cpu=None
    if sys.platform=='darwin':
        q=subprocess.run(['sysctl','-n','machdep.cpu.brand_string'],capture_output=True,text=True)
        cpu=q.stdout.strip() if q.returncode==0 else None
    cm.save(output/'manifest.json',dict(schema=1,protocol='digits-fixed-arena-budget-v2',source_revision=revision,
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),sources=sources,build=record,runs=runs,samples=samples,
        platform=dict(system=platform.system(),release=platform.release(),machine=platform.machine(),cpu=cpu),
        summaries=summaries(records),files={p.name:cm.digest(p) for p in output.glob('*.json')}))
    from verify import verify
    verify(output)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['build','test','sanitize','measure'])
    p.add_argument('--output',type=Path,default=ROOT/'results/local/digits');p.add_argument('--runs',type=int,default=6)
    p.add_argument('--samples',type=int,default=11);args=p.parse_args()
    if args.action=='build':build()
    elif args.action=='sanitize':build(True)
    elif args.action=='test':
        build();subprocess.run([sys.executable,str(HERE/'test_digits.py')],check=True)
        subprocess.run([sys.executable,'-O',str(HERE/'test_digits.py')],check=True)
    else:
        rt.require(1<=args.runs<=24 and 1<=args.samples<=20,'invalid counts');measure(args.output,args.runs,args.samples)
