"""Recompute liveness, references, measured output/address bounds and raw statistics."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

import run_benchmark as rb
import runtime as rt


def commands():
    flags=['cc','-Iinclude','-Isrc','-Itools','-std=c11','-O3','-Wall','-Wextra',
           '-Wpedantic','-Werror','-ffp-contract=off']
    out=[flags+['-fPIC','-c',src,'-o',obj] for src,obj in zip(
        ['src/graph.c','src/planner.c','src/execute.c','tools/workloads.c'],rb.OBJECTS)]
    out += [['cc','-shared']+rb.OBJECTS+['-lm','-o','build/libtiny_arena.so']]
    out += [flags+[src]+rb.OBJECTS+['-lm','-o',binary] for src,binary in
        [('tools/benchmark.c','build/benchmark'),('tests/test_native.c','build/test_native')]]
    return out


def verify(directory):
    directory=Path(directory);m=json.loads((directory/'manifest.json').read_text())
    rt.require(type(m['schema']) is int and m['schema']==1 and
        m['protocol']=='static-arena-inclusive-lifetimes-v1','report protocol mismatch')
    for field,lo,hi in [('runs',1,24),('samples',1,100),('repetitions',128,4096)]:
        rt.require(type(m[field]) is int and lo<=m[field]<=hi,'invalid count: '+field)
    expected={'correctness.json'}|{'run-%02d.json'%i for i in range(m['runs'])}
    rt.require(set(m['files'])==expected,'missing/extra report files')
    for name,sha in m['files'].items():rt.require(rb.digest(directory/name)==sha,'raw file hash mismatch')
    rt.require(set(m['sources'])==set(rb.SOURCES) and
        re.fullmatch('[0-9a-f]{40}',m['source_revision']),'source inventory/revision mismatch')
    for name,sha in m['sources'].items():
        blob=subprocess.check_output(['git','show',m['source_revision']+':'+name],cwd=rb.ROOT)
        rt.require(hashlib.sha256(blob).hexdigest()==sha,'historical source mismatch')
    build=m['build']
    rt.require(build['sources']==m['sources'] and build['commands']==commands() and
        set(build['objects'])==set(rb.OBJECTS) and set(build['binaries'])=={
        'build/libtiny_arena.so','build/benchmark','build/test_native'} and build['compiler'],
        'production build/source/flags mismatch')
    rt.require(all(re.fullmatch('[0-9a-f]{64}',x) for x in
        list(build['objects'].values())+list(build['binaries'].values())),'invalid production hashes')
    abi=m['abi']
    rt.require(abi==dict(size_t_bytes=rt.C.sizeof(rt.S),node_bytes=rt.C.sizeof(rt.Node),
        graph_bytes=rt.C.sizeof(rt.Graph),plan_bytes=rt.C.sizeof(rt.Plan),scratch_bytes=rt.C.sizeof(rt.Scratch),
        alignment=rt.ALIGN),'unsupported/mismatched recorded ABI')
    correctness=json.loads((directory/'correctness.json').read_text())
    rt.require(set(correctness)==set(rb.NAMES),'missing/extra graph correctness')
    for name,record in correctness.items():
        graph=record['graph'];gold=rt.oracle(graph)
        rt.require(gold==record['oracle_tensors'],'stored intermediate oracle mismatch')
        rt.require(set(record['plans'])==set(rb.MODES),'missing plan correctness')
        for mode,plan in record['plans'].items():
            rt.validate_layout(graph,plan,mode)
            rt.require(len(plan['intermediates'])==len(gold) and plan['observed_offsets']==plan['offset'],
                'missing actual intermediate/address evidence')
            for values,reference in zip(plan['intermediates'],gold):rt.close(values,reference)
            worst=max(abs(a-b) for values,reference in zip(plan['intermediates'],gold)
                      for a,b in zip(values,reference))
            rt.require(plan['guard_ok'] is True and plan['observed_tensors']==len(graph['nodes']) and
                plan['observed_highwater_bytes']==max(o+n['rows']*n['cols']*4 for o,n in
                    zip(plan['offset'],graph['nodes'])) and
                type(plan['max_intermediate_abs_error']) in (int,float) and
                math.isfinite(plan['max_intermediate_abs_error']) and plan['max_intermediate_abs_error']==worst,
                'actual intermediate trace/guard mismatch')
            rt.require(len(plan['outputs'])==len(graph['outputs']),'output count mismatch')
            for output,id in zip(plan['outputs'],graph['outputs']):rt.close(output,gold[id])
        rt.require(record['plans']['separate']['outputs']==record['plans']['reuse']['outputs'] and
            record['plans']['reuse']['arena_bytes']<=record['plans']['separate']['arena_bytes'],
            'same-graph execution mismatch')
    records=[]
    for run in range(m['runs']):
        r=json.loads((directory/('run-%02d.json'%run)).read_text())
        rt.require(r['schema']==1 and r['run']==run and r['samples']==m['samples'] and
            r['repetitions']==m['repetitions'] and r['plan_metadata_bytes']==abi['plan_bytes'] and
            r['planner_scratch_bytes']==abi['scratch_bytes'] and type(r['clock_resolution_ns']) is int and
            r['clock_resolution_ns']>0,'run/ABI/timer mismatch')
        rt.require([g['name'] for g in r['graphs']]==rb.NAMES,'graph order/inventory mismatch')
        for g in r['graphs']:
            saved=correctness[g['name']];graph=saved['graph']
            rt.require(g['node_count']==len(graph['nodes']) and g['graph_descriptor_bytes']==
                abi['graph_bytes']+len(graph['nodes'])*abi['node_bytes']+len(graph['outputs'])*abi['size_t_bytes'],
                'actual graph descriptor size mismatch')
            for mode in rb.MODES:
                plan=g[mode];rt.validate_layout(graph,plan,mode)
                reference=saved['plans'][mode]
                rt.require(all(plan[k]==reference[k] for k in
                    ['arena_bytes','live_lower_bound_bytes','offset','bytes','birth','death',
                     'observed_tensors','observed_highwater_bytes','guard_ok']),'C benchmark/test plan or trace mismatch')
                rt.require(plan['allocation_requested_bytes']==plan['arena_bytes']+2*rt.ALIGN and
                    plan['guard_bytes']==2*rt.ALIGN and type(plan['allocator_usable_bytes']) is int and
                    (plan['allocator_usable_bytes']==0 or plan['allocator_usable_bytes']>=plan['allocation_requested_bytes']),
                    'actual allocation size/guard mismatch')
                rt.require(len(plan['outputs'])==len(graph['outputs']),'measured output count mismatch')
                for output,id in zip(plan['outputs'],graph['outputs']):rt.close(output,saved['oracle_tensors'][id])
        rt.require([(r['sample'],r['graph'],r['mode']) for r in r['raw']]==rb.schedule(run,m['samples']),
                   'raw schedule mismatch')
        rt.require(all(type(row['elapsed_ns']) is int and row['elapsed_ns']>0 for row in r['raw']),
                   'invalid timing data')
        records.append(r)
    rt.require(m['summaries']==rb.summaries(records),'raw timing/summary mismatch')
    print('PASS: historical source/build, regenerated all-tensor oracle, inclusive lifetimes, actual allocations/outputs and timing statistics')
    return m,correctness,records


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    verify(p.parse_args().directory)
