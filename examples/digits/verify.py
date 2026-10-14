"""Verify real-model raw memory/capacity/numerical evidence and timing summaries."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import common as cm
from common import rt,ROOT,HERE
import run


def verify_checks(checks):
    model,f=cm.load_artifacts();rt.require(set(checks)=={'1','16'},'correctness batch inventory')
    for batch,budget in cm.BATCHES.items():
        record=checks[str(batch)];graph=record['graph']
        nodes=[dict(op=0,a=None,b=None,rows=batch,cols=64,input=[]),
               dict(op=1,a=0,b=None,rows=batch,cols=32,weights=sum(model['weights'][0],[]),bias=model['biases'][0]),
               dict(op=2,a=1,b=None,rows=batch,cols=32),
               dict(op=1,a=2,b=None,rows=batch,cols=10,weights=sum(model['weights'][1],[]),bias=model['biases'][1])]
        nodes[0]['input']=[v/16 for row in f['pixels'][:batch] for v in row]
        rt.require(graph==dict(nodes=nodes,outputs=[3]),'exported graph/weights/input mismatch')
        rt.require(record['budget_bytes']==budget and record['coverage']==f['test_ids'] and
            record['processed_rows']==math.ceil(450/batch)*batch and record['baseline_budget_status']==4 and
            record['reuse_budget_status']==0 and record['guards_ok'] is True and
            record['all_intermediates_bitwise_equal'] is True,'coverage/budget metadata mismatch')
        rt.require(len(record['plans'])==len(record['logits'])==len(record['predictions'])==
            len(record['intermediates'])==len(record['observed_offsets'])==2,'mode inventory')
        for mode in range(2):
            plan=record['plans'][mode];rt.validate_layout(graph,plan,('separate','reuse')[mode])
            rt.require((plan['arena_bytes']>budget if mode==0 else plan['arena_bytes']<=budget),'planned budget mismatch')
            rt.require(len(record['intermediates'][mode])==len(record['observed_offsets'][mode])==math.ceil(450/batch),
                       'intermediate coverage mismatch')
            flattened=[];worst=0.
            for call,start in enumerate(range(0,450,batch)):
                rows=min(batch,450-start);pixels=[v/16 for row in f['pixels'][start:start+rows] for v in row]
                graph['nodes'][0]['input']=pixels+[0.]*(batch*64-len(pixels));gold=rt.oracle(graph)
                actual=record['intermediates'][mode][call]
                rt.require(len(actual)==4 and record['observed_offsets'][mode][call]==plan['offset'],'tensor/address inventory')
                for values,ref in zip(actual,gold):
                    rt.close(values,ref);worst=max(worst,max(abs(a-b) for a,b in zip(values,ref)))
                flattened.extend([actual[3][i*10:(i+1)*10] for i in range(rows)])
            rt.require(record['logits'][mode]==flattened and record['max_intermediate_abs_error'][mode]==worst,
                       'final/intermediate error mismatch')
            rt.require(len(flattened)==450,'missing final tail')
            for actual,reference in zip(flattened,f['reference_logits']):rt.close(actual,reference)
            predictions=[max(range(10),key=lambda j:v[j]) for v in flattened]
            rt.require(predictions==record['predictions'][mode]==f['reference_predictions'],'class/reference mismatch')
            correct=sum(a==b for a,b in zip(predictions,f['labels']))
            rt.require(record['correct']==correct and correct/450>=.94,'accuracy mismatch')
        a,b=record['intermediates']
        def packed(v):return b''.join(struct.pack('<%df'%len(t),*t) for call in v for t in call)
        rt.require(packed(a)==packed(b),'layouts do not agree bitwise')
        graph['nodes'][0]['input']=nodes[0]['input']=[v/16 for row in f['pixels'][:batch] for v in row]


def schedule(run_id,samples):
    return [(sample,batch,(pos+sample+run_id)%2,kind)
        for order in range(2) for batch in [16 if (order+run_id)%2 else 1]
        for sample in range(samples) for pos in range(2) for kind in range(2)]


def verify_run(r,checks,run_id,samples):
    _,f=cm.load_artifacts()
    rt.require(r['run']==run_id and r['samples']==samples and r['planning_repetitions']==1024 and
        r['inference_repetitions']==4 and type(r['clock_resolution_ns']) is int and r['clock_resolution_ns']>0,'run protocol')
    rt.require([c['batch'] for c in r['cases']]==([16,1] if run_id%2 else [1,16]),'batch ordering')
    for case in r['cases']:
        batch=case['batch'];budget=cm.BATCHES[batch];ref=checks[str(batch)]
        fixed=rt.C.sizeof(rt.Plan)+rt.C.sizeof(rt.Scratch)+9640+rt.C.sizeof(rt.Graph)+4*rt.C.sizeof(rt.Node)+rt.C.sizeof(rt.S)+batch*64*4
        rt.require(case['budget_bytes']==budget and case['budget_request_bytes']==budget+128 and
            case['plan_bytes']==rt.C.sizeof(rt.Plan) and case['scratch_bytes']==rt.C.sizeof(rt.Scratch) and
            case['weight_bytes']==9640 and case['graph_bytes']==rt.C.sizeof(rt.Graph)+4*rt.C.sizeof(rt.Node)+rt.C.sizeof(rt.S) and
            case['input_buffer_bytes']==batch*64*4,'memory inventory')
        rt.require(type(case['budget_usable_bytes']) is int and
            (case['budget_usable_bytes']==0 or case['budget_usable_bytes']>=budget+128),'budget allocator size')
        rt.require(len(case['modes'])==2,'mode inventory')
        for mode,p in enumerate(case['modes']):
            rt.require(p['mode']==mode and p['arena_bytes']==ref['plans'][mode]['arena_bytes'] and
                p['allocation_request_bytes']==p['arena_bytes']+128 and p['guard_ok'] is True and
                p['accounted_bytes']==fixed+p['arena_bytes'] and p['accounted_budget_bytes']==fixed+budget and
                p['budget_status']==(4 if mode==0 else 0) and p['budget_observed_tensors']==(0 if mode==0 else 4) and
                p['rejection_untouched'] is (mode==0),'actual arena/accounting/budget evidence mismatch')
            rt.require(type(p['allocator_usable_bytes']) is int and (p['allocator_usable_bytes']==0 or
                p['allocator_usable_bytes']>=p['allocation_request_bytes']),'allocator size mismatch')
            rt.require(len(p['logits'])==4500 and len(p['predictions'])==450,'missing held-out outputs/tail')
            rt.close(p['logits'],sum(ref['logits'][mode],[]))
            pred=[max(range(10),key=lambda j:p['logits'][i*10+j]) for i in range(450)]
            rt.require(pred==p['predictions']==f['reference_predictions'] and
                p['correct']==ref['correct'] and p['budget_correct']==(0 if mode==0 else ref['correct']),'actual accuracy mismatch')
        rt.require(case['modes'][0]['logits']==case['modes'][1]['logits'],'C layouts final output mismatch')
    rt.require([(t['sample'],t['batch'],t['mode'],t['kind']) for t in r['raw']]==schedule(run_id,samples),'timing schedule mismatch')
    rt.require(all(type(t['elapsed_ns']) is int and t['elapsed_ns']>0 for t in r['raw']),'timing values')


def verify(directory,docs=False):
    directory=Path(directory);m=json.loads((directory/'manifest.json').read_text())
    rt.require(m['schema']==1 and m['protocol']=='digits-fixed-arena-budget-v1' and
        type(m['runs']) is int and 1<=m['runs']<=24 and type(m['samples']) is int and 1<=m['samples']<=20,'manifest protocol')
    rt.require(re.fullmatch('[0-9a-f]{40}',m['source_revision']) and set(m['sources'])==set(run.SOURCES),'source inventory')
    for name,sha in m['sources'].items():
        rt.require(hashlib.sha256(subprocess.check_output(['git','show',m['source_revision']+':'+name],cwd=ROOT)).hexdigest()==sha,
            'historical source mismatch')
        rt.require(cm.digest(ROOT/name)==sha,'current measured source differs: '+name)
    expected={'correctness.json'}|{'run-%02d.json'%i for i in range(m['runs'])}
    rt.require(set(m['files'])==expected,'report file inventory')
    for name,sha in m['files'].items():rt.require(cm.digest(directory/name)==sha,'raw hash mismatch')
    build=m['build'];rt.require(build['commands']==run.commands() and build['compiler'] and
        build['generated_header_sha256']==hashlib.sha256(cm.header().encode()).hexdigest() and
        set(build['objects'])=={'build/digits/'+n+'.o' for n in ('graph','planner','execute','fixture')} and
        set(build['binaries'])=={'build/digits/engine.so','build/digits/benchmark'},'build provenance')
    rt.require(all(re.fullmatch('[0-9a-f]{64}',v) for v in list(build['objects'].values())+list(build['binaries'].values())),
        'binary hash inventory')
    checks=json.loads((directory/'correctness.json').read_text());verify_checks(checks);records=[]
    for i in range(m['runs']):
        r=json.loads((directory/('run-%02d.json'%i)).read_text());verify_run(r,checks,i,m['samples']);records.append(r)
    rt.require(m['summaries']==run.summaries(records),'raw timing summary mismatch')
    if docs:
        sys_path=__import__('sys').path;sys_path.insert(0,str(ROOT/'tools'))
        from check_docs import table
        rows=[];cost=[];allocations=[]
        for case in sorted(records[0]['cases'],key=lambda c:c['batch']):
            batch=case['batch'];a,b=case['modes'];correct=checks[str(batch)]['correct']
            rows.append([str(batch),str(case['budget_bytes']),str(a['arena_bytes']),str(b['arena_bytes']),
                str(a['accounted_bytes']),str(b['accounted_bytes']),str(correct)+'/450'])
            allocations.append([str(batch)]+[str(p[k]) for p in (a,b) for k in
                ('allocation_request_bytes','allocator_usable_bytes')]+[
                str(case['budget_request_bytes']),str(case['budget_usable_bytes'])])
            cost.append([str(batch)]+['%.3f'%(m['summaries']['%d/%d/%d'%(batch,mode,kind)]['median_ns']/1000)
                for kind in (0,1) for mode in (0,1)])
        from visualize import svg
        for batch in cm.BATCHES:
            record=checks[str(batch)]
            figure=svg('digits_batch_'+str(batch),dict(graph=record['graph'],plans=dict(zip(
                ('separate','reuse'),record['plans']))))
            rt.require((HERE/('batch-%d.svg'%batch)).read_text()==figure,'real-model figure differs from actual plan')
        text=(HERE/'README.md').read_text()
        rt.require(table(text,'Batch')==rows and table(text,'Timing batch')==cost and
            table(text,'Allocation batch')==allocations and m['source_revision'][:7] in text,
                   'real-model README differs from measured evidence')
    print('PASS: real model, all intermediate tensors/tail, exact budgets, actual allocations, held-out accuracy and C timings')
    return m

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--docs',action='store_true');a=p.parse_args()
    verify(a.directory,a.docs)
