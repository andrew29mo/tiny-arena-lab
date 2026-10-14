import ctypes as C
import json
from pathlib import Path
import tempfile
import unittest
import common as cm
from common import rt,ROOT,HERE
import run

class RealModelTests(unittest.TestCase):
    def test_production_weights_every_tensor_every_example_budgets_and_tail(self):
        checks=run.correctness()
        self.assertEqual(checks['1']['correct'],checks['16']['correct'])
        self.assertEqual(checks['16']['processed_rows'],464)
        self.assertEqual(len(checks['16']['coverage']),450)
        self.assertEqual(checks['1']['plans'][0]['arena_bytes'],576)
        self.assertEqual(checks['1']['plans'][1]['arena_bytes'],384)
        self.assertEqual(checks['16']['plans'][0]['arena_bytes'],8832)
        self.assertEqual(checks['16']['plans'][1]['arena_bytes'],6144)

    def test_split_duplicate_data_and_model_corruptions(self):
        import copy
        import hashlib
        model,fixture=cm.load_artifacts()
        def overlap(m,f):m['test_ids'][0]=m['train_ids'][0]
        def duplicate(m,f):f['train_image_sha256'][0]=hashlib.sha256(bytes(f['pixels'][0])).hexdigest()
        def nonfinite(m,f):m['weights'][0][0][0]=float('inf')
        for change in [overlap,duplicate,nonfinite,lambda m,f:f['pixels'][0].__setitem__(0,17),
                       lambda m,f:m['training_parameters'].update(activation='identity'),
                       lambda m,f:m['training_parameters'].update(random_state=1)]:
            with tempfile.TemporaryDirectory() as tmp:
                m,f=copy.deepcopy(model),copy.deepcopy(fixture);change(m,f)
                mp,fp=Path(tmp)/'model.json',Path(tmp)/'fixture.json'
                mp.write_text(json.dumps(m));f['model_sha256']=cm.digest(mp);fp.write_text(json.dumps(f))
                with self.assertRaises(ValueError):cm.load_artifacts(mp,fp)

    def test_compiled_missing_relu_rejected_on_real_model(self):
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);source=(ROOT/'src/execute.c').read_text()
            anchor='y[j]=fmaxf(0.0f,a[j]);'
            self.assertEqual(source.count(anchor),1)
            (path/'execute.c').write_text(source.replace(anchor,'y[j]=a[j];'))
            subprocess.run(['cc','-I'+str(ROOT/'include')]+cm.FLAGS+['-fPIC','-c',str(path/'execute.c'),
                '-o',str(path/'execute.o')],check=True)
            library=path/'mutant.so'
            subprocess.run(['cc','-shared',str(path/'execute.o')]+[str(run.BUILD/(n+'.o')) for n in
                ('graph','planner','fixture')]+['-lm','-o',str(library)],check=True)
            with self.assertRaisesRegex(ValueError,'numerical mismatch'):run.correctness(cm.library(library))

    def test_compiled_uncaptured_tail_output_and_planning_faults_rejected(self):
        import subprocess
        original=(HERE/'benchmark.c').read_text()
        mutations=[
            ('uncaptured_tail','for(size_t start=0;start<DIGITS_COUNT;start+=batch) {',
             'for(size_t start=0;start<DIGITS_COUNT;start+=batch) {\n'
             'if(!saved && batch==16 && start+batch>DIGITS_COUNT) break;',
             'pass coverage/output fingerprint mismatch'),
            ('uncaptured_row','for(size_t i=0;i<rows;++i) {',
             'for(size_t i=0;i<rows;++i) {\nif(!saved && start+i==DIGITS_COUNT-1) continue;',
             'pass coverage/output fingerprint mismatch'),
            ('uncaptured_logit','const float *y=(const float *)(arena+p->offset[3]);',
             'const float *y=(const float *)(arena+p->offset[3]);\nif(!saved) ((float *)y)[0]+=0.0001f;',
             'pass coverage/output fingerprint mismatch'),
            ('half_planning','for(size_t rep=0;rep<1024;++rep)',
             'for(size_t rep=0;rep<512;++rep)','planning call count mismatch')]
        with tempfile.TemporaryDirectory() as tmp:
            for name,before,after,message in mutations:
                with self.subTest(fault=name):
                    self.assertEqual(original.count(before),1)
                    source=Path(tmp)/(name+'.c');binary=Path(tmp)/name
                    mutant=original.replace(before,after)
                    if name.startswith('uncaptured_'):
                        # Warmup remains correct: challenge the genuinely timed
                        # NULL-output branch, not just pre-measurement checks.
                        mutant=mutant.replace('typedef struct { size_t correct,rows,calls;',
                            'static int measurement_active=0;\ntypedef struct { size_t correct,rows,calls;')
                        mutant=mutant.replace('if(!saved', 'if(measurement_active && !saved')
                        mutant=mutant.replace('uint64_t begin=now();',
                            'measurement_active=(kind==1);\nuint64_t begin=now();')
                        mutant=mutant.replace('uint64_t elapsed=now()-begin;',
                            'uint64_t elapsed=now()-begin;measurement_active=0;')
                    source.write_text(mutant)
                    subprocess.run(['cc','-I'+str(ROOT/'include'),'-I'+str(HERE),'-I'+str(run.BUILD)]+
                        cm.FLAGS+[str(source)]+[str(run.BUILD/(n+'.o')) for n in
                        ('graph','planner','execute','fixture')]+['-lm','-o',str(binary)],check=True)
                    result=subprocess.run([str(binary),'0','1'],capture_output=True,text=True)
                    self.assertNotEqual(result.returncode,0)
                    self.assertIn(message,result.stderr)

    def test_bad_capacity_before_writes_and_exact_minimum(self):
        lib=cm.library()
        for batch in cm.BATCHES:
            g,nodes,out=cm.graph(lib,batch);p,s=rt.Plan(),rt.Scratch()
            self.assertEqual(lib.ta_plan_build(C.byref(g),1,C.byref(p),C.byref(s)),0)
            data=(rt.F*(batch*64))();inputs=(rt.P*1)(data);counts=(rt.S*1)(batch*64)
            for capacity,code in [(p.arena_bytes-1,4),(p.arena_bytes,0)]:
                arena=rt.Arena(capacity);before=C.string_at(arena.address,capacity)
                self.assertEqual(lib.ta_run(C.byref(g),C.byref(p),arena.address,capacity,inputs,counts,1,rt.OBS(),None),code)
                if code:self.assertEqual(C.string_at(arena.address,capacity),before)
                arena.guards()

    def test_c_harness_outputs_and_budgets(self):
        import subprocess
        from verify import verify_run
        report=json.loads(subprocess.check_output([str(run.BUILD/'benchmark'),'0','1'],text=True))
        verify_run(report,run.correctness(),0,1)

    def test_persisted_intermediate_and_coverage_corruptions(self):
        import copy
        from verify import verify_checks
        checks=run.correctness();verify_checks(copy.deepcopy(checks))
        changes=[lambda c:c['16']['coverage'].pop(),
                 lambda c:c['16']['intermediates'][1][-1][2].__setitem__(0,9999),
                 lambda c:c['1']['max_intermediate_abs_error'].__setitem__(1,0.),
                 lambda c:c['16']['observed_offsets'][1][-1].__setitem__(0,64)]
        for change in changes:
            bad=copy.deepcopy(checks);change(bad)
            with self.assertRaises(ValueError):verify_checks(bad)

    def test_rehashed_numeric_budget_and_tail_corruptions(self):
        import copy
        import subprocess
        from verify import verify_run
        checks=run.correctness();report=json.loads(subprocess.check_output([str(run.BUILD/'benchmark'),'0','1'],text=True))
        changes=[lambda r:r['cases'][0]['modes'][0].update(budget_status=0),
                 lambda r:r['cases'][1]['modes'][1]['logits'].__setitem__(-1,9999),
                 lambda r:r['cases'][1]['modes'][1]['predictions'].pop(),
                 lambda r:r['raw'].pop(),lambda r:r['cases'][0]['modes'][1].update(accounted_bytes=1),
                 lambda r:r['raw'][1].update(valid_rows=1792),
                 lambda r:r['raw'][0].update(plans_built=512),
                 lambda r:r['cases'][1]['modes'][1].update(fingerprint='0'*16)]
        for change in changes:
            bad=copy.deepcopy(report);change(bad)
            with self.assertRaises((ValueError,KeyError)):verify_run(bad,checks,0,1)

if __name__=='__main__':unittest.main()
