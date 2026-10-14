"""Compile genuinely wrong C implementations; real fixture checks must reject them."""
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
mutants=[('inclusive_lifetime','internal.h',
    's->birth[a] <= s->death[b] && s->birth[b] <= s->death[a]',
    's->birth[a] < s->death[b] && s->birth[b] < s->death[a]'),
    ('output_pin','graph.c','s->death[g->outputs[i]]=g->count;', '(void)g->outputs[i];'),
    ('second_operand','graph.c',
     'if (n->op==TA_ADD && s->death[n->b]<i) s->death[n->b]=i;', '(void)n->b;'),
    ('first_consumer_only','graph.c','if (s->death[n->a]<i) s->death[n->a]=i;',
     'if (s->death[n->a]==s->birth[n->a]) s->death[n->a]=i;'),
    ('missing_relu','execute.c','y[j]=fmaxf(0.0f,a[j]);','y[j]=a[j];')]
for optimization in ([],['-O']):
    subprocess.run([sys.executable]+optimization+['tests/test_planner.py','build/libtiny_arena.so',
                    '--fixtures-only'],cwd=ROOT,check=True)
with tempfile.TemporaryDirectory() as tmp:
    for name,file,before,after in mutants:
        directory=Path(tmp)/name;directory.mkdir()
        for source in ['graph.c','planner.c','execute.c','internal.h']:
            text=(ROOT/'src'/source).read_text()
            if source==file:
                if text.count(before)!=1:raise ValueError('mutation anchor changed: '+name)
                text=text.replace(before,after)
            (directory/source).write_text(text)
        library=directory/'mutant.so'
        command=['cc','-std=c11','-O3','-ffp-contract=off','-fPIC','-shared','-Iinclude','-Itools',
            '-I'+str(directory)]+[str(directory/f) for f in ['graph.c','planner.c','execute.c']]+[
            'tools/workloads.c','-lm','-o',str(library)]
        subprocess.run(command,cwd=ROOT,check=True)
        for optimization in ([],['-O']):
            result=subprocess.run([sys.executable]+optimization+['tests/test_planner.py',str(library),
                '--fixtures-only'],cwd=ROOT,capture_output=True,text=True)
            expected=['numerical mismatch'] if name=='missing_relu' else [
                'lifetime/size mismatch','simultaneously live tensors overlap','production planner failed']
            if result.returncode==0 or not any(message in result.stderr for message in expected):
                raise ValueError('mutant escaped numerical/liveness checks: '+name+' '+result.stderr)
            if name in ('second_operand','first_consumer_only'):
                directed=subprocess.run([sys.executable]+optimization+['tests/test_planner.py',str(library),
                    'PlannerTests.test_fanout_second_operand_dead_node_and_repeated_arena'],
                    cwd=ROOT,capture_output=True,text=True)
                if directed.returncode==0 or 'FAIL: test_fanout_second_operand' not in directed.stderr:
                    raise ValueError('directed consumer regression did not reject mutant: '+name+directed.stderr)
        print('PASS: compiled '+name+' rejected in normal and optimized Python')
