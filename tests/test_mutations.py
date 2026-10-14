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
        print('PASS: compiled '+name+' rejected in normal and optimized Python')
