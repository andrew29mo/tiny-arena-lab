"""Keep README values, diagram coordinates and viewer tied to verified evidence."""
import json
import hashlib
from pathlib import Path
import re
import tempfile

from runtime import require
from visualize import render

ROOT=Path(__file__).resolve().parents[1]


def check_measurement_sources(root,manifest):
    # Historical verification alone intentionally accepts older implementations.
    # A current README result additionally needs the same measured C code,
    # fixtures, build recipe, input/oracle bridge and timing driver.
    names=[name for name in manifest['sources'] if name.startswith(('src/','include/')) or
           name in ('tools/workloads.c','tools/workloads.h','tools/benchmark.c',
                    'tools/runtime.py','tools/run_benchmark.py','Makefile')]
    for name in names:
        require(hashlib.sha256((root/name).read_bytes()).hexdigest()==manifest['sources'][name],
                'current measurement implementation differs; remeasure before publishing: '+name)


def table(text,header):
    lines=text.splitlines();start=next((i for i,line in enumerate(lines) if line.startswith('| '+header+' |')),None)
    require(start is not None,'missing README table')
    rows=[]
    for line in lines[start+2:]:
        if not line.startswith('|'):break
        rows.append([v.strip() for v in line.strip('|').split('|')])
    return rows


def check(root=ROOT):
    directory=root/'results/published/host-20261014'
    m=json.loads((directory/'manifest.json').read_text());r=json.loads((directory/'run-00.json').read_text())
    check_measurement_sources(root,m)
    readme=(root/'README.md').read_text()
    rows=[];alloc=[];accounted=[]
    for g in r['graphs']:
        name=g['name'];a,b=g['separate'],g['reuse'];stats=m['summaries'][name]
        rows.append([name,str(a['arena_bytes']),str(b['arena_bytes']),str(b['live_lower_bound_bytes'])]+[
            '%.3f'%(stats[mode]['median_ns']/1000) for mode in ('separate','reuse')])
        alloc.append([name,str(a['allocation_requested_bytes']),str(a['allocator_usable_bytes']),
                      str(b['allocation_requested_bytes']),str(b['allocator_usable_bytes'])])
        fixed=r['plan_metadata_bytes']+r['planner_scratch_bytes']
        accounted.append([name,str(a['arena_bytes']+fixed),str(b['arena_bytes']+fixed)])
    require(table(readme,'Graph')==rows,'README memory/planning table disagrees with evidence')
    require(table(readme,'Allocation graph')==alloc,'README allocator table disagrees with evidence')
    require(table(readme,'Accounting graph')==accounted,'README accounting subtotal disagrees with evidence')
    require(m['source_revision'][:7] in readme and str(r['plan_metadata_bytes']) in readme and
            str(r['planner_scratch_bytes']) in readme,'README source/metadata missing')
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp);render(directory,path)
        for name in ['index.html']+[g['name']+'.svg' for g in r['graphs']]:
            require((root/'docs/figures'/name).read_bytes()==(path/name).read_bytes(),
                    'visualization differs from measured plan: '+name)
    for path in [root/'README.md',root/'CONTRIBUTING.md']+list((root/'docs').glob('*.md')):
        for link in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if '://' not in link and not link.startswith('#'):
                require((path.parent/link.split('#')[0]).exists(),'broken documentation link')
    print('PASS: README numerical tables/source, four SVGs, standalone viewer and local links')


if __name__=='__main__':check()
