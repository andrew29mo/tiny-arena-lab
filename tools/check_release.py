"""Check the complete public candidate, including superseded historical evidence."""
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile

ROOT=Path(__file__).resolve().parents[1]
REPORTS={'host-20261014','digits-budget-20261014','digits-budget-reviewed-20261014'}

def need(ok,message):
    if not ok:raise ValueError(message)

def command(arguments,**kwargs):
    subprocess.run(arguments,cwd=ROOT,check=True,**kwargs)

def inventory_and_notices():
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    for name in filter(None,tracked):
        path=ROOT/name
        if path.suffix=='.md':
            for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
                if '://' not in target and not target.startswith('#'):
                    need((path.parent/target.split('#')[0]).exists(),'broken documentation link: '+name)
    base=ROOT/'results/published'
    need({p.name for p in base.iterdir() if p.is_dir()}==REPORTS,'unreviewed/missing report directory')
    for name in REPORTS:need((base/name/'README.md').is_file(),'missing report status/attribution: '+name)
    need((ROOT/'LICENSE').read_text().startswith('MIT License\n\nCopyright (c) 2026 Andrew Yuan\n'),
         'unexpected project license/copyright')
    notices=(ROOT/'THIRD_PARTY_NOTICES.md').read_text()
    for term in ['E. Alpaydin','C. Kaynak','10.24432/C50P49','https://creativecommons.org/licenses/by/4.0/',
                 'examples/digits/model.json','examples/digits/fixture.json','results/published/digits-budget','MIT']:
        need(term in notices,'missing dataset attribution/license scope')
    modes=subprocess.check_output(['git','ls-files','-s'],cwd=ROOT,text=True).splitlines()
    need(all(row.startswith(('100644 ','100755 ')) for row in modes),'symlink/gitlink needs review')
    print('PASS: all Markdown local links, report status inventory, dataset/license notices and file modes',flush=True)

def historical_digits():
    directory=ROOT/'results/published/digits-budget-20261014'
    manifest=json.loads((directory/'manifest.json').read_text())
    revision=manifest['source_revision']
    need(re.fullmatch('[0-9a-f]{40}',revision) is not None and
         manifest['protocol']=='digits-fixed-arena-budget-v1','unexpected historical protocol/revision')
    # Run its own committed verifier against unchanged evidence. GIT_DIR only
    # supplies read-only object lookup; no checkout, branch or Git history edits.
    archive=subprocess.check_output(['git','archive',revision],cwd=ROOT)
    gitdir=subprocess.check_output(['git','rev-parse','--absolute-git-dir'],cwd=ROOT,text=True).strip()
    with tempfile.TemporaryDirectory(prefix='tiny-arena-history-') as tmp:
        path=Path(tmp)
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            for member in stream.getmembers():
                need((member.isfile() or member.isdir()) and not Path(member.name).is_absolute()
                     and '..' not in Path(member.name).parts,'unsafe historical archive member')
            stream.extractall(path)
        subprocess.run([sys.executable,str(path/'examples/digits/verify.py'),str(directory)],
            cwd=path,check=True,env={**os.environ,'GIT_DIR':gitdir,'GIT_WORK_TREE':str(path),
                                  'PYTHONDONTWRITEBYTECODE':'1'})
    print('PASS: superseded v1 evidence with its original verifier/source',flush=True)

def check():
    command([sys.executable,'tests/test_publication.py'])
    command([sys.executable,'tools/check_publication.py'])
    inventory_and_notices()
    command([sys.executable,'tools/check_docs.py'])
    command([sys.executable,'tools/verify_report.py','results/published/host-20261014'])
    command([sys.executable,'examples/digits/verify.py','results/published/digits-budget-reviewed-20261014','--docs'])
    historical_digits()
    print('PASS: complete release evidence checks; visibility is unchanged',flush=True)

if __name__=='__main__':check()
