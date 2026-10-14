"""Exercise publication boundaries without storing literal credential fixtures."""
from pathlib import Path
import subprocess
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from check_publication import findings

class PublicationTests(unittest.TestCase):
    def test_private_paths_credentials_and_binary_disguises_are_rejected(self):
        cases=[
            ('settings.txt',b'/' + b'Users' + b'/private-person/config'),
            ('settings.txt',b'/' + b'home' + b'/private-person/config'),
            ('settings.txt',b'C:'+bytes([92])+b'Users'+bytes([92])+b'private-person'+bytes([92])+b'config'),
            ('settings.txt',b'ghp_' + b'a'*35),
            ('settings.txt',b'github_pat_' + b'a'*40),
            ('settings.txt',b'-----BEGIN '+b'PRIVATE KEY-----'),
            ('settings.txt',b'https'+b'://name:password@example.invalid'),
            ('notes.txt',bytes([0x7f])+b'ELF'+bytes(12)),
            ('renamed.json',b'abc'+bytes([0])+b'content'),
            ('renamed.json',bytes([0xff,0xfe,0x81])),
            ('.venv/pyvenv.cfg',b'version = 3'),
            ('nested/.git/config',b'[core]'),
            ('AGENTS.md',b'personal rules'),
            ('test.a',b'archive'),
            ('build/generated.h',b'generated'),
            ('results/local/timing.json',b'{}')]
        for name,payload in cases:
            with self.subTest(file=name):self.assertTrue(findings(name,payload))

    def test_attributed_text_and_public_evidence_are_allowed(self):
        for name,payload in [
            ('README.md','Tensor arena → smaller storage'.encode()),
            ('results/published/example/manifest.json',b'{"sha256":"'+b'a'*64+b'"}'),
            ('examples/digits/fixture.json',b'{"pixels":[0,1,16],"labels":[3]}'),
            ('src/graph.c',b'#include "tiny_arena.h"\n')]:
            with self.subTest(file=name):self.assertEqual(findings(name,payload),[])

    def test_local_outputs_and_credentials_are_ignored(self):
        paths=['.env','.venv/pyvenv.cfg','venv/bin/python','build/engine.so',
               'libexample.a','example.obj','example.dSYM/Contents/Info.plist',
               'results/local/trial.json','AGENTS.md','token.json']
        result=subprocess.run(['git','check-ignore','--no-index','--stdin'],cwd=ROOT,
            input='\n'.join(paths)+'\n',text=True,capture_output=True,check=True)
        self.assertEqual(set(result.stdout.splitlines()),set(paths))

if __name__=='__main__':unittest.main()
