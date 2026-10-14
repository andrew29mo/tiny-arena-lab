"""Stored reports must survive independent recomputation and reject rehashed lies."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import run_benchmark as rb
import runtime as rt
from verify_report import verify
from visualize import svg
from check_docs import check_measurement_sources

PUBLISHED=ROOT/'results/published/host-20261014'


class ReportingTests(unittest.TestCase):
    def test_current_measurement_implementation_must_match(self):
        manifest=json.loads((PUBLISHED/'manifest.json').read_text())
        check_measurement_sources(ROOT,manifest)
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            for name in manifest['sources']:
                target=directory/name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ROOT/name,target)
            for name in ('src/graph.c','include/tiny_arena.h','tools/benchmark.c',
                         'tools/workloads.c','tools/run_benchmark.py','Makefile'):
                target=directory/name;original=target.read_bytes()
                with self.subTest(source=name):
                    target.write_bytes(original+b'\n')
                    with self.assertRaisesRegex(ValueError,'current measurement implementation differs'):
                        check_measurement_sources(directory,manifest)
                    target.write_bytes(original)
            # Additional coverage does not invalidate unchanged production timings.
            (directory/'tests/test_planner.py').write_text('# changed test coverage\n')
            check_measurement_sources(directory,manifest)

    def test_known_schedule(self):
        for run in range(6):
            order=rb.schedule(run,20)
            self.assertEqual(len(order),160)
            for graph in rb.NAMES:
                self.assertEqual(sum(n==graph and mode=='separate' for _,n,mode in order),20)

    def test_actual_persisted_report(self):
        if not PUBLISHED.exists():self.skipTest('published measurements not created yet')
        with contextlib.redirect_stdout(io.StringIO()):verify(PUBLISHED)

    def test_rehashed_corruptions_rejected(self):
        if not PUBLISHED.exists():self.skipTest('published measurements not created yet')
        def overlap(r):
            p=r['chain']['plans']['reuse'];p['offset'][2]=p['offset'][1]
        cases=[('run-00.json',lambda r:r['raw'].pop()),
               ('run-00.json',lambda r:r['graphs'][0]['reuse'].update(allocation_requested_bytes=1)),
               ('run-00.json',lambda r:r['graphs'][0]['reuse']['outputs'][0].__setitem__(0,999)),
               ('correctness.json',overlap),
               ('correctness.json',lambda r:r['chain']['plans']['reuse'].update(max_intermediate_abs_error=0)),
               ('correctness.json',lambda r:r['fork_outputs']['plans']['reuse']['death'].__setitem__(3,3))]
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/'report';shutil.copytree(PUBLISHED,directory)
            with contextlib.redirect_stdout(io.StringIO()):verify(directory)
            manifest=json.loads((directory/'manifest.json').read_text())
            for filename,change in cases:
                with self.subTest(file=filename):
                    value=json.loads((PUBLISHED/filename).read_text());change(value);rb.save(directory/filename,value)
                    m=json.loads(json.dumps(manifest));m['files'][filename]=rb.digest(directory/filename)
                    rb.save(directory/'manifest.json',m)
                    try:
                        with self.assertRaises(ValueError),contextlib.redirect_stdout(io.StringIO()):verify(directory)
                    finally:
                        shutil.copyfile(PUBLISHED/filename,directory/filename);rb.save(directory/'manifest.json',manifest)

    def test_svg_coordinates_come_from_plan(self):
        import xml.etree.ElementTree as ET
        graph=dict(nodes=[dict(op=0,a=None,b=None,rows=1,cols=1,input=[1])],outputs=[0])
        layout=dict(arena_bytes=64,live_lower_bound_bytes=64,birth=[0],death=[1],bytes=[64],offset=[0])
        xml=svg('fixture',dict(graph=graph,plans=dict(separate=layout,reuse=layout)))
        root=ET.fromstring(xml);rectangles=[e for e in root.iter() if e.attrib.get('class')=='bar']
        self.assertEqual(len(rectangles),3)
        for node in rectangles:
            self.assertEqual(node.attrib['data-tensor'],'0')
            self.assertEqual(float(node.attrib['x']),115)
            self.assertEqual(float(node.attrib['width']),1000)
            self.assertEqual(float(node.attrib['height']),205)

    def test_svg_same_scale_comparison_and_explicit_zoom(self):
        import xml.etree.ElementTree as ET
        graph=dict(nodes=[dict(op=0,a=None,b=None,rows=1,cols=1,input=[1]),
            dict(op=2,a=0,b=None,rows=1,cols=1),dict(op=2,a=1,b=None,rows=1,cols=1)],outputs=[2])
        common=dict(live_lower_bound_bytes=128,birth=[0,1,2],death=[1,2,3],bytes=[64,64,64])
        separate=dict(common,arena_bytes=192,offset=[0,64,128])
        reuse=dict(common,arena_bytes=128,offset=[0,64,0])
        xml=ET.fromstring(svg('scale',dict(graph=graph,plans=dict(separate=separate,reuse=reuse))))
        first=[e for e in xml.iter() if e.attrib.get('data-tensor')=='0']
        self.assertEqual(len(first),3)
        self.assertEqual(first[0].attrib['height'],first[1].attrib['height'])
        self.assertGreater(float(first[2].attrib['height']),float(first[1].attrib['height']))


if __name__=='__main__':unittest.main()
