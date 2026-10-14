"""Production C plans and every produced tensor vs independent Python oracles."""
import ctypes as C
from pathlib import Path
import random
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import runtime as rt

LIBRARY=Path(sys.argv.pop(1)) if len(sys.argv)>1 and not sys.argv[1].startswith('-') else ROOT/'build/libtiny_arena.so'


class OwnedGraph:
    def __init__(self,seed,count):
        rng=random.Random(seed);self.keep=[];nodes=[];inputs=[];counts=[]
        rows=rng.randint(1,3)
        for _ in range(rng.randint(1,min(3,count))):
            width=rng.randint(1,23)
            nodes.append(rt.Node(0,rt.NONE,rt.NONE,rows,width,None,None,0,0))
            data=(rt.F*(rows*width))(*(rng.uniform(-1,1) for _ in range(rows*width)))
            self.keep.append(data);inputs.append(C.cast(data,rt.P));counts.append(rows*width)
        while len(nodes)<count:
            a=rng.randrange(len(nodes));op=rng.randrange(1,4)
            width=nodes[a].cols;b=rt.NONE;wptr=bptr=None;nw=nb=0
            if op==1:
                width=rng.randint(1,31);nw=nodes[a].cols*width;nb=width
                w=(rt.F*nw)(*(rng.uniform(-.1,.1) for _ in range(nw)))
                bias=(rt.F*nb)(*(rng.uniform(-.1,.1) for _ in range(nb)))
                self.keep.extend([w,bias]);wptr=w;bptr=bias
            elif op==3:
                matches=[i for i,n in enumerate(nodes) if n.cols==width]
                b=rng.choice(matches)
            nodes.append(rt.Node(op,a,b,rows,width,wptr,bptr,nw,nb))
        outs=sorted(rng.sample(range(count),rng.randint(1,min(3,count))))
        # Always retain the last tensor; earlier outputs exercise pinning.
        outs=sorted(set(outs+[count-1]))
        self.nodes=(rt.Node*count)(*nodes);self.outputs=(rt.S*len(outs))(*outs)
        self.inputs=(rt.P*len(inputs))(*inputs);self.counts=(rt.S*len(counts))(*counts)
        self.graph=rt.Graph(count,len(outs),self.nodes,self.outputs)


class PlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.lib=rt.load(LIBRARY)

    def test_measured_fixtures_and_all_intermediates(self):
        records=rt.fixtures(self.lib)
        self.assertEqual(set(records),{'chain','residual','fork_outputs','residual_stack'})
        for record in records.values():
            a,b=record['plans']['separate'],record['plans']['reuse']
            self.assertEqual(a['outputs'],b['outputs'])
            self.assertEqual(a['observed_tensors'],len(record['graph']['nodes']))
        # The selected heuristic is demonstrably not always optimal.
        self.assertGreater(records['fork_outputs']['plans']['reuse']['arena_bytes'],
                           records['fork_outputs']['plans']['reuse']['live_lower_bound_bytes'])

    def test_100_varied_dags_with_multiple_inputs_and_pinned_outputs(self):
        for seed in range(100):
            owned=OwnedGraph(20261014+seed,1+seed%30)
            with self.subTest(seed=seed):
                rt.exercise(self.lib,owned.graph,owned.inputs,owned.counts,len(owned.inputs))

    def test_liveness_oracle_includes_last_read_and_pins(self):
        graph=dict(nodes=[dict(op=0,a=None,b=None,rows=1,cols=1,input=[-1]),
            dict(op=2,a=0,b=None,rows=1,cols=1),dict(op=2,a=1,b=None,rows=1,cols=1)],outputs=[0,2])
        birth,death,sizes,lower=rt.lifetimes(graph)
        self.assertEqual((birth,death,sizes,lower),([0,1,2],[3,2,3],[64,64,64],192))
        broken=dict(birth=birth,death=death,bytes=sizes,live_lower_bound_bytes=lower,
                    offset=[0,64,0],arena_bytes=128)
        with self.assertRaises(ValueError):rt.validate_layout(graph,broken,'reuse')

    def test_no_reuse_at_consumer_step(self):
        owner=OwnedGraph(7,3);g=owner.graph
        p,s=rt.Plan(),rt.Scratch()
        self.assertEqual(self.lib.ta_plan_build(C.byref(g),1,C.byref(p),C.byref(s)),0)
        for i,n in enumerate(owner.nodes):
            if n.op:
                self.assertFalse(p.offset[i]<p.offset[n.a]+p.bytes[n.a] and
                                 p.offset[n.a]<p.offset[i]+p.bytes[i])

    def test_maximum_graph_and_capacity_rejection(self):
        owner=OwnedGraph(909,128)
        rt.exercise(self.lib,owner.graph,owner.inputs,owner.counts,len(owner.inputs))
        owner.graph.count=129
        p,s=rt.Plan(),rt.Scratch()
        self.assertEqual(self.lib.ta_plan_build(C.byref(owner.graph),1,C.byref(p),C.byref(s)),1)


if __name__=='__main__':
    if '--fixtures-only' in sys.argv:
        rt.fixtures(rt.load(LIBRARY));print('PASS: real fixtures, independent lifetimes and every tensor')
    else:unittest.main()
