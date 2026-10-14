"""ctypes access to production objects; independent graph/liveness/numeric oracles."""
import ctypes as C
import math
from pathlib import Path

MAX=128
ALIGN=64
NONE=C.c_size_t(-1).value
F=C.c_float
P=C.POINTER(F)
S=C.c_size_t
SP=C.POINTER(S)


class Node(C.Structure):
    _fields_=[('op',C.c_int),('a',S),('b',S),('rows',S),('cols',S),('weights',P),('bias',P),
              ('weight_count',S),('bias_count',S)]


class Graph(C.Structure):
    _fields_=[('count',S),('output_count',S),('nodes',C.POINTER(Node)),('outputs',SP)]


class Plan(C.Structure):
    _fields_=[('mode',C.c_int),('count',S),('arena_bytes',S),('live_lower_bound_bytes',S),
              ('offset',S*MAX),('bytes',S*MAX),('birth',S*MAX),('death',S*MAX)]


class Scratch(C.Structure):
    _fields_=[('bytes',S*MAX),('birth',S*MAX),('death',S*MAX),('order',S*MAX)]


OBS=C.CFUNCTYPE(None,S,P,S,C.c_void_p)


def require(ok,message):
    if not ok:raise ValueError(message)


def load(path):
    lib=C.CDLL(str(Path(path).resolve()))
    lib.ta_plan_build.argtypes=[C.POINTER(Graph),C.c_int,C.POINTER(Plan),C.POINTER(Scratch)]
    lib.ta_plan_build.restype=C.c_int
    lib.ta_plan_validate.argtypes=[C.POINTER(Graph),C.POINTER(Plan),C.POINTER(Scratch)]
    lib.ta_plan_validate.restype=C.c_int
    lib.ta_run.argtypes=[C.POINTER(Graph),C.POINTER(Plan),C.c_void_p,S,C.POINTER(P),SP,S,OBS,C.c_void_p]
    lib.ta_run.restype=C.c_int
    lib.ta_fixture_create.argtypes=[S];lib.ta_fixture_create.restype=C.c_void_p
    lib.ta_fixture_free.argtypes=[C.c_void_p];lib.ta_fixture_free.restype=None
    lib.ta_fixture_graph.argtypes=[C.c_void_p];lib.ta_fixture_graph.restype=C.POINTER(Graph)
    lib.ta_fixture_inputs.argtypes=[C.c_void_p];lib.ta_fixture_inputs.restype=C.POINTER(P)
    lib.ta_fixture_input_counts.argtypes=[C.c_void_p];lib.ta_fixture_input_counts.restype=SP
    lib.ta_fixture_input_count.argtypes=[C.c_void_p];lib.ta_fixture_input_count.restype=S
    lib.ta_fixture_name.argtypes=[S];lib.ta_fixture_name.restype=C.c_char_p
    return lib


def snapshot(g,inputs):
    nodes=[];index=0
    for i in range(g.count):
        n=g.nodes[i]
        record=dict(op=int(n.op),a=None if n.a==NONE else int(n.a),b=None if n.b==NONE else int(n.b),
                    rows=int(n.rows),cols=int(n.cols))
        if n.op==0:
            record['input']=[float(inputs[index][j]) for j in range(n.rows*n.cols)];index+=1
        if n.op==1:
            record['weights']=[float(n.weights[j]) for j in range(n.weight_count)]
            record['bias']=[float(n.bias[j]) for j in range(n.bias_count)]
        nodes.append(record)
    return dict(nodes=nodes,outputs=[int(g.outputs[i]) for i in range(g.output_count)])


def lifetimes(graph):
    nodes=graph['nodes'];birth=[0 if n['op']==0 else i for i,n in enumerate(nodes)]
    death=birth.copy()
    for i,n in enumerate(nodes):
        for key in ('a','b'):
            if n[key] is not None:death[n[key]]=max(death[n[key]],i)
    for out in graph['outputs']:death[out]=len(nodes)
    sizes=[((n['rows']*n['cols']*4+ALIGN-1)//ALIGN)*ALIGN for n in nodes]
    lower=max(sum(size for size,a,b in zip(sizes,birth,death) if a<=step<=b)
              for step in range(len(nodes)+1))
    return birth,death,sizes,lower


def validate_layout(graph,layout,mode):
    birth,death,sizes,lower=lifetimes(graph)
    require(layout['birth']==birth and layout['death']==death and layout['bytes']==sizes and
            layout['live_lower_bound_bytes']==lower,'C/Python lifetime/size mismatch')
    offsets=layout['offset']
    require(len(offsets)==len(sizes) and all(type(x) is int and x>=0 and x%ALIGN==0 for x in offsets),
            'unaligned/invalid offsets')
    ends=[a+b for a,b in zip(offsets,sizes)]
    require(layout['arena_bytes']==max(ends) and layout['arena_bytes']>=lower,'arena size mismatch')
    if mode=='separate':
        total=0
        for offset,size in zip(offsets,sizes):
            require(offset==total,'separate layout unexpectedly reused storage');total+=size
    for i in range(len(graph['nodes'])):
        for j in range(i):
            if max(birth[i],birth[j])<=min(death[i],death[j]):
                require(max(offsets[i],offsets[j])>=min(ends[i],ends[j]),'simultaneously live tensors overlap')


def layout(p):
    return dict(arena_bytes=int(p.arena_bytes),live_lower_bound_bytes=int(p.live_lower_bound_bytes),
                **{name:list(getattr(p,name))[:p.count] for name in ['offset','bytes','birth','death']})


def oracle(graph):
    """No C planner/kernel calls: Python float64 fsum computes every tensor."""
    answer=[]
    for n in graph['nodes']:
        if n['op']==0:out=n['input'].copy()
        elif n['op']==1:
            a=answer[n['a']];columns=graph['nodes'][n['a']]['cols'];width=n['cols']
            out=[math.fsum([n['bias'][j]]+[a[row*columns+k]*n['weights'][k*width+j]
                 for k in range(columns)]) for row in range(n['rows']) for j in range(width)]
        elif n['op']==2:out=[max(0.0,v) for v in answer[n['a']]]
        else:out=[a+b for a,b in zip(answer[n['a']],answer[n['b']])]
        answer.append(out)
    return answer


def close(actual,expected):
    require(len(actual)==len(expected) and all(math.isfinite(a) and math.isfinite(b) and
        abs(a-b)<=1e-5+3e-5*abs(b) for a,b in zip(actual,expected)),'numerical mismatch')


class Arena:
    def __init__(self,size):
        self.storage=C.create_string_buffer(size+3*ALIGN)
        self.base=(C.addressof(self.storage)+ALIGN-1)//ALIGN*ALIGN
        self.address=self.base+ALIGN;self.size=size
        C.memset(self.base,0xa5,size+2*ALIGN)

    def guards(self):
        require(C.string_at(self.base,ALIGN)==bytes([0xa5])*ALIGN and
                C.string_at(self.address+self.size,ALIGN)==bytes([0xa5])*ALIGN,'arena guard overwritten')


def exercise(lib,g,inputs,counts,n_inputs):
    graph=snapshot(g,inputs);gold=oracle(graph);results={}
    for mode,name in enumerate(('separate','reuse')):
        p,s=Plan(),Scratch()
        require(lib.ta_plan_build(C.byref(g),mode,C.byref(p),C.byref(s))==0,'production planner failed')
        info=layout(p);validate_layout(graph,info,name)
        arena=Arena(p.arena_bytes);observed=[];addresses=[]
        @OBS
        def observer(i,data,count,_):
            observed.append((int(i),[float(data[j]) for j in range(count)]))
            addresses.append(C.cast(data,C.c_void_p).value-arena.address)
        require(lib.ta_run(C.byref(g),C.byref(p),arena.address,arena.size,inputs,counts,n_inputs,
                           observer,None)==0,'production execution failed')
        require([i for i,_ in observed]==list(range(g.count)) and addresses==info['offset'],
                'actual tensor traversal/address mismatch')
        worst=0.0
        for (i,values),reference in zip(observed,gold):
            close(values,reference);worst=max(worst,max(abs(a-b) for a,b in zip(values,reference)))
        outputs=[]
        for id in graph['outputs']:
            count=g.nodes[id].rows*g.nodes[id].cols
            values=list((F*count).from_address(arena.address+p.offset[id]))
            close(values,gold[id]);outputs.append(values)
        arena.guards()
        info.update(outputs=outputs,max_intermediate_abs_error=worst,
                    intermediates=[values for _,values in observed],observed_offsets=addresses,
                    observed_tensors=len(observed),observed_highwater_bytes=max(
                    o+len(values)*4 for o,(_,values) in zip(addresses,observed)),guard_ok=True)
        results[name]=info
    require(results['reuse']['arena_bytes']<=results['separate']['arena_bytes'],'reuse exceeds separate arena')
    return dict(graph=graph,oracle_tensors=gold,plans=results)


def fixtures(lib):
    records={}
    for case in range(4):
        handle=lib.ta_fixture_create(case);require(handle,'fixture allocation failed')
        try:
            g=lib.ta_fixture_graph(handle).contents
            records[lib.ta_fixture_name(case).decode()]=exercise(lib,g,lib.ta_fixture_inputs(handle),
                lib.ta_fixture_input_counts(handle),lib.ta_fixture_input_count(handle))
        finally:lib.ta_fixture_free(handle)
    return records
