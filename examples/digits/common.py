"""Dependency-free artifact audit, C fixture generation and production access."""
import ctypes as C
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
import runtime as rt
HERE=Path(__file__).resolve().parent
BATCHES={1:512,16:7168}  # predefined tensor-arena budgets, bytes
FLAGS=['-std=c11','-O3','-Wall','-Wextra','-Wpedantic','-Werror','-ffp-contract=off']


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def load_artifacts(model_path=None,fixture_path=None):
    model_path=Path(model_path or HERE/'model.json');fixture_path=Path(fixture_path or HERE/'fixture.json')
    m=json.loads(model_path.read_text());f=json.loads(fixture_path.read_text())
    rt.require(m['shape']==[64,32,10] and m['classes']==list(range(10)) and m['schema']==f['schema']==1,'model shape/classes')
    rt.require(f['model_sha256']==digest(model_path) and f['dataset_sha256']==m['dataset_sha256'],
               'model/dataset provenance mismatch')
    rt.require(m['seed']==20261014 and m['preprocessing']=='float32(pixel / 16)' and
        not m['training_parameters']['early_stopping'] and m['training_parameters']['solver']=='adam', 'training policy')
    rt.require(len(m['train_ids'])==1347 and len(m['test_ids'])==450 and
        sorted(m['train_ids']+m['test_ids'])==list(range(1797)) and f['test_ids']==m['test_ids'],'split overlap/coverage')
    rt.require(len(f['train_image_sha256'])==1347 and len(f['pixels'])==len(f['labels'])==
        len(f['reference_logits'])==len(f['reference_predictions'])==450,'fixture counts')
    train_hash=set(f['train_image_sha256'])
    for pixels,label,logits,pred in zip(f['pixels'],f['labels'],f['reference_logits'],f['reference_predictions']):
        rt.require(len(pixels)==64 and all(type(v) is int and 0<=v<=16 for v in pixels) and
                   type(label) is int and 0<=label<10,'pixel/label range')
        rt.require(hashlib.sha256(bytes(pixels)).hexdigest() not in train_hash,'duplicate image across split')
        rt.require(len(logits)==10 and pred==max(range(10),key=lambda j:logits[j]),'reference prediction mismatch')
    rt.require([len(w) for w in m['weights']]==[64,32] and
        all(len(row)==32 for row in m['weights'][0]) and all(len(row)==10 for row in m['weights'][1]) and
        [len(b) for b in m['biases']]==[32,10],'parameter shapes')
    flat=sum([sum(w,[])+b for w,b in zip(m['weights'],m['biases'])],[])
    rt.require(all(math.isfinite(x) and struct.unpack('<f',struct.pack('<f',x))[0]==x for x in flat),'parameters not exact finite float32')
    return m,f


def header():
    m,f=load_artifacts();arrays={
        'digits_w0':sum(m['weights'][0],[]),'digits_b0':m['biases'][0],
        'digits_w1':sum(m['weights'][1],[]),'digits_b1':m['biases'][1],
        'digits_x':[v/16 for row in f['pixels'] for v in row]}
    out=['/* Generated from attributed JSON artifacts; host fixture only. */']
    for name,values in arrays.items():
        tokens=[float(v).hex()+'f' for v in values]
        out.append('static const float %s[%d]={\n%s\n};'%(name,len(values),
            ',\n'.join(','.join(tokens[i:i+8]) for i in range(0,len(tokens),8))))
    out.append('static const unsigned char digits_y[450]={'+','.join(map(str,f['labels']))+'};')
    return '\n'.join(out)+'\n'


def library(path=ROOT/'build/digits/engine.so'):
    lib=C.CDLL(str(path))
    lib.ta_plan_build.argtypes=[C.POINTER(rt.Graph),C.c_int,C.POINTER(rt.Plan),C.POINTER(rt.Scratch)]
    lib.ta_plan_build.restype=C.c_int
    lib.ta_run.argtypes=[C.POINTER(rt.Graph),C.POINTER(rt.Plan),C.c_void_p,rt.S,C.POINTER(rt.P),rt.SP,rt.S,rt.OBS,C.c_void_p]
    lib.ta_run.restype=C.c_int
    lib.digits_graph.argtypes=[rt.S,C.POINTER(rt.Graph),C.POINTER(rt.Node),rt.SP]
    lib.digits_inputs.restype=rt.P
    return lib


def graph(lib,batch):
    nodes=(rt.Node*4)();out=rt.S();g=rt.Graph()
    lib.digits_graph(batch,C.byref(g),nodes,C.byref(out))
    return g,nodes,out
