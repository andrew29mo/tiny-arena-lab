"""Explicit optional sklearn reference/export check; never trains or tunes on test data."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent

def fixture():
    model=json.loads((HERE/'model.json').read_text());d=load_digits()
    expected=MLPClassifier(hidden_layer_sizes=(32,),activation='relu',solver='adam',
        max_iter=1500,random_state=20261014,early_stopping=False,tol=1e-5,n_iter_no_change=30)
    if model['training_parameters']!=json.loads(json.dumps(expected.get_params())):
        raise ValueError('recorded training policy differs from fixed upstream policy')
    raw=d.data.astype('<f4');labels=d.target.astype('<i4')
    digest=hashlib.sha256(raw.tobytes()+labels.tobytes()).hexdigest()
    if digest!=model['dataset_sha256']:raise ValueError('dataset provenance mismatch')
    train,test=train_test_split(np.arange(len(labels)),test_size=.25,random_state=model['seed'],stratify=labels)
    if train.tolist()!=model['train_ids'] or test.tolist()!=model['test_ids']:raise ValueError('split mismatch')
    train_hash=[hashlib.sha256(row.astype('u1').tobytes()).hexdigest() for row in raw[train]]
    if any(hashlib.sha256(row.astype('u1').tobytes()).hexdigest() in train_hash for row in raw[test]):
        raise ValueError('duplicate image across split')
    x=np.ascontiguousarray(raw[test]/16,dtype=np.float32)
    clf=MLPClassifier(hidden_layer_sizes=(32,),activation='relu')
    clf.coefs_=[np.array(w,dtype=np.float32) for w in model['weights']]
    clf.intercepts_=[np.array(b,dtype=np.float32) for b in model['biases']]
    clf.classes_=np.arange(10);clf.n_features_in_=64;clf.n_layers_=3;clf.n_outputs_=10;clf.out_activation_='softmax'
    with threadpool_limits(limits=1):
        probability=clf.predict_proba(x)
        hidden=np.maximum(0,x@clf.coefs_[0]+clf.intercepts_[0])
        logits=hidden@clf.coefs_[1]+clf.intercepts_[1]
    np.testing.assert_allclose(probability,model['export_reference_probabilities'],atol=1e-5,rtol=3e-5)
    return dict(schema=1,source_revision='1b5f64859433f83bf59962cd601911c4d769da26',
        source_repository='https://github.com/andrew29mo/tiny-infer-lab',
        model_sha256=hashlib.sha256((HERE/'model.json').read_bytes()).hexdigest(),
        dataset_sha256=digest,train_image_sha256=train_hash,test_ids=test.tolist(),
        pixels=raw[test].astype(int).tolist(),labels=labels[test].tolist(),
        reference_logits=logits.tolist(),reference_predictions=probability.argmax(axis=1).tolist(),
        versions=dict(numpy=np.__version__,sklearn=sklearn.__version__))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    value=fixture();path=HERE/'fixture.json'
    if args.check:
        saved=json.loads(path.read_text())
        for k in value:
            if k not in ('reference_logits','versions') and value[k]!=saved[k]:raise ValueError('fixture differs: '+k)
        np.testing.assert_allclose(value['reference_logits'],saved['reference_logits'],atol=1e-5,rtol=3e-5)
        print('PASS: installed sklearn dataset, fixed split, export probabilities and reference logits')
    else:
        path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
        print('Exported attributed held-out fixture; no training performed')
