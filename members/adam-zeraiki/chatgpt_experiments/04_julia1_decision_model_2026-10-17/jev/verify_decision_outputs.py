#!/usr/bin/env python3
"""Verify real IDs, requests, scores and hybrid routing without choosing models."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

# Load the same isolated dependency path and thread bounds as inference before
# importing NumPy/pandas, so a fresh environment uses the recorded runtime.
from decision_pipeline import make_requests,load_experts,QUESTION,WEIGHTS_SHA
import numpy as np
import pandas as pd

CLASSES=['apple','facebook','google','messenger','mozilla','samsung','whatsapp']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();r=args.run;b=args.package
    ids,_,expert,sources=load_experts(b,'validation')
    assert json.loads((r/'ids.json').read_text())==ids
    complete=np.load(r/'complete.npy')
    assert complete.shape==(1976,) and complete.all()
    probs=np.load(r/'probabilities.npy');logits=np.load(r/'logits.npy')
    hybrid=np.load(r/'hybrid_probabilities.npy')
    assert all(a.shape==(1976,7) and np.isfinite(a).all() for a in [probs,logits,hybrid])
    assert probs.dtype==hybrid.dtype==np.float32
    expected=np.exp(logits-logits.max(1,keepdims=True));expected/=expected.sum(1,keepdims=True)
    assert np.array_equal(expected,probs)
    assert np.allclose(probs.sum(1),1,atol=1e-6)
    route=expert[:,0].argmax(1)!=expert[:,1].argmax(1)
    expected_hybrid=(expert[:,0]+expert[:,1])*.5
    expected_hybrid[route]=probs[route]
    assert np.array_equal(expected_hybrid,hybrid)
    routing=json.loads((r/'hybrid_routing.json').read_text())
    assert routing['routed_count']==int(route.sum())==112
    assert routing['routed_ids']==np.asarray(ids)[route].tolist()
    rows=[json.loads(line) for line in (r/'requests.jsonl').read_text().splitlines()]
    assert len(rows)==1976
    expected_requests=make_requests(expert)
    for identifier,item,expected_row in zip(ids,rows,expected_requests):
        assert set(item)=={'Id','request'} and item['Id']==identifier
        request=item['request']
        assert request==expected_row
        assert set(request)=={'type','state','question','options'}
        assert request['question']==QUESTION and request['options']==CLASSES
        assert set(request['state'])=={'task','predictions'}
        for model_scores in request['state']['predictions'].values():
            assert list(model_scores)==CLASSES
            assert all(isinstance(v,float) for v in model_scores.values())
    prediction_csv=pd.read_csv(r/'predictions.csv',dtype={'Id':str})
    assert prediction_csv.Id.tolist()==ids
    assert prediction_csv.predicted_label.tolist()==np.asarray(CLASSES)[probs.argmax(1)].tolist()
    assert np.allclose(prediction_csv[['p_'+c for c in CLASSES]].to_numpy(),probs,atol=1e-7)
    receipt=json.loads((r/'prediction_receipt.json').read_text())
    assert receipt['count']==1976 and receipt['all_complete']
    assert receipt['probabilities_sha256']==sha(r/'probabilities.npy')
    assert receipt['source_probabilities_sha256']==sources
    result={'status':'complete_verified','count':1976,'routed_count':112,
            'classes':CLASSES,'ids_sha256':sha(r/'ids.json'),
            'probabilities_sha256':sha(r/'probabilities.npy'),
            'hybrid_probabilities_sha256':sha(r/'hybrid_probabilities.npy'),
            'requests_sha256':sha(r/'requests.jsonl'),'model_weights_sha256':WEIGHTS_SHA,
            'verified':['complete finite arrays','exact ID alignment','class order','logits softmax',
                        'fixed prediction-only routing','all request payloads match only the21suppliedscores',
                        'no labels or IDs in encoded state','CSV labels match probability argmax'],
            'accuracy_scored_here':False}
    tmp=r/'COMPLETE.json.tmp';tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(r/'COMPLETE.json')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
