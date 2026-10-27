#!/usr/bin/env python3
"""Verify the fixed disagreement-only experimental Julia competition CSV."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from decision_pipeline import CLASSES,WEIGHTS_SHA,load_experts,make_requests,save_json,sha
import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();run=args.run
    ids,truth,expert,sources=load_experts(args.package,'test')
    assert truth is None and len(ids)==9879
    route=expert[:,0].argmax(1)!=expert[:,1].argmax(1)
    selected=np.flatnonzero(route)
    assert len(selected)==574
    routed_ids=np.asarray(ids)[selected].tolist()
    assert json.loads((run/'ids.json').read_text())==routed_ids
    complete=np.load(run/'complete.npy')
    assert complete.shape==(574,) and complete.dtype==bool and complete.all()
    probabilities=np.load(run/'probabilities.npy');logits=np.load(run/'logits.npy')
    hybrid=np.load(run/'hybrid_probabilities.npy')
    assert probabilities.shape==logits.shape==(574,7)
    assert hybrid.shape==(9879,7)
    assert json.loads((run/'hybrid_ids.json').read_text())==ids
    assert all(a.dtype==np.float32 and np.isfinite(a).all() for a in [probabilities,logits,hybrid])
    expected=np.exp(logits-logits.max(1,keepdims=True));expected/=expected.sum(1,keepdims=True)
    assert np.array_equal(expected,probabilities)
    assert np.allclose(probabilities.sum(1),1,atol=1e-6)
    expected_hybrid=(expert[:,0]+expert[:,1])*.5
    expected_hybrid[selected]=probabilities
    assert np.array_equal(expected_hybrid,hybrid)
    requests=[json.loads(line) for line in (run/'requests.jsonl').read_text().splitlines()]
    expected_requests=make_requests(expert[selected])
    assert len(requests)==574
    for identifier,item,expected_request in zip(routed_ids,requests,expected_requests):
        assert set(item)=={'Id','request'} and item['Id']==identifier
        assert item['request']==expected_request
    csv_path=run/'submission_julia_decision_hybrid.csv'
    submission=pd.read_csv(csv_path,dtype=str)
    assert submission.columns.tolist()==['Id','Label']
    assert submission.Id.tolist()==ids and submission.Id.is_unique
    assert not submission.isna().any().any()
    assert submission.Label.tolist()==np.asarray(CLASSES)[hybrid.argmax(1)].tolist()
    assert set(submission.Label).issubset(CLASSES)
    receipt=json.loads((run/'prediction_receipt.json').read_text())
    assert receipt['count']==574 and receipt['all_complete']
    assert receipt['probabilities_sha256']==sha(run/'probabilities.npy')
    assert receipt['source_probabilities_sha256']==sources
    submission_receipt=json.loads((run/'hybrid_submission_receipt.json').read_text())
    assert submission_receipt['count']==9879 and submission_receipt['routed_count']==574
    assert submission_receipt['csv_sha256']==sha(csv_path)
    result={'status':'complete_verified','role':'experimental_not_recommended_for_kaggle',
            'count':9879,'routed_count':574,'classes':CLASSES,
            'ids_sha256':sha(run/'ids.json'),'hybrid_ids_sha256':sha(run/'hybrid_ids.json'),
            'probabilities_sha256':sha(run/'probabilities.npy'),
            'hybrid_probabilities_sha256':sha(run/'hybrid_probabilities.npy'),
            'requests_sha256':sha(run/'requests.jsonl'),'csv_sha256':sha(csv_path),
            'model_weights_sha256':WEIGHTS_SHA,
            'verified':['all 9879 test IDs exactly once in required order','complete finite arrays',
                        'class order','logits softmax','all 574 requests match the fixed score-state renderer',
                        'exact prediction-only routing','unrouted equal-ensemble outputs retained',
                        'CSV labels match hybrid probability argmax'],
            'kaggle_uploaded':False}
    save_json(run/'COMPLETE.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
