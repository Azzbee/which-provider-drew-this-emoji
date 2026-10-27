#!/usr/bin/env python3
"""Inspect a frozen validation run and compare a few outputs to upstream predict.

This is a post-evaluation implementation audit, not prompt or model selection.
It never changes the requests, checkpoint, exported predictions or routing.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from decision_pipeline import CLASSES,HERE,load_model,save_json
import numpy as np
import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--model',type=Path,default=HERE/'julia_model')
    args=parser.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.manual_seed(42);np.random.seed(42)
    args.out.mkdir(parents=True,exist_ok=True)
    saved_p=np.load(args.run/'probabilities.npy')
    saved_z=np.load(args.run/'logits.npy')
    winner=saved_p.argmax(1)
    rows=[json.loads(line) for line in (args.run/'requests.jsonl').read_text().splitlines()]
    # First example of each exported winner, plus fixed boundary/middle rows.
    # Correct labels are never consulted in choosing the audited examples.
    chosen={0,len(rows)//2,len(rows)-1}
    for index in range(len(CLASSES)):
        where=np.flatnonzero(winner==index)
        if len(where):chosen.add(int(where[0]))
    chosen=sorted(chosen)
    requests=[rows[i]['request'] for i in chosen]
    engine,sequence,_=load_model(SimpleNamespace(model=args.model,out=args.out,threads=2))
    fresh_z=np.asarray(engine.logits(requests),dtype=np.float32)
    upstream=engine.predict(requests)
    upstream_winner=np.asarray([result['index'] for result in upstream])
    fresh_winner=fresh_z.argmax(1)
    assert np.array_equal(upstream_winner,fresh_winner)
    assert np.array_equal(upstream_winner,winner[chosen])
    maximum_difference=float(np.abs(fresh_z-saved_z[chosen]).max())
    assert np.allclose(fresh_z,saved_z[chosen],atol=5e-5,rtol=1e-5)
    encoded=sequence(engine.tokenizer,requests[0],1024,256,strict=True)
    token_ranges=[]
    for j,pos in enumerate(encoded['markers']):
        count=encoded['option_tokens'][j]
        rendered=engine.tokenizer.decode(encoded['ids'][pos+1:pos+1+count])
        assert rendered.strip()==CLASSES[j]
        token_ranges.append({'index':j,'option':CLASSES[j],'marker_position':pos,
                             'decoded_option':rendered})
    summary={
        'purpose':'Post-evaluation implementation audit, with unchanged requests and weights',
        'classes':CLASSES,
        'predicted_label_histogram':dict(zip(CLASSES,np.bincount(winner,minlength=7).tolist())),
        'mean_probability':dict(zip(CLASSES,saved_p.mean(0).tolist())),
        'std_probability':dict(zip(CLASSES,saved_p.std(0).tolist())),
        'top_probability_quantiles':dict(zip(['minimum','q25','median','q75','maximum'],
                                           np.quantile(saved_p.max(1),[0,.25,.5,.75,1]).tolist())),
        'distinct_logit_rows':int(np.unique(saved_z,axis=0).shape[0]),
        'all_logit_rows_finite':bool(np.isfinite(saved_z).all()),
        'audited_request_count':len(chosen),
        'audited_rows':[{'row_index':i,'external_Id':rows[i]['Id'],
                        'saved_index':int(winner[i]),'saved_label':CLASSES[winner[i]],
                        'upstream_predict_index':int(upstream_winner[j]),
                        'fresh_logits_argmax':int(fresh_winner[j])} for j,i in enumerate(chosen)],
        'maximum_logit_difference_from_original_batch':maximum_difference,
        'upstream_predict_and_exported_argmax_agree_on_all_audited_rows':True,
        'decoded_option_marker_order':token_ranges,
        'qtype_index':encoded['qtype'],
        'orientation':'Upstream predict uses softmax and argmax over the caller-supplied option list, in the same order as exported columns',
        'interpretation':'No class-index reversal or constant-output implementation error found. Outputs vary, but their winners strongly favour google and messenger. This is a failed zero-shot decision-layer configuration; the audit alone does not identify the cause of the semantic bias or measure TypeSafe Jev.'}
    save_json(args.out/'diagnostics.json',summary)
    np.save(args.out/'fresh_logits.npy',fresh_z)
    save_json(args.out/'upstream_predict.json',upstream)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
