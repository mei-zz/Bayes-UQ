#!/usr/bin/env python3
"""Evaluate one frozen, training-selected MIRAGE decision. Never selects gamma."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import time
import traceback
import numpy as np
import pandas as pd

NEW='Downside-penalised Bayes-UQ'
GAIN='Gain-only fifth-percentile'
ALWAYS='Always posterior-mean'
BASE='Frozen BGE'
INC='Incumbent Bayes-UQ (xi=0.25)'
METRICS=['ndcg10','mrr10','recall10','QDR','ED','CVaR10','acceptance_rate','fallback_rate','order_change_rate']

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temp,path)
def progress(out,stage,**extra):
    record={'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'stage':stage,**extra}
    write(out/'EVALUATION_PROGRESS.json',record)
    print(json.dumps(record),flush=True)
def metric(order,target):
    rank=order.index(target)+1 if target in order else 0
    return ((1/math.log2(rank+1),1/rank,1.0) if 0<rank<=10 else (0.0,0.0,0.0)),rank

def main(args):
    out=Path(args.run);source=Path(args.source)
    lock=json.loads((out/'PREDICTIONS_LOCKED.json').read_text())
    if lock['status']!='PREDICTIONS_LOCKED' or lock['test_qrels_read']:
        raise RuntimeError('Missing pre-evaluation prediction lock')
    path=out/'predictions/predictions_locked.jsonl'
    if sha(path)!=lock['prediction_sha256']:
        raise RuntimeError('Predictions changed after locking')
    if sha(out/'SELECTED_RULE.json')!=lock['selection_sha256']:
        raise RuntimeError('Training selection changed after locking')
    if (out/'EVALUATION_COMPLETED.json').exists():
        raise RuntimeError('This run has already been evaluated; do not overwrite')
    result_dir=out/'results';result_dir.mkdir(exist_ok=True)
    predictions=[json.loads(line) for line in path.read_text().splitlines() if line]
    qids=[str(row['query_id']) for row in predictions]
    if len(qids)!=2000 or len(set(qids))!=2000:raise RuntimeError('Wrong query cohort')
    incumbent_root=Path(args.incumbent)
    incumbent_lock=json.loads((incumbent_root/'PREDICTIONS_LOCKED.json').read_text())
    incumbent_path=incumbent_root/'predictions/predictions_locked.jsonl'
    if incumbent_lock['selected_gamma']!=0.25 or sha(incumbent_path)!=incumbent_lock['prediction_sha256']:
        raise RuntimeError('Incumbent locked run differs')
    if lock['source_checkpoint_sha256']!=incumbent_lock['source_checkpoint_sha256']:
        raise RuntimeError('Full-fit posterior checkpoint differs across runs')
    incumbent={str(r['query_id']):r for r in (json.loads(line) for line in incumbent_path.read_text().splitlines() if line)}
    if set(incumbent)!=set(qids):raise RuntimeError('Incumbent query cohort differs')
    shared=[BASE,ALWAYS,GAIN,'Pointwise logistic','Adapted EAU']
    for row in predictions:
        previous=incumbent[str(row['query_id'])]
        if row['source_page_offset']!=previous['source_page_offset']:raise RuntimeError('Cluster assignment differs')
        for name in shared:
            if row['orders'][name]!=previous['orders'][name]:raise RuntimeError('Frozen comparator differs: '+name)
        row['orders'][INC]=previous['orders'][NEW]
        row['incumbent_accepted']=previous['selected_accepted']
    write(out/'EVALUATION_INPUT_LOCK.json',{'new_prediction_sha256':sha(path),
        'incumbent_prediction_sha256':sha(incumbent_path),'qrels_read':False,
        'common_comparators_and_clusters_identical':True,'test_grid_evaluated':False})
    methods=list(lock['methods'])+[INC];mi={m:i for i,m in enumerate(methods)}
    assert all(m in mi for m in [NEW,GAIN,ALWAYS,BASE])
    pool_path=source/'inputs/candidate_pools_test.csv'
    pool=pd.read_csv(pool_path,dtype={'query_id':str,'doc_id':str})
    pools={qid:set(group.doc_id) for qid,group in pool.groupby('query_id')}
    qrel_path=source/'evaluator_only/test_qrels.csv'
    progress(out,'READING_QRELS_AFTER_PREDICTION_LOCK',prediction_sha256=lock['prediction_sha256'])
    qrels=pd.read_csv(qrel_path,dtype={'query_id':str,'candidate_id':str})
    if len(qrels)!=100000 or qrels.duplicated(['query_id','candidate_id']).any() or not qrels.label.isin([0,1]).all():
        raise RuntimeError('Invalid 2000x50 qrels')
    grouped={qid:g for qid,g in qrels.groupby('query_id')}
    if set(grouped)!=set(qids) or set(pools)!=set(qids):raise RuntimeError('Qrel/pool/query mismatch')
    n=len(qids);m=len(methods)
    values=np.zeros((n,m,3));accepted=np.zeros((n,m),bool);changed=np.zeros((n,m),bool)
    reachable=np.zeros(n,bool);pages=np.zeros(n,dtype=np.int64);rows=[]
    for qi,row in enumerate(predictions):
        qid=qids[qi];group=grouped[qid];ids=pools[qid]
        if len(ids)!=50 or set(group.candidate_id)!=ids:raise RuntimeError('Candidate pool mismatch')
        positives=group.loc[group.label.eq(1),'candidate_id'].tolist()
        if len(positives)>1:raise RuntimeError('Single-reference task has multiple positives')
        target=positives[0] if positives else None
        reachable[qi]=bool(positives);pages[qi]=int(row['source_page_offset'])
        orders=row['orders'];reference=list(map(str,orders[BASE]));proposal=list(map(str,orders[ALWAYS]))
        base_triple,_=metric(reference,target)
        for method in methods:
            j=mi[method];order=list(map(str,orders[method]))
            if len(order)!=50 or set(order)!=ids:raise RuntimeError('Ranking is not a complete candidate permutation')
            acc=(bool(row['selected_accepted']) if method==NEW else
                 bool(row['gamma0_accepted']) if method==GAIN else
                 bool(row['incumbent_accepted']) if method==INC else method!=BASE)
            if method in [NEW,GAIN,INC] and order!=(proposal if acc else reference):
                raise RuntimeError('Exact fallback invariant failed')
            triple,rank=metric(order,target)
            values[qi,j,:]=triple;accepted[qi,j]=acc;changed[qi,j]=(order!=reference)
            rows.append({'query_id':qid,'source_page_offset':int(pages[qi]),'reachable':bool(reachable[qi]),
                         'method':method,'target_rank':rank if rank else None,'ndcg10':triple[0],
                         'mrr10':triple[1],'recall10':triple[2],'accepted':acc,'fallback':not acc,
                         'backbone_ndcg10':base_triple[0],'delta_ndcg10':triple[0]-base_triple[0],
                         'loss':max(0.0,base_triple[0]-triple[0]),
                         'actual_order_changed':bool(changed[qi,j]),'selected_gamma':row['selected_gamma'],
                         'decision_xi':0.25 if method==INC else float(row['selected_gamma']) if method==NEW else 0.0 if method==GAIN else None})
    if reachable.sum()!=1407 or len(set(pages))!=20:raise RuntimeError('Reachability/page support changed')
    base=values[:,mi[BASE],0]
    delta=values[:,:,0]-base[:,None];loss=np.maximum(-delta,0)
    summaries=[];diagnostics=[]
    for cohort,mask in [('all',np.ones(n,bool)),('reachable',reachable)]:
        for method in methods:
            j=mi[method];d=delta[mask,j];l=loss[mask,j];count=int(mask.sum());h=math.ceil(.1*count)
            rec={'cohort':cohort,'method':method,'N':count,'selected_gamma':float(lock['selected_gamma']),
                 **dict(zip(METRICS, [*values[mask,j,:].mean(axis=0),(d < -1e-12).mean(),l.mean(),
                 np.sort(l)[-h:].mean(),accepted[mask,j].mean(),(~accepted[mask,j]).mean(),changed[mask,j].mean()])),
                 'W':int((d>1e-12).sum()),'T':int((np.abs(d)<=1e-12).sum()),'L':int((d < -1e-12).sum()),
                 'accepted':int(accepted[mask,j].sum()),'fallback':int((~accepted[mask,j]).sum())}
            summaries.append(rec)
        always_delta=delta[:,mi[ALWAYS]]
        for method in [NEW,GAIN,INC]:
            a=accepted[:,mi[method]];rejected=(~a)&mask
            rec={'cohort':cohort,'method':method,'N':int(mask.sum()),
                 'always_losses':int(((always_delta < -1e-12)&mask).sum()),
                 'rejected_improvements':int((rejected&(always_delta>1e-12)).sum()),
                 'rejected_ties':int((rejected&(np.abs(always_delta)<=1e-12)).sum()),
                 'rejected_losses':int((rejected&(always_delta < -1e-12)).sum()),
                 'remaining_losses':int(((delta[:,mi[method]] < -1e-12)&mask).sum())}
            rec['loss_interception_recall']=rec['rejected_losses']/rec['always_losses'] if rec['always_losses'] else None
            diagnostics.append(rec)
    pd.DataFrame(rows).to_csv(result_dir/'query_outcomes.csv',index=False)
    pd.DataFrame(summaries).to_csv(result_dir/'summary.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(result_dir/'action_diagnostics.csv',index=False)
    progress(out,'POINT_ESTIMATES_WRITTEN_START_CLUSTER_BOOTSTRAP',gamma=lock['selected_gamma'],replicates=args.replicates)
    unique=np.unique(pages);mapping={p:i for i,p in enumerate(unique)}
    page_index=np.array([mapping[p] for p in pages])
    rng=np.random.Generator(np.random.PCG64(20261015))
    boots={c:np.zeros((args.replicates,m,len(METRICS))) for c in ['all','reachable']}
    for start in range(0,args.replicates,128):
        end=min(args.replicates,start+128)
        counts=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=end-start)
        for cohort,mask in [('all',np.ones(n,bool)),('reachable',reachable)]:
            weights=counts[:,page_index[mask]].astype(float);den=weights.sum(axis=1)
            if np.any(den==0):raise RuntimeError('Empty reachable bootstrap sample')
            b=boots[cohort][start:end];v=values[mask];l=loss[mask];d=delta[mask]
            for k in range(3):b[:,:,k]=(weights@v[:,:,k])/den[:,None]
            b[:,:,3]=(weights@(d < -1e-12))/den[:,None]
            b[:,:,4]=(weights@l)/den[:,None]
            h=np.ceil(.1*den).astype(int)
            for j in range(m):
                order=np.argsort(-l[:,j],kind='stable');w=weights[:,order]
                previous=np.cumsum(w,axis=1)-w
                take=np.minimum(w,np.maximum(h[:,None]-previous,0))
                b[:,j,5]=(take@l[order,j])/h
            b[:,:,6]=(weights@accepted[mask].astype(float))/den[:,None]
            b[:,:,7]=1-b[:,:,6]
            b[:,:,8]=(weights@changed[mask].astype(float))/den[:,None]
        if end%1024==0 or end==args.replicates:
            progress(out,'CLUSTER_BOOTSTRAP',completed=end,total=args.replicates)
    lookup={(r['cohort'],r['method']):r for r in summaries}
    absolute=[];paired=[]
    for cohort in boots:
        for method in methods:
            j=mi[method]
            for k,name in enumerate(METRICS):
                lo,hi=np.quantile(boots[cohort][:,j,k],[.025,.975])
                absolute.append({'cohort':cohort,'method':method,'metric':name,'estimate':lookup[(cohort,method)][name],
                                 'ci_lower':lo,'ci_upper':hi,'replicates':args.replicates,'cluster_unit':'source_page_offset'})
        for comparator in methods:
            if comparator==NEW:continue
            for k,name in enumerate(METRICS[:6]):
                lo,hi=np.quantile(boots[cohort][:,mi[NEW],k]-boots[cohort][:,mi[comparator],k],[.025,.975])
                paired.append({'cohort':cohort,'method':NEW,'comparator':comparator,'metric':name,
                    'difference':lookup[(cohort,NEW)][name]-lookup[(cohort,comparator)][name],
                    'ci_lower':lo,'ci_upper':hi,'replicates':args.replicates,'multiplicity_adjusted':False})
    pd.DataFrame(absolute).to_csv(result_dir/'absolute_ci.csv',index=False)
    pd.DataFrame(paired).to_csv(result_dir/'paired_ci.csv',index=False)
    np.savez_compressed(result_dir/'cluster_bootstrap_replicates.npz',all=boots['all'],reachable=boots['reachable'],methods=np.array(methods),metrics=np.array(METRICS))
    write(out/'EVALUATION_COMPLETED.json',{'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'selected_gamma':float(lock['selected_gamma']),'N':n,'reachable':int(reachable.sum()),'clusters':len(unique),
        'bootstrap_replicates':args.replicates,'seed':20261015,'prediction_sha256':sha(path),
        'qrels_sha256':sha(qrel_path),'pool_sha256':sha(pool_path),
        'statistical_scope':'nominal source-page-cluster percentile CI conditional on fixed training fit and selection',
        'test_gamma_grid_evaluated':False,'no_test_based_selection':True,
        'exploratory_refinement_on_previously_evaluated_test':True,
        'incumbent_prediction_sha256':sha(incumbent_path)})
    progress(out,'EVALUATION_COMPLETED',gamma=lock['selected_gamma'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--run',required=True);p.add_argument('--incumbent',required=True)
    p.add_argument('--replicates',type=int,default=10000);args=p.parse_args()
    try:main(args)
    except Exception:
        out=Path(args.run);out.mkdir(parents=True,exist_ok=True)
        write(out/'EVALUATION_FAILED.json',{'traceback':traceback.format_exc()})
        raise
