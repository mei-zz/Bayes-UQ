#!/usr/bin/env python3
"""Fixed-posterior decision-rule replay. Never fits or selects thresholds."""
import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-12
ALPHA = 0.05
NORMAL_Z = 1.6448536269514722
CANTELLI_K = math.sqrt(19.0)
POLICIES = ['always_proposal', 'positive_mean_gain', 'empirical_quantile_05',
            'normal_moment_05', 'cantelli_moment_05']
REFERENCE = 'empirical_quantile_05'
METRICS = ['ndcg10', 'mrr10', 'recall10', 'QDR', 'ED', 'CVaR10']

def utc():
    return datetime.now(timezone.utc).isoformat()

def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def progress(out, stage, **extra):
    record = dict(time_utc=utc(), pid=os.getpid(), stage=stage, **extra)
    write_json(out/'progress.json', record)
    print(json.dumps(record), flush=True)

def discount(rank):
    return 1.0 / math.log2(rank + 1.0) if rank <= 10 else 0.0

def summarize(ndcg, mrr, recall, base):
    delta = ndcg - base
    loss = np.maximum(0.0, -delta)
    tail = math.ceil(0.1 * len(base))
    return np.array([ndcg.mean(), mrr.mean(), recall.mean(), (delta < -EPS).mean(),
                     loss.mean(), np.sort(loss)[-tail:].mean()])

def main(args):
    out, source = Path(args.output), Path(args.source)
    out.mkdir(parents=True, exist_ok=True)
    (out/'predictions').mkdir(exist_ok=True)
    (out/'results').mkdir(exist_ok=True)
    if (out/'PROTOCOL_LOCK.json').exists():
        raise RuntimeError('Output already has a protocol lock; do not overwrite a run')
    protocol = {
        'created_utc': utc(), 'source_root': str(source), 'design':
        'Fixed-posterior decision-rule replay on the previously developed internal cohort',
        'expected_queries': 483, 'outer_folds': 5, 'rho': 2, 'draws_per_query': 128,
        'proposal': 'Saved rho=2 posterior-mean order, canonical-ID ties',
        'reference': 'Saved frozen backbone, canonical-ID ties',
        'gain': 'Same-draw expected discounted utility of proposal minus reference',
        'rules': {
            'always_proposal': 'Accept every saved proposal',
            'positive_mean_gain': 'mean(G)>0',
            'empirical_quantile_05': 'numpy linear empirical quantile(G,0.05)>0',
            'normal_moment_05': 'mean(G)-1.6448536269514722*std(G,ddof=0)>0',
            'cantelli_moment_05': 'mean(G)-sqrt(19)*std(G,ddof=0)>0'},
        'sd_definition': 'Empirical posterior SD of 128 gain draws, not SD/sqrt(128)',
        'threshold_selection': 'All coefficients and strict-zero thresholds fixed before this run; no test-qrel selection',
        'action_lock': 'All actions and complete selected rankings saved before reading qrels',
        'fallback': 'Complete saved reference order',
        'comparison_reference': REFERENCE,
        'primary_metrics': ['ndcg10','QDR','ED','CVaR10'],
        'secondary_metrics': ['mrr10','recall10'],
        'bootstrap': {'unit':'query','replicates':args.bootstrap,'seed':20261015,
                      'common_resamples':True,'tail':'ceil(0.1*N), retaining zeros',
                      'interval':'nominal paired percentile 95% CI; not multiplicity adjusted',
                      'scope':'conditional on fixed predictions; excludes refitting and selection uncertainty'},
        'p_values': 'Not computed; no significance claims will be made from this job',
        'limits': ['Not a mutual-information substitution experiment',
                   'No independent holdout or training-uncertainty evaluation',
                   'Normal-moment rule is a moment approximation, not an exact quantile',
                   'Cantelli concerns the empirical gain distribution, not realised nDCG risk',
                   'Acceptance levels need not match; all actions will be reported'],
        'python':sys.version, 'numpy':np.__version__, 'pandas':pd.__version__}
    write_json(out/'PROTOCOL_LOCK.json', protocol)
    progress(out,'protocol_locked')
    run = source/'02_CANONICAL_OOF/raw_locked_run'
    sources, query_rows, action_rows, rank_rows = [], [], [], []
    orders = {}
    # No qrels or observed evaluation outcomes are read in this phase.
    for fold in range(5):
        fd = run/'folds'/('outer_%d'%fold)
        paths = [fd/name for name in ['candidate_scores.csv','posterior_draw_candidate_probabilities.csv.gz',
                                      'posterior_utility_draws.csv','policy_decisions.csv']]
        frames=[]
        for p in paths:
            sources.append({'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)})
            frames.append(pd.read_csv(p,dtype={'query_id':str,'candidate_id':str},float_precision='round_trip'))
        scores, probabilities, utilities, decisions = frames
        probabilities = probabilities.loc[probabilities.rho.eq(2)]
        utilities = utilities.loc[utilities.rho.eq(2)]
        for qid, scores_q in scores.groupby('query_id',sort=True):
            key=(fold,qid)
            if qid in orders:
                raise RuntimeError('Query appears in multiple outer folds: '+qid)
            s=scores_q.sort_values('candidate_id',kind='mergesort')
            ids=s.candidate_id.tolist()
            base=s.sort_values(['base_rank','candidate_id'],kind='mergesort').candidate_id.tolist()
            proposal=s.sort_values(['rho2_bma_rank','candidate_id'],kind='mergesort').candidate_id.tolist()
            if len(set(ids))!=len(ids) or set(base)!=set(proposal):
                raise RuntimeError('Candidate identity inconsistency')
            p=probabilities.loc[probabilities.query_id.eq(qid)]
            pivot=p.pivot(index='draw_index',columns='candidate_id',values='relevance_probability').reindex(columns=ids).sort_index()
            if pivot.shape!=(128,len(ids)) or pivot.isna().any().any():
                raise RuntimeError('Incomplete posterior draws for '+qid)
            null=p.groupby('draw_index').null_probability.first().reindex(pivot.index).to_numpy()
            if (p.groupby('draw_index').null_probability.max()-p.groupby('draw_index').null_probability.min()).abs().max()>1e-10:
                raise RuntimeError('Inconsistent null probabilities')
            prob=pivot.to_numpy()
            if np.min(prob)<0 or np.min(null)<-1e-12 or not np.allclose(prob.sum(axis=1)+null,1.0,atol=1e-9):
                raise RuntimeError('Invalid categorical draws')
            br={cid:i+1 for i,cid in enumerate(base)}
            pr={cid:i+1 for i,cid in enumerate(proposal)}
            differences=np.array([discount(pr[c])-discount(br[c]) for c in ids])
            gains=prob@differences
            u=utilities.loc[utilities.query_id.eq(qid)].set_index('draw_index').reindex(pivot.index)
            if len(u)!=128 or not np.allclose(gains,u.expected_ndcg_gain.to_numpy(),rtol=1e-8,atol=1e-11):
                raise RuntimeError('Saved gains do not match same-draw proposal-reference gains for '+qid)
            mu=float(gains.mean()); sd=float(gains.std(ddof=0)); q05=float(np.quantile(gains,0.05,method='linear'))
            # Use original draw file precision for decisions, after reconstructing and validating it.
            gains=u.expected_ndcg_gain.to_numpy(float)
            mu=float(gains.mean()); sd=float(gains.std(ddof=0)); q05=float(np.quantile(gains,0.05,method='linear'))
            actions={'always_proposal':True,'positive_mean_gain':mu>0,'empirical_quantile_05':q05>0,
                     'normal_moment_05':mu-NORMAL_Z*sd>0,
                     'cantelli_moment_05':mu-CANTELLI_K*sd>0}
            saved=decisions.loc[decisions.query_id.eq(qid)&decisions.policy.eq('RHO2_LCB05_GATE')]
            if len(saved)!=1 or bool(int(saved.iloc[0].gate_applied))!=actions[REFERENCE]:
                raise RuntimeError('Fifth-percentile action differs from locked source for '+qid)
            expected=proposal if actions[REFERENCE] else base
            if saved.iloc[0].selected_order.split('|')!=expected:
                raise RuntimeError('Selected complete ranking differs from locked source for '+qid)
            orders[qid]={'fold':fold,'base':base,'proposal':proposal,'actions':actions}
            query_rows.append({'query_id':qid,'outer_fold':fold,'draws':128,'mean_gain':mu,'gain_sd_ddof0':sd,
                               'gain_variance_ddof0':sd*sd,'gain_q05_linear':q05,
                               'normal_moment_score':mu-NORMAL_Z*sd,'cantelli_moment_score':mu-CANTELLI_K*sd})
            for policy,accept in actions.items():
                selected=proposal if accept else base
                action_rows.append({'query_id':qid,'outer_fold':fold,'policy':policy,'accepted':int(accept),
                                    'fallback':int(not accept),'ranking_changed':int(selected!=base)})
                for rank,cid in enumerate(selected,1):
                    rank_rows.append({'query_id':qid,'outer_fold':fold,'policy':policy,'candidate_id':cid,'selected_rank':rank})
        progress(out,'loaded_fold',fold=fold,queries=len(orders))
    if len(orders)!=483:
        raise RuntimeError('Expected the full 483-query cohort')
    for name,rows in [('posterior_statistics.csv',query_rows),('policy_actions.csv',action_rows),('selected_rankings.csv',rank_rows)]:
        pd.DataFrame(rows).to_csv(out/'predictions'/name,index=False)
    write_json(out/'ACTIONS_LOCKED.json',{'time_utc':utc(),'queries':483,'policies':POLICIES,
        'qrels_read':False,'files':{name:digest(out/'predictions'/name) for name in ['posterior_statistics.csv','policy_actions.csv','selected_rankings.csv']}})
    progress(out,'actions_locked_before_qrels')
    qrel_path=source/'14_SOURCE_AND_HASHES/source_inputs/QRELS_SEPARATE.csv'
    sources.append({'path':str(qrel_path),'bytes':qrel_path.stat().st_size,'sha256':digest(qrel_path)})
    qrels=pd.read_csv(qrel_path,dtype={'query_id':str,'candidate_id':str})
    positive=qrels.loc[qrels.relevance.gt(0)].groupby('query_id').candidate_id.apply(set).to_dict()
    if any(len(v)>1 for v in positive.values()):
        raise RuntimeError('This evaluator requires the recorded single-positive task')
    def evaluate(order,target):
        ranks=[i+1 for i,c in enumerate(order) if c in target]
        rank=min(ranks) if ranks else 10**9
        return (discount(rank),1.0/rank if rank<=10 else 0.0,float(rank<=10))
    qids=sorted(orders)
    base=np.array([evaluate(orders[q]['base'],positive.get(q,set())) for q in qids])
    tensors=np.zeros((len(POLICIES),len(qids),3))
    outcomes=[]
    for pi,policy in enumerate(POLICIES):
        for qi,qid in enumerate(qids):
            rec=orders[qid];accept=rec['actions'][policy]
            selected=rec['proposal'] if accept else rec['base']
            value=evaluate(selected,positive.get(qid,set()))
            tensors[pi,qi,:]=value
            delta=value[0]-base[qi,0]
            outcomes.append({'query_id':qid,'outer_fold':rec['fold'],'policy':policy,'accepted':int(accept),
                'reachable':int(bool(set(rec['base'])&positive.get(qid,set()))),
                'ndcg10':value[0],'mrr10':value[1],'recall10':value[2],
                'backbone_ndcg10':base[qi,0],'delta_ndcg10':delta,'loss':max(0,-delta)})
    pd.DataFrame(outcomes).to_csv(out/'results/query_outcomes.csv',index=False)
    estimates=np.stack([summarize(a[:,0],a[:,1],a[:,2],base[:,0]) for a in tensors])
    summaries=[]
    for pi,policy in enumerate(POLICIES):
        subset=pd.DataFrame(outcomes).loc[lambda x:x.policy.eq(policy)]
        delta=subset.delta_ndcg10.to_numpy();acc=subset.accepted.to_numpy(bool)
        item={'policy':policy,'N':483,**dict(zip(METRICS,estimates[pi].tolist())),
              'W':int((delta>EPS).sum()),'T':int((np.abs(delta)<=EPS).sum()),'L':int((delta<-EPS).sum()),
              'accepted':int(acc.sum()),'fallback':int((~acc).sum()),'acceptance_rate':float(acc.mean())}
        proposal=np.array([evaluate(orders[q]['proposal'],positive.get(q,set()))[0] for q in qids])-base[:,0]
        for label,mask in [('improvements',proposal>EPS),('ties',np.abs(proposal)<=EPS),('losses',proposal<-EPS)]:
            item['rejected_proposal_'+label]=int(((~acc)&mask).sum())
        summaries.append(item)
    pd.DataFrame(summaries).to_csv(out/'results/summary.csv',index=False)
    write_json(out/'SOURCE_MANIFEST.json',{'source_files':sources,'qrels_used_only_after_action_lock':True})
    progress(out,'evaluation_complete_start_bootstrap',replicates=args.bootstrap)
    rng=np.random.Generator(np.random.PCG64(20261015))
    boot=np.empty((args.bootstrap,len(POLICIES),len(METRICS)))
    n=len(qids);tail=math.ceil(.1*n)
    for start in range(0,args.bootstrap,200):
        end=min(args.bootstrap,start+200)
        index=rng.integers(0,n,size=(end-start,n))
        for pi in range(len(POLICIES)):
            values=tensors[pi][index];delta=values[:,:,0]-base[index,0];loss=np.maximum(0,-delta)
            boot[start:end,pi,:3]=values.mean(axis=1)
            boot[start:end,pi,3]=(delta<-EPS).mean(axis=1)
            boot[start:end,pi,4]=loss.mean(axis=1)
            boot[start:end,pi,5]=np.partition(loss,n-tail,axis=1)[:,-tail:].mean(axis=1)
        if end%2000==0 or end==args.bootstrap:
            progress(out,'bootstrap',completed=end,total=args.bootstrap)
    intervals=[];paired=[];ri=POLICIES.index(REFERENCE)
    for pi,policy in enumerate(POLICIES):
        for mi,metric in enumerate(METRICS):
            lo,hi=np.quantile(boot[:,pi,mi],[.025,.975])
            intervals.append({'policy':policy,'metric':metric,'estimate':estimates[pi,mi],
                              'ci_lower':lo,'ci_upper':hi,'replicates':args.bootstrap})
            if policy!=REFERENCE:
                lo,hi=np.quantile(boot[:,pi,mi]-boot[:,ri,mi],[.025,.975])
                paired.append({'policy':policy,'reference':REFERENCE,'contrast':'policy minus reference',
                    'metric':metric,'difference':estimates[pi,mi]-estimates[ri,mi],
                    'ci_lower':lo,'ci_upper':hi,'replicates':args.bootstrap,'multiplicity_adjusted':False})
    pd.DataFrame(intervals).to_csv(out/'results/absolute_ci.csv',index=False)
    pd.DataFrame(paired).to_csv(out/'results/paired_ci_vs_quantile05.csv',index=False)
    np.savez_compressed(out/'results/bootstrap_replicates.npz',values=boot,policies=np.array(POLICIES),metrics=np.array(METRICS))
    write_json(out/'COMPLETED.json',{'completed_utc':utc(),'queries':n,'policies':POLICIES,
        'replicates':args.bootstrap,'original_sources_modified':False,
        'locked_quantile_policy_actions_reproduced':True})
    progress(out,'completed',queries=n,replicates=args.bootstrap)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--bootstrap',type=int,default=10000)
    args=parser.parse_args()
    try:
        main(args)
    except BaseException:
        output=Path(args.output)
        output.mkdir(parents=True,exist_ok=True)
        write_json(output/'FAILED.json',{'time_utc':utc(),'traceback':traceback.format_exc()})
        traceback.print_exc()
        sys.exit(1)
