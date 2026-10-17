#!/usr/bin/env python3
from pathlib import Path
import argparse
import importlib.util
import json
import math
import shutil
import sys
import traceback
import numpy as np

GRID=[k/200 for k in range(101)]+[1.0,2.0,4.0,8.0]
PROTOCOL='MIRAGE_FINE_TRAIN_OOF_GRID_20261015'

def main(args):
    output=Path(args.output);parent=Path(args.parent);source=Path(args.source)
    if (output/'SELECTED_RULE.json').exists():raise RuntimeError('Do not overwrite a selected run')
    spec=importlib.util.spec_from_file_location('coarse_runner',Path(__file__).parent/'coarse_runner.py')
    coarse=importlib.util.module_from_spec(spec);sys.modules[spec.name]=coarse;spec.loader.exec_module(coarse)
    coarse.PROTOCOL_ID=PROTOCOL
    output.mkdir(parents=True,exist_ok=True)
    coarse.atomic_json(output/'PREDECLARED_PROTOCOL.json',{'protocol':PROTOCOL,'time_utc':coarse.now(),
        'grid':GRID,'budget':0.005,'primary_selection_metric':'CVaR10%',
        'selection':['minimum CVaR','maximum nDCG','minimum xi'],
        'numeric_comparison':'Unrounded float64, exact lexicographic comparisons, tolerance=0',
        'source_train_oof':str(parent/'train_oof/oof_gain_downside.npz'),
        'test_cohort_previously_evaluated':True,'exploratory':True,'test_results_used_for_parameter_selection':False})
    coarse.progress(output,'FINE_GRID_STARTED',candidate_count=len(GRID))
    parent_rule=json.loads((parent/'SELECTED_RULE.json').read_text())
    if parent_rule['selected_gamma']!=0.25:raise RuntimeError('Incumbent identity changed')
    data_path=parent/'train_oof/oof_gain_downside.npz'
    with np.load(data_path,allow_pickle=False) as data:
        gain=data['gain_draws'];downside=data['downside_draws'];always=data['always_ndcg10'];base=data['frozen_bge_ndcg10']
        if gain.shape!=(1000,128) or downside.shape!=gain.shape:raise RuntimeError('Wrong OOF arrays')
        if len(np.unique(data['source_page_offset']))!=50 or set(data['fold'])!=set(range(5)):raise RuntimeError('Wrong page/fold structure')
        always_metrics=coarse.train_metrics(always,base)
        results=[];actions=[]
        for xi in GRID:
            q=np.quantile(gain-xi*downside,.05,axis=1,method='linear');accept=q>0
            value=np.where(accept,always,base);metrics=coarse.train_metrics(value,base,accept)
            mean_ok=metrics['nDCG@10']>=always_metrics['nDCG@10']-.005
            tail_ok=metrics['CVaR10%']<=always_metrics['CVaR10%']
            delta=always-base
            results.append({'gamma':xi,**metrics,'mean_constraint_passed':bool(mean_ok),
                'tail_constraint_passed':bool(tail_ok),'eligible':bool(mean_ok and tail_ok),
                'rejected_improvements':int(((~accept)&(delta>1e-12)).sum()),
                'rejected_ties':int(((~accept)&(np.abs(delta)<=1e-12)).sum()),
                'rejected_losses':int(((~accept)&(delta < -1e-12)).sum())})
            for i,qid in enumerate(data['query_ids']):
                actions.append({'query_id':str(qid),'fold':int(data['fold'][i]),'gamma':xi,
                                'accepted':bool(accept[i]),'utility_q05':float(q[i]),'ndcg10':float(value[i])})
    destination=output/'train_oof';destination.mkdir(exist_ok=True)
    coarse.write_csv(destination/'gamma_grid_results.csv',results)
    coarse.write_csv(destination/'gamma_grid_oof_actions.csv',actions)
    shutil.copy2(data_path,destination/data_path.name)
    eligible=[r for r in results if r['eligible']]
    selected=min(eligible,key=lambda r:(r['CVaR10%'],-r['nDCG@10'],r['gamma'])) if eligible else results[0]
    module,vendor_path=coarse.load_vendor()
    rule=dict(parent_rule)
    rule.update(protocol=PROTOCOL,selected_at_utc=coarse.now(),selected_gamma=selected['gamma'],
        selection_reason='fine_training_grid_min_cvar_then_max_ndcg_then_min_xi' if eligible else 'no_eligible_rule_fallback_xi0',
        selection_constraints_satisfied=bool(eligible),gamma_grid=GRID,selected_oof_metrics=selected,
        all_grid_results=results,always_oof_metrics=always_metrics,
        parent_selection_file=str(parent/'SELECTED_RULE.json'),parent_selection_sha256=coarse.sha256(parent/'SELECTED_RULE.json'),
        parent_oof_file=str(data_path),parent_oof_sha256=coarse.sha256(data_path),
        runner_script_sha256=coarse.sha256(Path(__file__)),vendor_script=str(vendor_path),
        vendor_script_sha256=coarse.sha256(vendor_path),
        selection_data='Saved training OOF only; no test predictions or test outcomes read before this lock',
        exploratory_test_reuse=True)
    coarse.atomic_json(output/'SELECTED_RULE.json',rule)
    coarse.progress(output,'FINE_RULE_FROZEN',selected_gamma=selected['gamma'],eligible_candidates=len(eligible))
    coarse.predict(module,source,output,rule)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--parent',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    try:main(a)
    except Exception:
        out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
        (out/'FAILED.json').write_text(json.dumps({'traceback':traceback.format_exc()},indent=2)+'\n')
        raise
