from pathlib import Path
import argparse,json,shutil,subprocess,sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=HERE/'reproduced_evaluation')
p.add_argument('--replicates',type=int,default=10000);a=p.parse_args()
if a.output.exists():raise RuntimeError('Choose a new output directory; existing results are preserved')
locked=HERE/'locked_run'
lock=json.loads((locked/'PREDICTIONS_LOCKED.json').read_text())
rule=json.loads((locked/'SELECTED_RULE.json').read_text())
# Reproduce the training-only selection from saved OOF arrays, without test outcomes.
with np.load(HERE/'training_reference/oof_gain_downside.npz',allow_pickle=False) as z:
    always=z['always_ndcg10'];base=z['frozen_bge_ndcg10'];grid=[]
    n=len(always);tail=int(np.ceil(.1*n));always_cvar=np.sort(np.maximum(base-always,0))[-tail:].mean()
    for gamma in rule['gamma_grid']:
        accept=np.quantile(z['gain_draws']-gamma*z['downside_draws'],.05,axis=1,method='linear')>0
        values=np.where(accept,always,base);cvar=np.sort(np.maximum(base-values,0))[-tail:].mean()
        if values.mean()>=always.mean()-rule['maximum_ndcg_reduction_vs_always_oof'] and cvar<=always_cvar:
            grid.append((float(cvar),-float(values.mean()),float(gamma)))
    selected=min(grid)[2] if grid else 0.0
    assert selected==float(rule['selected_gamma'])
pred=[json.loads(s) for s in (locked/'predictions/predictions_locked.jsonl').read_text().splitlines()]
with np.load(locked/'predictions/posterior_gain_downside_draws.npz',allow_pickle=False) as z:
    lookup={str(q):i for i,q in enumerate(z['query_ids'])}
    gain=np.quantile(z['gain_draws'],.05,axis=1,method='linear')
    penalised=np.quantile(z['gain_draws']-float(rule['selected_gamma'])*z['downside_draws'],.05,axis=1,method='linear')
for row in pred:
    i=lookup[str(row['query_id'])]
    assert bool(gain[i]>0)==row['gamma0_accepted']
    assert bool(penalised[i]>0)==row['selected_accepted']
shutil.copytree(locked,a.output)
subprocess.run([sys.executable,str(HERE/'evaluate_locked.py'),'--source',str(HERE/'evaluation_source'),'--run',str(a.output),'--incumbent',str(HERE/'coarse_control'),'--replicates',str(a.replicates)],check=True)
if a.replicates==10000:
    for name,keys in [('summary.csv',['cohort','method']),('paired_ci.csv',['cohort','method','comparator','metric'])]:
        expected=pd.read_csv(HERE/'expected_results'/name).sort_values(keys).reset_index(drop=True)
        actual=pd.read_csv(a.output/'results'/name).sort_values(keys).reset_index(drop=True)
        assert expected[keys].equals(actual[keys])
        numeric=expected.select_dtypes(include=[np.number]).columns
        assert np.allclose(expected[numeric],actual[numeric],rtol=0,atol=1e-12)
print('REPRODUCTION_PASSED: training selection, utility actions, exact ranking metrics and paired cluster intervals match.')
