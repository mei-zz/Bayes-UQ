# Bayes-UQ: Posterior-Utility Selective Reranking for Agricultural Knowledge Retrieval

## Project overview

This repository contains the reproducibility materials for the manuscript **“Bayes-UQ: Posterior-Utility Selective Reranking for Agricultural Knowledge Retrieval”** (Applied Intelligence, manuscript APIN-D-26-10051).

Bayes-UQ treats reranking as a query-level decision over a complete ranking. A candidate-versus-null relevance model produces a stabilised approximate posterior; the posterior-mean candidate probabilities define a complete ranking proposal; draw-wise reference-relative discounted utility is evaluated under the posterior draws; and a lower-tail acceptance rule either applies the complete proposal or returns the complete frozen backbone order exactly. Decision utility is gain alone internally and gain minus a training-selected expected-demotion penalty in the MIRAGE adaptation.

The bounded-residual controller that appears in the supporting analyses is a separate score-level control. It is not a second proposed method and is not an ablation of the whole-ranking acceptance rule.

## Main contribution

- A whole-ranking posterior-utility decision that evaluates whether a complete proposal should replace a frozen reference order.
- A lower-tail acceptance rule (empirical 5% quantile of draw-wise decision utility, strict `> 0`) with exact complete-order fallback.
- Matched policy, risk-weight, null-state and decision-variance controls that separate proposal construction from the selective action.
- An exploratory task adaptation to MIRAGE expert-response retrieval with training-only penalty selection, evaluated against Always, a coarse-grid control, gain-only, pointwise logistic and Adapted EAU.

## Reproduction scope

### Supported by this repository

- The synthetic candidate/null demonstration: fitting interface, posterior draws, posterior-mean proposal, 128-draw utility, strict Q05 acceptance, canonical tie-breaking and exact fallback.
- MIRAGE training-side coefficient selection from the supplied out-of-fold (OOF) gain/downside arrays: 105-point grid, the two training constraints, 59 eligible configurations and the selected `xi = 0.27`; the coarse control `xi = 0.25` is supplied as a frozen incumbent.
- Re-evaluation of the locked MIRAGE rankings and actions for all 2,000 test queries against the supplied candidate-ID qrels.
- All reported MIRAGE metrics, accept/fallback counts, W/T/L, loss-interception counts and 10,000-resample paired source-page-cluster percentile intervals.
- Reproduction of the synthetic decision fixtures, including positive-mean-negative-tail fallback and the strict zero boundary.

### Not reproducible from this repository alone

- Full retraining of the internal 483-query Bayes-UQ model. The internal features, predictions, qrels and OOF utility arrays are restricted and are not distributed.
- Feature-level reconstruction of the internal candidate/context features. The exact construction of several internal fields is not specified by the archived feature scripts.
- The bounded-residual supporting analyses, null-state ablation and internal utility-variance replays. Their source is under `supporting_analysis/` for provenance, but the required saved internal inputs are not included.
- Full MIRAGE model refitting. The cached training/test feature matrices and the full-fit checkpoint are not included.
- Any result that requires additional restricted records or checkpoints.

Reproduction in this repository means recomputation of reported metrics and selective decisions from supplied saved predictions, not independent retraining from raw data.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── run_reproduction.py                  # MIRAGE saved-output reproduction entry point
├── evaluate_locked.py                   # locked-prediction evaluator and cluster bootstrap
├── synthetic_core/
│   ├── run_synthetic.py                 # synthetic candidate/null demonstration
│   ├── vendor/                          # unchanged archived model/sorting/evaluation functions
│   └── README.md
├── locked_run/                          # selected xi=0.27 predictions, actions, locks, draws
├── coarse_control/                      # frozen xi=0.25 incumbent predictions and locks
├── evaluation_source/
│   ├── evaluator_only/test_qrels.csv    # candidate-ID relevance labels
│   └── inputs/candidate_pools_test.csv  # frozen pool IDs and cached text scores
├── training_reference/                  # OOF arrays, training-grid results, selection source
├── expected_results/                    # expected summary, actions, intervals, per-query outcomes
└── supporting_analysis/                 # archived source for analyses requiring absent inputs
```

## Requirements

- Python 3.10.15 tested in a clean environment; the recorded development environment is Python 3.11.11.
- `numpy==1.26.4`
- `pandas==2.3.3`
- `scipy==1.15.2`
- CPU only. No GPU, model download, API key or network access is required.

The pinned versions were installed from `requirements.txt` in a fresh virtual environment during release validation.

## Installation

```bash
git clone https://github.com/mei-zz/Bayes-UQ.git
cd Bayes-UQ
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
# .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Synthetic demonstration

```bash
python synthetic_core/run_synthetic.py --make-fixture
```

Expected terminal status: `SYNTHETIC_INTERFACE_DEMONSTRATION_NOT_PAPER_RESULTS`, with `accepted: 3`, `fallback: 5`, `rule_fixtures_passed: 3` (about 6 s on CPU).

Outputs: `synthetic_core/synthetic_fixture/` and `synthetic_core/synthetic_outputs/`. The synthetic data and outputs are explicitly toy examples and do not reproduce any reported paper metric.

To rerun the same fixture:

```bash
python synthetic_core/run_synthetic.py --fixture-dir synthetic_core/synthetic_fixture --output-dir synthetic_core/synthetic_outputs_repeat
```

## MIRAGE saved-output reproduction

```bash
python run_reproduction.py --output reproduced_evaluation
```

The command:

1. reproduces the training-only 105-point coefficient selection from `training_reference/oof_gain_downside.npz`;
2. checks the posterior gain/downside arrays and both decision policies for all 2,000 test queries;
3. evaluates every method against the supplied candidate-ID qrels;
4. repeats the 10,000-resample paired 20-source-page-cluster bootstrap;
5. compares the summary and paired intervals with `expected_results` at tolerance `1e-12`.

A successful run ends with:

```text
REPRODUCTION_PASSED: training selection, utility actions, exact ranking metrics and paired cluster intervals match.
```

Outputs: `reproduced_evaluation/` (lock-file copies, `results/summary.csv`, `paired_ci.csv`, `absolute_ci.csv`, `action_diagnostics.csv`, `query_outcomes.csv`, `cluster_bootstrap_replicates.npz`, and the `EVALUATION_*.json` records). Measured runtime: about 15–22 s on CPU.

## Expected results

All-cohort MIRAGE results for the selected `xi = 0.27` policy:

| Metric | Value |
|---|---|
| N | 2,000 |
| Reachable | 1,407 |
| nDCG@10 | 0.522297 |
| MRR@10 | 0.487242 |
| Recall@10 | 0.633500 |
| QDR | 0.020500 |
| ED | 0.004502 |
| CVaR$_{10\%}$ | 0.045021 |
| Accepted / fallback | 1,790 / 210 |
| W/T/L vs Frozen BGE | 358 / 1601 / 41 |
| Loss interception | 13 / 54 (24.07%) |
| Rejected improvements | 45 |

Paired contrasts (nominal 95% percentile intervals, 10,000 source-page-cluster resamples):

| Contrast | Metric | Difference | 95% CI |
|---|---|---|---|
| vs Always posterior-mean | nDCG@10 | −0.004573 | [−0.007289, −0.002165] |
| vs Always posterior-mean | CVaR$_{10\%}$ | −0.015170 | [−0.025516, −0.006451] |
| vs Frozen BGE | nDCG@10 | +0.056074 | [0.043629, 0.066964] |

The complete expected tables are in `expected_results/`.

## Statistical scope

- Posterior decision draws: 128 per query.
- Acceptance statistic: `np.quantile(gain - xi * downside, 0.05, method="linear") > 0`; with 128 draws the zero-based interpolation position is `0.05 * (128 - 1) = 6.35`.
- Fallback: the complete canonically tie-broken Frozen BGE order, not a partial or top-1 fallback.
- MIRAGE selection: 1,000 training queries, five page-grouped OOF folds, 50 training page clusters, 105 candidates, 59 eligible; selected `xi = 0.27`; coarse control `xi = 0.25`.
- Bootstrap: 10,000 resamples of 20 test source-page clusters, PCG64 seed 20261015, with the zero-inclusive CVaR tail recomputed in every replicate.
- Intervals are nominal percentile intervals conditional on the fixed predictions and training-side coefficient selection. They exclude refitting/selection uncertainty and are not multiplicity-adjusted or realised-risk guarantees.

## Limitations

- MIRAGE is an exploratory task adaptation. The coarse-grid rule was evaluated on the test cohort before the 105-point training-grid refinement; the refined result is not an independent confirmatory evaluation.
- The selected `xi = 0.27` and `xi = 0.275` tie on the two training objectives; the pre-declared minimum-`xi` rule selects 0.27.
- Test labels are opened only after the prediction lock. The absence of test-label use during upstream prediction generation is supported by the recorded provenance and SHA-256 locks, not independently provable from this repository.
- Internal 483-query results, the null-state ablation, bounded-residual analyses and internal utility-variance replays cannot be recomputed from the public inputs.
- The archived scripts under `supporting_analysis/` and `training_reference/vendor/` are included for source provenance. They require saved inputs or a Torch environment that are not part of this repository, and are not needed for the two commands above.

## Data availability and licensing

- **Code license:** no license is granted. All rights reserved. The repository is public for review and verification of the reported results; reuse or redistribution of the code requires permission from the authors.
- **MIRAGE-derived evaluation artifacts:** the authors have confirmed that the included derived evaluation artifacts (candidate-ID qrels, frozen candidate-pool IDs and cached text scores, locked rankings/actions, posterior utility arrays and statistical outputs) may be redistributed in this repository. No consultation images, expert-response text or other MIRAGE source content is included. Users must obtain the upstream MIRAGE dataset under its own terms and cite the original MIRAGE work.
- **Internal agricultural data:** restricted. Access to internal records and permitted query-level outputs requires corresponding-author approval, institutional permission and a data-use agreement. No internal 483-query records or features are included.
- The absolute `/home/ubuntu/...` paths and server hostname strings in some provenance JSON files are historical execution metadata, not runtime dependencies. Both commands resolve every input relative to the repository root.

## Reproducibility version

- Release: `v1.0.0`
- Repository: https://github.com/mei-zz/Bayes-UQ
- The release commit SHA and release URL are recorded on the GitHub Releases page.
- Validated environments: Python 3.10.15 with NumPy 1.26.4, Pandas 2.3.3 and SciPy 1.15.2 in a fresh virtual environment. The synthetic and MIRAGE workflows both passed from a fresh clone of this repository.

## Citation

If you use this code, please cite:

> Gang Mei, Qian Gao, Yutong Deng, Hongchun Qu. Bayes-UQ: Posterior-Utility Selective Reranking for Agricultural Knowledge Retrieval. Applied Intelligence, manuscript APIN-D-26-10051.

The MIRAGE dataset should be cited according to its original publication and repository terms.
