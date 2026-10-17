# Bayes-UQ minimal mechanism demonstration

This package runs the archived whole-ranking Bayes-UQ mechanism on synthetic cached features. It contains no restricted records, images, credentials, private corpus, trained real-data model, or complete experimental workspace. The synthetic output is **not a reproduction of any reported internal or MIRAGE result**. It must not be used to fill missing real-data measurements or confidence intervals.

## Scope

The demonstration executes training-only standardisation, the original candidate-versus-null downside-sensitive fit, a neutral-fit initialisation, the joint stabilised Gaussian posterior, 128 posterior draws, a posterior-mean complete ranking proposal, same-draw discounted utility gains, a strictly positive empirical fifth-percentile decision, canonical candidate-ID tie-breaking, and exact backbone fallback. Evaluation opens held-out qrels only after predictions and actions have been written.

The training objective, gradient, Hessian and probability functions are imported from unchanged source snapshots. The wrapper has no alternative learning objective and performs no parameter search. The development filenames in `vendor/` identify provenance, not additional manuscript methods.

## Environment and commands

Recorded environment supplied for the standalone validation: Python 3.11.11, NumPy 1.26.4, Pandas 2.3.3 and SciPy 1.15.2. These pins describe this demonstration environment; they are not a claim that all historical experiments used these versions.

```bash
python -m pip install -r requirements.txt
python run_synthetic.py --make-fixture
```

The default command creates `synthetic_fixture/` and writes `synthetic_outputs/`. It runs a small toy fit; it does not train on real experimental data. To rerun the same saved fixture:

```bash
python run_synthetic.py --fixture-dir synthetic_fixture --output-dir synthetic_outputs_repeat
```

No server connection, API key or encoder download is required. Do not invoke the archived scripts' original `main()` entry points: their original orchestration retains server paths, lock files and fixed-cohort checks. `run_synthetic.py` imports only their model, sorting, utility and evaluation functions.

## Files and source provenance

`SOURCE_PROVENANCE.json` records the exact original local paths and SHA-256 of the three verbatim source copies:

| File | Functions used |
|---|---|
| `vendor/track_f033_contextual_posterior.py` | `standardizer`, `query_design`, `fit_categorical`, `stable_seed`, `categorical_draw_probs`; the fit calls its original `objective_factory` and `numerical_hessian` |
| `vendor/run_frontier_f34_outer_once.py` | `FEATURE_NAMES`, `NULL_NAMES`, `order_from_scores`, `gain_vector`, `discounts` |
| `vendor/analyze_frontier_f34_outer_paired_v2.py` | `exact_metrics`, `method_summary` |
| `run_synthetic.py` | Synthetic fixture generation, portable input/output handling and direct application of the archived outer-run selection calculation |

The wrapper's `select_order` applies the calculation in original `run_frontier_f34_outer_once.py` lines 282–303 and 347–366: sort mean candidate probabilities, compute draw-wise gains against the frozen order, take `np.quantile(gains, 0.05, method="linear")`, accept only if the result exceeds zero. The null state participates in normalisation and fitting but never becomes a retrieved candidate. The joint draw contains the fitted tail threshold as in the original model; candidate probabilities use only the relevance-parameter coordinates.

## Cached-input interface

The toy fixture contains four CSV files and an explicit synthetic manifest. All identifiers start with `synthetic_`. There are 32 training queries and eight test queries, each with eight candidates. The model receives 12 candidate features and five query-context features; context `meta_score` is the per-query maximum of candidate metadata scores, as in the archived outer-run loader.

### `features.csv`

One row per query–candidate pair, with no relevance column:

| Columns | Input contract |
|---|---|
| `query_id`, `candidate_id` | String identifiers; candidate IDs are unique within each query and provide the canonical lexical tie-break |
| `base_score` | Finite frozen reference score, supplied independently of held-out qrels |
| `doc_cat_is_disease`, `doc_cat_is_guidance`, `doc_cat_is_phenology` | Candidate category indicators |
| `has_disease_label`, `has_stage_label` | Cached candidate metadata indicators |
| `image_rank`, `image_score`, `meta_image_consistency`, `meta_score`, `num_images`, `text_rank`, `text_score` | Finite numeric cached candidate inputs |
| `query_uncertainty_weight`, `text_top12_gap`, `top1_agree_text_meta`, `candidate_image_ratio` | Finite numeric context inputs, constant within a query |

Candidate feature order is exactly `runner.FEATURE_NAMES`. Context order is maximum `meta_score`, `query_uncertainty_weight`, `text_top12_gap`, `top1_agree_text_meta`, `candidate_image_ratio`.

The fixture generator uses arbitrary fixed toy constructions: uniform scores, reciprocal within-query ranks, randomly generated category/metadata indicators, random consistency values, a top-two text-score gap, text/metadata top-one agreement, the fraction with a nonzero toy image count, and a random ambiguity value. Its frozen backbone is `0.7 * text_score + 0.1 * meta_score`. These constructions define the synthetic example only. They do not supply or claim to reconstruct the unpublished original raw-data feature extractor.

### Other inputs

- `assignments.csv`: one `query_id,role` row per query; role is `train` or `test`.
- `train_qrels.csv`: `query_id,candidate_id,relevance` for training candidates only.
- `test_qrels.csv`: the same columns for held-out candidates; opened only after decision files are written.
- `fixture_manifest.json`: labels these files as `SYNTHETIC_INTERFACE_DEMONSTRATION_NOT_PAPER_RESULTS` and records generation settings.

Relevance is binary, with zero or one positive candidate per query. No positive denotes the null state. The toy target is generated by a fixed noisy score rule stated in the source, with predetermined null queries; it is not a clinical or agricultural annotation.

## Outputs and checks

Every summary JSON is labelled synthetic. Outputs include:

- `fit_diagnostics.json`: original optimiser diagnostics for the neutral and downside-sensitive fits, plus the fixed posterior-draw seed.
- `synthetic_model_state.npz`: toy parameters, covariance, joint draws and training-only scalers.
- `synthetic_candidate_predictions.csv`, `synthetic_actions.csv`, `synthetic_utility_draws.csv`: held-out toy probabilities, complete orders, actions and utility draws.
- `decision_order.json`: records that prediction/action files were written before opening test qrels.
- `synthetic_query_outcomes.csv`, `synthetic_evaluation.csv`: toy backbone, always-proposal and selective nDCG/MRR, plus QDR, ED, CVaR and W/T/L summaries using the archived evaluator. No real-data CI or significance claim is produced.
- `rule_fixtures.json`: independent controlled examples verifying positive-tail acceptance, positive-mean/negative-tail rejection, and zero-gain strict-boundary fallback with canonical ties.
- `run_summary.json`: observed action totals and the rule-check count. The fitted toy example is not required or tuned to exhibit both actions; its actual counts are reported unchanged.

The independent rule fixtures use controlled posterior-probability arrays, not fitted experimental predictions. They demonstrate the stated decision rule, not empirical method performance. No data or parameter search is performed to improve the toy evaluation.

## Reproducibility boundary

This package makes the mechanism independently executable without disclosing the entire private project. It is a cached-feature interface demonstration, not an end-to-end image/text preprocessing release or the full experimental reproduction bundle. Reproducing a reported real-data result additionally requires its permitted cached inputs, exact assignments and qrels, original run configuration and corresponding prediction outputs. The updated MIRAGE aggregates and their paired confidence intervals cannot be reconstructed from this synthetic fixture or from earlier-run files.
