#!/usr/bin/env python3
"""Reproduce locked F3.4 Q05 action and posterior-predictive diagnostics.

The script reads only saved predictions, posterior draws, query outcomes and
fixed qrels. It does not fit models or select/retune thresholds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

EPS = 1e-12
RISK_TAIL = 0.10
BOOT_REPS = 2000
BOOT_SEED = 20261011
PROB_BINS = [0.0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.50, 1.0000000001]


def ndcg_discount(rank: int) -> float:
    return 1.0 / math.log2(int(rank) + 1.0) if int(rank) <= 10 else 0.0


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def safe_corr(a: np.ndarray, b: np.ndarray, fn) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(fn(a, b).statistic)


def make_policy_row(dataset: str, qid: str, reachability: bool, policy: str,
                    base_order: list[str], selected_order: list[str], probs: dict,
                    delta: float, accepted, q05_gain: float, mean_gain: float,
                    posterior_negative_draw_share: float, group_id: str,
                    n_query: int) -> dict:
    ranks = {cid: rank for rank, cid in enumerate(selected_order, 1)}
    base_ranks = {cid: rank for rank, cid in enumerate(base_order, 1)}
    p_loss = p_gain = expected_gain = 0.0
    for cid, prob in probs.items():
        change = ndcg_discount(ranks[cid]) - ndcg_discount(base_ranks[cid])
        expected_gain += float(prob) * change
        if change < -EPS:
            p_loss += float(prob)
        elif change > EPS:
            p_gain += float(prob)
    p_tie = max(0.0, 1.0 - p_loss - p_gain)
    return {
        "dataset": dataset,
        "query_id": str(qid),
        "cluster_id": str(group_id),
        "reachable": bool(reachability),
        "policy": policy,
        "n_query": int(n_query),
        "observed_delta_ndcg10": float(delta),
        "observed_outcome": "loss" if delta < -EPS else ("gain" if delta > EPS else "tie"),
        "observed_loss": int(delta < -EPS),
        "predicted_posterior_predictive_loss": min(1.0, max(0.0, p_loss)),
        "predicted_posterior_predictive_gain": min(1.0, max(0.0, p_gain)),
        "predicted_posterior_predictive_tie": min(1.0, max(0.0, p_tie)),
        "posterior_predictive_expected_gain": float(expected_gain),
        "q05_accepted": accepted if policy == "F3.4 Q05" else np.nan,
        "policy_action_applied": accepted,
        "q05_gain_q05": float(q05_gain),
        "posterior_mean_utility_gain": float(mean_gain),
        "posterior_draw_share_negative_expected_gain": float(posterior_negative_draw_share),
        "actual_ranking_changed": int(selected_order != base_order),
        "model_support_note": "candidate/null model" if dataset == "internal" else ("conditional candidate-only; reachable" if reachability else "out-of-support: target absent"),
    }


def build_internal(root: Path) -> pd.DataFrame:
    run = root / "02_CANONICAL_OOF" / "raw_locked_run"
    src = root / "14_SOURCE_AND_HASHES" / "source_inputs"
    qrels = pd.read_csv(src / "QRELS_SEPARATE.csv", dtype={"query_id": str, "candidate_id": str})
    reach = qrels.groupby("query_id")["relevance"].sum().gt(0).to_dict()
    outcomes = pd.read_csv(run / "outer_query_metrics.csv", dtype={"query_id": str})
    decisions = pd.read_csv(run / "folds" / "outer_0" / "policy_decisions.csv", dtype={"query_id": str})
    decision_rows = []
    for fold in range(5):
        d = pd.read_csv(run / "folds" / ("outer_{}".format(fold)) / "policy_decisions.csv", dtype={"query_id": str})
        d["outer_fold"] = fold
        decision_rows.append(d)
    decisions = pd.concat(decision_rows, ignore_index=True)
    score_blocks, rank_blocks, post_blocks, utility_blocks = [], [], [], []
    for fold in range(5):
        fd = run / "folds" / ("outer_{}".format(fold))
        s = pd.read_csv(fd / "candidate_scores.csv", dtype={"query_id": str, "candidate_id": str})
        r = pd.read_csv(fd / "candidate_policy_ranks.csv", dtype={"query_id": str, "candidate_id": str})
        p = pd.read_csv(fd / "posterior_draw_candidate_probabilities.csv.gz", dtype={"query_id": str, "candidate_id": str})
        u = pd.read_csv(fd / "posterior_utility_draws.csv", dtype={"query_id": str})
        score_blocks.append(s[s["outer_fold"].eq(fold)])
        rank_blocks.append(r[r["outer_fold"].eq(fold)])
        post_blocks.append(p[(p["outer_fold"].eq(fold)) & (p["rho"].eq(2))])
        utility_blocks.append(u[(u["outer_fold"].eq(fold)) & (u["rho"].eq(2))])
    scores = pd.concat(score_blocks, ignore_index=True)
    ranks = pd.concat(rank_blocks, ignore_index=True)
    post = pd.concat(post_blocks, ignore_index=True)
    utilities = pd.concat(utility_blocks, ignore_index=True)
    policy_map = {
        "RHO2_ALWAYS_BMA": "Always posterior-mean",
        "RHO2_POSTERIOR_MEAN_GATE": "Posterior mean-gain gate",
        "RHO2_MAP_GAIN_GATE": "MAP-gain gate",
        "RHO2_LCB05_GATE": "F3.4 Q05",
    }
    result = []
    for (fold, qid), s in scores.groupby(["outer_fold", "query_id"], sort=True):
        s = s.sort_values("candidate_id", kind="mergesort")
        ids = s["candidate_id"].astype(str).tolist()
        base_order = s.sort_values(["base_rank", "candidate_id"])["candidate_id"].astype(str).tolist()
        probs = dict(zip(s["candidate_id"].astype(str), s["rho2_bma_probability"].astype(float)))
        draw = utilities[(utilities["outer_fold"].eq(fold)) & utilities["query_id"].eq(qid)]["expected_ndcg_gain"].to_numpy(float)
        mean_gain = float(draw.mean())
        neg_share = float(np.mean(draw < 0.0))
        for policy_id, policy_name in policy_map.items():
            sel = ranks[(ranks["outer_fold"].eq(fold)) & ranks["query_id"].eq(qid) & ranks["policy"].eq(policy_id)]
            selected_order = sel.sort_values("selected_rank")["candidate_id"].astype(str).tolist()
            out = outcomes[(outcomes["outer_fold"].eq(fold)) & outcomes["query_id"].eq(qid) & outcomes["policy"].eq(policy_id)]
            if len(out) != 1:
                raise ValueError("Expected one exact outcome for {} {}".format(qid, policy_id))
            d = decisions[(decisions["outer_fold"].eq(fold)) & decisions["query_id"].eq(qid) & decisions["policy"].eq(policy_id)]
            if len(d) != 1:
                raise ValueError("Expected one decision for {} {}".format(qid, policy_id))
            action = int(d.iloc[0]["gate_applied"]) if policy_id != "RHO2_ALWAYS_BMA" else 1
            result.append(make_policy_row(
                "internal", qid, reach.get(qid, False), policy_name, base_order, selected_order, probs,
                float(out.iloc[0]["delta_ndcg10"]), action, float(d.iloc[0]["posterior_lcb05_gain"]),
                mean_gain, neg_share, qid, len(scores[scores["query_id"].eq(qid)])))
    frame = pd.DataFrame(result)
    if frame["query_id"].nunique() != 483 or set(frame["policy"].unique()) != set(policy_map.values()):
        raise ValueError("Internal Q05 records do not cover the registered cohort/policies")
    return frame


def build_mirage(revision: Path, experiment: Path) -> pd.DataFrame:
    result_root = revision / "04_MIRAGE_RESULTS" / "results"
    server = experiment / "server_results"
    scores = pd.read_csv(server / "predictions" / "candidate_scores_locked.csv.gz", dtype={"query_id": str, "candidate_id": str})
    gains = pd.read_csv(server / "predictions" / "q05_posterior_gain_draws.csv.gz", dtype={"query_id": str})
    outcomes = pd.read_csv(result_root / "query_level_outcomes.csv", dtype={"query_id": str})
    with (server / "predictions" / "predictions_locked.jsonl").open("r", encoding="utf-8") as stream:
        locked = {str(row["query_id"]): row for row in (json.loads(line) for line in stream)}
    method_fields = {
        "Always posterior-mean": "always_posterior_mean",
        "F3.4 Q05": "enhanced_f34_q05",
        "F3.4 rho=0 MAP": "f34_rho0_map",
        "F3.4 rho=0 posterior-mean": "f34_rho0_posterior_mean",
    }
    row_lookup = outcomes.set_index("query_id")
    result = []
    for qid, s in scores.groupby("query_id", sort=True):
        s = s.sort_values("candidate_id", kind="mergesort")
        rec = row_lookup.loc[qid]
        base_order = locked[qid]["orders"]["Frozen BGE"]
        draw = gains.loc[gains["query_id"].eq(qid), "posterior_utility_gain"].to_numpy(float)
        mean_gain = float(draw.mean())
        neg_share = float(np.mean(draw < 0.0))
        accepted = int(bool(rec["q05_accepted"]))
        for policy, prob_field in method_fields.items():
            if policy == "Always posterior-mean":
                selected_order = locked[qid]["orders"]["Always posterior-mean"]
                probs = dict(zip(s["candidate_id"].astype(str), s[prob_field].astype(float)))
                delta = float(rec["always_posterior_mean_ndcg10"] - rec["frozen_bge_ndcg10"])
                action = 1
            elif policy == "F3.4 Q05":
                selected_order = locked[qid]["orders"]["Enhanced F3.4 + Q05"]
                probs = dict(zip(s["candidate_id"].astype(str), s["always_posterior_mean"].astype(float)))
                delta = float(rec["enhanced_f34_q05_ndcg10"] - rec["frozen_bge_ndcg10"])
                action = accepted
            else:
                selected_order = locked[qid]["orders"][policy]
                probs = dict(zip(s["candidate_id"].astype(str), s[prob_field].astype(float)))
                if policy == "F3.4 rho=0 MAP":
                    delta = float(rec["f34_rho0_map_ndcg10"] - rec["frozen_bge_ndcg10"])
                else:
                    delta = float(rec["f34_rho0_posterior_mean_ndcg10"] - rec["frozen_bge_ndcg10"])
                action = 1
            result.append(make_policy_row(
                "mirage", qid, bool(rec["target_reachable_at_50"]), policy, base_order,
                [str(x) for x in selected_order], probs, delta, action,
                float(rec["q05_gain_q05"]), mean_gain, neg_share,
                str(rec["source_page_offset"]), len(s)))
    frame = pd.DataFrame(result)
    if frame["query_id"].nunique() != 2000 or set(frame["policy"].unique()) != set(method_fields):
        raise ValueError("MIRAGE Q05 records do not cover fixed test cohort/methods")
    return frame


def summary_metrics(block: pd.DataFrame, scope: str) -> dict:
    y = block["observed_loss"].to_numpy(int)
    p = block["predicted_posterior_predictive_loss"].to_numpy(float)
    d = block["observed_delta_ndcg10"].to_numpy(float)
    loss = np.maximum(-d, 0.0)
    tail_n = max(1, int(math.ceil(RISK_TAIL * len(d))))
    brier = float(np.mean((p - y) ** 2))
    prevalence = float(np.mean(y))
    mean_baseline_brier = prevalence * (1 - prevalence)
    q05_actions = block["q05_accepted"].dropna()
    action_rate = float(block["policy_action_applied"].mean())
    return {
        "dataset": str(block["dataset"].iloc[0]), "scope": scope,
        "policy": str(block["policy"].iloc[0]), "n": int(len(block)),
        "reachable_n": int(block["reachable"].sum()), "observed_losses": int(y.sum()),
        "observed_loss_rate": prevalence,
        "posterior_predictive_loss_probability_mean": float(np.mean(p)),
        "posterior_predictive_loss_probability_median": float(np.median(p)),
        "loss_probability_brier": brier,
        "prevalence_only_brier": mean_baseline_brier,
        "brier_skill_vs_prevalence": (1.0 - brier / mean_baseline_brier) if mean_baseline_brier > 0 else float("nan"),
        "loss_probability_auc": safe_auc(y, p),
        "predicted_vs_observed_delta_pearson": safe_corr(block["posterior_predictive_expected_gain"].to_numpy(float), d, pearsonr),
        "predicted_vs_observed_delta_spearman": safe_corr(block["posterior_predictive_expected_gain"].to_numpy(float), d, spearmanr),
        "posterior_expected_gain_mean": float(block["posterior_mean_utility_gain"].mean()),
        "observed_delta_mean": float(np.mean(d)),
        "posterior_draw_negative_gain_share_mean": float(block["posterior_draw_share_negative_expected_gain"].mean()),
        "mean_observed_delta_ndcg10": float(np.mean(d)),
        "QDR": float(np.mean(d < -EPS)), "ED": float(np.mean(loss)),
        "CVaR10": float(np.sort(loss)[-tail_n:].mean()),
        "W": int(np.sum(d > EPS)), "T": int(np.sum(np.abs(d) <= EPS)), "L": int(np.sum(d < -EPS)),
        "q05_accept_count": int(np.nansum(q05_actions)) if len(q05_actions) else None,
        "q05_accept_rate": float(np.nanmean(q05_actions)) if len(q05_actions) else float("nan"),
        "fallback_count": int(np.sum(q05_actions == 0)) if len(q05_actions) else None,
        "policy_action_applied_rate": action_rate,
        "actual_ranking_change_rate": float(block["actual_ranking_changed"].mean()),
        "Q05_rejected_observed_losses": int(np.sum((block["q05_accepted"] == 0) & (d < -EPS))) if block["q05_accepted"].notna().any() else None,
        "Q05_rejected_observed_gains": int(np.sum((block["q05_accepted"] == 0) & (d > EPS))) if block["q05_accepted"].notna().any() else None,
        "q05_accepted_observed_losses": int(np.sum((block["q05_accepted"] == 1) & (d < -EPS))) if block["q05_accepted"].notna().any() else None,
    }


def calibration_bins(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, policy, scope), block in frame.groupby(["dataset", "policy", "calibration_scope"], sort=True):
        p = block["predicted_posterior_predictive_loss"].to_numpy(float)
        y = block["observed_loss"].to_numpy(int)
        b = pd.cut(p, bins=PROB_BINS, include_lowest=True, right=False)
        temp = pd.DataFrame({"bin": b.astype(str), "p": p, "y": y})
        for label, part in temp.groupby("bin", sort=False):
            if len(part) == 0:
                continue
            rows.append({"dataset": dataset, "policy": policy, "scope": scope, "predicted_probability_bin": label,
                         "n": int(len(part)), "mean_predicted_loss_probability": float(part["p"].mean()),
                         "observed_loss_rate": float(part["y"].mean()), "calibration_gap_observed_minus_predicted": float(part["y"].mean() - part["p"].mean())})
    return pd.DataFrame(rows)


def bootstrap_calibration(frame: pd.DataFrame, reps: int = BOOT_REPS, seed: int = BOOT_SEED) -> pd.DataFrame:
    rows = []
    for (dataset, policy, scope), block in frame.groupby(["dataset", "policy", "calibration_scope"], sort=True):
        if scope != "calibration" or len(block) < 30:
            continue
        y = block["observed_loss"].to_numpy(int)
        p = block["predicted_posterior_predictive_loss"].to_numpy(float)
        clusters = block["cluster_id"].astype(str).to_numpy()
        unique = np.unique(clusters)
        row_positions = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique}
        stable_key = "{}|{}".format(dataset, policy).encode("utf-8")
        key_seed = int(hashlib.sha256(stable_key).hexdigest()[:8], 16)
        rng = np.random.Generator(np.random.PCG64(seed + key_seed))
        point_values = {
            "brier": float(np.mean((y - p) ** 2)),
            "auc": safe_auc(y, p),
            "mean_predicted": float(p.mean()),
            "observed_rate": float(y.mean()),
        }
        values = {"brier": [], "auc": [], "mean_predicted": [], "observed_rate": []}
        for _ in range(reps):
            sampled = rng.choice(unique, size=len(unique), replace=True)
            ix = np.concatenate([row_positions[c] for c in sampled])
            if len(ix) == 0:
                continue
            yy, pp = y[ix], p[ix]
            values["brier"].append(float(np.mean((yy - pp) ** 2)))
            values["auc"].append(safe_auc(yy, pp))
            values["mean_predicted"].append(float(pp.mean()))
            values["observed_rate"].append(float(yy.mean()))
        for metric, vals in values.items():
            arr = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
            if len(arr):
                rows.append({"dataset": dataset, "policy": policy, "scope": scope, "metric": metric,
                             "estimate": float(point_values[metric]), "bootstrap_mean": float(np.mean(arr)),
                             "ci95_low": float(np.quantile(arr, 0.025)),
                             "ci95_high": float(np.quantile(arr, 0.975)), "bootstrap_replicates": int(len(arr)),
                             "resampling_unit": "query_id" if dataset == "internal" else "source_page_offset",
                             "seed": seed})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision-root", default=r"E:\tmp\APIN_D_26_10051_MIRAGE_INTEGRATED_REVISION")
    parser.add_argument("--f34-root", default=r"E:\tmp\experiments\apin_revision_f3_4_main_method_evidence")
    parser.add_argument("--mirage-root", default=r"E:\tmp\experiments\APIN_R3_3_EXTERNAL_VALIDATION_RECOVERY\05_NEW_EXTERNAL_EXPERIMENT\mirage_mmst")
    parser.add_argument("--out", default=r"E:\APIN_FINAL_SCIENTIFIC_CLOSURE\data")
    parser.add_argument("--bootstrap-reps", type=int, default=BOOT_REPS)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    internal = build_internal(Path(args.f34_root))
    mirage = build_mirage(Path(args.revision_root), Path(args.mirage_root))
    all_rows = pd.concat([internal, mirage], ignore_index=True)
    all_rows["calibration_scope"] = np.where(
        (all_rows["dataset"].eq("mirage") & ~all_rows["reachable"]), "outside_conditional_model_support", "calibration")
    # MIRAGE candidate-only task adaptation cannot represent an unreachable target.
    all_rows.to_csv(out / "F34_Q05_QUERY_LEVEL_DIAGNOSTICS.csv", index=False)
    summaries = []
    for keys, block in all_rows.groupby(["dataset", "policy"], sort=True):
        summaries.append(summary_metrics(block, "all recorded queries"))
        if block["reachable"].sum() > 0:
            summaries.append(summary_metrics(block[block["reachable"]], "reachable subset"))
        if (~block["reachable"]).sum() > 0:
            summaries.append(summary_metrics(block[~block["reachable"]], "unreachable subset"))
    pd.DataFrame(summaries).to_csv(out / "F34_Q05_RELIABILITY_SUMMARY.csv", index=False)
    bins = calibration_bins(all_rows[all_rows["calibration_scope"].eq("calibration")])
    bins.to_csv(out / "F34_Q05_PREDICTIVE_LOSS_CALIBRATION_BINS.csv", index=False)
    calibration_rows = []
    calibration_data = all_rows[all_rows["calibration_scope"].eq("calibration")]
    for (dataset, policy), block in calibration_data.groupby(["dataset", "policy"], sort=True):
        y = block["observed_loss"].to_numpy(int)
        p = block["predicted_posterior_predictive_loss"].to_numpy(float)
        bin_ids = pd.cut(p, bins=PROB_BINS, include_lowest=True, right=False)
        temp = pd.DataFrame({"bin": bin_ids.astype(str), "p": p, "y": y})
        weighted_abs_gap = 0.0
        occupied = 0
        for _, part in temp.groupby("bin", sort=False):
            if len(part):
                weighted_abs_gap += len(part) * abs(float(part["y"].mean() - part["p"].mean()))
                occupied += 1
        prevalence = float(y.mean())
        brier = float(np.mean((p - y) ** 2))
        calibration_rows.append({
            "dataset": dataset, "policy": policy, "scope": "supported calibration subset",
            "n_queries": len(block), "observed_losses": int(y.sum()), "observed_loss_rate": prevalence,
            "mean_predicted_loss_probability": float(p.mean()),
            "predicted_to_observed_rate_ratio": float(p.mean() / prevalence) if prevalence > 0 else np.nan,
            "brier": brier, "prevalence_only_brier": prevalence * (1 - prevalence),
            "brier_skill_vs_prevalence": float(1 - brier / (prevalence * (1 - prevalence))) if prevalence > 0 else np.nan,
            "auc": safe_auc(y, p), "ece_fixed_probability_bins": weighted_abs_gap / len(block),
            "occupied_fixed_bins": occupied, "fixed_probability_bins": json.dumps(PROB_BINS),
        })
    pd.DataFrame(calibration_rows).to_csv(out / "F34_Q05_POSTERIOR_PREDICTIVE_CALIBRATION.csv", index=False, float_format="%.12g")
    boot = bootstrap_calibration(all_rows, reps=args.bootstrap_reps)
    boot.to_csv(out / "F34_Q05_CALIBRATION_BOOTSTRAP_CI.csv", index=False)
    decision_rows = []
    for dataset, block in all_rows[all_rows["policy"].isin(["F3.4 Q05", "Always posterior-mean"])].groupby("dataset", sort=True):
        q = block[block["policy"].eq("F3.4 Q05")].set_index("query_id")
        a = block[block["policy"].eq("Always posterior-mean")].set_index("query_id")
        same = q.index.intersection(a.index)
        for label, df in [("Q05", q.loc[same]), ("Always posterior-mean", a.loc[same])]:
            if label == "Always posterior-mean":
                continue
            for reach_label, subset in [("all", df), ("reachable", df[df["reachable"]])]:
                for outcome in ["gain", "tie", "loss"]:
                    part = subset[subset["observed_outcome"].eq(outcome)]
                    if len(part):
                        decision_rows.append({"dataset": dataset, "scope": reach_label, "observed_outcome": outcome,
                                              "n": int(len(part)), "q05_accepted": int((part["q05_accepted"] == 1).sum()),
                                              "q05_rejected": int((part["q05_accepted"] == 0).sum()),
                                              "mean_q05_lower_quantile": float(part["q05_gain_q05"].mean()),
                                              "mean_posterior_predictive_loss_probability": float(part["predicted_posterior_predictive_loss"].mean())})
    pd.DataFrame(decision_rows).to_csv(out / "F34_Q05_ACTION_BY_OBSERVED_OUTCOME.csv", index=False)
    cross_rows = []
    for dataset in sorted(all_rows["dataset"].unique()):
        q05 = all_rows[(all_rows["dataset"].eq(dataset)) & all_rows["policy"].eq("F3.4 Q05")].set_index("query_id")
        always = all_rows[(all_rows["dataset"].eq(dataset)) & all_rows["policy"].eq("Always posterior-mean")].set_index("query_id")
        joined = q05[["reachable", "q05_accepted", "observed_outcome"]].join(
            always[["observed_outcome"]].rename(columns={"observed_outcome": "always_observed_outcome"}), how="inner")
        for scope_name, subset in [("all", joined), ("reachable", joined[joined["reachable"]])]:
            for outcome in ["gain", "tie", "loss"]:
                part = subset[subset["always_observed_outcome"].eq(outcome)]
                if len(part):
                    cross_rows.append({
                        "dataset": dataset, "scope": scope_name, "always_observed_outcome": outcome,
                        "n": int(len(part)), "q05_accepted": int((part["q05_accepted"] == 1).sum()),
                        "q05_fallback": int((part["q05_accepted"] == 0).sum()),
                        "q05_observed_gain_after_action": int((part["observed_outcome"].eq("gain")).sum()),
                        "q05_observed_tie_after_action": int((part["observed_outcome"].eq("tie")).sum()),
                        "q05_observed_loss_after_action": int((part["observed_outcome"].eq("loss")).sum()),
                    })
    pd.DataFrame(cross_rows).to_csv(out / "F34_Q05_ACCEPTANCE_VS_ALWAYS_OUTCOME.csv", index=False)
    manifest = {
        "status": "COMPLETED_DIAGNOSTIC_ONLY",
        "script": str(Path(__file__).resolve()),
        "threshold_changed": False,
        "test_outcomes_used_for_threshold_selection": False,
        "internal_query_count": int(internal["query_id"].nunique()),
        "mirage_query_count": int(mirage["query_id"].nunique()),
        "mirage_unreachable_queries_outside_candidate_only_model_support": int((~mirage.drop_duplicates("query_id")["reachable"]).sum()),
        "predictive_probability_definition": "Posterior-mean categorical mass on candidates whose selected rank has lower nDCG@10 discount than the frozen backbone; null state contributes zero. MIRAGE is conditional candidate-only and calibration is restricted to target-reachable queries.",
        "Q05_definition": "Strictly positive empirical 5th percentile of 128 posterior expected-utility-gain draws, inherited from the frozen protocol.",
        "bootstrap_replicates": int(args.bootstrap_reps),
        "bootstrap_seed": BOOT_SEED,
        "files": [p.name for p in sorted(out.glob("F34_Q05_*.csv"))],
    }
    (out / "F34_Q05_DIAGNOSTIC_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
