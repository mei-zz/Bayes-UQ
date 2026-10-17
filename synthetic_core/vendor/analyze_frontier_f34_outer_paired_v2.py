#!/usr/bin/env python3
"""Post-lock paired analysis of Frontier-F3.4 against fixed policies and R3-2 OOF."""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


REMOTE_ROOT = Path("/home/ubuntu/apin_revision_bayes_uq_frontier")
ROOT = REMOTE_ROOT if REMOTE_ROOT.exists() else Path(__file__).resolve().parent.parent
INPUTS = ROOT / "03_DATA_AND_PROTOCOL" / "inputs"
RUN = ROOT / "13_OUTER_OOF" / "frontier_f34_once"
LOCK_CONFIG = ROOT / "09_TRACK_F_AGRICULTURAL_ROBUSTNESS" / "v3_4" / "OUTER_EVALUATION_CONFIG.json"
R3_2_ORIGINAL = ROOT.parent / "apin_revision_r3_2_strong_baselines" / "predictions" / "R3_2_CANDIDATE_OOF_SCORES.csv"
R3_2_SCORES = RUN / "source_snapshots" / "R3_2_CANDIDATE_OOF_SCORES.csv"
REPS = 2000
SEED = 20261008
MAIN_POLICY = "RHO2_LCB05_GATE"
POLICIES = (
    "FROZEN_BACKBONE", "RHO2_ALWAYS_BMA", "RHO2_POSTERIOR_MEAN_GATE",
    "RHO2_MAP_GAIN_GATE", "RHO2_LCB05_GATE", "RHO0_LCB05_GATE", "RHO0_ALWAYS_BMA",
)
METRICS = ("ndcg10", "mrr10", "qdr", "ed", "cvar10", "worst_decline")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, float_format="%.17g")
    os.replace(tmp, path)


def d10(rank: int) -> float:
    return 1.0 / math.log2(rank + 1.0) if 1 <= rank <= 10 else 0.0


def exact_metrics(order: list[str], qrel: dict[str, int]) -> tuple[float, float]:
    positives = [cid for cid, rel in qrel.items() if rel > 0]
    if len(positives) > 1:
        raise ValueError("More than one positive qrel in query")
    if not positives:
        return 0.0, 0.0
    rank = order.index(positives[0]) + 1
    return d10(rank), (1.0 / rank if rank <= 10 else 0.0)


def method_summary(method: str, rows: list[dict]) -> dict:
    delta = np.asarray([x["delta_ndcg10"] for x in rows], dtype=float)
    loss = np.maximum(-delta, 0.0)
    n = len(rows)
    tail = max(1, int(math.ceil(0.10 * n)))
    wins, downs = int(np.sum(delta > 1e-12)), int(np.sum(delta < -1e-12))
    return {
        "method": method, "unique_queries": n,
        "ndcg10": float(np.mean([x["ndcg10"] for x in rows])),
        "mrr10": float(np.mean([x["mrr10"] for x in rows])),
        "qdr_count": downs, "qdr": float(np.mean(delta < -1e-12)),
        "ed": float(loss.mean()), "cvar10_tail_queries": tail,
        "cvar10": float(np.sort(loss)[-tail:].mean()),
        "worst_decline": float(loss.max()),
        "wtl": "{}/{}/{}".format(wins, n - wins - downs, downs),
    }


def metric_values(rows: list[dict], metric: str, indices: np.ndarray | None = None) -> float:
    if indices is not None:
        sample = [rows[int(i)] for i in indices]
    else:
        sample = rows
    delta = np.asarray([x["delta_ndcg10"] for x in sample], dtype=float)
    loss = np.maximum(-delta, 0.0)
    if metric == "ndcg10":
        return float(np.mean([x["ndcg10"] for x in sample]))
    if metric == "mrr10":
        return float(np.mean([x["mrr10"] for x in sample]))
    if metric == "qdr":
        return float(np.mean(delta < -1e-12))
    if metric == "ed":
        return float(loss.mean())
    if metric == "cvar10":
        tail = max(1, int(math.ceil(0.10 * len(sample))))
        return float(np.sort(loss)[-tail:].mean())
    if metric == "worst_decline":
        return float(loss.max())
    raise KeyError(metric)


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(np.asarray(pvalues, dtype=float))
    out = np.ones(len(pvalues), dtype=float)
    current = 0.0
    for rank, idx in enumerate(order):
        current = max(current, (len(pvalues) - rank) * float(pvalues[idx]))
        out[int(idx)] = min(1.0, current)
    return out.tolist()


def paired_bootstrap(a: list[dict], b: list[dict], rng: np.random.Generator) -> dict:
    if [r["query_id"] for r in a] != [r["query_id"] for r in b]:
        raise ValueError("Paired samples are not aligned by query ID")
    n = len(a)
    point = {m: metric_values(a, m) - metric_values(b, m) for m in METRICS}
    boot = {m: np.empty(REPS, dtype=float) for m in METRICS}
    for rep in range(REPS):
        idx = rng.integers(0, n, size=n)
        for metric in METRICS:
            boot[metric][rep] = metric_values(a, metric, idx) - metric_values(b, metric, idx)
    out = {}
    for metric in METRICS:
        vals = boot[metric]
        lo, hi = np.quantile(vals, [0.025, 0.975])
        centered = vals - point[metric]
        lower_tail = (np.sum(centered <= -abs(point[metric])) + 1.0) / (REPS + 1.0)
        upper_tail = (np.sum(centered >= abs(point[metric])) + 1.0) / (REPS + 1.0)
        p = min(1.0, 2.0 * min(lower_tail, upper_tail))
        out[metric] = {"difference": point[metric], "ci95_low": float(lo), "ci95_high": float(hi), "p_raw": float(p)}
    da = np.asarray([r["delta_ndcg10"] for r in a])
    db = np.asarray([r["delta_ndcg10"] for r in b])
    out["qdr_discordant"] = {
        "a_degraded_b_not": int(np.sum((da < -1e-12) & (db >= -1e-12))),
        "a_not_degraded_b": int(np.sum((da >= -1e-12) & (db < -1e-12))),
    }
    return out


def main() -> None:
    validation_path = RUN / "independent_validation_report.json"
    evaluation_path = RUN / "evaluation_manifest.json"
    lock_path = RUN / "PREDICTIONS_LOCKED.json"
    for p in (validation_path, evaluation_path, lock_path, R3_2_SCORES):
        if not p.is_file():
            raise FileNotFoundError(p)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    config = json.loads(LOCK_CONFIG.read_text(encoding="utf-8"))
    analysis_lock_path = RUN / "PAIRED_ANALYSIS_LOCK.json"
    analysis_lock = json.loads(analysis_lock_path.read_text(encoding="utf-8"))
    if sha256(ROOT / "code" / "analyze_frontier_f34_outer_paired.py") != config["source_code_hashes"]["paired_analysis"]:
        raise RuntimeError("Original pre-registered paired-analysis source hash changed")
    if sha256(Path(__file__)) != analysis_lock.get("corrected_analysis_sha256"):
        raise RuntimeError("Corrected post-lock source-path adapter differs from its pre-analysis lock")
    if sha256(ROOT / "code" / "run_frontier_f34_outer_once.py") != config["source_code_hashes"]["outer_training"]:
        raise RuntimeError("Outer trainer source hash differs from the locked code")
    if sha256(ROOT / "code" / "independent_validate_frontier_f34_outer.py") != config["source_code_hashes"]["independent_validator"]:
        raise RuntimeError("Independent validator source hash differs from the locked code")
    if sha256(ROOT / "09_TRACK_F_AGRICULTURAL_ROBUSTNESS" / "v3_4" / "OUTER_LOCK_PROTOCOL.md") != config.get("outer_lock_protocol_sha256"):
        raise RuntimeError("Outer lock protocol hash differs from the locked document")
    if sha256(ROOT / "code" / "independent_validate_frontier_f34_outer_v2.py") != analysis_lock.get("corrected_validator_sha256"):
        raise RuntimeError("Corrected validator source hash differs from the post-lock analysis record")
    if validation.get("status") != "PASS" or evaluation.get("status") != "OUTER_EVALUATED_ONCE":
        raise RuntimeError("Paired analysis requires a passing independent prediction validator")
    if evaluation.get("prediction_lock_sha256") != sha256(lock_path):
        raise RuntimeError("Evaluation manifest does not point to the current prediction lock")
    if analysis_lock.get("original_prediction_lock_sha256") != sha256(lock_path):
        raise RuntimeError("Paired-analysis lock is not bound to the frozen OOF predictions")
    if analysis_lock.get("validation_report_sha256") != sha256(validation_path):
        raise RuntimeError("Paired-analysis lock is not bound to the passing independent validator")
    if validation.get("validator_code_sha256") != analysis_lock.get("corrected_validator_sha256"):
        raise RuntimeError("Validation report does not identify the locked corrected validator")
    correction_lock = RUN / "VALIDATOR_CORRECTION_LOCK_V2.json"
    if validation.get("validator_correction_lock_sha256") != sha256(correction_lock):
        raise RuntimeError("Validation report does not match its qrels-blind correction lock")
    if analysis_lock.get("r3_2_scores_sha256") != sha256(R3_2_SCORES):
        raise RuntimeError("Staged R3-2 source hash differs from the pre-analysis lock")
    if analysis_lock.get("r3_2_scores_sha256") != analysis_lock.get("original_r3_2_scores_sha256"):
        raise RuntimeError("R3-2 source snapshot differs from its immutable source hash")
    if evaluation.get("outer_results_may_guide_method_revisions"):
        raise RuntimeError("Outer results are not allowed to guide method revisions")
    expected_input_hashes = lock.get("input_hashes", {})
    for name, digest in expected_input_hashes.items():
        if sha256(INPUTS / name) != digest:
            raise RuntimeError("A source input changed since prediction lock: " + name)

    # All outcome labels below are opened only after the immutable one-time lock and validator pass.
    features = pd.read_csv(INPUTS / "MODEL_FEATURES_NO_QRELS.csv", dtype={"query_id": str, "candidate_id": str})
    assignment = pd.read_csv(INPUTS / "OUTER_FOLD_ASSIGNMENTS.csv", dtype={"query_id": str, "role": str})
    assignment["outer_fold"] = assignment["outer_fold"].astype(int)
    fold_map = assignment.loc[assignment["role"].eq("test")].set_index("query_id")["outer_fold"].astype(int).to_dict()
    qrels = pd.read_csv(INPUTS / "QRELS_SEPARATE.csv", dtype={"query_id": str, "candidate_id": str})
    qrel_map: dict[str, dict[str, int]] = defaultdict(dict)
    for row in qrels.itertuples(index=False):
        qrel_map[str(row.query_id)][str(row.candidate_id)] = int(row.relevance)

    score = pd.read_csv(R3_2_SCORES, dtype={"query_id": str, "method": str, "candidate_id": str})
    required = {"query_id", "outer_fold", "method", "candidate_id", "relevance", "base_score", "score"}
    if not required.issubset(score.columns):
        raise ValueError("R3-2 OOF source is missing required candidate-level columns")
    score["outer_fold"] = score["outer_fold"].astype(int)
    score["relevance"] = score["relevance"].astype(int)
    if score.duplicated(["method", "query_id", "candidate_id"]).any():
        raise ValueError("Duplicate R3-2 method/query/candidate OOF row")
    if score["method"].nunique() < 2:
        raise ValueError("R3-2 source contains too few fixed baseline methods")
    if set(score["query_id"].astype(str)) != set(fold_map):
        raise ValueError("R3-2 query IDs do not match the locked five-fold outer pool")
    source_methods = sorted(score["method"].astype(str).unique())
    source_checks = []
    for method in source_methods:
        block = score.loc[score["method"].eq(method)].copy()
        if len(block) != 4813 or block["query_id"].nunique() != 483:
            raise ValueError("R3-2 method does not cover the complete frozen pool: " + method)
        if block["outer_fold"].to_dict() != block["query_id"].map(fold_map).astype(int).to_dict():
            raise ValueError("R3-2 outer fold mapping mismatch: " + method)
        keys = set(zip(block["query_id"].astype(str), block["candidate_id"].astype(str)))
        if keys != set(zip(features["query_id"].astype(str), features["candidate_id"].astype(str))):
            raise ValueError("R3-2 candidate IDs do not match the frozen pool: " + method)
        baseline_qrels = {(str(r.query_id), str(r.candidate_id)): int(r.relevance) for r in block.itertuples(index=False)}
        if any(baseline_qrels[(qid, cid)] != rel for qid, qm in qrel_map.items() for cid, rel in qm.items()):
            raise ValueError("R3-2 relevance labels differ from the locked qrel source: " + method)
        source_checks.append({"method": method, "query_count": 483, "candidate_pairs": 4813,
                              "outer_fold_mapping": "EXACT", "candidate_ids": "EXACT", "qrels": "EXACT"})

    rank_parts = []
    for fold in range(5):
        rank_path = RUN / "folds" / ("outer_{}".format(fold)) / "candidate_policy_ranks.csv"
        expected_rank_hash = lock["folds"][str(fold)]["files"]["candidate_policy_ranks.csv"]
        if sha256(rank_path) != expected_rank_hash:
            raise RuntimeError("Locked fold policy ranks changed: {}".format(fold))
        rank_parts.append(pd.read_csv(rank_path, dtype={"query_id": str, "candidate_id": str, "policy": str}))
    policy_ranks = pd.concat(rank_parts, ignore_index=True)
    f34_rows_by_method: dict[str, list[dict]] = {}
    for policy in POLICIES:
        block = policy_ranks.loc[policy_ranks["policy"].eq(policy)]
        rows = []
        for qid, qblock in block.groupby("query_id", sort=True):
            order = qblock.sort_values(["selected_rank", "candidate_id"])["candidate_id"].astype(str).tolist()
            ndcg, mrr = exact_metrics(order, qrel_map[str(qid)])
            base_order = features.loc[features["query_id"].eq(qid)].sort_values(["base_score", "candidate_id"], ascending=[False, True])["candidate_id"].astype(str).tolist()
            base_ndcg, base_mrr = exact_metrics(base_order, qrel_map[str(qid)])
            rows.append({"query_id": str(qid), "ndcg10": ndcg, "mrr10": mrr,
                         "delta_ndcg10": ndcg - base_ndcg, "delta_mrr10": mrr - base_mrr})
        f34_rows_by_method[policy] = sorted(rows, key=lambda x: x["query_id"])

    baseline_rows_by_method = {}
    for method in source_methods:
        block = score.loc[score["method"].eq(method)]
        rows = []
        for qid, qblock in block.groupby("query_id", sort=True):
            order = qblock.sort_values(["score", "candidate_id"], ascending=[False, True], kind="mergesort")["candidate_id"].astype(str).tolist()
            ndcg, mrr = exact_metrics(order, qrel_map[str(qid)])
            base_ndcg, base_mrr = exact_metrics(
                features.loc[features["query_id"].eq(qid)].sort_values(["base_score", "candidate_id"], ascending=[False, True], kind="mergesort")["candidate_id"].astype(str).tolist(),
                qrel_map[str(qid)])
            rows.append({"query_id": str(qid), "ndcg10": ndcg, "mrr10": mrr,
                         "delta_ndcg10": ndcg - base_ndcg, "delta_mrr10": mrr - base_mrr})
        baseline_rows_by_method[method] = sorted(rows, key=lambda x: x["query_id"])

    all_rows_by_method = {**f34_rows_by_method, **baseline_rows_by_method}
    summaries = pd.DataFrame([method_summary(method, rows) for method, rows in all_rows_by_method.items()])
    comparisons = [(MAIN_POLICY, policy) for policy in POLICIES if policy != MAIN_POLICY]
    comparisons += [(MAIN_POLICY, method) for method in source_methods]
    rng = np.random.Generator(np.random.PCG64(SEED))
    contrast_records = []
    raw_p_by_metric: dict[str, list[float]] = {m: [] for m in METRICS}
    contrast_ids = []
    for left, right in comparisons:
        result = paired_bootstrap(all_rows_by_method[left], all_rows_by_method[right], rng)
        contrast_id = "{}__minus__{}".format(left, right)
        contrast_ids.append(contrast_id)
        for metric in METRICS:
            item = result[metric]
            contrast_records.append({
                "contrast_id": contrast_id, "method_a": left, "method_b": right,
                "metric": metric, "difference_a_minus_b": item["difference"],
                "ci95_low": item["ci95_low"], "ci95_high": item["ci95_high"],
                "p_raw": item["p_raw"], "bootstrap_replicates": REPS, "seed": SEED,
            })
            raw_p_by_metric[metric].append(item["p_raw"])
        discord = result["qdr_discordant"]
        contrast_records.append({
            "contrast_id": contrast_id, "method_a": left, "method_b": right,
            "metric": "qdr_discordant_events", "difference_a_minus_b": np.nan,
            "ci95_low": np.nan, "ci95_high": np.nan, "p_raw": np.nan,
            "a_degraded_b_not": discord["a_degraded_b_not"],
            "a_not_degraded_b": discord["a_not_degraded_b"],
            "bootstrap_replicates": REPS, "seed": SEED,
        })
    stat = pd.DataFrame(contrast_records)
    for metric, pv in raw_p_by_metric.items():
        adj = holm(pv)
        rows_idx = stat.index[stat["metric"].eq(metric)].tolist()
        for row_idx, p_adj in zip(rows_idx, adj):
            stat.loc[row_idx, "p_holm_within_metric"] = p_adj

    query_rows = []
    for method, rows in all_rows_by_method.items():
        for r in rows:
            query_rows.append({"method": method, **r})
    source_report = {
        "status": "PASS", "r3_2_original_source_path": str(R3_2_ORIGINAL),
        "r3_2_staged_source_path": str(R3_2_SCORES),
        "paired_analysis_lock_sha256": sha256(analysis_lock_path),
        "paired_analysis_code_sha256": sha256(Path(__file__)),
        "independent_validation_report_sha256": sha256(validation_path),
        "r3_2_scores_sha256": sha256(R3_2_SCORES), "r3_2_method_count": len(source_methods),
        "methods": source_checks, "outer_fold_mapping": "EXACT per query against current locked assignments",
        "candidate_pool": "EXACT 4,813 query-candidate pairs", "qrels": "EXACT per query-candidate row",
        "original_table_2_comparison": "EXCLUDED: run comparability is not established",
        "outer_results_evidence_status": "EXPLORATORY_OUTER / historically exposed development pool",
        "independent_confirmation": False,
        "analysis_rule": "All methods in the validated R3-2 candidate-level file are reported; none is selected post hoc as the sole comparator.",
    }
    atomic_csv(RUN / "outer_method_summary.csv", summaries)
    atomic_csv(RUN / "outer_paired_statistics.csv", stat)
    atomic_csv(RUN / "outer_query_metrics_all_methods.csv", pd.DataFrame(query_rows))
    atomic_json(RUN / "r3_2_source_mapping_report.json", source_report)
    print("PAIRED_ANALYSIS_PASS baselines={} comparisons={} query_cluster_bootstrap={} independent_confirmation=false".format(
        len(source_methods), len(comparisons), REPS))
    print(summaries.to_string(index=False))


if __name__ == "__main__":
    main()
