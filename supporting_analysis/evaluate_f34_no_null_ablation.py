#!/usr/bin/env python3
"""Evaluate locked no-null predictions against canonical full F3.4 OOF.

Held-out qrels are opened only after the no-null prediction lock exists. No
model or gate parameter is selected in this evaluator.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-12
BOOT_REPS = 10000
BOOT_SEED = 20261012


def disc(rank: int) -> float:
    return 1.0 / math.log2(rank + 1.0) if 1 <= int(rank) <= 10 else 0.0


def stats(df: pd.DataFrame, policy: str) -> dict:
    d = df["delta_ndcg10"].to_numpy(float)
    losses = np.maximum(-d, 0.0)
    n = len(df)
    tail_n = max(1, int(math.ceil(0.10 * n)))
    win, lose = int((d > EPS).sum()), int((d < -EPS).sum())
    return {
        "policy": policy,
        "scope": df["scope"].iloc[0],
        "n_queries": n,
        "n_reachable": int(df["reachable"].sum()),
        "n_unreachable": int((~df["reachable"]).sum()),
        "ndcg10": float(df["ndcg10"].mean()),
        "mrr10": float(df["mrr10"].mean()),
        "recall10": float(df["recall10"].mean()),
        "qdr": float((d < -EPS).mean()),
        "ed": float(losses.mean()),
        "cvar10": float(np.sort(losses)[-tail_n:].mean()),
        "cvar10_tail_queries": tail_n,
        "wins": win,
        "ties": n - win - lose,
        "losses": lose,
        "wtl": "{}/{}/{}".format(win, n - win - lose, lose),
        "accept_rate": float(df["accepted"].mean()),
        "fallback_rate": float(1.0 - df["accepted"].mean()),
        "ranking_change_rate": float(df["ranking_changed"].mean()),
    }


def boot_effect(a: pd.DataFrame, b: pd.DataFrame, label: str, rng: np.random.Generator) -> list[dict]:
    """Paired query bootstrap: effect is no-null Q05 minus full F3.4 Q05."""
    if a["query_id"].tolist() != b["query_id"].tolist():
        raise ValueError("Paired rows must share query order")
    n = len(a)
    point_arrays = {
        "ndcg10": a["ndcg10"].to_numpy(float) - b["ndcg10"].to_numpy(float),
        "mrr10": a["mrr10"].to_numpy(float) - b["mrr10"].to_numpy(float),
        "recall10": a["recall10"].to_numpy(float) - b["recall10"].to_numpy(float),
        "qdr": (a["delta_ndcg10"].to_numpy(float) < -EPS).astype(float) - (b["delta_ndcg10"].to_numpy(float) < -EPS).astype(float),
        "ed": np.maximum(-a["delta_ndcg10"].to_numpy(float), 0.0) - np.maximum(-b["delta_ndcg10"].to_numpy(float), 0.0),
    }

    def cvar(x: np.ndarray) -> float:
        losses = np.maximum(-x, 0.0)
        k = max(1, int(math.ceil(0.10 * len(losses))))
        return float(np.sort(losses)[-k:].mean())

    point = {key: float(vals.mean()) for key, vals in point_arrays.items()}
    point["cvar10"] = cvar(a["delta_ndcg10"].to_numpy(float)) - cvar(b["delta_ndcg10"].to_numpy(float))
    samples = {key: np.empty(BOOT_REPS, dtype=float) for key in point}
    da = a["delta_ndcg10"].to_numpy(float)
    db = b["delta_ndcg10"].to_numpy(float)
    nd_a, nd_b = a["ndcg10"].to_numpy(float), b["ndcg10"].to_numpy(float)
    mr_a, mr_b = a["mrr10"].to_numpy(float), b["mrr10"].to_numpy(float)
    re_a, re_b = a["recall10"].to_numpy(float), b["recall10"].to_numpy(float)
    for rep in range(BOOT_REPS):
        ix = rng.integers(0, n, size=n)
        samples["ndcg10"][rep] = (nd_a[ix] - nd_b[ix]).mean()
        samples["mrr10"][rep] = (mr_a[ix] - mr_b[ix]).mean()
        samples["recall10"][rep] = (re_a[ix] - re_b[ix]).mean()
        samples["qdr"][rep] = ((da[ix] < -EPS).mean() - (db[ix] < -EPS).mean())
        samples["ed"][rep] = (np.maximum(-da[ix], 0.0) - np.maximum(-db[ix], 0.0)).mean()
        samples["cvar10"][rep] = cvar(da[ix]) - cvar(db[ix])
    out = []
    for metric, estimate in point.items():
        lo, hi = np.quantile(samples[metric], [0.025, 0.975])
        out.append({
            "contrast": label, "metric": metric, "effect_no_null_minus_full": estimate,
            "ci95_low": float(lo), "ci95_high": float(hi), "bootstrap_reps": BOOT_REPS,
            "bootstrap_seed": BOOT_SEED, "resampling_unit": "query_id",
        })
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    base = Path(r"E:\tmp\experiments\apin_revision_f3_4_main_method_evidence\02_CANONICAL_OOF\raw_locked_run")
    root = Path(r"E:\tmp\experiments\apin_revision_f3_4_main_method_evidence")
    out = Path(r"E:\APIN_FINAL_SCIENTIFIC_CLOSURE\data\no_null")
    pred_path = out / "F34_NO_NULL_PREDICTIONS_LOCKED.csv"
    lock_path = out / "F34_NO_NULL_PREDICTION_LOCK.json"
    if not pred_path.exists() or not lock_path.exists():
        raise FileNotFoundError("No-null label-blind prediction lock is missing")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "PREDICTIONS_LOCKED_LABEL_BLIND" or lock.get("test_qrels_read") is not False:
        raise ValueError("No-null predictions are not locked label-blind")

    # Outcomes and complete selected orders for full F3.4 are read from its
    # canonical OOF artifacts. Qrels are the only relevance source for eval.
    no = pd.read_csv(pred_path, dtype={"query_id": str, "candidate_id": str})
    qrel_path = root / "14_SOURCE_AND_HASHES" / "source_inputs" / "QRELS_SEPARATE.csv"
    qrels = pd.read_csv(qrel_path, dtype={"query_id": str, "candidate_id": str})
    if qrels.duplicated(["query_id", "candidate_id"]).any():
        raise ValueError("Duplicate fixed qrel candidate rows")
    targets = qrels[qrels["relevance"].astype(int).gt(0)].groupby("query_id")["candidate_id"].agg(list).to_dict()
    candidates = no.groupby("query_id")["candidate_id"].agg(set).to_dict()
    rows = []
    for qid, block in no.groupby("query_id", sort=True):
        block = block.copy()
        target_list = targets.get(qid, [])
        if len(target_list) > 1:
            raise ValueError("One-positive-or-null invariant failed: {}".format(qid))
        reachable = bool(target_list)
        if reachable and target_list[0] not in candidates[qid]:
            raise ValueError("Positive target absent from locked pool: {}".format(qid))
        target = target_list[0] if reachable else None
        base_order = block.sort_values("base_rank")["candidate_id"].astype(str).tolist()
        selected_order = block.sort_values("no_null_q05_rank")["candidate_id"].astype(str).tolist()
        proposal_order = block.sort_values("no_null_proposal_rank")["candidate_id"].astype(str).tolist()
        base_rank = base_order.index(target) + 1 if reachable else 0
        selected_rank = selected_order.index(target) + 1 if reachable else 0
        base_nd = disc(base_rank)
        nd = disc(selected_rank)
        rows.append({
            "query_id": qid, "outer_fold": int(block["outer_fold"].iloc[0]), "reachable": reachable,
            "scope": "ALL", "base_rank": base_rank, "selected_rank": selected_rank,
            "ndcg10": nd, "mrr10": 1.0 / selected_rank if 1 <= selected_rank <= 10 else 0.0,
            "recall10": float(1 <= selected_rank <= 10), "base_ndcg10": base_nd,
            "delta_ndcg10": nd - base_nd, "accepted": int(block["q05_accepted"].iloc[0]),
            "ranking_changed": int(base_order != selected_order),
            "proposal_changed": int(base_order != proposal_order),
            "no_null_order_sha256": str(block["q05_order_sha256"].iloc[0]),
            "no_null_order": "|".join(selected_order),
            "q05_gain_q05": float(block["q05_gain_q05"].iloc[0]),
            "posterior_mean_utility_gain": float(block["posterior_mean_utility_gain"].iloc[0]),
        })
    no_q = pd.DataFrame(rows).sort_values("query_id").reset_index(drop=True)
    if len(no_q) != 483 or no_q["query_id"].nunique() != 483:
        raise ValueError("Expected 483 distinct OOF queries")
    no_q["scope"] = "ALL"

    # Full F3.4 query-level effectiveness and fixed-gate action are canonical.
    outcomes = pd.read_csv(base / "outer_query_metrics.csv", dtype={"query_id": str})
    full = outcomes[outcomes["policy"].eq("RHO2_LCB05_GATE")].copy()
    if len(full) != 483 or full["query_id"].nunique() != 483:
        raise ValueError("Canonical full F3.4 Q05 outcomes do not cover 483 queries")
    full = full.set_index("query_id").loc[no_q["query_id"]].reset_index()
    full_decisions, full_ranks = [], []
    always_ranks = []
    for fold in range(5):
        fd = base / "folds" / ("outer_{}".format(fold))
        d = pd.read_csv(fd / "policy_decisions.csv", dtype={"query_id": str})
        full_decisions.append(d[(d["policy"].eq("RHO2_LCB05_GATE"))])
        r = pd.read_csv(fd / "candidate_policy_ranks.csv", dtype={"query_id": str, "candidate_id": str})
        full_ranks.append(r[r["policy"].eq("RHO2_LCB05_GATE")])
        always_ranks.append(r[r["policy"].eq("RHO2_ALWAYS_BMA")])
    full_decision = pd.concat(full_decisions).set_index("query_id").loc[no_q["query_id"]]
    rank_tbl = pd.concat(full_ranks)
    back_tbl = rank_tbl[rank_tbl["policy"].eq("FROZEN_BACKBONE")].set_index(["query_id", "candidate_id"])
    # rank table filter above contains only Q05; load backbone ranks separately
    # from each fold so exact ranking-change rate is checked.
    backs = []
    for fold in range(5):
        r = pd.read_csv(base / "folds" / ("outer_{}".format(fold)) / "candidate_policy_ranks.csv", dtype={"query_id": str, "candidate_id": str})
        backs.append(r[r["policy"].eq("FROZEN_BACKBONE")])
    back_tbl = pd.concat(backs).set_index(["query_id", "candidate_id"])
    gate_tbl = rank_tbl.set_index(["query_id", "candidate_id"])
    always_tbl = pd.concat(always_ranks).set_index(["query_id", "candidate_id"])
    full_rows = []
    for i, row in no_q.iterrows():
        qid = row.query_id
        target_rows = qrels[(qrels["query_id"].eq(qid)) & (qrels["relevance"].astype(int).gt(0))]
        reachable = len(target_rows) == 1
        target = str(target_rows["candidate_id"].iloc[0]) if reachable else None
        if reachable:
            base_rank = int(back_tbl.loc[(qid, target), "selected_rank"])
            rank = int(gate_tbl.loc[(qid, target), "selected_rank"])
        else:
            base_rank = rank = 0
        base_nd, nd = disc(base_rank), disc(rank)
        b_order = back_tbl.loc[qid].sort_values("selected_rank").index.astype(str).tolist()
        f_order = gate_tbl.loc[qid].sort_values("selected_rank").index.astype(str).tolist()
        full_rows.append({
            "query_id": qid, "outer_fold": int(full.iloc[i]["outer_fold"]), "reachable": reachable,
            "scope": "ALL", "base_rank": base_rank, "selected_rank": rank,
            "ndcg10": nd, "mrr10": 1.0 / rank if 1 <= rank <= 10 else 0.0,
            "recall10": float(1 <= rank <= 10), "base_ndcg10": base_nd,
            "delta_ndcg10": float(full.iloc[i]["delta_ndcg10"]),
            "accepted": int(full_decision.iloc[i]["gate_applied"]),
            "ranking_changed": int(b_order != f_order),
            "full_order_sha256": str(full_decision.iloc[i]["selected_order_sha256"]),
            "full_order": "|".join(f_order),
        })
    full_q = pd.DataFrame(full_rows)
    full_q["scope"] = "ALL"
    full_q = full_q.sort_values("query_id").reset_index(drop=True)
    no_q = no_q.sort_values("query_id").reset_index(drop=True)
    no_q["full_order_sha256"] = full_q["full_order_sha256"].to_numpy()
    no_q["full_order"] = full_q["full_order"].to_numpy()
    no_q["ranking_order_matches_full"] = no_q["no_null_order"].eq(no_q["full_order"])
    no_q["policy"] = "no-null F3.4 Q05"
    full_q["policy"] = "full F3.4 Q05"

    # Always-mean control remains included as a locked context row.
    always = outcomes[outcomes["policy"].eq("RHO2_ALWAYS_BMA")].copy().set_index("query_id").loc[no_q["query_id"]].reset_index()
    always_rows = []
    for i, row in no_q.iterrows():
        qid = row.query_id
        target_rows = qrels[(qrels["query_id"].eq(qid)) & (qrels["relevance"].astype(int).gt(0))]
        reachable = len(target_rows) == 1
        target = str(target_rows["candidate_id"].iloc[0]) if reachable else None
        # Reuse exact policy ranks, selected by the canonical candidate-rank file.
        rt = always_tbl.loc[qid].reset_index()
        br = back_tbl.loc[(qid, target), "selected_rank"] if reachable else 0
        rr = int(rt.loc[rt["candidate_id"].eq(target), "selected_rank"].iloc[0]) if reachable else 0
        nd, bnd = disc(rr), disc(int(br))
        order = rt.sort_values("selected_rank")["candidate_id"].astype(str).tolist()
        b_order = back_tbl.loc[qid].sort_values("selected_rank").index.astype(str).tolist()
        always_rows.append({"query_id": qid, "reachable": reachable, "scope": "ALL", "ndcg10": nd,
                            "mrr10": 1.0 / rr if 1 <= rr <= 10 else 0.0, "recall10": float(1 <= rr <= 10),
                            "base_ndcg10": bnd, "delta_ndcg10": float(always.iloc[i]["delta_ndcg10"]),
                            "accepted": 1, "ranking_changed": int(order != b_order)})
    always_q = pd.DataFrame(always_rows)
    always_q["scope"] = "ALL"
    always_q["policy"] = "Always posterior-mean"

    no_q.to_csv(out / "F34_NULL_STATE_ABLATION_QUERY_LEVEL.csv", index=False, float_format="%.12g")
    full_q.to_csv(out / "F34_NULL_STATE_FULL_MODEL_QUERY_LEVEL.csv", index=False, float_format="%.12g")
    all_policy = pd.concat([full_q, no_q, always_q], ignore_index=True)
    summary_rows = []
    for policy, block in all_policy.groupby("policy", sort=False):
        for scope, subset in [("ALL", block), ("REACHABLE", block[block["reachable"]]),
                              ("UNREACHABLE", block[~block["reachable"]])]:
            if len(subset):
                summary_rows.append(stats(subset.assign(scope=scope), policy))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "F34_NULL_STATE_ABLATION_SUMMARY.csv", index=False, float_format="%.12g")

    rng = np.random.default_rng(BOOT_SEED)
    effects = []
    for scope in ["ALL", "REACHABLE"]:
        selector_no = np.ones(len(no_q), dtype=bool) if scope == "ALL" else no_q["reachable"].to_numpy(bool)
        selector_full = np.ones(len(full_q), dtype=bool) if scope == "ALL" else full_q["reachable"].to_numpy(bool)
        a = no_q.loc[selector_no].sort_values("query_id").reset_index(drop=True)
        b = full_q.loc[selector_full].sort_values("query_id").reset_index(drop=True)
        effects.extend(boot_effect(a, b, "no-null Q05 minus full F3.4 Q05; " + scope, rng))
    pd.DataFrame(effects).to_csv(out / "F34_NULL_STATE_ABLATION_PAIRED_EFFECTS.csv", index=False, float_format="%.12g")
    paired_wtl = []
    for scope in ["ALL", "REACHABLE"]:
        selector_no = np.ones(len(no_q), dtype=bool) if scope == "ALL" else no_q["reachable"].to_numpy(bool)
        selector_full = np.ones(len(full_q), dtype=bool) if scope == "ALL" else full_q["reachable"].to_numpy(bool)
        a = no_q.loc[selector_no].sort_values("query_id").reset_index(drop=True)
        b = full_q.loc[selector_full].sort_values("query_id").reset_index(drop=True)
        diff = a["ndcg10"].to_numpy(float) - b["ndcg10"].to_numpy(float)
        wins, losses = int((diff > EPS).sum()), int((diff < -EPS).sum())
        paired_wtl.append({"scope": scope, "n_queries": len(diff), "no_null_better": wins,
                           "tie": len(diff) - wins - losses, "no_null_worse": losses,
                           "wtl_no_null_vs_full": "{}/{}/{}".format(wins, len(diff) - wins - losses, losses)})
    pd.DataFrame(paired_wtl).to_csv(out / "F34_NULL_STATE_ABLATION_PAIRED_WTL.csv", index=False)
    action_rows = []
    for scope in ["ALL", "REACHABLE", "UNREACHABLE"]:
        part = no_q if scope == "ALL" else no_q[no_q["reachable"]] if scope == "REACHABLE" else no_q[~no_q["reachable"]]
        if len(part):
            action_rows.append({
                "scope": scope, "n_queries": len(part),
                "full_q05_accept_count": int(full_q.loc[part.index, "accepted"].sum()),
                "no_null_q05_accept_count": int(part["accepted"].sum()),
                "full_q05_accept_rate": float(full_q.loc[part.index, "accepted"].mean()),
                "no_null_q05_accept_rate": float(part["accepted"].mean()),
                "exact_order_matches": int(part["ranking_order_matches_full"].sum()),
                "exact_order_match_rate": float(part["ranking_order_matches_full"].mean()),
                "same_observed_ndcg": int(np.isclose(part["ndcg10"], full_q.loc[part.index, "ndcg10"], atol=1e-12).sum()),
            })
    pd.DataFrame(action_rows).to_csv(out / "F34_NULL_STATE_ABLATION_ACTION_COMPARISON.csv", index=False, float_format="%.12g")

    manifest = {
        "status": "EVALUATION_COMPLETE_AFTER_LABEL_BLIND_PREDICTION_LOCK",
        "predictions_sha256": sha256(pred_path), "prediction_lock_sha256": sha256(lock_path),
        "qrels_sha256": sha256(qrel_path), "canonical_outcomes_sha256": sha256(base / "outer_query_metrics.csv"),
        "query_count": int(no_q["query_id"].nunique()), "reachable_count": int(no_q["reachable"].sum()),
        "unreachable_count": int((~no_q["reachable"]).sum()),
        "gate_threshold": "strict q05_gain_q05 > 0; locked before held-out evaluation",
        "uncertainty_interval": "paired percentile bootstrap over query_id; CVaR tail recomputed per replicate",
        "bootstrap_reps": BOOT_REPS, "bootstrap_seed": BOOT_SEED,
        "no_null_estimand_note": lock.get("no_null_training_rule"),
        "heldout_qrels_used_for_model_selection": False,
        "metric_definitions": {"QDR": "fraction of queries with nDCG@10 loss > 1e-12 vs frozen backbone",
                               "ED": "mean max(0, backbone nDCG@10 - policy nDCG@10)",
                               "CVaR10": "mean largest ceil(0.10*n) per-query nDCG@10 losses"},
    }
    (out / "F34_NULL_STATE_ABLATION_EVAL_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print("WROTE", out)


if __name__ == "__main__":
    main()
