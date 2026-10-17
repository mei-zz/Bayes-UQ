#!/usr/bin/env python3
"""Replay Original M2 with its explicit residual scale multiplier omitted.

Uses the already fitted R1-1 outer-fold predictions and residual_unit. It does
not fit or tune on held-out qrels. "No cap" here means no explicit selected
kappa multiplier; the residual factors still impose a natural unit bound.
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
BOOT_SEED = 20261013


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ndcg(rank: int) -> float:
    return 1 / math.log2(rank + 1) if 1 <= int(rank) <= 10 else 0.0


def summarize(df: pd.DataFrame, name: str) -> dict:
    d = df["delta_ndcg10"].to_numpy(float)
    losses = np.maximum(-d, 0.0)
    n = len(df)
    tail = max(1, int(math.ceil(0.10 * n)))
    wins, bad = int((d > EPS).sum()), int((d < -EPS).sum())
    return {
        "method": name, "n_queries": n,
        "nDCG@10": float(df["ndcg10"].mean()),
        "MRR@10": float(df["mrr10"].mean()),
        "Recall@10": float(df["recall10"].mean()),
        "QDR": float((d < -EPS).mean()), "QDR_count": bad,
        "ED": float(losses.mean()), "CVaR10": float(np.sort(losses)[-tail:].mean()),
        "CVaR10_tail_queries": tail, "worst_decline": float(losses.max()),
        "W/T/L": "{}/{}/{}".format(wins, n - wins - bad, bad),
        "ranking_change_rate": float(df["ranking_changed"].mean()),
        "top1_change_rate": float(df["top1_changed"].mean()),
        "target_rank_improved": int((df["target_rank_delta"] < 0).sum()),
        "target_rank_tied": int((df["target_rank_delta"] == 0).sum()),
        "target_rank_worsened": int((df["target_rank_delta"] > 0).sum()),
    }


def main() -> None:
    source = Path(r"E:\tmp\experiments\apin_revision_r1_1\results\all_outer_candidates.csv")
    out = Path(r"E:\APIN_FINAL_SCIENTIFIC_CLOSURE\data\m2_no_cap")
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(source)
    a0 = raw[raw["method"].eq("A0_full")].copy()
    if len(a0) != 4813 or a0["query_id"].nunique() != 483:
        raise ValueError("Original R1-1 A0 locked pool cardinality mismatch")
    if a0.duplicated(["query_id", "candidate_id"]).any():
        raise ValueError("Duplicate A0 candidate rows")
    for c in ["base_score", "residual_unit", "label", "selected_cap", "final_rank", "base_rank"]:
        a0[c] = pd.to_numeric(a0[c], errors="raise")
    max_abs_residual = float(a0["residual_unit"].abs().max())
    if max_abs_residual > 1.0 + 1e-10:
        raise ValueError("M2 residual factors violate the derived natural |r_i|<=1 bound")

    query_rows = []
    cand_rows = []
    for qid, block in a0.groupby("query_id", sort=True):
        block = block.copy()
        if int(block["label"].sum()) > 1:
            raise ValueError("One-positive-or-null target invariant failed for {}".format(qid))
        positive = block[block["label"].eq(1)]
        base_order = block.sort_values("base_rank", kind="mergesort")
        full_order = block.sort_values("final_rank", kind="mergesort")
        uncapped = block.assign(no_cap_score=block["base_score"] + block["residual_unit"])
        no_cap_order = uncapped.sort_values(["no_cap_score", "candidate_id"], ascending=[False, True], kind="mergesort")
        target = str(positive["candidate_id"].iloc[0]) if len(positive) else None
        rank_base = int(np.flatnonzero(base_order["candidate_id"].astype(str).to_numpy() == target)[0] + 1) if target else 0
        rank_full = int(block.loc[block["candidate_id"].astype(str).eq(target), "final_rank"].iloc[0]) if target else 0
        rank_no = int(np.flatnonzero(no_cap_order["candidate_id"].astype(str).to_numpy() == target)[0] + 1) if target else 0
        nd_base, nd_full, nd_no = ndcg(rank_base), ndcg(rank_full), ndcg(rank_no)
        base_ids = base_order["candidate_id"].astype(str).tolist()
        full_ids = full_order["candidate_id"].astype(str).tolist()
        no_ids = no_cap_order["candidate_id"].astype(str).tolist()
        query_rows.append({
            "query_id": str(qid), "outer_fold": int(block["outer_fold"].iloc[0]),
            "reachable": bool(target), "base_target_rank": rank_base,
            "selected_A0_target_rank": rank_full, "no_cap_target_rank": rank_no,
            "target_rank_delta_no_cap_vs_A0": rank_no - rank_full,
            "base_ndcg10": nd_base, "A0_ndcg10": nd_full, "A0_delta_ndcg10": nd_full - nd_base,
            "no_cap_ndcg10": nd_no, "no_cap_delta_ndcg10": nd_no - nd_base,
            "no_cap_minus_A0_ndcg10": nd_no - nd_full,
            "A0_ranking_changed": int(full_ids != base_ids),
            "no_cap_ranking_changed": int(no_ids != base_ids),
            "A0_top1_changed": int(full_ids[0] != base_ids[0]),
            "no_cap_top1_changed": int(no_ids[0] != base_ids[0]),
            "mean_abs_residual_unit": float(block["residual_unit"].abs().mean()),
            "max_abs_residual_unit_query": float(block["residual_unit"].abs().max()),
            "selected_cap_in_A0": float(block["selected_cap"].iloc[0]),
        })
        for row in uncapped.itertuples(index=False):
            cand_rows.append({"query_id": str(qid), "outer_fold": int(row.outer_fold),
                              "candidate_id": str(row.candidate_id), "label": int(row.label),
                              "base_score": float(row.base_score), "residual_unit": float(row.residual_unit),
                              "no_cap_score": float(row.no_cap_score), "base_rank": int(row.base_rank),
                              "A0_selected_rank": int(row.final_rank), "selected_cap_A0": float(row.selected_cap)})
    q = pd.DataFrame(query_rows).sort_values("query_id").reset_index(drop=True)
    if len(q) != 483 or q["reachable"].sum() != 397:
        raise ValueError("Reachability count differs from the locked R1-1 source")
    if not np.allclose(q["A0_ndcg10"], q["base_ndcg10"] + q["A0_delta_ndcg10"], atol=1e-12):
        raise ValueError("A0 source score replay failed")
    # Reproduce the reported full arm before interpreting the new variant.
    if abs(float(q["A0_ndcg10"].mean()) - 0.692209) > 1e-6:
        raise ValueError("A0 replay does not match the R1-1 canonical result")

    metrics = []
    for name, ndcol, rankcol, changed, top in [
        ("Original M2 A0 full (matched source)", "A0_ndcg10", "selected_A0_target_rank", "A0_ranking_changed", "A0_top1_changed"),
        ("Original M2 kappa-free unit residual", "no_cap_ndcg10", "no_cap_target_rank", "no_cap_ranking_changed", "no_cap_top1_changed"),
    ]:
        frame = q.copy()
        frame["ndcg10"] = frame[ndcol]
        frame["mrr10"] = np.where(frame[rankcol].between(1, 10), 1.0 / frame[rankcol], 0.0)
        frame["recall10"] = frame[rankcol].between(1, 10).astype(float)
        frame["delta_ndcg10"] = frame[ndcol] - frame["base_ndcg10"]
        frame["ranking_changed"] = frame[changed]
        frame["top1_changed"] = frame[top]
        frame["target_rank_delta"] = frame[rankcol] - frame["base_target_rank"]
        metrics.append(summarize(frame, name))
    summary = pd.DataFrame(metrics)
    summary.to_csv(out / "M2_NO_CAP_ABLATION_SUMMARY.csv", index=False, float_format="%.12g")
    q.to_csv(out / "M2_NO_CAP_ABLATION_QUERY_LEVEL.csv", index=False, float_format="%.12g")
    pd.DataFrame(cand_rows).to_csv(out / "M2_NO_CAP_ABLATION_CANDIDATE_LEVEL.csv", index=False, float_format="%.12g")

    # Paired query bootstrap for mean-effect and downside contrasts.
    rng = np.random.default_rng(BOOT_SEED)
    n = len(q)
    point = {
        "ndcg10": float((q["no_cap_ndcg10"] - q["A0_ndcg10"]).mean()),
        "mrr10": float((np.where(q["no_cap_target_rank"].between(1, 10), 1.0 / q["no_cap_target_rank"], 0.0) -
                         np.where(q["selected_A0_target_rank"].between(1, 10), 1.0 / q["selected_A0_target_rank"], 0.0)).mean()),
        "recall10": float((q["no_cap_target_rank"].between(1, 10).astype(float) - q["selected_A0_target_rank"].between(1, 10).astype(float)).mean()),
        "qdr": float(((q["no_cap_ndcg10"] - q["base_ndcg10"]) < -EPS).mean() - ((q["A0_ndcg10"] - q["base_ndcg10"]) < -EPS).mean()),
        "ed": float((np.maximum(q["base_ndcg10"] - q["no_cap_ndcg10"], 0.0) - np.maximum(q["base_ndcg10"] - q["A0_ndcg10"], 0.0)).mean()),
    }
    def cvar(loss: np.ndarray) -> float:
        k = max(1, int(math.ceil(0.10 * len(loss))))
        return float(np.sort(loss)[-k:].mean())
    point["cvar10"] = cvar(np.maximum(q["base_ndcg10"].to_numpy() - q["no_cap_ndcg10"].to_numpy(), 0.0)) - cvar(
        np.maximum(q["base_ndcg10"].to_numpy() - q["A0_ndcg10"].to_numpy(), 0.0))
    boot = {k: np.empty(BOOT_REPS, dtype=float) for k in point}
    a_nd, b_nd, base_nd = q["no_cap_ndcg10"].to_numpy(), q["A0_ndcg10"].to_numpy(), q["base_ndcg10"].to_numpy()
    a_rank, b_rank = q["no_cap_target_rank"].to_numpy(), q["selected_A0_target_rank"].to_numpy()
    a_mrr = np.where((a_rank >= 1) & (a_rank <= 10), 1 / np.maximum(a_rank, 1), 0.0)
    b_mrr = np.where((b_rank >= 1) & (b_rank <= 10), 1 / np.maximum(b_rank, 1), 0.0)
    a_rec, b_rec = ((a_rank >= 1) & (a_rank <= 10)).astype(float), ((b_rank >= 1) & (b_rank <= 10)).astype(float)
    a_loss, b_loss = np.maximum(base_nd - a_nd, 0.0), np.maximum(base_nd - b_nd, 0.0)
    for rep in range(BOOT_REPS):
        ix = rng.integers(0, n, size=n)
        boot["ndcg10"][rep] = (a_nd[ix] - b_nd[ix]).mean()
        boot["mrr10"][rep] = (a_mrr[ix] - b_mrr[ix]).mean()
        boot["recall10"][rep] = (a_rec[ix] - b_rec[ix]).mean()
        boot["qdr"][rep] = ((a_nd[ix] - base_nd[ix] < -EPS).mean() - (b_nd[ix] - base_nd[ix] < -EPS).mean())
        boot["ed"][rep] = (a_loss[ix] - b_loss[ix]).mean()
        boot["cvar10"][rep] = cvar(a_loss[ix]) - cvar(b_loss[ix])
    effect_rows = []
    for metric, estimate in point.items():
        lo, hi = np.quantile(boot[metric], [0.025, 0.975])
        effect_rows.append({"contrast": "kappa-free unit residual minus selected-cap A0", "metric": metric,
                            "effect": estimate, "ci95_low": float(lo), "ci95_high": float(hi),
                            "bootstrap_reps": BOOT_REPS, "bootstrap_seed": BOOT_SEED,
                            "resampling_unit": "query_id"})
    pd.DataFrame(effect_rows).to_csv(out / "M2_NO_CAP_ABLATION_PAIRED_EFFECTS.csv", index=False, float_format="%.12g")

    manifest = {
        "status": "COMPLETED_REPLAY_OF_LOCKED_OUTER_PREDICTIONS",
        "source_file": str(source), "source_sha256": sha256(source),
        "n_queries": int(q.query_id.nunique()), "n_candidate_rows": len(cand_rows),
        "reachable_queries": int(q.reachable.sum()), "unreachable_queries": int((~q.reachable).sum()),
        "formula": "score_i = b_i + r_i; r_i = w_q * (1 - H_i/ln(2)) * clip(2*pbar_i - 1,-1,1) * v_i",
        "uncapped_meaning": "The explicit selected/tuned kappa multiplier is absent and the formula adds r_i directly; no kappa sweep choice or residual clipping is applied. This is not an unbounded perturbation: since w_q, 1-H/ln2 and v_i are in [0,1] and the centered probability factor is in [-1,1], |r_i|<=1 naturally.",
        "protocol": "Reuses R1-1 A0 outer-fold posterior predictions, residual factors, folds, candidate pool, features, fitted regularization, and ranking tie rule. No fit, hyperparameter search, or test-qrel selection was performed.",
        "heldout_outcomes_used_for_model_selection": False,
        "paired_interval": "query-level percentile bootstrap; CVaR10 tail recomputed per replicate",
        "bootstrap_reps": BOOT_REPS, "bootstrap_seed": BOOT_SEED,
        "max_abs_residual_unit": max_abs_residual,
        "A0_replay_ndcg10": float(q.A0_ndcg10.mean()),
    }
    (out / "M2_NO_CAP_ABLATION_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(pd.DataFrame(effect_rows).to_string(index=False))
    print("WROTE", out)


if __name__ == "__main__":
    main()
