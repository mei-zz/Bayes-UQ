#!/usr/bin/env python3
"""Fit the fixed-protocol F3.4 no-null internal ablation, without test qrels."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

FEATURES = [
    "doc_cat_is_disease", "doc_cat_is_guidance", "doc_cat_is_phenology",
    "has_disease_label", "has_stage_label", "image_rank", "image_score",
    "meta_image_consistency", "meta_score", "num_images", "text_rank", "text_score",
]
LAMBDA = 1.0
RHO = 2.0
ALPHA = 0.10
TAU_DOWN = 0.05
TAU_TAIL = 0.05
SOFT_RANK_TEMPERATURE = 0.5
N_DRAWS = 128
EPS = 1e-12


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2 ** 63 - 1)


def discount(rank: int) -> float:
    return 1.0 / math.log2(rank + 1.0) if int(rank) <= 10 else 0.0


def hard_order(ids: list[str], scores: np.ndarray) -> list[int]:
    return sorted(range(len(ids)), key=lambda j: (-float(scores[j]), str(ids[j])))


def soft_loss_and_grad(X: np.ndarray, target: int, base_ndcg: float, beta: np.ndarray):
    scores = X @ beta
    others = np.asarray([j for j in range(len(scores)) if j != target], dtype=int)
    pair = expit((scores[others] - scores[target]) / SOFT_RANK_TEMPERATURE)
    soft_rank = 1.0 + float(pair.sum())
    soft_ndcg = 1.0 / math.log2(soft_rank + 1.0)
    raw = base_ndcg - soft_ndcg
    loss = TAU_DOWN * np.logaddexp(0.0, raw / TAU_DOWN)
    dloss_drank = math.log(2.0) / ((soft_rank + 1.0) * math.log(soft_rank + 1.0) ** 2)
    d_rank = ((pair * (1.0 - pair) / SOFT_RANK_TEMPERATURE)[:, None] *
              (X[others] - X[target])).sum(axis=0)
    grad = expit(raw / TAU_DOWN) * dloss_drank * d_rank
    return float(loss), grad


def objective_factory(groups: dict, train_qids: list[str], rho: float):
    p = len(FEATURES)
    n = len(train_qids)
    dim = p + (1 if rho > 0.0 else 0)

    def fn(theta: np.ndarray):
        beta = theta[:p]
        value = 0.5 * LAMBDA * float(beta @ beta)
        grad = np.zeros(dim, dtype=np.float64)
        grad[:p] = LAMBDA * beta
        losses = np.zeros(n, dtype=np.float64)
        loss_grads = np.zeros((n, p), dtype=np.float64)
        for row, qid in enumerate(train_qids):
            rec = groups[qid]
            target = int(rec["target"])
            if target < 0:
                # A no-null categorical model cannot assign an NLL target to an
                # unreachable positive. Its observed ranking change is zero,
                # so the downside-loss contribution remains the locked value 0.
                continue
            X = rec["X"]
            logits = X @ beta
            logz = float(logsumexp(logits))
            probs = np.exp(logits - logz)
            value += logz - float(logits[target])
            probs[target] -= 1.0
            grad[:p] += X.T @ probs
            losses[row], loss_grads[row] = soft_loss_and_grad(X, target, rec["base_ndcg"], beta)

        if rho > 0.0:
            eta = float(theta[p])
            tail_arg = (losses - eta) / TAU_TAIL
            tail_weight = expit(tail_arg)
            risk = eta + (TAU_TAIL / (ALPHA * n)) * float(np.logaddexp(0.0, tail_arg).sum())
            value += rho * n * risk
            grad[:p] += (rho / ALPHA) * (tail_weight[:, None] * loss_grads).sum(axis=0)
            grad[p] = rho * (n - float(tail_weight.sum()) / ALPHA)
        return float(value), grad
    return fn


def numerical_hessian(fn, theta: np.ndarray) -> np.ndarray:
    h = 2e-4
    p = len(theta)
    out = np.empty((p, p), dtype=np.float64)
    for j in range(p):
        delta = np.zeros(p, dtype=np.float64)
        delta[j] = h
        out[:, j] = (fn(theta + delta)[1] - fn(theta - delta)[1]) / (2.0 * h)
    return 0.5 * (out + out.T)


def fit_model(groups: dict, train_qids: list[str], rho: float, initial=None):
    fn = objective_factory(groups, train_qids, rho)
    p = len(FEATURES)
    dim = p + (1 if rho > 0.0 else 0)
    if initial is None:
        start = np.zeros(dim, dtype=np.float64)
        if rho > 0.0:
            start[p] = 0.05
    elif rho > 0.0 and len(initial) == p:
        start = np.r_[initial, 0.05]
    else:
        start = np.asarray(initial, dtype=np.float64).copy()
    began = time.time()
    result = minimize(fn, start, jac=True, method="L-BFGS-B",
                      options={"maxiter": 700, "ftol": 1e-11, "gtol": 1e-7, "maxls": 60, "maxcor": 20})
    grad = fn(result.x)[1]
    grad_inf = float(np.linalg.norm(grad, ord=np.inf))
    grad_per_train_query = grad_inf / max(len(train_qids), 1)
    if (not np.isfinite(result.fun) or not np.isfinite(result.x).all() or
            grad_per_train_query > 1e-4):
        raise RuntimeError("No-null optimizer convergence gate failed: success={} status={} grad/n={:.4g}".format(
            result.success, result.status, grad_per_train_query))
    hess = numerical_hessian(fn, result.x)
    eigenvalues, eigenvectors = np.linalg.eigh(hess)
    floor = max(LAMBDA, 1e-6)
    cov = (eigenvectors * (1.0 / np.maximum(eigenvalues, floor))) @ eigenvectors.T
    diagnostics = {
        "rho": rho, "objective": float(result.fun), "optimizer_success_flag": bool(result.success),
        "optimizer_status": int(result.status), "iterations": int(result.nit),
        "gradient_inf_norm": grad_inf, "gradient_inf_norm_per_all_train_query": grad_per_train_query,
        "hessian_min_eigenvalue_raw": float(eigenvalues.min()),
        "hessian_floor": floor, "hessian_clipped_fraction": float(np.mean(eigenvalues <= floor)),
        "fit_seconds": time.time() - began,
    }
    return result.x, cov, diagnostics


def candidate_probs(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    logits = X @ beta
    return np.exp(logits - logsumexp(logits))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--f34-root", default=r"E:\tmp\experiments\apin_revision_f3_4_main_method_evidence")
    parser.add_argument("--out", default=r"E:\APIN_FINAL_SCIENTIFIC_CLOSURE\data\no_null")
    args = parser.parse_args()
    f34_root = Path(args.f34_root)
    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    input_root = f34_root / "14_SOURCE_AND_HASHES" / "source_inputs"
    raw_root = f34_root / "02_CANONICAL_OOF" / "raw_locked_run"
    feature_path = input_root / "MODEL_FEATURES_NO_QRELS.csv"
    qrel_path = input_root / "QRELS_SEPARATE.csv"
    assignment_path = input_root / "OUTER_FOLD_ASSIGNMENTS.csv"
    features = pd.read_csv(feature_path, dtype={"query_id": str, "candidate_id": str})
    assignments = pd.read_csv(assignment_path, dtype={"query_id": str, "role": str})
    missing = sorted(set(FEATURES + ["base_score"]) - set(features.columns))
    if missing:
        raise ValueError("Missing locked feature columns: {}".format(missing))
    for col in FEATURES + ["base_score"]:
        features[col] = pd.to_numeric(features[col], errors="raise").astype(np.float64)
    if features["query_id"].nunique() != 483 or len(features) != 4813:
        raise ValueError("Locked candidate pool cardinality mismatch")
    all_groups = {}
    for qid, block in features.sort_values(["query_id", "candidate_id"]).groupby("query_id", sort=True):
        block = block.sort_values("candidate_id", kind="mergesort")
        all_groups[str(qid)] = {
            "ids": block["candidate_id"].astype(str).tolist(),
            "X_raw": block[FEATURES].to_numpy(np.float64),
            "base": block["base_score"].to_numpy(np.float64),
            "target": -1,
            "base_ndcg": 0.0,
        }

    prediction_rows, fit_rows, input_hashes = [], [], {
        "MODEL_FEATURES_NO_QRELS.csv": sha256(feature_path),
        "OUTER_FOLD_ASSIGNMENTS.csv": sha256(assignment_path),
        "QRELS_SEPARATE.csv": sha256(qrel_path),
    }
    for fold in range(5):
        role = assignments[assignments["outer_fold"].eq(fold)]
        train_qids = sorted(role.loc[role["role"].eq("train"), "query_id"].astype(str).unique())
        test_qids = sorted(role.loc[role["role"].eq("test"), "query_id"].astype(str).unique())
        if set(train_qids) & set(test_qids) or set(train_qids) | set(test_qids) != set(all_groups):
            raise ValueError("Outer fold is not an exact partition: {}".format(fold))

        # Read train labels only; no held-out qrel value enters fitting or ranking.
        train_set = set(train_qids)
        train_qrels = {}
        for chunk in pd.read_csv(qrel_path, dtype={"query_id": str, "candidate_id": str}, chunksize=50000):
            selected = chunk[chunk["query_id"].isin(train_set)]
            for row in selected.itertuples(index=False):
                train_qrels.setdefault(str(row.query_id), {})[str(row.candidate_id)] = int(row.relevance)
        if set(train_qrels) != train_set:
            raise ValueError("Training qrels do not cover fold {}".format(fold))

        train_groups = {}
        for qid in train_qids:
            rec = all_groups[qid]
            qmap = train_qrels[qid]
            if set(qmap) != set(rec["ids"]):
                raise ValueError("Training qrel/candidate mismatch for {}".format(qid))
            labels = np.asarray([qmap[cid] for cid in rec["ids"]], dtype=int)
            if int(labels.sum()) > 1:
                raise ValueError("One-positive-or-null target invariant failed")
            pos = np.flatnonzero(labels)
            target = int(pos[0]) if len(pos) else -1
            model_state = np.load(raw_root / "folds" / ("outer_{}".format(fold)) / "fold_model_state.npz")
            mean = model_state["X_mean"].astype(np.float64)
            std = model_state["X_std"].astype(np.float64)
            X = (rec["X_raw"] - mean) / std
            base_order = hard_order(rec["ids"], rec["base"])
            base_ndcg = discount(base_order.index(target) + 1) if target >= 0 else 0.0
            train_groups[qid] = {"ids": rec["ids"], "X": X, "target": target, "base_ndcg": base_ndcg}
        # The frozen F3.4 fold scalers are reused byte-for-byte for both arms.
        model_state = np.load(raw_root / "folds" / ("outer_{}".format(fold)) / "fold_model_state.npz")
        mean, std = model_state["X_mean"].astype(np.float64), model_state["X_std"].astype(np.float64)
        reachable_train = [q for q in train_qids if train_groups[q]["target"] >= 0]
        theta0, _, diag0 = fit_model(train_groups, train_qids, 0.0)
        theta2, cov2, diag2 = fit_model(train_groups, train_qids, RHO, initial=theta0)
        seed = stable_seed("APIN-F34-NO-NULL-ABLATION|fold={}|rho=2|draws={}".format(fold, N_DRAWS))
        rng = np.random.Generator(np.random.PCG64(seed))
        draws = rng.multivariate_normal(theta2, cov2, size=N_DRAWS, check_valid="raise")[:, :len(FEATURES)]
        for qid in test_qids:
            rec = all_groups[qid]
            X = (rec["X_raw"] - mean) / std
            ids = rec["ids"]
            base_order = hard_order(ids, rec["base"])
            draw_probs = np.asarray([candidate_probs(X, beta) for beta in draws])
            mean_probs = draw_probs.mean(axis=0)
            proposal = hard_order(ids, mean_probs)
            rank_map = {idx: rank for rank, idx in enumerate(proposal, 1)}
            base_rank = {idx: rank for rank, idx in enumerate(base_order, 1)}
            utility_delta = np.asarray([discount(rank_map[i]) - discount(base_rank[i]) for i in range(len(ids))])
            utility_draws = draw_probs @ utility_delta
            q05 = float(np.quantile(utility_draws, 0.05, method="linear"))
            accepted = int(q05 > 0.0)
            final_order = proposal if accepted else base_order
            for idx, cid in enumerate(ids):
                prediction_rows.append({
                    "outer_fold": fold, "query_id": qid, "candidate_id": cid,
                    "base_rank": int(base_rank[idx]), "no_null_proposal_rank": int(rank_map[idx]),
                    "no_null_q05_rank": int(final_order.index(idx) + 1),
                    "posterior_mean_candidate_probability": float(mean_probs[idx]),
                    "q05_accepted": accepted, "q05_gain_q05": q05,
                    "posterior_mean_utility_gain": float(utility_draws.mean()),
                    "posterior_draw_share_negative_expected_gain": float(np.mean(utility_draws < 0.0)),
                    "base_order_sha256": hashlib.sha256("|".join(ids[i] for i in base_order).encode()).hexdigest(),
                    "proposal_order_sha256": hashlib.sha256("|".join(ids[i] for i in proposal).encode()).hexdigest(),
                    "q05_order_sha256": hashlib.sha256("|".join(ids[i] for i in final_order).encode()).hexdigest(),
                })
        fit_rows.append({
            "outer_fold": fold, "train_queries_all": len(train_qids), "train_queries_reachable": len(reachable_train),
            "train_queries_unreachable": len(train_qids) - len(reachable_train),
            "test_queries": len(test_qids), "candidate_features": len(FEATURES),
            "lambda": LAMBDA, "rho": RHO, "posterior_draws": N_DRAWS, "posterior_seed": seed,
            "scaler_source": "reused exact F3.4 fold_model_state.npz X_mean/X_std",
            "rho0_fit": json.dumps(diag0, sort_keys=True), "rho2_fit": json.dumps(diag2, sort_keys=True),
            "test_qrels_read": False,
        })
        print("FOLD_COMPLETE {} reachable_train={} all_train={} test={}".format(
            fold, len(reachable_train), len(train_qids), len(test_qids)), flush=True)

    pred_path = out / "F34_NO_NULL_PREDICTIONS_LOCKED.csv"
    fit_path = out / "F34_NO_NULL_FIT_DIAGNOSTICS.csv"
    pd.DataFrame(prediction_rows).to_csv(pred_path, index=False)
    pd.DataFrame(fit_rows).to_csv(fit_path, index=False)
    manifest = {
        "status": "PREDICTIONS_LOCKED_LABEL_BLIND",
        "model": "F3.4 no-null conditional candidate categorical ablation",
        "unique_test_queries": int(pd.DataFrame(prediction_rows)["query_id"].nunique()),
        "prediction_rows": len(prediction_rows), "feature_count": len(FEATURES),
        "feature_names": FEATURES, "same_outer_folds": True,
        "same_candidate_pool_and_candidate_features": True,
        "same_train_only_scalers": True, "same_fixed_lambda_rho": {"lambda": LAMBDA, "rho": RHO},
        "no_null_training_rule": "Candidate-only categorical NLL is fit on reachable training queries; unreachable training queries have no candidate target and are excluded from the NLL. Their fixed zero downside losses remain in the same full-training-query RU denominator. No null logit/context is fitted.",
        "q05_rule": "Strictly positive linear empirical fifth percentile over 128 approximate posterior expected-gain draws; threshold fixed at zero.",
        "test_qrels_read": False, "test_qrels_used_for_model_selection": False,
        "inputs_sha256": input_hashes,
        "predictions_sha256": sha256(pred_path), "fit_diagnostics_sha256": sha256(fit_path),
        "script_sha256": sha256(Path(__file__)),
    }
    (out / "F34_NO_NULL_PREDICTION_LOCK.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
