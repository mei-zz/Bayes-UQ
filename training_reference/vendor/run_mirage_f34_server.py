#!/usr/bin/env python3
"""Fit locked MIRAGE methods and write label-blind test rankings on the server."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "32")
os.environ.setdefault("MKL_NUM_THREADS", "32")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import expit, logsumexp


FEATURES = ["text_score", "visual_score", "text_rank_score", "visual_rank_score",
            "text_visual_gap", "text_visual_product", "candidate_token_count"]
TEXT_EAU = ["text_score", "text_rank_score", "query_text_top12_gap", "query_text_mean",
            "query_text_std", "question_token_count", "candidate_token_count"]
VISUAL_EAU = ["visual_score", "visual_rank_score", "text_visual_gap", "text_visual_product",
              "query_visual_top12_gap", "query_visual_mean", "query_visual_std", "image_available"]
EAU_FEATURES = TEXT_EAU + VISUAL_EAU
LAMBDA = 1.0
RHO = 2.0
ALPHA = 0.10
TAU_DOWNSIDE = 0.05
TAU_TAIL = 0.05
TEMPERATURE = 0.5
DRAWS = 128
POINT_C_GRID = [0.01, 0.1, 1.0, 10.0]
EAU_CONFIGS = [
    {"learning_rate": 1e-4, "hidden_dim": 32, "dropout": 0.1, "weight_decay": 1e-5},
    {"learning_rate": 1e-4, "hidden_dim": 64, "dropout": 0.1, "weight_decay": 1e-5},
    {"learning_rate": 3e-4, "hidden_dim": 32, "dropout": 0.1, "weight_decay": 1e-5},
    {"learning_rate": 3e-4, "hidden_dim": 64, "dropout": 0.1, "weight_decay": 1e-5},
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def stable_seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")


def d10(rank: int | float) -> float:
    r = int(rank)
    return 1.0 / math.log2(r + 1.0) if 1 <= r <= 10 else 0.0


def metric_at_rank(rank: int, cutoff: int = 10) -> tuple[float, float, float]:
    if rank < 1 or rank > 50:
        return 0.0, 0.0, 0.0
    ndcg = d10(rank)
    mrr = 1.0 / rank if rank <= cutoff else 0.0
    recall = 1.0 if rank <= cutoff else 0.0
    return ndcg, mrr, recall


def make_pages(root: Path) -> tuple[dict[str, int], list[list[int]]]:
    train_queries = read_jsonl(root / "inputs" / "queries_train_selected.jsonl")
    q_page = {str(row["query_id"]): int(row["source_page_offset"]) for row in train_queries}
    pages = sorted(set(q_page.values()))
    shuffled = np.random.Generator(np.random.PCG64(20261010)).permutation(np.asarray(pages, dtype=np.int64))
    folds = [np.asarray(chunk, dtype=np.int64).tolist() for chunk in np.array_split(shuffled, 5)]
    if len(pages) != 50 or any(len(fold) != 10 for fold in folds):
        raise RuntimeError("fixed train-page grouping changed")
    return q_page, folds


def load_grouped_features(path: Path, query_path: Path, *, training: bool) -> dict:
    frame = pd.read_csv(path, dtype={"query_id": str, "candidate_id": str, "doc_id": str})
    queries = read_jsonl(query_path)
    meta = {str(q["query_id"]): q for q in queries}
    if set(frame.query_id.astype(str)) != set(meta):
        raise RuntimeError(f"feature/query IDs mismatch in {path.name}")
    if training:
        if "label" not in frame.columns or not set(frame.label.unique()).issubset({0, 1}):
            raise RuntimeError("training features lack binary labels")
    else:
        if "label" in frame.columns:
            raise RuntimeError("test labels leaked into the prediction feature file")
        forbidden = {"target_doc_id", "answer", "relevance", "label", "target_rank_text"}
        if forbidden & set(frame.columns):
            raise RuntimeError(f"test feature file contains forbidden labels: {sorted(forbidden & set(frame.columns))}")
    qids = sorted(meta, key=lambda x: int(x))
    x = np.zeros((len(qids), 50, len(FEATURES)), dtype=np.float64)
    ids = np.zeros((len(qids), 50), dtype=np.int64)
    base = np.zeros((len(qids), 50), dtype=np.float64)
    y = np.zeros((len(qids), 50), dtype=np.int8) if training else None
    pages = np.zeros(len(qids), dtype=np.int64)
    for qi, qid in enumerate(qids):
        group = frame[frame.query_id.astype(str).eq(qid)].copy()
        group["_doc"] = group.doc_id.astype(np.int64)
        group = group.sort_values("_doc", kind="mergesort")
        if len(group) != 50 or group.doc_id.astype(str).duplicated().any():
            raise RuntimeError(f"query {qid} does not have exactly 50 unique candidates")
        ids[qi] = group.doc_id.astype(np.int64).to_numpy()
        base[qi] = group.text_score.astype(float).to_numpy()
        x[qi] = group[FEATURES].to_numpy(dtype=np.float64)
        if training:
            y[qi] = group.label.astype(np.int8).to_numpy()
            if int(y[qi].sum()) != 1:
                raise RuntimeError(f"training query {qid} does not have one reachable target")
        pages[qi] = int(meta[qid]["source_page_offset"])
        if not np.array_equal(np.argsort(-base[qi], kind="stable"), np.argsort(group.text_rank.to_numpy())):
            # Equal-score ties are resolved by document ID in the locked pool. Verify exact ranks below.
            ranks_by_doc = group.assign(_rank=group.text_rank.astype(int)).set_index("_doc")["_rank"]
            ordered = sorted(range(50), key=lambda j: (-float(base[qi, j]), int(ids[qi, j])))
            computed = {int(ids[qi, pos]): rank for rank, pos in enumerate(ordered, start=1)}
            if any(computed[int(doc)] != int(ranks_by_doc.loc[int(doc)]) for doc in ids[qi]):
                raise RuntimeError(f"frozen BGE score/order mismatch for query {qid}")
    if not np.isfinite(x).all() or not np.isfinite(base).all():
        raise RuntimeError("non-finite server candidate features")
    return {"qids": qids, "meta": meta, "x": x, "ids": ids, "base": base, "y": y, "pages": pages, "frame": frame}


def query_gold_indices(y: np.ndarray) -> np.ndarray:
    return y.argmax(axis=1).astype(np.int64)


def standardize_candidates(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = x.reshape(-1, x.shape[-1])
    mean = flat.mean(axis=0, dtype=np.float64)
    scale = flat.std(axis=0, dtype=np.float64)
    scale[scale < 1e-12] = 1.0
    return ((x - mean) / scale), mean, scale


def make_f34_objective(x: np.ndarray, gold: np.ndarray, base_ndcg: np.ndarray, rho: float):
    n, c, p = x.shape
    gold_x = x[np.arange(n), gold]
    mask_other = np.ones((n, c), dtype=bool)
    mask_other[np.arange(n), gold] = False
    dim = p + (1 if rho > 0.0 else 0)

    def evaluate(theta: np.ndarray):
        beta = theta[:p]
        logits = np.einsum("ncp,p->nc", x, beta)
        logz = logsumexp(logits, axis=1)
        probs = np.exp(logits - logz[:, None])
        chosen = logits[np.arange(n), gold]
        objective = float(np.sum(logz - chosen) + 0.5 * LAMBDA * np.dot(beta, beta))
        grad_beta = np.einsum("ncp,nc->p", x, probs) - np.sum(gold_x, axis=0) + LAMBDA * beta
        if rho <= 0.0:
            return objective, grad_beta
        eta = float(theta[p])
        margins = expit((logits - chosen[:, None]) / TEMPERATURE)
        margins = np.where(mask_other, margins, 0.0)
        soft_rank = 1.0 + margins.sum(axis=1)
        log_rank = np.log(soft_rank + 1.0)
        soft_ndcg = math.log(2.0) / log_rank
        raw_downside = base_ndcg - soft_ndcg
        losses = TAU_DOWNSIDE * np.logaddexp(0.0, raw_downside / TAU_DOWNSIDE)
        dloss_drank = math.log(2.0) / ((soft_rank + 1.0) * log_rank ** 2)
        dloss_draw = expit(raw_downside / TAU_DOWNSIDE) * dloss_drank
        margin_grad = margins * (1.0 - margins) / TEMPERATURE
        margin_grad = np.where(mask_other, margin_grad, 0.0)
        drank_dbeta = np.einsum("nc,ncp->np", margin_grad, x) - margin_grad.sum(axis=1)[:, None] * gold_x
        dloss_dbeta = dloss_draw[:, None] * drank_dbeta
        tail_arg = (losses - eta) / TAU_TAIL
        tail_weight = expit(tail_arg)
        tail_risk = eta + (TAU_TAIL / (ALPHA * n)) * float(np.logaddexp(0.0, tail_arg).sum())
        objective += rho * n * tail_risk
        grad_beta += (rho / ALPHA) * np.einsum("n,np->p", tail_weight, dloss_dbeta)
        grad_eta = rho * (n - float(tail_weight.sum()) / ALPHA)
        return objective, np.r_[grad_beta, grad_eta]

    return evaluate, dim


def finite_difference_hessian(fn, theta: np.ndarray) -> np.ndarray:
    p = len(theta)
    h = 2e-4
    out = np.empty((p, p), dtype=np.float64)
    for j in range(p):
        step = np.zeros(p, dtype=np.float64)
        step[j] = h
        out[:, j] = (fn(theta + step)[1] - fn(theta - step)[1]) / (2.0 * h)
    return (out + out.T) * 0.5


def fit_f34(x_raw: np.ndarray, y_matrix: np.ndarray, rho: float, initial_beta: np.ndarray | None = None):
    x, mean, scale = standardize_candidates(x_raw)
    gold = query_gold_indices(y_matrix)
    base = np.empty(len(gold), dtype=np.float64)
    for qi, target_idx in enumerate(gold):
        base_rank = 1 + sum((x_raw[qi, j, 0] > x_raw[qi, target_idx, 0]) or
                            (x_raw[qi, j, 0] == x_raw[qi, target_idx, 0] and j < target_idx)
                            for j in range(50))
        # Candidate columns are ordered by numeric document ID. Ranking ties in the actual pool use that ID.
        base[qi] = d10(base_rank)
    fn, dim = make_f34_objective(x, gold, base, rho)
    if initial_beta is None:
        start = np.zeros(dim, dtype=np.float64)
        if rho > 0:
            start[-1] = 0.05
    else:
        start = np.r_[initial_beta, 0.05] if rho > 0 else initial_beta.copy()
    result = minimize(fn, start, jac=True, method="L-BFGS-B",
                      options={"maxiter": 700, "ftol": 1e-11, "gtol": 1e-7, "maxls": 60, "maxcor": 20})
    grad = fn(result.x)[1]
    grad_inf = float(np.linalg.norm(grad, ord=np.inf))
    grad_per_query = grad_inf / max(len(gold), 1)
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all() or grad_per_query > 1e-4:
        raise RuntimeError(f"F3.4 rho={rho} did not converge: success={result.success}, status={result.status}, grad/query={grad_per_query:.5g}")
    hessian = finite_difference_hessian(fn, result.x)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    floor = max(LAMBDA, 1e-6)
    covariance = (eigenvectors * (1.0 / np.maximum(eigenvalues, floor))) @ eigenvectors.T
    receipt = {"rho": rho, "queries": len(gold), "candidate_rows": int(x_raw.shape[0] * x_raw.shape[1]),
               "objective": float(result.fun), "success": bool(result.success), "status": int(result.status),
               "iterations": int(result.nit), "gradient_inf_norm": grad_inf,
               "gradient_inf_norm_per_query": grad_per_query,
               "hessian_eigen_min_raw": float(eigenvalues.min()), "hessian_eigen_max_raw": float(eigenvalues.max()),
               "hessian_eigenvalue_floor": floor,
               "hessian_nonpositive_eigenvalues": int(np.sum(eigenvalues <= 0)),
               "hessian_clipped_fraction": float(np.mean(eigenvalues <= floor))}
    return result.x, covariance, mean, scale, receipt


def f34_probabilities(x_raw: np.ndarray, params: tuple, draws: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    theta, cov, mean, scale = params
    x = (x_raw - mean) / scale
    p = x.shape[-1]
    if draws is None:
        logits = np.einsum("qcp,p->qc", x, theta[:p])
        return np.exp(logits - logsumexp(logits, axis=1, keepdims=True)), None
    logits = np.einsum("qcp,dp->dqc", x, draws[:, :p])
    probs = np.exp(logits - logsumexp(logits, axis=2, keepdims=True))
    return probs.mean(axis=0), probs


def candidate_order(scores: np.ndarray, ids: np.ndarray) -> list[int]:
    return sorted(range(len(ids)), key=lambda j: (-float(scores[j]), int(ids[j])))


def rank_ndcg_matrix(orders: list[list[int]], y: np.ndarray) -> np.ndarray:
    values = np.zeros(len(orders), dtype=np.float64)
    for qi, order in enumerate(orders):
        target = int(np.argmax(y[qi]))
        rank = order.index(target) + 1
        values[qi] = d10(rank)
    return values


def fit_pointwise(x_train: np.ndarray, y_train: np.ndarray, c_value: float):
    x = x_train.reshape(-1, x_train.shape[-1]).astype(np.float64)
    y = y_train.reshape(-1).astype(np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    xs = (x - mean) / scale
    design = np.column_stack([np.ones(len(xs)), xs])
    p = design.shape[1]
    penalty = np.ones(p, dtype=np.float64)
    penalty[0] = 0.0

    def objective(w):
        logits = design @ w
        probs = expit(logits)
        value = float(np.logaddexp(0.0, logits).sum() - np.dot(y, logits) + 0.5 / c_value * np.dot(w * penalty, w))
        grad = design.T @ (probs - y) + (w * penalty) / c_value
        return value, grad

    result = minimize(objective, np.zeros(p), jac=True, method="L-BFGS-B",
                      options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8, "maxls": 50})
    grad = objective(result.x)[1]
    if not np.isfinite(result.x).all() or np.linalg.norm(grad, ord=np.inf) / len(y) > 1e-5:
        raise RuntimeError(f"Pointwise logistic model failed to converge for C={c_value}")
    return {"mean": mean, "scale": scale, "weights": result.x, "C": c_value,
            "iterations": int(result.nit), "gradient_inf_per_row": float(np.linalg.norm(grad, ord=np.inf) / len(y))}


def pointwise_predict(model: dict, x: np.ndarray) -> np.ndarray:
    shape = x.shape[:2]
    xs = (x.reshape(-1, x.shape[-1]) - model["mean"]) / model["scale"]
    logits = model["weights"][0] + xs @ model["weights"][1:]
    return expit(logits).reshape(shape)


def add_eau_features(data: dict, token_counts: dict[str, int]) -> np.ndarray:
    x = data["x"]
    n, c, _ = x.shape
    cols = {name: x[:, :, i] for i, name in enumerate(FEATURES)}
    text = cols["text_score"]
    visual = cols["visual_score"]
    text_top = np.sort(text, axis=1)[:, ::-1]
    visual_top = np.sort(visual, axis=1)[:, ::-1]
    q_tokens = np.asarray([int(token_counts[qid]) for qid in data["qids"]], dtype=np.float64)
    q_features = {
        "query_text_top12_gap": np.repeat((text_top[:, 0] - text_top[:, 1])[:, None], c, axis=1),
        "query_text_mean": np.repeat(text.mean(axis=1)[:, None], c, axis=1),
        "query_text_std": np.repeat(text.std(axis=1)[:, None], c, axis=1),
        "query_visual_top12_gap": np.repeat((visual_top[:, 0] - visual_top[:, 1])[:, None], c, axis=1),
        "query_visual_mean": np.repeat(visual.mean(axis=1)[:, None], c, axis=1),
        "query_visual_std": np.repeat(visual.std(axis=1)[:, None], c, axis=1),
        "question_token_count": np.repeat(q_tokens[:, None], c, axis=1),
        "image_available": np.ones((n, c), dtype=np.float64),
    }
    all_cols = {**cols, **q_features}
    return np.stack([all_cols[name] for name in EAU_FEATURES], axis=2)


def eau_metric(probabilities: np.ndarray, y: np.ndarray, ids: np.ndarray) -> float:
    orders = [candidate_order(probabilities[i], ids[i]) for i in range(len(ids))]
    return float(rank_ndcg_matrix(orders, y).mean())


def train_eau_model(x_all: np.ndarray, y_all: np.ndarray, train_qidx: np.ndarray, val_qidx: np.ndarray,
                    ids: np.ndarray, config: dict, seed: int, max_epochs: int = 25,
                    patience: int = 5, fixed_epochs: int | None = None):
    from r3_3_eau_adapter import EAUNet, train_step

    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("Adapted EAU requires the server CUDA device")
    torch.cuda.set_device(0)
    np_rng = np.random.Generator(np.random.PCG64(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_rows = (train_qidx[:, None] * 50 + np.arange(50)[None, :]).reshape(-1)
    val_rows = (val_qidx[:, None] * 50 + np.arange(50)[None, :]).reshape(-1) if len(val_qidx) else np.asarray([], dtype=np.int64)
    flat_x = x_all.reshape(-1, x_all.shape[-1]).astype(np.float64)
    mean = flat_x[train_rows].mean(axis=0)
    scale = flat_x[train_rows].std(axis=0)
    scale[scale < 1e-10] = 1.0
    xs = ((flat_x - mean) / scale).astype(np.float32)
    xt = torch.as_tensor(xs, dtype=torch.float32, device="cuda:0")
    yt = torch.as_tensor(y_all.reshape(-1).astype(np.int64), dtype=torch.long, device="cuda:0")
    text_indices = [EAU_FEATURES.index(name) for name in TEXT_EAU]
    visual_indices = [EAU_FEATURES.index(name) for name in VISUAL_EAU]
    text_x, visual_x = xt[:, text_indices], xt[:, visual_indices]
    train_counts = np.bincount(y_all[train_qidx].reshape(-1), minlength=2).astype(np.float64)
    if np.any(train_counts == 0):
        raise RuntimeError("EAU training split must contain both target and nontarget candidates")
    class_weights = torch.as_tensor(len(train_rows) / (2.0 * train_counts), dtype=torch.float32, device="cuda:0")
    model = EAUNet(len(text_indices), len(visual_indices), int(config["hidden_dim"]), float(config["dropout"])).to("cuda:0")
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))

    def predict_rows(rows: np.ndarray) -> np.ndarray:
        model.eval()
        values = []
        with torch.inference_mode():
            for start in range(0, len(rows), 8192):
                ix = torch.as_tensor(rows[start:start + 8192], dtype=torch.long, device="cuda:0")
                out = model(text_x.index_select(0, ix), visual_x.index_select(0, ix))
                values.append(torch.softmax(out["fusion_logits"], dim=1)[:, 1].float().cpu().numpy())
        return np.concatenate(values) if values else np.asarray([], dtype=np.float64)

    epochs = int(fixed_epochs or max_epochs)
    best_epoch, best_metric, best_state, stale = epochs, -float("inf"), None, 0
    trace = []
    for epoch in range(epochs):
        model.train()
        order = np_rng.permutation(train_rows)
        losses = []
        for start in range(0, len(order), 512):
            selected = order[start:start + 512]
            idx = torch.as_tensor(selected, dtype=torch.long, device="cuda:0")
            mask = torch.ones(len(selected), dtype=torch.bool, device="cuda:0")
            loss, fit_loss = train_step(model, optimizer, text_x.index_select(0, idx), visual_x.index_select(0, idx),
                                        yt.index_select(0, idx), class_weights, epoch=epoch, zeta=0.01,
                                        candidate_mask=mask)
            if not np.isfinite(loss) or not np.isfinite(fit_loss):
                raise FloatingPointError("Adapted EAU returned non-finite loss")
            losses.append(loss)
        record = {"epoch": epoch + 1, "mean_loss": float(np.mean(losses))}
        if fixed_epochs is None:
            val_prob = predict_rows(val_rows).reshape(len(val_qidx), 50)
            metric = eau_metric(val_prob, y_all[val_qidx], ids[val_qidx])
            record["validation_ndcg10"] = metric
            if metric > best_metric + 1e-14:
                best_metric, best_epoch = metric, epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            trace.append(record)
            if stale >= patience:
                break
        else:
            trace.append(record)
    if fixed_epochs is None:
        if best_state is None:
            raise RuntimeError("EAU validation failed to select a checkpoint")
        model.load_state_dict(best_state)
        probabilities = predict_rows(val_rows).reshape(len(val_qidx), 50)
        selected_metric = best_metric
    else:
        probabilities = np.asarray([], dtype=np.float64)
        selected_metric = float("nan")
        best_epoch = epochs
    receipt = {"config": config, "seed": seed, "best_epoch": int(best_epoch),
               "best_validation_ndcg10": selected_metric, "trace": trace,
               "class_counts": train_counts.astype(int).tolist(),
               "train_queries": int(len(train_qidx)), "validation_queries": int(len(val_qidx)),
               "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(0)),
               "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
               "scaler_mean": mean.tolist(), "scaler_scale": scale.tolist()}
    return model, probabilities, receipt


def predict_eau(model, x_raw: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    from r3_3_eau_adapter import EAUNet
    flat = ((x_raw.reshape(-1, x_raw.shape[-1]) - mean) / scale).astype(np.float32)
    xt = torch.as_tensor(flat, dtype=torch.float32, device="cuda:0")
    text_indices = [EAU_FEATURES.index(name) for name in TEXT_EAU]
    visual_indices = [EAU_FEATURES.index(name) for name in VISUAL_EAU]
    text_x, visual_x = xt[:, text_indices], xt[:, visual_indices]
    model.eval()
    out = []
    with torch.inference_mode():
        for start in range(0, len(flat), 8192):
            stop = min(start + 8192, len(flat))
            pred = model(text_x[start:stop], visual_x[start:stop])
            out.append(torch.softmax(pred["fusion_logits"], dim=1)[:, 1].float().cpu().numpy())
    return np.concatenate(out).reshape(x_raw.shape[:2])


def verify_f34_gradient() -> dict:
    rng = np.random.default_rng(20261010)
    n, c, p = 8, 5, 3
    x = rng.normal(size=(n, c, p))
    gold = np.asarray([0, 1, 2, 3, 4, 0, 2, 1], dtype=np.int64)
    base = np.asarray([1.0, 0.5, 1 / math.log2(4), 0.0, 1 / math.log2(6), 1.0, 0.5, 1 / math.log2(3)])
    errors = {}
    for rho in (0.0, 2.0):
        fn, dim = make_f34_objective(x, gold, base, rho)
        theta = rng.normal(scale=0.1, size=dim)
        analytic = fn(theta)[1]
        numeric = np.empty(dim)
        h = 1e-6
        for j in range(dim):
            d = np.zeros(dim); d[j] = h
            numeric[j] = (fn(theta + d)[0] - fn(theta - d)[0]) / (2 * h)
        err = float(np.max(np.abs(analytic - numeric) / np.maximum(1.0, np.abs(numeric))))
        errors[str(rho)] = err
        if not np.isfinite(err) or err > 5e-5:
            raise RuntimeError(f"F3.4 objective gradient check failed for rho={rho}: {err}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    features_dir = root / "features"
    pred_dir = root / "predictions"
    model_dir = root / "checkpoints"
    logs_dir = root / "logs"
    pred_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    lock_path = root / "PREDICTION_LOCK.json"
    if lock_path.exists():
        raise RuntimeError("prediction lock already exists; refusing to rerun or overwrite test orders")
    if not torch.cuda.is_available():
        raise RuntimeError("model run must use the authorized server GPU")
    if not (features_dir / "feature_manifest_server.json").is_file():
        raise FileNotFoundError("server-side CLIP feature manifest is missing")
    if (root / "evaluator_only").exists():
        raise RuntimeError("evaluator-only labels were transferred before the prediction lock")

    started = time.perf_counter()
    torch.cuda.set_device(0)
    torch.set_num_threads(8)
    torch.cuda.reset_peak_memory_stats()
    train = load_grouped_features(features_dir / "candidate_features_train.csv",
                                  root / "inputs" / "queries_train_selected.jsonl", training=True)
    test = load_grouped_features(features_dir / "candidate_features_test_blind.csv",
                                 root / "inputs" / "queries_test_selected.jsonl", training=False)
    if len(train["qids"]) != 1000 or len(test["qids"]) != 2000:
        raise RuntimeError("fixed train/test query counts changed")
    token_counts = json.loads((features_dir / "query_token_counts_blind.json").read_text(encoding="utf-8"))
    if set(token_counts) != set(train["qids"]) | set(test["qids"]):
        raise RuntimeError("question-token count manifest differs from fixed query IDs")
    x_train, y_train, x_test = train["x"], train["y"], test["x"]
    fold_pages = [set(map(int, fold)) for fold in make_pages(root)[1]]
    cv_receipts = []

    # Fixed five-fold, page-grouped train-only validation for F3.4 fit diagnostics.
    for fold, val_pages in enumerate(fold_pages):
        val_idx = np.flatnonzero(np.isin(train["pages"], list(val_pages)))
        tr_idx = np.flatnonzero(~np.isin(train["pages"], list(val_pages)))
        beta0, _, _, _, rec0 = fit_f34(x_train[tr_idx], y_train[tr_idx], rho=0.0)
        theta2, cov2, mean2, scale2, rec2 = fit_f34(x_train[tr_idx], y_train[tr_idx], rho=RHO, initial_beta=beta0)
        p_val, _ = f34_probabilities(x_train[val_idx], (theta2, cov2, mean2, scale2))
        base_orders = [candidate_order(train["base"][qi], train["ids"][qi]) for qi in val_idx]
        risk_orders = [candidate_order(p_val[i], train["ids"][qi]) for i, qi in enumerate(val_idx)]
        cv_receipts.append({"fold": fold, "validation_pages": sorted(val_pages),
                            "train_queries": int(len(tr_idx)), "validation_queries": int(len(val_idx)),
                            "frozen_bge_ndcg10": float(rank_ndcg_matrix(base_orders, y_train[val_idx]).mean()),
                            "f34_rho2_mean_ndcg10": float(rank_ndcg_matrix(risk_orders, y_train[val_idx]).mean()),
                            "rho0_fit": rec0, "rho2_fit": rec2})
    atomic_json(logs_dir / "f34_internal_validation.json", {"status": "TRAIN_ONLY_DIAGNOSTIC", "folds": cv_receipts})

    # Train-only Pointwise logistic reranker; regularization C selected by page-grouped 5-fold nDCG@10.
    point_cv = []
    for c_value in POINT_C_GRID:
        fold_values = []
        for fold, val_pages in enumerate(fold_pages):
            val_idx = np.flatnonzero(np.isin(train["pages"], list(val_pages)))
            tr_idx = np.flatnonzero(~np.isin(train["pages"], list(val_pages)))
            point_model = fit_pointwise(x_train[tr_idx], y_train[tr_idx], c_value)
            val_scores = pointwise_predict(point_model, x_train[val_idx])
            orders = [candidate_order(val_scores[i], train["ids"][qi]) for i, qi in enumerate(val_idx)]
            fold_values.append(float(rank_ndcg_matrix(orders, y_train[val_idx]).mean()))
        point_cv.append({"C": c_value, "fold_ndcg10": fold_values, "mean_ndcg10": float(np.mean(fold_values))})
    point_c = sorted(point_cv, key=lambda r: (-r["mean_ndcg10"], r["C"]))[0]["C"]
    point_model = fit_pointwise(x_train, y_train, point_c)
    point_test = pointwise_predict(point_model, x_test)

    # Adapted EAU: the existing two-view architecture and published R3-3 training grid,
    # with a deterministic 40/10 page-group holdout for configuration/epoch selection.
    eau_x_train = add_eau_features(train, token_counts)
    eau_x_test = add_eau_features(test, token_counts)
    eau_val_pages = fold_pages[0]
    eau_val_idx = np.flatnonzero(np.isin(train["pages"], list(eau_val_pages)))
    eau_tr_idx = np.flatnonzero(~np.isin(train["pages"], list(eau_val_pages)))
    eau_trials = []
    for ci, config in enumerate(EAU_CONFIGS):
        _, _, receipt = train_eau_model(eau_x_train, y_train, eau_tr_idx, eau_val_idx, train["ids"], config,
                                        seed=stable_seed(f"MIRAGE-EAU-VAL|{ci}|20261010"), max_epochs=25, patience=5)
        eau_trials.append(receipt)
        print(json.dumps({"event": "eau_validation_complete", "config_index": ci,
                          "best_epoch": receipt["best_epoch"], "best_validation_ndcg10": receipt["best_validation_ndcg10"]}, sort_keys=True), flush=True)
    selected_trial = sorted(enumerate(eau_trials), key=lambda x: (-x[1]["best_validation_ndcg10"], x[0]))[0]
    eau_config_index, eau_selected = selected_trial
    eau_config = EAU_CONFIGS[eau_config_index]
    # Refit the selected EAU architecture on all 1,000 adaptation queries for exactly the train-selected epoch count.
    eau_model, _, eau_final_receipt = train_eau_model(eau_x_train, y_train, np.arange(len(train["qids"])),
                                                       np.asarray([], dtype=np.int64), train["ids"], eau_config,
                                                       seed=stable_seed("MIRAGE-EAU-FINAL|20261010"), max_epochs=25,
                                                       fixed_epochs=int(eau_selected["best_epoch"]))
    eau_mean = np.asarray(eau_final_receipt["scaler_mean"], dtype=np.float64)
    eau_scale = np.asarray(eau_final_receipt["scaler_scale"], dtype=np.float64)
    eau_test = predict_eau(eau_model, eau_x_test, eau_mean, eau_scale)
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in eau_model.state_dict().items()},
                "config": eau_config, "features": EAU_FEATURES, "scaler_mean": eau_mean, "scaler_scale": eau_scale,
                "selected_epoch": int(eau_selected["best_epoch"])}, model_dir / "adapted_eau.pt")

    # Final F3.4 full-training fit. rho=0 supplies a deterministic MAP point control;
    # rho=2 supplies Always posterior-mean and the fixed Q05 selective policy.
    theta0, cov0, mean0, scale0, receipt0 = fit_f34(x_train, y_train, rho=0.0)
    theta2, cov2, mean2, scale2, receipt2 = fit_f34(x_train, y_train, rho=RHO, initial_beta=theta0)
    draws0_rng = np.random.Generator(np.random.PCG64(stable_seed("Frontier-F3.4-MIRAGE|rho0|full|128")))
    draws2_rng = np.random.Generator(np.random.PCG64(stable_seed("Frontier-F3.4-MIRAGE|rho2|full|128")))
    draws0 = draws0_rng.multivariate_normal(theta0, cov0, size=DRAWS, check_valid="raise")
    draws2 = draws2_rng.multivariate_normal(theta2, cov2, size=DRAWS, check_valid="raise")
    map0_test, _ = f34_probabilities(x_test, (theta0, cov0, mean0, scale0))
    bma0_test, _ = f34_probabilities(x_test, (theta0, cov0, mean0, scale0), draws0)
    bma2_test, posterior_probs2 = f34_probabilities(x_test, (theta2, cov2, mean2, scale2), draws2)
    bge_orders = [candidate_order(test["base"][i], test["ids"][i]) for i in range(len(test["qids"]))]
    bma_orders = [candidate_order(bma2_test[i], test["ids"][i]) for i in range(len(test["qids"]))]
    q05_values = np.zeros(len(test["qids"]), dtype=np.float64)
    q05_draw_gains = np.zeros((len(test["qids"]), DRAWS), dtype=np.float64)
    q05_accept = np.zeros(len(test["qids"]), dtype=bool)
    for qi in range(len(test["qids"])):
        action_discount = np.zeros(50, dtype=np.float64)
        base_discount = np.zeros(50, dtype=np.float64)
        for rank, ci in enumerate(bma_orders[qi], start=1):
            action_discount[ci] = d10(rank)
        for rank, ci in enumerate(bge_orders[qi], start=1):
            base_discount[ci] = d10(rank)
        gain_by_draw = posterior_probs2[:, qi, :] @ (action_discount - base_discount)
        q05_draw_gains[qi] = gain_by_draw
        q05_values[qi] = float(np.quantile(gain_by_draw, 0.05, method="linear"))
        q05_accept[qi] = q05_values[qi] > 0.0

    point_orders = [candidate_order(point_test[i], test["ids"][i]) for i in range(len(test["qids"]))]
    eau_orders = [candidate_order(eau_test[i], test["ids"][i]) for i in range(len(test["qids"]))]
    map0_orders = [candidate_order(map0_test[i], test["ids"][i]) for i in range(len(test["qids"]))]
    bma0_orders = [candidate_order(bma0_test[i], test["ids"][i]) for i in range(len(test["qids"]))]
    q05_orders = [bma_orders[i] if q05_accept[i] else bge_orders[i] for i in range(len(test["qids"]))]

    methods = {
        "Frozen BGE": (test["base"], bge_orders, "always"),
        "Always posterior-mean": (bma2_test, bma_orders, "always"),
        "Enhanced F3.4 + Q05": (bma2_test, q05_orders, "q05"),
        "Pointwise logistic reranker": (point_test, point_orders, "always"),
        "Adapted EAU": (eau_test, eau_orders, "always"),
        "F3.4 rho=0 MAP": (map0_test, map0_orders, "always"),
        "F3.4 rho=0 posterior-mean": (bma0_test, bma0_orders, "always"),
    }
    score_path = pred_dir / "candidate_scores_locked.csv.gz"
    gain_path = pred_dir / "q05_posterior_gain_draws.csv.gz"
    gain_rows = [{"query_id": test["qids"][qi], "draw": draw, "posterior_utility_gain": float(q05_draw_gains[qi, draw])}
                 for qi in range(len(test["qids"])) for draw in range(DRAWS)]
    pd.DataFrame(gain_rows).to_csv(gain_path, index=False, compression="gzip", float_format="%.17g", lineterminator="\n")
    rows = []
    predictions = []
    for qi, qid in enumerate(test["qids"]):
        row = {"query_id": qid, "source_page_offset": int(test["pages"][qi]),
               "q05_gain_q05": float(q05_values[qi]), "q05_accepted": bool(q05_accept[qi])}
        for ci, doc in enumerate(test["ids"][qi]):
            rec = {"query_id": qid, "source_page_offset": int(test["pages"][qi]), "candidate_id": str(int(doc)),
                   "text_score": float(test["base"][qi, ci]), "q05_gain_q05": float(q05_values[qi]),
                   "q05_accepted": int(q05_accept[qi])}
            for method, (scores, _, _) in methods.items():
                safe_name = {"Frozen BGE": "frozen_bge", "Always posterior-mean": "always_posterior_mean",
                             "Enhanced F3.4 + Q05": "enhanced_f34_q05", "Pointwise logistic reranker": "pointwise_logistic",
                             "Adapted EAU": "adapted_eau", "F3.4 rho=0 MAP": "f34_rho0_map",
                             "F3.4 rho=0 posterior-mean": "f34_rho0_posterior_mean"}[method]
                rec[safe_name] = float(scores[qi, ci])
            rows.append(rec)
        orders = {method: [str(int(test["ids"][qi, idx])) for idx in order] for method, (_, order_list, _) in methods.items()
                  for order in [order_list[qi]]}
        predictions.append({"query_id": qid, "source_page_offset": int(test["pages"][qi]),
                            "q05_gain_q05": float(q05_values[qi]), "q05_accepted": bool(q05_accept[qi]),
                            "orders": orders})
    pd.DataFrame(rows).to_csv(score_path, index=False, compression="gzip", float_format="%.17g", lineterminator="\n")
    prediction_path = pred_dir / "predictions_locked.jsonl"
    prediction_path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in predictions), encoding="utf-8")

    train_manifest = {
        "status": "TRAIN_SELECTION_COMPLETE", "train_queries": len(train["qids"]),
        "train_candidates": int(len(train["qids"]) * 50), "train_page_folds": [sorted(x) for x in fold_pages],
        "f34_rho0_final": receipt0, "f34_rho2_final": receipt2,
        "f34_gradient_check_scaled_max_error": verify_f34_gradient(),
        "pointwise_logistic": {"selected_C": point_c, "grid": point_cv,
                               "mean": point_model["mean"].tolist(), "scale": point_model["scale"].tolist(),
                               "weights": point_model["weights"].tolist(), "iterations": point_model["iterations"],
                               "gradient_inf_per_row": point_model["gradient_inf_per_row"]},
        "adapted_eau": {"features": EAU_FEATURES, "text_features": TEXT_EAU, "visual_features": VISUAL_EAU,
                        "validation_page_fold": sorted(eau_val_pages), "trials": eau_trials,
                        "selected_config_index": eau_config_index, "selected_config": eau_config,
                        "final_refit": eau_final_receipt},
        "test_qrels_loaded": False,
        "selection_rule": "Point C selected only by 5-fold train-page nDCG@10; EAU config/epoch selected only on one fixed 10-page validation fold; F3.4 parameters/Q05 fixed a priori; no test labels used.",
    }
    atomic_json(logs_dir / "training_selection_manifest.json", train_manifest)
    np.savez_compressed(model_dir / "f34_parameters.npz", theta_rho0=theta0, covariance_rho0=cov0,
                        mean_rho0=mean0, scale_rho0=scale0, draws_rho0=draws0,
                        theta_rho2=theta2, covariance_rho2=cov2, mean_rho2=mean2,
                        scale_rho2=scale2, draws_rho2=draws2,
                        seed_rho0=np.asarray(stable_seed("Frontier-F3.4-MIRAGE|rho0|full|128"), dtype=np.uint64),
                        seed_rho2=np.asarray(stable_seed("Frontier-F3.4-MIRAGE|rho2|full|128"), dtype=np.uint64))
    np.savez_compressed(model_dir / "pointwise_logistic.npz", mean=point_model["mean"], scale=point_model["scale"],
                        weights=point_model["weights"], C=np.asarray(point_c))
    resource = {"hostname": platform.node(), "python": platform.python_version(), "cpu_count": os.cpu_count(),
                "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                "gpu_memory_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(0)),
                "elapsed_seconds": time.perf_counter() - started}
    try:
        resource["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
            text=True, timeout=15).strip()
    except Exception as exc:
        resource["nvidia_smi_error"] = str(exc)
    atomic_json(logs_dir / "resource_usage.json", resource)
    selection_path = logs_dir / "training_selection_manifest.json"
    features_path = features_dir / "feature_manifest_server.json"
    lock_payload = {
        "status": "PREDICTIONS_LOCKED_LABEL_BLIND", "protocol": "F34-MIRAGE-v1 + Amendment 03",
        "dataset": "MIRAGE-MMST_Standard", "train_queries": 1000, "test_queries": 2000,
        "candidate_count": 50, "test_qrels_transferred": False, "test_qrels_loaded": False,
        "methods": list(methods), "required_methods_present": ["Frozen BGE", "Always posterior-mean", "Enhanced F3.4 + Q05"],
        "q05_threshold": 0.0, "q05_posterior_quantile": 0.05, "posterior_draws": DRAWS,
        "test_pool_sha256": sha256(root / "inputs" / "candidate_pools_test.csv"),
        "test_feature_sha256": sha256(features_dir / "candidate_features_test_blind.csv"),
        "train_feature_sha256": sha256(features_dir / "candidate_features_train.csv"),
        "train_selection_manifest_sha256": sha256(selection_path),
        "feature_manifest_sha256": sha256(features_path),
        "prediction_sha256": sha256(prediction_path), "candidate_score_sha256": sha256(score_path),
        "posterior_gain_draws_sha256": sha256(gain_path),
        "model_parameters_sha256": sha256(model_dir / "f34_parameters.npz"),
        "point_model_sha256": sha256(model_dir / "pointwise_logistic.npz"),
        "adapted_eau_model_sha256": sha256(model_dir / "adapted_eau.pt"),
        "prediction_file": str(prediction_path), "candidate_score_file": str(score_path),
        "q05_accepted_queries": int(q05_accept.sum()), "q05_acceptance_rate": float(q05_accept.mean()),
        "q05_actual_order_change_rate": float(np.mean([q05_orders[i] != bge_orders[i] for i in range(len(q05_orders))])),
        "fixed_split_hash": sha256(root / "inputs" / "split_manifest.json"),
        "test_page_cluster_map_sha256": sha256(root / "inputs" / "page_cluster_manifest.json"),
        "scripts_sha256": {p.name: sha256(p) for p in sorted((root / "scripts").glob("*.py"))},
        "locked_at_unix": time.time(), "resource_usage": resource,
    }
    atomic_json(lock_path, lock_payload)
    print(json.dumps({"event": "prediction_lock_written", **lock_payload}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
