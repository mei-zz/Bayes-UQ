#!/usr/bin/env python3
"""Frontier-F3.3: modality-agreement contextual utility with a posterior interaction.

The only outputs are development-side inner-validation artifacts under the new
Frontier directory. No outer prediction or outer metric is created or read.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

ROOT = Path('/home/ubuntu/apin_revision_bayes_uq_frontier')
INPUTS = ROOT / '03_DATA_AND_PROTOCOL' / 'inputs'
TRACK = ROOT / '09_TRACK_F_AGRICULTURAL_ROBUSTNESS'
RUN = TRACK / 'v3_3'
PRED_DIR = RUN / 'predictions' / 'inner_splits'
TABLES = RUN / 'tables_data'
METRIC_SPLITS = TABLES / 'query_metrics_by_split'
RISK_SPLITS = TABLES / 'posterior_risk_by_split'
COST_SPLITS = TABLES / 'fit_diagnostics_by_split'
CHECKPOINTS = RUN / 'checkpoints'
CONFIG_PATH = RUN / 'config.json'
L2_GRID = (1.0,)
RHO_GRID = (0.0, 2.0)
ALPHA = 0.10
TEMPERATURE = 0.5
TAU_DOWNSIDE = 0.05
TAU_TAIL = 0.05
GRADIENT_TOL_PER_QUERY = 1e-4
GRADIENT_TOL_PER_CANDIDATE = 1e-4
POSTERIOR_DRAWS = 128
EPS = 1e-12
NULL_RAW_NAMES = (
    'meta_score', 'query_uncertainty_weight',
    'text_top12_gap', 'top1_agree_text_meta', 'candidate_image_ratio',
)
INTERACTION_FEATURE_NAMES = (
    'doc_cat_is_disease', 'doc_cat_is_guidance', 'doc_cat_is_phenology',
    'has_disease_label', 'has_stage_label', 'image_rank', 'image_score',
    'meta_image_consistency', 'meta_score', 'num_images',
    'text_rank', 'text_score',
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(label: str) -> int:
    return int(hashlib.sha256(label.encode('utf-8')).hexdigest()[:16], 16)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    tmp.replace(path)


def standardizer(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-12] = 1.0
    return mean, std


def hard_order(scores: np.ndarray, ids: list[str]) -> list[int]:
    return sorted(range(len(ids)), key=lambda j: (-float(scores[j]), ids[j]))


def ndcg_mrr(order: list[int], labels: np.ndarray) -> tuple[float, float]:
    pos = np.flatnonzero(labels)
    if len(pos) == 0:
        return 0.0, 0.0
    if len(pos) != 1:
        raise ValueError('Expected at most one relevant candidate per query')
    rank = order.index(int(pos[0])) + 1
    return 1.0 / math.log2(rank + 1.0), 1.0 / rank


def load_inputs() -> tuple[dict, list[str], pd.DataFrame]:
    feature_path = INPUTS / 'MODEL_FEATURES_NO_QRELS.csv'
    qrel_path = INPUTS / 'QRELS_SEPARATE.csv'
    inner_path = INPUTS / 'INNER_FOLD_ASSIGNMENTS.csv'
    for path in (feature_path, qrel_path, inner_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    features = pd.read_csv(feature_path, dtype={'query_id': str, 'candidate_id': str})
    qrels = pd.read_csv(qrel_path, dtype={'query_id': str, 'candidate_id': str})
    inner = pd.read_csv(inner_path, dtype={'query_id': str, 'role': str})
    keys = ['query_id', 'candidate_id']
    if features.duplicated(keys).any() or qrels.duplicated(keys).any():
        raise ValueError('Duplicate query-candidate input key')
    merged = features.merge(qrels[keys + ['relevance']], on=keys, how='outer', validate='one_to_one', indicator=True)
    if not merged['_merge'].eq('both').all():
        raise ValueError('Feature/qrel candidate coverage mismatch')
    merged['relevance'] = merged['relevance'].astype(int)
    if not merged['relevance'].isin([0, 1]).all():
        raise ValueError('Frontier-F3 requires binary labels')
    positive_counts = merged.groupby('query_id')['relevance'].sum()
    if int(positive_counts.max()) > 1:
        raise ValueError('Categorical one-positive-or-null target is violated')
    forbidden = {'query_id', 'candidate_id', 'base_score', 'relevance', '_merge', 'outer_fold', 'inner_fold', 'role'}
    source_feature_names = [c for c in features.columns if c not in forbidden]
    if len(source_feature_names) != 29 or 'text_image_gap' not in source_feature_names:
        raise ValueError('Expected 29 audited source features including redundant text_image_gap')
    feature_names = list(INTERACTION_FEATURE_NAMES)
    if len(feature_names) != 12 or not set(feature_names).issubset(source_feature_names):
        raise ValueError('Expected the 12 candidate-varying feature basis registered for F3.2')
    if set(NULL_RAW_NAMES) - set(features.columns):
        raise ValueError('A null-state context feature is absent')
    for col in feature_names + ['base_score']:
        merged[col] = pd.to_numeric(merged[col], errors='raise').astype(np.float64)
        if not np.isfinite(merged[col].to_numpy()).all():
            raise ValueError('Non-finite input in {}'.format(col))
    groups = {}
    for qid, block in merged.sort_values(keys).groupby('query_id', sort=True):
        block = block.sort_values('candidate_id', kind='mergesort')
        labels = block['relevance'].to_numpy(dtype=int)
        groups[str(qid)] = {
            'ids': block['candidate_id'].astype(str).tolist(),
            'X_raw': block[feature_names].to_numpy(dtype=np.float64),
            'base': block['base_score'].to_numpy(dtype=np.float64),
            'labels': labels,
            'agreement_group': int(block['top1_agree_text_meta'].iloc[0]),
            'null_raw': np.asarray([
                block['meta_score'].max(),
                block['query_uncertainty_weight'].iloc[0], block['text_top12_gap'].iloc[0],
                block['top1_agree_text_meta'].iloc[0], block['candidate_image_ratio'].iloc[0],
            ], dtype=np.float64),
        }
    if len(groups) != 483 or sum(len(v['ids']) for v in groups.values()) != 4813:
        raise ValueError('Unexpected cardinality: {} queries / {} pairs'.format(len(groups), sum(len(v['ids']) for v in groups.values())))
    if len(inner) != 7728 or set(inner['role']) != {'train', 'validation'}:
        raise ValueError('Unexpected inner-fold assignment format')
    if set(inner['query_id']) != set(groups):
        raise ValueError('Inner assignment query coverage differs from source data')
    return groups, feature_names, inner


def query_design(groups: dict, X_by_q: dict, Z_by_q: dict, train_qids: list[str],
                 interaction_indices: np.ndarray, agreement_mean: float) -> dict:
    p_main = next(iter(X_by_q.values())).shape[1]
    p_interaction = len(interaction_indices)
    candidate_dim = p_main + p_interaction
    g = next(iter(Z_by_q.values())).shape[0] + 1
    design = {}
    for qid in train_qids:
        Xq = X_by_q[qid]
        centered_agreement = float(groups[qid]['agreement_group']) - agreement_mean
        X_interaction = Xq[:, interaction_indices]
        rank_X = np.column_stack([Xq, centered_agreement * X_interaction])
        z = np.r_[1.0, Z_by_q[qid]]
        phi = np.zeros((len(Xq) + 1, candidate_dim + g), dtype=np.float64)
        phi[0, candidate_dim:] = z
        phi[1:, :candidate_dim] = rank_X
        pos = np.flatnonzero(groups[qid]['labels'])
        target = int(pos[0] + 1) if len(pos) else 0
        base_order = hard_order(groups[qid]['base'], groups[qid]['ids'])
        base_ndcg, _ = ndcg_mrr(base_order, groups[qid]['labels'])
        design[qid] = {
            'phi': phi, 'target': target, 'X': Xq, 'rank_X': rank_X,
            'centered_agreement': centered_agreement,
            'labels': groups[qid]['labels'], 'base_ndcg': base_ndcg,
            'agreement_group': groups[qid]['agreement_group'],
        }
    return design


def objective_factory(design: dict, train_qids: list[str], p_candidate: int,
                      p_interaction: int, l2: float, rho: float):
    candidate_dim = p_candidate + p_interaction
    null_dim = next(iter(design.values()))['phi'].shape[1] - candidate_dim
    model_dim = candidate_dim + null_dim
    n_params = model_dim + (1 if rho > 0.0 else 0)
    penalty = np.ones(n_params, dtype=np.float64)
    penalty[p_candidate:candidate_dim] = 4.0  # stronger, fixed shrinkage on interaction slopes
    penalty[candidate_dim] = 0.0  # null intercept only
    if rho > 0.0:
        penalty[model_dim:] = 0.0  # one auxiliary pooled RU threshold
    n_train = len(train_qids)

    def fn(theta: np.ndarray) -> tuple[float, np.ndarray]:
        value = 0.5 * l2 * float(np.dot(penalty * theta, theta))
        grad = l2 * penalty * theta
        losses = np.zeros(n_train, dtype=np.float64)
        loss_grads = np.zeros((n_train, candidate_dim), dtype=np.float64)
        model_theta = theta[:model_dim]
        for row, qid in enumerate(train_qids):
            rec = design[qid]
            phi = rec['phi']
            logits = phi @ model_theta
            logz = float(logsumexp(logits))
            probs = np.exp(logits - logz)
            value += logz - float(logits[rec['target']])
            grad[:model_dim] += phi.T @ probs - phi[rec['target']]

            pos = np.flatnonzero(rec['labels'])
            if len(pos) == 0:
                continue
            g_idx = int(pos[0])
            Xq = rec['rank_X']
            scores = Xq @ theta[:candidate_dim]
            other = np.asarray([j for j in range(len(scores)) if j != g_idx], dtype=int)
            margin_prob = expit((scores[other] - scores[g_idx]) / TEMPERATURE)
            soft_rank = 1.0 + float(margin_prob.sum())
            soft_ndcg = 1.0 / math.log2(soft_rank + 1.0)
            raw_downside = rec['base_ndcg'] - soft_ndcg
            losses[row] = TAU_DOWNSIDE * np.logaddexp(0.0, raw_downside / TAU_DOWNSIDE)
            dloss_drank = math.log(2.0) / ((soft_rank + 1.0) * math.log(soft_rank + 1.0) ** 2)
            d_rank_dtheta = ((margin_prob * (1.0 - margin_prob) / TEMPERATURE)[:, None] *
                             (Xq[other] - Xq[g_idx])).sum(axis=0)
            loss_grads[row] = expit(raw_downside / TAU_DOWNSIDE) * dloss_drank * d_rank_dtheta

        if rho > 0.0:
            eta = float(theta[model_dim])
            tail_arg = (losses - eta) / TAU_TAIL
            tail_weight = expit(tail_arg)
            risk = eta + (TAU_TAIL / (ALPHA * n_train)) * float(np.logaddexp(0.0, tail_arg).sum())
            value += rho * n_train * risk
            grad[:candidate_dim] += (rho / ALPHA) * (tail_weight[:, None] * loss_grads).sum(axis=0)
            grad[model_dim] += rho * (n_train - float(tail_weight.sum()) / ALPHA)
        return float(value), grad

    return fn


def synthetic_objective_gradient_check() -> dict:
    """Finite-difference audit of neutral and pooled-risk interaction objectives."""
    rng = np.random.RandomState(20261008)
    p_candidate, p_interaction = 3, 2
    candidate_dim = p_candidate + p_interaction
    design = {}
    train_qids = []
    for i in range(60):
        qid = 'synthetic_{:03d}'.format(i)
        train_qids.append(qid)
        X = rng.normal(size=(4, p_candidate))
        z = rng.normal(size=6)
        interaction_x = X[:, :p_interaction] * (0.5 if i < 30 else -0.5)
        rank_X = np.column_stack([X, interaction_x])
        phi = np.zeros((5, candidate_dim + 7), dtype=np.float64)
        phi[0, candidate_dim:] = np.r_[1.0, z]
        phi[1:, :candidate_dim] = rank_X
        labels = np.zeros(4, dtype=int)
        if i % 11 != 0:
            labels[i % 4] = 1
        target = int(np.flatnonzero(labels)[0] + 1) if labels.sum() else 0
        design[qid] = {
            'phi': phi, 'target': target, 'X': X, 'rank_X': rank_X, 'labels': labels,
            'base_ndcg': 0.6 + 0.1 * float(rng.rand()),
            'agreement_group': 0 if i < 30 else 1,
        }
    results = {}
    for rho in (0.0, 2.0):
        fn = objective_factory(design, train_qids, p_candidate, p_interaction, 1.0, rho)
        dim = candidate_dim + 7 + (1 if rho > 0.0 else 0)
        theta = rng.normal(scale=0.03, size=dim)
        analytic = fn(theta)[1]
        numeric = np.empty(dim, dtype=np.float64)
        h = 1e-6
        for j in range(dim):
            delta = np.zeros(dim, dtype=np.float64)
            delta[j] = h
            numeric[j] = (fn(theta + delta)[0] - fn(theta - delta)[0]) / (2.0 * h)
        abs_error = np.abs(analytic - numeric)
        max_abs = float(abs_error.max())
        max_scaled = float((abs_error / np.maximum(1.0, np.abs(numeric))).max())
        results['interaction_rho_{}'.format(rho)] = {'max_abs_error': max_abs, 'max_scaled_error': max_scaled}

    # The matched no-interaction control uses the same objective with a zero-width
    # interaction block; audit its neutral and risk-tilted gradients separately.
    design_no = {}
    for qid in train_qids:
        rec = design[qid]
        X = rec['X']
        phi = np.zeros((len(X) + 1, 3 + 7), dtype=np.float64)
        phi[0, 3:] = rec['phi'][0, candidate_dim:]
        phi[1:, :3] = X
        design_no[qid] = {
            'phi': phi, 'target': rec['target'], 'X': X, 'rank_X': X,
            'labels': rec['labels'], 'base_ndcg': rec['base_ndcg'],
            'agreement_group': rec['agreement_group'],
        }
    for rho in (0.0, 2.0):
        fn = objective_factory(design_no, train_qids, 3, 0, 1.0, rho)
        dim = 3 + 7 + (1 if rho > 0.0 else 0)
        theta = rng.normal(scale=0.03, size=dim)
        analytic = fn(theta)[1]
        numeric = np.empty(dim, dtype=np.float64)
        h = 1e-6
        for j in range(dim):
            delta = np.zeros(dim, dtype=np.float64)
            delta[j] = h
            numeric[j] = (fn(theta + delta)[0] - fn(theta - delta)[0]) / (2.0 * h)
        abs_error = np.abs(analytic - numeric)
        results['noint_rho_{}'.format(rho)] = {
            'max_abs_error': float(abs_error.max()),
            'max_scaled_error': float((abs_error / np.maximum(1.0, np.abs(numeric))).max()),
        }
    passed = all(v['max_scaled_error'] <= 2e-5 for v in results.values())
    return {
        'status': 'PASS' if passed else 'FAIL',
        'checks': results,
        'finite_difference_step': 1e-6,
        'synthetic_training_queries': 60,
        'synthetic_agreement_support': {'0': 30, '1': 30},
    }


def fit_categorical(design: dict, train_qids: list[str], p_candidate: int,
                    p_interaction: int, l2: float, rho: float,
                    initial: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    candidate_dim = p_candidate + p_interaction
    null_dim = next(iter(design.values()))['phi'].shape[1] - candidate_dim
    model_dim = candidate_dim + null_dim
    n_params = model_dim + (1 if rho > 0.0 else 0)
    fn = objective_factory(design, train_qids, p_candidate, p_interaction, l2, rho)
    if initial is None:
        start = np.zeros(n_params, dtype=np.float64)
        if rho > 0.0:
            start[model_dim:] = 0.05
    elif rho > 0.0 and len(initial) == model_dim:
        start = np.r_[initial.copy(), np.asarray([0.05], dtype=np.float64)]
    else:
        start = initial.copy()
    began = time.time()
    result = minimize(fn, start, jac=True, method='L-BFGS-B',
                      options={'maxiter': 700, 'ftol': 1e-11, 'gtol': 1e-7, 'maxls': 60, 'maxcor': 20})
    grad_norm = float(np.linalg.norm(fn(result.x)[1], ord=np.inf))
    grad_norm_per_query = grad_norm / max(len(train_qids), 1)
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all() or grad_norm_per_query > GRADIENT_TOL_PER_QUERY:
        raise RuntimeError('Categorical optimizer did not converge: success={} status={} grad_inf={:.3g} grad_inf_per_query={:.3g}'.format(
            result.success, result.status, grad_norm, grad_norm_per_query))

    # Smooth tail variables make the objective differentiable. A conservative
    # lambda-level precision floor stabilizes the local generalized-posterior Laplace.
    hess = numerical_hessian(fn, result.x)
    raw_eig, eigvec = np.linalg.eigh(0.5 * (hess + hess.T))
    floor = max(float(l2), 1e-6)
    clipped = np.maximum(raw_eig, floor)
    cov = (eigvec * (1.0 / clipped)) @ eigvec.T
    interaction_cov_trace = float(np.trace(cov[p_candidate:candidate_dim, p_candidate:candidate_dim]))
    interaction_norm = float(np.linalg.norm(result.x[p_candidate:candidate_dim]))
    diag = {
        'objective': float(result.fun), 'optimizer_success_flag': bool(result.success),
        'optimizer_status': int(result.status), 'optimizer_iterations': int(result.nit),
        'gradient_inf_norm': grad_norm, 'gradient_inf_norm_per_query': grad_norm_per_query,
        'hessian_min_eigenvalue_raw': float(raw_eig.min()),
        'hessian_max_eigenvalue_raw': float(raw_eig.max()),
        'hessian_nonpositive_eigenvalues': int(np.sum(raw_eig <= 0.0)),
        'hessian_eigenvalue_clip_fraction': float(np.mean(raw_eig <= floor)),
        'interaction_coefficient_norm': interaction_norm,
        'interaction_posterior_covariance_trace': interaction_cov_trace,
        'hessian_eigenvalue_floor': floor, 'elapsed_seconds': time.time() - began,
    }
    return result.x, cov, diag


def numerical_hessian(fn, theta: np.ndarray) -> np.ndarray:
    p = len(theta)
    h = 2e-4
    out = np.empty((p, p), dtype=np.float64)
    for j in range(p):
        delta = np.zeros(p, dtype=np.float64)
        delta[j] = h
        gp = fn(theta + delta)[1]
        gm = fn(theta - delta)[1]
        out[:, j] = (gp - gm) / (2.0 * h)
    return 0.5 * (out + out.T)


def fit_pointwise_map(groups: dict, rank_X_by_q: dict, p_candidate: int,
                      p_interaction: int, train_qids: list[str], l2: float) -> tuple[np.ndarray, float, dict]:
    X = np.vstack([rank_X_by_q[q] for q in train_qids])
    y = np.concatenate([groups[q]['labels'] for q in train_qids]).astype(np.float64)
    Xb = np.column_stack([np.ones(len(X)), X])
    penalty = np.ones(Xb.shape[1], dtype=np.float64)
    penalty[0] = 0.0
    penalty[1 + p_candidate:1 + p_candidate + p_interaction] = 4.0

    def fn(w: np.ndarray):
        eta = Xb @ w
        p = expit(eta)
        loss = np.logaddexp(0.0, eta).sum() - np.dot(y, eta) + 0.5 * l2 * np.dot(penalty * w, w)
        grad = Xb.T @ (p - y) + l2 * penalty * w
        return float(loss), grad

    began = time.time()
    res = minimize(fn, np.zeros(Xb.shape[1]), jac=True, method='L-BFGS-B',
                   options={'maxiter': 700, 'ftol': 1e-11, 'gtol': 1e-7, 'maxls': 60})
    grad_norm = float(np.linalg.norm(fn(res.x)[1], ord=np.inf))
    grad_norm_per_candidate = grad_norm / max(len(y), 1)
    if not np.isfinite(res.fun) or not np.isfinite(res.x).all() or grad_norm_per_candidate > GRADIENT_TOL_PER_CANDIDATE:
        raise RuntimeError('Pointwise optimizer did not converge; grad_inf={:.3g} grad_inf_per_candidate={:.3g}'.format(
            grad_norm, grad_norm_per_candidate))
    return res.x, time.time() - began, {
        'optimizer_iterations': int(res.nit), 'gradient_inf_norm': grad_norm,
        'gradient_inf_norm_per_candidate': grad_norm_per_candidate,
    }


def categorical_probs(rank_X: np.ndarray, z: np.ndarray, theta: np.ndarray) -> np.ndarray:
    candidate_dim = rank_X.shape[1]
    cand = rank_X @ theta[:candidate_dim]
    null_dim = len(z) + 1
    null = np.r_[1.0, z] @ theta[candidate_dim:candidate_dim + null_dim]
    logits = np.r_[null, cand]
    return np.exp(logits - logsumexp(logits))


def categorical_draw_probs(rank_X: np.ndarray, z: np.ndarray, draws: np.ndarray) -> np.ndarray:
    candidate_dim = rank_X.shape[1]
    cand = (rank_X @ draws[:, :candidate_dim].T).T
    z_aug = np.r_[1.0, z]
    null_dim = len(z) + 1
    null = np.sum(z_aug[None, :] * draws[:, candidate_dim:candidate_dim + null_dim], axis=1)
    logits = np.column_stack([null, cand])
    logits -= logsumexp(logits, axis=1)[:, None]
    return np.exp(logits)


def categorical_scores(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    pos = np.flatnonzero(labels)
    target = int(pos[0] + 1) if len(pos) else 0
    p = np.clip(probs, 1e-12, 1.0)
    one = np.zeros_like(p)
    one[target] = 1.0
    return -float(np.log(p[target])), float(np.sum((p - one) ** 2)), float((p[0] - one[0]) ** 2)


def pointwise_scores(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    p = np.clip(probs, 1e-12, 1.0 - 1e-12)
    ll = -float(np.mean(labels * np.log(p) + (1.0 - labels) * np.log1p(-p)))
    return ll, float(np.mean((p - labels) ** 2))


def posterior_risk(rec: dict, method_order: list[int], base_order: list[int], draw_probs: np.ndarray) -> dict:
    n = len(rec['ids'])
    discount = np.asarray([1.0 / math.log2(r + 1.0) for r in range(1, n + 1)])
    mrank = {idx: r for r, idx in enumerate(method_order)}
    brank = {idx: r for r, idx in enumerate(base_order)}
    delta = np.asarray([discount[mrank[j]] - discount[brank[j]] for j in range(n)])
    candidate_p = draw_probs[:, 1:]
    expected_delta = candidate_p @ delta
    down_prob = candidate_p @ (delta < -EPS).astype(float)
    down_severity = candidate_p @ np.maximum(-delta, 0.0)
    return {
        'posterior_expected_delta': float(expected_delta.mean()),
        'posterior_lcb05_expected_delta': float(np.quantile(expected_delta, 0.05)),
        'posterior_predictive_downside_probability': float(down_prob.mean()),
        'posterior_expected_downside': float(down_severity.mean()),
        'posterior_downside_probability_sd': float(down_prob.std(ddof=1)) if len(down_prob) > 1 else 0.0,
    }


def fit_predict_split(groups: dict, feature_names: list[str], train_qids: list[str], val_qids: list[str],
                      outer_fold: int, inner_fold: int):
    Xtrain = np.vstack([groups[q]['X_raw'] for q in train_qids])
    Xmean, Xstd = standardizer(Xtrain)
    X_by_q = {q: (groups[q]['X_raw'] - Xmean) / Xstd for q in train_qids + val_qids}
    Ztrain = np.vstack([groups[q]['null_raw'] for q in train_qids])
    Zmean, Zstd = standardizer(Ztrain)
    Z_by_q = {q: (groups[q]['null_raw'] - Zmean) / Zstd for q in train_qids + val_qids}
    missing = set(INTERACTION_FEATURE_NAMES) - set(feature_names)
    if missing:
        raise ValueError('Locked interaction feature absent: {}'.format(sorted(missing)))
    interaction_indices = np.asarray([feature_names.index(n) for n in INTERACTION_FEATURE_NAMES], dtype=int)
    agreement_mean = float(np.mean([groups[q]['agreement_group'] for q in train_qids]))
    rank_X_int = {}
    for qid in train_qids + val_qids:
        centered = float(groups[qid]['agreement_group']) - agreement_mean
        interaction_X = X_by_q[qid][:, interaction_indices]
        rank_X_int[qid] = np.column_stack([X_by_q[qid], centered * interaction_X])
    rank_X_noint = X_by_q
    design_int = query_design(groups, X_by_q, Z_by_q, train_qids, interaction_indices, agreement_mean)
    design_noint = query_design(groups, X_by_q, Z_by_q, train_qids, np.asarray([], dtype=int), agreement_mean)
    p_candidate = len(feature_names)
    p_interaction = len(interaction_indices)
    cand_rows, metric_rows, risk_rows, cost_rows = [], [], [], []

    for qid in val_qids:
        rec = groups[qid]
        base_order = hard_order(rec['base'], rec['ids'])
        base_ndcg, base_mrr = ndcg_mrr(base_order, rec['labels'])
        metric_rows.append({
            'hypothesis_id': 'H-F033', 'model_version': 'Frontier-F3.3',
            'outer_training_fold': outer_fold, 'inner_fold': inner_fold, 'query_id': qid,
            'agreement_group': rec['agreement_group'], 'variant': 'FROZEN_BACKBONE',
            'action': 'BASELINE', 'rho': np.nan, 'lambda': np.nan,
            'n_candidates': len(rec['ids']), 'backbone_ndcg10': base_ndcg, 'backbone_mrr10': base_mrr,
            'ndcg10': base_ndcg, 'mrr10': base_mrr, 'delta_ndcg10': 0.0, 'delta_mrr10': 0.0,
            'degraded': 0, 'reachable': int(rec['labels'].sum() == 1), 'expected_downside': 0.0,
            'worst_decline': 0.0, 'ranking_changed': 0, 'categorical_logloss': np.nan,
            'categorical_brier': np.nan, 'null_brier': np.nan,
            'status': 'INNER_VALIDATION / DEVELOPMENT_ONLY',
        })

    for l2 in L2_GRID:
        # Matched deterministic pointwise control uses the same reduced main-effect basis
        # and the same 12 context interactions as the full categorical model.
        point_w, point_secs, point_diag = fit_pointwise_map(
            groups, rank_X_int, p_candidate, p_interaction, train_qids, l2)
        cost_rows.append({
            'outer_training_fold': outer_fold, 'inner_fold': inner_fold, 'rho': np.nan, 'lambda': l2,
            'model': 'POINTWISE_INT_MAP', 'train_queries': len(train_qids), 'validation_queries': len(val_qids),
            'posterior_draws': 0, 'seed': np.nan, 'elapsed_seconds': point_secs,
            'objective': np.nan, 'optimizer_iterations': point_diag['optimizer_iterations'],
            'gradient_inf_norm': point_diag['gradient_inf_norm'], 'gradient_inf_norm_per_query': np.nan,
            'gradient_inf_norm_per_candidate': point_diag['gradient_inf_norm_per_candidate'],
            'hessian_min_eigenvalue_raw': np.nan, 'hessian_nonpositive_eigenvalues': np.nan,
            'hessian_eigenvalue_clip_fraction': np.nan,
            'interaction_coefficient_norm': float(np.linalg.norm(point_w[1 + p_candidate:1 + p_candidate + p_interaction])),
            'interaction_posterior_covariance_trace': np.nan, 'fit_status': 'PASS',
            'evidence_status': 'INNER_VALIDATION / DEVELOPMENT_ONLY', 'compute_device': 'CPU',
        })
        point_probs = {
            q: expit(np.column_stack([np.ones(len(rank_X_int[q])), rank_X_int[q]]) @ point_w)
            for q in val_qids
        }
        for qid in val_qids:
            rec = groups[qid]
            base_order = hard_order(rec['base'], rec['ids'])
            base_ndcg, base_mrr = ndcg_mrr(base_order, rec['labels'])
            probs = point_probs[qid]
            order = hard_order(probs, rec['ids'])
            ndcg, mrr = ndcg_mrr(order, rec['labels'])
            ll, brier = pointwise_scores(probs, rec['labels'])
            variant = 'POINTWISE_INT_MAP_rhoNA_lambda{}'.format(l2)
            metric_rows.append({
                'hypothesis_id': 'H-F033', 'model_version': 'Frontier-F3.3',
                'outer_training_fold': outer_fold, 'inner_fold': inner_fold, 'query_id': qid,
                'agreement_group': rec['agreement_group'], 'variant': variant, 'action': 'POINTWISE_MAP',
                'rho': np.nan, 'lambda': l2, 'n_candidates': len(rec['ids']),
                'backbone_ndcg10': base_ndcg, 'backbone_mrr10': base_mrr,
                'ndcg10': ndcg, 'mrr10': mrr, 'delta_ndcg10': ndcg - base_ndcg,
                'delta_mrr10': mrr - base_mrr, 'degraded': int(ndcg - base_ndcg < -EPS),
                'reachable': int(rec['labels'].sum() == 1), 'expected_downside': max(base_ndcg - ndcg, 0.0),
                'worst_decline': max(base_ndcg - ndcg, 0.0), 'ranking_changed': int(order != base_order),
                'categorical_logloss': ll, 'categorical_brier': brier, 'null_brier': np.nan,
                'agreement_center': float(groups[qid]['agreement_group'] - agreement_mean),
                'posterior_top1_stability': np.nan, 'posterior_pairwise_reversal_rate': np.nan,
                'posterior_map_bma_top1_agreement': np.nan,
                'status': 'INNER_VALIDATION / DEVELOPMENT_ONLY',
            })
            for rank, idx in enumerate(order, start=1):
                cand_rows.append({
                    'hypothesis_id': 'H-F033', 'model_version': 'Frontier-F3.3',
                    'outer_training_fold': outer_fold, 'inner_fold': inner_fold, 'query_id': qid,
                    'agreement_group': rec['agreement_group'], 'candidate_id': rec['ids'][idx],
                    'variant': variant, 'action': 'POINTWISE_MAP', 'rho': np.nan, 'lambda': l2,
                    'candidate_probability': float(probs[idx]), 'posterior_predictive_sd': np.nan,
                    'null_probability': np.nan, 'rank': rank, 'base_score': float(rec['base'][idx]),
                    'status': 'INNER_VALIDATION / DEVELOPMENT_ONLY',
                })

        for family, design, rank_X, n_interaction in (
            ('INT', design_int, rank_X_int, p_interaction),
            ('NOINT', design_noint, rank_X_noint, 0),
        ):
            neutral_theta, neutral_cov, neutral_diag = fit_categorical(
                design, train_qids, p_candidate, n_interaction, l2, 0.0)
            for rho in (0.0, 2.0):
                if rho == 0.0:
                    theta, cov, diag = neutral_theta, neutral_cov, neutral_diag
                else:
                    theta, cov, diag = fit_categorical(
                        design, train_qids, p_candidate, n_interaction, l2, rho, initial=neutral_theta)
                seed = stable_seed('Frontier-F3.3|{}|{}|{}|{}|{}|{}'.format(
                    family, outer_fold, inner_fold, l2, rho, POSTERIOR_DRAWS))
                rng = np.random.Generator(np.random.PCG64(seed))
                draws = rng.multivariate_normal(theta, cov, size=POSTERIOR_DRAWS, check_valid='warn')
                cost_rows.append({
                    'outer_training_fold': outer_fold, 'inner_fold': inner_fold, 'rho': rho, 'lambda': l2,
                    'model': 'CAT_{}_RU'.format(family), 'train_queries': len(train_qids),
                    'validation_queries': len(val_qids), 'posterior_draws': POSTERIOR_DRAWS,
                    'seed': seed, 'elapsed_seconds': diag['elapsed_seconds'], 'objective': diag['objective'],
                    'optimizer_iterations': diag['optimizer_iterations'],
                    'gradient_inf_norm': diag['gradient_inf_norm'],
                    'gradient_inf_norm_per_query': diag['gradient_inf_norm_per_query'],
                    'gradient_inf_norm_per_candidate': np.nan,
                    'hessian_min_eigenvalue_raw': diag['hessian_min_eigenvalue_raw'],
                    'hessian_nonpositive_eigenvalues': diag['hessian_nonpositive_eigenvalues'],
                    'hessian_eigenvalue_clip_fraction': diag['hessian_eigenvalue_clip_fraction'],
                    'interaction_coefficient_norm': diag['interaction_coefficient_norm'],
                    'interaction_posterior_covariance_trace': diag['interaction_posterior_covariance_trace'],
                    'fit_status': 'PASS', 'evidence_status': 'INNER_VALIDATION / DEVELOPMENT_ONLY',
                    'compute_device': 'CPU',
                })
                for qid in val_qids:
                    rec = groups[qid]
                    rank_features = rank_X[qid]
                    map_probs = categorical_probs(rank_features, Z_by_q[qid], theta)
                    draw_probs = categorical_draw_probs(rank_features, Z_by_q[qid], draws)
                    bma_probs = draw_probs.mean(axis=0)
                    base_order = hard_order(rec['base'], rec['ids'])
                    base_ndcg, base_mrr = ndcg_mrr(base_order, rec['labels'])
                    map_order = hard_order(map_probs[1:], rec['ids'])
                    map_position = np.empty(len(rec['ids']), dtype=int)
                    for rank, idx in enumerate(map_order):
                        map_position[idx] = rank
                    n_pairs = max(1, len(rec['ids']) * (len(rec['ids']) - 1) // 2)
                    reversal_counts = []
                    draw_orders = [hard_order(draw[1:], rec['ids']) for draw in draw_probs]
                    for draw_order in draw_orders:
                        draw_position = np.empty(len(rec['ids']), dtype=int)
                        for rank, idx in enumerate(draw_order):
                            draw_position[idx] = rank
                        reversals = 0
                        for i in range(len(rec['ids'])):
                            for j in range(i + 1, len(rec['ids'])):
                                reversals += int((map_position[i] < map_position[j]) !=
                                                 (draw_position[i] < draw_position[j]))
                        reversal_counts.append(reversals / float(n_pairs))
                    top1_stability = float(np.mean([o[0] == map_order[0] for o in draw_orders]))
                    bma_order = hard_order(bma_probs[1:], rec['ids'])
                    agreement = float(bma_order[0] == map_order[0])
                    reversal_rate = float(np.mean(reversal_counts))
                    pred_sd = draw_probs[:, 1:].std(axis=0, ddof=1)
                    for action, probs, action_name in (
                        ('MAP', map_probs, 'MAP'), ('BMA', bma_probs, 'POSTERIOR_MEAN')):
                        order = hard_order(probs[1:], rec['ids'])
                        ndcg, mrr = ndcg_mrr(order, rec['labels'])
                        delta = ndcg - base_ndcg
                        ll, brier, null_brier = categorical_scores(probs, rec['labels'])
                        variant = 'CAT_{}_{}_rho{}_lambda{}'.format(family, action, rho, l2)
                        metric_rows.append({
                            'hypothesis_id': 'H-F033', 'model_version': 'Frontier-F3.3',
                            'outer_training_fold': outer_fold, 'inner_fold': inner_fold, 'query_id': qid,
                            'agreement_group': rec['agreement_group'], 'variant': variant,
                            'action': action_name, 'rho': rho, 'lambda': l2,
                            'n_candidates': len(rec['ids']), 'backbone_ndcg10': base_ndcg,
                            'backbone_mrr10': base_mrr, 'ndcg10': ndcg, 'mrr10': mrr,
                            'delta_ndcg10': delta, 'delta_mrr10': mrr - base_mrr,
                            'degraded': int(delta < -EPS), 'reachable': int(rec['labels'].sum() == 1),
                            'expected_downside': max(-delta, 0.0), 'worst_decline': max(-delta, 0.0),
                            'ranking_changed': int(order != base_order),
                            'categorical_logloss': ll, 'categorical_brier': brier, 'null_brier': null_brier,
                            'agreement_center': float(groups[qid]['agreement_group'] - agreement_mean),
                            'posterior_top1_stability': top1_stability,
                            'posterior_pairwise_reversal_rate': reversal_rate,
                            'posterior_map_bma_top1_agreement': agreement,
                            'status': 'INNER_VALIDATION / DEVELOPMENT_ONLY',
                        })
                        cand_prob = probs[1:]
                        for rank, idx in enumerate(order, start=1):
                            cand_rows.append({
                                'hypothesis_id': 'H-F033', 'model_version': 'Frontier-F3.3',
                                'outer_training_fold': outer_fold, 'inner_fold': inner_fold,
                                'query_id': qid, 'agreement_group': rec['agreement_group'],
                                'candidate_id': rec['ids'][idx], 'variant': variant, 'action': action_name,
                                'rho': rho, 'lambda': l2, 'candidate_probability': float(cand_prob[idx]),
                                'posterior_predictive_sd': float(pred_sd[idx]),
                                'null_probability': float(probs[0]), 'rank': rank,
                                'base_score': float(rec['base'][idx]),
                                'status': 'INNER_VALIDATION / DEVELOPMENT_ONLY',
                            })
                        risk = posterior_risk(rec, order, base_order, draw_probs)
                        risk_rows.append({
                            'hypothesis_id': 'H-F033', 'model_version': 'Frontier-F3.3',
                            'outer_training_fold': outer_fold, 'inner_fold': inner_fold, 'query_id': qid,
                            'agreement_group': rec['agreement_group'], 'variant': variant,
                            'action': action_name, 'rho': rho, 'lambda': l2,
                            'actual_delta_ndcg10': delta, 'actual_degraded': int(delta < -EPS),
                            'posterior_top1_stability': top1_stability,
                            'posterior_pairwise_reversal_rate': reversal_rate,
                            'posterior_map_bma_top1_agreement': agreement,
                            'reachable': int(rec['labels'].sum() == 1), **risk,
                            'status': 'INNER_VALIDATION DIAGNOSTIC / DEVELOPMENT_ONLY',
                        })
    return pd.DataFrame(cand_rows), pd.DataFrame(metric_rows), pd.DataFrame(risk_rows), pd.DataFrame(cost_rows)

def risk_summary(loss: np.ndarray) -> tuple[float, float, float]:
    k = max(1, int(math.ceil(0.10 * len(loss))))
    return float(np.mean(loss > EPS)), float(loss.mean()), float(np.sort(loss)[-k:].mean())


def summarize(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    group_cols = ['outer_training_fold', 'variant', 'action', 'rho', 'lambda']
    for key, block in metrics.groupby(group_cols, dropna=False, sort=True):
        byq = block.groupby('query_id', sort=True).agg(
            ndcg10=('ndcg10', 'mean'), mrr10=('mrr10', 'mean'), delta=('delta_ndcg10', 'mean'),
            mrrdelta=('delta_mrr10', 'mean'), qdr_fraction=('degraded', 'mean'),
            logloss=('categorical_logloss', 'mean'), brier=('categorical_brier', 'mean'),
        )
        loss = np.maximum(-byq['delta'].to_numpy(), 0.0)
        qdr, ed, cvar = risk_summary(loss)
        rows.append({
            'outer_training_fold': int(key[0]), 'variant': key[1], 'action': key[2], 'rho': key[3], 'lambda': float(key[4]),
            'n_unique_queries': int(len(byq)), 'n_query_fit_rows': int(len(block)),
            'ndcg10': float(byq['ndcg10'].mean()), 'mrr10': float(byq['mrr10'].mean()),
            'qdr_query_averaged': qdr, 'mean_repeated_fit_degradation_fraction': float(byq['qdr_fraction'].mean()),
            'ed_query_averaged': ed, 'cvar10_query_averaged': cvar, 'worst_decline': float(loss.max()),
            'ranking_change_rate': float(block['ranking_changed'].mean()),
            'categorical_logloss': float(byq['logloss'].mean()), 'categorical_brier': float(byq['brier'].mean()),
            'status': 'INNER_VALIDATION SUMMARY / DEVELOPMENT_ONLY',
        })
    fold_summary = pd.DataFrame(rows)

    qmeans = metrics.groupby(['variant', 'action', 'rho', 'lambda', 'query_id'], dropna=False, sort=True).agg(
        ndcg10=('ndcg10', 'mean'), mrr10=('mrr10', 'mean'), delta=('delta_ndcg10', 'mean'),
        mrrdelta=('delta_mrr10', 'mean'), degraded_fraction=('degraded', 'mean'),
        logloss=('categorical_logloss', 'mean'), brier=('categorical_brier', 'mean'),
    ).reset_index()
    cluster_rows = []
    for key, block in qmeans.groupby(['variant', 'action', 'rho', 'lambda'], dropna=False, sort=True):
        loss = np.maximum(-block['delta'].to_numpy(), 0.0)
        qdr, ed, cvar = risk_summary(loss)
        cluster_rows.append({
            'variant': key[0], 'action': key[1], 'rho': key[2], 'lambda': float(key[3]),
            'unique_queries': int(block['query_id'].nunique()), 'mean_ndcg10_query_averaged': float(block['ndcg10'].mean()),
            'mean_mrr10_query_averaged': float(block['mrr10'].mean()), 'qdr_on_query_averaged_delta': qdr,
            'mean_repeated_fit_degradation_fraction': float(block['degraded_fraction'].mean()),
            'ed_query_averaged': ed, 'cvar10_query_averaged': cvar, 'worst_decline_query_averaged': float(loss.max()),
            'logloss_query_averaged': float(block['logloss'].mean()), 'brier_query_averaged': float(block['brier'].mean()),
            'status': 'INNER REPEATED-QUERY CLUSTER SUMMARY / DEVELOPMENT_ONLY',
        })
    cluster = pd.DataFrame(cluster_rows)

    selected = []
    for outer_fold, block in metrics.groupby('outer_training_fold', sort=True):
        candidates = fold_summary[fold_summary['outer_training_fold'] == outer_fold].copy()
        base = block.groupby('query_id', sort=True)['backbone_ndcg10'].mean()
        base_mean = float(base.mean())
        eligible = candidates[candidates['ndcg10'] > base_mean + EPS].copy()
        if eligible.empty:
            selected.append({
                'outer_training_fold': int(outer_fold), 'selected_variant': 'FROZEN_BACKBONE',
                'selected_rho': np.nan, 'selected_lambda': np.nan, 'backbone_ndcg10': base_mean,
                'selection_status': 'DEVELOPMENT_ONLY exact-backbone fallback',
            })
            continue
        best = eligible['ndcg10'].max()
        best_row = eligible.loc[eligible['ndcg10'].idxmax()]
        qbest = block[block['variant'] == best_row['variant']]
        qbest = qbest.groupby('query_id', sort=True)['ndcg10'].mean()
        se = float(qbest.std(ddof=1) / math.sqrt(max(1, qbest.size)))
        cutoff = best - se
        eligible = eligible[eligible['ndcg10'] >= cutoff - 1e-15].copy()
        eligible['complexity'] = eligible['variant'].map({
            'CAT_NOINT_MAP_rho0.0_lambda1.0': 0, 'CAT_NOINT_MAP_rho2.0_lambda1.0': 0,
            'CAT_INT_MAP_rho0.0_lambda1.0': 1, 'CAT_INT_MAP_rho2.0_lambda1.0': 1,
            'POINTWISE_INT_MAP_rhoNA_lambda1.0': 1,
            'CAT_NOINT_BMA_rho0.0_lambda1.0': 2, 'CAT_NOINT_BMA_rho2.0_lambda1.0': 2,
            'CAT_INT_BMA_rho0.0_lambda1.0': 3, 'CAT_INT_BMA_rho2.0_lambda1.0': 3,
        }).fillna(99)
        eligible = eligible.sort_values(
            ['cvar10_query_averaged', 'qdr_query_averaged', 'ed_query_averaged', 'complexity', 'lambda', 'rho'],
            ascending=[True, True, True, True, False, True], na_position='last')
        chosen = eligible.iloc[0]
        selected.append({
            'outer_training_fold': int(outer_fold), 'selected_variant': chosen['variant'],
            'selected_rho': chosen['rho'], 'selected_lambda': float(chosen['lambda']),
            'selected_action': chosen['action'], 'best_eligible_ndcg10': float(best),
            'one_se_cutoff': float(cutoff), 'selected_inner_ndcg10': float(chosen['ndcg10']),
            'selected_inner_mrr10': float(chosen['mrr10']), 'selected_inner_qdr': float(chosen['qdr_query_averaged']),
            'selected_inner_ed': float(chosen['ed_query_averaged']),
            'selected_inner_cvar10': float(chosen['cvar10_query_averaged']),
            'backbone_ndcg10': base_mean, 'selection_status': 'DEVELOPMENT_ONLY inner selector',
        })
    return fold_summary, cluster, pd.DataFrame(selected)


def summarize_agreement_groups(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ['variant', 'action', 'rho', 'lambda', 'agreement_group']
    for key, block in metrics.groupby(group_cols, dropna=False, sort=True):
        byq = block.groupby('query_id', sort=True).agg(
            ndcg10=('ndcg10', 'mean'), mrr10=('mrr10', 'mean'), delta=('delta_ndcg10', 'mean'))
        loss = np.maximum(-byq['delta'].to_numpy(), 0.0)
        qdr, ed, cvar = risk_summary(loss)
        rows.append({
            'variant': key[0], 'action': key[1], 'rho': key[2], 'lambda': float(key[3]),
            'agreement_group': int(key[4]), 'unique_queries': int(len(byq)),
            'mean_ndcg10_query_averaged': float(byq['ndcg10'].mean()),
            'mean_mrr10_query_averaged': float(byq['mrr10'].mean()),
            'qdr_query_averaged': qdr, 'ed_query_averaged': ed,
            'cvar10_query_averaged': cvar, 'worst_decline_query_averaged': float(loss.max()),
            'status': 'INNER FEATURE-STRATIFIED SUMMARY / DEVELOPMENT_ONLY',
        })
    return pd.DataFrame(rows)


def holm_adjust(pvalues: list[float]) -> list[float]:
    m = len(pvalues)
    order = np.argsort(np.asarray(pvalues))
    adjusted = np.ones(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * float(pvalues[idx]))
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def write_frame(path: Path, frame: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    frame.to_csv(tmp, index=False)
    tmp.replace(path)
    return sha256(path)


def main() -> None:
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    for path in (PRED_DIR, METRIC_SPLITS, RISK_SPLITS, COST_SPLITS, CHECKPOINTS):
        path.mkdir(parents=True, exist_ok=True)
    gradient_audit = synthetic_objective_gradient_check()
    atomic_json(CHECKPOINTS / 'objective_gradient_audit.json', gradient_audit)
    if gradient_audit.get('status') != 'PASS':
        raise RuntimeError('H-F033 synthetic objective gradient audit failed')
    preflight_path = TABLES / 'feature_only_preflight_audit.json'
    if not preflight_path.is_file():
        raise RuntimeError('H-F033 label-free support preflight is missing')
    preflight = json.loads(preflight_path.read_text(encoding='utf-8'))
    if (preflight.get('status') != 'PASS' or preflight.get('qrels_read') or
            preflight.get('outer_results_read') or not preflight.get('support_gate_pass') or
            not preflight.get('pairwise_rank_gate_pass') or not preflight.get('interaction_variation_gate_pass') or
            not preflight.get('candidate_vs_null_softmax_identifiability_pass') or
            not preflight.get('null_context_variation_gate_pass') or
            preflight.get('interaction_features') != list(INTERACTION_FEATURE_NAMES)):
        raise RuntimeError('H-F033 support preflight did not pass cleanly')
    for rel, expected_hash in preflight.get('source_sha256', {}).items():
        if sha256(ROOT / rel.replace('\\', '/')) != expected_hash:
            raise RuntimeError('H-F033 support preflight input hash changed: {}'.format(rel))
    code_hash = sha256(Path(__file__))
    input_hashes = {p.name: sha256(p) for p in sorted(INPUTS.glob('*.csv'))}
    config_hash = sha256(CONFIG_PATH)
    manifest_path = CHECKPOINTS / 'run_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('code_sha256') != code_hash or manifest.get('input_sha256') != input_hashes or manifest.get('config_sha256') != config_hash:
            raise RuntimeError('Checkpoint provenance mismatch; use a new model version')
    else:
        manifest = {
            'hypothesis_id': 'H-F033', 'model_version': 'Frontier-F3.3',
            'status': 'RUNNING', 'evidence_status': 'DEVELOPMENT_ONLY / INNER_VALIDATION',
            'code_sha256': code_hash, 'config_sha256': config_hash, 'input_sha256': input_hashes,
            'completed_splits': [], 'outer_results_read': False,
            'posterior_draws': POSTERIOR_DRAWS, 'rho_grid': list(RHO_GRID), 'lambda_grid': list(L2_GRID),
            'agreement_context': 'A_q=top1_agree_text_meta; training-center p_train computed from inner-training qids only',
            'interaction_feature_names': list(INTERACTION_FEATURE_NAMES),
            'interaction_prior_precision_multiplier': 4.0,
            'support_preflight_sha256': sha256(preflight_path),
            'model_families': ['CAT_INT', 'CAT_NOINT', 'POINTWISE_INT_MAP'],
            'candidate_feature_count': 12, 'candidate_utility_excludes_query_constant_source_features': True,
            'null_context_feature_names': list(NULL_RAW_NAMES),
            'objective_gradient_audit': gradient_audit,
        }
        atomic_json(manifest_path, manifest)
    groups, feature_names, inner = load_inputs()
    audit = {
        'unique_queries': len(groups), 'candidate_pairs': sum(len(x['ids']) for x in groups.values()),
        'reachable_queries': sum(int(x['labels'].sum() == 1) for x in groups.values()),
        'unreachable_queries': sum(int(x['labels'].sum() == 0) for x in groups.values()),
        'feature_count': len(feature_names), 'query_constant_candidate_features_excluded': 17,
        'null_context_features': list(NULL_RAW_NAMES), 'inner_assignment_rows': len(inner),
        'training_labels_used_only_for_training_qids': True, 'fold_local_scaling': True,
        'feature_only_pairwise_and_softmax_identifiability_preflight_pass': True,
        'null_context_variation_preflight_pass': True,
        'interaction_feature_names': list(INTERACTION_FEATURE_NAMES),
        'synthetic_objective_gradient_audit': gradient_audit,
        'outer_results_read': False, 'outer_predictions_created': False,
        'training_device': 'CPU', 'evidence_status': 'DEVELOPMENT_ONLY / INNER_VALIDATION',
    }
    atomic_json(CHECKPOINTS / 'input_audit.json', audit)
    split_keys = sorted(set(zip(inner['outer_fold'].astype(int), inner['inner_fold'].astype(int))))
    all_pred, all_metric, all_risk, all_cost = [], [], [], []
    for outer_fold, inner_fold in split_keys:
        key = '{}:{}'.format(outer_fold, inner_fold)
        pred_path = PRED_DIR / 'outer_{}_inner_{}.csv'.format(outer_fold, inner_fold)
        metric_path = METRIC_SPLITS / 'outer_{}_inner_{}.csv'.format(outer_fold, inner_fold)
        risk_path = RISK_SPLITS / 'outer_{}_inner_{}.csv'.format(outer_fold, inner_fold)
        cost_path = COST_SPLITS / 'outer_{}_inner_{}.csv'.format(outer_fold, inner_fold)
        paths = {'predictions': pred_path, 'metrics': metric_path, 'risk': risk_path, 'cost': cost_path}
        if key in manifest['completed_splits']:
            expected = manifest['split_sha256'][key]
            if any(not p.is_file() or sha256(p) != expected[name] for name, p in paths.items()):
                raise RuntimeError('Completed split is absent or hash-mismatched: {}'.format(key))
            all_pred.append(pred_path); all_metric.append(metric_path); all_risk.append(risk_path); all_cost.append(cost_path)
            continue
        split = inner[(inner['outer_fold'].astype(int) == outer_fold) & (inner['inner_fold'].astype(int) == inner_fold)]
        train_qids = sorted(split.loc[split['role'] == 'train', 'query_id'].astype(str).unique())
        val_qids = sorted(split.loc[split['role'] == 'validation', 'query_id'].astype(str).unique())
        if not train_qids or not val_qids or set(train_qids) & set(val_qids):
            raise ValueError('Invalid fold-local split {}'.format(key))
        started = time.time()
        cand, metr, risk, cost = fit_predict_split(groups, feature_names, train_qids, val_qids, outer_fold, inner_fold)
        hashes = {
            'predictions': write_frame(pred_path, cand), 'metrics': write_frame(metric_path, metr),
            'risk': write_frame(risk_path, risk), 'cost': write_frame(cost_path, cost),
        }
        manifest.setdefault('split_sha256', {})[key] = hashes
        manifest['completed_splits'].append(key)
        manifest['current_split'] = key
        manifest['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        atomic_json(manifest_path, manifest)
        all_pred.append(pred_path); all_metric.append(metric_path); all_risk.append(risk_path); all_cost.append(cost_path)
        print('INNER_SPLIT_COMPLETE {} train={} validation={} variants={} rows={} elapsed={:.1f}s'.format(
            key, len(train_qids), len(val_qids), len(metr['variant'].unique()), len(metr), time.time() - started), flush=True)

        progress_path = ROOT / 'FRONTIER_RESEARCH_PROGRESS.json'
        if progress_path.is_file():
            progress = json.loads(progress_path.read_text(encoding='utf-8'))
            progress['phase'] = 'track_f3_1_inner_validation_running'
            progress['current_research_direction'] = 'H-F033 fold-centered modality-agreement interaction in posterior candidate utility with pooled RU downside training objective'
            progress['completed_inner_splits'] = len(manifest['completed_splits'])
            progress['total_inner_splits'] = len(split_keys)
            progress['latest_checkpoint'] = str(manifest_path.relative_to(ROOT))
            progress['outer_results_read'] = False
            progress['training_device'] = 'CPU'
            progress['updated_local'] = time.strftime('%Y-%m-%d', time.gmtime())
            atomic_json(progress_path, progress)

    predictions = pd.concat([pd.read_csv(p) for p in all_pred], ignore_index=True)
    metrics = pd.concat([pd.read_csv(p) for p in all_metric], ignore_index=True)
    risks = pd.concat([pd.read_csv(p) for p in all_risk], ignore_index=True)
    costs = pd.concat([pd.read_csv(p) for p in all_cost], ignore_index=True)
    fold_summary, cluster_summary, selected = summarize(metrics)
    agreement_summary = summarize_agreement_groups(metrics)
    outputs = {
        TABLES / 'candidate_level_inner_predictions.csv': predictions,
        TABLES / 'inner_query_metrics.csv': metrics,
        TABLES / 'posterior_risk_diagnostics.csv': risks,
        TABLES / 'inner_fold_summary.csv': fold_summary,
        TABLES / 'inner_query_cluster_summary.csv': cluster_summary,
        TABLES / 'inner_selected_by_outer_training_fold.csv': selected,
        TABLES / 'agreement_group_summary.csv': agreement_summary,
        TABLES / 'fit_diagnostics.csv': costs,
    }
    output_hashes = {str(path.relative_to(ROOT)): write_frame(path, frame) for path, frame in outputs.items()}
    manifest.update({
        'status': 'COMPLETED', 'completed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'output_sha256': output_hashes, 'audit': audit,
        'inner_metric_rows': int(len(metrics)), 'candidate_prediction_rows': int(len(predictions)),
        'posterior_risk_rows': int(len(risks)),
    })
    atomic_json(manifest_path, manifest)
    atomic_json(CHECKPOINTS / 'final_audit.json', {
        'status': 'PASS', 'completed_splits': len(manifest['completed_splits']),
        'unique_queries': 483, 'candidate_pairs': 4813, 'outer_results_read': False,
        'outer_predictions_created': False, 'fold_local_scaling': True,
        'training_labels_used_only_for_inner_training_queries': True,
        'candidate_prediction_rows': len(predictions), 'inner_query_metric_rows': len(metrics),
        'posterior_risk_rows': len(risks), 'output_sha256': output_hashes,
    })
    progress_path = ROOT / 'FRONTIER_RESEARCH_PROGRESS.json'
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding='utf-8'))
        progress['phase'] = 'track_f3_1_inner_validation_complete'
        progress['current_research_direction'] = 'H-F033 fold-centered modality-agreement interaction in posterior candidate utility with pooled RU downside training objective'
        progress['completed_inner_splits'] = len(manifest['completed_splits'])
        progress['total_inner_splits'] = len(split_keys)
        progress['latest_checkpoint'] = str(manifest_path.relative_to(ROOT))
        progress['latest_inner_result'] = '09_TRACK_F_AGRICULTURAL_ROBUSTNESS/v3_3/tables_data/inner_query_cluster_summary.csv'
        progress['outer_results_read'] = False
        progress['training_device'] = 'CPU'
        progress['updated_local'] = time.strftime('%Y-%m-%d', time.gmtime())
        atomic_json(progress_path, progress)
    print('TRACK_F031_COMPLETE splits={} candidate_rows={} query_metric_rows={} posterior_risk_rows={}'.format(
        len(manifest['completed_splits']), len(predictions), len(metrics), len(risks)), flush=True)


if __name__ == '__main__':
    main()
