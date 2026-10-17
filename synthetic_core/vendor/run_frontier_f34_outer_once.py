#!/usr/bin/env python3
"""One locked, resumable exploratory nested-OOF run for Frontier-F3.4.

This program fits only on each outer fold's training qrels and emits test-side
scores, posterior draws, and decisions. It never calculates outer ranking
metrics and never uses held-out qrels to choose or change a prediction.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/ubuntu/apin_revision_bayes_uq_frontier")
INPUTS = ROOT / "03_DATA_AND_PROTOCOL" / "inputs"
LOCK_DIR = ROOT / "09_TRACK_F_AGRICULTURAL_ROBUSTNESS" / "v3_4"
RUN = ROOT / "13_OUTER_OOF" / "frontier_f34_once"
HELPER_PATH = ROOT / "code" / "track_f033_contextual_posterior.py"
TRAINER_PATH = ROOT / "code" / "run_frontier_f34_outer_once.py"
CONFIG_PATH = LOCK_DIR / "OUTER_EVALUATION_CONFIG.json"
EPS = 1e-12
N_DRAWS = 128
RHO_VALUES = (0.0, 2.0)
POLICIES = (
    "FROZEN_BACKBONE",
    "RHO2_ALWAYS_BMA",
    "RHO2_POSTERIOR_MEAN_GATE",
    "RHO2_MAP_GAIN_GATE",
    "RHO2_LCB05_GATE",
    "RHO0_LCB05_GATE",
    "RHO0_ALWAYS_BMA",
)
FEATURE_NAMES = (
    "doc_cat_is_disease", "doc_cat_is_guidance", "doc_cat_is_phenology",
    "has_disease_label", "has_stage_label", "image_rank", "image_score",
    "meta_image_consistency", "meta_score", "num_images", "text_rank", "text_score",
)
NULL_NAMES = (
    "meta_score", "query_uncertainty_weight", "text_top12_gap",
    "top1_agree_text_meta", "candidate_image_ratio",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_json(path: Path, obj: dict) -> None:
    atomic_bytes(path, (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))


def atomic_csv(path: Path, frame: pd.DataFrame, compression=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if compression == "gzip":
        with tmp.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
                zipped.write(frame.to_csv(index=False, float_format="%.17g").encode("utf-8"))
            raw.flush()
            os.fsync(raw.fileno())
    else:
        with tmp.open("w", encoding="utf-8", newline="") as f:
            frame.to_csv(f, index=False, float_format="%.17g")
            f.flush()
            os.fsync(f.fileno())
    os.replace(tmp, path)


def load_helper():
    spec = importlib.util.spec_from_file_location("frontier_f033_locked", str(HELPER_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import the locked F3.3 model primitives")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_lock(config: dict) -> None:
    expected = config["source_code_hashes"]
    actual = {
        "outer_training": sha256(TRAINER_PATH),
        "independent_validator": sha256(ROOT / "code" / "independent_validate_frontier_f34_outer.py"),
        "paired_analysis": sha256(ROOT / "code" / "analyze_frontier_f34_outer_paired.py"),
        "F3.3_model_source": sha256(HELPER_PATH),
    }
    if actual != expected:
        raise RuntimeError("Locked source-code SHA-256 mismatch; refusing to train")
    if sha256(LOCK_DIR / "OUTER_LOCK_PROTOCOL.md") != config.get("outer_lock_protocol_sha256"):
        raise RuntimeError("Locked outer protocol SHA-256 mismatch; refusing to train")
    if config.get("evidence_status") != "EXPLORATORY_OUTER" or config.get("outer_evaluation_count") != 1:
        raise RuntimeError("Unexpected outer lock configuration")
    if config.get("outer_results_read_before_lock") or config.get("outer_predictions_created_before_lock"):
        raise RuntimeError("Outer labels or predictions were already accessed before lock")


def verify_inputs() -> dict:
    manifest_path = INPUTS / "SOURCE_MANIFEST.json"
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {item["file"]: item["sha256"] for item in entries}
    required = ("MODEL_FEATURES_NO_QRELS.csv", "QRELS_SEPARATE.csv", "OUTER_FOLD_ASSIGNMENTS.csv")
    observed = {}
    for name in required:
        path = INPUTS / name
        if not path.is_file() or name not in expected:
            raise FileNotFoundError("Locked input absent from manifest: " + name)
        observed[name] = sha256(path)
        if observed[name] != expected[name]:
            raise RuntimeError("Input SHA-256 differs from the audited source manifest: " + name)
    return observed


def read_training_qrels(train_qids: set[str]) -> dict[str, dict[str, int]]:
    """Retain labels for this fold's training qids only; test qids never enter model data."""
    path = INPUTS / "QRELS_SEPARATE.csv"
    result: dict[str, dict[str, int]] = {}
    for chunk in pd.read_csv(path, dtype={"query_id": str, "candidate_id": str}, chunksize=50000):
        selected = chunk.loc[chunk["query_id"].isin(train_qids), ["query_id", "candidate_id", "relevance"]]
        for row in selected.itertuples(index=False):
            qid, cid, rel = str(row.query_id), str(row.candidate_id), int(row.relevance)
            if rel not in (0, 1):
                raise ValueError("Expected binary training qrels")
            result.setdefault(qid, {})[cid] = rel
    if set(result) != train_qids:
        raise ValueError("Training qrels do not cover the outer-training query IDs")
    return result


def load_fold_assignments() -> pd.DataFrame:
    frame = pd.read_csv(INPUTS / "OUTER_FOLD_ASSIGNMENTS.csv", dtype={"query_id": str, "role": str})
    if set(frame.columns) != {"outer_fold", "query_id", "role"}:
        raise ValueError("Unexpected outer-fold assignment schema")
    frame["outer_fold"] = frame["outer_fold"].astype(int)
    if frame.duplicated(["outer_fold", "query_id"]).any():
        raise ValueError("Duplicate query/fold assignment")
    if set(frame["outer_fold"]) != set(range(5)) or set(frame["role"]) != {"train", "test"}:
        raise ValueError("Expected five outer folds and train/test roles")
    return frame


def load_features(helper) -> tuple[dict, list[str], pd.DataFrame]:
    frame = pd.read_csv(INPUTS / "MODEL_FEATURES_NO_QRELS.csv", dtype={"query_id": str, "candidate_id": str})
    keys = ["query_id", "candidate_id"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate feature query-candidate key")
    if len(frame) != 4813 or frame["query_id"].nunique() != 483:
        raise ValueError("Unexpected locked frozen-pool dimensions")
    if set(FEATURE_NAMES) - set(frame.columns) or set(NULL_NAMES) - set(frame.columns):
        raise ValueError("A locked model feature is absent")
    for col in list(FEATURE_NAMES) + list(NULL_NAMES) + ["base_score"]:
        frame[col] = pd.to_numeric(frame[col], errors="raise").astype(np.float64)
        if not np.isfinite(frame[col].to_numpy()).all():
            raise ValueError("Non-finite source feature: " + col)
    groups = {}
    for qid, block in frame.sort_values(keys).groupby("query_id", sort=True):
        block = block.sort_values("candidate_id", kind="mergesort")
        groups[str(qid)] = {
            "ids": block["candidate_id"].astype(str).tolist(),
            "X_raw": block[list(FEATURE_NAMES)].to_numpy(dtype=np.float64),
            "base": block["base_score"].to_numpy(dtype=np.float64),
            "agreement_group": int(block["top1_agree_text_meta"].iloc[0]),
            "null_raw": np.asarray([
                block["meta_score"].max(), block["query_uncertainty_weight"].iloc[0],
                block["text_top12_gap"].iloc[0], block["top1_agree_text_meta"].iloc[0],
                block["candidate_image_ratio"].iloc[0],
            ], dtype=np.float64),
        }
    return groups, list(FEATURE_NAMES), frame


def discounts(n: int) -> np.ndarray:
    # The locked task has at most ten candidates per query; nDCG@10 discount is exact here.
    return np.asarray([1.0 / math.log2(r + 1.0) if r <= 10 else 0.0 for r in range(1, n + 1)])


def order_from_scores(ids: list[str], values: np.ndarray) -> list[int]:
    return sorted(range(len(ids)), key=lambda j: (-float(values[j]), ids[j]))


def gain_vector(ids: list[str], order: list[int], base_order: list[int]) -> np.ndarray:
    rank = np.empty(len(ids), dtype=int)
    base_rank = np.empty(len(ids), dtype=int)
    for pos, idx in enumerate(order, 1):
        rank[idx] = pos
    for pos, idx in enumerate(base_order, 1):
        base_rank[idx] = pos
    d = discounts(len(ids))
    return d[rank - 1] - d[base_rank - 1]


def write_fold(fold: int, groups: dict, features: pd.DataFrame, assignment: pd.DataFrame,
               input_hashes: dict, helper, lock_hash: str) -> dict:
    fold_dir = RUN / "folds" / ("outer_{}".format(fold))
    fold_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = fold_dir / "fold_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("config_hash") != lock_hash or existing.get("input_hashes") != input_hashes:
            raise RuntimeError("Existing fold checkpoint belongs to a different locked run")
        for name, digest in existing["files"].items():
            path = fold_dir / name
            if not path.is_file() or sha256(path) != digest:
                raise RuntimeError("Existing fold artifact failed its checkpoint hash: " + name)
        return existing

    role = assignment.loc[assignment["outer_fold"] == fold]
    train_qids = sorted(role.loc[role["role"] == "train", "query_id"].astype(str).unique())
    test_qids = sorted(role.loc[role["role"] == "test", "query_id"].astype(str).unique())
    if set(train_qids) & set(test_qids) or set(train_qids) | set(test_qids) != set(groups):
        raise ValueError("Outer training/test query partition is not an exact disjoint partition")
    train_qrels = read_training_qrels(set(train_qids))
    train_groups = {}
    for qid in groups:
        src = groups[qid]
        labels = np.zeros(len(src["ids"]), dtype=int)
        if qid in train_qids:
            qmap = train_qrels[qid]
            if set(qmap) != set(src["ids"]):
                raise ValueError("Training qrel/frozen candidate mismatch for " + qid)
            labels = np.asarray([qmap[cid] for cid in src["ids"]], dtype=int)
            if int(labels.sum()) > 1:
                raise ValueError("One-positive-or-null target invariant failed")
        train_groups[qid] = {**src, "labels": labels}

    started = time.time()
    Xtrain = np.vstack([train_groups[q]["X_raw"] for q in train_qids])
    Xmean, Xstd = helper.standardizer(Xtrain)
    X_by_q = {q: (train_groups[q]["X_raw"] - Xmean) / Xstd for q in groups}
    Ztrain = np.vstack([train_groups[q]["null_raw"] for q in train_qids])
    Zmean, Zstd = helper.standardizer(Ztrain)
    Z_by_q = {q: (train_groups[q]["null_raw"] - Zmean) / Zstd for q in groups}
    agreement_mean = float(np.mean([train_groups[q]["agreement_group"] for q in train_qids]))
    design = helper.query_design(
        train_groups, X_by_q, Z_by_q, train_qids, np.asarray([], dtype=int), agreement_mean)
    theta0, cov0, diag0 = helper.fit_categorical(design, train_qids, len(FEATURE_NAMES), 0, 1.0, 0.0)
    theta2, cov2, diag2 = helper.fit_categorical(
        design, train_qids, len(FEATURE_NAMES), 0, 1.0, 2.0, initial=theta0)
    states = {0.0: (theta0, cov0, diag0), 2.0: (theta2, cov2, diag2)}
    draws_by_rho = {}
    seeds = {}
    for rho, (theta, cov, _) in states.items():
        seed = helper.stable_seed("Frontier-F3.4-OUTER-LOCK|fold={}|rho={}|draws={}".format(fold, int(rho), N_DRAWS))
        rng = np.random.Generator(np.random.PCG64(seed))
        draws = rng.multivariate_normal(theta, cov, size=N_DRAWS, check_valid="raise")
        draws_by_rho[rho] = draws
        seeds[rho] = int(seed)

    candidate_rows, draw_rows, utility_rows, decision_rows, rank_rows = [], [], [], [], []
    qrels_training_rows = sum(len(v) for v in train_qrels.values())
    for qid in test_qids:
        rec = groups[qid]
        ids = rec["ids"]
        base_order = order_from_scores(ids, rec["base"])
        model_record = {}
        for rho, (theta, cov, diag) in states.items():
            rank_X = X_by_q[qid]
            map_all = helper.categorical_probs(rank_X, Z_by_q[qid], theta)
            sample_all = helper.categorical_draw_probs(rank_X, Z_by_q[qid], draws_by_rho[rho])
            bma_all = sample_all.mean(axis=0)
            map_prob = map_all[1:]
            bma_prob = bma_all[1:]
            map_order = order_from_scores(ids, map_prob)
            bma_order = order_from_scores(ids, bma_prob)
            delta = gain_vector(ids, bma_order, base_order)
            utility_draws = sample_all[:, 1:] @ delta
            mean_gain = float(utility_draws.mean())
            lcb = float(np.quantile(utility_draws, 0.05, method="linear"))
            map_delta = gain_vector(ids, map_order, base_order)
            map_gain = float(map_prob @ map_delta)
            model_record[rho] = {
                "map_all": map_all, "sample_all": sample_all, "bma_all": bma_all,
                "map_prob": map_prob, "bma_prob": bma_prob,
                "map_order": map_order, "bma_order": bma_order,
                "utility_draws": utility_draws, "mean_gain": mean_gain,
                "lcb": lcb, "map_gain": map_gain,
            }
            for draw_i in range(N_DRAWS):
                utility_rows.append({
                    "outer_fold": fold, "query_id": qid, "rho": int(rho),
                    "draw_index": draw_i, "expected_ndcg_gain": float(utility_draws[draw_i]),
                    "bma_order_sha256": canonical_hash([ids[i] for i in bma_order]),
                    "backbone_order_sha256": canonical_hash([ids[i] for i in base_order]),
                })
                null_prob = float(sample_all[draw_i, 0])
                for candidate_i, cid in enumerate(ids):
                    draw_rows.append({
                        "outer_fold": fold, "query_id": qid, "rho": int(rho),
                        "draw_index": draw_i, "candidate_id": cid,
                        "relevance_probability": float(sample_all[draw_i, candidate_i + 1]),
                        "null_probability": null_prob,
                    })

        m0, m2 = model_record[0.0], model_record[2.0]
        model_record_for_rows = {
            "rho0": m0, "rho2": m2,
        }
        base_rank = {idx: pos for pos, idx in enumerate(base_order, 1)}
        ranks_by_model = {}
        for key, model in model_record_for_rows.items():
            ranks_by_model[key] = {
                "map": {idx: pos for pos, idx in enumerate(model["map_order"], 1)},
                "bma": {idx: pos for pos, idx in enumerate(model["bma_order"], 1)},
            }
        for idx, cid in enumerate(ids):
            candidate_rows.append({
                "outer_fold": fold, "query_id": qid, "candidate_id": cid,
                "base_score": float(rec["base"][idx]), "base_rank": int(base_rank[idx]),
                "n_candidates": len(ids),
                "rho0_map_probability": float(m0["map_prob"][idx]),
                "rho0_bma_probability": float(m0["bma_prob"][idx]),
                "rho0_map_rank": int(ranks_by_model["rho0"]["map"][idx]),
                "rho0_bma_rank": int(ranks_by_model["rho0"]["bma"][idx]),
                "rho2_map_probability": float(m2["map_prob"][idx]),
                "rho2_bma_probability": float(m2["bma_prob"][idx]),
                "rho2_map_rank": int(ranks_by_model["rho2"]["map"][idx]),
                "rho2_bma_rank": int(ranks_by_model["rho2"]["bma"][idx]),
            })

        policy_specs = {
            "FROZEN_BACKBONE": ("BACKBONE", False, 0.0, 0.0),
            "RHO2_ALWAYS_BMA": ("RHO2_BMA", True, m2["mean_gain"], m2["lcb"]),
            "RHO2_POSTERIOR_MEAN_GATE": ("RHO2_BMA" if m2["mean_gain"] > 0.0 else "BACKBONE", m2["mean_gain"] > 0.0, m2["mean_gain"], m2["lcb"]),
            "RHO2_MAP_GAIN_GATE": ("RHO2_MAP" if m2["map_gain"] > 0.0 else "BACKBONE", m2["map_gain"] > 0.0, m2["mean_gain"], m2["lcb"]),
            "RHO2_LCB05_GATE": ("RHO2_BMA" if m2["lcb"] > 0.0 else "BACKBONE", m2["lcb"] > 0.0, m2["mean_gain"], m2["lcb"]),
            "RHO0_LCB05_GATE": ("RHO0_BMA" if m0["lcb"] > 0.0 else "BACKBONE", m0["lcb"] > 0.0, m0["mean_gain"], m0["lcb"]),
            "RHO0_ALWAYS_BMA": ("RHO0_BMA", True, m0["mean_gain"], m0["lcb"]),
        }
        for policy, (source, gate, mean_gain, lcb) in policy_specs.items():
            if source == "BACKBONE":
                order = base_order
            elif source == "RHO2_BMA":
                order = m2["bma_order"]
            elif source == "RHO2_MAP":
                order = m2["map_order"]
            else:
                order = m0["bma_order"]
            order_ids = [ids[i] for i in order]
            selected_score = np.zeros(len(ids), dtype=float)
            for pos, idx in enumerate(order):
                selected_score[idx] = float(len(ids) - pos)
            decision_rows.append({
                "outer_fold": fold, "query_id": qid, "policy": policy,
                "gate_applied": int(gate), "selected_source": source,
                "posterior_mean_gain": float(mean_gain), "posterior_lcb05_gain": float(lcb),
                "rho2_map_expected_gain": float(m2["map_gain"]),
                "selected_order": "|".join(order_ids),
                "selected_order_sha256": canonical_hash(order_ids),
                "posterior_draws": N_DRAWS,
            })
            for pos, idx in enumerate(order, 1):
                rank_rows.append({
                    "outer_fold": fold, "query_id": qid, "candidate_id": ids[idx],
                    "policy": policy, "selected_rank": pos,
                })

    fold_seconds = time.time() - started
    frames = {
        "candidate_scores.csv": pd.DataFrame(candidate_rows),
        "posterior_utility_draws.csv": pd.DataFrame(utility_rows),
        "policy_decisions.csv": pd.DataFrame(decision_rows),
        "candidate_policy_ranks.csv": pd.DataFrame(rank_rows),
    }
    for name, frame in frames.items():
        atomic_csv(fold_dir / name, frame)
    atomic_csv(fold_dir / "posterior_draw_candidate_probabilities.csv.gz",
               pd.DataFrame(draw_rows), compression="gzip")
    # Numerical model state and train-only scalers are retained for exact replay.
    state_tmp = fold_dir / "fold_model_state.npz.tmp"
    with state_tmp.open("wb") as stream:
        np.savez_compressed(
            stream, rho0_theta=theta0, rho0_cov=cov0, rho2_theta=theta2, rho2_cov=cov2,
            X_mean=Xmean, X_std=Xstd, Z_mean=Zmean, Z_std=Zstd,
            train_qids=np.asarray(train_qids, dtype="U"), test_qids=np.asarray(test_qids, dtype="U"),
            agreement_mean=np.asarray([agreement_mean]),
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(state_tmp, fold_dir / "fold_model_state.npz")
    diagnostics = {
        "outer_fold": fold, "model_version": "Frontier-F3.4", "hypothesis_id": "H-F034",
        "evidence_status": "EXPLORATORY_OUTER", "train_queries": len(train_qids),
        "test_queries": len(test_qids), "training_candidate_rows": qrels_training_rows,
        "rho0_fit": diag0, "rho2_fit": diag2, "posterior_draws": N_DRAWS,
        "posterior_seeds": {str(int(k)): v for k, v in seeds.items()},
        "train_only_standardizer_sha256": canonical_hash({
            "X_mean": Xmean.tolist(), "X_std": Xstd.tolist(),
            "Z_mean": Zmean.tolist(), "Z_std": Zstd.tolist(),
            "agreement_mean": agreement_mean,
        }),
        "fit_wall_seconds": fold_seconds, "compute_device": "CPU",
        "test_qrels_used": False, "outer_metrics_computed": False,
    }
    atomic_json(fold_dir / "fit_diagnostics.json", diagnostics)
    file_names = [*frames.keys(), "posterior_draw_candidate_probabilities.csv.gz", "fold_model_state.npz", "fit_diagnostics.json"]
    file_hashes = {name: sha256(fold_dir / name) for name in file_names}
    manifest = {
        "outer_fold": fold, "status": "FOLD_PREDICTIONS_COMPLETE",
        "config_hash": lock_hash, "input_hashes": input_hashes,
        "train_query_ids_sha256": canonical_hash(train_qids),
        "test_query_ids_sha256": canonical_hash(test_qids),
        "train_query_count": len(train_qids), "test_query_count": len(test_qids),
        "files": file_hashes, "test_qrels_used": False, "outer_metrics_computed": False,
    }
    atomic_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(5), help="Run/resume one fold; otherwise all folds")
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    verify_lock(config)
    input_hashes = verify_inputs()
    if input_hashes != config.get("input_hashes"):
        raise RuntimeError("Locked input hashes differ from the frozen outer-evaluation config")
    lock_hash = canonical_hash(config)
    helper = load_helper()
    assignment = load_fold_assignments()
    groups, _, features = load_features(helper)
    if set(assignment["query_id"].astype(str)) != set(groups):
        raise ValueError("Outer-fold assignments do not cover the frozen-pool query IDs")
    if RUN.exists():
        manifest_path = RUN / "run_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("Output directory exists without a resumable Frontier-F3.4 lock manifest")
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("config_hash") != lock_hash or prior.get("input_hashes") != input_hashes:
            raise RuntimeError("Output directory belongs to a different run; refusing overwrite")
    else:
        RUN.mkdir(parents=True)
    run_manifest = {
        "model_version": "Frontier-F3.4", "hypothesis_id": "H-F034",
        "status": "RUNNING_OUTER_PREDICTIONS", "evidence_status": "EXPLORATORY_OUTER",
        "config_hash": lock_hash, "input_hashes": input_hashes,
        "outer_results_read": False, "outer_test_qrels_used": False,
        "outer_metrics_computed": False, "outer_evaluation_count": 1,
        "folds_completed": [], "outer_results_may_guide_revisions": False,
        "python": sys.version.split()[0], "platform": platform.platform(),
        "started_unix": time.time(),
    }
    manifest_path = RUN / "run_manifest.json"
    if manifest_path.is_file():
        run_manifest.update(json.loads(manifest_path.read_text(encoding="utf-8")))
        if run_manifest.get("status") == "PREDICTIONS_LOCKED":
            lock_path = RUN / "PREDICTIONS_LOCKED.json"
            if not lock_path.is_file():
                raise RuntimeError("Locked status has no prediction lock record")
            print("PREDICTIONS_LOCKED; no work to resume")
            return
    atomic_json(manifest_path, run_manifest)
    folds = [args.fold] if args.fold is not None else list(range(5))
    for fold in folds:
        fold_manifest = write_fold(fold, groups, features, assignment, input_hashes, helper, lock_hash)
        run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        completed = set(run_manifest.get("folds_completed", []))
        completed.add(int(fold))
        run_manifest["folds_completed"] = sorted(completed)
        run_manifest["last_completed_fold"] = int(fold)
        run_manifest["last_fold_manifest_sha256"] = sha256(RUN / "folds" / ("outer_{}".format(fold)) / "fold_manifest.json")
        run_manifest["checkpoint_unix"] = time.time()
        atomic_json(manifest_path, run_manifest)
        print("FOLD_COMPLETE {} train={} test={} test_qrels_used=false".format(
            fold, fold_manifest["train_query_count"], fold_manifest["test_query_count"]))

    if args.fold is not None and len(set(run_manifest.get("folds_completed", []))) < 5:
        print("CHECKPOINT_SAVED; remaining outer folds were not run")
        return

    all_fold_manifests = {}
    for fold in range(5):
        path = RUN / "folds" / ("outer_{}".format(fold)) / "fold_manifest.json"
        if not path.is_file():
            raise RuntimeError("Cannot lock predictions before all five folds are complete")
        fm = json.loads(path.read_text(encoding="utf-8"))
        if fm.get("status") != "FOLD_PREDICTIONS_COMPLETE" or fm.get("test_qrels_used") or fm.get("outer_metrics_computed"):
            raise RuntimeError("Fold manifest violates pre-metric prediction lock policy")
        all_fold_manifests[str(fold)] = {
            "manifest_sha256": sha256(path), "files": fm["files"],
            "test_query_count": fm["test_query_count"],
        }
    lock_record = {
        "status": "PREDICTIONS_LOCKED", "model_version": "Frontier-F3.4",
        "hypothesis_id": "H-F034", "config_hash": lock_hash,
        "input_hashes": input_hashes, "folds": all_fold_manifests,
        "unique_oof_queries": 483, "candidate_pairs": 4813,
        "outer_test_qrels_read": False, "outer_metrics_computed": False,
        "locked_unix": time.time(),
        "instruction": "No method, threshold, model family, or hyperparameter changes after this lock; evaluation outputs are exploratory only.",
    }
    atomic_json(RUN / "PREDICTIONS_LOCKED.json", lock_record)
    run_manifest.update({
        "status": "PREDICTIONS_LOCKED", "folds_completed": list(range(5)),
        "prediction_lock_sha256": sha256(RUN / "PREDICTIONS_LOCKED.json"),
        "outer_results_read": False, "outer_test_qrels_used": False,
        "outer_metrics_computed": False, "completed_unix": time.time(),
    })
    atomic_json(manifest_path, run_manifest)
    print("PREDICTIONS_LOCKED folds=5 unique_queries=483 candidate_pairs=4813 outer_qrels_read=false outer_metrics_computed=false")


if __name__ == "__main__":
    main()
