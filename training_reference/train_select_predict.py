#!/usr/bin/env python3
"""Train-only page-grouped utility selection, followed by frozen blind prediction.

This runner never opens test qrels or result/metric files. It reuses the saved
full-training posterior and exact proposals, and does not tune against test
outcomes. The penalised utility is a new decision rule; it does not inherit an
optimality statement proved for unpenalised expected discounted gain.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np


GAMMA_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
DRAWS = 128
QUANTILE = 0.05
MAX_MEAN_REDUCTION = 0.005
PROTOCOL_ID = "MIRAGE-TRAIN-SELECTED-DOWNSIDE-UTILITY-20261015-v1"
OUTPUT_METHODS = [
    "Frozen BGE", "Always posterior-mean", "Pointwise logistic", "Adapted EAU",
    "Gain-only fifth-percentile", "Downside-penalised Bayes-UQ",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def progress(output: Path, stage: str, **values) -> None:
    receipt = {"time_utc": now(), "stage": stage, **values}
    atomic_json(output / "PROGRESS.json", receipt)
    with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(receipt, sort_keys=True), flush=True)


def load_vendor():
    path = Path(__file__).resolve().parent / "vendor" / "run_mirage_f34_server.py"
    if not path.is_file():
        raise FileNotFoundError(f"Required original training module is missing: {path}")
    spec = importlib.util.spec_from_file_location("mirage_original_training", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load original training module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if module.DRAWS != DRAWS or module.RHO != 2.0:
        raise RuntimeError("Original posterior draw count or rho differs from protocol")
    return module, path


def discounts(module, orders: list[list[int]], candidate_count: int) -> np.ndarray:
    values = np.zeros((len(orders), candidate_count), dtype=np.float64)
    for qi, order in enumerate(orders):
        if sorted(order) != list(range(candidate_count)):
            raise RuntimeError("Invalid candidate-index permutation")
        for rank, ci in enumerate(order, start=1):
            values[qi, ci] = module.d10(rank)
    return values


def utility_arrays(module, data: dict, bma: np.ndarray, probs: np.ndarray) -> dict:
    expected_shape = (DRAWS, len(data["qids"]), 50)
    if probs.shape != expected_shape or not np.isfinite(probs).all():
        raise RuntimeError("Invalid posterior probability array")
    if not np.allclose(probs.sum(axis=2), 1.0, atol=1e-12, rtol=1e-12):
        raise RuntimeError("Candidate posterior probabilities do not sum to one")
    proposal = [module.candidate_order(bma[i], data["ids"][i])
                for i in range(len(data["qids"]))]
    reference = [module.candidate_order(data["base"][i], data["ids"][i])
                 for i in range(len(data["qids"]))]
    delta = discounts(module, proposal, 50) - discounts(module, reference, 50)
    gain = np.einsum("dqc,qc->qd", probs, delta)
    downside = np.einsum("dqc,qc->qd", probs, np.maximum(-delta, 0.0))
    if not np.isfinite(gain).all() or not np.isfinite(downside).all():
        raise RuntimeError("Non-finite posterior utility draws")
    return {"gain": gain, "downside": downside,
            "proposal": proposal, "reference": reference}


def train_metrics(ndcg: np.ndarray, base_ndcg: np.ndarray,
                  accepted: np.ndarray | None = None) -> dict:
    delta = ndcg - base_ndcg
    loss = np.maximum(-delta, 0.0)
    tail_count = max(1, int(math.ceil(0.10 * len(loss))))
    result = {
        "queries": int(len(ndcg)), "nDCG@10": float(ndcg.mean()),
        "QDR": float(np.mean(delta < -1e-12)), "ED": float(loss.mean()),
        "CVaR10%": float(np.sort(loss)[-tail_count:].mean()),
        "CVaR_tail_count": tail_count,
    }
    if accepted is not None:
        result.update(accepted=int(accepted.sum()), fallback=int((~accepted).sum()),
                      acceptance_rate=float(accepted.mean()))
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def slice_data(data: dict, indices: np.ndarray) -> dict:
    return {"qids": [data["qids"][i] for i in indices],
            "ids": data["ids"][indices], "base": data["base"][indices]}


def choose_rule(module, source: Path, output: Path, vendor_path: Path) -> dict:
    train_feature = source / "features" / "candidate_features_train.csv"
    train_queries = source / "inputs" / "queries_train_selected.jsonl"
    train = module.load_grouped_features(train_feature, train_queries, training=True)
    if len(train["qids"]) != 1000:
        raise RuntimeError("Expected exactly 1,000 frozen training queries")
    page_folds = [sorted(map(int, fold)) for fold in module.make_pages(source)[1]]
    train_dir = output / "train_oof"
    checkpoint_dir = train_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    gains = np.full((1000, DRAWS), np.nan, dtype=np.float64)
    downsides = np.full_like(gains, np.nan)
    always_ndcg = np.full(1000, np.nan, dtype=np.float64)
    base_ndcg = np.full(1000, np.nan, dtype=np.float64)
    fold_indices = np.full(1000, -1, dtype=np.int64)
    diagnostics = []
    for fold, validation_pages in enumerate(page_folds):
        progress(output, "TRAIN_OOF_FIT_STARTED", fold=fold)
        is_validation = np.isin(train["pages"], validation_pages)
        validation = np.flatnonzero(is_validation)
        fitting = np.flatnonzero(~is_validation)
        if not len(validation) or not len(fitting):
            raise RuntimeError("Empty training or validation fold")
        if set(train["pages"][fitting]) & set(train["pages"][validation]):
            raise RuntimeError("Page-group leakage in training folds")
        if (fold_indices[validation] >= 0).any():
            raise RuntimeError("A query occurs in multiple validation folds")
        theta0, cov0, mean0, scale0, receipt0 = module.fit_f34(
            train["x"][fitting], train["y"][fitting], rho=0.0
        )
        theta, cov, mean, scale, receipt2 = module.fit_f34(
            train["x"][fitting], train["y"][fitting], rho=2.0,
            initial_beta=theta0,
        )
        # The inherited fitter may return success=False despite a small gradient.
        # This protocol stops on any reported fit failure instead of ignoring it.
        for receipt in (receipt0, receipt2):
            if not receipt.get("success", False):
                raise RuntimeError(f"Reported optimisation failure in fold {fold}: {receipt}")
        seed_label = f"MIRAGE-TRAIN-UTILITY|fold={fold}|20261015|128"
        seed = module.stable_seed(seed_label)
        draws = np.random.Generator(np.random.PCG64(seed)).multivariate_normal(
            theta, cov, size=DRAWS, check_valid="raise"
        )
        bma, probs = module.f34_probabilities(
            train["x"][validation], (theta, cov, mean, scale), draws
        )
        arrays = utility_arrays(module, slice_data(train, validation), bma, probs)
        gains[validation] = arrays["gain"]
        downsides[validation] = arrays["downside"]
        always_ndcg[validation] = module.rank_ndcg_matrix(
            arrays["proposal"], train["y"][validation]
        )
        base_ndcg[validation] = module.rank_ndcg_matrix(
            arrays["reference"], train["y"][validation]
        )
        fold_indices[validation] = fold
        fold_path = checkpoint_dir / f"fold_{fold}.npz"
        np.savez_compressed(
            fold_path, theta_rho0=theta0, covariance_rho0=cov0,
            mean_rho0=mean0, scale_rho0=scale0,
            theta_rho2=theta, covariance_rho2=cov, mean_rho2=mean,
            scale_rho2=scale, draws_rho2=draws,
            seed_rho2=np.asarray(seed, dtype=np.uint64),
            fitting_indices=fitting, validation_indices=validation,
        )
        receipt = {
            "fold": fold, "validation_pages": validation_pages,
            "fitting_queries": int(len(fitting)),
            "validation_queries": int(len(validation)),
            "seed_label": seed_label, "posterior_seed": seed,
            "rho0_fit": receipt0, "rho2_fit": receipt2,
            "checkpoint": str(fold_path), "checkpoint_sha256": sha256(fold_path),
        }
        diagnostics.append(receipt)
        atomic_json(train_dir / "fold_diagnostics.json", {"folds": diagnostics})
        progress(output, "TRAIN_OOF_FIT_COMPLETED", fold=fold,
                 validation_queries=int(len(validation)))
    if (fold_indices < 0).any() or not all(np.isfinite(a).all() for a in
                                          [gains, downsides, always_ndcg, base_ndcg]):
        raise RuntimeError("Incomplete OOF predictions")
    np.savez_compressed(
        train_dir / "oof_gain_downside.npz", query_ids=np.asarray(train["qids"]),
        source_page_offset=train["pages"], fold=fold_indices,
        gain_draws=gains, downside_draws=downsides,
        always_ndcg10=always_ndcg, frozen_bge_ndcg10=base_ndcg,
    )
    always_metrics = train_metrics(always_ndcg, base_ndcg)
    atomic_json(train_dir / "always_oof_metrics.json", always_metrics)
    grid_results = []
    oof_actions = []
    for gamma in GAMMA_GRID:
        quantiles = np.quantile(gains - gamma * downsides, QUANTILE,
                                axis=1, method="linear")
        accepted = quantiles > 0.0
        values = np.where(accepted, always_ndcg, base_ndcg)
        metrics = train_metrics(values, base_ndcg, accepted)
        mean_ok = metrics["nDCG@10"] >= always_metrics["nDCG@10"] - MAX_MEAN_REDUCTION
        tail_ok = metrics["CVaR10%"] <= always_metrics["CVaR10%"]
        grid_results.append({"gamma": gamma, **metrics,
                             "mean_constraint_passed": bool(mean_ok),
                             "tail_constraint_passed": bool(tail_ok),
                             "eligible": bool(mean_ok and tail_ok)})
        for qi, qid in enumerate(train["qids"]):
            oof_actions.append({"query_id": qid, "fold": int(fold_indices[qi]),
                                "gamma": gamma, "utility_q05": float(quantiles[qi]),
                                "accepted": bool(accepted[qi]),
                                "ndcg10": float(values[qi])})
    write_csv(train_dir / "gamma_grid_results.csv", grid_results)
    write_csv(train_dir / "gamma_grid_oof_actions.csv", oof_actions)
    eligible = [r for r in grid_results if r["eligible"]]
    if eligible:
        selected = min(eligible, key=lambda r: (r["CVaR10%"], -r["nDCG@10"], r["gamma"]))
        reason = "minimum_oof_cvar_then_maximum_ndcg_then_minimum_gamma"
    else:
        selected = next(r for r in grid_results if r["gamma"] == 0.0)
        reason = "no_eligible_rule_fallback_to_gamma_zero_constraints_not_satisfied"
    rule = {
        "protocol": PROTOCOL_ID, "selected_at_utc": now(),
        "selected_gamma": selected["gamma"], "selection_reason": reason,
        "selection_constraints_satisfied": bool(eligible),
        "gamma_grid": GAMMA_GRID, "quantile": QUANTILE,
        "quantile_method": "linear", "acceptance_operator": "> 0.0",
        "gain_definition": "sum_c p_s(c) * (d_proposal(c)-d_reference(c))",
        "downside_definition": "sum_c p_s(c) * max(d_reference(c)-d_proposal(c),0)",
        "posterior_draws": DRAWS, "rho": 2.0,
        "maximum_ndcg_reduction_vs_always_oof": MAX_MEAN_REDUCTION,
        "cvar_definition": "mean of largest ceil(0.10*n) nonnegative losses versus frozen BGE",
        "training_page_folds": page_folds, "always_oof_metrics": always_metrics,
        "selected_oof_metrics": selected, "all_grid_results": grid_results,
        "train_feature_file": str(train_feature), "train_feature_sha256": sha256(train_feature),
        "train_queries_file": str(train_queries), "train_queries_sha256": sha256(train_queries),
        "vendor_script": str(vendor_path), "vendor_script_sha256": sha256(vendor_path),
        "runner_script_sha256": sha256(Path(__file__).resolve()),
        "selection_data": "training OOF only; no test features, checkpoint, qrels or test outcomes read before this lock",
        "optimality_scope": "new penalised decision utility; no claim of inherited unpenalised sorting optimality",
    }
    atomic_json(output / "SELECTED_RULE.json", rule)
    progress(output, "TRAIN_RULE_FROZEN", selected_gamma=rule["selected_gamma"],
             constraints_satisfied=rule["selection_constraints_satisfied"])
    return rule


def project_blind_query_metadata(source: Path, output: Path) -> tuple[Path, Path]:
    # The staged name can be queries_test_selected.jsonl. Only the two fields
    # needed to align features and page groups are projected into the new job.
    # Other metadata fields, if present, never enter fitting, policy selection,
    # inference or prediction output.
    for name in ("queries_test_metadata_blind.jsonl", "queries_test_selected.jsonl"):
        path = source / "inputs" / name
        if not path.is_file():
            continue
        projected = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                projected.append({"query_id": str(row["query_id"]),
                                  "source_page_offset": int(row["source_page_offset"])})
        if len(projected) != 2000 or len({row["query_id"] for row in projected}) != 2000:
            raise RuntimeError("Unexpected test metadata query set")
        destination = output / "inputs" / "test_query_metadata_blind.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in projected),
                               encoding="utf-8")
        return path, destination
    raise FileNotFoundError("No staged blind test-query metadata file was found")


def predict(module, source: Path, output: Path, rule: dict) -> None:
    rule_path = output / "SELECTED_RULE.json"
    frozen_rule_sha = sha256(rule_path)
    if json.loads(rule_path.read_text(encoding="utf-8")) != rule:
        raise RuntimeError("Training-selection lock changed before test prediction")
    progress(output, "BLIND_TEST_PREDICTION_STARTED")
    feature_path = source / "features" / "candidate_features_test_blind.csv"
    source_query_path, query_path = project_blind_query_metadata(source, output)
    checkpoint_path = source / "checkpoints" / "f34_parameters.npz"
    source_lock_path = source / "PREDICTION_LOCK.json"
    original_path = source / "predictions" / "predictions_locked.jsonl"
    source_lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    required_hashes = {
        "test_feature_sha256": (feature_path, sha256(feature_path)),
        "model_parameters_sha256": (checkpoint_path, sha256(checkpoint_path)),
        "prediction_sha256": (original_path, sha256(original_path)),
        "train_feature_sha256": (Path(rule["train_feature_file"]), rule["train_feature_sha256"]),
    }
    for field, (path, actual) in required_hashes.items():
        if source_lock.get(field) != actual:
            raise RuntimeError(f"Source lock hash mismatch for {field}: {path}")
    test = module.load_grouped_features(feature_path, query_path, training=False)
    if len(test["qids"]) != 2000:
        raise RuntimeError("Expected exactly 2,000 frozen test queries")
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        params = tuple(checkpoint[key] for key in
                       ("theta_rho2", "covariance_rho2", "mean_rho2", "scale_rho2"))
        draws = checkpoint["draws_rho2"]
    if draws.shape != (128, 8) or not np.isfinite(draws).all():
        raise RuntimeError("Original fullfit posterior draws have changed")
    bma, probs = module.f34_probabilities(test["x"], params, draws)
    arrays = utility_arrays(module, test, bma, probs)
    gamma = float(rule["selected_gamma"])
    gain_q05 = np.quantile(arrays["gain"], QUANTILE, axis=1, method="linear")
    selected_q05 = np.quantile(arrays["gain"] - gamma * arrays["downside"],
                               QUANTILE, axis=1, method="linear")
    gamma0_accept = gain_q05 > 0.0
    selected_accept = selected_q05 > 0.0
    if (selected_accept & ~gamma0_accept).any():
        raise RuntimeError("Nonnegative downside penalty unexpectedly expanded the acceptance set")
    original_rows = [json.loads(line) for line in original_path.read_text(encoding="utf-8").splitlines() if line]
    original = {str(row["query_id"]): row for row in original_rows}
    if len(original_rows) != 2000 or len(original) != 2000 or set(original) != set(test["qids"]):
        raise RuntimeError("Original locked query set differs")
    pred_dir = output / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    records, action_rows = [], []
    baseline_keys = {"Pointwise logistic": "Pointwise logistic reranker", "Adapted EAU": "Adapted EAU"}
    for qi, qid in enumerate(test["qids"]):
        old = original[qid]
        ids = test["ids"][qi]
        proposal = [str(int(ids[i])) for i in arrays["proposal"][qi]]
        reference = [str(int(ids[i])) for i in arrays["reference"][qi]]
        if proposal != old["orders"]["Always posterior-mean"]:
            raise RuntimeError(f"Fullfit proposal differs from original lock for query {qid}")
        if reference != old["orders"]["Frozen BGE"]:
            raise RuntimeError(f"Frozen BGE order differs from original lock for query {qid}")
        if bool(gamma0_accept[qi]) != bool(old["q05_accepted"]):
            raise RuntimeError(f"Gamma-zero action differs from original lock for query {qid}")
        if int(test["pages"][qi]) != int(old["source_page_offset"]):
            raise RuntimeError(f"Source-page identity differs for query {qid}")
        orders = {"Frozen BGE": reference, "Always posterior-mean": proposal}
        for output_name, source_name in baseline_keys.items():
            old_order = list(map(str, old["orders"][source_name]))
            if len(old_order) != 50 or set(old_order) != set(reference):
                raise RuntimeError(f"Invalid locked baseline order for {qid}: {source_name}")
            orders[output_name] = old_order
        orders["Gain-only fifth-percentile"] = proposal if gamma0_accept[qi] else reference
        orders["Downside-penalised Bayes-UQ"] = proposal if selected_accept[qi] else reference
        if orders["Gain-only fifth-percentile"] != old["orders"]["Enhanced F3.4 + Q05"]:
            raise RuntimeError(f"Gamma-zero final order differs from original lock for query {qid}")
        record = {
            "query_id": qid, "source_page_offset": int(test["pages"][qi]),
            "orders": orders, "selected_gamma": gamma,
            "gamma0_accepted": bool(gamma0_accept[qi]),
            "selected_accepted": bool(selected_accept[qi]),
            "gain_q05": float(gain_q05[qi]),
            "penalised_gain_q05": float(selected_q05[qi]),
            "mean_expected_downside": float(arrays["downside"][qi].mean()),
        }
        records.append(record)
        action_rows.append({k: v for k, v in record.items() if k != "orders"})
    prediction_path = pred_dir / "predictions_locked.jsonl"
    prediction_path.write_text("".join(json.dumps(r, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
                                       for r in records), encoding="utf-8")
    action_path = pred_dir / "actions.csv"
    write_csv(action_path, action_rows)
    draw_path = pred_dir / "posterior_gain_downside_draws.npz"
    np.savez_compressed(draw_path, query_ids=np.asarray(test["qids"]),
                        source_page_offset=test["pages"],
                        gain_draws=arrays["gain"], downside_draws=arrays["downside"])
    if sha256(rule_path) != frozen_rule_sha:
        raise RuntimeError("Training-selection lock changed during prediction")
    lock = {
        "protocol": PROTOCOL_ID, "status": "PREDICTIONS_LOCKED",
        "locked_at_utc": now(), "selected_gamma": gamma,
        "selection_file": str(rule_path), "selection_sha256": frozen_rule_sha,
        "prediction_file": str(prediction_path), "prediction_sha256": sha256(prediction_path),
        "actions_file": str(action_path), "actions_sha256": sha256(action_path),
        "draws_file": str(draw_path), "draws_sha256": sha256(draw_path),
        "test_queries": 2000, "candidate_count": 50, "posterior_draws": DRAWS,
        "methods": OUTPUT_METHODS, "test_qrels_read": False,
        "test_results_read": False, "test_gamma_grid_evaluated": False,
        "source_root": str(source), "source_protocol": source_lock.get("protocol"),
        "source_lock_file": str(source_lock_path), "source_lock_sha256": sha256(source_lock_path),
        "source_checkpoint": str(checkpoint_path), "source_checkpoint_sha256": sha256(checkpoint_path),
        "source_predictions": str(original_path), "source_predictions_sha256": sha256(original_path),
        "source_test_features": str(feature_path), "source_test_features_sha256": sha256(feature_path),
        "source_test_metadata": str(source_query_path), "source_test_metadata_sha256": sha256(source_query_path),
        "projected_test_metadata": str(query_path), "projected_test_metadata_sha256": sha256(query_path),
        "test_metadata_fields_used": ["query_id", "source_page_offset"],
        "source_manifest": source_lock,
        "identity_checks": {"always_full_orders_equal": True, "frozen_bge_orders_equal": True,
                            "gamma_zero_actions_equal": True, "gamma_zero_final_orders_equal": True},
        "utility_scope": "Candidate-only posterior with preserved fullfit proposal. New downside-penalised acceptance utility.",
    }
    atomic_json(output / "PREDICTIONS_LOCKED.json", lock)
    atomic_json(output / "PREDICTION_COMPLETED.json", {
        "status": "PREDICTION_COMPLETED", "completed_at_utc": now(),
        "prediction_lock_sha256": sha256(output / "PREDICTIONS_LOCKED.json"),
        "selected_gamma": gamma, "test_queries": 2000,
    })
    progress(output, "PREDICTION_COMPLETED", selected_gamma=gamma, test_queries=2000)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise RuntimeError("Source and output must be separate, non-nested directories")
    protected = ["SELECTED_RULE.json", "PREDICTIONS_LOCKED.json", "PREDICTION_COMPLETED.json",
                 "PREDECLARED_PROTOCOL.json", "FAILED.json", "train_oof", "predictions"]
    if any((output / name).exists() for name in protected):
        raise RuntimeError("Output already contains experiment artifacts; refusing to overwrite")
    output.mkdir(parents=True, exist_ok=True)
    try:
        progress(output, "STARTED", protocol=PROTOCOL_ID)
        module, vendor_path = load_vendor()
        atomic_json(output / "PREDECLARED_PROTOCOL.json", {
            "protocol": PROTOCOL_ID, "recorded_before_fitting_at_utc": now(),
            "gamma_grid": GAMMA_GRID, "quantile": QUANTILE,
            "quantile_method": "linear", "threshold": 0.0,
            "posterior_draws": DRAWS, "rho": 2.0,
            "fold_seed_template": "MIRAGE-TRAIN-UTILITY|fold={i}|20261015|128",
            "selection_constraints": {"nDCG_at_least_always_oof_minus": MAX_MEAN_REDUCTION,
                                      "CVaR_no_greater_than_always_oof": True},
            "selection_priority": ["CVaR10% ascending", "nDCG@10 descending", "gamma ascending"],
            "no_eligible_rule": "gamma=0; record constraints not satisfied",
            "vendor_script_sha256": sha256(vendor_path),
            "runner_script_sha256": sha256(Path(__file__).resolve()),
            "numpy_version": np.__version__,
            "test_qrels_and_test_outcomes_used_for_selection": False,
        })
        rule = choose_rule(module, source, output, vendor_path)
        predict(module, source, output, rule)
    except Exception as exc:
        atomic_json(output / "FAILED.json", {
            "status": "FAILED", "failed_at_utc": now(),
            "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        progress(output, "FAILED", error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == "__main__":
    main()
