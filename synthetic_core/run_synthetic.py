#!/usr/bin/env python3
"""Run the archived Bayes-UQ mechanism on explicitly synthetic cached inputs.

This is an interface demonstration, not a reconstruction of reported results.
The fitted objective and posterior primitives are imported unchanged from vendor.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STATUS = "SYNTHETIC_INTERFACE_DEMONSTRATION_NOT_PAPER_RESULTS"


def load_vendor(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / "vendor" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


model = load_vendor("track_f033_contextual_posterior")
runner = load_vendor("run_frontier_f34_outer_once")
evaluation = load_vendor("analyze_frontier_f34_outer_paired_v2")


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def make_fixture(path: Path) -> None:
    """Create fixed-seed toy features; no real records or measured results."""
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.Generator(np.random.PCG64(20261015))
    features, assignments, train_qrels, test_qrels = [], [], [], []
    for qi in range(40):
        qid = "synthetic_q_{:03d}".format(qi)
        role = "train" if qi < 32 else "test"
        assignments.append({"query_id": qid, "role": role})
        n = 8
        text, image, meta = rng.uniform(0, 1, size=(3, n))
        image_count = rng.integers(0, 4, size=n)
        categories = rng.integers(0, 3, size=n)
        text_order = np.argsort(-text, kind="stable")
        image_order = np.argsort(-image, kind="stable")
        text_rank, image_rank = np.empty(n), np.empty(n)
        text_rank[text_order] = np.arange(1, n + 1)
        image_rank[image_order] = np.arange(1, n + 1)
        gap = float(text[text_order[0]] - text[text_order[1]])
        agreement = int(int(text_order[0]) == int(np.argmax(meta)))
        ambiguity = float(rng.uniform(0, 1))
        image_ratio = float(np.mean(image_count > 0))
        # This arbitrary fixed toy target rule is not a real relevance annotation.
        target = int(np.argmax(1.5 * text + 0.3 * image + rng.normal(0, 0.15, n)))
        if qi % 7 == 0:
            target = -1
        for j in range(n):
            cid = "synthetic_d_{:03d}_{:02d}".format(qi, j)
            features.append({
                "query_id": qid, "candidate_id": cid,
                "base_score": float(0.7 * text[j] + 0.1 * meta[j]),
                "doc_cat_is_disease": int(categories[j] == 0),
                "doc_cat_is_guidance": int(categories[j] == 1),
                "doc_cat_is_phenology": int(categories[j] == 2),
                "has_disease_label": int(rng.integers(0, 2)),
                "has_stage_label": int(rng.integers(0, 2)),
                "image_rank": float(1 / image_rank[j]), "image_score": float(image[j]),
                "meta_image_consistency": float(rng.uniform(0, 1)),
                "meta_score": float(meta[j]), "num_images": int(image_count[j]),
                "text_rank": float(1 / text_rank[j]), "text_score": float(text[j]),
                "query_uncertainty_weight": ambiguity, "text_top12_gap": gap,
                "top1_agree_text_meta": agreement, "candidate_image_ratio": image_ratio,
            })
            qrels = train_qrels if role == "train" else test_qrels
            qrels.append({"query_id": qid, "candidate_id": cid, "relevance": int(j == target)})
    for name, rows in (("features.csv", features), ("assignments.csv", assignments),
                       ("train_qrels.csv", train_qrels), ("test_qrels.csv", test_qrels)):
        pd.DataFrame(rows).to_csv(path / name, index=False)
    write_json(path / "fixture_manifest.json", {
        "status": STATUS, "seed": 20261015, "queries": 40, "train_queries": 32,
        "test_queries": 8, "candidates_per_query": 8,
        "warning": "All data are generated toy values. No paper metric is reproduced.",
    })


def select_order(ids, base_scores, draw_probabilities):
    """Direct application of archived outer-run lines 282-303, 347-366."""
    base_order = runner.order_from_scores(ids, np.asarray(base_scores))
    candidate_draws = np.asarray(draw_probabilities, dtype=float)
    if candidate_draws.shape != (128, len(ids)):
        raise ValueError("Expected 128 candidate-probability draws")
    proposal = runner.order_from_scores(ids, candidate_draws.mean(axis=0))
    gains = candidate_draws @ runner.gain_vector(ids, proposal, base_order)
    quantile = float(np.quantile(gains, 0.05, method="linear"))
    accepted = bool(quantile > 0.0)
    selected = proposal if accepted else base_order
    return base_order, proposal, selected, accepted, quantile, gains


def check_rule_fixtures():
    """Controlled rule checks are independent of the fitted toy example."""
    ids = ["candidate_b", "candidate_a"]
    fixtures = []
    # A proposal improving every draw must be accepted.
    draws = np.tile([0.1, 0.9], (128, 1))
    base, proposal, selected, accepted, q05, _ = select_order(ids, [1.0, 0.0], draws)
    assert accepted and selected == proposal and selected != base
    fixtures.append({"case": "positive_lower_tail_accept", "accepted": accepted, "q05": q05})
    # Its posterior-mean proposal has positive gain, but the lower tail is negative.
    draws = np.vstack([np.tile([0.9, 0.1], (16, 1)), np.tile([0.1, 0.9], (112, 1))])
    base, proposal, selected, accepted, q05, gains = select_order(ids, [1.0, 0.0], draws)
    assert gains.mean() > 0 and q05 < 0 and not accepted and selected == base
    fixtures.append({"case": "positive_mean_negative_tail_fallback", "accepted": accepted,
                     "q05": q05, "mean_gain": float(gains.mean())})
    # Canonical ties and strict > 0: identical proposal/reference yields fallback.
    draws = np.tile([0.5, 0.5], (128, 1))
    base, proposal, selected, accepted, q05, _ = select_order(ids, [0.5, 0.5], draws)
    assert [ids[j] for j in base] == ["candidate_a", "candidate_b"]
    assert q05 == 0 and not accepted and selected == base == proposal
    fixtures.append({"case": "zero_gain_strict_boundary_and_canonical_tie", "accepted": accepted,
                     "q05": q05, "selected_ids": [ids[j] for j in selected]})
    return fixtures


def run(fixture_dir: Path, output_dir: Path) -> None:
    manifest = json.loads((fixture_dir / "fixture_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != STATUS:
        raise ValueError("This entry point accepts only explicitly synthetic fixtures")
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments = pd.read_csv(fixture_dir / "assignments.csv", dtype={"query_id": str})
    frame = pd.read_csv(fixture_dir / "features.csv", dtype={"query_id": str, "candidate_id": str})
    train_ids = sorted(assignments.loc[assignments.role.eq("train"), "query_id"])
    test_ids = sorted(assignments.loc[assignments.role.eq("test"), "query_id"])
    if not train_ids or not test_ids or set(train_ids) & set(test_ids):
        raise ValueError("Expected a nonempty disjoint train/test split")
    if frame.duplicated(["query_id", "candidate_id"]).any():
        raise ValueError("Duplicate query/candidate key")
    if set(frame.query_id) != set(train_ids) | set(test_ids):
        raise ValueError("Assignment and feature query sets differ")
    train_qrels = pd.read_csv(fixture_dir / "train_qrels.csv",
                              dtype={"query_id": str, "candidate_id": str})
    if set(train_qrels.query_id) != set(train_ids):
        raise ValueError("Training qrels contain an unexpected query")
    qmap = {(r.query_id, r.candidate_id): int(r.relevance) for r in train_qrels.itertuples()}
    groups = {}
    for qid, block in frame.sort_values(["query_id", "candidate_id"]).groupby("query_id", sort=True):
        ids = block.candidate_id.tolist()
        labels = np.array([qmap[(qid, cid)] for cid in ids]) if qid in train_ids else np.zeros(len(ids), dtype=int)
        if labels.sum() > 1:
            raise ValueError("At most one positive target per query is supported")
        groups[qid] = {
            "ids": ids, "X_raw": block[list(runner.FEATURE_NAMES)].to_numpy(dtype=float),
            "base": block.base_score.to_numpy(dtype=float), "labels": labels,
            "agreement_group": int(block.top1_agree_text_meta.iloc[0]),
            "null_raw": np.asarray([block.meta_score.max(), block.query_uncertainty_weight.iloc[0],
                                    block.text_top12_gap.iloc[0], block.top1_agree_text_meta.iloc[0],
                                    block.candidate_image_ratio.iloc[0]], dtype=float),
        }
    xmean, xstd = model.standardizer(np.vstack([groups[q]["X_raw"] for q in train_ids]))
    zmean, zstd = model.standardizer(np.vstack([groups[q]["null_raw"] for q in train_ids]))
    xs = {q: (g["X_raw"] - xmean) / xstd for q, g in groups.items()}
    zs = {q: (g["null_raw"] - zmean) / zstd for q, g in groups.items()}
    design = model.query_design(groups, xs, zs, train_ids, np.array([], dtype=int),
                                float(np.mean([groups[q]["agreement_group"] for q in train_ids])))
    theta0, cov0, diag0 = model.fit_categorical(design, train_ids, 12, 0, 1.0, 0.0)
    theta, cov, diag = model.fit_categorical(design, train_ids, 12, 0, 1.0, 2.0, initial=theta0)
    seed = model.stable_seed("Bayes-UQ|synthetic-interface|seed=20261015|rho=2|draws=128")
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = rng.multivariate_normal(theta, cov, size=128, check_valid="raise")
    np.savez(output_dir / "synthetic_model_state.npz", theta=theta, covariance=cov,
             posterior_draws=draws, xmean=xmean, xstd=xstd, zmean=zmean, zstd=zstd)
    predictions, actions, utility_rows, selected_orders = [], [], [], {}
    for qid in test_ids:
        rec = groups[qid]
        ids = rec["ids"]
        probability = model.categorical_draw_probs(xs[qid], zs[qid], draws)
        base, proposal, selected, accepted, q05, gains = select_order(ids, rec["base"], probability[:, 1:])
        selected_orders[qid] = {"backbone": [ids[j] for j in base],
                                "always_proposal": [ids[j] for j in proposal],
                                "selective": [ids[j] for j in selected]}
        actions.append({"query_id": qid, "accepted": int(accepted), "fallback": int(not accepted),
                        "posterior_mean_gain": float(gains.mean()), "posterior_q05_gain": q05,
                        "base_order": "|".join(selected_orders[qid]["backbone"]),
                        "proposal_order": "|".join(selected_orders[qid]["always_proposal"]),
                        "selected_order": "|".join(selected_orders[qid]["selective"])})
        for j, cid in enumerate(ids):
            predictions.append({"query_id": qid, "candidate_id": cid,
                                "mean_candidate_probability": float(probability[:, j + 1].mean()),
                                "mean_null_probability": float(probability[:, 0].mean()),
                                "base_rank": base.index(j) + 1, "proposal_rank": proposal.index(j) + 1,
                                "selected_rank": selected.index(j) + 1})
        utility_rows.extend({"query_id": qid, "draw": i, "utility_gain": float(g)} for i, g in enumerate(gains))
    # Persist predictions/actions before opening held-out qrels.
    pd.DataFrame(predictions).to_csv(output_dir / "synthetic_candidate_predictions.csv", index=False)
    pd.DataFrame(actions).to_csv(output_dir / "synthetic_actions.csv", index=False)
    pd.DataFrame(utility_rows).to_csv(output_dir / "synthetic_utility_draws.csv", index=False)
    write_json(output_dir / "decision_order.json", {"status": STATUS, "test_qrels_opened": False,
                                                   "decisions_written_before_test_qrels": True})
    test_qrels = pd.read_csv(fixture_dir / "test_qrels.csv", dtype={"query_id": str, "candidate_id": str})
    if set(test_qrels.query_id) != set(test_ids):
        raise ValueError("Test qrels do not match held-out queries")
    outcomes = {name: [] for name in ("backbone", "always_proposal", "selective")}
    for qid in test_ids:
        qrels = {r.candidate_id: int(r.relevance) for r in test_qrels[test_qrels.query_id.eq(qid)].itertuples()}
        if set(qrels) != set(groups[qid]["ids"]):
            raise ValueError("Held-out qrel candidate set mismatch")
        base_ndcg, _ = evaluation.exact_metrics(selected_orders[qid]["backbone"], qrels)
        for policy, order in selected_orders[qid].items():
            ndcg, mrr = evaluation.exact_metrics(order, qrels)
            outcomes[policy].append({"query_id": qid, "ndcg10": ndcg, "mrr10": mrr,
                                     "delta_ndcg10": ndcg - base_ndcg})
    summaries = [evaluation.method_summary(name, rows) for name, rows in outcomes.items()]
    pd.DataFrame(summaries).to_csv(output_dir / "synthetic_evaluation.csv", index=False)
    pd.DataFrame([dict(policy=name, **row) for name, rows in outcomes.items() for row in rows]).to_csv(
        output_dir / "synthetic_query_outcomes.csv", index=False)
    write_json(output_dir / "fit_diagnostics.json", {"status": STATUS, "rho0": diag0, "rho2": diag,
                                                    "posterior_seed": seed, "posterior_draws": 128})
    rule_checks = check_rule_fixtures()
    write_json(output_dir / "rule_fixtures.json", {"status": STATUS, "checks": rule_checks})
    write_json(output_dir / "run_summary.json", {
        "status": STATUS, "train_queries": len(train_ids), "test_queries": len(test_ids),
        "candidate_features": list(runner.FEATURE_NAMES), "query_context": list(runner.NULL_NAMES),
        "accepted": sum(a["accepted"] for a in actions), "fallback": sum(a["fallback"] for a in actions),
        "decisions_written_before_test_qrels": True, "rule_fixtures_passed": len(rule_checks),
        "training_model": "unchanged archived candidate/null objective; lambda=1, rho=2, no interactions",
        "warning": "Synthetic examples validate execution and interfaces, not any published metric.",
    })
    print(json.dumps({"status": STATUS, "outputs": str(output_dir.resolve()),
                      "accepted": sum(a["accepted"] for a in actions),
                      "fallback": sum(a["fallback"] for a in actions),
                      "rule_fixtures_passed": len(rule_checks)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, default=HERE / "synthetic_fixture")
    parser.add_argument("--output-dir", type=Path, default=HERE / "synthetic_outputs")
    parser.add_argument("--make-fixture", action="store_true", help="Generate fixed-seed synthetic CSVs")
    args = parser.parse_args()
    if args.make_fixture:
        make_fixture(args.fixture_dir)
    run(args.fixture_dir, args.output_dir)
