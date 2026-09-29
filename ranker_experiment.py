"""Train a query-wise LightGBM ranker on the existing v10 candidate cache.

The script does not touch the production answer. It reconstructs only the
validation denominators from the documented pseudo-corpus sampling procedure,
then compares the new ranker with the saved CatBoost classifier on the same
query split. All models and data stay local; query IDs never enter features.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


DEFAULT_SOURCE = Path(__file__).resolve().parent
DEFAULT_DATA = DEFAULT_SOURCE / "dataset"


def truth_counts(source, data, val_keys):
    """Replay v10 corpus sampling to count all positives, including unretrieved ones."""
    sys.path.insert(0, str(source))
    from solution import contexts, sample_pseudo_corpus

    queries = pd.read_parquet(data / "benchmark_queries.parquet", columns=[
        "search_query", "search_location_id", "search_is_delivery_search",
        "search_infm_params_text", "search_category"])
    search_columns = sorted(c for c in queries if c.startswith("search_"))
    train = pd.read_parquet(data / "train.parquet", columns=search_columns + [
        "item_id", "item_location_id"])
    train["_context"] = contexts(train, search_columns)
    train = train.drop_duplicates(["_context", "item_id"]).reset_index(drop=True)
    items = pd.read_parquet(data / "benchmark_items.parquet", columns=["item_location_id"])
    pseudo = sample_pseudo_corpus(train.drop_duplicates("item_id").reset_index(drop=True),
                                  items, np.random.default_rng(2026), "location")
    eligible = train[train.item_id.isin(pseudo.item_id)]
    counts = eligible.groupby("_context", sort=False).size()
    result = counts.reindex(val_keys).to_numpy()
    assert np.isfinite(result).all() and np.min(result) >= 1
    return result


def recall_at_50(scores, labels, offsets, totals):
    """Match the challenge metric: average per-query recall over all relevant ads."""
    values = np.empty(len(totals), dtype=np.float64)
    for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        local = scores[start:end]
        best = np.argpartition(local, -min(50, len(local)))[-min(50, len(local)):]
        values[i] = labels[start:end][best].sum() / totals[i]
    return float(values.mean()), values


def choose_train_rows(base_scores, x, labels, offsets, top_baseline, top_text, random_rows):
    """Keep hard candidates, text candidates, and random negatives per query.

    Every observed positive is retained. Random negatives help the model score
    the full candidate set at inference, rather than only baseline's top list.
    """
    rng = np.random.default_rng(3105)
    chunks = []
    groups = []
    text = x["content_geo"].to_numpy()
    for start, end in zip(offsets[:-1], offsets[1:]):
        count = end - start
        def best(scores, k):
            k = min(k, count)
            return np.argpartition(scores[start:end], -k)[-k:]
        selected = np.concatenate([
            best(base_scores, top_baseline),
            best(text, top_text),
            rng.choice(count, min(random_rows, count), replace=False),
            np.flatnonzero(labels[start:end])])
        selected = np.unique(selected) + start
        chunks.append(selected)
        groups.append(len(selected))
    return np.concatenate(chunks), np.asarray(groups)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=Path("artifacts_ranker"))
    parser.add_argument("--rounds", type=int, default=350)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    v10 = args.source / "artifacts_v10"

    x_train, y_train, _, train_offsets = joblib.load(v10 / "rank_examples.joblib")
    x_val, y_val, _, val_offsets = joblib.load(v10 / "val_examples.joblib")
    val_keys = [int(key) for key in json.loads((v10 / "split.json").read_text())["validation_contexts"]]
    totals = truth_counts(args.source, args.data, val_keys)
    assert len(totals) == len(val_offsets) - 1
    assert np.all(totals >= np.add.reduceat(y_val, val_offsets[:-1]))

    baseline = CatBoostClassifier()
    baseline.load_model(v10 / "ranker.cbm")
    base_train = baseline.predict_proba(x_train)[:, 1]
    base_val = baseline.predict_proba(x_val)[:, 1]
    base_recall, base_values = recall_at_50(base_val, y_val, val_offsets, totals)
    print(f"Baseline {base_recall:.6f}; candidate recall "
          f"{np.mean(np.add.reduceat(y_val, val_offsets[:-1]) / totals):.6f}", flush=True)
    assert abs(base_recall - .83205) < 1e-6, "Pseudo-corpus reconstruction differs from v10"

    keep, groups = choose_train_rows(base_train, x_train, y_train, train_offsets,
                                     top_baseline=220, top_text=170, random_rows=80)
    print(f"Training rows: {len(keep)}; positives: {y_train[keep].sum()}", flush=True)
    train_set = lgb.Dataset(x_train.iloc[keep], label=y_train[keep], group=groups,
                            free_raw_data=False)
    model = lgb.train({
        "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [50],
        "lambdarank_truncation_level": 70, "label_gain": [0, 1],
        "learning_rate": .045, "num_leaves": 31, "max_depth": -1,
        "min_data_in_leaf": 80, "feature_fraction": .88, "bagging_fraction": .85,
        "bagging_freq": 1, "lambda_l2": 10, "max_bin": 127,
        "num_threads": 8, "verbosity": -1, "seed": 2026,
    }, train_set, num_boost_round=args.rounds,
    callbacks=[lgb.log_evaluation(period=100)])
    ranking = model.predict(x_val, num_threads=8)
    # Rank scores have a different scale than classifier log-odds. Within each
    # query, standardize both before trying their convex combinations.
    logit = np.log(np.clip(base_val, 1e-8, 1 - 1e-8) /
                   np.clip(1 - base_val, 1e-8, 1 - 1e-8))
    rank_scaled = ranking.copy()
    base_scaled = logit.copy()
    for start, end in zip(val_offsets[:-1], val_offsets[1:]):
        for values in (rank_scaled, base_scaled):
            part = values[start:end]
            part -= part.mean()
            part /= max(part.std(), 1e-5)
    report = {"baseline": base_recall, "candidate_recall": float(np.mean(
        np.add.reduceat(y_val, val_offsets[:-1]) / totals)),
        "train_rows": int(len(keep)), "rounds": args.rounds, "blends": {}}
    for weight in (0.0, .2, .4, .6, .8, 1.0):
        scores = weight * rank_scaled + (1 - weight) * base_scaled
        score, values = recall_at_50(scores, y_val, val_offsets, totals)
        report["blends"][str(weight)] = {"recall": score,
            "mean_delta": float((values - base_values).mean())}
        print(f"LightGBM weight {weight:.1f}: {score:.6f}", flush=True)
    best_weight = max(report["blends"], key=lambda w: report["blends"][w]["recall"])
    report["best_weight"] = best_weight
    model.save_model(str(args.output / "lambdarank.txt"))
    (args.output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved {args.output / 'lambdarank.txt'}", flush=True)


if __name__ == "__main__":
    main()
