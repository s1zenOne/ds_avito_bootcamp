"""Write a separate benchmark CSV with clicked-title prototype candidates.

The script reuses the existing unsupervised benchmark text index and cached
scores for the original candidate set. It adds corpus items retrieved from a
centroid of titles clicked for nearby historical search texts, computes the
same model features for only those new items, and blends the local CatBoost
and LightGBM scores exactly as in improved_predict.py. The benchmark query
ID is used only to identify the output row; no labels or answer-specific
rules are present. All models and computations run locally.

Example::

    python prototype_predict.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from prototype_experiment import prototype_scores
from retrieval import Retriever, topk
from solution import contexts, load_index, validate_answer


MODEL_NAMES = ("catboost_v5.cbm", "catboost_v6.cbm", "catboost_v9.cbm", "catboost_v10.cbm")
VIEW_PATTERN = re.compile(r"Вид услуги (.*?)(?= Тип услуги| Кто оказывает услуги| Онлайн-запись|$)")


def standardized(values):
    """Match the production blend's within-query scaling."""
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / max(values.std(), 1e-5)


def new_candidate_scores(features, models, ranker):
    """Compute scores on the same feature subsets used by each saved model."""
    probabilities = np.mean([
        model.predict_proba(features[model.feature_names_])[:, 1]
        for model in models
    ], axis=0)
    probabilities = np.clip(probabilities, 1e-8, 1 - 1e-8)
    logits = np.log(probabilities / (1 - probabilities)).astype(np.float32)
    ranking = ranker.predict(features[ranker.feature_name()], num_threads=8).astype(np.float32)
    return logits, ranking


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--index-cache", type=Path,
                        default=Path("artifacts_final/text_index_benchmark.joblib"))
    parser.add_argument("--score-cache", type=Path,
                        default=Path("artifacts_final/candidate_scores.npz"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, default=Path("answer_prototype.csv"))
    parser.add_argument("--limit", type=int, default=500,
                        help="Prototype candidates per raw and geography-weighted channel")
    parser.add_argument("--ranker-weight", type=float, default=.4)
    parser.add_argument("--view-boost", type=float, default=0.)
    args = parser.parse_args()
    if args.limit < 1 or not 0 <= args.ranker_weight <= 1:
        raise ValueError("Require positive --limit and --ranker-weight in [0, 1]")

    started = time.monotonic()
    queries = pd.read_parquet(args.data_dir / "benchmark_queries.parquet")
    items = pd.read_parquet(args.data_dir / "benchmark_items.parquet")
    train = pd.read_parquet(args.data_dir / "train.parquet")
    search_columns = sorted(column for column in queries if column.startswith("search_"))
    train["_context"] = contexts(train, search_columns)
    train = train.drop_duplicates(["_context", "item_id"]).reset_index(drop=True)
    # Я создаю папку и при самостоятельном запуске этого скрипта:
    # в свежем архиве промежуточный индекс ещё не существует.
    args.index_cache.parent.mkdir(parents=True, exist_ok=True)
    index = load_index(args.index_cache, items)
    if set(index.ids) != set(items.item_id):
        raise ValueError("Text index and benchmark item IDs differ")
    retriever = Retriever(index, train, train)
    params = index.items.item_infm_params_text.fillna("").astype(str).to_numpy()

    models = []
    for name in MODEL_NAMES:
        model = CatBoostClassifier()
        model.load_model(args.model_dir / name)
        models.append(model)
    ranker = lgb.Booster(model_file=str(args.model_dir / "lambdarank.txt"))
    with np.load(args.score_cache) as cache:
        cached_query_ids = [value.decode("ascii") for value in cache["query_ids"]]
        if cached_query_ids != queries.query_id.astype(str).tolist():
            raise ValueError("Score cache and benchmark queries have different order or IDs")
        offsets = cache["offsets"]
        cached_item_ids = cache["item_ids"]
        cached_classifier = cache["classifier"]
        cached_rank = cache["rank_scores"]
        cached_view = cache["view_matches"]
        if (len(offsets) != len(queries) + 1 or offsets[0] != 0 or
                np.any(np.diff(offsets) <= 0) or
                offsets[-1] != len(cached_item_ids) or
                not len(cached_item_ids) == len(cached_classifier) ==
                len(cached_rank) == len(cached_view)):
            raise ValueError("Invalid candidate score cache offsets or array lengths")
        if not np.isfinite(cached_classifier).all() or not np.isfinite(cached_rank).all():
            raise ValueError("Candidate score cache contains non-finite model scores")

        predictions = []
        changes = extra_total = 0
        for number, (_, query) in enumerate(queries.iterrows(), 1):
            start, end = offsets[number - 1:number + 1]
            old_ids = [value.decode("ascii") for value in cached_item_ids[start:end]]
            old_rows = np.fromiter((index.id_to_row[item_id] for item_id in old_ids),
                                   dtype=np.int32, count=len(old_ids))
            proto, geography = prototype_scores(query, retriever)
            proposed = np.union1d(topk(proto, args.limit), topk(proto * geography, args.limit))
            extra = np.setdiff1d(proposed, old_rows, assume_unique=True)
            extra_total += len(extra)
            if len(extra):
                score_fields, _ = retriever.all_scores(query)
                features = retriever.features(query, extra, score_fields)
                new_classifier, new_rank = new_candidate_scores(features, models, ranker)
                filter_match = VIEW_PATTERN.search(str(query.search_infm_params_text))
                phrase = "Вид услуги " + filter_match.group(1) if filter_match else None
                new_view = np.fromiter((phrase in params[row] for row in extra),
                                       dtype=bool, count=len(extra)) if phrase else np.zeros(len(extra), bool)
                all_rows = np.concatenate([old_rows, extra])
                classifier = np.concatenate([cached_classifier[start:end], new_classifier])
                ranking = np.concatenate([cached_rank[start:end], new_rank])
                view_matches = np.concatenate([cached_view[start:end], new_view])
            else:
                all_rows = old_rows
                classifier = cached_classifier[start:end]
                ranking = cached_rank[start:end]
                view_matches = cached_view[start:end]
            base_score = standardized(classifier + args.view_boost * view_matches)
            rank_score = standardized(ranking)
            blend = (1 - args.ranker_weight) * base_score + args.ranker_weight * rank_score
            selected = index.ids[all_rows[topk(blend, 50)]]
            predictions.append(" ".join(selected))
            # Count changes versus the existing cached candidate pool.
            old_base = standardized(cached_classifier[start:end] +
                                    args.view_boost * cached_view[start:end])
            old_rank = standardized(cached_rank[start:end])
            old_blend = (1 - args.ranker_weight) * old_base + args.ranker_weight * old_rank
            old_selected = set(index.ids[old_rows[topk(old_blend, 50)]])
            changes += int(set(selected) != old_selected)
            if number % 100 == 0:
                print(f"Prototype scored {number}/{len(queries)} in {time.monotonic()-started:.1f}s",
                      flush=True)

    args.output.parent.mkdir(exist_ok=True, parents=True)
    pd.DataFrame({"query_id": queries.query_id.astype(str), "answer": predictions}).to_csv(
        args.output, index=False)
    validation = validate_answer(args.output, queries, items)
    report = {
        "output": str(args.output), "limit_per_channel": args.limit,
        "ranker_weight": args.ranker_weight, "view_boost": args.view_boost,
        "queries_with_changed_top50": changes, "additional_candidates_scored": extra_total,
        "elapsed_seconds": round(time.monotonic() - started, 1), "validation": validation,
    }
    report_path = args.output.with_suffix(".metrics.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
