"""Generate a checked Stepik answer from local retrieval and ranking models.

Candidate retrieval uses only the supplied training interactions and item
fields. The four CatBoost models come from two pseudo-validation setups;
the LightGBM model learns within-query ranking from training-only candidates.
The benchmark query IDs are used solely to write the required CSV.

The candidate cache allows testing a ranking weight without rebuilding the
text index or querying the corpus again. It contains predictions, not labels.
"""

import argparse
import json
import re
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from retrieval import Retriever, topk
from solution import contexts, load_index, validate_answer


MODEL_NAMES = ("catboost_v5.cbm", "catboost_v6.cbm", "catboost_v9.cbm", "catboost_v10.cbm")
VIEW_PATTERN = re.compile(r"Вид услуги (.*?)(?= Тип услуги| Кто оказывает услуги| Онлайн-запись|$)")


def standardized(values):
    """Normalize model scores within one query; only order matters for Recall@50."""
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / max(values.std(), 1e-5)


def prepare_cache(data_dir, index_cache, model_dir, output_dir, candidate_cap):
    """Run content, history and microcategory retrieval, then score candidates."""
    train = pd.read_parquet(data_dir / "train.parquet")
    items = pd.read_parquet(data_dir / "benchmark_items.parquet")
    queries = pd.read_parquet(data_dir / "benchmark_queries.parquet")
    search_columns = sorted(column for column in queries if column.startswith("search_"))
    train["_context"] = contexts(train, search_columns)
    # Duplicate clicks do not create extra relevance labels or historical links.
    train = train.drop_duplicates(["_context", "item_id"]).reset_index(drop=True)

    # В распакованном архиве ещё нет artifacts_final/: создаю папку до
    # сохранения нового индекса, а не после вычисления предсказаний.
    index_cache.parent.mkdir(parents=True, exist_ok=True)
    index = load_index(index_cache, items)
    if set(index.ids) != set(items.item_id):
        raise ValueError("The cached text index was built for a different item corpus")
    retriever = Retriever(index, train, train)
    params = index.items.item_infm_params_text.fillna("").astype(str).to_numpy()
    catboost_models = []
    for name in MODEL_NAMES:
        model = CatBoostClassifier()
        model.load_model(model_dir / name)
        catboost_models.append(model)
    ranker = lgb.Booster(model_file=str(model_dir / "lambdarank.txt"))

    ids, classifier, rank_scores, view_matches, content_scores = [], [], [], [], []
    offsets = [0]
    for number, (_, query) in enumerate(queries.iterrows(), 1):
        rows, features = retriever.candidates(query, candidate_cap)
        # Each CatBoost model uses the subset of features present during its training.
        probabilities = np.mean([model.predict_proba(features[model.feature_names_])[:, 1]
                                 for model in catboost_models], axis=0)
        probabilities = np.clip(probabilities, 1e-8, 1 - 1e-8)
        logits = np.log(probabilities / (1 - probabilities))
        ranking = ranker.predict(features[ranker.feature_name()], num_threads=8)
        filter_match = VIEW_PATTERN.search(str(query.search_infm_params_text))
        phrase = "Вид услуги " + filter_match.group(1) if filter_match else None
        matches = np.fromiter((phrase in params[row] for row in rows), dtype=bool, count=len(rows)) \
            if phrase else np.zeros(len(rows), dtype=bool)

        ids.append(index.ids[rows].astype("S16"))
        classifier.append(logits.astype(np.float32))
        rank_scores.append(np.asarray(ranking, dtype=np.float32))
        view_matches.append(matches)
        content_scores.append(features.content_geo.to_numpy(np.float32))
        offsets.append(offsets[-1] + len(rows))
        if number % 100 == 0:
            print(f"Scored {number}/{len(queries)} queries", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "candidate_scores.npz"
    np.savez_compressed(cache_path, query_ids=queries.query_id.to_numpy(dtype="S16"),
                        item_ids=np.concatenate(ids), classifier=np.concatenate(classifier),
                        rank_scores=np.concatenate(rank_scores),
                        view_matches=np.concatenate(view_matches),
                        content_scores=np.concatenate(content_scores),
                        offsets=np.asarray(offsets, dtype=np.int64))
    (output_dir / "cache_manifest.json").write_text(json.dumps({
        "queries": len(queries), "candidates": offsets[-1], "candidate_cap": candidate_cap,
        "models": list(MODEL_NAMES) + ["lambdarank.txt"],
        "description": "Training-only retrieval and local model scores; no benchmark labels"}, indent=2))
    return cache_path


def write_answer(cache_path, data_dir, output_path, ranker_weight, view_boost):
    """Blend local model scores and validate every ID in the resulting CSV."""
    cache = np.load(cache_path)
    queries = pd.read_parquet(data_dir / "benchmark_queries.parquet")
    items = pd.read_parquet(data_dir / "benchmark_items.parquet", columns=["item_id"])
    cached_query_ids = [item.decode("ascii") for item in cache["query_ids"]]
    if cached_query_ids != queries.query_id.astype(str).tolist():
        raise ValueError("Score cache and benchmark queries have different order or IDs")
    offsets = cache["offsets"]
    item_ids = cache["item_ids"]
    base = cache["classifier"]
    rank = cache["rank_scores"]
    match = cache["view_matches"]
    # Кэш должен описывать непустой набор кандидатов для каждого запроса.
    # Эти проверки также работают при python -O, в отличие от assert.
    if (len(offsets) != len(queries) + 1 or offsets[0] != 0 or
            np.any(np.diff(offsets) <= 0) or
            offsets[-1] != len(item_ids) or
            not len(item_ids) == len(base) == len(rank) == len(match)):
        raise ValueError("Invalid candidate score cache offsets or array lengths")
    if not np.isfinite(base).all() or not np.isfinite(rank).all():
        raise ValueError("Candidate score cache contains non-finite model scores")
    answers = []
    for start, end in zip(offsets[:-1], offsets[1:]):
        # Apply the previously validated service-filter rule before blending.
        base_score = standardized(base[start:end] + view_boost * match[start:end])
        rank_score = standardized(rank[start:end])
        score = (1 - ranker_weight) * base_score + ranker_weight * rank_score
        selected = topk(score, 50)
        answers.append(" ".join(item.decode("ascii") for item in item_ids[start:end][selected]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    query_ids = [item.decode("ascii") for item in cache["query_ids"]]
    pd.DataFrame({"query_id": query_ids, "answer": answers}).to_csv(output_path, index=False)
    print("Validation:", validate_answer(output_path, queries, items), flush=True)
    cache.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--index-cache", type=Path, default=Path("artifacts_final/text_index_benchmark.joblib"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts_final"))
    parser.add_argument("--output", type=Path, default=Path("answer_candidate.csv"))
    parser.add_argument("--candidate-cap", type=int, default=600)
    parser.add_argument("--ranker-weight", type=float, default=.4)
    parser.add_argument("--view-boost", type=float, default=0.0)
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.ranker_weight <= 1:
        raise ValueError("ranker-weight must be between 0 and 1")
    cache_path = args.output_dir / "candidate_scores.npz"
    if not args.reuse_cache:
        cache_path = prepare_cache(args.data_dir, args.index_cache, args.model_dir,
                                   args.output_dir, args.candidate_cap)
    write_answer(cache_path, args.data_dir, args.output, args.ranker_weight, args.view_boost)


if __name__ == "__main__":
    main()
