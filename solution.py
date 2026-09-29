"""Train, validate and predict a reproducible local candidate generator.

Usage: python solution.py --data-dir data --output-dir artifacts
Training needs no internet, model download, or external inference API.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from retrieval import TextIndex, Retriever, topk


def contexts(frame, search_columns):
    """Canonical context keys preserve differences in location and filters."""
    return pd.util.hash_pandas_object(frame[search_columns].fillna("").astype(str), index=False).to_numpy()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def check_cache_manifest(args):
    """Never silently reuse candidate labels from another split or dataset."""
    manifest = {"seed": args.seed, "train_queries": args.train_queries,
                "validation_queries": args.validation_queries, "pipeline_version": 3,
                "split_style": args.split_style, "corpus_style": args.corpus_style,
                "inputs": {name: file_digest(args.data_dir / name) for name in
                           ("train.parquet", "benchmark_items.parquet", "benchmark_queries.parquet")}}
    if args.candidate_cap != 150:
        manifest["candidate_cap"] = args.candidate_cap
    if args.seen_style != "random":
        manifest["seen_style"] = args.seen_style
    if args.microcat_channel:
        manifest["microcat_channel"] = True
    path = args.output_dir / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Cache belongs to different data/settings. Choose a new --output-dir.")
    path.write_text(json.dumps(manifest, indent=2))


def build_examples(retriever, query_groups, truth, out_path, candidate_cap):
    if out_path.exists():
        return joblib.load(out_path)
    blocks, targets, query_rows, offsets = [], [], [], [0]
    started = time.monotonic()
    for n, (key, query) in enumerate(query_groups.iterrows(), 1):
        rows, features = retriever.candidates(query, candidate_cap)
        labels = np.isin(retriever.index.ids[rows], list(truth[key])).astype(np.int8)
        blocks.append(features)
        targets.append(labels)
        query_rows.append(rows)
        offsets.append(offsets[-1] + len(rows))
        if n % 100 == 0:
            print(f"Candidates {n}/{len(query_groups)}, {time.monotonic()-started:.0f}s", flush=True)
    result = (pd.concat(blocks, ignore_index=True), np.concatenate(targets), query_rows, offsets)
    joblib.dump(result, out_path)
    return result


def evaluate(scores, example, keys, truth):
    _, labels, rows, offsets = example
    recalls, ceilings = [], []
    for i, key in enumerate(keys):
        start, end = offsets[i:i+2]
        selected = topk(scores[start:end], 50)
        total = len(truth[key])
        recalls.append(float(labels[start:end][selected].sum()) / total)
        ceilings.append(float(labels[start:end].sum()) / total)
    values = np.asarray(recalls)
    rng = np.random.default_rng(2026)
    bootstrap = np.mean(rng.choice(values, (2000, len(values)), replace=True), axis=1)
    return {"recall_at_50": float(values.mean()),
            "bootstrap_95_interval": np.quantile(bootstrap, [.025, .975]).tolist(),
            "candidate_recall": float(np.mean(ceilings)),
            "queries": len(keys)}, values


def validate_answer(path, queries, items):
    answer = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert answer.columns.tolist() == ["query_id", "answer"]
    assert len(answer) == len(queries)
    assert answer.query_id.is_unique
    assert set(answer.query_id) == set(queries.query_id)
    valid_ids = set(items.item_id.astype(str))
    import re
    for query_id, value in answer.itertuples(index=False):
        assert len(query_id) == 16
        ids = value.split(" ") if value else []
        assert 0 < len(ids) <= 50 and len(set(ids)) == len(ids)
        assert all(re.fullmatch(r"[0-9a-f]{16}", x) and x in valid_ids for x in ids)
    return {"rows": len(answer), "candidates_per_query": answer.answer.str.split().str.len().value_counts().to_dict(),
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def sample_pseudo_corpus(train_items, benchmark_items, rng, style):
    """Sample labeled items with the benchmark's size and optional location mix."""
    if len(train_items) < len(benchmark_items):
        raise ValueError("Training corpus is smaller than benchmark corpus")
    if style == "random":
        chosen = rng.choice(len(train_items), len(benchmark_items), replace=False)
    elif style == "location":
        groups = train_items.groupby("item_location_id", sort=False).indices
        needs = benchmark_items.item_location_id.value_counts()
        selected = []
        for location, count in needs.items():
            available = groups.get(location, np.empty(0, dtype=int))
            if len(available):
                selected.extend(rng.choice(available, min(count, len(available)), replace=False))
        selected = np.asarray(selected, dtype=int)
        remaining = np.setdiff1d(np.arange(len(train_items)), selected, assume_unique=True)
        fill = rng.choice(remaining, len(benchmark_items) - len(selected), replace=False)
        chosen = rng.permutation(np.concatenate([selected, fill]))
    else:
        raise ValueError(f"Unknown corpus style: {style}")
    return train_items.iloc[chosen].reset_index(drop=True)


def query_profile(frame, top_locations):
    """Coarse observable query strata: geography, filters and text length."""
    locations = frame.search_location_id
    area = np.select([locations.isin(top_locations[:2]), locations.isin(top_locations[:10])],
                     ["top2", "top10"], default="other")
    filtered = frame.search_infm_params_text.fillna("").str.len().gt(0).to_numpy()
    words = frame.search_query.fillna("").str.split().str.len().clip(upper=7).to_numpy()
    return list(zip(area, filtered, words))


def sample_profile_contexts(grouped, benchmark_queries, rng, n_validation, n_ranker):
    """Draw unique training texts with the benchmark's observable query mix."""
    top_locations = benchmark_queries.search_location_id.value_counts().index.tolist()
    target = pd.Series(query_profile(benchmark_queries, top_locations)).value_counts(normalize=True)
    pools = {}
    source_strata = pd.Series(query_profile(grouped, top_locations), index=grouped.index)
    for stratum, part in grouped.groupby(source_strata, sort=False):
        pools[stratum] = (part, part.groupby("search_query", sort=False).indices)
    strata = list(target.index)
    proportions = target.to_numpy()
    used_texts = set()
    stages = []
    for size in (n_validation, n_ranker):
        quotas = rng.multinomial(size, proportions)
        selected = []
        # Scarce strata get first choice of texts shared across locations.
        order = sorted(range(len(strata)), key=lambda j: len(pools[strata[j]][1]) / max(quotas[j], 1))
        for j in order:
            count = quotas[j]
            if not count:
                continue
            part, by_text = pools[strata[j]]
            available = [value for value in by_text if value not in used_texts]
            if len(available) < count:
                raise ValueError(f"Not enough unique query texts in stratum {strata[j]}")
            chosen_texts = rng.choice(available, count, replace=False)
            for value in chosen_texts:
                rows = by_text[value]
                selected.append(part.index[rows[rng.integers(len(rows))]])
            used_texts.update(chosen_texts)
        stages.append(rng.permutation(np.asarray(selected, dtype=np.uint64)))
    return stages


def sample_seen_items(pseudo_items, interactions, benchmark_items, rng, style):
    """Match the size, and optionally the frequency, of the observable overlap."""
    counts = interactions.item_id.value_counts()
    real_seen = counts[counts.index.isin(benchmark_items.item_id)]
    if style == "random":
        return set(rng.choice(pseudo_items.item_id.to_numpy(), len(real_seen), replace=False))
    if style != "frequency":
        raise ValueError(f"Unknown seen style: {style}")
    pseudo_counts = counts.reindex(pseudo_items.item_id).to_numpy()
    pseudo_bins = np.minimum(pseudo_counts, 10)
    target_bins = np.minimum(real_seen.to_numpy(), 10)
    selected = []
    for bin_number, target in pd.Series(target_bins).value_counts().items():
        available = np.flatnonzero(pseudo_bins == bin_number)
        if len(available) < target:
            raise ValueError(f"Not enough pseudo items with interaction bin {bin_number}")
        selected.extend(rng.choice(available, target, replace=False))
    return set(pseudo_items.item_id.iloc[selected])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--train-queries", type=int, default=2500)
    parser.add_argument("--validation-queries", type=int, default=1000)
    parser.add_argument("--candidate-cap", type=int, default=600)
    parser.add_argument("--split-style", choices=("contexts", "unique_text", "benchmark_profile"), default="unique_text")
    parser.add_argument("--corpus-style", choices=("random", "location"), default="random")
    parser.add_argument("--seen-style", choices=("random", "frequency"), default="random")
    parser.add_argument("--microcat-channel", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--predict-only", action="store_true", help="Reuse the saved model and metrics; rebuild aggregates from input data")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(exist_ok=True, parents=True)
    check_cache_manifest(args)
    train = pd.read_parquet(args.data_dir / "train.parquet")
    items = pd.read_parquet(args.data_dir / "benchmark_items.parquet")
    queries = pd.read_parquet(args.data_dir / "benchmark_queries.parquet")
    search_columns = sorted(c for c in queries if c.startswith("search_"))
    train["_context"] = contexts(train, search_columns)
    # Deduplication precedes splitting, so repeat rows cannot leak into validation.
    train = train.drop_duplicates(["_context", "item_id"]).reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    if args.predict_only:
        report = json.loads((out / "metrics.json").read_text())
        model = CatBoostClassifier()
        model.load_model(out / "ranker.cbm")
        index = load_index(out / "text_index_benchmark.joblib", items)
        predict(index, train, queries, items, model, report, out, args.candidate_cap)
        return
    # A benchmark-sized pseudo corpus makes every held-out label observable.
    # Only as many of its items may contribute historical choices as are seen
    # in the real benchmark. This prevents a warm-item validation shortcut.
    unique_train_items = train.drop_duplicates("item_id").reset_index(drop=True)
    validation_items = sample_pseudo_corpus(unique_train_items, items, rng, args.corpus_style)
    eligible = train[train.item_id.isin(validation_items.item_id)].copy()
    seen_ids = sample_seen_items(validation_items, train, items, rng, args.seen_style)
    seen_count = len(seen_ids)
    historical = eligible[eligible.item_id.isin(seen_ids)].copy()
    grouped = eligible.drop_duplicates("_context").set_index("_context")[search_columns]
    truth = eligible.groupby("_context").item_id.agg(set).to_dict()
    if args.split_style == "benchmark_profile":
        val_keys, rank_keys = sample_profile_contexts(grouped, queries, rng,
                                                       args.validation_queries, args.train_queries)
        val_texts = grouped.loc[val_keys].search_query.to_numpy()
        rank_texts = grouped.loc[rank_keys].search_query.to_numpy()
        cold_val_texts = set(val_texts[rng.random(len(val_texts)) < .63])
        cold_rank_texts = set(rank_texts[rng.random(len(rank_texts)) < .63])
    elif args.split_style == "unique_text":
        # The benchmark contains one row per query text, with many unseen texts.
        # Sample texts uniformly and select one context for each. Prefer a mix of
        # filtered/unfiltered contexts close to the benchmark's observed mix.
        texts = rng.permutation(grouped.search_query.unique())
        val_texts = texts[:args.validation_queries]
        rank_texts = texts[args.validation_queries:args.validation_queries + args.train_queries]
        relevant_rows = grouped[grouped.search_query.isin(np.concatenate([val_texts, rank_texts]))]
        by_text = {text: part for text, part in relevant_rows.groupby("search_query", sort=False)}

        def choose_contexts(text_values):
            selected = []
            for text_value in text_values:
                contexts_for_text = by_text[text_value]
                wants_filter = rng.random() < .37
                matches = contexts_for_text[contexts_for_text.search_infm_params_text.str.len().gt(0) == wants_filter]
                pool = matches if len(matches) else contexts_for_text
                selected.append(pool.index[rng.integers(len(pool))])
            return np.asarray(selected, dtype=np.uint64)

        val_keys = choose_contexts(val_texts)
        rank_keys = choose_contexts(rank_texts)
        cold_val_texts = set(val_texts[rng.random(len(val_texts)) < .63])
        cold_rank_texts = set(rank_texts[rng.random(len(rank_texts)) < .63])
    else:
        keys = rng.permutation(grouped.index.to_numpy())
        val_keys = keys[:args.validation_queries]
        rank_keys = keys[args.validation_queries:args.validation_queries + args.train_queries]
        cold_val_texts = cold_rank_texts = set()
    if len(rank_keys) != args.train_queries:
        raise ValueError("Too few distinct query contexts for this split")
    (out / "split.json").write_text(json.dumps({"seed": args.seed, "search_columns": search_columns,
        "pseudo_corpus_items": len(validation_items), "pseudo_seen_items": seen_count,
        "candidate_cap": args.candidate_cap, "split_style": args.split_style,
        "corpus_style": args.corpus_style, "seen_style": args.seen_style,
        "microcat_channel": args.microcat_channel,
        "cold_validation_texts": len(cold_val_texts), "cold_ranker_texts": len(cold_rank_texts),
        "pseudo_corpus_id_sha256": hashlib.sha256("\n".join(sorted(validation_items.item_id)).encode()).hexdigest(),
        "validation_contexts": [str(x) for x in val_keys],
        "ranker_contexts": [str(x) for x in rank_keys]}, indent=2))
    index = load_index(out / "text_index_validation.joblib", validation_items)
    # The classifier sees candidates from a retriever that has not seen their labels.
    inner = historical[~historical._context.isin(np.concatenate([val_keys, rank_keys])) &
                       ~historical.search_query.isin(cold_val_texts | cold_rank_texts)]
    category_inner = (train[~train._context.isin(np.concatenate([val_keys, rank_keys])) &
                            ~train.search_query.isin(cold_val_texts | cold_rank_texts)]
                      if args.microcat_channel else None)
    rank_examples = build_examples(Retriever(index, inner, category_inner), grouped.loc[rank_keys],
                                   truth, out / "rank_examples.joblib", args.candidate_cap)
    outer = historical[~historical._context.isin(val_keys) &
                       ~historical.search_query.isin(cold_val_texts)]
    category_outer = (train[~train._context.isin(val_keys) &
                            ~train.search_query.isin(cold_val_texts)]
                      if args.microcat_channel else None)
    val_examples = build_examples(Retriever(index, outer, category_outer), grouped.loc[val_keys],
                                  truth, out / "val_examples.joblib", args.candidate_cap)
    x_train, y_train, _, _ = rank_examples
    x_val, y_val, _, _ = val_examples
    # Fixed hyperparameters; validation is reserved for comparing complete methods.
    model = CatBoostClassifier(iterations=450, depth=7, learning_rate=.06,
        loss_function="Logloss", random_seed=args.seed, thread_count=8,
        l2_leaf_reg=8, verbose=100, allow_writing_files=False)
    model.fit(x_train, y_train)
    model.save_model(out / "ranker.cbm")
    report = {"seed": args.seed, "train_contexts": len(rank_keys), "validation_contexts": len(val_keys),
              "deduplicated_pairs": len(train), "pseudo_corpus_items": len(validation_items),
              "pseudo_seen_items": seen_count, "candidate_cap": args.candidate_cap,
              "split_style": args.split_style, "corpus_style": args.corpus_style,
              "seen_style": args.seen_style, "microcat_channel": args.microcat_channel,
              "cold_validation_texts": len(cold_val_texts),
              "features": x_train.columns.tolist(), "methods": {}}
    per_query = pd.DataFrame({"context": val_keys})
    for name, scores in {"content": x_val.content.to_numpy(),
                         "content_geo": x_val.content_geo.to_numpy(),
                         "hybrid": x_val.combined.to_numpy(),
                         "catboost": model.predict_proba(x_val)[:, 1]}.items():
        metrics, values = evaluate(scores, val_examples, val_keys, truth)
        report["methods"][name] = metrics
        per_query[name] = values
        print(name, metrics, flush=True)
    winner = max(report["methods"], key=lambda x: report["methods"][x]["recall_at_50"])
    report["selected_method"] = winner
    report["feature_importance"] = dict(sorted(zip(x_train.columns, model.feature_importances_), key=lambda x:-x[1]))
    per_query = per_query.merge(grouped, left_on="context", right_index=True)
    per_query.to_csv(out / "validation_queries.csv", index=False)
    final_scores = model.predict_proba(x_val)[:, 1] if winner == "catboost" else x_val["combined" if winner == "hybrid" else winner].to_numpy()
    report["error_analysis"] = error_analysis(index, val_examples, final_scores, val_keys, truth, grouped, out)
    (out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    if args.skip_predict:
        return
    index = load_index(out / "text_index_benchmark.joblib", items)
    predict(index, train, queries, items, model, report, out, args.candidate_cap)


def load_index(path, items):
    if path.exists():
        return joblib.load(path)
    index = TextIndex(items)
    joblib.dump(index, path)
    return index


def error_analysis(index, examples, scores, keys, truth, grouped, out):
    """Separate failures of candidate retrieval from failures of final selection."""
    _, _, rows, offsets = examples
    missing_from_pool = missed_in_selection = total_positive = 0
    cases = []
    for i, key in enumerate(keys):
        start, end = offsets[i:i + 2]
        pool = rows[i]
        selected = pool[topk(scores[start:end], 50)]
        selected_ids = set(index.ids[selected])
        pool_ids = set(index.ids[pool])
        relevant = truth[key]
        missing_from_pool += len(relevant - pool_ids)
        missed_in_selection += len((relevant & pool_ids) - selected_ids)
        total_positive += len(relevant)
        if len(cases) < 20 and relevant - selected_ids:
            def describe(ids):
                return [{"item_id": item_id,
                         "title": str(index.items.iloc[index.id_to_row[item_id]].item_title_raw),
                         "location": str(index.items.iloc[index.id_to_row[item_id]].item_location_id),
                         "in_candidate_pool": item_id in pool_ids} for item_id in ids]
            cases.append({"query": grouped.loc[key].to_dict(),
                          "missed": describe(sorted(relevant - selected_ids)),
                          "top_five": describe(index.ids[selected[:5]])})
    (out / "error_examples.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2, default=str))
    return {"relevant_pairs": total_positive, "missing_from_candidates": missing_from_pool,
            "missed_during_selection": missed_in_selection}


def predict(index, train, queries, items, model, report, out, candidate_cap):
    winner = report["selected_method"]
    final = Retriever(index, train, train if report.get("microcat_channel", False) else None)
    predictions = []
    for n, (_, query) in enumerate(queries.iterrows(), 1):
        rows, features = final.candidates(query, candidate_cap)
        scores = model.predict_proba(features)[:, 1] if winner == "catboost" else features["combined" if winner == "hybrid" else winner].to_numpy()
        selected = rows[topk(scores, 50)]
        predictions.append(" ".join(index.ids[selected]))
        if n % 100 == 0:
            print(f"Predict {n}/{len(queries)}", flush=True)
    answer_path = out / "answer.csv"
    pd.DataFrame({"query_id": queries.query_id, "answer": predictions}).to_csv(answer_path, index=False)
    report["answer"] = validate_answer(answer_path, queries, items)
    (out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("Completed", report["answer"], flush=True)


if __name__ == "__main__":
    main()
