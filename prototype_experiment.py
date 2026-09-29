"""Evaluate clicked-title query expansion on the existing v10 validation split.

This is a candidate-generation experiment, not a benchmark submission. It
reconstructs the v10 pseudo corpus and history using the saved split settings,
then measures how many held-out clicked items an extra prototype channel adds
to the saved v10 candidate pools. The prototype contains only titles from
allowed historical interactions; held-out contexts and designated cold query
texts are excluded before the index is built.

Run locally after generating ``artifacts_v10`` with ``solution.py``::

    python prototype_experiment.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize as l2_normalize


ROOT = Path(__file__).resolve().parent
PRIOR = ROOT
sys.path.insert(0, str(PRIOR))
from retrieval import Retriever, topk  # noqa: E402
from solution import (  # noqa: E402
    contexts,
    sample_profile_contexts,
    sample_pseudo_corpus,
    sample_seen_items,
)

DATA = ROOT / "dataset"
SAVED = PRIOR / "artifacts_v10"
OUT = ROOT / "artifacts_prototype"


def same_split():
    """Replay v10's RNG exactly, and fail if its saved validation keys differ."""
    split = json.loads((SAVED / "split.json").read_text())
    train = pd.read_parquet(DATA / "train.parquet")
    benchmark_items = pd.read_parquet(DATA / "benchmark_items.parquet")
    benchmark_queries = pd.read_parquet(DATA / "benchmark_queries.parquet")
    search_columns = split["search_columns"]
    train["_context"] = contexts(train, search_columns)
    train = train.drop_duplicates(["_context", "item_id"]).reset_index(drop=True)
    rng = np.random.default_rng(split["seed"])
    unique_items = train.drop_duplicates("item_id").reset_index(drop=True)
    validation_items = sample_pseudo_corpus(unique_items, benchmark_items, rng, split["corpus_style"])
    item_digest = hashlib.sha256("\n".join(sorted(validation_items.item_id)).encode()).hexdigest()
    assert item_digest == split["pseudo_corpus_id_sha256"]
    eligible = train[train.item_id.isin(validation_items.item_id)].copy()
    seen_ids = sample_seen_items(validation_items, train, benchmark_items, rng, split["seen_style"])
    historical = eligible[eligible.item_id.isin(seen_ids)].copy()
    grouped = eligible.drop_duplicates("_context").set_index("_context")[search_columns]
    truth = eligible.groupby("_context").item_id.agg(set).to_dict()
    val_keys, rank_keys = sample_profile_contexts(
        grouped, benchmark_queries, rng, len(split["validation_contexts"]), len(split["ranker_contexts"])
    )
    assert [str(x) for x in val_keys] == split["validation_contexts"]
    assert [str(x) for x in rank_keys] == split["ranker_contexts"]
    val_texts = grouped.loc[val_keys].search_query.to_numpy()
    rank_texts = grouped.loc[rank_keys].search_query.to_numpy()
    cold_val_texts = set(val_texts[rng.random(len(val_texts)) < .63])
    _cold_rank_texts = set(rank_texts[rng.random(len(rank_texts)) < .63])
    assert len(cold_val_texts) == split["cold_validation_texts"]
    # Same history as v10's outer validation retriever: no held-out pair is used.
    outer = historical[~historical._context.isin(val_keys) &
                       ~historical.search_query.isin(cold_val_texts)].copy()
    category_outer = train[~train._context.isin(val_keys) &
                           ~train.search_query.isin(cold_val_texts)].copy()
    return grouped.loc[val_keys], val_keys, truth, outer, category_outer


def prototype_scores(query, retriever):
    """Map nearby search texts to clicked titles, then search corpus fields.

    Each nearby text contributes its clicked titles with the historical
    retriever's frequency correction. L2-normalized title centroids make
    expanded text comparable across queries of different lengths.
    """
    ix = retriever.index
    qw, qc, _ = ix.query_vectors(query)
    similarities = .55 * (qw @ retriever.hist_w).toarray().ravel()
    similarities += .45 * (qc @ retriever.hist_c).toarray().ravel()
    near = topk(np.clip(similarities, 0, 1), 40)
    weights = similarities[near] ** 4 / np.maximum(retriever.query_counts[near], 1) ** .35
    clicked_weights = np.asarray(retriever.links[near].T @ weights).ravel()
    active = np.flatnonzero(clicked_weights > 0)
    if len(active) == 0:
        return np.zeros(len(ix.ids), np.float32), np.ones(len(ix.ids), np.float32)
    weight_row = sparse.csr_matrix(clicked_weights[active].reshape(1, -1))
    word_centroid = l2_normalize(weight_row @ ix.title_w[:, active].T)
    char_centroid = l2_normalize(weight_row @ ix.title_c[:, active].T)
    title = (word_centroid @ ix.title_w).toarray().ravel()
    char = (char_centroid @ ix.title_c).toarray().ravel()
    desc = (word_centroid @ ix.desc_w).toarray().ravel()
    params = (word_centroid @ ix.params_w).toarray().ravel()
    score = (.50 * title + .20 * char + .20 * desc + .10 * params).astype(np.float32)

    sloc = str(query.get("search_location_id", -1))
    same_loc = (ix.loc == sloc).astype(np.float32)
    loc_counts = retriever.loc_counts.get(sloc, {})
    maximum = max(loc_counts.values(), default=1)
    loc_prior = np.array([loc_counts.get(v, 0) / maximum for v in ix.unique_locs],
                         np.float32)[ix.loc_codes]
    geography = .08 + .92 * np.maximum(loc_prior ** .35, same_loc)
    return score, geography


def main():
    OUT.mkdir(exist_ok=True)
    started = time.monotonic()
    grouped, keys, truth, outer, _category_outer = same_split()
    print(f"Replayed v10 split; {len(outer)} historical rows", flush=True)
    ix = joblib.load(SAVED / "text_index_validation.joblib")
    examples = joblib.load(SAVED / "val_examples.joblib")
    assert len(ix.ids) == 189_212
    assert len(examples[2]) == len(keys)
    retriever = Retriever(ix, outer)
    baseline_rows = examples[2]
    strategies = {
        "raw_100": [], "raw_200": [], "raw_500": [],
        "geo_100": [], "geo_200": [], "geo_500": [],
        "raw_geo_100": [], "raw_geo_200": [], "raw_geo_500": [],
    }
    base_recalls, totals = [], []
    added_pairs = {name: 0 for name in strategies}
    affected = {name: 0 for name in strategies}
    per_query = []
    for n, (key, query) in enumerate(grouped.iterrows()):
        score, geo = prototype_scores(query, retriever)
        raw_500 = topk(score, 500)
        geo_500 = topk(score * geo, 500)
        proposals = {
            "raw_100": raw_500[:100], "raw_200": raw_500[:200], "raw_500": raw_500,
            "geo_100": geo_500[:100], "geo_200": geo_500[:200], "geo_500": geo_500,
            "raw_geo_100": np.union1d(raw_500[:100], geo_500[:100]),
            "raw_geo_200": np.union1d(raw_500[:200], geo_500[:200]),
            "raw_geo_500": np.union1d(raw_500, geo_500),
        }
        relevant = {ix.id_to_row[item_id] for item_id in truth[key]}
        base = set(baseline_rows[n])
        base_hit = len(relevant & base)
        total = len(relevant)
        totals.append(total)
        base_recalls.append(base_hit / total)
        entry = {"context": str(key), "query": str(query.search_query),
                 "total_relevant": total, "baseline_hits": base_hit}
        for name, rows in proposals.items():
            new_hits = len(relevant & (set(rows) - base))
            added_pairs[name] += new_hits
            affected[name] += int(new_hits > 0)
            strategies[name].append((base_hit + new_hits) / total)
            entry[f"added_{name}"] = new_hits
        per_query.append(entry)
        if (n + 1) % 100 == 0:
            print(f"Prototype {n + 1}/{len(grouped)} in {time.monotonic() - started:.1f}s", flush=True)
    baseline = float(np.mean(base_recalls))
    result = {
        "validation_queries": len(keys), "historical_rows": len(outer),
        "relevant_pairs": int(sum(totals)), "baseline_candidate_recall": baseline,
        "new_candidate_recall": {name: float(np.mean(values)) for name, values in strategies.items()},
        "added_relevant_pairs": added_pairs, "queries_with_new_hits": affected,
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    pd.DataFrame(per_query).to_csv(OUT / "per_query.csv", index=False)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
