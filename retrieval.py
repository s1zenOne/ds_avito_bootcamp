"""Local sparse retrieval. No query IDs or benchmark labels enter the model.

The historical channel transfers observed choices from similar query texts.
All supervised aggregates are rebuilt using the training part of each split.
"""
from collections import defaultdict
from functools import lru_cache
import html
import re

import numpy as np
import pandas as pd
from nltk.stem.snowball import RussianStemmer
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

_stemmer = RussianStemmer()
_word_re = re.compile(r"[а-яa-z0-9]+")


def normalize(value):
    if pd.isna(value):
        return ""
    return " ".join(_word_re.findall(html.unescape(str(value)).lower().replace("ё", "е")))


@lru_cache(maxsize=400_000)
def stem(word):
    return _stemmer.stem(word)


def stem_text(value):
    return " ".join(stem(w) for w in normalize(value).split())


def topk(scores, k):
    """Stable top k: ties are broken by the sorted corpus row, hence item_id."""
    k = min(k, len(scores))
    if k == len(scores):
        candidates = np.arange(k)
    else:
        threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
        above = np.flatnonzero(scores > threshold)
        tied = np.flatnonzero(scores == threshold)[:k - len(above)]
        candidates = np.concatenate([above, tied])
    return candidates[np.lexsort((candidates, -scores[candidates]))[:k]]


class TextIndex:
    """Unsupervised corpus indexes can be reused across validation folds."""

    def __init__(self, items):
        self.items = items.sort_values("item_id").reset_index(drop=True).copy()
        self.ids = self.items.item_id.astype(str).to_numpy()
        self.id_to_row = dict(zip(self.ids, range(len(self.ids))))
        titles = self.items.item_title_raw.fillna("").map(stem_text)
        params = self.items.item_infm_params_text.fillna("").map(stem_text)
        descriptions = self.items.item_description_raw.fillna("").str.slice(0, 2500).map(stem_text)
        print("Fitting word index", flush=True)
        self.word = TfidfVectorizer(min_df=2, max_features=250_000, sublinear_tf=True,
                                   dtype=np.float32, token_pattern=r"(?u)\b\w+\b")
        self.word.fit(titles + " " + params + " " + descriptions)
        # Terms are rows, so retrieval touches only postings for query terms.
        self.title_w = self.word.transform(titles).T.tocsr()
        self.params_w = self.word.transform(params).T.tocsr()
        self.desc_w = self.word.transform(descriptions).T.tocsr()
        print("Fitting character index", flush=True)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                   max_features=160_000, sublinear_tf=True, dtype=np.float32)
        self.title_c = self.char.fit_transform(self.items.item_title_raw.fillna("").map(normalize)).T.tocsr()
        self.loc = self.items.item_location_id.fillna(-1).astype(str).to_numpy()
        self.unique_locs, self.loc_codes = np.unique(self.loc, return_inverse=True)
        self.micro = self.items.item_microcat_id.fillna(-1).astype(str).to_numpy()
        self.category = self.items.item_category_id.fillna(-1).astype(str).to_numpy()
        self.rating = pd.to_numeric(self.items.item_rating, errors="coerce").fillna(0).to_numpy(np.float32)
        self.reviews = np.log1p(pd.to_numeric(self.items.item_rating_reviews_count, errors="coerce").fillna(0).clip(lower=0)).to_numpy(np.float32)
        self.price = np.log1p(pd.to_numeric(self.items.item_price, errors="coerce").fillna(0).clip(lower=0)).to_numpy(np.float32)
        self.latitude = pd.to_numeric(self.items.item_latitude, errors="coerce").to_numpy(np.float32)
        self.longitude = pd.to_numeric(self.items.item_longitude, errors="coerce").to_numpy(np.float32)
        coordinates = self.items[["item_location_id"]].copy()
        coordinates["latitude"] = self.latitude
        coordinates["longitude"] = self.longitude
        self.loc_centers = coordinates.groupby("item_location_id")[["latitude", "longitude"]].median().to_dict("index")

    def query_vectors(self, query):
        text = str(query.get("search_query", ""))
        q_w = self.word.transform([stem_text(text)])
        q_c = self.char.transform([normalize(text)])
        p_w = self.word.transform([stem_text(query.get("search_infm_params_text", ""))])
        return q_w, q_c, p_w


class CategoryIntentIndex:
    """Transfer query intent to item microcategories using all allowed clicks."""

    def __init__(self, index, interactions):
        train = interactions.copy()
        train["_text"] = train.search_query.map(normalize)
        self.texts = sorted(train._text.unique())
        text_codes = {value: row for row, value in enumerate(self.texts)}
        self.categories, self.item_codes = np.unique(index.micro, return_inverse=True)
        category_codes = {value: row for row, value in enumerate(self.categories)}
        rows = train._text.map(text_codes).to_numpy()
        cols = train.item_microcat_id.astype(str).map(category_codes)
        present = cols.notna().to_numpy()
        self.links = sparse.csr_matrix((np.ones(present.sum(), np.float32),
                                        (rows[present], cols[present].to_numpy(dtype=int))),
                                       shape=(len(self.texts), len(self.categories)))
        self.counts = np.asarray(self.links.sum(axis=1)).ravel()
        self.word = index.word.transform([stem_text(value) for value in self.texts]).T.tocsr()
        self.char = index.char.transform(self.texts).T.tocsr()
        print(f"Category index: {len(self.texts)} texts, {len(self.categories)} microcategories", flush=True)

    def score(self, word_query, char_query):
        similarity = .55 * (word_query @ self.word).toarray().ravel()
        similarity += .45 * (char_query @ self.char).toarray().ravel()
        near = topk(similarity, 80)
        weights = similarity[near] ** 4 / np.maximum(self.counts[near], 1) ** .35
        prior = np.asarray(self.links[near].T @ weights).ravel()
        confidence = prior.max(initial=0) / max(prior.sum(), 1e-8)
        prior /= max(prior.max(initial=0), 1e-8)
        return prior[self.item_codes].astype(np.float32), float(confidence)


class Retriever:
    def __init__(self, index, interactions, category_interactions=None):
        self.index = index
        self.category_intent = (CategoryIntentIndex(index, category_interactions)
                                if category_interactions is not None else None)
        # Drop choices outside this retrieval corpus, but never invent IDs.
        train = interactions[interactions.item_id.isin(index.id_to_row)].copy()
        train["_text"] = train.search_query.map(normalize)
        self.texts = sorted(train._text.unique())
        text_to_row = {s: i for i, s in enumerate(self.texts)}
        qrows = train._text.map(text_to_row).to_numpy()
        irows = train.item_id.map(index.id_to_row).to_numpy()
        self.links = sparse.csr_matrix((np.ones(len(train), np.float32), (qrows, irows)),
                                       shape=(len(self.texts), len(index.ids)))
        self.hist_w = index.word.transform([stem_text(t) for t in self.texts]).T.tocsr()
        self.hist_c = index.char.transform(self.texts).T.tocsr()
        self.popularity = np.asarray(self.links.sum(axis=0)).ravel()
        self.query_counts = np.asarray(self.links.sum(axis=1)).ravel()
        loc_pairs = train.groupby(["search_location_id", "item_location_id"], dropna=False).size()
        self.loc_counts = defaultdict(dict)
        for (search_loc, item_loc), n in loc_pairs.items():
            self.loc_counts[str(search_loc)][str(item_loc)] = float(n)
        self.exact_text = text_to_row
        self.micro_links = train.groupby(["_text", "item_microcat_id"], dropna=False).size()
        print(f"Historical index: {len(train)} interactions, {len(self.texts)} texts", flush=True)

    def all_scores(self, query):
        ix = self.index
        qw, qc, pw = ix.query_vectors(query)
        title = (qw @ ix.title_w).toarray().ravel()
        char = (qc @ ix.title_c).toarray().ravel()
        desc = (qw @ ix.desc_w).toarray().ravel()
        params = (qw @ ix.params_w).toarray().ravel()
        filter_match = (pw @ ix.params_w).toarray().ravel()
        neighbors = .55 * (qw @ self.hist_w).toarray().ravel() + .45 * (qc @ self.hist_c).toarray().ravel()
        neighbors = np.clip(neighbors, 0, 1)
        near = topk(neighbors, 40)
        weights = neighbors[near] ** 4 / np.maximum(self.query_counts[near], 1) ** .35
        history = np.asarray(self.links[near].T @ weights).ravel()
        exact_idx = self.exact_text.get(normalize(query.get("search_query", "")))
        exact = np.zeros(len(ix.ids), np.float32) if exact_idx is None else self.links[exact_idx].toarray().ravel()
        sloc = str(query.get("search_location_id", -1))
        same_loc = (ix.loc == sloc).astype(np.float32)
        lc = self.loc_counts.get(sloc, {})
        max_count = max(lc.values(), default=1)
        # Soft geography permits nearby places and remote services.
        loc_prior = np.array([lc.get(v, 0) / max_count for v in ix.unique_locs], np.float32)[ix.loc_codes]
        center = ix.loc_centers.get(query.get("search_location_id"))
        distance = np.full(len(ix.ids), 20000, np.float32)
        if center is not None:
            lat, lon = center["latitude"], center["longitude"]
            lat1, lat2 = np.radians(lat), np.radians(ix.latitude)
            a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(np.radians(ix.longitude - lon) / 2) ** 2
            distance = np.nan_to_num(12742 * np.arcsin(np.sqrt(np.clip(a, 0, 1))), nan=20000)
        geography = .08 + .92 * np.maximum(loc_prior ** .35, same_loc)
        content = .45 * title + .30 * char + .18 * desc + .07 * params
        hist_norm = history / max(history.max(), 1e-8)
        combined = (.65 * hist_norm + .35 * content) * geography
        scores = dict(title=title, char=char, description=desc, params=params,
                      filter_match=filter_match, history=history, exact=exact,
                      same_location=same_loc, location_prior=loc_prior,
                      log_distance=np.log1p(distance),
                      same_category=(ix.category == str(query.get("search_category", -1))).astype(np.float32),
                      history_similarity=np.full(len(ix.ids), neighbors.max(initial=0), np.float32),
                      content=content, content_geo=content * geography,
                      history_geo=history * geography, combined=combined)
        if self.category_intent is not None:
            microcat_prior, confidence = self.category_intent.score(qw, qc)
            scores["microcat_prior"] = microcat_prior
            scores["microcat_confidence"] = np.full(len(ix.ids), confidence, np.float32)
        return scores, geography

    def candidates(self, query, n_per_channel=150):
        scores, geo = self.all_scores(query)
        # Union safeguards recall when one channel is unreliable.
        channels = [
            topk(scores["combined"], n_per_channel),
            topk(scores["content"] * geo, n_per_channel),
            topk(scores["history"] * geo, n_per_channel),
            topk(scores["content"], 50),
            topk(scores["history"], 50),
        ]
        if self.category_intent is not None and scores["microcat_prior"].max(initial=0) > 0:
            cat_scores = scores["microcat_prior"] * geo * (.05 + scores["content"])
            channels.append(topk(cat_scores, n_per_channel))
        rows = np.unique(np.concatenate(channels))
        return rows, self.features(query, rows, scores)

    def features(self, query, rows, scores):
        ix = self.index
        features = {name: values[rows] for name, values in scores.items()}
        features.update(popularity=np.log1p(self.popularity[rows]), rating=ix.rating[rows],
                        reviews=ix.reviews[rows], log_price=ix.price[rows])
        features["log_history"] = np.log1p(features["history"])
        features["log_exact"] = np.log1p(features["exact"])
        features["query_words"] = np.full(len(rows), len(normalize(query.get("search_query", "")).split()))
        features["delivery"] = np.full(len(rows), float(query.get("search_is_delivery_search", 0) or 0))
        for c in ("item_is_phone_hidden", "item_is_message_forbidden"):
            features[c] = ix.items[c].fillna(0).to_numpy(np.float32)[rows]
        return pd.DataFrame(features).fillna(0).astype(np.float32)
