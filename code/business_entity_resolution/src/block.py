"""
block.py - Scalable candidate generation for millions of entity pairs.

THREE blocking strategies, all within country buckets:
  A) TF-IDF char n-gram cosine on clean_name (top 30)
  B) Inverted token index on clean_name words (any shared word = candidate)
  C) TF-IDF char n-gram cosine on clean_address (top 20, only when non-empty)

Strategy B (inverted index) ensures high recall for fuzzy/typo matches — it
catches partial matches even when only a single keyword aligns. Strategy A
refines that with n-gram similarity. Strategy C adds address evidence.

For the very large datasets (millions of records), we:
  - Process one country at a time
  - Use batched sparse matrix multiplication for TF-IDF
  - Build inverted index in-memory per country bucket
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

from preprocess import clean_dataframe

# ─── Hyperparameters ─────────────────────────────────────────────────────────
TOP_NAME = 30   # Strategy A: top-K name TF-IDF candidates per S1 entity
TOP_ADDR = 20   # Strategy C: top-K address TF-IDF candidates
BATCH_S1 = 1000 # batch size for TF-IDF cosine computation
# ─────────────────────────────────────────────────────────────────────────────


def _tfidf_candidates(
    s1_texts: list[str],
    pool_texts: list[str],
    top_k: int,
    ngram_range=(2, 4),
) -> list[list[int]]:
    """
    Fit TF-IDF on pool_texts, project s1_texts, return top_k pool indices for each.
    Uses batched sparse dot products to avoid OOM on large corpora.
    """
    if not pool_texts or not s1_texts:
        return [[] for _ in s1_texts]

    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        min_df=1,
        sublinear_tf=True,
        dtype=np.float32,
    )
    # Fit on all texts for shared vocabulary
    vec.fit(pool_texts + s1_texts)
    pool_mat = vec.transform(pool_texts)   # (n_pool, vocab)
    query_mat = vec.transform(s1_texts)   # (n_s1, vocab)

    k = min(top_k, pool_mat.shape[0])
    results = []

    for i in range(0, query_mat.shape[0], BATCH_S1):
        batch = query_mat[i : i + BATCH_S1]
        # sparse @ sparse.T → dense
        sims = (batch @ pool_mat.T).toarray()   # (batch, n_pool)
        for row in sims:
            top_idx = np.argpartition(row, -k)[-k:] if k > 0 else np.array([], dtype=int)
            top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
            results.append(top_idx.tolist())

    return results


def _inverted_index_candidates(
    s1_ids: list[str],
    s1_names: list[str],
    pool_ids: list[str],
    pool_names: list[str],
) -> dict[str, set[str]]:
    """
    Build an inverted index on pool name tokens.
    For each S1 entity, return all pool IDs sharing ≥1 non-trivial name token.
    """
    # Stopwords to exclude from index (very common single-char / stop tokens)
    STOP = {"the", "of", "and", "a", "an", "in", "for", "to", "at", "by", "on",
            "is", "with", "de", "la", "le", "les", "des", "du", "el", "en", "y",
            "ka", "ki", "ke", "ko", "se"}

    # Build index: token → list of pool_ids
    index: dict[str, list[str]] = {}
    for pid, name in zip(pool_ids, pool_names):
        for tok in name.split():
            if len(tok) >= 3 and tok not in STOP:
                if tok not in index:
                    index[tok] = []
                index[tok].append(pid)

    # Lookup
    result: dict[str, set[str]] = {}
    for s1_id, name in zip(s1_ids, s1_names):
        cands = set()
        for tok in name.split():
            if len(tok) >= 3 and tok not in STOP:
                for pid in index.get(tok, []):
                    cands.add(pid)
        result[s1_id] = cands
    return result


def generate_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
) -> dict[str, list[str]]:
    """
    Generate candidate S2/S3 entity IDs for each S1 entity.

    Returns dict: s1_entity_id → deduplicated list of candidate entity_ids.
    """
    print("[block] Cleaning dataframes …")
    s1 = clean_dataframe(s1_df)
    s2 = clean_dataframe(s2_df)
    s3 = clean_dataframe(s3_df)
    s23 = pd.concat([s2, s3], ignore_index=True)

    candidates: dict[str, list[str]] = {eid: [] for eid in s1["entity_id"]}

    all_countries = s23["country"].unique().tolist()
    # Also process S1 entities whose country might not appear in S23
    s1_only_countries = set(s1["country"].unique()) - set(all_countries)
    if s1_only_countries:
        print(f"[block] Warning: S1 has countries not in S23: {s1_only_countries}")

    print(f"[block] Processing {len(all_countries)} country bucket(s) …")

    for country in tqdm(all_countries, desc="[block] Country buckets"):
        pool = s23[s23["country"] == country].reset_index(drop=True)
        s1_ctry = s1[s1["country"] == country].reset_index(drop=True)

        if s1_ctry.empty or pool.empty:
            continue

        pool_ids   = pool["entity_id"].tolist()
        pool_names = pool["clean_name"].tolist()
        pool_addrs = pool["clean_address"].tolist()
        s1_ids     = s1_ctry["entity_id"].tolist()
        s1_names   = s1_ctry["clean_name"].tolist()
        s1_addrs   = s1_ctry["clean_address"].tolist()

        # ── Strategy A: TF-IDF char n-gram on name ──────────────────────────
        name_results = _tfidf_candidates(s1_names, pool_names, TOP_NAME)
        for i, idx_list in enumerate(name_results):
            s1_id = s1_ids[i]
            for idx in idx_list:
                candidates[s1_id].append(pool_ids[idx])

        # ── Strategy B: Inverted token index on name ─────────────────────────
        inv_results = _inverted_index_candidates(s1_ids, s1_names, pool_ids, pool_names)
        for s1_id, cand_set in inv_results.items():
            candidates[s1_id].extend(cand_set)

        # ── Strategy C: TF-IDF on address (skip if mostly empty) ─────────────
        non_empty_mask = [bool(a.strip()) for a in pool_addrs]
        if sum(non_empty_mask) >= 10:
            valid_pool_ids   = [pool_ids[i] for i, m in enumerate(non_empty_mask) if m]
            valid_pool_addrs = [pool_addrs[i] for i, m in enumerate(non_empty_mask) if m]

            # Only S1 entities with non-empty address
            s1_with_addr = [(i, s1_ids[i], s1_addrs[i])
                            for i in range(len(s1_ids))
                            if s1_addrs[i].strip()]

            if s1_with_addr:
                s1_addr_texts = [x[2] for x in s1_with_addr]
                addr_results = _tfidf_candidates(s1_addr_texts, valid_pool_addrs, TOP_ADDR)
                for j, idx_list in enumerate(addr_results):
                    s1_id = s1_with_addr[j][1]
                    for idx in idx_list:
                        candidates[s1_id].append(valid_pool_ids[idx])

    # ── Deduplicate per S1 entity ─────────────────────────────────────────────
    for s1_id in candidates:
        candidates[s1_id] = list(dict.fromkeys(candidates[s1_id]))

    total = sum(len(v) for v in candidates.values())
    n = max(len(candidates), 1)
    print(f"[block] Generated candidates for {len(candidates)} S1 entities.")
    print(f"[block] Average candidates per S1 entity: {total / n:.1f}")
    print(f"[block] Total candidate pairs: {total:,}")
    return candidates


if __name__ == "__main__":
    BASE = Path(__file__).parent.parent.parent.parent
    s1 = pd.read_csv(BASE / "dataset/train/train_source1.tsv", sep="\t")
    s2 = pd.read_csv(BASE / "dataset/train/train_source2.tsv", sep="\t")
    s3 = pd.read_csv(BASE / "dataset/train/train_source3.tsv", sep="\t")
    cands = generate_candidates(s1, s2, s3)
    print(f"Sample: {list(cands.items())[:3]}")
