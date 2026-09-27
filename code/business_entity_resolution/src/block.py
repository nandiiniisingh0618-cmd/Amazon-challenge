"""
block.py - Scalable candidate generation for millions of entity pairs.

FOUR blocking strategies (all within country buckets):
  A) TF-IDF char n-gram cosine on clean_name (top 50)
  B) Exact first-token match (all S23 entities sharing same first word)
  C) Two-token intersection: require >=2 shared significant tokens in name
  D) TF-IDF char n-gram cosine on clean_address (top 20, non-empty only)

Design goals:
  - High recall: must not miss true matches
  - Manageable candidate count: targets <200 avg candidates per S1 entity
  - Scalable to millions of records
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

from preprocess import clean_dataframe

# ─── Hyperparameters ─────────────────────────────────────────────────────────
TOP_NAME = 50   # Strategy A: top-K name TF-IDF candidates per S1 entity
TOP_ADDR = 20   # Strategy D: top-K address TF-IDF candidates
BATCH_S1 = 2000 # batch size for TF-IDF cosine computation

# Stopwords: tokens too common to be discriminative in blocking
STOP_TOKENS = {
    "the", "of", "and", "a", "an", "in", "for", "to", "at", "by", "on",
    "is", "with", "de", "la", "le", "les", "des", "du", "el", "en", "y",
    "ka", "ki", "ke", "ko", "se", "pvt", "private", "limited", "ltd",
    "inc", "incorporated", "company", "co", "corp", "corporation",
    "llc", "limited", "liability", "services", "svc", "svcs",
    "group", "grp", "international", "intl", "enterprises",
}
MIN_TOKEN_LEN = 3  # tokens shorter than this are excluded from index
# ─────────────────────────────────────────────────────────────────────────────


def _tfidf_candidates(
    s1_texts: list[str],
    pool_texts: list[str],
    top_k: int,
    ngram_range=(2, 4),
) -> list[list[int]]:
    """
    Fit TF-IDF on pool_texts, project s1_texts, return indices of top_k
    most similar pool entries for each s1 query.
    Uses batched sparse dot-products to stay memory efficient.
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
    # Fit on combined vocab for proper transform of both sides
    vec.fit(pool_texts + s1_texts)
    pool_mat  = vec.transform(pool_texts)   # (n_pool, vocab)
    query_mat = vec.transform(s1_texts)     # (n_s1, vocab)

    k = min(top_k, pool_mat.shape[0])
    results: list[list[int]] = []

    for i in range(0, query_mat.shape[0], BATCH_S1):
        batch = query_mat[i : i + BATCH_S1]
        sims  = (batch @ pool_mat.T).toarray()  # (batch_sz, n_pool)
        for row in sims:
            if k == 0:
                results.append([])
            else:
                top_idx = np.argpartition(row, -k)[-k:]
                top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
                results.append(top_idx.tolist())
    return results


def _is_significant(token: str) -> bool:
    """Return True if the token is meaningful for blocking."""
    return (
        len(token) >= MIN_TOKEN_LEN
        and token not in STOP_TOKENS
        and not token.isdigit()
    )


def _first_token_candidates(
    s1_ids: list[str],
    s1_names: list[str],
    pool_ids: list[str],
    pool_names: list[str],
) -> dict[str, list[str]]:
    """
    Strategy B: Index pool by first significant token.
    For each S1 entity, return all pool IDs with the same first significant token.
    """
    # Build index: first_significant_token → list[pool_id]
    index: dict[str, list[str]] = {}
    for pid, name in zip(pool_ids, pool_names):
        for tok in name.split():
            if _is_significant(tok):
                index.setdefault(tok, []).append(pid)
                break  # only index on the FIRST significant token

    result: dict[str, list[str]] = {}
    for s1_id, name in zip(s1_ids, s1_names):
        cands = []
        for tok in name.split():
            if _is_significant(tok):
                cands = index.get(tok, [])
                break
        result[s1_id] = cands
    return result


def _two_token_candidates(
    s1_ids: list[str],
    s1_names: list[str],
    pool_ids: list[str],
    pool_names: list[str],
) -> dict[str, set[str]]:
    """
    Strategy C: Require >=2 shared significant tokens.
    Build inverted index, then for each S1 entity count token matches.
    Only include pool IDs that match on at least 2 significant tokens.
    """
    # Build index: token → set of pool_ids
    index: dict[str, set[str]] = {}
    for pid, name in zip(pool_ids, pool_names):
        for tok in name.split():
            if _is_significant(tok):
                if tok not in index:
                    index[tok] = set()
                index[tok].add(pid)

    result: dict[str, set[str]] = {}
    for s1_id, name in zip(s1_ids, s1_names):
        sig_tokens = [tok for tok in name.split() if _is_significant(tok)]
        # Count how many S1 tokens each pool ID matches
        match_count: dict[str, int] = {}
        for tok in sig_tokens:
            for pid in index.get(tok, set()):
                match_count[pid] = match_count.get(pid, 0) + 1
        # Only keep pool IDs matching on >=2 tokens
        result[s1_id] = {pid for pid, cnt in match_count.items() if cnt >= 2}
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
    s1  = clean_dataframe(s1_df)
    s2  = clean_dataframe(s2_df)
    s3  = clean_dataframe(s3_df)
    s23 = pd.concat([s2, s3], ignore_index=True)

    # Init result dict
    candidates: dict[str, list[str]] = {eid: [] for eid in s1["entity_id"]}

    all_countries = s23["country"].unique().tolist()
    print(f"[block] Processing {len(all_countries)} country bucket(s): {all_countries}")

    for country in tqdm(all_countries, desc="[block] Country buckets"):
        pool     = s23[s23["country"] == country].reset_index(drop=True)
        s1_ctry  = s1[s1["country"] == country].reset_index(drop=True)

        if s1_ctry.empty or pool.empty:
            continue

        pool_ids   = pool["entity_id"].tolist()
        pool_names = pool["clean_name"].tolist()
        pool_addrs = pool["clean_address"].tolist()
        s1_ids     = s1_ctry["entity_id"].tolist()
        s1_names   = s1_ctry["clean_name"].tolist()
        s1_addrs   = s1_ctry["clean_address"].tolist()

        print(f"\n  [{country}] S1={len(s1_ids):,}, Pool={len(pool_ids):,}")

        # ── Strategy A: TF-IDF char n-gram on name ──────────────────────────
        print(f"  [{country}] Strategy A: TF-IDF name …")
        name_results = _tfidf_candidates(s1_names, pool_names, TOP_NAME)
        for i, idx_list in enumerate(name_results):
            s1_id = s1_ids[i]
            for idx in idx_list:
                candidates[s1_id].append(pool_ids[idx])

        # ── Strategy B: First significant token match ────────────────────────
        print(f"  [{country}] Strategy B: First-token …")
        first_tok_res = _first_token_candidates(s1_ids, s1_names, pool_ids, pool_names)
        for s1_id, cand_list in first_tok_res.items():
            candidates[s1_id].extend(cand_list)

        # ── Strategy C: Two-token intersection ──────────────────────────────
        print(f"  [{country}] Strategy C: Two-token intersection …")
        two_tok_res = _two_token_candidates(s1_ids, s1_names, pool_ids, pool_names)
        for s1_id, cand_set in two_tok_res.items():
            candidates[s1_id].extend(cand_set)

        # ── Strategy D: TF-IDF char n-gram on address (non-empty) ────────────
        non_empty_mask = [bool(a.strip()) for a in pool_addrs]
        n_non_empty = sum(non_empty_mask)
        if n_non_empty >= 10:
            print(f"  [{country}] Strategy D: TF-IDF address ({n_non_empty:,} non-empty) …")
            valid_pool_ids   = [pool_ids[i]   for i, m in enumerate(non_empty_mask) if m]
            valid_pool_addrs = [pool_addrs[i] for i, m in enumerate(non_empty_mask) if m]

            s1_addr_items = [(i, s1_ids[i], s1_addrs[i])
                             for i in range(len(s1_ids)) if s1_addrs[i].strip()]
            if s1_addr_items:
                s1_addr_texts = [x[2] for x in s1_addr_items]
                addr_results  = _tfidf_candidates(s1_addr_texts, valid_pool_addrs, TOP_ADDR)
                for j, idx_list in enumerate(addr_results):
                    s1_id = s1_addr_items[j][1]
                    for idx in idx_list:
                        candidates[s1_id].append(valid_pool_ids[idx])

    # ── Deduplicate per S1 entity ─────────────────────────────────────────────
    for s1_id in candidates:
        candidates[s1_id] = list(dict.fromkeys(candidates[s1_id]))

    total = sum(len(v) for v in candidates.values())
    n = max(len(candidates), 1)
    print(f"\n[block] Generated candidates for {len(candidates):,} S1 entities.")
    print(f"[block] Average candidates per S1 entity: {total / n:.1f}")
    print(f"[block] Total candidate pairs: {total:,}")
    return candidates


if __name__ == "__main__":
    BASE = Path(__file__).parent.parent.parent.parent
    s1 = pd.read_csv(BASE / "dataset/train/train_source1.tsv", sep="\t")
    s2 = pd.read_csv(BASE / "dataset/train/train_source2.tsv", sep="\t")
    s3 = pd.read_csv(BASE / "dataset/train/train_source3.tsv", sep="\t")
    cands = generate_candidates(s1, s2, s3)
