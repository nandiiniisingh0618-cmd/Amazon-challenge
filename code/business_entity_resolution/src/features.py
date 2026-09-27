"""
features.py - Feature engineering for entity resolution pair classification.

For each (S1, candidate S2/S3) pair computes a rich feature vector:
- Multiple string similarity metrics on name and address
- Structural features
- Token-level overlap features  
- Numeric token matching (street numbers)
- Length and character-level features
"""

import sys
import re
from pathlib import Path
import numpy as np
from rapidfuzz import fuzz

SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _jaccard(a: str, b: str) -> float:
    """Jaccard similarity on word-token sets."""
    set_a = set(a.split())
    set_b = set(b.split())
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union


def _shared_tokens(a: str, b: str) -> int:
    """Count tokens in common between two strings."""
    return len(set(a.split()) & set(b.split()))


def _extract_numbers(text: str) -> set[str]:
    """Extract numeric tokens from text."""
    return set(re.findall(r'\d+', text))


def _number_overlap(a: str, b: str) -> float:
    """Fraction of numeric tokens that overlap between two strings."""
    na = _extract_numbers(a)
    nb = _extract_numbers(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    inter = len(na & nb)
    union = len(na | nb)
    return inter / union


def _first_token_match(a: str, b: str) -> float:
    """1 if first tokens match, else 0."""
    ta = a.split()
    tb = b.split()
    if not ta or not tb:
        return 0.0
    return 1.0 if ta[0] == tb[0] else 0.0


def _last_token_match(a: str, b: str) -> float:
    """1 if last tokens match, else 0."""
    ta = a.split()
    tb = b.split()
    if not ta or not tb:
        return 0.0
    return 1.0 if ta[-1] == tb[-1] else 0.0


def _prefix_sim(a: str, b: str, n: int = 4) -> float:
    """Fraction of character n-grams in common in first min(n,len) chars."""
    a4 = a[:n]
    b4 = b[:n]
    if not a4 and not b4:
        return 1.0
    if not a4 or not b4:
        return 0.0
    return fuzz.ratio(a4, b4) / 100.0


def _containment(a: str, b: str) -> float:
    """How much of the shorter token set is contained in the longer."""
    sa = set(a.split())
    sb = set(b.split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    shorter = sa if len(sa) <= len(sb) else sb
    longer = sb if len(sa) <= len(sb) else sa
    return len(shorter & longer) / len(shorter)


# ─── Main feature computation ─────────────────────────────────────────────────

def compute_features(row1: dict, row2: dict) -> dict:
    """
    Compute all features for a single (S1, candidate) pair.
    """
    n1 = row1.get("clean_name", "") or ""
    n2 = row2.get("clean_name", "") or ""
    a1 = row1.get("clean_address", "") or ""
    a2 = row2.get("clean_address", "") or ""
    c1 = row1.get("country", "") or ""
    c2 = row2.get("country", "") or ""

    # ── Name similarity features ─────────────────────────────────────────────
    name_lev_ratio   = fuzz.ratio(n1, n2) / 100.0
    name_token_sort  = fuzz.token_sort_ratio(n1, n2) / 100.0
    name_partial     = fuzz.partial_ratio(n1, n2) / 100.0
    name_jaro        = fuzz.WRatio(n1, n2) / 100.0
    name_jaccard     = _jaccard(n1, n2)
    name_containment = _containment(n1, n2)
    name_first_tok   = _first_token_match(n1, n2)
    name_last_tok    = _last_token_match(n1, n2)
    name_prefix_sim  = _prefix_sim(n1, n2, 6)
    shared_name_toks = float(_shared_tokens(n1, n2))
    name_token_set   = fuzz.token_set_ratio(n1, n2) / 100.0

    # ── Address similarity features ──────────────────────────────────────────
    addr_lev_ratio   = fuzz.ratio(a1, a2) / 100.0
    addr_token_sort  = fuzz.token_sort_ratio(a1, a2) / 100.0
    addr_partial     = fuzz.partial_ratio(a1, a2) / 100.0
    addr_jaccard     = _jaccard(a1, a2)
    addr_number_overlap = _number_overlap(a1, a2)
    addr_containment = _containment(a1, a2)

    # ── Structural features ──────────────────────────────────────────────────
    country_match   = 1.0 if c1 == c2 else 0.0

    max_len_n = max(len(n1), len(n2))
    name_len_ratio  = min(len(n1), len(n2)) / (max_len_n + 1)

    max_len_a = max(len(a1), len(a2))
    addr_len_ratio  = min(len(a1), len(a2)) / (max_len_a + 1)

    max_name_addr   = max(name_lev_ratio, addr_lev_ratio)
    both_empty_addr = 1.0 if (not a1.strip() and not a2.strip()) else 0.0

    # ── Combined/interaction features ────────────────────────────────────────
    name_addr_avg   = (name_lev_ratio + addr_lev_ratio) / 2.0
    name_addr_product = name_lev_ratio * addr_lev_ratio

    return {
        # Name features
        "name_lev_ratio":     name_lev_ratio,
        "name_token_sort":    name_token_sort,
        "name_partial":       name_partial,
        "name_jaro":          name_jaro,
        "name_jaccard":       name_jaccard,
        "name_token_set":     name_token_set,
        "name_containment":   name_containment,
        "name_first_tok":     name_first_tok,
        "name_last_tok":      name_last_tok,
        "name_prefix_sim":    name_prefix_sim,
        "shared_name_tokens": shared_name_toks,
        # Address features
        "addr_lev_ratio":     addr_lev_ratio,
        "addr_token_sort":    addr_token_sort,
        "addr_partial":       addr_partial,
        "addr_jaccard":       addr_jaccard,
        "addr_number_overlap":addr_number_overlap,
        "addr_containment":   addr_containment,
        # Structural
        "country_match":      country_match,
        "name_len_ratio":     name_len_ratio,
        "addr_len_ratio":     addr_len_ratio,
        "max_name_addr":      max_name_addr,
        "both_empty_addr":    both_empty_addr,
        "name_addr_avg":      name_addr_avg,
        "name_addr_product":  name_addr_product,
    }


# Feature names in a consistent order
FEATURE_NAMES = [
    "name_lev_ratio",
    "name_token_sort",
    "name_partial",
    "name_jaro",
    "name_jaccard",
    "name_token_set",
    "name_containment",
    "name_first_tok",
    "name_last_tok",
    "name_prefix_sim",
    "shared_name_tokens",
    "addr_lev_ratio",
    "addr_token_sort",
    "addr_partial",
    "addr_jaccard",
    "addr_number_overlap",
    "addr_containment",
    "country_match",
    "name_len_ratio",
    "addr_len_ratio",
    "max_name_addr",
    "both_empty_addr",
    "name_addr_avg",
    "name_addr_product",
]


def build_feature_matrix(
    pairs: list[tuple[str, str]],
    s1_lookup: dict[str, dict],
    s23_lookup: dict[str, dict],
) -> tuple[np.ndarray, list[str]]:
    """
    Build a feature matrix for a list of (s1_id, cand_id) pairs.

    Returns (X array of shape [n_pairs, n_features], feature_names list).
    """
    rows = []
    for s1_id, cand_id in pairs:
        r1 = s1_lookup.get(s1_id, {})
        r2 = s23_lookup.get(cand_id, {})
        feat = compute_features(r1, r2)
        rows.append([feat[f] for f in FEATURE_NAMES])

    if rows:
        X = np.array(rows, dtype=np.float32)
    else:
        X = np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

    return X, FEATURE_NAMES
