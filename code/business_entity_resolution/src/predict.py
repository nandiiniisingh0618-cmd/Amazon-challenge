"""
predict.py - Generate predictions on the test set.

Steps:
  1. Load test TSVs and clean.
  2. Load model.pkl and threshold.txt.
  3. Generate candidates using block.py.
  4. Compute features using features.py.
  5. Predict probabilities; apply threshold.
  6. Write output/candidate_pairs.tsv and output/matching_results.tsv.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

# ─── Path setup ──────────────────────────────────────────────────────────────
SRC_DIR = Path(__file__).parent
PROJECT_ROOT = SRC_DIR.parent.parent.parent  # …/amazon-entity-resolution/
sys.path.insert(0, str(SRC_DIR))

from preprocess import clean_dataframe
from block import generate_candidates
from features import build_feature_matrix

TEST_DIR   = PROJECT_ROOT / "dataset" / "test"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_PATH     = SRC_DIR / "model.pkl"
THRESHOLD_PATH = SRC_DIR / "threshold.txt"


def main():
    print("=" * 60)
    print("Entity Resolution — Prediction")
    print("=" * 60)

    # ── 1. Load test data ──────────────────────────────────────────────────────
    print("\n[1] Loading test data …")
    s1_raw = pd.read_csv(TEST_DIR / "test_source1.tsv", sep="\t")
    s2_raw = pd.read_csv(TEST_DIR / "test_source2.tsv", sep="\t")
    s3_raw = pd.read_csv(TEST_DIR / "test_source3.tsv", sep="\t")
    print(f"  S1: {len(s1_raw)} | S2: {len(s2_raw)} | S3: {len(s3_raw)}")

    # ── 2. Load model and threshold ────────────────────────────────────────────
    print("\n[2] Loading model and threshold …")
    clf       = joblib.load(MODEL_PATH)
    threshold = float(THRESHOLD_PATH.read_text().strip())
    print(f"  Threshold: {threshold:.2f}")

    # ── 3. Generate candidates ─────────────────────────────────────────────────
    print("\n[3] Generating candidates on test S1 …")
    cands_dict = generate_candidates(s1_raw, s2_raw, s3_raw)

    # Build lookup dicts from cleaned data
    s1_clean  = clean_dataframe(s1_raw)
    s2_clean  = clean_dataframe(s2_raw)
    s3_clean  = clean_dataframe(s3_raw)
    s23_clean = pd.concat([s2_clean, s3_clean], ignore_index=True)

    s1_lookup  = {row["entity_id"]: row.to_dict() for _, row in s1_clean.iterrows()}
    s23_lookup = {row["entity_id"]: row.to_dict() for _, row in s23_clean.iterrows()}

    # Valid S2/S3 entity IDs for integrity check
    valid_s23_ids = set(s23_clean["entity_id"].tolist())
    # Valid S1 entity IDs
    valid_s1_ids = set(s1_clean["entity_id"].tolist())

    # ── 4. Compute features ────────────────────────────────────────────────────
    print("\n[4] Computing features for candidate pairs …")
    # Flatten into ordered list for matrix computation
    all_s1_ids = s1_raw["entity_id"].tolist()
    all_pairs  = []
    pair_index: dict[str, list[tuple[str, int]]] = {}  # s1_id → list of (cand_id, global_idx)

    for s1_id in all_s1_ids:
        cand_list = cands_dict.get(s1_id, [])
        # Filter: must be valid S2/S3 IDs, must not be S1 IDs, no duplicates
        seen = set()
        clean_cands = []
        for cid in cand_list:
            if cid in seen:
                continue
            if cid in valid_s23_ids and cid not in valid_s1_ids:
                clean_cands.append(cid)
                seen.add(cid)

        pair_index[s1_id] = []
        for cid in clean_cands:
            idx = len(all_pairs)
            all_pairs.append((s1_id, cid))
            pair_index[s1_id].append((cid, idx))

    print(f"  Total candidate pairs: {len(all_pairs)}")

    if all_pairs:
        X, _ = build_feature_matrix(all_pairs, s1_lookup, s23_lookup)
        probs = clf.predict_proba(X)[:, 1]
    else:
        probs = np.array([])

    # ── 5 & 6. Apply threshold and write output ────────────────────────────────
    print(f"\n[5] Applying threshold {threshold:.2f} and writing output …")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    candidate_rows = []
    matching_rows  = []
    n_with_match   = 0
    n_singletons   = 0

    for s1_id in all_s1_ids:
        cand_pairs_for_entity = pair_index.get(s1_id, [])

        # All candidates (regardless of threshold)
        all_cands_for_entity = [cid for cid, _ in cand_pairs_for_entity]

        # Predicted matches (above threshold)
        matches = []
        for cid, idx in cand_pairs_for_entity:
            if probs[idx] >= threshold:
                matches.append(cid)

        candidate_rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(all_cands_for_entity),
        })
        matching_rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(matches),
        })

        if matches:
            n_with_match += 1
        else:
            n_singletons += 1

    # Write candidate_pairs.tsv
    cands_df = pd.DataFrame(candidate_rows, columns=["source1_entity_id", "candidate_entity_ids"])
    cands_df.to_csv(OUTPUT_DIR / "candidate_pairs.tsv", sep="\t", index=False)

    # Write matching_results.tsv
    match_df = pd.DataFrame(matching_rows, columns=["source1_entity_id", "matched_entity_ids"])
    match_df.to_csv(OUTPUT_DIR / "matching_results.tsv", sep="\t", index=False)

    print(f"\n  S1 entities with ≥1 predicted match : {n_with_match}")
    print(f"  S1 entities predicted as singletons  : {n_singletons}")
    print(f"\n  ✓ output/candidate_pairs.tsv  ({len(cands_df)} rows)")
    print(f"  ✓ output/matching_results.tsv ({len(match_df)} rows)")
    print("=" * 60)
    print("Prediction complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
