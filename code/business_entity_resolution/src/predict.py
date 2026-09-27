"""
predict.py - Generate predictions on test set using trained model.

Processes test data in country-level batches to stay memory efficient
on the very large test dataset (millions of entities per country).
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─── Path setup ──────────────────────────────────────────────────────────────
SRC_DIR      = Path(__file__).parent
PROJECT_ROOT = SRC_DIR.parent.parent.parent
sys.path.insert(0, str(SRC_DIR))

from preprocess import clean_dataframe
from block import generate_candidates
from features import build_feature_matrix, FEATURE_NAMES

TEST_DIR       = PROJECT_ROOT / "dataset" / "test"
OUTPUT_DIR     = PROJECT_ROOT / "output"
MODEL_PATH     = SRC_DIR / "model.pkl"
THRESHOLD_PATH = SRC_DIR / "threshold.txt"

# Feature computation batch size (pairs at a time)
FEAT_BATCH = 200_000


def main():
    print("=" * 70)
    print("Entity Resolution — Prediction")
    print("=" * 70)

    # ── 1. Load test data ─────────────────────────────────────────────────────
    print("\n[1] Loading test data …")
    s1_raw = pd.read_csv(TEST_DIR / "test_source1.tsv", sep="\t")
    s2_raw = pd.read_csv(TEST_DIR / "test_source2.tsv", sep="\t")
    s3_raw = pd.read_csv(TEST_DIR / "test_source3.tsv", sep="\t")
    print(f"  S1: {len(s1_raw):,} | S2: {len(s2_raw):,} | S3: {len(s3_raw):,}")

    # ── 2. Load model and threshold ────────────────────────────────────────────
    print("\n[2] Loading model and threshold …")
    clf       = joblib.load(MODEL_PATH)
    threshold = float(THRESHOLD_PATH.read_text().strip())
    print(f"  Model: {MODEL_PATH.name}  Threshold: {threshold:.2f}")

    # ── 3. Generate candidates ─────────────────────────────────────────────────
    print("\n[3] Generating candidates on test S1 …")
    cands_dict = generate_candidates(s1_raw, s2_raw, s3_raw)

    # ── 4. Build cleaned lookups ───────────────────────────────────────────────
    print("\n[4] Building feature lookups …")
    s1_clean  = clean_dataframe(s1_raw)
    s2_clean  = clean_dataframe(s2_raw)
    s3_clean  = clean_dataframe(s3_raw)
    s23_clean = pd.concat([s2_clean, s3_clean], ignore_index=True)

    s1_lookup  = s1_clean.set_index("entity_id").to_dict("index")
    s23_lookup = s23_clean.set_index("entity_id").to_dict("index")

    valid_s23_ids = set(s23_clean["entity_id"])
    valid_s1_ids  = set(s1_clean["entity_id"])

    # ── 5. Build ordered pair list ─────────────────────────────────────────────
    print("\n[5] Building ordered pair list …")
    all_s1_ids = s1_raw["entity_id"].tolist()

    # pair_index: s1_id → list of (cand_id, global_pair_index)
    pair_index: dict[str, list[tuple[str, int]]] = {}
    all_pairs:  list[tuple[str, str]] = []

    for s1_id in all_s1_ids:
        cand_list = cands_dict.get(s1_id, [])
        pair_index[s1_id] = []
        seen = set()
        for cid in cand_list:
            if cid in seen or cid not in valid_s23_ids or cid in valid_s1_ids:
                continue
            idx = len(all_pairs)
            all_pairs.append((s1_id, cid))
            pair_index[s1_id].append((cid, idx))
            seen.add(cid)

    print(f"  Total candidate pairs: {len(all_pairs):,}")

    # ── 6. Compute features and predict in batches ─────────────────────────────
    print(f"\n[6] Computing features & predicting in batches of {FEAT_BATCH:,} …")
    all_probs = np.empty(len(all_pairs), dtype=np.float32)

    for start in tqdm(range(0, len(all_pairs), FEAT_BATCH), desc="  Batches"):
        end   = min(start + FEAT_BATCH, len(all_pairs))
        batch = all_pairs[start:end]
        X, _  = build_feature_matrix(batch, s1_lookup, s23_lookup)
        if X.shape[0] > 0:
            all_probs[start:end] = clf.predict_proba(X)[:, 1]

    # ── 7. Apply threshold and write output ────────────────────────────────────
    print(f"\n[7] Applying threshold {threshold:.2f} and writing output …")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    candidate_rows: list[dict] = []
    matching_rows:  list[dict] = []
    n_with_match = 0
    n_singletons = 0

    for s1_id in all_s1_ids:
        pairs_for_entity = pair_index.get(s1_id, [])

        all_cands = [cid for cid, _ in pairs_for_entity]
        matches   = [cid for cid, idx in pairs_for_entity if all_probs[idx] >= threshold]

        candidate_rows.append({
            "source1_entity_id":   s1_id,
            "candidate_entity_ids": ",".join(all_cands),
        })
        matching_rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(matches),
        })

        if matches:
            n_with_match += 1
        else:
            n_singletons += 1

    # ── 8. Write TSVs ─────────────────────────────────────────────────────────
    print("\n[8] Writing output files …")
    cands_df = pd.DataFrame(candidate_rows, columns=["source1_entity_id", "candidate_entity_ids"])
    match_df  = pd.DataFrame(matching_rows,  columns=["source1_entity_id", "matched_entity_ids"])

    cands_df.to_csv(OUTPUT_DIR / "candidate_pairs.tsv",  sep="\t", index=False, encoding="utf-8")
    match_df.to_csv(OUTPUT_DIR / "matching_results.tsv", sep="\t", index=False, encoding="utf-8")

    print(f"\n  S1 entities with ≥1 predicted match : {n_with_match:,}")
    print(f"  S1 entities predicted as singletons  : {n_singletons:,}")
    print(f"\n  ✓ output/candidate_pairs.tsv  ({len(cands_df):,} rows)")
    print(f"  ✓ output/matching_results.tsv ({len(match_df):,} rows)")
    print("=" * 70)
    print("Prediction complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
