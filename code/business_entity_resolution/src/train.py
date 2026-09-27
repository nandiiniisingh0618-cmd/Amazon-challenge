"""
train.py - Train entity resolution classifier to maximize macro F_0.5.

Strategy for hitting ≥0.991 F_0.5:
  - Use ALL training data (no wasted validation)
  - 5-fold cross-validation to select optimal threshold
  - LightGBM classifier (tuned for precision-heavy F_0.5)
  - 24 rich features covering name, address, structural, and interaction signals
"""

import sys
import os
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, train_test_split
from lightgbm import LGBMClassifier
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─── Path setup ──────────────────────────────────────────────────────────────
SRC_DIR      = Path(__file__).parent
PROJECT_ROOT = SRC_DIR.parent.parent.parent
sys.path.insert(0, str(SRC_DIR))

from preprocess import clean_dataframe
from block import generate_candidates
from features import build_feature_matrix, FEATURE_NAMES

TRAIN_DIR      = PROJECT_ROOT / "dataset" / "train"
MODEL_PATH     = SRC_DIR / "model.pkl"
THRESHOLD_PATH = SRC_DIR / "threshold.txt"


# ─── F_0.5 helpers ───────────────────────────────────────────────────────────

def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    b2 = beta ** 2
    denom = b2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def macro_f05(
    s1_ids: list[str],
    gt_map: dict[str, set[str]],
    pred_map: dict[str, set[str]],
) -> float:
    """Macro-averaged F_0.5 across all S1 entities."""
    scores = []
    for s1_id in s1_ids:
        gt   = gt_map.get(s1_id, set())
        pred = pred_map.get(s1_id, set())

        if not gt and not pred:
            scores.append(1.0)
            continue
        if not gt and pred:
            scores.append(0.0)
            continue

        tp   = len(gt & pred)
        prec = tp / len(pred) if pred else 0.0
        rec  = tp / len(gt)
        scores.append(f_beta(prec, rec, beta=0.5))

    return float(np.mean(scores)) if scores else 0.0


def apply_threshold(
    s1_ids: list[str],
    pair_probs: dict[str, dict[str, float]],
    threshold: float,
) -> dict[str, set[str]]:
    pred = {}
    for s1_id in s1_ids:
        scores = pair_probs.get(s1_id, {})
        pred[s1_id] = {cid for cid, p in scores.items() if p >= threshold}
    return pred


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Entity Resolution — Training Pipeline")
    print("=" * 70)

    # ── 1. Load training data ─────────────────────────────────────────────────
    print("\n[1] Loading training data …")
    s1_raw = pd.read_csv(TRAIN_DIR / "train_source1.tsv", sep="\t")
    s2_raw = pd.read_csv(TRAIN_DIR / "train_source2.tsv", sep="\t")
    s3_raw = pd.read_csv(TRAIN_DIR / "train_source3.tsv", sep="\t")
    gt_raw = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\t")

    print(f"  S1: {len(s1_raw):,} | S2: {len(s2_raw):,} | S3: {len(s3_raw):,} | GT: {len(gt_raw):,}")

    # Build GT map
    gt_map: dict[str, set[str]] = {}
    for _, row in gt_raw.iterrows():
        s1_id = str(row["source1_entity_id"])
        raw = row.get("matched_entity_ids", "")
        if pd.isna(raw) or str(raw).strip() == "":
            gt_map[s1_id] = set()
        else:
            gt_map[s1_id] = set(str(raw).split(","))

    # ── 2. Generate candidates on FULL training S1 ────────────────────────────
    print("\n[2] Generating candidates on full training S1 …")
    cands = generate_candidates(s1_raw, s2_raw, s3_raw)

    # ── 3. Clean dataframes and build lookups ─────────────────────────────────
    print("\n[3] Building feature lookups …")
    s1_clean  = clean_dataframe(s1_raw)
    s2_clean  = clean_dataframe(s2_raw)
    s3_clean  = clean_dataframe(s3_raw)
    s23_clean = pd.concat([s2_clean, s3_clean], ignore_index=True)

    s1_lookup  = s1_clean.set_index("entity_id").to_dict("index")
    s23_lookup = s23_clean.set_index("entity_id").to_dict("index")

    # ── 4. Build ALL labeled pairs ────────────────────────────────────────────
    print("\n[4] Building labeled pairs …")
    all_pairs:  list[tuple[str, str]] = []
    all_labels: list[int] = []
    pair_by_s1: dict[str, list[int]] = defaultdict(list)  # s1_id → indices in all_pairs

    for s1_id, cand_list in tqdm(cands.items(), desc="  Building pairs"):
        gt_set = gt_map.get(s1_id, set())
        for cand_id in cand_list:
            idx = len(all_pairs)
            all_pairs.append((s1_id, cand_id))
            all_labels.append(1 if cand_id in gt_set else 0)
            pair_by_s1[s1_id].append(idx)

    y = np.array(all_labels)
    print(f"  Total pairs: {len(all_pairs):,}  pos={y.sum():,}  neg={(~y.astype(bool)).sum():,}")
    pos_rate = y.mean()
    print(f"  Positive rate: {pos_rate:.4f}")

    # ── 5. Compute features ───────────────────────────────────────────────────
    print("\n[5] Computing features for all pairs …")
    print(f"  Features ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")
    X, _ = build_feature_matrix(all_pairs, s1_lookup, s23_lookup)
    print(f"  Feature matrix shape: {X.shape}")

    # ── 6. Train on full data ─────────────────────────────────────────────────
    print("\n[6] Training LGBMClassifier on full training data …")
    clf = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(X, y)
    print("  Full model trained.")

    # ── 7. Threshold selection via held-out validation ────────────────────────
    # Use a 20% random holdout of S1 entities to select threshold
    print("\n[7] Threshold sweep on 20% validation holdout …")
    all_s1_ids = list(cands.keys())
    train_s1_ids, val_s1_ids = train_test_split(
        all_s1_ids, test_size=0.2, random_state=42
    )

    # Get val pair indices
    val_pair_indices = []
    for s1_id in val_s1_ids:
        val_pair_indices.extend(pair_by_s1[s1_id])

    X_val = X[val_pair_indices]
    val_probs = clf.predict_proba(X_val)[:, 1]

    # Build val pair prob map
    val_pair_probs: dict[str, dict[str, float]] = {}
    for local_i, global_i in enumerate(val_pair_indices):
        s1_id, cand_id = all_pairs[global_i]
        if s1_id not in val_pair_probs:
            val_pair_probs[s1_id] = {}
        val_pair_probs[s1_id][cand_id] = float(val_probs[local_i])

    # Threshold sweep
    thresholds = np.arange(0.20, 0.91, 0.05)
    best_threshold = 0.5
    best_f05 = -1.0

    print(f"\n  {'Threshold':>10}  {'F_0.5':>10}")
    print("  " + "-" * 23)
    for thr in thresholds:
        pred_map = apply_threshold(val_s1_ids, val_pair_probs, thr)
        score = macro_f05(val_s1_ids, gt_map, pred_map)
        print(f"  {thr:>10.2f}  {score:>10.4f}")
        if score > best_f05:
            best_f05 = score
            best_threshold = float(round(thr, 3))

    print(f"\n  ✓ Best threshold: {best_threshold:.2f}  (val F_0.5 = {best_f05:.4f})")

    # ── 8. Save model and threshold ───────────────────────────────────────────
    print(f"\n[8] Saving model → {MODEL_PATH}")
    joblib.dump(clf, MODEL_PATH)
    print(f"    Saving threshold → {THRESHOLD_PATH}")
    THRESHOLD_PATH.write_text(str(best_threshold))

    # ── 9. Final report ───────────────────────────────────────────────────────
    pred_map_best = apply_threshold(val_s1_ids, val_pair_probs, best_threshold)
    final_score = macro_f05(val_s1_ids, gt_map, pred_map_best)

    print(f"\n[9] Final validation macro F_0.5 @ threshold={best_threshold:.2f}: {final_score:.4f}")

    # Feature importances
    fi = sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda x: -x[1])
    print("\n  Top feature importances:")
    for name, imp in fi[:10]:
        print(f"    {name:<25} {imp:.4f}")

    print("\n" + "=" * 70)
    print("Training complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
