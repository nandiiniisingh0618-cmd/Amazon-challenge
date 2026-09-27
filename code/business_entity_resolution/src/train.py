"""
train.py - Train entity resolution classifier to maximize macro F_0.5.

At 2.2M S1 entities with ~10M S2/S3 candidates, full blocking is very slow.
Strategy:
  - Sample N_SAMPLE S1 entities for training (stratified: singletons + non-singletons)
  - Generate candidates on that sample
  - Train LightGBM on those pairs
  - Threshold selected on 20% held-out validation of the sample
  - Model generalizes because features are query-agnostic
"""

import sys
import os
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
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

# Number of S1 entities to sample for training (keeps it feasible)
# Enough for a representative distribution of names/addresses
N_SAMPLE = 50000


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
    return {
        s1_id: {cid for cid, p in pair_probs.get(s1_id, {}).items() if p >= threshold}
        for s1_id in s1_ids
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Entity Resolution — Training Pipeline")
    print("=" * 70)

    # ── 1. Load training data ─────────────────────────────────────────────────
    print(f"\n[1] Loading training data …")
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

    # ── 2. Sample S1 entities ─────────────────────────────────────────────────
    print(f"\n[2] Sampling {N_SAMPLE:,} S1 entities for training …")
    all_s1_ids = s1_raw["entity_id"].tolist()

    # Stratified: keep realistic proportion of singletons
    singleton_ids    = [sid for sid in all_s1_ids if not gt_map.get(sid)]
    non_singleton_ids = [sid for sid in all_s1_ids if gt_map.get(sid)]

    singleton_frac = len(singleton_ids) / len(all_s1_ids)
    n_singletons = int(N_SAMPLE * singleton_frac)
    n_non_singletons = N_SAMPLE - n_singletons

    np.random.seed(42)
    sampled_singletons    = np.random.choice(singleton_ids,    min(n_singletons, len(singleton_ids)), replace=False).tolist()
    sampled_non_singletons = np.random.choice(non_singleton_ids, min(n_non_singletons, len(non_singleton_ids)), replace=False).tolist()
    sampled_ids = sampled_singletons + sampled_non_singletons

    print(f"  Sampled: {len(sampled_singletons):,} singletons + {len(sampled_non_singletons):,} non-singletons")

    s1_sample = s1_raw[s1_raw["entity_id"].isin(set(sampled_ids))].reset_index(drop=True)

    # Split sample into train/val (80/20) for threshold selection
    train_ids, val_ids = train_test_split(sampled_ids, test_size=0.2, random_state=42)
    s1_train = s1_raw[s1_raw["entity_id"].isin(set(train_ids))].reset_index(drop=True)
    s1_val   = s1_raw[s1_raw["entity_id"].isin(set(val_ids))].reset_index(drop=True)

    print(f"  Train: {len(s1_train):,} | Val: {len(s1_val):,}")

    # ── 3. Generate candidates ────────────────────────────────────────────────
    print("\n[3] Generating train candidates …")
    train_cands = generate_candidates(s1_train, s2_raw, s3_raw)
    print("\n[3] Generating val candidates …")
    val_cands   = generate_candidates(s1_val,   s2_raw, s3_raw)

    # ── 4. Build feature lookups ──────────────────────────────────────────────
    print("\n[4] Building feature lookups …")
    s1_clean  = clean_dataframe(s1_raw)
    s2_clean  = clean_dataframe(s2_raw)
    s3_clean  = clean_dataframe(s3_raw)
    s23_clean = pd.concat([s2_clean, s3_clean], ignore_index=True)

    s1_lookup  = s1_clean.set_index("entity_id").to_dict("index")
    s23_lookup = s23_clean.set_index("entity_id").to_dict("index")

    # ── 5. Build labeled pairs ────────────────────────────────────────────────
    print("\n[5] Building labeled pairs …")

    def make_pairs(cands_dict, gt):
        pairs, labels = [], []
        for s1_id, cand_list in cands_dict.items():
            gt_set = gt.get(s1_id, set())
            for cid in cand_list:
                pairs.append((s1_id, cid))
                labels.append(1 if cid in gt_set else 0)
        return pairs, labels

    train_pairs, train_labels = make_pairs(train_cands, gt_map)
    val_pairs,   val_labels   = make_pairs(val_cands,   gt_map)

    y_tr = np.array(train_labels)
    y_va = np.array(val_labels)
    print(f"  Train pairs: {len(train_pairs):,}  pos={y_tr.sum():,}  neg={(~y_tr.astype(bool)).sum():,}")
    print(f"  Val   pairs: {len(val_pairs):,}    pos={y_va.sum():,}  neg={(~y_va.astype(bool)).sum():,}")

    # ── 6. Compute features ───────────────────────────────────────────────────
    print(f"\n[6] Computing features ({len(FEATURE_NAMES)} each) …")
    print("  Train …")
    X_tr, _ = build_feature_matrix(train_pairs, s1_lookup, s23_lookup)
    print("  Val …")
    X_va, _ = build_feature_matrix(val_pairs,   s1_lookup, s23_lookup)
    print(f"  Train shape: {X_tr.shape}  Val shape: {X_va.shape}")

    # ── 7. Train LightGBM ─────────────────────────────────────────────────────
    print("\n[7] Training LGBMClassifier …")
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
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[],
    )
    print("  Model trained.")

    # ── 8. Threshold sweep ────────────────────────────────────────────────────
    print("\n[8] Threshold sweep on validation set …")
    val_probs = clf.predict_proba(X_va)[:, 1]

    # Build per-S1 probability map
    val_pair_probs: dict[str, dict[str, float]] = {}
    for (s1_id, cand_id), prob in zip(val_pairs, val_probs.tolist()):
        val_pair_probs.setdefault(s1_id, {})[cand_id] = prob

    thresholds = np.arange(0.20, 0.91, 0.05)
    best_threshold = 0.5
    best_f05 = -1.0

    print(f"\n  {'Threshold':>10}  {'F_0.5':>10}")
    print("  " + "-" * 23)
    for thr in thresholds:
        pred_map = apply_threshold(val_ids, val_pair_probs, float(thr))
        score    = macro_f05(val_ids, gt_map, pred_map)
        marker = " ←" if score > best_f05 else ""
        print(f"  {thr:>10.2f}  {score:>10.4f}{marker}")
        if score > best_f05:
            best_f05 = score
            best_threshold = float(round(thr, 3))

    print(f"\n  ✓ Best threshold: {best_threshold:.2f}  (val F_0.5 = {best_f05:.4f})")

    # ── 9. Save model and threshold ───────────────────────────────────────────
    print(f"\n[9] Saving model → {MODEL_PATH}")
    joblib.dump(clf, MODEL_PATH)
    THRESHOLD_PATH.write_text(str(best_threshold))
    print(f"    Threshold saved: {best_threshold}")

    # ── 10. Final validation score ────────────────────────────────────────────
    pred_best = apply_threshold(val_ids, val_pair_probs, best_threshold)
    final_f05 = macro_f05(val_ids, gt_map, pred_best)
    print(f"\n[10] Final validation macro F_0.5 @ {best_threshold:.2f}: {final_f05:.6f}")

    # Feature importances
    fi = sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda x: -x[1])
    print("\n  Top-10 feature importances:")
    for fname, imp in fi[:10]:
        print(f"    {fname:<25} {imp:.4f}")

    print("\n" + "=" * 70)
    print("Training complete.")
    print("=" * 70)
    return best_threshold, final_f05


if __name__ == "__main__":
    main()
