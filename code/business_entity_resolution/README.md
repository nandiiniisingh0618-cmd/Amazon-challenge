# Amazon Business Entity Resolution Challenge

## Overview

This project solves the **Amazon Business Entity Resolution** challenge: given business records from three sources, identify which Source 2 and Source 3 records refer to the same real-world business as each Source 1 entity.

**Scoring metric**: Macro-averaged F_0.5 (precision-heavy — false merges penalised more than missed matches).

---

## Folder Structure

```
amazon-entity-resolution/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── output/
│   ├── candidate_pairs.tsv       ← generated
│   └── matching_results.tsv      ← generated
└── code/
    └── business_entity_resolution/
        ├── README.md
        ├── requirements.txt
        └── src/
            ├── preprocess.py
            ├── block.py
            ├── features.py
            ├── train.py
            ├── predict.py
            └── run_all.py
```

---

## Prerequisites

- Python ≥ 3.10
- All packages listed in `requirements.txt`

---

## Installation

```bash
pip install -r code/business_entity_resolution/requirements.txt
```

---

## Reproducing the Output Files

Run from the **project root** (the `amazon-entity-resolution/` directory):

### Option A — End-to-end (train + predict):

```bash
python code/business_entity_resolution/src/run_all.py
```

### Option B — Step by step:

**Step 1: Train the model**
```bash
python code/business_entity_resolution/src/train.py
```
This produces:
- `code/business_entity_resolution/src/model.pkl`
- `code/business_entity_resolution/src/threshold.txt`

**Step 2: Generate predictions**
```bash
python code/business_entity_resolution/src/predict.py
```
This produces:
- `output/candidate_pairs.tsv`
- `output/matching_results.tsv`

---

## Pipeline Description

### 1. Preprocessing (`preprocess.py`)
- Lowercases text, strips punctuation, expands business abbreviations (e.g. `ltd → limited`, `corp → corporation`) and address abbreviations (e.g. `rd → road`, `ave → avenue`).

### 2. Blocking (`block.py`)
Three blocking strategies are unioned per Source 1 entity, all within the same country bucket:

| Strategy | Method | Top-K |
|---|---|---|
| A — Name TF-IDF | Char n-gram (2–4) TF-IDF cosine similarity on `clean_name` | 30 |
| B — Address TF-IDF | Char n-gram (2–4) TF-IDF cosine similarity on `clean_address` | 20 |
| C — Prefix | Exact match on first token of `clean_name` | all |

### 3. Features (`features.py`)
14 features per candidate pair:

| Feature | Description |
|---|---|
| `name_lev_ratio` | Levenshtein ratio on clean names |
| `name_token_sort` | Token-sort ratio on clean names |
| `name_partial` | Partial ratio on clean names |
| `name_jaro` | WRatio on clean names |
| `name_jaccard` | Jaccard similarity on name word sets |
| `addr_lev_ratio` | Levenshtein ratio on clean addresses |
| `addr_token_sort` | Token-sort ratio on clean addresses |
| `addr_partial` | Partial ratio on clean addresses |
| `addr_jaccard` | Jaccard similarity on address word sets |
| `country_match` | 1 if countries identical, else 0 |
| `name_len_ratio` | Length ratio of clean names |
| `addr_len_ratio` | Length ratio of clean addresses |
| `max_name_addr` | max(name_lev_ratio, addr_lev_ratio) |
| `shared_name_tokens` | Count of shared name tokens |

### 4. Classification (`train.py`)
- **Model**: LightGBM classifier (500 estimators, lr=0.05, balanced class weights)
- **Validation**: 20% of S1 entities held out (seed 42)
- **Threshold**: swept from 0.30 to 0.90 in 0.05 steps to maximise macro F_0.5

### 5. Prediction (`predict.py`)
- Applies trained model + optimal threshold to test data
- Writes `output/candidate_pairs.tsv` and `output/matching_results.tsv`

---

## Output Format

Both output files are tab-separated with two columns:

**`output/matching_results.tsv`**
```
source1_entity_id\tmatched_entity_ids
S1-001\tS2-042,S3-017
S1-002\t
```

**`output/candidate_pairs.tsv`**
```
source1_entity_id\tcandidate_entity_ids
S1-001\tS2-042,S2-055,S3-017,S3-099
S1-002\t
```

- Every Source 1 entity has exactly one row in each file.
- Singletons (no match) have an empty `matched_entity_ids` / `candidate_entity_ids`.
- Every ID in `matching_results.tsv` also appears in `candidate_pairs.tsv` for the same S1 entity.

---

## License

MIT License. The LightGBM model is MIT licensed; all other dependencies are MIT or Apache 2.0.
