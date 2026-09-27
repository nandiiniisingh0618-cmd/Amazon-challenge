import pandas as pd
import numpy as np
import sys, time, re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

sys.stdout.reconfigure(encoding='utf-8')

print("Loading ground truth and sample training data...")
t0 = time.time()
gt_df = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t')
gt_matched = gt_df[gt_df['matched_entity_ids'].notna() & (gt_df['matched_entity_ids'] != '')]

# Sample 5,000 S1 with matches
sample_gt_matched = gt_matched.sample(5000, random_state=42)
sample_gt_singletons = gt_df[gt_df['matched_entity_ids'].isna() | (gt_df['matched_entity_ids'] == '')].sample(1000, random_state=42)
sample_gt = pd.concat([sample_gt_matched, sample_gt_singletons], ignore_index=True)

sample_s1_ids = set(sample_gt['source1_entity_id'])

s1_full = pd.read_csv('dataset/train/train_source1.tsv', sep='\t')
s1 = s1_full[s1_full['entity_id'].isin(sample_s1_ids)].copy().reset_index(drop=True)

s2 = pd.read_csv('dataset/train/train_source2.tsv', sep='\t')
s3 = pd.read_csv('dataset/train/train_source3.tsv', sep='\t')
s23 = pd.concat([s2, s3], ignore_index=True)

print(f"Loaded in {time.time()-t0:.2f}s. Sample S1: {len(s1):,}, Pool S23: {len(s23):,}")

# Ground truth dictionary
true_matches = {}
for _, row in sample_gt.iterrows():
    s1_id = row['source1_entity_id']
    m_str = str(row['matched_entity_ids']) if pd.notna(row['matched_entity_ids']) and str(row['matched_entity_ids']) != 'nan' else ''
    true_matches[s1_id] = set(m_str.split(',')) if m_str else set()

total_true_pairs = sum(len(m) for m in true_matches.values())
print(f"Total true matching pairs in sample: {total_true_pairs:,}\n")

# Normalization helper
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

s1['clean_name'] = s1['business_name'].apply(clean_text)
s1['clean_address'] = s1['business_address'].apply(clean_text)
s23['clean_name'] = s23['business_name'].apply(clean_text)
s23['clean_address'] = s23['business_address'].apply(clean_text)

# Run tests by Country
countries = s1['country'].unique()
print("Country distribution in S1 sample:", s1['country'].value_counts().to_dict())

def evaluate_candidates(candidates_dict, name="Strategy"):
    retrieved_true = 0
    total_cands = 0
    for s1_id, cands in candidates_dict.items():
        cand_set = set(cands)
        total_cands += len(cand_set)
        t_set = true_matches.get(s1_id, set())
        if t_set:
            retrieved_true += len(cand_set.intersection(t_set))
    
    recall = retrieved_true / max(total_true_pairs, 1)
    avg_cands = total_cands / max(len(candidates_dict), 1)
    print(f"[{name}] Recall: {recall*100:.2f}% ({retrieved_true}/{total_true_pairs}), Avg candidates/S1: {avg_cands:.1f}")
    return candidates_dict

# Test Strategy A: TF-IDF Name Top-K
print("\n--- Testing Strategy A: TF-IDF Name ---")
for K in [30, 50, 100]:
    cands = {eid: [] for eid in s1['entity_id']}
    for ctry in countries:
        s1_c = s1[s1['country'] == ctry].reset_index(drop=True)
        pool_c = s23[s23['country'] == ctry].reset_index(drop=True)
        if s1_c.empty or pool_c.empty: continue
        
        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, sublinear_tf=True, dtype=np.float32)
        # Fit on pool
        pool_mat = vec.fit_transform(pool_c['clean_name'])
        query_mat = vec.transform(s1_c['clean_name'])
        
        k_val = min(K, pool_mat.shape[0])
        # Compute in chunks
        chunk_sz = 1000
        for i in range(0, query_mat.shape[0], chunk_sz):
            q_chunk = query_mat[i:i+chunk_sz]
            sims = (q_chunk @ pool_mat.T).toarray()
            for row_idx in range(sims.shape[0]):
                s1_id = s1_c.loc[i + row_idx, 'entity_id']
                top_k_indices = np.argpartition(sims[row_idx], -k_val)[-k_val:]
                cands[s1_id].extend(pool_c.loc[top_k_indices, 'entity_id'].tolist())
    evaluate_candidates(cands, f"TF-IDF Name Top-{K}")

# Test Strategy B: TF-IDF Address Top-K
print("\n--- Testing Strategy B: TF-IDF Address ---")
for K in [20, 40]:
    cands = {eid: [] for eid in s1['entity_id']}
    for ctry in countries:
        s1_c = s1[s1['country'] == ctry].reset_index(drop=True)
        pool_c = s23[s23['country'] == ctry].reset_index(drop=True)
        if s1_c.empty or pool_c.empty: continue
        
        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, sublinear_tf=True, dtype=np.float32)
        pool_mat = vec.fit_transform(pool_c['clean_address'])
        query_mat = vec.transform(s1_c['clean_address'])
        
        k_val = min(K, pool_mat.shape[0])
        chunk_sz = 1000
        for i in range(0, query_mat.shape[0], chunk_sz):
            q_chunk = query_mat[i:i+chunk_sz]
            sims = (q_chunk @ pool_mat.T).toarray()
            for row_idx in range(sims.shape[0]):
                s1_id = s1_c.loc[i + row_idx, 'entity_id']
                top_k_indices = np.argpartition(sims[row_idx], -k_val)[-k_val:]
                cands[s1_id].extend(pool_c.loc[top_k_indices, 'entity_id'].tolist())
    evaluate_candidates(cands, f"TF-IDF Address Top-{K}")

