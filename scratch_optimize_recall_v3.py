import duckdb
import pandas as pd
import numpy as np
import sys, time
from sklearn.feature_extraction.text import TfidfVectorizer

sys.stdout.reconfigure(encoding='utf-8')

print("Starting Targeted Recall Optimization Benchmark v3...")
t0 = time.time()

con = duckdb.connect(database=':memory:')

# Read TSVs into DuckDB
con.execute("""
CREATE TABLE gt AS SELECT source1_entity_id, matched_entity_ids FROM read_csv('dataset/train/train_ground_truth.tsv', delim='\t', header=True);
CREATE TABLE s1_all AS SELECT entity_id, business_name, business_address, country FROM read_csv('dataset/train/train_source1.tsv', delim='\t', header=True);
CREATE TABLE s23 AS 
SELECT entity_id, business_name, business_address, country FROM read_csv('dataset/train/train_source2.tsv', delim='\t', header=True)
UNION ALL
SELECT entity_id, business_name, business_address, country FROM read_csv('dataset/train/train_source3.tsv', delim='\t', header=True);
""")

# Sample 1,500 S1 with matches
con.execute("""
CREATE TABLE s1_sample AS
SELECT s1.*, gt.matched_entity_ids
FROM s1_all s1
JOIN gt ON s1.entity_id = gt.source1_entity_id
WHERE gt.matched_entity_ids IS NOT NULL AND gt.matched_entity_ids != ''
USING SAMPLE 1500;
""")

con.execute("""
CREATE TABLE s1_feat AS
SELECT 
    entity_id, country, matched_entity_ids, business_name, business_address,
    lower(regexp_replace(business_name, '[^a-zA-Z0-9]', '', 'g')) AS norm_name,
    lower(regexp_replace(business_address, '[^a-zA-Z0-9]', '', 'g')) AS norm_addr,
    regexp_extract(business_address, '\\d+', 0) AS addr_num,
    lower(substr(regexp_replace(business_name, '[^a-zA-Z0-9]', '', 'g'), 1, 4)) AS prefix4
FROM s1_sample;

CREATE TABLE pool_feat AS
SELECT 
    entity_id, country, business_name, business_address,
    lower(regexp_replace(business_name, '[^a-zA-Z0-9]', '', 'g')) AS norm_name,
    lower(regexp_replace(business_address, '[^a-zA-Z0-9]', '', 'g')) AS norm_addr,
    regexp_extract(business_address, '\\d+', 0) AS addr_num,
    lower(substr(regexp_replace(business_name, '[^a-zA-Z0-9]', '', 'g'), 1, 4)) AS prefix4
FROM s23;
""")

# Rule 1: Exact norm_name
res1 = con.execute("SELECT s1.entity_id AS s1_id, p.entity_id AS cand_id FROM s1_feat s1 JOIN pool_feat p ON s1.country = p.country AND s1.norm_name = p.norm_name WHERE len(s1.norm_name) >= 3").df()

# Rule 2: Exact norm_addr
res2 = con.execute("SELECT s1.entity_id AS s1_id, p.entity_id AS cand_id FROM s1_feat s1 JOIN pool_feat p ON s1.country = p.country AND s1.norm_addr = p.norm_addr WHERE len(s1.norm_addr) >= 10").df()

# Rule 3: Prefix 4 + Addr Num
res3 = con.execute("SELECT s1.entity_id AS s1_id, p.entity_id AS cand_id FROM s1_feat s1 JOIN pool_feat p ON s1.country = p.country AND s1.prefix4 = p.prefix4 AND s1.addr_num = p.addr_num WHERE len(s1.prefix4) >= 3 AND len(s1.addr_num) >= 1").df()

# Rule 4: TF-IDF char n-gram similarity (Top-30) computed per country bucket
print("Computing Rule 4: TF-IDF Top-30 Name & Address Similarity...")
s1_df = con.execute("SELECT entity_id, country, business_name, business_address FROM s1_feat").df()
pool_df = con.execute("SELECT entity_id, country, business_name, business_address FROM pool_feat").df()

tfidf_pairs = []
for ctry in s1_df['country'].unique():
    s1_sub = s1_df[s1_df['country'] == ctry].reset_index(drop=True)
    p_sub = pool_df[pool_df['country'] == ctry].reset_index(drop=True)
    if s1_sub.empty or p_sub.empty: continue
    
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=5, max_features=25000, dtype=np.float32)
    p_mat = vec.fit_transform(p_sub['business_name'].fillna(''))
    q_mat = vec.transform(s1_sub['business_name'].fillna(''))
    
    K = 30
    k_val = min(K, p_mat.shape[0])
    chunk_sz = 1000
    for i in range(0, q_mat.shape[0], chunk_sz):
        sims = (q_mat[i:i+chunk_sz] @ p_mat.T).toarray()
        for r_idx in range(sims.shape[0]):
            s1_id = s1_sub.loc[i + r_idx, 'entity_id']
            top_k = np.argpartition(sims[r_idx], -k_val)[-k_val:]
            for idx in top_k:
                if sims[r_idx, idx] > 0.3:  # minimum TF-IDF similarity threshold
                    tfidf_pairs.append({'s1_id': s1_id, 'cand_id': p_sub.loc[idx, 'entity_id']})

res4 = pd.DataFrame(tfidf_pairs)

eval_df = con.execute("SELECT entity_id, matched_entity_ids FROM s1_feat").df()
gt_map = {}
total_gt_matches = 0
for _, row in eval_df.iterrows():
    s1_id = row['entity_id']
    m_set = set(str(row['matched_entity_ids']).split(','))
    gt_map[s1_id] = m_set
    total_gt_matches += len(m_set)

def eval_rule(df, name):
    if df.empty: return
    c_map = df.groupby('s1_id')['cand_id'].apply(set).to_dict()
    ret = sum(len(c_map.get(s1_id, set()).intersection(t_set)) for s1_id, t_set in gt_map.items())
    avg_c = sum(len(s) for s in c_map.values()) / len(gt_map)
    print(f"[{name}] Recall: {ret/total_gt_matches*100:.2f}% ({ret}/{total_gt_matches}), Avg Cands: {avg_c:.1f}")

eval_rule(res1, "Rule 1: Exact Norm Name")
eval_rule(res2, "Rule 2: Exact Norm Addr")
eval_rule(res3, "Rule 3: Prefix 4 + Addr Num")
eval_rule(res4, "Rule 4: TF-IDF Name Top-30 (>0.3)")

combined = pd.concat([res1, res2, res3, res4], ignore_index=True).drop_duplicates()
eval_rule(combined, "Combined Rules 1+2+3+4")
