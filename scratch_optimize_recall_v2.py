import duckdb
import pandas as pd
import numpy as np
import sys, time
from sklearn.feature_extraction.text import TfidfVectorizer

sys.stdout.reconfigure(encoding='utf-8')

print("Starting Targeted Recall Optimization Benchmark v2...")
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

# Sample 2,000 S1 with matches
con.execute("""
CREATE TABLE s1_sample AS
SELECT s1.*, gt.matched_entity_ids
FROM s1_all s1
JOIN gt ON s1.entity_id = gt.source1_entity_id
WHERE gt.matched_entity_ids IS NOT NULL AND gt.matched_entity_ids != ''
USING SAMPLE 2000;
""")

con.execute("""
CREATE TABLE s1_feat AS
SELECT 
    entity_id, country, matched_entity_ids,
    lower(regexp_replace(business_name, '[^a-zA-Z0-9]', '', 'g')) AS norm_name,
    lower(regexp_replace(business_address, '[^a-zA-Z0-9]', '', 'g')) AS norm_addr,
    regexp_extract(business_address, '\\d+', 0) AS addr_num,
    lower(substr(regexp_replace(business_name, '[^a-zA-Z0-9]', '', 'g'), 1, 4)) AS prefix4,
    lower(substr(regexp_replace(business_name, '[^a-zA-Z0-9]', '', 'g'), 1, 3)) AS prefix3,
    lower(substr(regexp_replace(business_name, '[^a-zA-Z0-9]', '', 'g'), 4, 3)) AS mid3
FROM s1_sample;

CREATE TABLE pool_feat AS
SELECT 
    entity_id, country,
    lower(regexp_replace(business_name, '[^a-zA-Z0-9]', '', 'g')) AS norm_name,
    lower(regexp_replace(business_address, '[^a-zA-Z0-9]', '', 'g')) AS norm_addr,
    regexp_extract(business_address, '\\d+', 0) AS addr_num,
    lower(substr(regexp_replace(business_name, '[^a-zA-Z0-9]', '', 'g'), 1, 4)) AS prefix4,
    lower(substr(regexp_replace(business_name, '[^a-zA-Z0-9]', '', 'g'), 1, 3)) AS prefix3,
    lower(substr(regexp_replace(business_name, '[^a-zA-Z0-9]', '', 'g'), 4, 3)) AS mid3
FROM s23;
""")

# Rule 1: Exact norm_name
res1 = con.execute("SELECT s1.entity_id AS s1_id, p.entity_id AS cand_id FROM s1_feat s1 JOIN pool_feat p ON s1.country = p.country AND s1.norm_name = p.norm_name WHERE len(s1.norm_name) >= 3").df()

# Rule 2: Exact norm_addr
res2 = con.execute("SELECT s1.entity_id AS s1_id, p.entity_id AS cand_id FROM s1_feat s1 JOIN pool_feat p ON s1.country = p.country AND s1.norm_addr = p.norm_addr WHERE len(s1.norm_addr) >= 10").df()

# Rule 3: Prefix 4 + Addr Num
res3 = con.execute("SELECT s1.entity_id AS s1_id, p.entity_id AS cand_id FROM s1_feat s1 JOIN pool_feat p ON s1.country = p.country AND s1.prefix4 = p.prefix4 AND s1.addr_num = p.addr_num WHERE len(s1.prefix4) >= 3 AND len(s1.addr_num) >= 1").df()

# Rule 4: Prefix 3 + Mid 3
res4 = con.execute("SELECT s1.entity_id AS s1_id, p.entity_id AS cand_id FROM s1_feat s1 JOIN pool_feat p ON s1.country = p.country AND s1.prefix3 = p.prefix3 AND s1.mid3 = p.mid3 WHERE len(s1.prefix3) >= 3 AND len(s1.mid3) >= 3").df()

eval_df = con.execute("SELECT entity_id, matched_entity_ids FROM s1_feat").df()
gt_map = {}
total_gt_matches = 0
for _, row in eval_df.iterrows():
    s1_id = row['entity_id']
    m_set = set(str(row['matched_entity_ids']).split(','))
    gt_map[s1_id] = m_set
    total_gt_matches += len(m_set)

def eval_rule(df, name):
    c_map = df.groupby('s1_id')['cand_id'].apply(set).to_dict()
    ret = sum(len(c_map.get(s1_id, set()).intersection(t_set)) for s1_id, t_set in gt_map.items())
    avg_c = sum(len(s) for s in c_map.values()) / len(gt_map)
    print(f"[{name}] Recall: {ret/total_gt_matches*100:.2f}% ({ret}/{total_gt_matches}), Avg Cands: {avg_c:.1f}")

eval_rule(res4, "Prefix 3 + Mid 3")

combined = pd.concat([res1, res2, res3, res4], ignore_index=True).drop_duplicates()
eval_rule(combined, "Combined Rules 1+2+3+4")
