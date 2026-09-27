import duckdb
import pandas as pd
import numpy as np
import sys, time

sys.stdout.reconfigure(encoding='utf-8')

print("Starting Fast DuckDB N-gram & Token Blocking...")
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

# Sample 3,000 S1 with matches
con.execute("""
CREATE TABLE s1_sample AS
SELECT s1.*, gt.matched_entity_ids
FROM s1_all s1
JOIN gt ON s1.entity_id = gt.source1_entity_id
WHERE gt.matched_entity_ids IS NOT NULL AND gt.matched_entity_ids != ''
USING SAMPLE 3000;
""")

print("Preparing token and prefix features...")
con.execute("""
CREATE TABLE s1_tokens AS
SELECT 
    entity_id, country, matched_entity_ids,
    lower(regexp_replace(business_name, '[^a-zA-Z0-9]', '', 'g')) AS norm_name,
    lower(regexp_replace(business_address, '[^a-zA-Z0-9]', '', 'g')) AS norm_addr,
    regexp_extract(business_address, '\\d+', 0) AS addr_num,
    lower(substr(regexp_replace(business_name, '[^a-zA-Z0-9]', '', 'g'), 1, 4)) AS prefix4,
    lower(substr(regexp_replace(business_name, '[^a-zA-Z0-9]', '', 'g'), 1, 3)) AS prefix3,
    lower(substr(regexp_replace(business_address, '[^a-zA-Z0-9]', '', 'g'), 1, 6)) AS addr_pref6
FROM s1_sample;

CREATE TABLE pool_tokens AS
SELECT 
    entity_id, country,
    lower(regexp_replace(business_name, '[^a-zA-Z0-9]', '', 'g')) AS norm_name,
    lower(regexp_replace(business_address, '[^a-zA-Z0-9]', '', 'g')) AS norm_addr,
    regexp_extract(business_address, '\\d+', 0) AS addr_num,
    lower(substr(regexp_replace(business_name, '[^a-zA-Z0-9]', '', 'g'), 1, 4)) AS prefix4,
    lower(substr(regexp_replace(business_name, '[^a-zA-Z0-9]', '', 'g'), 1, 3)) AS prefix3,
    lower(substr(regexp_replace(business_address, '[^a-zA-Z0-9]', '', 'g'), 1, 6)) AS addr_pref6
FROM s23;
""")

print("Executing fast multi-key blocking joins...")
t1 = time.time()

cands_df = con.execute("""
WITH matches AS (
    -- Rule 1: Exact norm_name match
    SELECT s1.entity_id AS s1_id, p.entity_id AS cand_id, 'exact_name' AS rule
    FROM s1_tokens s1 JOIN pool_tokens p ON s1.country = p.country AND s1.norm_name = p.norm_name
    WHERE len(s1.norm_name) >= 3

    UNION ALL

    -- Rule 2: Exact norm_addr match
    SELECT s1.entity_id AS s1_id, p.entity_id AS cand_id, 'exact_addr' AS rule
    FROM s1_tokens s1 JOIN pool_tokens p ON s1.country = p.country AND s1.norm_addr = p.norm_addr
    WHERE len(s1.norm_addr) >= 10

    UNION ALL

    -- Rule 3: Prefix 4 + Addr Num match
    SELECT s1.entity_id AS s1_id, p.entity_id AS cand_id, 'pref4_num' AS rule
    FROM s1_tokens s1 JOIN pool_tokens p ON s1.country = p.country AND s1.prefix4 = p.prefix4 AND s1.addr_num = p.addr_num
    WHERE len(s1.prefix4) >= 3 AND len(s1.addr_num) >= 1

    UNION ALL

    -- Rule 4: Prefix 3 + Address Prefix 6 match
    SELECT s1.entity_id AS s1_id, p.entity_id AS cand_id, 'pref3_addr6' AS rule
    FROM s1_tokens s1 JOIN pool_tokens p ON s1.country = p.country AND s1.prefix3 = p.prefix3 AND s1.addr_pref6 = p.addr_pref6
    WHERE len(s1.prefix3) >= 3 AND len(s1.addr_pref6) >= 5
)
SELECT s1_id, list(DISTINCT cand_id) AS cand_list, count(DISTINCT cand_id) AS n_cands
FROM matches
GROUP BY s1_id;
""").df()

print(f"DuckDB blocking finished in {time.time()-t1:.2f}s!")

# Evaluate Ground Truth Recall
eval_df = con.execute("SELECT entity_id, matched_entity_ids FROM s1_tokens").df()
gt_map = {}
total_gt_matches = 0

for _, row in eval_df.iterrows():
    s1_id = row['entity_id']
    m_set = set(str(row['matched_entity_ids']).split(','))
    gt_map[s1_id] = m_set
    total_gt_matches += len(m_set)

cand_map = dict(zip(cands_df['s1_id'], cands_df['cand_list']))

retrieved = 0
total_cands = 0
for s1_id, true_set in gt_map.items():
    cand_list = cand_map.get(s1_id, [])
    cand_set = set(cand_list)
    total_cands += len(cand_set)
    if true_set:
        retrieved += len(cand_set.intersection(true_set))

recall = retrieved / max(total_gt_matches, 1)
avg_c = total_cands / max(len(gt_map), 1)

print(f"\n==========================================")
print(f"DUCKDB DUAL-PREFIX BLOCKING (3,000 S1 SAMPLE)")
print(f"==========================================")
print(f"Recall: {recall*100:.2f}% ({retrieved}/{total_gt_matches})")
print(f"Average candidates per S1: {avg_c:.1f}")
print(f"Total candidate pairs: {total_cands:,}")
print(f"Execution Time: {time.time()-t1:.2f} seconds")
print(f"==========================================")
