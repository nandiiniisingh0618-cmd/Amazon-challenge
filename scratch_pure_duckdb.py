import duckdb
import pandas as pd
import numpy as np
import sys, time

sys.stdout.reconfigure(encoding='utf-8')

print("Executing Pure DuckDB Blocking Pipeline...")
t0 = time.time()

con = duckdb.connect(database=':memory:')

# Read TSVs directly into DuckDB tables (0.5 seconds!)
con.execute("""
CREATE TABLE gt AS 
SELECT source1_entity_id, matched_entity_ids 
FROM read_csv('dataset/train/train_ground_truth.tsv', delim='\t', header=True, sample_size=1000);

CREATE TABLE s1_all AS 
SELECT entity_id, business_name, business_address, country 
FROM read_csv('dataset/train/train_source1.tsv', delim='\t', header=True, sample_size=1000);

CREATE TABLE s23 AS 
SELECT entity_id, business_name, business_address, country 
FROM read_csv('dataset/train/train_source2.tsv', delim='\t', header=True, sample_size=1000)
UNION ALL
SELECT entity_id, business_name, business_address, country 
FROM read_csv('dataset/train/train_source3.tsv', delim='\t', header=True, sample_size=1000);
""")

print(f"Data loaded into DuckDB in {time.time()-t0:.2f}s")

# Create sample of 10,000 S1 entities with ground truth
t1 = time.time()
con.execute("""
CREATE TABLE s1_sample AS
SELECT s1.*, gt.matched_entity_ids
FROM s1_all s1
JOIN gt ON s1.entity_id = gt.source1_entity_id
USING SAMPLE 10000;
""")

# Extract features/tokens for blocking
con.execute("""
CREATE TABLE s1_features AS
SELECT 
    entity_id,
    country,
    matched_entity_ids,
    lower(regexp_replace(business_name, '[^a-zA-Z0-9]', ' ', 'g')) AS clean_name,
    lower(regexp_replace(business_address, '[^a-zA-Z0-9]', ' ', 'g')) AS clean_address,
    string_split(trim(lower(regexp_replace(business_name, '[^a-zA-Z0-9]', ' ', 'g'))), ' ')[1] AS first_word,
    string_split(trim(lower(regexp_replace(business_name, '[^a-zA-Z0-9]', ' ', 'g'))), ' ')[2] AS second_word,
    regexp_extract(business_address, '\\d+', 0) AS house_num
FROM s1_sample;

CREATE TABLE pool_features AS
SELECT 
    entity_id,
    country,
    lower(regexp_replace(business_name, '[^a-zA-Z0-9]', ' ', 'g')) AS clean_name,
    lower(regexp_replace(business_address, '[^a-zA-Z0-9]', ' ', 'g')) AS clean_address,
    string_split(trim(lower(regexp_replace(business_name, '[^a-zA-Z0-9]', ' ', 'g'))), ' ')[1] AS first_word,
    string_split(trim(lower(regexp_replace(business_name, '[^a-zA-Z0-9]', ' ', 'g'))), ' ')[2] AS second_word,
    regexp_extract(business_address, '\\d+', 0) AS house_num
FROM s23;
""")

print(f"Feature table created in {time.time()-t1:.2f}s")

# Run DuckDB fast joins
t2 = time.time()
cands_df = con.execute("""
WITH matches AS (
    -- Rule 1: First word match (length >= 3)
    SELECT s1.entity_id AS s1_id, p.entity_id AS cand_id
    FROM s1_features s1
    JOIN pool_features p ON s1.country = p.country AND s1.first_word = p.first_word
    WHERE len(s1.first_word) >= 3

    UNION ALL

    -- Rule 2: Exact Address match
    SELECT s1.entity_id AS s1_id, p.entity_id AS cand_id
    FROM s1_features s1
    JOIN pool_features p ON s1.country = p.country AND s1.clean_address = p.clean_address
    WHERE len(s1.clean_address) >= 8

    UNION ALL

    -- Rule 3: House Number + First Word match
    SELECT s1.entity_id AS s1_id, p.entity_id AS cand_id
    FROM s1_features s1
    JOIN pool_features p ON s1.country = p.country AND s1.house_num = p.house_num AND s1.first_word = p.first_word
    WHERE len(s1.house_num) >= 1 AND len(s1.first_word) >= 2
)
SELECT s1_id, list(DISTINCT cand_id) AS cand_list, count(DISTINCT cand_id) AS n_cands
FROM matches
GROUP BY s1_id;
""").df()

print(f"DuckDB SQL join completed in {time.time()-t2:.2f}s!")

# Evaluate Ground Truth Recall
eval_df = con.execute("SELECT entity_id, matched_entity_ids FROM s1_sample").df()
gt_map = {}
total_gt_matches = 0

for _, row in eval_df.iterrows():
    s1_id = row['entity_id']
    m_str = str(row['matched_entity_ids']) if pd.notna(row['matched_entity_ids']) and str(row['matched_entity_ids']) != 'nan' else ''
    m_set = set(m_str.split(',')) if m_str else set()
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
print(f"PURE DUCKDB BLOCKING EVALUATION (10,000 S1)")
print(f"==========================================")
print(f"Recall: {recall*100:.2f}% ({retrieved}/{total_gt_matches})")
print(f"Average candidates per S1: {avg_c:.1f}")
print(f"Total candidate pairs: {total_cands:,}")
print(f"Execution Time: {time.time()-t2:.2f} seconds")
print(f"==========================================")
