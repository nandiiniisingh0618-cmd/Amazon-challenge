import duckdb
import pandas as pd
import numpy as np
import sys, time

sys.stdout.reconfigure(encoding='utf-8')

print("Loading ground truth & datasets...")
t0 = time.time()

gt_df = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t')
gt_matched = gt_df[gt_df['matched_entity_ids'].notna() & (gt_df['matched_entity_ids'] != '')]

# Sample 5,000 S1 with matches
sample_gt = gt_matched.sample(5000, random_state=42)
sample_s1_ids = set(sample_gt['source1_entity_id'])

true_matches = {}
for row in sample_gt.itertuples(index=False):
    s1_id = row.source1_entity_id
    m_str = str(row.matched_entity_ids)
    true_matches[s1_id] = set(m_str.split(','))

total_true_pairs = sum(len(m) for m in true_matches.values())
print(f"Sample S1 count: 5,000, Total true pairs in GT: {total_true_pairs:,}")

s1_full = pd.read_csv('dataset/train/train_source1.tsv', sep='\t')
s1 = s1_full[s1_full['entity_id'].isin(sample_s1_ids)].copy().reset_index(drop=True)

s2 = pd.read_csv('dataset/train/train_source2.tsv', sep='\t')
s3 = pd.read_csv('dataset/train/train_source3.tsv', sep='\t')
s23 = pd.concat([s2, s3], ignore_index=True)

print(f"Data loaded in {time.time()-t0:.2f}s")

# Connect to DuckDB in memory
con = duckdb.connect(database=':memory:')

print("Registering dataframes & executing SQL transformations in DuckDB...")
t1 = time.time()

con.register('s1_raw', s1)
con.register('pool_raw', s23)

# SQL data cleaning & key extraction
con.execute("""
CREATE TABLE s1_tbl AS
SELECT 
    entity_id,
    country,
    lower(regexp_replace(business_name, '[^a-zA-Z0-9]', ' ', 'g')) AS clean_name,
    lower(regexp_replace(business_address, '[^a-zA-Z0-9]', ' ', 'g')) AS clean_address,
    regexp_extract(lower(business_name), '[a-z0-9]{3,}', 1) AS first_word,
    regexp_extract(lower(business_name), '[a-z0-9]{3,}', 2) AS second_word,
    regexp_extract(business_address, '\\d+', 0) AS house_num
FROM s1_raw;

CREATE TABLE pool_tbl AS
SELECT 
    entity_id,
    country,
    lower(regexp_replace(business_name, '[^a-zA-Z0-9]', ' ', 'g')) AS clean_name,
    lower(regexp_replace(business_address, '[^a-zA-Z0-9]', ' ', 'g')) AS clean_address,
    regexp_extract(lower(business_name), '[a-z0-9]{3,}', 1) AS first_word,
    regexp_extract(lower(business_name), '[a-z0-9]{3,}', 2) AS second_word,
    regexp_extract(business_address, '\\d+', 0) AS house_num
FROM pool_raw;
""")

print(f"DuckDB tables created in {time.time()-t1:.2f}s")

# Run Blocking Queries
t2 = time.time()
cands_df = con.execute("""
WITH matches AS (
    -- Rule 1: First word match (when first_word is >= 3 chars)
    SELECT s1.entity_id AS s1_id, p.entity_id AS cand_id
    FROM s1_tbl s1
    JOIN pool_tbl p ON s1.country = p.country AND s1.first_word = p.first_word
    WHERE len(s1.first_word) >= 3

    UNION ALL

    -- Rule 2: Exact Address match (non-empty address)
    SELECT s1.entity_id AS s1_id, p.entity_id AS cand_id
    FROM s1_tbl s1
    JOIN pool_tbl p ON s1.country = p.country AND s1.clean_address = p.clean_address
    WHERE len(s1.clean_address) >= 8

    UNION ALL

    -- Rule 3: House Number + First Word match
    SELECT s1.entity_id AS s1_id, p.entity_id AS cand_id
    FROM s1_tbl s1
    JOIN pool_tbl p ON s1.country = p.country AND s1.house_num = p.house_num AND s1.first_word = p.first_word
    WHERE len(s1.house_num) >= 1 AND len(s1.first_word) >= 2
)
SELECT s1_id, list(DISTINCT cand_id) AS cand_list, count(DISTINCT cand_id) AS n_cands
FROM matches
GROUP BY s1_id;
""").df()

print(f"DuckDB candidate generation completed in {time.time()-t2:.2f}s!")

# Evaluate against Ground Truth
cand_map = dict(zip(cands_df['s1_id'], cands_df['cand_list']))

retrieved = 0
total_cands = 0
for s1_id, true_set in true_matches.items():
    cand_list = cand_map.get(s1_id, [])
    cand_set = set(cand_list)
    total_cands += len(cand_set)
    if true_set:
        retrieved += len(cand_set.intersection(true_set))

recall = retrieved / total_true_pairs
avg_c = total_cands / max(len(true_matches), 1)

print(f"\n==========================================")
print(f"DUCKDB BLOCKING RESULTS")
print(f"==========================================")
print(f"Recall: {recall*100:.2f}% ({retrieved}/{total_true_pairs})")
print(f"Average candidates per S1: {avg_c:.1f}")
print(f"Total candidate pairs: {total_cands:,}")
print(f"Execution Time: {time.time()-t2:.2f} seconds!")
print(f"==========================================")
