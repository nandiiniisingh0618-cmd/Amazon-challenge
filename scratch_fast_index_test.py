import pandas as pd
import numpy as np
import sys, time, re
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

print("Loading ground truth...")
t0 = time.time()
gt_df = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t')
gt_matched = gt_df[gt_df['matched_entity_ids'].notna() & (gt_df['matched_entity_ids'] != '')]

# Sample 5,000 S1 with matches
sample_gt = gt_matched.sample(5000, random_state=42)
sample_s1_ids = set(sample_gt['source1_entity_id'])

# Ground truth dictionary
true_matches = {}
for _, row in sample_gt.iterrows():
    s1_id = row['source1_entity_id']
    m_str = str(row['matched_entity_ids'])
    true_matches[s1_id] = set(m_str.split(','))

total_true_pairs = sum(len(m) for m in true_matches.values())
print(f"Sample S1 count: 5,000, Total true pairs: {total_true_pairs:,}")

s1_full = pd.read_csv('dataset/train/train_source1.tsv', sep='\t')
s1 = s1_full[s1_full['entity_id'].isin(sample_s1_ids)].copy().reset_index(drop=True)

s2 = pd.read_csv('dataset/train/train_source2.tsv', sep='\t')
s3 = pd.read_csv('dataset/train/train_source3.tsv', sep='\t')
s23 = pd.concat([s2, s3], ignore_index=True)

print(f"Data loaded in {time.time()-t0:.2f}s")

STOPWORDS = {
    'the', 'of', 'and', 'a', 'an', 'in', 'for', 'to', 'at', 'by', 'on', 'with', 'from',
    'pvt', 'private', 'ltd', 'limited', 'inc', 'incorporated', 'corp', 'corporation',
    'co', 'company', 'llc', 'services', 'service', 'group', 'enterprises', 'solutions',
    'india', 'us', 'usa', 'llp', 'plc'
}

def get_tokens(text):
    if not isinstance(text, str): return []
    tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
    return [t for t in tokens if len(t) >= 2 and t not in STOPWORDS]

def get_digits(text):
    if not isinstance(text, str): return []
    return re.findall(r'\d+', text)

# Build Inverted Indexes on Pool (s23) by country
indexes = defaultdict(lambda: {
    'first_tok': defaultdict(list),
    'any_tok': defaultdict(list),
    'num_addr': defaultdict(list),
    'name_3gram': defaultdict(list)
})

print("Building pool index...")
t1 = time.time()

for row in s23.itertuples(index=False):
    eid = row.entity_id
    ctry = row.country
    name_toks = get_tokens(row.business_name)
    addr_toks = get_tokens(row.business_address)
    addr_nums = get_digits(row.business_address)
    
    ctry_idx = indexes[ctry]
    
    # 1. First token index
    if name_toks:
        ctry_idx['first_tok'][name_toks[0]].append(eid)
    
    # 2. Significant tokens index (tokens of length >= 4)
    for tok in name_toks:
        if len(tok) >= 4:
            ctry_idx['any_tok'][tok].append(eid)
            
    # 3. Address numeric + first token index
    if addr_nums and name_toks:
        key = f"{addr_nums[0]}_{name_toks[0][:3]}"
        ctry_idx['num_addr'][key].append(eid)

print(f"Index built in {time.time()-t1:.2f}s")

# Evaluate blocking rules
cands = defaultdict(set)
t2 = time.time()

for row in s1.itertuples(index=False):
    eid = row.entity_id
    ctry = row.country
    name_toks = get_tokens(row.business_name)
    addr_toks = get_tokens(row.business_address)
    addr_nums = get_digits(row.business_address)
    
    ctry_idx = indexes[ctry]
    cand_set = set()
    
    # Rule A: Match on first token
    if name_toks:
        cand_set.update(ctry_idx['first_tok'].get(name_toks[0], []))
        
    # Rule B: Match on any rare name token (len >= 4)
    for tok in name_toks:
        if len(tok) >= 4:
            pool_list = ctry_idx['any_tok'].get(tok, [])
            if len(pool_list) <= 500:  # don't explode on super common words
                cand_set.update(pool_list)
                
    # Rule C: Match on address number + name prefix
    if addr_nums and name_toks:
        key = f"{addr_nums[0]}_{name_toks[0][:3]}"
        cand_set.update(ctry_idx['num_addr'].get(key, []))
        
    cands[eid] = cand_set

print(f"Candidates generated in {time.time()-t2:.2f}s")

retrieved = 0
total_cands = 0
for s1_id, c_set in cands.items():
    total_cands += len(c_set)
    t_set = true_matches.get(s1_id, set())
    if t_set:
        retrieved += len(c_set.intersection(t_set))

recall = retrieved / total_true_pairs
avg_c = total_cands / len(cands)

print(f"=== FAST INDEX BLOCKING RESULTS ===")
print(f"Recall: {recall*100:.2f}% ({retrieved}/{total_true_pairs})")
print(f"Average candidates per S1: {avg_c:.1f}")
print(f"Total candidate pairs evaluated: {total_cands:,}")
