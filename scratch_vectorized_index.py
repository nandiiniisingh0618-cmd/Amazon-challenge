import pandas as pd
import numpy as np
import sys, time, re
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

print("Loading data...")
t0 = time.time()
gt_df = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t')
gt_matched = gt_df[gt_df['matched_entity_ids'].notna() & (gt_df['matched_entity_ids'] != '')]

sample_gt = gt_matched.sample(5000, random_state=42)
sample_s1_ids = set(sample_gt['source1_entity_id'])

true_matches = {}
for row in sample_gt.itertuples(index=False):
    s1_id = row.source1_entity_id
    m_str = str(row.matched_entity_ids)
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

# Build Inverted Indexes using fast list iteration over arrays
indexes = defaultdict(lambda: {
    'first_tok': defaultdict(list),
    'any_tok': defaultdict(list),
    'num_addr': defaultdict(list)
})

print("Building pool index via raw array iteration...")
t1 = time.time()

eids = s23['entity_id'].values
ctrys = s23['country'].values
names = s23['business_name'].values
addrs = s23['business_address'].values

for i in range(len(eids)):
    eid = eids[i]
    ctry = ctrys[i]
    n_str = names[i]
    a_str = addrs[i]
    
    name_toks = get_tokens(n_str)
    addr_nums = get_digits(a_str)
    
    ctry_idx = indexes[ctry]
    
    # 1. First token
    if name_toks:
        ctry_idx['first_tok'][name_toks[0]].append(eid)
    
    # 2. Significant tokens
    for tok in name_toks:
        if len(tok) >= 4:
            ctry_idx['any_tok'][tok].append(eid)
            
    # 3. Address numeric + name prefix
    if addr_nums and name_toks:
        key = f"{addr_nums[0]}_{name_toks[0][:3]}"
        ctry_idx['num_addr'][key].append(eid)

print(f"Index built in {time.time()-t1:.2f}s")

# Evaluate blocking rules
cands = defaultdict(set)
t2 = time.time()

s1_eids = s1['entity_id'].values
s1_ctrys = s1['country'].values
s1_names = s1['business_name'].values
s1_addrs = s1['business_address'].values

for i in range(len(s1_eids)):
    eid = s1_eids[i]
    ctry = s1_ctrys[i]
    name_toks = get_tokens(s1_names[i])
    addr_nums = get_digits(s1_addrs[i])
    
    ctry_idx = indexes[ctry]
    cand_set = set()
    
    if name_toks:
        cand_set.update(ctry_idx['first_tok'].get(name_toks[0], []))
        
    for tok in name_toks:
        if len(tok) >= 4:
            pool_list = ctry_idx['any_tok'].get(tok, [])
            if len(pool_list) <= 500:
                cand_set.update(pool_list)
                
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

print(f"\n=== FAST VECTORIZED INDEX BLOCKING RESULTS ===")
print(f"Recall: {recall*100:.2f}% ({retrieved}/{total_true_pairs})")
print(f"Average candidates per S1: {avg_c:.1f}")
print(f"Total candidate pairs: {total_cands:,}")
