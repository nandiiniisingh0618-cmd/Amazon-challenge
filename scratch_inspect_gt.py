import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t')
s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t').set_index('entity_id')
s2 = pd.read_csv('dataset/train/train_source2.tsv', sep='\t').set_index('entity_id')
s3 = pd.read_csv('dataset/train/train_source3.tsv', sep='\t').set_index('entity_id')

non_empty = gt[gt['matched_entity_ids'].notna() & (gt['matched_entity_ids'] != '')]
sample = non_empty.sample(15, random_state=42)

for _, row in sample.iterrows():
    s1_id = row['source1_entity_id']
    m_ids = row['matched_entity_ids'].split(',')
    
    s1_row = s1.loc[s1_id]
    print(f"=== S1 ({s1_id}) ===")
    print(f"  Name: {s1_row['business_name']}")
    print(f"  Addr: {s1_row['business_address']}")
    print(f"  Ctry: {s1_row['country']}")
    print("--- MATCHES ---")
    for m in m_ids:
        if m.startswith('S2-'):
            m_row = s2.loc[m]
        else:
            m_row = s3.loc[m]
        print(f"  [{m[:2]}] ({m})")
        print(f"     Name: {m_row['business_name']}")
        print(f"     Addr: {m_row['business_address']}")
        print(f"     Ctry: {m_row['country']}")
    print("="*70)
