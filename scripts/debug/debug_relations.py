#!/usr/bin/env python3
import json
from collections import Counter

with open('data/cache/triples.json') as f:
    data = json.load(f)

# Analyze the structure
print(f"Number of entries: {len(data)}")
print(f"Type of data: {type(data)}")

if isinstance(data, list) and len(data) > 0:
    print(f"\nFirst entry type: {type(data[0])}")
    if isinstance(data[0], dict):
        print(f"First entry keys: {data[0].keys()}")

# Flatten triples
all_triples = []
for i, item in enumerate(data):
    if isinstance(item, dict):
        if 'triples' in item:
            all_triples.extend(item['triples'])
    elif isinstance(item, list):
        # Maybe it's a flat list?
        all_triples.extend(item)
    else:
        print(f"Item {i}: {type(item)}")

print(f'\nTotal triples: {len(all_triples)}')

if all_triples:
    print(f'First triple: {all_triples[0]}')
    print(f'First triple type: {type(all_triples[0])}')
    
    # Check for empty relations
    empty_count = sum(1 for t in all_triples if (isinstance(t, dict) and not t.get('relation', '').strip()) or (isinstance(t, (list, tuple)) and len(t) < 2))
    print(f'Triples with empty/missing relation: {empty_count}')
    
    # Count unique relations
    if isinstance(all_triples[0], dict):
        rel_counter = Counter(t.get('relation', 'MISSING') for t in all_triples if isinstance(t, dict))
    else:
        rel_counter = Counter(t[1] if len(t) > 1 else 'MISSING' for t in all_triples if isinstance(t, (list, tuple)))
    
    print(f'\nUnique relations: {len(rel_counter)}')
    print(f'\nTop 20 relations:')
    for rel, count in rel_counter.most_common(20):
        print(f'  "{rel}": {count}')
