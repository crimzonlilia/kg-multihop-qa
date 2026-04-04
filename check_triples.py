#!/usr/bin/env python3
"""
Check triples cache - compare old vs new extracted triples
"""

import json
from pathlib import Path
from collections import Counter

TRIPLES_CACHE = "data/cache/triples.json"
SCHEMA_PATH = "src/schemas/relations.json"

def load_json(path):
    with open(path) as f:
        return json.load(f)

def main():
    print("=" * 70)
    print("CHECKING TRIPLES CACHE")
    print("=" * 70)
    
    if not Path(TRIPLES_CACHE).exists():
        print(f"\n❌ Cache not found: {TRIPLES_CACHE}")
        return
    
    # Load triples
    print(f"\n📄 Loading triples from cache...")
    triples_data = load_json(TRIPLES_CACHE)
    
    # Flatten triples from passage-based structure
    all_triples = []
    if isinstance(triples_data, dict) and 'triples' in triples_data:
        all_triples = triples_data['triples']
    elif isinstance(triples_data, list):
        # List of passage objects - extract triples from each
        for passage_obj in triples_data:
            if isinstance(passage_obj, dict) and 'triples' in passage_obj:
                all_triples.extend(passage_obj['triples'])
            elif isinstance(passage_obj, dict):
                # If it's already a triple dict, add it directly
                if 'relation' in passage_obj and 'subject' in passage_obj and 'object' in passage_obj:
                    all_triples.append(passage_obj)
    else:
        print("❌ Cannot parse triples format")
        return
    
    print(f"   Total triples: {len(all_triples)}")
    
    # Load schema
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    
    print(f"\n📋 Schema info:")
    print(f"   Total relations in schema: {len(schema)}")
    
    # NEW relations (expanded)
    new_relations = {'spouse', 'distributed_by', 'directed_by', 'owner_of', 'capital_of'}
    print(f"\n✨ New relations: {new_relations}")
    
    # Analyze triples
    print(f"\n🔍 Analyzing triples...")
    
    relations_count = Counter()
    old_triples = 0
    new_triples = 0
    
    for triple in all_triples:
        rel = triple.get('relation', '').lower()
        relations_count[rel] += 1
        
        if rel in new_relations:
            new_triples += 1
        else:
            old_triples += 1
    
    print(f"\n   OLD relations (standard): {old_triples}")
    print(f"   NEW relations (expanded): {new_triples}")
    print(f"   Ratio: {new_triples}/{len(all_triples)} = {100*new_triples/len(all_triples):.1f}%")
    
    print(f"\n📊 Top 10 relations by frequency:")
    for rel, count in relations_count.most_common(10):
        is_new = " ✨ NEW" if rel in new_relations else ""
        print(f"   {rel:25s}: {count:5d}{is_new}")
    
    print(f"\n   Total unique relations: {len(relations_count)}")
    
    # Show some sample new relation triples
    print(f"\n📝 Sample NEW relation triples:")
    sample_count = 0
    for triple in all_triples:
        if triple.get('relation', '').lower() in new_relations:
            print(f"   {triple.get('subject', '')}")
            print(f"      --[{triple.get('relation', '')}]-->")
            print(f"      {triple.get('object', '')}")
            print()
            sample_count += 1
            if sample_count >= 3:
                break
    
    print("=" * 70)
    print("✓ Analysis complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
