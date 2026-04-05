"""
Compare two triples versions to see what changed
- Count total triples, entities
- Find new/removed triples
- Analyze subject/object changes
"""

import json
from pathlib import Path
from collections import Counter
import sys

def load_triples_cache(file_path):
    """Load passages with triples"""
    with open(file_path, 'r', encoding='utf-8') as f:
        passages = json.load(f)
    
    all_triples = set()
    total_entities = 0
    
    for passage in passages:
        triples = passage.get('triples', [])
        for triple in triples:
            key = (
                triple.get('subject', '').lower(),
                triple.get('relation', '').lower(),
                triple.get('object', '').lower()
            )
            all_triples.add(key)
        
        entities = passage.get('entities', [])
        total_entities += len(entities)
    
    return all_triples, len(passages), total_entities

def find_timestamped_versions():
    """Find all timestamped triples files"""
    cache_dir = Path("data/cache")
    versioned = sorted(cache_dir.glob("triples_*.json"))
    return versioned

def compare_versions(file1, file2):
    """Compare two triples versions"""
    print(f"\n{'='*80}")
    print(f"COMPARING TRIPLES VERSIONS")
    print(f"{'='*80}\n")
    
    print(f"Version 1: {file1.name}")
    triples1, passages1, entities1 = load_triples_cache(file1)
    print(f"  Passages: {passages1}")
    print(f"  Unique triples: {len(triples1)}")
    print(f"  Total entities: {entities1}")
    
    print(f"\nVersion 2: {file2.name}")
    triples2, passages2, entities2 = load_triples_cache(file2)
    print(f"  Passages: {passages2}")
    print(f"  Unique triples: {len(triples2)}")
    print(f"  Total entities: {entities2}")
    
    # Differences
    new_triples = triples2 - triples1
    removed_triples = triples1 - triples2
    shared_triples = triples1 & triples2
    
    print(f"\n{'─'*80}")
    print(f"CHANGES")
    print(f"{'─'*80}\n")
    
    print(f"Shared triples:    {len(shared_triples)}")
    print(f"New triples:       {len(new_triples):5d} (+{100*len(new_triples)/len(triples1) if triples1 else 0:.1f}%)")
    print(f"Removed triples:   {len(removed_triples):5d} (-{100*len(removed_triples)/len(triples1) if triples1 else 0:.1f}%)")
    
    net_change = len(new_triples) - len(removed_triples)
    print(f"\nNet change: {net_change:+d} triples ({100*net_change/len(triples1) if triples1 else 0:+.1f}%)")
    
    # Sample new triples
    if new_triples:
        print(f"\n{'─'*80}")
        print(f"SAMPLE NEW TRIPLES (first 10)")
        print(f"{'─'*80}\n")
        
        for i, (s, p, o) in enumerate(sorted(new_triples)[:10], 1):
            print(f"{i}. {s:30s} → {p:20s} → {o}")
    
    # Sample removed triples
    if removed_triples:
        print(f"\n{'─'*80}")
        print(f"SAMPLE REMOVED TRIPLES (first 10)")
        print(f"{'─'*80}\n")
        
        for i, (s, p, o) in enumerate(sorted(removed_triples)[:10], 1):
            print(f"{i}. {s:30s} → {p:20s} → {o}")

if __name__ == "__main__":
    versions = find_timestamped_versions()
    
    if len(versions) < 2:
        if len(versions) == 0:
            print("❌ No timestamped triples versions found!")
        else:
            print(f"ℹ️  Only 1 version found: {versions[0].name}")
            print("Run rebuild_graph.py again to create a second version for comparison")
        sys.exit(1)
    
    # Compare latest two versions
    v2_path = versions[-1]
    v1_path = versions[-2]
    
    compare_versions(v1_path, v2_path)
