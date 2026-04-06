#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.retrieval.passage_ranking import load_passage_data, build_triple_to_passage_map
from src.graph.graph_utils import get_entity_id

passages = load_passage_data()
print(f"Loaded {len(passages)} passages from cache\n")

# Inspect first few passages
for i in range(min(3, len(passages))):
    p = passages[i]
    print(f"Passage {i}:")
    print(f"  Keys: {p.keys()}")
    print(f"  Has 'entities' key: {'entities' in p}")
    if 'entities' in p:
        ents = p['entities']
        print(f"  Entities count: {len(ents)}")
        print(f"  First entity: {ents[0] if ents else 'none'}")
    print()

# Try building the map and see what happens
print("\nBuilding triple-to-passage map...")
triple_to_passages, passage_texts, passage_entities, entity_to_passages = build_triple_to_passage_map(passages)

print(f"Results:")
print(f"  triple_to_passages: {len(triple_to_passages)} entries")
print(f"  passage_texts: {len(passage_texts)} entries")
print(f"  passage_entities: {len(passage_entities)} entries")
print(f"  entity_to_passages (ENTITY_ID → passages): {len(entity_to_passages)} entries")

# Check actual entity IDs and counts
if entity_to_passages:
    print(f"\nFirst 5 entity_id → passages mappings:")
    for ent_id, pids in list(entity_to_passages.items())[:5]:
        print(f"  {ent_id[:20]}... → {len(pids)} passages")
else:
    print(f"\n⚠️  entity_to_passages is empty!")
    print("\nDebugging: Check first passage entity structure...")
    if passages and 'entities' in passages[0]:
        sample_ents = passages[0]['entities'][:2]
        print(f"Sample entities: {sample_ents}")
        for ent in sample_ents:
            if isinstance(ent, dict) and 'text' in ent:
                ent_text = ent['text']
                ent_id = get_entity_id(ent_text)
                print(f"  Entity text='{ent_text}' => entity_id={ent_id[:20]}...")
