#!/usr/bin/env python3
"""Debug relation extraction with logging"""

import sys
import logging
sys.path.insert(0, ".")

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format='%(message)s')

from gliner2 import GLiNER2
import json

texts = [
    "John works for Apple Inc. in Cupertino.",
]

schema = {
    "born_in": "Person was born",
    "worked_at": "Person worked",
    "located_in": "Entity location",
}

print("Testing batch_extract_relations:\n")

extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

results = extractor.batch_extract_relations(
    texts,
    schema,
    include_confidence=True,
    batch_size=8
)

print("\nRaw result:")
print(json.dumps(results[0], indent=2))

print("\n\nNow test TYPE_CONSTRAINTS logic:")

# Simulate extraction processing
relation_dict = results[0].get("relation_extraction", {})
print(f"\nRelations in result: {list(relation_dict.keys())}")

from src.extraction.extract_triples import TYPE_CONSTRAINTS, CONFIDENCE_MIN

print(f"Relations in TYPE_CONSTRAINTS: {list(TYPE_CONSTRAINTS.keys())}")

for rel, pairs in relation_dict.items():
    print(f"\n  Relation: '{rel}' - {len(pairs)} pairs")
    
    if rel not in TYPE_CONSTRAINTS:
        print(f"    ✗ NOT in TYPE_CONSTRAINTS!")
    else:
        expected_subj, expected_objs = TYPE_CONSTRAINTS[rel]
        print(f"    ✓ Found in TYPE_CONSTRAINTS")
        print(f"      expected_subj={expected_subj}, expected_objs={expected_objs}")
    
    for i, pair in enumerate(pairs[:2]):
        print(f"    Pair {i}: {pair}")
