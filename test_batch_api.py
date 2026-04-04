#!/usr/bin/env python3
"""Test GLiNER2 batch API to understand the exact format"""

import sys
sys.path.insert(0, ".")

from gliner2 import GLiNER2
import json

print("=" * 70)
print("Testing GLiNER2 Batch API Format")
print("=" * 70)

# Load model
print("\n1. Loading GLiNER2 model...")
extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
print("   ✓ Model loaded")

# Test passages
texts = [
    "John works for Apple Inc. in Cupertino.",
    "Sarah founded TechStartup in 2020.",
]

entity_labels = ["person", "company", "location", "date"]

print(f"\n2. Testing batch_extract_entities with {len(texts)} texts...")
print(f"   Labels: {entity_labels}\n")

entity_results = extractor.batch_extract_entities(
    texts,
    entity_labels,
    include_confidence=True,
    batch_size=8
)

print("   Entity Results Type:", type(entity_results))
print("   Entity Results Length:", len(entity_results))
print("   Entity Results:\n")
for i, result in enumerate(entity_results):
    print(f"     [{i}] Type: {type(result)}")
    print(f"     [{i}] Content: {json.dumps(result, indent=8)}")
    print()

# Test relation extraction
relation_schema = {
    "works_for": "Person works for organization",
    "founded": "Person founded organization",
    "located_in": "Organization located in location"
}

print(f"\n3. Testing batch_extract_relations with {len(texts)} texts...")
print(f"   Relations: {list(relation_schema.keys())}\n")

relation_results = extractor.batch_extract_relations(
    texts,
    relation_schema,
    include_confidence=True,
    batch_size=8
)

print("   Relation Results Type:", type(relation_results))
print("   Relation Results Length:", len(relation_results))
print("   Relation Results:\n")
for i, result in enumerate(relation_results):
    print(f"     [{i}] Type: {type(result)}")
    print(f"     [{i}] Content: {json.dumps(result, indent=8)}")
    print()

print("=" * 70)
print("Format Analysis Complete!")
print("=" * 70)
