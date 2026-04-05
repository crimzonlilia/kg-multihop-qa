#!/usr/bin/env python3
"""Test extract_triples with corrected batch processing"""

import sys
sys.path.insert(0, ".")

from src.extraction.extract_triples import extract_triples_batch, get_extractor

# Test with 10 small passages
test_passages = [
    "John works for Apple Inc. in Cupertino.",
    "Sarah founded TechStartup in California.",
    "Microsoft CEO Satya Nadella announced new products.",
    "Google is located in Mountain View.",
    "Amazon started in Seattle.",
    "Tesla was founded by Elon Musk.",
    "Netflix streams movies and shows.",
    "Uber operates in multiple cities.",
    "Spotify has millions of songs.",
    "Intel manufactures computer chips.",
]

print("=" * 70)
print("Testing Extract Triples (Batch Mode)")
print("=" * 70)

print(f"\nTesting with {len(test_passages)} passages...")

# Load model once
extractor = get_extractor()

# Extract with batch_size=5
results = extract_triples_batch(
    test_passages,
    batch_size=5,
    skip_cache=True,
    deduplicate=False,
    extractor=extractor,
    save_cache=False
)

print(f"\n✓ Extracted from {len(results)} passages")
print("\nResults Summary:")
print("-" * 70)

total_entities = 0
total_triples = 0

for i, result in enumerate(results[:5]):  # Show first 5
    passage = result["passage"][:60]
    entities = result.get("entities", [])
    triples = result.get("triples", [])
    
    total_entities += len(entities)
    total_triples += len(triples)
    
    print(f"\n[{i+1}] Passage: {passage}...")
    print(f"    Entities: {len(entities)}")
    for e in entities[:3]:
        print(f"      - {e['text']} ({e['type']})")
    if len(entities) > 3:
        print(f"      ... and {len(entities)-3} more")
    
    print(f"    Triples: {len(triples)}")
    for t in triples[:3]:
        print(f"      - ({t['subject']}, {t['relation']}, {t['object']})")
    if len(triples) > 3:
        print(f"      ... and {len(triples)-3} more")

print(f"\n{'=' * 70}")
print(f"Total Entities: {total_entities}")
print(f"Total Triples: {total_triples}")
print(f"Average Entities per passage: {total_entities / min(5, len(results)):.1f}")
print(f"Average Triples per passage: {total_triples / min(5, len(results)):.1f}")
print("=" * 70)

if total_triples > 0:
    print("✓ SUCCESS - Extract is working!")
else:
    print("✗ FAILED - No triples extracted")
