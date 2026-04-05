"""Simple quick test of fixed extraction API."""
from src.extraction.model import get_extractor
from src.extraction.schema import DEFAULT_RELATION_SCHEMA

print("Loading extractor...")
ext = get_extractor()
print("✓ OK\n")

print("Testing batch_extract_relations with passage list...")
rels = list(DEFAULT_RELATION_SCHEMA.keys())
print(f"Relations: {rels[:5]}...\n")

result = ext.batch_extract_relations(['John works at Apple'], rels, batch_size=1)
print(f"✓ Result keys: {list(result[0].keys())}")
print(f"✓ Num relations returned: {len(result[0]['relation_extraction'])}")

# Check if any relations extracted
rel_extraction = result[0]['relation_extraction']
extracted_count = sum(len(v) for v in rel_extraction.values())
print(f"✓ Total extracted relation pairs: {extracted_count}")

print("\n✓ SUCCESS - API is fixed!")
