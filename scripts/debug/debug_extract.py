"""Debug extraction with few samples - test correct GLiNER2 API."""

import json
from pathlib import Path
from src.extraction.model import get_extractor
from src.extraction.schema import DEFAULT_RELATION_SCHEMA

# Load 2 simple passages
passages = [
    "Washington was born in Virginia in 1732.",
    "Albert Einstein was a German physicist born in 1879."
]

print(f"Testing extraction with {len(passages)} passages...")

try:
    print("Loading extractor...")
    extractor = get_extractor()
    print("✓ Extractor loaded")
    
    print("\nExtracting entities...")
    entity_results = extractor.batch_extract_entities(
        passages, 
        ["person", "location", "event"], 
        include_confidence=True, 
        batch_size=2
    )
    print(f"✓ Got {len(entity_results)} entity results")
    print(f"  Sample: {entity_results[0]}")
    
    print("\nExtracting relations (passing list of relation names)...")
    # Correct API: pass list of relation names, NOT dict
    relation_names = list(DEFAULT_RELATION_SCHEMA.keys())
    print(f"  Extracting {len(relation_names)} relations: {relation_names[:5]}...")
    
    relation_results = extractor.batch_extract_relations(
        passages, 
        relation_names,  # List of names, NOT dict
        include_confidence=True, 
        batch_size=2
    )
    print(f"✓ Got {len(relation_results)} relation results")
    print(f"  Result structure: {type(relation_results[0])}")
    print(f"  Result keys: {list(relation_results[0].keys()) if isinstance(relation_results[0], dict) else 'N/A'}")
    print(f"\n  Full sample result:\n{json.dumps(relation_results[0], indent=2, default=str)[:500]}")
    
except Exception as e:
    import traceback
    print(f"\n✗ Error: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
