"""Compare use_dynamic=True vs False extraction quality."""
import sys
sys.path.insert(0, '.')
from src.extraction.extract_triples import extract_triples_batch, DEFAULT_RELATION_SCHEMA

p5 = "Miquette Giraudy (born 9 February 1953, Nice, France) is a keyboard player and vocalist, best known for her work in Gong and with her partner Steve Hillage."
p10 = "Green is the fourth studio album by British progressive rock musician Steve Hillage. Written in spring 1977."

print("=== use_dynamic=True ===")
results = extract_triples_batch([p5, p10], relation_schema=DEFAULT_RELATION_SCHEMA, use_dynamic=True, skip_cache=True, deduplicate=False)
for i, r in enumerate(results):
    t_count = len(r["triples"])
    e_count = len(r["entities"])
    rels = [t["relation"] for t in r["triples"]]
    print(f"P{i}: {t_count} triples, {e_count} entities, relations={rels}")

print("\n=== use_dynamic=False ===")
results2 = extract_triples_batch([p5, p10], relation_schema=DEFAULT_RELATION_SCHEMA, use_dynamic=False, skip_cache=True, deduplicate=False)
for i, r in enumerate(results2):
    t_count = len(r["triples"])
    e_count = len(r["entities"])
    rels = [t["relation"] for t in r["triples"]]
    print(f"P{i}: {t_count} triples, {e_count} entities, relations={rels}")
