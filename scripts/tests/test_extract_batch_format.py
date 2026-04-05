"""Debug extract_triples_batch output format"""
from src.extraction.extract_triples import extract_triples_batch, DEFAULT_RELATION_SCHEMA
from src.data.musique_loader import load_musique

dev_data = load_musique('dev', max_samples=1)
passages = [p['text'] for qa in dev_data for p in qa['passages']]

print(f'Number of passages: {len(passages)}')
results = extract_triples_batch(passages[:5], relation_schema=DEFAULT_RELATION_SCHEMA, deduplicate=True, skip_cache=True)
print(f'Number of results: {len(results)}')
for i, r in enumerate(results[:3]):
    keys = list(r.keys())
    passage_len = len(r.get("passage", ""))
    triples_count = len(r.get("triples", []))
    print(f'Result {i}: keys={keys}, passage_len={passage_len}, triples_count={triples_count}')
