from src.graph.graph_utils import normalize, get_entity_id
from src.data.musique_loader import load_musique
from src.extraction.extract_triples_batch import extract_triples_batch, DEFAULT_RELATION_SCHEMA
from collections import defaultdict

# Load tiny sample
print('Loading 1 sample for debug...')
dev_data = load_musique('dev', max_samples=1)
passages_by_id = {}
for qa in dev_data:
    for i, para in enumerate(qa.get('passages', [])):
        p_id = f'q_0_p{i}'
        p_text = para.get('text', '')
        if p_text:
            passages_by_id[p_id] = p_text

print(f'Loaded {len(passages_by_id)} passages')

# Extract triples
passage_texts = list(passages_by_id.values())
results = extract_triples_batch(passage_texts, relation_schema=DEFAULT_RELATION_SCHEMA, skip_cache=True, batch_size=16, save_cache_to_disk=False, extract_entities=True)

print(f'Extracted {len(results)} results')

# Check triple structure
if results:
    sample_result = results[0]
    triples = sample_result.get('triples', [])
    print(f'\nSample extraction result:')
    print(f'  Triples count: {len(triples)}')
    if triples:
        print(f'\nFirst 3 triples:')
        for i, t in enumerate(triples[:3]):
            print(f'\n  Triple {i}:')
            print(f'    Subject: "{t.get("subject")}"')
            print(f'    Relation: "{t.get("relation")}"')
            print(f'    Object: "{t.get("object")}"')
            
            # Show what they normalize to
            s_norm = normalize(t.get("subject", ""))
            o_norm = normalize(t.get("object", ""))
            print(f'    Subject normalized: "{s_norm}"')
            print(f'    Object normalized: "{o_norm}"')
            
            # Show entity IDs
            s_id = get_entity_id(t.get("subject", ""))
            o_id = get_entity_id(t.get("object", ""))
            print(f'    Subject ID: {s_id}')
            print(f'    Object ID: {o_id}')
