"""
Diagnostic: Check entity extraction from questions
"""
import time
from src.extraction.extract_triples import extract_triples_batch, DEFAULT_RELATION_SCHEMA
from src.graph.build_graph import build_graph, merge_graphs
from src.data.musique_loader import load_musique
from src.retrieval.pagerank import extract_entities_from_question
import networkx as nx
import spacy

print("Step 1: Load 50 samples")
dev_data = load_musique("dev", max_samples=50)
print(f"Loaded {len(dev_data)} samples")

print("\nStep 2: Build passages")
passages_by_id = {}
for qa in dev_data:
    for i, para in enumerate(qa.get("passages", [])):
        p_text = para.get("text", "")
        if p_text:
            p_id = f"q{len(passages_by_id)}"
            passages_by_id[p_id] = p_text

print(f"Total passages: {len(passages_by_id)}")

print("\nStep 3: Extract triples")
passage_texts = list(passages_by_id.values())
results = extract_triples_batch(
    passage_texts,
    use_dynamic=True,
    skip_cache=True,
    deduplicate=False,
)

triples_by_passage = {}
for i, r in enumerate(results):
    triples_by_passage[i] = r.get("triples", [])

total_triples = sum(len(t) for t in triples_by_passage.values())
print(f"Total triples extracted: {total_triples}")



print("\nStep 4: Build graph")
passage_graphs = [
    build_graph(triples, add_cooccurrence_edges=True) 
    for triples in triples_by_passage.values() 
    if triples
]
final_graph = merge_graphs(passage_graphs)
print(f"Graph: {final_graph.number_of_nodes()} nodes, {final_graph.number_of_edges()} edges")

print("\nStep 5: Test entity extraction from questions")
nlp = spacy.load("en_core_web_sm")

sample_questions = [qa["question"] for qa in dev_data[:10]]
entity_counts = []

for q in sample_questions:
    ents = extract_entities_from_question(q, final_graph, nlp)
    entity_counts.append(len(ents))
    status = "OK" if ents else "FAIL"
    print(f"  [{status}] {q[:60]} -> {ents}")

print(f"\nEntity extraction success rate: {sum(1 for c in entity_counts if c > 0)}/{len(entity_counts)}")
print(f"Graph coverage:")
print(f"  - Nodes: {final_graph.number_of_nodes()}")
print(f"  - Edges: {final_graph.number_of_edges()}")
print(f"  - Density: {nx.density(final_graph):.4f}")

# Show some graph nodes
print(f"\nSample graph nodes (first 10):")
for node in list(final_graph.nodes())[:10]:
    print(f"  - {node}")

