import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.musique_loader import load_musique, get_all_passages
from src.extraction.relation_discovery import run_discovery
from src.extraction.extract_triples import GLiNER2, ENTITY_LABELS, build_schema, extract_triples_batch
from src.graph.build_graph import build_graph, graph_stats, save_graph, load_graph, normalize
from src.retrieval.pagerank import personalized_pagerank, get_subgraph
import spacy

GRAPH_PATH = "data/processed/kg.pkl"

def build_kg(passages, force_rebuild=False):
    """Pass 1 + Pass 2 + build graph. Cache ở mọi bước."""
    if not force_rebuild and Path(GRAPH_PATH).exists():
        print("Loading cached graph...")
        return load_graph(GRAPH_PATH)

    # Pass 1 — discover schema
    expanded_schema = run_discovery(passages, min_cluster_freq=3)

    # Pass 2 — extract triples
    results = extract_triples_batch(passages, relation_schema=expanded_schema)

    print("\n=== TRIPLE EXTRACTION DEBUG ===")
    print(f"Total triples extracted: {sum(len(r['triples']) for r in results)}")

    # Build graph
    all_triples = [t for r in results for t in r["triples"]]
    all_entities = [e for r in results for e in r["entities"]]

    G = build_graph(all_triples, all_entities, score_threshold=0.0)
    graph_stats(G)
    save_graph(G, GRAPH_PATH)
    return G


def run_qa(G, question, query_entities, top_k=10):
    """Retrieve subgraph + format context."""
    ranked = personalized_pagerank(G, query_entities, top_k=top_k)
    subG = get_subgraph(G, ranked)

    # Format context từ subgraph edges
    context_lines = []
    for u, v, data in subG.edges(data=True):
        for r in data.get("relations", []):
            rel = r["type"] if isinstance(r, dict) else r
            context_lines.append(f"{u} {rel} {v}")

    return "\n".join(context_lines), ranked


if __name__ == "__main__":
    samples = load_musique("dev", max_samples=100)
    passages = get_all_passages(samples)

    print(f"=== DATA LOADING ===")
    print(f"Samples: {len(samples)}, Passages: {len(passages)}")
    
    # Kiểm xem câu hỏi có trong samples không
    target_question = "Who is the spouse of the Green performer?"
    found_sample = None
    
    for i, s in enumerate(samples):
        if target_question.lower() in s["question"].lower():
            found_sample = (i, s)
            break
    
    if found_sample:
        idx, sample = found_sample
        print(f"\n✓ Found at sample #{idx}")
        print(f"  Q: {sample['question']}")
        print(f"  A: {sample['answer']}")
        print(f"  Passages: {len(sample['passages'])}")
        if sample['passages']:
            print(f"  First passage: {sample['passages'][0][:100]}...")
    else:
        print(f"\n✗ '{target_question}' NOT in first 100 samples!")
        print(f"\nFirst 5 questions:")
        for i, s in enumerate(samples[:5]):
            print(f"  [{i}] {s['question']}")
    
    # FORCE rebuild để test
    G = build_kg(passages, force_rebuild=True)

# Test QA trên sample #0
    sample = samples[0]
    question = sample["question"]
    answer = sample["answer"]

    print(f"\n{'='*60}")
    print(f"Q: {question}")
    print(f"A (gold): {answer}")

    # Extract entities from question using GLiNER2
    extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    schema = build_schema(extractor, {})
    results = extractor.extract(question, schema)

    query_entities = []
    for label, items in results.get("entities", {}).items():
        for item in items:
            text = item["text"] if isinstance(item, dict) else item
            query_entities.append(text)

    print(f"Extracted entities: {query_entities}")

    if query_entities:
        context, ranked = run_qa(G, question, query_entities, top_k=10)
        print(f"\nTop retrieved nodes:")
        for node, score in ranked[:5]:
            print(f"  {node}: {score:.4f}")
        
        print(f"\nContext retrieved:\n{context[:500]}")
    else:
        print("⚠️ No entities found! Using fallback...")
        query_entities = [w for w in question.split() if len(w) > 3 and w.lower() not in {"who", "what", "where", "when", "why"}]
        print(f"Fallback entities: {query_entities}")
        if query_entities:
            context, ranked = run_qa(G, question, query_entities, top_k=10)
            print(f"Context: {context[:500]}")

    # Check xem các answers có trong triples không
    target_answers = ["miquette giraudy", "fletcher webster", "marie de medici", "european integration"]
    for answer in target_answers:
        found = False
        for r in results:
            for t in r['triples']:
                if answer in t['subject'].lower() or answer in t['object'].lower():
                    found = True
                    print(f"✓ {answer} found in: {t}")
                    break
            if found:
                break
        if not found:
            print(f"✗ {answer} NOT in any triple")