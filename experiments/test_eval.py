import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.musique_loader import load_musique, get_all_passages
from src.extraction.extract_triples import GLiNER2, ENTITY_LABELS, build_schema
from src.graph.build_graph import load_graph, normalize
from src.retrieval.pagerank import personalized_pagerank, get_subgraph

GRAPH_PATH = "data/processed/kg.pkl"

def extract_query_entities(question):
    """Simple heuristic: extract content words"""
    stop_words = {"who", "what", "when", "where", "why", "how", "is", "the", "a", "an", "of", "in", "on", "at", "and", "or", "but", "by", "with", "for", "that", "this"}
    
    words = question.lower().split()
    query_entities = [w.rstrip('?.,;:') for w in words if len(w) > 2 and w.lower() not in stop_words]
    return list(set(query_entities))  # remove duplicates

def run_qa(G, question, query_entities, top_k=10):
    """Retrieve top k nodes"""
    from src.retrieval.pagerank import personalized_pagerank
    ranked = personalized_pagerank(G, query_entities, top_k=top_k)
    return ranked

if __name__ == "__main__":
    # Load data
    samples = load_musique("dev", max_samples=100)
    G = load_graph(GRAPH_PATH)
    
    # Setup extractor
    extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    schema = build_schema(extractor, {})
    
    print("="*70)
    print("EVALUATION on 10 SAMPLES")
    print("="*70)
    
    metrics = {
        "total": 10,
        "hit@10": 0,
        "extracted_entities": 0,
        "avg_rank": 0
    }
    
    ranks = []
    
    for i in range(min(10, len(samples))):
        sample = samples[i]
        question = sample["question"]
        answer = sample["answer"].lower()
        
        # Extract entities
        query_entities = extract_query_entities(question)
        
        if not query_entities:
            print(f"\n[{i+1}] ✗ No entities extracted")
            continue
        
        metrics["extracted_entities"] += 1
        
        # Retrieve
        ranked = run_qa(G, question, query_entities, top_k=10)
        
        # Check if answer in top 10
        answer_found = False
        answer_rank = -1
        
        for rank, (node, score) in enumerate(ranked):
            if answer in node.lower():
                answer_found = True
                answer_rank = rank + 1
                break
        
        if answer_found:
            metrics["hit@10"] += 1
            ranks.append(answer_rank)
            status = "✓"
        else:
            status = "✗"
        
        print(f"\n[{i+1}] {status}")
        print(f"  Q: {question[:60]}...")
        print(f"  A (gold): {answer}")
        print(f"  Entities: {query_entities}")
        if answer_found:
            print(f"  ✓ FOUND at rank {answer_rank}")
        else:
            print(f"  Top 3 nodes: {[n for n, _ in ranked[:3]]}")
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Hit@10: {metrics['hit@10']}/10 ({metrics['hit@10']*10}%)")
    print(f"Extracted entities: {metrics['extracted_entities']}/10 ({metrics['extracted_entities']*10}%)")
    if ranks:
        print(f"Avg Rank (when found): {sum(ranks)/len(ranks):.2f}")
    print("="*70)

def debug_answer_coverage(samples, G):
    """Check if ground truth answers exist in graph"""
    found = 0
    for i, sample in enumerate(samples[:10]):
        answer = normalize(sample["answer"])
        if answer in G:
            found += 1
            print(f"[{i+1}] ✓ {sample['answer']} in graph")
        else:
            print(f"[{i+1}] ✗ {sample['answer']} NOT in graph")
    
    print(f"\nAnswer coverage: {found}/10")

debug_answer_coverage(samples, G)