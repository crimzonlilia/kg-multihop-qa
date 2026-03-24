# experiments/test_eval_simple.py
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.musique_loader import load_musique, get_all_passages
from src.extraction.extract_triples import GLiNER2, ENTITY_LABELS, build_schema
from src.graph.build_graph import load_graph, normalize
from src.retrieval.pagerank import personalized_pagerank

GRAPH_PATH = "data/processed/kg.pkl"

def extract_query_entities(question):
    """Simple heuristic: extract content words"""
    stop_words = {"who", "what", "when", "where", "why", "how", "is", "the", "a", "an", "of", "in", "on", "at", "and", "or", "but", "by", "with", "for", "that", "this"}
    words = question.lower().split()
    query_entities = [w.rstrip('?.,;:') for w in words if len(w) > 2 and w.lower() not in stop_words]
    return list(set(query_entities))

if __name__ == "__main__":
    samples = load_musique("dev", max_samples=100)
    G = load_graph(GRAPH_PATH)
    
    # Chỉ test samples có answer trong graph
    target_answers = ["miquette giraudy", "fletcher webster", "marie de medici", "european integration"]
    
    print("="*70)
    print("EVALUATION - Samples với Answer IN GRAPH")
    print("="*70)
    
    hit = 0
    for i, sample in enumerate(samples):
        answer_norm = normalize(sample["answer"])
        if answer_norm not in target_answers:
            continue
        
        question = sample["question"]
        answer = sample["answer"]
        
        # Extract entities & retrieve
        query_entities = extract_query_entities(question)
        ranked = personalized_pagerank(G, query_entities, top_k=10)
        
        # Check if answer in top 10
        answer_found = False
        answer_rank = -1
        
        for rank, (node, score) in enumerate(ranked):
            if answer_norm in node.lower():
                answer_found = True
                answer_rank = rank + 1
                break
        
        status = "✓" if answer_found else "✗"
        hit += answer_found
        
        print(f"\n[{status}] Q: {question[:60]}...")
        print(f"    A: {answer}")
        print(f"    Entities: {query_entities}")
        if answer_found:
            print(f"    ✓ FOUND at rank {answer_rank}")
        else:
            print(f"    Top 3: {[n for n, _ in ranked[:3]]}")
    
    print("\n" + "="*70)
    print(f"Result: {hit}/{len(target_answers)} ({hit*100//len(target_answers)}%)")
    print("="*70)