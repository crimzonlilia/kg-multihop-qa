#!/usr/bin/env python3
"""
Comprehensive debug script để tìm ra tại sao evaluation returns 0%
"""
import sys, json, re
from pathlib import Path
sys.path.insert(0, ".")

from src.data.musique_loader import load_musique
from src.graph.build_graph import load_graph, normalize
from src.retrieval.pagerank import personalized_pagerank_fast

def check_graph():
    """Check if graph is loaded correctly"""
    print("\n" + "="*70)
    print("1. GRAPH LOADING CHECK")
    print("="*70)
    
    try:
        G = load_graph("data/processed/kg.pkl")
        print(f"✓ Graph loaded successfully")
        print(f"  - Nodes: {G.number_of_nodes()}")
        print(f"  - Edges: {G.number_of_edges()}")
        
        if G.number_of_nodes() == 0:
            print("  ❌ PROBLEM: Graph has 0 nodes!")
            return None
        
        # Sample nodes
        sample_nodes = list(G.nodes())[:10]
        print(f"  - Sample nodes: {sample_nodes}")
        
        return G
    except Exception as e:
        print(f"❌ Graph loading failed: {e}")
        return None


def check_data():
    """Check if data loads correctly"""
    print("\n" + "="*70)
    print("2. DATA LOADING CHECK")
    print("="*70)
    
    try:
        samples = load_musique("dev", max_samples=10)
        print(f"✓ Data loaded: {len(samples)} samples")
        
        if not samples:
            print("  ❌ PROBLEM: No samples loaded!")
            return None
        
        sample = samples[0]
        print(f"\n  Sample 0:")
        print(f"    - Question: {sample['question']}")
        print(f"    - Answer: {sample['answer']}")
        print(f"    - Hops: {sample['hops']}")
        print(f"    - Passages: {len(sample.get('passages', []))} passages")
        print(f"    - Supporting facts: {sample.get('supporting_facts', 'N/A')}")
        
        return samples
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return None


def extract_query_entities_debug(G, question):
    """Debug entity extraction"""
    print(f"\n  • Question: {question}")
    
    # Try spacy
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(question)
        spacy_entities = [ent.text.lower() for ent in doc.ents]
        print(f"  • Spacy entities: {spacy_entities}")
        
        # Check if they're in graph
        matched = [n for n in G.nodes() if any(e in n for e in spacy_entities)]
        print(f"  • Matched to graph: {matched[:3]}...")
        
        if matched:
            return sorted(matched, key=len, reverse=True)[:5]
    except Exception as e:
        print(f"  • Spacy failed: {e}")
    
    # Fallback: capitalized words
    words = question.split()
    candidates = [w.lower() for w in words if w and w[0].isupper()]
    print(f"  • Capitalized words (fallback): {candidates}")
    
    fallback_matched = [n for n in G.nodes() if n.lower() in candidates]
    print(f"  • Fallback matched: {fallback_matched[:3]}...")
    
    return [n for n in G.nodes() if n.lower() in candidates][:5]


def check_retrieval(G, samples):
    """Check retrieval on first 3 samples"""
    print("\n" + "="*70)
    print("3. RETRIEVAL CHECK")
    print("="*70)
    
    for i, sample in enumerate(samples[:3]):
        print(f"\n  --- Sample {i+1} ---")
        question = sample["question"]
        gold_answer = sample["answer"]
        
        # Extract entities
        query_entities = extract_query_entities_debug(G, question)
        
        if not query_entities:
            print(f"  ❌ No query entities found!")
            continue
        
        print(f"  • Query entities for retrieval: {query_entities}")
        
        # Run retrieval
        ranked = personalized_pagerank_fast(G, query_entities, top_k=10, neighborhood_hops=3)
        
        if not ranked:
            print(f"  ❌ No results from retrieval!")
            continue
        
        print(f"  • Top-5 retrieved nodes:")
        for rank, (node, score) in enumerate(ranked[:5], 1):
            print(f"    {rank}. {node} (score: {score:.4f})")
        
        # Check if gold answer is in there
        print(f"\n  • Gold answer: '{gold_answer}'")
        print(f"  • Normalized gold: '{normalize(gold_answer)}'")
        
        ranked_normalized = [normalize(n) for n, _ in ranked]
        normalized_gold = normalize(gold_answer)
        
        if normalized_gold in ranked_normalized:
            print(f"  ✓ Gold answer FOUND in top-10")
            idx = ranked_normalized.index(normalized_gold)
            print(f"    Rank: {idx + 1}")
        else:
            print(f"  ❌ Gold answer NOT in top-10")
            print(f"  • Ranked normalized: {ranked_normalized}")
            
            # Check if gold is in graph at all
            if normalized_gold in G:
                print(f"  • Note: Gold answer IS in graph, just not retrieved")
            else:
                print(f"  • Note: Gold answer NOT in graph")


def check_normalization():
    """Check normalization consistency"""
    print("\n" + "="*70)
    print("4. NORMALIZATION CHECK")
    print("="*70)
    
    test_answers = [
        "John Smith",
        "the United States",
        "Green performer",
        "Alice in Wonderland",
    ]
    
    for ans in test_answers:
        norm = normalize(ans)
        print(f"  '{ans}' → '{norm}'")


if __name__ == "__main__":
    G = check_graph()
    samples = check_data()
    check_normalization()
    
    if G and samples:
        check_retrieval(G, samples)
    
    print("\n" + "="*70)
    print("END DEBUG")
    print("="*70)
