import sys, json, re
from pathlib import Path
from collections import Counter
sys.path.insert(0, ".")

from src.data.musique_loader import load_musique
from src.graph.build_graph import load_graph, normalize
from src.retrieval.pagerank import personalized_pagerank, personalized_pagerank_fast, get_subgraph
from src.qa import triples_to_text
import nltk
import networkx as nx

def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def extract_query_entities_from_graph(G, question):
    """
    Extract entities from question using:
    1. Capitalized words (proper nouns)
    2. Ignore stop words
    3. Match to graph nodes
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    stop_words = {'who', 'what', 'where', 'when', 'why', 'how', 'is', 'are', 
                  'the', 'a', 'an', 'and', 'or', 'of', 'in', 'at', 'by', 'for',
                  'founded', 'distributed', 'spouse', 'owner', 'entity', 'performer'}
    
    # Extract capitalized words (proper nouns)
    words = question.split()
    candidates = []
    
    for word in words:
        # Clean punctuation
        clean_word = re.sub(r'[^\w]', '', word)
        
        # Check if capitalized and not stop word
        if clean_word and clean_word[0].isupper() and clean_word.lower() not in stop_words and len(clean_word) > 2:
            candidates.append(clean_word.lower())
    
    # Match to graph nodes
    matched = [n for n in G.nodes() if n.lower() in candidates]
    
    # Fallback: nếu không tìm được, try longer substrings từ question
    if not matched:
        # Split question by common delimiters
        phrases = re.split(r'\s+(?:of|in|is|the)\s+', question, flags=re.IGNORECASE)
        for phrase in phrases:
            phrase = phrase.strip().rstrip('?').lower()
            if len(phrase) > 6:  # Longer phrases likely to be entities
                # Try fuzzy match
                for node in G.nodes():
                    if node in phrase and len(node.split()) > 1:  # Multi-word entities
                        matched.append(node)
                if matched:
                    break
    
    return sorted(matched, key=len, reverse=True)[:5]


if __name__ == "__main__":
    samples = load_musique("dev", max_samples=10)  # All 2417 samples
    G = load_graph("data/processed/kg.pkl")

    # DEBUG: Check first 3 samples
    for sample_idx in range(min(3, len(samples))):
        sample = samples[sample_idx]
        question = sample["question"]
        gold = sample["answer"]
        
        print(f"\n{'='*60}")
        print(f"Sample {sample_idx+1}: {question}")
        print(f"Expected answer: {gold}")
        
        # Query entity extraction
        query_entities = extract_query_entities_from_graph(G, question)
        print(f"Query entities extracted: {query_entities}")
        
        if not query_entities:
            print("❌ No query entities found!")
            continue
        
        # Check if answer is in graph
        answer_norm = normalize(gold)
        print(f"Normalized answer: {answer_norm}")
        print(f"Answer in graph: {answer_norm in G}")
        
        # Run retrieval
        ranked = personalized_pagerank_fast(G, query_entities, top_k=10, neighborhood_hops=3)
        print(f"\nTop-10 results:")
        for rank, (node, score) in enumerate(ranked, 1):
            is_match = normalize(node) == answer_norm
            marker = "✓ MATCH" if is_match else ""
            print(f"  {rank}. {node} ({score:.4f}) {marker}")
        
        print(f"{'='*60}")

    # Tracking variables
    hit_at_10 = []
    supporting_fact_hits = []
    ranks_found = []
    hops_performance = Counter()
    hops_counts = Counter()
    skipped = 0

    for i, sample in enumerate(samples):
        question = sample["question"]
        gold = sample["answer"]
        hops = sample["hops"]
        supporting_facts = sample["supporting_facts"]
        
        hops_counts[hops] += 1

        query_entities = extract_query_entities_from_graph(G, question)
        if not query_entities:
            skipped += 1
            continue

        ranked = personalized_pagerank_fast(G, query_entities, top_k=10, neighborhood_hops=3)
        ranked_nodes = [normalize(n) for n, _ in ranked]
        
        # Check Hit@10
        normalized_gold = normalize(gold)
        is_hit = normalized_gold in ranked_nodes
        hit_at_10.append(is_hit)
        
        if is_hit:
            hops_performance[hops] += 1

        # Track rank if found
        answer_rank = None
        if is_hit:
            for rank, (node, score) in enumerate(ranked):
                if normalize(gold) == normalize(node):
                    answer_rank = rank + 1
                    ranks_found.append(answer_rank)
                    break

        # Check Supporting Facts Hit (more strict)
        supporting_fact_in_top10 = False
        if supporting_facts and ranked:
            # Extract supporting passage indices from ranked nodes
            retrieved_passages = [n for n, _ in ranked]
            # Check if any supporting fact title is in retrieved passages
            supporting_titles = [title for idx, title in supporting_facts]
            for title in supporting_titles:
                if any(title.lower() in node.lower() for node in retrieved_passages):
                    supporting_fact_in_top10 = True
                    break
        
        supporting_fact_hits.append(supporting_fact_in_top10)

        if (i + 1) % 20 == 0:
            print(f"[{i+1}] Q: {question[:60]}")
            print(f"     Gold: {gold} | Hit: {is_hit} | SF Hit: {supporting_fact_in_top10} | Rank: {answer_rank}")

    print(f"\n{'='*60}")
    print(f"OVERALL RESULTS")
    print(f"{'='*60}")
    print(f"Coverage (evaluated): {len(hit_at_10)}/{len(samples)} ({len(hit_at_10)*100/len(samples):.1f}%)")
    print(f"Skipped: {skipped} (no query entities)")
    print(f"\nHit@10:              {sum(hit_at_10)}/{len(hit_at_10)} ({sum(hit_at_10)*100/len(hit_at_10):.1f}%)")
    print(f"Supporting Fact Hit: {sum(supporting_fact_hits)}/{len(supporting_fact_hits)} ({sum(supporting_fact_hits)*100/len(supporting_fact_hits):.1f}%)")
    
    if ranks_found:
        print(f"\nAvg Rank (when found): {sum(ranks_found)/len(ranks_found):.1f}")
        print(f"MRR:                   {sum(1/r for r in ranks_found)/len(ranks_found):.3f}")

    # Hop-level analysis
    print(f"\n{'='*60}")
    print(f"HOP-BY-HOP ANALYSIS")
    print(f"{'='*60}")
    for hop in sorted(hops_counts.keys()):
        count = hops_counts[hop]
        correct = hops_performance[hop]
        acc = correct * 100 / count if count > 0 else 0
        print(f"{hop}-hop: {correct}/{count} ({acc:.1f}%)")

    print(f"{'='*60}")

    for s in samples[:3]:
        print(f"Hops: {s.get('hops')}, Q: {s['question'][:50]}")