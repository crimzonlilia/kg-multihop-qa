import sys, json, re
from pathlib import Path
from collections import Counter
sys.path.insert(0, ".")

from src.data.musique_loader import load_musique
from src.graph.build_graph import load_graph, normalize
from src.retrieval.pagerank import personalized_pagerank, get_subgraph
from src.qa import triples_to_text


def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def extract_query_entities_from_graph(G, question):
    q_lower = question.lower()
    matched = [n for n in G.nodes() if n.lower() in q_lower.split() and len(n) > 3]
    return sorted(matched, key=len, reverse=True)[:5]


if __name__ == "__main__":
    samples = load_musique("dev", max_samples=200)  # All 2417 samples
    G = load_graph("data/processed/kg.pkl")

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

        ranked = personalized_pagerank(G, query_entities, top_k=10)
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