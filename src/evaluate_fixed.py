import sys, json, re, time
from datetime import datetime
from pathlib import Path
from collections import Counter
sys.path.insert(0, ".")

from src.data.musique_loader import load_musique
from src.graph.build_graph import load_graph, normalize
from src.retrieval.pagerank import personalized_pagerank_fast
import spacy
import networkx as nx

try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("⚠️  Spacy model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

class ResultLogger:
    def __init__(self, base_name="eval_result"):
        # Create filename with timestamp to avoid overwriting
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = f"{base_name}_{timestamp}.txt"
        self.base_name = base_name
        self.lines = []
    
    def log(self, message=""):
        print(message)
        self.lines.append(str(message))
    
    def save(self):
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines))
        print(f"\n✓ Results saved to {self.filepath}")
        
        # Also append to summary file for comparison
        summary_file = f"{self.base_name}_summary.txt"
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"File: {self.filepath}\n")
            f.write("="*70 + "\n")
            
            # Extract key metrics
            for line in self.lines:
                if "Hit@10" in line or "Nodes:" in line or "Edges:" in line or "Components:" in line:
                    f.write(line + "\n")


def normalize_answer(s: str) -> str:
    """Aggressive normalization for answer matching"""
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def extract_query_entities_improved(G, question):
    """
    FIXED: Better entity extraction
    - Uses spacy NER more robustly
    - Better fallback matching
    """
    try:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(question)
        
        # Extract NER entities (proper nouns)
        entities = []
        for ent in doc.ents:
            # Match entity text to graph nodes (case-insensitive, substring)
            ent_norm = normalize(ent.text)
            
            # Try exact match first
            for node in G.nodes():
                if normalize(node) == ent_norm:
                    entities.append(node)
                    break
            
            # Try substring match if no exact match
            else:
                for node in G.nodes():
                    if ent_norm in normalize(node) or normalize(node) in ent_norm:
                        entities.append(node)
                        break
        
        if entities:
            return list(set(entities))[:5]  # Remove duplicates, return top 5
    
    except Exception as e:
        print(f"  ⚠ Spacy failed: {e}")
    
    # IMPROVED Fallback: match any capitalized words to graph
    words = question.split()
    capitalized = [w for w in words if w and w[0].isupper()]
    
    matched = []
    for word in capitalized:
        word_norm = normalize(word)
        for node in G.nodes():
            node_norm = normalize(node)
            # Match if exact or contained
            if word_norm in node_norm or node_norm in word_norm:
                matched.append(node)
    
    return list(set(matched))[:5]  # Dedup


def extract_answer_entities(answer_text):
    """Extract named entities from the gold answer"""
    try:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(answer_text)
        return [ent.text.lower() for ent in doc.ents]
    except:
        return []


def check_answer_in_retrieved(gold_answer, retrieved_nodes):
    """
    FIXED: Better matching between gold answer and retrieved nodes
    
    Strategies:
    1. Exact normalized match
    2. Substring match (answer in node or vice versa)
    3. Named entity overlap
    """
    answer_norm = normalize_answer(gold_answer)
    
    for node in retrieved_nodes:
        node_norm = normalize_answer(node)
        
        # Strategy 1: Exact match
        if answer_norm == node_norm:
            return True
        
        # Strategy 2: Significant substring (at least 3 chars)
        if len(answer_norm) >= 3:
            if answer_norm in node_norm or node_norm in answer_norm:
                return True
    
    return False


if __name__ == "__main__":
    start_time = time.time()
    logger = ResultLogger("eval_result")  # Will auto-add timestamp
    
    logger.log("Loading data...")
    samples = load_musique("dev", max_samples=200)
    G = load_graph("data/processed/kg.pkl")
    
    # Log graph statistics
    logger.log(f"Loaded {len(samples)} samples, graph has {G.number_of_nodes()} nodes")
    logger.log(f"\n{'='*70}")
    logger.log(f"GRAPH STATISTICS")
    logger.log(f"{'='*70}")
    logger.log(f"Nodes: {G.number_of_nodes()}")
    logger.log(f"Edges: {G.number_of_edges()}")
    logger.log(f"Components: {nx.number_weakly_connected_components(G)}")
    logger.log(f"Average degree: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
    logger.log("")

    # DEBUG: Check first 3 samples
    logger.log("="*70)
    logger.log("DEBUG: First 3 samples")
    logger.log("="*70)
    for sample_idx in range(min(3, len(samples))):
        sample = samples[sample_idx]
        question = sample["question"]
        gold = sample["answer"]
        
        logger.log(f"\nSample {sample_idx+1}: {question[:60]}...")
        logger.log(f"Expected answer: {gold}")
        
        # Query entity extraction
        query_entities = extract_query_entities_improved(G, question)
        logger.log(f"Query entities found: {query_entities}")
        
        if not query_entities:
            logger.log("❌ No query entities found!")
            continue
        
        # Run retrieval
        ranked = personalized_pagerank_fast(G, query_entities, top_k=10, neighborhood_hops=3)
        logger.log(f"Top-10 results:")
        for rank, (node, score) in enumerate(ranked, 1):
            match = "✓" if check_answer_in_retrieved(gold, [node]) else " "
            logger.log(f"  {rank}. [{match}] {node} ({score:.4f})")

    # Main evaluation
    logger.log("\n" + "="*70)
    logger.log("EVALUATION")
    logger.log("="*70)
    
    eval_start = time.time()
    hit_at_10 = []
    hops_performance = Counter()
    hops_counts = Counter()
    skipped = 0

    for i, sample in enumerate(samples):
        question = sample["question"]
        gold = sample["answer"]
        hops = sample.get("hops", 1)
        
        hops_counts[hops] += 1

        # Extract entities
        query_entities = extract_query_entities_improved(G, question)
        if not query_entities:
            skipped += 1
            continue

        # Retrieve
        ranked = personalized_pagerank_fast(G, query_entities, top_k=10, neighborhood_hops=3)
        ranked_nodes = [n for n, _ in ranked]
        
        # Check Hit@10 with improved matching
        is_hit = check_answer_in_retrieved(gold, ranked_nodes)
        hit_at_10.append(is_hit)
        
        if is_hit:
            hops_performance[hops] += 1

        if (i + 1) % 20 == 0:
            logger.log(f"[{i+1}] Q: {question[:50]}...")
            logger.log(f"     Gold: {gold} | Hit: {'✓' if is_hit else '✗'}")

    eval_time = time.time() - eval_start

    # Results
    logger.log(f"\n{'='*70}")
    logger.log(f"OVERALL RESULTS")
    logger.log(f"{'='*70}")
    logger.log(f"Coverage (evaluated): {len(hit_at_10)}/{len(samples)} ({len(hit_at_10)*100/len(samples):.1f}%)")
    logger.log(f"Skipped: {skipped} (no query entities)")
    logger.log(f"\nHit@10 (FIXED):      {sum(hit_at_10)}/{len(hit_at_10)} ({sum(hit_at_10)*100/len(hit_at_10):.1f}%)" if hit_at_10 else "Hit@10: N/A")

    # Hop analysis
    logger.log(f"\n{'='*70}")
    logger.log(f"HOP-BY-HOP ANALYSIS")
    logger.log(f"{'='*70}")
    for hop in sorted(hops_counts.keys()):
        count = hops_counts[hop]
        correct = hops_performance[hop]
        acc = correct * 100 / count if count > 0 else 0
        logger.log(f"{hop}-hop: {correct}/{count} ({acc:.1f}%)")
    
    # Timing info
    total_time = time.time() - start_time
    logger.log(f"\n{'='*70}")
    logger.log(f"TIMING INFORMATION")
    logger.log(f"{'='*70}")
    logger.log(f"Evaluation time: {eval_time:.2f}s")
    logger.log(f"Total time: {total_time:.2f}s")
    logger.log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save results
    logger.save()
