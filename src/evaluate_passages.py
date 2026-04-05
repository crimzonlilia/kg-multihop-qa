"""
Evaluate QA using passage-based retrieval (HippoRAG style)
Rank passages by entity coverage, check if answer is in passages
"""

import os
import sys, json, time, re
from datetime import datetime
from pathlib import Path
from collections import Counter
import statistics
from difflib import SequenceMatcher
sys.path.insert(0, ".")

from src.console_utils import configure_console_output
from src.cache_utils import resolve_graph_output_path
from src.data.musique_loader import load_musique
from src.graph.build_graph import load_graph, normalize
from src.retrieval.passage_ranking import (
    load_passage_data,
    build_triple_to_passage_map,
    rank_passages_by_entities,
    extract_answer_from_passages
)
import spacy
import networkx as nx

try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("⚠️  Spacy model not found")
    nlp = None

configure_console_output()

TRIPLES_CACHE_NAME = os.getenv("TRIPLES_CACHE_NAME", "full")  # e.g. full, 300


class ResultLogger:
    def __init__(self, base_name="eval_passages"):
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
                if "Found in top-10" in line or "Average rank" in line or "Median rank" in line:
                    f.write(line + "\n")


def normalize_answer(s: str) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def extract_query_entities(G, question):
    """Extract entities from question (improved with fallback strategies)"""
    if nlp is None:
        return []
    
    entities = set()
    graph_nodes = list(G.nodes())  # Cache for faster lookup
    graph_nodes_lower = {normalize(n): n for n in graph_nodes}  # Cache normalized
    
    try:
        doc = nlp(question)
        
        # Strategy 1: Named entities (PERSON, ORG, GPE, EVENT)
        for ent in doc.ents:
            text = ent.text
            text_norm = normalize(text)
            
            # Exact match (fast)
            if text_norm in graph_nodes_lower:
                entities.add(graph_nodes_lower[text_norm])
                continue
            
            # Fuzzy match for entities NOT found by exact match
            if len(text_norm) >= 2:
                best_match = None
                best_ratio = 0.7  # More relaxed threshold
                min_len = max(1, len(text_norm) - 5)
                max_len = len(text_norm) + 5
                
                for node in graph_nodes:
                    node_norm = normalize(node)
                    if min_len <= len(node_norm) <= max_len:
                        ratio = SequenceMatcher(None, text_norm, node_norm).ratio()
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_match = node
                
                if best_match:
                    entities.add(best_match)
            
            # Fallback: try individual words from multi-word entities
            if not entities and len(text.split()) > 1:
                for word in text.split():
                    word_norm = normalize(word)
                    if len(word_norm) >= 2 and word_norm in graph_nodes_lower:
                        entities.add(graph_nodes_lower[word_norm])
        
        # Strategy 2: Noun chunks (skip if already found many entities)
        if len(entities) < 5:
            for chunk in doc.noun_chunks:
                text = chunk.text
                text_norm = normalize(text)
                
                if len(text_norm) >= 3:
                    # Exact match first
                    if text_norm in graph_nodes_lower:
                        entities.add(graph_nodes_lower[text_norm])
                    else:
                        # Try individual words from chunks
                        words = chunk.text.split()
                        for word in words:
                            word_norm = normalize(word)
                            if len(word_norm) >= 2 and word_norm in graph_nodes_lower:
                                entities.add(graph_nodes_lower[word_norm])
        
        # Strategy 3: Pattern-based extraction for common question patterns
        if len(entities) < 3:
            # Pattern: "the X performer/actor/director/artist/singer"
            pattern_match = re.search(r'the\s+([^?]+?)\s+(?:performer|actor|actress|director|artist|singer|author|film|movie)', question, re.IGNORECASE)
            if pattern_match:
                text = pattern_match.group(1).strip()
                text_norm = normalize(text)
                if text_norm in graph_nodes_lower:
                    entities.add(graph_nodes_lower[text_norm])
                # Try individual words
                for word in text.split():
                    word_norm = normalize(word)
                    if len(word_norm) >= 2 and word_norm in graph_nodes_lower:
                        entities.add(graph_nodes_lower[word_norm])
        
        # Remove duplicates
        return list(entities)
    except Exception as e:
        return []


if __name__ == "__main__":
    start_time = time.time()
    logger = ResultLogger("eval_passages")  # Will auto-add timestamp
    
    logger.log("Loading data...")
    samples = load_musique("dev", max_samples=200)
    graph_path = resolve_graph_output_path(graph_name=TRIPLES_CACHE_NAME, must_exist=False)
    if not graph_path.exists():
        graph_path = resolve_graph_output_path(graph_name=None, must_exist=True)
    G = load_graph(str(graph_path))
    
    # Load passage data
    logger.log("Loading passages...")
    logger.log(f"Triples cache: {TRIPLES_CACHE_NAME}")
    logger.log(f"Graph file: {graph_path}")
    passages = load_passage_data(TRIPLES_CACHE_NAME)
    triple_to_passages, passage_texts, passage_entities, entity_to_passages = build_triple_to_passage_map(passages)
    
    # Log graph and passage statistics
    logger.log(f"Loaded {len(samples)} samples")
    logger.log(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    logger.log(f"Passages: {len(passages)}, Triples: {len(triple_to_passages)}")
    
    logger.log(f"\n{'='*70}")
    logger.log(f"SYSTEM CONFIGURATION")
    logger.log(f"{'='*70}")
    logger.log(f"Retrieval method: Passage-based ranking (HippoRAG style)")
    logger.log(f"Entity extraction strategies:")
    logger.log(f"  1. Spacy NER - exact & fuzzy-relaxed match (similarity >= 0.7, range ±5)")
    logger.log(f"  2. Noun chunks - extract individual words (word-level fallback)")
    logger.log(f"  3. Pattern-based extraction:")
    logger.log(f"     * 'the X performer/actor/director/film' + individual words")
    logger.log(f"  4. Caching strategies:")
    logger.log(f"     * Pre-normalized graph node index (O(1) exact lookup)")
    logger.log(f"     * SequenceMatcher for fuzzy scoring with relaxed thresholds")
    logger.log(f"  [PERF] Normalized node dictionary + fallback word-level extraction")
    logger.log(f"")
    logger.log(f"Ranking metric: Weighted entity coverage + frequency boost")
    logger.log(f"Top-k passages: 10")
    logger.log(f"Neighborhood hops: 4")
    logger.log(f"Weighting scheme:")
    logger.log(f"  - Hop distance weight: hop 0=1.0, hop 1=0.85, hop 2=0.7, hop 3=0.55, hop 4+=0.4")
    logger.log(f"  - Entity frequency boost: 1 entity=1.0x, 2 entities=1.4x, 3+ entities=1.8x")
    logger.log(f"Optimizations applied:")
    logger.log(f"  - BFS expansion to 3 hops for deeper entity neighborhood")
    logger.log(f"  - Weighted scoring by entity distance (closer = higher weight)")
    logger.log(f"  - Passage frequency multiplier for multi-entity coverage")
    logger.log(f"  - [PERF] Pre-computed entity-to-passages cache (O(1) lookup vs O(n^2) nested loop)")
    logger.log(f"")
    
    # DEBUG: First 3 samples
    logger.log("="*70)
    logger.log("DEBUG: First 3 samples")
    logger.log("="*70)
    
    for sample_idx in range(min(3, len(samples))):
        sample = samples[sample_idx]
        question = sample["question"]
        gold = sample["answer"]
        
        logger.log(f"\nSample {sample_idx+1}: {question[:60]}...")
        logger.log(f"Expected answer: {gold}")
        
        query_entities = extract_query_entities(G, question)
        logger.log(f"Query entities: {query_entities}")
        
        if not query_entities:
            logger.log("❌ No query entities found!")
            continue
        
        # Rank passages
        ranked = rank_passages_by_entities(
            G, query_entities, triple_to_passages, passage_entities, passage_texts,
            top_k=10, neighborhood_hops=3, entity_to_passages=entity_to_passages
        )
        
        logger.log(f"Top-3 passages:")
        for rank, (pid, text, score) in enumerate(ranked[:3], 1):
            found, _ = extract_answer_from_passages(gold, [(pid, text, score)])
            marker = "✓" if found else " "
            logger.log(f"  {rank}. [{marker}] Score: {score:.2f}")
            logger.log(f"      {text[:70]}...")
    
    # Main evaluation
    logger.log("\n" + "="*70)
    logger.log("EVALUATION")
    logger.log("="*70)
    
    eval_start = time.time()
    hit_at_top_k = []
    passage_rank_when_hit = []
    hops_performance = Counter()
    hops_counts = Counter()
    skipped = 0
    
    for i, sample in enumerate(samples):
        question = sample["question"]
        gold = sample["answer"]
        hops = sample.get("hops", 1)
        
        hops_counts[hops] += 1
        
        # Extract entities
        query_entities = extract_query_entities(G, question)
        if not query_entities:
            skipped += 1
            continue
        
        # Rank passages
        ranked_passages = rank_passages_by_entities(
            G, query_entities, triple_to_passages, passage_entities, passage_texts,
            top_k=10, neighborhood_hops=3, entity_to_passages=entity_to_passages
        )
        
        # Check if answer in passages
        found, rank = extract_answer_from_passages(gold, ranked_passages)
        
        hit_at_top_k.append(found)
        if found:
            passage_rank_when_hit.append(rank)
            hops_performance[hops] += 1
        
        if (i + 1) % 20 == 0:
            logger.log(f"[{i+1}] Q: {question[:50]}...")
            logger.log(f"     Gold: {gold} | Found: {'✓' if found else '✗'} {f'(rank {rank})' if found else ''}")
    
    eval_time = time.time() - eval_start
    
    # Results
    logger.log(f"\n{'='*70}")
    logger.log(f"OVERALL RESULTS")
    logger.log(f"{'='*70}")
    logger.log(f"Coverage (evaluated): {len(hit_at_top_k)}/{len(samples)} ({len(hit_at_top_k)*100/len(samples):.1f}%)")
    logger.log(f"Skipped: {skipped} (no query entities)")
    
    if hit_at_top_k:
        hit_count = sum(hit_at_top_k)
        total = len(hit_at_top_k)
        
        logger.log(f"\nFound in top-10 passages: {hit_count}/{total} ({hit_count*100/total:.1f}%)")
        
        # Rank statistics
        if passage_rank_when_hit:
            avg_rank = sum(passage_rank_when_hit) / len(passage_rank_when_hit)
            min_rank = min(passage_rank_when_hit)
            max_rank = max(passage_rank_when_hit)
            median_rank = statistics.median(passage_rank_when_hit)
            
            logger.log(f"\nRank Statistics (when answer found):")
            logger.log(f"  Average rank: {avg_rank:.2f}")
            logger.log(f"  Median rank: {median_rank:.1f}")
            logger.log(f"  Min rank: {min_rank}")
            logger.log(f"  Max rank: {max_rank}")
            
            # Rank distribution
            rank_dist = Counter(passage_rank_when_hit)
            logger.log(f"\nRank Distribution:")
            for rank in sorted(rank_dist.keys()):
                count = rank_dist[rank]
                pct = count * 100 / len(passage_rank_when_hit)
                logger.log(f"  Rank {rank}: {count} ({pct:.1f}%)")
    
    # Hop analysis
    logger.log(f"\n{'='*70}")
    logger.log(f"HOP-BY-HOP ANALYSIS")
    logger.log(f"{'='*70}")
    for hop in sorted(hops_counts.keys()):
        count = hops_counts[hop]
        correct = hops_performance[hop]
        acc = correct * 100 / count if count > 0 else 0
        logger.log(f"{hop}-hop: {correct}/{count} ({acc:.1f}%)")
    
    # Timing
    total_time = time.time() - start_time
    logger.log(f"\n{'='*70}")
    logger.log(f"TIMING INFORMATION")
    logger.log(f"{'='*70}")
    logger.log(f"Evaluation time: {eval_time:.2f}s")
    logger.log(f"Total time: {total_time:.2f}s")
    logger.log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save
    logger.save()
