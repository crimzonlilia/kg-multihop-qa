"""Quick test: rebuild graph with 200 samples only + evaluate."""

import json
import sys
import re
from pathlib import Path
from collections import defaultdict, Counter
import time
import networkx as nx
import tempfile

# Limit to 200 samples
DATA_LIMIT = 2

# Import pipeline
from src.extraction import extract_triples_batch
from src.extraction.schema import DEFAULT_RELATION_SCHEMA
from src.graph.build_graph import build_graph, merge_graphs, load_graph, normalize
from src.graph.entity_linking import link_entities_fast
from src.graph.graph_utils import graph_stats, save_graph
from src.data.musique_loader import load_musique
from src.retrieval.pagerank import rank_passages_by_ppr


def normalize_answer(s: str) -> str:
    """Aggressive normalization for answer matching"""
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def evaluate_on_graph(graph_obj, samples, passages_data, eval_ks=[1, 5, 10]):
    """Evaluate retrieval using Personalized PageRank (AiC@k metric)"""
    print("\n" + "=" * 70)
    print("EVALUATION ON SAMPLES (Personalized PageRank)")
    print("=" * 70)
    
    # Build passage mapping from cached data
    print("Building triple-to-passage mapping...")
    triple_to_passages, passage_texts, passage_entities, entity_to_passages = build_triple_to_passage_map_cached(passages_data)
    print(f"✓ Built mapping: {len(passage_texts)} passages, {len(triple_to_passages)} unique triples")
    
    # Track hits at each cutoff
    hits_at_k = {k: [] for k in eval_ks}
    hops_performance = Counter()
    hops_counts = Counter()
    skipped = 0
    result_counts = Counter()
    
    eval_start = time.time()
    
    max_k = max(eval_ks) * 2
    for i, sample in enumerate(samples):
        question = sample["question"]
        gold = sample["answer"]
        hops = sample.get("hops", 1)
        
        hops_counts[hops] += 1
        
        # Rank passages using PPR (entity extraction is done inside)
        ranked_passages = rank_passages_by_ppr(
            graph_obj, question,
            triple_to_passages, passage_texts,
            top_k=max_k,
            alpha=0.85,
            ppr_type="standard"
        )
        
        if not ranked_passages:
            skipped += 1
            continue
        
        result_counts[len(ranked_passages)] += 1
        
        # Check hits at each cutoff (AiC@k: Answer in Context metric)
        # Check if gold_answer appears as substring in passage
        for k in eval_ks:
            is_hit = False
            for passage_id, passage_text, score in ranked_passages[:k]:
                if gold.lower() in passage_text.lower():
                    is_hit = True
                    break
            
            hits_at_k[k].append(is_hit)
            if is_hit:
                hops_performance[hops] += 1
        
        # Debug samples
        if i < 3 or ((i + 1) % 50 == 0):
            hit_str = f"AiC@10: {'✓' if hits_at_k[10][-1] else '✗'}"
            print(f"[{i+1:3d}/{len(samples)}] {question[:50]:50s} | {hit_str} ({len(ranked_passages)} passages)")
            if i < 3 and ranked_passages:
                print(f"        Question: {question}")
                print(f"        Gold answer: {gold}")
                print(f"        Top passage: {passage_texts[ranked_passages[0][0]][:100]}...")
    
    eval_time = time.time() - eval_start
    
    # Results
    print(f"\n{'='*70}")
    print(f"PERSONALIZED PAGERANK RESULTS (AiC@k Metric)")
    print(f"{'='*70}")
    print(f"Evaluated: {len(hits_at_k[eval_ks[0]])}/{len(samples)} samples")
    print(f"Skipped:   {skipped} (no passages retrieved)")
    
    print(f"\nMetrics:")
    results = {}
    for k in sorted(eval_ks):
        if hits_at_k[k]:
            accuracy = sum(hits_at_k[k]) * 100 / len(hits_at_k[k])
            print(f"  AiC@{k:2d}: {sum(hits_at_k[k]):3d}/{len(hits_at_k[k]):3d} ({accuracy:5.1f}%)")
            results[f'aic@{k}'] = accuracy
        else:
            print(f"  AiC@{k:2d}: N/A")
    
    # Hop analysis
    if hops_counts:
        print(f"\n{'='*70}")
        print(f"HOP-BY-HOP BREAKDOWN (AiC@10)")
        print(f"{'='*70}")
        for hop in sorted(hops_counts.keys()):
            count = hops_counts[hop]
            correct = hops_performance[hop]
            acc = correct * 100 / count if count > 0 else 0
            print(f"  {hop}-hop: {correct:3d}/{count:3d} ({acc:5.1f}%)")
    
    print(f"\nEvaluation time: {eval_time:.2f}s")
    
    return {
        'hits_at_k': hits_at_k,
        'results': results,
        'skipped': skipped,
        'eval_time': eval_time,
        'result_counts': result_counts
    }


def build_triple_to_passage_map_cached(passages_data):
    """Build mapping from cached passages data (from triples.json)"""
    triple_to_passages = defaultdict(list)
    passage_texts = {}
    passage_entities = {}
    entity_to_passages = defaultdict(set)
    
    for passage_id, data in enumerate(passages_data):
        passage_text = data.get('passage', '')
        passage_texts[passage_id] = passage_text
        
        # Extract entity texts
        entities = data.get('entities', [])
        entity_texts = [e.get('text') for e in entities if isinstance(e, dict) and 'text' in e]
        passage_entities[passage_id] = entity_texts
        
        # Build entity cache
        for entity_text in entity_texts:
            entity_norm = entity_text.lower()
            entity_to_passages[entity_norm].add(passage_id)
        
        # Map triples to passages
        triples = data.get('triples', [])
        if triples:
            for triple in triples:
                if isinstance(triple, dict) and 'subject' in triple:
                    key = (
                        triple.get('subject', '').lower(),
                        triple.get('relation', '').lower(),
                        triple.get('object', '').lower()
                    )
                    triple_to_passages[key].append(passage_id)
    
    return triple_to_passages, passage_texts, passage_entities, entity_to_passages


# ALWAYS rebuild from 200 samples to avoid old cache conflicts
print("\n🔄 Building fresh graph from 200 samples...")
kg_path = Path("data/processed/kg.pkl")

# Clear old cache để avoid merging với data cũ
cache_path = Path("data/cache/triples.json")
if cache_path.exists():
    print(f"🗑️  Clearing old cache at {cache_path}...")
    cache_path.unlink()

# Load data
print("\n📚 Loading 200 samples...")
musique_path = Path("data/raw/musique/data/musique_ans_v1.0_train.jsonl")
passages_list = []
line_count = 0
para_count = 0

with open(musique_path) as f:
    for idx, line in enumerate(f):
        if idx >= DATA_LIMIT:
            break
        line_count += 1
        data = json.loads(line)
        if "paragraphs" in data:
            for para in data["paragraphs"]:
                if isinstance(para, dict) and "paragraph_text" in para:
                    passages_list.append(para["paragraph_text"])
                    para_count += 1

passages = passages_list
print(f"✓ Loaded {len(passages)} passages from {line_count} lines ({para_count} paragraphs with text)")

# Extract triples - dynamic schema để nhanh, deduplicate=False để keep per-passage data, save_cache_to_disk=False để ko merge cũ
print("\n🔗 Pass 1: Extracting triples (use_dynamic=True, deduplicate=False, no cache merge)...")
t0_extract = time.time()
results = extract_triples_batch(passages, relation_schema=DEFAULT_RELATION_SCHEMA, use_dynamic=True, k=8, skip_cache=True, deduplicate=False, save_cache_to_disk=False)
extract_time = time.time() - t0_extract

total_triples = sum(len(r['triples']) for r in results)
print(f"✓ Extracted {total_triples} triples in {extract_time:.2f}s")

# Build graphs
print("\n🌐 Pass 2: Building graphs...")
t0_build = time.time()
graphs = []
for result in results:
    triples = result.get('triples') or []
    if not triples:
        graphs.append(nx.DiGraph())
    else:
        g = build_graph(triples, result.get('entities', []))
        graphs.append(g)
build_time = time.time() - t0_build

# Merge graphs
print("\n🔀 Pass 3: Merging...")
t0_merge = time.time()
merged_graph = merge_graphs(graphs)
merge_time = time.time() - t0_merge

# Entity linking
print("\n🔗 Pass 4: Entity linking...")
t0_link = time.time()
linked_graph = link_entities_fast(merged_graph)
linking_time = time.time() - t0_link

print(f"✓ Graph ready: {linked_graph.number_of_nodes()} nodes, {linked_graph.number_of_edges()} edges")
save_graph(linked_graph, str(kg_path))
print(f"✓ Saved to {kg_path}")

# Load samples for evaluation
print("\n📋 Loading samples for evaluation...")
eval_samples = load_musique("dev", max_samples=2)
print(f"✓ Loaded {len(eval_samples)} samples")

# Use fresh results (just built) for passage mapping
print("\n📖 Building passage mapping from fresh extraction...")

passages_data = []
for result in results:
    passage = result.get('passage') or ""  # Handle None
    passages_data.append({
        'passage': passage,
        'entities': result.get('entities', []),
        'triples': result.get('triples', [])
    })
total_triples = sum(len(r['triples']) for r in passages_data if r['triples'])
print(f"✓ Prepared {len(passages_data)} passages (total triples: {total_triples})")

# Evaluate with passage ranking
eval_results = evaluate_on_graph(linked_graph, eval_samples, passages_data)

# Final summary
print("\n" + "=" * 70)
print("EVALUATION SUMMARY")
print("=" * 70)
print(f"Pipeline: Extraction → Graph Build → Entity Linking → PPR Retrieval")
print(f"Graph: {linked_graph.number_of_nodes():5d} nodes, {linked_graph.number_of_edges():5d} edges")
print(f"Evaluation time: {eval_results['eval_time']:7.2f}s")
print(f"\nRetrieval Metrics (Answer-in-Context):")
for metric, value in sorted(eval_results['results'].items()):
    print(f"  {metric.upper():8s}: {value:6.1f}%")
print(f"\nSamples: {len(eval_results['hits_at_k'][1])} / {len(eval_samples)} queries")
if eval_results['skipped'] > 0:
    print(f"Skipped:  {eval_results['skipped']} (PPR returned no passages)")

print("\n" + "=" * 70)
print("✓ EVALUATION COMPLETE - PPR-based Retrieval Test")
print("=" * 70)
