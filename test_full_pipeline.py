"""
Full end-to-end pipeline test from data loading to evaluation.
Tests with DATA_LIMIT samples. Evaluation metric: AiC@k (Answer-in-Context).
"""

import json
import sys
import time
import ctypes
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Prevent Windows from sleeping while pipeline runs
# ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
_ES_CONTINUOUS        = 0x80000000
_ES_SYSTEM_REQUIRED   = 0x00000001
if sys.platform == "win32":
    ctypes.windll.kernel32.SetThreadExecutionState(
        _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED
    )

from src.extraction.extract_triples import extract_triples_batch, DEFAULT_RELATION_SCHEMA
from src.graph.build_graph import build_graph, merge_graphs, normalize
from src.retrieval.pagerank import rank_passages_by_ppr
from src.data.musique_loader import load_musique
import networkx as nx

print("\n" + "=" * 80)
print("FULL PIPELINE TEST: Data Loading -> Extraction -> Graph Building -> Evaluation")
print("=" * 80)

# -- Configuration -----
DATA_LIMIT      = None  # None = full dataset (2417 QA pairs, ~48K passages)
SAMPLE_QUERIES  = None  # None = evaluate all queries
TOP_K           = 10    # retrieve top-k passages per query
PPR_TYPE        = "hipporag"  # "standard" | "hipporag" | "fast"
RELATION_DISCOVERY = True   # Pass 0: discover schema from corpus before extraction
PER_QUERY_GRAPH = True  # build a separate mini-graph per query to avoid
                        # entity ambiguity from unrelated QA pairs in the corpus
EXTRACTION_BATCH_SIZE = 16  # RTX 2060-safe default
REUSE_EXTRACTION_CACHE = True  # huge speed-up on reruns of the same corpus
EXTRACT_ENTITY_TYPES = True    # set False for faster but slightly noisier extraction
# -----------------------

# -- Step 1: Load data ---
print("\nSTEP 1: Loading MusiQue dataset...")
t0 = time.time()

dev_data = load_musique("dev", max_samples=DATA_LIMIT)
print(f"  Loaded {len(dev_data)} QA pairs from dev set")

# Build passage store  {passage_id -> passage_text}
# MuSiQue format: each QA pair has 'paragraphs' list with 'paragraph_text' + 'is_supporting'
passages_by_id   = {}   # {passage_id: text}
query_meta       = {}   # {query_id: {question, answer, gold_passage_ids}}

for qa in dev_data:
    qid      = qa.get("id", f"q_{len(query_meta)}")
    question = qa.get("question", "")
    answer   = qa.get("answer", "")
    gold_ids = []

    # loader wraps fields as: {"text": ..., "idx": ..., "is_supporting": ...}
    for i, para in enumerate(qa.get("passages", [])):
        p_text = para.get("text", "")
        p_id   = f"{qid}_p{i}"

        if p_text:
            passages_by_id[p_id] = p_text

        if para.get("is_supporting", False):
            gold_ids.append(p_id)

    query_meta[qid] = {
        "question":         question,
        "answer":           answer,
        "gold_passage_ids": gold_ids,
    }

load_time = time.time() - t0
print(f"  Unique passages   : {len(passages_by_id)}")
print(f"  Queries with gold : {sum(1 for m in query_meta.values() if m['gold_passage_ids'])}")
print(f"  Time              : {load_time:.2f}s")

# -- Step 2: Extract triples ---
print("\nSTEP 2: Extracting triples from passages...")
t0 = time.time()

passage_ids   = list(passages_by_id.keys())
passage_texts = list(passages_by_id.values())

# -- Step 0: Relation Discovery (Pass 0 - adaptive schema) ---
if RELATION_DISCOVERY:
    from src.extraction.relation_discovery import run_discovery
    from pathlib import Path as _Path
    _cache = _Path("data/cache/expanded_schema.json")
    if _cache.exists():
        import json as _json
        with open(_cache) as _f:
            extraction_schema = _json.load(_f)
        print(f"\nSTEP 0: Loaded cached schema ({len(extraction_schema)} relations) from {_cache}")
    else:
        print("\nSTEP 0: Running relation discovery on corpus...")
        extraction_schema = run_discovery(
            passage_texts,
            min_freq=3,
            min_cluster_freq=5,
            distance_threshold=0.35,
            top_k=80,
        )
        print(f"  Expanded schema: {len(extraction_schema)} relations")
else:
    extraction_schema = DEFAULT_RELATION_SCHEMA

# Clear GPU cache before extraction (58 labels × batches need headroom)
import os, torch as _torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if _torch.cuda.is_available():
    _torch.cuda.empty_cache()

print(f"  Extraction cache  : {'ON' if REUSE_EXTRACTION_CACHE else 'OFF'}")
print(f"  Entity filtering  : {'ON' if EXTRACT_ENTITY_TYPES else 'FAST MODE'}")
results_batch = extract_triples_batch(
    passage_texts,
    relation_schema=extraction_schema,
    use_dynamic=False,   # dynamic schema grouping severely hurts extraction quality
    skip_cache=not REUSE_EXTRACTION_CACHE,
    deduplicate=False,   # ← keep per-passage data intact
    batch_size=EXTRACTION_BATCH_SIZE,
    save_cache_to_disk=REUSE_EXTRACTION_CACHE,
    extract_entities=EXTRACT_ENTITY_TYPES,
)

# Map extraction results back to passage IDs
triples_by_passage = {}   # {passage_id: [triples]}
entities_by_passage = {}  # {passage_id: [entities]}

for i, result in enumerate(results_batch):
    p_id = passage_ids[i]
    triples_by_passage[p_id]  = result.get("triples", [])
    entities_by_passage[p_id] = result.get("entities", [])

extract_time  = time.time() - t0
total_triples = sum(len(t) for t in triples_by_passage.values())
non_empty     = sum(1 for t in triples_by_passage.values() if t)
avg           = total_triples / len(triples_by_passage) if triples_by_passage else 0

print(f"  Total triples     : {total_triples}")
print(f"  Passages with 1+  : {non_empty}/{len(triples_by_passage)}")
print(f"  Avg triples/pass  : {avg:.2f}")
print(f"  Time              : {extract_time:.2f}s")

assert len(results_batch) == len(passage_ids), \
    f"MISMATCH: {len(results_batch)} results vs {len(passage_ids)} passages"

# -- Step 3: Build per-passage graphs ---
print("\nSTEP 3: Building knowledge graphs per passage...")
t0 = time.time()

passage_graphs = {}
for p_id, triples in triples_by_passage.items():
    try:
        passage_graphs[p_id] = build_graph(
            triples,
            add_cooccurrence_edges=True  # Enable co-occurrence edges to improve connectivity
        )
    except Exception as e:
        print(f"  WARNING: {p_id}: {e}")
        passage_graphs[p_id] = nx.DiGraph()

build_time = time.time() - t0
print(f"  Graphs built      : {len(passage_graphs)}")
print(f"  Time              : {build_time:.2f}s")

# -- Step 4: Merge into unified KG ---
print("\nSTEP 4: Merging all graphs...")
t0 = time.time()

final_graph = merge_graphs(list(passage_graphs.values()))
merge_time  = time.time() - t0

print(f"  Nodes             : {final_graph.number_of_nodes()}")
print(f"  Edges             : {final_graph.number_of_edges()}")
print(f"  Time              : {merge_time:.2f}s")

# -- Build triple_to_passages lookup ---
# rank_passages_by_ppr needs {(s, r, o): [passage_ids]} to map nodes → passages
triple_to_passages = defaultdict(list)   # {(s,r,o): [p_id]}
passage_text_map   = passages_by_id      # alias for clarity

for p_id, triples in triples_by_passage.items():
    for t in triples:
        key = (
            normalize(t.get("subject", "")),
            t.get("relation", ""),
            normalize(t.get("object", "")),
        )
        if p_id not in triple_to_passages[key]:
            triple_to_passages[key].append(p_id)

print(f"\n  triple_to_passages: {len(triple_to_passages)} unique triples mapped")

# Build indexed node→passages lookup (O(1) per node vs O(all_triples) scan)
# Also include entity mentions for passages that have entities but no triples
node_to_passages = defaultdict(set)
for (s, r, o), pids in triple_to_passages.items():
    node_to_passages[s].update(pids)
    node_to_passages[o].update(pids)
for p_id, entities in entities_by_passage.items():
    for ent in entities:
        ent_norm = normalize(ent.get("text", "") if isinstance(ent, dict) else str(ent))
        if ent_norm and len(ent_norm) >= 3:
            node_to_passages[ent_norm].add(p_id)
print(f"  node_to_passages  : {len(node_to_passages)} unique nodes indexed")

# -- Step 4b: Build node embeddings + fact embeddings + passage embeddings ---
print("\nSTEP 4b: Building embeddings for semantic retrieval...")
from src.retrieval.pagerank import (
    build_node_embeddings,
    build_fact_embeddings,
    build_passage_embeddings,
    extract_entities_from_question,
)

t0 = time.time()
node_embeddings = build_node_embeddings(final_graph)
print(f"  Node embeddings  : {len(node_embeddings[0])} nodes in {time.time()-t0:.2f}s")

t0 = time.time()
fact_embeddings = build_fact_embeddings(triple_to_passages)
print(f"  Fact embeddings  : {len(fact_embeddings[0])} facts in {time.time()-t0:.2f}s")

t0 = time.time()
passage_embeddings = build_passage_embeddings(passage_text_map)
print(f"  Passage embeddings: {len(passage_embeddings[0])} passages in {time.time()-t0:.2f}s")

# -- Step 5: PPR Retrieval + AiC@k Evaluation ---
print(f"\nSTEP 5: Retrieval + AiC@k eval (top_k={TOP_K}, ppr={PPR_TYPE})...")
print(f"  Graph stats: {final_graph.number_of_nodes()} nodes, {final_graph.number_of_edges()} edges")
print(f"  triple_to_passages: {len(triple_to_passages)} unique triples")
print()

# Load spaCy for entity extraction debugging
import spacy
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

t0 = time.time()

hits_at = {1: 0, 5: 0, 10: 0}
total_queries = 0
results       = []

sample_qids = list(query_meta.keys()) if SAMPLE_QUERIES is None else list(query_meta.keys())[:SAMPLE_QUERIES]

for qid in sample_qids:
    meta     = query_meta[qid]
    question = meta["question"]
    answer   = meta["answer"].strip().lower()
    gold_ids = meta["gold_passage_ids"]

    if not question or not answer:
        continue

    # -- Per-query or global graph ----------------------------------------
    if PER_QUERY_GRAPH:
        # Filter to only this question's candidate passages (avoids cross-QA
        # node contamination that dilutes PPR scores for the correct passages)
        q_passage_texts = {pid: txt for pid, txt in passage_text_map.items()
                           if pid.startswith(qid + "_p")}
        q_t2p = defaultdict(list)
        for key, pids in triple_to_passages.items():
            local = [pid for pid in pids if pid in q_passage_texts]
            if local:
                q_t2p[key] = local
        q_n2p = defaultdict(set)
        for (s, r, o), pids in q_t2p.items():
            q_n2p[s].update(pids)
            q_n2p[o].update(pids)
        q_graphs = [passage_graphs[pid] for pid in q_passage_texts if pid in passage_graphs]
        q_graph  = merge_graphs(q_graphs) if q_graphs else final_graph
        q_node_emb = build_node_embeddings(q_graph)
        q_fact_emb = build_fact_embeddings(q_t2p)
        q_pass_emb = build_passage_embeddings(q_passage_texts)
        use_G, use_t2p, use_n2p, use_texts, use_emb, use_femb, use_pemb = (
            q_graph, q_t2p, q_n2p, q_passage_texts,
            q_node_emb, q_fact_emb, q_pass_emb,
        )
    else:
        use_G, use_t2p, use_n2p, use_texts, use_emb, use_femb, use_pemb = (
            final_graph, triple_to_passages, node_to_passages,
            passage_text_map, node_embeddings, fact_embeddings, passage_embeddings,
        )

    # use rank_passages_by_ppr (correct function)
    try:
        ranked = rank_passages_by_ppr(
            G                  = use_G,
            question           = question,
            triple_to_passages = use_t2p,
            passage_texts      = use_texts,
            node_to_passages   = use_n2p,
            node_embeddings    = use_emb,
            fact_embeddings    = use_femb,
            passage_embeddings = use_pemb,
            top_k              = TOP_K,
            alpha              = 0.85,
            ppr_type           = PPR_TYPE,
            nlp                = nlp,
        )
        # DEBUG first 3 queries: show which entities were found
        if total_queries < 3:
            seed_ents = extract_entities_from_question(question, use_G, nlp, node_embeddings=use_emb)
            print(f"  Q: {question[:70]}")
            print(f"     Entities(embed): {seed_ents}")
            print(f"     Graph size: {use_G.number_of_nodes()} nodes, {use_G.number_of_edges()} edges")
    except Exception as e:
        print(f"  ERROR in rank_passages_by_ppr: {e}")
        ranked = []
    
    # ranked = [(passage_id, passage_text, score), ...]
    if not ranked:
        print(f"  [NO RESULTS] {question[:60]}...")
        continue

    # AiC@k: does gold answer appear as substring in top-k passage texts?
    found_at = None
    for rank_pos, (p_id, p_text, score) in enumerate(ranked, start=1):
        if answer in p_text.lower():
            found_at = rank_pos
            break

    if found_at is not None:
        if found_at <= 1:  hits_at[1]  += 1
        if found_at <= 5:  hits_at[5]  += 1
        if found_at <= 10: hits_at[10] += 1

    total_queries += 1
    results.append({
        "query_id":    qid,
        "question":    question,
        "answer":      answer,
        "found_at":    found_at,
        "gold_ids":    gold_ids,
        "ranked_top5": [(pid, sc) for pid, _, sc in ranked[:5]],
    })

    status = f"found@{found_at}" if found_at else "MISS"
    if total_queries <= 5:  # Print first 5 results
        print(f"  [{status:>8}]  {question[:60]}...")

retrieval_time = time.time() - t0

# -- Step 6: Summary ---
print("\n" + "=" * 80)
total_time = load_time + extract_time + build_time + merge_time + retrieval_time

print("\nTiming Breakdown:")
print(f"  Loading        : {load_time:7.2f}s ({load_time/total_time*100:5.1f}%)")
print(f"  Extraction     : {extract_time:7.2f}s ({extract_time/total_time*100:5.1f}%)")
print(f"  Build graphs   : {build_time:7.2f}s ({build_time/total_time*100:5.1f}%)")
print(f"  Merge          : {merge_time:7.2f}s ({merge_time/total_time*100:5.1f}%)")
print(f"  Retrieval+eval : {retrieval_time:7.2f}s ({retrieval_time/total_time*100:5.1f}%)")
print(f"  {'-'*40}")
print(f"  TOTAL          : {total_time:7.2f}s")

print(f"\nEvaluation (AiC@k) - {total_queries} queries:")
for k in [1, 5, 10]:
    pct = hits_at[k] / total_queries * 100 if total_queries else 0
    print(f"  AiC@{k:2d} : {hits_at[k]}/{total_queries} ({pct:.1f}%)")

full_passages_count = 22316
estimated_time = extract_time * (full_passages_count / max(len(passage_texts), 1))
print(f"\nExtrapolated extraction time for ~{full_passages_count} passages:")
print(f"  ~{estimated_time/60:.1f} minutes ({estimated_time/3600:.2f} hours)")

print("\n" + "=" * 80)
print("FULL PIPELINE TEST COMPLETE")
print("=" * 80 + "\n")

# -- Save results -----
timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = f"full_pipeline_results_{timestamp}.json"

with open(results_file, "w") as f:
    json.dump({
        "timestamp": timestamp,
        "config": {
            "data_limit":     DATA_LIMIT,
            "sample_queries": SAMPLE_QUERIES,
            "top_k":          TOP_K,
            "ppr_type":       PPR_TYPE,
        },
        "timing": {
            "load":       load_time,
            "extract":    extract_time,
            "build":      build_time,
            "merge":      merge_time,
            "retrieval":  retrieval_time,
            "total":      total_time,
        },
        "evaluation": {
            "total_queries": total_queries,
            "hits_at_1":     hits_at[1],
            "hits_at_5":     hits_at[5],
            "hits_at_10":    hits_at[10],
            "aic_at_1":      hits_at[1]  / total_queries if total_queries else 0,
            "aic_at_5":      hits_at[5]  / total_queries if total_queries else 0,
            "aic_at_10":     hits_at[10] / total_queries if total_queries else 0,
        },
        "results": results,
    }, f, indent=2)

print(f"Results saved to: {results_file}\n")

# Restore normal Windows sleep behavior
if sys.platform == "win32":
    ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
