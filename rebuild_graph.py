#!/usr/bin/env python3
"""
REBUILD GRAPH WITH EXPANDED SCHEMA
=====================================

QUICK TEST (200 samples): ~1-2 min
  → Good for testing if fixes work

FULL BUILD (all samples): ~15-30 min  
  → For production evaluation

Change QUICK_TEST = False to build full dataset
"""

import sys, json, shutil, gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from src.data.musique_loader import load_musique, get_all_passages
from src.extraction.relation_discovery import run_discovery
from src.extraction.extract_triples import extract_triples_batch_dynamic, extract_triples_batch, build_schema, get_extractor, clear_extractor_cache
from src.graph.build_graph import build_graph, save_graph, graph_stats
from gliner2 import GLiNER2
import time
import psutil

GRAPH_PATH = "data/processed/kg.pkl"
BACKUP_PATH = "data/processed/kg.pkl.backup"
PROGRESS_FILE = Path("data/cache/progress.json")

# Memory optimization
BATCH_SIZE = 4000  # GPU with fp16: can handle 4x larger batches (RTX 2060 6GB)
ENABLE_GC = True  # Force garbage collection between batches

def get_memory_usage():
    """Get current memory usage in MB"""
    return psutil.Process().memory_info().rss / 1024 / 1024

def load_progress():
    """Load progress checkpoint"""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"last_batch": 0, "total_passages_processed": 0}

def save_progress(batch_num, total_processed):
    """Save progress checkpoint"""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump({
            "last_batch": batch_num,
            "total_passages_processed": total_processed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f)

def reset_progress():
    """Reset progress checkpoint (start fresh)"""
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
        print("✓ Progress file cleared - will start from beginning")

def load_cached_triples():
    """Load triples from existing cache (for graph-only mode)"""
    cache_path = Path("data/cache/triples.json")
    if not cache_path.exists():
        return None
    
    print("\n📂 Loading triples from cache...")
    with open(cache_path) as f:
        data = json.load(f)
    
    all_triples = []
    all_entities = []
    for item in data:
        if isinstance(item, dict):
            if 'triples' in item:
                all_triples.extend(item['triples'])
            if 'entities' in item:
                all_entities.extend(item['entities'])
    
    print(f"   ✓ Loaded {len(all_triples)} triples from {len(data)} passages")
    return all_triples, all_entities

def process_passages_in_batches(passages, relation_schema, extractor, batch_size=BATCH_SIZE):
    """Extract triples in batches to avoid memory spike - REUSE model"""
    all_results = []
    total_passages = len(passages)
    
    # Load progress checkpoint
    progress = load_progress()
    last_batch = progress["last_batch"]
    
    print(f"\n   Processing {total_passages} passages in batches of {batch_size}...")
    print(f"   Initial memory: {get_memory_usage():.0f}MB")
    
    # Calculate starting index based on last completed batch
    start_idx = last_batch * batch_size
    if start_idx > 0:
        print(f"   📍 Resuming from batch {last_batch + 1} (skipping first {start_idx} passages)")
    
    for i in range(start_idx, total_passages, batch_size):
        batch = passages[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_passages + batch_size - 1) // batch_size
        
        print(f"   [{batch_num}/{total_batches}] Processing {len(batch)} passages... ", end='', flush=True)
        
        # ← Extract with dynamic schema grouping (optimized batch processing)
        batch_results = extract_triples_batch_dynamic(
            batch, 
            relation_schema=relation_schema, 
            extractor=extractor,
            batch_size=min(64, len(batch)),  # Smaller batch size for dynamic (embeddings per batch)
            k=8,  # Select top-8 relations per passage
            skip_cache=True,  # Skip loading old cache within batch
            save_cache=True   # ← Save incrementally after each batch
        )
        all_results.extend(batch_results)
        
        # Save progress after each batch
        save_progress(batch_num, start_idx + len(all_results))
        
        mem = get_memory_usage()
        print(f"Memory: {mem:.0f}MB ✓")
        
        # Force garbage collection between batches
        if ENABLE_GC:
            del batch_results
            gc.collect()
    
    return all_results

def main(graph_only=False):
    print("=" * 70)
    if graph_only:
        print("BUILDING GRAPH FROM CACHED TRIPLES (GRAPH-ONLY MODE)")
        # Reset progress immediately for graph-only mode
        save_progress(0, 0)
    else:
        print("REBUILDING KNOWLEDGE GRAPH WITH EXPANDED SCHEMA")
    print("=" * 70)
    
    # Check if resuming
    progress = load_progress()
    if progress["last_batch"] > 0 and not graph_only:
        print(f"\n⏸️  RESUMING from batch {progress['last_batch'] + 1}")
        print(f"   Already processed: {progress['total_passages_processed']} passages")
        print(f"   Last updated: {progress.get('timestamp', 'unknown')}\n")
    
    print(f"  Optimization settings:")
    print(f"   Batch size: {BATCH_SIZE} passages per batch")
    print(f"   GC enabled: {ENABLE_GC}")
    print(f"   Starting memory: {get_memory_usage():.0f}MB")
    
    # Backup old graph
    if Path(GRAPH_PATH).exists():
        print(f"\n🔄 Backing up old graph...")
        shutil.copy(GRAPH_PATH, BACKUP_PATH)
        print(f"   Saved to: {BACKUP_PATH}")
    
    # SKIP extraction if graph-only mode
    if graph_only:
        print(f"\n📂 Loading cached triples...")
        cached_data = load_cached_triples()
        if cached_data is None:
            print(f"   ❌ No cached triples at data/cache/triples.json")
            print(f"   Run full rebuild first: python rebuild_graph.py")
            return
        all_triples, all_entities = cached_data
    else:
        # Load data
        print(f"\n📚 Loading data...")
        QUICK_TEST = False  # ← Set to False for FULL dataset (~15-30 min)
        max_samples = 200 if QUICK_TEST else None
        
        start = time.time()
        samples = load_musique("dev", max_samples=max_samples)
        passages = get_all_passages(samples, supporting_only=False)
        print(f"   Loaded {len(samples)} samples, {len(passages)} unique passages")
        if QUICK_TEST:
            print(f"   💡 Quick test mode (200 samples). Change QUICK_TEST=False for full dataset.")
        elapsed = time.time() - start
        print(f"   Time: {elapsed:.1f}s")
        
        # Pass 1: Relation Discovery
        print(f"\n🔍 Pass 1: Discovering relations...")
        start = time.time()
        
        with open("src/schemas/relations.json") as f:
            current_schema = json.load(f)
        print(f"   Current schema has {len(current_schema)} relations")
        
        try:
            expanded_schema = run_discovery(passages, min_cluster_freq=3)
            print(f"   Discovered schema has {len(expanded_schema)} relations")
        except Exception as e:
            print(f"   ⚠️  Discovery failed: {e}, using current schema")
            expanded_schema = current_schema
        
        elapsed = time.time() - start
        print(f"   Time: {elapsed:.1f}s")
        
        # Pass 2: Extract Triples
        print(f"\n🔗 Pass 2: Extracting triples with expanded schema...")
        start = time.time()
        
        print(f"   Loading GLiNER2 model (cached)...")
        extractor = get_extractor("fastino/gliner2-base-v1")
        print(f"   ✓ Model ready")
        
        results = process_passages_in_batches(passages, expanded_schema, extractor, batch_size=BATCH_SIZE)
        
        total_triples = sum(len(r.get('triples', [])) for r in results)
        total_entities = sum(len(r.get('entities', [])) for r in results)
        print(f"   Extracted {total_triples} triples from {total_entities} entities")
        
        new_relations = set()
        for r in results:
            for t in r.get('triples', []):
                rel = t.get('relation', '')
                if rel:
                    new_relations.add(rel)
        
        expanded_rels = ['spouse', 'distributed_by', 'directed_by', 'owner_of', 'capital_of']
        found_expanded = [r for r in expanded_rels if r in new_relations]
        print(f"   ✓ Found new relations: {found_expanded}")
        
        elapsed = time.time() - start
        print(f"   Time: {elapsed:.1f}s")
        
        all_triples = [t for r in results for t in r.get('triples', [])]
        all_entities = [e for r in results for e in r.get('entities', [])]
    
    # Build Graph
    print(f"\n🏗️  Building graph...")
    start = time.time()
    print(f"   Memory before: {get_memory_usage():.0f}MB")
    
    if not graph_only:
        # Clear results to free memory
        del results
        gc.collect()
    
    print(f"   Memory after prep: {get_memory_usage():.0f}MB")
    
    # Enable co-occurrence edges (optimized - max 10 per passage)
    G = build_graph(all_triples, all_entities, score_threshold=0.0, add_cooccurrence_edges=True)
    
    print(f"\n   Graph Statistics:")
    stats = graph_stats(G)
    print(f"   Nodes: {G.number_of_nodes()}")
    print(f"   Edges: {G.number_of_edges()}")
    print(f"   Memory after build: {get_memory_usage():.0f}MB")
    
    elapsed = time.time() - start
    print(f"   Time: {elapsed:.1f}s")
    
    # Save Graph
    print(f"\n💾 Saving graph...")
    save_graph(G, GRAPH_PATH)
    print(f"   Saved to: {GRAPH_PATH}")
    
    print("\n" + "=" * 70)
    print(" GRAPH REBUILD COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Final memory: {get_memory_usage():.0f}MB")
    
    # Clear progress after successful completion
    save_progress(0, 0)
    print("✓ Progress reset (ready for next run)")
    
    if not graph_only:
        if QUICK_TEST:
            print("\n⚡ Quick test graph built (200 samples)")
            print("   This is good for testing if fixes work")
            print("\n   For FULL evaluation:")
            print("   1. Edit rebuild_graph.py: Change QUICK_TEST = False")
            print("   2. Run again: python rebuild_graph.py")
            print("   3. Allow 15-30 min for full dataset processing")
            print("   4. If interrupted: just run again to resume! ⏸️→▶️")
        else:
            print("\n✓ FULL dataset graph built!")
            print("   Ready for production evaluation")
    else:
        print("\n✅ Graph built from cached triples!")
        print("   Ready for evaluation")
    
    print("\nNext steps:")
    print("1. Run: python src/evaluate_fixed.py")
    print("2. Check Hit@10 results")
    print("3. If issues, restore backup: cp data/processed/kg.pkl.backup data/processed/kg.pkl")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--reset":
            reset_progress()
            print("✓ Progress file cleared")
            sys.exit(0)
        elif sys.argv[1] == "--progress":
            p = load_progress()
            print(f"Last batch: {p['last_batch']}")
            print(f"Passages: {p['total_passages_processed']}")
            print(f"Timestamp: {p.get('timestamp', 'N/A')}")
            sys.exit(0)
        elif sys.argv[1] == "--graph-only":
            print("\n⚡ GRAPH-ONLY MODE: Building from cached triples (no extraction)\n")
            main(graph_only=True)
            sys.exit(0)
    
    main()
