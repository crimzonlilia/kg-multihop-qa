#!/usr/bin/env python
"""
Rebuild knowledge graph with hash-based entity IDs for consistent entity-passage mapping.

This script:
1. Loads all MusiQue passages
2. Extracts triples and entities using new normalize()
3. Builds graph using hash-based entity IDs via build_graph()
4. Merges into single KG
5. Saves to data/processed/kg.pkl

Run: python rebuild_graph_hash_based.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import logging
from src.data.musique_loader import load_musique, get_all_passages
from src.extraction import extract_triples_batch, DEFAULT_RELATION_SCHEMA
from src.graph.build_graph import build_graph, merge_graphs, save_graph

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def rebuild_kg(max_samples=None, skip_cache=False):
    """Rebuild KG from scratch with hash-based entity IDs."""
    
    print("🏗️  Rebuilding Knowledge Graph (hash-based entity IDs)\n")
    
    print("1️⃣  Loading MusiQue passages...")
    samples = load_musique("dev", max_samples=max_samples or 250)
    print(f"✓ Loaded {len(samples)} samples")

    print("\n2  Extracting passages...")
    passages = get_all_passages(samples)
    print(f"✓ Got {len(passages)} unique passages")

    print("\n3️⃣  Extracting triples from all passages...")
    results = extract_triples_batch(
        passages,
        relation_schema=DEFAULT_RELATION_SCHEMA,
        skip_cache=skip_cache
    )
    print(f"✓ Extracted from {len(results)} passages")

    print("\n4️⃣  Building graphs per passage (WITH HASH-BASED ENTITY IDs)...")
    graphs = []
    for i, result in enumerate(results):
        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(results)}] Built {len(graphs)} graphs so far")
        
        if result["triples"]:
            passage_idx = i
            passage_text = passages[passage_idx]
            
            G = build_graph(
                result["triples"],
                result["entities"],
                add_reverse_edges=False,
                add_cooccurrence_edges=True,
                passage_id=f"passage_{passage_idx}",
                passage_text=passage_text,
                add_passage_node=True,
            )
            graphs.append(G)

    print(f"✓ Built {len(graphs)} graphs")

    print("\n5️⃣  Merging graphs into single KG...")
    final_graph = merge_graphs(graphs)
    print(f"✓ Merged KG: {final_graph.number_of_nodes()} nodes, {final_graph.number_of_edges()} edges")

    print("\n6️⃣  Saving KG to data/processed/kg.pkl...")
    save_graph(final_graph, "data/processed/kg.pkl")
    print("✓ Saved!")

    print("\n✅ KG rebuild complete (hash-based entity IDs)")
    print(f"\nFinal KG stats:")
    print(f"  - Nodes: {final_graph.number_of_nodes()}")
    print(f"  - Edges: {final_graph.number_of_edges()}")
    print(f"  - Samples used: {max_samples or 'all'}")


if __name__ == "__main__":
    rebuild_kg(max_samples=250)
