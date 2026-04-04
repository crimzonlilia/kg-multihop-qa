"""
Knowledge Graph Building Pipeline (Orchestration Layer)

Imports modular components for backward compatibility.
For new code, import from submodules directly.
"""

import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import from modular components
from .graph_utils import normalize, normalize_relation_label, save_graph, load_graph, graph_stats
from .graph_core import build_graph
from .graph_merge import merge_graphs
from .entity_linking import link_similar_entities, link_entities_fast

# Re-export for backward compatibility
__all__ = [
    "normalize",
    "normalize_relation_label",
    "build_graph",
    "merge_graphs",
    "graph_stats",
    "save_graph",
    "load_graph",
    "link_similar_entities",
    "link_entities_fast",
    "discover_relations_from_passages",
]


def discover_relations_from_passages(passages):
    """
    Discover potential relations from passages using Pass 1 pipeline.
    
    Returns: expanded RELATION_SCHEMA dict
    """
    try:
        from src.extraction.relation_discovery import run_discovery
        
        logger.info("🔍 Discovering relations from passages...")
        expanded_schema = run_discovery(
            passages,
            min_freq=5,
            min_cluster_freq=10,
            distance_threshold=0.35,
            top_k=80
        )
        
        logger.info(f"✓ Discovered {len(expanded_schema) - 11} new relations (total {len(expanded_schema)})")
        return expanded_schema
    
    except Exception as e:
        logger.warning(f"⚠️  Relation discovery failed: {e}")
        return {}


# ======================
# MAIN PIPELINE
# ======================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    
    from src.data.musique_loader import load_musique, get_all_passages
    from src.extraction import extract_triples_batch, DEFAULT_RELATION_SCHEMA

    print("🏗️  Building Knowledge Graph from MusiQue...\n")
    
    print("1️⃣  Loading MusiQue samples...")
    samples = load_musique("dev", max_samples=200)
    print(f"✓ Loaded {len(samples)} samples")

    print("\n2️⃣  Extracting passages...")
    passages = get_all_passages(samples)
    print(f"✓ Got {len(passages)} unique passages")

    print("\n2️⃣.5️⃣  Discovering new relations from passages...")
    discovered_relations = discover_relations_from_passages(passages)
    if discovered_relations:
        print(f"✓ Extended schema with {len(discovered_relations)} new relations")
    else:
        print("  Using base relation schema only")

    print("\n3️⃣  Extracting triples from all passages...")
    results = extract_triples_batch(passages, relation_schema=DEFAULT_RELATION_SCHEMA, skip_cache=False)
    print(f"✓ Extracted from {len(results)} passages")

    print("\n4️⃣  Building graphs per passage...")
    graphs = []
    for i, result in enumerate(results):
        if result["triples"]:
            G = build_graph(result["triples"], result["entities"],
                           add_reverse_edges=False,
                           add_cooccurrence_edges=True)
            graphs.append(G)
        
        if (i + 1) % 20 == 0:
            logger.info(f"  [{i + 1}/{len(results)}] Built {len(graphs)} graphs")

    print(f"\n5️⃣  Merging {len(graphs)} graphs...")
    merged_graph = merge_graphs(graphs)

    print("\n6️⃣  Linking similar entities (fast mode)...")
    merged_graph = link_entities_fast(merged_graph, threshold=0.65)

    print("\n7️⃣  Final graph statistics:")
    graph_stats(merged_graph)

    # Save graph
    kg_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "kg.pkl"
    save_graph(merged_graph, str(kg_path))
    print(f"\n✅ Graph saved to {kg_path}")
