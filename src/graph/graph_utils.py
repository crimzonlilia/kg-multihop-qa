"""Graph utilities: normalization, I/O, statistics."""

import re
import pickle
import logging
from pathlib import Path
from collections import defaultdict
import networkx as nx

logger = logging.getLogger(__name__)

# Relation schema (loaded lazily)
_RELATION_SCHEMA = None


def get_relation_schema():
    """Load relation schema from file (cached)."""
    global _RELATION_SCHEMA
    
    if _RELATION_SCHEMA is None:
        schema_path = Path(__file__).resolve().parents[2] / "src" / "schemas" / "relations.json"
        
        if schema_path.exists():
            import json
            with open(schema_path) as f:
                _RELATION_SCHEMA = json.load(f)
        else:
            _RELATION_SCHEMA = {}
    
    return _RELATION_SCHEMA


def normalize(text: str) -> str:
    """
    Lightweight normalization:
    - lowercase
    - trim whitespace
    - compress multiple spaces
    (keeps punctuation to preserve entity integrity)
    """
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_relation_label(relation: str) -> str:
    """Convert relation type to human-readable label (cached via schema)."""
    schema = get_relation_schema()
    
    if relation in schema:
        return schema[relation]
    
    # Fallback formatting
    label = relation.lower().replace("_", " ")
    
    if "by" in label:
        return f"was {label}"
    if label.startswith(("born", "died")):
        return f"was {label}"
    
    return label


def save_graph(G, path: str):
    """Save graph to pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "wb") as f:
        pickle.dump(G, f)
    
    logger.info(f"✓ Saved graph to {path}")


def load_graph(path: str) -> nx.DiGraph:
    """Load graph from pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def graph_stats(G: nx.DiGraph) -> dict:
    """Compute and display graph statistics."""
    rel_dist = defaultdict(int)
    
    for _, _, data in G.edges(data=True):
        for r in data.get("relations", []):
            rel_dist[r.get("type")] += 1
    
    dangling = [n for n in G.nodes() if G.out_degree(n) == 0]
    
    stats = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "dangling_nodes": len(dangling),
        "components": nx.number_weakly_connected_components(G),
        "relation_dist": dict(rel_dist),
    }
    
    print("\n📊 Graph Statistics")
    print(f"  Nodes: {stats['nodes']}")
    print(f"  Edges: {stats['edges']}")
    print(f"  Density: {stats['density']:.4f}")
    print(f"  Dangling nodes: {stats['dangling_nodes']}")
    print(f"  Connected components: {stats['components']}")
    print(f"  Top relations: {sorted(rel_dist.items(), key=lambda x: x[1], reverse=True)[:5]}")
    
    return stats
