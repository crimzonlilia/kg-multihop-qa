"""Graph utilities: normalization, I/O, statistics."""

import re
import pickle
import logging
import hashlib
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
    Unified normalization for consistent entity representation across graph, cache, query:
    - Convert diacritics to ASCII (ā → a, é → e, etc.)
    - lowercase
    - remove punctuation and special chars (keep alphanumeric + spaces)
    - compress whitespace
    
    CRITICAL: This MUST be used everywhere (graph, passage, query)
    """
    if not text:
        return ""
    
    # Step 1: Normalize diacritics to ASCII
    # Convert Unicode combining characters to ASCII equivalents
    import unicodedata
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # Step 2: Lowercase
    text = text.lower().strip()
    
    # Step 3: Remove all non-alphanumeric except spaces
    text = re.sub(r"[^a-z0-9\s]", "", text)
    
    # Step 4: Compress multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_entity_id(text: str) -> str:
    """
    Generate stable entity ID using normalized text + MD5 hash.
    
    One entity → one normalized form → one hash → used everywhere
    
    Args:
        text: raw entity text (any case, punctuation)
    
    Returns:
        str: "entity-{md5_hex}" format, always same for same normalized entity
    """
    norm = normalize(text)
    if not norm:
        return "entity-empty"
    return "entity-" + hashlib.md5(norm.encode()).hexdigest()


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
