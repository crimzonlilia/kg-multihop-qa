import re
import pickle
import networkx as nx
from pathlib import Path
from collections import defaultdict, Counter
from difflib import SequenceMatcher
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ======================
# RELATION DISCOVERY
# ======================
def discover_relations_from_passages(passages):
    """
    Discover potential relations từ passages dùng full Pass 1 pipeline
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
# Load schema
# ======================
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "src" / "schemas" / "relations.json"
try:
    with open(SCHEMA_PATH, "r") as f:
        RELATION_SCHEMA = json.load(f)
except FileNotFoundError:
    RELATION_SCHEMA = {}


# ======================
# NORMALIZE (FIXED)
# ======================
def normalize(text: str) -> str:
    """
    Normalize nhẹ:
    - lowercase
    - trim space
    - KHÔNG xóa punctuation (tránh phá entity)
    """
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ======================
# BUILD GRAPH (IMPROVED)
# ======================
def build_graph(triples, entities=None, score_threshold=0.0, add_reverse_edges=True, add_cooccurrence_edges=False):
    """
    Build Knowledge Graph từ triples
    
    Improvements:
    - Fix entity_type_map (no dupe)
    - Track passage_entities từ triples only
    - Optional co-occurrence edges (DISABLED by default to prevent O(n²) explosion)
    - Better edge weight handling
    
    Args:
        add_reverse_edges: Thêm reverse edges để reduce dangling nodes
        add_cooccurrence_edges: Entities xuất hiện cùng passage → add edge (disabled for performance)
    """
    G = nx.DiGraph()
    entity_type_map = {}
    
    # ← FIX: Clean entity_type_map (no dupe)
    if entities:
        for e in entities:
            norm = normalize(e.get("text", ""))
            if norm:  # Only if not empty
                entity_type_map[norm] = e.get("type", "unknown")

    # ← FIX: Track entities FROM TRIPLES ONLY (not from entities list)
    passage_entities = set()
    
    logger.info(f"Building graph from {len(triples)} triples...")
    total_triples = len(triples)
    
    for idx, t in enumerate(triples):
        # Progress indicator every 5000 triples
        if (idx + 1) % 5000 == 0:
            logger.info(f"  [{idx + 1}/{total_triples}] {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        head = t.get("subject") or t.get("head", "")
        tail = t.get("object") or t.get("tail", "")
        relation = t.get("relation", "")
        score = t.get("score") if t.get("score") is not None else 1.0

        if not head or not tail or not relation:
            continue
        if score < score_threshold:
            continue

        head_norm = normalize(head)
        tail_norm = normalize(tail)
        
        if not head_norm or not tail_norm:
            continue
        
        # ← Track entities from triples
        passage_entities.add(head_norm)
        passage_entities.add(tail_norm)

        # ADD NODES
        if head_norm not in G:
            G.add_node(head_norm, entity_type=entity_type_map.get(head_norm, "unknown"), raw_text=head)
        if tail_norm not in G:
            G.add_node(tail_norm, entity_type=entity_type_map.get(tail_norm, "unknown"), raw_text=tail)

        # FORWARD EDGE
        label = normalize_relation_label(relation)
        if G.has_edge(head_norm, tail_norm):
            existing_relations = G[head_norm][tail_norm]["relations"]
            existing_types = {r["type"] for r in existing_relations}
            if relation not in existing_types:
                existing_relations.append({"type": relation, "label": label})
            
            # ← IMPROVE: Aggregate weights (average instead of max)
            existing_weight = G[head_norm][tail_norm]["weight"]
            new_weight = (existing_weight + score) / 2
            G[head_norm][tail_norm]["weight"] = new_weight
        else:
            G.add_edge(head_norm, tail_norm,
                relations=[{"type": relation, "label": label}],
                weight=score
            )
        
        # REVERSE EDGE
        if add_reverse_edges:
            reverse_label = f"← {label}"
            reverse_weight = score * 0.7
            
            if G.has_edge(tail_norm, head_norm):
                existing = G[tail_norm][head_norm]["relations"]
                existing_types = {r["type"] for r in existing}
                if f"reverse_{relation}" not in existing_types:
                    existing.append({"type": f"reverse_{relation}", "label": reverse_label})
                
                # ← Aggregate reverse weights too
                existing_weight = G[tail_norm][head_norm]["weight"]
                new_weight = (existing_weight + reverse_weight) / 2
                G[tail_norm][head_norm]["weight"] = new_weight
            else:
                G.add_edge(tail_norm, head_norm,
                    relations=[{"type": f"reverse_{relation}", "label": reverse_label}],
                    weight=reverse_weight
                )
    
    # CO-OCCURRENCE EDGES: Connect high-frequency entities from same passage
    if add_cooccurrence_edges and len(passage_entities) > 1:
        # ← OPTIMIZED: Only connect frequent, high-quality entities
        entities_list = sorted(list(passage_entities))
        
        # Filter to top entities by frequency (avoid O(n²) explosion)
        max_cooccurrence = min(10, len(entities_list))  # Max 10 entities per passage
        entities_list = entities_list[:max_cooccurrence]
        
        cooccur_weight = 0.3  # Low weight for co-occurrence
        
        for i in range(len(entities_list)):
            for j in range(i + 1, len(entities_list)):
                e1, e2 = entities_list[i], entities_list[j]
                
                # ← Bidirectional edges
                if not G.has_edge(e1, e2):
                    G.add_edge(e1, e2,
                        relations=[{"type": "co_occurs", "label": "appears with"}],
                        weight=cooccur_weight
                    )
                if not G.has_edge(e2, e1):
                    G.add_edge(e2, e1,
                        relations=[{"type": "co_occurs", "label": "appears with"}],
                        weight=cooccur_weight
                    )

    return G


# ======================
# MERGE GRAPHS (IMPROVED)
# ======================
def merge_graphs(graphs):
    """Merge multiple graphs with better weight aggregation."""
    merged = nx.DiGraph()

    for G in graphs:
        # nodes
        for node, data in G.nodes(data=True):
            if node in merged:
                if merged.nodes[node].get("entity_type") == "unknown":
                    merged.nodes[node]["entity_type"] = data.get("entity_type", "unknown")
            else:
                merged.add_node(node, **data)

        # edges
        for u, v, data in G.edges(data=True):
            if merged.has_edge(u, v):
                existing = merged[u][v]["relations"]
                existing_types = {r["type"] for r in existing}

                for r in data.get("relations", []):
                    if r["type"] not in existing_types:
                        existing.append(r)

                merged[u][v]["relations"] = existing

                # ← IMPROVE: Aggregate weights (average)
                existing_weight = merged[u][v].get("weight", 1.0)
                new_weight = data.get("weight", 1.0)
                merged[u][v]["weight"] = (existing_weight + new_weight) / 2

            else:
                merged.add_edge(
                    u,
                    v,
                    relations=list(data.get("relations", [])),
                    weight=data.get("weight", 1.0)
                )

    return merged


# ======================
# GRAPH STATS
# ======================
def graph_stats(G):
    rel_dist = defaultdict(int)

    for _, _, data in G.edges(data=True):
        for r in data.get("relations", []):
            rel_dist[r.get("type")] += 1

    dangling = [n for n in G.nodes if G.out_degree(n) == 0]

    stats = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "dangling_nodes": len(dangling),
        "components": nx.number_weakly_connected_components(G),
        "relation_dist": dict(rel_dist),
    }

    print("\n Graph Statistics")
    print(f"  Nodes: {stats['nodes']}")
    print(f"  Edges: {stats['edges']}")
    print(f"  Density: {stats['density']:.4f}")
    print(f"  Dangling nodes: {stats['dangling_nodes']}")
    print(f"  Connected components: {stats['components']}")

    return stats


# ======================
# SAVE / LOAD
# ======================
def save_graph(G, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(G, f)


def load_graph(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ======================
# RELATION LABEL
# ======================
def normalize_relation_label(relation: str) -> str:
    if relation in RELATION_SCHEMA:
        return RELATION_SCHEMA[relation]

    r = relation.lower().replace("_", " ")

    if "by" in r:
        return f"was {r}"
    if r.startswith(("born", "died")):
        return f"was {r}"

    return r


# ======================
# ENTITY LINKING (IMPROVED)
# ======================
def link_similar_entities(G, threshold=0.65):
    """
    Smart entity linking using fuzzy matching
    
    Improvements:
    - Use SequenceMatcher (Levenshtein-like)
    - Check all token combinations, not just first token
    - Lower threshold (0.65 vs 0.8)
    - Bidirectional edges
    """
    nodes = list(G.nodes())
    links_added = 0
    checked_pairs = set()
    
    # Check all pairs (O(n²) but worth it)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            n1, n2 = nodes[i], nodes[j]
            pair = tuple(sorted([n1, n2]))
            
            if pair in checked_pairs:
                continue
            checked_pairs.add(pair)
            
            # Skip if already connected
            if G.has_edge(n1, n2) or G.has_edge(n2, n1):
                continue
            
            # ← Fuzzy similarity (SequenceMatcher)
            ratio = SequenceMatcher(None, n1, n2).ratio()
            
            # ← Prefix/substring match
            prefix_match = n1 in n2 or n2 in n1
            
            # Link if fuzzy match OR prefix match
            if ratio >= threshold or (prefix_match and ratio >= 0.5):
                # Lower weight for fuzzy match vs prefix match
                if prefix_match:
                    weight = 0.25
                else:
                    weight = 0.15
                
                # ← Bidirectional
                G.add_edge(n1, n2, 
                    relations=[{"type": "similar_to", "label": "similar entity"}],
                    weight=weight
                )
                G.add_edge(n2, n1,
                    relations=[{"type": "similar_to", "label": "similar entity"}],
                    weight=weight
                )
                links_added += 1
    
    logger.info(f"✓ Entity linking added {links_added} bridge edges (threshold={threshold})")
    return G


# ======================
# DEBUG / MAIN
# ======================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    
    from src.data.musique_loader import load_musique, get_all_passages
    from src.extraction.extract_triples import extract_triples_batch, DEFAULT_RELATION_SCHEMA

    print(" Building Knowledge Graph from MusiQue...\n")
    
    print("1  Loading MusiQue samples...")
    samples = load_musique("dev", max_samples=200)
    print(f"✓ Loaded {len(samples)} samples")

    print("\n2  Extracting passages...")
    passages = get_all_passages(samples)
    print(f"✓ Got {len(passages)} unique passages")

    print("\n2.5 Discovering new relations from passages...")
    discovered_relations = discover_relations_from_passages(passages)
    if discovered_relations:
        RELATION_SCHEMA.update(discovered_relations)
        print(f"✓ Extended schema with {len(discovered_relations)} new relations")
    else:
        print("  Using base relation schema only")

    print("\n3  Extracting triples from all passages...")
    results = extract_triples_batch(passages, relation_schema=DEFAULT_RELATION_SCHEMA, skip_cache=False)
    print(f"✓ Extracted from {len(results)} passages")

    print("\n4  Building graphs per passage...")
    graphs = []
    for i, result in enumerate(results):
        if result["triples"]:
            G = build_graph(result["triples"], result["entities"], 
               add_reverse_edges=False,
               add_cooccurrence_edges=True)
            graphs.append(G)
        
        if (i + 1) % 20 == 0:
            logger.info(f"  [{i + 1}/{len(results)}] Built {len(graphs)} graphs")

    print(f"\n5  Merging {len(graphs)} graphs...")
    merged_graph = merge_graphs(graphs)

    print("\n6 Linking similar entities...")
    merged_graph = link_similar_entities(merged_graph, threshold=0.65)

    print("\n7  Final graph statistics:")
    graph_stats(merged_graph)

    # Save graph
    kg_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "kg.pkl"
    save_graph(merged_graph, str(kg_path))
    print(f"\n Graph saved to {kg_path}")