import re
import pickle
import networkx as nx
from pathlib import Path
from collections import defaultdict
import json

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
# BUILD GRAPH
# ======================
def build_graph(triples, entities=None, score_threshold=0.0, add_reverse_edges=True, add_cooccurrence_edges=True):
    """
    Build Knowledge Graph từ triples
    
    add_reverse_edges: Thêm reverse edges để reduce dangling nodes
    add_cooccurrence_edges: Entities xuất hiện cùng passage → add edge
    """
    G = nx.DiGraph()
    entity_type_map = {}
    
    if entities:
        for e in entities:
            raw = e.get("text", "")
            norm = normalize(raw)
            entity_type_map[norm] = e.get("type", "unknown")
            entity_type_map[raw.lower().strip()] = e.get("type", "unknown")

    # Track entities trong passage
    passage_entities = set(entity_type_map.keys()) if entity_type_map else set()
    
    for t in triples:
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
        
        # Track entities
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
            if score > G[head_norm][tail_norm]["weight"]:
                G[head_norm][tail_norm]["weight"] = score
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
                if reverse_weight > G[tail_norm][head_norm]["weight"]:
                    G[tail_norm][head_norm]["weight"] = reverse_weight
            else:
                G.add_edge(tail_norm, head_norm,
                    relations=[{"type": f"reverse_{relation}", "label": reverse_label}],
                    weight=reverse_weight
                )
    
    # CO-OCCURRENCE EDGES: Connect entities từ cùng passage
    if add_cooccurrence_edges and len(passage_entities) > 1:
        entities_list = sorted(list(passage_entities))
        for i in range(len(entities_list)):
            for j in range(i + 1, len(entities_list)):
                e1, e2 = entities_list[i], entities_list[j]
                cooccur_weight = 0.3  # Thấp hơn relation edges
                
                # Add bidirectional edges
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
# MERGE GRAPHS
# ======================
def merge_graphs(graphs):
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

                if data.get("weight", 0) > merged[u][v].get("weight", 0):
                    merged[u][v]["weight"] = data["weight"]

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

    print("\nGraph Statistics")
    print(f"Nodes: {stats['nodes']}")
    print(f"Edges: {stats['edges']}")
    print(f"Density: {stats['density']:.4f}")
    print(f"Dangling nodes: {stats['dangling_nodes']}")
    print(f"Connected components: {stats['components']}")

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
# DEBUG
# ======================
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    
    from src.data.musique_loader import load_musique, get_all_passages
    from src.extraction.extract_triples import extract_triples_batch, DEFAULT_RELATION_SCHEMA

    print("Loading 100 MusiQue samples...")
    samples = load_musique("dev", max_samples=100)
    print(f"✓ Loaded {len(samples)} samples")

    print("\nExtracting passages...")
    passages = get_all_passages(samples)
    print(f"✓ Got {len(passages)} unique passages")

    print("\nExtracting triples from all passages...")
    results = extract_triples_batch(passages, relation_schema=DEFAULT_RELATION_SCHEMA, skip_cache=True)
    print(f"✓ Extracted from {len(results)} passages")

    print("\nBuilding graphs...")
    graphs = []
    for i, result in enumerate(results):
        if result["triples"]:
            G = build_graph(result["triples"], result["entities"], add_reverse_edges=True)  # ← Thêm này
            graphs.append(G)
        
        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(results)}] Built {len(graphs)} graphs")

    print(f"\nMerging {len(graphs)} graphs...")
    merged_graph = merge_graphs(graphs)

    print("\nFinal graph statistics:")
    graph_stats(merged_graph)

    # Save graph to kg.pkl for evaluation
    kg_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "kg.pkl"
    save_graph(merged_graph, str(kg_path))
    print(f"\n✓ Graph saved to {kg_path}")