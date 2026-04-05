"""Core graph building logic."""

import logging
import networkx as nx
from .graph_utils import normalize, normalize_relation_label

logger = logging.getLogger(__name__)

# Common nouns / stopwords that should never become graph nodes
_NOISE_NODES = {
    "company", "organization", "person", "people", "man", "woman",
    "city", "country", "place", "location", "area", "region",
    "film", "movie", "show", "series", "book", "song", "album",
    "the", "this", "that", "he", "she", "it", "they", "his", "her",
    "work", "role", "part", "type", "kind", "name", "time", "year",
    "group", "team", "band", "member", "owner", "founder", "leader",
    "government", "state", "world", "war", "act", "law", "right",
}


def _is_noise_node(text: str) -> bool:
    """Return True if this normalized text should not be a graph node."""
    if not text:
        return True
    # Too short (single char or 2-char abbreviations without digits)
    if len(text) <= 2:
        return True
    # Pure digits or single token that is a stopword/common noun
    if text in _NOISE_NODES:
        return True
    # Pure number
    if text.isdigit():
        return True
    return False


def build_graph(triples: list, 
               entities: list = None,
               score_threshold: float = 0.0,
               add_reverse_edges: bool = True,
               add_cooccurrence_edges: bool = False) -> nx.DiGraph:
    """
    Build Knowledge Graph from triples.
    
    Args:
        triples: List of {'subject', 'object', 'relation', 'score'}
        entities: Optional list of {'text', 'type'} for entity type mapping
        score_threshold: Filter triples below this confidence
        add_reverse_edges: Add reverse edges for dangling node reduction
        add_cooccurrence_edges: Add edges between co-occurring entities (expensive)
    
    Returns: NetworkX DiGraph with entities as nodes and relations as edges
    """
    G = nx.DiGraph()
    entity_type_map = {}
    passage_entities = set()
    
    # Build entity type map (for fast O(1) lookup)
    if entities:
        for e in entities:
            norm = normalize(e.get("text", ""))
            if norm:
                entity_type_map[norm] = e.get("type", "unknown")
    
    logger.info(f"Building graph from {len(triples)} triples...")
    total_triples = len(triples)
    
    for idx, t in enumerate(triples):
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

        # ← Filter noise nodes before adding to graph
        if _is_noise_node(head_norm) or _is_noise_node(tail_norm):
            continue
        
        passage_entities.add(head_norm)
        passage_entities.add(tail_norm)
        
        if head_norm not in G:
            G.add_node(head_norm,
                      entity_type=entity_type_map.get(head_norm, "unknown"),
                      raw_text=head)
        if tail_norm not in G:
            G.add_node(tail_norm,
                      entity_type=entity_type_map.get(tail_norm, "unknown"),
                      raw_text=tail)
        
        label = normalize_relation_label(relation)
        _add_or_merge_edge(G, head_norm, tail_norm, relation, label, score)
        
        if add_reverse_edges:
            reverse_label = f"← {label}"
            reverse_weight = score * 0.7
            _add_or_merge_edge(G, tail_norm, head_norm, f"reverse_{relation}", reverse_label, reverse_weight)
    
    if add_cooccurrence_edges and len(passage_entities) > 1:
        _add_cooccurrence_edges(G, passage_entities)
    
    logger.info(f"✓ Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return G


def _add_or_merge_edge(G: nx.DiGraph, head: str, tail: str, relation: str, label: str, score: float):
    """Add edge or merge with existing edge (aggregate weights)."""
    if G.has_edge(head, tail):
        existing_relations = G[head][tail]["relations"]
        existing_types = {r["type"] for r in existing_relations}
        if relation not in existing_types:
            existing_relations.append({"type": relation, "label": label})
        existing_weight = G[head][tail]["weight"]
        G[head][tail]["weight"] = (existing_weight + score) / 2
    else:
        G.add_edge(head, tail,
                  relations=[{"type": relation, "label": label}],
                  weight=score)


def _add_cooccurrence_edges(G: nx.DiGraph, entities: set, max_per_passage: int = 10, weight: float = 0.3):
    """Add co-occurrence edges between entities (optimized)."""
    entities_list = sorted(list(entities))
    if len(entities_list) > max_per_passage:
        entities_list = entities_list[:max_per_passage]
    
    for i in range(len(entities_list)):
        for j in range(i + 1, len(entities_list)):
            e1, e2 = entities_list[i], entities_list[j]
            if G.has_edge(e1, e2) or G.has_edge(e2, e1):
                continue
            G.add_edge(e1, e2,
                      relations=[{"type": "co_occurs", "label": "appears with"}],
                      weight=weight)
            G.add_edge(e2, e1,
                      relations=[{"type": "co_occurs", "label": "appears with"}],
                      weight=weight)