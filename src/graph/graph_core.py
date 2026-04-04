"""Core graph building logic."""

import logging
import networkx as nx
from .graph_utils import normalize, normalize_relation_label

logger = logging.getLogger(__name__)


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
        # Progress every 5000 triples
        if (idx + 1) % 5000 == 0:
            logger.info(f"  [{idx + 1}/{total_triples}] {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Extract triple components
        head = t.get("subject") or t.get("head", "")
        tail = t.get("object") or t.get("tail", "")
        relation = t.get("relation", "")
        score = t.get("score") if t.get("score") is not None else 1.0
        
        # Validate triple
        if not head or not tail or not relation:
            continue
        if score < score_threshold:
            continue
        
        head_norm = normalize(head)
        tail_norm = normalize(tail)
        
        if not head_norm or not tail_norm:
            continue
        
        # Track entities in passage
        passage_entities.add(head_norm)
        passage_entities.add(tail_norm)
        
        # Add nodes if not present
        if head_norm not in G:
            G.add_node(head_norm,
                      entity_type=entity_type_map.get(head_norm, "unknown"),
                      raw_text=head)
        if tail_norm not in G:
            G.add_node(tail_norm,
                      entity_type=entity_type_map.get(tail_norm, "unknown"),
                      raw_text=tail)
        
        # Add forward edge
        label = normalize_relation_label(relation)
        _add_or_merge_edge(G, head_norm, tail_norm, relation, label, score)
        
        # Add reverse edge if requested
        if add_reverse_edges:
            reverse_label = f"← {label}"
            reverse_weight = score * 0.7
            _add_or_merge_edge(G, tail_norm, head_norm, f"reverse_{relation}", reverse_label, reverse_weight)
    
    # Add co-occurrence edges if requested
    if add_cooccurrence_edges and len(passage_entities) > 1:
        _add_cooccurrence_edges(G, passage_entities)
    
    logger.info(f"✓ Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return G


def _add_or_merge_edge(G: nx.DiGraph, head: str, tail: str, relation: str, label: str, score: float):
    """Add edge or merge with existing edge (aggregate weights)."""
    if G.has_edge(head, tail):
        # Merge with existing edge
        existing_relations = G[head][tail]["relations"]
        existing_types = {r["type"] for r in existing_relations}
        
        if relation not in existing_types:
            existing_relations.append({"type": relation, "label": label})
        
        # Aggregate weight (average)
        existing_weight = G[head][tail]["weight"]
        G[head][tail]["weight"] = (existing_weight + score) / 2
    else:
        # Create new edge
        G.add_edge(head, tail,
                  relations=[{"type": relation, "label": label}],
                  weight=score)


def _add_cooccurrence_edges(G: nx.DiGraph, entities: set, max_per_passage: int = 10, weight: float = 0.3):
    """
    Add co-occurrence edges between entities (optimized).
    
    Args:
        G: Graph to modify
        entities: Set of entity names in this passage
        max_per_passage: Max entities to connect (prevent O(n²) explosion)
        weight: Weight for co-occurrence edges
    """
    entities_list = sorted(list(entities))
    
    # Limit to top entities to avoid O(n²) explosion
    if len(entities_list) > max_per_passage:
        entities_list = entities_list[:max_per_passage]
    
    for i in range(len(entities_list)):
        for j in range(i + 1, len(entities_list)):
            e1, e2 = entities_list[i], entities_list[j]
            
            # Skip if already connected
            if G.has_edge(e1, e2) or G.has_edge(e2, e1):
                continue
            
            # Add bidirectional edges
            G.add_edge(e1, e2,
                      relations=[{"type": "co_occurs", "label": "appears with"}],
                      weight=weight)
            G.add_edge(e2, e1,
                      relations=[{"type": "co_occurs", "label": "appears with"}],
                      weight=weight)
