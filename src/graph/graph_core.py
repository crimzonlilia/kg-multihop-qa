"""Core graph building logic."""

import logging
import networkx as nx
from .graph_utils import normalize, normalize_relation_label, get_entity_id

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
    # Additional relationship/action words that shouldn't be entities
    "spouse", "husband", "wife", "parent", "child", "sibling", "brother", "sister",
    "tribute", "performance", "show", "film", "production",
    "creator", "artist", "actor", "director", "writer", "musician",
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


def _make_passage_node_id(passage_id: str) -> str:
    """Keep passage node ids namespaced so they never collide with entities."""
    passage_id = str(passage_id).strip()
    if passage_id.startswith("passage::"):
        return passage_id
    return f"passage::{passage_id}"


def build_graph(triples: list,
               entities: list = None,
               score_threshold: float = 0.0,
               add_reverse_edges: bool = True,
               add_cooccurrence_edges: bool = False,
               passage_id=None,
               passage_text=None,
               add_passage_node: bool = True,
               passage_edge_weight: float = 0.8) -> nx.DiGraph:
    """
    Build Knowledge Graph from triples using hash-based entity IDs for consistency.

    Args:
        triples: List of {'subject', 'object', 'relation', 'score'}
        entities: Optional list of {'text', 'type'} for entity type mapping
        score_threshold: Filter triples below this confidence
        add_reverse_edges: Add reverse edges for dangling node reduction
        add_cooccurrence_edges: Add edges between co-occurring entities (expensive)
        passage_id: Optional passage identifier to anchor this mini-graph
        passage_text: Optional raw passage text stored on the passage node
        add_passage_node: If True, attach a passage node linked to passage entities
        passage_edge_weight: Weight for passage↔entity structural edges

    Returns: NetworkX DiGraph with entity nodes (using hash IDs), relation edges, and optional passage anchors
    """
    G = nx.DiGraph()
    entity_id_to_type = {}      # entity_id -> entity_type
    entity_id_to_raw_text = {}  # entity_id -> original raw text
    entity_id_to_norm_text = {} # entity_id -> normalized text (for reference)
    passage_entity_ids = set()  # All entity IDs mentioned in this passage

    # Build entity type map using entity IDs
    if entities:
        for e in entities:
            raw_text = e.get("text", "")
            norm = normalize(raw_text)
            if norm and not _is_noise_node(norm):
                ent_id = get_entity_id(raw_text)
                entity_id_to_type[ent_id] = e.get("type", "unknown")
                entity_id_to_raw_text[ent_id] = raw_text
                entity_id_to_norm_text[ent_id] = norm
                passage_entity_ids.add(ent_id)

    logger.info(f"Building graph from {len(triples)} triples (using hash-based entity IDs)...")
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

        # Generate entity IDs using hash of normalized text (CRITICAL FIX)
        head_id = get_entity_id(head)
        tail_id = get_entity_id(tail)
        
        passage_entity_ids.add(head_id)
        passage_entity_ids.add(tail_id)
        
        # Store raw text and normalized text for reference
        entity_id_to_raw_text.setdefault(head_id, head)
        entity_id_to_raw_text.setdefault(tail_id, tail)
        entity_id_to_norm_text.setdefault(head_id, head_norm)
        entity_id_to_norm_text.setdefault(tail_id, tail_norm)

        # Add nodes with entity IDs as keys
        if head_id not in G:
            G.add_node(head_id,
                      node_type="entity",
                      entity_type=entity_id_to_type.get(head_id, "unknown"),
                      raw_text=head,
                      normalized_text=head_norm)
        if tail_id not in G:
            G.add_node(tail_id,
                      node_type="entity",
                      entity_type=entity_id_to_type.get(tail_id, "unknown"),
                      raw_text=tail,
                      normalized_text=tail_norm)

        label = normalize_relation_label(relation)
        _add_or_merge_edge(G, head_id, tail_id, relation, label, score)

        if add_reverse_edges:
            reverse_label = f"← {label}"
            reverse_weight = score * 0.7
            _add_or_merge_edge(G, tail_id, head_id, f"reverse_{relation}", reverse_label, reverse_weight)

    if add_cooccurrence_edges and len(passage_entity_ids) > 1:
        _add_cooccurrence_edges(G, passage_entity_ids)

    if add_passage_node and passage_id:
        passage_node = _make_passage_node_id(passage_id)
        G.add_node(
            passage_node,
            node_type="passage",
            passage_id=str(passage_id),
            text=passage_text or "",
            raw_text=str(passage_id),
        )

        for ent_id in sorted(passage_entity_ids):
            if ent_id not in G:
                G.add_node(
                    ent_id,
                    node_type="entity",
                    entity_type=entity_id_to_type.get(ent_id, "unknown"),
                    raw_text=entity_id_to_raw_text.get(ent_id, ent_id),
                    normalized_text=entity_id_to_norm_text.get(ent_id, ""),
                )
            _add_or_merge_edge(G, passage_node, ent_id, "contains", "contains", passage_edge_weight)
            _add_or_merge_edge(G, ent_id, passage_node, "mentioned_in", "mentioned in", max(0.3, passage_edge_weight * 0.8))

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