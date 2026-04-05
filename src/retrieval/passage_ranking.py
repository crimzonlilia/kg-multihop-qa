"""
Passage-based retrieval ranking (HippoRAG style)
- Map triples back to passages
- Rank passages based on entity coverage and triple connections
- Return top-k passages for context/generation
"""

from collections import defaultdict, Counter
import networkx as nx
from typing import List, Tuple, Set

from src.cache_utils import load_triples_cache


def load_passage_data(cache_path="data/cache/triples.json", cache_name=None):
    """Load passages with entity and triple info from a path or cache alias."""
    cache_arg = str(cache_path) if cache_path is not None else ""
    if cache_name is None and cache_arg and not cache_arg.endswith(".json") and "/" not in cache_arg and "\\" not in cache_arg:
        cache_name = cache_arg
        cache_path = None

    passages, _, _ = load_triples_cache(cache_path=cache_path, cache_name=cache_name)
    return passages


def build_triple_to_passage_map(passages):
    """
    Create mapping: (subject, predicate, object) -> passage_id
    Also returns passage text and entities mapping, plus entity-to-passages cache
    """
    triple_to_passages = defaultdict(list)
    passage_texts = {}
    passage_entities = {}  # passage_id -> list of entity texts
    entity_to_passages = defaultdict(set)  # entity (normalized) -> set of passage_ids (CACHE)
    
    for passage_id, passage_data in enumerate(passages):
        passage = passage_data.get('passage', '')
        passage_texts[passage_id] = passage
        
        # Extract entity texts
        entities = passage_data.get('entities', [])
        entity_texts = [e.get('text') for e in entities if 'text' in e]
        passage_entities[passage_id] = entity_texts
        
        # Build entity cache for fast lookup
        for entity_text in entity_texts:
            entity_norm = entity_text.lower()
            entity_to_passages[entity_norm].add(passage_id)
        
        # Map each triple to this passage
        triples = passage_data.get('triples', [])
        for triple in triples:
            if isinstance(triple, dict) and 'subject' in triple:
                key = (
                    triple.get('subject', '').lower(),
                    triple.get('relation', '').lower(),
                    triple.get('object', '').lower()
                )
                triple_to_passages[key].append(passage_id)
    
    return triple_to_passages, passage_texts, passage_entities, entity_to_passages


def rank_passages_by_entities(
    G: nx.DiGraph, 
    query_entities: List[str],
    triple_to_passages,
    passage_entities,
    passage_texts,
    top_k: int = 10,
    neighborhood_hops: int = 3,
    entity_to_passages: dict = None
) -> List[Tuple[int, str, float]]:
    """
    Rank passages based on entity coverage with weighted scoring
    
    Args:
        G: Knowledge graph
        query_entities: Starting entities from question
        triple_to_passages: Mapping from triples to passages
        passage_entities: Mapping from passage_id to entities
        passage_texts: Mapping from passage_id to text
        top_k: Return top-k passages
        neighborhood_hops: Expand to neighboring entities for n hops
        entity_to_passages: Cache mapping entity -> passages for fast lookup
    
    Returns:
        List of (passage_id, passage_text, score) tuples
    """
    
    # Find all relevant entities with hop information
    hop_weights = {}  # entity -> weight based on distance
    
    # BFS to find neighbors up to N hops
    visited = {}  # entity -> min_hops_found
    queue = [(ent, 0) for ent in query_entities if ent in G]
    
    while queue:
        entity, hops = queue.pop(0)
        
        if entity in visited and visited[entity] <= hops:
            continue
        visited[entity] = hops
        
        # Weight: closer entities (lower hops) get higher weight
        # hop 0 = 1.0, hop 1 = 0.85, hop 2 = 0.7, hop 3 = 0.55, hop 4+ = 0.4
        weight = max(0.4, 1.0 - hops * 0.15)
        hop_weights[entity] = weight
        
        if hops < neighborhood_hops:
            # Add neighbors
            for neighbor in G.neighbors(entity):
                if neighbor not in visited or visited[neighbor] > hops + 1:
                    queue.append((neighbor, hops + 1))
    
    # Score passages by weighted entity coverage (using cache if available)
    passage_scores = defaultdict(float)
    passage_match_count = Counter()
    
    for entity, entity_weight in hop_weights.items():
        entity_lower = entity.lower()
        
        # Fast lookup using cache
        if entity_to_passages and entity_lower in entity_to_passages:
            for passage_id in entity_to_passages[entity_lower]:
                passage_scores[passage_id] += entity_weight
                passage_match_count[passage_id] += 1
        else:
            # Fallback to slower matching if cache not available
            for passage_id, entities_in_passage in passage_entities.items():
                for e in entities_in_passage:
                    if e.lower() == entity_lower:
                        passage_scores[passage_id] += entity_weight
                        passage_match_count[passage_id] += 1
                        break
    
    # Boost passages with multiple entities
    for passage_id in passage_scores:
        entity_freq = passage_match_count[passage_id]
        # Boost: 1 entity = 1x, 2 entities = 1.4x, 3+ entities = 1.8x
        boost = min(1.8, 1.0 + entity_freq * 0.4)
        passage_scores[passage_id] *= boost
    
    # Sort passages by score
    ranked_passages = sorted(
        passage_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]
    
    # Return (passage_id, text, score)
    result = []
    for passage_id, score in ranked_passages:
        text = passage_texts.get(passage_id, "")
        result.append((passage_id, text, score))
    
    return result


def extract_answer_from_passages(
    gold_answer: str,
    passages: List[Tuple[int, str, float]],
    normalize_fn=None
) -> Tuple[bool, int]:
    """
    Check if gold answer appears in any passage
    Returns (found, passage_rank)
    """
    if normalize_fn is None:
        normalize_fn = lambda x: x.lower().strip()
    
    answer_norm = normalize_fn(gold_answer)
    
    for rank, (passage_id, text, score) in enumerate(passages, 1):
        text_lower = text.lower()
        
        # Multiple matching strategies
        if answer_norm in text_lower:
            return True, rank
        
        # Try word-by-word matching
        answer_words = answer_norm.split()
        if len(answer_words) > 0:
            if all(w in text_lower for w in answer_words):
                return True, rank
    
    return False, -1
