"""Schema definitions and dynamic schema grouping for relation extraction."""

import logging
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

# Entity and relation definitions
ENTITY_LABELS = ["person", "organization", "location", "event", "role"]

DEFAULT_RELATION_SCHEMA = {
    "born_in": "Person was born in a location",
    "died_in": "Person died in a location",
    "founded_by": "Organization was founded by a person",
    "located_in": "Entity located in a place",
    "part_of": "Organization is part of another organization",
    "occurred_in": "Event occurred in a location",
    "educated_at": "Person studied at an organization",
    "worked_at": "Person worked at an organization",
    "member_of": "Person is member of an organization",
}

TYPE_CONSTRAINTS = {
    "born_in": ("person", ["location"]),
    "died_in": ("person", ["location"]),
    "founded_by": (None, None),
    "located_in": (None, ["location"]),
    "part_of": ("organization", ["organization"]),
    "occurred_in": (None, ["location"]),
    "educated_at": ("person", ["organization"]),
    "worked_at": ("person", ["organization"]),
    "member_of": ("person", ["organization"]),
}

# Schema caching
_SCHEMA_CACHE = {}


def build_schema(extractor, relation_schema: dict, schema_id: str = None):
    """Build GLiNER2 schema from relation dict (cached)."""
    schema_key = schema_id or tuple(sorted((relation_schema or DEFAULT_RELATION_SCHEMA).keys()))
    
    if schema_key not in _SCHEMA_CACHE:
        _SCHEMA_CACHE[schema_key] = (
            extractor.create_schema()
            .entities(ENTITY_LABELS)
            .relations(relation_schema or DEFAULT_RELATION_SCHEMA)
        )
    
    return _SCHEMA_CACHE[schema_key]


def clear_schema_cache():
    """Clear cached schemas."""
    global _SCHEMA_CACHE
    _SCHEMA_CACHE = {}
    logger.info("Schema cache cleared")


def normalize_vectors(vectors):
    """Normalize vectors using L2 norm."""
    try:
        from sklearn.preprocessing import normalize as sklearn_normalize
        return sklearn_normalize(vectors, norm='l2', axis=1)
    except ImportError:
        # Manual L2 normalization
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norms + 1e-8)


def get_dynamic_schemas_grouped(passages: list[str],
                                relation_schema: dict,
                                embedder=None,
                                k: int = 8,
                                threshold: float = None) -> dict:
    """
    Group passages by semantically similar top-k relations.
    
    Args:
        passages: List of text passages
        relation_schema: Dict of {relation_name: description}
        embedder: Pre-loaded SentenceTransformer model
        k: Number of top relations to select per passage
        threshold: Confidence threshold (auto-detected if None)
    
    Returns:
        Dict with 'groups', 'schemas', and 'embedding_stats'
    """
    if not passages or not relation_schema:
        return {'groups': defaultdict(list), 'schemas': {}, 'embedding_stats': {}}
    
    if embedder is None:
        from .model import get_embedder
        embedder = get_embedder()
    
    logger.info(f"Embedding {len(passages)} passages + {len(relation_schema)} relations...")
    
    # Batch embed passages
    passage_embs = embedder.encode(passages, convert_to_numpy=True, batch_size=32, show_progress_bar=False)
    passage_embs = normalize_vectors(passage_embs).astype(np.float32)
    
    # Embed relation descriptions
    relation_names = list(relation_schema.keys())
    relation_descs = [relation_schema[r] for r in relation_names]
    relation_embs = embedder.encode(relation_descs, convert_to_numpy=True)
    relation_embs = normalize_vectors(relation_embs).astype(np.float32)
    
    # Compute cosine similarity (normalized)
    similarities = np.dot(passage_embs, relation_embs.T)  # [n_passages, n_relations]
    max_sims = np.max(similarities, axis=1)  # [n_passages]
    
    # Auto-detect threshold if not provided
    if threshold is None:
        threshold = np.percentile(max_sims, 25)
        logger.info(f"Auto-detected threshold: {threshold:.3f} (25th percentile)")
    
    # Select top-k relations per passage
    groups = defaultdict(list)
    schemas = {}
    
    for passage_idx, sims in enumerate(similarities):
        top_k_indices = np.argsort(sims)[-k:][::-1]
        reliable_indices = top_k_indices[sims[top_k_indices] > threshold]
        if len(reliable_indices) == 0:
            reliable_indices = top_k_indices[:min(1, k)]
        
        schema_key = tuple(sorted([relation_names[i] for i in reliable_indices]))
        groups[schema_key].append(passage_idx)
        
        if schema_key not in schemas:
            schemas[schema_key] = {
                relation_names[i]: relation_schema[relation_names[i]]
                for i in reliable_indices
            }
    
    stats = {
        'num_groups': len(groups),
        'max_group_size': max(len(v) for v in groups.values()) if groups else 0,
        'min_group_size': min(len(v) for v in groups.values()) if groups else 0,
        'avg_group_size': np.mean([len(v) for v in groups.values()]) if groups else 0,
        'threshold': threshold,
    }
    logger.info(f"Groups: {stats['num_groups']} | Size: {stats['min_group_size']}-{stats['max_group_size']}")
    
    return {'groups': groups, 'schemas': schemas, 'embedding_stats': stats}
