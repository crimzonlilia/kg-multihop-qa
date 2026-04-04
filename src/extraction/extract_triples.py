"""
Legacy wrapper for backward compatibility.
Import from modules instead.
"""

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import and re-export public API from modules
from . import (
    get_extractor,
    get_embedder,
    clear_model_cache,
    DEFAULT_RELATION_SCHEMA,
    TYPE_CONSTRAINTS,
    ENTITY_LABELS,
    build_schema,
    get_dynamic_schemas_grouped,
    extract_triples_batch,
    deduplicate_triples,
)

# For backward compatibility: additional exports
from .extract_core import (
    normalize_entity_text,
    filter_triple,
    extract_entities_from_batch,
    extract_relations_from_batch,
)

__all__ = [
    "get_extractor",
    "get_embedder",
    "clear_model_cache",
    "DEFAULT_RELATION_SCHEMA",
    "TYPE_CONSTRAINTS",
    "ENTITY_LABELS",
    "build_schema",
    "get_dynamic_schemas_grouped",
    "extract_triples_batch",
    "deduplicate_triples",
    "normalize_entity_text",
    "filter_triple",
    "extract_entities_from_batch",
    "extract_relations_from_batch",
]

logger.info("Using modular extraction pipeline (model.py, schema.py, extract_core.py, deduplicate.py)")


# For backward compatibility - add main block for direct usage
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data.musique_loader import load_musique, get_all_passages
    
    # Example usage
    samples = load_musique("dev", max_samples=20)
    passages = get_all_passages(samples)
    
    results = extract_triples_batch(
        passages,
        relation_schema=DEFAULT_RELATION_SCHEMA,
        deduplicate=True
    )
    
    # Sample output
    for r in results[:3]:
        if r.get('passage'):
            print(f"\nPassage: {r['passage'][:80]}...")
        for t in r["triples"]:
            freq_str = f" (freq: {t.get('freq', 1)})" if t.get('freq') else ""
            print(f"  ({t['subject']}, {t['relation']}, {t['object']}){freq_str}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data.musique_loader import load_musique, get_all_passages
    from src.extraction.relation_discovery import run_discovery

    samples = load_musique("dev", max_samples=20)
    passages = get_all_passages(samples)

    # Load expanded schema từ Pass 1
    expanded_schema = run_discovery(passages, min_cluster_freq=3)

    # Pass 2 với dynamic schema
    results = extract_triples_batch(passages, relation_schema=expanded_schema, deduplicate=True)

    # Sample output
    for r in results[:3]:
        if r['passage']:  # Nếu chưa deduplicate
            print(f"\nPassage: {r['passage'][:80]}...")
        for t in r["triples"]:
            freq_str = f" (freq: {t.get('freq', 1)})" if t.get('freq') else ""
            print(f"  ({t['subject']}, {t['relation']}, {t['object']}){freq_str}")