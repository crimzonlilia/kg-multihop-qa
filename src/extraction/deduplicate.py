"""Deduplication logic for extracted triples."""

import logging

logger = logging.getLogger(__name__)


def deduplicate_triples(results: list[dict]) -> list[dict]:
    """
    Merge duplicate triples from multiple passages.
    Aggregates confidence scores and counts frequencies.
    
    Args:
        results: List of {'passage': str, 'entities': list, 'triples': list}
    
    Returns:
        List with single element: {'passage': None, 'entities': [], 'triples': [dedup_list]}
    """
    triangle_groups = {}
    
    for r in results:
        for t in r["triples"]:
            key = (t["subject"].lower(), t["relation"], t["object"].lower())
            if key not in triangle_groups:
                triangle_groups[key] = []
            triangle_groups[key].append(t["score"])
    
    # Create deduplicated triples with averaged confidence and frequency
    dedup_triples = []
    for (subj, rel, obj), scores in triangle_groups.items():
        valid_scores = [s for s in scores if s is not None]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None
        
        dedup_triples.append({
            "subject": subj,
            "relation": rel,
            "object": obj,
            "score": avg_score,
            "freq": len(scores)
        })
    
    # Sort by frequency (highest first)
    dedup_triples.sort(key=lambda x: x["freq"], reverse=True)
    
    logger.info(f"Deduplicated {len(triangle_groups)} unique triples from {len(results)} passages")
    
    return [{"passage": None, "entities": [], "triples": dedup_triples}]
