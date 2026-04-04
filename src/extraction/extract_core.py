"""Core extraction logic for entities and relations."""

import logging
import torch
import json
import os
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)

# Entity filtering settings
_RAW_GENERIC = {
    'he', 'she', 'it', 'they', 'them', 'we', 'us', 'i', 'me', 'you',
    'one', 'this', 'that', 'these', 'those', 'the', 'a', 'an',
    'other', 'another', 'some', 'any', 'all', 'each', 'every',
    'person', 'people', 'thing', 'things', 'stuff', 'work'
}
GENERIC_ENTITIES = {e.lower() for e in _RAW_GENERIC}
MIN_ENTITY_LENGTH = 3
CONFIDENCE_MIN = 0.5

# Cache directory
CACHE_DIR = Path("data/cache")
TRIPLES_CACHE = CACHE_DIR / "triples.json"

# Setup environment
os.environ['OMP_NUM_THREADS'] = '8'


def normalize_entity_text(text: str) -> tuple[str, str]:
    """Returns (display_text, match_key) for entity matching."""
    display = text.strip()
    return display, display.lower()


def extract_entities_from_batch(entity_results: list, entity_labels: list) -> list[dict]:
    """Parse entity extraction results into standardized format."""
    entities = []
    
    for entity_result in entity_results:
        entity_dict = entity_result.get("entities", {})
        
        for label in entity_labels:
            if label not in entity_dict:
                continue
            
            items = entity_dict[label]
            for item in items:
                entity_text = item.get("text", "") if isinstance(item, dict) else str(item)
                confidence = item.get("confidence") if isinstance(item, dict) else None
                
                display, match_key = normalize_entity_text(entity_text)
                entities.append({
                    "text": display,
                    "text_lower": match_key,
                    "type": label,
                    "score": confidence,
                })
    
    return entities


def filter_triple(head: tuple, tail: tuple, score: float, relation: str,
                 entity_types: dict, type_constraints: dict) -> bool:
    """
    Determine if a triple should be filtered out.
    Returns True if triple SHOULD BE FILTERED (rejected), False if KEPT.
    """
    head_display, head_key = head
    tail_display, tail_key = tail
    
    # Filter by confidence
    if score < CONFIDENCE_MIN:
        return True
    
    # Filter generic entities
    if head_key in GENERIC_ENTITIES or tail_key in GENERIC_ENTITIES:
        return True
    
    # Filter short entities (likely abbreviations)
    if len(head_key) < MIN_ENTITY_LENGTH or len(tail_key) < MIN_ENTITY_LENGTH:
        return True
    
    # Filter by type constraints
    if relation in type_constraints:
        expected_subj, expected_objs = type_constraints[relation]
        
        # Skip type check if constraints are None (permissive relations)
        if expected_subj is None or expected_objs is None:
            return False
        
        # Check if entities have known types
        if head_key in entity_types and tail_key in entity_types:
            head_type = entity_types[head_key]
            tail_type = entity_types[tail_key]
            
            if head_type != expected_subj or tail_type not in expected_objs:
                return True
    
    return False


def extract_relations_from_batch(relation_results: list, entity_types: dict,
                                type_constraints: dict) -> list[list[dict]]:
    """Parse relation extraction results into standardized format."""
    batch_triples = []
    
    for relation_result in relation_results:
        triples = []
        relation_dict = relation_result.get("relation_extraction", {})
        
        for rel, pairs in relation_dict.items():
            for pair in pairs:
                head = pair.get("head", {})
                tail = pair.get("tail", {})
                
                head_text = head.get("text", "") if isinstance(head, dict) else str(head)
                tail_text = tail.get("text", "") if isinstance(tail, dict) else str(tail)
                head_score = head.get("confidence", 1.0) if isinstance(head, dict) else 1.0
                tail_score = tail.get("confidence", 1.0) if isinstance(tail, dict) else 1.0
                score = min(head_score, tail_score)
                
                head_display, head_key = normalize_entity_text(head_text)
                tail_display, tail_key = normalize_entity_text(tail_text)
                
                # Filter triple
                if filter_triple((head_display, head_key), (tail_display, tail_key),
                               score, rel, entity_types, type_constraints):
                    continue
                
                triples.append({
                    "subject": head_display,
                    "relation": rel,
                    "object": tail_display,
                    "score": score
                })
        
        batch_triples.append(triples)
    
    return batch_triples


def load_cache() -> list[dict] | None:
    """Load triples from cache if available."""
    if not TRIPLES_CACHE.exists():
        return None
    
    try:
        with open(TRIPLES_CACHE) as f:
            cached = json.load(f)
        logger.info(f"✓ Loaded cache: {len(cached)} passages")
        return cached
    except Exception as e:
        logger.warning(f"Could not load cache: {e}")
        return None


def save_cache(results: list[dict], merge_existing: bool = True):
    """Save extraction results to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    if merge_existing:
        existing = []
        if TRIPLES_CACHE.exists():
            try:
                with open(TRIPLES_CACHE) as f:
                    existing = json.load(f)
            except:
                pass
        
        all_results = existing + results
    else:
        all_results = results
    
    with open(TRIPLES_CACHE, "w") as f:
        json.dump(all_results, f)
    
    total_triples = sum(len(r['triples']) for r in all_results)
    logger.info(f"✓ Saved cache: {total_triples} triples from {len(all_results)} passages")


def extract_triples_batch(passages: list[str],
                         relation_schema: dict = None,
                         entity_labels: list[str] = None,
                         type_constraints: dict = None,
                         batch_size: int = 32,
                         use_dynamic: bool = False,
                         k: int = 8,
                         threshold: float = None,
                         skip_cache: bool = True,
                         deduplicate: bool = True,
                         extractor=None,
                         save_cache_to_disk: bool = True) -> list[dict]:
    """
    Extract triples from passages using GLiNER2.
    
    Args:
        passages: List of text passages
        relation_schema: Dict of relation definitions
        entity_labels: List of entity types to extract
        type_constraints: Dict of type constraints per relation
        batch_size: Batch size for extraction
        use_dynamic: Use dynamic schema grouping (slower but more accurate)
        k: Number of top relations for dynamic schema
        threshold: Confidence threshold for dynamic schema
        skip_cache: Ignore cache if exists
        deduplicate: Merge duplicate triples
        extractor: Pre-loaded GLiNER2 model
        save_cache_to_disk: Save results to cache file
    
    Returns: List of {'passage': str, 'entities': list, 'triples': list}
    """
    from .model import get_extractor
    from .schema import (
        DEFAULT_RELATION_SCHEMA, TYPE_CONSTRAINTS, ENTITY_LABELS,
        build_schema, get_dynamic_schemas_grouped
    )
    from .deduplicate import deduplicate_triples
    
    # Use defaults
    relation_schema = relation_schema or DEFAULT_RELATION_SCHEMA
    entity_labels = entity_labels or ENTITY_LABELS
    type_constraints = type_constraints or TYPE_CONSTRAINTS
    
    # Load cache
    if not skip_cache:
        cached = load_cache()
        if cached:
            return cached
    
    # Get or load model
    if extractor is None:
        extractor = get_extractor()
    
    schema = build_schema(extractor, relation_schema)
    
    results = []
    total = len(passages)
    logger.info(f"Extracting triples from {total} passages (batch_size={batch_size}, dynamic={use_dynamic})...")
    
    with torch.no_grad():
        # Dynamic schema grouping (optional)
        if use_dynamic:
            logger.info("Using dynamic schema grouping...")
            grouping = get_dynamic_schemas_grouped(
                passages, relation_schema, k=k, threshold=threshold
            )
            groups = grouping['groups']
            schemas = grouping['schemas']
            
            for schema_key, passage_indices in groups.items():
                group_passages = [passages[i] for i in passage_indices]
                group_schema = schemas[schema_key]
                
                # Extract for group
                entity_results = extractor.batch_extract_entities(
                    group_passages, entity_labels, include_confidence=True, batch_size=batch_size
                )
                relation_results = extractor.batch_extract_relations(
                    group_passages, group_schema, include_confidence=True, batch_size=batch_size
                )
                
                # Process results
                for group_idx, pass_idx in enumerate(passage_indices):
                    # Build entity type map
                    entity_types = {}
                    if group_idx < len(entity_results):
                        for ent in extract_entities_from_batch([entity_results[group_idx]], entity_labels):
                            entity_types[ent['text_lower']] = ent['type']
                    
                    # Extract triples
                    if group_idx < len(relation_results):
                        batch_triples = extract_relations_from_batch(
                            [relation_results[group_idx]], entity_types, type_constraints
                        )
                        triples = batch_triples[0] if batch_triples else []
                    else:
                        triples = []
                    
                    results.append({
                        "passage": passages[pass_idx],
                        "entities": [],
                        "triples": triples
                    })
        else:
            # Standard batch processing (simpler, faster)
            for i in range(0, total, batch_size):
                batch = passages[i:i+batch_size]
                batch_num = (i // batch_size) + 1
                
                try:
                    # Extract entities and relations
                    entity_results = extractor.batch_extract_entities(
                        batch, entity_labels, include_confidence=True, batch_size=batch_size
                    )
                    relation_results = extractor.batch_extract_relations(
                        batch, relation_schema, include_confidence=True, batch_size=batch_size
                    )
                    
                    # Process results
                    for idx, passage in enumerate(batch):
                        # Build entity type map
                        entity_types = {}
                        entities = []
                        if idx < len(entity_results):
                            entities = extract_entities_from_batch([entity_results[idx]], entity_labels)
                            for ent in entities:
                                entity_types[ent['text_lower']] = ent['type']
                        
                        # Extract triples
                        triples = []
                        if idx < len(relation_results):
                            batch_triples = extract_relations_from_batch(
                                [relation_results[idx]], entity_types, type_constraints
                            )
                            triples = batch_triples[0] if batch_triples else []
                        
                        results.append({
                            "passage": passage,
                            "entities": entities,
                            "triples": triples
                        })
                    
                    if batch_num % 5 == 0:
                        total_triples = sum(len(r['triples']) for r in results)
                        logger.info(f"[Batch {batch_num}: {i+len(batch)}/{total}] {total_triples} triples")
                
                except Exception as e:
                    logger.error(f"Batch {batch_num} failed: {type(e).__name__}: {str(e)}")
                    for passage in batch:
                        results.append({"passage": passage, "entities": [], "triples": []})
    
    # Save cache
    if save_cache_to_disk:
        save_cache(results, merge_existing=True)
    
    # Deduplicate if requested
    if deduplicate:
        results = deduplicate_triples(results)
    
    return results
