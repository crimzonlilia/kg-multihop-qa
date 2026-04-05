"""Core extraction logic for entities and relations."""

import logging
import torch
import json
import os
import hashlib
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
EXTRACTION_CHECKPOINT = CACHE_DIR / "extraction_checkpoint.json"

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


def _fingerprint_passages(passages: list[str]) -> str:
    """Build a stable fingerprint so we only reuse cache for the same corpus."""
    digest = hashlib.sha1()
    for passage in passages:
        digest.update(str(len(passage)).encode("utf-8"))
        digest.update(b"::")
        digest.update(passage.encode("utf-8", errors="ignore"))
        digest.update(b"\n")
    return digest.hexdigest()


def _build_cache_metadata(passages: list[str], relation_schema: dict,
                          entity_labels: list[str], batch_size: int,
                          use_dynamic: bool, k: int, threshold: float,
                          extract_entities: bool) -> dict:
    """Describe the current extraction run for safe cache reuse."""
    relation_keys = sorted((relation_schema or {}).keys())
    return {
        "version": 2,
        "num_passages": len(passages),
        "fingerprint": _fingerprint_passages(passages),
        "relation_keys": relation_keys,
        "entity_labels": sorted(entity_labels or []),
        "batch_size": batch_size,
        "use_dynamic": use_dynamic,
        "k": k,
        "threshold": threshold,
        "extract_entities": extract_entities,
    }


def _cache_meta_matches(expected_meta: dict, actual_meta: dict) -> bool:
    """Check whether a saved cache matches the current run settings."""
    keys_to_compare = [
        "version", "num_passages", "fingerprint", "relation_keys",
        "entity_labels", "use_dynamic", "k", "threshold", "extract_entities",
    ]
    return all(expected_meta.get(key) == actual_meta.get(key) for key in keys_to_compare)


def load_cache(expected_meta: dict | None = None) -> list[dict] | None:
    """Load triples from cache if available and compatible with the current run."""
    if not TRIPLES_CACHE.exists():
        return None

    try:
        with open(TRIPLES_CACHE) as f:
            cached = json.load(f)

        # Backward-compatible with the old plain-list cache format.
        if isinstance(cached, list):
            if expected_meta and len(cached) != expected_meta.get("num_passages", len(cached)):
                logger.info("Cache exists but does not match current passage count; ignoring it")
                return None
            logger.info(f"✓ Loaded legacy cache: {len(cached)} passages")
            return cached

        cached_meta = cached.get("meta", {})
        cached_results = cached.get("results", [])
        if expected_meta and not _cache_meta_matches(expected_meta, cached_meta):
            logger.info("Cache metadata mismatch; ignoring stale extraction cache")
            return None

        logger.info(f"✓ Loaded cache: {len(cached_results)} passages")
        return cached_results
    except Exception as e:
        logger.warning(f"Could not load cache: {e}")
        return None


def save_cache(results: list[dict], merge_existing: bool = False, metadata: dict | None = None):
    """Save extraction results to cache with run metadata for safe reuse."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    all_results = results
    if merge_existing and TRIPLES_CACHE.exists():
        try:
            with open(TRIPLES_CACHE) as f:
                existing_payload = json.load(f)
            existing_results = existing_payload if isinstance(existing_payload, list) else existing_payload.get("results", [])
            all_results = existing_results + results
        except Exception:
            all_results = results

    payload = {
        "meta": metadata or {"version": 2, "num_passages": len(all_results)},
        "results": all_results,
    }

    with open(TRIPLES_CACHE, "w") as f:
        json.dump(payload, f)

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
                         save_cache_to_disk: bool = True,
                         extract_entities: bool = True) -> list[dict]:
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
        extract_entities: Run entity extraction pass for type filtering.
            Set False for a faster but slightly noisier extraction mode.
    
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
    cache_meta = _build_cache_metadata(
        passages=passages,
        relation_schema=relation_schema,
        entity_labels=entity_labels,
        batch_size=batch_size,
        use_dynamic=use_dynamic,
        k=k,
        threshold=threshold,
        extract_entities=extract_entities,
    )
    if not skip_cache:
        cached = load_cache(expected_meta=cache_meta)
        if cached:
            return deduplicate_triples(cached) if deduplicate else cached
    
    # Get or load model
    if extractor is None:
        extractor = get_extractor()
    
    schema = build_schema(extractor, relation_schema)
    
    results = []
    total = len(passages)
    all_relations = list(relation_schema.keys())
    logger.info(
        f"Extracting triples from {total} passages "
        f"(batch_size={batch_size}, dynamic={use_dynamic}, entities={extract_entities})..."
    )
    
    with torch.inference_mode():
        # Dynamic schema grouping (optional)
        if use_dynamic:
            logger.info("Using dynamic schema grouping...")
            grouping = get_dynamic_schemas_grouped(
                passages, relation_schema, k=k, threshold=threshold
            )
            groups = grouping['groups']
            schemas = grouping['schemas']
            
            logger.info(f"Multi-relation extraction: {len(groups)} unique schemas (vs {total} passages)")
            
            # ← OPTIMIZATION: Batch by schema_key
            # All passages with same schema_key are extracted in ONE model call
            # Model internally handles sub-batching via batch_size parameter
            for schema_key, passage_indices in groups.items():
                all_passages_for_schema = [passages[i] for i in passage_indices]
                group_schema = schemas[schema_key]
                group_relations = list(group_schema.keys())  # Convert dict to list of relation names
                
                # Single model call for all passages with this schema
                # Model internally uses batch_size=32 to avoid GPU spike
                entity_results = []
                if extract_entities:
                    entity_results = extractor.batch_extract_entities(
                        all_passages_for_schema, entity_labels, include_confidence=True, batch_size=batch_size
                    )
                relation_results = extractor.batch_extract_relations(
                    all_passages_for_schema, group_relations, include_confidence=True, batch_size=batch_size
                )
                
                # Process all results from this schema batch
                for pos, pass_idx in enumerate(passage_indices):
                    # Build entity type map
                    entity_types = {}
                    if extract_entities and pos < len(entity_results):
                        for ent in extract_entities_from_batch([entity_results[pos]], entity_labels):
                            entity_types[ent['text_lower']] = ent['type']
                    
                    # Extract triples
                    if pos < len(relation_results):
                        batch_triples = extract_relations_from_batch(
                            [relation_results[pos]], entity_types, type_constraints
                        )
                        triples = batch_triples[0] if batch_triples else []
                    else:
                        triples = []
                    
                    results.append({
                        "passage": all_passages_for_schema[pos],  # Use actual passage text from batch
                        "entities": [],
                        "triples": triples
                    })
        else:
            # Standard batch processing with checkpoint/resume support
            checkpoint_every = 100  # save checkpoint every N batches

            # Resume from checkpoint if available
            resume_idx = 0
            if EXTRACTION_CHECKPOINT.exists():
                try:
                    with open(EXTRACTION_CHECKPOINT) as _f:
                        ckpt = json.load(_f)
                    if ckpt.get("total") == total:  # same dataset
                        resume_idx = ckpt["next_idx"]
                        results = ckpt["results"]
                        logger.info(f"Resuming from passage {resume_idx}/{total} ({len(results)} already done)")
                except Exception as _e:
                    logger.warning(f"Could not load checkpoint: {_e}")

            for i in range(resume_idx, total, batch_size):
                batch = passages[i:i+batch_size]
                batch_num = (i // batch_size) + 1
                
                try:
                    # Extract entities and relations
                    entity_results = []
                    if extract_entities:
                        entity_results = extractor.batch_extract_entities(
                            batch, entity_labels, include_confidence=True, batch_size=batch_size
                        )
                    relation_results = extractor.batch_extract_relations(
                        batch, all_relations, include_confidence=True, batch_size=batch_size
                    )
                    
                    # Process results
                    for idx, passage in enumerate(batch):
                        # Build entity type map
                        entity_types = {}
                        entities = []
                        if extract_entities and idx < len(entity_results):
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

                    # Save checkpoint every N batches
                    if batch_num % checkpoint_every == 0:
                        CACHE_DIR.mkdir(parents=True, exist_ok=True)
                        with open(EXTRACTION_CHECKPOINT, "w") as _f:
                            json.dump({"total": total, "next_idx": i + len(batch), "results": results}, _f)
                        logger.info(f"Checkpoint saved at passage {i + len(batch)}/{total}")
                
                except Exception as e:
                    logger.error(f"Batch {batch_num} failed: {type(e).__name__}: {str(e)}")
                    for passage in batch:
                        results.append({"passage": passage, "entities": [], "triples": []})

            # Remove checkpoint on successful completion
            if EXTRACTION_CHECKPOINT.exists():
                EXTRACTION_CHECKPOINT.unlink()
                logger.info("Checkpoint cleared after successful extraction")
    
    # Save cache
    if save_cache_to_disk:
        save_cache(results, merge_existing=False, metadata=cache_meta)
    
    # Deduplicate if requested
    if deduplicate:
        results = deduplicate_triples(results)
    
    return results
