"""Triple extraction pipeline with GLiNER2."""

from .model import get_extractor, get_embedder, clear_model_cache
from .schema import (
    DEFAULT_RELATION_SCHEMA,
    TYPE_CONSTRAINTS,
    ENTITY_LABELS,
    build_schema,
    get_dynamic_schemas_grouped,
)
from .extract_core import extract_triples_batch
from .deduplicate import deduplicate_triples

__all__ = [
    # Model
    "get_extractor",
    "get_embedder",
    "clear_model_cache",
    # Schema
    "DEFAULT_RELATION_SCHEMA",
    "TYPE_CONSTRAINTS",
    "ENTITY_LABELS",
    "build_schema",
    "get_dynamic_schemas_grouped",
    # Extraction
    "extract_triples_batch",
    # Deduplication
    "deduplicate_triples",
]
