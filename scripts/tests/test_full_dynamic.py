#!/usr/bin/env python3
"""
Full rebuild test with dynamic schema grouping.
Tests the optimized extraction on bigger sample.
"""
import sys
sys.path.insert(0, ".")

import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.data.musique_loader import load_musique, get_all_passages
from src.extraction import extract_triples_batch, get_extractor

# Load larger sample
logger.info("Loading MuSiQue dataset (full dev set)...")
samples = load_musique("dev")
passages = get_all_passages(samples)
logger.info(f"✓ Loaded {len(passages)} passages")

# Test schema
test_schema = {
    "born_in":     "Person was born in a location",
    "died_in":     "Person died in a location",
    "located_in":  "Entity located in a place",
    "worked_at":   "Person worked at an organization",
    "member_of":   "Person is member of an organization",
    "founded_by":  "Organization was founded by a person",
    "part_of":     "Organization is part of another organization",
    "educated_at": "Person studied at an organization",
    "occurred_in": "Event occurred in a location",
}

extractor = get_extractor()

logger.info("\n" + "="*70)
logger.info("Test 1: Dynamic extraction (full dataset, batched)")
logger.info("="*70)

start = time.time()
results_dynamic = extract_triples_batch(
    passages,
    relation_schema=test_schema,
    batch_size=100,  # Batch size for passage groups
    use_dynamic=True,
    k=8,
    skip_cache=True,
    deduplicate=False,
    extractor=extractor,
    save_cache_to_disk=False
)
dynamic_time = time.time() - start

dynamic_triples = sum(len(r.get("triples", [])) for r in results_dynamic)
logger.info(f"\n✓ Dynamic extraction completed:")
logger.info(f"  Time: {dynamic_time:.1f}s")
logger.info(f"  Triples: {dynamic_triples}")
logger.info(f"  Rate: {len(passages)/dynamic_time:.1f} passages/sec ({dynamic_triples/dynamic_time:.1f} triples/sec)")

logger.info("\n" + "="*70)
logger.info("Full rebuild ready!")
logger.info("="*70)
logger.info("Command to run full rebuild:")
logger.info("  python rebuild_graph.py")
