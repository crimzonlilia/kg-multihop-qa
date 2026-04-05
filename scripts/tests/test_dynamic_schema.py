#!/usr/bin/env python3
"""
Quick test of dynamic schema grouping extraction.
Tests the extract_triples_batch() function with use_dynamic=True.
"""
import sys
sys.path.insert(0, ".")

import logging
from datetime import datetime
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.data.musique_loader import load_musique, get_all_passages
from src.extraction import extract_triples_batch

# Load small sample for quick test
logger.info("Loading MuSiQue dataset (50 samples)...")
samples = load_musique("dev", max_samples=50)
passages = get_all_passages(samples)
logger.info(f"✓ Loaded {len(passages)} passages")

# Create a simple schema for testing
test_schema = {
    "born_in":     "Person was born in a location",
    "died_in":     "Person died in a location", 
    "located_in":  "Entity located in a place",
    "worked_at":   "Person worked at an organization",
    "member_of":   "Person is member of an organization",
}

logger.info("\n" + "="*60)
logger.info("Testing DYNAMIC SCHEMA EXTRACTION (optimized)")
logger.info("="*60)

start = time.time()
results_dynamic = extract_triples_batch(
    passages,
    relation_schema=test_schema,
    batch_size=25,
    use_dynamic=True,
    k=5,
    skip_cache=True,
    deduplicate=False,
    save_cache_to_disk=False
)
dynamic_time = time.time() - start

logger.info("\n" + "="*60)
logger.info("Testing STANDARD BATCH EXTRACTION")
logger.info("="*60)

start = time.time()
results_standard = extract_triples_batch(
    passages,
    relation_schema=test_schema,
    batch_size=25,
    skip_cache=True,
    deduplicate=False,
    save_cache_to_disk=False
)
standard_time = time.time() - start

logger.info("\n" + "="*60)
logger.info("RESULTS COMPARISON")
logger.info("="*60)

dynamic_triples = sum(len(r.get("triples", [])) for r in results_dynamic)
standard_triples = sum(len(r.get("triples", [])) for r in results_standard)

logger.info(f"Dynamic extraction:  {dynamic_triples} triples in {dynamic_time:.2f}s ({dynamic_triples/dynamic_time:.1f} per sec)")
logger.info(f"Standard extraction: {standard_triples} triples in {standard_time:.2f}s ({standard_triples/standard_time:.1f} per sec)")
logger.info(f"Speedup: {standard_time/dynamic_time:.2f}x")

# Show sample results
logger.info("\nSample triples (first 5 from dynamic extraction):")
count = 0
for r in results_dynamic:
    if r.get("passage"):
        for t in r.get("triples", [])[:2]:
            logger.info(f"  • ({t['subject']}, {t['relation']}, {t['object']}) [{t['score']:.2f}]")
            count += 1
            if count >= 5:
                break
    if count >= 5:
        break

logger.info("\n✓ Test complete!")
