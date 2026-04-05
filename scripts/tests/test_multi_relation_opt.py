"""Quick test: Multi-relation extraction optimization with 200 samples."""

import json
import time
import logging
from pathlib import Path
from src.extraction import extract_triples_batch
from src.extraction.model import get_extractor, get_embedder
from src.extraction.schema import get_dynamic_schemas_grouped, DEFAULT_RELATION_SCHEMA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load batch 1 data (musique)
musique_path = Path("data/raw/musique/data/musique_ans_v1.0_train.jsonl")
passages_list = []

with open(musique_path) as f:
    for idx, line in enumerate(f):
        if idx >= 50:  # Just 50 QA pairs for quick test = ~300-400 passages
            break
        data = json.loads(line)
        # Musique has "paragraphs" list with {"idx": ..., "title": ..., "paragraph_text": ...}
        if "paragraphs" in data:
            for para in data["paragraphs"]:
                if isinstance(para, dict) and "paragraph_text" in para:
                    passages_list.append(para["paragraph_text"])

passages = passages_list

logger.info(f"Loaded {len(passages)} passages for testing")

# ============================================================================
# TEST 1: Dynamic Schema Grouping Analysis
# ============================================================================
logger.info("\n" + "="*70)
logger.info("TEST 1: Dynamic Schema Grouping Analysis")
logger.info("="*70)

t0 = time.time()
embedder = get_embedder()
grouping = get_dynamic_schemas_grouped(passages, DEFAULT_RELATION_SCHEMA, embedder=embedder, k=8)
schema_time = time.time() - t0

groups = grouping['groups']
schemas = grouping['schemas']
stats = grouping['embedding_stats']

logger.info(f"✓ Schema grouping time: {schema_time:.2f}s")
logger.info(f"  - Unique schemas: {len(groups)}")
logger.info(f"  - Group sizes: min={stats['min_group_size']}, max={stats['max_group_size']}, avg={stats['avg_group_size']:.1f}")
logger.info(f"  - Passages covered: {sum(len(v) for v in groups.values())}")

# Show sample schemas
logger.info(f"\n  Sample schemas (top 5 by size):")
sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
for i, (schema_key, indices) in enumerate(sorted_groups[:5]):
    relations = list(schema_key)
    logger.info(f"    {i+1}. {len(indices):3d} passages | Relations: {', '.join(relations[:3])}")

# ============================================================================
# TEST 2: Multi-relation Extraction (Dynamic)
# ============================================================================
logger.info("\n" + "="*70)
logger.info("TEST 2: Multi-relation Extraction (Dynamic Mode)")
logger.info("="*70)

t0 = time.time()
results_dynamic = extract_triples_batch(passages, relation_schema=DEFAULT_RELATION_SCHEMA, use_dynamic=True, k=8)
extract_time_dynamic = time.time() - t0

total_triples_dynamic = sum(len(r['triples']) for r in results_dynamic)
logger.info(f"✓ Extraction time (dynamic): {extract_time_dynamic:.2f}s")
logger.info(f"  - Total triples: {total_triples_dynamic}")
logger.info(f"  - Avg triples/passage: {total_triples_dynamic / len(results_dynamic):.2f}")

# Sample triples
logger.info(f"\n  Sample triples (first 10):")
triple_count = 0
for result in results_dynamic:
    for triple in result['triples'][:3]:
        if triple_count < 10:
            logger.info(f"    • ({triple.get('head', '?')}, {triple.get('relation', '?')}, {triple.get('tail', '?')})")
            triple_count += 1

# ============================================================================
# TEST 3: Standard Extraction (No Dynamic) - For Comparison
# ============================================================================
logger.info("\n" + "="*70)
logger.info("TEST 3: Standard Extraction (No Dynamic) - For Baseline")
logger.info("="*70)

t0 = time.time()
results_standard = extract_triples_batch(passages, relation_schema=DEFAULT_RELATION_SCHEMA, use_dynamic=False)
extract_time_standard = time.time() - t0

total_triples_standard = sum(len(r['triples']) for r in results_standard)
logger.info(f"✓ Extraction time (standard): {extract_time_standard:.2f}s")
logger.info(f"  - Total triples: {total_triples_standard}")
logger.info(f"  - Avg triples/passage: {total_triples_standard / len(results_standard):.2f}")

# ============================================================================
# COMPARISON
# ============================================================================
logger.info("\n" + "="*70)
logger.info("COMPARISON & EFFICIENCY ANALYSIS")
logger.info("="*70)

speedup = extract_time_standard / extract_time_dynamic
logger.info(f"Speedup (dynamic vs standard): {speedup:.2f}x")
logger.info(f"  - Standard total time: {extract_time_standard:.2f}s")
logger.info(f"  - Dynamic total time:  {extract_time_dynamic:.2f}s")
logger.info(f"  - Time saved: {extract_time_standard - extract_time_dynamic:.2f}s")

time_overhead = extract_time_dynamic - schema_time
logger.info(f"\nTime breakdown (dynamic):")
logger.info(f"  - Schema grouping: {schema_time:.2f}s ({schema_time/extract_time_dynamic*100:.1f}%)")
logger.info(f"  - Extraction: {time_overhead:.2f}s ({time_overhead/extract_time_dynamic*100:.1f}%)")

# ============================================================================
# MODEL CALL ESTIMATION
# ============================================================================
logger.info("\n" + "="*70)
logger.info("MODEL CALL ESTIMATION")
logger.info("="*70)

# Standard mode: N batches of 32 passages = N/32 calls
standard_calls = (len(passages) + 31) // 32
logger.info(f"Standard mode model calls: ~{standard_calls} (batch of 32)")

# Dynamic mode: unique schemas
dynamic_calls = len(groups)
logger.info(f"Dynamic mode model calls: {dynamic_calls} (unique schemas)")

call_reduction = standard_calls - dynamic_calls
call_reduction_pct = (call_reduction / standard_calls) * 100
logger.info(f"Reduction: {call_reduction} calls ({call_reduction_pct:.1f}%)")

logger.info("\n" + "="*70)
logger.info("✓ TEST COMPLETE")
logger.info("="*70)
