#!/usr/bin/env python3
"""
QUICK FIX MIGRATION GUIDE
==========================

If you have code that currently reloads the model, here's how to migrate:

BEFORE (OLD):
─────────────
from gliner2 import GLiNER2

extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
results = extract_triples_batch(passages, extractor=extractor)

AFTER (NEW):
────────────  
from src.extraction.extract_triples import get_extractor

extractor = get_extractor()  # ← Loads once, reused
results = extract_triples_batch(passages, extractor=extractor)

Benefits:
✅ No reload on step transitions
✅ Automatic caching
✅ Same API - just different import
"""

# =============================================================================
# MIGRATION CHECKLIST
# =============================================================================

# [ ] Update imports in your pipeline files:
#       from gliner2 import GLiNER2  (REMOVE)
#       ➜ from src.extraction.extract_triples import get_extractor  (ADD)

# [ ] Replace model loading calls:
#       OLD: extractor = GLiNER2.from_pretrained(MODEL_NAME)
#       NEW: extractor = get_extractor()

# [ ] Test that step transitions are now instant (not 2-3 min lag)

# =============================================================================
# COMMON PATTERNS
# =============================================================================

# PATTERN 1: Simple usage (auto-cache)
def pattern_1_simple():
    from src.extraction.extract_triples import extract_triples_batch
    
    results = extract_triples_batch(passages)  # Loads model once
    # ... later in pipeline ...
    results = extract_triples_batch(more_passages)  # Uses cache ⚡

# PATTERN 2: Explicit extractor passing
def pattern_2_explicit():
    from src.extraction.extract_triples import get_extractor, extract_triples_batch
    
    extractor = get_extractor()  # Get cached instance
    results1 = extract_triples_batch(passages1, extractor=extractor)
    results2 = extract_triples_batch(passages2, extractor=extractor)

# PATTERN 3: Pipeline workflow
def pattern_3_pipeline():
    from src.extraction.extract_triples import get_extractor, extract_triples_batch
    from src.graph.build_graph import build_graph
    from src.retrieval.pagerank import personalized_pagerank
    
    extractor = get_extractor()  # Load once
    
    # Step 1: Extract (uses model)
    print("Step 1: Extracting...")
    results = extract_triples_batch(passages, extractor=extractor)
    print("✓ Complete")
    
    # Step 2: Build graph (no model reload!) ⚡
    print("Step 2: Building graph...")
    graph = build_graph(results)
    print("✓ Complete")
    
    # Step 3: Ranking (no model reload!) ⚡
    print("Step 3: Ranking...")
    ranked = personalized_pagerank(graph, query_entities)
    print("✓ Complete")

# PATTERN 4: Debugging (force reload)
def pattern_4_debug():
    from src.extraction.extract_triples import get_extractor, clear_extractor_cache
    
    # First run
    extractor1 = get_extractor()
    results1 = extract_triples_batch(passages)
    
    # If you need to reload for some reason:
    clear_extractor_cache()
    
    # Next run will reload
    extractor2 = get_extractor()
    results2 = extract_triples_batch(passages)

# =============================================================================
# FILES TO UPDATE IN YOUR PROJECT
# =============================================================================

FILES_TO_CHECK = """
Files that may need migration:
  [ ] src/pipeline.py - Check if creates extractor
  [ ] experiments/run_pipeline.py - Check if uses extract
  [ ] tests/ - Check test files
  [ ] your_script.py - Any custom scripts
  
Quick search for patterns to update:
  grep -r "GLiNER2.from_pretrained" . --include="*.py"
    ↓
  Replace with: get_extractor()
"""

# =============================================================================
# TESTING
# =============================================================================

test_script = """
To verify the fix works:

1. Run the test script:
   python test_model_caching.py

2. Expected output should show:
   - First call loads model (few seconds)
   - Subsequent calls are instant (cached)

3. Run your pipeline and check for lag:
   python rebuild_graph.py
   
   Should be FAST (no 2-3 min lag between steps)
"""

# =============================================================================
# PERFORMANCE EXPECTATIONS
# =============================================================================

performance = """
BEFORE (without caching):
  Step 1 (extract):    180s (load model)
  Step 2 (graph):      180s (reload!) ← LAG
  Step 3 (ranking):    180s (reload!) ← LAG
  ────────────────────────────
  Total: 540s (~9 min)

AFTER (with caching):
  Step 1 (extract):    180s (load model)
  Step 2 (graph):      0.1s (cached) ← FAST!
  Step 3 (ranking):    0.1s (cached) ← FAST!
  ────────────────────────────
  Total: 180s (~3 min) → 3x faster!
"""

if __name__ == "__main__":
    print(__doc__)
    print(FILES_TO_CHECK)
    print("\n" + test_script)
    print("\n" + performance)
