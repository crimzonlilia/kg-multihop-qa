#!/usr/bin/env python3
"""
Test script showing model caching performance improvement.

Shows why step transitions lag:
- WITHOUT caching: Model reloads ~2-3 min per step
- WITH caching: Subsequent steps are instant ⚡

Usage:
    python test_model_caching.py
"""

import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from src.extraction.extract_triples import get_extractor, clear_extractor_cache

TEST_PASSAGES = [
    "Albert Einstein was born in Ulm, Germany. He worked at Princeton University.",
    "Marie Curie discovered polonium in Paris. She won the Nobel Prize.",
    "Stephen Hawking was a theoretical physicist at Cambridge University.",
]

def test_without_caching():
    """Simulate old behavior: reload model each time."""
    print("\n" + "="*70)
    print("WITHOUT CACHING (Old behavior - SLOW)")
    print("="*70)
    
    # Clear cache to force reloads
    clear_extractor_cache()
    
    times = []
    for step in range(3):
        print(f"\n🔄 Step {step+1}: Extracting triples...")
        clear_extractor_cache()  # ← Force reload
        
        start = time.time()
        extractor = get_extractor()  # ← Reloads model (2-3 min)
        elapsed = time.time() - start
        times.append(elapsed)
        
        print(f"   Model load time: {elapsed:.1f}s")
    
    total = sum(times)
    print(f"\n📊 Total time for 3 steps: {total:.1f}s")
    print(f"   Average per step: {total/3:.1f}s")
    return total

def test_with_caching():
    """New behavior: load model once, reuse."""
    print("\n" + "="*70)
    print("WITH CACHING (New behavior - FAST) ⚡")
    print("="*70)
    
    clear_extractor_cache()
    
    times = []
    for step in range(3):
        print(f"\n🚀 Step {step+1}: Extracting triples...")
        
        start = time.time()
        extractor = get_extractor()  # ← First call loads, rest are instant
        elapsed = time.time() - start
        times.append(elapsed)
        
        print(f"   Model access time: {elapsed:.1f}s (cached)")
    
    total = sum(times)
    print(f"\n📊 Total time for 3 steps: {total:.1f}s")
    print(f"   Average per step: {total/3:.1f}s")
    return total

if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║           MODEL CACHING PERFORMANCE TEST                          ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    print("Testing first call (loads model)...")
    t0 = time.time()
    get_extractor()
    initial_load = time.time() - t0
    print(f"✓ Initial load: {initial_load:.1f}s")
    
    print("\nTesting cached access (should be instant)...")
    t0 = time.time()
    get_extractor()
    cached_access = time.time() - t0
    
    # Handle case where cached access is faster than timer precision
    if cached_access < 0.0001:
        print(f"✓ Cached access: <0.1ms (essentially instant)")
        speedup = initial_load / 0.0001  # Assume minimum precision
        print(f"  Speedup: >{speedup:.0f}x faster!")
    else:
        speedup = initial_load / cached_access
        print(f"✓ Cached access: {cached_access:.4f}s (speedup: {speedup:.0f}x)")
    
    print("\n" + "="*70)
    print("KEY INSIGHT")
    print("="*70)
    print("""
Why you experience lag when switching steps:

❌ OLD: Each step transition → Model reload (2-3 min lag)
   Step 1: extract_triples → LOAD MODEL (slow)
   Step 2: build_graph     → RELOAD MODEL (slow)  ← LAG!
   Step 3: run_pagerank    → RELOAD MODEL (slow)  ← LAG!

✅ NEW: Model loads once, reused across steps (instant)
   Step 1: extract_triples → LOAD MODEL (first time only)
   Step 2: build_graph     → USE CACHED MODEL ⚡ (instant)
   Step 3: run_pagerank    → USE CACHED MODEL ⚡ (instant)

Solution: Use get_extractor() instead of GLiNER2.from_pretrained()
    """)
    
    print("\nQUICK START:")
    print("─" * 70)
    print("""
# Old way (SLOW):
from gliner2 import GLiNER2
extractor = GLiNER2.from_pretrained("model_name")  # Reloads every time

# New way (FAST):
from src.extraction.extract_triples import get_extractor
extractor = get_extractor()  # Loads once, reuses automatically
    """)
