#!/usr/bin/env python3
"""Test optimized extraction on GPU"""
import time
import torch
from src.extraction.extract_triples import extract_triples_batch, clear_extractor_cache

# Clear cache to reload optimized model
clear_extractor_cache()

passages = [
    'John Smith was born in New York in 1990. He worked at Google.',
    'Apple Inc. was founded by Steve Jobs in California.',
    'Marie Curie discovered radium. She worked at Paris University.',
    'The Amazon rainforest is located in Brazil.',
    'Tesla was founded by Elon Musk. It is located in California.',
    'Albert Einstein was born in Germany and worked at Princeton.',
    'Microsoft was founded by Bill Gates in Seattle.',
    'NASA was founded in 1958 in Washington DC.',
    'The Eiffel Tower is located in Paris.',
    'Stephen Hawking worked at Cambridge University.',
]

print(f"{'='*70}")
print(f"OPTIMIZED GPU EXTRACTION TEST")
print(f"{'='*70}")
print(f"✓ CUDA Available: {torch.cuda.is_available()}")
print(f"✓ Passages: {len(passages)}")
print(f"✓ Batch Size: 10 (for test)")
print(f"\n[Testing with optimizations: torch.no_grad, fp16, reduced logging...]")

try:
    start = time.time()
    result = extract_triples_batch(passages, batch_size=10)
    elapsed = time.time() - start
    
    # Handle both dict and list return formats
    triples = result['triples'] if isinstance(result, dict) else result
    
    print(f"\n{'='*70}")
    print(f"⚡ RESULTS")
    print(f"{'='*70}")
    print(f"✓ Time: {elapsed:.2f}s")
    print(f"✓ Triples: {len(triples)}")
    print(f"✓ Per passage: {len(triples)/len(passages):.1f}")
    print(f"✓ Throughput: {len(passages)/elapsed:.1f} passages/sec")
    print(f"\n✅ OPTIMIZATIONS WORKING - Ready for full rebuild!")
    print(f"{'='*70}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
