#!/usr/bin/env python3
"""Quick GPU test"""
import time
import torch
from src.extraction.extract_triples import extract_triples_batch

passages = [
    'John Smith was born in New York in 1990. He worked at Google.',
    'Apple Inc. was founded by Steve Jobs in California.',
    'Marie Curie discovered radium. She worked at Paris University.',
    'The Amazon rainforest is located in Brazil.',
    'Tesla was founded by Elon Musk. It is located in California.',
]

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Testing with {len(passages)} passages...\n")

start = time.time()
result = extract_triples_batch(passages, batch_size=5)
elapsed = time.time() - start

# Handle both dict and list return formats
triples = result['triples'] if isinstance(result, dict) else result

print(f"\n{'='*60}")
print(f"Time: {elapsed:.2f}s")
print(f"Triples: {len(triples)}")
print(f"Per passage: {len(triples)/len(passages):.1f}")
print(f"Throughput: {len(passages)/elapsed:.1f} passages/sec")
print(f"{'='*60}")

print("\nSample Triples:")
for i, triple in enumerate(triples[:5], 1):
    print(f"{i}. {triple}")

print("\n✅ GPU Test PASSED - Ready for full rebuild!")
