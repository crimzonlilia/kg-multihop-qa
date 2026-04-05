#!/usr/bin/env python3
"""
Test GPU usage in actual extract_triples batch processing
"""
import torch
import time
from src.extraction.extract_triples import get_extractor, extract_triples_batch

def test_gpu_extraction():
    print("=" * 60)
    print("GPU EXTRACTION TEST - Before Full Rebuild")
    print("=" * 60)
    
    # 1. Check GPU
    print(f"\n✓ PyTorch Version: {torch.__version__}")
    print(f"✓ CUDA Available: {torch.cuda.is_available()}")
    print(f"✓ CUDA Version: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"✓ GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 2. Load model
    print("\n[Loading GLiNER2 Model...]")
    extractor = get_extractor()
    device = next(extractor.model.parameters()).device
    print(f"✓ Model Device: {device}")
    print(f"✓ Model Parameters: {sum(p.numel() for p in extractor.model.parameters()):,}")
    
    # 3. Test batch extraction with timing
    test_passages = [
        "Albert Einstein was born in Germany and worked in Princeton. He developed the theory of relativity.",
        "Marie Curie was a Polish physicist who won the Nobel Prize. She discovered radium.",
        "Stephen Hawking was born in Oxford and worked in Cambridge. He studied black holes.",
        "Isaac Newton was born in England and worked at Cambridge University. He discovered gravity.",
        "Galileo Galilei was born in Italy and worked in Padua and Pisa. He invented the telescope.",
    ]
    
    print(f"\n[Testing Batch Extraction with {len(test_passages)} passages...]")
    
    # Warm up (first run slower due to JIT)
    print("  Warm-up run...")
    start = time.time()
    result = extract_triples_batch(test_passages)
    warmup_time = time.time() - start
    print(f"  Warm-up time: {warmup_time:.2f}s")
    
    # Actual benchmark
    print("  Benchmark run...")
    start = time.time()
    result = extract_triples_batch(test_passages)
    batch_time = time.time() - start
    
    print(f"\n✓ Batch Processing Results:")
    print(f"  - Total time: {batch_time:.2f}s")
    print(f"  - Per passage: {batch_time/len(test_passages)*1000:.1f}ms")
    print(f"  - Triples extracted: {len(result['triples'])}")
    print(f"  - Average per passage: {len(result['triples'])/len(test_passages):.1f}")
    
    # 4. Show sample triples
    print(f"\n✓ Sample Triples Extracted:")
    for i, triple in enumerate(result['triples'][:5], 1):
        head, rel, tail = triple[0], triple[1], triple[2]
        conf = triple[3] if len(triple) > 3 else 1.0
        print(f"  {i}. ({head}, **{rel}**, {tail}) [conf: {conf:.2f}]")
    
    if torch.cuda.is_available():
        # 5. Check GPU memory usage
        print(f"\n✓ GPU Memory Usage:")
        print(f"  - Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"  - Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        print(f"  - Peak: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
    
    print("\n" + "=" * 60)
    print("✅ GPU EXTRACTION TEST PASSED - Ready for Full Rebuild!" if len(result['triples']) > 0 else "❌ TEST FAILED")
    print("=" * 60)

if __name__ == "__main__":
    test_gpu_extraction()
