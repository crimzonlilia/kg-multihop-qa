#!/usr/bin/env python3
"""
Test extract_triples with 200 sample passages before full rebuild
"""
import torch
import time
import json
from pathlib import Path
from src.extraction.extract_triples import get_extractor, extract_triples_batch

def load_sample_passages(num_samples=200):
    """Load sample passages from MusiQue dataset"""
    passage_file = Path("data/raw/musique/data/musique_full_v1.0_train.jsonl")
    
    if not passage_file.exists():
        print(f"❌ File not found: {passage_file}")
        return []
    
    passages = []
    try:
        with open(passage_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                try:
                    data = json.loads(line)
                    if 'paragraphs' in data:
                        for para in data['paragraphs']:
                            if 'paragraph_text' in para:
                                passages.append(para['paragraph_text'])
                                if len(passages) >= num_samples:
                                    break
                except:
                    continue
    except Exception as e:
        print(f"Error loading passages: {e}")
    
    return passages

def test_with_samples():
    print("=" * 70)
    print("GPU EXTRACTION TEST - 200 Sample Passages")
    print("=" * 70)
    
    # Check GPU
    print(f"\n{'='*70}")
    print("🔧 ENVIRONMENT CHECK")
    print(f"{'='*70}")
    print(f"✓ PyTorch Version: {torch.__version__}")
    print(f"✓ CUDA Available: {torch.cuda.is_available()}")
    print(f"✓ CUDA Version: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"✓ GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"⚠️  WARNING: CUDA not available - will run on CPU (slower)")
    
    # Load passages
    print(f"\n{'='*70}")
    print("📂 LOADING TEST DATA")
    print(f"{'='*70}")
    passages = load_sample_passages(200)
    
    if not passages:
        print("❌ Could not load passages!")
        return False
    
    print(f"✓ Loaded {len(passages)} passages")
    print(f"  Sample passage 1: {passages[0][:100]}...")
    
    # Load model
    print(f"\n{'='*70}")
    print("🤖 LOADING MODEL")
    print(f"{'='*70}")
    print("[Loading GLiNER2...]")
    start = time.time()
    extractor = get_extractor()
    load_time = time.time() - start
    
    device = next(extractor.parameters()).device
    print(f"✓ Model loaded in {load_time:.2f}s")
    print(f"✓ Model Device: {device}")
    print(f"✓ Model Parameters: {sum(p.numel() for p in extractor.parameters()):,}")
    
    # Test extraction
    print(f"\n{'='*70}")
    print("⚡ BATCH EXTRACTION - 200 PASSAGES")
    print(f"{'='*70}")
    print(f"[Processing {len(passages)} passages in batches...]")
    
    start = time.time()
    result = extract_triples_batch(passages, batch_size=50)
    total_time = time.time() - start
    
    # Results
    print(f"\n{'='*70}")
    print("📊 RESULTS")
    print(f"{'='*70}")
    print(f"✓ Total time: {total_time:.2f}s")
    print(f"✓ Per passage: {total_time/len(passages)*1000:.1f}ms")
    print(f"✓ Throughput: {len(passages)/total_time:.1f} passages/sec")
    print(f"✓ Triples extracted: {len(result['triples'])}")
    print(f"✓ Average per passage: {len(result['triples'])/len(passages):.2f}")
    
    if torch.cuda.is_available():
        print(f"\n✓ GPU Memory Usage:")
        print(f"  - Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"  - Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    # Sample output
    print(f"\n{'='*70}")
    print("📝 SAMPLE TRIPLES (first 10)")
    print(f"{'='*70}")
    for i, triple in enumerate(result['triples'][:10], 1):
        head, rel, tail = triple[0], triple[1], triple[2]
        conf = triple[3] if len(triple) > 3 else 1.0
        print(f"{i:2d}. ({head:20s} --[{rel:15s}]--> {tail:20s}) conf:{conf:.3f}")
    
    # Estimate full rebuild time
    print(f"\n{'='*70}")
    print("⏱️  ESTIMATED FULL REBUILD TIME")
    print(f"{'='*70}")
    full_passages = 21100  # Approximate from dataset
    estimated_time = (total_time / len(passages)) * full_passages
    
    print(f"Dataset: ~{full_passages:,} passages")
    print(f"Time per passage: {total_time/len(passages)*1000:.1f}ms")
    print(f"Estimated total: {estimated_time/60:.1f} minutes ({estimated_time/3600:.1f} hours)")
    
    # Final verdict
    print(f"\n{'='*70}")
    success = len(result['triples']) > 0
    if success:
        print("✅ TEST PASSED - Ready for Full Rebuild!")
    else:
        print("❌ TEST FAILED - No triples extracted")
    print(f"{'='*70}\n")
    
    return success

if __name__ == "__main__":
    test_with_samples()
