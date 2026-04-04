#!/usr/bin/env python3
import torch
import sys

print("=" * 70)
print("PyTorch GPU Check")
print("=" * 70)

print(f"\nPyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("⚠️  CUDA NOT available - will run on CPU (slower)")

print("\nTesting GLiNER2 on GPU...")

from gliner2 import GLiNER2

try:
    model = GLiNER2.from_pretrained("fastino/gliner2-base-v1", map_location="cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ GLiNER2 loaded successfully")
    print(f"  Model parameters: 205M (fp32) or ~100M (fp16)")
except Exception as e:
    print(f"✗ Failed to load: {e}")

print("\n" + "=" * 70)
