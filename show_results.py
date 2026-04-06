#!/usr/bin/env python3
"""Display results summary."""

import json

with open('full_pipeline_results_20260407_000833.json') as f:
    results = json.load(f)
    
# Print summary
print('=== EVALUATION SUMMARY ===')
print(f"AiC@1: {results['metrics']['aic@1']:.1%}")
print(f"AiC@5: {results['metrics']['aic@5']:.1%}")
print(f"AiC@10: {results['metrics']['aic@10']:.1%}")
print(f"\nF1@10 (partial): {results['metrics']['f1_partial@10']:.1%}")
print(f"F1@10 (strict): {results['metrics']['f1_strict@10']:.1%}")
print(f"\nGraph stats:")
print(f"  Nodes: {results['graph']['nodes']}")
print(f"  Edges: {results['graph']['edges']}")
