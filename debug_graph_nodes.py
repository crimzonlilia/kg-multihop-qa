#!/usr/bin/env python3
"""Debug script to check what nodes are in the graph."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from src.cache_utils import build_graph_output_path
import pickle

graph_path = build_graph_output_path('500')
if Path(graph_path).exists():
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    print(f'Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')
    
    # Check sample nodes
    nodes_sample = list(G.nodes(data=True))[:10]
    print("\nSample nodes from graph:")
    for node_id, data in nodes_sample:
        node_type = data.get("node_type", "unknown")
        raw_text = data.get("raw_text", "N/A")[:40] if "raw_text" in data else "N/A"
        print(f"  ID: {node_id[:40]}...")
        print(f"     Type: {node_type} | Raw: {raw_text}")
    
    # Check if there are nodes named 'Green'
    print("\nChecking for nodes named 'Green':")
    if 'Green' in G:
        print(f"  ✓ Found node 'Green': {G.nodes['Green']}")
    else:
        print("  ✗ No node named 'Green'")
        
    # Check if there's an entity node for Green
    from src.graph.graph_utils import get_entity_id
    green_id = get_entity_id("Green")
    print(f"\nExpected entity ID for 'Green': {green_id}")
    if green_id in G:
        print(f"  ✓ Found entity node: {G.nodes[green_id]}")
    else:
        print(f"  ✗ Entity node not found in graph")
        
else:
    print('No cached graph found')
