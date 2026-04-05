#!/usr/bin/env python3
"""Inspect a saved knowledge graph pickle and print useful stats."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.cache_utils import resolve_graph_output_path
from src.console_utils import configure_console_output
from src.graph.build_graph import load_graph

configure_console_output()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a saved KG pickle.")
    parser.add_argument("--graph", default=None, help="Named graph alias like 3, 500, or full.")
    parser.add_argument("--path", default=None, help="Explicit path to a graph .pkl file.")
    parser.add_argument("--top", type=int, default=15, help="Number of top items to show.")
    parser.add_argument("--sample-edges", type=int, default=15, help="Number of sample edges to print.")
    parser.add_argument("--json", action="store_true", help="Print stats as JSON.")
    return parser.parse_args()


def resolve_path(args: argparse.Namespace) -> Path:
    return resolve_graph_output_path(
        graph_name=args.graph,
        graph_path=args.path,
        must_exist=True,
    )


def top_items(items, top_n: int) -> list[tuple]:
    return sorted(items, key=lambda item: item[1], reverse=True)[:top_n]


def edge_relation_labels(attrs: dict) -> list[str]:
    """Extract one or more relation labels from a graph edge attribute dict."""
    if not isinstance(attrs, dict):
        return ["unknown"]

    relations = attrs.get("relations")
    if isinstance(relations, (list, tuple, set)):
        labels = [str(rel) for rel in relations if rel]
        if labels:
            return labels

    relation = attrs.get("relation") or attrs.get("type") or "unknown"
    return [str(relation)]


def collect_stats(G: nx.Graph, graph_path: Path, top_n: int, sample_edges: int) -> dict:
    relation_counts = Counter()
    for _, _, data in G.edges(data=True):
        relation_counts.update(edge_relation_labels(data))

    if G.is_directed():
        component_sets = list(nx.weakly_connected_components(G))
        component_count = nx.number_weakly_connected_components(G)
    else:
        component_sets = list(nx.connected_components(G))
        component_count = nx.number_connected_components(G)

    largest_component = max((len(component) for component in component_sets), default=0)
    isolated_nodes = list(nx.isolates(G))

    node_attr_keys = sorted({key for _, attrs in G.nodes(data=True) for key in attrs.keys()})
    edge_attr_keys = sorted({key for _, _, attrs in G.edges(data=True) for key in attrs.keys()})

    stats = {
        "graph_path": str(graph_path),
        "directed": G.is_directed(),
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G) if G.number_of_nodes() > 1 else 0.0,
        "avg_degree": (sum(dict(G.degree()).values()) / G.number_of_nodes()) if G.number_of_nodes() else 0.0,
        "components": component_count,
        "largest_component_size": largest_component,
        "distinct_relations": len(relation_counts),
        "isolated_nodes": len(isolated_nodes),
        "self_loops": nx.number_of_selfloops(G),
        "top_relations": relation_counts.most_common(top_n),
        "top_degree": top_items(G.degree, top_n),
        "top_in_degree": top_items(G.in_degree, top_n) if G.is_directed() else [],
        "top_out_degree": top_items(G.out_degree, top_n) if G.is_directed() else [],
        "node_attr_keys": node_attr_keys,
        "edge_attr_keys": edge_attr_keys,
        "sample_edges": [
            {
                "source": source,
                "target": target,
                "attrs": attrs,
            }
            for source, target, attrs in list(G.edges(data=True))[:sample_edges]
        ],
    }
    return stats


def print_pretty(stats: dict) -> None:
    print("\n" + "=" * 80)
    print("KNOWLEDGE GRAPH INSPECTION")
    print("=" * 80)
    print(f"Graph file            : {stats['graph_path']}")
    print(f"Directed              : {stats['directed']}")
    print(f"Nodes                 : {stats['nodes']}")
    print(f"Edges                 : {stats['edges']}")
    print(f"Density               : {stats['density']:.6f}")
    print(f"Average degree        : {stats['avg_degree']:.2f}")
    print(f"Connected components  : {stats['components']}")
    print(f"Largest component     : {stats['largest_component_size']}")
    print(f"Distinct relations    : {stats['distinct_relations']}")
    print(f"Isolated nodes        : {stats['isolated_nodes']}")
    print(f"Self loops            : {stats['self_loops']}")

    print("\nTop relations:")
    for relation, count in stats["top_relations"]:
        print(f"  - {relation:<24} {count}")

    print("\nTop degree nodes:")
    for node, degree in stats["top_degree"]:
        print(f"  - {node!r:<40} {degree}")

    if stats["top_in_degree"]:
        print("\nTop in-degree nodes:")
        for node, degree in stats["top_in_degree"]:
            print(f"  - {node!r:<40} {degree}")

    if stats["top_out_degree"]:
        print("\nTop out-degree nodes:")
        for node, degree in stats["top_out_degree"]:
            print(f"  - {node!r:<40} {degree}")

    print(f"\nNode attribute keys   : {stats['node_attr_keys']}")
    print(f"Edge attribute keys   : {stats['edge_attr_keys']}")

    print("\nSample edges:")
    for edge in stats["sample_edges"]:
        relation = ", ".join(edge_relation_labels(edge["attrs"]))
        weight = edge["attrs"].get("weight")
        weight_str = f" (weight={weight})" if weight is not None else ""
        print(f"  - {edge['source']!r} --[{relation}]--> {edge['target']!r}{weight_str}")

    print("=" * 80)


def main() -> None:
    args = parse_args()
    graph_path = resolve_path(args)
    G = load_graph(str(graph_path))
    stats = collect_stats(G, graph_path, top_n=args.top, sample_edges=args.sample_edges)

    if args.json:
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    else:
        print_pretty(stats)


if __name__ == "__main__":
    main()
