"""Graph merging and aggregation logic."""

import logging
import networkx as nx

logger = logging.getLogger(__name__)


def merge_graphs(graphs: list) -> nx.DiGraph:
    """
    Merge multiple graphs into one, aggregating nodes and edges.
    
    Args:
        graphs: List of NetworkX DiGraphs
    
    Returns: Merged DiGraph
    """
    if not graphs:
        return nx.DiGraph()
    
    merged = nx.DiGraph()
    
    logger.info(f"Merging {len(graphs)} graphs...")
    
    for idx, G in enumerate(graphs):
        # Merge nodes
        for node, data in G.nodes(data=True):
            if node in merged:
                # Update node type if existing is unknown
                if merged.nodes[node].get("entity_type") == "unknown":
                    merged.nodes[node]["entity_type"] = data.get("entity_type", "unknown")
            else:
                merged.add_node(node, **data)
        
        # Merge edges
        for u, v, data in G.edges(data=True):
            if merged.has_edge(u, v):
                # Merge relation types
                existing = merged[u][v]["relations"]
                existing_types = {r["type"] for r in existing}
                
                for r in data.get("relations", []):
                    if r["type"] not in existing_types:
                        existing.append(r)
                
                # Aggregate weight (average)
                existing_weight = merged[u][v].get("weight", 1.0)
                new_weight = data.get("weight", 1.0)
                merged[u][v]["weight"] = (existing_weight + new_weight) / 2
            else:
                merged.add_edge(u, v,
                              relations=list(data.get("relations", [])),
                              weight=data.get("weight", 1.0))
        
        if (idx + 1) % 50 == 0:
            logger.info(f"  [{idx + 1}/{len(graphs)}] {merged.number_of_nodes()} nodes, {merged.number_of_edges()} edges")
    
    logger.info(f"✓ Merged: {merged.number_of_nodes()} nodes, {merged.number_of_edges()} edges")
    
    return merged
