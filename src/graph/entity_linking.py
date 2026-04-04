"""Entity linking: smart fuzzy matching to reduce fragmentation."""

import logging
import networkx as nx
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


def link_similar_entities(G: nx.DiGraph,
                         threshold: float = 0.65,
                         prefix_weight: float = 0.25,
                         fuzzy_weight: float = 0.15,
                         max_links: int = None) -> nx.DiGraph:
    """
    Link similar entities using fuzzy matching (optimized).
    
    Improvements over original:
    - Cache pre-computed nodes list
    - Early termination if max_links reached
    - Skip nodes already connected
    - Configurable weights
    - Optional limit on total links
    
    Args:
        G: Input graph
        threshold: Fuzzy match threshold (0-1)
        prefix_weight: Weight for prefix/substring matches
        fuzzy_weight: Weight for fuzzy matches only
        max_links: Max links to add (None = unlimited)
    
    Returns: Graph with similarity links added
    """
    nodes = list(G.nodes())
    n = len(nodes)
    
    logger.info(f"🔗 Entity linking on {n} nodes (threshold={threshold})...")
    
    links_added = 0
    checked_pairs = set()
    
    # Early termination if max_links reached
    def should_continue():
        return max_links is None or links_added < max_links
    
    # Optimize pair checking - use sets for faster lookup
    existing_edges = set()
    for u, v in G.edges():
        existing_edges.add((u, v))
        existing_edges.add((v, u))
    
    # Check all pairs
    for i in range(n):
        if not should_continue():
            break
        
        for j in range(i + 1, n):
            if not should_continue():
                break
            
            n1, n2 = nodes[i], nodes[j]
            pair = tuple(sorted([n1, n2]))
            
            # Skip if already checked
            if pair in checked_pairs:
                continue
            checked_pairs.add(pair)
            
            # Skip if already connected
            if (n1, n2) in existing_edges or (n2, n1) in existing_edges:
                continue
            
            # Compute similarity
            ratio = SequenceMatcher(None, n1, n2).ratio()
            prefix_match = n1 in n2 or n2 in n1
            
            # Determine if should link
            should_link = False
            link_weight = 0.0
            
            if prefix_match and ratio >= 0.5:
                should_link = True
                link_weight = prefix_weight
            elif ratio >= threshold:
                should_link = True
                link_weight = fuzzy_weight
            
            # Add similarity edge
            if should_link:
                G.add_edge(n1, n2,
                          relations=[{"type": "similar_to", "label": "similar entity"}],
                          weight=link_weight)
                G.add_edge(n2, n1,
                          relations=[{"type": "similar_to", "label": "similar entity"}],
                          weight=link_weight)
                
                links_added += 1
                existing_edges.add((n1, n2))
                existing_edges.add((n2, n1))
    
    logger.info(f"✓ Entity linking added {links_added} bridge edges")
    
    return G


def link_entities_fast(G: nx.DiGraph,
                       threshold: float = 0.65,
                       weight: float = 0.2) -> nx.DiGraph:
    """
    Faster entity linking - only check high-degree nodes.
    
    This is a performance-optimized version that:
    - Only links nodes with high degree (likely core entities)
    - Skips isolated or low-degree nodes
    - Much faster on large graphs
    
    Args:
        G: Input graph
        threshold: Fuzzy match threshold
        weight: Weight for all similarity edges
    
    Returns: Graph with selective similarity links
    """
    nodes = list(G.nodes())
    n = len(nodes)
    
    # Filter to high-degree nodes (top 50% by degree)
    degrees = {node: G.degree(node) for node in nodes}
    avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0
    
    high_degree_nodes = [node for node, deg in degrees.items() if deg >= avg_degree]
    
    logger.info(f"🔗 Fast entity linking on {len(high_degree_nodes)}/{n} high-degree nodes...")
    
    links_added = 0
    checked_pairs = set()
    
    # Only check high-degree node pairs
    for i in range(len(high_degree_nodes)):
        for j in range(i + 1, len(high_degree_nodes)):
            n1, n2 = high_degree_nodes[i], high_degree_nodes[j]
            pair = tuple(sorted([n1, n2]))
            
            if pair in checked_pairs:
                continue
            checked_pairs.add(pair)
            
            if G.has_edge(n1, n2) or G.has_edge(n2, n1):
                continue
            
            ratio = SequenceMatcher(None, n1, n2).ratio()
            
            # Link if fuzzy match OR prefix match
            if ratio >= threshold or n1 in n2 or n2 in n1:
                G.add_edge(n1, n2,
                          relations=[{"type": "similar_to", "label": "similar entity"}],
                          weight=weight)
                G.add_edge(n2, n1,
                          relations=[{"type": "similar_to", "label": "similar entity"}],
                          weight=weight)
                
                links_added += 1
    
    logger.info(f"✓ Fast entity linking added {links_added} edges")
    
    return G
