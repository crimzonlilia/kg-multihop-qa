import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import networkx as nx
from src.graph.build_graph import load_graph, normalize


def personalized_pagerank(G, query_entities, top_k=10, alpha=0.85, entity_type_filter=None):
    """
    Personalized PageRank cho graph retrieval

    Args:
        G: networkx graph
        query_entities: list[str]
        top_k: số node trả về
        alpha: damping factor
        entity_type_filter: list of types to prioritize (e.g., ['person', 'work'])
                        sẽ loại bỏ generic types như 'occupation', 'location'

    Returns:
        List[(node, score)]
    """

    if G.number_of_nodes() == 0:
        return []

    # normalize query
    query_nodes = [normalize(e) for e in query_entities]

    # giữ node tồn tại
    valid_nodes = [n for n in query_nodes if n in G]

    if not valid_nodes:
        print(f"Không tìm thấy entity trong graph: {query_entities}")
        return []

    # personalization vector
    personalization = {node: 0.0 for node in G.nodes()}
    for node in valid_nodes:
        personalization[node] = 1.0 / len(valid_nodes)

    # chạy PageRank
    pr_scores = nx.pagerank(
        G,
        alpha=alpha,
        personalization=personalization,
        weight="weight"
    )

    # sort
    ranked = sorted(pr_scores.items(), key=lambda x: x[1], reverse=True)

    # Filter base trên entity_type nếu có
    if entity_type_filter:
        filtered = [
            (n, s) for n, s in ranked 
            if G.nodes[n].get("entity_type", "unknown") in entity_type_filter
        ]
        return filtered[:top_k]
    
    return ranked[:top_k]


def get_subgraph(G, ranked_nodes, hops=2):
    """
    Lấy subgraph quanh top nodes (để phục vụ multi-hop reasoning)
    """

    nodes = set()

    for node, _ in ranked_nodes:
        nodes.add(node)

        # BFS mở rộng multi-hop
        frontier = {node}
        for _ in range(hops):
            next_frontier = set()
            for n in frontier:
                next_frontier.update(G.successors(n))
                next_frontier.update(G.predecessors(n))
            nodes.update(next_frontier)
            frontier = next_frontier

    return G.subgraph(nodes)


if __name__ == "__main__":
    import json
    from src.data.musique_loader import load_musique
    from src.graph.build_graph import load_graph, normalize

    G = load_graph("data/processed/kg.pkl")

    # Load 100 samples từ musique dev set
    samples = load_musique("dev", max_samples=100)
    
    if not samples:
        print("Không tải được dataset!")
        sys.exit(1)
    
    # Chọn sample cần test (hoặc loop qua nhiều)
    sample_idx = 0 if len(sys.argv) <= 1 else int(sys.argv[1])
    
    if sample_idx >= len(samples):
        print(f"Sample index {sample_idx} vượt quá {len(samples)} samples")
        sys.exit(1)
    
    sample = samples[sample_idx]
    print(f"Question: {sample['question']}")
    print(f"Answer: {sample['answer']}")
    print(f"Supporting facts: {sample['supporting_facts']}\n")
    
    # Extract entities từ question và supporting facts
    # (có thể dùng NER hoặc tên từ supporting_facts)
    query_entities = [title for _, title in sample['supporting_facts']]
    
    if not query_entities:
        print("Không tìm thấy entities trong supporting facts!")
        sys.exit(1)
    
    print(f"Query entities: {query_entities}")
    ranked = personalized_pagerank(
        G, query_entities, top_k=30,
        entity_type_filter=['person', 'work', 'organization']  # Loại occupation, location
    )
    
    if not ranked:
        print("Không tìm thấy entity trong graph!")
    else:
        print("\nTop 30 nodes:")
        for i, (node, score) in enumerate(ranked, 1):
            print(f"  {i}. {node}: {score:.4f}")

        # Lấy top 10 cho subgraph
        top_10_ranked = ranked[:10]
        subG = get_subgraph(G, top_10_ranked, hops=4)
        print(f"\nSubgraph có {subG.number_of_nodes()} nodes, {subG.number_of_edges()} edges")