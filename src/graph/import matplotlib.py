import matplotlib.pyplot as plt
import networkx as nx

def visualize_subgraph(subG, max_nodes=80):
    # Nếu subgraph quá to thì lấy k-core / degree cao nhất
    if subG.number_of_nodes() > max_nodes:
        # Lấy top nodes theo degree
        nodes_sorted = sorted(subG.degree, key=lambda x: x[1], reverse=True)
        keep_nodes = [n for n, _ in nodes_sorted[:max_nodes]]
        H = subG.subgraph(keep_nodes).copy()
    else:
        H = subG

    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(H, k=0.3, seed=42)

    nx.draw_networkx_nodes(H, pos, node_size=80, node_color="skyblue")
    nx.draw_networkx_edges(H, pos, alpha=0.3, width=0.5)
    nx.draw_networkx_labels(H, pos, font_size=6)

    plt.axis("off")
    plt.tight_layout()
    plt.show()