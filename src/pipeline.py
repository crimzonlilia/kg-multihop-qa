from src.extraction.extract_triples import extract_information
from src.graph.build_graph import build_graph, merge_graphs
from src.retrieval.pagerank import personalized_pagerank, get_subgraph


def run_pipeline(texts, query_entities):
    graphs = []

    for idx, text in enumerate(texts):
        entities, triples = extract_information(text)
        G = build_graph(
            triples,
            entities,
            passage_id=f"passage_{idx}",
            passage_text=text,
        )
        graphs.append(G)

    final_graph = merge_graphs(graphs)

    ranked = personalized_pagerank(final_graph, query_entities)

    subgraph = get_subgraph(final_graph, ranked)

    return final_graph, ranked, subgraph