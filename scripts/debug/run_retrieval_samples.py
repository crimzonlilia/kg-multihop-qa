import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import random
import json

from src.graph.build_graph import load_graph, normalize
from src.data.musique_loader import load_musique
from src.retrieval.pagerank import (
    rank_passages_by_ppr,
    _build_ppr_personalization,
    personalized_pagerank_hipporag,
    extract_entities_from_question,
)
from src.retrieval.passage_ranking import load_passage_data, build_triple_to_passage_map


def run_samples(n=10, split='dev', max_samples=200):
    print(f"Loading KG and passage cache...")
    G = load_graph("data/processed/kg.pkl")

    passages = load_passage_data()
    triple_to_passages, passage_texts, passage_entities, entity_to_passages = build_triple_to_passage_map(passages)
    # Normalize node->passages keys to match graph.normalize used for graph nodes
    node_to_passages = {normalize(k): v for k, v in entity_to_passages.items()}

    samples = load_musique(split, max_samples=max_samples)
    if not samples:
        print("No samples loaded")
        return

    idxs = list(range(len(samples)))
    random.shuffle(idxs)
    idxs = idxs[:n]

    for i in idxs:
        sample = samples[i]
        question = sample.get('question')
        answer = sample.get('answer')
        sup = sample.get('supporting_facts')
        print('\n' + '='*80)
        print(f"Sample idx={i}")
        print(f"Q: {question}")
        print(f"A: {answer}")
        print(f"Supporting facts: {sup}\n")

        # Diagnostics: extract query entities and build personalization vector
        try:
            query_entities = extract_entities_from_question(question, G)
        except Exception:
            query_entities = []

        print(f"Query entities: {query_entities}")

        try:
            personalization, valid_nodes = _build_ppr_personalization(G, query_entities, passage_node_weight=0.05)
        except Exception:
            personalization, valid_nodes = {}, []

        print(f"Personalization nonzero keys (trim): {[k for k,v in personalization.items() if v>0][:10]}")

        # Run internal PPR to inspect top ranked nodes
        try:
            ranked_nodes = personalized_pagerank_hipporag(G, query_entities, top_k=min(100, G.number_of_nodes()))
        except Exception:
            ranked_nodes = []

        print("Top ranked nodes (type, has_text, passage_id if present):")
        for node, score in (ranked_nodes[:20] if ranked_nodes else []):
            data = G.nodes.get(node, {}) if node in G else {}
            ntype = data.get('node_type')
            has_text = bool(data.get('text'))
            pid = data.get('passage_id')
            print(f" - {node} score={score:.6f} type={ntype} has_text={has_text} passage_id={pid}")
            mapped = node_to_passages.get(node)
            print(f"    -> mapped passage ids count: {len(mapped) if mapped else 0}")

        results = rank_passages_by_ppr(
            G=G,
            question=question,
            triple_to_passages=triple_to_passages,
            passage_texts=passage_texts,
            node_to_passages=node_to_passages,
            top_k=10,
            ppr_type='hipporag',
        )

        print("Top passages:")
        for pid, text, score in results:
            print(f"- pid={pid} score={score:.4f} text_preview={text[:200]!r}")


if __name__ == '__main__':
    run_samples()
