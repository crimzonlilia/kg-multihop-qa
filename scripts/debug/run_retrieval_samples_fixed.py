import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import random
import json

from src.graph.build_graph import load_graph
from src.graph.graph_utils import get_entity_id
from src.data.musique_loader import load_musique
from src.retrieval.pagerank import rank_passages_by_ppr
from src.retrieval.passage_ranking import load_passage_data, build_triple_to_passage_map


def run_samples(n=10, split='dev', max_samples=200):
    print(f"Loading KG and passage cache...")
    G = load_graph("data/processed/kg.pkl")

    passages = load_passage_data()
    triple_to_passages, passage_texts, passage_entities, entity_to_passages = build_triple_to_passage_map(passages)

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

        results = rank_passages_by_ppr(
            G=G,
            question=question,
            triple_to_passages=triple_to_passages,
            passage_texts=passage_texts,
            node_to_passages=entity_to_passages,
            top_k=10,
            ppr_type='hipporag',
        )

        print("Top passages:")
        if results:
            for pid, text, score in results:
                print(f"- pid={pid} score={score:.4f} text_preview={text[:200]!r}")
        else:
            print("(no passages returned)")


if __name__ == '__main__':
    run_samples()
