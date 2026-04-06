#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Fix Windows encoding issues
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import random
from src.graph.build_graph import load_graph
from src.graph.graph_utils import get_entity_id, normalize
from src.data.musique_loader import load_musique
from src.retrieval.pagerank import rank_passages_by_ppr, extract_entities_from_question
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

    print(f"Graph node count: {G.number_of_nodes()}")
    print(f"Entity-to-passages mapping count: {len(entity_to_passages)}")
    print(f"Sample passage cache size: {len(passage_texts)}\n")

    idxs = list(range(len(samples)))
    random.shuffle(idxs)
    idxs = idxs[:n]

    for i in idxs:
        sample = samples[i]
        question = sample.get('question', '')
        answer = sample.get('answer', '')
        print('\n' + '='*80)
        print(f"Sample idx={i}, Q: {question[:100]}")
        print(f"A: {answer[:100]}\n")

        # Extract query entities
        try:
            query_entities = extract_entities_from_question(question, G)
        except Exception as e:
            query_entities = []
            print(f"  warn: entity extraction failed: {e}")

        # Show entity IDs and mapping info
        if query_entities:
            print(f"Extracted {len(query_entities)} entities:")
            for ent in query_entities[:3]:
                ent_id = get_entity_id(ent)
                has_mapping = ent_id in entity_to_passages
                count = len(entity_to_passages.get(ent_id, set())) if has_mapping else 0
                print(f"  - {ent} => {ent_id[:20]}... has_mapping={has_mapping} passage_count={count}")

        results = rank_passages_by_ppr(
            G=G,
            question=question,
            triple_to_passages=triple_to_passages,
            passage_texts=passage_texts,
            node_to_passages=entity_to_passages,
            top_k=5,
            ppr_type='hipporag',
        )

        print("\nTop passages:")
        if results:
            for j, (pid, text, score) in enumerate(results[:3]):
                preview = text[:100].replace('\n', ' ') if text else "(empty)"
                print(f"{j+1}. score={score:.4f} pid={pid} text={preview}...")
        else:
            print("(no passages returned)")


if __name__ == '__main__':
    run_samples(n=10)
