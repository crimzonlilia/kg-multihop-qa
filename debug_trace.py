"""
Trace through a single query to diagnose why PPR retrieval fails.
"""
import sys, json
sys.path.insert(0, '.')

from collections import defaultdict
from src.data.musique_loader import load_musique
from src.extraction.extract_triples import extract_triples_batch, DEFAULT_RELATION_SCHEMA
from src.graph.build_graph import build_graph, merge_graphs, normalize
from src.retrieval.pagerank import extract_entities_from_question, personalized_pagerank_hipporag
import spacy, networkx as nx

print("Loading data (5 QA pairs)...")
dev_data = load_musique("dev", max_samples=5)

passages_by_id = {}
query_meta = {}
for qa in dev_data:
    qid = qa.get("id", f"q_{len(query_meta)}")
    gold_ids = []
    for i, para in enumerate(qa.get("passages", [])):
        p_text = para.get("text", "")
        p_id = f"{qid}_p{i}"
        if p_text:
            passages_by_id[p_id] = p_text
        if para.get("is_supporting", False):
            gold_ids.append(p_id)
    query_meta[qid] = {"question": qa["question"], "answer": qa["answer"].strip().lower(), "gold_passage_ids": gold_ids}

print(f"Passages: {len(passages_by_id)}, Queries: {len(query_meta)}")

print("\nExtracting triples (use_dynamic=False for speed)...")
passage_ids = list(passages_by_id.keys())
passage_texts = list(passages_by_id.values())
results_batch = extract_triples_batch(
    passage_texts, relation_schema=DEFAULT_RELATION_SCHEMA,
    use_dynamic=False, skip_cache=True, deduplicate=False,
)
triples_by_passage = {passage_ids[i]: results_batch[i].get("triples", []) for i in range(len(results_batch))}
entities_by_passage = {passage_ids[i]: results_batch[i].get("entities", []) for i in range(len(results_batch))}

total_t = sum(len(t) for t in triples_by_passage.values())
print(f"Total triples: {total_t}, avg: {total_t/len(triples_by_passage):.2f}/passage")

print("\nBuilding graph...")
passage_graphs = {}
for p_id, triples in triples_by_passage.items():
    try:
        passage_graphs[p_id] = build_graph(triples, add_cooccurrence_edges=False)
    except Exception:
        passage_graphs[p_id] = nx.DiGraph()

final_graph = merge_graphs(list(passage_graphs.values()))
print(f"Graph: {final_graph.number_of_nodes()} nodes, {final_graph.number_of_edges()} edges")

# Build triple_to_passages
triple_to_passages = defaultdict(list)
for p_id, triples in triples_by_passage.items():
    for t in triples:
        key = (normalize(t.get("subject", "")), t.get("relation", ""), normalize(t.get("object", "")))
        if p_id not in triple_to_passages[key]:
            triple_to_passages[key].append(p_id)

# Build node_to_passages index
node_to_passages = defaultdict(set)
for (s, r, o), pids in triple_to_passages.items():
    node_to_passages[s].update(pids)
    node_to_passages[o].update(pids)

nlp = spacy.load("en_core_web_sm")

# Trace Q0
qid0 = list(query_meta.keys())[0]
meta0 = query_meta[qid0]
q = meta0["question"]
ans = meta0["answer"]
gold_ids = meta0["gold_passage_ids"]

print(f"\n=== Q0: {q}")
print(f"    Answer: {ans}")
print(f"    Gold passages: {gold_ids}")

# Check gold passage contents
for gid in gold_ids:
    txt = passages_by_id.get(gid, "NOT FOUND")
    print(f"\n    Gold {gid}: {txt[:120]}")
    print(f"    Triples: {triples_by_passage.get(gid, [])}")

# Entity extraction
query_entities = extract_entities_from_question(q, final_graph, nlp)
print(f"\n    Extracted entities: {query_entities}")

if not query_entities:
    print("    *** NO ENTITIES EXTRACTED - this is the failure point ***")
else:
    # Run PPR
    ranked_nodes = personalized_pagerank_hipporag(final_graph, query_entities, top_k=20, damping=0.5)
    print(f"\n    Top-10 PPR nodes: {[(n, round(s,4)) for n,s in ranked_nodes[:10]]}")

    # Map nodes to passages
    passage_scores = defaultdict(float)
    for node, score in ranked_nodes:
        for pid in node_to_passages.get(node, set()):
            passage_scores[pid] = max(passage_scores[pid], score)

    ranked_passages = sorted(passage_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\n    Top-10 passages (indexed):")
    for rank, (pid, score) in enumerate(ranked_passages, 1):
        is_gold = pid in gold_ids
        txt_preview = passages_by_id.get(pid, "")[:60]
        print(f"      [{rank}] {'***GOLD***' if is_gold else '          '} {pid} ({score:.4f}) {txt_preview}")
    
    # Check answer in passages
    for rank, (pid, score) in enumerate(ranked_passages, 1):
        if ans in passages_by_id.get(pid, "").lower():
            print(f"\n    *** Answer '{ans}' FOUND at rank {rank} ***")
            break
    else:
        print(f"\n    *** Answer '{ans}' NOT FOUND in top-10 ***")
        # Check if answer exists in any passage
        for pid, txt in passages_by_id.items():
            if ans in txt.lower():
                print(f"    (Answer exists in {pid}: {txt[:80]})")
                break
        else:
            print(f"    (Answer NOT IN ANY passage at all!)")
