import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests

from src.extraction.extract_triples import extract_information
from src.graph.build_graph import load_graph
from src.retrieval.pagerank import personalized_pagerank, get_subgraph, get_top_passages_from_ranked_nodes


def triples_to_text(subG):
    sentences = []
    for u, v, data in subG.edges(data=True):
        if subG.nodes[u].get("node_type") == "passage" or subG.nodes[v].get("node_type") == "passage":
            continue
        for r in data.get("relations", []):
            if r.get("type") in {"contains", "mentioned_in"}:
                continue
            label = r.get("label", r.get("type", ""))
            sentences.append(f"{u} {label} {v}")
    return ". ".join(sentences)


def passages_to_text(passages):
    return "\n\n".join(
        f"[{pid}] {text}" for pid, text, _ in passages if text
    )


def call_llm(prompt):
    url = "http://localhost:11434/api/generate"

    try:
        response = requests.post(
        url,
        data=json.dumps({
            "model": "gemma3:4b",
            "prompt": prompt,
            "stream": False,
            "temperature": 0.7,
            "top_p": 0.9
        }),
        headers={"Content-Type": "application/json"},
        timeout=120
    )
        response.raise_for_status()
        result = response.json()
        
        print(f"\nDEBUG - API Response: {result}")
        
        if "response" in result:
            return result["response"]
        else:
            return f"Error: Unexpected response format. Got: {list(result.keys())}"
            
    except requests.exceptions.Timeout:
        return "LLM Error: Request timeout (model takes too long)"
    except requests.exceptions.RequestException as e:
        return f"LLM Error: {str(e)}"


def answer_question(question):
    G = load_graph("data/processed/kg.pkl")

    # extract entity
    entities, _ = extract_information(question)
    query_entities = [e["text"] for e in entities]

    if not query_entities:
        return "Không tìm thấy entity."

    # retrieval
    ranked = personalized_pagerank(G, query_entities, top_k=30)
    subG = get_subgraph(G, ranked, hops=2)
    top_passages = get_top_passages_from_ranked_nodes(G, ranked, top_k=5)

    # context
    graph_context = triples_to_text(subG)
    passage_context = passages_to_text(top_passages)
    context = (
        f"Graph facts:\n{graph_context or '(none)'}\n\n"
        f"Relevant passages:\n{passage_context or '(none)'}"
    )

    print("\n=== CONTEXT ===")
    print(context)

    # prompt
    prompt = (
    "Answer the question using ONLY the provided graph facts and passages.\n"
    "If the answer is not in the context, say 'I don't know'.\n\n"
    f"{context}\n\n"
    f"Question:\n{question}\n\n"
    "Answer:"
)

    answer = call_llm(prompt)

    return answer


if __name__ == "__main__":
    question = "Where was Elon Musk born?"

    result = answer_question(question)

    print("\n=== ANSWER ===")
    print(result)