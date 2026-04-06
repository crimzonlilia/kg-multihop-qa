import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import networkx as nx
import numpy as np
import spacy
from collections import defaultdict
from src.graph.build_graph import load_graph, normalize
from src.graph.graph_utils import get_entity_id

# ── Embedding-based entity linker (module-level cache) ────────────────────────
EMBED_MODEL_OPTIONS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-base": "BAAI/bge-base-en-v1.5",
    "e5-small": "intfloat/e5-small-v2",
    "e5-base": "intfloat/e5-base-v2",
}

_embed_model = None
_embed_model_name = EMBED_MODEL_OPTIONS.get(
    os.getenv("KG_EMBED_MODEL", "minilm"),
    os.getenv("KG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
)
_graph_node_embeddings = None   # (node_list, embedding_matrix)


def set_embed_model(model_name_or_alias: str):
    """Switch retrieval embedding model by alias or full HF model name."""
    global _embed_model, _embed_model_name, _graph_node_embeddings
    resolved = EMBED_MODEL_OPTIONS.get(model_name_or_alias, model_name_or_alias)
    if resolved != _embed_model_name:
        _embed_model_name = resolved
        _embed_model = None
        _graph_node_embeddings = None
    return _embed_model_name


def get_embed_model_name() -> str:
    """Return the active retrieval embedding model name."""
    return _embed_model_name


def _prepare_texts_for_embedding(texts, text_type="passage"):
    """Apply model-specific prompt formatting for retrieval embeddings."""
    model_name = _embed_model_name.lower()
    if "e5" in model_name:
        prefix = "query: " if text_type == "query" else "passage: "
        return [prefix + text for text in texts]
    return list(texts)


def _encode_texts(texts, text_type="passage", batch_size=256):
    """Encode texts with the currently selected embedding model."""
    model = _get_embed_model()
    prepared = _prepare_texts_for_embedding(texts, text_type=text_type)
    return model.encode(
        prepared,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    ).astype(np.float32)


def _get_embed_model():
    """Lazy load SentenceTransformer (cached across calls)."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(_embed_model_name)
    return _embed_model

def build_fact_embeddings(triple_to_passages):
    """
    Encode all extracted triples as flat strings for query-to-fact similarity.

    HippoRAG insight: at retrieval time, comparing the query against fact strings
    ("subj pred obj") is more robust than comparing against bare entity names,
    because multi-hop questions often describe a relation rather than naming an
    entity directly.

    Args:
        triple_to_passages: dict mapping (s, r, o) tuple → list[passage_id]

    Returns:
        (fact_list, fact_matrix)
        • fact_list: list of (s, r, o) tuples, same order as rows in matrix
        • fact_matrix: np.ndarray shape (N, d) - L2-normalised embeddings
    """
    facts = list(triple_to_passages.keys())
    if not facts:
        return [], np.zeros((0, 384), dtype=np.float32)

    strings = [
        f"{s} {r.replace('_', ' ')} {o}"
        for s, r, o in facts
    ]
    matrix = _encode_texts(strings, text_type="fact", batch_size=256)
    return facts, matrix


def build_passage_embeddings(passage_texts):
    """
    Encode passage texts for dense-retrieval fallback.

    Used when PPR returns no results (graph had no entity matches).
    Returns passages ordered by cosine similarity to the query.

    Args:
        passage_texts: dict mapping passage_id → passage text string

    Returns:
        (pid_list, passage_matrix)
        • pid_list: list of passage_ids, same order as rows in matrix
        • passage_matrix: np.ndarray shape (N, d) – L2-normalised embeddings
    """
    pid_list = list(passage_texts.keys())
    if not pid_list:
        return [], np.zeros((0, 384), dtype=np.float32)

    matrix = _encode_texts(
        [passage_texts[pid] for pid in pid_list],
        text_type="passage",
        batch_size=256,
    )
    return pid_list, matrix


def add_synonymy_edges(G, threshold=0.85, batch_size=256):
    """
    Add soft KNN edges between entity nodes that are semantically similar
    but have different surface forms (e.g. "giraudy" ↔ "miquette giraudy").

    Edges are added with weight = cosine_similarity and relation "synonymy".
    This lets PPR spread probability across aliases found in different passages.

    Args:
        G: NetworkX DiGraph (modified in-place)
        threshold: cosine similarity above which to add a synonymy edge (default 0.85)
        batch_size: SentenceTransformer encoding batch size

    Returns:
        Number of synonymy edges added
    """
    node_list = [
        node for node, data in G.nodes(data=True)
        if data.get("node_type") != "passage"
    ]
    if len(node_list) < 2:
        return 0

    matrix = _encode_texts(
        node_list,
        text_type="node",
        batch_size=batch_size,
    )

    added = 0
    n = len(node_list)
    # Process in row chunks to avoid O(n²) memory at once
    chunk = 512
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sims = matrix[start:end] @ matrix.T  # (chunk, n)
        for local_i, i in enumerate(range(start, end)):
            for j in range(i + 1, n):
                if sims[local_i, j] >= threshold:
                    u, v = node_list[i], node_list[j]
                    if not G.has_edge(u, v):
                        G.add_edge(u, v, weight=float(sims[local_i, j]),
                                   relations=[{"type": "synonymy"}])
                        added += 1
                    if not G.has_edge(v, u):
                        G.add_edge(v, u, weight=float(sims[local_i, j]),
                                   relations=[{"type": "synonymy"}])
                        added += 1
    return added


def build_node_embeddings(G):
    """
    Build (and cache) embedding matrix for entity-like graph nodes.
    Passage nodes are skipped because their ids are structural anchors rather
    than semantic entities we want to match from the question.
    """
    node_list = [
        node for node, data in G.nodes(data=True)
        if data.get("node_type") != "passage"
    ]
    if not node_list:
        return node_list, np.zeros((0, 384), dtype=np.float32)
    matrix = _encode_texts(node_list, text_type="node", batch_size=256)
    return node_list, matrix


def _build_ppr_personalization(G, query_entities, passage_node_weight=0.05):
    """
    Build a PageRank personalization vector with strong entity seeds and a
    light prior over passage nodes so they can surface in the final ranking.
    
    CRITICAL FIX: query_entities are now entity IDs (hash format), not normalized strings.
    """
    if G.number_of_nodes() == 0:
        return {}, []

    # query_entities are already entity IDs from extract_entities_from_question
    # Check they exist in graph (should be direct lookups)
    valid_nodes = [
        node for node in query_entities
        if node in G and G.nodes[node].get("node_type") != "passage"
    ]
    if not valid_nodes:
        return {}, []

    personalization = {node: 0.0 for node in G.nodes()}
    for node in valid_nodes:
        personalization[node] += 1.0

    passage_nodes = [
        node for node, data in G.nodes(data=True)
        if data.get("node_type") == "passage"
    ]
    if passage_node_weight > 0 and passage_nodes:
        per_passage_weight = passage_node_weight / len(passage_nodes)
        for node in passage_nodes:
            personalization[node] += per_passage_weight

    total = sum(personalization.values())
    if total <= 0:
        return {}, []

    personalization = {
        node: (value / total if value >= 0 and not np.isnan(value) else 0.0)
        for node, value in personalization.items()
    }
    return personalization, valid_nodes


def get_top_passages_from_ranked_nodes(G, ranked_nodes, top_k=10, passage_texts=None):
    """
    Pull passage nodes directly from a ranked node list.
    Returns [(passage_id, passage_text, score), ...].
    """
    results = []
    seen = set()

    for node_id, score in ranked_nodes:
        if node_id not in G:
            continue
        data = G.nodes[node_id]
        if data.get("node_type") != "passage":
            continue

        passage_id = data.get("passage_id", str(node_id).replace("passage::", "", 1))
        passage_text = data.get("text") or (passage_texts or {}).get(passage_id, "")
        if not passage_text or passage_id in seen:
            continue

        seen.add(passage_id)
        results.append((passage_id, passage_text, float(score)))
        if len(results) >= top_k:
            break

    return results


_RERANK_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "by", "with",
    "is", "was", "were", "are", "be", "who", "what", "where", "when", "which",
    "whom", "whose", "that", "this", "these", "those", "from", "into", "as",
    "did", "does", "do", "has", "have", "had", "than", "then", "their", "his",
    "her", "its", "other", "another", "also", "about", "after", "before"
}


def _tokenize_query_terms(text):
    """Tokenize text for cheap lexical overlap scoring."""
    text = normalize(text)
    return {
        tok for tok in text.split()
        if len(tok) >= 3 and tok not in _RERANK_STOPWORDS and not tok.isdigit()
    }


def _normalize_score_dict(score_dict):
    """Min-max normalize a dict of scalar scores into [0, 1]."""
    if not score_dict:
        return {}
    values = list(score_dict.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-8:
        baseline = 1.0 if hi > 0 else 0.0
        return {k: baseline for k in score_dict}
    return {k: (v - lo) / (hi - lo) for k, v in score_dict.items()}


def _rerank_passage_candidates(
    question,
    query_entities,
    candidate_scores,
    passage_texts,
    node_hits_by_passage=None,
    fact_passage_scores=None,
    top_k=10,
):
    """
    Lightweight reranker / score fusion.

    Keeps runtime cheap by only rescoring the top candidate passages with:
      - graph/PPR score
      - fact similarity score
      - lexical overlap with the question
      - bridge / multi-hop support bonus from multiple matched nodes
    """
    query_terms = _tokenize_query_terms(question)
    seed_terms = {
        normalize(ent) for ent in query_entities
        if len(normalize(ent)) >= 3
    }
    base_norm = _normalize_score_dict(candidate_scores)
    fact_norm = _normalize_score_dict(fact_passage_scores or {})

    reranked = []
    for pid, base_score in candidate_scores.items():
        passage_text = passage_texts.get(pid, "")
        if not passage_text:
            continue

        passage_norm = normalize(passage_text)
        passage_terms = _tokenize_query_terms(passage_text)
        lexical_overlap = len(query_terms & passage_terms) / max(len(query_terms), 1)

        matched_seed_count = sum(1 for ent in seed_terms if ent and ent in passage_norm)
        seed_coverage = matched_seed_count / max(1, min(len(seed_terms), 3))

        node_hits = (node_hits_by_passage or {}).get(pid, set())
        node_diversity = min(1.0, len(node_hits) / 4.0)

        # Multi-hop flavored bridge bonus: reward passages backed by multiple
        # graph nodes and/or multiple seed mentions instead of a single local hit.
        bridge_bonus = min(1.0, 0.55 * seed_coverage + 0.45 * node_diversity)

        combined_score = (
            0.55 * base_norm.get(pid, 0.0)
            + 0.20 * fact_norm.get(pid, 0.0)
            + 0.15 * lexical_overlap
            + 0.10 * bridge_bonus
        )
        if lexical_overlap > 0 and bridge_bonus > 0:
            combined_score += 0.03

        reranked.append((pid, passage_text, float(combined_score), float(base_score)))

    reranked.sort(key=lambda x: (x[2], x[3]), reverse=True)
    return [(pid, text, score) for pid, text, score, _ in reranked[:max(top_k * 3, top_k)]]


def personalized_pagerank_hipporag(G, query_entities, top_k=10, damping=0.5, max_iter=20, tol=0.01):
    """
    HippoRAG style Personalized PageRank.

    Uses strong entity seeds plus a small prior on passage nodes so the final
    ranking can surface both bridge entities and supporting passages.
    """
    if G.number_of_nodes() == 0:
        return []

    reset_prob, valid_nodes = _build_ppr_personalization(
        G, query_entities, passage_node_weight=0.05
    )
    if not valid_nodes:
        return []

    pr_scores = nx.pagerank(
        G,
        alpha=damping,
        personalization=reset_prob,
        weight="weight",
        max_iter=max_iter,
        tol=tol
    )

    ranked = sorted(pr_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def personalized_pagerank(G, query_entities, top_k=10, alpha=0.85, entity_type_filter=None, max_iter=20, tol=0.01):
    """
    Standard Personalized PageRank (keep for backward compatibility).
    """

    if G.number_of_nodes() == 0:
        return []

    personalization, valid_nodes = _build_ppr_personalization(
        G, query_entities, passage_node_weight=0.05
    )
    if not valid_nodes:
        return []

    pr_scores = nx.pagerank(
        G,
        alpha=alpha,
        personalization=personalization,
        weight="weight",
        max_iter=max_iter,
        tol=tol
    )

    ranked = sorted(pr_scores.items(), key=lambda x: x[1], reverse=True)

    if entity_type_filter:
        filtered = [
            (n, s) for n, s in ranked
            if G.nodes[n].get("entity_type", "unknown") in entity_type_filter
        ]
        return filtered[:top_k]

    return ranked[:top_k]


def personalized_pagerank_fast(G, query_entities, top_k=10, alpha=0.85, neighborhood_hops=3):
    """
    Run PPR on a localized K-hop neighborhood around the query entities.
    Passage nodes receive a light prior and can still surface in the ranking.
    """
    if G.number_of_nodes() == 0:
        return []

    query_nodes = [normalize(e) for e in query_entities]
    valid_nodes = [
        n for n in query_nodes
        if n in G and G.nodes[n].get("node_type") != "passage"
    ]

    if not valid_nodes:
        return []

    nodes = set(valid_nodes)
    frontier = set(valid_nodes)

    for _ in range(neighborhood_hops):
        next_frontier = set()
        for n in frontier:
            next_frontier.update(G.successors(n))
            next_frontier.update(G.predecessors(n))
        nodes.update(next_frontier)
        frontier = next_frontier

    subgraph = G.subgraph(nodes)

    personalization = {node: 0.0 for node in subgraph.nodes()}
    base_score = 1.0 / len(valid_nodes)
    for node in valid_nodes:
        if node in personalization:
            personalization[node] = base_score

    neighbor_boost = 0.3 * base_score
    for node in valid_nodes:
        if node in G:
            for neighbor in set(G.successors(node)) | set(G.predecessors(node)):
                if neighbor in personalization:
                    personalization[neighbor] += neighbor_boost

    passage_nodes = [
        node for node, data in subgraph.nodes(data=True)
        if data.get("node_type") == "passage"
    ]
    if passage_nodes:
        passage_boost = 0.15 * base_score / len(passage_nodes)
        for node in passage_nodes:
            personalization[node] += passage_boost

    total = sum(personalization.values())
    if total > 0:
        personalization = {k: v / total for k, v in personalization.items()}

    pr_scores = nx.pagerank(
        subgraph,
        alpha=alpha,
        personalization=personalization,
        weight="weight",
        max_iter=100,
        tol=0.05
    )

    ranked = sorted(pr_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def get_subgraph(G, ranked_nodes, hops=2, include_passages=False):
    """
    Lấy subgraph quanh top nodes để phục vụ multi-hop reasoning.

    By default this keeps the triple/entity neighborhood and skips structural
    passage anchor nodes, since passages are usually concatenated separately
    into the final prompt.
    """

    nodes = set()

    for node, _ in ranked_nodes:
        if node not in G:
            continue

        if include_passages or G.nodes[node].get("node_type") != "passage":
            seed_nodes = {node}
        else:
            seed_nodes = {
                nbr for nbr in (set(G.successors(node)) | set(G.predecessors(node)))
                if G.nodes[nbr].get("node_type") != "passage"
            }

        nodes.update(seed_nodes)
        frontier = set(seed_nodes)
        for _ in range(hops):
            next_frontier = set()
            for n in frontier:
                next_frontier.update(G.successors(n))
                next_frontier.update(G.predecessors(n))
            if not include_passages:
                next_frontier = {
                    nbr for nbr in next_frontier
                    if G.nodes[nbr].get("node_type") != "passage"
                }
            nodes.update(next_frontier)
            frontier = next_frontier

    return G.subgraph(nodes)


def extract_entities_from_question(question, G, nlp=None, node_embeddings=None,
                                   top_k_embed=5, embed_threshold=0.60):
    """
    Extract entities from question and match to graph nodes (hash-based entity IDs).

    Strategy (HippoRAG-inspired, with entity ID fixes):
      Pass 1 – spaCy NER + entity ID matching (hash-based lookup)
      Pass 2 – embedding similarity: embed the question, score against
               pre-built node embedding matrix, keep nodes above threshold.

    CRITICAL FIX: Returns entity IDs (entity-hash format) not normalized strings,
    so they can be directly used in PPR seed set and lookups.

    Args:
        question: raw question string
        G: NetworkX graph (nodes are entity IDs like "entity-abc123")
        nlp: pre-loaded spaCy model (loaded lazily if None)
        node_embeddings: (node_list, matrix) from build_node_embeddings(G).
        top_k_embed: max nodes to add from embedding search
        embed_threshold: cosine similarity threshold (0-1) for embedding match

    Returns:
        list of matched entity node IDs (hash format), up to 10 total
    """
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            nlp = spacy.load("en_core_web_sm")

    doc = nlp(question)
    
    # Get all entity nodes from graph (exclude passage nodes)
    graph_entity_ids = {
        node for node, data in G.nodes(data=True)
        if data.get("node_type") != "passage"
    }
    
    # Build reverse lookup: normalized_text -> entity_id
    # This maps normalized entity text → entity ID (handles multiple surface forms → same ID)
    norm_to_entity_id = {}
    for ent_id in graph_entity_ids:
        # Entity IDs are in format "entity-abcdef123..."
        # Get the actual entity text from node data
        raw_text = G.nodes[ent_id].get("raw_text", "")
        if raw_text:
            norm = normalize(raw_text)
            if norm and not norm_to_entity_id.get(norm):
                norm_to_entity_id[norm] = ent_id

    matched_entity_ids = []

    # ── Pass 1a: spaCy NER ────────────────────────────────────────────────────
    for ent in doc.ents:
        ent_text = ent.text.strip()
        ent_id = get_entity_id(ent_text)
        
        # Direct lookup: if this entity ID exists in graph
        if ent_id in graph_entity_ids and ent_id not in matched_entity_ids:
            matched_entity_ids.append(ent_id)
        # Fallback: check normalized form
        elif ent_id not in matched_entity_ids:
            ent_norm = normalize(ent_text)
            if len(ent_norm) >= 3 and ent_norm in norm_to_entity_id:
                found_id = norm_to_entity_id[ent_norm]
                if found_id not in matched_entity_ids:
                    matched_entity_ids.append(found_id)

    # ── Pass 1b: question-word substring scan (with hash matching) ──────────
    question_norm = normalize(question)
    for norm_text, ent_id in norm_to_entity_id.items():
        if len(norm_text) >= 4 and norm_text in question_norm:
            if ent_id not in matched_entity_ids:
                matched_entity_ids.append(ent_id)

    # ── Pass 1c: noun chunks (if still empty) ────────────────────────────────
    if not matched_entity_ids:
        for chunk in doc.noun_chunks:
            chunk_text = chunk.root.text.strip()
            chunk_id = get_entity_id(chunk_text)
            if chunk_id in graph_entity_ids and chunk_id not in matched_entity_ids:
                matched_entity_ids.append(chunk_id)

    # ── Pass 1d: title-case tokens fallback ───────────────────────────────────
    if not matched_entity_ids:
        for token in doc:
            if token.is_alpha and token.is_title and len(token.text) >= 3:
                token_id = get_entity_id(token.text)
                if token_id in graph_entity_ids and token_id not in matched_entity_ids:
                    matched_entity_ids.append(token_id)

    # ── Pass 2: embedding similarity (HippoRAG-style) ────────────────────────
    # Pre-filter: only embed nodes that are entity IDs (not passages)
    if node_embeddings is not None:
        node_list, matrix = node_embeddings
    elif G.number_of_nodes() <= 5000:
        # Build on-the-fly for small graphs (warn: slow for large graphs)
        node_list, matrix = build_node_embeddings(G)
    else:
        node_list, matrix = [], np.zeros((0, 384))

    if len(node_list) > 0:
        q_vec = _encode_texts([question], text_type="query", batch_size=32)[0]
        sims = matrix @ q_vec  # cosine similarity (both L2-normalized)
        top_indices = np.argsort(sims)[::-1][:top_k_embed * 3]
        added = 0
        for idx in top_indices:
            if added >= top_k_embed:
                break
            if sims[idx] < embed_threshold:
                break
            node_id = node_list[idx]
            if node_id not in matched_entity_ids:
                matched_entity_ids.append(node_id)
                added += 1

    # Return up to 10 unique entity IDs (preserving order)
    return list(dict.fromkeys(matched_entity_ids))[:10]


def rank_passages_by_ppr(
    G,
    question,
    triple_to_passages,
    passage_texts,
    node_to_passages=None,
    node_embeddings=None,
    fact_embeddings=None,
    passage_embeddings=None,
    top_k=10,
    alpha=0.85,
    ppr_type="standard",
    nlp=None,
    fact_top_k=30,
    fact_threshold=0.35,
):
    """
    Rank passages using Personalized PageRank based on query question.

    Algorithm (HippoRAG-style with fact-embedding seeds):
    1. If fact_embeddings provided:
         a. Encode query, compute cosine similarity against all fact strings
         b. Top-k matching facts → extract their entity nodes as PPR seeds
         c. Merge with NER-based seeds (NER seeds kept as fallback)
       Else: use 4-pass NER + node-embedding seeds (legacy path)
    2. PPR with merged seed set
    3. Map top-k PPR nodes → passages via node_to_passages (O(1))
    4. Fallback: if PPR returns 0 passages and passage_embeddings provided,
       return top-k passages by dense query-passage cosine similarity
    
    Args:
        G: NetworkX knowledge graph
        question: Query question string
        triple_to_passages: Dict mapping (s,r,o) tuple → list[passage_id]
        passage_texts: Dict mapping passage_id → passage text
        node_to_passages: Optional {node: set(passage_ids)} for O(1) lookup.
        node_embeddings: Optional (node_list, matrix) from build_node_embeddings(G).
        fact_embeddings: Optional (fact_list, fact_matrix) from build_fact_embeddings().
            fact_list is list of (s, r, o) tuples; fact_matrix is L2-normalised.
        passage_embeddings: Optional (pid_list, passage_matrix) from
            build_passage_embeddings(). Used as dense fallback when PPR finds nothing.
        top_k: Number of passages to return.
        alpha: PPR damping factor (0.85 = standard, 0.5 = HippoRAG).
        ppr_type: "standard", "hipporag", or "fast".
        nlp: Pre-loaded spaCy model.
        fact_top_k: How many top-scoring facts to use as seed source (default 30).
        fact_threshold: Min cosine similarity between query and a fact string
            to accept that fact's entities as PPR seeds (default 0.35).

    Returns:
        List of (passage_id, passage_text, ppr_score) tuples, sorted by score.
    """

    # ── Seed extraction ─────────────────────────────────────────────────
    # Always run NER-based extractor (fast, covers exact matches)  
    # Returns entity IDs in hash format from extract_entities_from_question
    query_entities = extract_entities_from_question(
        question, G, nlp, node_embeddings=node_embeddings
    )
    fact_passage_scores = defaultdict(float)

    # Augment seeds from fact-embedding similarity (HippoRAG core idea)
    # CRITICAL: fact_list contains (s_norm, r, o_norm) tuples where s_norm, o_norm are NORMALIZED STRINGS
    # Entity IDs are: entity_id = hash(normalize(raw_text))
    # Since fact_list already has NORMALIZED strings, we can directly get entity IDs
    if fact_embeddings is not None:
        fact_list, fact_matrix = fact_embeddings
        if len(fact_list) > 0:
            q_vec = _encode_texts([question], text_type="query", batch_size=32)[0]
            sims = fact_matrix @ q_vec          # (N,)
            top_idxs = np.argsort(sims)[::-1][:fact_top_k]
            graph_nodes = set(G.nodes())
            
            for idx in top_idxs:
                if sims[idx] < fact_threshold:
                    break
                s_norm, r, o_norm = fact_list[idx]
                
                # Convert normalized strings directly to entity IDs
                # get_entity_id(normalized_string) = hash(normalized_string) 
                # (since normalize(normalized_string) = normalized_string)
                for norm_text in (s_norm, o_norm):
                    if not norm_text or len(norm_text) < 3:
                        continue
                    entity_id = get_entity_id(norm_text)
                    if entity_id in graph_nodes and entity_id not in query_entities:
                        query_entities.append(entity_id)
                
                # Map fact passages with fact's normalized key
                for pid in triple_to_passages.get((s_norm, r, o_norm), []):
                    fact_passage_scores[pid] = max(fact_passage_scores[pid], float(sims[idx]))

    def _dense_passage_fallback():
        """Return top-k passages by query-passage cosine similarity."""
        if passage_embeddings is not None:
            pid_list, pass_matrix = passage_embeddings
            if len(pid_list) > 0:
                q_v = _encode_texts([question], text_type="query", batch_size=32)[0]
                scores = pass_matrix @ q_v
                top_idxs = np.argsort(scores)[::-1][:top_k]
                return [
                    (pid_list[i], passage_texts[pid_list[i]], float(scores[i]))
                    for i in top_idxs
                    if pid_list[i] in passage_texts
                ]
        # Last resort: first top_k passages
        pids = list(passage_texts.keys())[:top_k]
        return [(pid, passage_texts[pid], 0.0) for pid in pids]

    if not query_entities:
        return _dense_passage_fallback()

    # Run PPR
    if ppr_type == "hipporag":
        ranked_nodes = personalized_pagerank_hipporag(
            G, query_entities, top_k=min(100, G.number_of_nodes()), damping=0.5
        )
    elif ppr_type == "fast":
        ranked_nodes = personalized_pagerank_fast(
            G, query_entities, top_k=min(100, G.number_of_nodes()), alpha=alpha
        )
    else:  # standard
        ranked_nodes = personalized_pagerank(
            G, query_entities, top_k=min(100, G.number_of_nodes()), alpha=alpha
        )

    if not ranked_nodes:
        return _dense_passage_fallback()

    # Map nodes to passages with lightweight score fusion.
    # Prefer direct passage nodes if the graph contains them, then fuse with the
    # legacy node_to_passages lookup for additional entity evidence.
    passage_max_scores = defaultdict(float)
    passage_sum_scores = defaultdict(float)
    node_hits_by_passage = defaultdict(set)

    direct_passages = get_top_passages_from_ranked_nodes(
        G,
        ranked_nodes,
        top_k=max(top_k * 5, 25),
        passage_texts=passage_texts,
    )
    for pid, _, score in direct_passages:
        passage_max_scores[pid] = max(passage_max_scores[pid], score)
        passage_sum_scores[pid] += score
        node_hits_by_passage[pid].add(f"passage::{pid}")

    if node_to_passages is not None:
        # Fast O(1) indexed lookup (preferred)
        for node, score in ranked_nodes:
            if node in G and G.nodes[node].get("node_type") == "passage":
                continue
            for pid in node_to_passages.get(node, set()):
                passage_max_scores[pid] = max(passage_max_scores[pid], score)
                passage_sum_scores[pid] += score
                node_hits_by_passage[pid].add(node)
    else:
        # Fallback: O(n*m) linear scan over all triples
        for node, score in ranked_nodes:
            if node in G and G.nodes[node].get("node_type") == "passage":
                continue
            for (s, r, o), passage_ids in triple_to_passages.items():
                if normalize(s) == node or normalize(o) == node:
                    for pid in passage_ids:
                        passage_max_scores[pid] = max(passage_max_scores[pid], score)
                        passage_sum_scores[pid] += score
                        node_hits_by_passage[pid].add(node)

    passage_scores = {
        pid: passage_max_scores[pid] + 0.20 * passage_sum_scores[pid] + 0.03 * len(node_hits_by_passage[pid])
        for pid in passage_max_scores
    }

    # Dense fallback if PPR found no passages
    if not passage_scores:
        return _dense_passage_fallback()

    # Lightweight reranking / multi-hop score fusion on only the top candidates.
    top_candidates = dict(
        sorted(passage_scores.items(), key=lambda x: x[1], reverse=True)[:max(top_k * 5, 25)]
    )
    ranked_passages = _rerank_passage_candidates(
        question=question,
        query_entities=query_entities,
        candidate_scores=top_candidates,
        passage_texts=passage_texts,
        node_hits_by_passage=node_hits_by_passage,
        fact_passage_scores=fact_passage_scores,
        top_k=top_k,
    )

    # Deduplicate (exact text + normalized)
    seen_texts = set()
    result = []

    for passage_id, passage_text, score in ranked_passages:
        if passage_id not in passage_texts:
            continue

        text_norm = normalize(passage_text)
        if text_norm not in seen_texts:
            seen_texts.add(text_norm)
            result.append((passage_id, passage_text, score))

        if len(result) >= top_k:
            break

    return result


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
    
    # ← Test HippoRAG style PPR
    print("\n=== HippoRAG Style PPR (damping=0.5) ===")
    ranked_hipporag = personalized_pagerank_hipporag(
        G, query_entities, top_k=30
    )
    
    if not ranked_hipporag:
        print("Không tìm thấy entity trong graph!")
    else:
        print("Top 30 nodes:")
        for i, (node, score) in enumerate(ranked_hipporag, 1):
            print(f"  {i}. {node}: {score:.4f}")

        # Lấy top 10 cho subgraph
        top_10_ranked = ranked_hipporag[:10]
        subG = get_subgraph(G, top_10_ranked, hops=4)
        print(f"\nSubgraph có {subG.number_of_nodes()} nodes, {subG.number_of_edges()} edges")
    
    # ← Test traditional PPR (for comparison)
    print("\n=== Traditional PPR (damping=0.85) ===")
    ranked_traditional = personalized_pagerank(
        G, query_entities, top_k=30,
        entity_type_filter=['person', 'work', 'organization']
    )
    
    if ranked_traditional:
        print("Top 10 nodes:")
        for i, (node, score) in enumerate(ranked_traditional[:10], 1):
            print(f"  {i}. {node}: {score:.4f}")