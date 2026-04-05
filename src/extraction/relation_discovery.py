"""
Pass 1 — Relation Discovery (True Adaptive)
Extract free-form (subject, verb, object) từ text với NER filter
→ normalize → embed → cluster → expand schema
"""

import spacy
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
import json
from pathlib import Path

nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

CACHE_DIR = Path("data/cache")

SEED_SCHEMA = {
    "born_in":     "Person was born in a location",
    "died_in":     "Person died in a location",
    "nationality": "Person holds citizenship of a country",
    "occupation":  "Person has a job or role",
    "founded_by":  "Organization was founded by a person",
    "located_in":  "Entity is located in a place",
    "part_of":     "Entity is part of another entity",
    "occurred_in": "Event occurred in a location",
    "educated_at": "Person studied at an organization",
    "worked_at":   "Person worked at an organization",
    "member_of":   "Person is member of an organization",
    "spouse":      "Person is married to another person",
    "sibling":     "Person is sibling of another person",
    "parent_of":   "Person is parent of another person",
    "child_of":    "Person is child of another person",
    "distributed_by": "Work was distributed by an organization",
    "directed_by": "Work was directed by a person",
    "written_by":  "Work was written by a person",
    "produced_by": "Work was produced by a person",
    "starred_in":  "Person starred in a work",
    "performed_in": "Person performed in a work",
    "owner_of":    "Organization owns something",
    "owned_by":    "Entity is owned by an organization",
    "capital_of":  "Location is capital of a country",
}

STOP_VERBS = {
    "be", "have", "do", "say", "get", "make", "go", "know", "take",
    "see", "come", "think", "look", "want", "give", "use", "find",
    "tell", "ask", "seem", "feel", "try", "leave", "call", "keep",
    "include", "become", "show", "consider", "allow", "move", "play",
    "increase", "receive", "report", "describe", "refer", "remain",
    "continue", "result", "follow", "lead", "provide", "require"
}


# ── Cache helpers ─────────────────────────────────────────────────────────────

def save_cache(data, name):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_DIR / name, "w") as f:
        json.dump(data, f)

def load_cache(name):
    path = CACHE_DIR / name
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

def save_checkpoint(freq, start_idx):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_DIR / "checkpoint.json", "w") as f:
        json.dump({"freq": dict(freq), "start_idx": start_idx}, f)

def load_checkpoint():
    data = load_cache("checkpoint.json")
    if data:
        return Counter(data["freq"]), data["start_idx"]
    return Counter(), 0


# ── Step 1: Extract raw relation strings ─────────────────────────────────────

def token_in_ent(token, doc) -> bool:
    """Check token có nằm trong một NER span không."""
    return any(token.i >= ent.start and token.i < ent.end for ent in doc.ents)


def extract_verb_triples(text: str) -> list[str]:
    """
    Dùng spaCy dependency parsing + NER để lấy relations.
    Chỉ giữ verb nếu cả subject VÀ object đều là named entity.
    Trả về list relation strings (verb_lemma hoặc verb_lemma_prep).
    """
    doc = nlp(text)
    relations = []

    for token in doc:
        if token.pos_ != "VERB":
            continue

        verb_lemma = token.lemma_.lower()
        if verb_lemma in STOP_VERBS or len(verb_lemma) < 3:
            continue

        # Chỉ lấy verb nếu subject là NER
        has_ent_subj = any(
            token_in_ent(child, doc)
            for child in token.children
            if child.dep_ in ("nsubj", "nsubjpass")
        )
        if not has_ent_subj:
            continue

        # Tìm prep có pobj là NER → verb_prep style
        found_prep = False
        for child in token.children:
            if child.dep_ != "prep":
                continue
            pobjs = [c for c in child.children if c.dep_ == "pobj"]
            if pobjs and token_in_ent(pobjs[0], doc):
                relations.append(f"{verb_lemma}_{child.text.lower()}")
                found_prep = True
                break  # chỉ lấy prep đầu tiên có entity object

        # Nếu không có prep thì check direct object là NER
        if not found_prep:
            has_ent_obj = any(
                token_in_ent(child, doc)
                for child in token.children
                if child.dep_ in ("dobj", "attr")
            )
            if has_ent_obj:
                relations.append(verb_lemma)

    return relations


def collect_raw_relations(passages: list[str],
                           skip_cache: bool = False) -> Counter:
    """Chạy spaCy trên toàn corpus, đếm frequency raw relation strings."""
    if not skip_cache:
        cached = load_cache("raw_relations_freq.json")
        if cached:
            print(f"✓ Loaded cached raw relations: {len(cached)} types")
            return Counter(cached)

    freq, start_idx = load_checkpoint()
    if start_idx > 0:
        print(f"✓ Resuming from passage {start_idx}")

    for i, passage in enumerate(passages[start_idx:], start=start_idx):
        try:
            rels = extract_verb_triples(passage)
            freq.update(rels)
        except Exception as e:
            print(f"  [!] Passage {i} failed: {e}")
            continue

        if (i + 1) % 200 == 0:
            save_checkpoint(freq, i + 1)
            print(f"  [{i+1}/{len(passages)}] {len(freq)} unique relation types")

    cp = CACHE_DIR / "checkpoint.json"
    if cp.exists():
        cp.unlink()

    save_cache(dict(freq), "raw_relations_freq.json")
    return freq


# ── Step 2: Filter + Normalize ────────────────────────────────────────────────

def normalize_relation(rel: str) -> str:
    """
    Lemmatize verb part của relation string.
    written_by → write_by, directed_by → direct_by
    Giúp cluster gom được các surface forms cùng nghĩa.
    """
    parts = rel.split("_")
    try:
        parts[0] = nlp(parts[0])[0].lemma_
    except Exception:
        pass
    return "_".join(parts)


def filter_candidates(freq: Counter,
                       min_freq: int = 5,
                       top_k: int = 80) -> list[str]:
    """Giữ top-k relations có freq >= min_freq, loại seed schema."""
    candidates = [
        r for r, count in freq.most_common(top_k * 2)
        if count >= min_freq
        and r not in SEED_SCHEMA
        and not any(char.isupper() for char in r) 
    ]
    return candidates[:top_k]


# ── Step 3: Cluster ───────────────────────────────────────────────────────────

def cluster_relations(candidates: list[str],
                       freq: Counter,
                       distance_threshold: float = 0.35) -> dict[str, list[str]]:
    """
    Normalize → embed → Agglomerative cluster.
    Canonical = member có frequency cao nhất trong cluster.
    Returns { canonical: [surface_forms] }
    """
    if len(candidates) < 2:
        print("Not enough candidates to cluster.")
        return {}

    # Normalize trước khi embed để gom surface forms cùng nghĩa
    normalized = [normalize_relation(r) for r in candidates]

    print(f"Clustering {len(candidates)} candidates (threshold={distance_threshold})...")
    embeddings = embedder.encode(normalized, normalize_embeddings=True)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average"
    )
    labels = clustering.fit_predict(embeddings)

    # Group by cluster label, dùng original (non-normalized) string
    clusters: dict[int, list[str]] = {}
    for rel, label in zip(candidates, labels):
        clusters.setdefault(label, []).append(rel)

    # Canonical = highest freq member
    discovered = {}
    for label, rels in clusters.items():
        canonical = max(rels, key=lambda r: freq[r])
        discovered[canonical] = rels

    return discovered


# ── Step 4: Build expanded schema ─────────────────────────────────────────────

def build_expanded_schema(discovered: dict[str, list[str]],
                           freq: Counter,
                           min_cluster_freq: int = 10,
                           dedup_threshold: float = 0.75) -> dict[str, str]:
    """Seed schema + discovered clusters đủ lớn, loại redundant với seed."""
    expanded = dict(SEED_SCHEMA)

    # Relations that are too generic / not useful as KG hops
    NOISE_RELATIONS = {
        "spend", "spruce", "invade", "enter", "meet", "die", "replace",
        "represent", "merge_in", "cede_to", "grow_in",
        # near-synonyms of seed (catch what embedding misses)
        "bear_in", "locate_in", "die_in", "educate_at",
    }

    # Embed seed relation names once for dedup comparison
    seed_keys = list(SEED_SCHEMA.keys())
    seed_embs = embedder.encode(seed_keys, normalize_embeddings=True)  # (n_seed, dim)

    added = skipped_freq = skipped_dup = 0
    for canonical, surface_forms in discovered.items():
        total_freq = sum(freq[r] for r in surface_forms)
        if total_freq < min_cluster_freq:
            skipped_freq += 1
            continue
        if canonical in expanded or canonical in NOISE_RELATIONS:
            skipped_dup += 1
            continue

        # Semantic dedup: skip if too similar to any existing seed relation
        cand_emb = embedder.encode([canonical], normalize_embeddings=True)  # (1, dim)
        sims = (seed_embs @ cand_emb.T).flatten()  # cosine sim via dot (normalized)
        if sims.max() >= dedup_threshold:
            most_similar = seed_keys[int(sims.argmax())]
            print(f"  [~] {canonical:20s} ~= {most_similar} (sim={sims.max():.2f}) skip")
            skipped_dup += 1
            continue

        desc = f"Relation expressed as: {', '.join(surface_forms[:4])}"
        expanded[canonical] = desc
        print(f"  [+] {canonical:20s} freq={total_freq:4d}  forms={surface_forms[:3]}")
        added += 1

    print(f"\nSchema: {len(SEED_SCHEMA)} seed -> {len(expanded)} total "
          f"(+{added} new, {skipped_dup} deduped/noise, {skipped_freq} low-freq)")
    return expanded


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_discovery(passages: list[str],
                  min_freq: int = 5,
                  min_cluster_freq: int = 10,
                  distance_threshold: float = 0.35,
                  top_k: int = 80) -> dict[str, str]:
    """Full Pass 1 pipeline. Returns expanded RELATION_SCHEMA."""
    print(f"\n=== Pass 1: Relation Discovery ({len(passages)} passages) ===")

    freq = collect_raw_relations(passages)
    print(f"\nRaw relation types : {len(freq)}")
    print(f"Top 20             : {freq.most_common(20)}")

    candidates = filter_candidates(freq, min_freq=min_freq, top_k=top_k)
    print(f"\nCandidates after filter: {len(candidates)}")

    discovered = cluster_relations(candidates, freq, distance_threshold)
    print(f"Clusters found         : {len(discovered)}")

    expanded_schema = build_expanded_schema(discovered, freq, min_cluster_freq)

    save_cache(expanded_schema, "expanded_schema.json")
    return expanded_schema


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data.musique_loader import load_musique, get_all_passages

    samples = load_musique("dev", max_samples=200, answerable_only=True)
    passages = get_all_passages(samples, supporting_only=False)

    expanded_schema = run_discovery(
        passages,
        min_freq=5,
        min_cluster_freq=10,
        distance_threshold=0.35,
        top_k=80
    )

    print("\n=== Final Schema ===")
    for rel, desc in expanded_schema.items():
        marker = "NEW" if rel not in SEED_SCHEMA else "   "
        print(f"  [{marker}] {rel:20s} {desc}")