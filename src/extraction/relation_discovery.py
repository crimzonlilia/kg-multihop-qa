"""
Pass 1 — Relation Discovery
Dùng GLiNER2 với broad relation list → collect → cluster → expand schema
"""

from gliner2 import GLiNER2
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
import numpy as np
import json
from pathlib import Path

# Chỉ keep relations phổ biến trong Wikipedia & QA datasets
BROAD_RELATIONS = [
    # Core biographical
    "born_in", "died_in", "nationality", "located_in", "educated_at",
    # Work/role
    "worked_at", "occupation", "member_of",
    # Organization
    "founded_by", "part_of",
    # Creative works
    "directed_by", "written_by", "performed_by",
    # Relationships
    "associated_with", "known_for", "participated_in"
]

SEED_SCHEMA = {
    "born_in", "died_in", "nationality", "occupation",
    "founded_by", "located_in", "part_of", "occurred_in",
    "educated_at", "worked_at", "member_of"
}

MODEL_NAME = "fastino/gliner2-base-v1"
embedder = SentenceTransformer("all-MiniLM-L6-v2")

CACHE_DIR = Path("data/cache")

def save_freq(freq, name="relations_freq.json"):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)  # ← Thêm dòng này
    with open(CACHE_DIR / name, "w") as f:
        json.dump(dict(freq), f)

def load_freq(name="relations_freq.json"):
    path = CACHE_DIR / name
    if path.exists():
        with open(path) as f:
            return Counter(json.load(f))
    return None

def save_checkpoint(freq, start_idx, name="checkpoint.json"):
    """Save progress để resume khi crash"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)  # ← Thêm dòng này
    checkpoint = {
        "freq": dict(freq),
        "start_idx": start_idx
    }
    with open(CACHE_DIR / name, "w") as f:
        json.dump(checkpoint, f)

def load_checkpoint(name="checkpoint.json"):
    """Load checkpoint nếu có"""
    path = CACHE_DIR / name
    if path.exists():
        with open(path) as f:
            data = json.load(f)
            return Counter(data["freq"]), data["start_idx"]
    return None, 0


def collect_relations(passages: list[str], extractor: GLiNER2, batch_size=32, skip_cache=False) -> Counter:
    """
    Chạy GLiNER2 với broad relations trên toàn corpus.
    Trả về Counter của các relation strings được extract ra.
    """
    if not skip_cache:
        cached = load_freq()
        if cached:
            print(f"✓ Loaded cached: {len(cached)} relations")
            return cached
    
    # Load checkpoint để resume
    freq, start_idx = load_checkpoint()
    if start_idx > 0:
        print(f"✓ Resuming from checkpoint: batch {start_idx // batch_size}")
    else:
        freq = Counter()
    
    total_batches = (len(passages) + batch_size - 1) // batch_size
    
    for batch_idx, i in enumerate(range(start_idx, len(passages), batch_size)):
        batch = passages[i:i+batch_size]
        try:
            results = extractor.batch_extract_relations(batch, BROAD_RELATIONS)
            for result in results:
                relations = result.get("relation_extraction", {})
                for rel, pairs in relations.items():
                    if pairs:
                        freq[rel] += len(pairs)
        except Exception as e:
            continue
        
        # Save checkpoint + cache mỗi 5 batches
        if (batch_idx + 1) % 5 == 0:
            save_checkpoint(freq, i + batch_size)  # Save next start_idx
            save_freq(freq)
            print(f"  [{batch_idx+1}/{total_batches}] {len(freq)} relations (checkpoint saved)")
    
    # Clean up checkpoint khi xong
    checkpoint_path = CACHE_DIR / "checkpoint.json"
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    
    save_freq(freq)
    return freq


def cluster_relations(freq: Counter,
                      distance_threshold: float = 0.35) -> dict:
    """
    Cluster relations tương đồng về semantic.
    Trả về { canonical: [surface_forms] }
    """
    # Chỉ cluster những gì KHÔNG có trong seed
    candidates = [r for r in freq if r not in SEED_SCHEMA]

    if len(candidates) < 2:
        print("Not enough candidates to cluster.")
        return {}

    print(f"Clustering {len(candidates)} candidate relations...")
    embeddings = embedder.encode(candidates, normalize_embeddings=True)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average"
    )
    labels = clustering.fit_predict(embeddings)

    clusters = {}
    for rel, label in zip(candidates, labels):
        clusters.setdefault(label, []).append(rel)

    # Canonical = relation có frequency cao nhất trong cluster
    discovered = {}
    for label, rels in clusters.items():
        canonical = max(rels, key=lambda r: freq[r])
        discovered[canonical] = rels

    return discovered


def build_expanded_schema(discovered: dict,
                           freq: Counter,
                           min_cluster_freq: int = 5) -> dict:
    """
    Merge seed schema + discovered relations đủ lớn.
    Trả về RELATION_SCHEMA dict để pass vào GLiNER2 Pass 2.
    """
    expanded = {
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
        "member_of":   "Person is member of an organization"
    }

    added = 0
    for canonical, surface_forms in discovered.items():
        total_freq = sum(freq[r] for r in surface_forms)
        if total_freq >= min_cluster_freq and canonical not in expanded:
            expanded[canonical] = f"Relation involving: {', '.join(surface_forms[:3])}"
            print(f"  [+] {canonical} (freq={total_freq}, forms={surface_forms})")
            added += 1

    print(f"Schema: {len(SEED_SCHEMA)} → {len(expanded)} (+{added} new)")
    return expanded


def run_discovery(passages: list[str],
                  min_cluster_freq: int = 5,
                  distance_threshold: float = 0.35) -> dict:
    """
    Full Pass 1 pipeline.
    Returns: expanded RELATION_SCHEMA dict
    """
    print(f"\n=== Pass 1: Relation Discovery ({len(passages)} passages) ===")

    extractor = GLiNER2.from_pretrained(MODEL_NAME)

    freq = collect_relations(passages, extractor)
    print(f"\nRelations found: {len(freq)}")
    print(f"Top 15: {freq.most_common(15)}")

    discovered = cluster_relations(freq, distance_threshold)
    expanded_schema = build_expanded_schema(discovered, freq, min_cluster_freq)

    return expanded_schema


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data.musique_loader import load_musique, get_all_passages

    samples = load_musique("dev", max_samples=200)
    passages = get_all_passages(samples)

    expanded_schema = run_discovery(passages, min_cluster_freq=3)

    print("\n=== Final Schema ===")
    for rel, desc in expanded_schema.items():
        marker = "NEW" if rel not in SEED_SCHEMA else "   "
        print(f"  [{marker}] {rel}: {desc}")