from gliner2 import GLiNER2
from pathlib import Path
import json
import os

MODEL_NAME = "fastino/gliner2-base-v1"

ENTITY_LABELS = ["person", "organization", "location", "event", "role"]

DEFAULT_RELATION_SCHEMA = {
    "born_in":     "Person was born in a location",
    "died_in":     "Person died in a location",
    "nationality": "Person holds citizenship of a country",
    "occupation":  "Person has a job role",
    "founded_by":  "Organization was founded by a person",
    "located_in":  "Entity located in a place",
    "part_of":     "Organization is part of another organization",
    "occurred_in": "Event occurred in a location",
    "educated_at": "Person studied at an organization",
    "worked_at":   "Person worked at an organization",
    "member_of":   "Person is member of an organization",
}

TYPE_CONSTRAINTS = {
    "born_in":     ("person",       ["location"]),
    "died_in":     ("person",       ["location"]),
    "nationality": ("person",       ["location"]),
    "occupation":  ("person",       ["role"]),
    "founded_by":  ("organization", ["person"]),
    "located_in":  ("organization", ["location"]),
    "part_of":     ("organization", ["organization"]),
    "occurred_in": ("event",        ["location"]),
    "educated_at": ("person",       ["organization"]),
    "worked_at":   ("person",       ["organization"]),
    "member_of":   ("person",       ["organization"]),
}

CACHE_DIR = Path("data/cache")
TRIPLES_CACHE = CACHE_DIR / "triples.json"

# Thêm vào extract_triples_batch
os.environ['OMP_NUM_THREADS'] = '8'  # Dùng 8 CPU cores

def build_schema(extractor, relation_schema: dict):
    """Build GLiNER2 schema từ dynamic relation dict."""
    return (
        extractor.create_schema()
        .entities(ENTITY_LABELS)
        .relations(relation_schema)
    )


def extract_information(text: str, extractor, schema) -> tuple[list, list, int]:
    """Extract entities + typed triples từ 1 passage.
    Returns: entities, triples, filtered_count
    """
    results = extractor.extract(text, schema, include_confidence=True)

    entities = []
    for label, items in results.get("entities", {}).items():
        for item in items:
            entities.append({
                "text": item["text"] if isinstance(item, dict) else item,
                "type": label,
                "score": item.get("confidence") if isinstance(item, dict) else None,
            })

    entity_type = {e["text"]: e["type"] for e in entities}

    triples = []
    filtered_count = 0
    for rel, pairs in results.get("relation_extraction", {}).items():
        for pair in pairs:
            if isinstance(pair, (list, tuple)):
                head, tail, score = pair[0], pair[1], None
            elif isinstance(pair, dict):
                head = pair["head"]["text"] if isinstance(pair["head"], dict) else pair["head"]
                tail = pair["tail"]["text"] if isinstance(pair["tail"], dict) else pair["tail"]
                head_score = pair["head"].get("confidence", 1.0) if isinstance(pair["head"], dict) else 1.0
                tail_score = pair["tail"].get("confidence", 1.0) if isinstance(pair["tail"], dict) else 1.0
                score = min(head_score, tail_score)
            else:
                continue

            # Type constraint filtering
            if rel in TYPE_CONSTRAINTS:
                expected_subj, expected_objs = TYPE_CONSTRAINTS[rel]
                head_type = entity_type.get(head, "unknown")  # Default "unknown"
                tail_type = entity_type.get(tail, "unknown")
                
                # Chỉ filter nếu type khác với expected, bỏ qua nếu unknown
                if head_type != "unknown" and head_type != expected_subj:
                    filtered_count += 1
                    continue
                
                if tail_type != "unknown" and tail_type not in expected_objs:
                    filtered_count += 1
                    continue

            triples.append({"subject": head, "relation": rel, "object": tail, "score": score})

    return entities, triples, filtered_count


def extract_triples_batch(passages: list[str],
                           relation_schema: dict = None,
                           batch_size: int = 32,
                           skip_cache: bool = True) -> list[dict]:
    """
    Chạy Pass 2 trên toàn corpus.
    Returns: list of { passage, entities, triples }
    """
    if not skip_cache and TRIPLES_CACHE.exists():
        with open(TRIPLES_CACHE) as f:
            cached = json.load(f)
        print(f"  Loaded triples cache: {len(cached)} passages")
        return cached

    schema_dict = relation_schema or DEFAULT_RELATION_SCHEMA
    extractor = GLiNER2.from_pretrained(MODEL_NAME)
    schema = build_schema(extractor, schema_dict)

    results = []
    total = len(passages)
    total_filtered = 0

    for i in range(0, total, batch_size):
        batch = passages[i : i + batch_size]
        batch_filtered = 0
        
        for passage in batch:
            try:
                entities, triples, filtered = extract_information(passage, extractor, schema)
                results.append({"passage": passage, "entities": entities, "triples": triples})
                batch_filtered += filtered
                total_filtered += filtered
            except Exception as e:
                results.append({"passage": passage, "entities": [], "triples": []})

        if (i // batch_size) % 1 == 0:
            total_triples = sum(len(r['triples']) for r in results)
            print(f"  [{i + len(batch)}/{total}] {total_triples} triples so far (bỏ {total_filtered} vì constraints)")

    # Cache kết quả
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRIPLES_CACHE, "w") as f:
        json.dump(results, f)
    total_kept = sum(len(r['triples']) for r in results)
    print(f"  Done. {total_kept} triples giữ lại, {total_filtered} bỏ đi từ {len(results)} passages")
    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data.musique_loader import load_musique, get_all_passages
    from src.extraction.relation_discovery import run_discovery

    samples = load_musique("dev", max_samples=None)
    passages = get_all_passages(samples)

    # Load expanded schema từ Pass 1
    expanded_schema = run_discovery(passages, min_cluster_freq=3)

    # Pass 2 với dynamic schema
    results = extract_triples_batch(passages, relation_schema=expanded_schema)

    # Sample output
    for r in results[:3]:
        print(f"\nPassage: {r['passage'][:80]}...")
        for t in r["triples"]:
            print(f"  ({t['subject']}, {t['relation']}, {t['object']})")