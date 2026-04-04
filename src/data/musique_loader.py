import json
from pathlib import Path

DATA_PATH = Path("data/raw/musique/data")


def load_musique(split="dev", max_samples=None, answerable_only=False):
    """
    split: 'dev' | 'train' | 'test'
    answerable_only: chỉ lấy samples có answerable=True (recommended khi eval)
    """
    file_path = DATA_PATH / f"musique_ans_v1.0_{split}.jsonl"

    samples = []
    with open(file_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            item = json.loads(line)

            if answerable_only and not item["answerable"]:
                continue

            # Giữ flag is_supporting trong từng passage
            passages = [
                {
                    "text": p["paragraph_text"],
                    "title": p.get("title", ""),
                    "idx": p.get("idx", i),
                    "is_supporting": p["is_supporting"],
                }
                for i, p in enumerate(item["paragraphs"])
            ]

            samples.append({
                "id": item["id"],
                "question": item["question"],
                "answer": item["answer"],
                "answerable": item["answerable"],
                "passages": passages,
                "hops": len(item["question_decomposition"]),
                "decomposition": item.get("question_decomposition", []),
            })

    return samples


def get_all_passages(samples: list, supporting_only=False) -> list[str]:
    """
    Gom unique passage texts từ samples để build KG.
    supporting_only=True: chỉ lấy passages is_supporting=True (nhanh hơn, ít noise hơn)
    supporting_only=False: lấy hết kể cả distractor (recall cao hơn)
    """
    seen = set()
    passages = []
    for s in samples:
        for p in s["passages"]:
            if supporting_only and not p["is_supporting"]:
                continue
            if p["text"] not in seen:
                seen.add(p["text"])
                passages.append(p["text"])
    return passages


def get_supporting_passages(sample: dict) -> list[str]:
    """Lấy supporting passages của 1 sample cụ thể (dùng khi eval từng câu)"""
    return [p["text"] for p in sample["passages"] if p["is_supporting"]]


if __name__ == "__main__":
    # Dev split — dùng cho eval
    samples = load_musique("dev", max_samples=100, answerable_only=True)
    print(f"Loaded: {len(samples)} answerable samples")
    print(f"Hops distribution: {sorted({s['hops'] for s in samples})}")

    all_passages = get_all_passages(samples, supporting_only=False)
    supporting_passages = get_all_passages(samples, supporting_only=True)
    print(f"All unique passages:        {len(all_passages)}")
    print(f"Supporting-only passages:   {len(supporting_passages)}")

    print(f"\nSample question: {samples[0]['question']}")
    print(f"Answer:          {samples[0]['answer']}")
    print(f"Hops:            {samples[0]['hops']}")
    print(f"Supporting passages ({len(get_supporting_passages(samples[0]))}):")
    for p in get_supporting_passages(samples[0]):
        print(f"  - {p[:80]}...")