import json
from pathlib import Path

DATA_PATH = Path("data/raw/musique/data")

def load_musique(split="dev", max_samples=None, supporting_only=False):
    """
    split: 'dev' | 'train' | 'test'
    supporting_only: chỉ lấy passages is_supporting=True để test nhanh
    """
    file_path = DATA_PATH / f"musique_ans_v1.0_{split}.jsonl"
    
    samples = []
    with open(file_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            item = json.loads(line)
            
            if supporting_only:
                passages = [p["paragraph_text"] for p in item["paragraphs"] 
                           if p["is_supporting"]]
            else:
                passages = [p["paragraph_text"] for p in item["paragraphs"]]
            
            # Extract supporting facts: list of (idx, title)
            supporting_facts = [(p["idx"], p["title"]) for p in item["paragraphs"] 
                               if p["is_supporting"]]
            
            samples.append({
                "id": item["id"],
                "question": item["question"],
                "answer": item["answer"],
                "answerable": item["answerable"],
                "passages": passages,
                "hops": len(item["question_decomposition"]),
                "supporting_facts": supporting_facts,
                "decomposition": item.get("question_decomposition", [])
            })
    
    return samples


def get_all_passages(samples: list, supporting_only=False) -> list[str]:
    """Gom tất cả passages từ samples để build KG"""
    seen = set()
    passages = []
    for s in samples:
        for p in s["passages"]:
            if p not in seen:
                seen.add(p)
                passages.append(p)
    return passages


if __name__ == "__main__":
    samples = load_musique("dev", max_samples=100)
    print(f"Loaded: {len(samples)} samples")
    print(f"Hops distribution: { {s['hops'] for s in samples} }")
    
    passages = get_all_passages(samples)
    print(f"Unique passages: {len(passages)}")
    print(f"\nSample question: {samples[0]['question']}")
    print(f"Answer: {samples[0]['answer']}")