"""Debug AiC@k failures — check extraction and entity linking for sample questions."""
import sys
sys.path.insert(0, '.')

from src.extraction.extract_triples import extract_triples_batch, DEFAULT_RELATION_SCHEMA
from src.graph.build_graph import normalize
import spacy

# Gold passages for Q0: "Who is the spouse of the Green performer?"
passages = [
    "Miquette Giraudy (born 9 February 1953, Nice, France) is a keyboard player and vocalist, best known for her work in Gong and with her partner Steve Hillage. She and Hillage currently form the core of the ambient band System 7.",
    "Green is the fourth studio album by British progressive rock musician Steve Hillage. Written in spring 1977 at the same time as his previous album, Green was recorded alone, primarily in Dorking, Surrey.",
]

print("=== Extraction from gold passages ===")
results = extract_triples_batch(
    passages,
    relation_schema=DEFAULT_RELATION_SCHEMA,
    use_dynamic=False,
    skip_cache=True,
    deduplicate=False,
)
for i, r in enumerate(results):
    print(f"\nPassage {i}:")
    for t in r.get("triples", []):
        print(f"  ({t['subject']}, {t['relation']}, {t['object']})")
    ents = [e["text"] for e in r.get("entities", [])]
    print(f"  entities: {ents}")

print("\n=== spaCy on question ===")
nlp = spacy.load("en_core_web_sm")
q = "Who is the spouse of the Green performer?"
doc = nlp(q)
print("NER:", [(e.text, e.label_) for e in doc.ents])
print("Noun chunks:", [c.text for c in doc.noun_chunks])
print("Title tokens:", [t.text for t in doc if t.is_alpha and t.is_title])
