#!/usr/bin/env python3
"""Debug entity extraction."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from src.cache_utils import build_graph_output_path
from src.retrieval.pagerank import extract_entities_from_question
import pickle
import spacy

# Load graph
graph_path = build_graph_output_path('500')
with open(graph_path, 'rb') as f:
    G = pickle.load(f)

# Load spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Test questions
test_questions = [
    "Who is the spouse of the Green performer?",
    "Who founded the company that distributed the film UHF?",
    "What administrative territorial entity is the owner of Ciudad Deportivo?",
]

print("Testing entity extraction:")
print("=" * 80)

for q in test_questions:
    print(f"\nQuestion: {q}")
    print(f"Question length: {len(q)}")
    
    extracted = extract_entities_from_question(q, G, nlp=nlp)
    print(f"  Raw extracted (entity IDs): {extracted}")
    print(f"  Number of entities: {len(extracted)}")
    
    # Get the actual text for each 
    for ent_id in extracted:
        if ent_id in G:
            raw_text = G.nodes[ent_id].get("raw_text", "N/A")
            print(f"    {ent_id[:30]}... -> '{raw_text}'")
        else:
            print(f"    {ent_id} NOT IN GRAPH")
    
    if not extracted:
        print(f"  ⚠️  NO ENTITIES EXTRACTED")
        
        # Debug: Let's see what spaCy finds
        doc = nlp(q)
        print(f"    spaCy NER found: {[ent.text for ent in doc.ents]}")
        print(f"    Noun chunks: {[chunk.text for chunk in doc.noun_chunks]}")
