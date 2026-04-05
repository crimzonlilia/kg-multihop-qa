# Knowledge Graph Pipeline - Comprehensive Documentation

**Last Updated**: April 4, 2026  
**Purpose**: Technical documentation of the current KG pipeline architecture  
**For session notes, issues & optimization history**: See [DEVLOG.md](DEVLOG.md)

---

## Overview

This document describes the complete end-to-end pipeline for multi-hop question answering over MusiQue dataset:

```
Raw Passages
    ↓
[Stage 1] Data Loading (MusiQue Loader)
    ↓
[Stage 2] Triple Extraction (GLiNER2 + Dynamic Schema Grouping)
    ↓
[Stage 3] Graph Building (Per-Passage Graphs + Merge)
    ↓
[Stage 4] Entity Linking (Consolidate Similar Entities)
    ↓
[Stage 5] Retrieval (PASSAGE RANKING by Entity Coverage)
    ↓
[Stage 6] Evaluation (Hit@1, Hit@5, Hit@10)
    ↓
Answer
```

---

## Stage 1: Data Loading

**File**: `src/data/musique_loader.py`  
**Entry Point**: `load_musique(split, max_samples=None)`

### Purpose
Load MusiQue dataset samples and extract all passages from supporting context.

### Process
1. **Load JSONL file**:
   - Train: `data/raw/musique/train.jsonl`
   - Dev: `data/raw/musique/dev.jsonl`
   - Each line is a JSON object with structure:
     ```json
     {
       "question": "Which city was the actor who played X born in?",
       "answer": "Berlin",
       "paragraphs": [
         {
           "paragraph_text": "...",
           "is_supporting": true/false
         }
       ]
     }
     ```

2. **Extract Sample Data**:
   - question: Multi-hop query requiring multiple reasoning steps
   - answer: String answer (typically entity name or span)
   - paragraphs: List of supporting documents
   - (metadata): hop count for multi-hop classification

3. **Get All Passages** (via `get_all_passages()`):
   - Collect all `paragraph_text` from all samples
   - Return flat list of unique passages
   - **Current Issue**: Only ~40 passages loaded from 200 samples (needs investigation)

### Output
```python
{
  "question": str,
  "answer": str,
  "paragraphs": List[str],  # passage texts
  "supporting_fact_indices": List[int],
  "hops": int  # number of reasoning steps
}
```

---

## Stage 2: Triple Extraction

**File**: `src/extraction/extract_core.py`  
**Entry Point**: `extract_triples_batch(passages, relation_schema, use_dynamic=True, ...)`

### Architecture

#### 2.1 Base Model: GLiNER2
- **Model**: `fastino/gliner2-base-v1`
- **Task**: Jointly extract named entities AND relations from text
- **Hardware**: CUDA with FP16 for efficiency
- **Batch Size**: 32 (internal batching to prevent GPU spikes)
- **Methods**:
  - `batch_extract_entities(texts, entity_labels)` → Extract named entities
  - `batch_extract_relations(texts, relations)` → Extract entity relations

#### 2.2 Entity Extraction

**File**: `src/extraction/schema.py`  
**Entity Labels**: `["person", "organization", "location", "event", "role"]`

Process:
1. Parse GLiNER2 output for each entity type
2. Filter generic pronouns: `{he, she, it, they, ...}`
3. Filter entities < 3 characters
4. Build `entity_type` map: `{text_lower: entity_type}`

#### 2.3 Relation Schema

**Base Schema** (9 relations):
```python
DEFAULT_RELATION_SCHEMA = {
    "born_in": "Person was born in a location",
    "died_in": "Person died in a location",
    "founded_by": "Organization was founded by a person",
    "located_in": "Entity located in a place",
    "part_of": "Organization is part of another organization",
    "occurred_in": "Event occurred in a location",
    "educated_at": "Person studied at an organization",
    "worked_at": "Person worked at an organization",
    "member_of": "Person is member of an organization",
}
```

**Type Constraints**: Filter invalid triples (e.g., `object` must be location for `born_in`)

#### 2.4 TWO EXTRACTION MODES

##### Mode A: Standard Batch Processing (use_dynamic=False)
```
Passage 1⎯⎯→ GLiNER2 [Entity + Relation Extraction] ⎯⎯→ Triples 1
Passage 2⎯⎯→ GLiNER2 [Entity + Relation Extraction] ⎯⎯→ Triples 2
Passage 3⎯⎯→ GLiNER2 [Entity + Relation Extraction] ⎯⎯→ Triples 3
...
```
- **Speed**: Fast (batch of 32)
- **Accuracy**: Can miss relations not in ALL-relations schema

##### Mode B: Dynamic Schema Grouping (use_dynamic=True) ✓ **CURRENT**
```
[Embedding + Clustering Phase]
Passage 1,3,5,7 ──┐
                  ├──→ Schema Group A (relations: born_in, died_in, located_in)
Passage 2,4,6    ─┴→ GLiNER2 [Targeted Extraction]
...
```

**Process**:
1. **Embed Passages**: SentenceTransformer (`all-MiniLM-L6-v2`)
2. **Embed Relations**: Average word embeddings of relation names
3. **Find Optimal Relations for Each Passage** (k-NN):
   - Find k=8 relations most similar to passage semantic content
   - Use cosine similarity
   - Threshold: 0.5 (skip if similarity < threshold)
4. **Group Passages by Relation Schema**:
   - Passages with same optimal relations → same group
   - Creates ~17-20 groups from 40 passages (more granular than single schema)
5. **Extract Per-Group**:
   - All passages in group extracted with specific relations
   - Model focuses only on relevant relations (higher precision)
   - Single GLiNER2 call batches all passages with same relations

**Benefits**:
- Reduces extraction noise (don't look extract unrelated relations)
- Typically increases triples per passage: 1.4→3-5
- Slight slowdown (embedding overhead), but per-group batching compensates

**Code Flow**:
```python
grouping = get_dynamic_schemas_grouped(
    passages, 
    relation_schema=DEFAULT_RELATION_SCHEMA,
    k=8,                    # Top 8 most similar relations per passage
    threshold=0.5           # Min similarity cutoff
)

for schema_key, passage_indices in grouping['groups'].items():
    all_passages_for_schema = [passages[i] for i in passage_indices]
    group_schema = grouping['schemas'][schema_key]
    
    # Single GLiNER2 call for all passages + group's relations
    entity_results = extractor.batch_extract_entities(
        all_passages_for_schema, entity_labels, batch_size=32
    )
    relation_results = extractor.batch_extract_relations(
        all_passages_for_schema, list(group_schema.keys()), batch_size=32
    )
    
    # Process results (pos = position in this group's passages)
    for pos, pass_idx in enumerate(passage_indices):
        results.append({
            "passage": all_passages_for_schema[pos],  # ← FIXED: use actual passage text
            "entities": [...],
            "triples": [...]
        })
```

#### 2.5 Triple Format

Each triple is a dict:
```python
{
    "subject": "Albert Einstein",       # str
    "relation": "born_in",              # str (from schema)
    "object": "Ulm",                    # str
    "confidence": 0.94                  # float [0,1]
}
```

#### 2.6 Caching & Output

**Cache File**: `data/cache/triples.json`  
**Output Format**: List of dicts
```python
[
    {
        "passage": "Einstein was born in Ulm, Germany in 1879...",
        "entities": [...],
        "triples": [
            {"subject": "Albert Einstein", "relation": "born_in", "object": "Ulm", ...},
            ...
        ]
    },
    ...
]
```

**Cache Control**:
- `skip_cache=False`: Use cached triples if exists
- `skip_cache=True`: Recompute from scratch
- `save_cache_to_disk=False`: Don't save results (prevent contamination)
- `deduplicate=False`: Keep per-passage data intact (don't merge passages)

---

## Stage 3: Graph Building

**File**: `src/graph/build_graph.py`  
**Core**: `src/graph/graph_core.py`

### 3.1 Per-Passage Graph Construction

**Component**: `build_graph(triples, entities, add_reverse_edges=False, add_cooccurrence_edges=True)`

For each passage with triples:
```
Triples:
  (Albert Einstein, born_in, Ulm)
  (Ulm, located_in, Germany)
  (Albert Einstein, worked_at, Princeton)

Creates Graph:
  Nodes: {Albert Einstein, Ulm, Germany, Princeton}
  Edges:
    Albert Einstein --born_in-→ Ulm
    Ulm --located_in-→ Germany
    Albert Einstein --worked_at-→ Princeton
    
  [OPTIONAL] Cooccurrence edges:
    Albert Einstein ←--cooccurrence--→ Ulm (same passage)
    Ulm ←--cooccurrence--→ Germany
```

**Graph Type**: `networkx.DiGraph`  
**Node Attributes**: `{"type": "person"|"location"|...}`  
**Edge Attributes**: `{"relation": "born_in", "source": "passage_id"}`

### 3.2 Graph Merging

**Component**: `merge_graphs(graphs_list)`

Steps:
1. Union all nodes + edges (keep attributes)
2. Deduplicate edges (same subject-relation-object triple)
3. Handle node conflicts:
   - Same node name different types: keep union of types
   - Same node name same type: merge attributes

**Result**: Single unified `networkx.DiGraph` representing entire knowledge base
```
40 Individual Graphs
  (avg 2 nodes, 3 edges each)
       ↓
    [MERGE]
       ↓
Single Unified Graph
  (74 nodes, 94 edges from 40 passages)
```

### 3.3 Normalization

**Function**: `normalize(text)`  
- Lowercase
- Strip whitespace
- Remove punctuation
- Collapse multiple spaces

**Purpose**: Enable entity matching despite capitalization/spacing variations

---

## Stage 4: Entity Linking

**File**: `src/graph/entity_linking.py`  
**Component**: `link_entities_fast(G)`

### Purpose
Consolidate similar or duplicate entities:
- `Albert Einstein` vs `Einstein` (coreference)
- `New York` vs `New York City` (abbreviation)
- `USA` vs `United States of America` (alias)

### Algorithm: Semantic Similarity + String Matching

```
1. For each entity pair (e1, e2):
   
   a) String matching:
      - Same after normalization? → Link
      - e1 substring of e2 (>80%)? → Link
   
   b) Semantic similarity (spaCy):
      - Load entity representations
      - Cosine similarity > 0.85? → Link
   
2. Merge all similar entities:
   - Redirect all edges from e2 to e1
   - Keep unified node with merged attributes
```

**Impact**:
- Reduces node count (e.g., 80→70 due to aliases)
- Increases edge count (consolidates paths)
- Improves retrieval (better entity matching with queries)

---

## Stage 5: Retrieval - PASSAGE RANKING

**File**: `src/retrieval/passage_ranking.py`  
**Entry Point**: `rank_passages_by_entities(G, query_entities, triple_to_passages, passage_entities, passage_texts, top_k=10, neighborhood_hops=3)`

### ⚠️ CRITICAL: This is NOT PageRank

**Old Method (PageRank)**: Rank *nodes* by network influence  
**Current Method (PASSAGE RANKING)**: Rank *passages* by entity coverage  

### Passage Ranking Algorithm

#### Step 1: Entity Expansion (BFS)
Starting from query entities, expand to neighboring entities in graph:
```
Query: "Which city was Einstein born in?"
Query Entities: [Albert Einstein]

Hop 0: {Albert Einstein}
  ↓ (outgoing edges)
Hop 1: {Ulm, Princeton}  (born_in, worked_at)
  ↓ (from objects, reverse/forward)
Hop 2: {Germany, New Jersey}  (located_in)
  ↓
Hop 3: {...expand further...}
```

#### Step 2: Hop-Weighted Scoring
Each entity gets score based on minimum hop distance from query:
```
score(entity) = decay^min_hops
  where decay = 1.0 for hop 0
        decay = 0.85 for hop 1
        decay = 0.7 for hop 2
        ...

Albert Einstein: 1.0  (hop 0, query entity)
Ulm: 0.85  (hop 1, directly related)
Germany: 0.7  (hop 2, indirectly related)
```

#### Step 3: Passage Scoring
For each passage, sum scores of entities it contains:
```
Passage 1 entities: {Albert Einstein, Ulm, Germany}
score(Passage 1) = 1.0 + 0.85 + 0.7 = 2.55

Passage 2 entities: {Einstein, Princeton}
score(Passage 2) = 1.0 + 0.85 = 1.85
```

#### Step 4: Ranking
Rank passages by score, return top-k:
```
rank_passages_by_entities(..., top_k=10)
  ↓
[
  (passage_id=1, text="Einstein was born in Ulm...", score=2.55),
  (passage_id=2, text="Einstein worked at Princeton...", score=1.85),
  (passage_id=5, text="...", score=1.4),
  ...
]
```

### Data Structures

**triple_to_passages**: Built before ranking  
```python
{
    ("albert einstein", "born_in", "ulm"): [passage_1, passage_3, passage_7],
    ("ulm", "located_in", "germany"): [passage_1, passage_2],
    ...
}
```

**passage_texts**:
```python
{
    0: "Albert Einstein was born in Ulm, Germany...",
    1: "Einstein worked at Princeton University...",
    ...
}
```

**passage_entities**: Maps passage to entities it contains  
```python
{
    0: ["Albert Einstein", "Ulm", "Germany"],
    1: ["Einstein", "Princeton"],
    ...
}
```

**entity_to_passages**: Cache for fast entity lookup  
```python
{
    "albert einstein": {0, 1, 5, ...},
    "ulm": {0, 3, 7},
    ...
}
```

### Why Passage Ranking?

1. **Task-Aligned**: Multi-hop QA needs passages (context), not nodes
2. **Entity Coverage**: Passages with more query-relevant entities rank higher
3. **Semantic Proximity**: Hop distance encodes indirect relevance
4. **Scalable**: Faster than computing PageRank on large graphs
5. **Interpretable**: Easy to explain which entities matched query

---

## Stage 6: Evaluation

**File**: `test_rebuild_200.py`  
**Evaluation Metrics**: Hit@1, Hit@5, Hit@10

### Process

#### 6.1 Query Entity Extraction
From question, extract entities present in graph:
```python
question = "Which city was the actor who played Hamlet born in?"
  ↓ [spaCy NER + graph matching]
query_entities = ["actor", "Hamlet"]  # or actual actor/character names if in graph
```

#### 6.2 Passage Ranking
Rank passages using Stage 5 algorithm:
```python
ranked_passages = rank_passages_by_entities(
    G, 
    query_entities, 
    triple_to_passages, 
    passage_entities, 
    passage_texts,
    top_k=20,               # Get top 20 for Hit@10 evaluation
    neighborhood_hops=3
)
```

#### 6.3 Hit@k Evaluation
Check if gold answer appears in top-k passages:
```
ranked_passages (top 20):
  [
    (passage_id=1, text="... born in Berlin ...", score=2.5),
    (passage_id=3, text="... worked in Munich ...", score=1.8),
    ...
  ]

gold_answer = "Berlin"

Hit@1: Is answer in top-1 passages? 
  → Check passage_1 for "Berlin" → YES ✓ (Hit)

Hit@5: Is answer in top-5 passages?
  → Check passages 1-5 for "Berlin" → YES ✓ (Hit)

Hit@10: Is answer in top-10 passages?
  → ... → YES ✓ (Hit)
```

### Metrics Interpretation

- **Hit@1**: How many questions answered from single most relevant passage (10% threshold)
- **Hit@5**: Passage appears in top 5 (50% threshold - good for retrieval)
- **Hit@10**: Passage appears in top 10 (100% threshold - comprehensive check)

**Coverage %**: (num_hits / total_samples) × 100

**Example Results**:
```
Coverage: 5/10 (50.0%)
Hit@1: 1/10 (10.0%)
Hit@5: 3/10 (30.0%)
Hit@10: 5/10 (50.0%)
Skipped: 0 (no query entities found in graph)
```

---

## Configuration Parameters

### Data Loading
| Parameter | Value | Note |
|-----------|-------|------|
| Split | "dev" | MusiQue dev set |
| Max Samples | 200 | Subset for quick testing |
| Max Passages | ~40 | Extracted from samples |

### Extraction
| Parameter | Value | Note |
|-----------|-------|------|
| use_dynamic | True | Use schema grouping |
| k | 8 | Top 8 relations per passage |
| threshold | 0.5 | Min similarity for relation inclusion |
| batch_size | 32 | GLiNER2 internal batch size |
| skip_cache | True | Force recomputation |
| deduplicate | False | Keep per-passage data |
| save_cache_to_disk | False | Don't persist to file |

### Graph Building
| Parameter | Value | Note |
|-----------|-------|------|
| add_reverse_edges | False | Don't add reverse relations |
| add_cooccurrence_edges | True | Add same-passage cooccurrence |

### Entity Linking
| Parameter | Value | Note |
|-----------|-------|------|
| similarity_threshold | 0.85 | spaCy semantic similarity cutoff |
| substring_threshold | 0.8 | String overlap threshold |

### Passage Ranking
| Parameter | Value | Note |
|-----------|-------|------|
| top_k | 20 | Retrieve top 20 passages |
| neighborhood_hops | 3 | Expand to 3-hop neighbors |
| decay | [1.0, 0.85, 0.7, ...] | Hop-based score decay |



---

## Code Entry Points

### Quick Test (CURRENT)
```bash
python test_full_pipeline.py
```

Runs complete end-to-end pipeline on 5 QA pairs (20 passages):

**Expected Output**:
```
================================================================================
FULL PIPELINE TEST: Data Loading -> Extraction -> Graph Building -> Evaluation
================================================================================

STEP 1: Loading MusiQue dataset...
  Loaded 5 QA pairs from dev set
  - Unique passages: 20
  - Query-passage mappings: 5
  - Time: 0.00s

STEP 2: Extracting triples from passages...
  Extracted 36 triples from 20 passages
  - Multi-relation extraction: 12 unique schemas (vs 20 passages)
  - Avg: 1.80 triples/passage
  - Time: 10.71s

STEP 3: Building knowledge graphs per passage...
  Built 20 graphs
  - Time: 0.00s

STEP 4: Merging all graphs...
  Merged into single graph
  - Nodes: 44
  - Edges: 62
  - Time: 0.00s

STEP 5: PageRank-based retrieval on sample queries...
  Completed retrieval for 5 sample queries
  - Time: 0.01s

Timing Breakdown:
  Loading:         0.01s ( 0.1%)
  Extraction:     10.71s (99.8%)
  Build graphs:    0.00s ( 0.0%)
  Merge:           0.00s ( 0.0%)
  Retrieval:       0.01s ( 0.1%)
  ────────────────────────────────
  TOTAL:          10.73s

Retrieval Results:
  - Gold passages found: 0/10 (0%)  [known issue: retrieval bug]

Extrapolated timing (for ~22316 passages):
  Estimated time: 199.5 minutes (3.33 hours)

================================================================================
FULL PIPELINE TEST COMPLETE
================================================================================

Results saved to: full_pipeline_results_20260404_204547.json
```

**Customizing the Test**:
```python
# Edit test_full_pipeline.py:
DATA_LIMIT = 200  # Change to test more samples (default: 5)
SAMPLE_QUERIES = 5  # Change number of queries to evaluate

# Then run:
python test_full_pipeline.py
```

### Custom Pipeline
```python
from src.data.musique_loader import load_musique, get_all_passages
from src.extraction import extract_triples_batch
from src.graph.build_graph import build_graph, merge_graphs

# 1. Load data
samples = load_musique("dev", max_samples=200)
passages = get_all_passages(samples)

# 2. Extract triples
results = extract_triples_batch(
    passages, 
    use_dynamic=True, 
    skip_cache=True
)

# 3. Build graphs
graphs = [
    build_graph(r['triples'], r['entities'])
    for r in results if r['triples']
]
merged_graph = merge_graphs(graphs)

# 4. Rank & evaluate
ranked = rank_passages_by_entities(
    merged_graph, 
    query_entities,
    triple_to_passages,
    passage_entities,
    passage_texts,
    top_k=10
)
```



---

## Debugging

### Check Extraction Quality
```python
results = extract_triples_batch(passages, use_dynamic=True)
triple_counts = [len(r['triples']) for r in results]
print(f"Mean triples per passage: {sum(triple_counts) / len(triple_counts)}")
print(f"Total triples: {sum(triple_counts)}")
```
- Expected: 5-10 per passage
- Actual: 1.4 (too low)

### Check Graph Size
```python
print(f"Nodes: {len(merged_graph.nodes())}")
print(f"Edges: {len(merged_graph.edges())}")
print(f"Nodes per passage: {len(merged_graph.nodes()) / num_passages}")
```
- Expected: 3-5 nodes per passage
- Actual: ~2 (sparse)

### Check Passage Ranking
```python
ranked = rank_passages_by_entities(..., top_k=10)
for passage_id, text, score in ranked[:5]:
    print(f"{score:.2f}: {text[:50]}...")
```
