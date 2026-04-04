# 🐛 BUG REPORT: 0% Hit Rate in Multi-Hop QA Evaluation

## Summary
The evaluation system is returning 0% Hit@10 and 0% Supporting Fact Hit. Investigation reveals **multiple critical issues** in the evaluation logic and pipeline.

---

## 🔴 CRITICAL ISSUES FOUND

### Issue #1: **WRONG EVALUATION LOGIC** ⚠️ MOST CRITICAL
**File**: `src/evaluate.py` (lines 113-117)

**Problem**: The evaluation checks if the gold *answer text* matches graph *node names*.
```python
ranked = personalized_pagerank_fast(G, query_entities, top_k=10, neighborhood_hops=3)
ranked_nodes = [normalize(n) for n, _ in ranked]
normalized_gold = normalize(gold)
is_hit = normalized_gold in ranked_nodes  # ← THIS IS WRONG!
```

**Why it fails**:
- Gold answers are free-form text: `"John Smith"`, `"United States"`
- Graph nodes are entity names extracted from passages: `"John Smith (politician)"`, `"United States of America"`
- These almost never match exactly after normalization

**Example**:
- Question: `"Who is the spouse of the Green performer?"`
- Gold answer: `"Alice Black"` (free-form text)
- Retrieved node: `"alice black (spouse)"` (from KG)
- After normalization: `"alice black"` ≠ `"alice black spouse"` → **NO HIT**

**Fix**: Use better matching strategy:
1. Extract named entities from both gold answer and retrieved nodes
2. Compare entities instead of full text
3. Allow substring matching for answer phrases found in node names

---

### Issue #2: **WEAK ENTITY EXTRACTION**
**File**: `src/evaluate.py` (lines 24-45)

**Problem**: The `extract_query_entities_from_graph()` function:
```python
try:
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(question)
    entities = [ent.text.lower() for ent in doc.ents]
    matched = [n for n in G.nodes() if any(e in n for e in entities)]
    if matched:
        return sorted(matched, key=len, reverse=True)[:5]
except:  # ← Silently fails!
    pass
```

**Issues**:
- If spacy fails, it silently falls back with `except: pass`
- Spacy NER might not find all entities
- Fallback only looks for capitalized words
- For questions like `"Who is the Green performer?"`, it might not extract `"Green performer"` correctly
- Substring matching (`any(e in n for e in entities)`) is too loose

**Test case that probably fails**:
```
Question: "Who founded the company that distributed the film"
- Expects to find: "Company Name", "Film Title", etc.
- Extracted entities might be: ["Company", "Film"] 
- But these won't match graph names like "MGM Films" or "Universal Pictures"
```

**Fix**: 
- Better entity typing and matching to graph
- Catch spacy errors explicitly
- Better fallback strategy

---

### Issue #3: **INCONSISTENT NORMALIZATION**
**File**: `src/evaluate.py` has TWO different normalize functions:

```python
def normalize_answer(s: str) -> str:  # Line 12 - AGGRESSIVE
    s = s.lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)      # Remove articles
    s = re.sub(r'[^\w\s]', ' ', s)             # Remove punctuation  
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# But in main loop (line 116) uses:
from src.graph.build_graph import normalize  # MINIMAL
def normalize(text: str) -> str:  # Line 45 in build_graph.py
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)           # Only this!
    return text
```

**Problem**: 
- `normalize_answer()` is defined but NEVER USED in main eval loop
- Main loop uses `normalize()` from build_graph which is too minimal
- This inconsistency means answers with punctuation never match

**Example**:
- Answer: `"O'Brien"` 
- Normalized by build_graph: `"o'brien"`
- Graph node: `"obrien"`
- No match!

---

### Issue #4: **MISSING `supporting_facts` DATA**
**File**: `src/evaluate.py` (lines 126-127)

```python
supporting_facts = sample.get("supporting_facts", [])
# ... later ...
supporting_titles = [title for idx, title in supporting_facts]  # Line 126
```

**Problem**:
- MusiQue loader doesn't extract `supporting_facts` as a separate field!
- `load_musique()` only loads: `question`, `answer`, `passages` (with `is_supporting` flag)
- But `supporting_facts` is never computed, so it defaults to `[]`
- Therefore Supporting Fact Hit metric is **ALWAYS 0**

**From musique_loader.py** (lines 26-33):
```python
samples.append({
    "id": item["id"],
    "question": item["question"],
    "answer": item["answer"],
    "answerable": item["answerable"],
    "passages": passages,      # ← Has is_supporting flag
    "hops": len(item["question_decomposition"]),
    "decomposition": item.get("question_decomposition", []),
    # ← NO supporting_facts field!
})
```

**Fix**: Extract supporting fact titles:
```python
supporting_facts = [
    (i, p["title"]) 
    for i, p in enumerate(item["paragraphs"]) 
    if p["is_supporting"]
]
samples.append({
    ...
    "supporting_facts": supporting_facts,
})
```

---

### Issue #5: **GRAPH MIGHT BE EMPTY OR POORLY BUILT**
**File**: Various

**Possible causes**:
- If `data/processed/kg.pkl` doesn't exist, graph has 0 nodes
- If triple extraction fails, graph construction fails
- If entities aren't being extracted properly, no edges in graph

**Evidence from results**:
- 55% coverage (110/200) → 90 samples skipped due to "no query entities"
- 0% hit rate → Either no retrieval results or all wrong

**Hypothesis**: With 45% of samples failing to extract query entities, the graph might have very few nodes or poor entity matching.

---

## 🔧 IMMEDIATE FIXES NEEDED

### Fix 1: Use correct normalization consistently
```python
# Use normalize_answer() everywhere, not normalize()
```

### Fix 2: Fix entity extraction matching
```python
def extract_query_entities_improved(G, question):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(question)
    
    matched = []
    for ent in doc.ents:
        ent_norm = normalize_answer(ent.text)
        # Try exact match first
        for node in G.nodes():
            if normalize_answer(node) == ent_norm:
                matched.append(node)
                break
        # Fallback: substring or loose match
        else:
            for node in G.nodes():
                if ent_norm in normalize_answer(node):
                    matched.append(node)
                    break
    return list(set(matched))
```

### Fix 3: Fix answer matching in evaluation
```python
def check_answer_in_retrieved(gold, retrieved_nodes):
    gold_norm = normalize_answer(gold)
    for node in retrieved_nodes:
        node_norm = normalize_answer(node)
        # Exact match
        if gold_norm == node_norm:
            return True
        # Substring match (answer is part of node name)
        if gold_norm in node_norm or node_norm in gold_norm:
            return True
    return False

# In main loop:
is_hit = check_answer_in_retrieved(gold, ranked_nodes)
```

### Fix 4: Add supporting_facts to data loader
```python
# In musique_loader.py
supporting_facts = [
    (i, p.get("title", ""))
    for i, p in enumerate(item["paragraphs"])
    if p["is_supporting"]
]

samples.append({
    ...
    "supporting_facts": supporting_facts,
})
```

### Fix 5: Debug graph construction
```python
print(f"Graph nodes: {G.number_of_nodes()}")
print(f"Sample nodes: {list(G.nodes())[:20]}")
# Check if graph is actually built
```

---

## 📊 Expected Results After Fixes

With proper evaluation logic:
- **Hit@10**: Should increase from **0%** to **~30-50%** range (realistic for unsupervised retrieval)
- **Supporting Fact Hit**: Should show non-zero values
- **Coverage**: Should improve as entity extraction improves

---

## 📝 Files Created

1. **evaluate_fixed.py** - Fixed evaluation script with:
   - Better entity extraction
   - Correct normalization
   - Proper answer matching
   - Entity-based comparison

2. **debug_pipeline.py** - Debug script to verify:
   - Graph loading
   - Entity extraction
   - Retrieval results
   - Normalization consistency

---

## ✅ Recommended Action

Run: `src/evaluate_fixed.py` to get accurate results and identify remaining issues
