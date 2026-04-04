# 🎯 Action Plan: Fixing 0% Hit@10 in Multi-Hop QA

## 📋 Summary of Issues & Fixes

### What Was Wrong:
1. ❌ **Schema too limited** - Missing relations like `spouse`, `distributed_by`, `directed_by`
2. ❌ **Wrong evaluation logic** - Comparing answer text to node names directly
3. ❌ **Weak entity extraction** - Failing silently, returning wrong entities
4. ❌ **Graph has generic type nodes** - Diluting relevant entity retrieval

### What's Fixed:
1. ✅ **Schema expanded** from 11 → 25 relations (includes all needed relations)
2. ✅ **Evaluation logic improved** - Better answer matching in `evaluate_fixed.py`
3. ✅ **Error handling added** - Better robustness
4. ✅ **Type constraints made permissive** - Some relations can connect any entity types

---

## 🚀 Action Steps (IN ORDER)

### Step 1: Verify Schema Changes (2 min)
```bash
python test_extraction.py
```
Should show: ✓ Needed relations now in schema

---

### Step 2: CRITICAL - Rebuild Graph (1-2 min for quick test!)
**WHY**: Current graph (kg.pkl) was built with OLD schema that didn't have spouse/distributed_by relations

```bash
python rebuild_graph.py
```

**Quick test mode** (DEFAULT - 1-2 min):
- Rebuilds with 200 samples
- Good for testing if fixes work
- Change `QUICK_TEST = False` in rebuild_graph.py to do full dataset (15-30 min)

**What it does**:
- Loads 200 passages from MusiQue dev set (quick test) or ALL passages (full)
- Extracts triples using NEW schema (with 25 relations instead of 11)
- Builds new graph
- Backs up old graph to `kg.pkl.backup`

**Expected output**:
```
Loaded 200 samples, X unique passages
Extracted Y triples from Z entities
✓ Found new relations: ['spouse', 'distributed_by', 'directed_by', ...]
Graph Statistics:
   Nodes: ~1000-2000 (quick) or 5000-10000 (full)
   Edges: ~2000-5000 (quick) or 10000-20000 (full)
```

⏱️ **TIME**: 
- Quick test (200 samples): ~1-2 min
- Full build (all samples): ~15-30 min

---

### Step 3: Test Improved Evaluation (2 min)
```bash
python src/evaluate_fixed.py
```

**What to look for**:
- Should see "Query entities found" for all samples
- Top-10 results should include entity nodes, not just types
- Check if any "✓" marks appear (means answer was found)

**Expected output**:
```
DEBUG: First 3 samples
======================================================================

Sample 1: Who is the spouse of the Green performer?...
Expected answer: Miquette Giraudy
Query entities found: ['peter green', ...]
Top-10 results:
  1. [ ] peter green (0.xxx)
  2. [✓] miquette giraudy (spouse) ...   ← IF EXTRACTION WORKED
  ...

======================================================================
OVERALL RESULTS
======================================================================
Coverage (evaluated): 110+/200 (55%+)
Hit@10 (FIXED): X/110 (X%)  ← Should be > 0%!
```

---

### Step 4: If Still 0%, Debug
If Hit@10 is still 0%, run:
```bash
python debug_pipeline.py
```

This will show:
- ✓ Graph loaded correctly
- ✓ Entities being extracted
- ✓ Retrieval results
- ✗ Why answers aren't matching

---

## 📊 Expected Results

### Before (Current):
```
Hit@10: 0/110 (0%)
Coverage: 55%
Skipped: 90 (no entities)
```

### After Fixes (Optimistic):
If GLiNER properly extracts new relations:
```
Hit@10: 30-50/110 (27-45%)
Coverage: 75%+
Skipped: <50
```

### After Fixes (Pessimistic):
If GLiNER struggles:
```
Hit@10: 5-15/110 (5-15%)
Coverage: 65%
Skipped: 70
```

---

## 📁 Files Modified

| File | What | Why |
|------|------|-----|
| `src/schemas/relations.json` | Added 14 relations | Schema too limited |
| `src/extraction/extract_triples.py` | Updated schemas + type constraints | Extract new relations |
| `src/extraction/relation_discovery.py` | Updated discovery schema | Support dynamic discovery |
| `src/evaluate_fixed.py` | Fixed answer matching | Evaluation logic was wrong |
| `rebuild_graph.py` | NEW - Rebuild tool | Need new schema in graph |
| `debug_pipeline.py` | NEW - Debug tool | For troubleshooting |
| `FIXES_APPLIED.md` | NEW - Detailed analysis | Technical reference |

---

## ⚙️ How the Fix Works

### Before:
```
Question: "Who is spouse of Green performer?"
  ↓ (Schema only has 11 relations, NO "spouse")
  ↓ Can't extract spouse relationship
  ↓ Graph has no path from "Peter Green" → "Miquette Giraudy"
  ↓ Retrieval returns random nodes
  ↓ Hit@10 = 0%
```

### After:
```
Question: "Who is spouse of Green performer?"
  ↓ (Schema has 25 relations, includes "spouse")
  ↓ GLiNER extracts: "Peter Green" --spouse--> "Miquette Giraudy"
  ↓ Graph has the correct edge
  ↓ Retrieval finds both Peter Green AND Miquette Giraudy
  ✓ Hit@10 should be > 0%
```

---

## 💡 Key Points

### What Will Improve:
- ✅ Multi-hop questions with family relations (spouse, parent, sibling)
- ✅ Questions about media (directed_by, distributed_by, written_by)
- ✅ Questions about ownership and location

### What Still Needs Work:
- ⚠️ Entity linking (currently substring matching - imperfect)
- ⚠️ Natural language question understanding
- ⚠️ Multi-hop reasoning (2+ hops is still hard)
- ⚠️ Answer extraction accuracy

### Troubleshooting:

**If Hit@10 still 0%:**
1. Check graph nodes: `python debug_pipeline.py` → "Graph Nodes" section
2. Check if needed relations were extracted
3. Verify passages contain the information
4. Check entity extraction quality

**If Coverage drops:**
1. Entity extraction might have issues
2. Run: `python debug_pipeline.py` → "Entity Extraction Check"
3. May need to reinstall spacy: `python -m spacy download en_core_web_sm`

**If performance is worse:**
1. Type constraints might be too strict
2. Restore backup: `mv data/processed/kg.pkl.backup data/processed/kg.pkl`
3. Open an issue

---

## 🔄 Full Testing Workflow

```bash
# 1. Verify schema
python test_extraction.py

# 2. Quick test - rebuild graph (1-2 min, 200 samples)
python rebuild_graph.py

# 3. Test evaluation
python src/evaluate_fixed.py

# 4. If results good, do FULL rebuild (change in rebuild_graph.py)
#    - Edit: QUICK_TEST = False (line 18)
#    - Run: python rebuild_graph.py (15-30 min)
#    - Run: python src/evaluate_fixed.py

# 5. If issues, debug
python debug_pipeline.py

# 6. If broken, restore
mv data/processed/kg.pkl.backup data/processed/kg.pkl
```

---

## ✅ Checklist

- [ ] Run `test_extraction.py` - Schema is updated
- [ ] Run `rebuild_graph.py` - New graph built
- [ ] Run `evaluate_fixed.py` - Evaluation logic fixed
- [ ] Check results - Should be > 0%
- [ ] If 0%, run `debug_pipeline.py` for troubleshooting

---

## 📞 If You Get Stuck

1. **No entities extracted**: Check spacy model installed
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **Graph rebuild takes too long**: Stop it (Ctrl+C) and restore:
   ```bash
   mv data/processed/kg.pkl.backup data/processed/kg.pkl
   ```

3. **Still 0% after rebuild**: Run debug script to see where it breaks:
   ```bash
   python debug_pipeline.py
   ```

4. **Memory issues**: Reduce `max_samples` in rebuild_graph.py

---

**Good luck! The fix should get you from 0% to something measurable.** 🚀
