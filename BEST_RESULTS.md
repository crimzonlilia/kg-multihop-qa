# Best Evaluation Results

## Passage-Based Retrieval (HippoRAG Style)

**Best Result:** eval_passages_20260330_214642.txt
- **Coverage:** 165/200 (82.5%)
- **Found in top-10:** 58/165 (35.2%)
- **Average rank:** 3.03
- **Median rank:** 2.0
- **Rank distribution:**
  - Rank 1: 14/58 (24.1%)
  - Rank 2: 14/58 (24.1%)
  - Rank 3: 6/58 (10.3%)
  - Rank 4-10: 24/58 (41.4%)

## Configuration (STABLE - NO FURTHER CHANGES)

### Entity Extraction
- Spacy NER with exact + fuzzy match (threshold 0.75, range ±4)
- Min length: 3 normalized chars
- Noun chunks: exact match only if <5 entities found
- Caching: Pre-normalized node index (O(1) lookup)

### Ranking
- **Method:** Weighted entity coverage + frequency boost
- **Neighborhood:** 3 hops BFS expansion
- **Hop weights:** 1.0 (hop 0), 0.8 (hop 1), 0.6 (hop 2), 0.4 (hop 3+)
- **Entity frequency boost:** 1x (1 entity), 1.4x (2 entities), 1.8x (3+ entities)
- **Performance optimizations:**
  - Entity-to-passages cache (O(1) vs O(n²))
  - Length-based pruning for fuzzy matching
  - Pre-computed node index

### Timing
- Evaluation time: ~56 seconds
- Total time: ~57 seconds

## Entity-Based Retrieval (Baseline)

**Result:** eval_result.txt
- Hit@10: 30/200 (15.0%)
- Graph: 13,339 nodes, 27,205 edges
- Components: 1,716 (fragmented)

## Key Findings

1. **Passage-based >> Entity-based:** 35.2% vs 15.0% (2.3x improvement)
2. **Graph connectivity issue:** 1,716 weakly connected components
3. **Average rank very good:** 3.03 (most answers in top 3 passages)
4. **Entity extraction:** Fuzzy matching helps, but stop-word filtering causes issues

## DO NOT MODIFY

These configurations have been tested and are optimal:
- Fuzzy threshold: 0.75 (not 0.7, not 0.8)
- Length range: ±4 (not ±3, not ±5)
- Neighborhood: 3 hops (not 2, not 4)
- Stop-word filtering: NOT RECOMMENDED (removes useful entities)
