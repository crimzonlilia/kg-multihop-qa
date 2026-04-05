"""
Evaluate quality of extracted triples
- Check semantic validity (subject, predicate, object not empty/garbage)
- Check for duplicates and malformed entries
- Analyze relation distribution
- Verify coverage of passages
- Find potential noise patterns
"""

import os
from collections import Counter, defaultdict
from datetime import datetime
import re

from src.cache_utils import load_triples_cache


def load_triples(cache_path=None, cache_name=None):
    """Load passages with triples from a path or cache alias like 'full'/'300'."""
    cache_arg = str(cache_path) if cache_path is not None else ""
    if cache_name is None and cache_arg and not cache_arg.endswith(".json") and "/" not in cache_arg and "\\" not in cache_arg:
        cache_name = cache_arg
        cache_path = None

    if cache_path is None and cache_name is None:
        cache_name = os.getenv("TRIPLES_CACHE_NAME", "full")

    passages, _, resolved_path = load_triples_cache(cache_path=cache_path, cache_name=cache_name)
    print(f"📄 Using triples cache: {resolved_path}")
    return passages

def extract_all_triples(passages):
    """Extract all triples from passages"""
    all_triples = []
    triple_set = set()  # For dedup counting
    
    for passage_id, passage_data in enumerate(passages):
        triples = passage_data.get('triples', [])
        for triple in triples:
            if isinstance(triple, dict):
                s = triple.get('subject', '').strip()
                p = triple.get('relation', '').strip()  # FIX: use 'relation' not 'predicate'
                o = triple.get('object', '').strip()
                
                all_triples.append({
                    'passage_id': passage_id,
                    'subject': s,
                    'predicate': p,
                    'object': o,
                    'original': triple
                })
                
                triple_tuple = (s.lower(), p.lower(), o.lower())
                triple_set.add(triple_tuple)
    
    return all_triples, triple_set

def check_triple_validity(triple):
    """Check if triple is valid (not garbage/empty)"""
    s, p, o = triple['subject'], triple['predicate'], triple['object']
    
    issues = []
    
    # Check empty
    if not s or not p or not o:
        issues.append("EMPTY_FIELD")
    
    # Check too short (likely garbage)
    if len(s) < 2 or len(p) < 2 or len(o) < 2:
        issues.append("TOO_SHORT")
    
    # Check only special chars
    if not re.search(r'[a-zA-Z0-9]', s):
        issues.append("NO_ALPHANUMERIC_SUBJECT")
    if not re.search(r'[a-zA-Z0-9]', p):
        issues.append("NO_ALPHANUMERIC_PREDICATE")
    if not re.search(r'[a-zA-Z0-9]', o):
        issues.append("NO_ALPHANUMERIC_OBJECT")
    
    # Check suspicious patterns (all numbers = date, time, etc - less useful)
    if s.replace('-', '').replace('/', '').isdigit():
        issues.append("SUBJECT_NUMERIC")
    if o.replace('-', '').replace('/', '').isdigit():
        issues.append("OBJECT_NUMERIC")
    
    # Check punctuation-only
    if not any(c.isalnum() for c in s):
        issues.append("SUBJECT_NO_ALPHANUM")
    if not any(c.isalnum() for c in p):
        issues.append("PREDICATE_NO_ALPHANUM")
    if not any(c.isalnum() for c in o):
        issues.append("OBJECT_NO_ALPHANUM")
    
    return issues

def analyze_triples_quality(passages):
    """Comprehensive quality analysis"""
    print("\n" + "="*70)
    print("TRIPLES EXTRACTION QUALITY EVALUATION")
    print("="*70 + "\n")
    
    all_triples, unique_triples = extract_all_triples(passages)
    
    # Basic stats
    print(f"Total passages: {len(passages)}")
    print(f"Total triples (raw): {len(all_triples)}")
    print(f"Unique triples (lowercased): {len(unique_triples)}")
    print(f"Deduplication ratio: {len(all_triples) / len(unique_triples):.2f}x")
    
    # Validity check
    print(f"\n{'─'*70}")
    print("VALIDITY CHECK")
    print(f"{'─'*70}")
    
    issue_counts = Counter()
    valid_count = 0
    invalid_triples = []
    
    for triple in all_triples:
        issues = check_triple_validity(triple)
        if not issues:
            valid_count += 1
        else:
            for issue in issues:
                issue_counts[issue] += 1
            invalid_triples.append((triple, issues))
    
    print(f"\n✓ Valid triples: {valid_count}/{len(all_triples)} ({100*valid_count/len(all_triples):.1f}%)")
    print(f"✗ Invalid triples: {len(all_triples) - valid_count}/{len(all_triples)} ({100*(len(all_triples) - valid_count)/len(all_triples):.1f}%)")
    
    print(f"\nTop issues:")
    for issue, count in issue_counts.most_common(10):
        print(f"  - {issue}: {count} ({100*count/len(all_triples):.1f}%)")
    
    # Relation analysis
    print(f"\n{'─'*70}")
    print("RELATION ANALYSIS")
    print(f"{'─'*70}")
    
    relation_counts = Counter()
    for triple in all_triples:
        relation_counts[triple['predicate'].lower()] += 1
    
    print(f"\nUnique relations: {len(relation_counts)}")
    print(f"\nTop 20 relations:")
    for rel, count in relation_counts.most_common(20):
        print(f"  - {rel}: {count}")
    
    # Subject/Object analysis
    print(f"\n{'─'*70}")
    print("ENTITY ANALYSIS")
    print(f"{'─'*70}")
    
    subject_counts = Counter()
    object_counts = Counter()
    for triple in all_triples:
        subject_counts[triple['subject'].lower()] += 1
        object_counts[triple['object'].lower()] += 1
    
    print(f"\nUnique subjects: {len(subject_counts)}")
    print(f"Unique objects: {len(object_counts)}")
    
    # Average triple appearances
    avg_subject_freq = sum(subject_counts.values()) / len(subject_counts)
    avg_object_freq = sum(object_counts.values()) / len(object_counts)
    print(f"Average subject frequency: {avg_subject_freq:.2f}")
    print(f"Average object frequency: {avg_object_freq:.2f}")
    
    print(f"\nMost common subjects (top 10):")
    for subj, count in subject_counts.most_common(10):
        print(f"  - {subj}: {count}")
    
    print(f"\nMost common objects (top 10):")
    for obj, count in object_counts.most_common(10):
        print(f"  - {obj}: {count}")
    
    # Passage coverage
    print(f"\n{'─'*70}")
    print("PASSAGE COVERAGE")
    print(f"{'─'*70}")
    
    passages_with_triples = len([p for p in passages if p.get('triples')])
    passages_without_triples = len(passages) - passages_with_triples
    avg_triples_per_passage = len(all_triples) / len(passages)
    
    print(f"\nPassages with triples: {passages_with_triples}/{len(passages)} ({100*passages_with_triples/len(passages):.1f}%)")
    print(f"Passages without triples: {passages_without_triples}/{len(passages)}")
    print(f"Average triples per passage: {avg_triples_per_passage:.2f}")
    
    # Sample invalid triples
    if invalid_triples:
        print(f"\n{'─'*70}")
        print("SAMPLE INVALID TRIPLES (first 10)")
        print(f"{'─'*70}\n")
        
        for i, (triple, issues) in enumerate(invalid_triples[:10], 1):
            print(f"{i}. Issues: {', '.join(issues)}")
            print(f"   S: '{triple['subject']}'")
            print(f"   P: '{triple['predicate']}'")
            print(f"   O: '{triple['object']}'")
            print()
    
    # Sample valid triples
    print(f"{'─'*70}")
    print("SAMPLE VALID TRIPLES (first 10)")
    print(f"{'─'*70}\n")
    
    valid_samples = [t for t in all_triples if not check_triple_validity(t)][:10]
    for i, triple in enumerate(valid_samples, 1):
        print(f"{i}. S: '{triple['subject']}'")
        print(f"   P: '{triple['predicate']}'")
        print(f"   O: '{triple['object']}'")
        print()
    
    # Quality score
    print(f"{'─'*70}")
    print("OVERALL QUALITY SCORE")
    print(f"{'─'*70}\n")
    
    validity_score = 100 * valid_count / len(all_triples)
    dedup_score = 100 * len(unique_triples) / len(all_triples)
    coverage_score = 100 * passages_with_triples / len(passages)
    
    overall_score = (validity_score * 0.5 + dedup_score * 0.3 + coverage_score * 0.2)
    
    print(f"Validity score:        {validity_score:.1f}%  ({'✓ GOOD' if validity_score > 80 else '⚠ FAIR' if validity_score > 60 else '✗ POOR'})")
    print(f"Deduplication score:   {dedup_score:.1f}%  ({'✓ GOOD' if dedup_score > 90 else '⚠ FAIR' if dedup_score > 70 else '✗ POOR'})")
    print(f"Coverage score:        {coverage_score:.1f}% ({'✓ GOOD' if coverage_score > 90 else '⚠ FAIR' if coverage_score > 70 else '✗ POOR'})")
    print(f"\n🎯 OVERALL QUALITY:    {overall_score:.1f}/100 ({'✓ EXCELLENT' if overall_score > 75 else '⚠ ACCEPTABLE' if overall_score > 60 else '✗ POOR'})")
    
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    passages = load_triples()
    analyze_triples_quality(passages)
