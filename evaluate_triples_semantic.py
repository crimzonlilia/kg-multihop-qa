"""
Evaluate semantic quality of extracted triples
- Check if triple entities appear in passage text
- Check if triple is meaningless (all generic terms)
- Find noise patterns (common vague subjects/predicates)
- Suggest filtering rules
"""

import json
from collections import Counter, defaultdict
from datetime import datetime
import re

def load_triples_with_passages(cache_path="data/cache/triples.json"):
    """Load passages with triples"""
    with open(cache_path, 'r', encoding='utf-8') as f:
        passages = json.load(f)
    return passages

# Generic/vague terms that don't add info
GENERIC_SUBJECTS = {
    'he', 'she', 'it', 'they', 'the', 'a', 'an', 'this', 'that', 'these', 'those',
    'one', 'some', 'any', 'each', 'every', 'other', 'another', 'who', 'what', 'which',
    'person', 'people', 'thing', 'group', 'type', 'kind', 'sort', 'example', 'case',
    'work', 'project', 'effort', 'way', 'method', 'process', 'system', 'order',
    'part', 'section', 'area', 'region', 'place', 'location', 'site', 'point',
    'time', 'period', 'date', 'year', 'month', 'day', 'hour', 'moment', 'season'
}

GENERIC_RELATIONS = {
    '', 'is', 'are', 'be', 'have', 'has', 'do', 'does', 'make', 'made', 'make', 'was', 'were',
    'has', 'have', 'etc', 'and', 'or', 'but', 'also', 'as', 'at', 'in', 'on', 'by', 'with',
    'to', 'from', 'of', 'for', 'about', 'through', 'during', 'before', 'after', 'under',
    'over', 'above', 'below', 'between', 'among', 'around', 'near', 'like', 'other',
    'unknown', 'related', 'reference', 'link', 'connection', 'mention', 'said', 'told'
}

GENERIC_OBJECTS = {
    'he', 'she', 'it', 'they', 'the', 'a', 'an', 'this', 'that', 'these', 'those',
    'one', 'yes', 'no', 'true', 'false', 'maybe', 'perhaps', 'possibly', 'probably',
    'more', 'less', 'other', 'another', 'different', 'same', 'similar', 'different',
    'thing', 'stuff', 'things', 'people', 'persons', 'example', 'etc', 'information'
}

def normalize_text(text):
    """Normalize text for matching"""
    return re.sub(r'\s+', ' ', text.lower().strip())

def extract_named_entities_simple(text):
    """Extract capitalized words (simple NER)"""
    # Match capitalized words
    entities = set()
    for match in re.finditer(r'\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\b', text):
        entity = match.group(1).lower()
        if len(entity) > 2:  # Skip short entities
            entities.add(entity)
    return entities

def check_triple_semantic_quality(triple, passage_text, passage_entities):
    """
    Check semantic quality of a triple
    Returns (quality_score, issues)
    """
    s = triple.get('subject', '').lower().strip()
    p = triple.get('relation', '').lower().strip()
    o = triple.get('object', '').lower().strip()
    
    issues = []
    quality_score = 1.0
    
    # Check if subject/object in passage
    passage_lower = normalize_text(passage_text)
    entities_in_passage = extract_named_entities_simple(passage_text)
    
    # Score based on issues
    
    # 1. Generic subject (low info)
    if s in GENERIC_SUBJECTS:
        issues.append("GENERIC_SUBJECT")
        quality_score *= 0.5
    
    # 2. Generic relation (weak semantic)
    if p in GENERIC_RELATIONS or not p:
        issues.append("GENERIC_RELATION")
        quality_score *= 0.4
    
    # 3. Generic object (low semantic)
    if o in GENERIC_OBJECTS:
        issues.append("GENERIC_OBJECT")
        quality_score *= 0.6
    
    # 4. Very short (vague or typo)
    if len(s) < 3 or len(p) < 3 or len(o) < 3:
        issues.append("TOO_SHORT")
        quality_score *= 0.3
    
    # 5. Subject not in passage (mismatch)
    if s not in passage_lower and s not in str(entities_in_passage):
        issues.append("SUBJECT_NOT_IN_PASSAGE")
        quality_score *= 0.6
    
    # 6. Object not in passage (mismatch)
    if o not in passage_lower and o not in str(entities_in_passage):
        issues.append("OBJECT_NOT_IN_PASSAGE")
        quality_score *= 0.6
    
    # 7. All numeric (dates/numbers, less useful for semantic)
    if s.replace('-', '').isdigit() or o.replace('-', '').isdigit():
        issues.append("NUMERIC_FIELD")
        quality_score *= 0.7
    
    # 8. Very long (probably extraction error)
    if len(s) > 100 or len(p) > 100 or len(o) > 100:
        issues.append("TOO_LONG")
        quality_score *= 0.3
    
    # 9. Special characters only
    if not any(c.isalnum() for c in s) or not any(c.isalnum() for c in p) or not any(c.isalnum() for c in o):
        issues.append("SPECIAL_CHARS_ONLY")
        quality_score *= 0.1
    
    # 10. Suspicious patterns (e.g., "is the" as relation)
    if re.search(r'is\s+the|the\s+is|are\s+the|was\s+the', f"{s} {p} {o}"):
        issues.append("SUSPICIOUS_PATTERN")
        quality_score *= 0.3
    
    return max(0.0, quality_score), issues

def analyze_semantic_quality(passages):
    """Comprehensive semantic quality analysis"""
    print("\n" + "="*80)
    print("SEMANTIC QUALITY EVALUATION OF EXTRACTED TRIPLES")
    print("="*80 + "\n")
    
    total_triples = 0
    high_quality = 0
    medium_quality = 0
    low_quality = 0
    noise_triples = []
    
    issue_frequency = Counter()
    subject_issues = Counter()
    predicate_issues = Counter()
    
    for passage_id, passage_data in enumerate(passages):
        passage_text = passage_data.get('passage', '')
        entities = passage_data.get('entities', [])
        passage_entities = set(e.get('text', '').lower() for e in entities if 'text' in e)
        
        triples = passage_data.get('triples', [])
        for triple in triples:
            if isinstance(triple, dict) and 'subject' in triple:
                total_triples += 1
                
                quality_score, issues = check_triple_semantic_quality(
                    triple, passage_text, passage_entities
                )
                
                # Categorize
                if quality_score >= 0.8:
                    high_quality += 1
                elif quality_score >= 0.4:
                    medium_quality += 1
                else:
                    low_quality += 1
                
                # Track issues
                for issue in issues:
                    issue_frequency[issue] += 1
                
                # Track problematic subjects/predicates
                if quality_score < 0.5:
                    noise_triples.append({
                        'passage_id': passage_id,
                        'triple': triple,
                        'quality_score': quality_score,
                        'issues': issues,
                        'passage': passage_text[:100]
                    })
                    
                    if 'GENERIC_SUBJECT' in issues or 'SUBJECT_NOT_IN_PASSAGE' in issues:
                        subject_issues[triple.get('subject', 'UNKNOWN')] += 1
                    if 'GENERIC_RELATION' in issues or 'SUSPICIOUS_PATTERN' in issues:
                        predicate_issues[triple.get('relation', 'UNKNOWN')] += 1
    
    # Print summary
    print(f"Total triples analyzed: {total_triples}")
    print()
    print(f"Quality Distribution:")
    print(f"  ✓ High quality (≥0.8):   {high_quality:5d} ({100*high_quality/total_triples:.1f}%)")
    print(f"  ~ Medium quality (0.4-0.8): {medium_quality:5d} ({100*medium_quality/total_triples:.1f}%)")
    print(f"  ✗ Low quality (<0.4):    {low_quality:5d} ({100*low_quality/total_triples:.1f}%)")
    print()
    
    # Most common issues
    print(f"{'─'*80}")
    print("MOST COMMON QUALITY ISSUES")
    print(f"{'─'*80}\n")
    
    for issue, count in issue_frequency.most_common(15):
        print(f"  {issue:30s}: {count:5d} ({100*count/total_triples:.1f}%)")
    
    # Most problematic subjects
    print(f"\n{'─'*80}")
    print("MOST PROBLEMATIC SUBJECTS (low-quality triples)")
    print(f"{'─'*80}\n")
    
    for subject, count in subject_issues.most_common(15):
        print(f"  '{subject}': {count} triples")
    
    # Most problematic predicates
    print(f"\n{'─'*80}")
    print("MOST PROBLEMATIC PREDICATES (weak semantic)")
    print(f"{'─'*80}\n")
    
    for predicate, count in predicate_issues.most_common(15):
        print(f"  '{predicate}': {count} triples")
    
    # Sample noise triples
    print(f"\n{'─'*80}")
    print("SAMPLE LOW-QUALITY TRIPLES (quality_score < 0.5)")
    print(f"{'─'*80}\n")
    
    for i, noise in enumerate(noise_triples[:20], 1):
        triple = noise['triple']
        print(f"{i}. Score: {noise['quality_score']:.2f} | Issues: {', '.join(noise['issues'])}")
        print(f"   S: '{triple.get('subject', '')}'")
        print(f"   P: '{triple.get('relation', '')}'")
        print(f"   O: '{triple.get('object', '')}'")
        print(f"   Passage: {noise['passage']}...")
        print()
    
    # Filtering recommendations
    print(f"{'─'*80}")
    print("FILTERING RECOMMENDATIONS")
    print(f"{'─'*80}\n")
    
    recommendations = []
    
    # Recommend filtering by issue frequency
    for issue, count in issue_frequency.most_common(5):
        pct = 100 * count / total_triples
        if pct > 5:  # If issue affects >5% of triples
            recommendations.append(f"Filter triples with '{issue}' (affects {pct:.1f}% of data)")
    
    # Recommend filtering generic subjects
    generic_subject_count = sum(c for s, c in subject_issues.items() if s in GENERIC_SUBJECTS)
    if generic_subject_count > 0:
        pct = 100 * generic_subject_count / total_triples
        recommendations.append(f"Filter generic subjects like {tuple(GENERIC_SUBJECTS)[:5]} (affects {pct:.1f}%)")
    
    # Recommend filtering by passage mismatch
    mismatch_count = issue_frequency.get('SUBJECT_NOT_IN_PASSAGE', 0) + issue_frequency.get('OBJECT_NOT_IN_PASSAGE', 0)
    if mismatch_count > 0:
        pct = 100 * mismatch_count / total_triples
        recommendations.append(f"Filter passage-mismatched triples (S/O not in text) - affects {pct:.1f}%")
    
    if recommendations:
        print("To improve triple quality, consider:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    else:
        print("Triple quality looks good - minimal filtering needed!")
    
    # Quality metrics
    print(f"\n{'─'*80}")
    print("QUALITY METRICS")
    print(f"{'─'*80}\n")
    
    avg_quality = (high_quality * 0.9 + medium_quality * 0.6 + low_quality * 0.2) / total_triples * 100
    print(f"Average quality score: {avg_quality:.1f}%")
    print(f"'Noise' ratio (low-quality): {100*low_quality/total_triples:.1f}%")
    print(f"Estimated usable triples: {total_triples - low_quality} ({100*(total_triples-low_quality)/total_triples:.1f}%)")
    
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    passages = load_triples_with_passages()
    analyze_semantic_quality(passages)
