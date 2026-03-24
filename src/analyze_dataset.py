# src/analyze_dataset.py
import sys, json
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, ".")

from src.data.musique_loader import load_musique

def plot_distributions(samples):
    """Create comprehensive visualizations"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('MusiQue Dataset Analysis', fontsize=16, fontweight='bold')
    
    # 1. Hops Distribution
    hops_dist = Counter(s.get("hops", 0) for s in samples)
    axes[0, 0].bar(hops_dist.keys(), hops_dist.values(), color='steelblue')
    axes[0, 0].set_title('Multi-hop Distribution')
    axes[0, 0].set_xlabel('Number of Hops')
    axes[0, 0].set_ylabel('Count')
    for k, v in hops_dist.items():
        axes[0, 0].text(k, v, str(v), ha='center', va='bottom')
    
    # 2. Question Length Distribution
    q_lengths = [len(s["question"].split()) for s in samples]
    axes[0, 1].hist(q_lengths, bins=30, color='coral', edgecolor='black')
    axes[0, 1].set_title('Question Length Distribution')
    axes[0, 1].set_xlabel('Words')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(np.mean(q_lengths), color='red', linestyle='--', label=f'Mean: {np.mean(q_lengths):.1f}')
    axes[0, 1].legend()
    
    # 3. Answer Length Distribution
    a_lengths = [len(s["answer"].split()) for s in samples]
    axes[0, 2].hist(a_lengths, bins=30, color='lightgreen', edgecolor='black')
    axes[0, 2].set_title('Answer Length Distribution')
    axes[0, 2].set_xlabel('Words')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].axvline(np.mean(a_lengths), color='red', linestyle='--', label=f'Mean: {np.mean(a_lengths):.1f}')
    axes[0, 2].legend()
    
    # 4. Passages per Sample
    passage_counts = [len(s["passages"]) for s in samples]
    axes[1, 0].hist(passage_counts, bins=30, color='lightyellow', edgecolor='black')
    axes[1, 0].set_title('Passages per Sample')
    axes[1, 0].set_xlabel('Count')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].axvline(np.mean(passage_counts), color='red', linestyle='--', label=f'Mean: {np.mean(passage_counts):.1f}')
    axes[1, 0].legend()
    
    # 5. Supporting Facts Distribution
    support_counts = [len(s.get("supporting_facts", [])) if isinstance(s.get("supporting_facts"), list) else 0 
                      for s in samples]
    axes[1, 1].hist(support_counts, bins=30, color='plum', edgecolor='black')
    axes[1, 1].set_title('Supporting Facts Distribution')
    axes[1, 1].set_xlabel('Count')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].axvline(np.mean(support_counts), color='red', linestyle='--', label=f'Mean: {np.mean(support_counts):.1f}')
    axes[1, 1].legend()
    
    # 6. Summary Stats Table
    axes[1, 2].axis('off')
    summary_text = f"""
SUMMARY STATISTICS

Questions:
  Avg length: {np.mean(q_lengths):.1f} words
  Range: {min(q_lengths)}-{max(q_lengths)}

Answers:
  Avg length: {np.mean(a_lengths):.1f} words
  Range: {min(a_lengths)}-{max(a_lengths)}

Passages:
  Avg: {np.mean(passage_counts):.1f}
  Range: {min(passage_counts)}-{max(passage_counts)}

Supporting Facts:
  Avg: {np.mean(support_counts):.1f}
  Range: {min(support_counts)}-{max(support_counts)}
"""
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('data/analysis/dataset_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved to: data/analysis/dataset_analysis.png")
    plt.close()

def analyze_question_complexity(samples):
    """Analyze question complexity patterns"""
    print(f"\n{'='*60}")
    print("QUESTION COMPLEXITY ANALYSIS (for Multi-hop Reasoning)")
    print(f"{'='*60}")
    
    hops_dist = Counter(s.get("hops", 0) for s in samples)
    
    for hop in sorted(hops_dist.keys()):
        hop_samples = [s for s in samples if s.get("hops", 0) == hop]
        q_lens = [len(s["question"].split()) for s in hop_samples]
        a_lens = [len(s["answer"].split()) for s in hop_samples]
        passages = [len(s["passages"]) for s in hop_samples]
        
        print(f"\n{hop}-HOP QUESTIONS ({len(hop_samples)} samples):")
        print(f"  Question length: {np.mean(q_lens):.1f} ± {np.std(q_lens):.1f} words")
        print(f"  Answer length: {np.mean(a_lens):.1f} ± {np.std(a_lens):.1f} words")
        print(f"  Passages provided: {np.mean(passages):.1f} ± {np.std(passages):.1f}")
        
        # Check reasoning chain
        support_counts = [len(s.get("supporting_facts", [])) if isinstance(s.get("supporting_facts"), list) else 0 
                          for s in hop_samples]
        print(f"  Supporting facts: {np.mean(support_counts):.1f} (reasoning chain length)")

def analyze_answerable(samples):
    """Analyze answerable vs unanswerable"""
    print(f"\n{'='*60}")
    print("ANSWERABILITY ANALYSIS")
    print(f"{'='*60}")
    
    answerable = [s for s in samples if s.get("answerable", True)]
    unanswerable = [s for s in samples if not s.get("answerable", True)]
    
    print(f"\nAnswerable: {len(answerable)} ({len(answerable)*100/len(samples):.1f}%)")
    print(f"Unanswerable: {len(unanswerable)} ({len(unanswerable)*100/len(samples):.1f}%)")
    
    if unanswerable:
        print("\nUnanswerable questions characteristics:")
        q_lens = [len(s["question"].split()) for s in unanswerable]
        passages = [len(s["passages"]) for s in unanswerable]
        print(f"  Avg question length: {np.mean(q_lens):.1f} words")
        print(f"  Avg passages provided: {np.mean(passages):.1f}")

def analyze_coverage(samples):
    """Analyze answer coverage in passages"""
    print(f"\n{'='*60}")
    print("ANSWER COVERAGE ANALYSIS (Critical for Graph Construction)")
    print(f"{'='*60}")
    
    answer_in_passages = []
    
    for s in samples:
        answer = s["answer"].lower()
        # passages is already a list of strings, not dicts
        passages_text = " ".join([p.lower() for p in s["passages"]])
        
        found = answer in passages_text
        answer_in_passages.append(found)
    
    coverage_pct = sum(answer_in_passages) * 100 / len(samples)
    print(f"\nAnswers found in passages: {sum(answer_in_passages)}/{len(samples)} ({coverage_pct:.1f}%)")
    print(f"Answers NOT in passages: {len(answer_in_passages) - sum(answer_in_passages)} ({100-coverage_pct:.1f}%)")
    print("\n⚠️  Critical: This explains why graph Hit@10 is limited!")
    print(f"    Only {coverage_pct:.1f}% of answers can be extracted from provided passages.")
    print(f"    With KG containing only these passages, max achievable Hit@10 ≈ {coverage_pct:.1f}%")

def analyze_passage_quality(samples):
    """Analyze passage diversity"""
    print(f"\n{'='*60}")
    print("PASSAGE QUALITY ANALYSIS")
    print(f"{'='*60}")
    
    total_passages = sum(len(s["passages"]) for s in samples)
    unique_passages = len(set(p for s in samples for p in s["passages"]))
    
    print(f"\nTotal passage pairs: {total_passages}")
    print(f"Unique passages: {unique_passages}")
    print(f"Duplication ratio: {(1 - unique_passages/total_passages)*100:.1f}%")
    
    # Average passage length
    passage_lengths = []
    for s in samples:
        for p in s["passages"]:
            passage_lengths.append(len(p.split()))
    
    print(f"\nAverage passage length: {np.mean(passage_lengths):.1f} words")
    print(f"Min/Max: {min(passage_lengths)}/{max(passage_lengths)} words")

if __name__ == "__main__":
    # Create output directory
    Path("data/analysis").mkdir(parents=True, exist_ok=True)
    
    samples = load_musique("dev", max_samples=None)  # Load all
    
    print(f"\n{'='*60}")
    print(f"MusiQue Dataset Analysis ({len(samples)} samples)")
    print(f"{'='*60}")
    
    # Basic stats
    print(f"\nDataset Size:")
    print(f"  Total samples: {len(samples)}")
    print(f"  Answerable: {sum(1 for s in samples if s.get('answerable', True))}")
    print(f"  Unanswerable: {sum(1 for s in samples if not s.get('answerable', True))}")
    
    # Run all analyses
    analyze_answerable(samples)
    analyze_coverage(samples)
    analyze_passage_quality(samples)
    analyze_question_complexity(samples)
    
    # Generate visualizations
    print(f"\n{'='*60}")
    print("Generating Visualizations...")
    try:
        plot_distributions(samples)
    except Exception as e:
        print(f"⚠️  Visualization failed: {e}")
        print("   (Continue with text-based analysis)")
    
    print(f"\n{'='*60}")
    print("✓ Analysis Complete!")
    print(f"{'='*60}\n")