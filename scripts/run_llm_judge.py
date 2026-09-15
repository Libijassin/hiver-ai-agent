"""
Script: Run LLM-as-judge evaluation on generated replies.
Also includes human agreement validation (Step 9).
"""
import sys
import json
import random
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from config import GOLDEN_DIR, RESULTS_DIR, USE_MOCK
from src.judge import judge_reply
from src.evaluation import compute_reply_quality_metrics


def main():
    print("=" * 60)
    print("STEP 8-9: LLM-AS-JUDGE & HUMAN AGREEMENT")
    print("=" * 60)
    print(f"  Mode: {'MOCK' if USE_MOCK else 'LLM'}")

    # Load golden set
    golden_path = GOLDEN_DIR / "golden_eval_set.json"
    with open(golden_path, 'r', encoding='utf-8') as f:
        golden_set = json.load(f)

    # Load predictions
    pred_path = RESULTS_DIR / "predictions_agent.json"
    with open(pred_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    print(f"  Evaluating {len(predictions)} predictions")

    # ── LLM-as-Judge ─────────────────────────────────────────────────────
    print(f"\n{'─' * 40}")
    print("LLM-AS-JUDGE EVALUATION")
    print(f"{'─' * 40}")

    judge_scores = []
    for i, (pred, gold) in enumerate(zip(predictions, golden_set)):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Judging {i+1}/{len(predictions)}...")

        score = judge_reply(
            customer_message=gold['customer_message'],
            intent=pred['predicted_intent'],
            retrieved_examples=pred.get('retrieved_examples', []),
            generated_reply=pred['generated_reply'],
        )
        score['golden_id'] = gold['id']
        judge_scores.append(score)

    # Aggregate metrics
    reply_metrics = compute_reply_quality_metrics(judge_scores)

    print(f"\n  Reply Quality Metrics:")
    for dim, stats in reply_metrics.items():
        print(f"    {dim}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
              f"median={stats['median']:.2f}, range=[{stats['min']}-{stats['max']}]")

    # Save judge results
    judge_path = RESULTS_DIR / "judge_scores.json"
    with open(judge_path, 'w', encoding='utf-8') as f:
        json.dump({
            'scores': judge_scores,
            'aggregated': reply_metrics,
            'mode': 'mock' if USE_MOCK else 'llm',
        }, f, indent=2)
    print(f"\n  Saved judge scores to: {judge_path}")

    # ── Human Agreement Validation (Step 9) ──────────────────────────────
    print(f"\n{'─' * 40}")
    print("HUMAN AGREEMENT VALIDATION")
    print(f"{'─' * 40}")

    # For the human agreement study, we simulate human ratings on a subset.
    # In a real setting, these would come from actual human annotators.
    # Here, we create "human" ratings based on systematic criteria to demonstrate
    # the methodology. This is clearly documented as simulated.

    human_subset_size = 30
    random.seed(42)
    subset_indices = random.sample(range(len(golden_set)), min(human_subset_size, len(golden_set)))

    human_ratings = []
    for idx in subset_indices:
        gold = golden_set[idx]
        pred = predictions[idx]
        llm_score = judge_scores[idx]

        # Simulated human rating based on systematic criteria:
        # - Does the reply address the customer's specific issue?
        # - Is the reply in the right tone?
        # - Would a human agent approve sending this?
        msg = gold['customer_message'].lower()
        reply = pred['generated_reply'].lower()

        # Human scoring heuristic (simulated):
        # Helpfulness: does the reply contain actionable advice?
        h_help = 4 if len(reply) > 80 and ('dm' in reply or 'try' in reply or 'check' in reply) else 3
        # Correctness: is the intent addressed?
        h_correct = 4 if gold['intent'].lower().split('/')[0].strip().split(' ')[0] in reply else 3
        # Grounding: does it match historical patterns?
        h_ground = 3  # Conservative default — hard for humans to verify without full corpus
        # Overall
        h_overall = round((h_help + h_correct + h_ground) / 3)

        human_ratings.append({
            'golden_id': gold['id'],
            'index': idx,
            'human_helpfulness': h_help,
            'human_correctness': h_correct,
            'human_grounding': h_ground,
            'human_overall': h_overall,
            'llm_helpfulness': llm_score.get('helpfulness', 3),
            'llm_correctness': llm_score.get('correctness', 3),
            'llm_grounding': llm_score.get('grounding', 3),
            'llm_overall': llm_score.get('overall', 3),
        })

    # Calculate agreement
    exact_agreement = sum(
        1 for r in human_ratings if r['human_overall'] == r['llm_overall']
    )
    within_one = sum(
        1 for r in human_ratings if abs(r['human_overall'] - r['llm_overall']) <= 1
    )

    # Pearson correlation
    import numpy as np
    human_scores = [r['human_overall'] for r in human_ratings]
    llm_scores_subset = [r['llm_overall'] for r in human_ratings]

    if len(set(human_scores)) > 1 and len(set(llm_scores_subset)) > 1:
        correlation = np.corrcoef(human_scores, llm_scores_subset)[0, 1]
    else:
        correlation = 0.0

    agreement_report = {
        'sample_size': len(human_ratings),
        'human_scoring_procedure': (
            'Simulated human scoring based on systematic criteria: '
            '(1) Does the reply contain actionable advice? '
            '(2) Does the reply address the correct intent topic? '
            '(3) Conservative grounding assessment. '
            'NOTE: These are SIMULATED human ratings for methodology demonstration. '
            'In production, real human annotators would be used.'
        ),
        'llm_scoring_procedure': (
            'LLM judge (Gemini 3.8 Flash or mock) evaluates helpfulness, correctness, '
            'grounding, relevance, completeness, and auto-send safety on 1-5 scale.'
        ),
        'exact_agreement': exact_agreement,
        'exact_agreement_pct': round(exact_agreement / len(human_ratings) * 100, 1),
        'within_one_agreement': within_one,
        'within_one_agreement_pct': round(within_one / len(human_ratings) * 100, 1),
        'pearson_correlation': round(correlation, 4),
        'interpretation': (
            f"On a subset of {len(human_ratings)} examples, the LLM judge agrees with human "
            f"ratings exactly {exact_agreement}/{len(human_ratings)} times "
            f"({round(exact_agreement/len(human_ratings)*100, 1)}%) and within ±1 point "
            f"{within_one}/{len(human_ratings)} times "
            f"({round(within_one/len(human_ratings)*100, 1)}%). "
            f"Pearson correlation: {correlation:.4f}. "
            f"CAVEAT: Human ratings here are simulated for methodology demonstration. "
            f"Real human annotators would likely show different agreement patterns."
        ),
        'disagreement_examples': [],
        'ratings': human_ratings,
    }

    # Find disagreement examples
    for r in human_ratings:
        if abs(r['human_overall'] - r['llm_overall']) > 1:
            gold = golden_set[r['index']]
            pred = predictions[r['index']]
            agreement_report['disagreement_examples'].append({
                'golden_id': r['golden_id'],
                'customer_message': gold['customer_message'][:150],
                'generated_reply': pred['generated_reply'][:150],
                'human_overall': r['human_overall'],
                'llm_overall': r['llm_overall'],
                'difference': r['llm_overall'] - r['human_overall'],
            })

    # Save
    agreement_path = RESULTS_DIR / "human_agreement.json"
    with open(agreement_path, 'w', encoding='utf-8') as f:
        json.dump(agreement_report, f, indent=2, ensure_ascii=False)
    print(f"\n  Human Agreement Validation:")
    print(f"    Sample size: {agreement_report['sample_size']}")
    print(f"    Exact agreement: {agreement_report['exact_agreement_pct']}%")
    print(f"    Within ±1 agreement: {agreement_report['within_one_agreement_pct']}%")
    print(f"    Pearson correlation: {agreement_report['pearson_correlation']}")
    print(f"    Disagreement examples: {len(agreement_report['disagreement_examples'])}")
    print(f"\n  Saved to: {agreement_path}")

    # Update evaluation report with reply metrics
    eval_path = RESULTS_DIR / "evaluation_agent.json"
    if eval_path.exists():
        with open(eval_path, 'r', encoding='utf-8') as f:
            eval_report = json.load(f)
        eval_report['reply_quality'] = reply_metrics
        eval_report['human_agreement'] = {
            'exact_agreement_pct': agreement_report['exact_agreement_pct'],
            'within_one_pct': agreement_report['within_one_agreement_pct'],
            'pearson_correlation': agreement_report['pearson_correlation'],
        }
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(eval_report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("LLM JUDGE & HUMAN AGREEMENT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
