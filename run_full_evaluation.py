#!/usr/bin/env python3
"""
Run Full SpotifyCares Evaluation Pipeline
-----------------------------------------
Executes data preparation, baseline models, agent inference, metric evaluation,
LLM-as-a-judge scoring, and failure mode analysis in sequence.
"""
import subprocess
import sys
import os

SCRIPTS = [
    ("1/6 Preparing dataset splits (0% data leakage)", ["python", "scripts/prepare_data.py"]),
    ("2/6 Building 200-sample hand-labelled golden set", ["python", "scripts/build_golden_set.py"]),
    ("3/6 Running baseline models (Trivial & Simple)", ["python", "scripts/run_baselines.py"]),
    ("4/6 Running SpotifyCares Agent inference", ["python", "scripts/run_agent.py"]),
    ("5/6 Computing evaluation metrics & comparisons", ["python", "scripts/evaluate.py"]),
    ("6/6 Running LLM-as-a-Judge & Failure analysis", ["python", "scripts/run_llm_judge.py", "--n-samples", "200"]),
]

def main():
    print("=" * 65)
    print("🚀 Running Full SpotifyCares Support Agent Evaluation Pipeline")
    print("=" * 65)
    
    cwd = os.path.dirname(os.path.abspath(__file__))
    
    for title, cmd in SCRIPTS:
        print(f"\n▶ [{title}]...")
        result = subprocess.run(cmd, cwd=cwd)
        if result.returncode != 0:
            print(f"❌ Step failed with returncode {result.returncode}: {' '.join(cmd)}")
            sys.exit(result.returncode)
            
    print("\n" + "=" * 65)
    print("✅ All evaluation steps completed successfully!")
    print("📊 Results saved to results/")
    print("🌐 Start the web dashboard: python server.py --port 8000")
    print("=" * 65)

if __name__ == "__main__":
    main()
