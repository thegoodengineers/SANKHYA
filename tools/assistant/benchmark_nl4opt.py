"""NL4Opt accuracy benchmark for the SANKHYA assistant.

Runs the NL4Opt test set (Ramamonjison et al., NeurIPS 2022, arXiv:2303.08233)
against a locally served model and reports how often the generated LP reaches the
reference optimum, parses but is wrong, or fails to parse.

Usage:
    SANKHYA_ASSISTANT_URL=http://localhost:11434/v1/chat/completions \\
        python benchmark_nl4opt.py --dataset nl4opt_test.json --model llama3 --out results.csv

Each row in results.csv carries the instance id, model name, prompt version,
git commit, and one of: correct | wrong | parse_error.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import assistant

# arXiv:2303.08233 — NL4Opt competition, NeurIPS 2022
CITATION = "Ramamonjison et al., NeurIPS 2022, arXiv:2303.08233"

PROMPT_VERSION = "v1"


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def _run_instance(problem_text: str, reference_obj: float, tol: float = 1e-4) -> str:
    """Return 'correct', 'wrong', or 'parse_error'."""
    endpoint = os.environ.get("SANKHYA_ASSISTANT_URL")
    if not endpoint:
        return "parse_error"

    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            result = assistant.ask_assistant(problem_text)
    except Exception:
        return "parse_error"

    if result is None:
        return "parse_error"
    if result.status not in ("optimal", "feasible"):
        return "wrong"
    if abs(result.objective - reference_obj) <= tol * max(1.0, abs(reference_obj)):
        return "correct"
    return "wrong"


def main():
    ap = argparse.ArgumentParser(description="NL4Opt benchmark")
    ap.add_argument("--dataset", required=True, help="Path to nl4opt_test.json")
    ap.add_argument("--model", default="unknown", help="Model name served at the endpoint")
    ap.add_argument("--out", default="nl4opt_results.csv", help="Output CSV path")
    ap.add_argument("--tol", type=float, default=1e-4, help="Relative/absolute tolerance")
    args = ap.parse_args()

    with open(args.dataset) as f:
        instances = json.load(f)

    commit = _commit()
    counts = {"correct": 0, "wrong": 0, "parse_error": 0}

    with open(args.out, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["id", "model", "prompt_version", "commit", "outcome"])
        for inst in instances:
            inst_id = inst.get("id", "?")
            problem = inst.get("problem", "")
            ref_obj = float(inst.get("obj_value", 0.0))
            outcome = _run_instance(problem, ref_obj, args.tol)
            counts[outcome] += 1
            writer.writerow([inst_id, args.model, PROMPT_VERSION, commit, outcome])
            print(f"{inst_id}: {outcome}")

    total = sum(counts.values())
    print(f"\n--- Results ({args.model}, prompt {PROMPT_VERSION}, commit {commit}) ---")
    print(f"  correct     : {counts['correct']}/{total}")
    print(f"  wrong       : {counts['wrong']}/{total}")
    print(f"  parse_error : {counts['parse_error']}/{total}")
    print(f"  CSV written : {args.out}")
    print(f"\nCitation: {CITATION}")


if __name__ == "__main__":
    main()
