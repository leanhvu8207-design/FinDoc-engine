"""CLI script to execute the FinDoc-Engine retrieval and guardrail benchmark (Task 5.2)."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tests.evaluation.evaluate_retrieval import run_benchmark_evaluation  # noqa: E402

if __name__ == "__main__":
    metrics = run_benchmark_evaluation()
    print(metrics.to_markdown_table())
