"""Unit tests for the quantitative evaluation harness (Phase 5 - Task 5.2)."""

from pathlib import Path

from tests.evaluation.evaluate_retrieval import run_benchmark_evaluation


def test_benchmark_evaluation_harness_achieves_target_slas() -> None:
    """Verify the 30-query benchmark suite executes and achieves target SLAs."""
    gt_path = Path("data/test_suite/ground_truth.json")
    assert gt_path.exists()

    metrics = run_benchmark_evaluation(ground_truth_path=gt_path)
    assert metrics.total_queries == 30
    assert metrics.hit_at_1 >= 0.85
    assert metrics.hit_at_3 >= 0.90
    assert metrics.hit_at_5 >= 0.95
    assert metrics.mrr >= 0.85
    assert metrics.ndcg_at_5 >= 0.88
    assert metrics.hallucination_detection_rate == 1.0
    md_table = metrics.to_markdown_table()
    assert "Hit Rate @ 1" in md_table
    assert "NDCG @ 5" in md_table
