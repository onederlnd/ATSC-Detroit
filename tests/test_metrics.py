"""
test_metrics.py
---------------
Unit tests for utils/metrics.py.

Run:
    python -m pytest tests/test_metrics.py -v
"""

import pytest
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.metrics import (
    compute_episode_metrics,
    compare_agents,
    plot_wait_over_time,
    plot_phase_distribution,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic info_history
# ---------------------------------------------------------------------------


def make_info_history(n_steps=100, waiting=5, phase=0):
    """Create a synthetic info_history list for testing."""
    return [
        {
            "step": i,
            "current_phase": phase if i < n_steps // 2 else (phase + 1) % 4,
            "waiting_vehicles": waiting,
            "total_wait_so_far": float(waiting * (i + 1)),
        }
        for i in range(n_steps)
    ]


def make_episode_result(n_steps=100, waiting=5, total_wait=None):
    """Create a synthetic episode result dict."""
    info_history = make_info_history(n_steps=n_steps, waiting=waiting)
    return {
        "total_wait": total_wait
        if total_wait is not None
        else float(waiting * n_steps),
        "total_reward": float(-waiting * n_steps),
        "n_steps": n_steps,
        "info_history": info_history,
    }


# ---------------------------------------------------------------------------
# compute_episode_metrics tests
# ---------------------------------------------------------------------------


def test_compute_metrics_raises_on_empty():
    """Should raise ValueError when given an empty list."""
    with pytest.raises(ValueError):
        compute_episode_metrics([])


def test_compute_metrics_total_wait():
    """total_wait should equal the last info's total_wait_so_far."""
    history = make_info_history(n_steps=10, waiting=3)
    metrics = compute_episode_metrics(history)
    assert metrics["total_wait"] == history[-1]["total_wait_so_far"]


def test_compute_metrics_avg_wait_per_step():
    """avg_wait_per_step should be the mean of waiting_vehicles."""
    history = make_info_history(n_steps=10, waiting=4)
    metrics = compute_episode_metrics(history)
    assert metrics["avg_wait_per_step"] == pytest.approx(4.0)


def test_compute_metrics_max_wait_step():
    """max_wait_step should be the highest single-step waiting count."""
    history = make_info_history(n_steps=10, waiting=2)
    # Inject a spike
    history[5]["waiting_vehicles"] = 20
    metrics = compute_episode_metrics(history)
    assert metrics["max_wait_step"] == 20


def test_compute_metrics_n_phases_switched():
    """n_phases_switched should count phase changes correctly."""
    # make_info_history switches phase at the halfway point — 1 switch
    history = make_info_history(n_steps=10, waiting=1)
    metrics = compute_episode_metrics(history)
    assert metrics["n_phases_switched"] == 1


def test_compute_metrics_no_phase_switches():
    """If phase never changes, n_phases_switched should be 0."""
    history = [
        {
            "step": i,
            "current_phase": 0,
            "waiting_vehicles": 1,
            "total_wait_so_far": float(i + 1),
        }
        for i in range(10)
    ]
    metrics = compute_episode_metrics(history)
    assert metrics["n_phases_switched"] == 0


def test_compute_metrics_returns_all_keys():
    """Result dict should contain all four expected keys."""
    history = make_info_history()
    metrics = compute_episode_metrics(history)
    for key in [
        "total_wait",
        "avg_wait_per_step",
        "max_wait_step",
        "n_phases_switched",
    ]:
        assert key in metrics, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# compare_agents tests
# ---------------------------------------------------------------------------


def test_compare_agents_improvement_positive():
    """When RL agent waits less, improvement should be positive."""
    baseline = [make_episode_result(total_wait=1000) for _ in range(3)]
    rl = [make_episode_result(total_wait=700) for _ in range(3)]
    improvement = compare_agents(baseline, rl)
    assert improvement == pytest.approx(30.0)


def test_compare_agents_improvement_negative():
    """When RL agent waits more, improvement should be negative."""
    baseline = [make_episode_result(total_wait=700) for _ in range(3)]
    rl = [make_episode_result(total_wait=1000) for _ in range(3)]
    improvement = compare_agents(baseline, rl)
    assert improvement < 0


def test_compare_agents_returns_float():
    """compare_agents should return a float."""
    baseline = [make_episode_result() for _ in range(3)]
    rl = [make_episode_result() for _ in range(3)]
    result = compare_agents(baseline, rl)
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# plot_wait_over_time tests
# ---------------------------------------------------------------------------


def test_plot_wait_over_time_saves_file():
    """When save_path is given, a file should be created."""
    history = make_info_history()
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_plot.png")
        plot_wait_over_time(history, label="Test", save_path=save_path)
        assert os.path.exists(save_path), "Plot file was not created"


def test_plot_wait_over_time_no_crash_without_save():
    """plot_wait_over_time should not crash when save_path is None."""
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend for testing
    history = make_info_history()
    try:
        plot_wait_over_time(history, label="Test", save_path=None)
    except Exception as e:
        pytest.fail(f"plot_wait_over_time raised an exception: {e}")


# ---------------------------------------------------------------------------
# plot_phase_distribution tests
# ---------------------------------------------------------------------------


def test_plot_phase_distribution_saves_file():
    """When save_path is given, a file should be created."""
    history = make_info_history()
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_phase_plot.png")
        plot_phase_distribution(history, num_phases=4, save_path=save_path)
        assert os.path.exists(save_path), "Phase distribution plot was not created"


def test_plot_phase_distribution_no_crash_without_save():
    """plot_phase_distribution should not crash when save_path is None."""
    import matplotlib

    matplotlib.use("Agg")
    history = make_info_history()
    try:
        plot_phase_distribution(history, num_phases=4, save_path=None)
    except Exception as e:
        pytest.fail(f"plot_phase_distribution raised an exception: {e}")
