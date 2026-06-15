"""
metrics.py
----------
Metric computation, comparison, and plotting for ATSC-Detroit.

Both the fixed baseline and the RL agent funnel their episode results
through here so all performance numbers are computed the same way.

Usage:
    from utils.metrics import compute_episode_metrics, compare_agents, plot_wait_over_time

    metrics = compute_episode_metrics(info_history)
    compare_agents(baseline_results, rl_results)
    plot_wait_over_time(info_history, label="PPO Agent")
"""

import matplotlib.pyplot as plt


def compute_episode_metrics(info_history):
    """
    Compute summary statistics from one episode's worth of step info dicts.

    Parameters
    ----------
    info_history : list of dict
        The list of `info` dicts returned by env.step() across one episode.
        Each dict contains at minimum:
            "waiting_vehicles"   (int)   — vehicles halted this step
            "current_phase"      (int)   — active phase index this step
            "total_wait_so_far"  (float) — cumulative wait up to this step

    Returns
    -------
    dict with keys:
        total_wait        (float) — final cumulative waiting vehicles
        avg_wait_per_step (float) — mean waiting vehicles per step
        max_wait_step     (int)   — highest single-step waiting vehicle count
        n_phases_switched (int)   — how many times the phase changed
    """
    if not info_history:
        raise ValueError("info_history cannot be empty")

    total_wait = info_history[-1]["total_wait_so_far"]

    waiting_vehicles = [info["waiting_vehicles"] for info in info_history]
    avg_wait_per_step = sum(waiting_vehicles) / len(waiting_vehicles)
    max_wait_step = max(waiting_vehicles)

    counter = 0
    for i in range(1, len(info_history)):
        if info_history[i]["current_phase"] != info_history[i - 1]["current_phase"]:
            counter += 1

    n_phases_switched = counter

    return {
        "total_wait": total_wait,
        "avg_wait_per_step": avg_wait_per_step,
        "max_wait_step": max_wait_step,
        "n_phases_switched": n_phases_switched,
    }


def compare_agents(baseline_results, rl_results):
    """
    Print a side-by-side comparison table and return the improvement percentage.

    Parameters
    ----------
    baseline_results : list of dict
        One dict per episode from FixedTimingAgent.run_episode().
    rl_results : list of dict
        One dict per episode from RLAgent.run_episode().

    Returns
    -------
    float
        Percentage improvement in total_wait of the RL agent over the baseline.
        Positive means the RL agent waited less (better).
        Formula: (baseline_avg - rl_avg) / baseline_avg * 100
    """
    avg_baseline_wait = sum(r["total_wait"] for r in baseline_results) / len(
        baseline_results
    )
    avg_rl_wait = sum(r["total_wait"] for r in rl_results) / len(rl_results)

    if avg_baseline_wait == 0:
        improvement = 0.0
    else:
        improvement = (avg_baseline_wait - avg_rl_wait) / avg_baseline_wait * 100

    print("\n┌─────────────────────────────────────────┐")
    print("│           Agent Comparison              │")
    print("├──────────────────────┬──────────────────┤")
    print(f"│ Fixed baseline       │ {avg_baseline_wait:>12,.0f} veh-steps │")
    print(f"│ PPO agent            │ {avg_rl_wait:>12,.0f} veh-steps │")
    print(f"│ Improvement          │ {improvement:>15.1f}% │")
    print("└──────────────────────┴──────────────────┘\n")

    return float(improvement)


def plot_wait_over_time(info_history, label="Agent", save_path=None):
    """
    Line chart of waiting vehicles per simulation step.

    Parameters
    ----------
    info_history : list of dict
        One episode's worth of env.step() info dicts.
    label : str
        Legend label for the line (e.g. "PPO Agent" or "Fixed Baseline").
    save_path : str or None
        If provided, save the figure to this path instead of displaying it.
    """
    waiting = [info["waiting_vehicles"] for info in info_history]
    steps = range(len(waiting))

    plt.plot(steps, waiting, label=label)
    plt.title(f"Waiting Vehicles Over Time — {label}")
    plt.xlabel("Simulation Step")
    plt.ylabel("Waiting Vehicles")
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()


def plot_phase_distribution(info_history, num_phases=4, save_path=None):
    """
    Bar chart showing how many steps each signal phase was active.

    Parameters
    ----------
    info_history : list of dict
        One episode's worth of env.step() info dicts.
    num_phases : int
        Total number of signal phases (used to label the x-axis).
    save_path : str or None
        If provided, save the figure instead of displaying it.
    """
    counts = [0] * num_phases
    for info in info_history:
        phase = info["current_phase"]
        if 0 <= phase < num_phases:
            counts[phase] += 1

    plt.bar(range(num_phases), counts)
    plt.title("Phase Distribution")
    plt.xlabel("Phase Index")
    plt.ylabel("Steps Active")
    plt.xticks(range(num_phases))

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()
