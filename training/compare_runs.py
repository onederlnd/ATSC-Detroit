"""
compare_runs.py
---------------
Evaluates all tuned models and ranks them by average total wait time.

Loads every model saved by tune.py, runs each for N episodes, and prints
a ranked comparison table so you can identify the best hyperparameter config.

Run:
    python training/compare_runs.py

    # Evaluate each model for 10 episodes
    python training/compare_runs.py --episodes 10

    # Only evaluate models whose run name contains a string
    python training/compare_runs.py --filter lr0.0001
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.metrics import compute_episode_metrics
from env.traffic_env import TrafficSignalEnv
from agents.rl_agent import RLAgent

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def find_tuned_models(filter_str=None):
    """Scan MODEL_DIR for models saved by tune.py."""
    folders = [f for f in os.listdir(MODEL_DIR) if f.startswith("tune_")]

    models = []
    for folder in folders:
        path = os.path.join(MODEL_DIR, folder, "final.zip")
        if os.path.exists(path):
            models.append({"run_name": folder, "model_path": path})

    if filter_str:
        models = [m for m in models if filter_str in m["run_name"]]

    return sorted(models, key=lambda m: m["run_name"])


def evaluate_model(model_path, n_episodes=5):
    """Run a saved model for N episodes and return aggregated metrics."""
    agent = RLAgent(model_path=model_path)

    waits = []
    rewards = []
    switches = []

    for _ in range(n_episodes):
        env = TrafficSignalEnv(use_gui=False)
        try:
            episode = agent.run_episode(env)
        finally:
            env.close()

        metrics = compute_episode_metrics(episode["info_history"])
        waits.append(metrics["total_wait"])
        rewards.append(episode["total_reward"])
        switches.append(metrics["n_phases_switched"])

    mean_wait = sum(waits) / len(waits)
    std_wait = (sum((x - mean_wait) ** 2 for x in waits) / len(waits)) ** 0.5

    return {
        "avg_total_wait": mean_wait,
        "std_total_wait": std_wait,
        "avg_reward": sum(rewards) / len(rewards),
        "avg_phase_switches": sum(switches) / len(switches),
    }


def print_ranked_table(results):
    """Print all models ranked by avg_total_wait ascending."""
    ranked = sorted(results, key=lambda r: r["avg_total_wait"])

    print("\n" + "=" * 100)
    print("=== Model Rankings — lower wait is better ===")
    print("=" * 100)
    print(
        f"{'Rank':<5} {'Run Name':<45} {'Avg Wait':>10} {'Std Wait':>10} "
        f"{'Avg Reward':>12} {'Phase Switches':>15}"
    )
    print("-" * 100)

    for i, r in enumerate(ranked, start=1):
        marker = " <-- BEST" if i == 1 else ""
        print(
            f"{i:<5} {r['run_name']:<45} "
            f"{r['avg_total_wait']:>10,.0f} "
            f"{r['std_total_wait']:>10,.0f} "
            f"{r['avg_reward']:>12,.0f} "
            f"{r['avg_phase_switches']:>15.1f}"
            f"{marker}"
        )

    print("=" * 100)
    best = ranked[0]
    print(f"\nBest model : {best['run_name']}")
    print(f"Model path : {best['model_path']}")
    print("\nTo watch the best model:")
    print(f'    python main.py watch --model "{best["model_path"]}"')
    print("\nTo compare against baseline:")
    print(f'    python main.py compare --model "{best["model_path"]}"')


def main():
    parser = argparse.ArgumentParser(description="Rank tuned ATSC-Detroit models.")
    parser.add_argument(
        "--episodes", type=int, default=5, help="Episodes per model (default: 5)."
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Only evaluate run names containing this string.",
    )
    args = parser.parse_args()

    models = find_tuned_models(filter_str=args.filter)

    if not models:
        print("No tuned models found. Run 'python training/tune.py' first.")
        sys.exit(1)

    print(
        f"\nFound {len(models)} model(s). Evaluating each for {args.episodes} episode(s)..."
    )

    results = []
    for i, m in enumerate(models, start=1):
        print(f"  Evaluating {m['run_name']} ({i}/{len(models)})...")
        metrics = evaluate_model(m["model_path"], n_episodes=args.episodes)
        results.append({**m, **metrics})

    print_ranked_table(results)


if __name__ == "__main__":
    main()
