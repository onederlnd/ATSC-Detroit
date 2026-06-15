"""
evaluate.py
-----------
Standalone evaluation script for ATSC-Detroit.

Loads a saved model (or runs the fixed baseline) and evaluates it for N
episodes, printing a metrics table and saving plots to output/.

Run:
    # Evaluate a trained model
    python training/evaluate.py --model models/best_model.zip

    # Evaluate the fixed baseline
    python training/evaluate.py --baseline

    # Run 10 episodes with the GUI visible
    python training/evaluate.py --model models/best_model.zip --episodes 10 --gui

    # Compare a model against the baseline in one run
    python training/evaluate.py --model models/best_model.zip --compare
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
from env.traffic_env import TrafficSignalEnv, NUM_PHASES
from agents.rl_agent import RLAgent
from agents.fixed_baseline import FixedTimingAgent
from utils.metrics import (
    compute_episode_metrics,
    compare_agents,
    plot_wait_over_time,
    plot_phase_distribution,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


def run_evaluation(agent, n_episodes, use_gui=False):
    """Run an agent for N episodes and return per-episode metric dicts."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []

    for e in range(n_episodes):
        env = TrafficSignalEnv(use_gui=use_gui)
        try:
            episode = agent.run_episode(env)
        finally:
            env.close()

        metrics = compute_episode_metrics(episode["info_history"])
        print(
            f"Episode {e + 1}/{n_episodes} — "
            f"total wait: {metrics['total_wait']:,.0f} "
            f"reward: {episode['total_reward']:,.0f} "
            f"steps: {episode['n_steps']}"
        )

        results.append({**episode, **metrics})

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ATSC-Detroit agents.")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--compare", action="store_true")

    args = parser.parse_args()
    if not any([args.model, args.baseline, args.compare]):
        parser.print_help()
        sys.exit(1)

    if args.compare:
        if not args.model:
            print("Error --compare requries --model <path>.")
            sys.exit(1)
        baseline_agent = FixedTimingAgent(num_phases=NUM_PHASES, cycle_length=None)
        baseline_results = run_evaluation(baseline_agent, args.episodes, args.gui)

        rl_agent = RLAgent(model_path=args.model)
        rl_results = run_evaluation(rl_agent, args.episodes, args.gui)

        compare_agents(baseline_results, rl_results)

        plot_wait_over_time(
            baseline_results[-1]["info_history"],
            label="Fixed Baseline",
            save_path=os.path.join(OUTPUT_DIR, "baseline_wait_over_time.png"),
        )
        plot_wait_over_time(
            rl_results[-1]["info_history"],
            label="PPO Agent",
            save_path=os.path.join(OUTPUT_DIR, "rl_wait_over_time.png"),
        )
        plot_phase_distribution(
            baseline_results[-1]["info_history"],
            num_phases=NUM_PHASES,
            save_path=os.path.join(OUTPUT_DIR, "baseline_phase_distribution.png"),
        )
        plot_phase_distribution(
            rl_results[-1]["info_history"],
            num_phases=NUM_PHASES,
            save_path=os.path.join(OUTPUT_DIR, "rl_phase_distribution.png"),
        )
        print(f"Plots saved to {OUTPUT_DIR}/")

    if args.baseline:
        agent = FixedTimingAgent(num_phases=NUM_PHASES, cycle_length=None)
        results = run_evaluation(agent, args.episodes, args.gui)

        plot_wait_over_time(
            results[-1]["info_history"],
            label="Fixed Baseline",
            save_path=os.path.join(OUTPUT_DIR, "baseline_wait_over_time.png"),
        )
        plot_phase_distribution(
            results[-1]["info_history"],
            num_phases=NUM_PHASES,
            save_path=os.path.join(OUTPUT_DIR, "baseline_phase_distribution.png"),
        )
        print(f"Plots saved to {OUTPUT_DIR}/")

    if args.model:
        agent = RLAgent(model_path=args.model)
        results = run_evaluation(agent, args.episodes, args.gui)

        plot_wait_over_time(
            results[-1]["info_history"],
            label="PPO Agent",
            save_path=os.path.join(OUTPUT_DIR, "rl_wait_over_time.png"),
        )
        plot_phase_distribution(
            results[-1]["info_history"],
            num_phases=NUM_PHASES,
            save_path=os.path.join(OUTPUT_DIR, "rl_phase_distribution.png"),
        )
        print(f"Plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
