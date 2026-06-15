"""
train.py
--------
Train a PPO agent on the TrafficSignalEnv and compare against a fixed-timing baseline.

Run:
    python training/train.py

---------------------------------------------------------------------------
Best hyperparameters from tune.py grid search (June 2026)
---------------------------------------------------------------------------
Winner : lr=5e-4, gamma=0.95, reward_shaping=True
Avg total wait : 6,766 vehicle-steps (5-episode eval, deterministic)
Phase switches : 135 per episode
Notes:
  - Reward shaping (phase switch penalty=-2.0) improved stability
  - gamma=0.95 outperformed 0.99 — shorter horizon suits single intersection
  - Full rankings: python training/compare_runs.py
---------------------------------------------------------------------------
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

# Add project root to path if running from subdirectory
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.traffic_env import TrafficSignalEnv, NUM_PHASES, MIN_GREEN_STEPS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TOTAL_TIMESTEPS = 500_000  # Train for 500k steps — increase for better results
EVAL_FREQ = 10_000  # Evaluate every 10k steps
N_EVAL_EPISODES = 5
MODEL_SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Baseline: Fixed timing
# ---------------------------------------------------------------------------


def run_fixed_baseline(n_episodes: int = 3) -> float:
    """
    Run the fixed-timing baseline (cycle through phases every MIN_GREEN_STEPS).
    Returns average total wait time per episode.
    """
    print("\n--- Running Fixed Timing Baseline ---")
    env = TrafficSignalEnv(use_gui=False)
    total_waits = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        step = 0

        while not done:
            # Fixed: cycle phases on a fixed schedule
            phase = (step // MIN_GREEN_STEPS) % NUM_PHASES
            obs, reward, terminated, truncated, info = env.step(phase)
            total_reward += reward
            done = terminated or truncated
            step += 1

        total_waits.append(info["total_wait_so_far"])
        print(
            f"  Episode {ep + 1}: total wait = {info['total_wait_so_far']:.0f} vehicle-steps"
        )

    env.close()
    avg = float(np.mean(total_waits))
    print(f"  Baseline avg total wait: {avg:.0f} vehicle-steps\n")
    return avg


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train():
    print("=== Adaptive Traffic Signal Control — Training ===\n")
    # Create vectorized training environment (1 env for now — increase for speed)
    train_env = make_vec_env(
        lambda: TrafficSignalEnv(use_gui=False),
        n_envs=1,
    )

    # --- PPO Agent ---
    # PPO is a solid default for discrete action spaces.
    # Tune learning_rate, n_steps, batch_size if results are poor.
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,  # Discount factor — high because wait time accumulates
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=MODEL_SAVE_DIR,
        name_prefix="atsc_ppo",
    )

    # --- Train ---
    print("--- Training PPO Agent ---")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback],
        progress_bar=True,
    )

    # Save final model
    final_path = os.path.join(MODEL_SAVE_DIR, "atsc_ppo_final")
    model.save(final_path)
    print(f"\nModel saved to: {final_path}")

    train_env.close()
    print(f"\nTo visualize training: tensorboard --logdir {LOG_DIR}")


if __name__ == "__main__":
    train()
