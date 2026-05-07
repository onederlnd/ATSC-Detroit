"""
train.py
--------
Train a PPO agent on the TrafficSignalEnv and compare against a fixed-timing baseline.

Run:
    python training/train.py
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Add project root to path if running from subdirectory
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.traffic_env import TrafficSignalEnv, NUM_PHASES, MIN_GREEN_STEPS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TOTAL_TIMESTEPS = 500_000       # Train for 500k steps — increase for better results
EVAL_FREQ = 10_000              # Evaluate every 10k steps
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
        print(f"  Episode {ep + 1}: total wait = {info['total_wait_so_far']:.0f} vehicle-steps")

    env.close()
    avg = float(np.mean(total_waits))
    print(f"  Baseline avg total wait: {avg:.0f} vehicle-steps\n")
    return avg


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    print("=== Adaptive Traffic Signal Control — Training ===\n")

    # Run baseline first so we have a comparison number
    baseline_wait = run_fixed_baseline()

    # Create vectorized training environment (1 env for now — increase for speed)
    train_env = make_vec_env(
        lambda: TrafficSignalEnv(use_gui=False),
        n_envs=1,
    )

    # Separate eval environment
    eval_env = TrafficSignalEnv(use_gui=False)

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
        gamma=0.99,           # Discount factor — high because wait time accumulates
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    # --- Callbacks ---
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_SAVE_DIR,
        log_path=LOG_DIR,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        verbose=1,
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
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    # Save final model
    final_path = os.path.join(MODEL_SAVE_DIR, "atsc_ppo_final")
    model.save(final_path)
    print(f"\nModel saved to: {final_path}")

    # --- Quick evaluation of trained agent ---
    print("\n--- Evaluating Trained Agent ---")
    trained_waits = []
    for ep in range(N_EVAL_EPISODES):
        obs, _ = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = eval_env.step(int(action))
            done = terminated or truncated
        trained_waits.append(info["total_wait_so_far"])
        print(f"  Episode {ep + 1}: total wait = {info['total_wait_so_far']:.0f} vehicle-steps")

    eval_env.close()
    train_env.close()

    avg_trained = float(np.mean(trained_waits))
    improvement = (baseline_wait - avg_trained) / baseline_wait * 100

    print(f"\n=== Results ===")
    print(f"  Fixed baseline avg wait : {baseline_wait:.0f} vehicle-steps")
    print(f"  Trained agent avg wait  : {avg_trained:.0f} vehicle-steps")
    print(f"  Improvement             : {improvement:.1f}%")
    print(f"\nTo visualize training: tensorboard --logdir {LOG_DIR}")


if __name__ == "__main__":
    train()