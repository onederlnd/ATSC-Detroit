"""
tune.py
-------
Grid search over PPO hyperparameters for ATSC-Detroit.

Trains one model per config combination, saves each to models/,
and logs each to TensorBoard with a distinct run name.

Run:
    python training/tune.py

Results:
    Models saved to : models/tune_lr{lr}_gamma{g}_rs{reward_shaping}/
    TensorBoard logs: logs/tune_lr{lr}_gamma{g}_rs{reward_shaping}/
    Summary table   : printed to stdout when all runs complete

View all runs in TensorBoard:
    tensorboard --logdir logs/
"""

import os
import sys
import itertools
import gymnasium as gym

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from env.traffic_env import TrafficSignalEnv

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")

TOTAL_TIMESTEPS = 50_000

GRID = {
    "learning_rate": [1e-4, 3e-4, 5e-4],
    "gamma": [0.95, 0.99],
    "reward_shaping": [False, True],
}

PHASE_SWITCH_PENALTY = -2.0


class RewardShapedEnv(gym.Env):
    """Thin wrapper around TrafficSignalEnv that adds a phase-switch penalty."""

    def __init__(self, penalty=PHASE_SWITCH_PENALTY):
        self._penalty = penalty
        self._env = TrafficSignalEnv(use_gui=False)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self, **kwargs):
        obs, info = self._env.reset(**kwargs)
        self._last_phase = info.get("current_phase", 0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        if self._last_phase != info["current_phase"]:
            reward += self._penalty
        self._last_phase = info["current_phase"]
        return obs, reward, terminated, truncated, info

    def close(self):
        self._env.close()

    def render(self):
        pass


def run_training(
    learning_rate, gamma, reward_shaping, combo_number=None, total_combos=None
):
    """Train one PPO model with the given hyperparameters."""
    run_name = f"tune_lr{learning_rate}_gamma{gamma}_rs{int(reward_shaping)}"
    run_model_dir = os.path.join(MODEL_DIR, run_name)
    run_log_dir = os.path.join(LOG_DIR, run_name)

    os.makedirs(run_model_dir, exist_ok=True)
    os.makedirs(run_log_dir, exist_ok=True)

    if reward_shaping:
        train_env = make_vec_env(lambda: RewardShapedEnv(), n_envs=1)
    else:
        train_env = make_vec_env(lambda: TrafficSignalEnv(use_gui=False), n_envs=1)

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=learning_rate,
        gamma=gamma,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=run_log_dir,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=run_model_dir,
        name_prefix="ckpt",
    )

    counter = f"({combo_number}/{total_combos})" if combo_number else ""
    print(f"\n=== Run: {run_name} {counter} ===")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback],
        progress_bar=True,
    )

    final_path = os.path.join(run_model_dir, "final")
    model.save(final_path)
    print(f"Model saved to: {final_path}.zip")

    train_env.close()

    return {
        "run_name": run_name,
        "model_path": final_path + ".zip",
        "lr": learning_rate,
        "gamma": gamma,
        "rs": reward_shaping,
    }


def print_summary(results):
    """Print a formatted table of all runs and their configs."""
    print("\n" + "=" * 80)
    print("=== Tuning Summary ===")
    print("=" * 80)
    print(f"{'Run Name':<45} {'LR':<8} {'Gamma':<6} {'RS':<5} {'Model Path'}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['run_name']:<45} "
            f"{r['lr']:<8} "
            f"{r['gamma']:<6} "
            f"{str(r['rs']):<5} "
            f"{r['model_path']}"
        )
    print("=" * 80)
    print("\nTo compare training curves:")
    print(f"    tensorboard --logdir {LOG_DIR}")
    print("\nTo rank models by performance:")
    print("    python training/compare_runs.py")


def main():
    keys = list(GRID.keys())
    values = list(GRID.values())
    combos = list(itertools.product(*values))

    total = len(combos)
    est_mins = total * (TOTAL_TIMESTEPS / 500_000) * 15
    est_hrs = est_mins / 60

    print("\n=== ATSC-Detroit Hyperparameter Tuning ===")
    print(f"  Configs to run : {total}")
    print(f"  Timesteps each : {TOTAL_TIMESTEPS:,}")
    print(f"  Estimated time : ~{est_hrs:.1f} hours")
    print(f"  TensorBoard    : tensorboard --logdir {LOG_DIR}\n")
    input("Press Enter to start, or Ctrl+C to cancel...")

    results = []
    for i, combo in enumerate(combos, start=1):
        lr, gamma, rs = combo
        result = run_training(
            learning_rate=lr,
            gamma=gamma,
            reward_shaping=rs,
            combo_number=i,
            total_combos=total,
        )
        results.append(result)

    print_summary(results)


if __name__ == "__main__":
    main()
