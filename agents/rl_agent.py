# agents/rl_agent.py

"""
Thin wrapper around a saved Stable-Baselines3 PPO model.
"""

import os
from stable_baselines3 import PPO


class RLAgent:
    """Loads and runs a trained PPO model for inference."""

    def __init__(self, model_path):
        self._model_path = model_path

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No model found at '{model_path}'. Run 'python main.py train' first to generate one."
            )
        self._model = PPO.load(model_path)

    def predict(self, obs, deterministic=True):
        """Return the model's action for the current observation."""
        return self._model.predict(obs, deterministic=deterministic)

    def run_episode(self, env):
        """Run one full episode and return summary metrics."""
        obs, info = env.reset()
        total_reward = 0.0
        n_steps = 0
        done = False
        info_history = []

        while not done:
            action, _ = self.predict(obs)
            action = int(action)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            n_steps += 1
            info_history.append(info)
            done = terminated or truncated

        return {
            "total_wait": info["total_wait_so_far"],
            "total_reward": total_reward,
            "n_steps": n_steps,
            "info_history": info_history,
        }

    @property
    def model_path(self):
        """The path this agent's model was loaded from."""
        return self._model_path
