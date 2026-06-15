"""
fixed_baseline.py
-----------------
A reusable fixed-timing traffic signal agent.

Fixed-timing controllers cycle through all signal phases on a predetermined
schedule regardless of actual traffic conditions. This serves as the
performance baseline that the RL agent must beat.

This class intentionally mirrors the Stable-Baselines3 model API so it can
be swapped in anywhere the PPO model is used — no special-casing needed.

Usage:
    from agents.fixed_baseline import FixedTimingAgent
    from env.traffic_env import TrafficSignalEnv

    env = TrafficSignalEnv()
    agent = FixedTimingAgent(num_phases=4, cycle_length=40)
    result = agent.run_episode(env)
    print(result)
"""

from env.traffic_env import MIN_GREEN_STEPS


class FixedTimingAgent:
    """Cycles through signal phases on a fixed schedule."""

    def __init__(self, num_phases, cycle_length):
        self.num_phases = num_phases
        self.cycle_length = cycle_length or MIN_GREEN_STEPS
        self._step = 0

    def predict(self, obs, deterministic=True):
        """Return the action for the current timestep."""
        phase = (self._step // self.cycle_length) % self.num_phases

        self._step += 1
        return (phase, None)

    def reset(self):
        """Reset the internal step counter between episodes."""
        self._step = 0

    def run_episode(self, env):
        """Run one full episode and return summary metrics."""
        self.reset()
        obs, info = env.reset()

        total_reward = 0.0
        n_steps = 0
        done = False
        info_history = []

        while not done:
            action, _ = self.predict(obs)
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
