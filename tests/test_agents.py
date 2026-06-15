"""
test_agents.py
--------------
Unit tests for FixedTimingAgent and RLAgent.

Run:
    python -m pytest tests/test_agents.py -v
"""

import pytest
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.fixed_baseline import FixedTimingAgent
from env.traffic_env import NUM_PHASES, MIN_GREEN_STEPS


# ---------------------------------------------------------------------------
# FixedTimingAgent tests
# ---------------------------------------------------------------------------


@pytest.fixture
def fixed_agent():
    return FixedTimingAgent(num_phases=NUM_PHASES, cycle_length=None)


def test_fixed_agent_predict_returns_valid_action(fixed_agent):
    """predict() should return a valid phase index and None."""
    obs = np.zeros(9, dtype=np.float32)
    action, states = fixed_agent.predict(obs)
    assert 0 <= action < NUM_PHASES, f"Action {action} out of valid range"
    assert states is None


def test_fixed_agent_cycles_phases(fixed_agent):
    """Agent should cycle through all phases given enough steps."""
    obs = np.zeros(9, dtype=np.float32)
    seen_phases = set()

    for _ in range(NUM_PHASES * fixed_agent.cycle_length):
        action, _ = fixed_agent.predict(obs)
        seen_phases.add(action)

    assert seen_phases == set(range(NUM_PHASES)), (
        f"Not all phases were visited. Saw: {seen_phases}"
    )


def test_fixed_agent_reset_clears_step_counter(fixed_agent):
    """After reset(), the agent should produce the same action sequence."""
    obs = np.zeros(9, dtype=np.float32)

    # Record first N actions
    first_run = []
    for _ in range(fixed_agent.cycle_length * 2):
        action, _ = fixed_agent.predict(obs)
        first_run.append(action)

    # Reset and record again
    fixed_agent.reset()
    second_run = []
    for _ in range(fixed_agent.cycle_length * 2):
        action, _ = fixed_agent.predict(obs)
        second_run.append(action)

    assert first_run == second_run, "Action sequence differed after reset"


def test_fixed_agent_run_episode_returns_expected_keys(env):
    """run_episode() should return a dict with the required keys."""
    agent = FixedTimingAgent(num_phases=NUM_PHASES, cycle_length=None)
    result = agent.run_episode(env)

    assert "total_wait" in result
    assert "total_reward" in result
    assert "n_steps" in result
    assert "info_history" in result


def test_fixed_agent_run_episode_info_history_not_empty(env):
    """info_history should contain at least one entry after an episode."""
    agent = FixedTimingAgent(num_phases=NUM_PHASES, cycle_length=None)
    result = agent.run_episode(env)
    assert len(result["info_history"]) > 0


def test_fixed_agent_run_episode_n_steps_matches_history(env):
    """n_steps should equal the length of info_history."""
    agent = FixedTimingAgent(num_phases=NUM_PHASES, cycle_length=None)
    result = agent.run_episode(env)
    assert result["n_steps"] == len(result["info_history"])


# ---------------------------------------------------------------------------
# RLAgent tests
# ---------------------------------------------------------------------------


def test_rl_agent_raises_on_missing_model():
    """RLAgent should raise FileNotFoundError if the model file doesn't exist."""
    from agents.rl_agent import RLAgent

    with pytest.raises(FileNotFoundError):
        RLAgent(model_path="models/nonexistent_model.zip")


def test_rl_agent_model_path_property():
    """model_path property should return the path the agent was loaded from."""
    from agents.rl_agent import RLAgent

    path = "models/nonexistent.zip"
    with pytest.raises(FileNotFoundError):
        agent = RLAgent(model_path=path)
    # We can't test the property without a real model,
    # but we can confirm the error message contains the path.
    try:
        RLAgent(model_path=path)
    except FileNotFoundError as e:
        assert path in str(e)


@pytest.mark.skipif(
    not os.path.exists("models/atsc_ppo_final.zip"),
    reason="Trained model not found — run 'python main.py train' first",
)
def test_rl_agent_predict_returns_valid_action():
    """With a real model, predict() should return a valid action."""
    from agents.rl_agent import RLAgent

    agent = RLAgent(model_path="models/atsc_ppo_final.zip")
    obs = np.zeros(9, dtype=np.float32)
    action, states = agent.predict(obs)
    assert 0 <= int(action) < NUM_PHASES
    assert states is None


@pytest.mark.skipif(
    not os.path.exists("models/atsc_ppo_final.zip"),
    reason="Trained model not found — run 'python main.py train' first",
)
def test_rl_agent_run_episode_returns_expected_keys(env):
    """run_episode() should return a dict with the required keys."""
    from agents.rl_agent import RLAgent

    agent = RLAgent(model_path="models/atsc_ppo_final.zip")
    result = agent.run_episode(env)

    assert "total_wait" in result
    assert "total_reward" in result
    assert "n_steps" in result
    assert "info_history" in result
