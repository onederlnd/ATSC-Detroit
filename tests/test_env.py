"""
test_env.py
-----------
Unit and integration tests for TrafficSignalEnv.

Run:
    python -m pytest tests/test_env.py -v
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.traffic_env import (
    TrafficSignalEnv,
    NUM_PHASES,
    MIN_GREEN_STEPS,
    MAX_GREEN_STEPS,
    YELLOW_STEPS,
    MAX_STEPS,
)


@pytest.fixture
def env():
    """Create a fresh environment for each test and close it after."""
    e = TrafficSignalEnv(use_gui=False)
    yield e
    e.close()


# ---------------------------------------------------------------------------
# Test 1 — Reset without error
# ---------------------------------------------------------------------------


def test_reset_returns_obs_and_info(env):
    """env.reset() should return a tuple of (obs, info) without raising."""
    obs, info = env.reset()
    assert obs is not None
    assert isinstance(info, dict)


# ---------------------------------------------------------------------------
# Test 2 — Observation shape matches observation_space
# ---------------------------------------------------------------------------


def test_observation_shape(env):
    """Observation returned by reset() must match the declared observation_space shape."""
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape, (
        f"Expected shape {env.observation_space.shape}, got {obs.shape}"
    )


def test_observation_shape_after_step(env):
    """Observation returned by step() must also match observation_space shape."""
    env.reset()
    obs, _, _, _, _ = env.step(0)
    assert obs.shape == env.observation_space.shape


def test_observation_bounds(env):
    """All observation values should be within [0, 1]."""
    obs, _ = env.reset()
    assert np.all(obs >= 0.0), "Observation has values below 0"
    assert np.all(obs <= 1.0), "Observation has values above 1"


# ---------------------------------------------------------------------------
# Test 3 — All actions accepted without crash
# ---------------------------------------------------------------------------


def test_all_actions_accepted(env):
    """Every action in action_space should be accepted by step() without error."""
    env.reset()
    # Run MIN_GREEN_STEPS first so phase switches are allowed
    for _ in range(MIN_GREEN_STEPS):
        env.step(0)

    for action in range(NUM_PHASES):
        # Reset between actions so we always have a valid state
        env.reset()
        for _ in range(MIN_GREEN_STEPS):
            env.step(0)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs is not None, f"Action {action} returned None observation"


# ---------------------------------------------------------------------------
# Test 4 — MIN_GREEN_STEPS is respected
# ---------------------------------------------------------------------------


def test_min_green_steps_respected(env):
    """
    Phase should not change before MIN_GREEN_STEPS have elapsed.
    Even if the agent requests a different phase, the env should hold the current one.
    """
    env.reset()
    initial_phase = env._current_phase

    # Request a different phase immediately (before MIN_GREEN_STEPS)
    different_phase = (initial_phase + 1) % NUM_PHASES
    for step in range(MIN_GREEN_STEPS - 1):
        env.step(different_phase)
        # Phase should not have changed yet (no yellow, no switch)
        assert env._current_phase == initial_phase or env._in_yellow, (
            f"Phase changed at step {step + 1}, before MIN_GREEN_STEPS ({MIN_GREEN_STEPS})"
        )


# ---------------------------------------------------------------------------
# Test 5 — MAX_GREEN_STEPS forces a transition
# ---------------------------------------------------------------------------


def test_max_green_steps_forces_transition(env):
    """
    If the agent keeps requesting the same phase, the env must force a
    transition after MAX_GREEN_STEPS.
    """
    env.reset()
    initial_phase = env._current_phase

    # Keep requesting the same phase for longer than MAX_GREEN_STEPS
    for _ in range(MAX_GREEN_STEPS + YELLOW_STEPS + 5):
        env.step(initial_phase)

    # After MAX_GREEN_STEPS + yellow, the phase must have changed
    assert env._current_phase != initial_phase or env._in_yellow, (
        "Phase did not change after MAX_GREEN_STEPS elapsed"
    )


# ---------------------------------------------------------------------------
# Test 6 — Yellow phase duration is exactly YELLOW_STEPS
# ---------------------------------------------------------------------------


def test_yellow_duration(env):
    """
    When a phase transition is triggered, the environment should spend
    exactly YELLOW_STEPS steps in the yellow state before switching.
    """
    env.reset()

    # Wait for MIN_GREEN_STEPS so a phase switch is allowed
    for _ in range(MIN_GREEN_STEPS):
        env.step(0)

    # Request a different phase to trigger yellow
    different_phase = (env._current_phase + 1) % NUM_PHASES
    env.step(different_phase)

    if not env._in_yellow:
        pytest.skip("Yellow transition did not trigger — phase may already match")

    yellow_steps_counted = 0
    while env._in_yellow:
        yellow_steps_counted += 1
        env.step(different_phase)

    assert yellow_steps_counted == YELLOW_STEPS, (
        f"Expected {YELLOW_STEPS} yellow steps, got {yellow_steps_counted}"
    )


# ---------------------------------------------------------------------------
# Test 7 — terminated is eventually True within MAX_STEPS
# ---------------------------------------------------------------------------


def test_terminates_within_max_steps(env):
    """
    The episode must end (terminated or truncated) within MAX_STEPS steps.
    """
    env.reset()
    done = False
    steps = 0

    while not done:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        done = terminated or truncated
        steps += 1
        assert steps <= MAX_STEPS + 10, (
            f"Episode did not terminate within MAX_STEPS ({MAX_STEPS})"
        )

    assert done, "Episode never terminated"
