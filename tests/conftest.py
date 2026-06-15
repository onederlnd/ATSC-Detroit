"""
conftest.py
-----------
Shared pytest fixtures for ATSC-Detroit tests.

Fixtures defined here are automatically available to all test files
without needing to import them.
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.traffic_env import TrafficSignalEnv


@pytest.fixture
def env():
    """
    Create a fresh TrafficSignalEnv for each test and close it after.

    Used by:
        - test_env.py
        - test_agents.py (for run_episode tests)
    """
    e = TrafficSignalEnv(use_gui=False)
    yield e
    e.close()
