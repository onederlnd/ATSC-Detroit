"""
traffic_env.py
--------------
A Gymnasium environment that wraps a SUMO simulation via TraCI.
Trains an RL agent to control a single intersection's signal phases.

Dependencies:
    pip install gymnasium stable-baselines3 sumo-rl
    (SUMO must also be installed: https://sumo.dlr.de/docs/Installing/index.html)

Usage:
    from env.traffic_env import TrafficSignalEnv
    env = TrafficSignalEnv()
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# SUMO_HOME must be set in your environment (e.g. /usr/share/sumo or /opt/homebrew/opt/sumo)
if "SUMO_HOME" not in os.environ:
    raise EnvironmentError(
        "SUMO_HOME is not set. Install SUMO and set: export SUMO_HOME=/path/to/sumo"
    )

tools_path = os.path.join(os.environ["SUMO_HOME"], "tools")
if tools_path not in sys.path:
    sys.path.append(tools_path)

import traci  # noqa: E402 — must come after sys.path update


# ---------------------------------------------------------------------------
# Constants — adjust these to match your actual SUMO network
# ---------------------------------------------------------------------------

# Path to your SUMO config file
SUMO_CFG = os.path.join(
    os.path.dirname(__file__), "..", "simulation", "intersection.sumocfg"
)

# The traffic light ID in your SUMO network (check your .net.xml)
TL_ID = "A0"

# How many signal phases your intersection has (e.g. 4 for a basic 4-way)
NUM_PHASES = 4

# Detector IDs — inductive loop detectors on each approach lane
# These must match detector IDs defined in your SUMO network/additional files
DETECTOR_IDS = [
    "det_north_in",
    "det_south_in",
    "det_east_in",
    "det_west_in",
]

# Simulation step length (seconds) — must match SUMO config
STEP_LENGTH = 1.0

# Max simulation steps per episode
MAX_STEPS = 3600  # 1 simulated hour

# Min/max green time per phase (safety constraints — real controllers enforce these)
MIN_GREEN_STEPS = 10  # 10 seconds minimum green
MAX_GREEN_STEPS = 60  # 60 seconds maximum green

# Yellow phase duration (fixed — not controlled by AI)
YELLOW_STEPS = 4


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class TrafficSignalEnv(gym.Env):
    """
    Single-intersection adaptive traffic signal control environment.

    Observation space:
        - Queue length on each approach (normalized 0-1)
        - Current phase index (one-hot encoded)
        - Time elapsed in current phase (normalized 0-1)

    Action space:
        Discrete — which phase to switch to next.
        The environment handles the yellow transition automatically.

    Reward:
        Negative sum of waiting vehicles across all detectors.
        Minimizing wait time = maximizing reward.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, use_gui: bool = False, render_mode=None):
        super().__init__()

        self.use_gui = use_gui
        self.render_mode = render_mode
        self._sumo_running = False

        # --- Observation space ---
        # [queue per detector (normalized)] + [current phase one-hot] + [phase elapsed time]
        obs_size = len(DETECTOR_IDS) + NUM_PHASES + 1
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_size,),
            dtype=np.float32,
        )

        # --- Action space ---
        # Agent picks which phase to activate next
        self.action_space = spaces.Discrete(NUM_PHASES)

        # Internal state
        self._step_count = 0
        self._current_phase = 0
        self._phase_step_count = 0  # Steps spent in current phase
        self._in_yellow = False
        self._yellow_step_count = 0
        self._pending_phase = None  # Phase waiting behind a yellow

        # Metrics tracking
        self._total_wait = 0.0
        self._episode_rewards = []

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self._sumo_running:
            traci.close()
            self._sumo_running = False

        self._start_sumo()

        self._step_count = 0
        self._current_phase = 0
        self._phase_step_count = 0
        self._in_yellow = False
        self._yellow_step_count = 0
        self._pending_phase = None
        self._total_wait = 0.0

        obs = self._get_observation()
        info = {}
        return obs, info

    def step(self, action: int):
        assert self._sumo_running, "Call reset() before step()."
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # --- Phase transition logic ---
        if not self._in_yellow:
            if (
                action != self._current_phase
                and self._phase_step_count >= MIN_GREEN_STEPS
            ):
                # Start yellow transition
                self._in_yellow = True
                self._yellow_step_count = 0
                self._pending_phase = action
                self._set_yellow()
            elif self._phase_step_count >= MAX_GREEN_STEPS:
                # Force transition at max green (safety constraint)
                next_phase = (self._current_phase + 1) % NUM_PHASES
                self._in_yellow = True
                self._yellow_step_count = 0
                self._pending_phase = next_phase
                self._set_yellow()
        else:
            self._yellow_step_count += 1
            if self._yellow_step_count >= YELLOW_STEPS:
                # Yellow done — switch to pending phase
                self._in_yellow = False
                self._current_phase = self._pending_phase
                self._pending_phase = None
                self._phase_step_count = 0
                traci.trafficlight.setPhase(TL_ID, self._current_phase)

        # Advance simulation one step
        traci.simulationStep()
        self._step_count += 1
        self._phase_step_count += 1

        # --- Reward ---
        wait = self._get_total_waiting_vehicles()
        reward = -float(wait)  # Minimize waiting vehicles
        self._total_wait += wait

        # --- Termination ---
        terminated = traci.simulation.getMinExpectedNumber() <= 0
        truncated = self._step_count >= MAX_STEPS

        obs = self._get_observation()
        info = {
            "step": self._step_count,
            "current_phase": self._current_phase,
            "waiting_vehicles": wait,
            "total_wait_so_far": self._total_wait,
        }

        if terminated or truncated:
            traci.close()
            self._sumo_running = False

        return obs, reward, terminated, truncated, info

    def render(self):
        # GUI rendering is handled by SUMO itself when use_gui=True
        pass

    def close(self):
        if self._sumo_running:
            traci.close()
            self._sumo_running = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _start_sumo(self):
        """Launch SUMO as a subprocess and connect TraCI."""
        binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_cmd = [
            binary,
            "-c",
            SUMO_CFG,
            "--step-length",
            str(STEP_LENGTH),
            "--no-warnings",  # Suppress SUMO console noise
            "--waiting-time-memory",
            "1000",
        ]
        traci.start(sumo_cmd)
        self._sumo_running = True
        traci.trafficlight.setPhase(TL_ID, self._current_phase)

    def _get_observation(self) -> np.ndarray:
        """Build the observation vector."""
        # Queue lengths — vehicles halted on each approach (normalized to 0-1)
        max_queue = 20.0  # Normalization constant — tune to your intersection
        queues = []
        for det_id in DETECTOR_IDS:
            try:
                # getLastStepHaltingNumber: vehicles stopped at detector
                halted = traci.inductionloop.getLastStepVehicleNumber(det_id)
            except traci.exceptions.TraCIException:
                halted = 0
            queues.append(min(halted / max_queue, 1.0))

        # Current phase — one-hot encoded
        phase_one_hot = [0.0] * NUM_PHASES
        phase_one_hot[self._current_phase] = 1.0

        # Time in current phase (normalized)
        phase_time_norm = min(self._phase_step_count / MAX_GREEN_STEPS, 1.0)

        obs = np.array(queues + phase_one_hot + [phase_time_norm], dtype=np.float32)
        return obs

    def _get_total_waiting_vehicles(self) -> int:
        """Sum of halted vehicles across all monitored detectors."""
        total = 0
        for det_id in DETECTOR_IDS:
            try:
                total += traci.inductionloop.getLastStepVehicleNumber(det_id)
            except traci.exceptions.TraCIException:
                pass
        return total

    def _set_yellow(self):
        """
        Switch to the yellow phase for the current direction.
        In SUMO, yellow phases are typically interleaved between green phases.
        This is a simplified approach — production systems use explicit yellow phase IDs.
        """
        ## A common SUMO convention: yellow phase = current_phase * 2 + 1
        ## Adjust based on your actual phase structure in the .net.xml
        # yellow_phase_id = self._current_phase * 2 + 1
        # try:
        #    traci.trafficlight.setPhase(TL_ID, yellow_phase_id)
        # except traci.exceptions.TraCIException:
        #    # Fallback: just hold current phase if yellow phase ID doesn't exist
        pass
