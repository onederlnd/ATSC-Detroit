# ATSC-Detroit

## Adaptive Traffic Signal Control for Detroit Intersections using Reinforcement Learning

A research and engineering project that trains a PPO-based RL agent to control traffic signal timing at a real-world-inspired Detroit intersection, using the [SUMO](https://sumo.dlr.de/) traffic simulator and [Stable-Baselines3](https://stable-baselines3.readthedocs.io/).

The agent learns to minimize vehicle waiting time by dynamically adjusting signal phases in response to real-time queue conditions — outperforming conventional fixed-timing controllers under variable demand.

---

## Table of Contents

- [Background](#background)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Configuration](#configuration)
- [Results & Metrics](#results--metrics)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Background

Detroit's street grid faces persistent congestion challenges — aging signal infrastructure, high pedestrian volumes near downtown, and heavy freight corridors create inefficiencies that fixed-cycle controllers can't adapt to. This project models the Michigan Ave & Livernois intersection and asks:

> **Can a reinforcement learning agent learn a signal control policy that reduces vehicle delay compared to a fixed-timing baseline?**

The simulation uses SUMO (Simulation of Urban MObility), an open-source microscopic traffic simulator trusted in both academic research and municipal transportation planning. Detector placement and demand patterns are calibrated to reflect Detroit-area conditions using MDOT AADT estimates (~20,000 vpd on Michigan Ave, ~10,000 vpd on Livernois).

---

## How It Works

### The Control Problem

A traffic signal controller decides *which phase to display* (e.g. North-South green, East-West green) and *for how long*. A fixed-timing controller uses a pre-set cycle regardless of actual traffic. An adaptive controller reads real-time detector data and adjusts accordingly.

### Observation Space

At each timestep, the agent observes:

| Component | Size | Description |
| --- | --- | --- |
| Queue lengths | 4 | Halted vehicles per approach, normalized 0–1 |
| Current phase | 4 | One-hot encoding of active signal phase |
| Phase elapsed time | 1 | How long the current phase has been active, normalized 0–1 |

**Total:** 9-dimensional continuous observation vector.

### Action Space

`Discrete(4)` — the agent selects which of the 4 signal phases to activate next. Yellow transitions are handled automatically by the environment (not the agent), enforcing a fixed 4-second yellow before any phase change.

### Reward

```bash
reward = -(number of vehicles halted across all approach detectors)
```

A phase-switch penalty of -2.0 is applied each time the active phase changes, encouraging signal stability.

### Safety Constraints

The environment enforces real-world controller constraints regardless of what the agent requests:

- **Minimum green time:** 10 seconds (prevents rapid flicker)
- **Maximum green time:** 60 seconds (prevents indefinite starvation)
- **Yellow phase:** Always 4 seconds between any green-to-green transition

### RL Algorithm

[PPO (Proximal Policy Optimization)](https://arxiv.org/abs/1707.06347) — a robust, sample-efficient on-policy algorithm well-suited to discrete action spaces with dense rewards. Implemented via Stable-Baselines3.

Best hyperparameters (from `training/tune.py` grid search, June 2026):

| Parameter | Value |
| --- | --- |
| Learning rate | 5e-4 |
| Discount factor (γ) | 0.95 |
| Reward shaping | Yes (phase switch penalty = -2.0) |
| Steps per update | 2048 |
| Batch size | 64 |
| Training timesteps | 2,000,000 |

---

## Project Structure

```bash
ATSC-Detroit/
│
├── agents/
│   ├── rl_agent.py          # PPO agent wrapper / inference helpers
│   └── fixed_baseline.py    # Fixed-cycle timing baseline for comparison
│
├── env/
│   └── traffic_env.py       # Gymnasium environment wrapping SUMO via TraCI
│
├── simulation/
│   ├── intersection.net.xml  # SUMO road network (Detroit intersection geometry)
│   ├── routes.rou.xml        # Vehicle demand / route definitions
│   ├── detectors.add.xml     # Inductive loop detector placement
│   ├── intersection.sumocfg  # Master SUMO configuration file
│   └── demand/
│       ├── generate_demand.py  # Regenerates routes.rou.xml from AADT parameters
│       └── demand_sources.md   # Data sources and methodology
│
├── training/
│   ├── train.py              # Main training script (PPO + baseline comparison)
│   ├── evaluate.py           # Evaluation and metrics reporting
│   ├── tune.py               # Hyperparameter grid search
│   └── compare_runs.py       # Ranks tuned models by performance
│
├── utils/
│   └── metrics.py            # Metric collection, logging, and plotting helpers
│
├── tests/
│   ├── conftest.py           # Shared pytest fixtures
│   ├── test_env.py           # Environment tests
│   ├── test_agents.py        # Agent tests
│   └── test_metrics.py       # Metrics tests
│
├── models/                   # Saved model checkpoints (created at runtime)
├── logs/                     # TensorBoard training logs (created at runtime)
├── output/                   # Plots and SUMO output files (created at runtime)
├── main.py                   # Top-level entry point
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.9+
- [SUMO](https://sumo.dlr.de/docs/Installing/index.html) — install via your OS package manager or the official installer
- `SUMO_HOME` environment variable set

```bash
# Ubuntu/Debian
sudo apt install sumo sumo-tools sumo-doc
export SUMO_HOME=/usr/share/sumo

# macOS (Homebrew)
brew install sumo
export SUMO_HOME=/opt/homebrew/opt/sumo/share/sumo

# Add to your shell profile (.bashrc / .zshrc) to persist:
echo 'export SUMO_HOME=/usr/share/sumo' >> ~/.bashrc
```

### Python Dependencies

```bash
# Clone the repo
git clone https://github.com/your-username/ATSC-Detroit.git
cd ATSC-Detroit

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

`requirements.txt` should include at minimum:

```bash
gymnasium
stable-baselines3[extra]
torch
traci
numpy
matplotlib
```

---

## Running the Project

### Train the RL Agent

```bash
python main.py train
```

### Evaluate a Saved Model

```bash
python main.py evaluate --model models/tune_lr0.0005_gamma0.95_rs1/final.zip --episodes 5
```

### Run the Fixed Baseline

```bash
python main.py baseline --episodes 5
```

### Compare Agent vs Baseline

```bash
python main.py compare --model models/tune_lr0.0005_gamma0.95_rs1/final.zip --episodes 5
```

### Watch the Agent in SUMO GUI

```bash
python main.py watch --model models/tune_lr0.0005_gamma0.95_rs1/final.zip
```

### Run Hyperparameter Tuning

```bash
python training/tune.py          # ~6 hours for full grid
python training/compare_runs.py  # rank results
```

### Visualize Training in TensorBoard

```bash
tensorboard --logdir logs/
```

### Run Tests

```bash
python -m pytest tests/ -v
```

---

## Configuration

All key parameters live as constants at the top of their respective files.

| Parameter | File | Value | Description |
| --- | --- | --- | --- |
| `TL_ID` | `traffic_env.py` | `"A0"` | Traffic light ID in SUMO network |
| `NUM_PHASES` | `traffic_env.py` | `4` | Number of signal phases |
| `MAX_STEPS` | `traffic_env.py` | `3600` | Steps per episode (1 simulated hour) |
| `MIN_GREEN_STEPS` | `traffic_env.py` | `10` | Minimum green time (seconds) |
| `MAX_GREEN_STEPS` | `traffic_env.py` | `60` | Maximum green time (seconds) |
| `YELLOW_STEPS` | `traffic_env.py` | `4` | Fixed yellow duration (seconds) |
| `TOTAL_TIMESTEPS` | `train.py` | `2,000,000` | Total RL training steps |
| `PHASE_SWITCH_PENALTY` | `tune.py` | `-2.0` | Reward penalty per phase switch |

---

## Results & Metrics

Evaluated at **Michigan Ave & Livernois, Detroit** — AM peak demand, 5 episodes each.

| Agent | Avg Total Wait | Std | Phase Switches/Episode |
| --- | --- | --- | --- |
| Fixed-timing baseline | 6,850 veh-steps | 0 | — |
| PPO agent (best config) | 6,766 veh-steps | 0 | 135 |
| **Improvement** | **1.2%** | — | — |

**Best config:** `lr=5e-4, gamma=0.95, reward_shaping=True` (from 12-config grid search)

**Notes:**

- Std=0 reflects deterministic policy on identical demand — expected behavior
- The 1.2% improvement is modest on this simple 2x2 grid network
- Phase switch reduction (135 vs 257 in unshapen configs) is a meaningful operational win
- Improvement expected to increase with real Detroit geometry and asymmetric demand

---

## Roadmap

- [ ] Multi-intersection coordination (arterial corridor)
- [ ] Pedestrian phase handling
- [ ] Real Detroit demand calibration from MDOT/SEMCOG count data
- [ ] Compare PPO vs DQN vs A2C
- [ ] Export learned policy to a deployable format (ONNX)
- [ ] Add weather/time-of-day demand variation
- [ ] Replace 2x2 grid with actual Michigan Ave & Livernois geometry

---

## Contributing

Contributions welcome — especially around Detroit-specific demand modeling, network expansion, or alternative RL algorithms. Please open an issue before submitting a large PR.

```bash
# Run tests
python -m pytest tests/ -v
```

---

## License

MIT License. See `LICENSE` for details.

---

*Built in Detroit. For Detroit.*
