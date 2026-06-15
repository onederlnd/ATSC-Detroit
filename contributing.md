# Contributing to ATSC-Detroit

Thanks for your interest in contributing. This document covers how to get set up, how the codebase is organized, and what kinds of contributions are most useful.

---

## Getting Started

### Prerequisites

- Python 3.9+
- SUMO traffic simulator
- Git

### Setup

```bash
# 1. Fork and clone the repo
git clone https://github.com/your-username/ATSC-Detroit.git
cd ATSC-Detroit

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install SUMO (Ubuntu/Debian)
sudo apt install sumo sumo-tools sumo-doc
export SUMO_HOME=/usr/share/sumo

# 5. Verify the setup
python main.py baseline --episodes 1
```

If the baseline runs without error, you're set up correctly.

---

## Project Structure

```
ATSC-Detroit/
├── agents/         # Agent classes (FixedTimingAgent, RLAgent)
├── env/            # Gymnasium environment wrapping SUMO
├── simulation/     # SUMO network, routes, detectors, config
├── training/       # Training, evaluation, and tuning scripts
├── utils/          # Shared metrics and plotting
├── tests/          # pytest test suite
└── main.py         # CLI entry point
```

Read `simulation/README.md` for details on the network geometry and demand calibration.

---

## Running Tests

All contributions should pass the test suite before submitting a PR:

```bash
python -m pytest tests/ -v
```

Tests require SUMO to be installed and `SUMO_HOME` to be set. Tests that require a trained model are automatically skipped if no model is found.

---

## What to Work On

The most impactful areas for contribution:

**Detroit-specific improvements:**

- Replace the 2x2 grid network with real Michigan Ave & Livernois geometry from OpenStreetMap
- Calibrate demand with actual MDOT/SEMCOG count data for the intersection
- Add pedestrian phase handling

**RL improvements:**

- Compare PPO against DQN and A2C
- Multi-intersection coordination along the Michigan Ave corridor
- Add time-of-day and weather demand variation

**Infrastructure:**

- Export trained policy to ONNX for deployment
- Add a web dashboard for visualizing agent performance
- CI/CD pipeline with GitHub Actions

Please open an issue before starting a large piece of work so we can discuss approach and avoid duplicated effort.

---

## Submitting a PR

1. Create a branch from `main`: `git checkout -b feature/your-feature`
2. Make your changes
3. Run the test suite: `python -m pytest tests/ -v`
4. Commit with a clear message: `git commit -m "Add pedestrian phase handling"`
5. Push and open a PR against `main`

PR descriptions should explain what changed, why, and how to test it.

---

## Code Style

- Follow existing patterns in each file
- Add docstrings to all public classes and functions
- Keep constants at the top of files, not buried in functions
- Prefer clarity over cleverness

---

## Questions

Open a GitHub issue with the `question` label.
