# ATSC-Detroit — Build Roadmap

A phase-by-phase guide to completing the project from current state to a
fully trained, evaluated, and documented RL-based adaptive traffic signal
controller.

---

## Current State

| File | Status |
| --- | --- |
| `env/traffic_env.py` | Complete |
| `training/train.py` | Complete |
| `simulation/*.xml / *.sumocfg` | Complete |
| `agents/rl_agent.py` | Empty |
| `agents/fixed_baseline.py` | Empty |
| `training/evaluate.py` | Empty |
| `utils/metrics.py` | Empty |
| `main.py` | Empty |
| `tests/` | Empty |

---

## Phase 1 — Complete the Core Agents

These are needed before anything can run end-to-end.

### 1.1 `agents/fixed_baseline.py`

A clean, reusable class wrapping the fixed-timing logic that currently
lives inline inside `train.py`. Extracting it here makes the codebase
easier to extend with other baselines later (Webster, SCATS-style, etc.).

Tasks:

- [x] Define a `FixedTimingAgent` class with a `predict(obs)` method
      that mirrors the SB3 model API so it's drop-in replaceable
- [x] Accept `cycle_length` and `num_phases` as constructor arguments
- [x] Add a `run_episode(env)` helper that returns total wait and reward
- [x] Remove the inline baseline logic from `train.py` and import this instead

### 1.2 `agents/rl_agent.py`

A thin wrapper around the SB3 PPO model for inference and loading.
Training lives in `train.py` — this file is for deployment-time use.

Tasks:

- [x] Define an `RLAgent` class that loads a saved `.zip` model
- [x] Expose a `predict(obs, deterministic=True)` method
- [x] Add a `run_episode(env)` helper (mirrors `FixedTimingAgent`)
- [x] Handle missing model file gracefully with a clear error message

---

## Phase 2 — Metrics and Evaluation

### 2.1 `utils/metrics.py`

Central place for computing and logging performance numbers. Both the
baseline and the RL agent will use this.

Tasks:

- [x] `compute_episode_metrics(info_history)` — takes the list of `info`
      dicts returned by `env.step()` and returns a summary dict:
      `total_wait`, `avg_wait_per_step`, `max_wait_step`, `n_phases_switched`
- [x] `compare_agents(baseline_results, rl_results)` — prints a formatted
      side-by-side table and returns improvement percentage
- [x] `plot_wait_over_time(info_history, label, save_path=None)` — line
      chart of waiting vehicles per step using matplotlib
- [x] `plot_phase_distribution(info_history, save_path=None)` — bar chart
      showing how often each phase was active
- [x] Add an `output/` directory to `.gitignore` since plots get saved there

### 2.2 `training/evaluate.py`

Standalone script to evaluate any saved model (or the baseline) without
re-running training.

Tasks:

- [x] Accept `--model` path argument (or `--baseline` flag) via `argparse`
- [x] Accept `--episodes` argument (default 5)
- [x] Accept `--gui` flag to watch the sim
- [x] Run the specified agent for N episodes using the helpers from
      `agents/` and `utils/metrics.py`
- [x] Print the metrics table and save plots to `output/`
- [x] Example usage in a docstring at the top of the file

---

## Phase 3 — Entry Point

### 3.1 `main.py`

Top-level script that ties everything together. Should be the only file a
new user needs to read to understand how to operate the project.

Tasks:

- [x] Subcommands via `argparse`: `train`, `evaluate`, `baseline`, `watch`
- [x] `train` — calls `training/train.py:train()`
- [x] `evaluate` — calls `training/evaluate.py` with a model path
- [x] `baseline` — runs just the fixed-timing agent for N episodes
- [x] `watch` — loads a saved model and runs one episode with `use_gui=True`
- [x] Print a helpful usage block when called with no arguments

Intended usage after this phase:

```bash
python3 main.py train
python3 main.py evaluate --model models/atsc_ppo_final.zip
python3 main.py watch --model models/atsc_ppo_final.zip
python3 main.py baseline --episodes 5
```

---

## Phase 4 — Tests

### 4.1 `tests/test_env.py`

Tasks:

- [x] Test that `TrafficSignalEnv` resets without error
- [x] Test that observation shape matches `observation_space`
- [x] Test that all actions in `action_space` are accepted without crash
- [x] Test that `MIN_GREEN_STEPS` is respected (agent can't switch phase
      before minimum green time elapses)
- [x] Test that `MAX_GREEN_STEPS` forces a transition
- [x] Test that yellow phase duration is always exactly `YELLOW_STEPS`
- [x] Test that `terminated` is eventually `True` within `MAX_STEPS`

### 4.2 `tests/test_agents.py`

Tasks:

- [x] Test that `FixedTimingAgent.predict()` returns a valid action for any obs
- [x] Test that `RLAgent` raises a clear error when model file is missing
- [x] Test that `run_episode()` returns the expected keys in the result dict

### 4.3 `tests/test_metrics.py`

Tasks:

- [x] Test `compute_episode_metrics()` with a synthetic `info_history`
- [x] Test that `compare_agents()` computes improvement percentage correctly
- [x] Test that plot functions run without error (use a temp directory for saves)

Run all tests:

```bash
python -m pytest tests/ -v
```

---

## Phase 5 — Detroit Demand Calibration

The current `routes.rou.xml` likely uses generic or placeholder demand.
This phase makes the simulation Detroit-specific.

Tasks:

- [x] Identify the target intersection (e.g. Michigan Ave & Livernois,
      Woodward & Grand Blvd, or a MDOT-counted location)
- [x] Pull hourly volume counts from MDOT Traffic Monitoring or SEMCOG
- [x] Rewrite `routes.rou.xml` to reflect:
      - AM peak (7–9 AM) volume
      - PM peak (4–6 PM) volume
      - Off-peak baseline volume
      - Realistic directional splits (not uniform 25% per approach)
- [x] Add a `simulation/demand/` subfolder with raw count data and a
      generation script so the `.rou.xml` is reproducible
- [x] Validate that SUMO trip counts roughly match real-world volumes

---

## Phase 6 — Extended Training and Tuning

Run once the full pipeline is wired up and tests pass.

Tasks:

- [ ] Increase `TOTAL_TIMESTEPS` to 1–2M and re-run training
- [ ] Try `n_envs=4` in `make_vec_env` to speed up data collection
- [ ] Tune `learning_rate` — try 1e-4 and 5e-4 alongside default 3e-4
- [ ] Tune `gamma` — try 0.95 for shorter planning horizon
- [ ] Experiment with reward shaping: penalize phase switches to encourage
      stability (add a small negative reward each time the phase changes)
- [ ] Log all hyperparameter runs to TensorBoard with distinct run names
- [ ] Document the best-performing config in `training/train.py` comments

---

## Phase 7 — Results and Documentation

### 7.1 Results

- [ ] Run baseline for 10 episodes and record mean ± std of total wait
- [ ] Run best trained model for 10 episodes and record same
- [ ] Fill in the Results table in `README.md` with real numbers
- [ ] Save representative plots to `output/` and reference them in the README

### 7.2 Documentation

- [ ] Add docstrings to every public class and function that doesn't have one
- [ ] Write `simulation/README.md` explaining the network geometry, detector
      placement, and demand calibration approach
- [ ] Add a `CONTRIBUTING.md` with setup instructions for new contributors
- [ ] Tag `v0.1.0` once Phase 4 tests pass and Phase 6 produces a positive result

---

## Suggested Build Order

```bash
Phase 1 → Phase 2 → Phase 3 → Phase 4 → (first full run)
                                              ↓
                                         Phase 5 → Phase 6 → Phase 7
```

Complete Phases 1–4 first so you have a working, tested pipeline end-to-end
— even with placeholder demand. Then layer in Detroit-specific calibration
and extended training once you know the plumbing works.
