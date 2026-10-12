# HIRO Meta-Period Study (PointMaze)

A PyTorch implementation of HIRO (Hierarchical Reinforcement Learning with Off-policy Correction, [Nachum et al., 2018](https://arxiv.org/abs/1805.08296)) with TD3 at both levels. It comes with the experiment pipeline from my MSc thesis, which studies how the **meta-period** _c_ affects learning across three PointMaze mazes of increasing difficulty. The meta-period is the number of low-level steps between high-level subgoal decisions.

The pipeline trains a grid of (maze × meta-period × seed) runs, logs every run to its own folder, and produces learning curves, sensitivity plots and a results table.

## Contents

| File                      | Purpose                                                                                                                                 |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `hiro_agent.py`           | Two-level HIRO agent: subgoal selection, subgoal transition, intrinsic reward, off-policy correction, TD3 updates                       |
| `td3.py`                  | TD3 controller (twin critics, delayed actor updates, target policy smoothing)                                                           |
| `networks.py`             | Goal-conditioned actor and critic MLPs                                                                                                  |
| `replay_buffer.py`        | Base replay buffer                                                                                                                      |
| `hiro_buffers.py`         | Low-level buffer (with subgoals) and high-level buffer (with the per-window state/action trajectories needed for off-policy correction) |
| `subgoal.py`              | Bounded subgoal space                                                                                                                   |
| `train_pointmaze_hiro.py` | Training and evaluation for one configuration                                                                                           |
| `run_sweep.py`            | Runs the full meta-period × seed matrix for one maze; resumable                                                                         |
| `run_logger.py`           | Per-run logging (`config.json`, CSVs, `summary.json`, model checkpoints)                                                                |
| `visualize_results.py`    | Figures and results table from a finished sweep                                                                                         |
| `watch_pointmaze_hiro.py` | Render a trained agent                                                                              |

## Installation

Python 3.10–3.12.

```bash
git clone <repolink and name>
cd <repo-name>
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

The default `torch` wheel is CPU-only on Windows and macOS. A GPU is optional (the networks are small), but if you want the CUDA build, see the note in `requirements.txt`. The code uses CUDA automatically when it is available.

To check the install, run `python hiro_agent.py`. It runs a short self-test on random data and should end with `HiroAgent works correctly.`

## Usage

### 1. Run the sweep

```bash
python run_sweep.py --maze umaze --dry-run   # print the plan without training
python run_sweep.py --maze umaze
python run_sweep.py --maze medium
python run_sweep.py --maze large
```

| Option           | Default        | Description                                                    |
| ---------------- | -------------- | -------------------------------------------------------------- |
| `--maze`         | (required)     | `umaze`, `medium` or `large`                                   |
| `--windows`      | `3 5 10 20 40` | Meta-period values to sweep                                    |
| `--seeds`        | `0 1 2`        | Random seeds                                                   |
| `--total-steps`  | per maze       | Override the step budget (UMaze 120k, Medium 300k, Large 500k) |
| `--results-root` | `results`      | Where run folders are written                                  |
| `--dry-run`      | off            | Print the plan and exit                                        |

The sweep can be resumed. A run that already has a `summary.json` is skipped, so you can stop and restart freely. A run interrupted part-way starts over from scratch.

Run folders never collide, so you can run several sweeps in parallel in separate terminals, e.g. one per maze or one per seed (`--seeds 0`, `--seeds 1`, ...).

**Rough runtime:** about 15–20 minutes per 60k environment steps on a single CPU core, including evaluation. That is roughly 35 minutes per UMaze run, 1.5 hours per Medium run and 2.5 hours per Large run. The full 45-run matrix is about 70 CPU-hours.

### 2. Run a single configuration

`train_pointmaze_hiro.py` has no command-line interface; call `train()` from Python:

```python
from train_pointmaze_hiro import train

train(env_name="PointMaze_UMaze-v3", total_steps=120_000, window_length=10, seed=0)
```

### 3. Plot the results

```bash
python visualize_results.py
python visualize_results.py --results-root results --out figures --threshold 0.5
```

This prints a summary table to the console and writes the following to `figures/`:

| Output                       | Content                                                                         |
| ---------------------------- | ------------------------------------------------------------------------------- |
| `fig1_learning_curves.png`   | Eval success vs. environment steps, one panel per maze, mean ± 1 std over seeds |
| `fig2_sensitivity_curve.png` | Final success vs. meta-period, one line per maze                                |
| `fig3_sample_efficiency.png` | Steps to reach the success threshold (`n/r` = not reached)                      |
| `fig4_stability.png`         | Across-seed std of final success                                                |
| `fig5_summary_heatmap.png`   | Meta-period × maze grid of mean final success                                   |
| `results_table.csv`          | Mean, std and per-seed final success for every cell                             |

### 4. Watch a trained agent

Edit the `watch(...)` call at the bottom of `watch_pointmaze_hiro.py` to point at a run folder, e.g. `model_prefix="results/hiro_td3td3_pointmaze_umaze_v3/c10/seed0/model"`, then run `python watch_pointmaze_hiro.py`. This needs a display. Read [Known issues](#known-issues) first.

## Output format

```
results/
└── hiro_td3td3_pointmaze_umaze_v3/
    └── c10/                    # meta-period
        └── seed0/
            ├── config.json     # every hyperparameter for the run
            ├── train_log.csv   # one row per training episode
            ├── eval_log.csv    # one row per evaluation point
            ├── summary.json    # headline metrics (written when the run completes)
            ├── model_low.pth
            └── model_high.pth
```

`summary.json` contains these metrics:

- **`final_success_rate`**: the mean of the last 3 evaluation points.
- **`steps_to_threshold`**: the first environment step at which eval success reached 0.5 (and 0.25). It is `null` if the threshold was never reached.
- **`stability_post_rise_std`**: the standard deviation of eval success after it first exceeds 0.05. Lower means a steadier learning curve.

## Method

**Environment.** PointMaze v3 from [Gymnasium Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics): UMaze, Medium and Large. The reward is sparse (1 at the goal, 0 otherwise), and the episode ends on reaching the goal (`continuing_task=False`) or after 300 steps. Start and goal cells are randomised every episode. The observation is (x, y, vx, vy), so position is the first two state dimensions.

**High level.** A TD3 agent takes (state, goal) as input and every _c_ steps outputs a 2-D subgoal: a desired (x, y) offset from the current position, bounded to ±5. Its reward is the sum of environment rewards over the window.

**Low level.** A TD3 agent takes (state, subgoal) and outputs a force action every step. Its intrinsic reward is −‖s + g − s′‖ over the position dimensions. Between high-level decisions, the subgoal is shifted with HIRO's transition g′ = s + g − s′, so it keeps pointing at the same absolute target.

**Off-policy correction.** When the high level is trained, each stored subgoal is relabelled with the candidate that best explains the window's observed actions under the _current_ low-level policy (Gaussian log-likelihood). There are 8 candidates: the original subgoal, the achieved displacement, and 6 Gaussian perturbations of the displacement.

### Default hyperparameters

| Parameter                    | Value                                                                 |
| ---------------------------- | --------------------------------------------------------------------- |
| Meta-period _c_              | 3, 5, 10, 20, 40                                                      |
| Step budget                  | UMaze 120k · Medium 300k · Large 500k                                 |
| Episode horizon              | 300 steps                                                             |
| Seeds                        | 0, 1, 2                                                               |
| Networks                     | 2 hidden layers × 300 units, ReLU; tanh-scaled actor output           |
| Optimiser                    | Adam; actor lr 1e-4, critic lr 1e-3                                   |
| Discount γ / target update τ | 0.99 / 0.005 at both levels                                           |
| TD3                          | policy noise 0.2, noise clip 0.5, actor update every 2 critic updates |
| Exploration noise (std)      | low level 0.1; high level 0.5 (0.1 × subgoal bound)                   |
| Subgoal bounds               | ±5 in x and y                                                         |
| Replay buffers               | 200k each, batch size 128                                             |
| Warm-up                      | 2,000 steps with random subgoals and random actions                   |
| Training frequency           | low level every step; high level every 10 environment steps           |
| Evaluation                   | 25 deterministic episodes after every 5,000 environment steps         |

Evaluation runs at the first episode boundary after each 5,000-step milestone, so logged evaluation steps can fall up to 300 steps past the milestone. The evaluation environment is seeded, so every meta-period with the same seed is scored on the same start/goal pairs.

### Design notes for anyone building on this

These settings were held fixed across meta-periods, but their _effect_ changes with _c_. Keep them in mind when interpreting meta-period comparisons:

- **Update ratio.** The high level is updated every 10 environment steps regardless of _c_. It therefore gets about 0.3 updates per high-level transition at _c_ = 3 and 4 at _c_ = 40.
- **Discounting.** γ = 0.99 is applied per high-level transition, so measured in environment steps the high level's effective horizon grows with _c_.
- **Subgoal reach.** The subgoal bound of ±5 is large compared with how far the point mass can move in one window (roughly 0.01–1.5 units over 3–40 steps). Subgoals therefore behave more like directions than reachable positions.

## Relationship to the thesis

The results in the thesis were produced with the code at tag **`thesis-2026`** (`git checkout thesis-2026`). That version contained bugs that are fixed in the current code:

1. **High-level buffer.** It stored the previous window's subgoal (or a random placeholder at the start of each episode) as the high-level action, instead of the subgoal actually chosen.
2. **Evaluation meta-period.** Evaluation queried the high level on every step (effectively _c_ = 1) and never applied the subgoal transition.
3. **Evaluation timing.** Evaluation ran in the middle of training episodes and overwrote the episode's goal and partial window.
4. **Padding.** The off-policy correction scored the padding added to windows that end early (on reaching the goal); it is now masked out.

The current version also seeds the evaluation environment (see above). Results from the current code are therefore not directly comparable with the numbers reported in the thesis.

## Citation

If you use this code, please cite the thesis:

```bibtex
@mastersthesis{osbourne2026metaperiod,
  author = {Osbourne, Oneil},
  title  = {A Controlled Empirical Study of Meta-Period Sensitivity in Off-Policy
            Hierarchical Reinforcement Learning for Maze Navigation},
  school = {The University of the West Indies, Mona},
  type   = {MSc thesis},
  year   = {2026}
}
```

This work builds on:

- Nachum, O., Gu, S., Lee, H., & Levine, S. (2018). Data-efficient hierarchical reinforcement learning. _NeurIPS 31_.
- Fujimoto, S., van Hoof, H., & Meger, D. (2018). Addressing function approximation error in actor-critic methods. _ICML, PMLR 80_.
- Fu, J., Kumar, A., Nachum, O., Tucker, G., & Levine, S. (2020). D4RL: Datasets for deep data-driven reinforcement learning. _arXiv:2004.07219_.

