"""
Sweep driver for the meta-period (window_length) sensitivity study.

Runs the full matrix of  window_length x seed  for a chosen PointMaze difficulty,
using the step-based training in train_pointmaze_hiro.py. Each run logs into its
own tagged folder (results/hiro_td3td3_{env}/c{wl}/seed{N}/), so nothing collides
and the whole thing is resumable: a run whose summary.json already exists is skipped.

USAGE (from the hiro_mine folder):

    # one maze at a time (recommended — do the cheap ones first):
    python run_sweep.py --maze umaze
    python run_sweep.py --maze medium
    python run_sweep.py --maze large

    # options:
    python run_sweep.py --maze medium --seeds 0 1 2 --windows 3 5 10 20 40
    python run_sweep.py --maze large  --seeds 0 1 2 --windows 5 10 20   # reduced set on Large
    python run_sweep.py --maze umaze --dry-run    # just print the plan, run nothing

Per-maze defaults (env name + total step budget + per-episode horizon) are set in
MAZES below, based on the validated difficulty gradient:
  - UMaze : saturates fast        -> 120k steps
  - Medium: discriminates, slower -> 300k steps
  - Large : hard but learnable    -> 500k steps
The per-episode horizon is 300 for all three (verified sufficient: on Large, all
successful episodes terminated in <250 steps).
"""

import argparse
import os
import time
from datetime import timedelta

from train_pointmaze_hiro import train


# Per-maze configuration: env name, total environment-step budget, episode horizon.
MAZES = {
    "umaze":  {"env_name": "PointMaze_UMaze-v3",  "total_steps": 120_000, "max_steps_per_episode": 300},
    "medium": {"env_name": "PointMaze_Medium-v3", "total_steps": 300_000, "max_steps_per_episode": 300},
    "large":  {"env_name": "PointMaze_Large-v3",  "total_steps": 500_000, "max_steps_per_episode": 300},
}

DEFAULT_WINDOWS = [3, 5, 10, 20, 40]
DEFAULT_SEEDS = [0, 1, 2]


def env_tag(env_name):
    """Mirror RunLogger's folder-naming so we can check for completed runs."""
    return env_name.lower().replace("-", "_").replace(".", "_")


def run_dir_for(env_name, window_length, seed, results_root="results"):
    return os.path.join(
        results_root,
        f"hiro_td3td3_{env_tag(env_name)}",
        f"c{window_length}",
        f"seed{seed}",
    )


def already_done(env_name, window_length, seed, results_root="results"):
    """A run counts as done if its summary.json exists (written only on completion)."""
    return os.path.exists(os.path.join(
        run_dir_for(env_name, window_length, seed, results_root), "summary.json"
    ))


def main():
    parser = argparse.ArgumentParser(description="Meta-period sweep driver")
    parser.add_argument("--maze", required=True, choices=list(MAZES.keys()),
                        help="Which PointMaze difficulty to sweep.")
    parser.add_argument("--windows", type=int, nargs="+", default=DEFAULT_WINDOWS,
                        help="Window-length values to sweep (default: 3 5 10 20 40).")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help="Seeds to run (default: 0 1 2).")
    parser.add_argument("--total-steps", type=int, default=None,
                        help="Override the per-maze step budget.")
    parser.add_argument("--results-root", default="results",
                        help="Root results directory (default: results).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and exit without running.")
    args = parser.parse_args()

    cfg = MAZES[args.maze]
    env_name = cfg["env_name"]
    total_steps = args.total_steps if args.total_steps is not None else cfg["total_steps"]
    max_steps = cfg["max_steps_per_episode"]

    # Build the list of (window, seed) cells, in a sensible order: all seeds of one
    # window together, so a partial sweep still yields complete windows to look at.
    cells = [(wl, s) for wl in args.windows for s in args.seeds]
    todo = [(wl, s) for (wl, s) in cells if not already_done(env_name, wl, s, args.results_root)]
    done = [(wl, s) for (wl, s) in cells if already_done(env_name, wl, s, args.results_root)]

    print("=" * 70)
    print(f"SWEEP PLAN  —  maze={args.maze}  ({env_name})")
    print(f"  windows      : {args.windows}")
    print(f"  seeds        : {args.seeds}")
    print(f"  total cells  : {len(cells)}")
    print(f"  already done : {len(done)}  (skipped)")
    print(f"  to run       : {len(todo)}")
    print(f"  step budget  : {total_steps:,} env steps/run   (episode horizon {max_steps})")
    print("=" * 70)
    if done:
        print("Skipping completed:", ", ".join(f"c{wl}/seed{s}" for wl, s in done))
    if not todo:
        print("Nothing to run — every cell already has a summary.json. Done.")
        return
    print("Will run:", ", ".join(f"c{wl}/seed{s}" for wl, s in todo))
    print("=" * 70)

    if args.dry_run:
        print("(dry run — exiting without training)")
        return

    sweep_start = time.perf_counter()
    for i, (wl, seed) in enumerate(todo, 1):
        print("\n" + "#" * 70)
        print(f"# RUN {i}/{len(todo)}  —  maze={args.maze}  window_length={wl}  seed={seed}")
        print("#" * 70)
        run_start = time.perf_counter()

        train(
            env_name=env_name,
            total_steps=total_steps,
            max_steps_per_episode=max_steps,
            window_length=wl,
            seed=seed,
            results_root=args.results_root,
        )

        run_elapsed = timedelta(seconds=time.perf_counter() - run_start)
        total_elapsed = timedelta(seconds=time.perf_counter() - sweep_start)
        print(f"\n[run {i}/{len(todo)} done in {run_elapsed}]  [sweep elapsed {total_elapsed}]")

    print("\n" + "=" * 70)
    print(f"SWEEP COMPLETE — {len(todo)} runs in {timedelta(seconds=time.perf_counter() - sweep_start)}")
    print("=" * 70)


if __name__ == "__main__":
    main()