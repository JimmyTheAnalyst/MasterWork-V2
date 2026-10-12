import os
import csv
import json


class RunLogger:
    """
    Minimal experiment logger shared across all training conditions
    (flat TD3, flat SAC, TD3-TD3 HIRO, SAC-SAC HIRO).

    Creates a self-contained run folder:

        results/{method}_{env_tag}/seed{seed}/
            config.json     - all hyperparameters + env + seed
            train_log.csv   - one row per episode
            eval_log.csv    - one row per evaluation point
            model.pth                  (flat agents)
            model_low.pth, model_high.pth   (hierarchical agents)

    It writes to disk incrementally (each episode/eval is flushed), so a
    crash mid-run does not lose the data collected so far. It also prints
    to the terminal so you keep the live view.

    Usage:
        logger = RunLogger(method="hiro_td3td3",
                           env_name="PointMaze_UMaze-v3",
                           seed=0, config=cfg_dict)
        ... each episode:
        logger.log_episode(episode, global_step, reward, success, success_rate)
        ... each eval:
        logger.log_eval(episode, global_step, eval_success_rate)
        ... at the end:
        logger.save_model(agent, hierarchical=True)
        logger.close()
    """

    TRAIN_FIELDS = [
        "episode", "global_step", "reward", "success",
        "success_rate_20", "elapsed_sec",
    ]
    EVAL_FIELDS = ["episode", "global_step", "eval_success_rate"]

    def __init__(self, method, env_name, seed, config=None, results_root="results",
                 run_tag=None):
        self.method = method
        self.env_name = env_name
        self.seed = seed

        # Build a filesystem-safe env tag, e.g. "PointMaze_UMaze-v3" -> "pointmaze_umaze_v3"
        env_tag = env_name.lower().replace("-", "_").replace(".", "_")

        # run_tag disambiguates runs that share method+env+seed but differ in the
        # swept parameter (e.g. window length). Without it, every value of a swept
        # parameter would map to the SAME folder and overwrite the previous run.
        # e.g. run_tag="c10" -> results/hiro_td3td3_pointmaze_umaze_v3/c10/seed0/
        if run_tag is not None:
            tag = str(run_tag).replace(" ", "").replace("/", "_")
            self.run_dir = os.path.join(
                results_root, f"{method}_{env_tag}", tag, f"seed{seed}"
            )
        else:
            self.run_dir = os.path.join(
                results_root, f"{method}_{env_tag}", f"seed{seed}"
            )
        os.makedirs(self.run_dir, exist_ok=True)

        # ---- config.json ----
        full_config = {
            "method": method,
            "env_name": env_name,
            "seed": seed,
            "run_tag": run_tag,
        }
        if config:
            full_config.update(config)
        self._config = full_config  # retained for the end-of-run summary
        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(full_config, f, indent=2)

        # In-memory record of eval points so write_summary() needn't re-read the CSV
        self._eval_points = []  # list of (global_step, eval_success_rate)

        # ---- open CSV writers ----
        self._train_file = open(os.path.join(self.run_dir, "train_log.csv"), "w", newline="")
        self._train_writer = csv.DictWriter(self._train_file, fieldnames=self.TRAIN_FIELDS)
        self._train_writer.writeheader()
        self._train_file.flush()

        self._eval_file = open(os.path.join(self.run_dir, "eval_log.csv"), "w", newline="")
        self._eval_writer = csv.DictWriter(self._eval_file, fieldnames=self.EVAL_FIELDS)
        self._eval_writer.writeheader()
        self._eval_file.flush()

        print(f"[RunLogger] logging to {self.run_dir}")

    # ------------------------------------------------------------------ logging

    def log_episode(self, episode, global_step, reward, success,
                    success_rate, elapsed_sec, verbose=True):
        """Record one training episode. `success` is bool/0-1, success_rate in [0,100]."""
        row = {
            "episode": episode,
            "global_step": global_step,
            "reward": float(reward),
            "success": int(bool(success)),
            "success_rate_20": float(success_rate),
            "elapsed_sec": round(float(elapsed_sec), 2),
        }
        self._train_writer.writerow(row)
        self._train_file.flush()

        if verbose:
            success_str = "YES" if success else "no"
            print(f"{episode:>8} {float(reward):>8.2f} {success_str:>8} "
                  f"{float(success_rate):>8.0f}% {global_step:>9} {float(elapsed_sec):>7.1f}s")

    def log_eval(self, episode, global_step, eval_success_rate, verbose=True):
        """Record one evaluation point. eval_success_rate in [0,1]."""
        row = {
            "episode": episode,
            "global_step": global_step,
            "eval_success_rate": float(eval_success_rate),
        }
        self._eval_writer.writerow(row)
        self._eval_file.flush()
        self._eval_points.append((int(global_step), float(eval_success_rate)))

        if verbose:
            print(f"  [Eval] Success rate: {eval_success_rate*100:.0f}%")

    # ------------------------------------------------------------------ models

    def model_path(self, name="model"):
        """Return a path inside the run folder for saving a model file."""
        return os.path.join(self.run_dir, name)

    def save_model(self, agent, hierarchical):
        """
        Save the trained agent into the run folder.

        hierarchical=False -> flat agent with .save(path) producing one file.
                              Saves results/.../seedN/model.pth
        hierarchical=True  -> HIRO agent whose .save(prefix) produces
                              prefix_low.pth and prefix_high.pth.
                              Saves model_low.pth and model_high.pth.
        """
        if hierarchical:
            prefix = os.path.join(self.run_dir, "model")
            agent.save(prefix)  # HiroAgent.save appends _low.pth / _high.pth
            print(f"[RunLogger] saved model_low.pth and model_high.pth in {self.run_dir}")
        else:
            path = os.path.join(self.run_dir, "model.pth")
            agent.save(path)
            print(f"[RunLogger] saved model.pth in {self.run_dir}")

    # ------------------------------------------------------------------ summary

    def write_summary(self, final_k=3, thresholds=(0.5, 0.25), rise_level=0.05):
        """
        Compute and write summary.json with the three headline metrics from the
        reporting guide, derived from the eval points collected this run.

        Metrics:
          final_success_rate : mean of the last `final_k` eval points (headline performance).
          steps_to_threshold : for each threshold, the first global_step whose eval
                               success >= threshold, or null if never reached (censored).
          stability_post_rise_std : std of eval success over points AFTER the curve
                               first rises above `rise_level` (lower = more stable);
                               null if it never rises.

        Also echoes key config fields (window_length, env, seed, subgoal_bounds)
        so each summary.json is self-describing for the per-seed appendix table.
        Returns the summary dict.
        """
        pts = sorted(self._eval_points)  # by global_step
        steps = [s for s, _ in pts]
        rates = [r for _, r in pts]

        # final success: mean of last final_k eval points
        if rates:
            k = min(final_k, len(rates))
            final_success = sum(rates[-k:]) / k
        else:
            final_success = None

        # steps to threshold (first crossing), per threshold; None = never reached
        steps_to_threshold = {}
        for thr in thresholds:
            hit = next((s for s, r in pts if r >= thr), None)
            steps_to_threshold[str(thr)] = hit  # int or None

        # stability: std of rates after the curve first rises above rise_level
        rise_idx = next((i for i, r in enumerate(rates) if r > rise_level), None)
        if rise_idx is not None and len(rates) - rise_idx >= 2:
            post = rates[rise_idx:]
            mean_post = sum(post) / len(post)
            var = sum((x - mean_post) ** 2 for x in post) / (len(post) - 1)  # sample std
            stability = var ** 0.5
        else:
            stability = None

        summary = {
            "method": self.method,
            "env_name": self.env_name,
            "seed": self.seed,
            "window_length": self._config.get("window_length"),
            "subgoal_bounds": self._config.get("subgoal_bounds"),
            "eval_episodes": self._config.get("eval_episodes"),
            "num_eval_points": len(pts),
            "final_success_rate": final_success,
            "final_k": final_k,
            "steps_to_threshold": steps_to_threshold,
            "stability_post_rise_std": stability,
            "rise_level": rise_level,
            "last_global_step": steps[-1] if steps else None,
        }
        with open(os.path.join(self.run_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        fs = "n/a" if final_success is None else f"{final_success:.3f}"
        st = "n/a" if stability is None else f"{stability:.3f}"
        print(f"[RunLogger] summary.json written  "
              f"(final={fs}, stability_std={st})")
        return summary

    # ------------------------------------------------------------------ cleanup

    def close(self):
        self._train_file.close()
        self._eval_file.close()


if __name__ == "__main__":
    # Tiny self-test
    import shutil
    logger = RunLogger(
        method="selftest",
        env_name="PointMaze_UMaze-v3",
        seed=0,
        config={"num_episodes": 3, "subgoal_bounds": [5.0, 5.0]},
        results_root="_logger_selftest",
    )
    logger.log_episode(1, 300, 0.0, False, 0.0, 0.1)
    logger.log_episode(2, 600, 1.0, True, 50.0, 0.2)
    logger.log_eval(2, 600, 0.5)
    logger.close()

    # Verify files exist and have content
    run_dir = logger.run_dir
    for fname in ["config.json", "train_log.csv", "eval_log.csv"]:
        path = os.path.join(run_dir, fname)
        assert os.path.exists(path), f"missing {fname}"
        print(f"--- {fname} ---")
        with open(path) as f:
            print(f.read())
    shutil.rmtree("_logger_selftest")
    print("RunLogger self-test passed.")