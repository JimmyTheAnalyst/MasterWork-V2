import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics
from collections import deque
import time
from datetime import timedelta

from hiro_agent import HiroAgent
from subgoal import SubgoalSpace
from run_logger import RunLogger


gym.register_envs(gymnasium_robotics)


# --- Subgoal bounds for PointMaze ---------------------------------------------
# PointMaze is a small maze (cells are ~1 unit). The subgoal is a 2D (x, y)
# position OFFSET that the low level should achieve within `window_length` steps.
# Bounds of +-5 keep subgoals within reach over a 10-step window, addressing the
# "subgoal bounds too large for the per-window reachable distance" concern that
# hurt the AntMaze runs (where bounds were +-10). If the low level struggles to
# achieve subgoals (intrinsic reward stays strongly negative), reduce further.
POINTMAZE_SUBGOAL_BOUNDS = np.array([5.0, 5.0], dtype=np.float32)


def unpack_observation(obs):
    """
    PointMaze observations are dicts:
      observation   = (x, y, vx, vy)   -> position is ALREADY the first 2 dims
      desired_goal  = (x, y)            -> the target
      achieved_goal = (x, y)            -> current position

    Unlike AntMaze, we do NOT augment the state, because position is already
    state[:2]. This keeps the invariant the HIRO machinery relies on:
    state[:subgoal_dim] == position.
    """
    state = obs['observation'].astype(np.float32)
    final_goal = obs['desired_goal'].astype(np.float32)
    achieved = obs['achieved_goal'].astype(np.float32)
    return state, final_goal, achieved


def train(
    env_name="PointMaze_UMaze-v3",   # sparse reward + (set below) episodic termination
    total_steps=120000,              # TOTAL environment-step budget (the comparison axis).
                                     # Per-maze guidance: UMaze ~120k, Medium ~300k, Large ~500k.
    max_steps_per_episode=300,       # per-episode horizon. 300 verified sufficient for all
                                     # PointMaze mazes incl. Large (successes terminate <250 steps).
    start_training_steps=2000,
    buffer_size=200000,
    batch_size=128,
    window_length=10,
    reward_scaling=1.0,              # PointMaze sparse reward is already 0/1; no scale-down
    eval_every_steps=5000,           # evaluate every N ENVIRONMENT STEPS (not episodes), so
                                     # eval points land at comparable steps across all runs.
    eval_episodes=25,                # test episodes per eval point. 25 -> 0.04 resolution
    seed=42,
    results_root="results_fixed",          # root dir for logged runs (sweep driver may override)
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # continuing_task=False -> episode TERMINATES on goal-reach (within 0.45 units).
    # reset_target=False    -> one goal per episode (standard maze navigation).
    env = gym.make(env_name, continuing_task=False, reset_target=False)
    env.reset(seed=seed)

    obs, _ = env.reset()
    state, _, _ = unpack_observation(obs)
    state_dim = state.shape[0]                 # 4 for PointMaze
    goal_dim = obs['desired_goal'].shape[0]    # 2
    action_dim = env.action_space.shape[0]     # 2
    action_scale = env.action_space.high       # [1, 1]

    subgoal_space = SubgoalSpace(POINTMAZE_SUBGOAL_BOUNDS)

    print(f"Environment: {env_name}")
    print(f"  State dim:     {state_dim} (x, y, vx, vy)")
    print(f"  Goal dim:      {goal_dim}")
    print(f"  Action dim:    {action_dim}")
    print(f"  Action scale:  {action_scale}")
    print(f"  Subgoal dim:   {subgoal_space.dim}  bounds +-{subgoal_space.bounds}")

    agent = HiroAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        goal_dim=goal_dim,
        action_scale=action_scale,
        subgoal_space=subgoal_space,
        device=device,
        window_length=window_length,
        buffer_size=buffer_size,
        batch_size=batch_size,
        reward_scaling=reward_scaling,
        start_training_steps=start_training_steps,
    )

    # ---- Logger: writes config.json, train_log.csv, eval_log.csv, and models
    # into results/hiro_td3td3_{env}/seed{seed}/ ----
    logger = RunLogger(
        method="hiro_td3td3",
        env_name=env_name,
        seed=seed,
        run_tag=f"c{window_length}",   # separates sweep runs: .../c3/seed0, .../c10/seed0, ...
        results_root=results_root,
        config={
            "total_steps": total_steps,
            "max_steps_per_episode": max_steps_per_episode,
            "start_training_steps": start_training_steps,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "window_length": window_length,
            "reward_scaling": reward_scaling,
            "eval_every_steps": eval_every_steps,
            "eval_episodes": eval_episodes,
            "subgoal_bounds": subgoal_space.bounds.tolist(),
            "continuing_task": False,
            "low_expl_noise": agent.low_controller.expl_noise,
            "high_expl_noise": agent.high_controller.expl_noise,
        },
    )

    global_step = 0
    episode = 0
    recent_rewards = deque(maxlen=20)
    recent_successes = deque(maxlen=20)
    start_time = time.time()
    next_eval_at = eval_every_steps  # step milestone for the next evaluation

    print(f"\nTraining for {total_steps} environment steps...")
    print(f"{'Episode':>8} {'Reward':>8} {'Success':>8} {'SuccRate':>9} {'Steps':>9} {'Time':>8}")
    print("-" * 60)

    # Step-based stopping: train until the environment-step budget is exhausted,
    # however many episodes that takes. This makes the step budget identical across
    # all window lengths and seeds, which is required for the env-steps x-axis to be
    # a fair comparison (good policies finish episodes faster, so fixing episode count
    # would give different runs different amounts of experience).
    while global_step < total_steps:
        episode += 1
        obs, _ = env.reset()
        state, final_goal, _ = unpack_observation(obs)

        agent.reset_episode(state, final_goal)

        episode_reward = 0.0
        reached_goal = False

        for step in range(max_steps_per_episode):
            action = agent.select_action(state, global_step, noisy=True)

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state, _, _ = unpack_observation(next_obs)

            # terminated and truncated kept SEPARATE: only true goal-reach
            # termination zeroes the bootstrap; truncation keeps it.
            agent.store_transition(state, action, reward, next_state, terminated, truncated)

            agent.train(global_step)

            state = next_state
            episode_reward += reward
            global_step += 1

            if info.get("success", False) or terminated:
                reached_goal = True

            if terminated or truncated:
                break
            if global_step >= total_steps:
                break

        recent_rewards.append(episode_reward)
        recent_successes.append(1.0 if reached_goal else 0.0)
        success_rate = np.mean(recent_successes) * 100
        elapsed = time.time() - start_time

        logger.log_episode(
            episode, global_step, episode_reward,
            reached_goal, success_rate, elapsed,
        )

        # FIX (bug C): evaluate at the episode boundary. evaluate() calls agent.reset_episode(),
        # which overwrites final_goal / the half-built window of a live training episode.
        # Eval points now land within one episode (<=300 steps) of the milestone; the logged
        # global_step is what the plots use, so nothing else changes.
        if global_step >= next_eval_at:
            eval_success = evaluate(agent, env_name, num_episodes=eval_episodes,
                                    max_steps=max_steps_per_episode, seed=10_000 + seed)
            logger.log_eval(episode, global_step, eval_success)
            while next_eval_at <= global_step:
                next_eval_at += eval_every_steps

    env.close()
    print("\nTraining complete.")
    logger.save_model(agent, hierarchical=True)
    logger.write_summary()
    logger.close()


def evaluate(agent, env_name, num_episodes=25, max_steps=300, seed=None):
    env = gym.make(env_name, continuing_task=False, reset_target=False)
    if seed is not None:
        env.reset(seed=seed)   # same start/goal draws for every run -> lower-variance comparisons
    successes = 0

    for _ in range(num_episodes):
        obs, _ = env.reset()
        state, final_goal, _ = unpack_observation(obs)
        agent.reset_episode(state, final_goal)

        for _ in range(max_steps):
            action = agent.select_action(state, global_step=10**9, noisy=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state, _, _ = unpack_observation(next_obs)
            agent.advance(state, next_state)   # FIX (bug B): shift subgoal, count the window

            if info.get("success", False) or terminated:
                successes += 1
                break
            if terminated or truncated:
                break

            state = next_state

    env.close()
    return successes / num_episodes


if __name__ == "__main__":
    start = time.perf_counter()
    # Quick manual run; the sweep driver (run_sweep.py) is the normal entry point.
    #train(total_steps=120000, env_name="PointMaze_UMaze-v3", window_length=10, seed=0)
    print(f"Total training time: {timedelta(seconds=time.perf_counter() - start)}")