import numpy as np
import torch
import torch.nn.functional as F

from td3 import TD3Controller
from hiro_buffers import LowReplayBuffer, HighReplayBuffer
from subgoal import SubgoalSpace


class HiroAgent:
    """
    HIRO agent: two-level hierarchy with TD3 at each level.
    
    High level: outputs subgoals at a slower timescale (every window_length steps)
    Low level:  outputs environment actions, conditioned on current subgoal
    
    The low level receives intrinsic reward (distance to subgoal).
    The high level receives extrinsic reward (accumulated environment reward).
    
    Subgoals shift each step via the subgoal transition function to remain
    pointing at the same absolute state target as the agent moves.
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        goal_dim,
        action_scale,
        subgoal_space,  # SubgoalSpace object
        device,
        window_length=10,
        buffer_size=200000,
        batch_size=128,
        reward_scaling=0.1,
        start_training_steps=2500,
        train_freq_high=10,
    ):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.subgoal_space = subgoal_space
        self.subgoal_dim = subgoal_space.dim
        self.window_length = window_length
        self.reward_scaling = reward_scaling
        self.start_training_steps = start_training_steps
        self.train_freq_high = train_freq_high
        
        # ---- Low level: TD3 agent that takes (state, subgoal) -> action ----
        self.low_controller = TD3Controller(
            state_dim=state_dim,
            action_dim=action_dim,
            action_scale=action_scale,
            goal_dim=self.subgoal_dim,  # subgoal plays the role of "goal" for the low level
            device=device,
            expl_noise=0.1,
        )
        
        # ---- High level: TD3 agent that takes (state, final_goal) -> subgoal ----
        # High-level's "action" is a subgoal, so its action_scale is the subgoal bounds.
        #
        # NOTE on expl_noise: in TD3Controller.select_action the exploration noise std
        # is (expl_noise * action_scale). With subgoal bounds = 10, an expl_noise of 1.0
        # would give a noise std of 10 — i.e. noise spanning the ENTIRE subgoal range,
        # which drowns out the learned policy and makes high-level rollouts ~uniform
        # random. The HIRO paper uses an absolute high-level noise std of ~1.0, so with
        # action_scale=10 we want expl_noise=0.1 (0.1 * 10 = 1.0 absolute std).
        self.high_controller = TD3Controller(
            state_dim=state_dim,
            action_dim=self.subgoal_dim,  # "action" = subgoal
            action_scale=subgoal_space.bounds,
            goal_dim=goal_dim,
            device=device,
            expl_noise=0.1,  # absolute subgoal-space std ~= 0.1 * bounds
        )
        
        # ---- Replay buffers ----
        self.low_buffer = LowReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            subgoal_dim=self.subgoal_dim,
            buffer_size=buffer_size,
            batch_size=batch_size,
            device=device,
        )
        
        self.high_buffer = HighReplayBuffer(
            state_dim=state_dim,
            goal_dim=goal_dim,
            subgoal_dim=self.subgoal_dim,
            action_dim=action_dim,
            buffer_size=buffer_size,
            batch_size=batch_size,
            window_length=window_length,
            device=device,
        )
        
        # ---- Per-episode state ----
        self.final_goal = None      # the episode's fixed target (from environment)
        self.current_subgoal = None # current active subgoal (shifts each step)
        self.steps_in_window = 0    # how many low-level steps since last subgoal was picked
        
        # Buffer for building up high-level experiences over a window
        self._window_start_state = None
        self._window_subgoal = None      # the subgoal picked at window start
        self._window_accumulated_reward = 0.0
        self._window_states = []         # trajectory of states during window
        self._window_actions = []        # trajectory of actions during window
    
    # ==================== Core hierarchical logic ====================
    
    def reset_episode(self, initial_state, final_goal):
        """Call at the start of each episode."""
        self.final_goal = final_goal
        self.current_subgoal = self.subgoal_space.sample()  # placeholder; will be overwritten
        self.steps_in_window = 0
        self._reset_window(initial_state)
    
    def _reset_window(self, state):
        """Start a new high-level window."""
        self._window_start_state = state.copy()
        self._window_subgoal = self.current_subgoal.copy()
        self._window_accumulated_reward = 0.0
        self._window_states = []
        self._window_actions = []
    
    def select_action(self, state, global_step, noisy=True):
        """
        Called every environment step. Handles hierarchy internally:
        - If at the start of a new window, pick a new subgoal
        - Otherwise, shift the current subgoal via the transition function
        - Let the low level choose an environment action conditioned on the subgoal
        """
        # Pick new subgoal at the start of each window
        if self.steps_in_window == 0:
            if global_step < self.start_training_steps:
                # Warm-up: random subgoal
                self.current_subgoal = self.subgoal_space.sample()
            else:
                self.current_subgoal = self.high_controller.select_action(
                    state, goal=self.final_goal, noisy=noisy
                )
                self.current_subgoal = self.subgoal_space.clip(self.current_subgoal)
            # FIX (bug A): record the subgoal that was ACTUALLY emitted for this window.
            # Previously _window_subgoal was copied in _reset_window() *before* this
            # selection happened, so the high-level buffer stored the previous window's
            # (shifted) subgoal — or the random placeholder from reset_episode().
            self._window_subgoal = np.array(self.current_subgoal, dtype=np.float32).copy()
        
        # Low level chooses action given (state, current_subgoal)
        if global_step < self.start_training_steps:
            # Warm-up: random action — sampled from [-action_scale, +action_scale]
            action_scale = self.low_controller.action_scale.cpu().numpy()
            action = np.random.uniform(-action_scale, action_scale).astype(np.float32)
        else:
            action = self.low_controller.select_action(
                state, goal=self.current_subgoal, noisy=noisy
            )
        
        return action
    
    def advance(self, state, next_state):
        """
        FIX (bug B): advance the hierarchy WITHOUT storing anything. Call this after every
        env.step() in evaluation / watch loops. Without it, steps_in_window never leaves 0,
        so select_action() queries the high level on every step (effective c = 1) and the
        subgoal transition h(s, g, s') is never applied at evaluation time.
        """
        self.current_subgoal = self.subgoal_space.clip(
            self._subgoal_transition(state, self.current_subgoal, next_state)
        )
        self.steps_in_window += 1
        if self.steps_in_window >= self.window_length:
            self.steps_in_window = 0

    def store_transition(self, state, action, reward, next_state, terminated, truncated):
        """
        Called after each environment step. Handles:
        - Computing intrinsic reward
        - Storing low-level experience
        - Shifting the subgoal for the next step
        - Accumulating data for the high-level window
        - Storing high-level experience when window completes

        IMPORTANT: terminated and truncated are kept separate.
        - terminated = the episode reached a true terminal state (goal reached).
          We must NOT bootstrap past it, so not_done = 0.
        - truncated = the episode hit a time limit but could have continued.
          We MUST still bootstrap, so not_done = 1 (truncation is treated as
          non-terminal for the value target).
        Only `terminated` zeroes out the Bellman bootstrap. Both close the window.
        """
        # The episode is over (for window/loop purposes) if either is true.
        episode_over = terminated or truncated

        # Compute intrinsic reward (negative distance to subgoal-indicated target)
        intrinsic_reward = self._intrinsic_reward(state, self.current_subgoal, next_state)
        
        # Compute next subgoal via transition (subgoal shifts with state change)
        next_subgoal = self._subgoal_transition(state, self.current_subgoal, next_state)
        next_subgoal = self.subgoal_space.clip(next_subgoal)
        
        # Store low-level experience.
        # NOTE: the low level's notion of "terminal" is the subgoal horizon, not the
        # environment episode. We bootstrap the low level across env truncation, and
        # only treat true environment termination as terminal.
        self.low_buffer.add(
            state=state,
            subgoal=self.current_subgoal,
            action=action,
            next_state=next_state,
            next_subgoal=next_subgoal,
            reward=intrinsic_reward,
            done=terminated,  # only true termination zeroes the bootstrap
        )
        
        # Accumulate for high-level experience
        self._window_states.append(state.copy())
        self._window_actions.append(action.copy())
        self._window_accumulated_reward += self.reward_scaling * reward
        
        self.steps_in_window += 1
        
        # If window complete, store high-level experience and start a new window.
        # The window closes on horizon OR on episode end (terminated or truncated),
        # but the stored not_done uses `terminated` only (handled downstream).
        window_complete = (self.steps_in_window >= self.window_length) or episode_over
        if window_complete:
            self._store_high_level_experience(next_state, terminated)
            self._reset_window(next_state)
            self.steps_in_window = 0
        
        # Update current subgoal for next step
        self.current_subgoal = next_subgoal
    
    def _store_high_level_experience(self, window_end_state, terminated):
        """Pack up the accumulated window and add it to the high-level buffer.

        `terminated` is true environment termination only (goal reached), NOT
        truncation — so the high-level Bellman target bootstraps correctly across
        time-limit endings.
        """
        # Pad trajectory to fixed length if window ended early (episode done)
        states_arr = np.array(self._window_states, dtype=np.float32)
        actions_arr = np.array(self._window_actions, dtype=np.float32)
        traj_len = len(states_arr)   # FIX: remember the real length so padding can be masked

        if len(states_arr) < self.window_length:
            # Pad with final values to keep shape consistent
            pad_length = self.window_length - len(states_arr)
            state_pad = np.tile(states_arr[-1:], (pad_length, 1))
            action_pad = np.tile(actions_arr[-1:], (pad_length, 1))
            states_arr = np.concatenate([states_arr, state_pad], axis=0)
            actions_arr = np.concatenate([actions_arr, action_pad], axis=0)
        
        self.high_buffer.add(
            state=self._window_start_state,
            goal=self.final_goal,
            subgoal=self._window_subgoal,
            next_state=window_end_state,
            reward=self._window_accumulated_reward,
            done=terminated,
            state_trajectory=states_arr,
            action_trajectory=actions_arr,
            traj_len=traj_len,
        )
    
    def _subgoal_transition(self, state, subgoal, next_state):
        """
        HIRO's subgoal transition: shift subgoal so it still points at the same 
        absolute target as the agent moves.
        
        Formula: new_subgoal = state[:sg_dim] + subgoal - next_state[:sg_dim]
        
        Only applies to the first `subgoal_dim` components of the state.
        """
        return state[:self.subgoal_dim] + subgoal - next_state[:self.subgoal_dim]
    
    def _intrinsic_reward(self, state, subgoal, next_state):
        """
        Low-level reward: negative Euclidean distance from where the low level
        ended up to where it was supposed to end up (state + subgoal).
        """
        desired_end = state[:self.subgoal_dim] + subgoal
        actual_end = next_state[:self.subgoal_dim]
        return -float(np.linalg.norm(desired_end - actual_end))
    
    # ==================== Training ====================
    
    def train(self, global_step):
        """
        Train one step of both levels (with high level trained less frequently).
        Returns a dict of losses.
        """
        losses = {}
        
        if global_step < self.start_training_steps:
            return losses
        
        # Low level trains every step if buffer has enough samples
        if len(self.low_buffer) >= self.low_buffer.batch_size:
            low_losses = self._train_low()
            for k, v in low_losses.items():
                losses[f'low_{k}'] = v
        
        # High level trains less often
        if global_step % self.train_freq_high == 0:
            if len(self.high_buffer) >= self.high_buffer.batch_size:
                high_losses = self._train_high()
                for k, v in high_losses.items():
                    losses[f'high_{k}'] = v
        
        return losses
    
    def _train_low(self):
        """One TD3 update on the low-level controller using the low buffer."""
        states, subgoals, actions, next_states, next_subgoals, rewards, not_dones = \
            self.low_buffer.sample()
        
        return self._td3_update(
            controller=self.low_controller,
            states=states,
            goals=subgoals,
            actions=actions,
            next_states=next_states,
            next_goals=next_subgoals,
            rewards=rewards,
            not_dones=not_dones,
        )
    def _off_policy_correction(self, states, subgoals, next_states,
                            state_trajectories, action_trajectories,
                            num_candidates=8, traj_lens=None):
    
        """
        HIRO off-policy correction.
    
        For each experience in the batch, relabel the stored subgoal with the one
        most likely to have induced the observed action sequence under the 
        *current* low-level policy.
        
        Args:
            states: (batch, state_dim) — starts of each window
            subgoals: (batch, subgoal_dim) — originally chosen subgoals
            next_states: (batch, state_dim) — ends of each window
            state_trajectories: (batch, window, state_dim) — states during each window
            action_trajectories: (batch, window, action_dim) — actions during each window
            num_candidates: how many candidate subgoals to consider per experience
        
        Returns:
            relabeled_subgoals: (batch, subgoal_dim) tensor on device
        """
        
        batch_size = states.shape[0]
        window = state_trajectories.shape[1]
        subgoal_dim = subgoals.shape[1]
        
        # Work in numpy for candidate generation (cleaner), then convert for policy query
        states_np = states.cpu().numpy()
        subgoals_np = subgoals.cpu().numpy()
        next_states_np = next_states.cpu().numpy()
        state_traj_np = state_trajectories.cpu().numpy()
        action_traj_np = action_trajectories.cpu().numpy()
        
        # --- Generate candidate subgoals ---
        # Candidate 1: the original subgoal
        # Candidate 2: the state difference (end_state - start_state for first sg_dim dims)
        # Candidates 3..N: random perturbations around the state-difference
        
        state_diff = next_states_np[:, :subgoal_dim] - states_np[:, :subgoal_dim]
        # shape (batch, subgoal_dim)
        
        # Stack candidates: (batch, num_candidates, subgoal_dim)
        candidates = np.zeros((batch_size, num_candidates, subgoal_dim), dtype=np.float32)
        candidates[:, 0] = subgoals_np               # original
        candidates[:, 1] = state_diff                # state difference
        
        # Fill remaining with Gaussian noise around state_diff
        std = 0.5 * self.subgoal_space.bounds  # half of subgoal bounds as std
        for i in range(2, num_candidates):
            noise = np.random.normal(0, std, size=(batch_size, subgoal_dim)).astype(np.float32)
            candidates[:, i] = state_diff + noise
        
        # Clip all candidates to the valid subgoal space
        candidates = np.clip(candidates, self.subgoal_space.low, self.subgoal_space.high)
        
        # --- Score each candidate ---
        # For each candidate, we simulate what the current low level would do along the 
        # stored trajectory. We measure the difference from the actual actions taken.
        # Best candidate = one whose predicted actions best match the observed actions.
        
        # Reshape for efficient batched policy queries
        # We want, for each (batch_item, timestep, candidate):
        #   current_subgoal_at_t = apply subgoal_transition from candidate through trajectory
        #   predicted_action = low_level_policy(state_at_t, current_subgoal_at_t)
        
        # Flatten batch and timesteps for easier policy calls: (batch * window, ...)
        flat_states = state_traj_np.reshape(-1, state_traj_np.shape[-1])
        # flat_states shape: (batch * window, state_dim)

        # Loop-invariant precomputations (do these ONCE, not per candidate):
        # - the observed actions to compare against
        # - the initial sub-state used in the subgoal shift
        # - the states tensor on device
        flat_actual_actions = action_traj_np.reshape(-1, action_traj_np.shape[-1])
        initial_state_sub = state_traj_np[:, 0, :subgoal_dim]  # (batch, sg_dim)
        state_t = torch.from_numpy(flat_states).float().to(self.device)
        # FIX: mask out padded timesteps of windows that ended early (goal reached), otherwise
        # the final (s, a) pair is counted (window - traj_len + 1) times in the score.
        if traj_lens is None:
            valid = np.ones((batch_size, window), dtype=np.float32)
        else:
            tl = traj_lens.cpu().numpy().reshape(-1)
            valid = (np.arange(window)[None, :] < tl[:, None]).astype(np.float32)

        with torch.no_grad():
            log_probs = np.zeros((batch_size, num_candidates), dtype=np.float32)
            
            for c in range(num_candidates):
                candidate_sg = candidates[:, c]  # (batch, subgoal_dim)
                
                # Compute the shifted subgoal at each timestep of the trajectory.
                # At t=0, subgoal = candidate.
                # At t>0, subgoal shifts via transition function (as agent moves).
                # 
                # For each window, we need the subgoal values at all `window` timesteps.
                # Subgoal transition: sg_{t+1} = s_t + sg_t - s_{t+1}
                # Equivalently: sg_t = candidate + s_0 - s_t  (in the relevant dims)
                
                shifted_sgs = (
                    candidate_sg[:, np.newaxis, :]        # (batch, 1, sg_dim)
                    + initial_state_sub[:, np.newaxis, :]  # (batch, 1, sg_dim)
                    - state_traj_np[:, :, :subgoal_dim]    # (batch, window, sg_dim)
                )
                # shape: (batch, window, sg_dim)
                
                # Flatten for policy query: (batch * window, sg_dim)
                flat_sgs = shifted_sgs.reshape(-1, subgoal_dim)
                
                # Query current low-level actor for what it would have done
                sg_t = torch.from_numpy(flat_sgs).float().to(self.device)
                predicted_actions = self.low_controller.actor(state_t, sg_t)
                predicted_actions = predicted_actions.cpu().numpy()
                # shape: (batch * window, action_dim)
                
                # Compare to actual actions taken
                action_diff = predicted_actions - flat_actual_actions
                # shape: (batch * window, action_dim)
                
                # Log-probability under Gaussian assumption: -0.5 * ||diff||^2
                # Sum over action dimensions and over timesteps within window
                squared_diff = np.sum(action_diff ** 2, axis=-1)  # (batch * window,)
                squared_diff = squared_diff.reshape(batch_size, window)  # (batch, window)
                log_probs[:, c] = -0.5 * np.sum(squared_diff * valid, axis=-1)  # (batch,)
        
        # Pick the best candidate for each batch item
        best_indices = np.argmax(log_probs, axis=-1)  # (batch,)
        best_subgoals = candidates[np.arange(batch_size), best_indices]  # (batch, subgoal_dim)
        
        return torch.from_numpy(best_subgoals).float().to(self.device)
    
    
    def _train_high(self):
        """
        One TD3 update on the high-level controller, WITH off-policy correction.
        Relabels stored subgoals to ones most likely to have induced observed actions
        under the current low-level policy.
        """
        states, goals, subgoals, next_states, rewards, not_dones, state_traj, action_traj, traj_lens = \
            self.high_buffer.sample()

        # Off-policy correction: relabel subgoals
        corrected_subgoals = self._off_policy_correction(
            states=states,
            subgoals=subgoals,
            next_states=next_states,
            state_trajectories=state_traj,
            action_trajectories=action_traj,
            traj_lens=traj_lens,
        )
        
        return self._td3_update(
            controller=self.high_controller,
            states=states,
            goals=goals,
            actions=corrected_subgoals,  # Use CORRECTED subgoals, not original
            next_states=next_states,
            next_goals=goals,
            rewards=rewards,
            not_dones=not_dones,
        )
    
    def _td3_update(self, controller, states, goals, actions, next_states, 
                    next_goals, rewards, not_dones):
        """
        Run one TD3 update on a given controller.
        Factored out because low and high levels share the same update logic
        (just with different data and different goal semantics).
        """
        controller.total_updates += 1
        
        # Target Q
        with torch.no_grad():
            noise = torch.randn_like(actions) * controller.policy_noise
            noise = torch.clamp(noise, -controller.noise_clip, controller.noise_clip)
            
            next_actions = controller.actor_target(next_states, next_goals) + noise
            next_actions = torch.clamp(next_actions, -controller.action_scale, controller.action_scale)
            
            target_q1 = controller.critic1_target(next_states, next_actions, next_goals)
            target_q2 = controller.critic2_target(next_states, next_actions, next_goals)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + not_dones * controller.gamma * target_q
        
        # Update critics
        current_q1 = controller.critic1(states, actions, goals)
        current_q2 = controller.critic2(states, actions, goals)
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        controller.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        controller.critic1_optimizer.step()
        
        controller.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        controller.critic2_optimizer.step()
        
        losses = {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
        }
        
        # Delayed actor update
        if controller.total_updates % controller.policy_freq == 0:
            actor_loss = -controller.critic1(states, controller.actor(states, goals), goals).mean()
            
            controller.actor_optimizer.zero_grad()
            actor_loss.backward()
            controller.actor_optimizer.step()
            
            controller._soft_update(controller.actor, controller.actor_target)
            controller._soft_update(controller.critic1, controller.critic1_target)
            controller._soft_update(controller.critic2, controller.critic2_target)
            
            losses['actor_loss'] = actor_loss.item()
        
        return losses
    
    # ==================== Save / Load ====================
    
    def save(self, path_prefix):
        self.low_controller.save(f"{path_prefix}_low.pth")
        self.high_controller.save(f"{path_prefix}_high.pth")
    
    def load(self, path_prefix):
        self.low_controller.load(f"{path_prefix}_low.pth")
        self.high_controller.load(f"{path_prefix}_high.pth")


if __name__ == "__main__":
    # Sanity check
    from subgoal import SubgoalSpace, ANTMAZE_SUBGOAL_BOUNDS
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    state_dim = 27
    action_dim = 8
    goal_dim = 2
    action_scale = np.ones(action_dim, dtype=np.float32)
    
    subgoal_space = SubgoalSpace(ANTMAZE_SUBGOAL_BOUNDS)
    
    agent = HiroAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        goal_dim=goal_dim,
        action_scale=action_scale,
        subgoal_space=subgoal_space,
        device=device,
        window_length=10,
        buffer_size=1000,
        batch_size=64,
        start_training_steps=100,
    )
    
    print(f"Subgoal dim: {agent.subgoal_dim}")
    print(f"Window length: {agent.window_length}")
    
    # Simulate an episode
    print("\nSimulating an episode of 25 steps...")
    initial_state = np.random.randn(state_dim).astype(np.float32)
    final_goal = np.random.randn(goal_dim).astype(np.float32)
    
    agent.reset_episode(initial_state, final_goal)
    state = initial_state
    
    for step in range(25):
        action = agent.select_action(state, global_step=step)
        next_state = np.random.randn(state_dim).astype(np.float32)
        reward = np.random.randn()
        terminated = (step == 24)  # pretend goal reached at end
        truncated = False
        
        agent.store_transition(state, action, reward, next_state, terminated, truncated)
        state = next_state
    
    print(f"Low buffer size:  {len(agent.low_buffer)} (should be 25)")
    print(f"High buffer size: {len(agent.high_buffer)} (should be ~3, from 25 steps / 10 window)")
    
    # Try training (will fail if buffers are too small, but shouldn't crash)
    print("\nAttempting training steps...")
    for step in range(150, 155):
        losses = agent.train(global_step=step)
        if losses:
            loss_str = ", ".join(f"{k}={v:.3f}" for k, v in losses.items() if 'loss' in k)
            print(f"  Step {step}: {loss_str}")
    
    print("\nHiroAgent works correctly.")


    """  # Continuation of sanity check: add more experiences, then train
    print("\nAdding more experiences to enable training...")
    for _ in range(200):
        state = np.random.randn(state_dim).astype(np.float32)
        action = np.random.uniform(-1, 1, size=action_dim).astype(np.float32)
        next_state = np.random.randn(state_dim).astype(np.float32)
        reward = np.random.randn()
        done = False
        
        # Directly poke the buffers to avoid needing full episode logic
        subgoal = subgoal_space.sample()
        next_subgoal = subgoal_space.sample()
        agent.low_buffer.add(
            state=state, subgoal=subgoal, action=action,
            next_state=next_state, next_subgoal=next_subgoal,
            reward=reward, done=done,
        )
    
    # Also pad the high buffer
    for _ in range(100):
        agent.high_buffer.add(
            state=np.random.randn(state_dim).astype(np.float32),
            goal=np.random.randn(goal_dim).astype(np.float32),
            subgoal=subgoal_space.sample(),
            next_state=np.random.randn(state_dim).astype(np.float32),
            reward=np.random.randn(),
            done=False,
            state_trajectory=np.random.randn(10, state_dim).astype(np.float32),
            action_trajectory=np.random.randn(10, action_dim).astype(np.float32),
        )
    
    print(f"Low buffer size:  {len(agent.low_buffer)}")
    print(f"High buffer size: {len(agent.high_buffer)}")
    
    print("\nAttempting training steps now:")
    for step in range(150, 155):
        losses = agent.train(global_step=step)
        if losses:
            loss_str = ", ".join(f"{k}={v:.3f}" for k, v in losses.items() if 'loss' in k)
            print(f"  Step {step}: {loss_str}")
        else:
            print(f"  Step {step}: (no training happened)")


    print("\nTesting off-policy correction...")
    
    # Sample a batch from the high buffer
    states, goals, subgoals, next_states, rewards, not_dones, state_traj, action_traj = \
        agent.high_buffer.sample()
    
    corrected = agent._off_policy_correction(
        states=states,
        subgoals=subgoals,
        next_states=next_states,
        state_trajectories=state_traj,
        action_trajectories=action_traj,
    )
    
    print(f"  Original subgoals shape:   {subgoals.shape}")
    print(f"  Corrected subgoals shape:  {corrected.shape}")
    print(f"  Max difference per sample: {(corrected - subgoals).abs().max(dim=-1).values[:5].tolist()}")
    print("  Correction works.") """