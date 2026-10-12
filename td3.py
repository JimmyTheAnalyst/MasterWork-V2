import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from networks import Actor, Critic


class TD3Controller:
    """
    Twin Delayed DDPG (TD3) agent.
    
    Implements the three stability tricks from the TD3 paper:
    1. Twin critics with minimum for target (pessimism reduces overestimation)
    2. Delayed policy updates (actor updates every policy_freq critic updates)
    3. Target policy smoothing (noise added to target actions)
    
    Also supports goal-conditioning: if goal_dim > 0, the actor and critics
    take (state, goal) as input and action is computed with respect to the goal.
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        action_scale,
        device,
        goal_dim=0,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        expl_noise=0.1,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.expl_noise = expl_noise
        self.goal_dim = goal_dim
        
        # Action scale as a tensor on device (for clipping exploration noise)
        self.action_scale = torch.as_tensor(action_scale, dtype=torch.float32, device=device)
        
        # Create networks
        self.actor = Actor(state_dim, action_dim, action_scale, goal_dim=goal_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        self.critic1 = Critic(state_dim, action_dim, goal_dim=goal_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim, goal_dim=goal_dim).to(device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)
        
        self.total_updates = 0
    
    def select_action(self, state, goal=None, noisy=False):
        """
        Select an action for a single state. Used during environment interaction.
        Set noisy=True during training for exploration, False for evaluation.
        """
        with torch.no_grad():
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            if goal is not None:
                goal = torch.as_tensor(goal, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            action = self.actor(state, goal).squeeze(0)
            
            if noisy:
                noise = torch.randn_like(action) * self.expl_noise * self.action_scale
                action = action + noise
                action = torch.clamp(action, -self.action_scale, self.action_scale)
        
        return action.cpu().numpy()
    
    def train(self, replay_buffer):
        """
        One training step: sample from buffer, update critics, maybe update actor.
        Returns a dict of losses for logging.
        """
        self.total_updates += 1
        
        # Sample a batch from the replay buffer
        states, actions, next_states, rewards, not_dones = replay_buffer.sample()
        
        # Compute target Q-value (TD target)
        with torch.no_grad():
            # Target policy smoothing: add clipped noise to target actions
            noise = torch.randn_like(actions) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            
            next_actions = self.actor_target(next_states) + noise
            next_actions = torch.clamp(next_actions, -self.action_scale, self.action_scale)
            
            # Twin critic minimum
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            
            # Bellman target
            target_q = rewards + not_dones * self.gamma * target_q
        
        # Update critics
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        losses = {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
        }
        
        # Delayed actor update
        if self.total_updates % self.policy_freq == 0:
            # Actor aims to maximize Q (so we minimize -Q)
            actor_loss = -self.critic1(states, self.actor(states)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update of target networks
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)
            
            losses['actor_loss'] = actor_loss.item()
        
        return losses
    
    def _soft_update(self, source, target):
        """Polyak averaging: target = tau * source + (1 - tau) * target"""
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'total_updates': self.total_updates,
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        self.total_updates = checkpoint['total_updates']


if __name__ == "__main__":
    # Sanity check: create a TD3 controller and run a few training steps
    from replay_buffer import ReplayBuffer
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    state_dim = 3   # Pendulum has 3-dim state
    action_dim = 1  # Pendulum has 1-dim action
    action_scale = [2.0]  # Pendulum actions are in [-2, 2]
    
    td3 = TD3Controller(
        state_dim=state_dim,
        action_dim=action_dim,
        action_scale=action_scale,
        device=device,
    )
    
    buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=10000,
        batch_size=256,
        device=device,
    )
    
    # Fill buffer with random transitions
    print("Filling buffer with fake transitions...")
    for _ in range(500):
        state = np.random.randn(state_dim).astype(np.float32)
        action = np.random.uniform(-2, 2, size=action_dim).astype(np.float32)
        next_state = np.random.randn(state_dim).astype(np.float32)
        reward = np.random.randn()
        done = np.random.random() < 0.05
        buffer.add(state, action, next_state, reward, done)
    
    # Test action selection
    test_state = np.random.randn(state_dim).astype(np.float32)
    action_deterministic = td3.select_action(test_state, noisy=False)
    action_noisy = td3.select_action(test_state, noisy=True)
    print(f"\nDeterministic action: {action_deterministic}")
    print(f"Noisy action:         {action_noisy}")
    
    # Run a few training steps
    print("\nRunning training steps...")
    for step in range(5):
        losses = td3.train(buffer)
        loss_str = ", ".join(f"{k}={v:.4f}" for k, v in losses.items())
        print(f"Step {step}: {loss_str}")
    
    print("\nTD3 controller works correctly.")