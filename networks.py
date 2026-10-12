import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """
    TD3 Actor network. Takes a state and optional goal, outputs an action
    scaled to the environment's action range.
    
    When goal_dim=0, this behaves as a standard state-only actor (for flat TD3).
    When goal_dim>0, state and goal are concatenated as input (for goal-conditioned HIRO).
    """
    def __init__(self, state_dim, action_dim, action_scale, goal_dim=0, hidden_dim=300):
        super().__init__()
        input_dim = state_dim + goal_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        
        # Register action_scale as a non-trainable parameter so it moves with .to(device)
        action_scale = torch.as_tensor(action_scale, dtype=torch.float32)
        self.register_buffer('action_scale', action_scale)
    
    def forward(self, state, goal=None):
        if goal is not None:
            x = torch.cat([state, goal], dim=-1)
        else:
            x = state
        raw_action = self.net(x)
        return self.action_scale * torch.tanh(raw_action)


class Critic(nn.Module):
    """
    TD3 Critic network (single Q-function).
    We'll instantiate two of these for TD3's twin-critic trick.
    Takes (state, goal, action) and outputs a scalar Q-value.
    """
    def __init__(self, state_dim, action_dim, goal_dim=0, hidden_dim=300):
        super().__init__()
        input_dim = state_dim + goal_dim + action_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, state, action, goal=None):
        if goal is not None:
            x = torch.cat([state, goal, action], dim=-1)
        else:
            x = torch.cat([state, action], dim=-1)
        return self.net(x)


if __name__ == "__main__":
    # Quick sanity check — create networks and run a forward pass
    state_dim = 27
    action_dim = 8
    goal_dim = 15
    action_scale = torch.ones(action_dim)
    
    actor = Actor(state_dim, action_dim, action_scale, goal_dim=goal_dim)
    critic = Critic(state_dim, action_dim, goal_dim=goal_dim)
    
    batch_size = 4
    state = torch.randn(batch_size, state_dim)
    goal = torch.randn(batch_size, goal_dim)
    
    action = actor(state, goal)
    q_value = critic(state, action, goal)
    
    print(f"State shape:   {state.shape}")
    print(f"Goal shape:    {goal.shape}")
    print(f"Action shape:  {action.shape}")
    print(f"Q-value shape: {q_value.shape}")
    print(f"Action range:  [{action.min():.3f}, {action.max():.3f}] (should be in [-1, 1])")