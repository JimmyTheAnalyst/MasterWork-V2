import numpy as np
import torch


class ReplayBuffer:
    """
    Basic replay buffer for TD3.
    Stores (state, action, next_state, reward, done) transitions.
    Samples uniformly at random for training.
    """
    def __init__(self, state_dim, action_dim, buffer_size, batch_size, device):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        
        self.ptr = 0
        self.size = 0
        
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.not_dones = np.zeros((buffer_size, 1), dtype=np.float32)
    
    def add(self, state, action, next_state, reward, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.next_states[self.ptr] = next_state
        self.rewards[self.ptr] = reward
        self.not_dones[self.ptr] = 1.0 - float(done)
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def sample(self):
        idx = np.random.randint(0, self.size, size=self.batch_size)
        
        return (
            torch.from_numpy(self.states[idx]).to(self.device),
            torch.from_numpy(self.actions[idx]).to(self.device),
            torch.from_numpy(self.next_states[idx]).to(self.device),
            torch.from_numpy(self.rewards[idx]).to(self.device),
            torch.from_numpy(self.not_dones[idx]).to(self.device),
        )
    
    def __len__(self):
        return self.size


if __name__ == "__main__":
    # Sanity check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    buffer = ReplayBuffer(
        state_dim=27,
        action_dim=8,
        buffer_size=1000,
        batch_size=32,
        device=device,
    )
    
    # Add 100 fake transitions
    for _ in range(100):
        state = np.random.randn(27).astype(np.float32)
        action = np.random.randn(8).astype(np.float32)
        next_state = np.random.randn(27).astype(np.float32)
        reward = np.random.randn()
        done = np.random.random() < 0.1
        buffer.add(state, action, next_state, reward, done)
    
    print(f"Buffer size after 100 adds: {len(buffer)}")
    
    # Sample a batch
    states, actions, next_states, rewards, not_dones = buffer.sample()
    print(f"Batch shapes:")
    print(f"  states:      {states.shape}")
    print(f"  actions:     {actions.shape}")
    print(f"  next_states: {next_states.shape}")
    print(f"  rewards:     {rewards.shape}")
    print(f"  not_dones:   {not_dones.shape}")
    print(f"  On device:   {states.device}")