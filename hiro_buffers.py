import numpy as np
import torch

from replay_buffer import ReplayBuffer


class LowReplayBuffer(ReplayBuffer):
    """
    Replay buffer for the low-level controller.
    
    Extends ReplayBuffer with subgoal and next_subgoal fields. The low level
    is a goal-conditioned TD3 agent where the 'goal' is the current subgoal
    set by the high level (which shifts as the agent moves).
    """
    def __init__(self, state_dim, action_dim, subgoal_dim, buffer_size, batch_size, device):
        super().__init__(state_dim, action_dim, buffer_size, batch_size, device)
        self.subgoals = np.zeros((buffer_size, subgoal_dim), dtype=np.float32)
        self.next_subgoals = np.zeros((buffer_size, subgoal_dim), dtype=np.float32)
    
    def add(self, state, subgoal, action, next_state, next_subgoal, reward, done):
        # Fill parent fields
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.next_states[self.ptr] = next_state
        self.rewards[self.ptr] = reward
        self.not_dones[self.ptr] = 1.0 - float(done)
        
        # Fill subgoal fields
        self.subgoals[self.ptr] = subgoal
        self.next_subgoals[self.ptr] = next_subgoal
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def sample(self):
        idx = np.random.randint(0, self.size, size=self.batch_size)
        
        return (
            torch.from_numpy(self.states[idx]).to(self.device),
            torch.from_numpy(self.subgoals[idx]).to(self.device),
            torch.from_numpy(self.actions[idx]).to(self.device),
            torch.from_numpy(self.next_states[idx]).to(self.device),
            torch.from_numpy(self.next_subgoals[idx]).to(self.device),
            torch.from_numpy(self.rewards[idx]).to(self.device),
            torch.from_numpy(self.not_dones[idx]).to(self.device),
        )


class HighReplayBuffer(ReplayBuffer):
    """
    Replay buffer for the high-level controller.
    
    Each entry represents one high-level window (typically 10 low-level steps).
    Stores the start state, final goal (for goal-conditioning), subgoal chosen,
    end state, accumulated reward, and — critically — the full trajectory of
    low-level states and actions during the window. That trajectory is needed
    for the off-policy correction we'll add later.
    """
    def __init__(self, state_dim, goal_dim, subgoal_dim, action_dim, 
                 buffer_size, batch_size, window_length, device):
        # window_length = number of low-level steps per high-level step
        
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.window_length = window_length
        
        self.ptr = 0
        self.size = 0
        
        # Single-transition fields at high-level timescale
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.goals = np.zeros((buffer_size, goal_dim), dtype=np.float32)
        self.subgoals = np.zeros((buffer_size, subgoal_dim), dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.not_dones = np.zeros((buffer_size, 1), dtype=np.float32)
        
        # Trajectory fields — sequences of low-level states and actions during window
        self.state_trajectories = np.zeros(
            (buffer_size, window_length, state_dim), dtype=np.float32
        )
        self.action_trajectories = np.zeros(
            (buffer_size, window_length, action_dim), dtype=np.float32
        )
        # FIX: number of real (unpadded) steps in each stored window
        self.traj_lens = np.full((buffer_size, 1), window_length, dtype=np.int64)

    def add(self, state, goal, subgoal, next_state, reward, done,
            state_trajectory, action_trajectory, traj_len=None):
        self.states[self.ptr] = state
        self.goals[self.ptr] = goal
        self.subgoals[self.ptr] = subgoal
        self.next_states[self.ptr] = next_state
        self.rewards[self.ptr] = reward
        self.not_dones[self.ptr] = 1.0 - float(done)
        self.state_trajectories[self.ptr] = state_trajectory
        self.action_trajectories[self.ptr] = action_trajectory
        self.traj_lens[self.ptr] = self.window_length if traj_len is None else traj_len

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def sample(self):
        idx = np.random.randint(0, self.size, size=self.batch_size)
        
        return (
            torch.from_numpy(self.states[idx]).to(self.device),
            torch.from_numpy(self.goals[idx]).to(self.device),
            torch.from_numpy(self.subgoals[idx]).to(self.device),
            torch.from_numpy(self.next_states[idx]).to(self.device),
            torch.from_numpy(self.rewards[idx]).to(self.device),
            torch.from_numpy(self.not_dones[idx]).to(self.device),
            # Trajectories kept as tensors for off-policy correction later
            torch.from_numpy(self.state_trajectories[idx]).to(self.device),
            torch.from_numpy(self.action_trajectories[idx]).to(self.device),
            torch.from_numpy(self.traj_lens[idx]).to(self.device),
        )
    
    def __len__(self):
        return self.size


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test LowReplayBuffer
    print("Testing LowReplayBuffer...")
    low_buffer = LowReplayBuffer(
        state_dim=27, action_dim=8, subgoal_dim=15,
        buffer_size=1000, batch_size=32, device=device,
    )
    
    for _ in range(100):
        low_buffer.add(
            state=np.random.randn(27).astype(np.float32),
            subgoal=np.random.randn(15).astype(np.float32),
            action=np.random.randn(8).astype(np.float32),
            next_state=np.random.randn(27).astype(np.float32),
            next_subgoal=np.random.randn(15).astype(np.float32),
            reward=np.random.randn(),
            done=False,
        )
    
    batch = low_buffer.sample()
    print(f"  Low buffer size: {len(low_buffer)}")
    print(f"  Batch shapes: {[b.shape for b in batch]}")
    
    # Test HighReplayBuffer
    print("\nTesting HighReplayBuffer...")
    high_buffer = HighReplayBuffer(
        state_dim=27, goal_dim=2, subgoal_dim=15, action_dim=8,
        buffer_size=1000, batch_size=32, window_length=10, device=device,
    )
    
    for _ in range(100):
        high_buffer.add(
            state=np.random.randn(27).astype(np.float32),
            goal=np.random.randn(2).astype(np.float32),
            subgoal=np.random.randn(15).astype(np.float32),
            next_state=np.random.randn(27).astype(np.float32),
            reward=np.random.randn(),
            done=False,
            state_trajectory=np.random.randn(10, 27).astype(np.float32),
            action_trajectory=np.random.randn(10, 8).astype(np.float32),
        )
    
    batch = high_buffer.sample()
    print(f"  High buffer size: {len(high_buffer)}")
    print(f"  Batch shapes: {[b.shape for b in batch]}")
    
    print("\nBoth buffers working correctly.")