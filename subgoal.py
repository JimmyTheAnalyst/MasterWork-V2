import numpy as np


class SubgoalSpace:
    """
    Defines the bounded space of subgoals for the high-level policy.
    
    For AntMaze, the subgoal is a 15-dim vector representing target changes
    in a subset of state dimensions (position, orientation, joint angles).
    Each dimension has its own range.
    
    For other environments, pass different bounds.
    """
    def __init__(self, bounds):
        """
        bounds: np.array of shape (dim,) giving the symmetric bound for each 
                dimension. The space is [-bounds[i], +bounds[i]] per dimension.
        """
        self.bounds = np.asarray(bounds, dtype=np.float32)
        self.dim = len(bounds)
        self.low = -self.bounds
        self.high = self.bounds
    
    def sample(self):
        """Sample a random subgoal uniformly from the space."""
        return np.random.uniform(self.low, self.high).astype(np.float32)
    
    def clip(self, subgoal):
        """Clip a subgoal to the valid range."""
        return np.clip(subgoal, self.low, self.high)


# Default bounds for AntMaze (copied from HIRO paper / reference implementation)
# 15 dimensions: x, y position offsets, z (height), quaternion orientation (4), 
# then 8 joint angles/velocities
#ANTMAZE_SUBGOAL_BOUNDS = np.array([
#    10.0, 10.0,                    # x, y position
#    0.5,                           # z height
#    1.0, 1.0, 1.0, 1.0,            # quaternion orientation
#    0.5, 0.3, 0.5, 0.3,            # hip joints
#    0.5, 0.3, 0.5, 0.3,            # knee joints
#], dtype=np.float32)
ANTMAZE_SUBGOAL_BOUNDS = np.array([10.0, 10.0], dtype=np.float32)


if __name__ == "__main__":
    # Sanity check with AntMaze subgoal space
    space = SubgoalSpace(ANTMAZE_SUBGOAL_BOUNDS)
    print(f"Subgoal dim: {space.dim}")
    print(f"Bounds: {space.bounds}")
    
    # Sample a few subgoals
    print("\nSampled subgoals:")
    for i in range(3):
        sg = space.sample()
        print(f"  {sg}")
    
    # Test clipping
    out_of_bounds = np.ones(space.dim, dtype=np.float32) * 20.0  # way too big
    clipped = space.clip(out_of_bounds)
    print(f"\nClipped 20s to: {clipped}")