import numpy as np
class UniformExperienceReplay:
    def __init__(self, capacity, input_shape):
        self._states = np.zeros([capacity, *input_shape], dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.float32)
        self._next_states = np.zeros([capacity, *input_shape], dtype=np.float32)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.int64)
        
        self._pointer = 0
        self._size = 0
        self._capacity = capacity

    def __len__(self):
        return self._size

    def Append(self, state, action, next_state, reward, done):            
        self._states[self._pointer] = state
        self._actions[self._pointer] = action
        self._next_states[self._pointer] = next_state
        self._rewards[self._pointer] = reward
        self._dones[self._pointer] = done
        
        self._pointer = (self._pointer + 1) % self._capacity
        self._size = min(self._size + 1 , self._capacity)

    def Sample(self, batch_size):
        indices_np = np.random.choice(self._size, batch_size, replace=False)

        states_np = self._states[indices_np]
        actions_np = self._actions[indices_np]
        next_states_np = self._next_states[indices_np]
        rewards_np = self._rewards[indices_np]
        dones_np = self._dones[indices_np]

        weights_np = np.ones(batch_size)

        return states_np, actions_np, next_states_np, rewards_np, dones_np, indices_np, weights_np
    
    # just do nothing because PER requires this and having checks is more cpu than running nothing.
    def BatchUpdate(self, tree_idx, errors):
        pass
