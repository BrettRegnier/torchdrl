import numpy as np

from .ExperienceReplay import ExperienceReplay

class UniformExperienceReplay:
    def __init__(self, capacity, input_shape):
        # super(UniformExperienceReplay, self).__init__()
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
        
    def BatchAppend(self, states, actions, next_states, rewards, dones, priorities, batch_size):
        for i in range(batch_size):
            self.Append(states[i], actions[i], next_states[i], rewards[i], dones[i], priorities[i])

    def Append(self, state, action, next_state, reward, done):
        self._ready = False

        self._states[self._pointer] = state
        self._actions[self._pointer] = action
        self._next_states[self._pointer] = next_state
        self._rewards[self._pointer] = reward
        self._dones[self._pointer] = done
        
        self._pointer = (self._pointer + 1) % self._capacity
        self._size = min(self._size + 1 , self._capacity)

        self._ready = True

    def Sample(self, batch_size):
        self._ready = False

        indices_np = np.random.choice(self._size, batch_size, replace=False)

        self._ready = True
        return self._GetMemories(indices_np, batch_size)
    
    def Pop(self, batch_size):
        self._ready = False

        indices_np = np.array(range(batch_size))
        states_np, actions_np, next_states_np, rewards_np, dones_np, indices_np, weights_np = self._GetMemories(indices_np, batch_size)

        self._states[:-batch_size] = self._states[batch_size:]
        self._states[-batch_size:] = 0

        self._actions[:-batch_size] = self._actions[batch_size:]
        self._actions[-batch_size:] = 0

        self._next_states[:-batch_size] = self._next_states[batch_size:]
        self._next_states[-batch_size:] = 0

        self._rewards[:-batch_size] = self._rewards[batch_size:]
        self._rewards[-batch_size:] = 0

        self._dones[:-batch_size] = self._dones[batch_size:]
        self._dones[-batch_size:] = 0

        self._pointer = max(0, self._pointer - batch_size)
        self._size -= batch_size

        self._ready = True
        return states_np, actions_np, next_states_np, rewards_np, dones_np, indices_np, weights_np

    def _GetMemories(self, indices_np, batch_size):
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
