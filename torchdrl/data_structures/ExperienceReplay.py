import numpy as np
import time

class ExperienceReplay:
    def __init__(self, capacity, input_shape):
        self._ready = True

        self._capacity = capacity
        self._pointer = 0
        self._size = 0
        
        self._states = np.zeros([capacity, *input_shape], dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.float32)
        self._next_states = np.zeros([capacity, *input_shape], dtype=np.float32)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.int64)

        # needed for PER not the most optimized for when its not needed but that doesn't matter right now
        self._errors = np.zeros(capacity, dtype=np.float32)
    
    def __len__(self):
        return self._size
        
    def BatchAppend(self, states, actions, next_states, rewards, dones, priorities, batch_size):
        for i in range(batch_size):
            self.Append(states[i], actions[i], next_states[i], rewards[i], dones[i], priorities[i])

    def Append(self, state, action, next_state, reward, done, error=0):
        self.WaitStatus()

        # set the ready flag off
        self.ReadyDown()

        self._states[self._pointer] = state
        self._actions[self._pointer] = action
        self._next_states[self._pointer] = next_state
        self._rewards[self._pointer] = reward
        self._dones[self._pointer] = done
        self._errors[self._pointer] = error
        
        self._pointer = (self._pointer + 1) % self._capacity

        if self._size < self._capacity:
            self._size += 1

        # set the ready flag on
        self.ReadyUp()

    def Pop(self, batch_size):
        self.WaitStatus()

        # set the ready flag off
        self.ReadyDown()

        indices_np = np.array(range(batch_size))
        states_np, actions_np, next_states_np, rewards_np, dones_np, errors_np, indices_np, weights_np = self._GetMemories(indices_np, batch_size)

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

        self._errors[:-batch_size] = self._errors[batch_size:]
        self._errors[-batch_size:] = 0

        self._pointer = max(0, self._pointer - batch_size)
        self._size -= batch_size

        # set the ready flag on
        self.ReadyUp()

        return states_np, actions_np, next_states_np, rewards_np, dones_np, errors_np, indices_np, weights_np

    def _GetMemories(self, indices_np, batch_size):
        states_np = self._states[indices_np]
        actions_np = self._actions[indices_np]
        next_states_np = self._next_states[indices_np]
        rewards_np = self._rewards[indices_np]
        dones_np = self._dones[indices_np]
        errors_np = self._errors[indices_np]

        weights_np = np.ones(batch_size)

        return states_np, actions_np, next_states_np, rewards_np, dones_np, errors_np, indices_np, weights_np

    def WaitStatus(self):
        while not self._ready:
            time.sleep(0.1)
    
    def ReadyDown(self):
        self._ready = False

    def ReadyUp(self):
        self._ready = True