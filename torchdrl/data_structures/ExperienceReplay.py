import numpy as np
import collections

import time

class ExperienceReplay:
    def __init__(self, capacity, input_shape, n_step=1, gamma=0.99):
        self._ready = True

        self._capacity = capacity
        self._n_step = n_step
        self._gamma = gamma
        self._pointer = 0
        self._size = 0
        
        self._states = np.zeros([self._capacity, *input_shape], dtype=np.float32)
        self._actions = np.zeros(self._capacity, dtype=np.float32)
        self._next_states = np.zeros([self._capacity, *input_shape], dtype=np.float32)
        self._rewards = np.zeros(self._capacity, dtype=np.float32)
        self._dones = np.zeros(self._capacity, dtype=np.int64)

        # TODO move into its own class.
        # needed for Ape-X PER not the most optimized for when its not needed but that doesn't matter right now
        # self._errors = np.zeros(self._capacity, dtype=np.float32)

        self._n_step_buffer = collections.deque(maxlen=self._n_step)
    
    def __len__(self):
        return self._size
        
    def BatchAppend(self, states, actions, next_states, rewards, dones, priorities, batch_size):
        for i in range(batch_size):
            self.Append(states[i], actions[i], next_states[i], rewards[i], dones[i], priorities[i])

    def Append(self, state, action, next_state, reward, done):
        # Wait until ready
        self._WaitStatus()

        # set the ready flag off
        self._ReadyDown()
    
        transition = (state, action, next_state, reward, done)
        self._n_step_buffer.append(transition)

        if len(self._n_step_buffer) >= self._n_step:
            # create a n-step transition
            n_next_state, n_reward, n_done = self._GetNStepInfo(self._n_step_buffer, self._gamma)
            n_state, n_action = self._n_step_buffer[0][:2]

            self._states[self._pointer] = n_state
            self._actions[self._pointer] = n_action
            self._next_states[self._pointer] = n_next_state
            self._rewards[self._pointer] = n_reward
            self._dones[self._pointer] = n_done
            
            self._pointer = (self._pointer + 1) % self._capacity
            if self._size < self._capacity:
                self._size += 1

            transition = self._n_step_buffer[0]
        else:
            transition = ()

        # set the ready flag on
        self._ReadyUp()

        return transition
        
    def Sample(self, batch_size):
        self._WaitStatus()

        self._ReadyDown()

        indices = np.random.choice(self._size, batch_size, replace=False)
        states_np, actions_np, next_states_np, rewards_np, dones_np = self._GetMemories(indices)
        weights_np = np.ones(batch_size)

        self._ReadyUp()

        return (states_np, actions_np, next_states_np, rewards_np, dones_np, weights_np)

    def SampleBatchFromIndices(self, indices):
        self._WaitStatus()

        self._ReadyDown()

        states_np, actions_np, next_states_np, rewards_np, dones_np = self._GetMemories(indices)

        self._ReadyUp()

        return (states_np, actions_np, next_states_np, rewards_np, dones_np)

    def Pop(self, batch_size):
        self._WaitStatus()

        # set the ready flag off
        self._ReadyDown()

        indices_np = np.array(range(batch_size))
        states_np, actions_np, next_states_np, rewards_np, dones_np = self._GetMemories(indices_np)

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

        weights_np = np.ones(batch_size)

        self._pointer = max(0, self._pointer - batch_size)
        self._size -= batch_size

        # set the ready flag on
        self._ReadyUp()

        return states_np, actions_np, next_states_np, rewards_np, dones_np, weights_np, indices_np

    def _GetMemories(self, indices_np):
        states_np = self._states[indices_np]
        actions_np = self._actions[indices_np]
        next_states_np = self._next_states[indices_np]
        rewards_np = self._rewards[indices_np]
        dones_np = self._dones[indices_np]
        # errors_np = self._errors[indices_np]

        return states_np, actions_np, next_states_np, rewards_np, dones_np

    def _GetNStepInfo(self, n_step_buffer, gamma):
        next_state, reward, done = n_step_buffer[-1][-3:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            n_s, r, d = transition[-3:]

            reward = r + gamma * reward * (1-d)
            next_state, done = (n_s, d) if d else (next_state, done)
        
        return next_state, reward, done

    def _WaitStatus(self):
        while not self._ready:
            time.sleep(0.1)
    
    def _ReadyDown(self):
        self._ready = False

    def _ReadyUp(self):
        self._ready = True