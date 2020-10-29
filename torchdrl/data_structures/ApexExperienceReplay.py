import numpy as np
import collections

import time

from .ExperienceReplay import ExperienceReplay

class ApexExperieceReplay(ExperienceReplay):
    def __init__(self, capacity, input_shape, n_step=1, gamma=0.99):
        super(ApexExperieceReplay, self).__init__(capacity, input_shape, n_step, gamma)
        self._errors = np.zeros(self._capacity, dtype=np.float32)
            
    def BatchAppend(self, states, actions, next_states, rewards, dones, errors, batch_size):
        for i in range(batch_size):
            self.Append(states[i], actions[i], next_states[i], rewards[i], dones[i], errors[i])

    def Append(self, state, action, next_state, reward, done, error):
        # Wait until ready
        self._WaitStatus()

        # set the ready flag off
        self._ReadyDown()
    
        transition = (state, action, next_state, reward, done, error)
        self._n_step_buffer.append(transition)

        if len(self._n_step_buffer) >= self._n_step:
            # create a n-step transition
            n_next_state, n_reward, n_done, n_error = self._GetNStepInfo(self._n_step_buffer, self._gamma)
            n_state, n_action = self._n_step_buffer[0][:2]

            self._states[self._pointer] = n_state
            self._actions[self._pointer] = n_action
            self._next_states[self._pointer] = n_next_state
            self._rewards[self._pointer] = n_reward
            self._dones[self._pointer] = n_done
            self._errors[self._pointer] = n_error
            
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
        errors_np = self._errors[indices]

        self._ReadyUp()

        return (states_np, actions_np, next_states_np, rewards_np, dones_np, errors_np)

    def SampleBatchFromIndices(self, indices):
        self._WaitStatus()

        self._ReadyDown()

        states_np, actions_np, next_states_np, rewards_np, dones_np = self._GetMemories(indices)
        errors_np = self._errors[indices]

        self._ReadyUp()

        return (states_np, actions_np, next_states_np, rewards_np, dones_np, errors_np)

    def Pop(self, batch_size):
        self._WaitStatus()

        # set the ready flag off
        self._ReadyDown()

        indices_np = np.array(range(batch_size))
        states_np, actions_np, next_states_np, rewards_np, dones_np = self._GetMemories(indices_np)
        errors_np = self._errors[indices_np]

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
        self._ReadyUp()

        return states_np, actions_np, next_states_np, rewards_np, dones_np, errors_np, indices_np

    def _GetNStepInfo(self, n_step_buffer, gamma):
        next_state, reward, done, error = n_step_buffer[-1][-4:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            n_s, r, d, err = transition[-4:]

            reward = r + gamma * reward * (1-d)
            next_state, done, error = (n_s, d, err) if d else (next_state, done, error)
        
        return next_state, reward, done, error

    def _WaitStatus(self):
        while not self._ready:
            time.sleep(0.1)
    
    def _ReadyDown(self):
        self._ready = False

    def _ReadyUp(self):
        self._ready = True