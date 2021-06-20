from typing import Iterable
import numpy as np
import collections

import time

from torchdrl.data_structures.PrioritizedExperienceReplay import PrioritizedExperienceReplay

class ApexExperieceReplay(PrioritizedExperienceReplay):            
    def BatchAppend(self, states, actions, next_states, rewards, dones, errors, batch_size):
        for i in range(batch_size):
            self.Append(states[i], actions[i], next_states[i], rewards[i], dones[i], errors[i])

    def Append(self, state, action, next_state, reward, done, error):
        # Wait until ready
        self._WaitStatus()

        # set the ready flag off
        self._ReadyDown()

        # constantly update this value so that it will be returned implicitly
        self._max_priority = error
        transition = super().Append(state, action, next_state, reward, done)
        
        # set the ready flag on
        self._ReadyUp()

        return transition

    def _CalculatePriority(self, priority):
        return priority
        
    def Sample(self, batch_size):
        self._WaitStatus()

        self._ReadyDown()
        
        states_np, actions_np, next_states_np, rewards_np, dones_np, weights_np, indices_np = super().Sample(batch_size)

        self._ReadyUp()

        return (states_np, actions_np, next_states_np, rewards_np, dones_np, weights_np, indices_np)

    def SampleBatchFromIndices(self, indices):
        self._WaitStatus()

        self._ReadyDown()

        states_np, actions_np, next_states_np, rewards_np, dones_np, weights_np = self.SampleBatchFromIndices(indices)

        self._ReadyUp()

        return (states_np, actions_np, next_states_np, rewards_np, dones_np, weights_np)

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

        for i in range((self._sum_tree._capacity+1) - batch_size):
            self._sum_tree.UpdatePriority(i, self._sum_tree[i+batch_size])
            self._sum_tree.UpdatePriority(i+batch_size, 0)
        
        # Slide the pointer back a little or move to the now new open spots are
        self._sum_tree._pointer = self._sum_tree._capacity - (self._sum_tree._capacity - batch_size)
        self._pointer = self._capacity - (self._capacity - batch_size)
        self._size -= batch_size

        # set the ready flag on
        self._ReadyUp()

        return states_np, actions_np, next_states_np, rewards_np, dones_np, errors_np, indices_np

    def _Rollout(self):
        # get the last experience
        next_state, reward, done, error = self._n_step_buffer[-1][-3:]

        for experience in reversed(list(self._n_step_buffer)[:-1]):
            n_next_state, n_reward, n_done, n_err = experience[-3:]

            reward = n_reward + self._gamma * reward * (1 - n_done)
            
            if n_done:
                next_state = n_next_state
                done = n_done
                error = n_err

        return next_state, reward, done, error

    def _WaitStatus(self):
        while not self._ready:
            time.sleep(0.1)
    
    def _ReadyDown(self):
        self._ready = False

    def _ReadyUp(self):
        self._ready = True