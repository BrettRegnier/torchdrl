import collections
from typing import Iterable
import numpy as np

# source from https://github.com/Curt-Park/rainbow-is-all-you-need 

class ExperienceReplay:    
    """
    Uniform Experience Replay
    Inspired from https://github.com/Curt-Park/rainbow-is-all-you-need 
    
    Arguments:
        input_shape {tuple:int, list:int} -- The input shape of the 
            observation space
        capacity {int} -- Number of memories stored in the buffer
        n_step {int} -- Number of steps to rollout
        gamma {float} -- Discount factor for rollout
    """

    def __init__(
        self, 
        input_shape:Iterable,
        capacity:int=1, 
        n_step:int=1,
        gamma:float=0.99
    ):
        self._capacity = capacity
        self._n_step = n_step
        self._gamma = gamma
        self._pointer = 0
        self._size = 0
        
        if any(isinstance(el, (list, tuple)) for el in input_shape):
            self._n_states = len(input_shape)
            # self._states = np.zeros(self._capacity, dtype=object)
            # self._next_states = np.zeros(self._capacity, dtype=object)
        else:
            self._n_states = 1
            input_shape = [input_shape]
        
        
        self._states = []
        self._next_states = []
        for i in range(self._n_states):
            self._states.append(np.zeros([self._capacity, *input_shape[i]], dtype=np.float32))
            self._next_states.append(np.zeros([self._capacity, *input_shape[i]], dtype=np.float32))

        self._actions = np.zeros(self._capacity, dtype=np.float32)
        self._rewards = np.zeros(self._capacity, dtype=np.float32)
        self._dones = np.zeros(self._capacity, dtype=np.int64)

        self._n_step_buffer = collections.deque(maxlen=self._n_step)

    def Append(self, state, action, next_state, reward, done):
        experience = (state, action, next_state, reward, done)
        self._n_step_buffer.append(experience)

        if len(self._n_step_buffer) >= self._n_step:
            # create an n-step experience
            next_state, reward, done = self._Rollout()
            state, action = self._n_step_buffer[0][:2]
            
            if self._n_states > 1:
                for i in range(self._n_states):
                    self._states[i][self._pointer] = state[i]
                    self._next_states[i][self._pointer] = next_state[i]
            else:
                self._states[0][self._pointer] = state
                self._next_states[0][self._pointer] = next_state

            self._actions[self._pointer] = action
            self._rewards[self._pointer] = reward
            self._dones[self._pointer] = done

            self._pointer = (self._pointer + 1) % self._capacity
            if self._size < self._capacity:
                self._size += 1

            experience = self._n_step_buffer[0]
        else:
            experience = ()

        return experience

    def Sample(self, batch_size):
        indices_np = np.random.choice(self._size, size=batch_size, replace=False)
        
        return self.SampleBatchFromIndices(indices_np) + (indices_np,)

    def SampleBatchFromIndices(self, indices):
        return self._GetMemories(indices)

    def _Rollout(self):
        # get the last experience
        next_state, reward, done = self._n_step_buffer[-1][-3:]

        for experience in reversed(list(self._n_step_buffer)[:-1]):
            n_next_state, n_reward, n_done = experience[-3:]

            reward = n_reward + self._gamma * reward * (1 - n_done)
            
            if n_done:
                next_state = n_next_state
                done = n_done

        return next_state, reward, done
    
    def _GetMemories(self, indices):
        weights = np.ones(len(indices), dtype=np.float32)
        
        if self._n_states > 1:
            states = []
            next_states = []
            for i in range(self._n_states):
                states.append(self._states[i][indices])
                next_states.append(self._next_states[i][indices])
        else:
            states = self._states[0][indices]
            next_states = self._next_states[0][indices]
        
        return (
            states,
            self._actions[indices],
            next_states,
            self._rewards[indices],
            self._dones[indices],
            weights
        )

    def __len__(self):
        return self._size

    def GetNStep(self):
        """
        Returns the n_step private parameter
        """
        return self._n_step

    # just do nothing because PER requires this and having checks is more cpu than running nothing.
    def BatchUpdate(self, indices, errors):
        pass

