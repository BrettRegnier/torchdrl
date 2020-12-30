import collections
import numpy as np

# source from https://github.com/Curt-Park/rainbow-is-all-you-need 

class ExperienceReplay:
    def __init__(
        self, 
        capacity:int, 
        input_shape:(tuple, list),
        n_step:int=3,
        gamma:float=0.99
    ):
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

        self._n_step_buffer = collections.deque(maxlen=self._n_step)

    def Append(self, state, action, next_state, reward, done):
        experience = (state, action, next_state, reward, done)
        self._n_step_buffer.append(experience)

        if len(self._n_step_buffer) >= self._n_step:
            # create an n-step experience
            next_state, reward, done = self._Rollout()
            state, action = self._n_step_buffer[0][:2]

            self._states[self._pointer] = state
            self._actions[self._pointer] = action
            self._next_states[self._pointer] = next_state
            self._rewards[self._pointer] = reward
            self._dones[self._pointer] = done

            self._pointer = (self._pointer + 1) % self._capacity
            if self._size < self._capacity:
                self._size += 1

            experience = self._n_step_buffer[0]
        else:
            experience = ()

        return experience

    def SampleBatch(self, batch_size):
        indices = np.random.choice(self._size, size=batch_size, replace=False)

        weights = np.ones(batch_size, dtype=np.float32)

        return (
            self._states[indices],
            self._actions[indices],
            self._next_states[indices],
            self._rewards[indices],
            self._dones[indices],
            weights,
            indices
        )

    def SampleBatchFromIndices(self, indices):
        weights = np.ones(len(indices), dtype=np.float32)
        return (
            self._states[indices],
            self._actions[indices],
            self._next_states[indices],
            self._rewards[indices],
            self._dones[indices],
            weights
        )

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

    def __len__(self):
        return self._size

