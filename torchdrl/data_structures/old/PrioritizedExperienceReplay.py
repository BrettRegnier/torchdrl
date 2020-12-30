import collections
import numpy as np
import random
import math

from .ExperienceReplay import ExperienceReplay
from .SegmentTree_v2 import SegmentTree_v2

class PrioritizedExperienceReplay(ExperienceReplay):
    def __init__(self, capacity, input_shape, alpha, beta, priority_epsilon, n_step=1, gamma=0.99):
        super(PrioritizedExperienceReplay, self).__init__(capacity, input_shape, n_step, gamma)
        self._capacity = capacity
        self._input_shape = input_shape
        self._n_step = n_step
        self._gamma = gamma

        tree_capacity = 1
        while tree_capacity < self._capacity:
            tree_capacity *= 2
        self._sum_tree = SegmentTree_v2(tree_capacity)

        # tranisition buffer for N-step learning
        self._n_step_buffer = collections.deque(maxlen=self._n_step)

        # avoid - hyperparameter to avoid some experiences that have 0 probabilty
        self._priority_epsilon = priority_epsilon
        # greedy action probability
        self._alpha = alpha
        # sampling, from initial value increasing to 1
        self._beta = beta

        # TODO maybe change
        self._beta_inc = 0.00001 # TODO maybe parameter?

        self._max_priority = 1 
        self._min_priority = math.inf

        self._tree_pointer = 0

    def BatchAppend(self, states, actions, next_states, rewards, dones, priorities, batch_size):
        for i in range(batch_size):
            self.Append(states[i], actions[i], next_states[i], rewards[i], dones[i], self._GetPriority(priorities[i]))

    def Append(self, state, action, next_state, reward, done, priority=0):
        transition = super().Append(state, action, next_state, reward, done)
        
        if transition:
            # priority = transition[5]
            if priority == 0:
                priority = self._max_priority ** self._alpha
            self.UpdateMaxMinPriority(priority)
            self._sum_tree[self._tree_pointer] = priority
            self._tree_pointer = (self._tree_pointer + 1) % self._capacity

        return transition

    def Sample(self, batch_size):
        indices = []
        priorities = []

        # priority_segment = (self._sum_tree.Total() - self._min_priority) / batch_size
        priority_segment = self._sum_tree.Total() / batch_size
        
        for i in range(batch_size):
            a = priority_segment * i
            b = priority_segment * (i + 1)

            # pick a random value between a and b
            value = random.uniform(a, b)

            idx, priority = self._sum_tree.GetLeaf(value)

            priorities.append(priority)
            indices.append(idx)

        weights = []
        for i, idx in enumerate(indices):
            p_min = (self._min_priority / self._sum_tree.Total()) + 1e-5

            max_weight = (p_min * self._size) ** (-self._beta)

            # calculate weights
            p_sample = priorities[i] / self._sum_tree.Total()
            weight = ((p_sample * self._size) + 1e-5) ** (-self._beta)
            weight = weight / max_weight
            weights.append(weight)

        
        indices_np = np.array(indices)
        weights_np = np.array(weights)
        states_np, actions_np, next_states_np, rewards_np, dones_np = self._GetMemories(indices_np)

        # update beta
        self._beta = np.min([1.0, self._beta + self._beta_inc])

        return states_np, actions_np, next_states_np, rewards_np, dones_np, indices_np, weights_np

    def BatchUpdate(self, indices, errors):
        for idx, error in zip(indices, errors):
            self._sum_tree[idx] = self._GetPriority(error)

    def _GetPriority(self, error):
        return (error + self._priority_epsilon) ** self._alpha

    def UpdateMaxMinPriority(self, priority):
        if priority > self._max_priority:
            self._max_priority = priority
        
        if priority < self._min_priority:
            self._min_priority = priority

    def __len__(self):
        return self._sum_tree._entries
