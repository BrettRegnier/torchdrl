import collections
import numpy as np
import random

from .ExperienceReplay import ExperienceReplay
from .SumSegmentTree import SumSegmentTree

class PrioritizedExperienceReplay(ExperienceReplay):
    def __init__(
        self, 
        capacity,
        input_shape, 
        n_step=1,
        gamma=0.99,
        alpha=0.6,
        beta=0.2,
        beta_inc=0.0001
    ):

        assert alpha >= 0

        super(PrioritizedExperienceReplay, self).__init__(
            capacity, input_shape, n_step, gamma
        )

        self._priority_epsilon = 1e-6
        
        self._max_priority = 1.0
        self._min_priority = float("inf")
        self._tree_pointer = 0
        self._alpha = alpha
        self._beta = beta
        self._beta_inc = beta_inc

        tree_capacity = 1 
        while tree_capacity < self._capacity:
            tree_capacity *= 2
        
        self._sum_tree = SumSegmentTree(self._capacity)

    def Append(
        self,
        state,
        action,
        next_state,
        reward,
        done
    ):
        experience = super().Append(state, action, next_state, reward, done)

        if experience:
            priority = self._CalculatePriority(self._max_priority)
            self._sum_tree.Add(priority)
            self._UpdateBounds(priority)

        return experience

    def Sample(self, batch_size):
        indices = []
        segment_priority = self._sum_tree.Total() / batch_size

        for i in range(batch_size):
            a = segment_priority * i 
            b = segment_priority * (i + 1)
            value = random.uniform(a, b)
            idx = self._sum_tree.Retrieve(value)
            indices.append(idx)
        
        # states = self._states[indices]
        # actions = self._actions[indices]
        # next_states = self._next_states[indices]
        # rewards = self._rewards[indices]
        # dones = self._dones[indices]
        
        states, actions, next_states, rewards, dones, weights = self.SampleBatchFromIndices(indices)
        indices = np.array(indices, dtype=np.int64)

        self._beta = np.min([1.0, self._beta + self._beta_inc])

        return (
            states,
            actions,
            next_states,
            rewards, 
            dones,
            weights,
            indices
        )
        
    def SampleBatchFromIndices(self, indices):
        states, actions, next_states, rewards, dones, _ = super().SampleBatchFromIndices(indices)
        weights = self._CalculateWeights(indices)
        
        return (
            states,
            actions,
            next_states,
            rewards,
            dones,
            weights            
        )

    def BatchUpdate(self, indices, errors):
        for idx, error in zip(indices, errors):
            self._sum_tree.UpdatePriority(idx, error)
            
    def _CalculateWeights(self, indices):
        weights = []
        for i in indices:
            p_min = self._min_priority / self._sum_tree.Total() + self._priority_epsilon
            max_weight = (p_min * len(self)) ** (-self._beta)

            p_sample = self._sum_tree.Get(i) / self._sum_tree.Total()
            weight = (p_sample * len(self) + self._priority_epsilon) ** (-self._beta)
            weight = weight / max_weight
            weights.append(weight)
        
        weights_np = np.array(weights, dtype=np.float32)
        return weights_np

    def _UpdateBounds(self, priority):
        if priority > self._max_priority:
            self._max_priority = priority
        if priority < self._min_priority:
            self._min_priority = priority

    def _CalculateWeight(self, idx, beta):
        p_min = self._min_priority / self._sum_tree.Total() + self._priority_epsilon
        max_weight = (p_min * len(self)) ** (-beta)

        p_sample = self._sum_tree.Get(idx) / self._sum_tree.Total()
        weight = (p_sample * len(self) + self._priority_epsilon) ** (-beta)
        weight = weight / max_weight

        return weight

    def _CalculatePriority(self, priority):
        return (priority + self._priority_epsilon) ** self._alpha
