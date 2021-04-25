import collections
import numpy as np
import random

from .ExperienceReplay import ExperienceReplay
from .SumSegmentTree import SumSegmentTree

class PrioritizedExperienceReplay(ExperienceReplay):
    """
    Implemenation of the Prioritized Experience Replay paper https://arxiv.org/abs/1511.05952
    Inspired from https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/03.per.ipynb
    
    Arguments:
        input_shape {tuple:int, list:int} -- The input shape of the 
            observation space
        capacity {int} -- Number of memories stored in the buffer
        n_step {int} -- Number of steps to rollout
        gamma {float} -- Discount factor for rollout
        alpha {float} -- Max priority multiplier [0-1]
        beta {float} -- Importance sampling, linearly increasing to 1. 
            Where 1 is max importance
        beta_inc {float} -- Beta increment value per sample retrieval
    """

    def __init__(
        self, 
        input_shape, 
        capacity=1,
        n_step=1,
        gamma=0.99,
        alpha=0.2,
        beta=0.6,
        beta_inc=0.001
    ):
        assert alpha >= 0

        super(PrioritizedExperienceReplay, self).__init__(
            input_shape, capacity, n_step, gamma
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
