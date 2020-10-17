import torch
import numpy as np

from .SumTreeTorch import SumTreeTorch

class PrioritizedExperienceReplayTorch:
    def __init__(self, capacity, input_shape, device='cpu'):
        self._sum_tree = SumTreeTorch(capacity, input_shape, device)

        # avoid - hyperparameter to avoid some experiences that have 0 probabilty
        self._epsilon = 0.01 
        # greedy action probability
        self._alpha = 0.6 
        # self._beta-sampling, from initial value increasing to 1
        self._beta = 0.4 
        self._beta_inc = 0.001

        self._absolute_error_upper = 1 

    def Append(self, state, action, next_state, reward, done, priority=0):
        self._ready = False

        if priority == 0:
            max_priority = torch.max(self._sum_tree._tree[-self._sum_tree._capacity:])
            if max_priority == 0:
                max_priority = self._absolute_error_upper
            
            priority = max_priority
            
        self._sum_tree.Add(priority, state, action, next_state, reward, done)
        self._ready = True

    def Sample(self, batch_size):
        values = []

        priority_segment = (self._sum_tree.Total() / batch_size).cpu()

        for i in range(batch_size):
            a = priority_segment * i
            b = priority_segment * (i+1)

            values.append(np.random.uniform(a, b))

        
        indices, priorities, states, actions, next_states, rewards, dones = self._sum_tree.GetLeaves(values)

        # get weights
        sampling_probs = priorities / self._sum_tree.Total()
        weights = (self._sum_tree._entries * sampling_probs) ** -self._beta
        weights /= weights.max()

        self._beta = np.min([1,0, self._beta + self._beta_inc])

        return states, actions, next_states, rewards, dones, indices, weights

    def BatchUpdate(self, tree_idx, errors):
        abs_errors = abs(errors + self._epsilon)
        clipped_errors = np.minimum(abs_errors, self._absolute_error_upper)
        priorities = np.power(clipped_errors, self._alpha)

        for ti, p in zip(tree_idx, priorities):
            self._sum_tree.Update(ti, p)
        
    def __len__(self):
        return self._sum_tree._entries
