import numpy as np

from .SumTree import SumTree

class PrioritizedExperienceReplay:
    avoid = 0.01 # hyperparameters to avoid some experiences that have 0 probabilty
    epsilon = 0.6 # greedy
    importance = 0.4 # importance-sampl;ing, from initial value increasing to 1

    importance_inc = 0.001

    absolute_error_upper = 1 

    def __init__(self, capacity):
        self.sum_tree = SumTree(capacity)

    def Store(self, experience):
        max_priority = np.max(self.sum_tree.tree[-self.sum_tree.capacity:])

        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.sum_tree.Add(max_priority, experience)

    def Sample(self, batch_size):
        minibatch = []

        batch_idx = np.empty((batch_size,), dtype=np.int32)

        priority_segment = self.sum_tree.TotalPriority / batch_size

        for i in range(batch_size):
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            idx, priority, data = self.sum_tree.GetLeaf(value)

            batch_idx[i] = idx

            minibatch.append([data[0], data[1], data[2], data[3], data[4]])

        return batch_idx, minibatch

    def BatchUpdate(self, tree_idx, abs_errors):
        abs_errors += self.avoid
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        priorities = np.power(clipped_errors, self.epsilon)

        for ti, p in zip(tree_idx, priorities):
            self.sum_tree.Update(ti, p)
