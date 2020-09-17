import numpy as np

class SumTree:
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity
        
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def Add(self, priority, data):
        tree_idx = self.data_pointer + self.capacity - 1

        self.data[self.data_pointer] = data

        self.Update(tree_idx, priority)

        self.data_pointer += 1

        # overwrite old data
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def Update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority

        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def GetLeaf(self, p_value):
        parent_idx = 0 

        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1

            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if p_value <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    v -= self.tree[left_child_idx]
                    parent_idx = right_child_idx
            
        data_index = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_index]

    @property
    def TotalPriority(self):
        return self.tree[0]