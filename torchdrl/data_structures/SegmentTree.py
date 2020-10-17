import numpy as np

class SegmentTree:
    def __init__(self, capacity):
        # TODO make sure is > 0 and is a power of 2

        self._capacity = capacity
        self._data_capacity = (capacity * 2) - 1

        self._tree = np.zeros(self._capacity)
        self._data = np.zeros(self._data_capacity, dtype=object)

        self._pointer = 0
        self._entries = 0
