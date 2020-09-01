import gym
from gym import spaces
import random
import math


class FiveNumSort(gym.Env):
    def __init__(self, **kwargs):
        super(FiveNumSort, self).__init__()

        self._numbers = [4, 1, 2, 5, 3]
        if "numbers" in kwargs.keys():
            self._numbers = kwargs['numbers']

        largest = max(self._numbers)
        self._compressed_numbers = [x/largest for x in self._numbers]

        self._actions = []
        for x in range(len(self._numbers)):
            for y in range(x+1, len(self._numbers)):
                self._actions.append((x,y))

        self.action_space = spaces.Discrete(len(self._actions))
        self.observation_space = spaces.Box(low=0, high=10, shape=(1, len(self._numbers)))
        self._steps = 0

    def step(self, action):
        done = True
        self._steps += 1
        reward = max(0.1, 1 / self._steps)

        i, j = self._actions[action]
        self.swap(i, j)

        for x in range(0, len(self._numbers) -1):
            if self._numbers[x] > self._numbers[x+1]:
                done = False
                reward = -0.0001

        # self.render()
        return self._compressed_numbers, reward, done, {}

    def render(self, mode="human"):
        print(self._numbers)

    def swap(self, i, j):
        tmp = self._numbers[i]
        self._numbers[i] = self._numbers[j]
        self._numbers[j] = tmp

        tmp = self._compressed_numbers[i]
        self._compressed_numbers[i] = self._compressed_numbers[j]
        self._compressed_numbers[j] = tmp

    def reset(self):
        self._steps = 0
        
        in_order = True
        while in_order:
            for i in range(len(self._numbers) * 10):
                i, j = random.choice(self._actions)
                self.swap(i, j)

            for x in range(0, len(self._numbers) - 1):
                if self._numbers[x] > self._numbers[x+1]:
                    in_order = False
        

        return self._compressed_numbers

    def seed(self, seed):
        self._seed = seed
        random.seed(seed)

# test = FiveNumSort()
# test.render()
# test.swap(0, 1)
# test.render()
