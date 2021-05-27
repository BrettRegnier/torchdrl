import numpy as np
from numpy import random

import torchdrl.tools.Helper as Helper

class EpsilonGreedy:
    name="EpsilonGreedy"
    def __init__(self, epsilon:float=1.0, epsilon_min:float=0.01, epsilon_decay:float=0.99):
        self._epsilon = epsilon
        self._epsilon_min = epsilon_min
        self._epsilon_max = epsilon
        self._epsilon_decay = epsilon_decay

    def __call__(self, q_values, evaluate=False):
        if not isinstance(q_values, (np.ndarray,)):
            q_values = q_values.detach().cpu().numpy()

        batch_size, n_actions = q_values.shape

        # Get the greedy actions
        actions = np.argmax(q_values, axis=1)

        if not evaluate:
            # check if any of the actions will be epsilon selected
            randoms = np.random.random(size=batch_size) < self._epsilon
            actions[randoms] = np.random.choice(n_actions, sum(randoms))

            # update epsilon
            self._epsilon = max(self._epsilon_min, self._epsilon * self._epsilon_decay)
        return actions

    def Save(self, folderpath, filename):
        data = self.state_dict()

        Helper.SaveAgent(folderpath, filename, data)

    def state_dict(self):
        data = {}
        data['name'] = self.name
        data['epsilon'] = self._epsilon
        data['epsilon_min'] = self._epsilon_min
        data['epsilon_max'] = self._epsilon_max
        data['epsilon_decay'] = self._epsilon_decay

        return data

    def Load(self, path):
        checkpoint = Helper.LoadAgent(path)

        self.load_state_dict(checkpoint)

    def load_state_dict(self, dict):
        self._epsilon = dict['epsilon']
        self._epsilon_min = dict['epsilon_min']
        self._epsilon_max = dict['epsilon_max']
        self._epsilon_decay = dict['epsilon_decay']

