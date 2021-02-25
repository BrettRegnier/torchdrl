import os

import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import random

from torchdrl.agents.markov.DQL import DQL

from torchdrl.neural_networks.FullyConnectedNetwork import FullyConnectedNetwork
from torchdrl.representations.Plotter import Plotter


class DoubleDQL(DQL):
    def __init__(self, env, oracle=None, **kwargs):
        super(DoubleDQL, self).__init__(env, oracle, **kwargs)

    def CalculateErrors(self, states_t, actions_t, next_states_t, rewards_t, dones_t, indices_np, weights_t, batch_size, gamma):
        q_values = self._net(states_t).gather(1, actions_t.unsqueeze(1))

        with torch.no_grad():
            next_q_values = self._target_net(next_states_t)
            next_actions = self._net(next_states_t).argmax(dim=1, keepdim=True)
            next_state_action_pair_values = next_q_values.gather(
                1, next_actions).detach()

        q_target = (rewards_t + gamma * next_state_action_pair_values *
                    (1-dones_t)).to(self._device)

        errors = F.smooth_l1_loss(q_values, q_target, reduction="none")
        return errors
