import torch
import torch.nn as nn 

class ActorCriticNetwork(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        
        self._actor = actor
        self._critic = critic

    def forward(self, state):
        action_pred = self._actor(state)
        value_pred = self._critic(state)

        return action_pred, value_pred