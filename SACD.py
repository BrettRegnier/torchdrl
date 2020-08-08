import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from torch.optim import Adam

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class BaseNetwork(nn.Module):
    def __init__(self, input_shape, n_actions, convo=False):
        super(BaseNetwork, self).__init__()
        
        self._convo = convo
        self._out_features = 64

        if self._convo:
            self._net = nn.Sequential(
                nn.Conv2d(input_shape[0], 80, kernel_size=3),
                nn.ReLU(),
                Flatten() # flatten for the linear
            )
            out = self._conv(torch.zeros(1, *input_shape))
            self._out_features = [int(np.prod(out.size()))]
        else:
            total_inputs = 1
            for x in input_shape:
                total_inputs *= x
            
            input_shape = [total_inputs]

            self._net = nn.Sequential(
                nn.Linear(*input_shape, 64),
                nn.ReLU(),
                # nn.Linear(1024, 1024),
                # nn.ReLU(),
                nn.Linear(64, self._out_features),
                nn.ReLU(),
            )

    def forward(self, states):
        return self._net(states)

class QNetwork(nn.Module):
    def __init__(self, input_shape, n_actions, convo=False, dueling=False):
        super(QNetwork, self).__init__()		
        
        self._body = BaseNetwork(input_shape, n_actions, convo)

        if dueling:
            self._action = nn.Sequential(
                nn.Linear(self._body._out_features, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, n_actions),
            )

            self._value = nn.Sequential(
                nn.Linear(self._body._out_features, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 1),
            )
        else:
            self._head = nn.Sequential(
                # nn.Linear(self._body._out_features, 1024),
                # nn.ReLU(inplace=True),
                nn.Linear(64, n_actions),
            )

        self._dueling = dueling
    
    def forward(self, states):
        x = self._body(states)
        
        if not self._dueling:
            return self._head(x)
        else:
            x = self._body(x)
            action = self._body(x)
            value = self._body(x)
            return value + action - action.mean(1, keepdim=True)

# sac
class TwinnedQNetwork(nn.Module):
    def __init__(self, input_shape, n_actions, convo=False, dueling=False):
        super(TwinnedQNetwork, self).__init__()
        self._Q1 = QNetwork(input_shape, n_actions, convo, dueling)
        self._Q2 = QNetwork(input_shape, n_actions, convo, dueling)

    def forward(self, states):
        return self._Q1(states), self._Q2(states)

# actor
class CategoricalPolicy(nn.Module):
    def __init__(self, input_shape, n_actions, convo=False):
        super(CategoricalPolicy, self).__init__()

        self._body = BaseNetwork(input_shape, n_actions, convo)

        self._head = nn.Sequential(
            nn.Linear(self._body._out_features, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, n_actions)
        )

    def forward(self, states):
        x = self._body(states)

        return self._head(x)

    def Sample(self, states):
        out = self.forward(states)

        action_probs = F.softmax(out, dim=1)
        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1, 1)

        # stabilize by adding a small value z
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs

# TODO make a base agent.

class AgentSACD:
    def __init__(self, input_shape, n_actions, lr, device, convo=False, dueling=False, gamma=0.99, target_entropy_ratio=0.98):
        self._device = device
        self._gamma = gamma

        self._actor = CategoricalPolicy(input_shape, n_actions, convo).to(self._device)
        self._online_critic = TwinnedQNetwork(input_shape, n_actions, convo, dueling).to(self._device)
        self._target_critic = TwinnedQNetwork(input_shape, n_actions, convo, dueling).to(self._device).eval()

        # load target and disable grad
        self._target_critic.load_state_dict(self._online_critic.state_dict())
        for param in self._target_critic.parameters():
            param.requires_grad = False
        
        self._actor_optimizer = Adam(self._actor.parameters(), lr=lr)
        self._q1_optimizer = Adam(self._online_critic._Q1.parameters(), lr=lr)
        self._q2_optimizer = Adam(self._online_critic._Q2.parameters(), lr=lr)

        # target entropy
        self._target_entropy = -np.log(1.0/n_actions) * target_entropy_ratio

        # optimize log(alpha)
        self._log_alpha = torch.zeros(1, requires_grad=True, device=self._device)
        self._alpha = self._log_alpha.exp()
        self._alpha_optimizer = Adam([self._log_alpha], lr=lr)

    def Act(self, state):
        state_t = torch.tensor([state], dtype=torch.float32).to(self._device)
        out = self._actor(state_t)
        action_t = torch.argmax(out, dim=1, keepdim=True)
        return action_t.item()


    # TODO might need to convert to torch tensors here
    def Value(self, states, actions):
        state_value1, state_value2 = self._online_critic(states)
        return state_value1.gather(1, actions.long()), state_value2.gather(1, actions.long())
    
    # TODO multistep gamma... gamma ** multistep
    @torch.no_grad()
    def TargetValue(self, rewards, next_states, dones):
        _, action_probs, log_action_probs = self._actor.Sample(next_states)
        next_state_value1, next_state_value2 = self._target_critic(next_states)
        next_value = (action_probs * 
        (torch.min(next_state_value1, next_state_value2) - self._alpha 
        * log_action_probs)).sum(dim=1, keepdim=True)

        return rewards + (1.0 - dones) * self._gamma

    # weights are for PER if used other wise weights=1.0
    def Loss(self, batch, weights):
        states, actions, rewards, next_states, dones = batch

        states_t = torch.tensor(states, dtype=torch.float32).to(self._device)
        actions_t = torch.tensor(actions, dtype=torch.float32).to(self._device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(self._device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32).to(self._device)
        dones_t = torch.tensor(dones, dtype=torch.float32).to(self._device)

        ### critic loss ###
        state_values1, state_values2 = self.Value(states_t, actions_t)
        target_values = self.TargetValue(rewards_t, next_states_t, dones_t)

        # td errors for priority weighting in PER
        errors = torch.abs(state_values1.detach() - target_values)

        # get mean of values
        value_mean1 = state_values1.mean().item()
        value_mean2 = state_values2.mean().item()

        # mean squared TD error with priority weight
        value_loss1 = torch.mean((state_values1 - target_values).pow(2) * weights)
        value_loss2 = torch.mean((state_values2 - target_values).pow(2) * weights)


        ### actor loss ###
        _, action_probs, log_action_probs = self._actor.Sample(states_t)

        with torch.no_grad():
            value1, value2 = self._online_critic(states_t)
            q_value = torch.min(value1, value2)

        # expectations of entropy
        entropies = -torch.sum(action_probs * log_action_probs, dim=1, keepdim=True)

        # expected Q
        expected_q_value = torch.sum(q_value * action_probs, dim=1, keepdim=True)

        # loss with priority weights
        actor_loss = (weights * (-q_value - self._alpha * entropies)).mean()


        ### entropy loss ###
        entropies = entropies.detach()
        entropy_loss = -torch.mean(self._log_alpha * (self._target_entropy - entropies) * weights)


        return value_loss1, value_loss2, actor_loss, entropy_loss

    def Learn(self, batch, weights):
        value_loss1, value_loss2, actor_loss, entropy_loss = self.Loss(batch, weights)

        self.UpdateParams(self._q1_optimizer, value_loss1)
        self.UpdateParams(self._q2_optimizer, value_loss2)
        self.UpdateParams(self._actor_optimizer, actor_loss)
        self.UpdateParams(self._alpha_optimizer, entropy_loss)

        self._alpha = self._log_alpha.exp()

    def UpdateParams(self, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

