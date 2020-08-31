import torch
import torch.nn as nn
import torch.distributions as dis 
import torch.nn.functional as F 
import torch.optim as optim

import numpy as np

from agents.SAC import SAC
from agents.BaseAgent import BaseAgent

from models.ConvolutionNetwork import ConvolutionNetwork as CN
from models.FullyConnectedNetwork import FullyConnectedNetwork as FCN
from models.TwoHeadedNetwork import TwoHeadedNetwork as THN

# TODO remove the connection of SAC and SACD due to the creation of the SAC network and then deletion waste of resources.
# TODO generalize creating a convo network (using the config file)

class SACD(SAC):
    name="SACD"
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        assert self._action_type == "DISCRETE", "Action types must be discrete. Use SAC for continuous actions"

        fc_in_shape = self._input_shape
        
        self._critic_convo = None
        if self._hyperparameters['critic_convo'] is not None:
            cnc = self._hyperparameters['critic_convo']
            self._critic_convo = CN(self._input_shape, cnc['filters'], cnc['kernels'], cnc['strides'], cnc['paddings'], cnc['activations'], cnc['pools'], cnc['flatten']).to(self._device)
            fc_in_shape = (self._critic_convo.OutputSize(),)

        critic_fc = self._hyperparameters['critic_fc']
        
        self._actor_convo = None
        if self._hyperparameters['actor_convo'] is not None:
            cnc = self._hyperparameters['actor_convo']
            self._actor_convo = CN(self._input_shape, cnc['filters'], cnc['kernels'], cnc['strides'], cnc['paddings'], cnc['activations'], cnc['pools'], cnc['flatten']).to(self._device)

        actor_fc = self._hyperparameters['actor_fc']
        
        # self._critic1 = FCN(fc_in_shape, self._n_actions, critic_fc['hidden_layers'], critic_fc['activations'], critic_fc['final_activation']).to(self._device)
        # self._critic2 = FCN(fc_in_shape, self._n_actions, critic_fc['hidden_layers'], critic_fc['activations'], critic_fc['final_activation']).to(self._device)

        # self._critic_optimizer1 = optim.Adam(self._critic1.parameters(), lr=self._hyperparameters['critic_lr'], eps=1e-4)
        # self._critic_optimizer2 = optim.Adam(self._critic2.parameters(), lr=self._hyperparameters['critic_lr'], eps=1e-4)

        # self._critic_target1 = FCN(fc_in_shape, self._n_actions, critic_fc['hidden_layers'], critic_fc['activations'], critic_fc['final_activation']).to(self._device)
        # self._critic_target2 = FCN(fc_in_shape, self._n_actions, critic_fc['hidden_layers'], critic_fc['activations'], critic_fc['final_activation']).to(self._device)
        
        # self.CopyNetwork(self._critic1, self._critic_target1)
        # self.CopyNetwork(self._critic2, self._critic_target2)

        self._critics = THN(self._input_shape, self._n_actions, critic_fc['hidden_layers'], critic_fc['activations'], critic_fc['final_activation'], self._hyperparameters['critic_convo']).to(self._device)
        self._critics_optimizer = optim.Adam(self._critics.parameters(), lr=self._hyperparameters['critic_lr'], eps=1e-4)
        self._critics_targets = THN(self._input_shape, self._n_actions, critic_fc['hidden_layers'], critic_fc['activations'], critic_fc['final_activation'], self._hyperparameters['critic_convo']).to(self._device)
        self.CopyNetwork(self._critics, self._critics_targets)
        
        self._actor = FCN(self._input_shape, self._n_actions, actor_fc['hidden_layers'], actor_fc['activations'], actor_fc['final_activation'], self._hyperparameters['actor_convo']).to(self._device)
        self._actor_optimizer = optim.Adam(self._actor.parameters(), lr=self._hyperparameters['actor_lr'], eps=1e-4)

        self._target_entropy = -torch.prod(torch.tensor(self._env.action_space.shape).to(self._device)).item()
        self._log_alpha = torch.zeros(1, requires_grad=True, device=self._device)
        self._alpha = self._log_alpha.exp().detach()
        self._alpha_optimizer = optim.Adam([self._log_alpha], lr=self._hyperparameters['alpha_lr'])

        self._add_noise = False
        # TODO add noise to output

        self._evaluate = True #TODO eval

        self._gamma = config['gamma']

    def ActionInfo(self, state_t):
        
        # if self._actor_convo is not None:
        #     state_t = self._actor_convo(state_t)

        action_probs = self._actor(state_t)
        max_prop_action = torch.argmax(action_probs).unsqueeze(0)
        action_distribution = dis.Categorical(action_probs)

        action = action_distribution.sample()

        # small eps to avoid 0s
        z = (action_probs == 0.0).float() * 1e-8 

        log_action_probabilities = torch.log(action_probs + z)
        return action, (action_probs, log_action_probabilities), max_prop_action

    def CriticLoss(self, states_t, actions_t, next_states_t, rewards_t, dones_t):
        # if self._critic_convo is not None:
        #     states_t = self._critic_convo(states_t)

        with torch.no_grad():
            next_state_action, (action_probs, log_action_probs), _ = self.ActionInfo(next_states_t)

            # if self._critic_convo is not None:
            #     next_states_t = self._critic_convo(next_states_t)

            # next_q_values1 = self._critic_target1(next_states_t)
            # next_q_values2 = self._critic_target2(next_states_t) 

            next_q_values1, next_q_values2 = self._critics_targets(next_states_t)

            min_q_next_value = action_probs * (torch.min(next_q_values1, next_q_values2) - self._alpha * log_action_probs)
            min_q_next_value = min_q_next_value.mean(dim=1)
            
        next_q_value = rewards_t + (1.0 - dones_t) * self._gamma * (min_q_next_value)

        actions_t_usq = actions_t.unsqueeze(-1)
        # actions_t_usq2 = actions_t.unsqueeze(-1)

        # states_t1 = states_t.clone()
        # states_t2 = states_t.clone()

        # out_values1 = self._critic1(states_t1)
        # gather_values1 = out_values1.gather(1, actions_t_usq1)
        # q_value1 = gather_values1.squeeze(-1)

        # out_values2 = self._critic2(states_t2)
        # gather_values2 = out_values2.gather(1, actions_t_usq2)
        # q_value2 = gather_values2.squeeze(-1)

        # q_value1 = self._critic1(states_t).gather(1, actions_t_usq).squeeze(-1)
        # q_value2 = self._critic2(states_t).gather(1, actions_t_usq).squeeze(-1)

        q_value1, q_value2 = self._critics(states_t)
        q_value1 = q_value1.gather(1, actions_t_usq).squeeze(-1)
        q_value2 = q_value2.gather(1, actions_t_usq).squeeze(-1)

        critic_loss1 = F.mse_loss(q_value1, next_q_value)
        critic_loss2 = F.mse_loss(q_value2, next_q_value)
        # critic_loss2 = torch.prod(torch.tensor([0.00000000001, 1], requires_grad=True))

        return critic_loss1, critic_loss2

    def ActorLoss(self, states_t):
        actions, (action_probs, log_action_probs), _ = self.ActionInfo(states_t)

        
        if self._actor_convo is not None:
            states_t = self._actor_convo(states_t)
        q_pi1 = self._critic1(states_t)
        q_pi2 = self._critic2(states_t)

        min_q_pi = torch.min(q_pi1, q_pi2)

        actor_loss = (action_probs * (self._alpha * log_action_probs - min_q_pi)).mean()
        log_action_probs = torch.sum(log_action_probs * action_probs, dim=1)

        return actor_loss, log_action_probs
