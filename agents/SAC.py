import torch
import torch.distributions as distrib
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import sys
sys.path.insert(1, '../')# TODO remove

from agents.BaseAgent import BaseAgent
# from FullyConnectedNetwork import FullyConnected
from models.FullyConnectedNetwork import FullyConnectedNetwork as FCN
from models.TwoHeadedNetwork import TwoHeadedNetwork as THN

class SAC(BaseAgent):
    def __init__(self, config):
        super(SAC, self).__init__(config)
        assert self._action_type == "CONTINUOUS", "Action type must be continous, use SACD for discrete actions"

        name = "SAC"
        critic_in_shape = [self._input_shape[0] + self._n_actions]
        self._critic1 = FCN(critic_in_shape, 1).to(self._device)
        self._critic2 = FCN(critic_in_shape, 1).to(self._device)

        self._critic_optimizer1 = optim.Adam(self._critic1.parameters(), lr=lr)
        self._critic_optimizer2 = optim.Adam(self._critic2.parameters(), lr=lr)

        self._critic_target1 = FCN(critic_in_shape, 1).to(self._device)
        self._critic_target2 = FCN(critic_in_shape, 1).to(self._device)

        self._critic_target1.CopyModel(self._critic1)
        self._critic_target2.CopyModel(self._critic2)

        # self._actor = FCN(self._input_shape, self._n_actions*2).to(self._device)
        self._actor = THN(self._input_shape, self._n_actions).to(self._device)
        self._actor_optimizer = optim.Adam(self._actor.parameters(), lr=lr)

        # self._target_entropy = -torch.prod(torch.tensor(self._env.action_space.shape).to(self._device)).item()
        target_entropy_ratio = 0.98
        self._target_entropy = -np.log(1.0/self._n_actions) * target_entropy_ratio
        self._log_alpha = torch.zeros(1, requires_grad=True, device=self._device)
        self._alpha = self._log_alpha.exp()
        self._alpha_optimizer = optim.Adam([self._log_alpha], lr=lr)

        self._add_noise = False
        # TODO add noise to output

        self._evaluate = True #TODO eval

        self._gamma = gamma
    
    def Evaluate(self):
        pass
    

    def PlayEpisode(self, evaluate=False):
        done = False
        steps = 0
        episode_reward = 0

        state = self._env.reset()
        while steps != self._max_steps and not done:
            if self._total_steps < self._warm_up:
                action = self._env.action_space.sample()
            else:
                action = self.Act(state, evaluate=evaluate)

            next_state, reward, done, info = self._env.step(action)
            self._memory.Append(state, action, next_state, reward, done)

            if self._total_steps >= self._warm_up and self._total_steps > self._batch_size:
                self.Learn()

            episode_reward += reward
            state = next_state
            steps += 1
            self._total_steps += 1

        return episode_reward, steps, info

    # TODO the evaluation
    def Act(self, state, evaluate=False):
        ''' input: state -> list, eval -> bool
        Returns an action'''
        if evaluate:
            raise NotImplemented("Eval step not implemented")
        else:
            state_t = torch.tensor([state], dtype=torch.float32).to(self._device)
            action, _, _ = self.ActionInfo(state_t)

        action = action.detach().cpu().numpy()
        return action[0]

    def ActionInfo(self, state_t):
        ''' Action, log probabilities, and tanh of the mean '''
        output = self._actor(state_t)
        mean, log_std = output

        std = log_std.exp()
        normal = distrib.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        
        log_prob = normal.log_prob(z)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        # log_prob = (normal.log_prob(z) - torch.log(1 - (torch.tanh(z)).pow(2) + 1e-6)).sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)

    def Learn(self):

        # will crash if memory is empty...
        # TODO add error checking in memory
        states_t, actions_t, next_states_t, rewards_t, dones_t = self.SampleMemoryT(self._batch_size)
        
        critic_loss1, critic_loss2 = self.CriticLoss(states_t, actions_t, next_states_t, rewards_t, dones_t)
        
        self.OptimizationStep(self._critic_optimizer1, self._critic1, critic_loss1, self._hyperparameters["critic_gradient_clipping_norm"])     
        self.OptimizationStep(self._critic_optimizer2, self._critic2, critic_loss2, self._hyperparameters["critic_gradient_clipping_norm"])     

        actor_loss, log_pi = self.ActorLoss(states_t)
        self.OptimizationStep(self._actor_optimizer, self._actor, actor_loss, self._hyperparameters["actor_gradient_clipping_norm"])     

        alpha_loss = self.EntropyLoss(log_pi)
        self.OptimizationStep(self._alpha_optimizer, None, alpha_loss, None)     

        # update the target networks
        tau = self._hyperparameters['critic_tau']
        for target_param, local_param in zip(self._critic_target1.parameters(), self._critic1.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

        for target_param, local_param in zip(self._critic_target2.parameters(), self._critic2.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

        self._alpha = self._log_alpha.exp()


    def CriticLoss(self, states_t, actions_t, next_states_t, rewards_t, dones_t):
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.ActionInfo(next_states_t)
            next_q_values1 = self._critic_target1(torch.cat((next_states_t, next_state_actions), 1))
            next_q_values2 = self._critic_target2(torch.cat((next_states_t, next_state_actions), 1))

            min_q_next_value = torch.min(next_q_values1, next_q_values2) - self._alpha * next_state_log_pi
            next_q_value = rewards_t + (1.0 - dones_t) * self._gamma * min_q_next_value

        q_value1 = self._critic1(torch.cat((states_t, actions_t), 1))
        q_value2 = self._critic2(torch.cat((states_t, actions_t), 1))

        critic_loss1 = F.mse_loss(q_value1, next_q_value.detach())
        critic_loss2 = F.mse_loss(q_value2, next_q_value.detach())

        return critic_loss1, critic_loss2

    def ActorLoss(self, states_t):
        actions, log_pis, _ = self.ActionInfo(states_t)
        q_pi1 = self._critic1(torch.cat((states_t, actions), 1))
        q_pi2 = self._critic2(torch.cat((states_t, actions), 1))

        min_q_pi = torch.min(q_pi1, q_pi2)
        actor_loss = ((self._alpha * log_pis) - min_q_pi).mean()
        return actor_loss, log_pis

    def EntropyLoss(self, log_pi):
        alpha_loss = -(self._log_alpha * (log_pi + self._target_entropy).detach()).mean()
        return alpha_loss

