import torch
from torch.nn import MSELoss
from torch.optim import Adam
import numpy as np
import random
from agents.BaseAgent import BaseAgent

from models.FullyConnectedNetwork import FullyConnectedNetwork as FCN 
from models.ConvolutionNetwork import ConvolutionNetwork as CN
from data_structures.UniformExperienceReplay import NewUEP as NUER
from data_structures.UniformExperienceReplay import UniformExperienceReplay as UER


class DQL(BaseAgent):
    def __init__(self, config):
        super(DQL, self).__init__(config)
        self._epsilon = self._hyperparameters['epsilon']
        self._epsilon_decay = self._hyperparameters['epsilon_decay']
        self._epsilon_min = self._hyperparameters['epsilon_min']
        self._target_update = self._hyperparameters['target_update']
        self._gamma = self._hyperparameters['gamma']

        # for soft update
        self._soft_update = self._hyperparameters['soft_update']
        self._tau = self._hyperparameters['tau'] 
        
        convo = None
        if self._hyperparameters['convo'] is not None:
            cnc = self._hyperparameters['convo']
            convo = CN(self._input_shape, cnc['filters'], cnc['kernels'], cnc['strides'], cnc['paddings'], cnc['activations'], cnc['pools'], cnc['flatten'])

        fcc = self._hyperparameters['fc']
        self._net = FCN(self._input_shape, self._n_actions, fcc["hidden_layers"], fcc['activations'], fcc['final_activation'], convo).to(self._device)
        self._target_net = FCN(self._input_shape, self._n_actions, fcc["hidden_layers"], fcc['activations'], fcc['final_activation'], convo).to(self._device)
        self._net_optimizer = Adam(self._net.parameters(), lr=self._hyperparameters['lr'])
    
    def PlayEpisode(self, evaluate=False):
        done = False
        steps = 0
        episode_reward = 0
        
        if self._enable_seed:
            self._env.seed(self._seed)
        state = self._env.reset()
        while steps != self._max_steps and not done:
            action = self.Act(state)
                
            next_state, reward, done, info = self._env.step(action)
            
            self._memory.Append(state, action, next_state, reward, done)

            if len(self._memory) > self._batch_size and self._total_steps > self._warm_up:
                self.Learn()

                # update epsilon
                self._epsilon = max(self._epsilon * self._epsilon_decay, self._epsilon_min)
            
            episode_reward += reward
            state = next_state

            steps += 1
            self._total_steps += 1
            
        return episode_reward, steps, info

    @torch.no_grad()
    def Act(self, state):            
        if random.random() < self._epsilon:
            action = self._env.action_space.sample()
        else:
            state_t = torch.tensor(state).float().detach()
            state_t = state_t.to(self._device)
            state_t = state_t.unsqueeze(0)
            
            q_values = self._net(state_t)
            action = torch.argmax(q_values).item()

        return action
    
    # https://wegfawefgawefg.github.io/tutorials/rl/doubledeepql/doubledeepql.html
    def Learn(self):        
        states_np, actions_np, next_states_np, rewards_np, dones_np = self._memory.Sample(self._batch_size)

        states_t = torch.tensor(states_np, dtype=torch.float32).to(self._device)
        actions_t = torch.tensor(actions_np, dtype=torch.int64).to(self._device)
        next_states_t = torch.tensor(next_states_np, dtype=torch.float32).to(self._device)
        rewards_t = torch.tensor(rewards_np, dtype=torch.float32).to(self._device)
        dones_t = torch.tensor(dones_np, dtype=torch.bool).to(self._device)

        batch_indices = np.arange(self._batch_size, dtype=np.int64)
        q_values = self._net(states_t)[batch_indices, actions_t]

        next_q_values = self._target_net(next_states_t)
        next_actions = self._net(next_states_t).max(dim=1)[1]
        next_action_qs = next_q_values[batch_indices, next_actions]
        next_action_qs[dones_t] = 0.0

        q_target = rewards_t + self._gamma * next_action_qs

        td = q_target - q_values

        self._net_optimizer.zero_grad()

        loss = ((td ** 2.0)).mean()
        loss.backward()
        self._net_optimizer.step()

        # update target
        if self._total_steps % self._target_update == 0:
            self.CopyNetwork(self._net, self._target_net)

    # currently broken
    # https://towardsdatascience.com/deep-q-learning-for-the-cartpole-44d761085c2f
    def SoftTargetUpdate(self, network, target_network, tau=0.1):
        model_params = network.named_parameters()
        target_params = target_network.named_parameters()

        updated_params = dict(target_params)

        for model_name, model_param in model_params:
            if model_name in target_params:
                updated_params[model_name].data.copy_((tau) * model_param.data + (1-tau) * target_params[model_param].data)
        
        target_network.load_state_dict(updated_params)
