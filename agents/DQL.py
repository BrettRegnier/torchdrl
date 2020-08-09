import torch
from torch.nn import MSELoss
from torch.optim import Adam
import numpy as np
import random
from agents.BaseAgent import BaseAgent

from models.FullyConnectedNetwork import FullyConnectedNetwork as FCN 


class DQL(BaseAgent):
	def __init__(self, config):
		self._epsilon = config['epsilon']
		self._target_update = config['target_update']
		self._soft = config['soft']
		self._tau = config['tau']
		self._gamma = config['gamma']
		
		self._net = FCN(self._input_shape, self._n_actions).to(self._device)
		self._target_net = FCN(self._input_shape, self._n_actions).to(self._device)
		self._net_optimizer = Adam(self._net.parameters(), lr=config['lr'], eps=1e-6)
	
	def PlayEpisode(self, evaluate=False):
		done = False
		steps = 0
		episode_reward = 0
		
		state = self._env.reset()
		while steps != self._max_steps and not done:
			if random.random() < self._epsilon:
				action = self._env.action_space.sample()
			else:
				action = self.Act(state)
				
			next_state, reward, done, info = self._env.step(action)
			
			self._memory.Append(state, action, next_state, reward, done)
			
			self.Learn()
			
			episode_reward += reward
			state = next_state
			
		if self._episode % self._target_update == 0:
			if self._soft:
				self.SoftTargetUpdate()
			else:
				self.TargetUpdate()
			
	
	@torch.no_grad()
	def Act(self, state):
		state_t = torch.tensor(np.array(state, copy=False)).to(self._device)
		
		q_value = self._net(state_t)
		action = torch.argmax(q_value, dim=1)
		return action.item()
		
	def Learn(self):
		# TODO add situation without replay
		self._net_optimizer.zero_grad()
		
		states_np, actions_np, next_states_np, rewards_np, dones_np = self._memory.Sample(self._batch_size)

		states_t = torch.tensor(np.array(states, copy=False), dtype=torch.float32).to(self._device)
		next_states_t = torch.tensor(np.array(next_states, copy=False), dtype=torch.float32).to(self._device)
		actions_t = torch.tensor(np.array(actions), dtype=torch.int64).to(self._device)
		rewards_t = torch.tensor(np.array(rewards), dtype=torch.float32).to(self._device)
		dones_t = torch.tensor(np.array(dones), dtype=torch.bool).to(self._device)
	 
		state_values = self._net(states_t).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)
		
		with torch.no_grad():
			next_state_values = self._target_net(next_states_t).max(1)[0]
			next_state_values[dones_t] = 0.0
			next_state_values = next_state_values.detach()
			
		expected_values = next_state_values * self._gamma + rewards_t
		
		loss = MSELoss(state_values, expected_values)
		
		loss.backward()
		self._net_optimizer.step()
		
	def SoftTargetUpdate(self):
		raise NotImplementedError("Soft target update is not implemented")
	
	def TargetUpdate(self):
		raise NotImplementedError("Target update is not implemented")