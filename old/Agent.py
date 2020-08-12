import torch
import torch.nn.functional as F
import numpy as np

from collections import namedtuple
from Network import A2C 

class AgentA2C:
	action_value = namedtuple('action_value', ['log_probs', 'value'])
	def __init__(self, lr, input_shape, n_actions, gamma, device):
		self._gamma = gamma 
		self._device = device
		self._network = A2C(lr, input_shape, n_actions, convo=False).to(self._device)
		
		self._actions = []
		self._rewards = []
		
		self._epsilon = np.finfo(np.float32).eps.item()
		
	def Act(self, state):
		state_np = np.array(state, copy=False)
		state_t = torch.tensor(state_np, dtype=torch.float32).to(self._device)
		
		probabilities = self._network.Policy(state_t)
		state_value = self._network.Value()
		
		action_probs = torch.distributions.Categorical(probabilities)
		action = action_probs.sample()
		
		self._log_probs = action_probs.log_prob(action)
		
		self._actions.append(self.action_value(self._log_probs, state_value))
		
		return action.item()
	
	def Learn(self):
		true_reward = 0 #R
		saved_actions = self._actions
		
		actor_losses = []
		critic_losses = []
		returns = [] # true values
		
		# calculate the true values 
		# go by reverse order
		for reward in self._rewards[::-1]:
			# calculate discounted value
			true_reward = reward + self._gamma * true_reward
			returns.insert(0, true_reward)
				
		returns_t = torch.tensor(returns)
		returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + self._epsilon).to(self._device)
		
		for (log_prob, state_value), delta in zip(saved_actions, returns):
			temporal_difference = delta - state_value.item()
			actor_losses.append(-log_prob * temporal_difference)
			critic_losses.append(F.smooth_l1_loss(state_value, torch.tensor([delta]).to(self._device)))
			
		self._network._optimizer.zero_grad()
		actor_stack = torch.stack(actor_losses)
		actor_stack_sum = actor_stack.sum()
		loss = torch.stack(actor_losses).sum() + torch.stack(critic_losses).sum()
		
		loss.backward()
		self._network._optimizer.step()
		
		del self._actions[:]
		del self._rewards[:]
		
	def Learn_o(self, state, next_state, reward, done):
		state_t = torch.tensor(np.array(state), dtype=torch.float32).to(self._device)
		next_state_t = torch.tensor(np.array(next_state), dtype=torch.float32).to(self._device)
		reward_t = torch.tensor(np.array(reward), dtype=torch.float32).to(self._device)
		# done_t = torch.tensor(np.array(done), dtype=torch.bool).to(self._device)
		
		state_value = self._network.Value() # current state on body has been calculated from self.Act(state)
		next_state_value = self._network.Value(next_state_t)
		
		delta = reward_t + (self._gamma * next_state_value * (1 - int(done)))
		temporal_difference = delta - state_value
		
		actor_loss = -self._log_probs * temporal_difference
		critic_loss = temporal_difference ** 2
		
		loss = (actor_loss + critic_loss)
		
		self._network._optimizer.zero_grad()
		loss.backward()
		self._network._optimizer.step()
		