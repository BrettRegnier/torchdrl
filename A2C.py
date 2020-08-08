import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class A2C(nn.Module):
	def __init__(self, lr, input_shape, n_actions, convo=False):
		super(A2C, self).__init__()
		self._conv = None
		self._body_out = None
		
		if convo:
			self._conv = nn.Sequential(
				nn.Conv2d(input_shape[0], 80, kernel_size=3),
				nn.ReLU()
			)
			out = self._conv(torch.zeros(1, *input_shape))
			input_shape = [int(np.prod(out.size()))]
		else:
			total_inputs = 1
			for x in input_shape:
				total_inputs *= x
			
			input_shape = [total_inputs]
		
		self._body = nn.Sequential(
			nn.Linear(*input_shape, 2048),
			nn.ReLU(),
			nn.Linear(2048, 2048),
			nn.ReLU(),
		)
		
		self._actor = nn.Linear(2048, n_actions)
		self._critic = nn.Linear(2048, 1)
		
		self._optimizer = optim.Adam(self.parameters(), lr=lr)

	def forward(self, obs):
		out = obs	
		if self._conv is not None:
			out = self._conv(out).view(out.size()[0], -1)
		
		self._body_out = self._body(out)
		return self._body_out	
	
	def Policy(self, obs=None):
		if obs is not None:
			self.forward(obs)
			
		return F.softmax(self._actor(self._body_out), -1)
		
	def Value(self, obs=None):
		if obs is not None:
			self.forward(obs)

		return self._critic(self._body_out)