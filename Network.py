import torch
import torch.nn as nn
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
		
		self._body = nn.Sequential(
			nn.Linear(*input_shape, 1024),
			nn.ReLU(),
			nn.Linear(1024, 1024),
			nn.ReLU(),
		)
		
		self._actor = nn.Linear(1024, n_actions)
		self._critic = nn.Linear(1024, 1)
		
		self._optimizer = optim.Adam(self.parameters(), lr=lr)

	def forward(self, obs):
		out = obs	
		if self._conv is not None:
			out = self._conv(out).view(out.size()[0], -1)
		
		self._body_out = self._body(out)
		return self._body_out	
	
	def Policy(self, obs=None):
		if obs is None:
			return self._actor(self._body_out)
		self.forward(obs)
		return self._actor(self._body_out)
		
	def Value(self, obs=None):
		if obs is None:
			return self._critic(self._body_out)
		self.forward(obs)
		return self._critic(self._body_out)