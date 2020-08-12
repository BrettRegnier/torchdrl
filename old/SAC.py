import torch
import torch.nn as nn
import torch.distributions as dis
import torch.nn.functional as F
import torch.optim as optim
# https://towardsdatascience.com/in-depth-review-of-soft-actor-critic-91448aba63d4
# https://arxiv.org/abs/1801.01290
# https://github.com/ku2482/sac-discrete.pytorch

# critic
class Critic(nn.Module):
    def __init__(self, input_shape, n_actions, convo=False, init_weight=3e-3):
        super(Critic, self).__init__()		
        
        self._conv = None
                
        # TODO figure out how this will work with convo
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
            
            input_shape = [total_inputs + n_actions]

        self._fc1 = nn.Linear(*input_shape, 1024)
        self._fc2 = nn.Linear(1024, 1024)
        self._fc3 = nn.Linear(1024, 1)

        # initialize the weights
        self._fc3.weight.data.uniform_(-init_weight, init_weight)
        self._fc3.bias.data.uniform_(-init_weight, init_weight)

    def Forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self._fc1(x))
        x = F.relu(self._fc2(x))
        x = self._fc3(x)
        return x

# actor
class Actor(nn.Module):
    def __init__(self, input_shape, n_actions, convo=False, init_weight=3e-3, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        self._log_std_min = log_std_min
        self._log_std_max = log_std_max        
        
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

        self._fc1 = nn.Linear(*input_shape, 2048)
        self._fc2 = nn.Linear(2048, 2048)

        # head that ouputs the mean of the guassian distribution
        self._mean_linear = nn.Linear(2048, n_actions)
        self._mean_linear.weight.data.uniform_(-init_weight, init_weight)
        self._mean_linear.bias.data.uniform_(-init_weight, init_weight)

        # head that outputs the log(covariance) of the gaussian distribution
        self._log_std_linear = nn.Linear(2048, n_actions)
        self._log_std_linear.weight.data.uniform_(-init_weight, init_weight)
        self._log_std_linear.bias.data.uniform_(-init_weight, init_weight)

    def Forward(self, state):
        x = F.relu(self._fc1(state))
        x = F.relu(self._fc2(x))

        mean = self._mean_linear(x)
        log_std = self._log_std_linear(x)
        log_std = torch.clamp(log_std, self._log_std_min, self._log_std_max)

        return mean, log_std

    def Sample(self, state, epsilon=1e-6):
        mean, log_std = self.Forward(state)
        std = log_std.exp()

        normal = dis.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)

        log_pi = (normal.log_prob(z) - torch.log(1 - (torch.tanh(z)).pow(2) + epsilon)).sum(1, keepdim=True)

        return action, log_pi

class AgentSAC:
    def __init__(self, input_shape, n_actions, gamma, tau, alpha, critic_lr, actor_lr, a_lr, device):
        self._gamma = gamma
        self._tau = tau
        self._alpha = alpha
        self._device = device

        self._update_step = 0
        self._delay_step = 2

        self._first_critic = Critic(input_shape, n_actions).to(self._device)
        self._second_critic = Critic(input_shape, n_actions).to(self._device)
        self._first_target_critic = Critic(input_shape, n_actions).to(self._device)
        self._second_target_critic = Critic(input_shape, n_actions).to(self._device)
        self._actor = Actor(input_shape, n_actions).to(self._device)

        self._first_target_critic.state_dict(self._first_critic.state_dict())      
        self._second_target_critic.state_dict(self._second_critic.state_dict())

        # optimizers
        self._first_critic_optimizer = optim.Adam(self._first_critic.parameters(), lr=critic_lr)
        self._second_critic_optimizer = optim.Adam(self._second_critic.parameters(), lr=critic_lr)
        self._actor_optimizer = optim.Adam(self._actor.parameters(), actor_lr)
        
        # entropy temperature
        self._target_entropy = -torch.prod(torch.tensor(input_shape).to(self._device)).item()
        self._log_alpha = torch.zeros(1, requires_grad=True, device=self._device)
        self._alpha_optimizer = optim.Adam([self._log_alpha], lr=a_lr)

        # implement outside
        # self._replay = UniformExperienceReplay(buffer_maxlen)

        # TODO
        # things I omitted 
        # 1. Env
        # 2. Memory

    def Act(self, state):
        # might need to add [] around state here
        state_t = torch.tensor(state, dtype=torch.float32).to(self._device)
        mean, log_std = self._actor.Forward(state_t)
        std = log_std.exp()

        normal = dis.Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        # TODO change this to be a discrete number
        # the example is using for both discrete and continuous
        # action = torch.argmax(action)
        action = action.cpu().detach().squeeze(0).numpy()

        #returns the list of actions need for later use as numpy array
        return action
    
    def Learn(self, batch):
        states, actions, rewards, next_states, dones = batch

        states_t = torch.tensor(states, dtype=torch.float32).to(self._device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32).to(self._device)
        actions_t = torch.tensor(actions, dtype=torch.float32).to(self._device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(self._device)
        dones_t = torch.tensor(dones, dtype=torch.float32).to(self._device)
        dones_t = dones_t.view(dones_t.size(0), -1)

        next_actions, next_log_pis = self._actor.Sample(next_states_t)
        target_value1 = self._first_target_critic.Forward(next_states_t, next_actions)
        target_value2 = self._second_target_critic.Forward(next_states_t, next_actions)
        next_q_target = torch.min(target_value1, target_value2) - self._alpha * next_log_pis
        expected_q = rewards_t + (1 - dones_t) * self._gamma * next_q_target

        # Loss
        q_value1 = self._first_critic.Forward(states_t, actions_t)
        q_value2 = self._second_critic.Forward(states_t, actions_t)
        q_loss1 = F.mse_loss(q_value1, expected_q.detach())
        q_loss2 = F.mse_loss(q_value2, expected_q.detach())

        self._first_critic_optimizer.zero_grad()
        q_loss1.backward()
        self._first_critic_optimizer.step()

        self._second_critic_optimizer.zero_grad()
        q_loss2.backward()
        self._second_critic_optimizer.step()

        # delated update for actor and targets
        new_actions, log_pis = self._actor.Forward(states_t)
        if self._update_step % self._delay_step == 0:
            # prevent overflowing
            self._update_step = 0
            min_q_value = torch.min(
                self._first_critic.Forward(states_t, new_actions),
                self._second_critic.Forward(states_t, new_actions)
            )
            
            actor_loss = (self._alpha * log_pis - min_q_value).mean()
            
            self._actor_optimizer.zero_grad()
            actor_loss.backward()
            self._actor_optimizer.step()

            # update targets
            self._first_target_critic.state_dict(self._first_critic.state_dict())      
            self._second_target_critic.state_dict(self._second_critic.state_dict())

        # update temperature
        alpha_loss = (self._log_alpha * (-log_pis - self._target_entropy).detach()).mean()

        self._alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self._alpha_optimizer.step()
        self._alpha = self._log_alpha.exp()

        self._update_step += 1


        

