import torch 
import torch.nn.functional as F
import torch.distributions as distributions

import torchdrl.tools.Helper as Helper
from torchdrl.agents.monte_carlo.MonteCarloAgent import MonteCarloAgent

class GAE(MonteCarloAgent):
    def __init__(self, trace_decay, *args, **kwargs):
        super(GAE, self).__init__(*args, **kwargs)

        self._trace_decay = trace_decay

    def Train(self):
        self._policy.train()

        return self.PlayEpisode()

    def TrainNoYield(self):
        pass

    def Learn(self, advantages_t, log_prob_actions_t, rewards_t, values_t):
        advantages_t = advantages_t.detach()
        rewards_t = rewards_t.detach()

        policy_loss = -(advantages_t * log_prob_actions_t).sum()
        value_loss = F.smooth_l1_loss(rewards_t, values_t).sum()

        self._optimizer.zero_grad()

        policy_loss.backward()
        value_loss.backward()

        self._optimizer.step()

        return policy_loss.item(), value_loss.item()

    def Evaluate(self):
        self._policy.eval()

        steps = 0
        done = False
        info = {}
        episode_reward = 0

        state = self._test_env.reset()

        while not done and steps != self._max_steps:
            with torch.no_grad():
                action_pred, _ = self.GetAction(state, evaluate=True)
                action_prob = F.softmax(action_pred, dim=-1)

            # self._test_env.Print()
            # print(action_pred)
            # print(action_prob)

            action = torch.argmax(action_prob, dim=-1)
            # action = torch.argmax(action_pred)
            state, reward, done, info = self._test_env.step(action.item())

            # print(action.item())

            steps += 1
            episode_reward += reward

        # self._test_env.Print()
        # exit()

        return episode_reward, info

    def GetAction(self, state, evaluate=False):
        return self.Act(state)

    def Act(self, state):
        state_t = Helper.ConvertStateToTensor(state, self._device)
        return self._policy(state_t)

    def Save(self, folderpath, filename):
        pass

    def Load(self, filepath):
        pass

    def PlayEpisode(self):
        log_prob_actions = []
        values = []
        rewards = []

        steps = 0
        done = False
        episode_reward = 0

        state = self._env.reset()

        while not done and steps != self._max_steps:
            action_pred, value_pred = self.GetAction(state, evaluate=False)

            action_prob = F.softmax(action_pred, dim=-1)
            dist = distributions.Categorical(action_prob)
            action = dist.sample()
            log_prob_action = dist.log_prob(action)

            next_state, reward, done, info = self._env.step(action.item())

            log_prob_actions.append(log_prob_action)
            values.append(value_pred)
            rewards.append(reward)

            state = next_state

            steps += 1
            episode_reward += reward

        log_prob_actions_t = torch.cat(log_prob_actions)
        values_t = torch.cat(values).squeeze(-1)

        discounted_rewards_t = self.Rollout(rewards, normalize=True)
        advantages_t = self.Advantages(rewards, values_t, normalize=True)
        policy_loss, value_loss = self.Learn(advantages_t, log_prob_actions_t, discounted_rewards_t, values_t)

        return policy_loss, value_loss, episode_reward
    
    def Advantages(self, rewards, values_t, normalize=True):
        advantages = []
        advantage = 0
        next_value = 0

        for reward, value in zip(reversed(rewards), reversed(values_t)):
            td_error = reward + next_value * self._gamma - value
            advantage = td_error + advantage * self._gamma * self._trace_decay
            next_value = value
            advantages.insert(0, advantage)

        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self._device)

        if normalize:
            advantages_t = (advantages_t - advantages_t.mean()) / advantages_t.std()

        return advantages_t