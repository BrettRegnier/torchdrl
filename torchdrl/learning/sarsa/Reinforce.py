import torch
import torch.nn.functional as F 
import torch.distributions as distributions

import torchdrl.tools.Helper as Helper
from torchdrl.agents.SARSAAgent import SARSAAgent

class Reinforce(SARSAAgent):
    def __init__(self, *args, **kwargs):
        super(Reinforce, self).__init__(*args, **kwargs)

    def Train(self):
        self._policy.train()

        return self.PlayEpisode()

    def TrainNoYield(self):
        pass

    def Learn(self, rewards_t, log_prob_actions_t):
        rewards_t = rewards_t.detach()
        loss = -(rewards_t * log_prob_actions_t).sum()

        self._optimizer.zero_grad()

        loss.backward()

        self._optimizer.step()

        return loss.item()

    def Evaluate(self):
        self._policy.eval()

        steps = 0
        done = False
        info = {}
        episode_reward = 0

        state = self._test_env.reset()

        while not done and steps != self._max_steps:
            with torch.no_grad():
                action_pred = self.GetAction(state, evaluate=True)
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
        rewards = []

        steps = 0
        done = False
        episode_reward = 0

        state = self._env.reset()

        while not done and steps != self._max_steps:
            action_pred = self.GetAction(state, evaluate=False)

            action_prob = F.softmax(action_pred, dim=-1)
            dist = distributions.Categorical(action_prob)
            action = dist.sample()
            log_prob_action = dist.log_prob(action)

            next_state, reward, done, info = self._env.step(action.item())

            log_prob_actions.append(log_prob_action)
            rewards.append(reward)

            state = next_state

            steps += 1
            episode_reward += reward

        log_prob_actions_t = torch.cat(log_prob_actions)
        discounted_rewards_t = self.Rollout(rewards, normalize=False)

        loss = self.Learn(discounted_rewards_t, log_prob_actions_t)

        return loss, episode_reward