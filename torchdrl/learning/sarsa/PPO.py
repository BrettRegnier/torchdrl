import torch
import torch.nn.functional as F
import torch.distributions as distributions

import torchdrl.tools.Helper as Helper
from torchdrl.agents.SARSAAgent import SARSAAgent

class PPO(SARSAAgent):
    def __init__(self, ppo_steps, ppo_clip, *args, **kwargs):
        super(PPO, self).__init__(*args, **kwargs)

        self._ppo_steps = ppo_steps
        self._ppo_clip = ppo_clip

    def Train(self):
        self._policy.train()

        return self.PlayEpisode()

    def TrainNoYield(self):
        pass

    def Learn(self, states_t, actions_t, log_prob_actions_t, advantages_t, rewards_t):
        total_policy_loss = 0
        total_value_loss = 0

        advantages_t = advantages_t.detach()
        log_prob_actions_t = log_prob_actions_t.detach()
        actions_t = actions_t.detach()

        for _ in range(self._ppo_steps):
            # get new log prob of actions for all input states
            action_pred, value_pred = self._policy(states_t)

            value_pred = value_pred.squeeze(-1)
            action_prob = F.softmax(action_pred, dim=-1)
            dist = distributions.Categorical(action_prob)

            # new log prob using old actions
            new_log_prob_actions = dist.log_prob(actions_t)

            policy_ratio = (new_log_prob_actions - log_prob_actions_t).exp()

            policy_loss_1 = policy_ratio * advantages_t
            policy_loss_2 = torch.clamp(policy_ratio, min=1.0 - self._ppo_clip, max=1.0+self._ppo_clip) * advantages_t

            policy_loss = -torch.min(policy_loss_1, policy_loss_2).sum()
            value_loss = F.smooth_l1_loss(rewards_t, value_pred).sum()

            self._optimizer.zero_grad()

            policy_loss.backward()
            value_loss.backward()

            self._optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        avg_policy_loss = total_policy_loss / self._ppo_steps
        avg_value_loss = total_value_loss / self._ppo_steps

        return avg_policy_loss, avg_value_loss

    def Evaluate(self):
        self._policy.eval()

        steps = 0
        done = False
        info = {}
        episode_reward = 0

        state = self._test_env.reset()

        while not done and steps != self._max_steps:
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
        states = []
        actions = []
        log_prob_actions = []
        values = []
        rewards_t = []

        steps = 0
        done = False
        episode_reward = 0
        
        state = self._env.reset()

        while not done and steps != self._max_steps:
            # print(state)
            action_pred, value_pred = self.GetAction(state)
            # print(action_pred)
            # print(value_pred)
            # input()
            action_prob = F.softmax(action_pred, dim=-1)
            dist = distributions.Categorical(action_prob)
            action = dist.sample()
            log_prob_action = dist.log_prob(action)

            next_state, reward, done, _ = self._env.step(action.item())

            states.append(state)
            actions.append(action)
            log_prob_actions.append(log_prob_action)
            values.append(value_pred)
            rewards_t.append(reward)

            state = next_state

            steps += 1
            episode_reward += reward

        states_t = torch.tensor(states, dtype=torch.float32, device=self._device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self._device)
        log_prob_actions_t = torch.tensor(log_prob_actions, dtype=torch.float32, device=self._device)
        values_t = torch.tensor(values, dtype=torch.float32, device=self._device).squeeze(-1)

        discounted_rewards_t = self.Rollout(rewards_t, False)
        advantages_t = self.Advantages(discounted_rewards_t, values_t, False)

        policy_loss, value_loss = self.Learn(states_t, actions_t, log_prob_actions_t, advantages_t, discounted_rewards_t)

        return policy_loss, value_loss, episode_reward

    def Advantages(self, discounted_rewards_t, values_t, normalize=True):
        advantages_t = discounted_rewards_t - values_t

        if normalize:
            advantages_t = (advantages_t - advantages_t.mean()) / advantages_t.std()

        return advantages_t