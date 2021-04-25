import torch
import torch.nn.functional as F

from torchdrl.agents.markov.DQL import DQL

class DoubleDQL(DQL):
    def CalculateErrors(self, states_t, actions_t, next_states_t, rewards_t, dones_t, indices_np, weights_t, batch_size, gamma):
        q_values = self._model(states_t).gather(1, actions_t.unsqueeze(1))

        with torch.no_grad():
            next_q_values = self._target_model(next_states_t)
            next_actions = self._model(next_states_t).argmax(dim=1, keepdim=True)
            next_state_action_pair_values = next_q_values.gather(
                1, next_actions).detach()

        q_target = (rewards_t + gamma * next_state_action_pair_values *
                    (1-dones_t)).to(self._device)

        errors = F.smooth_l1_loss(q_values, q_target, reduction="none")
        return errors
