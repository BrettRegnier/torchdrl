import torch.nn.functional as F

class DQL:
    name="DQL"
    def __init__(self, device="cpu"):
        self._device = device

    def CalculateErrors(self, network, target_network, states_t, actions_t, next_states_t, rewards_t, dones_t, batch_size, gamma):
        q_values = network(states_t).gather(1, actions_t.unsqueeze(1))
        
        next_q_values = target_network(next_states_t).max(
            dim=1, keepdim=True)[0].detach()

        q_targets = (rewards_t + gamma * next_q_values *
                     (1-dones_t)).to(self._device)

        errors = F.smooth_l1_loss(q_values, q_targets, reduction="none")

        return errors

    def __call__(self, network, target_network, states_t, actions_t, next_states_t, rewards_t, dones_t, batch_size, gamma):
        return self.CalculateErrors(network, target_network, states_t, actions_t, next_states_t, rewards_t, dones_t, batch_size, gamma)