import torch

# DQN
# Double DQN
# Dueling DQN
# Prioritized Experience Replay
# Noisy net
# Categorical DQN
# N step
class RainbowDQL:
    name="Rainbow"
    def __init__(self, device="cpu"):
        self._device = device
        self.atom_size = None
        self.v_min = None
        self.v_max = None
        self.support = None

    def CalculateErrors(self, network, target_network, states_t, actions_t, next_states_t, rewards_t, dones_t, batch_size, gamma):
        # get atoms
        if self.atom_size is None:
            try:
                self.atom_size = network.GetAtomSize()
            except:
                raise Exception("Provided network must have a function called GetAtomSize that returns the number of atoms")

        if self.v_min is None and self.v_max is None:
            try:
                self.v_min, self.v_max = network.GetSupportBounds()
            except:
                raise Exception("Provided network must have a function called GetSupportBounds that return tuple (value min, value max)")

        if self.support is None:
            try:
                self.support = network.GetSupport()
            except:
                raise Exception("Provided network must have a function called GetSupport that returns a torch tensor support (linspace)")


        # categorical
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # double dqn
            next_actions = network(next_states_t).argmax(1)
            next_dists = target_network.DistributionForward(next_states_t)
            next_dists = next_dists[range(batch_size), next_actions]

            # categorical
            t_z = rewards_t + (1 - dones_t) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(0, (batch_size - 1) *
                               self.atom_size, batch_size)
                .long()
                .unsqueeze(1)
                .expand(batch_size, self.atom_size)
                .to(self._device)
            )

            proj_dist = torch.zeros(next_dists.size(), device=self._device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dists * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dists * (b - l.float())).view(-1)
            )

        dist = network.DistributionForward(states_t)
        log_p = torch.log(dist[range(batch_size), actions_t])

        errors = -(proj_dist * log_p).sum(1)
        return errors
