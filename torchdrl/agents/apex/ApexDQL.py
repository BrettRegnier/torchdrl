from torchdrl.agents.q_learning.QLearningAgent import QLearningAgent
from torchdrl.agents.q_learning.DQL import DQL

class ApexQLearningAgent(QLearningAgent):
    def __init__(self, *args, **kwargs):
        super(ApexQLearningAgent, self).__init__(*args, **kwargs)

class ApexDQL(QLearningAgent, DQL):
    def __init__(self, *args, **kwargs):
        super(ApexDQL, self).__Init__(*args, **kwargs)