
from .BaseAgent import BaseAgent
from .DQL import DQL

class DuelingDQN(DQL):
    def __init__(self, config):
        super(DuelingDQL, self).__init__(config)