# TODO Finish
from torchdrl.agents.markov.DQL import DQL
from torchdrl.agents.markov.DoubleDQL import DoubleDQL
from torchdrl.agents.markov.RainbowDQL import RainbowDQL

from torchdrl.agents.monte_carlo.Reinforce import Reinforce
from torchdrl.agents.monte_carlo.GAE import GAE
from torchdrl.agents.monte_carlo.PPO import PPO

def CreateAgent(agent_type, args, kwargs):
    agent = None
    if agent_type.lower() == 'dql':
        agent = DQL(*args, **kwargs)
    elif agent_type.lower() == 'doubledql':
        agent = DoubleDQL(*args, **kwargs)
    elif agent_type.lower() == 'rainbowdql':
        agent = DoubleDQL(*args, **kwargs)
    elif agent_type.lower() == 'reinforce':
        agent = Reinforce(*args, **kwargs)
    elif agent_type.lower() == 'gae':
        agent = GAE(*args, **kwargs)
    elif agent_type.lower() == 'ppo':
        agent = PPO(*args, **kwargs)

    return agent