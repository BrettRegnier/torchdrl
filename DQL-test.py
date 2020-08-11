import gym
import torch
from data_structures.UniformExperienceReplay import UniformExperienceReplay
from agents.DQL import DQL

device = torch.device('cuda') # move into config.
env = gym.make("CartPole-v0") # move into config
# memory = UniformExperienceReplay(1000) # TODO generalize this. #move to config
 
config = {
    "env": env,
    "device": device,
    "reward_goal": 150,
    "reward_window": 100,
    "memory_size": 100000,
    "batch_size": 64,
    "warm_up": 0,
    "n_step_update": 1,
    "n_updates_per_learn": 1,
    "max_steps": -1,
    "visualize": False,
    "visualize_frequency": 10,
    "hyperparameters": {
        "hidden_layers": [1024, 512],
        "activations": ['relu', 'relu'],
        "final_activation": None,
        "lr": 0.001,
        "epsilon": 0.1,
        "epsilon_decay": 0.00005,
        "epsilon_min": 0.001,
        "target_update": 10,
        "tau": 0.1,
        "gamma": 0.99,
        "soft_update": True,
    }
}

agent = DQL(config)
agent.Train()
