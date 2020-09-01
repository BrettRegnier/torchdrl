import gym
from data_structures.UniformExperienceReplay import UniformExperienceReplay
from agents.SAC import SAC
from agents.SACD import SACD
import torch

import Config

torch.autograd.set_detect_anomaly(True)

# device = torch.device('cuda')
# env = gym.make("CartPole-v0")
# memory = UniformExperienceReplay(1000000) # TODO generalize this.

config = {
    "env": "CartPole-v0",
    "device": "cuda",
    "reward_goal": 150,
    "reward_window": 100,
    "gamma": 0.99,
    "memory_size": 100000,
    "batch_size": 256,
    "warm_up": 400,
    "max_steps": -1,
    "n_step_update": 1,
    "n_updates_per_learn": 1,
    "visualize": False,
    "visualize_frequency": 100,
    "hyperparameters": {
        # actor
        "actor_lr": 0.0003,
        "actor_hidden_layers": [1024, 1024],
        "actor_activations": ['relu', 'relu'],
        "actor_final_activation": 'softmax',
        "actor_initializer": "Xavier",
        "actor_gradient_clipping_norm": 5,
        # critic
        "critic_lr": 0.0003,
        "critic_hidden_layers": [1024, 1024],
        "critic_activations": ['relu', 'relu'],
        "critic_final_activation": None,
        "critic_tau": 0.005,
        "critic_initializer": "Xavier",
        "critic_gradient_clipping_norm": 5,
        # alpha entropy
        "alpha_lr": 0.0003,
    }
}

# config = Config.Load("configs/SACD_CartPole-v0")
config["env"] = gym.make(config["env"])

# general hyperparams
# gamma = 0.99
# batch_size = 256
# warm_up = 400
# n_step_update = 1
# n_updates_per_learn = 1


# # hyperparams for actor
# actor_lr = 0.0003
# actor_initializer = "Xavier"
# actor_gradient_clipping = 5

# # hyperparams for critic
# critic_lr = 0.0003
# critic_tau = 0.005
# critic_initializer = "Xavier"
# critic_gradient_clipping = 5

test = SACD(config)
test.Train()

# TODO 
# hyperparams, and do gradient clipping