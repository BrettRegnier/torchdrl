import argparse
import threading

import gym
import torch

from ui.Commandline import Commandline

from data_structures import Config
from envs import RegisteredEnvs

from agents.DQL import DQL
from agents.SAC import SAC
from agents.SACD import SACD

# TODO add visualization
# config = Config.Load("configs/Minesweeper-v0_SACD.txt")

# TODO make this modular? or move into the agent...

def Main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--ui", default="commandline", action="store_true")
    args = parser.parse_args()

    ui = None
    if args.ui == "commandline":
        ui = Commandline()

        config_path = ui.GetConfig()
        config = Config.Load(config_path)
        config['env'] = RegisteredEnvs.BuildEnv(config['env_name'], config['env_kwargs'])
        agent = MakeAgent(config['agent'], config)

        train_thread = threading.Thread(target=agent.Train)
        train_thread.daemon = True
        train_thread.start()
        
        ui.Begin()

    elif args.ui == "gui":
        raise NotImplementedError("GUI not implemented")

def MakeAgent(agent_name, config):
    agent = None
    if agent_name == "DQL":
        agent = DQL(config)
    elif agent_name == "SAC":
        agent = SAC(config)
    elif agent_name == "SACD":
        agent = SACD(config)
    else:
        raise AssertionError("No agent with that name")

    return agent

if __name__ == "__main__":
    Main()


# TODO change loss calculation to hinge loss