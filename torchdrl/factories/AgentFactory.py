import torch.nn as nn
import torch.utils.data as data_utils

from torchdrl.agents.q_learning.DQL import DQL
from torchdrl.agents.q_learning.DoubleDQL import DoubleDQL
from torchdrl.agents.q_learning.RainbowDQL import RainbowDQL

from torchdrl.agents.sarsa.Reinforce import Reinforce
from torchdrl.agents.sarsa.GAE import GAE
from torchdrl.agents.sarsa.PPO import PPO

from torchdrl.agents.supervised.Supervised import Supervised

import torchdrl.factories.NeuralNetworkFactory as NeuralNetworkFactory
import torchdrl.factories.OptimizerFactory as OptimizerFactory
import torchdrl.factories.SchedulerFactory as SchedulerFactory
import torchdrl.factories.MemoryFactory as MemoryFactory
import torchdrl.factories.AgentFactory as AgentFactory

import DragonFruit.datagen.BoatDatasetRetriever as BoatDatasetRetriever
import DragonFruit.datagen.BoatDataGenerator as BoatDataGenerator

def CreateAgent(agent_type, args, kwargs):
    agent = None
    if agent_type.lower() == 'dql':
        agent = DQL(*args, **kwargs)
    elif agent_type.lower() == 'doubledql':
        agent = DoubleDQL(*args, **kwargs)
    elif agent_type.lower() == 'rainbowdql':
        agent = RainbowDQL(*args, **kwargs)
    elif agent_type.lower() == 'reinforce':
        agent = Reinforce(*args, **kwargs)
    elif agent_type.lower() == 'gae':
        agent = GAE(*args, **kwargs)
    elif agent_type.lower() == 'ppo':
        agent = PPO(*args, **kwargs)

    return agent 

def CreateQLearningAgent(config, env):
    q_learning_config = config
    device = q_learning_config['device']

    # reinforcement learning
    model = NeuralNetworkFactory.CreateNetwork(
        q_learning_config['model'], env.observation_space, env.action_space, device)
    q_learning_optimizer = OptimizerFactory.CreateOptimizer(
        q_learning_config['optimizer']['name'], (model.parameters(),), q_learning_config['optimizer']['kwargs'])
    q_learning_scheduler = SchedulerFactory.CreateScheduler(
        q_learning_config['scheduler']['name'], q_learning_optimizer, q_learning_config['scheduler']['kwargs'])
    memory = MemoryFactory.CreateMemory(
        q_learning_config['memories']['memory']['name'], env.observation_space, q_learning_config['memories']['memory']['kwargs'])

    memory_n_step = None
    if 'memory_n_step' in q_learning_config['memories']:
        memory_n_step = MemoryFactory.CreateMemory(
            q_learning_config['memories']['memory_n_step']['name'], env.observation_space, q_learning_config['memories']['memory_n_step']['kwargs'])

    q_learning_args = (q_learning_config['name'], env, model,
                   q_learning_optimizer, q_learning_config['batch_size'], memory)
    q_learning_kwargs = q_learning_config['kwargs']
    q_learning_kwargs['memory_n_step'] = memory_n_step
    q_learning_kwargs['scheduler'] = q_learning_scheduler
    q_learning_kwargs['device'] = device

    q_learning_agent = CreateAgent(q_learning_config['type'], q_learning_args, q_learning_kwargs)

    return q_learning_agent

def CreateSupervisedAgent(config, env):
    supervised_config = config
    device = supervised_config['device']

    # supervised learning
    model = NeuralNetworkFactory.CreateNetwork(
        supervised_config['model'], env.observation_space, env.action_space, device)
    supervised_optimizer = OptimizerFactory.CreateOptimizer(
        supervised_config['optimizer']['name'], (model.parameters(),), supervised_config['optimizer']['kwargs'])
    criterion = nn.CrossEntropyLoss()  # TODO move into a factory? // Note this is categorical cross entropy loss
    trainset = BoatDatasetRetriever.GetDataset(
        supervised_config['dataset']['train_set_dir'], supervised_config['dataset']['multi_keys'])
    trainset_dataloader = data_utils.DataLoader(
        trainset, shuffle=supervised_config['shuffle'], batch_size=supervised_config['batch_size'], num_workers=0)
    supervised_scheduler = SchedulerFactory.CreateScheduler(
        supervised_config['scheduler']['name'], supervised_optimizer, supervised_config['scheduler']['kwargs'])

    supervised_args = (model, supervised_optimizer,
                       criterion, trainset_dataloader, env)
    supervised_kwargs = supervised_config['kwargs']
    # supervised_kwargs["scheduler"] = supervised_scheduler

    supervised_agent = Supervised(
        *supervised_args, **supervised_kwargs, device=device)

    return supervised_agent