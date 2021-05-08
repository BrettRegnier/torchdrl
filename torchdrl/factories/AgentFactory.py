import torch.utils.data as data_utils

from torchdrl.agents.markov.DQL import DQL
from torchdrl.agents.markov.DoubleDQL import DoubleDQL
from torchdrl.agents.markov.RainbowDQL import RainbowDQL

from torchdrl.agents.monte_carlo.Reinforce import Reinforce
from torchdrl.agents.monte_carlo.GAE import GAE
from torchdrl.agents.monte_carlo.PPO import PPO

import torchdrl.factories.NeuralNetworkFactory as NeuralNetworkFactory
import torchdrl.factories.OptimizerFactory as OptimizerFactory
import torchdrl.factories.SchedulerFactory as SchedulerFactory
import torchdrl.factories.MemoryFactory as MemoryFactory
import torchdrl.factories.AgentFactory as AgentFactory

import DragonFruit.datagen.BoatDatasetRetriever as BoatDatasetRetriever
import DragonFruit.datagen.BoatDataGenerator as BoatDataGenerator

def CreateMarkovAgent(config, env):
    markov_config = config
    device = markov_config['device']

    # reinforcement learning
    model = NeuralNetworkFactory.CreateNetwork(
        markov_config['model'], env.observation_space, env.action_space, device)
    markov_optimizer = OptimizerFactory.CreateOptimizer(
        markov_config['optimizer']['name'], (model.parameters(),), markov_config['optimizer']['kwargs'])
    markov_scheduler = SchedulerFactory.CreateScheduler(
        markov_config['scheduler']['name'], markov_optimizer, markov_config['scheduler']['kwargs'])
    memory = MemoryFactory.CreateMemory(
        markov_config['memories']['memory']['name'], env.observation_space, markov_config['memories']['memory']['kwargs'])

    memory_n_step = None
    if 'memory_n_step' in markov_config['memories']:
        memory_n_step = MemoryFactory.CreateMemory(
            markov_config['memories']['memory_n_step']['name'], env.observation_space, markov_config['memories']['memory_n_step']['kwargs'])

    markov_args = (markov_config['name'], env, model,
                   markov_optimizer, markov_config['batch_size'], memory)
    markov_kwargs = markov_config['kwargs']
    markov_kwargs['memory_n_step'] = memory_n_step
    markov_kwargs['scheduler'] = markov_scheduler
    markov_kwargs['device'] = device

    markov_agent = CreateAgent(markov_config['type'], markov_args, markov_kwargs)

    return markov_agent

def CreateSupervisedAgent(config, env):
    supervised_config = config
    device = supervised_config['device']

    # supervised learning
    model = NeuralNetworkFactory.CreateNetwork(
        supervised_config['model'], env.observation_space, env.action_space, device)
    supervised_optimizer = OptimizerFactory.CreateOptimizer(
        supervised_config['optimizer']['name'], (model.parameters(),), supervised_config['optimizer']['kwargs'])
    criterion = nn.CrossEntropyLoss()  # TODO move into a factory?
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