import gym
from torchdrl.neural_networks.ConvolutionNetwork1D import ConvolutionNetwork1D
from torchdrl.neural_networks.ConvolutionNetwork2D import ConvolutionNetwork2D
from torchdrl.neural_networks.FullyConnectedNetwork import FullyConnectedNetwork
from torchdrl.neural_networks.DuelingNetwork import DuelingNetwork
from torchdrl.neural_networks.NoisyDuelingNetwork import NoisyDuelingNetwork
from torchdrl.neural_networks.NoisyDuelingCategoricalNetwork import NoisyDuelingCategoricalNetwork
from torchdrl.neural_networks.CombineNetwork import CombineNetwork

# TODO make this recursive and rely on id of groups and sequentials so that there can be more than one group, sequential and head.
def CreateNetwork(network_kwargs, observation_space, action_space, device="cpu"):
    network = None
    input_shape = observation_space
    n_actions = action_space.n # TODO more complex
    
    if isinstance(input_shape, (gym.spaces.Tuple, gym.spaces.Dict)):
        if 'group' in network_kwargs:
            networks = []

            for i, (net_type, net_kwargs) in enumerate(network_kwargs['group'].items()):
                networks.append(NetworkSelectionFactory(
                    net_type, input_shape[i].shape, net_kwargs, device=device))
            network = CombineNetwork(networks, device)
            input_shape = network.OutputSize()
        else:
            raise Exception(
                "Error. Gym tuple/dict detected, requires a grouping of networks")
    else:
        input_shape = input_shape.shape

    if 'sequential' in network_kwargs:
        for i, (net_type, net_kwargs) in enumerate(network_kwargs['sequential'].items()):
            network = NetworkSelectionFactory(
                net_type, input_shape, net_kwargs, network, device=device)
            input_shape = network.OutputSize()

    if 'head' in network_kwargs:
        for i, (net_type, net_kwargs) in enumerate(network_kwargs['head'].items()):
            net_kwargs['out_features'] = n_actions
            network = NetworkSelectionFactory(
                net_type, input_shape, net_kwargs, network, device=device)

    return network
        
def NetworkSelectionFactory(network_type, input_shape, kwargs, network=None, device='cpu'):
    if network_type == "conv1d":
        if 'out_features' in kwargs and 'filters' not in kwargs:
            kwargs['filters'] = kwargs['out_features']
        return ConvolutionNetwork1D(input_shape, kwargs['filters'], kwargs['kernels'], kwargs['strides'], kwargs['paddings'], kwargs['activations'], kwargs['pools'], kwargs['flatten'], network, device)
    elif network_type == "conv2d":
        if 'out_features' in kwargs and 'filters' not in kwargs:
            kwargs['filters'] = kwargs['out_features']
        return ConvolutionNetwork2D(input_shape, kwargs['filters'], kwargs['kernels'], kwargs['strides'], kwargs['paddings'], kwargs['activations'], kwargs['pools'], kwargs['flatten'], network, device)
    elif network_type == "fullyconnected":
        return FullyConnectedNetworkFactory(input_shape, kwargs['out_features'], kwargs["hidden_layers"], kwargs['activations'], kwargs['dropouts'], kwargs['final_activation'], network, device)
    elif network_type == "dueling":
        return DuelingNetworkFactory(input_shape, kwargs['out_features'], kwargs["hidden_layers"], kwargs['activations'], kwargs['dropouts'], kwargs['final_activation'], network, device)
    elif network_type == "noisydueling":
        raise NotImplementedError()
        return NoisyDuelingNetwork(input_shape, kwargs['out_features'], kwargs["hidden_layers"], kwargs['activations'], kwargs['dropouts'], kwargs['final_activation'], network, device)
    elif network_type == "noisyduelingcategorical":
        return NoisyDuelingCategoricalNetworkFactory(input_shape, kwargs['out_features'], kwargs['v_min'], kwargs['v_max'], kwargs['atom_size'], kwargs["hidden_layers"], kwargs['activations'], kwargs['dropouts'], kwargs['final_activation'], network, device)
    else:
        raise NotImplementedError(network_type + "is not implemented")

def ConvolutionNetwork1DFactory(input_shape, filters, kernels, strides, paddings, activations, pools, flatten=True, network=None, device='cpu'):
    return ConvolutionNetwork1D(input_shape, filters, kernels, strides, paddings, activations, pools, flatten, network, device)

def ConvolutionNetwork2DFactory(input_shape, kernels, strides, paddings, activations, pools, flatten=True, network=None, device='cpu'):
    return ConvolutionNetwork2D(input_shape, filters, kernels, strides, paddings, activations, pools, flatten, network, device)

def FullyConnectedNetworkFactory(input_shape, n_actions, hidden_layers, activations, dropouts, final_activation, network=None, device='cpu'):
    return FullyConnectedNetwork(input_shape, n_actions, hidden_layers, activations, dropouts, final_activation, network, device)

def DuelingNetworkFactory(input_shape, n_actions, hidden_layers, activations, dropouts, final_activation, network=None, device='cpu'):
    return DuelingNetwork(input_shape, n_actions, hidden_layers, activations, dropouts, final_activation, network, device)

def NoisyDuelingNetworkFactory(input_shape, n_actions, hidden_layers, activations, dropouts, final_activation, network=None, device='cpu'):
    return DuelingNetwork(input_shape, n_actions, hidden_layers, activations, dropouts, final_activation, network, device)

def NoisyDuelingCategoricalNetworkFactory(input_shape, n_actions, v_min, v_max, atom_size, hidden_layers, activations, dropouts, final_activation, network=None, device='cpu'):
    return NoisyDuelingCategoricalNetwork(input_shape, n_actions, v_min, v_max, atom_size, hidden_layers, activations, dropouts, final_activation, network, device)
