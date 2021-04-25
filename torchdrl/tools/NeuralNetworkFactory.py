from torchdrl.neural_networks.ConvolutionNetwork1D import ConvolutionNetwork1D
from torchdrl.neural_networks.ConvolutionNetwork2D import ConvolutionNetwork2D
from torchdrl.neural_networks.FullyConnectedNetwork import FullyConnectedNetwork
from torchdrl.neural_networks.DuelingNetwork import DuelingNetwork
from torchdrl.neural_networks.NoisyDuelingNetwork import NoisyDuelingNetwork
from torchdrl.neural_networks.NoisyDuelingCategoricalNetwork import NoisyDuelingCategoricalNetwork

# TODO make this recursive and rely on id of groups and sequentials so that there can be more than one group, sequential and head.
def CreateNetwork(self, body_kwargs, in_features, out_features):
    network = None
    input_shape = in_features
    
    if isinstance(input_shape, (gym.spaces.Tuple, gym.spaces.Dict)):
        if 'group' in body_kwargs:
            networks = []

            for i, (net_type, net_kwargs) in enumerate(body_kwargs['group'].items()):
                networks.append(NetworkSelectionFactory(
                    net_type, input_shape[i].shape, net_kwargs, device=self._device))
            network = CombineNetwork(networks, self._device)
            input_shape = network.OutputSize()
        else:
            raise Exception(
                "Error. Gym tuple/dict detected, requires a grouping of networks")
    else:
        input_shape = input_shape.shape

    if 'sequential' in body_kwargs:
        for i, (net_type, net_kwargs) in enumerate(body_kwargs['sequential'].items()):
            network = NetworkSelectionFactory(
                net_type, input_shape, net_kwargs, network, device=self._device)
            input_shape = network.OutputSize()

    if 'head' in body_kwargs:
        for i, (net_type, net_kwargs) in enumerate(body_kwargs['head'].items()):
            net_kwargs['out_features'] = out_features
            network = NetworkSelectionFactory(
                net_type, input_shape, net_kwargs, network, device=self._device)
        
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
        return NoisyDuelingCategoricalNetworkFactory(input_shape, kwargs['out_features'], kwargs['atom_size'], kwargs['support'], kwargs["hidden_layers"], kwargs['activations'], kwargs['dropouts'], kwargs['final_activation'], network, device)
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

def NoisyDuelingCategoricalNetworkFactory(input_shape, n_actions, atom_size, support, hidden_layers, activations, dropouts, final_activation, network=None, device='cpu'):
    return NoisyDuelingCategoricalNetwork(input_shape, n_actions, atom_size, support, hidden_layers, activations, dropouts, final_activation, network, device)
