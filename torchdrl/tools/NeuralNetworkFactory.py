from torchdrl.neural_networks.ConvolutionNetwork1D import ConvolutionNetwork1D
from torchdrl.neural_networks.ConvolutionNetwork2D import ConvolutionNetwork2D
from torchdrl.neural_networks.FullyConnectedNetwork import FullyConnectedNetwork
from torchdrl.neural_networks.DuelingNetwork import DuelingNetwork
from torchdrl.neural_networks.NoisyDuelingNetwork import NoisyDuelingNetwork
from torchdrl.neural_networks.NoisyDuelingCategoricalNetwork import NoisyDuelingCategoricalNetwork


def ConvolutionNetwork1DFactory(input_shape, filters, kernels, strides, paddings, activations, pools, flatten=True, body=None, device='cpu'):
    return ConvolutionNetwork1D(input_shape, filters, kernels, strides, paddings, activations, pools, flatten, body, device)


def ConvolutionNetwork2DFactory(input_shape, kernels, strides, paddings, activations, pools, flatten=True, body=None, device='cpu'):
    return ConvolutionNetwork2D(input_shape, filters, kernels, strides, paddings, activations, pools, flatten, body, device)


def FullyConnectedNetworkFactory(input_shape, n_actions, hidden_layers, activations, dropouts, final_activation, body=None, device='cpu'):
    return FullyConnectedNetwork(input_shape, n_actions, hidden_layers, activations, dropouts, final_activation, body, device)


def DuelingNetworkFactory(input_shape, n_actions, hidden_layers, activations, dropouts, final_activation, body=None, device='cpu'):
    return DuelingNetwork(input_shape, n_actions, hidden_layers, activations, dropouts, final_activation, body, device)


def NoisyDuelingNetworkFactory(input_shape, n_actions, hidden_layers, activations, dropouts, final_activation, body=None, device='cpu'):
    return DuelingNetwork(input_shape, n_actions, hidden_layers, activations, dropouts, final_activation, body, device)


def NoisyDuelingCategoricalNetworkFactory(input_shape, n_actions, atom_size, support, hidden_layers, activations, dropouts, final_activation, body=None, device='cpu'):
    return NoisyDuelingCategoricalNetwork(input_shape, n_actions, atom_size, support, hidden_layers, activations, dropouts, final_activation, body, device)


def NetworkSelectionFactory(network_type, input_shape, kwargs, body=None, device='cpu'):
    if network_type == "conv1d":
        return ConvolutionNetwork1D(input_shape, kwargs['filters'], kwargs['kernels'], kwargs['strides'], kwargs['paddings'], kwargs['activations'], kwargs['pools'], kwargs['flatten'], body, device)
    elif network_type == "conv2d":
        return ConvolutionNetwork2D(input_shape, kwargs['filters'], kwargs['kernels'], kwargs['strides'], kwargs['paddings'], kwargs['activations'], kwargs['pools'], kwargs['flatten'], body, device)
    elif network_type == "fullyconnected":
        return FullyConnectedNetworkFactory(input_shape, kwargs['out_features'], kwargs["hidden_layers"], kwargs['activations'], kwargs['dropouts'], kwargs['final_activation'], body, device)
    elif network_type == "dueling":
        return DuelingNetworkFactory(input_shape, kwargs['out_features'], kwargs["hidden_layers"], kwargs['activations'], kwargs['dropouts'], kwargs['final_activation'], body, device)
    elif network_type == "noisydueling":
        raise NotImplementedError()
        return NoisyDuelingNetwork(input_shape, kwargs['out_features'], kwargs["hidden_layers"], kwargs['activations'], kwargs['dropouts'], kwargs['final_activation'], body, device)
    elif network_type == "noisyduelingcategorical":
        return NoisyDuelingCategoricalNetworkFactory(input_shape, kwargs['out_features'], kwargs['atom_size'], kwargs['support'], kwargs["hidden_layers"], kwargs['activations'], kwargs['dropouts'], kwargs['final_activation'], body, device)
    else:
        raise NotImplementedError(network_type + "is not implemented")

# def NetworkFactory(input_shape, network_args, list_type=None):
#     body = []
#     for net_k, net_v in network_args.items():
#         if net_k == "group":
#             NetworkFactory(net_v, net_k)
#         elif net_k == "sequential":
#             NetworkFactory(net_v, net_k)
#         else:
#             if list_type == "sequential":
#                 body = [NetworkSelectionFactory(net_k, (net_v,), body)]
#             elif list_type == "group":
#                 body.append(NetworkSelectionFactory(net_k, (net_v,), body))

#     return body
