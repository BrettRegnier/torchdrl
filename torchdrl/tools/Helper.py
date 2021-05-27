import os
import torch
import torch.nn as nn 
import torch.optim as optim
import numpy as np

def AppendExtension(filepath, ext):
    if ext[0] != ".":
        ext = "." + ext

    if ext.lower() != filepath[-len(ext):].lower():
        filepath += ext
    
    return filepath

def AppendSlash(folderpath):
    folderpath = folderpath if folderpath[len(folderpath) - 1] == "/" else folderpath + "/"
    return folderpath

def CreateSaveLocation(folderpath):
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)

def CheckFileLocation(filepath):
    return os.path.exists(filepath)

def SplitFilename(filepath):
    idx = filepath.rfind("/")
    filename = filepath[idx+1:]
    folderpath = filepath[:idx]
    return (filename, folderpath)

# TODO rename to SaveTorchFile
def SaveAgent(folderpath, filename, agent_dict):
    filename = AppendExtension(filename, "pt")
    folderpath = AppendSlash(folderpath)
    CreateSaveLocation(folderpath)

    torch.save(agent_dict, folderpath + filename)

# TODO rename to LoadTorchFile
def LoadAgent(filepath):
    if CheckFileLocation(filepath):
        checkpoint = torch.load(filepath)
        return checkpoint 
    else:
        filename, folderpath = SplitFilename(filepath)
        raise Exception(f"{filename} does not exist at {folderpath}")

# TODO Fix
# I need to give it some way of distinguishing between a list of lists
# and a list of different states objects, which could be based on shape?
def ConvertStateToTensor(states, device="cpu", state_is_tuple=False):
    if state_is_tuple:
        states_t = []
        for state in states:
            if isinstance(state, object) and not isinstance(state, np.ndarray):
                tuple_states = []
                for val in state:
                    tuple_state_t = torch.tensor(val, dtype=torch.float32,
                                    device=device).unsqueeze(0).detach()
                    tuple_states.append(tuple_state_t)
                states_t.append(tuple_states)
            else:
                raise Exception("The states array is not an instance of np.ndarray")
    else:
        states_t = torch.tensor(states, dtype=torch.float32,
                            device=device).detach()
        
    return states_t

def CreateOptimizer(optimizer_name, network_params, optimizer_args):
    if optimizer_name.lower() == "adam":
        return optim.Adam(network_params, **optimizer_args)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(network_params, **optimizer_args)

def UpdateNetwork(from_net, to_net, tau=1.0):
    if tau == 1.0:
        to_net.load_state_dict(from_net.state_dict())
    else:
        for from_param, to_param in zip(from_net.parameters(), to_net.parameters()):
            to_param.data.copy_(tau*from_param.data + (1.0-tau) * to_param.data)