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

def SaveAgent(folderpath, filename, agent_dict):
    filename = AppendExtension(filename, "pt")
    folderpath = AppendSlash(folderpath)
    CreateSaveLocation(folderpath)

    torch.save(agent_dict, folderpath + filename)

def LoadAgent(filepath):
    if CheckFileLocation(filepath):
        checkpoint = torch.load(filepath)
        return checkpoint 
    else:
        filename, folderpath = Helper.GetFileName(filepath)
        raise Exception(f"{filename} does not exist at {folderpath}")

def ConvertStateToTensor(state, device="cpu"):
    if isinstance(state, object) and not isinstance(state, np.ndarray):
        states_t = []
        for val in state:
            state_t = torch.tensor(val, dtype=torch.float32,
                            device=device).unsqueeze(0).detach()
            states_t.append(state_t)
    else:
        states_t = torch.tensor(state, dtype=torch.float32,
                            device=device).detach()
        states_t = states_t.unsqueeze(0)
        
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