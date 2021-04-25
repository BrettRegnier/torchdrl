import torch.optim as optim

def CreateScheduler(name, args, kwargs):
    if name == "steplr":
        return optim.lr_scheduler.StepLR(*args, **kwargs)
    elif name == "None":
        return None