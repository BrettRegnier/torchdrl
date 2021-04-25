import torch.optim as optim

def CreateOptimizer(name, args, kwargs):
    if name.lower() == "adam":
        return optim.Adam(*args, **kwargs)
    elif name.lower() == "adagrad":
        return optim.Adagrad(*args, **kwargs)
    elif name.lower() == "sgd":
        return optim.SGD(*args, **kwargs)