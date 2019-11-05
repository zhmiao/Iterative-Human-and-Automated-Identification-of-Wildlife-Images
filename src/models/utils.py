import torch

models = {}


def register_model(name):

    """
    Model register
    """

    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def get_model(name, **args):

    """
    Model getter
    """

    net = models[name](**args)
    if torch.cuda.is_available():
        net = net.cuda()
    return net

