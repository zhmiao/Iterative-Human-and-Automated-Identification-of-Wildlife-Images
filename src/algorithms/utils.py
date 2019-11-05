algorithms = {}


def register_algorithm(name):

    """
    Algorithm register
    """

    def decorator(cls):
        algorithms[name] = cls
        return cls
    return decorator


def get_algorithm(name, args):

    """
    Algorithm getter
    """

    alg = algorithms[name](args)
    return alg
