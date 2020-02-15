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


class Algorithm:

    """
    Base Algorithm class for reference.
    """

    name = None

    def __init__(self, args):
        self.args = args
        self.logger = self.args.logger
        self.weights_path = './weights/{}/{}_{}.pth'.format(self.args.algorithm, self.args.conf_id, self.args.session)

    def train_epoch(self, epoch):
        pass

    def train(self):
        pass

    def evaluate_epoch(self, loader):
        pass

    def evaluate(self, loader):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass

