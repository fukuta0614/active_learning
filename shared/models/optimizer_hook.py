from chainer import cuda


class DifferentLearningRate(object):
    """Optimizer hook function for different learning rate.

    This hook function multiple scaled parameter to the specified gradient.

    Args:
        target (dict) : {key: value} = {layer name: rate}

    Attributes:
        target (dict) : {key: value} = {layer name: rate}

    """
    name = 'DifferentLearningRate'

    def __init__(self, target):
        self.target = target

    def __call__(self, opt):
        for name, param in opt.target.namedparams():
            if name in self.target:
                rate = self.target[name]
                grad = param.grad
                with cuda.get_device(grad):
                    grad *= rate