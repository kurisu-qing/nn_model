import numpy as np


def relu6(x):
    x = np.asarray(x)
    return np.clip(x, 0, 6)


def sigmoid(x):
    """
    stable sigmoid
    """
    x = np.asarray(x)
    return np.where(x > 0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))


class Net():
    """
    Network inference numpy version
    """
    def __init__(self, wi, alpha, wo, act=relu6):
        self.wi = wi
        self.alpha = alpha
        self.wo = wo
        self.act = act

    def forward(self, x, hx):
        g = self.alpha
        act = self.act
        i = act(np.dot(self.wi, x))
        h = g * hx + (1 - g) * i
        y = np.dot(self.wo, act(h))
        return y, h


class Infer():

    def __init__(self):
        pass

    def fearture(self):
        pass

    def predict(self):
        pass
