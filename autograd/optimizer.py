"""
Optimizers are used to adjust the parameters of a neural network.
Popular optimizers include SGD, Adam, and RMSProp. 
Implemented optimizers: SGD.
"""
import numpy as np

from autograd.module import Module
from autograd.tensor import Tensor


class Optimizer:
    def step(self, module: Module):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr=0.001) -> None:
        self.lr = lr

    def step(self, module: Module):
        for param in module.parameters():
            param -= param.grad * self.lr