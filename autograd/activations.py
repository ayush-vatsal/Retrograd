import numpy as np

from autograd.tensor import Tensor, Dependency

# defining common activations
def tanh(tensor: Tensor) -> Tensor:
    data = np.tanh(tensor.data)
    requires_grad = tensor.requires_grad
    depends_on = []

    if requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (1 - data * data)

        depends_on = [Dependency(tensor, grad_fn)]

    return Tensor(data, requires_grad, depends_on)


def sigmoid(tensor: Tensor) -> Tensor:
    data = 1 / (1 + np.exp(-tensor.data))
    requires_grad = tensor.requires_grad
    depends_on = []

    if requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * data * (1 - data)

        depends_on = [Dependency(tensor, grad_fn)]

    return Tensor(data, requires_grad, depends_on)


def relu(tensor: Tensor) -> Tensor:
    data = np.maximum(tensor.data, 0)
    requires_grad = tensor.requires_grad
    depends_on = []

    if requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * int(data > 0)

        depends_on = [Dependency(tensor, grad_fn)]

    return Tensor(data, requires_grad, depends_on)
