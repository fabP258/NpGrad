import numpy as np


class Node:

    def __init__(
        self, value: np.ndarray, requires_grad: bool = False, _children=(), _op=""
    ):
        self.value = value
        self.grad = np.zeros(value.shape)
        self._requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __matmul__(self, other):
        # do we need to check shapes?
        return Node(self.value @ other.value, _children=(self, other), _op="@")

    # activation functions
    def relu(self):

        assert (
            len(self.value.shape) < 3
        ), "ReLu is only supported up to two dimensional arrays"

        res = self.value
        res[res < 0] = 0

        return Node(res, _children=(self,), _op="ReLu")

    def softmax(self):

        assert (
            len(self.value.shape) < 3
        ), "Softmax is only supported up to two dimensional arrays"

        # TODO: Consider subtraction of max for numerical stability

        # exponentiate
        res = np.exp(self.value)

        # normalize
        res = res / np.sum(
            res, axis=0 if len(self.value.shape) == 1 else 1, keepdims=True
        )

        out = Node(res, _children=(self,), _op="Softmax")

        # TODO: Test this with a unit test
        def _backward():
            self.grad += (
                res * (-np.sum(res, axis=1, keepdims=True, initial=-1)) * out.grad
            )

        out._backward = _backward

        return out
