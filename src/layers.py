from abc import ABC, abstractmethod

from functions import Function
from tensors import Matrix, Vector


class Layer(ABC):
    @abstractmethod
    def predict(self):
        pass
        
    @abstractmethod
    def backprop(self):
        pass
        
    @abstractmethod
    def gd(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass


class DenseLayer(Layer):
    name = "Dense Layer"

    def __init__(self, from_n: int, to_n: int, activation_func: Function, /,
                 random_seed: int = None, param_default_val: float = None, last=False):
        if param_default_val:
            self.weights = Matrix.filled(param_default_val, from_n, to_n)
            self.biases = Vector.filled(param_default_val, to_n)
        else:
            if random_seed:
                seed(random_seed)
            self.weights = Matrix.random(from_n, to_n)
            self.biases = Vector.random(to_n)

        self.activ = activation_func
        self.last = last

        self.z = None
        self.z_partial = None

        self.A = None

    def predict(self, prev_A: Vector) -> Vector:
        self.z = prev_A * self.weights
        self.A = self.activ.compute(self.z)

        return self.A
        
    def backprop(self, next_layer: Vector | Layer) -> Vector:
        if self.last:
            self.z_partial = self.A - next_layer  # Last layer, so next is actual labels.
        else:
            self.z_partial = (next_layer.z * next_layer.weights.transpose()
                              ).hadamard(self.activ.compute_deriv(self.z))

        return self  # For folding in actual model.

    def gd(self, prev_A: Vector, alpha: float, n: int) -> Vector:
        self.weights -= (prev_A.transpose() * self.z_partial) * alpha / n
        self.biases -=  self.z_partial * alpha / n

        return self.A
        
    def __repr__(self):
        return f"({l.name}, {l.activ.name})\n" \
               f"   weights:\n{l.weights}\n\n" \
               f"   biases: {l.biases}\n\n"


