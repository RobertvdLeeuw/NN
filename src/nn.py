from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce
import operator
from random import random, shuffle, seed

from functions import Function, ReLU, Sigmoid, Identity
from layers import Layer, DenseLayer
from tensors import Matrix, Vector


class NN:
    def __init__(self, shape_in: tuple[int], /, alpha: float, endlabels: list = None):
        self.shape_in = shape_in
        self.layers = []

        self.alpha = alpha
        
        self.MSEs = []
        self.MAEs = []
        self.corrects = []

        self._epoch_counter = 0

        self.endlabels = None
        self.baked = False

    def add_layer(self, layer: Layer, *args, **kwargs) -> "NN":
        prev_shape_out = self.shape_in if not self.layers else self.layers[-1].shape_out
        new_layer = layer(prev_shape_out, *args, **kwargs)

        self.layers.append(new_layer)
        self.baked = False

        return self  # For chaining.

    def bake(self, endlabels: list = None) -> "NN":  # Final settings upon model building completion.
        self.layers[-1].last = True  # Needed for proper GD formula selection.

        if endlabels:
            if len(endlabels) != self.layers[-1].shape_out:
                raise Exception("Number of endlabels doesn't match length of output layer.")
                    
            self.endlabels = endlabels

        self.baked = True
        return self

    def train(self, features: list[Vector], labels: list[Vector], eval_func: callable = None, interval_k: int = 1000) -> dict:
        self.MSEs = []
        self.MAEs = []
        self.corrects = []

        n = len(features)
        for f, l in zip(features, labels):
            _ = self.predict(f)

            self.backprop_gd(l, n, eval_func)
            self._epoch_counter += 1

            if self._epoch_counter % interval_k == 0:
                print(f"Training at {self._epoch_counter} epochs.")

    def predict(self, features: Vector) -> Vector | dict:
        if not self.baked:
            raise Exception("Model isn't baked (call nn.bake() after adding all layers).")

        out = reduce(lambda prev, l: l.predict(prev), self.layers, features)
        return dict(zip(self.endlabels, out)) if self.endlabels else out

    def correct(self, labels: Vector, n: int, eval_func: callable = None):
        if eval_func:
            A = self.layers[-1].A
            last_out = dict(zip(self.endlabels, A)) if self.endlabels else A
            
            self.corrects.append(eval_func(last_out, labels) * 100)
            # *100 so mean/moving avg becomes % correct.
        
        e = self.layers[-1].A - labels
        self.MAEs.append(abs(e).sum())
        self.MSEs.append((e * e.transpose()).unwrap())
        
        reduce(lambda next, l: l.backprop(next), reversed(self.layers), labels)
        reduce(lambda prev, l: l.gd(prev, self.alpha, n), self.layers, labels)

    def __repr__(self) -> str:
        text = "==================== NN ====================\n" \
              f"    alpha: {self.alpha}\n" \
              f"    n_layers: {len(self.layers)}\n\n" \
               " - Layers ----------------------------------\n"
        
        for i, l in enumerate(self.layers, start=1):
            text += f" {i} " + repr(l)
        text += "============================================"

        return text


if __name__ == "__main__":
    X = Vector([0.5, 0.8])
    Y = Vector([0.4])

    random_seed = 2
    nn = (NN(X.shape, alpha=0.1)
          .add_layer(DenseLayer, 3, ReLU, param_default_val=0.5)
          .add_layer(DenseLayer, 1, Sigmoid, param_default_val=0.5)
          .bake())

    nn.predict(X)
    nn.correct(Y, 1)
    print(nn)

