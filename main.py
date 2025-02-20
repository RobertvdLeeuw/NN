from functools import reduce
import operator
from random import random, shuffle

from tensors import Matrix, Vector


class Function:
    def __init__(self, name: str, func: callable, derivative: callable):
        self.name = name
        self.func = func
        self.deriv = derivative

    def compute(self, data: Vector) -> Vector:
        return Vector([self.func(x) for x in data])

    def compute_deriv(self, data: Vector) -> Vector:
        return Vector([self.deriv(x) for x in data])


class ANN:
    def __init__(self, layer_sizes: list[int], activation_funcs: list[Function], alpha: float, endlabels: list = None):
        self.weights = [Matrix.random(x, y) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [Vector.random(x) for x in layer_sizes[1:]]
        self.activation_funcs = activation_funcs

        self.alpha = alpha

        self.last_run_z = None  # Saved for backprop.
        self.last_run_A = None  # Saved for backprop.
        self.MSEs = None

        self._epoch_counter = 0

        self.endlabels = endlabels
        if endlabels and len(endlabels) != layer_sizes[-1]:
            raise Exception("Number of endlabels doesn't match length of output layer.")

    def __repr__(self):
        weights_str = "\n".join(f"{w}\n" for w in self.weights)
        biases_str = "\n".join(f"{b}\n" for b in self.biases)
        return f"Weights:\n{weights_str}\n\nBiases:\n{biases_str}\n"

    def train(self, features: list[Vector], labels: list[Vector], interval_k: int = 1000) -> dict:
        self.MSEs = []

        n = len(features)
        for f, l in zip(features, labels):
            _ = self.predict(f)

            self.correct(l, n)
            self._epoch_counter += 1

            if self._epoch_counter % interval_k == 0:
                print(f"Training at {self._epoch_counter} epochs.")

    def predict(self, features: Vector) -> list | dict:
        self.last_run_z = []
        self.last_run_A = []
        z = features

        for weights, biases, activ, in zip(self.weights, self.biases, self.activation_funcs):
            A = activ.compute(z)
            z = A * weights + biases
            self.last_run_A.append(A)
            self.last_run_z.append(z)

        A = self.activation_funcs[-1].compute(z)  # TODO: Better loop so this last case doesn't have to be excluded.
        self.last_run_A.append(A)

        return dict(zip(self.endlabels, A)) if self.endlabels else A

    def correct(self, labels: Vector, n: int):
        # Start with output layer error
        e = self.last_run_A[-1] - labels
        self.MSEs.append(e * e.transpose() / n)

        z_partials = [e]  # z_L = A_L - Y

        # Compute z_partials backwards through layers
        for i in range(len(self.weights) - 1, 0, -1):  # Go from L-1 to 1
            z = self.last_run_z[i - 1]  # Get z for current layer
            weights = self.weights[i]  # Get weights for next layer
            activ = self.activation_funcs[i]  # Get activation for current layer

            # Calculate partial using the formula: (∂z_{l+1} · W_{l+1}^T) ⊙ f'(z_l)
            z_partial = (z_partials[-1] * weights.transpose()).hadamard(activ.compute_deriv(z))
            z_partials.append(z_partial)

        z_partials = list(reversed(z_partials))

        # Calculate weight partials using A_{l-1}^T · ∂z_l
        w_partials = [A.transpose() * z_p / n for A, z_p in zip(self.last_run_A, z_partials)]

        # Calculate bias partials
        b_partials = [z / n for z in z_partials]

        # Update weights and biases
        self.weights = [w - w_p * self.alpha for w, w_p in zip(self.weights, w_partials)]
        self.biases = [b - b_p * self.alpha for b, b_p in zip(self.biases, b_partials)]

    def save_parameters(self, name: str):
        pass


ReLU = Function(name="ReLU",
                func=lambda x: x if x > 0 else 0,
                derivative=lambda x: int(x > 0))

e = 2.71828
sig = lambda x: 1 / (1 + e ** (-x))
Sigmoid = Function(name="Sigmoid",
                   func=sig,
                   derivative=lambda x: sig(x) * (1 - sig(x)))

Identity = Function(name="Identity",
                    func=lambda x: x,
                    derivative=lambda x: 1)

layer_sizes = [11, 11, 6]
activation_funcs = [Identity, ReLU, Sigmoid]
alpha = 0.16

# Load CSV
features, labels = [], []
qualities = list(range(3, 9))
with open('WineQT.csv', 'r') as f:
    for line in f.readlines()[1:]:
        *feat, label, _ = line.split(',')

        features.append(Vector([float(f) for f in feat]))
        labels.append(Vector([float(int(label) == q) for q in qualities]))

from sklearn.model_selection import train_test_split

training_features, testing_features, training_labels, testing_labels = train_test_split(features, labels,
                                                                                        random_state=1,
                                                                                        test_size=0.15,
                                                                                        shuffle=True)

nn = ANN(layer_sizes, activation_funcs, alpha, endlabels=qualities)
nn.train(training_features, training_labels)


def argmax_label(d: dict) -> str:
    # return the key of which the value is highest of all values
    return [label for label in d if d[label] == max(d.values())][0]


correct = 0
for f, l in zip(testing_features, testing_labels):
    output = nn.predict(f)
    pred = max(output.keys(), key=lambda k: output[k])
    label = float(max(range(3, 9), key=lambda x: l[x - 3]))

    correct += pred == label

print(f"SCORE:{correct / len(testing_features) * 100:2f}%")

# MSE and such
from pprint import pprint

pprint(nn.MSEs[0])
pprint(nn.MSEs[-1])
