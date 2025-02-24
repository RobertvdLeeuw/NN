from dataclasses import dataclass
from tensors import Vector, Matrix

@dataclass
class Function:
    name: str
    func: callable
    deriv: callable

    def compute(self, data: Vector) -> Vector:
        return Vector([self.func(x) for x in data])

    def compute_deriv(self, data: Vector) -> Vector:
        return Vector([self.deriv(x) for x in data])


ReLU = Function(name="ReLU",
                func=lambda x: x if x > 0 else 0,
                deriv=lambda x: int(x > 0))

e = 2.71828
sig = lambda x: 1 / (1 + e ** (-x))
Sigmoid = Function(name="Sigmoid",
                   func=sig,
                   deriv=lambda x: sig(x) * (1 - sig(x)))

Identity = Function(name="Identity",
                    func=lambda x: x,
                    deriv=lambda x: 1)

