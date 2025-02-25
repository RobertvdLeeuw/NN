import operator
from random import random


def cartprod(*lists):
    if len(lists) < 2:
        return lists

    x, y = lists.pop(), lists.pop()
    cp = [(a, b) for a in x for b in y]
    
    if len(lists) == 2:   
        return cp

    return cartprod(cp, *lists)


def length_check(func):
    def wrapper(*args, **kwargs):
        if not isinstance(args[1], Vector):  # Vector +-/* int or float
            return func(*args, **kwargs)

        if len(args[0]) != len(args[1]):
            raise Exception(f"Tried to perform operator on differing length vectors ({args[0]}, {args[1]}).")
        
        return func(*args, **kwargs)
    return wrapper

    
def round_number(x: float, n_digits: int) -> str:
    if x.is_integer():
        return str(x)

    if len(str(x).split(".")[1]) > n_digits:
        return str(round(x, n_digits)) + "…"
        
    return str(x)
    

class Vector:
    __slots__ = ['values', '_len', 'shape']
    n_digits_clamp = 4

    def __init__(self, values: list):
        self.values = values
        self._len = len(values)
        self.shape = (1, self._len)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, i) -> int | float:
        return self.values[i]

    def __iter__(self):
        return self.values.__iter__()

    def transpose(self):
        a = Matrix([[x] for x in self.values])
        return a

    def copy(self) -> "Vector":
        return Vector(self.values)
    
    @staticmethod
    def random(n: int) -> "Vector":
        return Vector([random() for _ in range(n)])

    def dotprod(self, other) -> int:
        if len(self) == 1:            
            return other * self[0]  # Handling it as matrix multiplication.

        if len(self) != len(other):
            raise Exception(f"Tried to perform operator on differing length vectors ({self}, {other}).")

        return sum(self[i] * other[i] for i in range(len(self)))

    @length_check
    def crossprod(self, other) -> "Vector":
        if len(self) != 3:
            raise Exception("Cross products are only implemented for the case of len 3.")

        return Vector([self[1] * other[2] - self[2] * other[1],
                       self[2] * other[0] - self[0] * other[2],
                       self[0] * other[1] - self[1] * other[0]])

    def unwrap(self) -> float | int:
        if len(self) > 1:
            raise Exception("Tried to unwrap vector of length", len(self))

        return self[0]

    @staticmethod
    def filled(val: float, n: int) -> "Vector":
        return Vector([val] * n)

    @length_check
    def _element_wise_operation(self, other, operand) -> "Vector":
        if isinstance(other, Vector):
            return Vector([operand(self[i], other[i]) for i in range(len(self))])

        return Vector([operand(x, other) for x in self])

    def __add__(self, other) -> "Vector":
        return self._element_wise_operation(other, operator.add)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other) -> "Vector":
        return self._element_wise_operation(other, operator.sub)

    def __rsub__(self, other):
        return self - other

    def __truediv__(self, other) -> "Vector":
        return self._element_wise_operation(other, operator.truediv)

    def __rtruediv__(self, other):
        return self / other

    def __mul__(self, other) -> "Vector":
        if isinstance(other, Vector):
            return self.dotprod(other)
        if isinstance(other, Matrix):
            return NotImplemented

        return Vector([x * other for x in self])  # Fixed multiplication with scalar
    
    def __rmul__(self, other):
        return self * other

    def __repr__(self):
        return "[" +  ", ".join([round_number(x, Vector.n_digits_clamp) for x in self]) + "]"
        
    def __abs__(self) -> "Vector":
        return Vector([abs(x) for x in self])
        
    def sum(self) -> float:
        return sum(self.values)

    def hadamard(self, other) -> "Vector":
        return self._element_wise_operation(other, operator.mul)

    def pad(self, n: int, val: float = 0.0) -> "Vector":
        padding = [val] * n
        return Vector(padding + self.values + padding)


def shape_check(func):
    def wrapper(*args, **kwargs):
        if not isinstance(args[1], Matrix):  # Vector +-/* int or float
            return func(*args, **kwargs)

        if args[0].shape != args[1].shape:
            raise Exception(f"Tried to get the dot product of differently shaped matrices ({args[0].shape, args[1].shape}).")
        
        return func(*args, **kwargs)
    return wrapper


def matrix_mul(self, other):  # Kinda messy, but better than duplication for edgecases with __rmul__().
    if not isinstance(other, Matrix):
        return Matrix([x * other for x in self])

    if self.shape[0] != other.shape[1]:
        raise Exception(f"Tried to multiply matrices of unsupported shapes ({self.shape}, {other.shape})")

    other_tp = other.transpose()
    return Matrix([[x.dotprod(y) for y in other_tp] for x in self])


# IMPORTANT: values is a list of rows, so len(val) = n_rows and len(val[0]) = n_cols.
class Matrix:
    __slots__ = ['values', 'shape']

    def __new__(cls, values):  # 1 row matrix -> vector coercion.
        if len(values) == 1:
            return Vector(values[0])
        return super().__new__(cls)
        
    def __init__(self, values):
        if any(len(values[0]) != len(values[i]) for i in range(1, len(values))):
            raise ValueError(f"Mismatching rows lengths for matrix ({[len(x) for x in values]}).")
        self.values = [Vector(row) if not isinstance(row, Vector) else row for row in values]
        self.shape = (len(self.values), len(values[0]))

    def __getitem__(self, i) -> Vector:
        return self.values[i]

    def __iter__(self):
        return self.values.__iter__()

    @staticmethod
    def random(n_rows: int, n_cols: int) -> "Vector":
        return Matrix([Vector.random(n_cols) for _ in range(n_rows)])

    def copy(self) -> "Matrix":
        return Matrix(*[x.copy() for x in self])

    def transpose(self) -> "Matrix":
        return Matrix([list(x) for x in zip(*self)])
    
    def determinant(self) -> float:
        n_rows, n_cols = self.shape

        if n_rows != n_cols:
            raise Exception("Determinants are only supported for square matrices.")
            
        if n_rows == 1:
            return self[0][0]
        if n_rows == 2:
            return self[0][0] * self[1][1] - self[0][1] * self[1][0]
        
        # Laplace expansion along first row.
        def cofactor(j: int) -> float | int:
            # Excluding first row and current column.
            minor = [[self[i][k] for k in range(n_cols) if k != j] 
                     for i in range(1, n_rows)]

            return self[0][j] * ((-1) ** j) * Matrix(minor).determinant()
        
        return sum(cofactor(j) for j in range(n_cols))

    @staticmethod
    def filled(val: float, n_rows: int, n_cols: int) -> "Matrix":
        return Matrix([Vector.filled(val, n_cols)] * n_rows)

    @shape_check
    def _element_wise_operation(self, other, operand) -> "Matrix":
        if isinstance(other, Matrix):
            return Matrix([operand(self[i], other[i]) for i in range(self.shape[0])])
                         
        return Matrix([operand(x, other) for x in self])

    def __add__(self, other) -> "Matrix":
        return self._element_wise_operation(other, operator.add)

    def __radd__(self, other):
        return self + other

    @shape_check
    def __truediv__(self, other) -> "Matrix":
        return self._element_wise_operation(other, operator.truediv)

    def __rtruediv__(self, other):
        return self / other

    @shape_check
    def __sub__(self, other) -> "Matrix":
        return self._element_wise_operation(other, operator.sub)

    def __rsub__(self, other):
        return self - other

    def __mul__(self, other):
        if isinstance(other, Vector):  # Treat as row vector
            return Matrix([Vector([row.unwrap() * x for x in other]) for row in self])

            # Treat vector as column vector (Nx1)
            if self.shape[1] != len(other):
                raise Exception(f"Incompatible shapes for matrix-vector multiplication: {self.shape} and {len(other)}")
            return Vector([row.dotprod(other) for row in self])
        return matrix_mul(self, other)

    def __rmul__(self, other):
        if isinstance(other, Vector):
            # Treat vector as 1xN matrix multiplying an NxM matrix
            return Vector([sum(other[i] * self[i][j] for i in range(len(other))) 
                          for j in range(len(self[0]))])
        return matrix_mul(other, self)

    def __repr__(self):
        return "\n".join(str(x) for x in self)
        
    def __abs__(self) -> "Matrix":
        return Matrix([abs(v) for v in self])

    def sum(self) -> float:
        return sum(v.sum() for v in self)

    def hadamard(self, other) -> "Matrix":
        return self._element_wise_operation(self, other, operator.mul)
        
    def flatten(self) -> Vector:
        return Vector([x for x in v for v in self])

    def convolve(self, kernel: "Matrix", stride: tuple[int, int] = (1, 1)) -> "Matrix":
        n_rows, n_cols = self.shape[0] - kernel.shape[0] + 1, self.shape[1] - kernel.shape[1] + 1
        f = [[0 for _ in range(n_cols)] for _ in range(n_rows)]

        for i, j in cartprod(range(n_rows), range(n_cols)):
            for x, y in cartprod(range(kernel.shape[0]), range(kernel.shape[1])):
                f[i][j] += self[i*stride[0] + x][j*stride[1] + y] * kernel[x][y]

        return Matrix(f)
    
    def max_convolve(self, window: tuple[int, int]) -> "Matrix":
        n_rows, n_cols = self.shape[0] - window[0] + 1, self.shape[1] - window[1] + 1
        f = [[None for _ in range(n_cols)] for _ in range(n_rows)]

        for i, j in cartprod(range(n_rows), range(n_cols)):
            for x, y in cartprod(range(window[0]), range(window[1])):
                if not f[i][j]:
                    f[i][j] = self[i*window[0] + x][j*window[1] + y]
                else:
                    f[i][j] = max(f[i][j], self[i*window[0] + x][j*window[1] + y])

        return Matrix(f)
    
    def pad(self, n: int, val: float = 0.0) -> "Matrix":
        _, n_cols = self.shape
        
        return Matrix([Vector.filled(val, n_cols + n*2)] * n +
                      [v.pad(n, val) for v in self] +
                      [Vector.filled(val, n_cols + n*2)] * n)


if __name__ == "__main__":
    m1 = Matrix([[1, 2, 3], 
                 [4, 5, 6], 
                 [7, 8, 9]])
    m2 = Matrix([[10, 11], [12, 13]])

    print(m1.convolve(m2))

    print(m2.pad(2))

