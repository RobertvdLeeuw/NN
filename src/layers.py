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

    def __init__(self, shape_in: tuple[int], to_n: int, activation_func: Function, **kwargs):
        self.shape_in = shape_in
        self.shape_out = (1, to_n)
                 
        if val := kwargs.get("param_default_val"): 
            self.weights = Matrix.filled(val, shape_in[1], to_n)
            self.biases = Vector.filled(val, to_n)
        else:
            seed(kwargs.get("random_seed"))
            
            self.weights = Matrix.random(shape_in[1], to_n)
            self.biases = Vector.random(to_n)

        self.activ = activation_func
        self.last = False

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
        return f"{self.name} ({self.activ.name})\n" \
               f"   weights:\n{self.weights}\n\n" \
               f"   biases: {self.biases}\n\n"

        
class ConvolvingLayer(Layer):
    name = "Convolving Layer"
    
    # TODO: BIAS!!!!!!!!!!!

    def __init__(self, input_shape: tuple[int, int, int], kernel_n: int, kernel_shape: tuple[int, int], **kwargs):
        self.shape_in = input_shape
        self.shape_out = self._calc_output_shape()

        if val := kwargs.get("param_default_value"):
            self.kernels = [Matrix.filled(val, *kernel_shape) for _ in range(kernel_n)] 
        else:
            random.seed(kwargs.get("random_seed"))
            self.kernels = [Matrix.random(*kernel_shape) for _ in range(kernel_n)] 
        
        self.kernel_shape = kernel_shape
        self.stride = stride
        
        self.activ = kwargs.get("activation_func", Identity)
    
        self.padding_size = kwargs.get("padding_size", 0)
        self.padding_value = kwargs.get("padding_value", 0)
        self.stride = kwargs.get("stride", (1, 1))

        self.z = None
        self.A = None
    
        # Used for sum, don't have to init a new matrix for every forward prop this way.
        feature_maps_shape = input_shape[0] - kernels[0].shape[0] + 1, input_shape[1] - kernels[0].shape[1] + 1
        self.zero_feature = Matrix.filled(0, *feature_maps_shape)

    def _calc_output_shape(self) -> tuple[int, int, int]:  # https://www.baeldung.com/cs/convolutional-layer-size
        shape = ((x_in) / stride for x_in, stride in zip(self.shape_in[0:3], self.stride))
        depth = self.input_shape[2] * len(self.kernels)
        
        return ((self.shape_in[0] - self.kernel_shape[0] + 2*self.padding_size) / stride[0] + 1, 
                (self.shape_in[1] - self.kernel_shape[1] + 2*self.padding_size) / stride[1] + 1, 
                self.input_shape[2] * len(self.kernels))

    def predict(self, prev_A: list[Matrix]) -> list[Matrix]:
        if self.padding_size:
            prev_A = [a.pad(self.padding_size, self.padding_value) for a in prev_A]            
        
        self.z = [a.convolve(k, self.stride) for a in prev_A for k in self.kernels]
        self.A = self.activ.compute(self.z)

        return self.A
    
    def backprop(self, next_layer: list[Matrix]) -> list[Matrix]:
        # W_partial = next_layer.z_partial * 

        return self

    def gd(self, prev_A: list[Matrix], alpha: float, n: int) -> list[Matrix]:
        # W -= alpha * w_partial

        return self.A
        
    def __repr__(self):
        kernel_str = "\n\n".join([str(k) for k in self.kernels])
        
        return f"{self.name} ({l.activ.name})\n" \
               f"   shape:\n{self.shape}\n" \
               f"   kernels:\n{kernel_str}\n\n" 
        

class PoolingLayer(ConvolvingLayer):
    name = "Pooling Layer"
    hows = ['max', 'avg']
    
    def __init__(self, shape_in: tuple[int, int, int], window: tuple[int, int], how: str):
        self.how = how
        
        super().__init__(input_shape, 1, window, stride=window)

        if how == 'avg':
            size = window[0] * window[1]
            self.kernels = [Matrix.filled(1 / size, *window)] 
        elif how != 'max':
            raise Exception(f"Invalid pooling method: {how}")    
            
    def predict(self, prev_A: list[Matrix]) -> list[Matrix]:
        if how == 'max':
            self.z = [a.max_convolve(k, self.stride) for a in prev_A for k in self.kernels]
            self.A = self.activ.compute(self.z)

            return self.A
        elif how == 'avg':
            return super().predict(prev_A)
    
    def backprop(self, next_layer: list[Matrix]) -> list[Matrix]:
        pass

    def gd(self, prev_A: list[Matrix], alpha: float, n: int) -> list[Matrix]:
        pass

    def __repr__(self):
        return f"{self.name} ({self.stride[0]}x{self.stride[1]} {self.how})\n\n"
               

class FlatteningLayer(Layer):
    name = "Flattening Layer"
    
    def __init__(self, shape_in: tuple[int, int, int]):
        self.shape_in = shape_in
        self.shape_out = (1, reduce(operator.mul, shape_in))
        
        self.z = None
        self.z_partial = None
        
        self.A = None

    def predict(self, prev_A: list[Matrix] | Matrix) -> Vector:
        if isinstance(prev_A, list):
            self.A = Vector([x for x in m.flatten() for m in prev_A])
        else:
            self.A = prev_A.flatten()

        return self.A

    def unflatten(self, data: Vector) -> Matrix | list[Matrix]:
        rows, cols, depth = map(range, self.shape_in)
        
        f = [[[0 for _ in cols] for _ in rows] for _ in depth]

        row_interval = self.shape_in[1]
        layer_interval = self.shape[0] * layer_interval
        for r, c, l in cartprod(rows, cols, depth):
            f[l][r][c] = data[l*layer_interval + r*row_interval + c]

        f = [Matrix(x) for x in f]
        
        return f[0] if len(f) == 1 else f
        
    def backprop(self, next_layer: Vector) -> Matrix | list[Matrix]:
        return self

    def gd(self, prev_A: Matrix | list[Matrix], alpha: float, n: int) -> Vector:
        return self.A
        
    def __repr__(self):
        return f"{l.name}\n\n"
        
