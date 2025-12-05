import numpy as np


def unbroadcast(broadcasted_grad, target_shape):
        # gradient has more dims than original 
    # this is needed bcoz numpy automatically broadcasts the biases to the shape of the output
    # but the gradient is the sum over the broadcasted dimensions
    # sp we shrink the gradient back to the original shape by summing over the broadcasted dimensions

    grad = broadcasted_grad
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)
    # sum over added dimensions
    # dimensions match but lengths differ

    for axis, (current_dim, target_dim) in enumerate(zip(grad.shape, target_shape)):
        if target_dim == 1 and current_dim > 1:
            grad = grad.sum(axis=axis, keepdims=True)# sum over broadcasted dimensions

    return grad
 

class GradMode:
    enabled = True
    # global switch for enabling/disabling gradient tracking


class no_grad:
    def __enter__(self):
                # is run when user writes  "with no_grad():"

        self.previous_mode = GradMode.enabled
        GradMode.enabled = False
        
    def __exit__(self, *args):
        GradMode.enabled = self.previous_mode
        # is run at the end of the with block


class Context:
    # save information for the backward pass used in Function.forward to be used in Function.backward

    def __init__(self):
        self.saved_tensors = ()

    def save_tens_for_backward(self, *tensors):
        self.saved_tensors = tensors


class Function:
        # class for defining mathematical operations

    @staticmethod
    def forward(ctx, *args):
        raise NotImplementedError
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError
    
    @classmethod
    def apply(cls, *args, **kwargs):  # 1. Standard 'cls'
        ctx = Context()
        # create the individual context to store info for backward pass
        raw_args = [t.data if isinstance(t, Tensor) else t for t in args]# get data not tensors
        
        output_data = cls.forward(ctx, *raw_args, **kwargs)# call forward
        
        parents = [t for t in args if isinstance(t, Tensor)]# get the tensors with which this operation is called
        requires_grad = any(p.requires_grad for p in parents)# if any I do too xdd
        
        out = Tensor(output_data, _prev=tuple(parents), requires_grad=requires_grad)
        # create output tensor
        if requires_grad and GradMode.enabled:
            def _backward():
                grad_output = out.grad# this is the grad coming from the next node in the graph - we calculate backwards xddd
                
                if grad_output is None:
                    return

                grads = cls.backward(ctx, grad_output)# calculate gradients wrt inputs
                
                if not isinstance(grads, tuple):
                    grads = (grads, )
                
                for parent, grad in zip(parents, grads):
                    if parent.requires_grad:
                        if grad is None:
                            continue
                                
                        if parent.grad is None:
                            parent.grad = np.zeros_like(parent.data)
                        parent.grad += grad# pass the gradient to the parents - backward
                        
            out._backward = _backward# this is not called yet, only assigned - will be called during Tensor.backward()
        return out



class Add(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.shape_x = x.shape
        ctx.shape_y = y.shape
        
        return x + y 

    @staticmethod
    def backward(ctx, grad_output):
        
        grad_x = unbroadcast(grad_output, ctx.shape_x)# the derivate of addition is 1, but we need to unbroadcast
        grad_y = unbroadcast(grad_output, ctx.shape_y)
        
        return grad_x, grad_y


class Subtract(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.shape_x = x.shape
        ctx.shape_y = y.shape

        return x - y

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = unbroadcast(grad_output, ctx.shape_x)
        grad_y = unbroadcast(-grad_output, ctx.shape_y) 
        return grad_x, grad_y


class Multiply(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_tens_for_backward(x, y)
        
        return x * y

    @staticmethod
    def backward(ctx, grad_output):

        x, y = ctx.saved_tensors
        
        grad_x = grad_output * y
        grad_y = grad_output * x
        
        grad_x = unbroadcast(grad_x, x.shape)
        grad_y = unbroadcast(grad_y, y.shape)
        
        return grad_x, grad_y
    

class MatMul(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_tens_for_backward(x, y)
        return x @ y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        
        
        grad_x = grad_output @ y.T
        grad_y = x.T @ grad_output
        
        return grad_x, grad_y


class ReLU(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_tens_for_backward(x)
        return np.maximum(0, x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        
        grad = grad_output.copy()
        grad[x <= 0] = 0 
        return grad
    

class Sigmoid(Function):
    @staticmethod
    def forward(ctx, x):
        pos_mask = (x >= 0)
        neg_mask = ~pos_mask
        
        z = np.zeros_like(x)
        
        z[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
        
        exp_x = np.exp(x[neg_mask])
        z[neg_mask] = exp_x / (1 + exp_x)
        
        ctx.save_tens_for_backward(z)
        return z


    @staticmethod
    def backward(ctx, grad_output):
        sigmoid_x, = ctx.saved_tensors
        grad = grad_output * sigmoid_x * (1 - sigmoid_x)
        return grad

class Sum(Function):
    @staticmethod
    def forward(ctx, x, axis=None, keepdims=False):
        ctx.input_shape = x.shape
        ctx.axis = axis
        ctx.keepdims = keepdims
        
        return np.sum(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx, grad_output):
        input_shape = ctx.input_shape
        axis = ctx.axis
        keepdims = ctx.keepdims
        
        if axis is not None and not keepdims:
            grad_output = np.expand_dims(grad_output, axis)
            
        grad = np.ones(input_shape) * grad_output
        
        return grad
    

class Pow(Function):
    @staticmethod
    def forward(ctx, x, c):
        ctx.save_tens_for_backward(x, c)
        return x ** c

    @staticmethod
    def backward(ctx, grad_output):
        x, c = ctx.saved_tensors
        
        grad_x = c * (x ** (c - 1)) * grad_output
        
        return grad_x

class Transpose(Function):
    @staticmethod
    def forward(ctx, x):
        return x.T

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.T

class Tensor:
    __array_priority__ = 1000
    def __init__(self, data, _prev=(),requires_grad=False):
        self.data = np.array(data) if not isinstance(data, np.ndarray) else data
        
        self.requires_grad = requires_grad
        self.grad = None                                                                                                                                                                                                                                                      

        self._prev = set(_prev) 

        self._backward = lambda: None
        
    
    
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data, dtype=float)

        topo_order = self._build_topological_sort()

        for node in reversed(topo_order):                                                                   
            node._backward()                                                                                                                        

    def _build_topological_sort(self):
        # simple topological sort

        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        return topo

    def __repr__(self):
        return f"[data = {self.data}, grad = {self.grad}, requires_grad = {self.requires_grad}]"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Add.apply(self, other) 
    def __radd__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Add.apply(other, self)
    

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Subtract.apply(self, other)
    
    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Subtract.apply(other, self) 

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Multiply.apply(self, other)

    def __rmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Multiply.apply(other, self)


    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return MatMul.apply(self, other)

    def __rmatmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return MatMul.apply(other, self)

    def __pow__(self, other):
        return Pow.apply(self, other)


    def sum(self, axis=None, keepdims=False):
        return Sum.apply(self, axis=axis, keepdims=keepdims)
    
    def relu(self):
        return ReLU.apply(self)
    
    def sigmoid(self):
        return Sigmoid.apply(self)
    
    @property
    def T(self):
        return Transpose.apply(self)
    

class Parameter(Tensor):
    # just a tensor that wants gradients
    # the weights and biases are this class

    def __init__(self, data):
        super().__init__(data, requires_grad=True)



class Dropout(Function):
    @staticmethod
    def forward(ctx, x, p=0.5, training=True):
        if not training:
            return x
        
        mask = np.random.binomial(1, 1-p, size=x.shape)
        
        scale = 1.0 / (1.0 - p)
        mask = mask * scale
        
        ctx.save_tens_for_backward(mask)
        
        return x * mask

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.saved_tensors:
            return grad_output
            
        mask, = ctx.saved_tensors
        return grad_output * mask
    
    
if __name__ == "__main__":


    x = Tensor(np.array([[1.0, 2.0, 3.0, 4.0]]))
    W1 = Tensor(np.random.randn(4, 2), requires_grad=True) 
    b1 = Tensor(np.random.randn(1, 2), requires_grad=True)
    W2 = Tensor(np.random.randn(2, 1), requires_grad=True)
    b2 = Tensor(np.random.randn(1, 1), requires_grad=True)

    z1 = x @ W1 + b1
    h  = z1.relu()

    y_hat = h @ W2 + b2
    