import numpy as np
import engine
from engine import Tensor, Parameter, Function

class Module:
    def __init__(self):
        self._modules = {}
        self._parameters = {}
        self.training = True

    def train(self, mode=True):
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
    
    def eval(self):
        self.train(False)

    def forward(self, x):
        raise NotImplementedError

    def __call__(self, x, *args, **kwargs):
        return self.forward(x, *args, **kwargs)

    def parameters(self):
        # recursively find all parameters
        # this is called in the training loop the get all the tensor so that we can push their data wrt their gradients!

        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def __setattr__(self, name, value):
        # this is run when we do "self.name = value" in the Tensor init func
        # if we are setting a Parameter (tensor with gradients), save it to _parameters
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        
        super().__setattr__(name, value)
        
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # yes it is initialized transposed xd 

        init = "he"
        if init == "he":
            k = np.sqrt(2.0 / in_features)
            

            self.weight = Parameter((np.random.randn(out_features, in_features) * k).astype(np.float32))
            
            if bias:
                self.bias = Parameter(np.zeros((1, out_features)).astype(np.float32))
            else:
                self.bias = None
        elif init == "glorot":
            rng = np.random.default_rng()
            scale = np.sqrt(2.0 / (in_features + out_features))
            self.weight = Parameter((rng.standard_normal((out_features, in_features)) * scale).astype(np.float32))
            if bias:
                self.bias = Parameter(np.zeros((1, out_features)).astype(np.float32))
            else:
                self.bias = None

    def forward(self, x):

        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out

class BatchNorm1d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.weight = Parameter(np.ones((1, num_features), dtype=np.float32))
        self.bias = Parameter(np.zeros((1, num_features), dtype=np.float32))
        
        self.running_mean = np.zeros((1, num_features), dtype=np.float32)
        self.running_var = np.ones((1, num_features), dtype=np.float32)

    def forward(self, x):
        
        if self.training:
            # standardize batch 
            N = x.data.shape[0]
            
            batch_mean = x.sum(axis=0, keepdims=True) * (1.0 / N)
            
            x_centered = x - batch_mean
            batch_var = (x_centered ** 2).sum(axis=0, keepdims=True) * (1.0 / N)
            
            m = self.momentum
            self.running_mean = (1 - m) * self.running_mean + m * batch_mean.data
            self.running_var = (1 - m) * self.running_var + m * batch_var.data
            
            std = (batch_var + self.eps) ** 0.5
            x_norm = x_centered * (std ** -1.0)
            
            return x_norm * self.weight + self.bias
            
        else:
            
            r_mean = engine.Tensor(self.running_mean, requires_grad=False)
            r_var = engine.Tensor(self.running_var, requires_grad=False)
            
            x_centered = x - r_mean
            std = (r_var + self.eps) ** 0.5
            x_norm = x_centered * (std ** -1.0)
            
            return x_norm * self.weight + self.bias


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers
        for i, layer in enumerate(layers):
            self.add_module(f'layer_{i}', layer)
            
    def add_module(self, name, module):
        setattr(self, name, module)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    

class ReLU(Module):
    def forward(self, x):
        return x.relu()
    
class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()
    


class MSELoss(Module):
    def forward(self, preds, targets):

        diff = preds - targets
        return (diff ** 2).sum() * (1.0 / preds.data.shape[0])
    

class CrossEntropyFunction(Function):
    @staticmethod
    def forward(ctx, logits, targets):

        max_logits = np.max(logits, axis=1, keepdims=True)
        shifted_logits = logits - max_logits
        
        exp_logits = np.exp(shifted_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        ctx.save_tens_for_backward(probs, targets)
        
        batch_size = logits.shape[0]
        log_probs = np.log(probs + 1e-9)
        
        loss = -np.sum(targets * log_probs) / batch_size
        return loss.astype(np.float32)

    @staticmethod
    def backward(ctx, grad_output):
        probs, targets = ctx.saved_tensors
        batch_size = probs.shape[0]
        
        grad_input = (probs - targets) / batch_size
        
        return grad_input * grad_output, None


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        return engine.Dropout.apply(x, p=self.p, training=self.training)


# 2. The Module (The Wrapper)
class CrossEntropyLoss(Module):
    def forward(self, logits, targets):
        return CrossEntropyFunction.apply(logits, targets)