import numpy as np


class Optimizer:
    def __init__(self, parameters, lr):
        self.parameters = list(parameters)
        self.lr = lr

    def zero_grad(self):
        for p in self.parameters:
            p.grad = None
            
    def step(self):
        raise NotImplementedError
    

class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay 
        self.state = {}

    def step(self):
        for p in self.parameters:
            if p.grad is None:
                continue
            
            d_p = p.grad
            
            if self.weight_decay != 0:
                d_p = d_p + (self.weight_decay * p.data)

            if self.momentum > 0:
                if p not in self.state:
                    self.state[p] = np.zeros_like(p.data)
                
                buf = self.state[p]
                buf = self.momentum * buf + d_p
                self.state[p] = buf
                d_p = buf

            p.data -= self.lr * d_p



class AdamW(Optimizer):
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        self.state = {}
        
    

    def step(self):
        for p in self.parameters:
            if p.grad is None:
                continue

            if p not in self.state:
                self.state[p] = {
                    'step': 0,
                    'exp_avg': np.zeros_like(p.data),
                    'exp_avg_sq': np.zeros_like(p.data)
                }

            state = self.state[p]
            state['step'] += 1
            
            grad = p.grad
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            beta1, beta2 = self.beta1, self.beta2

            
            p.data -= self.lr * self.weight_decay * p.data
            exp_avg[:] = beta1 * exp_avg + (1 - beta1) * grad

            exp_avg_sq[:] = beta2 * exp_avg_sq + (1 - beta2) * (grad ** 2)

            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']

            step_size = self.lr / bias_correction1
            
            denom = (np.sqrt(exp_avg_sq) / np.sqrt(bias_correction2)) + self.eps

            p.data -= step_size * (exp_avg / denom)