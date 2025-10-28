class NaiveSGD:
    
    def __init__(self, params, lr=0.01, weight_decay=0):
        self.defaults = dict(
            lr=lr,
            weight_decay=weight_decay
        )
        if isinstance(params, list):
            assert all(isinstance(group, dict) for group in params)
            self.param_groups = params
        else:
            self.param_groups = [{'params': list(params)}]
        
    def step(self):
        for group in self.param_groups:
            params = group['params']
            lr = group.get('lr', self.defaults['lr'])
            weight_decay = group.get('weight_decay', self.defaults['weight_decay'])
            
            for param in params:
                if param.grad is None:
                    continue
                param.data -= lr * (param.grad + weight_decay * param.data)
        
    def zero_grad(self):
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.zero_()
