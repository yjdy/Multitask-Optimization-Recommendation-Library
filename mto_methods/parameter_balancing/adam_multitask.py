import math
import torch
from torch.optim.optimizer import Optimizer
from typing import List, Union


class AdamMultiTask(Optimizer):
    r"""
        Implements Adam with AdaTask algorithm.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, args=None,
                 device='cpu', n_tasks=2, task_weight=[1, 1]):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamMultiTask, self).__init__(params, defaults)

        self.n_tasks = n_tasks
        self.device = device
        self.betas = betas
        self.eps = eps
        self.task_weight = torch.Tensor(task_weight).to(device)

    def set_task_weight(self, task_weights):
        self.task_weight = torch.Tensor(task_weights).to(self.device)

    def zero_grad_modules(self, modules_parameters):
        for p in modules_parameters:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def backward_and_step(self,
                          losses: torch.Tensor,
                          shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
                          task_specific_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
                          last_shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None, ):

        shared_grads = []
        if shared_parameters is not None:
            for i in range(len(losses)):
                self.zero_grad_modules(shared_parameters)
                losses[i].backward(retain_graph=True)
                grad = [p.grad.detach().clone() if (p.requires_grad is True and p.grad is not None) else None for p in
                        shared_parameters]
                shared_grads.append(grad)
        task_specific_grads = []
        if task_specific_parameters is not None:
            self.zero_grad_modules(task_specific_parameters)
            losses.sum().backward()
            task_specific_grads = [p.grad.detach().clone() if (p.requires_grad is True and p.grad is not None) else None
                                   for p in task_specific_parameters]

        return self.step(shared_parameters, task_specific_parameters, shared_grads, task_specific_grads)

    @torch.no_grad()
    def step(self, shared_parameters, task_specific_parameters, shared_grads, task_specific_grads):
        # lr
        for group in self.param_groups:
            step_lr = group['lr']
            weight_decay = group.get('weight_decay', None)

        # shared param
        for pi in range(len(shared_parameters)):
            p = shared_parameters[pi]
            state = self.state[p]
            # State initialization
            if len(state) == 0:
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

            state['step'] += 1
            beta1, beta2 = self.betas
            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

            for t in range(self.n_tasks):
                grad = shared_grads[t][pi]
                if grad is None:
                    continue
                _exp_avg = exp_avg * beta1 + (1 - beta1) * grad
                _exp_avg_sq = exp_avg_sq * beta2 + (1 - beta2) * grad.pow(2)
                denom = (_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(self.eps)
                step_size = step_lr / bias_correction1
                if weight_decay:
                    p.add_(p, alpha=-weight_decay * step_lr)
                p.addcdiv_(_exp_avg, denom, value=-self.task_weight[t] * step_size)
            grad = []
            for t in range(self.n_tasks):
                grad.append(
                    self.task_weight[t] * shared_grads[t][pi] if shared_grads[t][pi] is not None else torch.zeros_like(
                        p, memory_format=torch.preserve_format)
                )
            grad = sum(grad)
            exp_avg.mul_(beta1).add_(grad, alpha=1-beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
        # task specific param
        for pi in range(len(task_specific_parameters)):
            p = task_specific_parameters[pi]
            state = self.state[p]
            # State initialization
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

            state['step'] += 1
            beta1, beta2 = self.betas
            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']

            grad = task_specific_grads[pi]
            if grad is None:
                continue
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(self.eps)
            step_size = step_lr / bias_correction1
            if weight_decay:
                p.add_(p, alpha=-step_lr * weight_decay)
            p.addcdiv_(exp_avg, denom, value=-step_size)

        return None
