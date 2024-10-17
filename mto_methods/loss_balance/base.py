import torch
import torch.nn.functional as F
from typing import Union, List, Tuple, Dict

from ..weighted_methods import WeightMethod


class LinearScalarization(WeightMethod):
    """Linear scalarization baseline L = sum_j w_j * l_j where l_j is the loss for task j and w_h"""

    def __init__(
            self,
            n_tasks: int,
            model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device = 'cpu',
            task_weights: Union[List[float], torch.Tensor] = None,
    ):
        super().__init__(n_tasks, model,optimizer,device=device)
        if task_weights is None:
            task_weights = torch.ones((n_tasks,))
        if not isinstance(task_weights, torch.Tensor):
            task_weights = torch.tensor(task_weights)
        assert len(task_weights) == n_tasks
        self.task_weights = task_weights.to(device)

    def get_weighted_loss(self, losses, *args, **kwargs):
        loss = torch.sum(losses * self.task_weights)
        return loss, dict(weights=self.task_weights)

    def backward_and_step(
            self,
            categorical_fields,
            numerical_fields,
            train_labels,
            criterion,
            **kwargs
    ) -> Tuple[torch.Tensor,Dict]:

        losses = self.forward(categorical_fields,numerical_fields,train_labels,criterion)

        weighted_loss, extra_outputs = self.get_weighted_loss(losses)
        weighted_loss.backward()
        self.optimizer.step()
        return weighted_loss,extra_outputs

class ScaleInvariantLinearScalarization(WeightMethod):
    """Linear scalarization baseline L = sum_j w_j * l_j where l_j is the loss for task j and w_h"""

    def __init__(
            self,
            n_tasks: int,
            model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device = 'cpu',
            task_weights: Union[List[float], torch.Tensor] = None,
    ):
        super().__init__(n_tasks, model, optimizer, device=device)
        if task_weights is None:
            task_weights = torch.ones((n_tasks,))
        if not isinstance(task_weights, torch.Tensor):
            task_weights = torch.tensor(task_weights)
        assert len(task_weights) == n_tasks
        self.task_weights = task_weights.to(device)

    def get_weighted_loss(self, losses, *args,**kwargs):
        loss = torch.sum(torch.log(losses) * self.task_weights)
        return loss, dict(weights=self.task_weights)

class STL(WeightMethod):
    """Single task learning"""

    def __init__(self, n_tasks, device: torch.device, main_task):
        super().__init__(n_tasks, device=device)
        self.main_task = main_task
        self.weights = torch.zeros(n_tasks, device=device)
        self.weights[main_task] = 1.0

    def get_weighted_loss(self, losses: torch.Tensor, *args,**kwargs):
        assert len(losses) == self.n_tasks
        loss = losses[self.main_task]

        return loss, dict(weights=self.weights)

class RLW(WeightMethod):
    """Random loss weighting: https://arxiv.org/pdf/2111.10603.pdf"""

    def __init__(self, n_tasks, device: torch.device):
        super().__init__(n_tasks, device=device)

    def get_weighted_loss(self, losses: torch.Tensor, **kwargs):
        assert len(losses) == self.n_tasks
        weight = (F.softmax(torch.randn(self.n_tasks), dim=-1)).to(self.device)
        loss = torch.sum(losses * weight)

        return loss, dict(weights=weight)