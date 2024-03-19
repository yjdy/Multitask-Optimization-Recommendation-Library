from ..weighted_methods import WeightMethod
import torch

from typing import List

class Uncertainty(WeightMethod):
    """Implementation of `Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics`
    Source: https://github.com/yaringal/multi-task-learning-example/blob/master/multi-task-learning-example-pytorch.ipynb
    """

    def __init__(self, n_tasks, model, optimizer, device: torch.device, lr=0.01):
        super(Uncertainty,self).__init__(n_tasks, model, optimizer,device=device)
        self.logsigma = torch.tensor([0.0] * n_tasks, device=device, requires_grad=True)
        self.optimizer.add_param_group({'param':[self.logsigma],'lr':lr})
        self.optim_lr = lr

    def get_weighted_loss(self, losses: torch.Tensor, **kwargs):
        loss = sum(
            [
                0.5 * (torch.exp(-logs) * loss + logs)
                for loss, logs in zip(losses, self.logsigma)
            ]
        )

        return loss, dict(
            weights=torch.exp(-self.logsigma)
        )  # NOTE: not exactly task weights

    def parameters(self) -> List[torch.Tensor]:
        return [self.logsigma]