import torch
import torch.nn.functional as F
from typing import Union, List,Tuple

from ..weighted_methods import WeightMethod

class FAMO(WeightMethod):
    """Linear scalarization baseline L = sum_j w_j * l_j where l_j is the loss for task j and w_h"""

    def __init__(
            self,
            n_tasks: int,
            model:torch.nn.Module,
            optimizer,
            device: torch.device,
            gamma: float = 1e-5,
            w_lr: float = 0.025,
            task_weights: Union[List[float], torch.Tensor] = None,
            max_norm: float = 1.0,
    ):
        super(FAMO, self).__init__(n_tasks, model,optimizer,device=device)
        self.min_losses = torch.zeros(n_tasks).to(device)
        self.w = torch.tensor([0.0] * n_tasks, device=device, requires_grad=True)
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=gamma)
        self.max_norm = max_norm

    def backward_and_step(
            self,
            losses: torch.Tensor,
            train_data,
            train_label,
            calc_loss,
            **kwargs,
    ) -> Tuple[Union[torch.Tensor, None], Union[dict, None]]:
        loss, extra_outputs = self.get_weighted_loss(losses=losses, **kwargs)
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            train_pred = self.model(train_data, return_representation=False)
            new_losses = torch.stack(
                [calc_loss(train_pred[i], train_label[i])
                 for i in range(len(train_pred))]
            )
            self.update(new_losses.detach())

    def set_min_losses(self, losses):
        self.min_losses = losses

    def get_weighted_loss(self, losses, **kwargs):
        self.prev_loss = losses
        z = F.softmax(self.w, -1)
        D = losses - self.min_losses + 1e-8
        c = (z / D).sum().detach()
        loss = (D.log() * z / c).sum()
        return loss, {"weights": z, "logits": self.w.detach().clone()}

    def update(self, curr_loss):
        delta = (self.prev_loss - self.min_losses + 1e-8).log() - \
                (curr_loss - self.min_losses + 1e-8).log()
        with torch.enable_grad():
            d = torch.autograd.grad(F.softmax(self.w, -1),
                                    self.w,
                                    grad_outputs=delta.detach())[0]
        self.w_opt.zero_grad()
        self.w.grad = d
        self.w_opt.step()