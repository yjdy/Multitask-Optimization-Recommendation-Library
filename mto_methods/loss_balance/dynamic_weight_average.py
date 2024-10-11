import torch
from ..weighted_methods import WeightMethod
import numpy as np

class DynamicWeightAverage(WeightMethod):
    """Dynamic Weight Average from `End-to-End Multi-Task Learning with Attention`.
    Modification of: https://github.com/lorenmt/mtan/blob/master/im2im_pred/model_segnet_split.py#L242
    """

    def __init__(
            self, n_tasks, device: torch.device, iteration_window: int = 25, temp=2.0
    ):
        """

        Parameters
        ----------
        n_tasks :
        iteration_window : 'iteration' loss is averaged over the last 'iteration_window' losses
        temp :
        """
        super().__init__(n_tasks, device=device)
        self.iteration_window = iteration_window
        self.temp = temp
        self.running_iterations = 0
        self.costs = np.ones((iteration_window * 2, n_tasks), dtype=np.float32)
        self.weights = np.ones(n_tasks, dtype=np.float32)

    def get_weighted_loss(self, losses, **kwargs):
        cost = losses.detach().cpu().numpy()

        # update costs - fifo
        self.costs[:-1, :] = self.costs[1:, :]
        self.costs[-1, :] = cost

        if self.running_iterations > self.iteration_window:
            ws = self.costs[self.iteration_window:, :].mean(0) / self.costs[
                                                                 : self.iteration_window, :
                                                                 ].mean(0)
            self.weights = (self.n_tasks * np.exp(ws / self.temp)) / (
                np.exp(ws / self.temp)
            ).sum()

        task_weights = torch.from_numpy(self.weights.astype(np.float32)).to(
            losses.device
        )
        loss = (task_weights * losses).mean()

        self.running_iterations += 1

        return loss, dict(weights=task_weights)