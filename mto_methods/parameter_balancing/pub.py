import torch
import numpy as np
from mto_methods.weighted_methods import WeightMethod

import cvxpy as cp
from mto_methods.utils import get_shared_adam_updates
from typing import List, Tuple, Dict
from .adam_multitask import AadmMultiTask


class ParameterUpdateBalancing(WeightMethod):
    def __init__(
            self,
            n_tasks: int,
            model: torch.nn.Module,
            optimizer: AadmMultiTask,
            device: torch.device,
            max_norm: float = 1.0,
            update_weights_frequency: int = 10,
            optim_niter=20,
    ):
        super(ParameterUpdateBalancing, self).__init__(
            n_tasks=n_tasks,
            device=device,
            model=model,
            optimizer=optimizer
        )

        self.optim_niter = optim_niter
        self.update_weights_frequency = update_weights_frequency
        self.max_norm = max_norm

        self.prvs_alpha_param = None
        self.normalization_factor = np.ones((1,))
        self.init_dtd = np.eye(self.n_tasks)
        self.step = 0.0
        self.prvs_alpha = np.ones(self.n_tasks, dtype=np.float32)

    def _stop_criteria(self, dtd, alpha_t):
        return (
                (self.alpha_param.value is None)
                or (np.linalg.norm(dtd @ alpha_t - 1 / (alpha_t + 1e-10)) < 1e-3)
                or (
                        np.linalg.norm(self.alpha_param.value - self.prvs_alpha_param.value)
                        < 1e-6
                )
        )

    def solve_optimization(self, dtd: np.array):
        self.G_param.value = dtd
        self.normalization_factor_param.value = self.normalization_factor

        alpha_t = self.prvs_alpha
        for _ in range(self.optim_niter):
            self.alpha_param.value = alpha_t
            self.prvs_alpha_param.value = alpha_t

            try:
                self.prob.solve(solver=cp.QSOP, warm_start=True, max_iters=100, ignore_dpp=True)
            except:
                self.alpha_param.value = self.prvs_alpha_param.value

            if self._stop_criteria(dtd, alpha_t):
                break

            alpha_t = self.alpha_param.value

        if alpha_t is not None:
            self.prvs_alpha = alpha_t

        return self.prvs_alpha

    def _calc_phi_alpha_linearization(self):
        G_prvs_alpha = self.G_param @ self.prvs_alpha_param
        prvs_phi_tag = 1 / self.prvs_alpha_param + (1 / G_prvs_alpha) @ self.G_param
        phi_alpha = prvs_phi_tag @ (self.alpha_param - self.prvs_alpha_param)
        return phi_alpha

    def _init_optim_problem(self):
        self.alpha_param = cp.Variable(shape=(self.n_tasks,), nonneg=True)
        self.prvs_alpha_param = cp.Parameter(
            shape=(self.n_tasks,), value=self.prvs_alpha
        )
        self.G_param = cp.Parameter(
            shape=(self.n_tasks, self.n_tasks), value=self.init_dtd
        )
        self.normalization_factor_param = cp.Parameter(
            shape=(1,), value=np.array([1.0])
        )

        self.phi_alpha = self._calc_phi_alpha_linearization()

        G_alpha = self.G_param @ self.alpha_param
        constraint = []
        for i in range(self.n_tasks):
            constraint.append(
                -cp.log(self.alpha_param[i] * self.normalization_factor_param)
                - cp.log(G_alpha[i])
                <= 0
            )
        obj = cp.Minimize(
            cp.sum(G_alpha) + self.phi_alpha / self.normalization_factor_param
        )
        self.prob = cp.Problem(obj, constraint)

    def get_weighted_loss(
            self,
            losses: List,
            shared_parameters,
            *args,
            **kwargs,
    ):
        """

        Parameters
        ----------
        losses :
        shared_parameters : shared parameters
        kwargs :

        Returns
        -------

        """

        extra_outputs = dict()
        if self.step == 0:
            self._init_optim_problem()

        if (self.step % self.update_weights_frequency) == 1:  # 需要正常跑第一个step
            self.step += 1

            D = get_shared_adam_updates(losses, self.optimizer, format='torch')
            DTD = torch.mm(D, D.t())

            self.normalization_factor = (
                torch.norm(DTD).detach().cpu().numpy().reshape((1,))
            )
            DTD = DTD / self.normalization_factor.item()
            extra_outputs["DTD"] = DTD.detach().cpu().numpy()
            alpha = self.solve_optimization(DTD.cpu().detach().numpy())
            alpha = torch.from_numpy(alpha)
        else:
            self.step += 1
            alpha = self.prvs_alpha

        weighted_loss = sum([losses[i] * alpha[i] for i in range(len(alpha))])
        extra_outputs["weights"] = alpha
        return weighted_loss, extra_outputs

    def backward_and_step(
            self,
            categorical_fields,
            numerical_fields,
            train_labels,
            criterion,
            **kwargs
    ) -> Tuple[torch.Tensor, Dict]:

        losses = self.forward(categorical_fields, numerical_fields, train_labels, criterion)

        weighted_loss, extra_outputs = self.get_weighted_loss(losses, self.model.shared_module.parameters())
        self.optimizer.set_task_weight(extra_outputs["weights"])

        # weighted_loss.backward()
        self.optimizer.backward_and_step(losses, self.model.shared_module.parameters(),
                                         self.model.task_specific_module.parameters())
        return weighted_loss, extra_outputs
