import torch
from ..weighted_methods import WeightMethod
import numpy as np
from scipy.optimize import minimize
from typing import Union, List,Tuple,Dict
class CAGrad(WeightMethod):
    def __init__(self, n_tasks, device: torch.device, c=0.4, max_norm=1.0):
        super().__init__(n_tasks, device=device)
        self.c = c
        self.max_norm = max_norm

    def get_weighted_loss(
            self,
            losses,
            shared_parameters,
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
        # NOTE: we allow only shared params for now. Need to see paper for other options.
        grad_dims = []
        for param in shared_parameters:
            grad_dims.append(param.data.numel())
        grads = torch.Tensor(sum(grad_dims), self.n_tasks).to(self.device)

        for i in range(self.n_tasks):
            if i < self.n_tasks:
                losses[i].backward(retain_graph=True)
            else:
                losses[i].backward()
            self.grad2vec(shared_parameters, grads, grad_dims, i)
            # multi_task_model.zero_grad_shared_modules()
            for p in shared_parameters:
                p.grad = None

        g, GTG, w_cpu = self.cagrad(grads, alpha=self.c, rescale=1)
        self.overwrite_grad(shared_parameters, g, grad_dims)
        return GTG, w_cpu

    def backward_and_step(
            self,
            categorical_fields,
            numerical_fields,
            train_labels,
            criterion,
            **kwargs
    ) -> Tuple[torch.Tensor,Dict]:
        losses = self.forward(categorical_fields, numerical_fields, train_labels, criterion)
        GTG, w_cpu = self.get_weighted_loss(losses, self.model.share_module.parameters())
        # weighted_loss.backward()
        self.optimizer.step()
        return None, {"GTG": GTG, "weights": w_cpu}

    def cagrad(self, grads, alpha=0.5, rescale=1):
        GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
        g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

        x_start = np.ones(self.n_tasks) / self.n_tasks
        bnds = tuple((0, 1) for x in x_start)
        cons = {"type": "eq", "fun": lambda x: 1 - sum(x)}
        A = GG.numpy()
        b = x_start.copy()
        c = (alpha * g0_norm + 1e-8).item()

        def objfn(x):
            return (
                    x.reshape(1, self.n_tasks).dot(A).dot(b.reshape(self.n_tasks, 1))
                    + c
                    * np.sqrt(
                x.reshape(1, self.n_tasks).dot(A).dot(x.reshape(self.n_tasks, 1))
                + 1e-8
            )
            ).sum()

        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(grads.device)
        gw = (grads * ww.view(1, -1)).sum(1)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm + 1e-8)
        g = grads.mean(1) + lmbda * gw
        if rescale == 0:
            return g, GG.numpy(), w_cpu
        elif rescale == 1:
            return g / (1 + alpha ** 2), GG.numpy(), w_cpu
        else:
            return g / (1 + alpha), GG.numpy(), w_cpu

    @staticmethod
    def grad2vec(shared_params, grads, grad_dims, task):
        # store the gradients
        grads[:, task].fill_(0.0)
        cnt = 0
        # for mm in m.shared_modules():
        #     for p in mm.parameters():

        for param in shared_params:
            grad = param.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[: cnt + 1])
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1

    def overwrite_grad(self, shared_parameters, newgrad, grad_dims):
        newgrad = newgrad * self.n_tasks  # to match the sum loss
        cnt = 0

        # for mm in m.shared_modules():
        #     for param in mm.parameters():
        for param in shared_parameters:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[: cnt + 1])
            this_grad = newgrad[beg:en].contiguous().view(param.data.size())
            param.grad = this_grad.data.clone()
            cnt += 1

    def backward(
            self,
            losses: torch.Tensor,
            parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
            shared_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
            task_specific_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
            **kwargs,
    ):
        GTG, w = self.get_weighted_loss(losses, shared_parameters)
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        return None, {"GTG": GTG, "weights": w}  # NOTE: to align with all other weight methods