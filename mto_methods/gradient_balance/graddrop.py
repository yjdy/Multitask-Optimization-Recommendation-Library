import torch
from ..weighted_methods import WeightMethod
from typing import Union, List,Tuple,Dict

class GradDrop(WeightMethod):
    def __init__(self, n_tasks, device: torch.device, max_norm=1.0):
        super().__init__(n_tasks, device=device)
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

        P = 0.5 * (1. + grads.sum(1) / (grads.abs().sum(1) + 1e-8))
        U = torch.rand_like(grads[:, 0])
        M = P.gt(U).view(-1, 1) * grads.gt(0) + P.lt(U).view(-1, 1) * grads.lt(0)
        g = (grads * M.float()).mean(1)
        self.overwrite_grad(shared_parameters, g, grad_dims)

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

    def backward_and_step(
            self,
            categorical_fields,
            numerical_fields,
            train_labels,
            criterion,
            **kwargs
    ) -> Tuple[torch.Tensor,Dict]:
        losses = self.forward(categorical_fields, numerical_fields, train_labels, criterion)
        self.get_weighted_loss(losses, self.model.share_module.parameters())
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.share_module.parameters(), self.max_norm)
        self.optimizer.step()
        return None, None  # NOTE: to align with all other weight methods

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
        # GTG, w = self.get_weighted_loss(losses, shared_parameters)
        self.get_weighted_loss(losses, shared_parameters)
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        return None, None  # NOTE: to align with all other weight methods