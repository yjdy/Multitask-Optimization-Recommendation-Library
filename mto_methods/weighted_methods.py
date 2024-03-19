
from typing import List, Tuple, Union,Dict
from abc import abstractmethod

import torch
class WeightMethod:
    def __init__(self, n_tasks: int, model:torch.nn.Module, optimizer:torch.optim.optimizer, device: torch.device='cpu',**kwargs):
        super().__init__()
        self.n_tasks = n_tasks
        self.device = device
        self.optimizer = optimizer
        self.model = model

    @abstractmethod
    def get_weighted_loss(
            self,
            losses: torch.Tensor,
            shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
            task_specific_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ],
            **kwargs,
    ):
        pass

    def zero_grad_modules(self, modules_parameters):
        for p in modules_parameters:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def backward_and_step(
            self,
            categorical_fields,
            numerical_fields,
            train_labels,
            criterion,
            **kwargs
    ) -> Tuple[torch.Tensor,Dict]:
        self.optimizer.zero_grad()
        y = self.model(categorical_fields, numerical_fields)
        losses = torch.stack([criterion(y[i], train_labels[:, i].float()) for i in range(train_labels.size(1))])
        weighted_loss, extra_outputs = self.get_weighted_loss(losses,self.model.share_module.parameters(),self.model.task_specific_module.parameters())
        weighted_loss.backward()
        self.optimizer.step()
        return weighted_loss,extra_outputs


    def backward(
            self,
            losses: torch.Tensor,
            shared_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
            task_specific_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
            last_shared_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
            representation: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
            **kwargs,
    ) -> Tuple[Union[torch.Tensor, None], Union[dict, None]]:
        """

        Parameters
        ----------
        losses :
        shared_parameters :
        task_specific_parameters :
        last_shared_parameters : parameters of last shared layer/block
        representation : shared representation
        kwargs :

        Returns
        -------
        Loss, extra outputs
        """
        loss, extra_outputs = self.get_weighted_loss(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            last_shared_parameters=last_shared_parameters,
            representation=representation,
            **kwargs,
        )

        # if self.max_norm > 0:
        #     torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)

        loss.backward()
        return loss, extra_outputs

    def __call__(
            self,
            categorical_fields,
            numerical_fields,
            train_labels,
            criterion,
            **kwargs
    ):
        return self.backward_and_step(
            self,
            categorical_fields,
            numerical_fields,
            train_labels,
            criterion,
            **kwargs
        )
    def set_mode(self,mode):
        if mode=='train':
            self.model.train()
        else:
            self.model.eval()

    def parameters(self) -> List[torch.Tensor]:
        """return learnable parameters"""
        return []