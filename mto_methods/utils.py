import torch
import numpy as np

EPS=1e-8
@torch.no_grad
def get_shared_grads(loss_list, model, optimizer,format='np'):
    task_num = len(loss_list)
    grads = [[] for _ in  range(task_num)]
    for j in range(task_num):
        optimizer.zero_grad()
        task_loss = loss_list[j]
        task_loss.backward(retain_graph=True)
        for param in model.shared_module.parameters():
            if param.grad is not None:
                grads[j].append(param.grad.data.clone().detach().flatten().cpu().numpy())
            else:
                grads[j].append(torch.zeros_like(param).detach().flatten().cpu().numpy())
    if format=='np':
        return np.stack([np.concatenate(grads[i]) for i in range(task_num)])
    if format=='torch':
        return torch.stack([torch.cat(grads[i]) for i in range(task_num)])