import torch
import numpy as np

EPS=1e-8

@torch.no_grad
def get_shared_adam_updates(loss_list, optimizer,format='np'):
    task_num = len(loss_list)
    updates = [[] for _ in  range(task_num)]
    for j in range(task_num):
        optimizer.zero_grad()
        task_loss = loss_list[j]
        task_loss.backward(retain_graph=True)
        for group in optimizer.param_groups:
            beta1, beta2 = group['betas']
            for param in group:
                if param.grad is not None:
                    state = optimizer.state[param]
                    step = state.get("step", torch.tensor(1., device=param.device))
                    m_t = (beta1 * state['exp_avg'] + (1-beta1)*param.grad) / (1-beta1**step)
                    v_t = (beta2 * state['exp_avt_sq'] + (1-beta2) * torch.pow(param.grad,2)) / (1-beta2**step)
                    updates[j].append(m_t/(v_t.sqrt()+EPS)).detach().flatten().cpu().numpy()
                else:
                    updates[j].append(torch.zeros_like(param).detach().flatten().cpu().numpy())
    if format=='np':
        return np.stack([np.concatenate(updates[i]) for i in range(task_num)])
    if format=='torch':
        return torch.stack([torch.cat(updates[i]) for i in range(task_num)])

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

@torch.no_grad
def get_grads(loss_list, model, optimizer,format='np'):
    task_num = len(loss_list)
    grads = [[] for _ in  range(task_num)]
    for j in range(task_num):
        optimizer.zero_grad()
        task_loss = loss_list[j]
        task_loss.backward(retain_graph=True)
        for param in model.parameters():
            if param.grad is not None:
                grads[j].append(param.grad.data.clone().detach().flatten().cpu().numpy())
            else:
                grads[j].append(torch.zeros_like(param).detach().flatten().cpu().numpy())
    if format=='np':
        return np.stack([np.concatenate(grads[i]) for i in range(task_num)])
    if format=='torch':
        return torch.stack([torch.cat(grads[i]) for i in range(task_num)])