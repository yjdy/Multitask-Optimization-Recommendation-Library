import os

import numpy as np
import torch
import random

from datasets.aliexpress import AliExpressDataset,split_dataset
from models.sharedbottom import SharedBottomModel
from models.singletask import SingleTaskModel
from models.omoe import OMoEModel
from models.mmoe import MMoEModel
from models.ple import PLEModel
from models.aitm import AITMModel

def get_dataset(name, path, ratio=0.0):
    if 'AliExpress' in name and ratio<=0:
        return AliExpressDataset(path)
    elif ratio > 0 and ratio < 1:
        return split_dataset(path, ratio)
    else:
        raise ValueError('unknown dataset name: ' + name)


def get_model(name, categorical_field_dims, numerical_num, task_num, expert_num, embed_dim):
    """
    Hyperparameters are empirically determined, not opitmized.
    """

    if name == 'sharedbottom':
        print("Model: Shared-Bottom")
        return SharedBottomModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256),
                                 tower_mlp_dims=(128, 64), task_num=task_num, dropout=0.2)
    elif name == 'singletask':
        print("Model: SingleTask")
        return SingleTaskModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256),
                               tower_mlp_dims=(128, 64), task_num=task_num, dropout=0.2)
    elif name == 'omoe':
        print("Model: OMoE")
        return OMoEModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256),
                         tower_mlp_dims=(128, 64), task_num=task_num, expert_num=expert_num, dropout=0.2)
    elif name == 'mmoe':
        print("Model: MMoE")
        return MMoEModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256),
                         tower_mlp_dims=(128, 64), task_num=task_num, expert_num=expert_num, dropout=0.2)
    elif name == 'ple':
        print("Model: PLE")
        return PLEModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256),
                        tower_mlp_dims=(128, 64), task_num=task_num, shared_expert_num=int(expert_num / 2),
                        specific_expert_num=int(expert_num / 2), dropout=0.2)
    elif name == 'aitm':
        print("Model: AITM")
        return AITMModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256),
                         tower_mlp_dims=(128, 64), task_num=task_num, dropout=0.2)
    else:
        raise ValueError('unknown model name: ' + name)


class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model.state_dict(), self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False

@torch.no_grad
def get_adam_updates(loss_list, model, optimizer:torch.optim.Adam):
    task_num = len(loss_list)
    updates = [[] for _ in range(task_num)]
    beta1, beta2 = optimizer.defaults['betas']

    for j in range(task_num):
        optimizer.zero_grad()
        task_loss = loss_list[j]
        task_loss.backward(retain_graph=True)
        for param in model.shared_module.parameters():
            if param.grad is not None:
                state = optimizer.state[param]
                step = state.get('step', torch.tensor(0., device=param.device)) + 1
                m_t = (beta1 * state('exp_avg') + (1 - beta1) * param.grad ) / (1 - beta1 ** step)
                v_t = (beta2 * state('exp_avg_sq') + (1 - beta2) * torch.pow(param.grad, 2)) / (1 - beta2 ** step)
                updates[j].append((m_t/(v_t.sqrt()+1e-8)).detach().flatten().cpu().numpy())
            else:
                updates[j].append(torch.zeros_like(param).detach().flatten().cpu().numpy())
    return np.stack([np.concatenate(updates[i]) for i in range(task_num)])

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

