import torch
from torch import nn

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        return noisy_tensor


class AddUniformNoise:
    def __init__(self, std=1):
        self.std = std

    def __call__(self, tensor):
        noise = (torch.rand_like(tensor) - 0.5) * self.std
        noisy_tensor = tensor + noise
        return noisy_tensor

class BaseModel(nn.Module):
    """
    A pytorch implementation of MMoE Model.

    Reference:
        Ma, Jiaqi, et al. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. KDD 2018.
    """

    def __init__(self, categorical_field_dims, numerical_num, embed_dim, bottom_mlp_dims, tower_mlp_dims, task_num, expert_num, dropout):
        NotImplemented


    def get_shared_output(self, categorical_x, numerical_x,training=True,aug=False):
        categorical_emb = self.embedding(categorical_x)
        numerical_emb = self.numerical_layer(numerical_x).unsqueeze(1)
        emb = torch.cat([categorical_emb, numerical_emb], 1).view(-1, self.embed_output_dim)
        if training and aug:
            emb = self.aug(emb)
        return emb

    def forward(self, categorical_x, numerical_x):
        """
        :param
        categorical_x: Long tensor of size ``(batch_size, categorical_field_dims)``
        numerical_x: Long tensor of size ``(batch_size, numerical_num)``
        """
        NotImplemented


    def get_gate_output(self,  categorical_x, numerical_x):
        NotImplemented
