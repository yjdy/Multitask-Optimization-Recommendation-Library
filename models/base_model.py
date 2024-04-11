import torch
from torch import nn

from .layers import EmbeddingLayer, MultiLayerPerceptron

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.1,**kwargs):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise.to(tensor.device)
        return noisy_tensor


class AddUniformNoise:
    def __init__(self, std=1,**kwargs):
        self.std = std

    def __call__(self, tensor):
        noise = (torch.rand_like(tensor) - 0.5) * self.std
        noisy_tensor = tensor + noise.to(tensor.device)
        return noisy_tensor

class BaseModel(nn.Module):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if kwargs.get('aug',None):
            if kwargs.get('aug_type',None) == 'uniform':
                self.aug = AddUniformNoise(**kwargs)
            else:
                self.aug = AddGaussianNoise(**kwargs)


    def get_embeding_output(self, categorical_x, numerical_x,training=True,aug=False):
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
