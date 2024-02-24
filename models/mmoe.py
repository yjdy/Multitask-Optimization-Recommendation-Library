import torch
import torch.nn as nn
from .layers import EmbeddingLayer, MultiLayerPerceptron


class MMoEModel(torch.nn.Module):
    """
    A pytorch implementation of MMoE Model.

    Reference:
        Ma, Jiaqi, et al. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. KDD 2018.
    """

    def __init__(self, categorical_field_dims, numerical_num, embed_dim, bottom_mlp_dims, tower_mlp_dims, task_num,
                 expert_num, dropout, gain=0.1):
        super().__init__()
        self.embedding = EmbeddingLayer(categorical_field_dims, embed_dim, gain=gain)
        self.numerical_layer = torch.nn.Linear(numerical_num, embed_dim)
        self.embed_output_dim = (len(categorical_field_dims) + 1) * embed_dim
        self.task_num = task_num
        self.expert_num = expert_num

        self.expert = torch.nn.ModuleList(
            [MultiLayerPerceptron(self.embed_output_dim, bottom_mlp_dims, dropout, output_layer=False) for i in
             range(expert_num)])
        self.tower = torch.nn.ModuleList(
            [MultiLayerPerceptron(bottom_mlp_dims[-1], tower_mlp_dims, dropout) for i in range(task_num)])
        self.gate = torch.nn.ModuleList(
            [torch.nn.Sequential(torch.nn.Linear(self.embed_output_dim, expert_num), torch.nn.Softmax(dim=1)) for i in
             range(task_num)])

        self.shared_module = nn.ModuleList([self.embedding, self.expert, self.numerical_layer])
        self.task_specific_module = nn.ModuleList([self.tower, self.gate])


    def forward(self, categorical_x, numerical_x):
        """
        :param
        categorical_x: Long tensor of size ``(batch_size, categorical_field_dims)``
        numerical_x: Long tensor of size ``(batch_size, numerical_num)``
        """
        categorical_emb = self.embedding(categorical_x)
        numerical_emb = self.numerical_layer(numerical_x).unsqueeze(1)
        emb = torch.cat([categorical_emb, numerical_emb], 1).view(-1, self.embed_output_dim)
        gate_value = [self.gate[i](emb).unsqueeze(1) for i in range(self.task_num)]
        fea = torch.cat([self.expert[i](emb).unsqueeze(1) for i in range(self.expert_num)], dim=1)
        task_fea = [torch.bmm(gate_value[i], fea).squeeze(1) for i in range(self.task_num)]

        results = [torch.sigmoid(self.tower[i](task_fea[i]).squeeze(1)) for i in range(self.task_num)]
        return results

    def get_shared_output(self, categorical_x, numerical_x):
        categorical_emb = self.embedding(categorical_x)
        numerical_emb = self.numerical_layer(numerical_x).unsqueeze(1)
        emb = torch.cat([categorical_emb, numerical_emb], 1).view(-1, self.embed_output_dim)
        return emb

    def get_gate_output(self, categorical_x, numerical_x):
        emb = self.get_shared_output(categorical_x, numerical_x)
        gate_value = [self.gate[i](emb).unsqueeze(1) for i in range(self.task_num)]
        return gate_value