import torch
from .layers import EmbeddingLayer, MultiLayerPerceptron

class ESMMModel(torch.nn.Module):
    """
    """

    def __init__(self, categorical_field_dims, numerical_num, embed_dim, bottom_mlp_dims, tower_mlp_dims, task_num, dropout, gain):
        super().__init__()

        self.embedding = EmbeddingLayer(categorical_field_dims, embed_dim)
        self.numerical_layer = torch.nn.Linear(numerical_num, embed_dim)
        self.embed_output_dim = (len(categorical_field_dims)+1) * embed_dim
        self.task_num = task_num

        self.bottom = torch.nn.ModuleList([MultiLayerPerceptron(self.embed_output_dim, bottom_mlp_dims, dropout, output_layer=False) for _ in range(task_num)])
        self.tower = torch.nn.ModuleList([MultiLayerPerceptron(bottom_mlp_dims[-1], tower_mlp_dims, dropout) for _ in range(task_num)])

    def forward(self, categorical_x, numerical_x):
        """
        :param
        categorical_x: Long tensor of size ``(batch_size, categorical_field_dims)``
        numerical_x: Long tensor of size ``(batch_size, numerical_num)``
        """
        categorical_emb = self.embedding(categorical_x)
        numerical_emb = self.numerical_layer(numerical_x).unsqueeze(1)
        emb = torch.cat([categorical_emb, numerical_emb], 1).view(-1, self.embed_output_dim)
        fea = [self.bottom[i](emb) for i in range(self.task_num)]

        p_ctr, p_cvr = [torch.sigmoid(self.tower[i](fea[i]).squeeze(1)) for i in range(self.task_num)]
        results = [p_ctr, p_ctr*p_cvr, p_cvr]
        return results