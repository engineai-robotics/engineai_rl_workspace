from engineai_rl.utils import get_activation
import numpy as np
from .network_base import NetworkBase
import torch.nn as nn


class Mlp(NetworkBase):
    def __init__(
        self,
        num_input_dim,
        num_output_dim,
        hidden_dims=[256, 256],
        activation="elu",
        orthogonal_init=False,
        normalizer=None,
    ):
        super().__init__(num_input_dim, num_output_dim, orthogonal_init, normalizer)

        activation = get_activation(activation)

        # MLP
        mlp_trunk_layers = []
        mlp_trunk_layers.append(nn.Linear(num_input_dim, hidden_dims[0]))
        if self.orthogonal_init:
            nn.init.orthogonal_(mlp_trunk_layers[-1].weight, np.sqrt(2))
        mlp_trunk_layers.append(activation)
        for layer in range(len(hidden_dims)):
            if layer == len(hidden_dims) - 1:
                self.head = nn.Linear(hidden_dims[layer], num_output_dim)
                if self.orthogonal_init:
                    nn.init.orthogonal_(self.head.weight, 0.01)
                    nn.init.constant_(self.head.bias, 0.0)
            else:
                mlp_trunk_layers.append(
                    nn.Linear(hidden_dims[layer], hidden_dims[layer + 1])
                )
                if self.orthogonal_init:
                    nn.init.orthogonal_(mlp_trunk_layers[-1].weight, np.sqrt(2))
                    nn.init.constant_(mlp_trunk_layers[-1].bias, 0.0)
                mlp_trunk_layers.append(activation)
        self.trunk = nn.Sequential(*mlp_trunk_layers)

    def pure_forward(self, x):
        return self.head(self.trunk(x))
