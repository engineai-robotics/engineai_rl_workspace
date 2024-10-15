import torch.nn as nn
from engineai_rl.utils import get_activation


class NetworkBase(nn.Module):
    def __init__(
        self, num_input_dim, num_output_dim, orthogonal_init=False, normalizer=None
    ):
        super().__init__()
        self.orthogonal_init = orthogonal_init
        self.num_input_dim = num_input_dim
        self.num_output_dim = num_output_dim
        self.normalizer = normalizer
        if normalizer is not None and not normalizer.require_grad:
            for param in self.normalizer.parameters():
                param.requires_grad = False

    def forward(self, x):
        if self.normalizer is not None:
            x = self.normalizer(x)
        return self.pure_forward(x)

    def pure_forward(self, x):
        raise NotImplementedError
