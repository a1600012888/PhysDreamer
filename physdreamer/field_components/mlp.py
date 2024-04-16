"""
Mostly from nerfstudio: https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/field_components/mlp.py
"""
from typing import Optional, Set, Tuple, Union

import torch
from jaxtyping import Float
from torch import Tensor, nn


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_layers: int,
        layer_width: int,
        out_dim: Optional[int] = None,
        skip_connections: Optional[Tuple[int]] = None,
        activation: Optional[nn.Module] = nn.ReLU(),
        out_activation: Optional[nn.Module] = None,
        zero_init = False,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        assert self.in_dim > 0
        self.out_dim = out_dim if out_dim is not None else layer_width
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.skip_connections = skip_connections
        self._skip_connections: Set[int] = (
            set(skip_connections) if skip_connections else set()
        )
        self.activation = activation
        self.out_activation = out_activation
        self.net = None
        self.zero_init = zero_init

        self.build_nn_modules()

    def build_nn_modules(self) -> None:
        """Initialize multi-layer perceptron."""
        layers = []
        if self.num_layers == 1:
            layers.append(nn.Linear(self.in_dim, self.out_dim))
        else:
            for i in range(self.num_layers - 1):
                if i == 0:
                    assert (
                        i not in self._skip_connections
                    ), "Skip connection at layer 0 doesn't make sense."
                    layers.append(nn.Linear(self.in_dim, self.layer_width))
                elif i in self._skip_connections:
                    layers.append(
                        nn.Linear(self.layer_width + self.in_dim, self.layer_width)
                    )
                else:
                    layers.append(nn.Linear(self.layer_width, self.layer_width))
            layers.append(nn.Linear(self.layer_width, self.out_dim))
        self.layers = nn.ModuleList(layers)

        if self.zero_init:
            torch.nn.init.zeros_(self.layers[-1].weight)
            torch.nn.init.zeros_(self.layers[-1].bias)

    def pytorch_fwd(
        self, in_tensor: Float[Tensor, "*bs in_dim"]
    ) -> Float[Tensor, "*bs out_dim"]:
        """Process input with a multilayer perceptron.

        Args:
            in_tensor: Network input

        Returns:
            MLP network output
        """
        x = in_tensor
        for i, layer in enumerate(self.layers):
            # as checked in `build_nn_modules`, 0 should not be in `_skip_connections`
            if i in self._skip_connections:
                x = torch.cat([in_tensor, x], -1)
            x = layer(x)
            if self.activation is not None and i < len(self.layers) - 1:
                x = self.activation(x)
        if self.out_activation is not None:
            x = self.out_activation(x)
        return x

    def forward(
        self, in_tensor: Float[Tensor, "*bs in_dim"]
    ) -> Float[Tensor, "*bs out_dim"]:
        return self.pytorch_fwd(in_tensor)
