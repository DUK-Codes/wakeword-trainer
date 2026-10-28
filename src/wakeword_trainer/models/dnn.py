"""The fully-connected classifier used by openWakeWord models.

The architecture intentionally mirrors the original openWakeWord DNN so that
exported models are drop-in compatible with existing inference code:

    flatten -> Linear -> LayerNorm -> ReLU -> [FC block] * n_blocks -> Linear -> Sigmoid

Input  is ``(batch, frames, 96)`` embedding features.
Output is ``(batch, 1)`` wake word probability.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

from ..config import ModelConfig


class FCNBlock(nn.Module):
    def __init__(self, layer_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fcn_layer = nn.Linear(layer_dim, layer_dim)
        self.layer_norm = nn.LayerNorm(layer_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.relu(self.layer_norm(self.fcn_layer(x))))


class WakeWordNet(nn.Module):
    """Small fully-connected wake word classifier."""

    def __init__(
        self,
        input_frames: int = 16,
        embedding_dim: int = 96,
        layer_dim: int = 64,
        n_blocks: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_frames = int(input_frames)
        self.embedding_dim = int(embedding_dim)
        self.layer_dim = int(layer_dim)
        self.n_blocks = int(n_blocks)

        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(self.input_frames * self.embedding_dim, self.layer_dim)
        self.layer_norm1 = nn.LayerNorm(self.layer_dim)
        self.relu1 = nn.ReLU()
        self.blocks = nn.ModuleList([FCNBlock(self.layer_dim, dropout) for _ in range(self.n_blocks)])
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.last_layer = nn.Linear(self.layer_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.squeeze(1) if x.shape[1] == 1 else x.reshape(x.shape[0], x.shape[1], -1)
        x = self.relu1(self.layer_norm1(self.layer1(self.flatten(x))))
        for block in self.blocks:
            x = block(x)
        return self.sigmoid(self.last_layer(self.dropout(x)))

    @torch.no_grad()
    def predict_proba(self, features) -> torch.Tensor:
        """Score numpy features of shape ``(frames, 96)`` or ``(batch, frames, 96)``."""
        tensor = torch.as_tensor(features, dtype=torch.float32)
        single = tensor.ndim == 2
        if single:
            tensor = tensor.unsqueeze(0)
        self.eval()
        scores = self.forward(tensor).squeeze(-1)
        return scores[0] if single else scores


def build_model(input_frames: int, config: ModelConfig | None = None, *, embedding_dim: int = 96) -> WakeWordNet:
    config = config or ModelConfig()
    return WakeWordNet(
        input_frames=input_frames,
        embedding_dim=embedding_dim,
        layer_dim=config.layer_dim,
        n_blocks=config.n_blocks,
        dropout=config.dropout,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def average_state_dicts(state_dicts: list[OrderedDict]) -> OrderedDict:
    """Average model weights (simple checkpoint ensembling)."""
    if not state_dicts:
        raise ValueError("No state dictionaries to average.")
    averaged = OrderedDict()
    for key in state_dicts[0]:
        tensors = [state[key] for state in state_dicts]
        averaged[key] = torch.stack([tensor.float() for tensor in tensors], dim=0).mean(dim=0)
    return averaged
