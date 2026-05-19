import torch
import torch.nn as nn
from src.env.env import Environment


class Value(nn.Module):

    DEFAULT_HIDDEN_DIMENSION: int = 64

    def __init__(
        self,
        num_observations: int = Environment.NUM_OBSERVATIONS,
        hidden_dimension: int = DEFAULT_HIDDEN_DIMENSION,
    ) -> None:
        super().__init__()

        self.value: nn.Module = nn.Sequential(
            nn.Linear(num_observations, hidden_dimension),
            nn.Tanh(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.Tanh(),
            nn.Linear(hidden_dimension, 1),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.value(observation)
