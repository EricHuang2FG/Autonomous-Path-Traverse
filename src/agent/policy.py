import torch
import torch.nn as nn
from src.env.env import Environment


class Policy(nn.Module):

    DEFAULT_HIDDEN_DIMENSION: int = 64

    LOG_STD_RANGE: tuple[int, int] = (-5, 2)

    def __init__(
        self,
        num_observations: int = Environment.NUM_OBSERVATIONS,
        hidden_dimension: int = DEFAULT_HIDDEN_DIMENSION,
        num_actions: int = Environment.NUM_ACTIONS,
    ) -> None:
        super().__init__()

        self.policy: nn.Module = nn.Sequential(
            nn.Linear(num_observations, hidden_dimension),
            nn.Tanh(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.Tanh(),
        )

        self.mean: nn.Linear = nn.Linear(hidden_dimension, num_actions)
        self.log_std: nn.Linear = nn.Linear(hidden_dimension, num_actions)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output: torch.Tensor = self.policy(observation)
        mean: torch.Tensor = self.mean(output)
        std: torch.Tensor = self.log_std(output).clamp(*Policy.LOG_STD_RANGE).exp()
        return mean, std
