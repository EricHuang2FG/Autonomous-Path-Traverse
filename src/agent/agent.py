import torch

from src.agent.actor import Actor
from src.agent.critic import Critic
from src.utils.constants import DEVICE_CPU
from torchrl.objectives.value import GAE
from torchrl.objectives import ClipPPOLoss
from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement


class Agent:

    GAMMA: float = 0.99
    LAMBDA: float = 0.95

    CLIP_EPSILON: float = 0.2
    ENTROPY_BONUS: bool = True
    ENTROPY_COEFFICIENT: float = 0.01

    LEARNING_RATE: float = 3e-4
    FRAMES_PER_BATCH: int = 8192
    TOTAL_FRAMES: int = 10000000
    NUM_BATCHES: int = TOTAL_FRAMES // FRAMES_PER_BATCH
    SUB_BATCH_SIZE: int = 64

    def __init__(self, device: str = DEVICE_CPU) -> None:
        self.actor: Actor = Actor(device=device)
        self.critic: Critic = Critic(device=device)

        self.advantage_module: GAE = GAE(
            gamma=Agent.GAMMA, lmbda=Agent.LAMBDA, value_network=self.critic.critic
        )

        self.loss_module: ClipPPOLoss = ClipPPOLoss(
            actor_network=self.actor.actor,
            critic_network=self.critic.critic,
            clip_epsilon=Agent.CLIP_EPSILON,
            entropy_bonus=Agent.ENTROPY_BONUS,
            entropy_coef=Agent.ENTROPY_COEFFICIENT,
        )

        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(
            self.loss_module.parameters(),
            lr=Agent.LEARNING_RATE,
        )

        self.replay_buffer: ReplayBuffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=Agent.FRAMES_PER_BATCH),
            sampler=SamplerWithoutReplacement(),
            batch_size=Agent.SUB_BATCH_SIZE,
        )
