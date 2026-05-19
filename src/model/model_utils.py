import torch

from src.agent.agent import Agent
from src.env.env import Environment
from src.utils.constants import (
    DEVICE_CPU,
    DEVICE_GPU,
    NUM_EPOCHS,
    LOSS_OBJECTIVE,
    LOSS_CRITIC,
    LOSS_ENTROPY,
    MAX_GRAD_NORM,
    SEED,
    BATCH_TERMINATED,
    BATCH_REWARD,
    BATCH_NEXT,
)

from tensordict import TensorDict
from torchrl.collectors import SyncDataCollector


def train_model(
    device: str = DEVICE_CPU, destination_path: str = "models/model.model"
) -> None:
    device = DEVICE_GPU if torch.cuda.is_available() else DEVICE_CPU

    environment: Environment = Environment(device=device, seed=SEED)
    agent: Agent = Agent(device=device)
    collector: SyncDataCollector = SyncDataCollector(
        environment,
        agent.actor.actor,
        frames_per_batch=Agent.FRAMES_PER_BATCH,
        total_frames=Agent.TOTAL_FRAMES,
    )

    i: int
    batch: TensorDict
    for i, batch in enumerate(collector):

        for _ in range(NUM_EPOCHS):
            agent.advantage_module(batch)
            data_view: TensorDict = batch.reshape(-1)
            agent.replay_buffer.extend(data_view.cpu())

            for _ in range(Agent.FRAMES_PER_BATCH // Agent.SUB_BATCH_SIZE):
                sub_data: TensorDict = agent.replay_buffer.sample(Agent.SUB_BATCH_SIZE)
                loss_values: TensorDict = agent.loss_module(sub_data.to(device))
                total_loss: torch.Tensor = (
                    loss_values[LOSS_OBJECTIVE]
                    + loss_values[LOSS_CRITIC]
                    + loss_values[LOSS_ENTROPY]
                )
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    agent.loss_module.parameters(), MAX_GRAD_NORM
                )
                agent.optimizer.step()
                agent.optimizer.zero_grad()

        mean_reward: float = batch[BATCH_NEXT, BATCH_REWARD].mean().item()
        terminated: torch.Tensor = batch[BATCH_NEXT, BATCH_TERMINATED].squeeze(-1)
        mean_episode_length: float = (
            terminated.numel() / terminated.sum().item() if terminated.any() else 0.0
        )
        print(
            f"Batch {i}/{Agent.NUM_BATCHES}, reward: {mean_reward:.4f} | mean episode length: {mean_episode_length}"
        )

    torch.save(
        {
            "actor": agent.actor.actor.state_dict(),
            "critic": agent.critic.critic.state_dict(),
        },
        destination_path,
    )
