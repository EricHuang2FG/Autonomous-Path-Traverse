import torch
import pygame
from src.objects.path import Path
from src.objects.entity import Entity
from src.agent.agent import Agent
from src.env.env import Environment
from src.model.model_utils import load_model_from_file_for_eval
from src.utils.constants import WINDOW_SIZE, FPS, COLOUR_BLACK, DEVICE_CPU, DEVICE_GPU

from tensordict import TensorDict


class Simulation:
    SIMULATION_STATE_RUNNING: str = "running"
    SIMULATION_STATE_RESET: str = "reset"

    SIMULATION_TRAVERSAL_COMPLETE_DISTANCE_THRESHOLD: float = Path.PATH_WIDTH / 2

    def __init__(
        self,
        controlled_by_keyboard: bool = True,
        model_path: str = "models/model.model",
        device: str = DEVICE_CPU,
    ) -> None:
        pygame.init()

        self.window: pygame.Surface = pygame.display.set_mode(WINDOW_SIZE)
        self.clock: pygame.time.Clock = pygame.time.Clock()
        self.run_sim: bool = True
        self.simulation_state: str = Simulation.SIMULATION_STATE_RESET
        self.controlled_by_keyboard: bool = controlled_by_keyboard

        if not controlled_by_keyboard:
            self.device: str = (
                DEVICE_GPU
                if device == DEVICE_GPU and torch.cuda.is_available()
                else DEVICE_CPU
            )
            self.model_path: str = model_path
            self.agent: Agent = load_model_from_file_for_eval(
                path=model_path,
                device=self.device,
            )

        self.path: Path = Path()
        self.entity: Entity = Entity(list(self.path.get_starting_position()))

        pygame.display.set_caption("Autonomous Path Traverse")

    def draw_background(self, colour: tuple[int, int, int]):
        self.window.fill(colour)

    def check_run(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def run(self) -> None:
        while self.run_sim:
            self.clock.tick(FPS)
            self.run_sim = self.check_run()

            if self.simulation_state == Simulation.SIMULATION_STATE_RESET:
                self.path.generate_path()
                self.simulation_state = Simulation.SIMULATION_STATE_RUNNING
                self.entity.reset_entity(list(self.path.get_starting_position()))

            elif self.simulation_state == Simulation.SIMULATION_STATE_RUNNING:
                steering: float = 0.0
                throttle: float = 0.0
                if not self.controlled_by_keyboard:
                    observation: tuple[float, float, float, float] = (
                        Environment.get_observation(self.path, self.entity)
                    )
                    with torch.no_grad():
                        input_tensordict: TensorDict = TensorDict(
                            {
                                "observation": torch.tensor(
                                    observation, dtype=torch.float32, device=self.device
                                )
                            },
                            batch_size=[]
                        )
                        action_tensordict: TensorDict = self.agent.actor.actor(
                            input_tensordict
                        )
                    action: torch.Tensor = action_tensordict["action"]
                    steering = action[0].item()
                    throttle = action[1].item()
                self.entity.move(
                    controlled_by_keyboard=self.controlled_by_keyboard,
                    steering=steering,
                    throttle=throttle,
                )

                if self.path.is_traversal_complete(self.entity):
                    self.simulation_state = Simulation.SIMULATION_STATE_RESET

            self.draw_background(COLOUR_BLACK)
            self.path.draw_path(self.window)
            self.entity.draw(self.window)

            pygame.display.flip()
        pygame.quit()
