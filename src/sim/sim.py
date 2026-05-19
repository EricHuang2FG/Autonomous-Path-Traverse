import pygame
from src.objects.path import Path
from src.objects.entity import Entity
from src.utils.constants import WINDOW_SIZE, FPS, COLOUR_BLACK


class Simulation:
    SIMULATION_STATE_RUNNING: str = "running"
    SIMULATION_STATE_RESET: str = "reset"

    SIMULATION_TRAVERSAL_COMPLETE_DISTANCE_THRESHOLD: float = Path.PATH_WIDTH / 2

    def __init__(self) -> None:
        pygame.init()

        self.window: pygame.Surface = pygame.display.set_mode(WINDOW_SIZE)
        self.clock: pygame.time.Clock = pygame.time.Clock()
        self.run_sim: bool = True
        self.simulation_state: str = Simulation.SIMULATION_STATE_RESET

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

    def run(self, controlled_by_keyboard: bool = False) -> None:
        while self.run_sim:
            self.clock.tick(FPS)
            self.run_sim = self.check_run()

            if self.simulation_state == Simulation.SIMULATION_STATE_RESET:
                self.path.generate_path()
                self.simulation_state = Simulation.SIMULATION_STATE_RUNNING
                self.entity.reset_entity(list(self.path.get_starting_position()))

            elif self.simulation_state == Simulation.SIMULATION_STATE_RUNNING:
                self.entity.move(controlled_by_keyboard=controlled_by_keyboard)

                if self.path.is_traversal_complete(self.entity):
                    self.simulation_state = Simulation.SIMULATION_STATE_RESET

            self.draw_background(COLOUR_BLACK)
            self.path.draw_path(self.window)
            self.entity.draw(self.window)

            pygame.display.flip()
        pygame.quit()
