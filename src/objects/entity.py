import math
import pygame
from src.utils.constants import (
    FPS,
    WINDOW_SIZE,
    COLOUR_RED,
    COLOUR_BLUE,
)


class Entity:

    MAX_LINEAR_SPEED: float = 750
    MAX_LINEAR_ACCELERATION: float = 90.0
    MAX_ANGULAR_SPEED: float = 0.95
    FRICTION: float = 0.98
    TIME_STEP: float = 1 / FPS  # dt

    ENTITY_RADIUS: int = 10
    ENTITY_DIRECTION_VECTOR_WIDTH: int = 5
    ENTITY_DIRECTION_VECTOR_LENGTH: int = 30

    def __init__(self, initial_position: list[float]) -> None:
        self.position: list[float] = initial_position.copy()
        self.speed: float = 0.0
        self.direction: float = math.pi / 2  # angle clockwise from the rightward horizontal
        self.velocity: tuple[float, float] = (0.0, 0.0)
    
    def reset_entity(self, initial_position: list[float]) -> None:
        self.set_position(initial_position)
        self.speed = 0.0
        self.direction = math.pi / 2
        self.velocity = (0.0, 0.0)

    def set_position(self, position: list[float]) -> None:
        self.position = position
        self.constrain_position()

    def get_distance_to_point(
        self, target_point: tuple[float | int, float | int]
    ) -> float:
        return math.dist(self.position, target_point)

    def constrain_position(self) -> None:
        for i, _ in enumerate(self.position):
            self.position[i] = max(0, min(WINDOW_SIZE[i], self.position[i]))

    def move(
        self, controlled_by_keyboard: bool, steering: float = 0.0, throttle: float = 0.0
    ) -> None:
        if controlled_by_keyboard:
            keys: pygame.key.ScancodeWrapper = pygame.key.get_pressed()
            if keys[pygame.K_a]:
                steering = -1.0

            if keys[pygame.K_d]:
                steering = 1.0

            if keys[pygame.K_w]:
                throttle = 1.0

            if keys[pygame.K_s]:
                throttle = -1.0

        # determine direction
        self.direction += steering * Entity.MAX_ANGULAR_SPEED * Entity.TIME_STEP
        self.direction %= 2 * math.pi # constrain angle between 0 and 2pi

        # determine and clamp speed
        self.speed += throttle * Entity.MAX_LINEAR_ACCELERATION * Entity.TIME_STEP
        self.speed *= Entity.FRICTION
        self.speed = max(
            -Entity.MAX_LINEAR_SPEED, min(Entity.MAX_LINEAR_SPEED, self.speed)
        )

        # determine velocity
        self.velocity = (
            math.cos(self.direction) * self.speed,
            math.sin(self.direction) * self.speed,
        )

        # determine position
        for i, _ in enumerate(self.position):
            self.position[i] += self.velocity[i] * Entity.TIME_STEP

        self.constrain_position()

    def draw(self, window: pygame.Surface) -> None:
        pygame.draw.circle(
            window,
            COLOUR_RED,
            tuple(int(num) for num in self.position),
            Entity.ENTITY_RADIUS,
        )

        # draw a direction vector
        if self.speed == 0:
            return

        tail: list[float] = []
        tip: list[float] = []
        for i, velocity in enumerate(self.velocity):
            unit_direction: float = velocity / self.speed
            tail.append(self.position[i] + unit_direction * Entity.ENTITY_RADIUS)
            tip.append(
                self.position[i]
                + unit_direction * Entity.ENTITY_DIRECTION_VECTOR_LENGTH
            )

        pygame.draw.line(
            window,
            COLOUR_BLUE,
            tuple(tail),
            tuple(tip),
            Entity.ENTITY_DIRECTION_VECTOR_WIDTH,
        )
