import math
import random
import pygame
from src.utils.constants import (
    WINDOW_SIZE,
    FPS,
    COLOUR_WHITE,
    COLOUR_BLACK,
    COLOUR_RED,
    COLOUR_BLUE,
)


class Path:

    PATH_WIDTH: int = 35
    MINIMUM_DOWNWARD_PATH_SEGMENT_LENGTH: int = (
        max(WINDOW_SIZE[0] // 10, PATH_WIDTH * 2) // 2
    )
    MAXIMUM_PATH_SEGMENT_LENGTH: int = 4 * MINIMUM_DOWNWARD_PATH_SEGMENT_LENGTH

    def __init__(self) -> None:
        self.vertices: list[tuple[int, int]] = []

    def get_starting_position(self) -> tuple[int, int]:
        if not self.vertices:
            return (0, 0)
        return self.vertices[0]

    def get_ending_position(self) -> tuple[int, int]:
        if not self.vertices:
            return (0, 0)
        return self.vertices[-1]

    def generate_path(self) -> None:
        half_path_width: int = Path.PATH_WIDTH // 2

        x: int = random.randint(half_path_width, WINDOW_SIZE[0] - half_path_width)
        y: int = 0
        self.vertices.clear()
        self.vertices.append((x, y))

        while y < WINDOW_SIZE[1]:
            # randomly determine dx, bounded by the remaining distance to
            # the left and right boundaries of the screen
            dx: int = random.randint(
                -min(Path.MAXIMUM_PATH_SEGMENT_LENGTH, x - half_path_width),
                min(
                    Path.MAXIMUM_PATH_SEGMENT_LENGTH,
                    WINDOW_SIZE[0] - x - half_path_width,
                ),
            )

            # if the path is already closer to the bottom of the screen
            # than the minimum allowable path length, then simply set dy
            # to the remaining distance to prevent going off the screen.
            # Otherwise, randomly determine dy, bounded by the maximum
            # and minimum allowable lengths
            dy: int = min(
                WINDOW_SIZE[1] - y,
                random.randint(
                    Path.MINIMUM_DOWNWARD_PATH_SEGMENT_LENGTH,
                    Path.MAXIMUM_PATH_SEGMENT_LENGTH,
                ),
            )

            x += dx
            y += dy
            self.vertices.append((x, y))

    def erase_path(self, window: pygame.Surface) -> None:
        window.fill(COLOUR_BLACK)

    def draw_path(self, window: pygame.Surface) -> None:
        pygame.draw.lines(window, COLOUR_WHITE, False, self.vertices, Path.PATH_WIDTH)


class Entity:

    MAX_LINEAR_SPEED: float = 450
    MAX_LINEAR_ACCELERATION: float = 25.0
    MAX_ANGULAR_SPEED: float = 0.7
    FRICTION: float = 0.98
    TIME_STEP: float = 1 / FPS  # dt

    ENTITY_RADIUS: int = 10
    ENTITY_DIRECTION_VECTOR_WIDTH: int = 5
    ENTITY_DIRECTION_VECTOR_LENGTH: int = 30

    def __init__(self, initial_position: list[float]) -> None:
        self.position: list[float] = initial_position.copy()
        self.speed: float = 0.0
        self.direction: float = 0.0  # angle clockwise from the rightward horizontal
        self.velocity: tuple[float, float] = (0.0, 0.0)

    def set_position(self, position: list[float]) -> None:
        self.position = position

    def get_distance_to_point(
        self, target_point: tuple[float | int, float | int]
    ) -> float:
        return math.dist(self.position, target_point)

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
        for i in range(2):
            self.position[i] += self.velocity[i] * Entity.TIME_STEP

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
