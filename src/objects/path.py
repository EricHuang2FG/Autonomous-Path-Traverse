import math
import random
import pygame
import numpy as np
from src.objects.entity import Entity
from src.utils.constants import (
    WINDOW_SIZE,
    COLOUR_WHITE,
)


class Path:

    PATH_WIDTH: int = 35
    MINIMUM_DOWNWARD_PATH_SEGMENT_LENGTH: int = (
        max(WINDOW_SIZE[0] // 10, PATH_WIDTH * 2) // 2
    )
    MAXIMUM_PATH_SEGMENT_LENGTH: int = 4 * MINIMUM_DOWNWARD_PATH_SEGMENT_LENGTH

    PATH_TRAVERSAL_COMPLETE_DISTANCE_THRESHOLD: float = PATH_WIDTH / 2

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

    def is_traversal_complete(self, entity: Entity) -> bool:
        return (
            entity.get_distance_to_point(self.get_ending_position())
            < Path.PATH_TRAVERSAL_COMPLETE_DISTANCE_THRESHOLD
        )

    def get_segment_direction(self, segment_vector: np.ndarray) -> float:
        rightward_horizontal: np.ndarray = np.array([1, 0])
        return math.acos(
            np.clip(
                np.dot(segment_vector, rightward_horizontal)
                / (
                    np.linalg.norm(rightward_horizontal)
                    * np.linalg.norm(segment_vector)
                ),
                -1.0,
                1.0,
            )
        )

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

    def draw_path(self, window: pygame.Surface) -> None:
        pygame.draw.lines(window, COLOUR_WHITE, False, self.vertices, Path.PATH_WIDTH)
