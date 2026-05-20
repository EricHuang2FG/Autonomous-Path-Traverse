import math
import torch
import numpy as np
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data import BoundedTensorSpec, DiscreteTensorSpec, CompositeSpec
from src.objects.path import Path
from src.objects.entity import Entity
from src.utils.constants import DEVICE_CPU, WINDOW_SIZE


class Environment(EnvBase):

    DOWNWARD_PROGRESS_REWARD_SCALE: float = 0.25
    LINEAR_DEVIATION_PENALTY_SCALE: float = 0.0025
    ANGULAR_DEVIATION_PENALTY_SCALE: float = 0.005

    MAX_LINEAR_DEVIATION: float = (
        50.0  # max linear deviation for episode to be terminated
    )

    NUM_OBSERVATIONS: int = 4
    NUM_ACTIONS: int = 2

    def __init__(self, device: str = DEVICE_CPU, seed: int | None = 42) -> None:
        super().__init__(device=device)

        self.set_seed(seed)

        self.path: Path = Path()
        self.entity: Entity = Entity(list(self.path.get_starting_position()))

        # observation contains four floats, which respectively are
        # linear deviation, angular deviation, angle to next segment, and normalized speed
        self.observation_spec: CompositeSpec = CompositeSpec(
            observation=BoundedTensorSpec(
                low=torch.tensor([-torch.inf, -math.pi, -math.pi, -1.0]),
                high=torch.tensor([torch.inf, math.pi, math.pi, 1.0]),
                shape=(Environment.NUM_OBSERVATIONS,),
                dtype=torch.float32,
            ),
            device=device,
        )

        # action is [steering, throttle], each bounded between [-1, 1]
        self.action_spec: CompositeSpec = CompositeSpec(
            action=BoundedTensorSpec(
                low=-1.0,
                high=1.0,
                shape=(Environment.NUM_ACTIONS,),
                dtype=torch.float32,
            ),
            device=device,
        )

        # reward is a float
        self.reward_spec: CompositeSpec = CompositeSpec(
            reward=BoundedTensorSpec(
                low=-torch.inf, high=torch.inf, shape=(1,), dtype=torch.float32
            ),
            device=device,
        )

        # terminated is a boolean
        self.done_spec: CompositeSpec = CompositeSpec(
            terminated=DiscreteTensorSpec(n=2, shape=(1,), dtype=torch.bool),
            truncated=DiscreteTensorSpec(n=2, shape=(1,), dtype=torch.bool),
            device=device,
        )

    def _set_seed(self, seed: int | None) -> None:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

    def _reset(self, _: TensorDict) -> TensorDict:
        self.path.generate_path()
        self.entity.reset_entity(list(self.path.get_starting_position()))

        return TensorDict(
            {
                "observation": torch.tensor(
                    Environment.get_observation(self.path, self.entity),
                    dtype=torch.float32,
                )
            },
            batch_size=[],
        )

    def _step(self, tensordict: TensorDict) -> TensorDict:
        action: torch.Tensor = tensordict["action"]

        steering: float = action[0].item()
        throttle: float = action[1].item()

        prev_y_coordinate: float = self.entity.position[1]
        self.entity.move(
            controlled_by_keyboard=False, steering=steering, throttle=throttle
        )

        observation: tuple[float, float, float, float] = Environment.get_observation(
            self.path, self.entity
        )
        reward: float = self._get_reward(
            self.entity.position[1] - prev_y_coordinate, observation[0], observation[1]
        )
        terminated: bool = self._is_terminated(observation[0])

        if terminated:
            print(
                f"Progress to the bottom of the space: {((self.entity.position[1] / WINDOW_SIZE[1]) * 100):4f}%"
            )

        return TensorDict(
            {
                "observation": torch.tensor(observation, dtype=torch.float32),
                "reward": torch.tensor([reward], dtype=torch.float32),
                "terminated": torch.tensor([terminated], dtype=torch.bool),
                "truncated": torch.tensor([False], dtype=torch.bool),
            },
            batch_size=[],
        )

    @staticmethod
    def get_observation(
        path: Path, entity: Entity
    ) -> tuple[float, float, float, float]:
        """
        Gets necessary observations for the entity. Using the path segment that the entity is currently the
        closest to, it determines, 1), the signed linear deviation of position from the line segment, 2), the
        angular deviation of the direction of the entity's movement (i.e. the angle that it makes with
        the line segment), 3), the angle between the current and the next line segment, and 4), the speed
        of the entity normalized by the maximum speed.

        Args:
            None

        Returns:
            tuple[float, float, float, float]: a tuple containing the signed linear deviation, the angular deviation,
            the angle between the current and the next line segment, and the normalized speed of the entity.
        """

        position: np.ndarray = np.array(entity.position)

        # -1 represents left when moving in the downward direction
        # 1 represents right when moving in the downward direction
        min_linear_deviation_sign: int = -1
        min_linear_deviation: float = float("inf")
        angular_deviation: float = float("inf")
        angle_to_next_segment: float = 0.0

        for index, start in enumerate(path.vertices):
            if index == len(path.vertices) - 1:  # if at the last point
                break

            start: np.ndarray = np.array(start)
            end: np.ndarray = np.array(path.vertices[index + 1])

            segment_vector: np.ndarray = end - start
            segment_length: float = np.linalg.norm(segment_vector)

            if segment_length == 0:
                continue

            # to determine whether position is closest to the starting point,
            # ending point, or a point on the line, we project position on the line
            projection_normalized: float = (
                np.dot(segment_vector, position - start) / segment_length**2
            )
            if projection_normalized > 1:  # move on to the next segment
                continue

            cross_product: float = np.cross(segment_vector, position - start)
            if projection_normalized < 0:  # closer to start
                curr_linear_deviation: float = np.linalg.norm(start - position)
            else:  # distance from segment less than distant to endpoint
                curr_linear_deviation: float = (
                    np.linalg.norm(cross_product) / segment_length
                )

            # is this the segment that the entity is currently considered to be on?
            if curr_linear_deviation < min_linear_deviation:
                min_linear_deviation = curr_linear_deviation

                # determine whether position is to the left or right of the line
                # we do this by looking at the sign of the magnitude of the cross product
                min_linear_deviation_sign = -1 if cross_product < 0 else 1

                # determine the angular deviation
                # note that the segment_direction is constrained between 0 and pi
                segment_direction: float = path.get_segment_direction(segment_vector)
                # compute angular_deviation and wrap the angle such that
                # the deviation takes on [-pi, pi]
                angular_deviation = (segment_direction - entity.direction + math.pi) % (
                    2 * math.pi
                ) - math.pi

                # determine the angle between the current and the next segment
                if index != len(path.vertices) - 2:  # if there are still segments ahead
                    next_segment_vector: np.ndarray = np.array(
                        path.vertices[index + 2]
                    ) - np.array(path.vertices[index + 1])
                    angle_to_next_segment = (
                        path.get_segment_direction(next_segment_vector)
                        - segment_direction
                    )

        return (
            min_linear_deviation * min_linear_deviation_sign,
            angular_deviation,
            angle_to_next_segment,
            entity.speed / Entity.MAX_LINEAR_SPEED,
        )

    def _get_reward(
        self, delta_y: float, linear_deviation: float, angular_deviation: float
    ) -> float:
        # reward for moving downwards toward end of path (penalty otherwise)
        # penalty for linear deviation
        # penalty for angular deviation
        return (
            delta_y * Environment.DOWNWARD_PROGRESS_REWARD_SCALE
            - abs(linear_deviation) * Environment.LINEAR_DEVIATION_PENALTY_SCALE
            - abs(angular_deviation) * Environment.ANGULAR_DEVIATION_PENALTY_SCALE
        )

    def _is_terminated(self, linear_deviation: float) -> bool:
        return (
            self.path.is_traversal_complete(self.entity)
            or abs(linear_deviation) > Environment.MAX_LINEAR_DEVIATION
        )
