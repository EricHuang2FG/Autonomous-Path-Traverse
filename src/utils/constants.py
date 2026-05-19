# display parameters
WINDOW_SIZE: tuple[int, int] = (450, 400)
FPS: int = 60

# device types
DEVICE_CPU: str = "cpu"
DEVICE_GPU: str = "cuda"

# RGB colour definitions
COLOUR_WHITE: tuple[int, int, int] = (255, 255, 255)
COLOUR_BLACK: tuple[int, int, int] = (0, 0, 0)
COLOUR_RED: tuple[int, int, int] = (255, 0, 0)
COLOUR_BLUE: tuple[int, int, int] = (0, 0, 244)

# training parametres
NUM_EPOCHS: int = 5
MAX_GRAD_NORM: float = 1.0
SEED: int = 42

# loss parameters for training
LOSS_OBJECTIVE: str = "loss_objective"
LOSS_CRITIC: str = "loss_critic"
LOSS_ENTROPY: str = "loss_entropy"

# batch keys for displaying training information
BATCH_NEXT: str = "next"
BATCH_REWARD: str = "reward"
BATCH_TERMINATED: str = "terminated"
