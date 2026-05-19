from src.agent.value import Value
from src.utils.constants import DEVICE_CPU
from tensordict.nn import TensorDictModule


class Critic:

    VALUE_MODULE_IN_KEYS: list[str] = ["observation"]
    VALUE_MODULE_OUT_KEYS: list[str] = ["state_value"]

    def __init__(self, device: str = DEVICE_CPU) -> None:
        self.critic: TensorDictModule = TensorDictModule(
            Value().to(device),
            in_keys=Critic.VALUE_MODULE_IN_KEYS,
            out_keys=Critic.VALUE_MODULE_OUT_KEYS,
        ).to(device)
