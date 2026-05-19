from src.agent.policy import Policy
from src.utils.constants import DEVICE_CPU
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, TanhNormal


class Actor:

    POLICY_MODULE_IN_KEYS: list[str] = ["observation"]
    POLICY_MODULE_OUT_KEYS: list[str] = ["loc", "scale"]
    ACTOR_OUT_KEYS: list[str] = ["action"]

    def __init__(self, device: str = DEVICE_CPU) -> None:
        self.policy_module: TensorDictModule = TensorDictModule(
            Policy().to(device),
            in_keys=Actor.POLICY_MODULE_IN_KEYS,
            out_keys=Actor.POLICY_MODULE_OUT_KEYS,
        )

        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=self.policy_module,
            in_keys=Actor.POLICY_MODULE_OUT_KEYS,
            out_keys=Actor.ACTOR_OUT_KEYS,
            distribution_class=TanhNormal,
            return_log_prob=True,
        ).to(device)
