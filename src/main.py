from src.model.model_utils import train_model
from src.sim.sim import Simulation
from src.utils.constants import DEVICE_CPU, DEVICE_GPU


def main() -> None:
    train_model(device=DEVICE_CPU, destination_path="models/traverse.v2.model")
    # simulator: Simulation = Simulation(
    #     controlled_by_keyboard=False,
    #     model_path="models/traverse.v1.model",
    #     device=DEVICE_CPU,
    # )
    # simulator.run()


if __name__ == "__main__":
    main()
