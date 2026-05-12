from src.sim.sim import Simulation


def main() -> None:
    simulator: Simulation = Simulation()
    simulator.run(controlled_by_keyboard=True)


if __name__ == "__main__":
    main()
