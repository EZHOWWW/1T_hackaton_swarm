from simulation import Simulation
from swarm import Swarm


def main():
    sim = Simulation()
    swarm = Swarm(sim)
    swarm.start()


if __name__ == "__main__":
    main()
