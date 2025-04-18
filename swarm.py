from simulation import Simulation
from drone import Drone


class Swarm:
    def __init__(self, sim: Simulation, unit_count: int = 5):
        self.sim = sim
        self.unit_count = unit_count
        self.units = [Drone(self.sim) for _ in range(self.unit_count)]
        self.fireplaces = None

    def start(self):
        self.fireplaces = self.sim
