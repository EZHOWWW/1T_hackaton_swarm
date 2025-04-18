from simulation import Simulation
from drone import Drone


class Swarm:
    def __init__(self, sim: Simulation, unit_count: int = 5):
        self.sim = sim
        self.unit_count = unit_count
        self.units = [Drone() for _ in range(self.unit_count)]
    

    def start(self):
        pass
