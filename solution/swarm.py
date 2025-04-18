from solution.simulation import Simulation
from solution.drone import Drone


class Swarm:
    def __init__(self, sim: Simulation, unit_count: int = 5):
        self.sim = sim
        self.unit_count = unit_count
        self.units = [Drone(i, self.sim, self) for i in range(self.unit_count)]
        self.fireplaces = None

    def start(self):
        self.fireplaces = [[i, -1] for i in self.sim.get_fireplaces_info()]
        while True:
            for u in self.units:
                u.update()
