from solution.simulation import Simulation
from solution.drone import Drone
from solution.geometry import Vector
import time


class Swarm:
    def __init__(self, sim: Simulation, unit_count: int = 5):
        self.sim = sim
        self.unit_count = unit_count
        self.units = [Drone(i, self.sim, self) for i in range(self.unit_count)]
        self.fireplaces = None

    def start(self):
        self.fireplaces = [[i, -1] for i in self.sim.get_fireplaces_info()]
        print(self.fireplaces)

        dt = 1
        start = time.time()
        while True:
            self.update_drones_info()
            dt = time.time() - start
            start = time.time()
            for u in self.units:
                u.update(dt * 10)
            self.upload_drones_info()
            time.sleep(0.1)

    def update_drones_info(self):
        info = self.sim.get_drones_info()
        for i, v in enumerate(self.units):
            v.params = info[i]
            v.engines = self.sim.last_engines[i]

    def calculate_home_squere(self):
        pass

    def get_home_pos(self, target: Vector) -> Vector:
        pass

    def upload_drones_info(self):
        eng_drones = [v.engines for i, v in enumerate(self.units)]
        drop_drones = [v.need_drop for i, v in enumerate(self.units)]

        self.sim.set_drones(eng_drones, drop_drones)
