from numpy import clip

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
            if not self.update_drones_info():
                continue
            if not self.any_drone_alive():
                break
            dt = time.time() - start
            start = time.time()
            for u in self.units:
                u.update(dt)
            self.upload_drones_info()
            time.sleep(0.1)

    def update_drones_info(self):
        info = self.sim.get_drones_info()
        if not info: return False
        for i, v in enumerate(self.units):
            v.params = info[i]
            v.engines = self.sim.last_engines[i]
        return True

    def calculate_home_squere(self):
        # TODO
        pass

    def any_drone_alive(self) -> bool:
        return any([d.params.is_alive for d in self.units])
    
    def get_home_pos(self, target: Vector) -> Vector:
        # Определяем вершины параллелепипеда
        p1 = Vector(-74, 0, 78)
        p2 = Vector(-74, 1000, 72)
        p3 = Vector(-80, 1000, 72)
        p4 = Vector(-80, 0, 78)

        # Параметры параллелепипеда
        min_x = min(p1.x, p2.x, p3.x, p4.x)
        max_x = max(p1.x, p2.x, p3.x, p4.x)
        min_y = min(p1.y, p2.y, p3.y, p4.y)
        max_y = max(p1.y, p2.y, p3.y, p4.y)
        min_z = min(p1.z, p2.z, p3.z, p4.z)
        max_z = max(p1.z, p2.z, p3.z, p4.z)

        # Проецируем точку на параллелепипед
        closest_x = clip(target.x, min_x, max_x)
        closest_y = clip(target.y, min_y, max_y)
        closest_z = clip(target.z, min_z, max_z)

        # Проверяем, находится ли проекция внутри "рамки" (боковых граней)
        # Если да, то нужно выбрать ближайшую грань
        if min_x < target.x < max_x and min_z < target.z < max_z:
            # Точка проекции внутри рамки, выбираем ближайшую грань по Y
            if abs(target.y - max_y) < abs(target.y - min_y):
                closest_y = max_y
            else:
                closest_y = min_y
        elif min_x < target.x < max_x and min_y < target.y < max_y:
            # Точка проекции между min_z и max_z, выбираем ближайшую Z грань
            if abs(target.z - max_z) < abs(target.z - min_z):
                closest_z = max_z
            else:
                closest_z = min_z
        elif min_z < target.z < max_z and min_y < target.y < max_y:
            # Точка проекции между min_x и max_x, выбираем ближайшую X грань
            if abs(target.x - max_x) < abs(target.x - min_x):
                closest_x = max_x
            else:
                closest_x = min_x

        return Vector(closest_x, closest_y, closest_z)

    def upload_drones_info(self):
        eng_drones = [v.engines for i, v in enumerate(self.units)]
        drop_drones = [v.need_drop for i, v in enumerate(self.units)]

        self.sim.set_drones(eng_drones, drop_drones)
