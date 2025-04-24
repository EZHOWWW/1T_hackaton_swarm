from numpy import clip

from solution.simulation import Simulation
from solution.drone import Drone
from solution.geometry import Vector
import time


def get_points() -> dict[Vector, list[Vector]]:
    def get_point_1():
        return {
            Vector(-72.53, 0, 93.89): [  # 1 - 13 самая ближняя
                Vector(-72.9, 3.00, 84.03),
                Vector(-71.9, 3.00, 89.02),
                Vector(-71.3, 3.00, 93.89),
            ]
        }

    def get_point_2():
        return {
            Vector(-70.81, 0.00, 100.86): [  # 2 - 4 2ая ближаня
                Vector(-77.09, 8.50, 83.92),
                Vector(-74.30, 8.00, 88.25),
                Vector(-74.43, 8.50, 94.36),
                Vector(-72.83, 8.50, 98.50),
                Vector(-70.81, 8.50, 100.86),
            ]
        }

    def get_point_3():
        return {
            Vector(-69.93, 0.00, 113.52): [  # 3 - 7 3ая ближняя
                Vector(-77.02, 9.60, 83.47),
                Vector(-76.80, 9.60, 90.36),
                Vector(-75.65, 9.60, 97.36),
                Vector(-74.13, 9.60, 103.45),
                Vector(-72.81, 9.60, 108.97),
                Vector(-69.93, 9.60, 113.52),
            ]
        }

    def get_point_4():
        return {
            Vector(-60.32, 0.00, 113.95): [  # 4 - 3 4ая ближнаяя
                Vector(-71.02, 12.00, 81.74),
                Vector(-66.73, 12.00, 84.34),
                Vector(-63.47, 12.00, 89.26),
                Vector(-63.71, 12.00, 95.79),
                Vector(-63.93, 12.00, 101.28),
                Vector(-63.70, 12.00, 107.79),
                Vector(-62.32, 12.00, 113.95),
            ]
        }

    def get_point_5():
        return {
            Vector(-34.17, 0.00, 100.55):  # 5 - 0 слева нижняя
            [
                Vector(-67.93, 6.00, 82.84),
                Vector(-57.18, 6.00, 83.03),
                Vector(-50.13, 6.00, 96.99),
                Vector(-34.17, 6.00, 100.55),
            ]
        }

    def get_point_6():
        return {
            Vector(-24.3, 0.00, 99.83): [  # 6 - 1 слева верхняя
                Vector(-67.93, 9.00, 75.84),
                Vector(-57.18, 9.00, 75.03),
                Vector(-50.13, 12.00, 96.99),
                Vector(-30.13, 12.00, 96.55),
                Vector(-24.3, 12.00, 99.83),
            ]
        }

    def get_point_7():
        return {
            Vector(-30.8, 0, 48): [  # 7 -2 Под мостом на полу
                Vector(-71.14, 3.00, 68.59),
                Vector(-64.98, 3.00, 68.39),
                Vector(-56.01, 3.00, 67.57),
                Vector(-50.08, 3.00, 57.41),
                Vector(-47.10, 3.00, 50.25),
                Vector(-40.10, 3.00, 47.55),
                Vector(-30.8, 3.00, 48.34),
                Vector(-30.8, 3.00, 48),
            ]
        }

    def get_point_8():
        return {
            Vector(-26.97, 0, 54.39): [  # 8 - 11 Под мостом на каробке
                Vector(-69.63, 4.80, 68.87),
                Vector(-62.78, 4.80, 68.66),
                Vector(-52.41, 4.80, 69.68),
                Vector(-43.89, 4.80, 71.11),
                Vector(-36.27, 4.80, 71.38),
                Vector(-26.75, 4.80, 71.11),
                Vector(-14.69, 4.80, 70.90),
                Vector(-9.16, 4.80, 63.10),
                Vector(-9.16, 4.80, 51),
                Vector(-26.97, 4.90, 54.39),
            ]
        }

    def get_point_9():
        return {
            Vector(-3.26, 0, 38.55): [  # 9 - 14  у моста между коробками
                Vector(-74.13, 13, 65),
                Vector(-71.90, 13.00, 58.59),
                Vector(-49.52, 12.00, 49.84),
                Vector(-19.48, 12.00, 39.06),
                Vector(-5.56, 10.00, 38.8),
                Vector(-3.26, 10, 38.55),
            ]
        }

    def get_point_10():
        return {
            Vector(19.9, 0, 51.29): [  # 10 - 6 Под мостом далеко
                Vector(-71.90, 7, 68.59),
                Vector(-19.48, 7, 69.06),
                Vector(-10, 7, 60.06),
                Vector(18.7, 7, 51.29),
            ]
        }

    def get_point_11():
        return {
            Vector(-26.36, 0, 24.28): [  # 11 - 10
                Vector(-72.90, 14.00, 62.59),
                Vector(-71.90, 15.00, 58.59),
                Vector(-49.52, 15.00, 49.84),
                Vector(-30.48, 15.00, 39.06),
                Vector(-26.37, 13, 24.28),
            ]
        }

    def get_point_12():
        return {
            Vector(-47.44, 0, 11.34): [
                Vector(-75.55, 4.00, 70.46),
                Vector(-57.86, 4.00, 68.47),
                Vector(-54.09, 4.00, 50.52),
                Vector(-51.63, 5.00, 26.25),
                Vector(-47.44, 5, 11.34),
            ]
        }

    def get_point_13():
        return {
            Vector(-43.82, 0, 6.51): [
                Vector(-71.55, 6.00, 68.46),
                Vector(-57.86, 6.00, 68.47),
                Vector(-56.09, 6.00, 50.52),
                Vector(-55.09, 6.00, 35.52),
                Vector(-54.63, 7.00, 26.25),
                Vector(-43.82, 8, 6.51),
            ]
        }

    def get_point_14():
        return {
            Vector(41.43, 3, 66.17):  # Заглушка для точки 12
            [
                Vector(-73.56, 10.00, 80.39),
                Vector(-44.53, 12.00, 78.24),
                Vector(-12.54, 14.00, 78.60),
                Vector(15.53, 14.00, 78.40),
                Vector(40.15, 14.00, 75.73),
                Vector(41.43, 12, 66.17),
            ]
        }

    def get_point_15():
        return {
            Vector(-20.17, 0, -48.13): [
                Vector(-84.07, 13.00, 68.12),
                Vector(-85.57, 13.00, 51.01),
                Vector(-84.88, 13.00, 0.44),
                Vector(-61.04, 7.00, 1.50),
                Vector(-30.42, 7.00, -3.68),
                Vector(-21.17, 7.00, -48.13),
            ]
        }

    p = {}
    # p |= get_point_1()
    # p |= get_point_2()
    # p |= get_point_3()
    # p |= get_point_4()
    # p |= get_point_5()
    p |= get_point_6()
    # p |= get_point_7()
    # p |= get_point_8()
    # p |= get_point_9()
    # p |= get_point_10()
    # p |= get_point_11()
    # p |= get_point_12()
    # p |= get_point_13()
    # p |= get_point_14()
    # p |= get_point_15()
    return p


"""
[Fireplace(number=8, possition=Vector(-43.82, 0, 6.51), active=False), -1], 


[Fireplace(number=0, possition=Vector(-34.17, 3, 100.55), active=False), -1], 
[Fireplace(number=1, possition=Vector(-24.3, 0, 99.83), active=False), -1], 
[Fireplace(number=2, possition=Vector(-30.8, 0, 48), active=False), -1], 
[Fireplace(number=3, possition=Vector(-60.32, 0, 113.95), active=False), -1], +
[Fireplace(number=4, possition=Vector(-70.85, 0, 100.86), active=False), -1], +
[Fireplace(number=5, possition=Vector(41.43, 3, 66.17), active=False), -1], 
[Fireplace(number=6, possition=Vector(19.9, 0, 51.29), active=False), -1], 
[Fireplace(number=7, possition=Vector(-69.93, 0, 113.52), active=False), -1],  +
[Fireplace(number=8, possition=Vector(-43.82, 0, 6.51), active=False), -1], 
[Fireplace(number=9, possition=Vector(-47.44, 0, 11.34), active=False), -1], 
[Fireplace(number=10, possition=Vector(-26.37, 0, 24.28), active=False), -1], 
[Fireplace(number=11, possition=Vector(-26.97, 2.99, 54.39), active=False), -1], 
[Fireplace(number=12, possition=Vector(-20.17, 0, -48.13), active=False), -1],
 [Fireplace(number=13, possition=Vector(-72.53, 0, 93.89), active=False), -1],  +
 [Fireplace(number=14, possition=Vector(-3.26, 2, 38.55), active=False), -1]]
"""

from solution.points import get_points


class Swarm:
    def __init__(self, sim: Simulation, unit_count: int = 5):
        self.sim = sim
        self.unit_count = unit_count
        self.units = [Drone(i, self.sim, self) for i in range(self.unit_count)]
        self.fireplaces = None

        self.points = get_points()

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
                # break
            self.upload_drones_info()
            time.sleep(0.1)

    def update_drones_info(self):
        info = self.sim.get_drones_info()
        if not info:
            return False
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
        p1 = Vector(-74.5, 0, 77.6)
        p2 = Vector(-74.5, 1000, 72.5)
        p3 = Vector(-79.5, 1000, 72.5)
        p4 = Vector(-79.5, 0, 77.6)

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

        return Vector(closest_x, 5, closest_z)

    def upload_drones_info(self):
        eng_drones = [v.engines for i, v in enumerate(self.units)]
        drop_drones = [v.need_drop for i, v in enumerate(self.units)]

        self.sim.set_drones(eng_drones, drop_drones)

    def get_fireplace_points(
        self, fireplace_pos: Vector
    ) -> tuple[Vector, list[Vector]]:
        for k, v in self.points.items():
            print(k, (k - fireplace_pos).length())
        closest_key = min(
            self.points, key=lambda key: (key - fireplace_pos).replace(y=0).length()
        )
        return closest_key, self.points[closest_key]
