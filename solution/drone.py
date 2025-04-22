from numpy import clip

import asyncio
from solution.simulation import Simulation, Fireplace, DroneInfo
from abc import ABC
from solution.geometry import Vector
from solution.executor import DroneExecutor


class Task(ABC):
    pass


class FindFireplace(Task):
    pass


class Sleep(Task):
    pass


class GoToTask(Task):
    def __init__(self, pos: Vector):
        self.pos = pos


class GoToFireplace(GoToTask):
    def __init__(self, fireplace_pos: Vector, fireplace_index: int):
        super().__init__(fireplace_pos)
        self.fireplace_index = fireplace_index


class GoToHome(GoToTask):
    def __init__(self, home_pos):
        super().__init__(home_pos)


DISTANCE_TO_DROP = 1.2


class Drone:
    def __init__(self, id: int, sim: Simulation, swarm):
        self.id = id
        self.sim = sim
        self.swarm = swarm
        self.task: Task = FindFireplace()
        # Main params
        self.params: DroneInfo = None
        self.engines: list[float] = [0.0] * 8
        self.need_drop: bool = False

        self.have_bomb: bool = True
        self.is_dead = False

        self.my_height = 8 + self.id * 2
        self.executor = DroneExecutor(self)

    def update(self, dt: float):
        """now self.params and self.engines is actual"""
        if self.is_dead:
            return
        if self.need_drop:
            self.have_bomb = False
            self.need_drop = False
        self.solve_task(self.task, dt)

        if not self.params.is_alive:
            self.dead()

        self.log()

    def solve_task(self, task: Task, dt: float):
        if isinstance(self.task, FindFireplace):
            pos, idx = self.find_fireplace(self.swarm.fireplaces)
            if pos is not None:
                self.task = GoToFireplace(pos, idx)
            else:
                self.task = Sleep()
        if isinstance(self.task, GoToFireplace):
            self.go_to_fireplace(self.task, dt)
        if isinstance(self.task, GoToHome):
            self.go_to_home(self.task, dt)
            if self.is_at_home():
                self.task = FindFireplace()

    def go_to_fireplace(self, fireplace_task: GoToFireplace, dt: float):
        if (self.params.possition - fireplace_task.pos).replace(
            y=0
        ).length() <= DISTANCE_TO_DROP:
            self.need_drop = True
            self.task = GoToHome(self.swarm.get_home_pos(self.params.possition))
        else:
            self.go_to(fireplace_task, dt)

    def go_to_home(self, home_task: GoToHome, dt: float):
        pos = home_task.pos
        self.engines = self.executor.move_to(pos, self.my_height, 1, dt)

    def go_to(self, go_to_task: GoToTask, dt: float):
        pos = go_to_task.pos
        self.engines = self.executor.move_to(pos, self.my_height, 1, dt)

    def is_at_home(self) -> bool:
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

        # Проверяем, находится ли точка в пределах всех границ
        is_in_x = min_x <= self.params.possition.x <= max_x
        is_in_y = min_y <= self.params.possition.y <= max_y
        is_in_z = min_z <= self.params.possition.z <= max_z

        return is_in_x and is_in_y and is_in_z

    def find_fireplace(
        self, fireplaces: list[list[Fireplace, int]]
    ) -> tuple[Vector, int] | None:
        """Ищет ближайший свободный и активный камин и назначает его себе."""
        if self.is_dead:
            return None, None
        best_fp_index = -1
        min_dist = float("inf")

        for i, fp_info in enumerate(fireplaces):
            fireplace, assigned_drone_id = fp_info
            if not fireplace.active and assigned_drone_id == -1:  # Свободен и активен
                dist = (self.params.possition - fireplace.possition).length()
                if dist < min_dist:
                    min_dist = dist
                    best_fp_index = i

        if best_fp_index != -1:
            # Нашли камин, назначаем его себе
            fireplaces[best_fp_index][1] = self.params.id
            chosen_pos = fireplaces[best_fp_index][0].possition
            print(
                f"Drone: {self.params.id} (Pos: {self.params.possition}) assigned to fireplace index: {best_fp_index} at pos: {chosen_pos}, distance: {min_dist:.2f}"
            )
            return chosen_pos, best_fp_index
        else:
            # Свободных активных каминов нет
            return None, None

    def dead(self):
        self.is_dead = True
        if isinstance(self.task, GoToFireplace):
            self.swarm.fireplaces[self.task.fireplace_index][1] = -1
            print(self.swarm.fireplaces)
            print("\n\n\n\n\n")
        print(f"DRONE {self.id} IS DEAD!")

    def log(self):
        if self.params.is_alive:
            print("=" * 10 + f"DRONE: {self.params.id}" + "=" * 10)
            if (
                isinstance(self.task, GoToTask)
                and self.task is not None
                and self.params.possition is not None
            ):
                print(self.params.possition, type(self.task))
                dist = self.params.possition - self.task.pos
                print(
                    f"distance to target: {dist.length()} \t | x : {abs(dist.x)}, \t | y : {abs(dist.y)}"
                )
            print(type(self.task))
            print(self.params)
            print(self.engines)
            print("\n\n")
