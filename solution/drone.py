from numpy import clip

import asyncio
from solution.simulation import Simulation, Fireplace, DroneInfo
from abc import ABC
from solution.geometry import Vector
from solution.executor import DroneExecutor
import time


class Task(ABC):
    pass


class FindFireplace(Task):
    pass


class Sleep(Task):
    pass


class GoOnPoints(Task):
    def __init__(self, start_pos: Vector, points: list[Vector], current_point_index=0):
        self.start_pos = start_pos
        self.points: list[Vector] = points
        self.current_point_index = current_point_index

    def get_cur_point(self) -> Vector:
        return self.points[self.current_point_index]

    def update_cur_point(self, target: Vector):
        cur_point = self.get_cur_point()
        if self.current_point_index == len(self.points) - 1:
            return cur_point
        last_point = (
            self.start_pos
            if self.current_point_index < 1
            else self.points[self.current_point_index - 1]
        )
        sgn_x = -1 if (last_point - cur_point).x <= 0 else 1
        sgn_y = 1 if (last_point - cur_point).y <= 0 else -1
        sgn_z = -1 if (last_point - cur_point).z <= 0 else 1

        if (
            target.x * sgn_x <= cur_point.x * sgn_x
            # target.y * sgn_y > cur_point.y
            and target.z * sgn_z <= cur_point.z * sgn_z
            or (target - cur_point).length() <= 0.7
        ):
            self.current_point_index += 1


class GoToFireplaceOnPoints(GoOnPoints):
    def __init__(self, start_pos: Vector, points: list[Vector], fireplace_index: int):
        super().__init__(start_pos, points)
        self.fireplace_index = fireplace_index


class GoToHomeOnPoints(GoOnPoints):
    def __init__(self, start_pos: Vector, points: list[Vector]):
        super().__init__(start_pos, points[::-1])


class GoToTask(Task):
    def __init__(self, pos: Vector):
        self.pos = pos


class GoToFireplace(GoToTask):
    def __init__(self, fireplace_pos: Vector):
        super().__init__(fireplace_pos)


class GoToHome(GoToTask):
    def __init__(self, home_pos):
        super().__init__(home_pos)


DISTANCE_TO_DROP = 1.5


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

        # self.my_height = self.id * 3 + 14
        self.my_height = 12
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
            points, idx = self.find_fireplace_point(self.swarm.fireplaces)
            if points is not None:
                self.task = GoToFireplaceOnPoints(self.params.possition, points[1], idx)
            else:
                self.task = Sleep()
        if isinstance(self.task, GoToFireplaceOnPoints):
            self.go_to_fireplace_on_points(self.task, dt)
        if isinstance(self.task, GoToHomeOnPoints):
            self.go_to_home(self.task, dt)
            if self.is_at_home():
                self.task = FindFireplace()

    def go_to_fireplace_on_points(
        self, fireplace_task: GoToFireplaceOnPoints, dt: float
    ):
        fireplace_task.update_cur_point(self.params.possition)
        self.go_to_on_points(fireplace_task, dt)
        if fireplace_task.current_point_index == len(fireplace_task.points) - 1:
            if (self.params.possition - fireplace_task.get_cur_point()).replace(
                y=0
            ).length() < DISTANCE_TO_DROP:
                self.need_drop = True
                self.task = GoToHomeOnPoints(
                    self.params.possition,
                    [self.swarm.get_home_pos(self.params.possition)]
                    + fireplace_task.points[: len(fireplace_task.points) - 1],
                )

    def go_to_home(self, home_task: GoToHome, dt: float):
        home_task.update_cur_point(self.params.possition)
        self.go_to_on_points(home_task, dt)
        if self.is_at_home():
            self.task = FindFireplace()

    def go_to_on_points(self, go_to_task: GoOnPoints, dt: float):
        pos = go_to_task.get_cur_point()
        print(pos, go_to_task.current_point_index)
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
        is_in_y = self.params.possition.y <= 6
        is_in_z = min_z <= self.params.possition.z <= max_z

        return is_in_x and is_in_y and is_in_z

    def find_fireplace(
        self, fireplaces: list[list[Fireplace, int]]
    ) -> tuple[Vector, int]:
        """Ищет ближайший свободный и активный камин и назначает его себе."""
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
            import os

            with open("chosens.txt", "a") as f:
                f.write(
                    f"Drone: {self.params.id} (Pos: {self.params.possition}) assigned to fireplace index: {best_fp_index} at pos: {chosen_pos}, distance: {min_dist:.2f}\n"
                )

            return chosen_pos, best_fp_index
        else:
            # Свободных активных каминов нет
            return None, None

    def find_fireplace_point(
        self, fireplaces: list[list[Fireplace, int]]
    ) -> tuple[Vector, list[Vector]] | None:
        pos, idx = self.find_fireplace(fireplaces)
        if pos is not None:
            return self.swarm.get_fireplace_points(pos), idx
        return None, None

    def dead(self):
        self.is_dead = True
        with open(f"./logs/drone_dead_{self.id}_{time.time()}.txt", 'w') as f:
            if isinstance(self.task, GoToFireplace):
                self.swarm.fireplaces[self.task.fireplace_index][1] = -1
                print(self.swarm.fireplaces, file=f)
                print("\n\n\n\n\n", file=f)
                print(f"DRONE {self.id} IS DEAD!", file=f)
                print(f"{self.my_height=}", file=f)
                print(f"{self.task}", file=f)
                print(f"{self.start_pos=}", file=f)
                print(f"{self.points=}", file=f)
                print(f"{self.current_point_index=}", file=f)
                print(f"{self.params.position=}", file=f)
                print(f"Dist: {(self.params.position - self.task.get_cur_point()).length()}", file=f)
            except Exception as exc:
                pass
                print(f"{self.task.start_pos=}", file=f)
                print(f"{self.task.points=}", file=f)
                print(f"{self.task.current_point_index=}", file=f)
                print(f"{self.params.possition=}", file=f)
                print(
                    f"Dist: {(self.params.possition - self.task.get_cur_point()).length()}",
                    file=f,
                )

    def log(self):
        pass
        # if self.params.is_alive:
        #     print("=" * 10 + f"DRONE: {self.params.id}" + "=" * 10)
        #     if (
        #         isinstance(self.task, GoToTask)
        #         and self.task is not None
        #         and self.params.possition is not None
        #     ):
        #         print(self.params.possition, type(self.task))
        #         dist = self.params.possition - self.task.pos
        #         print(
        #             f"distance to target: {dist.length()} \t | x : {abs(dist.x)}, \t | y : {abs(dist.y)}"
        #         )
        #     if isinstance(self.task, GoOnPoints):
        #         print(
        #             f"{type(self.task)} task: {self.task.get_cur_point()} \t {self.task.points}"
        #         )
        #     print(self.params)
        #     print(self.engines)
        #     print("\n\n")
