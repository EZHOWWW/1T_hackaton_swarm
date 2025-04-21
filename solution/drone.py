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
    def __init__(self, fireplace_pos: Vector):
        super().__init__(fireplace_pos)


class GoToHome(GoToTask):
    def __init__(self, home_pos):
        super().__init__(home_pos)


DISTANCE_TO_DROP = 1.7


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

        # self.my_height = self.id * 3 + 14
        self.my_height = 12
        self.executor = DroneExecutor(self)

    def update(self, dt: float):
        """now self.params and self.engines is actual"""
        if self.need_drop:
            self.need_drop = False
        self.solve_task(self.task, dt)
        self.log()

    def solve_task(self, task: Task, dt: float):
        if isinstance(self.task, FindFireplace):
            pos = self.find_fireplace(self.swarm.fireplaces)
            if pos is not None:
                self.task = GoToFireplace(pos)
            else:
                self.task = Sleep()
        if isinstance(self.task, GoToFireplace):
            self.go_to_fireplace(self.task, dt)
        if isinstance(self.task, GoToHome):
            self.go_to_home(self.task, dt)

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
        target_height = (
            1 if (self.params.possition - pos).length() < 13 else self.my_height
        )
        self.engines = self.executor.move_to(pos, target_height, 1, dt)

    def go_to(self, go_to_task: GoToTask, dt: float):
        pos = go_to_task.pos
        self.engines = self.executor.move_to(pos, self.my_height, 1, dt)

    def find_fireplace(self, fireplaces: list[list[Fireplace, int]]) -> Vector | None:
        """Ищет ближайший свободный и активный камин и назначает его себе."""
        # return Vector(-77, 10, 68)
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
            return chosen_pos
        else:
            # Свободных активных каминов нет
            return None

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
            print(type(self.task), self.task.pos)
            print(self.params)
            print(self.engines)
            print("\n\n")
