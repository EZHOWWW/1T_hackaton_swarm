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

        self.my_height = 10
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

    def go_to_fireplace(self, fireplace_task: GoToFireplace, dt):
        if (self.params.possition - fireplace_task.pos).replace(
            y=0
        ).length() <= DISTANCE_TO_DROP:
            self.need_drop = True
            self.task = GoToHome(self.swarm.get_home_pos(self.params.possition))
        else:
            self.go_to(fireplace_task, dt)

    def go_to(self, go_to_task: GoToTask, dt: float):
        # direction = go_to_task.pos - self.params.possition
        # direction = self.params.possition
        direction = Vector(0.1, 0.9, 0.0).normalize()
        self.engines = self.executor.move_to_direction(direction, self.my_height, dt, 1)

    def find_fireplace(self, fireplaces: list[list[Fireplace, int]]) -> Vector | None:
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
            return chosen_pos
        else:
            # Свободных активных каминов нет
            return None

    def log(self):
        print("=" * 10 + f"DRONE: {self.params.id}" + "=" * 10)
        print(type(self.task), self.task.pos)
        print(self.params)
        print(self.engines)
        print("\n\n")
