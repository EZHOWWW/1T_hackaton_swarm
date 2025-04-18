import asyncio
from simulation import Simulation
from abc import ABC


class Task(ABC):
    pass


class FindFireplace(Task):
    pass


class GoToTask(Task):
    def __init__(self, pos):
        self.pos = pos


class GoToFireplace(GoToTask):
    def __init__(self, fireplace_pos):
        super().__init__(fireplace_pos)


class GoToHome(GoToTask):
    def __init__(self, home_pos):
        super().__init__(home_pos)


class DroneExecutor:
    def __init__(self, number: int, sim: Simulation):
        self.number = number
        self.sim = sim
        self.cur_task = None

    def set_motors(self, motors: tuple[float]):
        self.sim.set_drone_motors(self.number, motors)


class Drone:
    def __init__(self, number: int, sim: Simulation):
        self.active = True
        self.number = number
        self.executor = DroneExecutor(sim)
        self.sim = sim
        self.task = FindFireplace()

    def start(self):
        while self.active:
            self.update()

    def update(self):
        match type(self.task):
            case FindFireplace:
                pass
            case GoToTask:
                pass
