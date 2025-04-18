import asyncio
from simulation import Simulation, Fireplace
from abc import ABC
from swarm import Swarm


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
    def __init__(self, number: int, sim: Simulation, swarm: Swarm):
        self.active = True
        self.number = number
        self.executor = DroneExecutor(sim)
        self.sim = sim
        self.swarm = swarm
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

    def find_fireplace(self, fireplaces: list[tuple(Fireplace, int)]) -> int:
        """
        get list(Fireplace(pos, active), drone number which work on in)
        return number Fireplace which we choose
        """
        pass
