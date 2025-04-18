import asyncio
from solution.simulation import Simulation, Fireplace
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
    def __init__(self, number: int, sim: Simulation, swarm):
        self.active = True
        self.number = number
        self.executor = DroneExecutor(self.number, sim)
        self.sim = sim
        self.swarm = swarm
        self.task = FindFireplace()
        self.self_info = None

    def start(self):
        while self.active:
            self.update()

    def update(self):
        self.self_info = self.sim.get_drone_info(self.number)
        # match type(self.task):
        #     case FindFireplace:
        #         self.find_fireplace(self.swarm.fireplaces)
        #     case GoToTask:
        #         pass

    def find_fireplace(self, fireplaces: list[list[Fireplace, int]]) -> int:
        """
        get list(Fireplace(pos, active), drone number which work on in)
        return number Fireplace which we choose
        """
        # l =
        pass
