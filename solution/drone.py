import asyncio
from solution.executor import DroneExecutor
from solution.simulation import Simulation, Fireplace
from abc import ABC
from solution.geometry import Vector


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


class Drone:
    def __init__(self, id: int, sim: Simulation, swarm):
        self.active = True
        self.id = id
        self.executor = DroneExecutor(self)
        self.sim = sim
        self.swarm = swarm
        self.task = FindFireplace()
        # Main params
        self.params = None
        self.engines = None
        self.need_drop = False

    def update(self):
        """now self.params and self.engines is actual"""
        print(self.params)
        print(self.engines)
        print()

        if isinstance(self.task, FindFireplace):
            pos = self.find_fireplace(self.swarm.fireplaces)
            if pos is not None:
                self.task = GoToFireplace(pos)
            else:
                self.task = Sleep()
        if isinstance(self.task, GoToFireplace):
            self.go_to_fireplace(self.task, 0.1)

    def go_to_fireplace(self, fireplace_task: GoToFireplace, dt: float):
        self.engines = self.executor.move_to(self.params, fireplace_task.pos, dt)

    def find_fireplace(self, fireplaces: list[list[Fireplace, int]]) -> Vector | None:
        """
        get list(Fireplace(pos, active), drone number which work on in)
        return number Fireplace which we choose
        """
        fireplace_id = 0
        l = [
            (self.params.possition - v[0].possition).length()
            for i, v in enumerate(fireplaces)
            if v[1] == -1
        ]
        if len(l) == 0:
            return None
        print(l)
        fireplace_id = l.index(min(l))
        fireplaces[fireplace_id][1] = self.id

        print(
            f"drone: {self.id} with pos: {self.params.possition} \nfind fireplace id: {fireplace_id} \nwith pos: {fireplaces[fireplace_id][0].possition}, \ndistance: {l[fireplace_id]}\n"
        )
        return fireplaces[fireplace_id][0].possition
