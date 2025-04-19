import asyncio
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


class DroneExecutor:
    def __init__(self, drone):
        self.drone = drone
        self.engine_possition_ration = t = 0.25
        z = 1 - (t**2 + (1 - t) ** 2)  # vector len = 1
        self.engines_vector_effect = [
            Vector(t, 1 - t, z),  # 1 fr
            Vector(-t, 1 - t, z),  # 2 fl
            -Vector(-t, 1 - t, z),  # 3 br
            -Vector(t, 1 - t, z),  # 4 bl
            Vector(1 - t, t, z),  # 5 rf
            Vector(1 - t, -t, z),  # 6 rb
            -Vector(1 - t, -t, z),  # 7 lf
            -Vector(1 - t, t, z),  # 8 lb
        ]
        self.desired_speed = Vector()

    def engines_for_speed(self, desired_speed: Vector = Vector()) -> list[float]:
        from numpy import clip

        self.desired_speed = desired_speed

        engines = [0] * 8
        for i, v in enumerate(self.engines_vector_effect):
            engines[i] = self.desired_speed.dot(v)
        # if < 0 need up other engine
        return clip(engines, 0, 1)


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
            self.go_to_fireplace(self.task)

    def go_to_fireplace(self, fireplace_task: GoToFireplace):
        direction = fireplace_task.pos - self.params.possition
        direction = Vector(0, 0, 1)
        print(direction)
        self.engines = self.executor.engines_for_speed(direction)

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
