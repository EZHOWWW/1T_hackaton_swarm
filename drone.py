import asyncio
from simulation import Simulation


class DroneExecutor:
    def __init__(self, number: int, sim: Simulation):
        self.number = number
        self.sim = sim
        self.cur_task = None

    def set_motors(self, motors: tuple[float]):
        self.sim.set_drone_motors(self.number, motors)


class Drone:
    def __init__(self, number: int, sim: Simulation):
        self.number = number
        self.executor = DroneExecutor(sim)
        self.sim = sim
    
