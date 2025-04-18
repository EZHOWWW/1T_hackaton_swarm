from dataclasses import dataclass
from solution.geometry import Vector
from algorithm.PID import move, get_data, equal, concat_engines, concat_engine
from connection.SocketConnection import SocketConnection
import json


@dataclass
class Fireplace:
    number: int
    possition: Vector = Vector()
    active: bool = False


@dataclass
class DroneInfo:
    possition: Vector = Vector()
    velocity: Vector = Vector()
    angle: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )
    angular_velocity: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )
    lidars: tuple[float, ...] = tuple([0] * 10)  # len = 10
    is_alive: bool = True


class Simulation:
    def __init__(self):
        print("init")

    async def connect_to_server(self):
        self.connection = SocketConnection()
        await self.connection.set_connection()

    def get_fireplaces_info(self) -> list[Fireplace]:
        res = [
            Fireplace(i, Vector(v["x"], v["y"], v["z"]))
            for i, v in enumerate(
                json.loads(self.connection.receive_data())["firesPositions"]
            )
        ]
        return res

    def get_drone_info(self, n: int) -> DroneInfo:
        # print(self.connection.send_data("0"))
        # print(self.connection.receive_data())
        # print(self.connection.receive_data())
        print(
            self.connection.send_data(
                concat_engines(concat_engine([0 for _ in range(8)], {"id": 0}), 0)
            )
        )
        return DroneInfo()

    def set_drone(self, n: int, motors: tuple[float], drop: bool = False):
        pass

    def close_connection(self):
        print("close")
