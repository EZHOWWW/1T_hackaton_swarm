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
    id: int = -1
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
    lidars: dict[str, list[float]] = None  # len = 10
    is_alive: bool = True


def parse_drone_data(i: dict) -> DroneInfo:
    def parse_vector(v: dict) -> tuple[float, float, float]:
        return (v["x"], v["y"], v["z"])

    return DroneInfo(
        i["id"],
        Vector(*parse_vector(i["droneVector"])),
        Vector(*parse_vector(i["linearVelocity"])),
        parse_vector(i["droneAxisRotation"]),
        parse_vector(i["angularVelocity"]),
        i["lidarInfo"],
        not i["isDroneCrushed"],
    )


class Simulation:
    def __init__(self):
        self.connection = SocketConnection()
        self.fireplaces = None
        self.last_drones_info = None
        self.last_engines = None
        self.can_get_drones_info = False
        print("init")

    async def connect_to_server(self):
        await self.connection.set_connection()

    def get_fireplaces_info(self) -> list[Fireplace]:
        if self.fireplaces is None:
            self.fireplaces = [
                Fireplace(i, Vector(v["x"], v["y"], v["z"]))
                for i, v in enumerate(
                    json.loads(self.connection.receive_data())["firesPositions"]
                )
            ]

        return self.fireplaces

    def get_drones_info(self) -> list[DroneInfo]:
        if self.last_drones_info is None:
            self.set_drones([[0] * 8] * 5)
        if not self.can_get_drones_info:
            self.set_drones(self.last_engines)
        res = json.loads(self.connection.receive_data()).get("dronesData", None)
        if not res:
            return []
        self.last_drones_info = [parse_drone_data(i) for i in res]
        self.can_get_drones_info = False

        return self.last_drones_info

    def set_drones(self, engines: list[list[float]], drop: list[bool] = [False] * 8):
        """engines in [0, 1]"""
        self.last_engines = engines
        eng = [
            concat_engine([e * 100 for e in v], {"id": i}, drop[i])
            for i, v in enumerate(engines)
        ]
        self.connection.send_data(concat_engines(eng, 0.1))
        self.can_get_drones_info = True

    def close_connection(self):
        print("close")  # ахха, хорош, мегахорош))
        self.connection.close_connection()
