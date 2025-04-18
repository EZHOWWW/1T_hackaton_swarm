from dataclasses import dataclass


@dataclass
class Fireplace:
    possition: tuple[float] = tuple()  # len =  3
    active: bool = False


@dataclass
class DroneInfo:
    possition: tuple[float] = tuple()  # len = 3
    lidars: tuple[float] = tuple()  # len = 10
    angle: tuple[float] = tuple()  # len 3


class Simulation:
    def __init__(self):
        self.connect_to_server()
        self.fireplaces = None

    def connect_to_server(self):
        pass

    def get_fireplaces_info(self) -> list[Fireplace]:
        return []

    def get_drone_info(self, n: int) -> DroneInfo:
        return DroneInfo()

    def set_drone_motors(self, n: int, motors: tuple[float]):
        pass
