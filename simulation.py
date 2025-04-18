from dataclasses import dataclass

@dataclass
class Fireplace:
    possition: tuple[float, float, float] = (0.0, 0.0, 0.0)
    active: bool = False


@dataclass
class DroneInfo:
    possition: tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    ) 
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
    lidars: tuple[float, ...] = tuple(
        [0] * 10
    )  # len = 10
    is_alive: bool = True


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

    def set_drone(self, n: int, motors: tuple[float], drop: bool = False):
        pass
