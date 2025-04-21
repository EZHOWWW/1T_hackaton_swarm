import math


class Vector:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar):
        return Vector(self.x / scalar, self.y / scalar, self.z / scalar)

    def dot(self, other):
        """Скалярное произведение"""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        """Векторное произведение"""
        return Vector(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def __mod__(self, other):
        """Скалярное произведение"""
        return dot(other)

    def __pow__(self, other):
        """Векторное произведение"""
        return cross(other)

    def length(self):
        """Длина вектора"""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self):
        """Нормализация вектора (возвращает новый вектор)"""
        mag = self.length()
        if mag == 0:
            return Vector()
        return self / mag

    def __str__(self):
        return f"Vector({self.x}, {self.y}, {self.z})"

    def __repr__(self):
        return self.__str__()

    def __neg__(self):
        return Vector(-self.x, -self.y, -self.z)

    def distance_to(self, other) -> float:
        return (other - self).length()

    def distance_xz_to(self, other) -> float:
        return Vector(other.x - self.x, 0, other.z - self.z).length()
