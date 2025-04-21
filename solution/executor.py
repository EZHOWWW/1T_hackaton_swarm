from solution.geometry import Vector
import math
import numpy as np


class PIDController:
    """
    Простой ПИД-регулятор.
    """

    def __init__(
        self,
        Kp,
        Ki,
        Kd,
        setpoint=0,
        output_limits=(-1, 1),
        integral_limits=None,
        angle_wrap_degrees=None,
    ):
        """
        Инициализация PID контроллера.

        :param Kp: Коэффициент пропорционального усиления.
        :param Ki: Коэффициент интегрального усиления.
        :param Kd: Коэффициент дифференциального усиления.
        :param setpoint: Желаемое значение (цель).
        :param output_limits: Кортеж (min_output, max_output) для ограничения выхода.
        :param integral_limits: Кортеж (min_integral, max_integral) для ограничения интегральной суммы (Anti-windup).
                                 Если None, используется output_limits.
        :param angle_wrap_degrees: Градусы для обработки ошибки угла (например, 360 для углов 0-360, или None).
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self._setpoint = setpoint
        self.output_limits = output_limits
        self.integral_limits = (
            integral_limits if integral_limits is not None else output_limits
        )
        self.angle_wrap_degrees = (
            angle_wrap_degrees  # Для корректной обработки углов (например, рысканья)
        )

        self._integral = 0.0
        self._previous_error = 0.0
        self._previous_measurement = (
            None  # Используется для D-члена на основе измерения
        )

        # Флаг для инициализации _previous_error при первом вызове update
        self._first_run = True

    def reset(self):
        """Сбрасывает состояние PID контроллера."""
        self._integral = 0.0
        self._previous_error = 0.0
        self._previous_measurement = None
        self._first_run = True

    @property
    def setpoint(self):
        """Возвращает текущее заданное значение."""
        return self._setpoint

    @setpoint.setter
    def setpoint(self, new_setpoint):
        """Устанавливает новое заданное значение и опционально сбрасывает PID."""
        # Можно добавить логику сброса, если нужно: self.reset()
        self._setpoint = new_setpoint

    def _calculate_error(self, measurement):
        """Вычисляет ошибку, учитывая переполнение угла, если необходимо."""
        error = self._setpoint - measurement
        if self.angle_wrap_degrees:
            # Коррекция ошибки для углов (например, от -180 до +180)
            error = (
                error + self.angle_wrap_degrees / 2
            ) % self.angle_wrap_degrees - self.angle_wrap_degrees / 2
        return error

    def update(self, measurement, dt):
        """
        Вычисляет управляющее воздействие PID на основе измерения.

        :param measurement: Текущее измеренное значение (например, высота, угол).
        :param dt: Время, прошедшее с предыдущего вызова (в секундах). Должно быть > 0.
        :return: Управляющее воздействие.
        """
        if dt <= 0:
            # Возвращаем 0 или последнее значение, если dt некорректно
            print("Warning: dt <= 0 in PID update. Skipping step.")
            return 0.0  # Безопасное значение

        error = self._calculate_error(measurement)

        # --- Пропорциональный член ---
        P_term = self.Kp * error

        # --- Интегральный член (с Anti-windup) ---
        # Накапливаем интеграл
        self._integral += error * dt
        # Ограничиваем интеграл
        self._integral = max(
            self.integral_limits[0], min(self.integral_limits[1], self._integral)
        )
        I_term = self.Ki * self._integral

        # --- Дифференциальный член ---
        # Вариант 1: На основе ошибки (чувствителен к смене setpoint)
        # derivative = (error - self._previous_error) / dt

        # Вариант 2: На основе изменения измерения (менее чувствителен к смене setpoint)
        # Инициализация при первом запуске
        if self._first_run:
            self._previous_measurement = measurement
            derivative = 0.0
            self._first_run = False
        else:
            # Учитываем переполнение угла при вычислении разницы измерений
            measurement_diff = measurement - self._previous_measurement
            if self.angle_wrap_degrees:
                measurement_diff = (
                    measurement_diff + self.angle_wrap_degrees / 2
                ) % self.angle_wrap_degrees - self.angle_wrap_degrees / 2
            derivative = (
                -measurement_diff / dt
            )  # Знак минус, т.к. D = -Kd * d(measurement)/dt

        D_term = self.Kd * derivative  # Используем вариант 2

        # --- Расчет выхода и его ограничение ---
        output = P_term + I_term + D_term
        output = max(self.output_limits[0], min(self.output_limits[1], output))

        # --- Обновление для следующего шага ---
        self._previous_error = error
        self._previous_measurement = measurement

        return output

    def update_from_error(self, error, dt):
        """
        Альтернативный метод обновления, если ошибка уже вычислена
        (Полезно, если внешняя логика обрабатывает переполнение углов).
        ВНИМАНИЕ: D-член в этой реализации будет менее точным или потребует передачи d(error)/dt.
                 Рекомендуется использовать основной метод update().
        """
        if dt <= 0:
            print("Warning: dt <= 0 in PID update_from_error. Skipping step.")
            return 0.0

        # P
        P_term = self.Kp * error

        # I
        self._integral += error * dt
        self._integral = max(
            self.integral_limits[0], min(self.integral_limits[1], self._integral)
        )
        I_term = self.Ki * self._integral

        # D (на основе ошибки - может вызвать скачки при смене setpoint)
        if self._first_run:
            derivative = 0.0
            self._first_run = False
        else:
            derivative = (error - self._previous_error) / dt
        D_term = self.Kd * derivative

        # Output
        output = P_term + I_term + D_term
        output = max(self.output_limits[0], min(self.output_limits[1], output))

        # Update
        self._previous_error = error
        return output


class Engine:
    def __init__(
        self,
        name: str,
        number: int,
        impact_direction: Vector,
        pid_params: dict | None = None,
    ):
        self.name = name
        self.number = number
        if pid_params is None:
            pid_params = {
                "Kp": 0.1,
                "Ki": 0.3,
                "Kd": 0.7,
                "setpoint": 0.0,
                "integral_limits": None,
                "angle_wrap_degrees": None,
                "output_limits": (0, 1),
            }
        self.pid = PIDController(**pid_params)
        self.impact_direction = impact_direction
        self.last_impact = 0

    def impact_on_direction(
        self, target: Vector, current_up_vector: Vector, dt: float
    ) -> float:
        setpoint = self.impact_direction.dot(target)  # проекция цели на вектор мотора
        self.pid.setpoint = setpoint
        measurement = target.dot(current_up_vector)  # проекция текущего положения
        impact = self.pid.update(measurement, dt)
        return impact

    def impact_to_direction_up(
        self, target: Vector, current_up_vector: Vector, speed: float, dt: float
    ) -> float:
        """
        rotate drone to look up on target
        Parameters
            ----------
            target : Vector length = 1
            current_up_vector : Vector lenght = 1
            speed : float in [0, 1]
            df : float
        """
        setpoint = self.impact_direction.dot(target)
        self.pid.setpoint = setpoint
        measurement = self.impact_direction.dot(current_up_vector)
        result_impact = self.pid.update(measurement, dt) * speed
        print(
            f"{self.name} {self.number} | setpoint: {setpoint},\t measurement: {measurement}, impact: {result_impact}"
        )
        self.last_impact = result_impact
        return result_impact


class DroneExecutor:
    # pid for each engine
    # direction + antigrav
    # наклон изменяет угол
    def __init__(self, drone):
        self.drone = drone
        self.attitude_motor = t = 0.6
        engines_data = [
            ("fr", -t, t - 1),  # 0
            ("fl", -t, 1 - t),  # 1
            ("lf", t - 1, t),  # 2
            ("lb", 1 - t, t),  # 3
            ("bl", t, 1 - t),  # 4
            ("br", t, t - 1),  # 5
            ("rb", 1 - t, -t),  # 6
            ("rf", t - 1, -t),  # 7
        ]
        self.engines = [
            Engine(v[0], i, Vector(v[1], 1, v[2]).normalize())
            for i, v in enumerate(engines_data)
        ]

        # pitch roll
        # --- Параметры управления --
        self.max_tilt_angle = 20.0

        self.lidar_effects = {
            "f": Vector(1, 0, 0),
            "fr": Vector(1, 0, -1),
            "r": Vector(0, 0, -1),
            "br": Vector(-1, 0, -1),
            "b": Vector(-1, 0, 0),
            "bl": Vector(-1, 0, 1),
            "l": Vector(0, 0, 1),
            "fl": Vector(1, 0, 1),
            "up": Vector(0, -1, 0),
            "d": Vector(0, 1, 0),
        }
        for k, v in self.lidar_effects.items():
            self.lidar_effects[k] = v.normalize()
        self.lidar_mult = 25

    def move_to_direction(
        self,
        direction: Vector,
        target_height: float,
        dt: float,
        target_speed: float | None = None,
        compensate_gravity=True,
    ) -> list[float]:
        print(direction)
        # if compensate_gravity:
        #     direction += self.gravity_compince(direction)
        # if target_speed is not None:
        #     direction = direction.normalize() * target_speed

        # print(direction)

        impact_engines = [0] * 8
        up_vector = self.get_up_vector(self.drone.params.angle)
        print(up_vector, direction)
        for i, v in enumerate(self.engines):
            impact_engines[i] = (
                v.impact_to_direction_up(direction, up_vector, 0.5, dt) + 0.5
            )

        return np.clip(impact_engines, 0, 1)
        # return self.apply_negativ_to_opposite(impact_engines)

    def apply_negativ_to_opposite(self, impact_engines: list[float]) -> list[float]:
        for i, v in enumerate(impact_engines):
            if v < 0:
                impact_engines[(i + 4) % 8] -= v
                impact_engines[i] = 0
        return impact_engines

    def get_up_vector(self, rotate: list[float] | None = None) -> Vector:
        """
        Вычисляет единичный вектор, направленный вверх, на основе углов Эйлера дрона.

        Args:
            rotate: Список из трех углов (тангаж, рысканье, крен) в градусах.
                    Порядок: [тангаж (pitch), рысканье (yaw), крен (roll)].

        Returns:
            Единичный вектор (numpy array) [x, y, z], представляющий направление "вверх".
        """
        if rotate is None:
            rotate = self.drone.params.angle
        pitch_rad = np.deg2rad(rotate[0])
        yaw_rad = np.deg2rad(rotate[1])
        roll_rad = np.deg2rad(rotate[2])

        # Матрицы вращения вокруг осей X, Y и Z
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
                [0, np.sin(pitch_rad), np.cos(pitch_rad)],
            ]
        )

        Ry = np.array(
            [
                [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
                [0, 1, 0],
                [-np.sin(yaw_rad), 0, np.cos(yaw_rad)],
            ]
        )

        Rz = np.array(
            [
                [np.cos(roll_rad), -np.sin(roll_rad), 0],
                [np.sin(roll_rad), np.cos(roll_rad), 0],
                [0, 0, 1],
            ]
        )
        # Матрица вращения, представляющая ориентацию дрона (в мировых координатах)
        # Порядок умножения матриц важен: сначала крен, затем тангаж, затем рысканье (ZYX)
        R = Ry @ Rx @ Rz

        up_local = np.array([0, 1, 0])

        up_global = R @ up_local

        return Vector(*up_global)

    def gravity_compince(self, direction: Vector) -> Vector:
        return Vector(y=direction.length() * 1)

    def correct_from_lidars(self, direction: Vector, dt: float) -> Vector:
        correction = Vector()
        for k, v in self.drone.params.lidars.items():
            if v == 0:
                continue
            correction += self.lidar_effects[k] / v * self.lidar_mult
        return correction
