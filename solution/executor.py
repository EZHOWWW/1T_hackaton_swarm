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


class DroneExecutor:
    def __init__(self, drone):
        self.drone = drone
        # --- Параметры управления ---
        self.altitude_pid = PIDController(
            Kp=0.2,
            Ki=0.6,
            Kd=0.2,
            setpoint=0,
            output_limits=(0, 0.8),
        )
        pitch_yaw_pid_params = {
            "Kp": 0.3,
            "Ki": 0.8,
            "Kd": 0.1,
            "setpoint": 0.0,
            "output_limits": (-0.5, 1.0),
        }
        self.max_tilt_angle = 35.0
        # -----------

        self.max_tilt_procent = self.max_tilt_angle / 90
        self.pitch_pid = PIDController(**pitch_yaw_pid_params)
        self.yaw_pid = PIDController(**pitch_yaw_pid_params)

    def get_pitch_correction(
        self,
        target_up: Vector,
        current_up_vector: Vector,
        target_speed: float,
        dt: float,
    ):
        forwared_vec = Vector(1, 0, 0).normalize()
        direction = target_up
        setpoint = forwared_vec.dot(direction)
        # setpoint = direction.x
        self.pitch_pid.setpoint = setpoint
        measurement = forwared_vec.dot(current_up_vector)
        # measurement = current_up_vector.x
        impact = self.pitch_pid.update(measurement, dt)

        # print(
        #     f"pitch correction: setpoint : {setpoint}, meas : {measurement}, impact : {impact} "
        # )

        return np.array([-1, -1, 0, 0, +1, +1, 0, 0]) * impact

    def get_yaw_correction(
        self,
        target_up: Vector,
        current_up_vector: Vector,
        target_speed: float,
        dt: float,
    ):
        forwared_vec = Vector(0, 0, +1).normalize()
        direction = target_up
        setpoint = forwared_vec.dot(direction)
        self.yaw_pid.setpoint = setpoint
        measurement = forwared_vec.dot(current_up_vector)
        impact = self.yaw_pid.update(measurement, dt)

        # print(
        #     f"yaw correction: setpoint : {setpoint}, meas : {measurement}, impact : {impact} "
        # )

        return np.array([0, 0, +1, +1, 0, 0, -1, -1]) * impact

    def get_altitude_correction(
        self, target_altitude: float, current_altitude: float, dt: float
    ):
        self.altitude_pid.setpoint = target_altitude
        return np.full(8, self.altitude_pid.update(current_altitude, dt))

    def move_to(
        self, target: Vector, target_height: float, target_speed: float, dt: float
    ) -> list[float]:
        direction = self.drone.params.possition - target
        direction = Vector(x=direction.z, y=0, z=direction.x).normalize()

        engines = np.zeros(8)
        direction = self.correct_direction(direction, target_speed, dt)
        print(direction)
        vector_up = self.get_up_vector()
        print(vector_up)
        engines += self.get_pitch_correction(direction, vector_up, target_speed, dt)
        engines += self.get_yaw_correction(direction, vector_up, target_speed, dt)
        engines += self.get_altitude_correction(
            target_height, self.drone.params.possition.y, dt
        )

        engines = np.clip(engines, 0, 1)
        return engines

    def apply_negativ_to_opposite(self, impact_engines: list[float]) -> list[float]:
        for i, v in enumerate(impact_engines):
            if v < 0:
                impact_engines[(i + 4) % 8] -= v
                impact_engines[i] = 0
        return impact_engines

    def correct_direction(
        self, direction: Vector, target_speed: float, dt: float
    ) -> Vector:
        direction = self.correct_direction_from_lidars(
            direction, self.drone.params.lidars, dt
        )
        direction = self.correct_gravity(direction, target_speed, dt)
        direction = self.correct_direction_from_other_drones(direction, dt)
        return direction

    def correct_direction_from_lidars(
        self, direction: Vector, lidars: dict, dt: float
    ) -> Vector:
        # TODO
        return direction

    def correct_height_from_lidars(
        self, direction: Vector, lidars: dict, dt: float
    ) -> Vector:
        # TODO
        return direction

    def correct_direction_from_other_drones(
        self, direction: Vector, dt: float
    ) -> Vector:
        # TODO
        return direction

    def correct_gravity(
        self, direction: Vector, target_speed: float, dt: float
    ) -> Vector:
        """
        Params
        ---------
        direction : Vector len = 1, y=0
        """
        direction = direction.replace(y=0)
        return direction.replace(
            y=direction.length() / self.max_tilt_procent
        ).normalize()

    def correct_height(
        target_height: float, direction: Vector, lidars: dict, df: float
    ) -> float:
        return target_height

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
        pitch_rad = np.deg2rad(rotate[2])
        yaw_rad = np.deg2rad(rotate[1])
        roll_rad = np.deg2rad(rotate[0])

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
