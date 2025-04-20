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
        self.altitude_pid = PIDController(
            Kp=0.2,
            Ki=0.2,
            Kd=0.5,
            setpoint=self.drone.my_height,
            output_limits=(0, 0.6),
        )
        pitch_yaw_pid_params = {
            "Kp": 0.1,
            "Ki": 0.2,
            "Kd": 0.1,
            "setpoint": 0.0,
            "output_limits": (-0.1, 0.5),
        }
        self.pitch_pid = PIDController(**pitch_yaw_pid_params)
        self.yaw_pid = PIDController(**pitch_yaw_pid_params)
        self.attitude_motor = t = 0.6
        self.engines_effects = [
            Vector(*i, z=1)
            for i in [
                (-t, t - 1),  # 0
                (-t, 1 - t),  # 1
                (t - 1, t),  # 2
                (1 - t, t),  # 3
                (t, 1 - t),  # 4
                (t, t - 1),  # 5
                (1 - t, -t),  # 6
                (t - 1, -t),
            ]
        ]  # 7
        # pitch roll
        # --- Параметры управления ---
        self.max_tilt_angle = 10.0

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
        self.lidar_mult = 100
        self.last_correction = None
        for k, v in self.lidar_effects.items():
            self.lidar_effects[k] = v.normalize()

    def get_altitude_correction(self, target_height: float, dt: float) -> list[float]:
        self.altitude_pid.setpoint = target_height
        return np.full(
            8,
            self.altitude_pid.update(self.drone.params.possition.y, dt),
            dtype=np.float64,
        )

    def get_pitch_correction(
        self, direction: Vector, target_speed: float, dt: float
    ) -> list[float]:
        cur_angle = self.drone.params.angle[0]
        val = direction.z / direction.length() * self.max_tilt_angle
        self.pitch_pid.setpoint = val
        currection = self.pitch_pid.update(cur_angle, dt)

        return np.array([1, 1, 0, 0, -1, -1, 0, 0]) * currection

    def get_yaw_correction(
        self, direction: Vector, target_speed: float, dt: float
    ) -> list[float]:
        cur_angle = self.drone.params.angle[2]
        val = -direction.x / direction.length() * self.max_tilt_angle
        self.yaw_pid.setpoint = val
        currection = self.yaw_pid.update(cur_angle, dt)

        return np.array([0, 0, 1, 1, 0, 0, -1, -1]) * currection

    def move_to_direction(
        self, direction: Vector, target_height: float, target_speed: float, dt: float
    ) -> list[float]:
        engines = np.zeros(8)
        print(direction)
        print(self.drone.params.lidars)
        lidar_correction = self.correct_from_lidars(direction, dt)
        print(direction + lidar_correction)
        engines += self.get_altitude_correction(target_height + lidar_correction.y, dt)
        engines += self.get_pitch_correction(
            direction + lidar_correction, target_speed, dt
        )
        engines += self.get_yaw_correction(
            direction + lidar_correction, target_speed, dt
        )

        engines = np.clip(engines, 0, 1)
        return engines

    def correct_from_lidars(self, direction: Vector, dt: float) -> Vector:
        correction = Vector()
        for k, v in self.drone.params.lidars.items():
            if v == 0:
                continue
            correction += self.lidar_effects[k] / v * self.lidar_mult
        return correction
