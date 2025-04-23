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
            Kp=0.1,
            Ki=0.6,
            Kd=0.2,
            setpoint=0,
            output_limits=(0, 0.7),
        )

        pitch_yaw_pid_params = {
            "Kp": 1.0,
            "Ki": 1.1,
            "Kd": 0.5,
            "setpoint": 0.0,
            "integral_limits": None,
            "output_limits": (-0.8, +0.8),
        }

        # - Максимальный угол наклона дрона (скорость) -
        self.max_tilt_angle = 20
        # TODO best params
        # -----------

        self.max_tilt_procent = self.max_tilt_angle / 90
        self.pitch_pid = PIDController(**pitch_yaw_pid_params)
        self.yaw_pid = PIDController(**pitch_yaw_pid_params)

        self._safety_radius = 3.0  # Радиус безопасности use in self.correct_direction_from_other_drones, correct_direction_from_lidars
        self._repulsion_strength = (
            10.0  # Сила отталкивания use in self.correct_direction_from_other_drones
        )

    def get_pitch_correction(
        self,
        target_up: Vector,
        current_up_vector: Vector,
        target_speed: float,
        dt: float,
    ):
        forwared_vec = self.get_forwored_vec()
        direction = target_up.normalize()
        setpoint = forwared_vec.dot(direction)
        self.pitch_pid.setpoint = setpoint
        measurement = forwared_vec.dot(current_up_vector)
        impact = self.pitch_pid.update(measurement, dt)

        return np.array([-1, -1, 0, 0, +1, +1, 0, 0]) * impact

    def get_yaw_correction(
        self,
        target_up: Vector,
        current_up_vector: Vector,
        target_speed: float,
        dt: float,
    ):
        right_vec = self.get_right_vec()
        direction = target_up.normalize()
        setpoint = right_vec.dot(direction)
        self.yaw_pid.setpoint = setpoint
        measurement = right_vec.dot(current_up_vector)
        impact = self.yaw_pid.update(measurement, dt)

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

        target_height = self.correct_height(target, None, dt)
        engines = np.zeros(8)
        direction = self.correct_direction(direction, target_speed, dt)
        vector_up = self.get_up_vector()
        engines += self.get_pitch_correction(direction, vector_up, target_speed, dt)
        engines += self.get_yaw_correction(direction, vector_up, target_speed, dt)
        engines += self.get_altitude_correction(
            target_height, self.drone.params.possition.y, dt
        )

        engines = np.clip(engines, 0, 1)
        return engines

    def correct_direction(
        self, direction: Vector, target_speed: float, dt: float
    ) -> Vector:
        # direction = self.correct_direction_from_lidars(
        #     direction, self.drone.params.lidars, dt
        # )
        # direction = self.correct_direction_from_other_drones(
        #     direction,
        #     self.drone.params.possition,
        #     [i.params.possition for i in self.drone.swarm.units],
        #     dt,
        # )
        direction = self.correct_gravity(direction, target_speed, dt)
        return direction

    def correct_height_from_lidars(
        self, direction: Vector, lidars: dict, dt: float
    ) -> Vector:
        # TODO
        return direction

    def correct_direction_from_other_drones(
        self,
        direction: Vector,
        my_possition: Vector,  # Позиция текущего дрона (можно использовать self.position)
        drones_possitions: list[Vector],
        dt: float,
        drones_speed: list[Vector] | None = None,
    ) -> Vector:
        """
        Корректирует направление для избегания столкновений с другими дронами (APF).
        """
        avoidance_vector = (
            Vector()
        )  # Используем конструктор по умолчанию для нулевого вектора
        EPSILON = 1e-9  # Малый допуск для сравнения float и избегания деления на 0

        for other_pos in drones_possitions:
            # Проверка, является ли other_pos позицией текущего дрона
            # Сравниваем координаты напрямую из-за отсутствия __eq__ и для надежности
            is_self = (
                abs(other_pos.x - my_possition.x) < EPSILON
                and abs(other_pos.y - my_possition.y) < EPSILON
                and abs(other_pos.z - my_possition.z) < EPSILON
            )
            if is_self:
                continue

            relative_pos = my_possition - other_pos
            distance = relative_pos.length()

            if distance < self._safety_radius and distance > EPSILON:
                # Сила отталкивания (увеличивается при приближении)
                repulsion_magnitude = self._repulsion_strength * (
                    self._safety_radius / distance - 1.0
                )
                # Направление отталкивания
                repulsion_direction = (
                    relative_pos.normalize()
                )  # Ваш normalize() обрабатывает нулевую длину
                avoidance_vector += repulsion_direction * repulsion_magnitude
            elif distance <= EPSILON:
                # Дроны слишком близко или в одной точке - небольшой "толчок"
                # Можно сделать его случайным или зависящим от индекса дрона, чтобы избежать симметрии
                avoidance_vector += (
                    Vector(0.1, -0.1, 0) * self._repulsion_strength * 0.1
                )  # Пример масштабированного толчка

        corrected_direction = direction + avoidance_vector

        # corrected_direction = corrected_direction.normalize()

        return corrected_direction

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
        target_height: float, target_pos: Vector, lidars: dict, df: float
    ) -> float:
        return target_pos.y

    def get_up_vector(self, rotate: list[float] | None = None) -> Vector:
        """Глобальный вектор "вверх" дрона."""

        if rotate is None:
            rotate = self.drone.params.angle

        return self._rotate_local_to_global(Vector(0, 1, 0), rotate)

    def get_forwored_vec(self, rotate: list[float] | None = None) -> Vector:
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

        forward_local = np.array([1, 0, 0])

        forward_global = R @ forward_local

        return Vector(*forward_global)

    def get_right_vec(self, rotate: list[float] | None = None) -> Vector:
        """

        Вычисляет единичный вектор, направленный вправо, на основе углов Эйлера дрона.



        Args:

            rotate: Список из трех углов (тангаж, рысканье, крен) в градусах.

                    Порядок: [тангаж (pitch), рысканье (yaw), крен (roll)].



        Returns:

            Единичный вектор (numpy array) [x, y, z], представляющий направление "вправо".

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

        right_local = np.array([0, 0, 1])

        right_global = R @ right_local

        return Vector(*right_global)

    def up_to_forwored(self, vector_up: Vector) -> Vector:
        """

        Преобразует вектор, направленный вверх в локальной системе координат дрона,

        в вектор, направленный вперед в мировой системе координат.



        Args:

            vector_up: Единичный вектор (numpy array) [x, y, z], представляющий направление "вверх"

                       в локальной системе координат дрона (обычно [0, 1, 0]).



        Returns:

            Единичный вектор (numpy array) [x, y, z], представляющий направление "вперед"

            в мировой системе координат.

        """

        rotate = self.drone.params.angle

        pitch_rad = np.deg2rad(rotate[2])

        yaw_rad = np.deg2rad(rotate[1])

        roll_rad = np.deg2rad(rotate[0])

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

        R = Ry @ Rx @ Rz

        forward_local = np.array([1, 0, 0])

        forward_global = R @ forward_local

        return Vector(*forward_global)

    def up_to_right(self, vector_up: Vector) -> Vector:
        """

        Преобразует вектор, направленный вверх в локальной системе координат дрона,

        в вектор, направленный вправо в мировой системе координат.



        Args:

            vector_up: Единичный вектор (numpy array) [x, y, z], представляющий направление "вверх"

                       в локальной системе координат дрона (обычно [0, 1, 0]).



        Returns:

            Единичный вектор (numpy array) [x, y, z], представляющий направление "вправо"

            в мировой системе координат.

        """

        rotate = self.drone.params.angle

        pitch_rad = np.deg2rad(rotate[2])

        yaw_rad = np.deg2rad(rotate[1])

        roll_rad = np.deg2rad(rotate[0])

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

        R = Ry @ Rx @ Rz

        right_local = np.array([0, 0, 1])

        right_global = R @ right_local

        return Vector(*right_global)

    def _get_rotation_matrix(self, rotate: list[float]) -> np.ndarray:
        """Матрица вращения ZYX из углов [roll, yaw, pitch] в градусах."""

        roll_rad, yaw_rad, pitch_rad = map(np.deg2rad, rotate)  # [0], [1], [2]

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

        # Порядок ZYX: v_global = Ry @ Rx @ Rz @ v_local

        R = Ry @ Rx @ Rz

        return R

    def _rotate_local_to_global(
        self, local_vector: Vector, rotate: list[float]
    ) -> Vector:
        """Вращает локальный вектор в глобальную систему координат."""

        R = self._get_rotation_matrix(rotate)

        local_np = np.array([local_vector.x, local_vector.y, local_vector.z])

        global_np = R @ local_np

        return Vector(x=global_np[0], y=global_np[1], z=global_np[2])
