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


class Lidar:
    def __init__(
        self,
        direction_name: str,
        direction_vector: Vector,
        force: float,
        active_distance: float,
    ):
        self.direction_name = direction_name
        self.direction_vector = (
            direction_vector.normalize()
        )  # Локальное направление лидара
        self.force = force
        self.active_distance = active_distance
        self.MAX_LIDAR_DISTANCE = 10  # Физический предел дальности

    def get_correction(
        self,
        actual_distance: float,
        rotate: list[float],
        rotation_func,  # Функция DroneExecutor._rotate_local_to_global
    ) -> Vector:
        """Рассчитывает вектор коррекции в ГЛОБАЛЬНОЙ системе координат."""
        if actual_distance < 0 or actual_distance > self.active_distance:
            return Vector()  # Нет коррекции

        # Множитель силы (0 до 1), 1 при actual_distance = 0
        distance_mult = max(0.0, 1.0 - (actual_distance / self.active_distance))

        # Локальный вектор силы (против направления лидара)
        local_force_vector = (-self.direction_vector) * distance_mult * self.force

        # Преобразование в глобальный вектор коррекции
        global_correction_vector = rotation_func(local_force_vector, rotate)
        return global_correction_vector


class DroneExecutor:
    def __init__(self, drone):
        self.drone = drone
        # --- Параметры управления ---
        # - Pid регуляторы -
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
        self.max_tilt_angle = 12

        # - Параметры отталкивания дроннов (избежание сталкновений) -
        self._safety_radius = 6  # Радиус безопасности use in self.correct_direction_from_other_drones, correct_direction_from_lidars
        self._repulsion_strength = (
            5  # Сила отталкивания use in self.correct_direction_from_other_drones
        )

        # - Параметры лидаров -
        up_lidar_force = u = 0.2
        self.lidar_directions = {
            "f": -Vector(-1, u, 0),  # вперед
            "fr": -Vector(-1, u, -1),  # вперед-вправо
            "r": -Vector(0, u, -1),  # вправо
            "br": -Vector(1, u, -1),  # назад-вправо
            "b": -Vector(1, u, 0),  # назад
            "bl": -Vector(1, u, +1),  # назад-влево
            "l": -Vector(0, u, +1),  # влево
            "fl": -Vector(-1, u, +1),  # вперед-влево
            "up": Vector(0, 1, 0),  # вверх
            "d": Vector(0, -1, 0),  # вниз
        }
        self.lidar_force = 1
        self.lidar_active_distance = 8  # дистанция с которой лидар начинает оказывать воздейстиве <= MAX_LIDAR_DISTANCE = 10
        self.MIN_FLIGHT_HEIGHT = 0.5  # Пример
        self.MAX_FLIGHT_HEIGHT = 25.0  # Пример
        self.MAX_TOTAL_LIDAR_CORRECTION_NORM = self.lidar_force * 2

        # TODO best params
        # -----------

        self.max_tilt_procent = self.max_tilt_angle / 90
        self.pitch_pid = PIDController(**pitch_yaw_pid_params)
        self.yaw_pid = PIDController(**pitch_yaw_pid_params)

        self._lidars = [
            Lidar(l[0], l[1], self.lidar_force, self.lidar_active_distance)
            for l in self.lidar_directions.items()
        ]

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
        right_vec = self.get_right_vec()
        direction = target_up.normalize()
        setpoint = right_vec.dot(direction)
        self.yaw_pid.setpoint = setpoint
        measurement = right_vec.dot(current_up_vector)
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

        print(target_height, end="\t")
        target_height = self.correct_height(
            target_height, target, self.drone.params.lidars, dt
        )
        print(target_height)

        engines = np.zeros(8)
        direction = self.correct_direction(direction, target_speed, dt)
        # print(direction)
        vector_up = self.get_up_vector()
        # print(vector_up)
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
        direction = self.correct_direction_from_other_drones(
            direction,
            self.drone.params.possition,
            [i.params.possition for i in self.drone.swarm.units if not i.is_dead],
            dt,
        )
        direction = self.correct_gravity(direction, target_speed, dt)
        return direction

    def correct_direction_from_lidars(
        self, direction: Vector, lidars_data: dict, dt: float
    ) -> Vector:
        """
        Корректирует желаемое направление движения на основе показаний лидаров.

        Args:
            direction: Исходный желаемый вектор направления (ожидается нормализованный).
            lidars_data: Словарь с данными лидаров {имя: дистанция}.
            dt: Дельта времени (пока не используется).

        Returns:
            Новый вектор направления (не обязательно нормализованный),
            учитывающий силы отталкивания от препятствий.
            Может вернуть нулевой вектор при сильном конфликте направлений.
        """
        total_correction_vector = Vector(0, 0, 0)
        current_rotation = self.drone.params.angle  # Углы [roll, yaw, pitch]

        for lidar_sensor in self._lidars:
            direction_name = lidar_sensor.direction_name
            if (
                direction_name in lidars_data and lidars_data[direction_name] != -1
                # and direction_name not in ["up", "d"]
            ):
                actual_distance = lidars_data[direction_name]
                global_correction_vector = lidar_sensor.get_correction(
                    actual_distance, current_rotation, self._rotate_local_to_global
                )
                total_correction_vector += global_correction_vector

        # Ограничение на общую силу коррекции (чтобы избежать слишком резких маневров)
        correction_norm = total_correction_vector.length()
        if correction_norm > self.MAX_TOTAL_LIDAR_CORRECTION_NORM:
            total_correction_vector = total_correction_vector * (
                self.MAX_TOTAL_LIDAR_CORRECTION_NORM / correction_norm
            )
            print(f"Lidar correction capped at {self.MAX_TOTAL_LIDAR_CORRECTION_NORM}")

        corrected_direction = direction + total_correction_vector

        return corrected_direction

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

        corrected_direction = corrected_direction.normalize()

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
        self,
        target_height: float,
        target_pos: Vector,
        lidars: dict,  # Данные лидаров {имя: дистанция}
        dt: float,
    ) -> float:
        """Объединяет коррекции высоты."""
        # 1. Коррекция по дистанции до цели (из вашего кода)
        pos_diff = self.drone.params.possition - target_pos
        distance_xz = Vector(pos_diff.x, 0, pos_diff.z).length()

        target_height = self.correct_height_to_distance(
            target_height, distance_xz, target_pos.y
        )

        # 2. Коррекция по лидарам
        target_height = self.correct_height_from_lidars(target_height, lidars, dt)

        # 3. Ограничение высоты полета
        final_height = max(
            self.MIN_FLIGHT_HEIGHT,
            min(self.MAX_FLIGHT_HEIGHT, target_height),
        )

        return final_height

    def correct_height_to_distance(
        self, height: float, distance_xz: float, target_pos_y: float
    ) -> float:
        """Коррекция высоты при приближении к цели по горизонтали."""
        DISTANCE_TO_GO_DOWN = 4
        UP_TO_TARGET = 5
        if distance_xz < DISTANCE_TO_GO_DOWN:
            # Приближаемся к высоте цели
            return target_pos_y + UP_TO_TARGET
        return height

    def correct_height_from_lidars(
        self, current_target_height: float, lidars_data: dict, dt: float
    ) -> float:
        """Корректирует целевую высоту на основе лидаров и ориентации дрона."""
        total_vertical_correction = 0.0
        current_rotation = self.drone.params.angle  # Углы [roll, yaw, pitch]

        for lidar_sensor in self._lidars:
            direction_name = lidar_sensor.direction_name
            if direction_name in lidars_data and lidars_data[direction_name] != -1:
                actual_distance = lidars_data[direction_name]
                # Получаем глобальный вектор коррекции от лидара
                global_correction_vector = lidar_sensor.get_correction(
                    actual_distance,
                    current_rotation,
                    self._rotate_local_to_global,  # Передаем метод вращения
                )
                # Суммируем только вертикальную (Y) компоненту
                total_vertical_correction += global_correction_vector.y
            # else: Игнорируем отсутствующие данные лидара

        # Применяем суммарную коррекцию
        corrected_height = current_target_height + total_vertical_correction
        return corrected_height

    def get_up_vector(self, rotate: list[float] | None = None) -> Vector:
        """Глобальный вектор "вверх" дрона."""
        if rotate is None:
            rotate = self.drone.params.angle
        return self._rotate_local_to_global(Vector(0, 1, 0), rotate)

    def get_forwored_vec(self, rotate: list[float] | None = None) -> Vector:
        """
        Вычисляет единичный вектор, направленный вперед, на основе углов Эйлера дрона.

        Args:
            rotate: Список из трех углов (тангаж, рысканье, крен) в градусах.
                    Порядок: [тангаж (pitch), рысканье (yaw), крен (roll)].

        Returns:
            Единичный вектор (numpy array) [x, y, z], представляющий направление "вперед".
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
