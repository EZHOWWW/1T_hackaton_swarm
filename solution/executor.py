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

        # --- Конфигурация PID (НУЖНО ТЩАТЕЛЬНО НАСТРОИТЬ!) ---
        self.base_thrust = 0.5
        self.altitude_pid = PIDController(
            Kp=1.0, Ki=0.05, Kd=0.6, setpoint=10.0, output_limits=(-0.5, 0.5)
        )
        self.pitch_pid = PIDController(
            Kp=0.04, Ki=0.01, Kd=0.03, setpoint=0.0, output_limits=(-0.3, 0.3)
        )
        self.roll_pid = PIDController(
            Kp=0.04, Ki=0.01, Kd=0.03, setpoint=0.0, output_limits=(-0.3, 0.3)
        )
        self.target_yaw = 0.0
        self.yaw_pid = PIDController(
            Kp=0.05,
            Ki=0.005,
            Kd=0.02,
            setpoint=self.target_yaw,
            output_limits=(-0.2, 0.2),
            angle_wrap_degrees=360,
        )

        # --- Параметры управления ---
        self.max_tilt_angle = 15.0

        # --- Конфигурация моторов (Круговая, 8 моторов) ---
        # ВАЖНО: Проверьте предположения о нумерации и вращении!
        # Предположения:
        # 1. Мотор 0 - спереди (положительное направление оси X дрона).
        # 2. Нумерация идет ПРОТИВ часовой стрелки (CCW). (0=Перед, 1=Перед-Лево, 2=Лево, ..., 7=Перед-Право)
        # 3. Четные моторы (0, 2, 4, 6) вращаются CCW (создают реактивный момент CW на раму = -Yaw).
        # 4. Нечетные моторы (1, 3, 5, 7) вращаются CW (создают реактивный момент CCW на раму = +Yaw).
        # 5. Симуляция моделирует реактивный момент.
        # 6. Положительный Roll = правый борт вниз. Положительный Pitch = нос вверх.

        num_motors = 8
        self.motor_effects = []
        # Углы моторов против часовой стрелки от оси X+ (вперед)
        angles_rad = [i * (2 * math.pi / num_motors) for i in range(num_motors)]
        # Пример: 0: 0, 1: pi/4, 2: pi/2, 3: 3pi/4, 4: pi, 5: 5pi/4, 6: 3pi/2, 7: 7pi/4

        print("Initializing Motor Effects (Circular Octo):")  # Отладочный вывод
        print(
            "Assumptions: 0=Front, numbering CCW. Even motors=CCW(-Yaw), Odd motors=CW(+Yaw)"
        )

        for i in range(num_motors):
            angle = angles_rad[i]
            # Roll Effect = sin(angle) (Момент вокруг оси X = r_y * Fz)
            roll_effect = math.sin(angle)
            # Pitch Effect = -cos(angle) (Момент вокруг оси Y = -r_x * Fz)
            pitch_effect = -math.cos(angle)
            # Yaw Effect (Реактивный момент на раму): +1 для CW моторов (нечетные), -1 для CCW моторов (четные)
            yaw_effect = +1 if i % 2 != 0 else -1

            # Округление для наглядности (можно убрать)
            roll_effect = round(roll_effect, 3)
            pitch_effect = round(pitch_effect, 3)

            self.motor_effects.append([roll_effect, pitch_effect, yaw_effect])
            # Отладка:
            # print(f"  Motor {i} ({(math.degrees(angle)):.0f} deg): Roll={roll_effect:.2f}, Pitch={pitch_effect:.2f}, Yaw={yaw_effect}")

        # Результат расчета (примерный):
        # Motor 0 (0 deg):   Roll=0.00, Pitch=-1.00, Yaw=-1 (CCW)
        # Motor 1 (45 deg):  Roll=0.71, Pitch=-0.71, Yaw=+1 (CW)
        # Motor 2 (90 deg):  Roll=1.00, Pitch=-0.00, Yaw=-1 (CCW)
        # Motor 3 (135 deg): Roll=0.71, Pitch=0.71,  Yaw=+1 (CW)
        # Motor 4 (180 deg): Roll=0.00, Pitch=1.00,  Yaw=-1 (CCW)
        # Motor 5 (225 deg): Roll=-0.71,Pitch=0.71,  Yaw=+1 (CW)
        # Motor 6 (270 deg): Roll=-1.00,Pitch=0.00,  Yaw=-1 (CCW)
        # Motor 7 (315 deg): Roll=-0.71,Pitch=-0.71, Yaw=+1 (CW)
        # !!! ПРОВЕРЬТЕ ЭТУ МАТРИЦУ И ПРЕДПОЛОЖЕНИЯ В СИМУЛЯЦИИ !!!

    # --- Методы update_attitude_pids, update_altitude_pid ---
    # (Остаются без изменений по сравнению с предыдущим ответом)
    def update_attitude_pids(self, target_pitch, target_roll, dt):
        # ... (код как прежде) ...
        if self.drone.params is None:
            return 0, 0, 0
        current_pitch = self.drone.params.angle[0]
        current_yaw = self.drone.params.angle[1]
        current_roll = self.drone.params.angle[2]
        self.pitch_pid.setpoint = target_pitch
        pitch_correction = self.pitch_pid.update(current_pitch, dt)
        self.roll_pid.setpoint = target_roll
        roll_correction = self.roll_pid.update(current_roll, dt)
        self.yaw_pid.setpoint = self.target_yaw
        yaw_correction = self.yaw_pid.update(current_yaw, dt)
        return pitch_correction, roll_correction, yaw_correction

    def update_altitude_pid(self, target_height, dt):
        # ... (код как прежде) ...
        if self.drone.params is None:
            return self.base_thrust
        current_height = self.drone.params.possition.z
        self.altitude_pid.setpoint = target_height
        altitude_correction = self.altitude_pid.update(current_height, dt)
        total_thrust = self.base_thrust + altitude_correction
        return total_thrust

    # --- Метод move_to_direction ---
    # (Логика расчета целевых углов и вызова PID остается)
    # (Изменяется только применение коррекций в цикле смешивания моторов)
    def move_to_direction(
        self, direction: Vector, target_height: float, target_speed: float, dt: float
    ) -> list[float]:
        if self.drone.params is None or not self.drone.params.is_alive:
            return [0.0] * 8

        # --- Шаг 1: Вычисление целевых углов ---
        # ... (код расчета target_pitch, target_roll как прежде) ...
        if direction.length() > 1e-6:
            desired_velocity_xy = direction.replace(z=0).normalize() * target_speed
        else:
            desired_velocity_xy = Vector(0, 0, 0)
        target_yaw_rad = math.radians(self.target_yaw)
        cos_yaw = math.cos(-target_yaw_rad)
        sin_yaw = math.sin(-target_yaw_rad)
        target_vel_x_drone = (
            desired_velocity_xy.x * cos_yaw - desired_velocity_xy.y * sin_yaw
        )
        target_vel_y_drone = (
            desired_velocity_xy.x * sin_yaw + desired_velocity_xy.y * cos_yaw
        )
        gain_vel_to_angle = 0.8
        # Проверьте знаки для вашей системы координат!
        target_pitch = -target_vel_x_drone * gain_vel_to_angle
        target_roll = target_vel_y_drone * gain_vel_to_angle
        target_pitch = max(-self.max_tilt_angle, min(self.max_tilt_angle, target_pitch))
        target_roll = max(-self.max_tilt_angle, min(self.max_tilt_angle, target_roll))

        # --- Шаг 2: Обновление PID регуляторов ---
        pitch_correction, roll_correction, yaw_correction = self.update_attitude_pids(
            target_pitch, target_roll, dt
        )
        thrust = self.update_altitude_pid(target_height, dt)

        # --- Шаг 3: Распределение тяги и коррекций по моторам ---
        # Используем НОВУЮ матрицу self.motor_effects
        engines = [0.0] * 8
        for i in range(8):
            motor_thrust = thrust
            # Применяем коррекции моментов от PIDов
            motor_thrust += self.motor_effects[i][0] * roll_correction  # Крен
            motor_thrust += self.motor_effects[i][1] * pitch_correction  # Тангаж
            motor_thrust += (
                self.motor_effects[i][2] * yaw_correction
            )  # Рысканье (Реактивный момент)
            engines[i] = motor_thrust

        # --- Шаг 4: Нормализация и ограничение выхода моторов ---
        engines = np.clip(np.array(engines), 0.0, 1.0)

        return engines.tolist()
