import numpy as np

# Входные данные (замените своими фактическими значениями)
world_coords_square = [(-74.0, 78.0), (-74.0, 72.0), (-80.0, 72.0), (-80.0, 78.0)]

pixel_coords_square = [
    (342, 787),  # Замените на пиксельные координаты угла 1
    (366, 786),  # Замените на пиксельные координаты угла 2
    (343, 811),  # Замените на пиксельные координаты угла 3
    (368, 810),  # Замените на пиксельные координаты угла 4
]

pixel_fire_coords = [
    (378, 737),  # Замените на пиксельные координаты первого огня
    (378, 680),  # Замените на пиксельные координаты второго огня
    (372, 598),  # Замените на пиксельные координаты третьего огня
    # Добавьте координаты для всех ваших огней
]


def estimate_reverse_affine_transform(world_coords, pixel_coords_square):
    """
    Оценивает обратное аффинное преобразование из пиксельных координат в мировые.

    Args:
        world_coords (np.array): Массив мировых координат углов квадрата (x, z).
        pixel_coords_square (np.array): Массив пиксельных координат углов квадрата (u, v).

    Returns:
        np.array: Матрица обратного аффинного преобразования (2x3).
    """
    A = []
    b = []
    for s, d in zip(pixel_coords_square, world_coords):
        u, v = s
        x, z = d
        A.append([u, v, 1, 0, 0, 0])
        A.append([0, 0, 0, u, v, 1])
        b.append(x)
        b.append(z)
    A = np.array(A)
    b = np.array(b)
    x_sol, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    return x_sol.reshape((2, 3))


def apply_reverse_transform(pixel_coord, reverse_transform):
    """
    Применяет обратное аффинное преобразование к пиксельным координатам.

    Args:
        pixel_coord (np.array): Пиксельные координаты точки (u, v).
        reverse_transform (np.array): Матрица обратного аффинного преобразования (2x3).

    Returns:
        np.array: Мировые координаты точки (x, z).
    """
    u, v = pixel_coord
    transformed = np.dot(reverse_transform, np.array([u, v, 1]))
    return transformed


def get_real_fire_coordinates(
    pixel_fire_coords, world_coords_square, pixel_coords_square
):
    """
    Находит реальные координаты огней в симуляции по их пиксельным координатам.

    Args:
        pixel_fire_coords (list[tuple[int, int]]): Список пиксельных координат огней (u, v).
        world_coords_square (list[tuple[float, float]]): Список мировых координат углов квадрата (x, z).
        pixel_coords_square (list[tuple[int, int]]): Список пиксельных координат углов квадрата (u, v).

    Returns:
        list[tuple[float, float, float]]: Список реальных координат огней (x, 0.0, z).
    """
    world_coords_np = np.array(world_coords_square)
    pixel_coords_square_np = np.array(pixel_coords_square)
    pixel_fire_coords_np = np.array(pixel_fire_coords)

    # Оценка матрицы обратного преобразования
    reverse_transform_matrix = estimate_reverse_affine_transform(
        world_coords_np, pixel_coords_square_np
    )
    print("Матрица обратного преобразования:\n", reverse_transform_matrix)

    real_fire_coords = []
    for pixel_coord in pixel_fire_coords_np:
        real_xz = apply_reverse_transform(pixel_coord, reverse_transform_matrix)
        real_fire_coords.append((real_xz[0], 0.0, real_xz[1]))

    return real_fire_coords


# Получение реальных координат огней
real_coordinates = get_real_fire_coordinates(
    pixel_fire_coords, world_coords_square, pixel_coords_square
)

# Вывод результатов
print("\nРеальные координаты огней (x, 0, z):")
for i, coord in enumerate(real_coordinates):
    print(f"Огонь {i + 1}: x={coord[0]:.2f}, y={coord[1]:.1f}, z={coord[2]:.2f}")
