from solution.geometry import Vector


def get_points() -> dict[Vector, list[Vector]]:
    def get_point_1():
        return {
            Vector(-72.53, 0, 93.89): [  # 1 - 13 самая ближняя
                Vector(-72.9, 3.00, 84.03),
                Vector(-71.9, 3.00, 89.02),
                Vector(-71.3, 3.00, 93.89),
            ]
        }

    def get_point_2():
        return {
            Vector(-70.81, 0.00, 100.86): [  # 2 - 4 2ая ближаня
                Vector(-74.30, 8.00, 88.25),
                Vector(-74.43, 8.50, 94.36),
                Vector(-72.83, 8.50, 98.50),
                Vector(-70.81, 8.50, 100.86),
            ]
        }

    def get_point_3():
        return {
            Vector(-69.93, 0.00, 113.52): [  # 3 - 7 3ая ближняя
                Vector(-77.02, 9.60, 83.47),
                Vector(-76.80, 9.60, 90.36),
                Vector(-75.65, 9.60, 97.36),
                Vector(-74.13, 9.60, 103.45),
                Vector(-72.81, 9.60, 108.97),
                Vector(-69.93, 9.60, 113.52),
            ]
        }

    def get_point_4():
        return {
            Vector(-60.32, 0.00, 113.95): [  # 4 - 3 4ая ближнаяя
                Vector(-75.02, 12.00, 81.74),
                Vector(-66.73, 12.00, 84.34),
                Vector(-63.47, 12.00, 89.26),
                Vector(-63.71, 12.00, 95.79),
                Vector(-63.93, 12.00, 101.28),
                Vector(-63.70, 12.00, 107.79),
                Vector(-62.32, 12.00, 113.95),
            ]
        }

    def get_point_5():
        return {
            Vector(-34.17, 0.00, 100.55):  # 5 - 0 слева нижняя
            [
                Vector(-67.93, 6.00, 82.84),
                Vector(-57.18, 6.00, 83.03),
                Vector(-50.13, 6.00, 96.99),
                Vector(-34.17, 6.00, 100.55),
            ]
        }

    def get_point_6():
        return {
            Vector(-24.3, 0.00, 99.83): [  # 6 - 1 слева верхняя
                Vector(-67.93, 9.00, 75.84),
                Vector(-57.18, 9.00, 75.03),
                Vector(-50.13, 12.00, 98.00),
                Vector(-30.13, 12.00, 98.0),
                Vector(-24.3, 12.00, 99.83),
            ]
        }

    def get_point_7():
        return {
            Vector(-30.8, 0, 48): [  # 7 -2 Под мостом на полу
                Vector(-71.14, 3.00, 68.59),
                Vector(-64.98, 3.00, 68.39),
                Vector(-56.01, 3.00, 67.57),
                Vector(-50.08, 3.00, 57.41),
                Vector(-47.10, 3.00, 50.25),
                Vector(-40.10, 3.00, 47.55),
                Vector(-30.8, 3.00, 48.34),
                Vector(-30.8, 3.00, 48),
            ]
        }

    def get_point_8():
        return {
            Vector(-26.97, 0, 54.39): [  # 8 - 11 Под мостом на каробке
                Vector(-69.63, 4.80, 68.87),
                Vector(-62.78, 4.80, 68.66),
                Vector(-52.41, 4.80, 69.68),
                Vector(-43.89, 4.80, 71.11),
                Vector(-36.27, 4.80, 71.38),
                Vector(-26.75, 4.80, 71.11),
                Vector(-14.69, 4.80, 70.90),
                Vector(-9.16, 4.80, 63.10),
                Vector(-9.16, 4.80, 51),
                Vector(-26.97, 4.90, 53.39),
            ]
        }

    def get_point_9():
        return {
            Vector(-3.26, 0, 38.55): [  # 9 - 14  у моста между коробками
                Vector(-74.13, 13, 65),
                Vector(-71.90, 13.00, 58.59),
                Vector(-49.52, 12.00, 49.84),
                Vector(-19.48, 12.00, 39.06),
                Vector(-5.56, 10.00, 38.8),
                Vector(-3.26, 10, 38.55),
            ]
        }

    def get_point_10():
        return {
            Vector(19.9, 0, 51.29): [  # 10 - 6 Под мостом далеко
                Vector(-71.90, 9, 73.59),
                Vector(-49, 7, 70.19),
                Vector(-19.48, 7, 69.06),
                Vector(-10, 7, 65.06),
                Vector(18.7, 7, 51.29),
            ]
        }

    def get_point_11():
        return {
            Vector(-26.36, 0, 24.28): [  # 11 - 10
                Vector(-72.90, 14.00, 62.59),
                Vector(-71.90, 15.00, 58.59),
                Vector(-49.52, 15.00, 49.84),
                Vector(-30.48, 15.00, 39.06),
                Vector(-26.37, 13, 24.28),
            ]
        }

    def get_point_12():
        return {
            Vector(-47.44, 0, 11.34): [
                Vector(-75.55, 4.00, 70.46),
                Vector(-57.86, 4.00, 68.47),
                Vector(-54.09, 4.00, 50.52),
                Vector(-51.63, 5.00, 26.25),
                Vector(-47.44, 5, 11.34),
            ]
        }

    def get_point_13():
        return {
            Vector(-43.82, 0, 6.51): [
                Vector(-71.55, 6.00, 68.46),
                Vector(-57.86, 6.00, 68.47),
                Vector(-56.09, 6.00, 50.52),
                Vector(-55.09, 6.00, 35.52),
                Vector(-54.63, 7.00, 26.25),
                Vector(-43.82, 8, 6.51),
            ]
        }

    def get_point_14():
        return {
            Vector(41.43, 3, 66.17):  # Заглушка для точки 12
            [
                Vector(-73.56, 10.00, 80.39),
                Vector(-44.53, 12.00, 78.24),
                Vector(-12.54, 14.00, 78.60),
                Vector(15.53, 14.00, 78.40),
                Vector(40.15, 14.00, 75.73),
                Vector(41.43, 12, 66.17),
            ]
        }

    def get_point_15():
        return {
            Vector(-20.17, 0, -48.13): [
                Vector(-84.07, 13.00, 68.12),
                Vector(-85.57, 13.00, 51.01),
                Vector(-84.88, 13.00, 0.44),
                Vector(-61.04, 7.00, 1.50),
                Vector(-30.42, 7.00, -3.68),
                Vector(-21.17, 7.00, -48.13),
            ]
        }

    p = {}
    p |= get_point_1()
    p |= get_point_2()
    p |= get_point_3()
    p |= get_point_4()
    p |= get_point_5()
    p |= get_point_6()
    p |= get_point_7()
    p |= get_point_8()
    p |= get_point_9()
    p |= get_point_10()
    p |= get_point_11()
    p |= get_point_12()
    p |= get_point_13()
    p |= get_point_14()
    p |= get_point_15()
    return p
