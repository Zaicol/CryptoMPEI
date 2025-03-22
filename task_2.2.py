# Задание 2.2
from tabulate import tabulate


def create_orthogonal_array(p):
    # Количество строк и столбцов
    rows = p * p
    cols = p + 1
    # Создаем пустой массив
    OA = [[0] * cols for _ in range(rows)]
    # Заполняем первый столбец
    for i in range(rows):
        OA[i][0] = i // p

    # Заполняем второй столбец
    for i in range(rows):
        OA[i][1] = i % p

    # Заполняем остальные столбцы
    for col in range(2, cols):
        for row in range(rows):
            OA[row][col] = (OA[row][0] + (col - 1) * OA[row][1]) % p

    return OA


def is_orthogonal_array(OA, p):
    # Количество строк и столбцов
    rows = p * p
    cols = p + 1

    # Проверяем каждую пару столбцов
    for i in range(cols):
        for j in range(i + 1, cols):
            # Создаем множество всех возможных пар
            pairs = set()
            for row in range(rows):
                pair = (OA[row][i], OA[row][j])
                if pair in pairs:
                    # Если пара уже встречалась, массив не ортогональный
                    return False
                pairs.add(pair)
            # Проверяем, что все возможные пары присутствуют
            if len(pairs) != p * p:
                return False
    return True


while True:
    p = int(input("Введите простое число p: "))
    # Проверка, что p - простое число
    if p < 2 or any(p % i == 0 for i in range(2, int(p ** 0.5) + 1)):
        print("Ошибка: число должно быть простым.")
    else:
        oa = create_orthogonal_array(p)
        # Вывод результата с использованием tabulate
        print(f"\nОртогональный массив OA({p}, {p + 1}, 1):")
        print(tabulate(oa, tablefmt="pretty"))
