import numpy as np
from fractions import Fraction
from tabulate import tabulate

# Распределение вероятностей ключей
p_K = np.array([Fraction(1, 2), Fraction(1, 4), Fraction(1, 4)])

# Матрица преобразований аутентификации e_k(x)
e_matrix = np.array([
    [1, 1, 1, 2],  # k=1
    [2, 2, 1, 2],  # k=2
    [1, 2, 2, 1]  # k=3
], dtype=object)

# Размерности
n_x = 4  # Число сообщений
n_a = 2  # Число билетов аутентификации
n_k = 3  # Число ключей

# Распределение вероятностей сообщений (равномерное)
p_X = Fraction(1, n_x)

indices_combinations = [(x+1, a+1) for x in range(n_x) for a in range(n_a)]

# Шаг 1: Вычисление p_им(x, a)
p_им = np.zeros((n_x, n_a), dtype=object)
for x in range(n_x):
    for a in range(n_a):
        # Суммируем вероятности ключей, для которых e_k(x) = a
        p_им[x, a] = np.sum(p_K[e_matrix[:, x] == a + 1])

# Шаг 2: Оптимальная стратегия имитации
opt_imitation = []
max_p_им = np.max(p_им)
for x in range(n_x):
    for a in range(n_a):
        if p_им[x, a] == max_p_им:
            opt_imitation.append((x + 1, a + 1))

# Шаг 3: Вычисление p_подм(x', a'; x, a)
p_подм_2d = np.zeros((n_x * n_a, n_x * n_a), dtype=object)

# Заполняем массив
for x in range(n_x):
    for a in range(n_a):
        index_1 = x * n_a + a  # Первый индекс
        for x_prime in range(n_x):
            for a_prime in range(n_a):
                index_2 = x_prime * n_a + a_prime  # Второй индекс
                if x_prime == x:
                    p_подм_2d[index_1, index_2] = 0
                    continue
                # Ищем индексы строк, где в столбце x значение равно a
                indices1 = np.where(e_matrix[:, x_prime] == a_prime + 1)[0]
                indices2 = np.where(e_matrix[:, x] == a + 1)[0]

                # Находим пересечение
                intersection = np.intersect1d(indices1, indices2)
                result_up = Fraction(np.sum(p_K[intersection]))

                # Вероятность подмены
                p_подм_2d[index_1, index_2] = result_up / p_им[x, a]


# Шаг 4: Оптимальная стратегия подмены
opt_substitution = {}
for x in range(n_x):
    for a in range(n_a):
        # Находим максимальную вероятность подмены для данной пары (x, a)
        max_p_subm = np.max(p_подм_2d[x * n_a + a])
        max_p_arg_subm = np.argmax(p_подм_2d[x * n_a + a])
        opt_substitution[(x + 1, a + 1)] = (max_p_arg_subm // n_a + 1,  max_p_arg_subm % n_a + 1)

# Высчитываем p(x, a) и p(x, a) * p_подм(x, a)
p_xa = (p_им * p_X).flatten()
p_подм_max = np.max(p_подм_2d, axis=1)
p_xa_subm = p_xa * p_подм_max
p_подм = np.sum(p_xa_subm)


def print_fraction_table(arr, splices=False):
    # Преобразуем элементы массива в дроби
    fraction_arr = np.vectorize(lambda x: str(Fraction(x).limit_denominator()))(arr)

    # Добавляем столбец с ID
    ids = np.arange(1, len(arr) + 1).reshape(-1, 1)  # Создаем столбец ID
    if splices:
        ids = np.array([f"{x}" for x in indices_combinations]).reshape(-1, 1)
    table = np.hstack((ids, fraction_arr))  # Объединяем ID и массив дробей

    # Создаем заголовки
    headers = ["x\\a"] + [f"{i + 1}" for i in range(len(arr[0]))]
    if splices:
        headers = ["(x, a)\\(x', a')"] + [f"{x}" for x in indices_combinations]

    # Выводим таблицу
    print(tabulate(table, headers=headers, tablefmt="pretty"))


# Вывод результатов
print("p_им(x, a):")
print_fraction_table(p_им)

print("\nОптимальная стратегия имитации:")
print(str(opt_imitation) + ", где p_им = ", Fraction(max_p_им))

print("\np_подм(x', a'; x, a):")
print_fraction_table(p_подм_2d, True)

print("\np(x, a) * p_подм(x, a):")
table_data = np.array([
    [f"{x}" for x in indices_combinations],
    [str(frac) for frac in p_xa],
    [str(frac) for frac in p_подм_max],
    [str(frac) for frac in p_xa_subm]
], dtype=object)
headers = ["(x,a)", "p(x,a)", "p_подм(x,a)", "p(x,a) × p_подм(x,a)"]
print(tabulate(table_data.T, headers=headers, tablefmt="pretty"))

print("\np_подм:", Fraction(p_подм))

print("\nОптимальная стратегия подмены третьей стороны:")
table_data = []
for key, value in opt_substitution.items():
    table_data.append([str(key), str(value)])

# Заголовки таблицы
headers = ["(x, a)", "(x', a')"]

# Выводим таблицу с использованием tabulate
print(tabulate(table_data, headers=headers, tablefmt="grid"))

# Задание 2.2
def generate_oa(p):
    """
    Генерация ортогонального массива OA(p, p+1, 1).
    :param p: простое число
    :return: массив OA(p, p+1, 1)
    """
    import numpy as np

    # Создаем массив размера p x (p+1), заполняем нулями
    oa = np.zeros((p, p + 1), dtype=int)

    # Заполняем первый столбец нулями
    oa[:, 0] = 0

    # Заполняем остальные столбцы
    for j in range(1, p + 1):  # j - номер столбца
        oa[:, j] = [(i + j - 1) % p for i in range(p)]

    return oa


# Ввод простого числа p
p = int(input("Введите простое число p: "))

# Проверка, что p - простое число
if p < 2 or any(p % i == 0 for i in range(2, int(p**0.5) + 1)):
    print("Ошибка: число должно быть простым.")
else:
    # Генерация ортогонального массива
    oa = generate_oa(p)

    # Вывод результата
    print(f"Ортогональный массив OA({p}, {p+1}, 1):")
    print(oa)