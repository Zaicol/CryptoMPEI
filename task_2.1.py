import numpy as np
from fractions import Fraction
from tabulate import tabulate


def print_fraction_table(arr, splices=False):
    # Добавляем столбец с ID
    ids = np.arange(1, len(arr) + 1).reshape(-1, 1)  # Создаем столбец ID
    if splices:
        ids = np.array([f"{x}" for x in indices_combinations]).reshape(-1, 1)
    table = np.hstack((ids, arr))  # Объединяем ID и массив дробей

    # Создаем заголовки
    headers = ["x\\a"] + [f"{i + 1}" for i in range(len(arr[0]))]
    if splices:
        headers = ["(x, a)\\(x', a')"] + [f"{x}" for x in indices_combinations]

    # Выводим таблицу
    print(tabulate(table, headers=headers, tablefmt="pretty"))


# Ввод размерностей
n_x = int(input("Введите число сообщений (n_x): "))
n_a = int(input("Введите число билетов аутентификации (n_a): "))
n_k = int(input("Введите число ключей (n_k): "))

# Ввод распределения вероятностей ключей
print("Введите распределение вероятностей ключей (p_K):")
p_K = []
for i in range(n_k):
    while True:
        try:
            value = input(f"p_K[{i + 1}]: ")
            prob = Fraction(value)
            p_K.append(prob)
            break
        except ValueError:
            print("Ошибка: введите дробное число в формате 'числитель/знаменатель' или целое число.")

# Проверка, что сумма вероятностей равна 1
if not sum(p_K) == 1:
    print("Ошибка: сумма вероятностей должна быть равна 1.")
    exit()

p_K = np.array(p_K)

# Ввод матрицы преобразований аутентификации e_k(x)
print("Введите матрицу преобразований аутентификации e_k(x):")
e_matrix = []
for i in range(n_k):
    while True:
        try:
            row = input(f"Строка {i + 1} (введите {n_x} элементов через пробел): ").split()
            if len(row) != n_x:
                raise ValueError
            row = [Fraction(x) for x in row]
            e_matrix.append(row)
            break
        except ValueError:
            print(f"Ошибка: введите ровно {n_x} элементов в строке.")

e_matrix = np.array(e_matrix, dtype=object)

# Распределение вероятностей сообщений (равномерное)
p_X = Fraction(1, n_x)

indices_combinations = [(x+1, a+1) for x in range(n_x) for a in range(n_a)]

# Шаг 1: Вычисление p_им(x, a)
p_им = np.zeros((n_x, n_a), dtype=object)
for x in range(n_x):
    for a in range(n_a):
        # Суммируем вероятности ключей, для которых e_k(x) = a
        p_им[x, a] = np.sum(p_K[e_matrix[:, x] == a + 1])

print("p_им(x, a):")
print_fraction_table(p_им)

# Шаг 2: Оптимальная стратегия имитации
opt_imitation = []
max_p_им = np.max(p_им)
for x in range(n_x):
    for a in range(n_a):
        if p_им[x, a] == max_p_им:
            opt_imitation.append((x + 1, a + 1))

print("\nОптимальная стратегия имитации:")
print(f"Навязывание одного из сообщений {opt_imitation}, где p_им = {Fraction(max_p_им)}")

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

print("\np_подм(x', a'; x, a):")
print_fraction_table(p_подм_2d, True)


# Шаг 4: Оптимальная стратегия подмены

# Высчитываем p(x, a) и p(x, a) * p_подм(x, a)
p_xa = (p_им * p_X).flatten()
p_подм_max = np.max(p_подм_2d, axis=1)
p_xa_prod_подм = p_xa * p_подм_max
p_подм = np.sum(p_xa_prod_подм)

print("\np(x, a) * p_подм(x, a):")
table_data = np.array([
    [f"{x}" for x in indices_combinations],
    [str(frac) for frac in p_xa],
    [str(frac) for frac in p_подм_max],
    [str(frac) for frac in p_xa_prod_подм]
], dtype=object)
headers = ["(x,a)", "p(x,a)", "p_подм(x,a)", "p(x,a) × p_подм(x,a)"]
print(tabulate(table_data.T, headers=headers, tablefmt="pretty"))

print("\np_подм:", Fraction(p_подм))

# Оптимальная стратегия подмены, обеспечивающая максимальную вероятность
opt_substitution = {}
for x in range(n_x):
    for a in range(n_a):
        # Находим максимальную вероятность подмены для данной пары (x, a)
        max_p_subm = np.max(p_подм_2d[x * n_a + a])
        max_p_arg_subm = np.argmax(p_подм_2d[x * n_a + a])
        opt_substitution[(x + 1, a + 1)] = (max_p_arg_subm // n_a + 1,  max_p_arg_subm % n_a + 1)

print("\nОптимальная стратегия подмены третьей стороны:")
table_data = []
for key, value in opt_substitution.items():
    table_data.append([str(key), str(value)])
headers = ["(x, a)", "(x', a')"]
print(tabulate(table_data, headers=headers, tablefmt="grid"))
