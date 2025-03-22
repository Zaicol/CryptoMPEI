import random
import math

from tabulate import tabulate


def getSieve(n):
    """
    Реализация решета Эратосфена для нахождения всех простых чисел до n.
    Возвращает список простых чисел.
    """
    # Создаем массив isprime, где isprime[i] = True, если i предположительно простое число
    isprime = [True for _ in range(n)]
    prime = []
    spf = [None for _ in range(n)]

    # Числа 0 и 1 не являются простыми
    isprime[0] = isprime[1] = False

    # Проходим по всем числам от 2 до n-1
    for i in range(2, n):
        if isprime[i]:  # Если i предположительно простое
            prime.append(i)  # Добавляем его в список простых чисел
            spf[i] = i  # Наименьший простой делитель числа i — это само число i

        # Обновляем массив isprime, помечая составные числа как False
        j = 0
        while j < len(prime) and i * prime[j] < n and prime[j] <= spf[i]:
            isprime[i * prime[j]] = False
            spf[i * prime[j]] = prime[j]
            j += 1

    return prime  # Возвращаем список простых чисел


def is_probably_prime(n, sieve):
    """
    Проверяет, является ли число n вероятно простым, используя список простых чисел из решета.
    Возвращает False, если n делится на любое из простых чисел из решета.
    """
    for x in sieve:
        if n % x == 0:
            return False
    return True


def generatePrime(n: int):
    """
    Генерирует простое число длиной n десятичных цифр.
    Использует алгоритм, основанный на решете Эратосфена и тестах на простоту.
    """
    up_limit = 10 ** n  # Верхний предел для генерации числа (10^n)
    lower_limit = up_limit // 10  # Нижний предел для генерации числа (10^n)
    primes = getSieve(1000)
    s = primes[-1]  # Начинаем с наибольшего простого числа из решета

    # Основной цикл для генерации простого числа
    while s < lower_limit:
        lo = (lower_limit - 1) // s + 1  # Минимальное значение r
        hi = (up_limit - 1) // s  # Максимальное значение r

        while True:
            # Генерируем случайное число r и вычисляем кандидата на простое число n
            try:
                r = random.randint(lo, hi) << 1  # r — четное случайное число
            except ValueError:
                print((up_limit - 1), s, hi)
                r = random.randint(lo, hi + 1)
            cand = s * r + 1  # Формула для нового кандидата на простое число

            # Проверяем, является ли n вероятно простым
            if not is_probably_prime(cand, primes):
                continue

            # Проводим дополнительную проверку на простоту с помощью теста Ферма
            while True:
                a = random.randint(2, cand - 1)
                if pow(a, cand - 1, cand) != 1:  # Проверяем малую теорему Ферма
                    break

                # Проверяем НОД для дополнительной уверенности
                d = math.gcd((pow(a, r, cand) - 1) % cand, cand)
                if d != cand:
                    if d == 1:  # Если НОД равен 1, n вероятно простое
                        s = cand
                    break

            if s == cand:
                break

    if s > up_limit:
        return generatePrime(n)
    return s


def get_new_matrix():
    return [[random.randint(10 ** 10, 10 ** 15), random.randint(3, 30)],
            [random.randint(3, 30), random.randint(3, 30)]]


def determinant(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def extended_gcd(a, b):
    if b == 0:
        return a, 1, 0
    else:
        g, x, y = extended_gcd(b, a % b)
        return g, y, x - (a // b) * y


def mod_inverse_euclidean(a, n):
    g, x, y = extended_gcd(a, n)
    if g != 1:
        return None  # Обратного элемента не существует
    else:
        return x % n  # Приводим результат к положительному значению


def inverse_matrix(matrix, p):
    det = determinant(matrix)
    det_inv = mod_inverse_euclidean(det, p)
    if det_inv is None:
        raise ValueError("Матрица необратима")

    # Строим обратную матрицу
    inv_matrix = [
        [(matrix[1][1] * det_inv) % p, (p - matrix[0][1] * det_inv) % p],
        [(p - matrix[1][0] * det_inv) % p, (matrix[0][0] * det_inv) % p]
    ]
    return inv_matrix


def multiply_matrices(matrix1, matrix2, p):
    # Извлекаем элементы матриц
    a11, a12 = matrix1[0][0], matrix1[0][1]
    a21, a22 = matrix1[1][0], matrix1[1][1]

    b11, b12 = matrix2[0][0], matrix2[0][1]
    b21, b22 = matrix2[1][0], matrix2[1][1]

    # Вычисляем элементы результирующей матрицы
    c11 = a11 * b11 + a12 * b21
    c12 = a11 * b12 + a12 * b22
    c21 = a21 * b11 + a22 * b21
    c22 = a21 * b12 + a22 * b22

    # Возвращаем результирующую матрицу
    return [[c11 % p, c12 % p], [c21 % p, c22 % p]]


def encode(ciphersize, matrix, p):
    # Извлекаем элементы
    a11, a12 = ciphersize[0], ciphersize[1]

    b11, b12 = matrix[0][0], matrix[0][1]
    b21, b22 = matrix[1][0], matrix[1][1]
    # Вычисляем элементы результата
    c11 = a11 * b11 + a12 * b21
    c12 = a11 * b12 + a12 * b22

    # Возвращаем результирующую матрицу
    return c11 % p, c12 % p


def print_matrix(matrix, name):
    print(name)
    str_matrix = [[str(e) for e in row] for row in matrix]
    # Используем tabulate для форматированного вывода матрицы
    print(tabulate(str_matrix, tablefmt="grid"))
    print()


p = generatePrime(49)
print("p =", p)
matrix_M = get_new_matrix()
det_M = determinant(matrix_M)

while det_M == 1:
    matrix_M = get_new_matrix()
    det_M = determinant(matrix_M)

print_matrix(matrix_M, "Оригинальная матрица M:")

matrix_inv = inverse_matrix(matrix_M, p)
print_matrix(matrix_inv, "Обратная матрица:")

print_matrix(multiply_matrices(matrix_M, matrix_inv, p), "Перемножение M и M^-1")

u1 = (random.randint(10 ** 2, 10 ** 6), random.randint(10 ** 2, 10 ** 6))
c1 = encode(u1, matrix_M, p)
u_dec = encode(c1, matrix_inv, p)
print("Открытый текст:", u1)
print("Шифртекст:", c1)
print("Расшифрованный текст:", u_dec)

if u1 == u_dec:
    print("Открытый и расшифрованный текст - совпадают")
else:
    print("Открытый и расшифрованный текст - не совпадают!")

print("\n" + "=" * 60 + "\n\nЗадание 2\n\n")


def solve_mod_system(a1, b1, c1, a2, b2, c2, p):
    """
    Решает систему двух линейных уравнений по модулю p:
        a1 * x + b1 * y ≡ c1 (mod p),
        a2 * x + b2 * y ≡ c2 (mod p).
    Возвращает кортеж (x, y).
    """
    # Вычисляем определитель системы
    det = (a1 * b2 - a2 * b1) % p
    if det == 0:
        raise ValueError("Система не имеет единственного решения")

    # Находим обратный элемент для определителя
    det_inv = mod_inverse_euclidean(det, p)
    if det_inv is None:
        raise ValueError("Определитель необратим по модулю p")

    # Решаем систему
    x = (det_inv * (c1 * b2 - c2 * b1)) % p
    y = (det_inv * (a1 * c2 - a2 * c1)) % p

    return x, y


def crypto_analysis(u1, c1, u2, c2, p):
    # Распаковываем пары
    x11, x12 = u1
    y11, y12 = c1
    x21, x22 = u2
    y21, y22 = c2

    # Решаем первую подсистему для m11 и m21
    m11, m21 = solve_mod_system(
        x11, x12, y11,  # Коэффициенты первого уравнения
        x21, x22, y21,  # Коэффициенты второго уравнения
        p  # Модуль
    )

    # Решаем вторую подсистему для m12 и m22
    m12, m22 = solve_mod_system(
        x11, x12, y12,  # Коэффициенты первого уравнения
        x21, x22, y22,  # Коэффициенты второго уравнения
        p  # Модуль
    )

    # Возвращаем восстановленную матрицу M
    return [[m11, m12], [m21, m22]]


# Генерируем вторую пару (u2, c2)
u2 = (random.randint(10 ** 3, 10 ** 9), random.randint(10 ** 3, 10 ** 9))
c2 = encode(u2, matrix_M, p)

print("Первая пара (u1, c1):")
print("Открытый текст u1:", u1)
print("Шифртекст c1:", c1)
print()
print("Вторая пара (u2, c2):")
print("Открытый текст u2:", u2)
print("Шифртекст c2:", c2)
print()

# Выполняем криптоанализ
recovered_matrix = crypto_analysis(u1, c1, u2, c2, p)
print_matrix(recovered_matrix, "Восстановленная матрица M:")

# Проверяем, совпадает ли восстановленная матрица с исходной
if recovered_matrix == matrix_M:
    print("Восстановленная матрица совпадает с исходной!")
else:
    print("Ошибка: восстановленная матрица не совпадает с исходной.")

print("\n" + "=" * 60 + "\n\nЗадание 3\n\n")

u1 = (0, 1)
c1 = encode(u1, matrix_M, p)
u2 = (1, 0)
c2 = encode(u2, matrix_M, p)

recovered_matrix = [list(c2), list(c1)]
print_matrix(recovered_matrix, "Восстановленная матрица M:")

if recovered_matrix == matrix_M:
    print("Восстановленная матрица совпадает с исходной!")
else:
    print("Ошибка: восстановленная матрица не совпадает с исходной.")

u1 = (0, 1)
c1 = encode(u1, matrix_inv, p)
u2 = (1, 0)
c2 = encode(u2, matrix_inv, p)

recovered_matrix = [list(c2), list(c1)]
print_matrix(recovered_matrix, "Восстановленная матрица M^-1:")

if recovered_matrix == matrix_inv:
    print("Восстановленная матрица совпадает с исходной!")
else:
    print("Ошибка: восстановленная матрица не совпадает с исходной.")