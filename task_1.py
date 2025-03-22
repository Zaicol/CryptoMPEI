import random
from sympy import solve


def is_prime(num):
    if num < 2:
        return False
    # Проверяем делители до sqrt(num)
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            return False
    return True


# Функция для поиска первого простого числа длиной 49 символов
def find_prime_of_length_12():
    # Начинаем с числа 10^48 (первое число длиной 49 символов)
    num = 10 ** 12
    # Убедимся, что начинаем с нечётного числа
    if num % 2 == 0:
        num += 1
    # Ищем первое простое число
    while True:
        if is_prime(num):
            return num
        num += 2  # Переходим к следующему нечётному числу


def get_new_matrix():
    return [[random.randint(10 ** 7, 10 ** 10), random.randint(3, 30)],
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
        [(matrix[1][1] * det_inv) % p, (p-matrix[0][1] * det_inv) % p],
        [(p-matrix[1][0] * det_inv) % p, (matrix[0][0] * det_inv) % p]
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
    s = [[str(e) for e in row] for row in matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))
    print()


p = find_prime_of_length_12()
matrix_M = get_new_matrix()
det_M = determinant(matrix_M)

while det_M == 1:
    matrix_M = get_new_matrix()
    det_M = determinant(matrix_M)

print_matrix(matrix_M, "Оригинальная матрица M:")

matrix_inv = inverse_matrix(matrix_M, p)
print_matrix(matrix_inv, "Обратная матрица:")

print_matrix(multiply_matrices(matrix_M, matrix_inv, p), "Перемножение M и M^-1")

u1 = (random.randint(10 ** 3, 10 ** 9), random.randint(10 ** 3, 10 ** 9))
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
        p               # Модуль
    )

    # Решаем вторую подсистему для m12 и m22
    m12, m22 = solve_mod_system(
        x11, x12, y12,  # Коэффициенты первого уравнения
        x21, x22, y22,  # Коэффициенты второго уравнения
        p               # Модуль
    )

    # Возвращаем восстановленную матрицу M
    return [[m11, m12], [m21, m22]]

# Функция для криптоанализа
def find_original_matrix(u1, c1, u2, c2, p):
    # Формируем матрицу X из открытых текстов
    X = [[u1[0], u1[1]], [u2[0], u2[1]]]
    # Формируем матрицу C из шифртекстов
    C = [[c1[0], c1[1]], [c2[0], c2[1]]]

    # Находим обратную матрицу для X по модулю p
    X_inv = inverse_matrix(X, p)

    # Вычисляем матрицу M как M = X_inv * C mod p
    M = multiply_matrices(X_inv, C, p)
    return M


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



print("\n" + "=" * 60 + "\n\nЗадание 4\n\n")

# Определяем алфавит
alphabet = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
n = len(alphabet)


# Функция для преобразования текста в числовую форму
def text_to_numbers(text):
    return [alphabet.index(char) for char in text]


# Функция для преобразования чисел в текст
def numbers_to_text(numbers):
    return "".join([alphabet[num] for num in numbers])


# Расширенный алгоритм Евклида для нахождения обратного элемента
def mod_inverse(a, n):
    g, x, _ = extended_gcd(a, n)
    if g != 1:
        return None  # Обратного элемента нет
    return x % n


# Расширенный алгоритм Евклида
def extended_gcd(a, b):
    if b == 0:
        return a, 1, 0
    else:
        g, x, y = extended_gcd(b, a % b)
        return g, y, x - (a // b) * y


# Функция расшифровки
def decrypt_affine(ciphertext, a, b):
    a_inv = mod_inverse(a, n)
    if a_inv is None:
        raise ValueError("Обратный элемент для 'a' не существует")

    ciphertext_numbers = text_to_numbers(ciphertext)
    plaintext_numbers = [(a_inv * (y - b)) % n for y in ciphertext_numbers]
    return numbers_to_text(plaintext_numbers)


# Шифртексты
ciphertext_1 = "ШЭКЧРЧЦБКЭДЦНЦНЭЦБЧЪЗОЭЪЭШЭСКХ"
ciphertext_2 = "СЙОЧЛУДЦФЁЬРЧЙЮЕЧДГЮЕСЁРЙСЁЯЁГ"


def check_combinations(combs, text):
    for c in combs:
        if c in text:
            return False
    return True


# Подбор ключей a и b
def brute_force_decrypt(ciphertext):
    for a in [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 23, 25, 26, 28, 29, 31, 32]:
        for b in range(n):
            try:
                plaintext = decrypt_affine(ciphertext, a, b)
                if check_combinations(["ЁЯ", "ЦЩ", "ХЯ"] + [n + "Ь" for n in "АОУИЕЙЪЭЫЯЮ"], plaintext):
                    print(f"a = {a}, b = {b}: {plaintext}")
            except ValueError:
                continue


# Запускаем подбор для первого шифртекста
print("Подбор ключей для первого шифртекста:")
brute_force_decrypt(ciphertext_1)

# Запускаем подбор для второго шифртекста
print("\nПодбор ключей для второго шифртекста:")
brute_force_decrypt(ciphertext_2)
