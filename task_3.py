# Преобразование Уолша
def walsh_transform(a, N):
    w = a[:]  # Копируем список для преобразования
    for n in range(1, N + 1):  # Проходим по всем уровням преобразования
        for i in range(0, 2 ** (N - n)):
            for j in range(2 ** (n - 1)):
                t = 2 ** n * i + j
                tt = t + 2 ** (n - 1)
                if tt < len(w):  # Проверяем, чтобы индексы не выходили за пределы
                    d = w[t]
                    w[t] = d + w[tt]
                    w[tt] = d - w[tt]
    return w


# Преобразование Фурье (использует преобразование Уолша)
def fourier_transform(f, N):
    return walsh_transform(f, N)


# Вычисление статистической структуры функции
def statistical_structure(a, N):
    d = [(a[2 * j] + a[2 * j + 1]) // 2 for j in range(2 ** (N - 1))]
    for n in range(2, N + 1):
        for i in range(0, 2 ** (N - n)):
            for j in range(2 ** (n - 1)):
                t = 2 ** n * i + j
                tt = t + 2 ** (n - 1)
                # Проверяем, чтобы индексы не выходили за пределы
                if t < len(d) and tt < len(d):
                    d[t], d[tt] = d[t] + d[tt], d[t] - d[tt]
    return d


# Линейная функция от куба
def linear_function(cube, u):
    return [(sum(int(ci) * int(ui) for ci, ui in zip(it, u)) % 2) for it in cube]


# Аффинная функция от куба
def affine_function(cube, u):
    return [(1 + sum(int(ci) * int(ui) for ci, ui in zip(it, u))) % 2 for it in cube]


# Вычисление расстояний Хемминга до линейных и аффинных функций
def hamming_distance(f, cube):
    hdl, hda = [], []
    for it in cube:
        l = linear_function(cube, it)
        # Расстояние до линейной функции
        wl = sum((l[i] + f[i]) % 2 for i in range(len(f)))
        # Расстояние до аффинной функции
        wa = sum((l[i] + f[i] + 1) % 2 for i in range(len(f)))
        hdl.append(wl)
        hda.append(wa)
    return hdl, hda


# Генерация куба истинности для N переменных
def generate_cube(N):
    return [bin(i)[2:].zfill(N) for i in range(2 ** N)]


# Булева функция для 6 переменных
def boolean_function(cube):
    return [(int(it[0]) ^ (int(it[1]) * int(it[2])) ^ (int(it[3]) * int(it[4]) * int(it[5]))) for it in cube]


# Функция для преобразования сигнального вектора в характеристическую последовательность
def sign_to_char(f):
    return [(-1) ** bit for bit in f]


N = 6  # Число переменных
cube = generate_cube(N)  # Генерация куба истинности для 6 переменных
f = boolean_function(cube)  # Вычисление сигнального вектора
F = sign_to_char(f)  # Вычисление характеристической последовательности

w = walsh_transform(F, N)  # Преобразование Уолша
fft_result = fourier_transform(f, N)  # Преобразование Фурье
d = statistical_structure(F, N)  # Статистическая структура
hd = hamming_distance(f, cube)  # Расстояния Хемминга

# Вывод результатов
print("Преобразование Уолша:", w)
print("Преобразование Фурье:", fft_result)
print("Статистическая структура:", d)
print("Расстояния Хемминга до линейных функций:", hd[0])
print("Расстояния Хемминга до аффинных функций:", hd[1])
