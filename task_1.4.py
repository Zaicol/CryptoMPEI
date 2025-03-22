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
    impossible_combinations = [n + "Ь" for n in "АГЕЁИЙОУЦЪЫЬЭЮЯ"]
    impossible_combinations += ["Ь" + n for n in "АЙЪЫЬЭ"]
    impossible_combinations += [n + "Ъ" for n in "АЕЁИЙОУЪЫЬЭЮЯ"]
    impossible_combinations += ["Ъ" + n for n in "АБГДЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭ"]
    impossible_combinations += [n + "Й" for n in "БВГДЖЗИЙКЛМНПРСТФХЦЧШЩЪЬ"]
    for a in [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 23, 25, 26, 28, 29, 31, 32]:
        for b in range(n):
            try:
                plaintext = decrypt_affine(ciphertext, a, b)
                if check_combinations(impossible_combinations, plaintext):
                    print(f"a = {a}, b = {b}: {plaintext}")
            except ValueError:
                continue


# Запускаем подбор для первого шифртекста
print("Подбор ключей для первого шифртекста:")
brute_force_decrypt(ciphertext_1)

# Запускаем подбор для второго шифртекста
print("\nПодбор ключей для второго шифртекста:")
brute_force_decrypt(ciphertext_2)


def encrypt_affine(plaintext, a=10, b=6):
    plaintext_numbers = text_to_numbers(plaintext)
    ciphertext_numbers = [(a * x + b) % n for x in plaintext_numbers]
    return numbers_to_text(ciphertext_numbers)


print(ciphertext_1 == encrypt_affine(decrypt_affine(ciphertext_1, 10, 6)))

print(mod_inverse(7, n))