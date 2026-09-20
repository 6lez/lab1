"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput

KK = "aeiou"

def count_vowels(data: TextInput) -> int:
    text = data.value
    return sum(1 for ch in text.lower() if ch in KK)

def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(set(text)) == len(text)

def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return bin(number).count("1")

def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0
    while number >= 10:
        product = 1
        while number:
            product *= number % 10
            number //= 10
        number = product
        steps += 1
    return steps

def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    return sum((x - y) ** 2 for x, y in zip(predicted, expected)) / len(predicted)

def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    parts: list[str] = []
    divisor = 2
    while divisor * divisor <= number:
        if number % divisor == 0:
            power = 0
            while number % divisor == 0:
                number //= divisor
                power += 1
            parts.append(f"({divisor})" if power == 1 else f"({divisor}**{power})")
        divisor += 1
    if number > 1:
        parts.append(f"({number})")
    return "".join(parts)


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    total = 0
    k = 0
    while total < cube_count:
        k += 1
        total += k * k
    return k if total == cube_count else "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    digits = [int(ch) for ch in str(data.value)]
    count = len(digits)
    middle = count // 2
    if count % 2:  # нечётное число цифр: одна средняя цифра не учитывается
        left, right = digits[:middle], digits[middle + 1:]
    else:  # чётное число цифр: две средние цифры не учитываются
        left, right = digits[:middle - 1], digits[middle + 1:]
    return sum(left) == sum(right)

print("=== Задача 1. count_vowels ===")
for text in ("abeba", "vifdudif", "", "bcdfg", "AEIOU", "AeIoU", "xyz"):
       print(f"count_vowels({text!r}) = {count_vowels(TextInput(value=text))}")

print("\n=== Задача 2. has_unique_characters ===")
for text in ("abefs", "vifdudif", "", "a", "aa", "abcABC"):
    print(f"has_unique_characters({text!r}) = {has_unique_characters(TextInput(value=text))}")

print("\n=== Задача 3. count_one_bits ===")
for number in (1, 2, 22, 312, 353, 255, 1024):
    print(f"count_one_bits({number}) = {count_one_bits(PositiveIntegerInput(value=number))}")

print("\n=== Задача 4. multiplicative_persistence ===")
for number in (4, 39, 999, 1, 10, 277777788888899):
    print(f"multiplicative_persistence({number}) = "
              f"{multiplicative_persistence(PositiveIntegerInput(value=number))}")

print("\n=== Задача 5. mse ===")
for pair in (((20, 20), (11, 23)), ((30, 10), (32, 44)), ((1, 2, 3), (1, 2, 3)), ((0,), (5,))):
    print(f"mse{pair} = {mse(VectorPairInput(predicted=pair[0], expected=pair[1]))}")

print("\n=== Задача 6. prime_factorization ===")
for number in (1, 2, 12, 86240, 21340, 997, 1024):
    print(f"prime_factorization({number}) = "
              f"{prime_factorization(PositiveIntegerInput(value=number))!r}")

print("\n=== Задача 7. pyramid ===")
for number in (1, 5, 10, 14, 30, 55, 2):
    print(f"pyramid({number}) = {pyramid(PositiveIntegerInput(value=number))!r}")

print("\n=== Задача 8. is_balanced_number ===")
for number in (1234, 12321, 4404, 1234006, 123456, 7, 11):
    print(f"is_balanced_number({number}) = "
              f"{is_balanced_number(PositiveIntegerInput(value=number))}")