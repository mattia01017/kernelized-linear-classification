from functools import reduce
from itertools import combinations_with_replacement
from typing import Sequence, Iterable
from datatypes import Label


def zo_loss(Y: list[Label], Y_pred: list[Label]):
    return sum(1 if y != y_pred else 0 for y, y_pred in zip(Y, Y_pred)) / len(Y)


def dot_prod(a: Iterable[float], b: Iterable[float]) -> float:
    return reduce(lambda p, c: p + float(c[0]) * float(c[1]), zip(a, b), 0.0)


def norm(a: Iterable[float]) -> float:
    return reduce(lambda p, c: p + c**2, a, 0)


def quadratic_expansion(a: Sequence[float]) -> tuple[float, ...]:
    return tuple(
        a[i] * a[j] for i, j in combinations_with_replacement(range(len(a)), 2)
    )