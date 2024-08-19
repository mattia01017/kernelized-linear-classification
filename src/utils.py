from functools import reduce
from itertools import combinations_with_replacement
from typing import Sequence, Iterable
from datatypes import Label, DataPoint


def zo_loss(Y: Sequence[Label], Y_pred: Sequence[Label]):
    assert len(Y) == len(Y_pred), f"Vectors {Y} and {Y_pred} are not of the same length"
    return sum(1 if y != y_pred else 0 for y, y_pred in zip(Y, Y_pred)) / len(Y)


def dot_prod(a: Sequence[float], b: Sequence[float]) -> float:
    assert len(a) == len(b), f"Vectors {a} and {b} are not of the same length"
    return reduce(lambda p, c: p + float(c[0]) * float(c[1]), zip(a, b), 0.0)


def norm(a: Iterable[float]) -> float:
    return reduce(lambda p, c: p + c**2, a, 0)


def quadratic_extraction(a: DataPoint) -> DataPoint:
    return (1.0,) +  tuple(
        a[i] * a[j] for i, j in combinations_with_replacement(range(len(a)), 2)
    )