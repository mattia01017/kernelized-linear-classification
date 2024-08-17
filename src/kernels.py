from utils import dot_prod, norm
from typing import Callable
from math import exp
from typing import Sequence


def gaussian(gamma: float) -> Callable[[Sequence[float], Sequence[float]], float]:
    return lambda x, xp: exp(-norm((x[i] - xp[i] for i in range(len(x))))) ** 2 / (2 * gamma)


def polynomial(degree: int) -> Callable[[Sequence[float], Sequence[float]], float]:
    return lambda x, xp: (1 + dot_prod(x, xp)) ** degree
