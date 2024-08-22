from utils import dot_prod, norm
from typing import Callable
from math import exp
from typing import Sequence


def gaussian(gamma: float) -> Callable[[Sequence[float], Sequence[float]], float]:
    """
    Create a gaussian kernel function

    Parameters
    ----------
    gamma: float
        the gamma parameter of the gaussian kernel

    Returns
    -------
    Callable[[Sequence[float], Sequence[float]], float]
        the kernel function
    """
    return lambda x, xp: exp(
        -norm((x[i] - xp[i] for i in range(len(x)))) ** 2 / (2 * gamma)
    )


def polynomial(degree: int) -> Callable[[Sequence[float], Sequence[float]], float]:
    """
    Create a polynomial kernel function

    Parameters
    ----------
    degree: int
        the degree of the polynomial kernel

    Returns
    -------
    Callable[[Sequence[float], Sequence[float]], float]
        the kernel function
    """
    return lambda x, xp: (1 + dot_prod(x, xp)) ** degree
