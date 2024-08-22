from functools import reduce
from math import sqrt
from typing import Sequence, Iterable
from datatypes import Label, DataPoint


def zo_loss(Y: Sequence[Label], Y_pred: Sequence[Label]) -> float:
    """
    Compute the zero-one loss

    Parameters
    ----------
    Y: Sequence[Label]
        the actual labels
    Y_pred: Sequence[Label]
        the predicted labels

    Returns
    -------
    float:
        the average zero-one loss
    """
    assert len(Y) == len(Y_pred), f"Vectors {Y} and {Y_pred} are not of the same length"
    return sum(1 for y, y_pred in zip(Y, Y_pred) if y != y_pred) / len(Y)


def dot_prod(V: Sequence[float], U: Sequence[float]) -> float:
    """
    Compute the dot product

    Parameters
    ----------
    V: Sequence[float]
        the first vector
    U: Sequence[float]
        the second vector

    Returns
    -------
    float:
        the result of the dot product
    """
    assert len(V) == len(U), f"Vectors {V} and {U} are not of the same length"
    return sum(float(v) * float(u) for v, u in zip(V, U))


def scalar_prod(k: float, V: Sequence[float]) -> list[float]:
    """
    Compute the product between a scalar and a vector

    Parameters
    ----------
    k: float
        the scalar
    V: Sequence[float]
        the vector

    Returns
    -------
    float:
        the result of the product
    """
    return [k * v for v in V]


def norm(V: Iterable[float]) -> float:
    """
    Compute the euclidean norm

    Parameters
    ----------
    V: Iterable[float]
        the vector

    Returns
    -------
    float:
        the norm of V
    """
    return sqrt(sum(v**2 for v in V))


def quadratic_extraction(x: DataPoint) -> DataPoint:
    """
    Compute the quadratic feature extraction

    Parameters
    ----------
    x: DataPoint
        the data point to map

    Returns
    -------
    DataPoint
        the expanded vector
    """
    return (
        (1.0,)
        + tuple(x)
        + tuple(x[i] * x[j] for i in range(len(x)) for j in range(i, len(x)))
    )


def remove_outliers(trainX, trainY, alpha=1.5):
    for i in range(len(trainX[0])):
        sortedXi = sorted(trainX, key=lambda x: x[i])
        q1 = sortedXi[round(len(trainX) * 0.25)][i]
        q3 = sortedXi[round(len(trainX) * 0.75)][i]
        iqr = q3 - q1
        lb = q1 - alpha * iqr
        ub = q3 + alpha * iqr
        filtered = list(filter(lambda x: lb <= x[0][i] <= ub, zip(trainX, trainY)))
        trainX = [x[0] for x in filtered]
        trainY = [x[1] for x in filtered]

    return trainX, trainY


class ZScore:
    """
    Helper class for performing standardization on dataset
    """
    def __init__(self) -> None:
        self.mean = []
        self.std = []
    
    def fit_transform(self, X: list[DataPoint]) -> list[DataPoint]:
        """
        Apply standardization using mean and standard deviation of provided
        data

        Parameters
        ----------
        X: list[DataPoint]
            the list of datapoints to standardize

        Returns
        -------
        list[DataPoint]
            the standardized data 
        """
        self.mean = [self._mean([x[i] for x in X]) for i in range(len(X[0]))]
        self.std = [self._std([x[i] for x in X], self.mean[i]) for i in range(len(X[0]))]
        return self.transform(X)
        

    def transform(self, X: list[DataPoint]) -> list[DataPoint]:
        """
        Apply standardization considering mean and standard deviation only of
        data provided with fit_transform method

        Parameters
        ----------
        X: list[DataPoint]
            the list of datapoints to standardize

        Returns
        -------
        list[DataPoint]
            the standardized data 
        """
        return [
            tuple((x[i] - self.mean[i]) / (self.std[i] if self.std[i] != 0 else 1) for i in range(len(X[0])))
            for x in X
        ]

    def _mean(self, xi: Sequence[float]):
        return sum(xi) / len(xi)
    
    def _std(self, xi: Sequence[float], mean_i: float):
        return sum((e - mean_i)**2 for e in xi) / (len(xi)-1)
