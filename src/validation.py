from itertools import product
from multiprocess.pool import Pool
from typing import Type, Any

from datatypes import BinaryClassifier, DataPoint, Label
from utils import zo_loss

def nested_cross_validation(
    Model: Type[BinaryClassifier], 
    X: list[DataPoint], 
    Y: list[Label], 
    K: int, 
    params: dict[str, list[Any]],
    log: bool = False
) -> tuple[dict[str, Any], float]:
    """
    Perform K-fold nested cross-validation

    Parameters
    ----------
    Model: Type[BinaryClassifier]
        The type of the model to test
    X: list[DataPoint]
        The list of datapoints
    Y: list[Label]
        The list of labels
    K: int
        the number of folds
    params: dict[str, list[Any]]
        pairs of parameter name and list of values. The internal validation
        is performed on each value of the cartesian product among the parameter values lists.
        Example: given { "a": [1,2], "b": [3,4], "c": [5] }, the validation will be performed
        considering models with the following parameters:
        a=1, b=3, c=5
        a=1, b=4, c=5
        a=2, b=3, c=5
        a=2, b=4, c=5

    Returns
    -------
    tuple[dict[str, Any], float]
        the best parameters configuration and the relative 0-1 loss
    """
    X_folds = _split(X, K)
    Y_folds = _split(Y, K)

    prod = list(product(*params.values()))
    keys = params.keys()

    configs = [dict(zip(keys, e)) for e in prod]

    best_loss = float('inf')
    best_param = {}

    for i in range(K):
        if log:
            print("fold", i)
        trainX = sum(X_folds[:i] + X_folds[i+1:], start=[])
        trainY = sum(Y_folds[:i] + Y_folds[i+1:], start=[])
        testX = X_folds[i]
        testY = Y_folds[i]

        with Pool() as pool:
            losses = pool.map(lambda p: cross_validation(Model(**p), trainX, trainY, K), configs)

        _, local_best_param = min(zip(losses, configs), key=lambda x: x[0])
        
        m = Model(**local_best_param)
        m.fit(trainX, trainY)
        predY = m.predict(testX)
        loss = zo_loss(testY, predY)

        if loss < best_loss:
            best_loss = loss
            best_param = local_best_param
    
    return best_param, best_loss

def cross_validation(
    model: BinaryClassifier,
    X: list[DataPoint],
    Y: list[Label],
    K: int
) -> float:
    """
    Perform external cross-validation

    Parameters
    ----------
    model: BinaryClassifier
        the classifier to evaluate
    X: list[DataPoint]
        The list of datapoints
    Y: list[Label]
        The list of labels
    K: int
        the number of folds

    Returns
    -------
    float
        the average 0-1 loss on the folds
    """
    X_folds = _split(X, K)
    Y_folds = _split(Y, K)

    loss_sum = 0

    for i in range(K):
        trainX = sum(X_folds[:i] + X_folds[i+1:], start=[])
        trainY = sum(Y_folds[:i] + Y_folds[i+1:], start=[])
        testX = X_folds[i]
        testY = Y_folds[i]
        model.fit(trainX, trainY)
        predY = model.predict(testX)
        loss_sum += zo_loss(testY, predY)
    return loss_sum / K

def _split(values: list, K: int):
    folds_size = len(values) // K
    folds = [values[folds_size*i : folds_size*(i+1)] for i in range(K)]

    remaining = len(values) % K

    if remaining != 0:
        for i in range(remaining):
            folds[i].append(values[-1-i])
    return folds
