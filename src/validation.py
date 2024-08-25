from itertools import product
from multiprocess.pool import Pool
from typing import Callable, Type, Any

from datatypes import BinaryClassifier, DataPoint, Label
from utils import remove_outliers, zo_loss, ZScore

def grid_search_cv(
    Model: Type[BinaryClassifier], 
    X: list[DataPoint], 
    Y: list[Label], 
    K: int, 
    params: dict[str, list[Any]],
    Scaler: Type[ZScore] | None = None,
    extraction: Callable[[DataPoint], DataPoint] | None = None,
) -> tuple[dict[str, Any], float]:
    """
    Perform a grid search, evaluating models through cross validation

    Parameters
    ----------
    Model: Type[BinaryClassifier]:
        the model type to build and evaluate
    X: list[DataPoint]:
        The data points
    Y: list[Label]: 
        The labels associated to data points
    K: int:
        The number of folds to use for cross validation
    params: dict[str, list[Any]]:
        pairs of parameter name and list of values. The search evauluate
        each value of the cartesian product among the parameter values lists.
        Example: given { "a": [1,2], "b": [3,4], "c": [5] }, the evaluation will be performed
        considering models with the following parameters:
        a=1, b=3, c=5
        a=1, b=4, c=5
        a=2, b=3, c=5
        a=2, b=4, c=5
    Scaler: Type[ZScore] | None, default=None,
        The scaler to construct and apply on data for normalization/standardization
    extraction: Callable[[DataPoint], DataPoint], default=None
        The function used to map data points in different feature spaces
    
    Returns
    -------
    tuple[dict[str, Any], float]
        key value pairs respresenting the best parameters configuration found and the loss.
    """
    prod = list(product(*params.values()))
    keys = params.keys()

    configs = [dict(zip(keys, e)) for e in prod]

    with Pool() as pool:
        losses = pool.map(
            lambda p: cross_validation(Model(**p), X, Y, K, Scaler, extraction),
            configs
        )
    
    return min(zip(configs, losses), key=lambda x: x[1])
    

def nested_cross_validation(
    Model: Type[BinaryClassifier], 
    X: list[DataPoint], 
    Y: list[Label], 
    K: int, 
    params: dict[str, list[Any]],
    Scaler: Type[ZScore] | None = None,
    extraction: Callable[[DataPoint], DataPoint] | None = None,
    log: bool = False,
) -> float:
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
    Scaler: Type[ZScore] | None, default=None,
        The scaler to construct and apply on data for normalization/standardization
    extraction: Callable[[DataPoint], DataPoint], default=None
        The function used to map data points in different feature spaces

    Returns
    -------
    float
        the average losses 
    """
    X_folds = _split(X, K)
    Y_folds = _split(Y, K)

    prod = list(product(*params.values()))
    keys = params.keys()

    configs = [dict(zip(keys, e)) for e in prod]

    loss = 0
    for i in range(K):
        if log:
            print("fold", i, end="\r")
        trainX = sum(X_folds[:i] + X_folds[i+1:], start=[])
        trainY = sum(Y_folds[:i] + Y_folds[i+1:], start=[])
        testX = X_folds[i]
        testY = Y_folds[i]

        with Pool() as pool:
            losses = pool.map(lambda p: cross_validation(Model(**p), trainX, trainY, K, Scaler, extraction), configs)

        _, local_best_param = min(zip(losses, configs), key=lambda x: x[0])
        
        m = Model(**local_best_param)
        m.fit(trainX, trainY)
        predY = m.predict(testX)
        loss += zo_loss(testY, predY)

    if log:
        print()
    return loss / K

def cross_validation(
    model: BinaryClassifier,
    X: list[DataPoint],
    Y: list[Label],
    K: int,
    Scaler: Type[ZScore] | None = None,
    extraction: Callable[[DataPoint], DataPoint] | None = None,
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
    Scaler: Type[ZScore] | None, default=None,
        The scaler to construct and apply on data for normalization/standardization
    extraction: Callable[[DataPoint], DataPoint], default=None
        The function used to map data points in different feature spaces

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

        if Scaler:
            trainX, trainY = remove_outliers(trainX, trainY)
            scaler = Scaler()
            trainX = scaler.fit_transform(trainX)
            testX = scaler.transform(testX)
        
        if extraction:
            trainX = [extraction(x) for x in trainX]
            testX = [extraction(x) for x in testX]

        model.fit(trainX, trainY)
        predY = model.predict(testX)
        loss_sum += zo_loss(testY, predY)
    return loss_sum / K

def _split(values: list, K: int):
    folds_size = len(values) // K
    folds = [values[folds_size*i : folds_size*(i+1)] for i in range(K)]

    remaining = len(values) % K

    for i in range(remaining):
        folds[i].append(values[-1-i])
    return folds
