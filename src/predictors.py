from math import log, sqrt, exp
from utils import dot_prod, norm
from datatypes import Label, DataPoint, BinaryClassifier
from typing import Literal, Callable
import random
from collections import Counter

class Perceptron(BinaryClassifier):
    """
    The perceptron in the linearly separable case

    Parameters
    ----------
    epochs: int
        the maximum number of scans of the entire dataset
    expansion: Callable[[DataPoint], DataPoint], default=None
        a function that perform feature expansion. It's applied before training 
        and before predicting. If None, no expansion is applied

    Attributes
    ----------
    w: list[float] | None
        the weight vector. When the model isn't fitted yet, 
        the vector length is 0
    """

    def __init__(
        self,
        epochs: int = 1000,
        expansion: Callable[[DataPoint], DataPoint] | None = None,
    ) -> None:
        self.expansion = expansion
        self.epochs = epochs
        self.w = []

    def fit(self, X: list[DataPoint], Y: list[Label], warm_start: bool = False) -> None:
        if self.expansion:
            X = [self.expansion(x) for x in X]
        if not warm_start:
            self.w = [0.0] * len(X[0])

        for _ in range(self.epochs):
            updated = False
            for xt, yt in zip(X, Y):
                if float(yt) * dot_prod(self.w, xt) <= 0:
                    updated = True
                    self._update_w(xt, yt)
            if not updated:
                print("Convergence")
                break

    def predict(self, X: list[DataPoint]) -> list[Label]:
        X = X if self.expansion is None else [self.expansion(x) for x in X]
        return [1 if dot_prod(xt, self.w) >= 0 else -1 for xt in X]

    def _update_w(self, xt: DataPoint, yt: Label):
        for i in range(len(self.w)):
            self.w[i] += float(yt) * float(xt[i])


class SVM(BinaryClassifier):
    """
    Support vector machines for binary classification

    Parameters
    ----------
    epochs: int
        the number of steps of Pegasos
    learning_rate: float, default=0.02
        the learning rate
    regularization: float, default=0.0001
        the weight of the regularization parameter, that is half the squared norm 
        of the weight vector
    loss_func: "hinge" | "logistic", default="hinge"
        the loss function to optimize in the training phase
    expansion: Callable[[DataPoint], DataPoint], default=None
        a function that perform feature expansion. It's applied before training 
        and before predicting. If None, no expansion is applied

    Attributes
    ----------
    w: list[float] | None
        the weight vector. When the model isn't fitted yet, 
        the vector length is 0
    """

    def __init__(
        self,
        epochs: int = 1000,
        learning_rate: float = 0.02,
        regularization: float = 0.0001,
        loss_func: Literal["hinge"] | Literal["logistic"] = "hinge",
        expansion: Callable[[DataPoint], DataPoint] | None = None,
        random: random.Random = random.Random()
    ) -> None:
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.loss_func = loss_func
        self.expansion = expansion
        self.w = []
        self._curr_w = []
        self._iter_count = 0
        self._rand = random

    def fit(self, X: list[DataPoint], Y: list[Label], warm_start: bool = False) -> None:
        size = len(X[0] if self.expansion is None else self.expansion(X[0]))
        if not warm_start:
            self.w = [0.0] * size
            self._curr_w = [0.0] * size
            self._iter_count = 0
        cumulative_w = [0.0] * size

        for t in range(self._iter_count, self._iter_count + self.epochs):
            z = self._rand.randint(0, len(X) - 1)
            new_X = X[z] if self.expansion is None else self.expansion(X[z])

            for i in range(len(X[z])):
                if self.loss_func == "hinge":
                    loss_grad = (
                        -new_X[i] * float(Y[z])
                        if float(Y[z]) * self._curr_w[i] * new_X[i] < 1
                        else 0.0
                    )
                else:
                    exponent = Y[z] * self._curr_w[i] * new_X[i]
                    loss_grad = -Y[z] * new_X[i] / ((1 + exp(709 if exponent > 709 else exponent)) * log(2))

                self._curr_w[i]  = self._curr_w[i] - (self.learning_rate / sqrt(t + 1)) * (
                    loss_grad + self.regularization * self._curr_w[i]
                )

            w_norm = norm(self._curr_w)
            for i in range(len(self._curr_w)):
                self._curr_w[i] /= w_norm
                cumulative_w[i] += self._curr_w[i]

        self._iter_count += self.epochs

        for i in range(len(self.w)):
            cumulative_w[i] /= self.epochs

        for i in range(len(self.w)):
            self.w[i] = ((self._iter_count - self.epochs) / self._iter_count) * self.w[i] + (self.epochs / self._iter_count) * cumulative_w[i]

    def predict(self, X: list[DataPoint]) -> list[Label]:
        X = X if self.expansion is None else [self.expansion(x) for x in X]
        return [1 if dot_prod(xt, self.w) >= 0 else -1 for xt in X]


class KernelPerceptron(BinaryClassifier):
    """
    The kernel perceptron algorithm

    Parameters
    ----------
    kernel: Callable[[DataPoint, DataPoint], float]
        the kernel function K(x, x')
    
    Attributes
    ----------
    S: list[tuple[DataPoint, Label]]
        the list of couples (Datapoint, Label) filled during the training phase
        when a label isn't predicted correctly
    """

    def __init__(
        self, 
        kernel: Callable[[DataPoint, DataPoint], float]
    ) -> None:
        self.kernel = kernel
        self.S: list[tuple[DataPoint, Label]] = []

    def fit(self, X: list[DataPoint], Y: list[Label], warm_start: bool = False) -> None:
        if warm_start:
            self.S.extend(
                [(xt, yt) for xt, yt in zip(X, Y) if self._compute_prediction(xt) != yt]
            )
        else:
            self.S = [
                (xt, yt) for xt, yt in zip(X, Y) if self._compute_prediction(xt) != yt
            ]

    def _compute_prediction(self, xt: DataPoint) -> Label:
        return 1 if sum(map(lambda z: z[1] * self.kernel(z[0], xt), self.S)) >= 0 else -1

    def predict(self, X: list[DataPoint]) -> list[Label]:
        return [self._compute_prediction(xt) for xt in X]


class KernelSVM(BinaryClassifier):
    """
    The kernel version of SVM

    Parameters
    ----------
    kernel: Callable[[DataPoint, DataPoint], float]
        the kernel function K(x, x')
    epochs: int, default=1000
        the number of iterations of Pegasos
    regularization: float, default=1.0
        the weight of the regularization parameter, that is half the squared norm 
        of the weight vector

    Attributes
    ----------
    alpha: Counter[tuple[DataPoint, Label]]
        A counter of couples (DataPoint, Label) incremented during the
        training phase when a mistake is found 
    """
    def __init__(
        self,
        kernel: Callable[[DataPoint, DataPoint], float],
        epochs: int = 1000,
        regularization: float = 0.0001,
    ) -> None:
        self.kernel = kernel
        self.epochs = epochs
        self.regularization = regularization
        self.alpha: Counter[DataPoint] = Counter()
        self.iter_count = 0

    def fit(self, X: list[DataPoint], Y: list[Label], warm_start: bool = False) -> None:
        if not warm_start:
            self.iter_count = 0
            self.alpha = Counter()
        for t in range(self.iter_count, self.iter_count + self.epochs):
            z = random.randint(0, len(X) - 1)
            pred = self._compute_prediction(X[z]) * Y[z] / (self.regularization * (t+1))

            if pred < 1:
                self.alpha[X[z]] += Y[z]
        
        self.iter_count += self.epochs

    def _compute_prediction(self, xt: DataPoint) -> float:
        return sum(map(
            lambda e: e[1] * self.kernel(xt, e[0]),
            self.alpha.items(),
        ))
    
    def predict(self, X: list[DataPoint]) -> list[Label]:
        return [1 if self._compute_prediction(xt) > 0 else -1 for xt in X]

