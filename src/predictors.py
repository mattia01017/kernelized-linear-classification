import json
from math import log, sqrt, exp
from utils import dot_prod, scalar_prod, norm
from datatypes import Label, DataPoint, BinaryClassifier
from typing import Literal, Callable
import random
from collections import Counter

class Perceptron(BinaryClassifier):
    """
    The perceptron in the linearly separable case

    Parameters
    ----------
    epochs_per_step: int
        the maximum number of scans of the entire dataset in a fit method call
    expansion: Callable[[DataPoint], DataPoint], default=None

    Attributes
    ----------
    w: list[float] | None
        the weight vector. When the model isn't fitted yet, 
        the vector length is 0
    """

    def __init__(
        self,
        epochs_per_step: int = 1000,
    ) -> None:
        self.epochs_per_step = epochs_per_step
        self.w = []
        self.epochs = 0

    def fit(self, X: list[DataPoint], Y: list[Label], warm_start: bool = False) -> None:
        assert len(X) == len(Y)
        if not warm_start:
            self.w = [0.0] * len(X[0])
            self.epochs = 0

        for _ in range(self.epochs_per_step):
            updated = False
            for xt, yt in zip(X, Y):
                if float(yt) * dot_prod(self.w, xt) <= 0:
                    updated = True
                    self._update_w(xt, yt)
            if not updated:
                print("Convergence")
                break
        
        self.epochs += self.epochs_per_step

    def predict(self, X: list[DataPoint]) -> list[Label]:
        return [1 if dot_prod(xt, self.w) >= 0 else -1 for xt in X]

    def _update_w(self, xt: DataPoint, yt: Label):
        for i in range(len(self.w)):
            self.w[i] += float(yt) * float(xt[i])


class SVM(BinaryClassifier):
    """
    Support vector machines for binary classification

    Parameters
    ----------
    epochs_per_step: int
        the number of iterations of Pegasos performed in a fit method call
    learning_rate: float, default=0.02
        the learning rate
    regularization: float, default=0.0001
        the weight of the regularization parameter, that is half the squared norm 
        of the weight vector
    loss_func: "hinge" | "logistic", default="hinge"
        the loss function to optimize in the training phase
    rand: random.Random, default=random.Random()
        The instance of random variable generator to use

    Attributes
    ----------
    w: list[float] | None
        the weight vector. When the model isn't fitted yet, 
        the vector length is 0
    iter_count: int
        the total number of iterations performed. The counter is reset
        with fit method call with warm_start=False
    """

    def __init__(
        self,
        epochs_per_step: int = 1000,
        learning_rate: float = 0.02,
        regularization: float = 0.0001,
        loss_func: Literal["hinge"] | Literal["logistic"] = "hinge",
        random: random.Random = random.Random()
    ) -> None:
        self.epochs_per_step = epochs_per_step
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.loss_func = loss_func
        self.w = []
        self._curr_w = []
        self.iter_count = 0
        self._rand = random

    def fit(self, X: list[DataPoint], Y: list[Label], warm_start: bool = False) -> None:
        assert len(X) == len(Y)
        if not warm_start:
            self.w = [0.0] * len(X[0])
            self._curr_w = [0.0] * len(X[0])
            self.iter_count = 0
        cumulative_w = [0.0] * len(X[0])

        for t in range(self.iter_count, self.iter_count + self.epochs_per_step):
            z = self._rand.randint(0, len(X) - 1)

            if self.loss_func == "hinge":
                loss_grad = self._hinge_gradient(X[z], Y[z], self._curr_w)
            else:
                loss_grad = self._logistic_gradient(X[z], Y[z], self._curr_w)

            update_step = scalar_prod(
                (self.learning_rate / sqrt(t + 1)), 
                [loss_grad[i] + self.regularization * self._curr_w[i] for i in range(len(loss_grad))]
            )

            for i in range(len(self._curr_w)):
                self._curr_w[i] -= update_step[i]

            for i in range(len(self._curr_w)):
                cumulative_w[i] += self._curr_w[i]

        self.iter_count += self.epochs_per_step

        for i in range(len(self.w)):
            cumulative_w[i] /= self.epochs_per_step

        for i in range(len(self.w)):
            self.w[i] = ((self.iter_count - self.epochs_per_step) / self.iter_count) * self.w[i] + (self.epochs_per_step / self.iter_count) * cumulative_w[i]

    def _hinge_gradient(self, x: DataPoint, y: Label, w: list[float]) -> list[float]:
        return (
            scalar_prod(-y, x)
            if float(y) * dot_prod(w, x) < 1
            else [0.0] * len(x)
        )

    def _logistic_gradient(self, x: DataPoint, y: Label, w: list[float]) -> list[float]:
        exponent = y * dot_prod(w, x)
        sigma = ((1 + exp(709 if exponent > 709 else exponent)) * log(2))
        loss_grad = [-y * xi / sigma for xi in x]
        return loss_grad

    def predict(self, X: list[DataPoint]) -> list[Label]:
        return [1 if dot_prod(xt, self.w) >= 0 else -1 for xt in X]


class KernelPerceptron(BinaryClassifier):
    """
    The kernel perceptron algorithm

    Parameters
    ----------
    kernel: Callable[[DataPoint, DataPoint], float]
        the kernel function K(x, x')
    epochs_per_step: int, default=1
        the number of epochs_per_step in a fit method call
    
    Attributes
    ----------
    S: list[tuple[DataPoint, Label]]
        the list of couples (Datapoint, Label) filled during the training phase
        when a label isn't predicted correctly
    iter_count: int
        the total number of iterations performed. The counter is reset
        with fit method call with warm_start=False
    """

    def __init__(
        self, 
        kernel: Callable[[DataPoint, DataPoint], float],
        epochs_per_step: int = 1
    ) -> None:
        self._kernel = kernel
        self.epochs_per_step = epochs_per_step
        self.S: Counter[DataPoint] = Counter()
        self.iter_count = 0

    def fit(self, X: list[DataPoint], Y: list[Label], warm_start: bool = False) -> None:
        assert len(X) == len(Y)
        if not warm_start:
            self.S = Counter()
        for _ in range(self.epochs_per_step):
            for xt, yt in zip(X, Y):
                if yt != self._compute_prediction(xt):
                    self.S[xt] += yt

        self.iter_count += self.epochs_per_step

    def _compute_prediction(self, xt: DataPoint) -> Label:
        return 1 if sum(map(lambda z: z[1] * self._kernel(z[0], xt), self.S.items())) >= 0 else -1

    def predict(self, X: list[DataPoint]) -> list[Label]:
        return [self._compute_prediction(xt) for xt in X]


class KernelSVM(BinaryClassifier):
    """
    The kernel version of SVM

    Parameters
    ----------
    kernel: Callable[[DataPoint, DataPoint], float]
        the kernel function K(x, x')
    epochs_per_step: int, default=1000
        the number of iterations of Pegasos in a single fit method call
    regularization: float, default=1.0
        the weight of the regularization parameter, that is half the squared norm 
        of the weight vector
    rand: random.Random, default=random.Random()
        The instance of random variable generator to use

    Attributes
    ----------
    alpha: Counter[DataPoint]
        A counter associated to data points incremented during the
        training phase when a mistake is found
    """
    def __init__(
        self,
        kernel: Callable[[DataPoint, DataPoint], float],
        epochs_per_step: int = 1000,
        regularization: float = 0.0001,
        rand: random.Random = random.Random()
    ) -> None:
        self.kernel = kernel
        self.epochs_per_step = epochs_per_step
        self.regularization = regularization
        self.rand = rand
        self.alpha: Counter[DataPoint] = Counter()
        self.iter_count = 0

    def fit(self, X: list[DataPoint], Y: list[Label], warm_start: bool = False) -> None:
        assert len(X) == len(Y)
        if not warm_start:
            self.iter_count = 0
            self.alpha = Counter()
        for t in range(self.iter_count, self.iter_count + self.epochs_per_step):
            z = self.rand.randint(0, len(X) - 1)
            pred = self._compute_prediction(X[z]) * Y[z] / (self.regularization * (t+1))

            if pred < 1:
                self.alpha[X[z]] += Y[z]
        
        self.iter_count += self.epochs_per_step

    def _compute_prediction(self, xt: DataPoint) -> float:
        return sum(ys * self.kernel(xt, xs) for xs, ys in self.alpha.items())
    
    def predict(self, X: list[DataPoint]) -> list[Label]:
        return [1 if self._compute_prediction(xt) > 0 else -1 for xt in X]

