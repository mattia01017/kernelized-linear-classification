from abc import ABC, abstractmethod
from typing import Literal, Sequence

Label = Literal[1] | Literal[-1]
DataPoint = Sequence[float]


class BinaryClassifier(ABC):
    """
    Abstract class of a generic binary classifier
    """

    @abstractmethod
    def fit(self, X: list[DataPoint], Y: list[Label], warm_start: bool = False) -> None:
        """
        Takes data points and labels of a dataset for learning

        Parameters
        ----------
        X: list[DataPoint]
            the list of data points
        Y: list[Prediction]
            the list of labels associated to data points
        warm_start: bool, default=False
            if true, the model won't be reinitialized before training. If this method was never
            called before on this model there could be unexpected behaviors if warm_start is set
            to True

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def predict(self, X: list[DataPoint]) -> list[Label]:
        """
        Makes predictions on the given data points. Can be called always 
        after fit method

        Parameters
        ----------
        X: list[DataPoint]
            the list of data points.

        Returns
        -------
        list[Prediction]
            the list of predicted labels
        """
        pass
