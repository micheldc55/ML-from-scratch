import numpy as np
import pandas as pd

from tree_based_algorithms.decision_trees.loss_functions import GiniImpurity, Entropy


class DecisionTreeClassifier:
    """A decision tree classifier.

    Args:
        max_depth (int): The maximum depth of the tree.
        min_samples_split (int): The minimum number of samples required to split an internal node.
        min_samples_leaf (int): The minimum number of samples required to be at a leaf node.
        loss_function (str): The loss function to use when calculating impurity.
    """

    def __init__(
            self,
            max_depth: int = 3,
            min_samples_split: int = 2,
            min_samples_leaf: int = 1,
            criterion: str = "gini"
        ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.loss_function = self._implement_loss_function(criterion)

        self._tree = None

    @staticmethod
    def _implement_loss_function(criterion: str) -> callable:
        """Implements the loss function.

        Args:
            criterion (str): The loss function to use when calculating impurity.

        Returns:
            callable: The loss function.
        """
        if criterion == "gini":
            return GiniImpurity()
        elif criterion == "entropy":
            return Entropy()
        else:
            raise NotImplementedError("Loss function not implemented yet.")

    def fit(self, X: np.ndarray, y: np.array) -> None:
        """Build a decision tree classifier from the training set (X, y).

        Args:
            X (np.ndarray): The training input samples.
            y (np.array): The target values.
        """
        pass