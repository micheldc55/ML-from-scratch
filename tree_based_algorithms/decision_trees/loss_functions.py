import numpy as np 


class GiniImpurity:
    """Contains the entropy loss function. The entropy is a measure of the uncertainty of a random variable.
    """
    def __init__(self, criterion: str = "gini"):
        self.criterion = criterion

    def __call__(self, x: np.ndarray, y: np.array) -> float:
        raise NotImplementedError("Entropy loss function not implemented yet.")
    
    def _compute_categorical_probabilities(
            self, x: np.array, y: np.array, category: np.int64 or np.str_ or np.float64
        ) -> np.array:
        """Computes the probability of a given category.

        Args:
            x (np.array): The array that contains the feature for which we want to compute the impurity.
            y (np.array): Array that contains the target variable. We want to measure the impact of the
            split of feature x on the target variable y.
            category (np.int64 or np.str_ or np.float64): The category for which we want to compute the probability.

        Returns:
            np.array: The probability of a given category.
        """
        indexes = np.where(x == category)[0]
        target = y[indexes]

        _, target_counts = np.unique(target, return_counts=True)

        return target_counts / len(target)
    
    def _compute_gini_for_category(self, x: np.array, y: np.array) -> float:
        """Computes the entropy for a given category.

        Args:
            x (np.array): The array that contains the feature for which we want to compute the impurity.
            y (np.array): Array that contains the target variable. We want to measure the impact of the
            split of feature x on the target variable y.

        Returns:
            float: The floating point value of the Gini Impurity for a given category.
        """
        distinct_x, distinct_x_counts = np.unique(x, return_counts=True)

        # Compute the weight of each category.
        weights = distinct_x_counts / len(x)

        gini_impurity = np.array([])

        # Compute the gini impurity for each category.
        # TODO: Vectorize this loop.
        for idx, category in enumerate(distinct_x):
            prob = self._compute_categorical_probabilities(x, y, category)

            gini_impurity = np.append(gini_impurity, 1 - np.sum(prob ** 2))

        return np.sum(weights * gini_impurity)



