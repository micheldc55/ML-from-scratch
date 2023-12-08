import numpy as np 


class GiniImpurity:
    """Contains the entropy loss function. The entropy is a measure of the uncertainty of a random variable.
    """
    def __init__(self):
        pass

    def __call__(self, X: np.ndarray, y: np.array) -> float:
        """Computes the Gini Impurity for a given feature.

        Args:
            X (np.ndarray): The array that contains the feature for which we want to compute the impurity.
            y (np.array): Array that contains the target variable. We want to measure the impact of the
            split of feature X on the target variable y.

        Returns:
            float: The floating point value of the Gini Impurity for a given feature.
        """
        gini_impurities = np.array([])

        for col in range(X.shape[1]):
            gini = self._compute_gini_for_category(X[:, col], y)
            gini_impurities = np.append(gini_impurities, gini)

        return gini_impurities
    
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


class Entropy:
    """Contains the entropy loss function. The entropy is a measure of the uncertainty of a random variable.
    """
    pass

    def __init__(self):
        raise NotImplementedError("Entropy loss function not implemented yet.")


if __name__ == "__main__":
    import pandas as pd

    df_test = pd.DataFrame({
        "a": [1, 1, 1, 1, 2, 2, 2, 2],
        "b": [1, 1, 1, 0, 1, 0, 0, 0],
        "y": [1, 1, 0, 0, 1, 0, 0, 1]
    })

    X = df_test[["a", "b"]].values
    y = df_test["y"].values

    gini_impurity = GiniImpurity()

    impurity = gini_impurity(X, y)

    print(impurity)