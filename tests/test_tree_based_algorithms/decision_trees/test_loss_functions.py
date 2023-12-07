import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from tree_based_algorithms.decision_trees.loss_functions import GiniImpurity


class TestGiniImpurity(unittest.TestCase):

    def setUp(self) -> None:
        self.gini_impurity = GiniImpurity()

    def test_compute_categorical_probabilities(self):
        x = np.array([1, 1, 1, 1, 2, 2, 2, 2])
        y = np.array([1, 1, 0, 0, 1, 0, 0, 1])

        probabilities = self.gini_impurity._compute_categorical_probabilities(x, y, 1)

        self.assertTrue(np.allclose(probabilities, np.array([0.5, 0.5])))

    def test_compute_gini_for_category(self):
        x = np.array([1, 1, 1, 1, 2, 2, 2, 2])
        y = np.array([1, 1, 0, 0, 1, 0, 0, 1])

        gini_impurity = self.gini_impurity._compute_gini_for_category(x, y)

        self.assertTrue(np.isclose(gini_impurity, 0.5))


if __name__ == "__main__":
    unittest.main()