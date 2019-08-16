from unittest import TestCase
import pandas as pd
import numpy as np
from fastlvm import CoverTree
from fastlvm.covertree import HyperParams as onehp
from sklearn.neighbors import NearestNeighbors


class TestCoverTree(TestCase):
    def setUp(self) -> None:
        # Generate training and test data
        np.random.seed(seed=3)
        N = 100
        K = 10
        D = 1000
        means = 20 * np.random.rand(K, D) - 10
        x = np.vstack([np.random.randn(N, D) + means[i] for i in range(K)])
        np.random.shuffle(x)
        self.x = np.require(x, requirements=['A', 'C', 'O', 'W'])
        x2 = np.vstack([np.random.randn(N // 10, D) + means[i] for i in range(K)])
        self.x2 = np.require(x2, requirements=['A', 'C', 'O', 'W'])

    def test_produce(self):
        """
        Find the k-nearest neighbor in x for every element in x2
        Compare the results from our model and the baseline
        """
        # fit model
        hp = onehp(trunc=-1, k=1)
        self.coverTree = CoverTree(hyperparams=hp)
        self.coverTree.set_training_data(inputs=pd.DataFrame(self.x))
        self.coverTree.fit()

        # model prediction
        a = self.coverTree.produce(inputs=pd.DataFrame(self.x2))
        first = np.squeeze(self.x[a.value])

        # baseline model
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute').fit(self.x)
        distances, indices = nbrs.kneighbors(self.x2)
        second = np.squeeze(self.x[indices])

        self.assertTrue(np.array_equiv(first, second))
