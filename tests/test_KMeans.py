from unittest import TestCase
from fastlvm import KMeans
from fastlvm.kmeans import HyperParams
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans as sKMeans


class TestKMeans(TestCase):
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
        K = 10
        hp = HyperParams(k=K, iters=100, initialization='covertree')
        ctm = KMeans(hyperparams=hp)
        ctm.set_training_data(inputs=pd.DataFrame(self.x))
        ctm.fit()
        a = ctm.evaluate(inputs=pd.DataFrame(self.x2))

        # baseline model
        skm = sKMeans(K, 'k-means++', 1, 10, verbose=0)
        skm.fit(self.x, self.x2)
        b = skm.score(self.x2)

        self.assertAlmostEqual(a, b, places=1)

    def test_get_set_param(self):
        K = 10
        hp = HyperParams(k=K, iters=100, initialization='covertree')
        ctm = KMeans(hyperparams=hp)
        ctm.set_training_data(inputs=pd.DataFrame(self.x))
        ctm.fit()
        a = ctm.evaluate(inputs=pd.DataFrame(self.x2))

        # call get_params
        p = ctm.get_params()
        ctm = None

        # call set_params
        ct_new = KMeans(hyperparams=hp)
        ct_new.set_params(params=p)
        a_new = ct_new.evaluate(inputs=pd.DataFrame(self.x2))

        self.assertAlmostEqual(a, a_new)
