from unittest import TestCase
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from fastlvm.gmm import HyperParams
from fastlvm import GMM


class TestGMM(TestCase):
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
        hp = HyperParams(k=K, iters=10, initialization='covertree')
        ctm = GMM(hyperparams=hp)
        ctm.set_training_data(inputs=pd.DataFrame(self.x))
        a = ctm.evaluate(inputs=pd.DataFrame(self.x2))

        # baseline model
        skm = GaussianMixture(K, covariance_type='diag', max_iter=10, init_params='kmeans', verbose=0)
        skm.fit(self.x, self.x2)
        b = skm.score(self.x2)

        self.assertAlmostEqual(a, b)

    def test_get_set_param(self):
        K = 10
        hp = HyperParams(k=K, iters=100, initialization='covertree')
        ctm = GMM(hyperparams=hp)
        ctm.set_training_data(inputs=pd.DataFrame(self.x))
        ctm.fit()
        a = ctm.evaluate(inputs=pd.DataFrame(self.x2))

        # call get_params
        p = ctm.get_params()
        ctm = None

        # call set_params
        ct_new = GMM(hyperparams=hp)
        ct_new.set_params(params=p)
        a_new = ct_new.evaluate(inputs=pd.DataFrame(self.x2))

        self.assertAlmostEqual(a, a_new)
