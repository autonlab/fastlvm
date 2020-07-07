import os
from typing import Any

import covertreec
import numpy as np
from d3m import container, utils
from d3m.metadata import hyperparams, base as metadata_base
from d3m.metadata import params
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase

Inputs = container.DataFrame  # type: DataFrame
Outputs = container.DataFrame  # type: DataFrame


class Params(params.Params):
    tree: bytes  # Byte stream represening the tree.
    training_outputs: Any


class HyperParams(hyperparams.Hyperparams):
    trunc = hyperparams.UniformInt(lower=-1, upper=100, default=-1,
                                   semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                                   description='Level of truncation of the tree. -1 means no truncation.')
    k = hyperparams.UniformInt(lower=1, upper=100, default=1,
                               semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                               description='Number of neighbors.')


class CoverTreeRegressor(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, HyperParams]):
    """
    Regressor based on the k-nearest neighbors search using Cover Trees.
    """

    metadata = metadata_base.PrimitiveMetadata({
        "id": "92360c43-6e6f-4ff3-b1e6-5851792d8fcc",
        "version": "3.1.1",
        "name": "Nearest Neighbor Regressor with Cover Trees",
        "description": "Regressor based on the k-nearest neighbors search using Cover Trees..",
        "python_path": "d3m.primitives.regression.cover_tree.Fastlvm",
        "primitive_family": metadata_base.PrimitiveFamily.REGRESSION,
        "algorithm_types": ["K_NEAREST_NEIGHBORS"],
        "keywords": ["regression", "cover trees", "fast nearest neighbor search"],
        "source": {
            "name": "CMU",
            "contact": "mailto:donghanw@cs.cmu.edu",
            "uris": ["https://gitlab.datadrivendiscovery.org/cmu/fastlvm", "https://github.com/autonlab/fastlvm"]
        },
        "installation": [
            {
                "type": "PIP",
                "package_uri": 'git+https://github.com/autonlab/fastlvm.git@{git_commit}#egg=fastlvm'.format(
                    git_commit=utils.current_git_commit(os.path.dirname(__file__)))
            }
        ]
    })

    def __init__(self, *, hyperparams: HyperParams) -> None:
        super().__init__(hyperparams=hyperparams)
        # super(CoverTree, self).__init__()
        self._this = None
        self._trunc = hyperparams['trunc']
        self._k = hyperparams['k']
        self._training_inputs = None  # type: Inputs
        self._training_outputs = None
        self._fitted = False
        self.hyperparams = hyperparams
        self._INDEX_OUT_OF_BOUND_COUNT_MAX = 3
        self._index_out_of_bound_count = 0

    def __del__(self):
        if self._this is not None:
            covertreec.delete(self._this)

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        """
        Sets training data for CoverTree.

        Parameters
        ----------
        inputs : Inputs
            A NxD DataFrame of data points for training.
            :param outputs:
        """

        self._training_inputs = inputs.values
        self._training_outputs = outputs.values
        self._fitted = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        """
        Construct the tree
        """
        if self._fitted:
            return

        if self._training_inputs is None:
            raise ValueError("Missing training data.")

        self._this = covertreec.new(self._training_inputs, self._trunc)
        self._fitted = True

        return base.CallResult(None)

    def get_call_metadata(self) -> bool:
        """
        Returns metadata about the last ``fit`` call if it succeeded

        Returns
        -------
        Status : bool
            True/false status of fitting.

        """
        return self.fitted

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        """
        Finds the closest points for the given set of test points using the tree constructed.

        Parameters
        ----------
        inputs : Inputs
            A NxD DataFrame of data points.

        Returns
        -------
        Outputs
            The k nearest neighbours of each point.

        """
        self._index_out_of_bound_count = 0

        if self._this is None:
            raise ValueError('Fit model first')

        _y = self._training_outputs

        # Turn an index to a label. Used when k==1
        def to_label(i):  # FIXME: this may be a bug in the underlying code.
            if i < len(_y):
                return _y[i]
            else:
                if self._index_out_of_bound_count < self._INDEX_OUT_OF_BOUND_COUNT_MAX:
                    self.logger.warning("Index out of bound: index %(i)s is greater than %(max)s. This may be a bug. "
                                        "We'll use the first label.", {
                                            'i': i,
                                            'max': len(_y)
                                        })
                self._index_out_of_bound_count += 1
                return _y[0]

        # Turn indices to labels. Used when k > 1
        def to_labels(row):
            labels = []
            for i in row:
                if i < len(_y):
                    labels.append(_y[i])
                else:  # FIXME: this may be a bug in the underlying code.
                    if self._index_out_of_bound_count < self._INDEX_OUT_OF_BOUND_COUNT_MAX:
                        self.logger.warning("Index out of bounds: index %(idx)s is greater than %(max)s. "
                                            "This may be a bug. We'll replace it with the first value in the row that is "
                                            "less than %(max)s.", {
                                                'idx': i,
                                                'max': len(_y)
                                            })
                    self._index_out_of_bound_count += 1
                    for j in row:
                        if j < len(_y):
                            labels.append(_y[j])
                            break
            return np.squeeze(labels)

        k = self._k
        if k == 1:
            results, _ = covertreec.NearestNeighbour(self._this, inputs.values)
            mode = [to_label(i) for i in results]
        else:
            results, _ = covertreec.kNearestNeighbours(self._this, inputs.values, k)
            predicted = np.apply_along_axis(to_labels, 1, results)
            mode = np.mean(predicted, axis=1)

        if self._index_out_of_bound_count >= self._INDEX_OUT_OF_BOUND_COUNT_MAX:
            self.logger.warning("And {} more index out of bounds warnings.".format(1 + self._index_out_of_bound_count -
                                                                                   self._INDEX_OUT_OF_BOUND_COUNT_MAX))

        output = container.DataFrame(mode, generate_metadata=True)
        # output.metadata = inputs.metadata.clear(source=self, for_value=output, generate_metadata=True)

        return base.CallResult(output)

    def get_params(self) -> Params:
        """
        Get parameters of KMeans.
OB
        Parameters are basically the cluster centres in byte stream.

        Returns
        ----------
        params : Params
            A named tuple of parameters.
        """

        return Params(tree=covertreec.serialize(self._this),
                      training_outputs=self._training_outputs)

    def set_params(self, *, params: Params) -> None:
        """
        Set parameters of cover tree.

        Parameters are basically the tree structure in byte stream.

        Parameters
        ----------
        params : Params
            A named tuple of parameters.
        """
        self._this = covertreec.deserialize(params['tree'])
        self._training_outputs = params['training_outputs']
