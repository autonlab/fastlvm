import covertreec

import numpy as np
import pdb
import typing, sys, os

from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces import base
from d3m import container, utils
import d3m.metadata
from d3m.metadata import hyperparams, base as metadata_base
from d3m.metadata import params
from scipy import stats


Inputs = container.DataFrame  # type: DataFrame
Outputs = container.DataFrame  # type: DataFrame

class Params(params.Params):
    tree: bytes # Byte stream represening the tree.

class HyperParams(hyperparams.Hyperparams):
    trunc = hyperparams.UniformInt(lower=-1, upper=100,default=-1,semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],description='Level of truncation of the tree. -1 means no truncation.')
    k = hyperparams.UniformInt(lower=1, upper=10,default=3,semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],description='Number of neighbors.')


class CoverTreeClassifier(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, HyperParams]):
    """
    Classifier implementing the k-nearest neighbors vote using Cover Trees.
    """

    metadata = metadata_base.PrimitiveMetadata({
         "id": "e770fae6-da6d-45f8-86bf-38a121a4e65a",
         "version": "3.0.0",
         "name": "Nearest Neighbor Classification with Cover Trees",
         "description": "Classifier implementing the k-nearest neighbors vote using Cover Trees.",
         "python_path": "d3m.primitives.classification.cover_tree.Fastlvm",
         "primitive_family": metadata_base.PrimitiveFamily.CLASSIFICATION,
         "algorithm_types": [ "K_NEAREST_NEIGHBORS" ],
         "keywords": ["classification", "cover trees", "fast nearest neighbor search"],
         "source": {
             "name": "CMU",
             "contact": "mailto:donghanw@cs.cmu.edu",
             "uris": ["https://gitlab.datadrivendiscovery.org/cmu/fastlvm", "https://github.com/autonlab/fastlvm"]
         },
         "installation": [
         {
             "type": "PIP",
             "package_uri": 'git+https://github.com/autonlab/fastlvm/fastlvm.git@{git_commit}#egg=fastlvm'.format(
                                                        git_commit=utils.current_git_commit(os.path.dirname(__file__)))
         }
         ]
     })

    def __init__(self, *, hyperparams: HyperParams) -> None:
        super().__init__(hyperparams = hyperparams)
        #super(CoverTree, self).__init__()
        self._this = None
        self._trunc = hyperparams['trunc']
        self._k = hyperparams['k']
        self._training_inputs = None  # type: Inputs
        self._training_outputs = None
        self._fitted = False
        self.hyperparams = hyperparams

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
        if self._this is None:
            raise ValueError('Fit model first')

        _y = self._training_outputs

        # Turn an index to a label. Used when k==1
        def to_label(i):  # FIXME: this may be a bug in the underlying code.
            if i < len(_y):
                return _y[i]
            else:
                self.logger.warning("Index out of bound: index %(i)s is greater than %(max)s. This may be a bug. "
                                    "We'll use the first label.", {
                                        'i': i,
                                        'max': len(_y)
                                    })
                return _y[0]

        # Turn indices to labels. Used when k > 1
        def to_labels(row):
            labels = []
            for i in row:
                if i < len(_y):
                    labels.append(_y[i])
                else:  # FIXME: this may be a bug in the underlying code.
                    self.logger.warning("Index out of bounds: index %(idx)s is greater than %(max)s in row %(row)s. "
                                        "This may be a bug. We'll replace it with the first value in the row that is "
                                        "less than %(max)s.", {
                                            'idx': i,
                                            'max': len(_y),
                                            'row': str(row)
                                        })
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
            mode, _ = stats.mode(predicted, axis=1)

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

        return Params(tree=covertreec.serialize(self._this))

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

