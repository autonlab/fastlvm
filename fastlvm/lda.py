import ldac

import numpy as np
import pdb
import typing, os, sys

from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces import base
from d3m import container, utils
import d3m.metadata
from d3m.metadata import hyperparams, base as metadata_base
from d3m.metadata import params

Inputs = container.List  # type: list of np.ndarray
Outputs = container.List  # type: list of np.ndarray
Predicts = container.ndarray  # type: np.ndarray
VocabularyInputs = container.DataFrame  # DataFrame: one column, one word per row

class Params(params.Params):
    topic_matrix: bytes  # Byte stream represening topics

class HyperParams(hyperparams.Hyperparams):
    k = hyperparams.UniformInt(lower=1, upper=10000, default=10, semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], description='The number of clusters to form as well as the number of centroids to generate.')
    iters = hyperparams.UniformInt(lower=1, upper=10000, default=100, semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], description='The number of iterations of inference.')
    num_top = hyperparams.UniformInt(lower=1, upper=10000, default=1, semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], description='The number of top words requested')
    seed = hyperparams.UniformInt(lower=-1000000, upper=1000000, default=1, semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], description='A random seed to use')


class LDA(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, HyperParams]):
    """
    This class provides functionality for unsupervised inference on latent Dirichlet allocation, which is a
    probabilistic topic model of corpora of documents which seeks to represent the underlying thematic structure of
    the document collection. They have emerged as a powerful new technique of finding useful structure in an
    unstructured collection as it learns distributions over words. The high probability words in each distribution
    gives us a way of understanding the contents of the corpus at a very high level. In LDA, each document of the
    corpus is assumed to have a distribution over K topics, where the discrete topic distributions are drawn from a
    symmetric dirichlet distribution. Standard packages, like those in scikit learn are inefficient in addition to
    being limited to a single machine. Whereas our underlying C++ implementation can be distributed to run on
    multiple machines. To enable the distribution through python interface is work in progress. The API is similar to
    sklearn.decomposition.LatentDirichletAllocation.
    """

    metadata = metadata_base.PrimitiveMetadata({
        "id": "f410b951-1cb6-481c-8d95-2d97b31d411d",
        "version": "1.0",
        "name": "Latent Dirichlet Allocation Topic Modelling",
        "description": "This class provides functionality for unsupervised inference on latent Dirichlet allocation, which is a probabilistic topic model of corpora of documents which seeks to represent the underlying thematic structure of the document collection. They have emerged as a powerful new technique of finding useful structure in an unstructured collection as it learns distributions over words. The high probability words in each distribution gives us a way of understanding the contents of the corpus at a very high level. In LDA, each document of the corpus is assumed to have a distribution over K topics, where the discrete topic distributions are drawn from a symmetric dirichlet distribution. Standard packages, like those in scikit learn are inefficient in addition to being limited to a single machine. Whereas our underlying C++ implementation can be distributed to run on multiple machines. To enable the distribution through python interface is work in progress. The API is similar to sklearn.decomposition.LatentDirichletAllocation.",
        "python_path": "d3m.primitives.cmu.fastlvm.LDA",
        "primitive_family": metadata_base.PrimitiveFamily.CLUSTERING,
        "algorithm_types": [ "LATENT_DIRICHLET_ALLOCATION" ],
        "keywords": ["large scale LDA", "topic modeling", "clustering"],
        "source": {
            "name": "CMU",
            "uris": [ "https://gitlab.datadrivendiscovery.org/cmu/fastlvm" ]
        },
        "installation": [
        {
            "type": "PIP",
            "package_uri": 'git+https://gitlab.datadrivendiscovery.org/cmu/fastlvm.git@{git_commit}#egg=fastlvm'.format(
                                          git_commit=utils.current_git_commit(os.path.dirname(__file__)))
        }
        ]
    })


    def __init__(self, *, hyperparams: HyperParams) -> None:
        #super(LDA, self).__init__()
        super().__init__(hyperparams = hyperparams)
        self._this = None
        self._k = hyperparams['k']
        self._iters = hyperparams['iters']
        self._num_top = hyperparams['num_top']
        self._seed = hyperparams['seed']

        self._training_inputs = None  # type: Inputs
        self._validation_inputs = None # type: Inputs
        self._fitted = False
        self._ext = None

        self.hyperparams = hyperparams


    def __del__(self):
        if self._this is not None:
            ldac.delete(self._this, self._ext)

    def set_training_data(self, *, training_inputs: Inputs, validation_inputs: Inputs, vocabulary:VocabularyInputs) -> None:
        """
        Sets training data for LDA.

        Parameters
        ----------
        training_inputs : Inputs
            A list of 1d numpy array of dtype uint32. Each numpy array contains a document with each token mapped to its word id.
        validation_inputs : Inputs
            A list of 1d numpy array of dtype uint32. Each numpy array contains a document with each token mapped to its word id. This represents validation docs to validate the results learned after each iteration of canopy algorithm.
        vocabulary : VocabularyInputs
            An one-column DataFrame. Each row contains a word.
        """

        self._training_inputs = training_inputs
        self._validation_inputs = validation_inputs

        vocab_size = len(vocabulary.index)
        vocab = [''.join(['w',str(i)]) for i in range(vocab_size)]
        self._this = ldac.new(self._k, self._iters, vocab)

        self._fitted = False


    def fit(self) -> None:
        """
        Inference on the latent Dirichley allocation model
        """
        if self._fitted:
            return

        if self._training_inputs is None:
            raise ValueError("Missing training data.")

        ldac.fit(self._this, self._training_inputs, self._validation_inputs)
        self._fitted = True

    def get_call_metadata(self) -> bool:
        """
        Returns metadata about the last ``fit`` call if it succeeded

        Returns
        -------
        Status : bool
            True/false status of fitting.

        """
        return self._fitted

    def produce(self, *, inputs: Inputs) -> base.CallResult[Outputs]:
        """
        Finds the token topic assignment (and consequently topic-per-document distribution) for the given set of docs using the learned model.

        Parameters
        ----------
        inputs : Inputs
            A list of 1d numpy array of dtype uint32. Each numpy array contains a document with each token mapped to its word id.

        Returns
        -------
        Outputs
            A list of 1d numpy array which represents index of the topic each token belongs to.

        """
        return base.CallResult(ldac.predict(self._this, inputs))

    def evaluate(self, *, inputs: Inputs) -> float:
        """
        Finds the per-token log likelihood (-ve log perplexity) of learned model on a set of test docs.

        Parameters
        ----------
        inputs : Inputs
            A list of 1d numpy array of dtype uint32. Each numpy array contains a document with each token mapped to its word id. This represents test docs to test the learned model.

        Returns
        -------
        score : float
            Final per-token log likelihood (-ve log perplexity).
        """
        return ldac.evaluate(self._this, inputs)

    def produce_top_words(self) -> Outputs:
        """
        Get the top words of each topic for this model.

        Returns
        ----------
        topic_matrix : list
            A list of size k containing list of size num_top words.
        """

        return ldac.top_words(self._this, self._num_top)

    def produce_topic_matrix(self) -> Predicts:
        """
        Get current word|topic distribution matrix for this model.

        Returns
        ----------
        topic_matrix : numpy.ndarray
            A numpy array of shape (vocab_size,k) with each column containing the word|topic distribution.
        """

        if self._ext is None:
            self._ext = ldac.topic_matrix(self._this)
        return self._ext

    def multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: Inputs, timeout: float = None, iterations: int = None) -> base.MultiCallResult:
        pass

    def fit_multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: Inputs, training_inputs: Inputs,
                          validation_inputs: Inputs, vocabulary: VocabularyInputs,
                          timeout: float = None, iterations: int = None) -> base.MultiCallResult:  # type: ignore
        return self._fit_multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations,
                                       inputs=inputs,
                                       training_inputs=training_inputs, validation_inputs=validation_inputs,
                                       vocabulary=vocabulary)

    def get_params(self) -> Params:
        """
        Get parameters of LDA.

        Parameters are basically the topic matrix in byte stream.

        Returns
        ----------
        params : Params
            A named tuple of parameters.
        """

        return Params(topic_matrix=ldac.serialize(self._this))

    def set_params(self, *, params: Params) -> None:
        """
        Set parameters of LDA.

        Parameters are basically the topic matrix in byte stream.

        Parameters
        ----------
        params : Params
            A named tuple of parameters.
        """
        self._this = ldac.deserialize(params['topic_matrix'])

    def set_random_seed(self) -> None:
        """
        NOT SUPPORTED YET
        Sets a random seed for all operations from now on inside the primitive.

        By default it sets numpy's and Python's random seed.

        Parameters
        ----------
        seed : int
            A random seed to use.
        """

        raise NotImplementedError("Not supported yet")

